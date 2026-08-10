"""
Learning Rate Finder Tests
==========================

The finder runs a real training loop, so the tests supply a deterministic
`BaseTrainer` whose loss curve is known in advance. That makes the suggestion
checkable: on a curve whose steepest descent sits at a chosen learning rate, the
suggestion must land near it rather than merely somewhere in range.

The early-exit paths get equal weight. A finder that never stops on divergence
spends its whole budget training a model that has already blown up, and reports
the resulting garbage as a suggestion.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.tuning.learning_rate_finder import (
    BaseTrainer,
    FunctionTrainer,
    LearningRateFinder,
    suggest_learning_rate_schedule,
)


class CurveTrainer(BaseTrainer):
    """
    Replays a loss as a fixed function of the learning rate, with no model state.

    Being a pure function of `lr` is what makes the suggestion assertable: the
    curve's shape is chosen by the test, so the location of the steepest descent
    is known before the finder runs.
    """

    def __init__(self, curve):
        self.curve = curve
        self.calls = []
        self.resets = 0

    def train_batch(self, learning_rate):
        self.calls.append(learning_rate)
        return self.curve(learning_rate)

    def reset_model(self):
        self.resets += 1


def valley(centre=1e-2, width=1.0):
    """A loss that falls to a minimum at `centre` and rises again after it."""
    return lambda lr: 1.0 + ((np.log10(lr) - np.log10(centre)) / width) ** 2


class TestLearningRateGrid:
    def test_exponential_mode_is_geometric_between_the_bounds(self):
        finder = LearningRateFinder(
            CurveTrainer(valley()), min_lr=1e-4, max_lr=1e-1, num_iterations=4
        )
        rates = finder._generate_learning_rates()
        np.testing.assert_allclose(rates, [1e-4, 1e-3, 1e-2, 1e-1], rtol=1e-9)

    def test_linear_mode_is_evenly_spaced_between_the_bounds(self):
        finder = LearningRateFinder(
            CurveTrainer(valley()),
            min_lr=0.0,
            max_lr=1.0,
            num_iterations=5,
            step_mode="linear",
        )
        np.testing.assert_allclose(
            finder._generate_learning_rates(), [0.0, 0.25, 0.5, 0.75, 1.0]
        )

    def test_the_step_mode_is_case_insensitive(self):
        """
        Regression test. `step_mode` is lowercased into the attribute, then the
        raw argument was validated, so the normalization could never take effect
        and "EXP" raised instead of being accepted.
        """
        finder = LearningRateFinder(CurveTrainer(valley()), step_mode="EXP")
        assert finder.step_mode == "exp"

    def test_an_unknown_step_mode_is_rejected(self):
        with pytest.raises(ValueError):
            LearningRateFinder(CurveTrainer(valley()), step_mode="quadratic")

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(ValueError):
            LearningRateFinder(CurveTrainer(valley()), min_lr=1.0, max_lr=1e-3)

    @pytest.mark.parametrize("beta", [0.0, 1.0, -0.5, 1.5])
    def test_a_smoothing_factor_outside_the_open_unit_interval_is_rejected(self, beta):
        with pytest.raises(ValueError):
            LearningRateFinder(CurveTrainer(valley()), smooth_beta=beta)


class TestSmoothing:
    def test_the_smoothed_curve_starts_at_the_first_raw_loss(self):
        finder = LearningRateFinder(CurveTrainer(valley()), smooth_beta=0.9)
        losses = np.array([5.0, 1.0, 1.0, 1.0])
        assert finder._smooth_losses(losses)[0] == 5.0

    def test_smoothing_follows_the_seeded_exponential_average(self):
        finder = LearningRateFinder(CurveTrainer(valley()), smooth_beta=0.5)
        losses = np.array([4.0, 0.0, 0.0])
        np.testing.assert_allclose(finder._smooth_losses(losses), [4.0, 2.0, 1.0])

    def test_a_constant_curve_smooths_to_itself(self):
        """Seeding at the first sample is what makes this exact, with no warm-up bias."""
        finder = LearningRateFinder(CurveTrainer(valley()), smooth_beta=0.98)
        losses = np.full(20, 3.0)
        np.testing.assert_allclose(finder._smooth_losses(losses), 3.0)

    def test_more_smoothing_tracks_the_raw_curve_less_closely(self):
        finder_light = LearningRateFinder(CurveTrainer(valley()), smooth_beta=0.1)
        finder_heavy = LearningRateFinder(CurveTrainer(valley()), smooth_beta=0.95)
        losses = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
        light = np.abs(finder_light._smooth_losses(losses) - losses).sum()
        heavy = np.abs(finder_heavy._smooth_losses(losses) - losses).sum()
        assert heavy > light


class TestDivergenceDetection:
    def test_no_divergence_is_reported_before_ten_iterations(self):
        """A short history is noise; calling it divergence aborts every run."""
        finder = LearningRateFinder(CurveTrainer(valley()), divergence_threshold=2.0)
        losses = np.array([1.0] + [100.0] * 9)
        assert finder._detect_divergence(losses, 9) is False

    def test_a_loss_above_the_threshold_times_the_minimum_diverges(self):
        finder = LearningRateFinder(CurveTrainer(valley()), divergence_threshold=4.0)
        losses = np.array([1.0] * 10 + [4.1])
        assert finder._detect_divergence(losses, 10) is True

    def test_a_loss_below_the_threshold_does_not_diverge(self):
        finder = LearningRateFinder(CurveTrainer(valley()), divergence_threshold=4.0)
        losses = np.array([1.0] * 10 + [3.9])
        assert finder._detect_divergence(losses, 10) is False

    def test_the_threshold_is_relative_to_the_minimum_not_the_first_loss(self):
        """
        The curve descends before it blows up, so anchoring on `losses[0]` would
        let the loss climb back to its starting value unnoticed.
        """
        finder = LearningRateFinder(CurveTrainer(valley()), divergence_threshold=2.0)
        losses = np.array([10.0] + [1.0] * 10 + [2.5])
        assert finder._detect_divergence(losses, 11) is True


class TestFind:
    def test_the_model_is_reset_before_the_sweep(self):
        """
        Each learning rate must be judged from the same starting point. A sweep
        that inherits the previous run's weights measures history, not the rate.
        """
        trainer = CurveTrainer(valley())
        LearningRateFinder(trainer, min_lr=1e-5, max_lr=1e-1, num_iterations=20).find(
            verbose=False
        )
        assert trainer.resets == 1

    def test_every_learning_rate_in_the_grid_is_trained_on(self):
        trainer = CurveTrainer(valley())
        finder = LearningRateFinder(
            trainer, min_lr=1e-5, max_lr=1e-1, num_iterations=20
        )
        result = finder.find(verbose=False)
        np.testing.assert_allclose(trainer.calls, finder._generate_learning_rates())
        assert len(result.losses) == 20

    def test_the_suggestion_tracks_the_steepest_descent_of_the_curve(self):
        """
        The documented rule is `min_gradient_lr / 10`. Moving the valley must move
        the suggestion with it, which a hardcoded or accidentally constant
        suggestion would not do.
        """
        suggestions = {}
        for centre in (1e-3, 1e-1):
            trainer = CurveTrainer(valley(centre=centre))
            result = LearningRateFinder(
                trainer, min_lr=1e-6, max_lr=1.0, num_iterations=60
            ).find(verbose=False)
            suggestions[centre] = result.suggested_lr
            assert result.suggested_lr == pytest.approx(
                result.min_gradient_lr / 10, rel=1e-9
            )
        assert suggestions[1e-3] < suggestions[1e-1]

    def test_the_suggestion_sits_below_the_loss_minimum(self):
        """
        The whole point of the test: pick a rate on the descending slope, not the
        one at the bottom, which is already on the edge of instability.
        """
        trainer = CurveTrainer(valley(centre=1e-2))
        result = LearningRateFinder(
            trainer, min_lr=1e-6, max_lr=1.0, num_iterations=60
        ).find(verbose=False)
        assert result.suggested_lr < result.analysis["min_loss_lr"]

    def test_the_sweep_stops_when_the_loss_diverges(self):
        trainer = CurveTrainer(lambda lr: 1.0 if lr < 1e-2 else 1e6)
        result = LearningRateFinder(
            trainer,
            min_lr=1e-6,
            max_lr=1.0,
            num_iterations=100,
            divergence_threshold=4.0,
        ).find(verbose=False)
        assert len(result.losses) < 100
        assert len(result.learning_rates) == len(result.losses)

    def test_the_sweep_stops_on_a_non_finite_loss(self):
        trainer = CurveTrainer(lambda lr: 1.0 if lr < 1e-2 else np.nan)
        result = LearningRateFinder(
            trainer, min_lr=1e-6, max_lr=1.0, num_iterations=100
        ).find(verbose=False)
        assert np.all(np.isfinite(result.losses))
        assert len(result.learning_rates) == len(result.losses)

    def test_a_raising_trainer_ends_the_sweep_rather_than_the_process(self):
        """An out-of-memory or overflow at a high rate is data, not a crash."""

        def explode(lr):
            if lr > 1e-2:
                raise OverflowError("simulated blow-up")
            return 1.0

        result = LearningRateFinder(
            CurveTrainer(explode), min_lr=1e-6, max_lr=1.0, num_iterations=100
        ).find(verbose=False)
        assert len(result.losses) >= 5

    def test_too_few_usable_losses_is_an_error_not_a_suggestion(self):
        """
        Returning a number from three points would look like a result. There is
        no curve to analyse, so the finder has to say so.
        """
        trainer = CurveTrainer(lambda lr: 1.0 if lr < 1e-5 else np.nan)
        with pytest.raises(RuntimeError):
            LearningRateFinder(
                trainer, min_lr=1e-6, max_lr=1.0, num_iterations=10
            ).find(verbose=False)

    def test_the_result_carries_matching_arrays_and_analysis(self):
        trainer = CurveTrainer(valley())
        result = LearningRateFinder(
            trainer, min_lr=1e-5, max_lr=1e-1, num_iterations=30
        ).find(verbose=False)

        assert (
            len(result.learning_rates)
            == len(result.losses)
            == len(result.smoothed_losses)
        )
        assert result.analysis["min_loss"] == pytest.approx(result.losses.min())
        assert result.analysis["max_loss"] == pytest.approx(result.losses.max())
        assert result.suggested_lr == result.analysis["suggested_lr"]
        assert result.min_gradient_lr == result.analysis["min_gradient_lr"]

    def test_verbose_false_prints_nothing(self, capsys):
        LearningRateFinder(
            CurveTrainer(valley()), min_lr=1e-5, max_lr=1e-1, num_iterations=20
        ).find(verbose=False)
        assert capsys.readouterr().out == ""


def test_the_function_trainer_forwards_to_the_supplied_callbacks():
    trained, reset = [], []
    trainer = FunctionTrainer(
        lambda lr: trained.append(lr) or 0.5, lambda: reset.append(True)
    )
    assert trainer.train_batch(0.25) == 0.5
    trainer.reset_model()
    assert trained == [0.25]
    assert reset == [True]


class TestScheduleSuggestions:
    @pytest.fixture
    def result(self):
        trainer = CurveTrainer(valley(centre=1e-2))
        return LearningRateFinder(
            trainer, min_lr=1e-6, max_lr=1.0, num_iterations=60
        ).find(verbose=False)

    @pytest.mark.parametrize("schedule_type", ["onecycle", "cyclic", "cosine", "step"])
    def test_each_schedule_is_labelled_and_described(self, result, schedule_type):
        suggestion = suggest_learning_rate_schedule(result, schedule_type)
        assert suggestion["schedule_type"] == schedule_type
        assert suggestion["description"]

    def test_the_cyclical_bounds_come_from_the_finder_and_are_ordered(self, result):
        for schedule_type in ("onecycle", "cyclic"):
            suggestion = suggest_learning_rate_schedule(result, schedule_type)
            assert suggestion["base_lr"] == result.suggested_lr
            assert suggestion["max_lr"] == result.min_gradient_lr
            assert suggestion["base_lr"] < suggestion["max_lr"]

    def test_cosine_anneals_from_the_high_rate_down_to_the_low_one(self, result):
        suggestion = suggest_learning_rate_schedule(result, "cosine")
        assert suggestion["min_lr"] < suggestion["initial_lr"]

    def test_an_unknown_schedule_is_rejected(self, result):
        with pytest.raises(ValueError):
            suggest_learning_rate_schedule(result, "triangular9")
