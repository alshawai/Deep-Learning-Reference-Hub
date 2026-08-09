"""
Random Search Tests
===================

A search that samples from the wrong distribution still returns a best-of-N
configuration, still beats its own mean, and still looks like it worked. So the
distributions are tested against their defining property rather than their range:
log-uniform must be uniform in the exponent, and the power law must put its mass
where the docstring says it does.

The convenience wrapper's shorthand parser gets the same treatment. It decides
between a numeric range and a categorical choice by inspecting the value, and a
misread there silently changes what space is being searched.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest
from conftest import load

rs = load("hyperparameter_tuning/random_search.py")
ParameterDistribution = rs.ParameterDistribution
UniformDistribution = rs.UniformDistribution
LogUniformDistribution = rs.LogUniformDistribution
IntegerDistribution = rs.IntegerDistribution
ChoiceDistribution = rs.ChoiceDistribution
PowerDistribution = rs.PowerDistribution
RandomSearchOptimizer = rs.RandomSearchOptimizer
RandomSearchResult = rs.RandomSearchResult
random_search = rs.random_search
analyze_parameter_importance = rs.analyze_parameter_importance

SAMPLES = 4000


@pytest.fixture(autouse=True)
def seeded():
    """Every distribution here draws through the global numpy seed."""
    np.random.seed(0)


class TestDistributions:
    def test_the_base_class_is_abstract(self):
        with pytest.raises(NotImplementedError):
            ParameterDistribution().sample()

    def test_uniform_stays_in_range_and_fills_it(self):
        dist = UniformDistribution(-2.0, 5.0)
        draws = np.array([dist.sample() for _ in range(SAMPLES)])
        assert draws.min() >= -2.0 and draws.max() <= 5.0
        assert draws.mean() == pytest.approx(1.5, abs=0.15)

    def test_log_uniform_is_uniform_in_the_exponent(self):
        """
        The property that makes it the right choice for a learning rate: each
        decade gets equal probability. A plain uniform over the same bounds
        would put ~90% of its draws in the top decade.
        """
        dist = LogUniformDistribution(1e-5, 1e-1)
        decades = np.floor(np.log10([dist.sample() for _ in range(SAMPLES)]))
        counts = np.array([(decades == d).sum() for d in (-5, -4, -3, -2)])
        assert counts.min() > 0.7 * SAMPLES / 4

    def test_log_uniform_rejects_non_positive_bounds(self):
        for low, high in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)):
            with pytest.raises(ValueError):
                LogUniformDistribution(low, high)

    def test_integer_draws_are_ints_in_the_half_open_range(self):
        dist = IntegerDistribution(3, 7)
        draws = [dist.sample() for _ in range(500)]
        assert all(isinstance(value, int) for value in draws)
        assert set(draws) == {3, 4, 5, 6}, "high is exclusive"

    def test_choice_draws_only_from_the_given_options(self):
        dist = ChoiceDistribution(["adam", "sgd", "rmsprop"])
        assert set(dist.sample() for _ in range(200)) <= {"adam", "sgd", "rmsprop"}

    def test_choice_weights_are_normalized_and_respected(self):
        dist = ChoiceDistribution(["a", "b"], probabilities=[3.0, 1.0])
        assert sum(dist.probabilities) == pytest.approx(1.0)
        draws = [dist.sample() for _ in range(SAMPLES)]
        assert draws.count("a") / SAMPLES == pytest.approx(0.75, abs=0.03)

    def test_choice_rejects_a_weight_per_option_mismatch(self):
        with pytest.raises(ValueError):
            ChoiceDistribution(["a", "b", "c"], probabilities=[0.5, 0.5])

    def test_power_above_one_concentrates_near_the_lower_bound(self):
        """
        Regression test. The exponent was applied as `u ** (1 / power)`, which is
        the standard power-function distribution and skews the other way: the
        default power=2 drew a mean of 2/3 of the range. The class exists to
        favour small values, and every weight-decay sweep using it was in fact
        sampling large ones.
        """
        draws = np.array(
            [PowerDistribution(0.0, 1.0, power=2.0).sample() for _ in range(SAMPLES)]
        )
        assert draws.mean() == pytest.approx(1 / 3, abs=0.03)
        assert (draws < 0.5).mean() > 0.6

    def test_power_of_one_is_uniform(self):
        draws = np.array(
            [PowerDistribution(0.0, 1.0, power=1.0).sample() for _ in range(SAMPLES)]
        )
        assert draws.mean() == pytest.approx(0.5, abs=0.03)

    def test_a_higher_power_pushes_the_mass_further_down(self):
        def mean_of(power):
            return np.mean(
                [
                    PowerDistribution(0.0, 1.0, power=power).sample()
                    for _ in range(SAMPLES)
                ]
            )

        assert mean_of(4.0) < mean_of(2.0) < mean_of(1.0)

    def test_power_stays_within_its_bounds(self):
        draws = np.array(
            [PowerDistribution(1e-6, 1e-2, power=3.0).sample() for _ in range(1000)]
        )
        assert draws.min() >= 1e-6 and draws.max() <= 1e-2

    @pytest.mark.parametrize(
        "dist",
        [
            UniformDistribution(0.0, 1.0),
            LogUniformDistribution(1e-4, 1e-1),
            IntegerDistribution(1, 5),
            ChoiceDistribution([1, 2]),
            PowerDistribution(0.0, 1.0),
        ],
        ids=["uniform", "loguniform", "integer", "choice", "power"],
    )
    def test_every_distribution_reports_its_configuration(self, dist):
        """`repr` lands in logs and notebooks, so it has to say what was searched."""
        assert dist.__class__.__name__ in repr(dist)


class TestSampling:
    def test_a_sample_covers_exactly_the_search_space_keys(self):
        opt = RandomSearchOptimizer(
            lambda p: 0.0,
            {
                "lr": LogUniformDistribution(1e-4, 1e-1),
                "units": IntegerDistribution(8, 64),
            },
            n_iter=1,
        )
        assert set(opt.sample_parameters()) == {"lr", "units"}

    def test_sampling_n_configurations_returns_n_of_them(self):
        opt = RandomSearchOptimizer(
            lambda p: 0.0, {"x": UniformDistribution(0.0, 1.0)}, n_iter=1
        )
        assert len(opt.sample_multiple_parameters(7)) == 7

    def test_the_random_state_makes_a_run_reproducible(self):
        def run():
            return random_search(
                lambda p: p["x"],
                {"x": UniformDistribution(0.0, 1.0)},
                n_iter=20,
                random_state=7,
                verbose=False,
            )

        first, second = run(), run()
        assert first.all_scores == second.all_scores
        assert first.best_params == second.best_params


class TestOptimize:
    def test_the_search_maximizes_and_reports_the_argmax(self):
        """
        Sign convention: higher is better. A minimizing search would return the
        worst configuration while every other field still looked sane.
        """
        result = random_search(
            lambda p: -((p["x"] - 2.0) ** 2),
            {"x": UniformDistribution(-5.0, 5.0)},
            n_iter=200,
            random_state=0,
            verbose=False,
        )
        assert result.best_score == max(result.all_scores)
        assert result.best_params["x"] == pytest.approx(2.0, abs=0.2)

    def test_every_iteration_is_recorded_once(self):
        result = random_search(
            lambda p: p["x"],
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=35,
            random_state=0,
            verbose=False,
        )
        assert len(result.all_params) == len(result.all_scores) == 35
        assert result.statistics["total_evaluations"] == 35

    def test_a_raising_objective_scores_worst_instead_of_ending_the_search(self):
        """
        A configuration that cannot be trained is a data point, not a crash. It
        must also never win, or the search reports an unusable best.
        """

        def objective(params):
            if params["x"] > 0.5:
                raise RuntimeError("simulated OOM")
            return params["x"]

        with pytest.warns(UserWarning):
            result = random_search(
                objective,
                {"x": UniformDistribution(0.0, 1.0)},
                n_iter=40,
                random_state=0,
                verbose=False,
            )
        assert result.statistics["failed_evaluations"] > 0
        assert np.isfinite(result.best_score)
        assert result.best_params["x"] <= 0.5

    def test_a_non_finite_score_is_treated_as_a_failure(self):
        result = random_search(
            lambda p: np.nan if p["x"] > 0.5 else p["x"],
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=40,
            random_state=0,
            verbose=False,
        )
        assert result.statistics["failed_evaluations"] > 0
        assert np.isfinite(result.best_score)

    def test_the_statistics_summarize_only_the_successful_evaluations(self):
        result = random_search(
            lambda p: 1.0 if p["x"] > 0.5 else np.inf,
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=30,
            random_state=0,
            verbose=False,
        )
        stats = result.statistics
        assert stats["successful_evaluations"] + stats["failed_evaluations"] == 30
        assert stats["mean_score"] == pytest.approx(1.0)

    def test_early_stopping_ends_the_search_before_the_budget(self):
        result = random_search(
            lambda p: 0.0,  # never improves after the first evaluation
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=500,
            random_state=0,
            early_stopping=True,
            patience=5,
            verbose=False,
        )
        assert result.statistics["iterations_completed"] < 500
        assert result.statistics["early_stopped"] is True

    def test_without_early_stopping_the_whole_budget_is_spent(self):
        result = random_search(
            lambda p: 0.0,
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=40,
            random_state=0,
            verbose=False,
        )
        assert result.statistics["iterations_completed"] == 40
        assert result.statistics["early_stopped"] is False

    def test_a_steadily_improving_objective_is_never_stopped_early(self):
        counter = {"n": 0}

        def improving(params):
            counter["n"] += 1
            return float(counter["n"])

        result = random_search(
            improving,
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=30,
            random_state=0,
            early_stopping=True,
            patience=3,
            verbose=False,
        )
        assert result.statistics["iterations_completed"] == 30

    def test_verbose_false_prints_nothing(self, capsys):
        random_search(
            lambda p: p["x"],
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=10,
            random_state=0,
            verbose=False,
        )
        assert capsys.readouterr().out == ""


class TestShorthandSearchSpace:
    def parse(self, spec, n=1):
        """
        Runs one search and hands back every value that was sampled.

        Drawn inside a single search rather than one per call: `random_state`
        reseeds the global generator, so repeated one-iteration searches would
        return the same value every time and hide the shape of the space.
        """
        seen = []

        def objective(params):
            seen.append(params["p"])
            return 0.0

        random_search(objective, {"p": spec}, n_iter=n, random_state=0, verbose=False)
        return seen[0] if n == 1 else seen

    def test_a_two_element_list_of_strings_is_categorical(self):
        """
        Regression test. The length-2 branch was checked before the list branch,
        so any two-option categorical list was read as a numeric range and
        raised. `["relu", "tanh"]` is an ordinary search space.
        """
        assert self.parse(["relu", "tanh"]) in {"relu", "tanh"}

    def test_a_two_element_numeric_list_is_still_categorical(self):
        """A list means choose-one, so it must not become a range of its bounds."""
        assert set(self.parse([16, 256], n=50)) == {16, 256}

    def test_a_longer_list_is_categorical(self):
        assert set(self.parse([16, 32, 64, 128], n=100)) == {16, 32, 64, 128}

    def test_a_float_tuple_becomes_a_uniform_range(self):
        value = self.parse((0.0, 0.5))
        assert 0.0 <= value <= 0.5
        assert not isinstance(value, int)

    def test_an_integer_tuple_covers_both_endpoints(self):
        """`(64, 512)` reads as inclusive, so the top of the range is reachable."""
        assert set(self.parse((1, 3), n=200)) == {1, 2, 3}

    def test_a_wide_positive_range_is_sampled_logarithmically(self):
        values = self.parse((1e-5, 1e-1), n=400)
        decades = np.floor(np.log10(values))
        assert len(set(decades)) >= 3, "a uniform draw would sit in the top decade"

    def test_a_distribution_object_passes_through_untouched(self):
        value = self.parse(LogUniformDistribution(1e-3, 1e-2))
        assert 1e-3 <= value <= 1e-2

    def test_a_non_numeric_tuple_is_rejected(self):
        with pytest.raises(ValueError):
            self.parse(("relu", "tanh"))

    def test_an_unsupported_specification_is_rejected(self):
        for spec in (0.5, "uniform", (1, 2, 3)):
            with pytest.raises(ValueError):
                self.parse(spec)


class TestParameterImportance:
    def test_a_parameter_driving_the_score_outranks_an_ignored_one(self):
        result = random_search(
            lambda p: p["signal"],
            {
                "signal": UniformDistribution(0.0, 1.0),
                "noise": UniformDistribution(0.0, 1.0),
            },
            n_iter=120,
            random_state=0,
            verbose=False,
        )
        importance = analyze_parameter_importance(result, top_n=60)
        assert importance["signal"] > importance["noise"]

    def test_the_scores_are_ranked_high_to_low(self):
        result = random_search(
            lambda p: p["signal"],
            {
                "signal": UniformDistribution(0.0, 1.0),
                "noise": UniformDistribution(0.0, 1.0),
            },
            n_iter=80,
            random_state=1,
            verbose=False,
        )
        values = list(analyze_parameter_importance(result).values())
        assert values == sorted(values, reverse=True)

    def test_a_categorical_parameter_is_left_out_rather_than_coerced(self):
        result = random_search(
            lambda p: p["x"],
            {"x": UniformDistribution(0.0, 1.0), "opt": ChoiceDistribution(["a", "b"])},
            n_iter=40,
            random_state=0,
            verbose=False,
        )
        assert "opt" not in analyze_parameter_importance(result)

    def test_a_search_too_short_to_correlate_returns_nothing(self):
        result = random_search(
            lambda p: p["x"],
            {"x": UniformDistribution(0.0, 1.0)},
            n_iter=1,
            random_state=0,
            verbose=False,
        )
        assert analyze_parameter_importance(result) == {}
