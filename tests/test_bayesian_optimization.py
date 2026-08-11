"""
Bayesian Optimization Tests
===========================

The surrogate and the acquisition function are tested separately from the loop
that drives them, because each fails quietly. A GP that does not interpolate its
own observations still returns plausible numbers; an Expected Improvement that
drops its exploration term still returns a positive value everywhere and simply
stops exploring.

The loop itself is checked against the property that justifies the method: on a
smooth objective it must beat an equal budget of random draws. Anything less and
the extra machinery is costing evaluations for nothing.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest
from scipy.stats import norm

from dlhub.tuning.bayesian import (
    BayesianOptimizer,
    GaussianProcess,
    bayesian_optimize,
)


def quadratic(params):
    """Smooth, single-peaked, maximum of 0 at (2, -1)."""
    return -((params["x"] - 2.0) ** 2) - (params["y"] + 1.0) ** 2


class TestKernel:
    def test_a_point_with_itself_is_the_kernel_variance(self):
        gp = GaussianProcess(kernel_variance=2.5)
        X = np.array([[0.3, 0.7]])
        assert gp.rbf_kernel(X, X)[0, 0] == pytest.approx(2.5)

    def test_similarity_falls_off_with_distance(self):
        gp = GaussianProcess()
        near = gp.rbf_kernel(np.array([[0.0]]), np.array([[0.1]]))[0, 0]
        far = gp.rbf_kernel(np.array([[0.0]]), np.array([[5.0]]))[0, 0]
        assert near > far
        assert far == pytest.approx(0.0, abs=1e-5)

    def test_the_lengthscale_sets_how_fast(self):
        """
        A short lengthscale means neighbouring configurations tell you little
        about each other, which is what turns the surrogate back into noise.
        """
        distance = np.array([[1.0]]), np.array([[0.0]])
        short = GaussianProcess(kernel_lengthscale=0.1).rbf_kernel(*distance)[0, 0]
        long = GaussianProcess(kernel_lengthscale=10.0).rbf_kernel(*distance)[0, 0]
        assert short < long

    def test_the_matrix_has_one_entry_per_pair(self):
        gp = GaussianProcess()
        K = gp.rbf_kernel(np.zeros((4, 2)), np.zeros((7, 2)))
        assert K.shape == (4, 7)

    def test_the_kernel_is_symmetric(self):
        gp = GaussianProcess()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 3))
        np.testing.assert_allclose(gp.rbf_kernel(X, X), gp.rbf_kernel(X, X).T)


class TestGaussianProcess:
    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(ValueError):
            GaussianProcess().predict(np.array([[0.0]]))

    def test_the_posterior_passes_through_the_observations(self):
        """
        With near-zero noise the GP is an interpolator. If it does not reproduce
        what it was told, every acquisition value computed from it is fiction.
        """
        gp = GaussianProcess(noise_variance=1e-10)
        X = np.array([[0.1], [0.4], [0.9]])
        y = np.array([1.0, -0.5, 2.0])
        gp.fit(X, y)
        mean, _ = gp.predict(X)
        np.testing.assert_allclose(mean, y, atol=1e-4)

    def test_uncertainty_collapses_at_observed_points_and_grows_away_from_them(self):
        """The property the whole method rests on: the model knows what it does not know."""
        gp = GaussianProcess(noise_variance=1e-10)
        gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 0.0]))
        _, std_observed = gp.predict(np.array([[0.0]]))
        _, std_between = gp.predict(np.array([[0.5]]))
        assert std_observed[0] < std_between[0]

    def test_the_standard_deviation_is_never_negative_or_nan(self):
        """
        Subtracting two nearly equal quantities can go slightly negative in
        floating point, and the square root of that is a nan that silently
        poisons every acquisition value.
        """
        gp = GaussianProcess()
        X = np.array([[0.0], [1e-9], [2e-9]])  # nearly duplicated points
        gp.fit(X, np.array([1.0, 1.0, 1.0]))
        _, std = gp.predict(X)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)

    def test_a_singular_kernel_matrix_falls_back_instead_of_raising(self):
        gp = GaussianProcess(noise_variance=0.0)
        X = np.array([[0.5], [0.5], [0.5]])  # exact duplicates
        with pytest.warns(UserWarning, match="singular"):
            gp.fit(X, np.array([1.0, 1.0, 1.0]))
        mean, std = gp.predict(X)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))

    def test_the_training_data_is_copied_not_aliased(self):
        gp = GaussianProcess()
        X = np.array([[0.0], [1.0]])
        y = np.array([1.0, 2.0])
        gp.fit(X, y)
        y[0] = 99.0
        assert gp.y_train[0] == 1.0

    def test_predictions_have_one_entry_per_test_point(self):
        gp = GaussianProcess()
        gp.fit(np.zeros((3, 2)), np.zeros(3))
        mean, std = gp.predict(np.zeros((6, 2)))
        assert mean.shape == std.shape == (6,)


def fitted_optimizer(acquisition="ei", xi=0.01, kappa=2.576):
    """An optimizer with three observations already in hand and its GP fitted."""
    opt = BayesianOptimizer(
        quadratic, {"x": (0.0, 1.0)}, acquisition=acquisition, xi=xi, kappa=kappa
    )
    opt.X_observed = [np.array([0.1]), np.array([0.5]), np.array([0.9])]
    opt.y_observed = [0.0, 1.0, 0.0]
    opt.gp.fit(np.array(opt.X_observed), np.array(opt.y_observed))
    return opt


class TestAcquisition:
    def test_expected_improvement_matches_its_closed_form(self):
        opt = fitted_optimizer(xi=0.01)
        X = np.array([[0.3], [0.7]])
        mean, std = opt.gp.predict(X)
        improvement = mean - max(opt.y_observed) - 0.01
        z = improvement / std
        expected = improvement * norm.cdf(z) + std * norm.pdf(z)
        np.testing.assert_allclose(opt._expected_improvement(X), expected, rtol=1e-9)

    def test_expected_improvement_is_never_negative(self):
        opt = fitted_optimizer()
        grid = np.linspace(0, 1, 50).reshape(-1, 1)
        assert np.all(opt._expected_improvement(grid) >= 0.0)

    def test_expected_improvement_is_near_zero_at_an_observed_point(self):
        """
        No uncertainty and no headroom there, so there is nothing to gain by
        spending another training run on it.
        """
        opt = fitted_optimizer()
        assert opt._expected_improvement(np.array([[0.5]]))[0] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_a_larger_xi_raises_the_bar_for_improvement(self):
        """`xi` is the exploration knob: it discounts small predicted gains."""
        grid = np.linspace(0, 1, 50).reshape(-1, 1)
        greedy = fitted_optimizer(xi=0.0)._expected_improvement(grid)
        cautious = fitted_optimizer(xi=1.0)._expected_improvement(grid)
        assert cautious.sum() < greedy.sum()

    def test_upper_confidence_bound_matches_its_closed_form(self):
        opt = fitted_optimizer(acquisition="ucb", kappa=2.0)
        X = np.array([[0.3], [0.7]])
        mean, std = opt.gp.predict(X)
        np.testing.assert_allclose(
            opt._upper_confidence_bound(X), mean + 2.0 * std, rtol=1e-9
        )

    def test_a_larger_kappa_favours_the_uncertain_regions(self):
        opt_low = fitted_optimizer(acquisition="ucb", kappa=0.0)
        opt_high = fitted_optimizer(acquisition="ucb", kappa=10.0)
        grid = np.linspace(0, 1, 200).reshape(-1, 1)
        assert opt_high._upper_confidence_bound(grid).argmax() != (
            opt_low._upper_confidence_bound(grid).argmax()
        )

    def test_an_empty_history_gives_every_point_the_same_score(self):
        """Nothing is known yet, so the first pick must not be biased by the surrogate."""
        opt = BayesianOptimizer(quadratic, {"x": (0.0, 1.0)})
        grid = np.linspace(0, 1, 10).reshape(-1, 1)
        assert len(set(opt._expected_improvement(grid))) == 1

    @pytest.mark.parametrize("acquisition", ["ei", "ucb", "EI", "UCB"])
    def test_the_acquisition_name_is_case_insensitive(self, acquisition):
        opt = BayesianOptimizer(quadratic, {"x": (0.0, 1.0)}, acquisition=acquisition)
        assert opt.acquisition == acquisition.lower()

    def test_an_unknown_acquisition_is_rejected_at_construction(self):
        """
        Regression test. It was only checked inside the acquisition call, which
        happens after the initial design has run. Each of those points is a full
        training run, so a typo used to cost the whole initial budget first.
        """
        with pytest.raises(ValueError):
            BayesianOptimizer(quadratic, {"x": (0.0, 1.0)}, acquisition="poi")


class TestParameterMapping:
    def test_normalization_maps_the_bounds_to_zero_and_one(self):
        opt = BayesianOptimizer(quadratic, {"x": (-5.0, 5.0), "y": (0.0, 100.0)})
        np.testing.assert_allclose(
            opt._normalize_params(np.array([-5.0, 0.0])), [0.0, 0.0]
        )
        np.testing.assert_allclose(
            opt._normalize_params(np.array([5.0, 100.0])), [1.0, 1.0]
        )

    def test_denormalization_inverts_normalization(self):
        opt = BayesianOptimizer(quadratic, {"a": (-3.0, 7.0), "b": (1e-4, 1e-1)})
        raw = np.array([2.5, 0.03])
        np.testing.assert_allclose(
            opt._denormalize_params(opt._normalize_params(raw)), raw
        )

    def test_the_array_is_labelled_in_search_space_order(self):
        opt = BayesianOptimizer(quadratic, {"lr": (0.0, 1.0), "wd": (0.0, 1.0)})
        assert opt._array_to_dict(np.array([0.25, 0.75])) == {"lr": 0.25, "wd": 0.75}


class TestOptimize:
    def test_the_reported_best_is_the_best_of_the_history(self):
        result = bayesian_optimize(
            quadratic,
            {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            n_iterations=10,
            n_initial=4,
            random_state=0,
            verbose=0,
        )
        assert result.best_score == max(score for _, score in result.history)
        assert (result.best_params, result.best_score) in result.history

    def test_the_search_spends_its_whole_budget(self):
        result = bayesian_optimize(
            quadratic,
            {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            n_iterations=8,
            n_initial=3,
            random_state=0,
            verbose=0,
        )
        assert len(result.history) == 11
        assert result.convergence_data["n_evaluations"] == 11
        assert result.convergence_data["n_failed"] == 0

    def test_every_suggestion_stays_inside_the_search_space(self):
        """
        The acquisition is maximized in normalized coordinates, so a mapping
        error shows up as configurations outside the bounds the caller set.
        """
        result = bayesian_optimize(
            quadratic,
            {"x": (-2.0, 3.0), "y": (10.0, 20.0)},
            n_iterations=10,
            n_initial=3,
            random_state=0,
            verbose=0,
        )
        for params, _ in result.history:
            assert -2.0 <= params["x"] <= 3.0
            assert 10.0 <= params["y"] <= 20.0

    @pytest.mark.parametrize("acquisition", ["ei", "ucb"])
    def test_the_search_beats_an_equal_budget_of_random_draws(self, acquisition):
        """
        The claim the method makes. Averaged over seeds because a single run of
        either can get lucky, and on a smooth surface where a surrogate is
        supposed to help at all.
        """
        bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        n_initial, n_iterations = 5, 10
        guided, blind = [], []
        for seed in range(3):
            result = bayesian_optimize(
                quadratic,
                bounds,
                n_iterations=n_iterations,
                n_initial=n_initial,
                acquisition=acquisition,
                random_state=seed,
                verbose=0,
            )
            guided.append(result.best_score)
            rng = np.random.default_rng(seed)
            draws = rng.uniform(-5, 5, size=(n_initial + n_iterations, 2))
            blind.append(max(quadratic({"x": x, "y": y}) for x, y in draws))
        assert np.mean(guided) > np.mean(blind)

    def test_the_improvement_is_measured_against_the_initial_random_design(self):
        result = bayesian_optimize(
            quadratic,
            {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            n_iterations=10,
            n_initial=5,
            random_state=0,
            verbose=0,
        )
        initial_mean = np.mean([s for _, s in result.history[:5]])
        assert result.convergence_data["improvement_over_random"] == pytest.approx(
            result.best_score - initial_mean
        )

    def test_a_failing_configuration_is_skipped_rather_than_recorded(self):
        """
        A configuration that cannot be trained carries no information about the
        objective. Feeding a placeholder score to the GP would teach it a
        cliff that is not there.
        """

        def unstable(params):
            if params["x"] > 0.0:
                raise RuntimeError("simulated divergence")
            return -(params["x"] ** 2)

        with pytest.warns(UserWarning):
            result = bayesian_optimize(
                unstable,
                {"x": (-5.0, 5.0)},
                n_iterations=10,
                n_initial=6,
                random_state=0,
                verbose=0,
            )
        assert all(params["x"] <= 0.0 for params, _ in result.history)
        assert result.convergence_data["n_failed"] > 0
        assert len(result.history) + result.convergence_data["n_failed"] == 16

    def test_a_non_finite_score_is_treated_as_a_failure(self):
        result = bayesian_optimize(
            lambda p: np.inf if p["x"] > 0 else p["x"],
            {"x": (-5.0, 5.0)},
            n_iterations=6,
            n_initial=6,
            random_state=0,
            verbose=0,
        )
        assert all(np.isfinite(score) for _, score in result.history)

    def test_an_objective_that_always_fails_is_an_error_not_an_empty_result(self):
        """
        Returning a result with no best parameters would push the failure into
        whatever code unpacks it, far from the cause.
        """

        def always_fails(params):
            raise RuntimeError("nothing works")

        with pytest.warns(UserWarning), pytest.raises(RuntimeError):
            bayesian_optimize(
                always_fails,
                {"x": (0.0, 1.0)},
                n_iterations=3,
                n_initial=3,
                random_state=0,
                verbose=0,
            )

    def test_the_random_state_makes_a_run_reproducible(self):
        def run():
            return bayesian_optimize(
                quadratic,
                {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
                n_iterations=8,
                n_initial=3,
                random_state=11,
                verbose=0,
            )

        first, second = run(), run()
        assert first.best_score == second.best_score
        assert first.convergence_data["scores"] == second.convergence_data["scores"]

    def test_verbose_zero_prints_nothing(self, capsys):
        bayesian_optimize(
            quadratic,
            {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            n_iterations=3,
            n_initial=2,
            random_state=0,
            verbose=0,
        )
        assert capsys.readouterr().out == ""

    def test_higher_verbosity_reports_more(self, capsys):
        printed = {}
        for level in (1, 2):
            bayesian_optimize(
                quadratic,
                {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
                n_iterations=5,
                n_initial=2,
                random_state=0,
                verbose=level,
            )
            printed[level] = capsys.readouterr().out
        assert len(printed[2]) > len(printed[1]) > 0
