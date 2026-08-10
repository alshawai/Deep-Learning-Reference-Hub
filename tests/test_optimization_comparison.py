"""
Optimization Comparison Harness Tests
=====================================

The three benchmark problems carry gradients derived by hand, and Beale's runs to
nine terms. A slip in one of them produces a harness that still runs, still draws
a descending curve, and quietly ranks the optimizers on the wrong surface. Every
gradient here is differenced against its own loss function.

The rest of the file pins the harness contract: a run records one loss per
iteration, convergence is detected rather than assumed, and divergence ends the
run instead of filling the history with infinities.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.optimizers.comparison import (
    BealeFunction,
    OptimizationAnalytics,
    OptimizationComparison,
    OptimizationProblem,
    OptimizationResult,
    OptimizerType,
    QuadraticBowl,
    RosenbrockFunction,
)

PROBLEMS = [
    QuadraticBowl(),
    QuadraticBowl(a=1.0, b=20.0),
    RosenbrockFunction(),
    BealeFunction(),
]
PROBLEM_IDS = ["bowl", "bowl-ill-conditioned", "rosenbrock", "beale"]

# Points away from the axes and away from each minimum, so no term drops out.
PROBE_POINTS = [
    {"x": 0.7, "y": -1.3},
    {"x": -2.0, "y": 2.0},
    {"x": 1.4, "y": 0.6},
    {"x": 3.0, "y": 0.25},
]


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEM_IDS)
@pytest.mark.parametrize("point", PROBE_POINTS, ids=lambda p: f"x{p['x']}y{p['y']}")
def test_analytic_gradients_match_finite_differences(problem, point):
    """
    Two-sided differencing of each problem's own loss. This is the only check
    that can see a wrong coefficient or a dropped chain-rule factor: the harness
    runs, descends, and ranks optimizers either way.
    """
    eps = 1e-6
    analytic = problem.gradients(point)

    for key in point:
        plus = dict(point)
        plus[key] += eps
        minus = dict(point)
        minus[key] -= eps
        numeric = (problem.loss_function(plus) - problem.loss_function(minus)) / (
            2 * eps
        )
        assert analytic[key] == pytest.approx(numeric, rel=1e-5, abs=1e-6), (
            f"d/d{key} of {problem.name} at {point}"
        )


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEM_IDS)
def test_each_problem_starts_somewhere_and_names_itself(problem):
    start = problem.initial_parameters()
    assert set(start) == {"x", "y"}
    assert np.isfinite(problem.loss_function(start))
    assert problem.name


def test_the_known_minima_are_where_the_literature_says():
    """
    Anchors the problems to their published definitions. Rosenbrock's minimum is
    (a, a^2) and Beale's is (3, 0.5); both are zero there.
    """
    assert QuadraticBowl().loss_function({"x": 0.0, "y": 0.0}) == 0.0
    assert RosenbrockFunction().loss_function({"x": 1.0, "y": 1.0}) == 0.0
    assert BealeFunction().loss_function({"x": 3.0, "y": 0.5}) == pytest.approx(0.0)


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEM_IDS)
def test_the_gradient_vanishes_at_the_minimum(problem):
    minima = {
        "Quadratic Bowl (a=1.0, b=1.0)": {"x": 0.0, "y": 0.0},
        "Quadratic Bowl (a=1.0, b=20.0)": {"x": 0.0, "y": 0.0},
        "Rosenbrock Function (a=1.0, b=100.0)": {"x": 1.0, "y": 1.0},
        "Beale Function": {"x": 3.0, "y": 0.5},
    }
    grads = problem.gradients(minima[problem.name])
    for key, value in grads.items():
        assert value == pytest.approx(0.0, abs=1e-9), f"d/d{key}"


def test_the_abstract_problem_refuses_to_be_used_directly():
    problem = OptimizationProblem("abstract")
    for call in (
        lambda: problem.loss_function({"x": 0.0}),
        lambda: problem.gradients({"x": 0.0}),
        problem.initial_parameters,
    ):
        with pytest.raises(NotImplementedError):
            call()


class TestOptimizerFactory:
    @pytest.mark.parametrize("kind", list(OptimizerType), ids=lambda k: k.value)
    def test_every_enum_member_can_be_built(self, kind):
        harness = OptimizationComparison(verbose=False)
        optimizer = harness.create_optimizer(kind, learning_rate=0.02)
        assert optimizer.learning_rate == 0.02
        assert optimizer.name

    def test_the_names_are_distinct(self):
        """The results dict is keyed by name, so a collision silently drops a run."""
        harness = OptimizationComparison(verbose=False)
        names = [harness.create_optimizer(k).name for k in OptimizerType]
        assert len(set(names)) == len(names)

    def test_an_unknown_optimizer_type_is_rejected(self):
        with pytest.raises(ValueError):
            OptimizationComparison(verbose=False).create_optimizer("adam")


class TestRunOptimization:
    def test_the_loss_history_starts_at_the_initial_point(self):
        harness = OptimizationComparison(max_iterations=50, verbose=False)
        problem = QuadraticBowl()
        result = harness.run_optimization(
            problem, harness.create_optimizer(OptimizerType.SGD, learning_rate=0.1)
        )
        assert result.losses[0] == problem.loss_function(problem.initial_parameters())

    def test_one_loss_and_one_parameter_snapshot_per_iteration(self):
        harness = OptimizationComparison(
            max_iterations=30, tolerance=0.0, verbose=False
        )
        result = harness.run_optimization(
            QuadraticBowl(),
            harness.create_optimizer(OptimizerType.SGD, learning_rate=0.1),
        )
        assert len(result.losses) == len(result.parameters) == 30

    def test_the_parameter_history_is_snapshotted_not_aliased(self):
        """
        A history of references to one mutating dict reads as a model that never
        moved, which would make every trajectory plot a single point.
        """
        harness = OptimizationComparison(
            max_iterations=20, tolerance=0.0, verbose=False
        )
        result = harness.run_optimization(
            RosenbrockFunction(),
            harness.create_optimizer(OptimizerType.ADAM, learning_rate=0.05),
        )
        first, last = result.parameters[0], result.parameters[-1]
        assert first is not last
        assert (first["x"], first["y"]) != (last["x"], last["y"])

    def test_sgd_descends_the_bowl_to_its_minimum(self):
        harness = OptimizationComparison(
            max_iterations=500, tolerance=1e-12, verbose=False
        )
        result = harness.run_optimization(
            QuadraticBowl(),
            harness.create_optimizer(OptimizerType.SGD, learning_rate=0.1),
        )
        assert result.final_loss < result.losses[0]
        assert result.final_loss == pytest.approx(0.0, abs=1e-6)

    def test_a_flat_run_is_reported_as_converged(self):
        """
        Convergence is a loss that stopped moving. A harness that only ever
        reports max_iterations makes the speed ranking meaningless.
        """
        harness = OptimizationComparison(
            max_iterations=1000, tolerance=1e-6, verbose=False
        )
        result = harness.run_optimization(
            QuadraticBowl(),
            harness.create_optimizer(OptimizerType.SGD, learning_rate=0.1),
        )
        assert result.iterations_to_converge < 1000
        assert len(result.losses) == result.iterations_to_converge

    def test_a_divergent_learning_rate_ends_the_run_early(self):
        harness = OptimizationComparison(
            max_iterations=1000, tolerance=0.0, verbose=False
        )
        result = harness.run_optimization(
            QuadraticBowl(),
            harness.create_optimizer(OptimizerType.SGD, learning_rate=5.0),
        )
        assert len(result.losses) < 1000
        assert np.all(np.isfinite(result.losses))

    def test_the_optimizer_is_reset_so_runs_do_not_inherit_momentum(self):
        """
        Reusing one optimizer across problems must not carry velocity between
        them, or the second run starts mid-flight and looks faster than it is.
        """
        harness = OptimizationComparison(
            max_iterations=40, tolerance=0.0, verbose=False
        )
        optimizer = harness.create_optimizer(OptimizerType.MOMENTUM, learning_rate=0.05)
        first = harness.run_optimization(QuadraticBowl(), optimizer)
        second = harness.run_optimization(QuadraticBowl(), optimizer)
        assert first.losses == second.losses

    def test_verbose_false_prints_nothing(self, capsys):
        harness = OptimizationComparison(max_iterations=20, verbose=False)
        harness.run_optimization(
            QuadraticBowl(), harness.create_optimizer(OptimizerType.ADAM)
        )
        assert capsys.readouterr().out == ""


def test_comparing_optimizers_returns_one_result_per_configuration():
    harness = OptimizationComparison(max_iterations=100, verbose=False)
    results = harness.compare_optimizers(
        QuadraticBowl(),
        {
            OptimizerType.SGD: {"learning_rate": 0.1},
            OptimizerType.ADAM: {"learning_rate": 0.1},
        },
    )
    assert len(results) == 2
    for result in results.values():
        assert isinstance(result, OptimizationResult)
        assert result.losses


class TestAnalytics:
    def make_result(self, losses):
        return OptimizationResult(
            optimizer_name="probe",
            losses=list(losses),
            parameters=[{} for _ in losses],
            convergence_time=0.01,
            final_loss=losses[-1],
            iterations_to_converge=len(losses),
        )

    def test_the_metrics_follow_their_definitions(self):
        metrics = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([10.0, 6.0, 4.0, 2.0])
        )
        assert metrics["initial_loss"] == 10.0
        assert metrics["final_loss"] == 2.0
        assert metrics["loss_reduction"] == 8.0
        assert metrics["relative_improvement"] == pytest.approx(80.0)
        assert metrics["convergence_rate"] == pytest.approx(8.0 / 4)

    def test_a_steady_descent_scores_more_stable_than_an_erratic_one(self):
        steady = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([10.0, 8.0, 6.0, 4.0, 2.0])
        )
        erratic = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([10.0, 1.0, 9.0, 1.5, 2.0])
        )
        assert steady["stability_score"] > erratic["stability_score"]

    def test_a_perfectly_linear_descent_is_maximally_stable(self):
        metrics = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([10.0, 8.0, 6.0, 4.0, 2.0])
        )
        assert metrics["stability_score"] == pytest.approx(1.0)

    def test_the_stability_score_never_goes_negative(self):
        metrics = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([1.0, 100.0, 1.0, 100.0, 1.0])
        )
        assert metrics["stability_score"] >= 0.0

    def test_a_single_point_run_does_not_divide_by_zero(self):
        metrics = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([5.0])
        )
        assert metrics["convergence_rate"] == 0.0
        assert metrics["stability_score"] == 0.0

    def test_a_zero_initial_loss_does_not_divide_by_zero(self):
        metrics = OptimizationAnalytics.compute_convergence_metrics(
            self.make_result([0.0, 0.0])
        )
        assert metrics["relative_improvement"] == 0.0

    def test_the_best_final_loss_ranks_first(self):
        rankings = OptimizationAnalytics.rank_optimizers(
            {
                "bowl": {
                    "good": self.make_result([10.0, 0.1]),
                    "bad": self.make_result([10.0, 9.0]),
                }
            }
        )
        assert rankings["bowl"]["good_loss_rank"] == 1
        assert rankings["bowl"]["bad_loss_rank"] == 2

    def test_the_fewest_iterations_ranks_fastest(self):
        rankings = OptimizationAnalytics.rank_optimizers(
            {
                "bowl": {
                    "quick": self.make_result([10.0, 1.0]),
                    "slow": self.make_result([10.0, 5.0, 3.0, 1.0]),
                }
            }
        )
        assert rankings["bowl"]["quick_speed_rank"] == 1
        assert rankings["bowl"]["slow_speed_rank"] == 2

    def test_a_problem_with_no_results_is_skipped(self):
        assert OptimizationAnalytics.rank_optimizers({"empty": {}}) == {}
