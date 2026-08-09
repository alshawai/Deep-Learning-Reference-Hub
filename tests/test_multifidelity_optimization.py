"""
Multi-Fidelity Optimization Tests
=================================

ASHA earns its keep through one behaviour: many configurations start cheap and
only the promising ones get expensive. A run that never promotes still returns a
best configuration, still reports a plausible score, and has quietly degenerated
into random search at the lowest fidelity. The rung-geometry tests below exist to
catch exactly that.

The evaluator used throughout is a pure function of the configuration and the
fidelity, so which configuration deserves promotion is known before the run
starts.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from collections import Counter

import numpy as np
import pytest
from conftest import load

mf = load("hyperparameter_tuning/multifidelity_optimization.py")
ASHAOptimizer = mf.ASHAOptimizer
FidelityEvaluator = mf.FidelityEvaluator
FunctionEvaluator = mf.FunctionEvaluator
CandidateResult = mf.CandidateResult
MultiFidelityResult = mf.MultiFidelityResult
asha_optimize = mf.asha_optimize
analyze_fidelity_correlation = mf.analyze_fidelity_correlation


def ranked_eval(hyperparams, fidelity):
    """
    Score rises with the configuration's own quality and with the budget spent.

    Deterministic on purpose: the ordering of configurations is fixed, so a test
    can name which ones should survive to the top rung.
    """
    return hyperparams["q"] + 0.01 * fidelity, {"fidelity": fidelity}


def configs(n):
    return [{"q": float(i)} for i in range(n)]


def run(n_configs=27, **kwargs):
    options = {
        "min_fidelity": 1,
        "max_fidelity": 81,
        "reduction_factor": 3,
        "max_iterations": 300,
        "max_concurrent": 1,
        "verbose": False,
    }
    options.update(kwargs)
    return asha_optimize(ranked_eval, configs(n_configs), **options)


class TestRungLadder:
    def test_the_budgets_climb_by_the_reduction_factor(self):
        opt = ASHAOptimizer(
            FunctionEvaluator(ranked_eval, 1, 81),
            reduction_factor=3,
            min_budget=1,
            max_budget=81,
        )
        assert [r["budget"] for r in opt.rungs] == [1, 3, 9, 27, 81]

    @pytest.mark.parametrize("factor", [2, 3, 4])
    def test_the_promotion_threshold_follows_the_reduction_factor(self, factor):
        """
        Regression test. A rung promotes its top 1/factor, so it needs at least
        `factor` results before that fraction means anything. The threshold was
        derived from the rung count instead, so a taller ladder demanded more
        results at the bottom rung and the run stalled there -- 27 configurations
        on a five-rung ladder never got past the second rung, and every reported
        best came from the cheapest fidelity.
        """
        opt = ASHAOptimizer(
            FunctionEvaluator(ranked_eval, 1, 64),
            reduction_factor=factor,
            min_budget=1,
            max_budget=64,
        )
        assert [r["n_required"] for r in opt.rungs[:-1]] == [factor] * (
            len(opt.rungs) - 1
        )

    def test_the_top_rung_promotes_nowhere(self):
        opt = ASHAOptimizer(
            FunctionEvaluator(ranked_eval, 1, 81),
            reduction_factor=3,
            min_budget=1,
            max_budget=81,
        )
        assert opt.rungs[-1]["n_required"] == 0

    def test_a_budget_range_outside_the_evaluator_is_rejected(self):
        with pytest.raises(ValueError):
            ASHAOptimizer(
                FunctionEvaluator(ranked_eval, 4, 40), min_budget=1, max_budget=81
            )

    def test_a_budget_maps_back_to_its_rung(self):
        opt = ASHAOptimizer(
            FunctionEvaluator(ranked_eval, 1, 81),
            reduction_factor=3,
            min_budget=1,
            max_budget=81,
        )
        assert opt._get_rung_for_budget(9) == 2
        assert opt._get_rung_for_budget(10) is None


class TestPromotion:
    def test_the_population_thins_by_the_reduction_factor_at_each_rung(self):
        """
        The whole point of ASHA. 27 configurations at the cheapest fidelity should
        become 9, then 3, then 1 -- not 27 evaluations and a stop.
        """
        result = run(n_configs=27)
        per_fidelity = Counter(r.fidelity for r in result.all_results)
        assert per_fidelity[1] == 27
        assert per_fidelity[3] == 9
        assert per_fidelity[9] == 3
        assert per_fidelity[27] == 1

    def test_the_survivors_are_the_high_scorers(self):
        """
        Promotion has to be by score. Promoting arbitrarily costs the same budget
        and produces the same shaped result, so only the identity of the
        survivors distinguishes a working ladder from a broken one.
        """
        result = run(n_configs=27)
        top_rung = [r for r in result.all_results if r.fidelity == 9]
        assert {r.hyperparams["q"] for r in top_rung} == {26.0, 25.0, 24.0}

    def test_a_configuration_is_promoted_out_of_a_rung_only_once(self):
        result = run(n_configs=27)
        seen = Counter((r.config_id, r.fidelity) for r in result.all_results)
        assert max(seen.values()) == 1

    def test_a_promotion_carries_the_original_hyperparameters_upward(self):
        result = run(n_configs=27)
        by_id = {}
        for record in result.all_results:
            by_id.setdefault(record.config_id, []).append(record.hyperparams)
        for history in by_id.values():
            assert all(h == history[0] for h in history)

    def test_climbing_the_ladder_costs_less_than_evaluating_everything_fully(self):
        """
        The economic claim behind multi-fidelity search, stated as a number: 27
        full evaluations would be 27 * 81 units of budget.
        """
        result = run(n_configs=27)
        assert result.total_budget_used < 27 * 81 / 10

    def test_the_reported_best_comes_from_the_highest_fidelity_reached(self):
        result = run(n_configs=27)
        assert result.best_fidelity == max(r.fidelity for r in result.all_results)
        assert result.best_score == max(r.score for r in result.all_results)


class TestBudgetAndLimits:
    def test_max_iterations_caps_the_number_of_evaluations(self):
        result = run(n_configs=27, max_iterations=10)
        assert len(result.all_results) <= 10

    def test_the_run_stops_when_the_ladder_is_exhausted(self):
        """With three configurations there is one promotion available, then none."""
        result = run(n_configs=3, max_iterations=300)
        assert len(result.all_results) == 4

    def test_the_recorded_budget_is_the_sum_of_the_fidelities_evaluated(self):
        result = run(n_configs=9)
        assert result.total_budget_used == sum(r.fidelity for r in result.all_results)

    def test_concurrency_does_not_change_which_configurations_win(self):
        """
        The promotion bookkeeping is shared mutable state behind a lock. Four
        workers claiming promotions must reach the same survivors as one.
        """
        serial = run(n_configs=27, max_concurrent=1)
        parallel = run(n_configs=27, max_concurrent=4)
        assert parallel.best_config == serial.best_config
        assert Counter(r.fidelity for r in parallel.all_results) == Counter(
            r.fidelity for r in serial.all_results
        )


class TestFailureHandling:
    def test_a_raising_evaluation_is_recorded_as_the_worst_possible_score(self):
        def explodes(hyperparams, fidelity):
            if hyperparams["q"] == 0.0:
                raise RuntimeError("simulated training failure")
            return ranked_eval(hyperparams, fidelity)

        with pytest.warns(UserWarning, match="Evaluation failed"):
            result = asha_optimize(
                explodes,
                configs(9),
                min_fidelity=1,
                max_fidelity=9,
                reduction_factor=3,
                max_iterations=100,
                max_concurrent=1,
                verbose=False,
            )
        failed = [r for r in result.all_results if r.hyperparams["q"] == 0.0]
        assert failed and all(r.score == -np.inf for r in failed)
        assert "error" in failed[0].metadata

    def test_a_failing_configuration_is_never_promoted(self):
        def explodes(hyperparams, fidelity):
            if hyperparams["q"] > 6.0:  # the would-be winners all fail
                raise RuntimeError("simulated training failure")
            return ranked_eval(hyperparams, fidelity)

        with pytest.warns(UserWarning):
            result = asha_optimize(
                explodes,
                configs(9),
                min_fidelity=1,
                max_fidelity=9,
                reduction_factor=3,
                max_iterations=100,
                max_concurrent=1,
                verbose=False,
            )
        promoted = [r for r in result.all_results if r.fidelity > 1]
        assert promoted
        assert all(r.hyperparams["q"] <= 6.0 for r in promoted)

    def test_a_non_finite_score_is_normalized_to_negative_infinity(self):
        result = asha_optimize(
            lambda hp, fid: (np.nan if hp["q"] == 0.0 else hp["q"], {}),
            configs(6),
            min_fidelity=1,
            max_fidelity=9,
            reduction_factor=3,
            max_iterations=50,
            max_concurrent=1,
            verbose=False,
        )
        scores = [r.score for r in result.all_results if r.hyperparams["q"] == 0.0]
        assert scores and all(s == -np.inf for s in scores)

    def test_failures_are_counted_apart_from_successes(self):
        result = asha_optimize(
            lambda hp, fid: (np.inf if hp["q"] < 2.0 else hp["q"], {}),
            configs(6),
            min_fidelity=1,
            max_fidelity=9,
            reduction_factor=3,
            max_iterations=50,
            max_concurrent=1,
            verbose=False,
        )
        stats = result.statistics
        assert stats["failed_evaluations"] == 2
        assert (
            stats["successful_evaluations"] + stats["failed_evaluations"]
            == stats["total_evaluations"]
        )


class TestStatistics:
    def test_the_per_fidelity_breakdown_matches_the_results(self):
        result = run(n_configs=27)
        analysis = result.statistics["fidelity_analysis"]
        for fidelity, summary in analysis.items():
            scores = [r.score for r in result.all_results if r.fidelity == fidelity]
            assert summary["n_evaluations"] == len(scores)
            assert summary["max_score"] == pytest.approx(max(scores))
            assert summary["mean_score"] == pytest.approx(np.mean(scores))

    def test_rungs_used_counts_the_rungs_that_saw_a_result(self):
        result = run(n_configs=27)
        assert result.statistics["rungs_used"] == len(
            {r.fidelity for r in result.all_results}
        )

    def test_a_run_with_no_results_reports_no_statistics(self):
        opt = ASHAOptimizer(
            FunctionEvaluator(ranked_eval, 1, 9), min_budget=1, max_budget=9
        )
        assert opt._compute_statistics() == {}


class TestFidelityCorrelation:
    def test_a_fidelity_agnostic_objective_correlates_perfectly_across_rungs(self):
        """
        Score is the configuration's quality alone, so the cheap rung ranks the
        configurations exactly as the expensive one does. That is the assumption
        multi-fidelity search rests on, and this is what it looks like when it
        holds.
        """
        result = asha_optimize(
            lambda hp, fid: (hp["q"], {}),
            configs(27),
            min_fidelity=1,
            max_fidelity=81,
            reduction_factor=3,
            max_iterations=300,
            max_concurrent=1,
            verbose=False,
        )
        correlations = analyze_fidelity_correlation(result)
        assert correlations
        assert all(c == pytest.approx(1.0) for c in correlations.values())

    def test_fewer_than_three_shared_configurations_yields_no_correlation(self):
        result = run(n_configs=3)
        assert analyze_fidelity_correlation(result) == {}


class TestEvaluatorInterface:
    def test_the_abstract_evaluator_refuses_to_be_instantiated(self):
        with pytest.raises(TypeError):
            FidelityEvaluator()

    def test_the_function_evaluator_forwards_both_arguments(self):
        seen = []
        evaluator = FunctionEvaluator(
            lambda hp, fid: (seen.append((hp, fid)) or 1.0, {"ok": True}), 2, 20
        )
        score, metadata = evaluator.evaluate({"q": 1.0}, 5)
        assert score == 1.0
        assert metadata == {"ok": True}
        assert seen == [({"q": 1.0}, 5)]
        assert evaluator.get_fidelity_range() == (2, 20)


def test_verbose_false_prints_nothing(capsys):
    run(n_configs=9, verbose=False)
    assert capsys.readouterr().out == ""


def test_the_result_is_the_documented_dataclass():
    result = run(n_configs=9)
    assert isinstance(result, MultiFidelityResult)
    assert isinstance(result.all_results[0], CandidateResult)
    assert result.best_config in [r.hyperparams for r in result.all_results]
    assert result.total_time > 0
