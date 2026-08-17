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

Concurrency gets its own class, because the interesting property of ASHA is one
it deliberately gives up. Promotion happens on partial information, so the
promotion set depends on arrival order and two worker counts need not agree.
`TestConcurrency` asserts what does survive extra workers and documents what
does not.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import threading
import time
from collections import Counter

import numpy as np
import pytest

from dlhub.tuning.multifidelity import (
    ASHAOptimizer,
    CandidateResult,
    FidelityEvaluator,
    FunctionEvaluator,
    MultiFidelityResult,
    analyze_fidelity_correlation,
    asha_optimize,
)

# The rung budgets for min_fidelity=1, max_fidelity=81, reduction_factor=3.
LADDER = [1, 3, 9, 27, 81]


def ranked_eval(hyperparams, fidelity):
    """
    Score rises with the configuration's own quality and with the budget spent.

    Deterministic on purpose: the ordering of configurations is fixed, so a test
    can name which ones should survive to the top rung.

    The quality term also dominates the fidelity term -- the whole ladder is
    worth 0.01 * 81 = 0.81, less than the gap of 1.0 between adjacent `q`. So no
    amount of extra budget lets one configuration overtake a better one, and the
    winner of a run is known without knowing how far up the ladder it climbed.
    """
    return hyperparams["q"] + 0.01 * fidelity, {"fidelity": fidelity}


def configs(n):
    return [{"q": float(i)} for i in range(n)]


def run(n_configs=27, eval_function=ranked_eval, **kwargs):
    options = {
        "min_fidelity": 1,
        "max_fidelity": 81,
        "reduction_factor": 3,
        "max_iterations": 300,
        "max_concurrent": 1,
        "verbose": False,
    }
    options.update(kwargs)
    return asha_optimize(eval_function, configs(n_configs), **options)


class HoldsBackTheStrongest:
    """
    Evaluator that forces the arrival order concurrency makes possible.

    An evaluator that returns instantly arrives in submission order, so a
    parallel run lands on the serial answer every time and the asynchronous path
    goes untested -- 200 runs of the old test in isolation produced the serial
    result 200 times, while a full-suite run failed roughly one time in ten.
    Reproducing the skew deliberately is the only way to assert anything about
    it: the strongest configurations are withheld at the cheapest fidelity until
    `release_after_promotions` promotions have been claimed without them, which
    is what a loaded machine does by accident.

    Needs at least two workers, one to hold and one to claim promotions. Given
    one worker the held evaluations wait out `timeout` instead, which leaves the
    run correct and merely slow.

    Parameters
    ----------
    hold_at_or_above : float
        Configurations whose `q` is at least this are held at the cheapest
        fidelity.
    release_after_promotions : int
        How many promotions to let through before releasing them.
    timeout : float, default=2.0
        Upper bound on a single hold, so that a change to the scheduler cannot
        hang the suite waiting for a promotion that never comes.
    """

    def __init__(self, hold_at_or_above, release_after_promotions, timeout=2.0):
        self.hold_at_or_above = hold_at_or_above
        self.release_after_promotions = release_after_promotions
        self.timeout = timeout
        self._released = threading.Event()
        self._lock = threading.Lock()
        self._promotions = 0

    def __call__(self, hyperparams, fidelity):
        if fidelity > LADDER[0]:  # anything above the cheapest rung is a promotion
            with self._lock:
                self._promotions += 1
                reached = self._promotions >= self.release_after_promotions
            if reached:
                self._released.set()
        elif hyperparams["q"] >= self.hold_at_or_above:
            self._released.wait(timeout=self.timeout)
        return ranked_eval(hyperparams, fidelity)


def promoted_out_of_the_cheapest_rung(result):
    """The `q` of every configuration that reached the second rung, best first."""
    return sorted(
        (r.hyperparams["q"] for r in result.all_results if r.fidelity == LADDER[1]),
        reverse=True,
    )


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

    @pytest.mark.parametrize("max_concurrent", [1, 2, 4, 8])
    def test_max_iterations_means_the_same_at_every_worker_count(self, max_concurrent):
        """
        Regression test (issue #11). `max_iterations` is counted at submission,
        and the pool computes every submission whether or not the loop is still
        watching -- `ThreadPoolExecutor.__exit__` joins its workers. So the
        evaluations in flight when the limit tripped used to be paid for and then
        dropped, leaving `max_iterations` a cap on evaluations *performed* when
        serial and on evaluations *recorded* when parallel, short by up to
        `max_concurrent - 1`.
        """
        counted = []
        lock = threading.Lock()

        def counting_eval(hyperparams, fidelity):
            with lock:
                counted.append(fidelity)
            return ranked_eval(hyperparams, fidelity)

        result = run(
            n_configs=27,
            eval_function=counting_eval,
            max_iterations=10,
            max_concurrent=max_concurrent,
        )

        assert len(counted) == 10
        assert len(result.all_results) == len(counted)

    def test_the_run_stops_when_the_ladder_is_exhausted(self):
        """With three configurations there is one promotion available, then none."""
        result = run(n_configs=3, max_iterations=300)
        assert len(result.all_results) == 4

    def test_the_recorded_budget_is_the_sum_of_the_fidelities_evaluated(self):
        result = run(n_configs=9)
        assert result.total_budget_used == sum(r.fidelity for r in result.all_results)

    def test_the_budget_accounts_for_every_evaluation_the_run_paid_for(self):
        """
        The module's headline claim. `budget_efficiency` and the ladder-is-cheaper
        test both rest on `total_budget_used` being what the run actually spent,
        so an evaluation computed by the pool and dropped by the scheduler makes
        the efficiency number flattering by exactly the fidelity it cost.
        """
        spent = []
        lock = threading.Lock()

        def counting_eval(hyperparams, fidelity):
            with lock:
                spent.append(fidelity)
            return ranked_eval(hyperparams, fidelity)

        result = run(
            n_configs=27,
            eval_function=counting_eval,
            max_iterations=10,
            max_concurrent=4,
        )

        assert result.total_budget_used == sum(spent)

    def test_a_timeout_still_records_the_evaluations_already_running(self):
        """
        The timeout stops new submissions; it cannot un-spend an evaluation the
        pool is already running.

        The slow evaluation has to outlast the scheduler's one-second poll, or
        `as_completed` hands it back inside the loop and the drain is never
        reached -- at half a second this test passes against the unfixed
        scheduler.
        """

        def slow_on_the_strongest(hyperparams, fidelity):
            if hyperparams["q"] == 0.0:
                time.sleep(1.3)
            return ranked_eval(hyperparams, fidelity)

        result = run(
            n_configs=3,
            eval_function=slow_on_the_strongest,
            max_fidelity=9,
            max_concurrent=2,
            timeout=0.1,
        )

        assert 0.0 in {r.hyperparams["q"] for r in result.all_results}
        assert result.total_budget_used == sum(r.fidelity for r in result.all_results)


class TestARunThatRecordsNothing:
    """
    The empty run (issue #12).

    `best_result` starts as None and only `_add_result` assigns it, so a run that
    records nothing used to dereference None and raise `AttributeError` naming
    `NoneType` -- from the return statement, or one line earlier from the verbose
    summary. Three inputs reach it, and they do not all mean the same thing:

    No candidates is caller error. An empty list is more often a search space
    that filtered down to nothing than a deliberate no-op, so it is reported as a
    `ValueError` naming the argument.

    No budget is a limit doing its job. `max_iterations=0` grants none, and a
    `timeout` already elapsed stops the first submission; both are legitimate and
    return the dataclass with the three `best_*` fields None.
    """

    def test_no_candidates_to_search_is_rejected(self):
        with pytest.raises(ValueError, match="initial_configurations is empty"):
            run(n_configs=0)

    def test_the_rejection_names_the_argument_the_caller_passed(self):
        """
        The point of the change. `AttributeError: 'NoneType' object has no
        attribute 'hyperparams'` names an internal that no caller passed; the
        message has to name `initial_configurations` to be actionable.
        """
        with pytest.raises(ValueError) as excinfo:
            run(n_configs=0)
        assert "initial_configurations" in str(excinfo.value)

    @pytest.mark.parametrize(
        "label, options",
        [
            ("no budget", {"max_iterations": 0}),
            ("timeout already elapsed", {"max_iterations": 10, "timeout": 1e-9}),
        ],
    )
    def test_a_run_granted_no_budget_reports_an_empty_run(self, label, options):
        result = run(n_configs=27, **options)

        assert result.all_results == []
        assert result.best_config is None
        assert result.best_score is None
        assert result.best_fidelity is None
        assert result.total_budget_used == 0
        assert result.statistics == {}

    def test_the_three_best_fields_are_absent_together(self):
        """
        The invariant the optional fields carry: either all three describe a real
        evaluation or all three are None. A half-populated result would let a
        caller read `best_score` as a number while `best_config` is None.
        """
        empty = run(n_configs=27, max_iterations=0)
        populated = run(n_configs=9)

        assert [empty.best_config, empty.best_score, empty.best_fidelity] == [
            None,
            None,
            None,
        ]
        assert all(
            field is not None
            for field in (
                populated.best_config,
                populated.best_score,
                populated.best_fidelity,
            )
        )

    def test_an_empty_run_is_recognizable_without_touching_the_best_fields(self):
        """
        `all_results` being empty and `best_config` being None are the same
        condition, so a caller may branch on whichever reads better.
        """
        result = run(n_configs=27, max_iterations=0)
        assert (result.best_config is None) == (not result.all_results)

    def test_the_verbose_summary_of_an_empty_run_says_so(self, capsys):
        """
        `verbose=True` raised one line before the return statement, for the same
        reason. It has its own guard, so it needs its own test.
        """
        run(n_configs=27, max_iterations=0, verbose=True)
        assert "no best configuration to report" in capsys.readouterr().out


class TestConcurrency:
    """
    What extra workers must preserve, and what they are allowed to change.

    The "A" in ASHA is asynchronous. A rung promotes the top
    1 / reduction_factor of whatever has arrived, without waiting for the rung to
    fill, so the promotion set is a function of arrival order and arrival order
    is a function of thread scheduling. Pinning the set to the serial run's
    asserts something the algorithm does not offer, and the assertion that did
    so failed roughly one full-suite run in ten (issue #4). Exact
    cross-worker reproducibility would be a change to the scheduler -- drain
    each rung before promoting -- and it would cost the asynchrony that makes
    ASHA worth using.

    What concurrency does have to preserve is the bookkeeping: the lock's job,
    the shape of the ladder, and the identity of the winner. Every test here
    holds back the strongest configurations, because a parallel run whose
    evaluator returns instantly reproduces the serial run and proves nothing.
    """

    def test_no_configuration_is_evaluated_twice_at_one_fidelity(self):
        """
        The lock's real job. Two workers claiming the same promotion would spend
        the budget twice and report the same configuration twice at one rung.
        """
        result = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        seen = Counter((r.config_id, r.fidelity) for r in result.all_results)
        assert max(seen.values()) == 1

    def test_no_configuration_is_dropped_before_it_is_judged(self):
        """
        A promotion claimed and then discarded costs a configuration its only
        evaluation, and the run would still look healthy -- one fewer result at
        the cheapest rung is invisible in the reported best.
        """
        result = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        evaluated = {r.config_id for r in result.all_results if r.fidelity == LADDER[0]}
        assert evaluated == set(range(27))

    def test_a_promotion_climbs_the_ladder_one_rung_at_a_time(self):
        """
        Rung geometry. Every fidelity is a rung budget, and a configuration that
        reached rung k was evaluated at every rung below it -- no configuration
        skips a rung to arrive at an expensive fidelity unjudged.
        """
        result = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        reached = {}
        for record in result.all_results:
            reached.setdefault(record.config_id, set()).add(record.fidelity)
        for fidelities in reached.values():
            top = max(LADDER.index(f) for f in fidelities)
            assert sorted(fidelities) == LADDER[: top + 1]

    def test_every_configuration_the_serial_run_promotes_is_still_promoted(self):
        """
        The strong are never lost. A run ends only once no promotion is
        claimable, and by then the cheapest rung holds every result -- so
        whichever configurations the full rung ranks in its top third have all
        been promoted, however the arrivals were ordered.
        """
        serial = run(max_concurrent=1)
        parallel = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        assert set(promoted_out_of_the_cheapest_rung(serial)) <= set(
            promoted_out_of_the_cheapest_rung(parallel)
        )

    def test_the_winner_is_the_one_the_serial_run_finds(self):
        """
        Not luck, and not a weaker claim than the old assertion: `ranked_eval`
        gives the whole ladder less weight than one step of `q`, and every
        configuration is evaluated at the cheapest fidelity, so the best `q`
        wins whatever the scheduling did. Only `best_score` is free to vary,
        with how far the winner climbed.
        """
        serial = run(max_concurrent=1)
        parallel = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        assert parallel.best_config == serial.best_config

    def test_a_partial_rung_may_promote_more_than_the_serial_run(self):
        """
        The behaviour the old assertion forbade, pinned here so that it is not
        quietly forbidden again.

        Seven promotions are claimed while the three strongest configurations
        are still reporting, so the rung is judged on 24 results and promotes
        its top 24 // 3 = 8 -- reaching q=17, which the full rung of 27 ranks
        tenth and would not have promoted. The three then arrive and are
        promoted as the top of 27 // 3 = 9. Ten promotions where the serial run
        makes nine, and both are correct ASHA.

        The overshoot is at the margin, not arbitrary: the promoted set stays a
        contiguous block from the top, so asynchrony costs a little wasted
        budget on near-misses rather than promoting a bad configuration.
        """
        serial = run(max_concurrent=1)
        parallel = run(max_concurrent=4, eval_function=HoldsBackTheStrongest(24.0, 6))
        serial_promoted = promoted_out_of_the_cheapest_rung(serial)
        parallel_promoted = promoted_out_of_the_cheapest_rung(parallel)

        assert serial_promoted == [float(q) for q in range(26, 17, -1)]
        assert len(parallel_promoted) > len(serial_promoted)
        assert parallel_promoted == [
            float(q) for q in range(26, 26 - len(parallel_promoted), -1)
        ]

    def test_an_evaluation_slower_than_the_scheduler_poll_completes(self):
        """
        Regression test. The scheduler collects results with
        `as_completed(..., timeout=1.0)`, meaning it to be a poll interval that
        returns control to the loop so the iteration and time limits get
        re-checked. `as_completed` reports an elapsed timeout by raising, and
        that was not caught, so a run aborted with `TimeoutError` whenever no
        evaluation happened to finish inside a second.

        Which is every run this module is for -- its own notes recommend it when
        "training time is expensive". Every other test here passes only because
        its evaluator returns in microseconds.

        Only the first evaluation needs to outlast the poll: what is under test
        is that the scheduler survives an elapsed poll, not how many it survives.
        """
        slow_calls = []

        def slow_on_the_first_call(hyperparams, fidelity):
            slow_calls.append(fidelity)
            if len(slow_calls) == 1:
                time.sleep(1.2)
            return ranked_eval(hyperparams, fidelity)

        result = run(
            n_configs=3,
            eval_function=slow_on_the_first_call,
            max_fidelity=9,
            max_concurrent=2,
        )
        assert max(r.training_time for r in result.all_results) > 1.0
        # Three at the cheapest rung, then the one promotion the ladder allows.
        assert len(result.all_results) == 4
        assert result.best_fidelity == LADDER[1]


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
