"""
Learning Rate Scheduler Tests
=============================

Twelve strategies behind one `step()` call. Every one has a closed form, so the
tests assert the formula rather than the shape of the curve: a scheduler that
decays at the wrong rate still produces a plausible descending list.

Two properties are checked for every strategy at once, at the bottom of the file,
because they are the ones a new strategy is most likely to break: the learning
rate stays non-negative, and `history` stays consistent with `current_lr`.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import math

import pytest
from conftest import load

lrs = load("learning_rate_schedulers.py")
LearningRateScheduler = lrs.LearningRateScheduler
SchedulerType = lrs.SchedulerType


def test_constant_never_moves():
    sched = LearningRateScheduler(0.1, SchedulerType.CONSTANT)
    assert [sched.step() for _ in range(5)] == [0.1] * 5


class TestDecaySchedules:
    """Each of these is a one-line formula in the step count."""

    def test_step_decay_multiplies_by_gamma_every_step_size_steps(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.STEP_DECAY, step_size=3, gamma=0.5
        )
        got = [sched.step() for _ in range(9)]
        expected = [1.0 * 0.5 ** (t // 3) for t in range(1, 10)]
        assert got == pytest.approx(expected)

    def test_step_decay_defaults_gamma_to_a_tenth(self):
        sched = LearningRateScheduler(1.0, SchedulerType.STEP_DECAY, step_size=2)
        assert sched.kwargs["gamma"] == 0.1

    def test_exponential_decay_is_gamma_to_the_step_count(self):
        sched = LearningRateScheduler(1.0, SchedulerType.EXPONENTIAL_DECAY, gamma=0.9)
        got = [sched.step() for _ in range(5)]
        assert got == pytest.approx([0.9**t for t in range(1, 6)])

    def test_polynomial_decay_reaches_zero_at_total_steps(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.POLYNOMIAL_DECAY, total_steps=10, power=2.0
        )
        got = [sched.step() for _ in range(10)]
        assert got[:-1] == pytest.approx([(1 - t / 10) ** 2 for t in range(1, 10)])
        assert got[-1] == 0.0

    def test_polynomial_decay_stays_at_zero_past_total_steps(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.POLYNOMIAL_DECAY, total_steps=3, power=1.0
        )
        for _ in range(6):
            sched.step()
        assert sched.get_lr() == 0.0


class TestCosineSchedules:
    def test_cosine_annealing_runs_from_initial_lr_down_to_eta_min(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.COSINE_ANNEALING, T_max=10, eta_min=0.01
        )
        got = [sched.step() for _ in range(10)]
        expected = [
            0.01 + (1.0 - 0.01) * (1 + math.cos(math.pi * t / 10)) / 2
            for t in range(1, 11)
        ]
        assert got == pytest.approx(expected)
        assert got[-1] == pytest.approx(0.01)

    def test_cosine_annealing_falls_back_to_total_steps_for_t_max(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.COSINE_ANNEALING, total_steps=10
        )
        assert [sched.step() for _ in range(10)][-1] == pytest.approx(0.0)

    def test_cosine_annealing_needs_a_period_from_somewhere(self):
        with pytest.raises(ValueError):
            LearningRateScheduler(1.0, SchedulerType.COSINE_ANNEALING)

    def test_warm_restarts_jump_back_up_at_each_cycle_boundary(self):
        """The defining behaviour of SGDR: the rate recovers instead of decaying to zero."""
        sched = LearningRateScheduler(
            1.0,
            SchedulerType.COSINE_ANNEALING_WARM_RESTARTS,
            T_0=10,
            T_mult=2,
            eta_min=0.0,
        )
        seen = [sched.step() for _ in range(31)]
        assert seen[8] < 0.1, "should be near the floor just before the restart"
        assert seen[9] == pytest.approx(1.0), "step 10 restarts the cycle"
        assert seen[29] == pytest.approx(1.0), "step 30 restarts the second cycle"

    def test_the_restart_count_counts_restarts(self):
        """
        Regression test. The cycle is recomputed from `step_count` on every call,
        so incrementing the counter inside that search loop re-adds the whole
        history each step: 39 steps reported 40 restarts where 2 had happened.
        The value is serialized by `get_state`, so the wrong number persists.
        """
        sched = LearningRateScheduler(
            1.0, SchedulerType.COSINE_ANNEALING_WARM_RESTARTS, T_0=10, T_mult=2
        )
        for _ in range(39):
            sched.step()
        assert sched._restart_count == 2

    def test_the_restart_count_does_not_depend_on_how_it_was_reached(self):
        """It is a function of step_count, so stepping there must be idempotent."""
        one = LearningRateScheduler(
            1.0, SchedulerType.COSINE_ANNEALING_WARM_RESTARTS, T_0=4, T_mult=2
        )
        for _ in range(20):
            one.step()

        two = LearningRateScheduler(
            1.0, SchedulerType.COSINE_ANNEALING_WARM_RESTARTS, T_0=4, T_mult=2
        )
        two.step_count = 19
        two.step()

        assert one._restart_count == two._restart_count


class TestWarmup:
    def test_linear_warmup_ramps_then_holds(self):
        sched = LearningRateScheduler(0.5, SchedulerType.LINEAR_WARMUP, warmup_steps=4)
        got = [sched.step() for _ in range(6)]
        assert got == pytest.approx([0.125, 0.25, 0.375, 0.5, 0.5, 0.5])

    def test_linear_warmup_requires_warmup_steps(self):
        with pytest.raises(ValueError):
            LearningRateScheduler(0.5, SchedulerType.LINEAR_WARMUP)

    def test_warmup_cosine_peaks_at_the_end_of_warmup(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.WARMUP_COSINE, total_steps=20, warmup_steps=5
        )
        seen = [sched.step() for _ in range(20)]
        assert seen[4] == pytest.approx(1.0), "warmup ends at the full rate"
        assert seen.index(max(seen)) == 4
        assert seen[-1] == pytest.approx(0.0, abs=1e-12)

    def test_warmup_cosine_requires_total_steps(self):
        with pytest.raises(ValueError):
            LearningRateScheduler(1.0, SchedulerType.WARMUP_COSINE, warmup_steps=5)


class TestCyclical:
    def test_triangular_rises_to_max_lr_and_returns_to_base_lr(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.CYCLICAL, base_lr=0.1, max_lr=1.0, step_size_up=5
        )
        seen = [sched.step() for _ in range(10)]
        assert seen[4] == pytest.approx(1.0), "peak at the top of the ramp"
        assert seen[9] == pytest.approx(0.1), "back to base after the full cycle"

    def test_triangular2_halves_the_amplitude_each_cycle(self):
        sched = LearningRateScheduler(
            1.0,
            SchedulerType.CYCLICAL,
            base_lr=0.0,
            max_lr=1.0,
            step_size_up=5,
            mode="triangular2",
        )
        seen = [sched.step() for _ in range(20)]
        assert seen[4] == pytest.approx(1.0)
        assert seen[14] == pytest.approx(0.5)

    def test_the_cycle_count_tracks_the_current_cycle(self):
        """
        Regression test. `_cycle_count` is serialized by `get_state` and was never
        assigned, so it read 0 for the whole run while the cycle index was already
        being computed one line away.
        """
        sched = LearningRateScheduler(
            1.0, SchedulerType.CYCLICAL, base_lr=0.0, max_lr=1.0, step_size_up=5
        )
        for _ in range(4):
            sched.step()
        assert sched._cycle_count == 1
        for _ in range(10):
            sched.step()
        assert sched._cycle_count == 2

    def test_an_unknown_mode_is_rejected(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.CYCLICAL, step_size_up=2, mode="spiral"
        )
        with pytest.raises(ValueError):
            sched.step()


class TestOneCycle:
    def test_the_rate_peaks_at_pct_start_and_comes_back_down(self):
        sched = LearningRateScheduler(
            0.1, SchedulerType.ONE_CYCLE, total_steps=100, max_lr=1.0, pct_start=0.3
        )
        seen = [sched.step() for _ in range(100)]
        assert seen.index(max(seen)) == pytest.approx(29, abs=1)
        assert max(seen) == pytest.approx(1.0, rel=1e-3)
        assert seen[-1] < 0.2

    def test_one_cycle_requires_total_steps(self):
        with pytest.raises(ValueError):
            LearningRateScheduler(0.1, SchedulerType.ONE_CYCLE, max_lr=1.0)


class TestReduceOnPlateau:
    def test_a_plateau_longer_than_patience_reduces_the_rate(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.REDUCE_ON_PLATEAU, patience=2, factor=0.5
        )
        seen = [sched.step(1.0) for _ in range(6)]
        assert seen[0] == 1.0, "the first call only records the baseline"
        assert seen[-1] < 1.0

    def test_a_still_improving_metric_is_never_reduced(self):
        sched = LearningRateScheduler(
            1.0, SchedulerType.REDUCE_ON_PLATEAU, patience=1, factor=0.5
        )
        for metric in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            sched.step(metric)
        assert sched.get_lr() == 1.0

    def test_the_rate_never_falls_below_min_lr(self):
        sched = LearningRateScheduler(
            1.0,
            SchedulerType.REDUCE_ON_PLATEAU,
            patience=0,
            factor=0.1,
            min_lr=0.05,
        )
        for _ in range(30):
            sched.step(1.0)
        assert sched.get_lr() == pytest.approx(0.05)

    def test_cooldown_holds_the_rate_after_a_reduction(self):
        """
        Regression test. `cooldown` was read into a local and never used, so a
        long plateau reduced the rate every `patience + 1` steps regardless of it.
        A reduction needs time to show up in the metric; cooling down is how the
        scheduler avoids collapsing the rate while it waits.
        """
        without = LearningRateScheduler(
            1.0, SchedulerType.REDUCE_ON_PLATEAU, patience=1, factor=0.5, cooldown=0
        )
        with_cooldown = LearningRateScheduler(
            1.0, SchedulerType.REDUCE_ON_PLATEAU, patience=1, factor=0.5, cooldown=5
        )
        for _ in range(12):
            without.step(1.0)
            with_cooldown.step(1.0)
        assert with_cooldown.get_lr() > without.get_lr()

    def test_mode_max_treats_a_rising_metric_as_improvement(self):
        sched = LearningRateScheduler(
            1.0,
            SchedulerType.REDUCE_ON_PLATEAU,
            patience=1,
            factor=0.5,
            mode="max",
        )
        for metric in (0.1, 0.2, 0.3, 0.4, 0.5):
            sched.step(metric)
        assert sched.get_lr() == 1.0

    def test_a_missing_metric_warns_and_holds(self):
        sched = LearningRateScheduler(1.0, SchedulerType.REDUCE_ON_PLATEAU)
        with pytest.warns(UserWarning):
            assert sched.step() == 1.0


class TestCustom:
    def test_the_custom_function_drives_the_rate(self):
        sched = LearningRateScheduler(
            0.5,
            SchedulerType.CUSTOM,
            custom_func=lambda step, initial_lr, **kw: initial_lr / step,
        )
        assert [sched.step() for _ in range(3)] == pytest.approx([0.5, 0.25, 0.5 / 3])

    def test_custom_requires_a_function(self):
        with pytest.raises(ValueError):
            LearningRateScheduler(0.5, SchedulerType.CUSTOM)


def test_a_non_positive_initial_rate_is_rejected():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            LearningRateScheduler(bad)


def scheduler_of(kind):
    """One valid instance per strategy, for the cross-strategy invariants below."""
    common = {
        SchedulerType.CONSTANT: {},
        SchedulerType.STEP_DECAY: {"step_size": 3, "gamma": 0.5},
        SchedulerType.EXPONENTIAL_DECAY: {"gamma": 0.9},
        SchedulerType.POLYNOMIAL_DECAY: {"total_steps": 20},
        SchedulerType.COSINE_ANNEALING: {"T_max": 20},
        SchedulerType.COSINE_ANNEALING_WARM_RESTARTS: {"T_0": 5},
        SchedulerType.CYCLICAL: {"step_size_up": 4},
        SchedulerType.ONE_CYCLE: {"total_steps": 20},
        SchedulerType.REDUCE_ON_PLATEAU: {"patience": 1},
        SchedulerType.WARMUP_COSINE: {"total_steps": 20, "warmup_steps": 5},
        SchedulerType.LINEAR_WARMUP: {"warmup_steps": 5},
        SchedulerType.CUSTOM: {"custom_func": lambda s, lr, **kw: lr / s},
    }[kind]
    return LearningRateScheduler(0.1, kind, **common)


@pytest.mark.parametrize("kind", list(SchedulerType), ids=lambda k: k.value)
class TestEveryStrategy:
    """
    Invariants that hold whatever the curve does. A new strategy added to the
    enum is covered by these without anyone writing a new test, which is the
    point of parametrizing over `SchedulerType` itself.
    """

    def test_the_rate_never_goes_negative(self, kind):
        sched = scheduler_of(kind)
        for _ in range(25):
            assert sched.step(metric=1.0) >= 0.0

    def test_history_tracks_every_step_and_ends_at_the_current_rate(self, kind):
        sched = scheduler_of(kind)
        for _ in range(10):
            sched.step(metric=1.0)
        assert len(sched.history) == 11, "history starts with the initial rate"
        assert sched.history[-1] == sched.get_lr()

    def test_reset_returns_to_the_initial_rate(self, kind):
        sched = scheduler_of(kind)
        for _ in range(10):
            sched.step(metric=1.0)
        sched.reset()
        assert sched.get_lr() == 0.1
        assert sched.step_count == 0
        assert sched.history == [0.1]

    def test_state_round_trips(self, kind):
        sched = scheduler_of(kind)
        for _ in range(7):
            sched.step(metric=1.0)

        resumed = scheduler_of(kind)
        resumed.load_state(sched.get_state())
        assert resumed.get_lr() == sched.get_lr()
        assert resumed.step_count == sched.step_count
        assert resumed.step(metric=1.0) == sched.step(metric=1.0)
