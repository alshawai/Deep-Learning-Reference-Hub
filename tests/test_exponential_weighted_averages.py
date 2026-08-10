"""
Exponential Weighted Average Tests
============================

The module's contract is bias-corrected averages whose first value is
approximately the first sample, whatever the beta. These tests assert the
closed form and that each strategy is genuinely different.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.optimizers.exponential_weighted_averages import (
    AveragingStrategy,
    ExponentialWeightedAverage,
)


def test_first_bias_corrected_average_is_approximately_the_first_sample():
    """
    Bias correction exists to remove the (1 - beta) warm-up artifact. The first
    corrected average must be near the first sample, not one tenth of it.

    The tolerance is the epsilon in the denominator, which costs a relative
    eps/(1-beta) — 1e-5 at beta = 0.999.
    """
    for beta in (0.9, 0.99, 0.999):
        avg = ExponentialWeightedAverage(beta=beta)
        out = avg.update(5.0)
        np.testing.assert_allclose(out, 5.0, rtol=1e-4)


def test_bias_corrected_average_matches_the_closed_form():
    """
    For a constant stream x, m_t = (1 - beta^t) * x, so the corrected average
    is exactly x at every step. Asserting the unrolled accumulator catches an
    error the corrected output alone would mask.
    """
    beta, x, steps = 0.9, 3.0, 8
    avg = ExponentialWeightedAverage(beta=beta)
    for _ in range(steps):
        avg.update(x)
    np.testing.assert_allclose(avg.v, (1 - beta**steps) * x, rtol=1e-9)
    np.testing.assert_allclose(avg.get_current_average(), x, rtol=1e-6)


def test_tracked_weight_matches_the_closed_form_for_a_constant_beta():
    """
    The implementation tracks the applied weight rather than computing
    1 - beta**t. For a constant beta the two must agree, or every corrected
    value is wrong by the difference.
    """
    beta = 0.9
    avg = ExponentialWeightedAverage(beta=beta)
    for t in range(1, 10):
        avg.update(1.0)
        np.testing.assert_allclose(avg.weight, 1 - beta**t, rtol=1e-12)


def test_uncorrected_average_stays_damped():
    avg = ExponentialWeightedAverage(beta=0.9, bias_correction=False)
    for _ in range(5):
        avg.update(10.0)
    assert avg.get_current_average() < 10.0


def test_simple_strategy_has_no_bias_correction_at_all():
    """The uncorrected average starts at (1 - beta) * x, a tenth of the sample."""
    avg = ExponentialWeightedAverage(beta=0.9, strategy=AveragingStrategy.SIMPLE)
    avg.update(5.0)
    np.testing.assert_allclose(avg.get_current_average(), 0.5, rtol=1e-12)


def test_variance_corrected_strategy_reports_a_variance():
    avg = ExponentialWeightedAverage(
        beta=0.9, strategy=AveragingStrategy.VARIANCE_CORRECTED
    )
    for x in (1.0, 2.0, 3.0):
        avg.update(x)
    assert len(avg.variance_history) == 3
    assert all(np.isfinite(v) for v in avg.variance_history)


def test_ewa_is_more_responsive_than_the_mean():
    """
    The whole point of a weighted average: recent samples matter more. After a
    step change the EWA must sit between the old level and the new one, closer
    to the new one for large beta.
    """
    avg = ExponentialWeightedAverage(beta=0.9)
    for _ in range(10):
        avg.update(0.0)
    for _ in range(10):
        avg.update(1.0)
    level = avg.get_current_average()
    assert 0.0 < level < 1.0
    assert level > 0.5


def test_works_on_arrays_elementwise():
    """
    Each coordinate must average independently. Running the same stream through
    three scalar averages must reproduce the vector result exactly, which fails
    if any reduction leaks across coordinates.
    """
    stream = [np.array([1.0, 20.0, 300.0]), np.array([2.0, 30.0, 100.0])]

    vector = ExponentialWeightedAverage(beta=0.9)
    for x in stream:
        out = vector.update(x)
    assert out.shape == (3,)

    for i in range(3):
        scalar = ExponentialWeightedAverage(beta=0.9)
        for x in stream:
            expected = scalar.update(float(x[i]))
        np.testing.assert_allclose(out[i], expected, rtol=1e-12)


def test_warmup_keeps_a_constant_stream_at_its_own_level():
    """
    Regression test. Warm-up sets beta_t = (t-1)/t, which makes the accumulator
    already unbiased: the applied weight is 1 at every step. Correcting it by the
    constant-beta form `1 - beta**t` then divides by a number well under 1 and
    inflates the result — a constant stream of 10.0 read back as 100.0 at t=1 and
    52.6 at t=2.

    A weighted average of a constant stream is that constant, whatever beta does.
    """
    avg = ExponentialWeightedAverage(beta=0.9, warmup_steps=10)
    for t in range(1, 8):
        returned = avg.update(10.0)
        np.testing.assert_allclose(returned, 10.0, rtol=1e-6)
        np.testing.assert_allclose(avg.weight, 1.0, rtol=1e-9)


def test_update_and_get_current_average_never_disagree():
    """
    Both apply bias correction, so they must apply the same one. They read the
    same state and previously computed the correction factor two different ways.
    """
    for warmup in (0, 5):
        avg = ExponentialWeightedAverage(beta=0.9, warmup_steps=warmup)
        for x in (3.0, -1.0, 7.5, 0.25, 2.0):
            returned = avg.update(x)
            np.testing.assert_allclose(returned, avg.get_current_average(), rtol=1e-12)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        ExponentialWeightedAverage(beta=1.0)
    with pytest.raises(ValueError):
        ExponentialWeightedAverage(beta=0.0)
    with pytest.raises(ValueError):
        ExponentialWeightedAverage(epsilon=0.0)
    with pytest.raises(ValueError):
        ExponentialWeightedAverage(warmup_steps=-1)


def test_reset_returns_to_pristine_state():
    avg = ExponentialWeightedAverage(beta=0.9)
    for x in range(5):
        avg.update(float(x))
    avg.reset()
    assert avg.t == 0
    assert avg.v is None
    assert avg.weight == 0.0
    assert not avg.history


def test_a_constant_stream_never_averages_above_itself():
    """
    A weighted average of a constant is bounded by that constant, under every
    strategy, because the weights are non-negative and sum to at most one. An
    over-correction shows up here as a value above the stream.

    Only the corrected strategies reach the constant. SIMPLE and
    EXPONENTIAL_DECAY approach it from below and stay biased low, which is what
    bias correction exists to repair.
    """
    corrected = {
        AveragingStrategy.BIAS_CORRECTED,
        AveragingStrategy.VARIANCE_CORRECTED,
    }
    for strategy in AveragingStrategy:
        avg = ExponentialWeightedAverage(beta=0.9, strategy=strategy)
        for _ in range(200):
            out = avg.update(4.0)
        assert 0.0 < out <= 4.0 + 1e-9, f"{strategy} exceeded the stream: {out}"
        if strategy in corrected:
            np.testing.assert_allclose(out, 4.0, rtol=1e-6, err_msg=str(strategy))


def test_uncorrected_strategies_are_monotone_toward_the_constant():
    """They may lag, but they must always be closing the gap, never widening it."""
    for strategy in (AveragingStrategy.SIMPLE, AveragingStrategy.EXPONENTIAL_DECAY):
        avg = ExponentialWeightedAverage(beta=0.9, strategy=strategy)
        seen = [avg.update(4.0) for _ in range(50)]
        for earlier, later in zip(seen, seen[1:]):
            assert later >= earlier - 1e-12, f"{strategy} moved away from the stream"


def test_state_round_trips():
    avg = ExponentialWeightedAverage(beta=0.9)
    for x in range(5):
        avg.update(float(x))
    resumed = ExponentialWeightedAverage(beta=0.9)
    resumed.load_state(avg.get_state())
    assert resumed.get_current_average() == avg.get_current_average()
    assert resumed.t == avg.t
