"""
Adam Optimizer Tests
============================

Checks the Adam update against closed forms derived by hand, not against
recorded output. Every expected value here can be read off the equations in
`training-techniques/Optimization Algorithms.md`.

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

adam = load("optimization_algorithms/adam_optimizer.py")
AdamOptimizer = adam.AdamOptimizer


def test_first_step_size_equals_the_learning_rate():
    """
    With zero-initialized moments, the first bias-corrected step is
    `lr * g / (|g| + eps)`, so its magnitude is the learning rate whatever the
    gradient's scale. This is the property bias correction exists to produce.

    The tolerance is the epsilon term: the step falls short of `lr` by a relative
    `eps / (|g| + eps)`, which is largest for the smallest gradient. At g = 1e-3
    and eps = 1e-8 that is 1e-5, so 1e-4 passes on the exact arithmetic while
    still failing on a missing or misplaced bias correction, which costs a factor
    of ten at t = 1.
    """
    for scale in (1e-3, 1.0, 1e3):
        opt = AdamOptimizer(learning_rate=0.01)
        p = {"W": np.zeros((2, 2))}
        g = {"W": np.full((2, 2), scale)}
        moved = p["W"] - opt.update(p, g)["W"]
        np.testing.assert_allclose(moved, 0.01, rtol=1e-4)


def test_constant_gradient_gives_a_constant_step():
    """
    Under a constant gradient the corrected moments satisfy m_hat = g and
    v_hat = g^2 at every t, because the (1 - beta^t) factors cancel exactly.
    So every step has the same size, not a growing or shrinking one.
    """
    opt = AdamOptimizer(learning_rate=0.01)
    p = {"W": np.zeros((3,))}
    g = {"W": np.full((3,), 2.0)}

    steps = []
    for _ in range(5):
        new = opt.update(p, g)
        steps.append(float((p["W"] - new["W"])[0]))
        p = new

    np.testing.assert_allclose(steps, [0.01] * 5, rtol=1e-6)


def test_moments_match_the_closed_form():
    """
    For constant g the recursions unroll to m_t = (1 - beta1^t) * g and
    v_t = (1 - beta2^t) * g^2. Asserting the raw state, not just the step,
    localizes a fault to the accumulator rather than the update.
    """
    beta1, beta2, g_val, steps = 0.9, 0.999, 3.0, 7
    opt = AdamOptimizer(learning_rate=0.01, beta1=beta1, beta2=beta2)
    p = {"W": np.zeros((2,))}
    g = {"W": np.full((2,), g_val)}
    for _ in range(steps):
        p = opt.update(p, g)

    np.testing.assert_allclose(opt.m["W"], (1 - beta1**steps) * g_val, rtol=1e-9)
    np.testing.assert_allclose(opt.v["W"], (1 - beta2**steps) * g_val**2, rtol=1e-9)
    assert opt.t == steps


def test_step_opposes_the_gradient():
    """
    A sign error here is the defect this suite exists to catch: it still trains
    on a symmetric fixture, and it teaches the update backwards.
    """
    opt = AdamOptimizer(learning_rate=0.1)
    p = {"W": np.array([5.0, -5.0])}
    g = {"W": np.array([2.0, -2.0])}
    new = opt.update(p, g)["W"]
    assert new[0] < p["W"][0]
    assert new[1] > p["W"][1]


def test_descends_a_quadratic_to_its_minimum():
    """f(x) = sum(x^2) has its minimum at the origin and gradient 2x."""
    opt = AdamOptimizer(learning_rate=0.1)
    p = {"x": np.array([1.0, -2.0, 3.0])}
    for _ in range(500):
        p = opt.update(p, {"x": 2 * p["x"]})
    np.testing.assert_allclose(p["x"], 0.0, atol=1e-3)


def test_weight_decay_pulls_a_zero_gradient_parameter_toward_zero():
    """
    Decay enters as `grad + weight_decay * param`, so a parameter with no
    gradient of its own still shrinks, and shrinks toward zero from either side.
    """
    opt = AdamOptimizer(learning_rate=0.01, weight_decay=0.1)
    p = {"W": np.array([1.0, -1.0])}
    new = opt.update(p, {"W": np.zeros(2)})["W"]
    assert abs(new[0]) < 1.0
    assert abs(new[1]) < 1.0
    assert new[0] > 0 and new[1] < 0


def test_amsgrad_denominator_never_shrinks():
    """
    AMSGrad's guarantee is a non-decreasing v_hat_max, which is what bounds the
    effective learning rate. A decreasing entry means the maximum is not held.
    """
    opt = AdamOptimizer(learning_rate=0.01, beta2=0.9, amsgrad=True)
    p = {"W": np.zeros((2,))}
    seen = []
    for g_val in (5.0, 0.1, 0.1, 0.1, 3.0, 0.01):
        p = opt.update(p, {"W": np.full((2,), g_val)})
        seen.append(opt.v_hat_max["W"].copy())
    for earlier, later in zip(seen, seen[1:]):
        assert np.all(later >= earlier - 1e-12)


def test_clip_by_norm_caps_the_global_norm():
    """Clipping is global across parameters, not per-tensor."""
    opt = AdamOptimizer(learning_rate=0.01, gradient_clip_norm=1.0)
    g = {"a": np.array([3.0, 4.0]), "b": np.array([12.0])}
    clipped = opt._clip_gradients(g)
    total = np.sqrt(sum(np.sum(v**2) for v in clipped.values()))
    assert total <= 1.0 + 1e-6


def test_clip_by_value_bounds_each_entry():
    opt = AdamOptimizer(learning_rate=0.01, gradient_clip_value=0.5)
    clipped = opt._clip_gradients({"a": np.array([-9.0, 0.1, 9.0])})
    np.testing.assert_allclose(clipped["a"], [-0.5, 0.1, 0.5])


def test_a_parameter_without_a_gradient_is_returned_unchanged():
    opt = AdamOptimizer(learning_rate=0.1)
    p = {"W": np.array([1.0]), "frozen": np.array([7.0])}
    new = opt.update(p, {"W": np.array([1.0])})
    np.testing.assert_array_equal(new["frozen"], [7.0])
    assert new["frozen"] is not p["frozen"], "must not alias the input array"


def test_reset_state_clears_moments_and_the_step_counter():
    opt = AdamOptimizer(learning_rate=0.01)
    p = opt.update({"W": np.zeros(2)}, {"W": np.ones(2)})
    assert opt.t == 1 and opt.m
    opt.reset_state()
    assert opt.t == 0 and not opt.m and not opt.v

    again = opt.update({"W": np.zeros(2)}, {"W": np.ones(2)})
    np.testing.assert_allclose(again["W"], p["W"], rtol=1e-12)


def test_repeated_runs_agree():
    """Adam holds no randomness, so two identical runs must agree exactly."""

    def run():
        opt = AdamOptimizer(learning_rate=0.01)
        p = {"W": np.linspace(-1, 1, 6).reshape(2, 3)}
        for i in range(10):
            p = opt.update(p, {"W": np.full((2, 3), 0.1 * (i + 1))})
        return p["W"]

    np.testing.assert_array_equal(run(), run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": 1.5},
        {"beta1": 1.0},
        {"beta1": -0.1},
        {"beta2": 1.0},
        {"epsilon": 0.0},
        {"weight_decay": -0.1},
    ],
)
def test_rejects_hyperparameters_outside_their_valid_range(kwargs):
    with pytest.raises(ValueError):
        AdamOptimizer(**kwargs)


def test_factory_passes_arguments_through():
    opt = adam.create_adam_optimizer(learning_rate=0.5, beta1=0.5, amsgrad=True)
    assert opt.learning_rate == 0.5
    assert opt.beta1 == 0.5
    assert opt.amsgrad is True


def test_state_round_trips():
    """A reloaded optimizer must continue the trajectory, not restart it."""
    opt = AdamOptimizer(learning_rate=0.01)
    p = {"W": np.zeros(3)}
    g = {"W": np.array([0.5, -0.5, 1.5])}
    for _ in range(4):
        p = opt.update(p, g)

    resumed = AdamOptimizer(learning_rate=0.01)
    resumed.load_state(opt.get_state())
    np.testing.assert_allclose(resumed.update(p, g)["W"], opt.update(p, g)["W"])
