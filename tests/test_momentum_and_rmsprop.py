"""
Momentum and RMSprop Optimizer Tests
============================

Both optimizers accumulate an exponential average and both are one sign error
away from ascending. These tests assert the recursions in closed form and check
the direction of travel explicitly.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
from conftest import load

from dlhub.optimizers.momentum import MomentumOptimizer
from dlhub.optimizers.rmsprop import RMSpropOptimizer


class TestMomentum:
    """v_t = beta * v_{t-1} + (1 - beta) * g_t, then theta -= lr * v_hat."""

    def test_velocity_matches_the_closed_form(self):
        """Constant g unrolls to v_t = (1 - beta^t) * g."""
        beta, g_val, steps = 0.9, 2.0, 6
        opt = MomentumOptimizer(learning_rate=0.01, beta=beta)
        p = {"W": np.zeros((2,))}
        for _ in range(steps):
            p = opt.update_parameters(p, {"W": np.full((2,), g_val)})
        np.testing.assert_allclose(opt.v["W"], (1 - beta**steps) * g_val, rtol=1e-9)

    def test_bias_corrected_first_step_is_the_full_learning_rate(self):
        """
        At t = 1 the correction divides by (1 - beta), which exactly undoes the
        (1 - beta) factor in the velocity. So the first step is lr * g, the same
        step plain gradient descent would take.
        """
        opt = MomentumOptimizer(learning_rate=0.1, beta=0.9, bias_correction=True)
        moved = 0.0 - opt.update_parameters({"W": np.zeros(1)}, {"W": np.ones(1)})["W"]
        np.testing.assert_allclose(moved, 0.1, rtol=1e-9)

    def test_without_correction_the_first_step_is_damped(self):
        """
        The uncorrected first step is lr * (1 - beta) * g — a tenth of the size at
        beta = 0.9. This is the warm-up artifact bias correction removes.
        """
        opt = MomentumOptimizer(learning_rate=0.1, beta=0.9, bias_correction=False)
        moved = 0.0 - opt.update_parameters({"W": np.zeros(1)}, {"W": np.ones(1)})["W"]
        np.testing.assert_allclose(moved, 0.1 * 0.1, rtol=1e-9)

    def test_step_opposes_the_gradient(self):
        opt = MomentumOptimizer(learning_rate=0.1)
        new = opt.update_parameters(
            {"W": np.array([5.0, -5.0])}, {"W": np.array([2.0, -2.0])}
        )["W"]
        assert new[0] < 5.0 and new[1] > -5.0

    def test_momentum_carries_through_a_zero_gradient(self):
        """
        The point of momentum: motion persists after the gradient vanishes. A
        plain SGD step would stop dead here.
        """
        opt = MomentumOptimizer(learning_rate=0.1, beta=0.9)
        p = opt.update_parameters({"W": np.zeros(1)}, {"W": np.ones(1)})
        before = p["W"].copy()
        after = opt.update_parameters(p, {"W": np.zeros(1)})["W"]
        assert after[0] < before[0] - 1e-6

    def test_descends_a_quadratic(self):
        opt = MomentumOptimizer(learning_rate=0.1)
        p = {"x": np.array([1.0, -2.0])}
        for _ in range(2000):
            p = opt.update_parameters(p, {"x": 2 * p["x"]})
        np.testing.assert_allclose(p["x"], 0.0, atol=1e-3)

    def test_gradient_norm_is_the_euclidean_norm_over_all_parameters(self):
        opt = MomentumOptimizer()
        got = opt.compute_gradient_norm(
            {"a": np.array([3.0, 4.0]), "b": np.array([12.0])}
        )
        np.testing.assert_allclose(got, 13.0, rtol=1e-12)

    def test_reset_clears_state(self):
        opt = MomentumOptimizer(learning_rate=0.01)
        opt.update_parameters({"W": np.zeros(2)}, {"W": np.ones(2)})
        opt.reset_optimizer_state()
        assert opt.t == 0 and not opt.v


class TestRMSprop:
    """s_t = beta * s_{t-1} + (1 - beta) * g_t^2, then theta -= lr * g / (sqrt(s) + eps)."""

    def test_second_moment_matches_the_closed_form(self):
        beta, g_val, steps = 0.9, 3.0, 5
        opt = RMSpropOptimizer(learning_rate=0.01, beta=beta)
        p = {"W": np.zeros((2,))}
        for _ in range(steps):
            p = opt.update_parameters(p, {"W": np.full((2,), g_val)})
        np.testing.assert_allclose(opt.s["W"], (1 - beta**steps) * g_val**2, rtol=1e-9)

    def test_step_size_is_scale_invariant(self):
        """
        RMSprop's defining property: dividing by sqrt(s) cancels the gradient's
        magnitude, so rescaling every gradient by a constant leaves the step
        unchanged. A per-parameter learning rate that still tracked scale would
        fail here.
        """
        steps = []
        for scale in (1e-2, 1.0, 1e2):
            opt = RMSpropOptimizer(learning_rate=0.01, beta=0.9)
            p = {"W": np.zeros(1)}
            for _ in range(3):
                p = opt.update_parameters(p, {"W": np.full(1, scale)})
            steps.append(float(p["W"][0]))
        np.testing.assert_allclose(steps, steps[0], rtol=1e-4)

    def test_step_opposes_the_gradient(self):
        opt = RMSpropOptimizer(learning_rate=0.1)
        new = opt.update_parameters(
            {"W": np.array([5.0, -5.0])}, {"W": np.array([2.0, -2.0])}
        )["W"]
        assert new[0] < 5.0 and new[1] > -5.0

    def test_larger_second_moment_means_a_smaller_step(self):
        """A consistently noisy coordinate must be stepped more cautiously."""
        opt = RMSpropOptimizer(learning_rate=0.01, beta=0.9)
        p = {"W": np.zeros(2)}
        for _ in range(20):
            p = opt.update_parameters(p, {"W": np.array([10.0, 10.0])})
        quiet, loud = p["W"].copy(), p["W"].copy()
        assert opt.s["W"][0] > 0

        step_after_history = abs(
            opt.update_parameters({"W": quiet}, {"W": np.array([1.0, 1.0])})["W"][0]
            - quiet[0]
        )
        fresh = RMSpropOptimizer(learning_rate=0.01, beta=0.9)
        step_from_zero = abs(
            fresh.update_parameters({"W": loud}, {"W": np.array([1.0, 1.0])})["W"][0]
            - loud[0]
        )
        assert step_after_history < step_from_zero

    def test_learning_rate_decay_follows_lr0_over_one_plus_decay_t(self):
        opt = RMSpropOptimizer(learning_rate=1.0, decay=0.5)
        p = {"W": np.zeros(1)}
        for t in range(1, 5):
            p = opt.update_parameters(p, {"W": np.ones(1)})
            np.testing.assert_allclose(
                opt.learning_rate, 1.0 / (1 + 0.5 * t), rtol=1e-12
            )

    def test_descends_a_quadratic(self):
        opt = RMSpropOptimizer(learning_rate=0.01)
        p = {"x": np.array([1.0, -2.0])}
        for _ in range(3000):
            p = opt.update_parameters(p, {"x": 2 * p["x"]})
        np.testing.assert_allclose(p["x"], 0.0, atol=1e-2)

    def test_reset_clears_state(self):
        opt = RMSpropOptimizer(learning_rate=0.01, decay=0.1)
        opt.update_parameters({"W": np.zeros(2)}, {"W": np.ones(2)})
        opt.reset_optimizer_state()
        assert opt.t == 0 and not opt.s
        assert opt.learning_rate == opt.initial_learning_rate


def test_the_three_optimizers_agree_on_direction():
    """
    Different step sizes are expected; a different sign is not. This is the
    cheapest cross-module check that no single optimizer has drifted.
    """
    p = {"W": np.array([1.0, -1.0])}
    g = {"W": np.array([0.5, -0.5])}

    adam = load("optimization_algorithms/adam_optimizer.py").AdamOptimizer(
        learning_rate=0.01
    )
    results = [
        adam.update(p, g)["W"],
        MomentumOptimizer(learning_rate=0.01).update_parameters(p, g)["W"],
        RMSpropOptimizer(learning_rate=0.01).update_parameters(p, g)["W"],
    ]
    for out in results:
        assert out[0] < p["W"][0]
        assert out[1] > p["W"][1]
