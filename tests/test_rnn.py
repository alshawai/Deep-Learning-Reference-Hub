"""
Vanilla RNN Reference Tests
===========================

This module is the correctness proof for the hub's RNN reference, and the
framework parity port is checked against it. The one failure it exists to catch
is a flipped sign in backpropagation through time: a sign error trains the model
uphill while every shape assertion still passes, so it teaches confidently wrong
math. The gradient-agreement test therefore pins every analytic gradient against
a central finite-difference approximation on the dossier's fixture, and a sign
flip drives the relative error to order one.

The other two contracts guard the fixture's determinism and the shapes the topic
promises, including the single-timestep edge where the recurrence degenerates.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from collections.abc import Callable

import numpy as np
import pytest

from dlhub.nn.sequence.rnn import (
    bidirectional_rnn_forward,
    clip_gradients,
    compute_loss,
    deep_rnn_forward,
    make_fixture,
    rnn_backward,
    rnn_forward,
    softmax,
    update_parameters,
)

# The dossier's finite-difference contract: central differences with this step,
# analytic-vs-numeric relative error below this tolerance, for every parameter.
FD_EPSILON = 1e-7
FD_TOLERANCE = 1e-7


def numeric_gradient(
    loss: Callable[[], float], theta: np.ndarray, eps: float = FD_EPSILON
) -> np.ndarray:
    """
    Central finite-difference gradient of ``loss`` with respect to ``theta``.

    ``loss`` must read ``theta`` in place (it is a closure over the same array),
    so each entry is perturbed by +/- eps, the loss re-evaluated, and the entry
    restored. The array is unchanged on return.
    """
    grad = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        original = theta[idx]

        theta[idx] = original + eps
        loss_plus = loss()
        theta[idx] = original - eps
        loss_minus = loss()
        theta[idx] = original

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Norm-based relative difference, the metric the hub's gradient check uses."""
    numerator = np.linalg.norm(analytic - numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    return 0.0 if denominator == 0 else float(numerator / denominator)


def global_norm(gradients: dict[str, np.ndarray]) -> float:
    """L2 norm of all gradients viewed as one concatenated vector."""
    return float(np.sqrt(sum(np.sum(g**2) for g in gradients.values())))


# --- Gradient agreement (the sign-flip catcher) --------------------------

# Each analytic gradient the dossier names, paired with the fixture tensor it
# differentiates and the key it is returned under.
GRADIENT_TARGETS = [
    ("Wax", "dWax"),
    ("Waa", "dWaa"),
    ("Wya", "dWya"),
    ("ba", "dba"),
    ("by", "dby"),
    ("a0", "da0"),
    ("x", "dx"),
]


@pytest.mark.parametrize("tensor_name, grad_key", GRADIENT_TARGETS)
def test_bptt_gradient_matches_finite_difference(tensor_name, grad_key):
    """
    Every analytic BPTT gradient agrees with central differences on the fixture.

    This is the test the reference exists to pass. A dropped term, a missing
    1/m, or -- the worst case -- a flipped sign, all push the relative error far
    above the tolerance instead of failing silently at training time.
    """
    fixture = make_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )

    _, _, caches = rnn_forward(x, parameters, a0)
    analytic = rnn_backward(y, caches, parameters)

    tensor = {**parameters, "a0": a0, "x": x}[tensor_name]

    def loss() -> float:
        _, y_pred, _ = rnn_forward(x, parameters, a0)
        return compute_loss(y_pred, y)

    numeric = numeric_gradient(loss, tensor)
    error = relative_error(analytic[grad_key], numeric)
    assert error < FD_TOLERANCE, f"{grad_key}: relative error {error:.2e}"


def test_a_flipped_sign_is_rejected():
    """
    The defect a hub must never ship. If the sign of any gradient were flipped,
    its agreement with finite differences would collapse -- confirming the check
    discriminates, rather than passing everything.
    """
    fixture = make_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    _, _, caches = rnn_forward(x, parameters, a0)
    analytic = rnn_backward(y, caches, parameters)

    def loss() -> float:
        _, y_pred, _ = rnn_forward(x, parameters, a0)
        return compute_loss(y_pred, y)

    numeric = numeric_gradient(loss, parameters["Waa"])
    assert relative_error(-analytic["dWaa"], numeric) > 0.9


# --- Determinism ---------------------------------------------------------


def test_the_seeded_fixture_reproduces_exactly():
    """The pinned draw order under seed(1) must give bit-identical arrays."""
    first = make_fixture()
    second = make_fixture()
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["a0"], second["a0"])
    for key in ("Wax", "Waa", "Wya", "ba", "by"):
        np.testing.assert_array_equal(
            first["parameters"][key], second["parameters"][key]
        )
    np.testing.assert_array_equal(first["y"], second["y"])


def test_the_forward_trajectory_is_deterministic():
    """Re-running the forward pass on one fixture yields the same trajectory."""
    fixture = make_fixture()
    x, a0, parameters = fixture["x"], fixture["a0"], fixture["parameters"]
    a1, y1, _ = rnn_forward(x, parameters, a0)
    a2, y2, _ = rnn_forward(x, parameters, a0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(y1, y2)


# --- Shape and edge contracts --------------------------------------------


def test_forward_produces_the_declared_shapes():
    """Hidden states are (n_a, m, T_x) and outputs are (n_y, m, T_x)."""
    fixture = make_fixture()
    d = fixture["dims"]
    a, y_pred, caches = rnn_forward(fixture["x"], fixture["parameters"], fixture["a0"])
    assert a.shape == (d["n_a"], d["m"], d["T_x"])
    assert y_pred.shape == (d["n_y"], d["m"], d["T_x"])
    assert len(caches) == d["T_x"]


def test_backward_gradients_match_their_parameter_shapes():
    """Each gradient carries the shape of the thing it differentiates."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    _, _, caches = rnn_forward(x, parameters, a0)
    grads = rnn_backward(y, caches, parameters)

    for key in ("Wax", "Waa", "Wya", "ba", "by"):
        assert grads[f"d{key}"].shape == parameters[key].shape
    assert grads["da0"].shape == (d["n_a"], d["m"])
    assert grads["dx"].shape == (d["n_x"], d["m"], d["T_x"])


def test_softmax_outputs_are_distributions():
    """Every output column is strictly positive and sums to one."""
    fixture = make_fixture()
    _, y_pred, _ = rnn_forward(fixture["x"], fixture["parameters"], fixture["a0"])
    assert np.all(y_pred > 0)
    column_sums = np.sum(y_pred, axis=0)
    np.testing.assert_allclose(column_sums, 1.0, atol=1e-12)


def test_softmax_is_shift_invariant_and_stable():
    """The max-subtraction shift leaves the result unchanged and avoids overflow."""
    z = np.array([[1.0, 1000.0], [2.0, 1000.0], [3.0, 999.0]])
    probs = softmax(z)
    np.testing.assert_allclose(np.sum(probs, axis=0), 1.0)
    assert np.all(np.isfinite(probs))
    # Adding a per-column constant must not change the distribution.
    shifted = softmax(z + np.array([[5.0, -7.0]]))
    np.testing.assert_allclose(probs, shifted)


def test_the_default_initial_state_is_the_zero_vector():
    """Omitting a0 matches passing an explicit zero state (a^{<0>} = 0)."""
    fixture = make_fixture()
    x, parameters, d = fixture["x"], fixture["parameters"], fixture["dims"]
    a_default, y_default, _ = rnn_forward(x, parameters)
    a_zeros, y_zeros, _ = rnn_forward(x, parameters, np.zeros((d["n_a"], d["m"])))
    np.testing.assert_array_equal(a_default, a_zeros)
    np.testing.assert_array_equal(y_default, y_zeros)


def test_a_single_timestep_sequence_works():
    """
    T_x = 1 is the degenerate edge: the recurrence has no successor, so da_next
    starts and stays zero. Shapes and gradient agreement must still hold.
    """
    fixture = make_fixture()
    d = fixture["dims"]
    x1 = fixture["x"][:, :, :1]
    y1 = fixture["y"][:, :, :1]
    parameters, a0 = fixture["parameters"], fixture["a0"]

    a, y_pred, caches = rnn_forward(x1, parameters, a0)
    assert a.shape == (d["n_a"], d["m"], 1)
    assert y_pred.shape == (d["n_y"], d["m"], 1)

    analytic = rnn_backward(y1, caches, parameters)

    def loss() -> float:
        _, y_pred_inner, _ = rnn_forward(x1, parameters, a0)
        return compute_loss(y_pred_inner, y1)

    numeric = numeric_gradient(loss, parameters["Waa"])
    assert relative_error(analytic["dWaa"], numeric) < FD_TOLERANCE


def test_rnn_forward_rejects_a_two_dimensional_input():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        rnn_forward(fixture["x"][:, :, 0], fixture["parameters"], fixture["a0"])


def test_rnn_forward_rejects_a_mismatched_initial_state():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        rnn_forward(fixture["x"], fixture["parameters"], np.zeros((99, 10)))


# --- Gradient clipping ---------------------------------------------------


def test_clip_scales_an_oversized_gradient_to_the_threshold():
    """A gradient above the cap is rescaled to exactly the cap, direction kept."""
    gradients = {
        "dWax": np.array([[3.0, 4.0]]),  # norm 5 on its own
        "dWaa": np.array([[12.0]]),
    }  # combined norm 13
    clipped = clip_gradients(gradients, max_norm=1.0)
    assert global_norm(clipped) == pytest.approx(1.0)
    # Direction preserved: every entry scaled by the same factor 1/13.
    for key in gradients:
        np.testing.assert_allclose(clipped[key], gradients[key] / 13.0)


def test_clip_leaves_a_small_gradient_untouched_but_copied():
    """Below the cap the values are unchanged, yet a fresh dict is returned."""
    gradients = {"dWax": np.array([[0.3, 0.4]])}  # norm 0.5 < 1
    clipped = clip_gradients(gradients, max_norm=1.0)
    np.testing.assert_array_equal(clipped["dWax"], gradients["dWax"])
    assert clipped["dWax"] is not gradients["dWax"]


# --- Parameter update ----------------------------------------------------


def test_update_applies_one_gradient_descent_step():
    """Each shared weight moves by -lr * its gradient; the input is untouched."""
    fixture = make_fixture()
    parameters = fixture["parameters"]
    before = {k: v.copy() for k, v in parameters.items()}
    gradients = {f"d{k}": np.ones_like(v) for k, v in parameters.items()}

    updated = update_parameters(parameters, gradients, learning_rate=0.1)

    for key in parameters:
        np.testing.assert_allclose(updated[key], before[key] - 0.1)
        np.testing.assert_array_equal(parameters[key], before[key])


# --- Compositional variants (forward-pass shape walkthroughs) ------------


def test_bidirectional_concatenates_two_direction_states():
    """
    Output width is 2 * n_a, the forward half matches a plain forward pass, and
    the backward half is the reverse-time recurrence realigned to forward order.
    """
    fixture = make_fixture()
    d = fixture["dims"]
    x = fixture["x"]
    rng = np.random.default_rng(0)
    params_fwd = fixture["parameters"]
    params_bwd = {
        "Wax": rng.standard_normal((d["n_a"], d["n_x"])),
        "Waa": rng.standard_normal((d["n_a"], d["n_a"])),
        "Wya": rng.standard_normal((d["n_y"], d["n_a"])),
        "ba": rng.standard_normal((d["n_a"], 1)),
        "by": rng.standard_normal((d["n_y"], 1)),
    }

    a_bi, parts = bidirectional_rnn_forward(x, params_fwd, params_bwd)
    assert a_bi.shape == (2 * d["n_a"], d["m"], d["T_x"])

    a_fwd_ref, _, _ = rnn_forward(x, params_fwd)
    a_bwd_rev, _, _ = rnn_forward(x[:, :, ::-1], params_bwd)
    a_bwd_ref = a_bwd_rev[:, :, ::-1]

    np.testing.assert_array_equal(parts["a_forward"], a_fwd_ref)
    np.testing.assert_array_equal(parts["a_backward"], a_bwd_ref)
    np.testing.assert_array_equal(a_bi[: d["n_a"]], a_fwd_ref)
    np.testing.assert_array_equal(a_bi[d["n_a"] :], a_bwd_ref)


def test_deep_rnn_stacks_layers_and_propagates_widths():
    """
    A layer's activation sequence is the input to the layer above, so hidden
    widths flow upward and only the top layer produces the model output.
    """
    fixture = make_fixture()
    d = fixture["dims"]
    x = fixture["x"]
    rng = np.random.default_rng(1)
    n_a0, n_a1 = 5, 6

    layer0 = {
        "Wax": rng.standard_normal((n_a0, d["n_x"])),
        "Waa": rng.standard_normal((n_a0, n_a0)),
        "Wya": rng.standard_normal((d["n_y"], n_a0)),
        "ba": rng.standard_normal((n_a0, 1)),
        "by": rng.standard_normal((d["n_y"], 1)),
    }
    layer1 = {
        "Wax": rng.standard_normal((n_a1, n_a0)),  # consumes layer0's width
        "Waa": rng.standard_normal((n_a1, n_a1)),
        "Wya": rng.standard_normal((d["n_y"], n_a1)),
        "ba": rng.standard_normal((n_a1, 1)),
        "by": rng.standard_normal((d["n_y"], 1)),
    }

    activations, y_pred_top = deep_rnn_forward(x, [layer0, layer1])

    assert activations[0].shape == (n_a0, d["m"], d["T_x"])
    assert activations[1].shape == (n_a1, d["m"], d["T_x"])
    assert y_pred_top.shape == (d["n_y"], d["m"], d["T_x"])

    # Layer 1 is a full RNN over layer 0's activations.
    a1_ref, y1_ref, _ = rnn_forward(activations[0], layer1)
    np.testing.assert_array_equal(activations[1], a1_ref)
    np.testing.assert_array_equal(y_pred_top, y1_ref)


def test_deep_rnn_rejects_an_empty_stack():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        deep_rnn_forward(fixture["x"], [])
