"""
GRU Reference Tests
===================

This module is the correctness proof for the hub's GRU reference, and the
TensorFlow parity port is checked against it. The one failure it exists to catch
is a flipped sign in backpropagation through time: a sign error trains the model
uphill while every shape assertion still passes, so it teaches confidently wrong
math. The GRU adds a second place for one to hide -- the previous state reaches
the next one along three separate paths, and one wrong sign among them cancels
part of the gradient rather than all of it -- so the gradient-agreement test
pins every analytic gradient against a central finite-difference approximation
on the dossier's fixture.

Two tests state what the cell is for. With the update gate shut, the state is
copied forward and the gradient arriving at ``a^{<0>}`` is exactly the undamped
sum of the per-step read-out gradients -- no geometric decay. And the reset gate
is pinned to act *before* the hidden transform: the reference is shown to
disagree with the reset-after form that PyTorch and cuDNN-mode Keras compute, so
a later "simplification" to that form fails here rather than silently changing
the function the hub documents.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.nn.sequence._common import sigmoid, split_gate_matrix
from dlhub.nn.sequence._gradient_check import numeric_gradient, relative_error
from dlhub.nn.sequence.gru import (
    gru_backward,
    gru_cell_forward,
    gru_forward,
    make_fixture,
)
from dlhub.nn.sequence.rnn import compute_loss

# The dossier's finite-difference contract: analytic-vs-numeric relative error
# below this tolerance, for every parameter.
FD_TOLERANCE = 1e-7

# The step is one decade larger than the shared default the vanilla RNN uses,
# and it has to be. A central difference subtracts two float64 losses of order
# 3, so the roundoff floor on the relative error of a gradient `g` is about
# ulp(L) / (2 eps ||g||). The reset gate's gradient has norm 0.06 on this
# fixture, which puts its floor at 1.2e-7 for eps = 1e-7 -- above the tolerance
# no matter how correct the analytic gradient is. At 1e-6 the floor drops
# tenfold, the truncation error it trades against stays near 1e-12, and every
# gradient below lands at or under 1.1e-8.
FD_EPSILON = 1e-6

# Each analytic gradient the dossier names, paired with the fixture tensor it
# differentiates and the key it is returned under. Written out rather than
# derived from the module's own key list, so dropping a gate there cannot
# quietly drop it from the check.
GRADIENT_TARGETS = [
    ("Wr", "dWr"),
    ("Wu", "dWu"),
    ("Wc", "dWc"),
    ("Wy", "dWy"),
    ("br", "dbr"),
    ("bu", "dbu"),
    ("bc", "dbc"),
    ("by", "dby"),
    ("a0", "da0"),
    ("x", "dx"),
]


def loss_closure(x, parameters, a0, y):
    """
    A zero-argument loss over the given arrays, for the finite-difference check.

    Every array is read in place on each call, so perturbing an entry of any one
    of them and re-calling is what moves the loss.
    """

    def loss() -> float:
        _, y_pred, _ = gru_forward(x, parameters, a0)
        return compute_loss(y_pred, y)

    return loss


def unpack(fixture):
    """The four arrays every test below starts from."""
    return fixture["x"], fixture["a0"], fixture["parameters"], fixture["y"]


# --- Gradient agreement (the sign-flip catcher) --------------------------


@pytest.mark.parametrize("tensor_name, grad_key", GRADIENT_TARGETS)
def test_bptt_gradient_matches_finite_difference(tensor_name, grad_key):
    """
    Every analytic BPTT gradient agrees with central differences on the fixture.

    This is the test the reference exists to pass. A dropped term, a missing
    1/m, the reset gate's product rule applied to only one of its two factors,
    or -- the worst case -- a flipped sign, all push the relative error far above
    the tolerance instead of failing silently at training time.
    """
    x, a0, parameters, y = unpack(make_fixture())

    _, _, caches = gru_forward(x, parameters, a0)
    analytic = gru_backward(y, caches, parameters)

    tensor = {**parameters, "a0": a0, "x": x}[tensor_name]
    numeric = numeric_gradient(loss_closure(x, parameters, a0, y), tensor, FD_EPSILON)

    error = relative_error(analytic[grad_key], numeric)
    assert error < FD_TOLERANCE, f"{grad_key}: relative error {error:.2e}"


def test_a_flipped_sign_is_rejected():
    """
    The defect a hub must never ship. If the sign of the update gate's gradient
    were flipped, its agreement with finite differences would collapse --
    confirming the check discriminates, rather than passing everything.
    """
    x, a0, parameters, y = unpack(make_fixture())
    _, _, caches = gru_forward(x, parameters, a0)
    analytic = gru_backward(y, caches, parameters)

    numeric = numeric_gradient(
        loss_closure(x, parameters, a0, y), parameters["Wu"], FD_EPSILON
    )
    assert relative_error(-analytic["dWu"], numeric) > 0.9


# --- The thesis: the (1 - Gamma_u) carry path ----------------------------


def test_a_shut_update_gate_copies_the_state_forward():
    """
    With Gamma_u = 0 the blend degenerates to a^{<t>} = a^{<t-1>}.

    The candidate is still computed and still discarded, so the state arrives at
    the last timestep unchanged. A gate wired to the wrong side of the blend
    would overwrite it instead.
    """
    x, a0, parameters, _ = unpack(make_fixture())
    shut = {key: value.copy() for key, value in parameters.items()}
    shut["Wu"] = np.zeros_like(shut["Wu"])  # gate depends on the bias alone
    shut["bu"] = np.full_like(shut["bu"], -15.0)  # sigma(-15) = 3e-7

    a, _, _ = gru_forward(x, shut, a0)

    for t in range(x.shape[2]):
        np.testing.assert_allclose(a[:, :, t], a0, atol=1e-5)


def test_the_carry_path_delivers_gradient_undamped():
    """
    With the update gate shut, d L / d a^{<0>} is the *sum* of every step's
    read-out gradient, with no per-step decay factor at all.

    That is the GRU's answer to the vanishing gradient in one equation. Because
    a^{<t>} = a^{<t-1>} for every t, the initial state is the state the read-out
    saw at all T_x steps, so each step's gradient W_y^T (y_hat - y) / m arrives
    at a^{<0>} with weight one. A carry path that damped the gradient -- through
    a weight matrix or a saturating nonlinearity, as the vanilla RNN's does --
    would make the T_x terms shrink geometrically and this equality fail.
    """
    x, a0, parameters, y = unpack(make_fixture())
    m, T_x = x.shape[1], x.shape[2]

    shut = {key: value.copy() for key, value in parameters.items()}
    shut["Wu"] = np.zeros_like(shut["Wu"])
    shut["bu"] = np.full_like(shut["bu"], -15.0)

    _, y_pred, caches = gru_forward(x, shut, a0)
    da0 = gru_backward(y, caches, shut)["da0"]

    undamped_sum = sum(
        shut["Wy"].T @ ((y_pred[:, :, t] - y[:, :, t]) / m) for t in range(T_x)
    )
    np.testing.assert_allclose(da0, undamped_sum, atol=1e-5)


# --- The reset gate's placement (the framework divergence) ---------------


def test_the_reset_gate_acts_before_the_hidden_transform():
    """
    The hub computes tanh(W_c [Gamma_r * a^{<t-1>}; x]), not the reset-after form
    tanh(W_cx x + Gamma_r * (W_ca a^{<t-1>})) that torch.nn.GRU and
    tf.keras.layers.GRU(reset_after=True) compute.

    The two are not algebraically equal, and this pins the difference as a
    measured fact rather than a remark: on the fixture's first step they differ
    by more than one, far outside any tolerance. Without this test, "simplifying"
    the candidate to the framework form would pass every other assertion here
    while silently changing the function the hub documents.
    """
    x, a0, parameters, _ = unpack(make_fixture())
    n_a = parameters["Wu"].shape[0]
    x_t = x[:, :, 0]

    a_reference, _, cache = gru_cell_forward(x_t, a0, parameters)

    # The same weights, read the framework's way: split W_c into its recurrent
    # and input halves and let the gate act on the transformed state instead.
    Wca, Wcx = split_gate_matrix(parameters["Wc"], n_a)
    gamma_r, gamma_u = cache["gamma_r"], cache["gamma_u"]
    a_tilde_after = np.tanh(Wcx @ x_t + gamma_r * (Wca @ a0) + parameters["bc"])
    a_reset_after = gamma_u * a_tilde_after + (1 - gamma_u) * a0

    assert np.max(np.abs(a_reference - a_reset_after)) > 0.1


# --- Determinism ---------------------------------------------------------


def test_the_seeded_fixture_reproduces_exactly():
    """The pinned draw order under seed(1) must give bit-identical arrays."""
    first, second = make_fixture(), make_fixture()
    for key in ("x", "a0", "y"):
        np.testing.assert_array_equal(first[key], second[key])
    for key in first["parameters"]:
        np.testing.assert_array_equal(
            first["parameters"][key], second["parameters"][key]
        )


def test_the_forward_trajectory_is_deterministic():
    """Re-running the forward pass on one fixture yields the same trajectory."""
    x, a0, parameters, _ = unpack(make_fixture())
    a1, y1, _ = gru_forward(x, parameters, a0)
    a2, y2, _ = gru_forward(x, parameters, a0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(y1, y2)


# --- Shape and edge contracts --------------------------------------------


def test_forward_produces_the_declared_shapes():
    """Hidden states are (n_a, m, T_x) and outputs are (n_y, m, T_x)."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, a0, parameters, _ = unpack(fixture)

    a, y_pred, caches = gru_forward(x, parameters, a0)

    assert a.shape == (d["n_a"], d["m"], d["T_x"])
    assert y_pred.shape == (d["n_y"], d["m"], d["T_x"])
    assert len(caches) == d["T_x"]


def test_backward_gradients_match_their_parameter_shapes():
    """Each gradient carries the shape of the thing it differentiates."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, a0, parameters, y = unpack(fixture)

    _, _, caches = gru_forward(x, parameters, a0)
    grads = gru_backward(y, caches, parameters)

    for key, value in parameters.items():
        assert grads[f"d{key}"].shape == value.shape
    assert grads["da0"].shape == (d["n_a"], d["m"])
    assert grads["dx"].shape == (d["n_x"], d["m"], d["T_x"])


def test_the_sequence_loop_agrees_with_stepping_the_cell_by_hand():
    """
    gru_forward is exactly the cell applied T_x times with the state threaded.

    Pinning this keeps the unrolled loop honest: a state carried from the wrong
    step would still produce arrays of the right shape.
    """
    x, a0, parameters, _ = unpack(make_fixture())
    a, y_pred, _ = gru_forward(x, parameters, a0)

    a_prev = a0
    for t in range(x.shape[2]):
        a_next, y_t, _ = gru_cell_forward(x[:, :, t], a_prev, parameters)
        np.testing.assert_array_equal(a[:, :, t], a_next)
        np.testing.assert_array_equal(y_pred[:, :, t], y_t)
        a_prev = a_next


def test_the_gates_are_probabilities_and_the_readout_is_a_distribution():
    """Both gates lie strictly in (0, 1); every output column sums to one."""
    x, a0, parameters, _ = unpack(make_fixture())
    _, y_pred, caches = gru_forward(x, parameters, a0)

    for cache in caches:
        for gate in ("gamma_r", "gamma_u"):
            assert np.all(cache[gate] > 0.0) and np.all(cache[gate] < 1.0)
        assert np.all(np.abs(cache["a_tilde"]) < 1.0)  # tanh, not sigmoid

    np.testing.assert_allclose(np.sum(y_pred, axis=0), 1.0, atol=1e-12)


def test_the_blend_is_a_convex_combination():
    """
    a^{<t>} lies between a^{<t-1>} and the candidate, entry by entry.

    The two blend weights are Gamma_u and 1 - Gamma_u, so the new state can
    never leave the interval its two inputs span. Swapping the weights would
    keep that true, which is why the gradient check above is the real guard;
    this pins the weaker property that they are complements and not, say, both
    equal to Gamma_u.
    """
    x, a0, parameters, _ = unpack(make_fixture())
    _, _, caches = gru_forward(x, parameters, a0)

    for cache in caches:
        lower = np.minimum(cache["a_prev"], cache["a_tilde"])
        upper = np.maximum(cache["a_prev"], cache["a_tilde"])
        assert np.all(cache["a_next"] >= lower - 1e-12)
        assert np.all(cache["a_next"] <= upper + 1e-12)


def test_the_default_initial_state_is_the_zero_vector():
    """Omitting a0 matches passing an explicit zero state (a^{<0>} = 0)."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, parameters = fixture["x"], fixture["parameters"]

    a_default, y_default, _ = gru_forward(x, parameters)
    a_zeros, y_zeros, _ = gru_forward(x, parameters, np.zeros((d["n_a"], d["m"])))

    np.testing.assert_array_equal(a_default, a_zeros)
    np.testing.assert_array_equal(y_default, y_zeros)


def test_a_single_timestep_sequence_works():
    """
    T_x = 1 is the degenerate edge: the recurrence has no successor, so da_next
    starts and stays zero. Shapes and gradient agreement must still hold.
    """
    fixture = make_fixture()
    d = fixture["dims"]
    x1, y1 = fixture["x"][:, :, :1], fixture["y"][:, :, :1]
    a0, parameters = fixture["a0"], fixture["parameters"]

    a, y_pred, caches = gru_forward(x1, parameters, a0)
    assert a.shape == (d["n_a"], d["m"], 1)
    assert y_pred.shape == (d["n_y"], d["m"], 1)

    analytic = gru_backward(y1, caches, parameters)
    numeric = numeric_gradient(
        loss_closure(x1, parameters, a0, y1), parameters["Wu"], FD_EPSILON
    )
    assert relative_error(analytic["dWu"], numeric) < FD_TOLERANCE


def test_gru_forward_rejects_a_two_dimensional_input():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        gru_forward(fixture["x"][:, :, 0], fixture["parameters"])


def test_gru_forward_rejects_a_mismatched_initial_state():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        gru_forward(fixture["x"], fixture["parameters"], np.zeros((99, 10)))


def test_the_gate_nonlinearity_is_the_shared_sigmoid():
    """
    The gates are the package's stable sigmoid, not a locally rolled one.

    Recomputing one gate from the cached inputs with `sigmoid` reproduces it
    exactly, which is what keeps the cell and its framework port agreeing about
    what "sigma" means.
    """
    x, a0, parameters, _ = unpack(make_fixture())
    _, _, caches = gru_forward(x, parameters, a0)
    cache = caches[0]

    expected = sigmoid(parameters["Wr"] @ cache["concat"] + parameters["br"])
    np.testing.assert_array_equal(cache["gamma_r"], expected)
