"""
LSTM Reference Tests
====================

This module is the correctness proof for the hub's LSTM reference, and the
TensorFlow parity port is checked against it. The one failure it exists to catch
is a flipped sign in backpropagation through time: a sign error trains the model
uphill while every shape assertion still passes, so it teaches confidently wrong
math. With four gates there are four more places for one to hide than in the
vanilla RNN, so the gradient-agreement test pins every analytic gradient against
a central finite-difference approximation on the dossier's fixture, and a sign
flip drives the relative error to order one.

Two tests state the architecture's thesis as an assertion rather than a claim.
The cell-state Jacobian is checked to be exactly diagonal with the forget gate
on that diagonal -- an elementwise factor, not a repeated Jacobian product --
and an open forget gate is shown to deliver gradient to ``c^{<0>}`` that a shut
one destroys. The remaining contracts guard the fixture's determinism and the
shapes the topic promises, including the single-timestep edge.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.nn.sequence._gradient_check import numeric_gradient, relative_error
from dlhub.nn.sequence.lstm import (
    lstm_backward,
    lstm_cell_forward,
    lstm_forward,
    make_fixture,
)
from dlhub.nn.sequence.rnn import compute_loss

# The dossier's finite-difference contract: analytic-vs-numeric relative error
# below this tolerance, for every parameter.
FD_TOLERANCE = 1e-7

# The step is one decade larger than the shared default the vanilla RNN uses,
# and deliberately so. A central difference subtracts two float64 losses of
# order 5, so the roundoff floor on the relative error of a gradient `g` is
# about ulp(L) / (2 eps ||g||). At eps = 1e-7 the smallest gradients on this
# fixture sit at 9e-8 -- under the tolerance, but with no margin, which is a
# test that would go red on another BLAS rather than on a real defect. At 1e-6
# the floor drops tenfold, the truncation error it trades against stays near
# 1e-12, and every gradient below lands at or under 1e-8.
FD_EPSILON = 1e-6

# Each analytic gradient the dossier names, paired with the fixture tensor it
# differentiates and the key it is returned under. Written out rather than
# derived from the module's own key list, so dropping a gate there cannot
# quietly drop it from the check.
GRADIENT_TARGETS = [
    ("Wf", "dWf"),
    ("Wu", "dWu"),
    ("Wc", "dWc"),
    ("Wo", "dWo"),
    ("Wy", "dWy"),
    ("bf", "dbf"),
    ("bu", "dbu"),
    ("bc", "dbc"),
    ("bo", "dbo"),
    ("by", "dby"),
    ("a0", "da0"),
    ("c0", "dc0"),
    ("x", "dx"),
]


def loss_closure(x, parameters, a0, c0, y):
    """
    A zero-argument loss over the given arrays, for the finite-difference check.

    Every array is read in place on each call, so perturbing an entry of any one
    of them and re-calling is what moves the loss.
    """

    def loss() -> float:
        _, _, y_pred, _ = lstm_forward(x, parameters, a0, c0)
        return compute_loss(y_pred, y)

    return loss


def unpack(fixture):
    """The five arrays every test below starts from."""
    return (
        fixture["x"],
        fixture["a0"],
        fixture["c0"],
        fixture["parameters"],
        fixture["y"],
    )


# --- Gradient agreement (the sign-flip catcher) --------------------------


@pytest.mark.parametrize("tensor_name, grad_key", GRADIENT_TARGETS)
def test_bptt_gradient_matches_finite_difference(tensor_name, grad_key):
    """
    Every analytic BPTT gradient agrees with central differences on the fixture.

    This is the test the reference exists to pass. A dropped term, a missing
    1/m, a gate paired with the wrong pre-activation, or -- the worst case -- a
    flipped sign, all push the relative error far above the tolerance instead of
    failing silently at training time.
    """
    x, a0, c0, parameters, y = unpack(make_fixture())

    _, _, _, caches = lstm_forward(x, parameters, a0, c0)
    analytic = lstm_backward(y, caches, parameters)

    tensor = {**parameters, "a0": a0, "c0": c0, "x": x}[tensor_name]
    numeric = numeric_gradient(
        loss_closure(x, parameters, a0, c0, y), tensor, FD_EPSILON
    )

    error = relative_error(analytic[grad_key], numeric)
    assert error < FD_TOLERANCE, f"{grad_key}: relative error {error:.2e}"


def test_a_flipped_sign_is_rejected():
    """
    The defect a hub must never ship. If the sign of the forget gate's gradient
    were flipped, its agreement with finite differences would collapse --
    confirming the check discriminates, rather than passing everything.
    """
    x, a0, c0, parameters, y = unpack(make_fixture())
    _, _, _, caches = lstm_forward(x, parameters, a0, c0)
    analytic = lstm_backward(y, caches, parameters)

    numeric = numeric_gradient(
        loss_closure(x, parameters, a0, c0, y), parameters["Wf"], FD_EPSILON
    )
    assert relative_error(-analytic["dWf"], numeric) > 0.9


# --- The thesis: the additive cell path ----------------------------------


def test_the_cell_state_jacobian_is_the_forget_gate():
    """
    The cell-state Jacobian is the elementwise factor Gamma_f, and nothing else.

    That is, ``d c^{<t>} / d c^{<t-1>} = Gamma_f^{<t>}``.

    This is the whole argument for the architecture, measured rather than
    asserted in prose. The Jacobian of one cell step with respect to the
    incoming cell state is computed column by column with central differences:
    it comes out *diagonal* -- unit i of c^{<t-1>} reaches unit i of c^{<t>} and
    no other -- with the forget gate on the diagonal. Contrast the vanilla RNN,
    where the same Jacobian is a dense `W_aa^T` scaled by a tanh derivative and
    its repeated product decays geometrically.
    """
    x, a0, c0, parameters, _ = unpack(make_fixture())
    n_a = parameters["Wf"].shape[0]
    example = 3  # any column of the batch; they are independent

    _, _, _, cache = lstm_cell_forward(x[:, :, 0], a0, c0, parameters)

    jacobian = np.zeros((n_a, n_a))
    for i in range(n_a):
        c_plus = c0.copy()
        c_plus[i, example] += FD_EPSILON
        _, c_next_plus, _, _ = lstm_cell_forward(x[:, :, 0], a0, c_plus, parameters)

        c_minus = c0.copy()
        c_minus[i, example] -= FD_EPSILON
        _, c_next_minus, _, _ = lstm_cell_forward(x[:, :, 0], a0, c_minus, parameters)

        jacobian[:, i] = (c_next_plus[:, example] - c_next_minus[:, example]) / (
            2 * FD_EPSILON
        )

    off_diagonal = jacobian - np.diag(np.diag(jacobian))
    assert np.max(np.abs(off_diagonal)) == 0.0, "the cell path must not mix units"
    np.testing.assert_allclose(
        np.diag(jacobian), cache["gamma_f"][:, example], atol=1e-8
    )


def test_an_open_forget_gate_carries_gradient_a_shut_one_destroys():
    """
    The same network, twice, differing only in the forget gate's bias.

    With the gate open (Gamma_f = 0.9997) the gradient reaching the initial cell
    state is four orders of magnitude larger than with it shut (Gamma_f =
    0.0003), because the shut gate multiplies the carried gradient by 3e-4 at
    every one of the T_x steps. This is the vanishing-gradient mechanism the
    topic is about, made to happen and not happen on demand.
    """
    x, a0, c0, parameters, y = unpack(make_fixture())

    def dc0_norm(forget_bias: float) -> float:
        tuned = {key: value.copy() for key, value in parameters.items()}
        tuned["Wf"] = np.zeros_like(tuned["Wf"])  # gate depends on the bias alone
        tuned["bf"] = np.full_like(tuned["bf"], forget_bias)
        _, _, _, caches = lstm_forward(x, tuned, a0, c0)
        return float(np.linalg.norm(lstm_backward(y, caches, tuned)["dc0"]))

    assert dc0_norm(+8.0) > 100 * dc0_norm(-8.0)


# --- Determinism ---------------------------------------------------------


def test_the_seeded_fixture_reproduces_exactly():
    """The pinned draw order under seed(1) must give bit-identical arrays."""
    first, second = make_fixture(), make_fixture()
    for key in ("x", "a0", "c0", "y"):
        np.testing.assert_array_equal(first[key], second[key])
    for key in first["parameters"]:
        np.testing.assert_array_equal(
            first["parameters"][key], second["parameters"][key]
        )


def test_the_forward_trajectory_is_deterministic():
    """Re-running the forward pass on one fixture yields the same trajectory."""
    x, a0, c0, parameters, _ = unpack(make_fixture())
    a1, c1, y1, _ = lstm_forward(x, parameters, a0, c0)
    a2, c2, y2, _ = lstm_forward(x, parameters, a0, c0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(y1, y2)


# --- Shape and edge contracts --------------------------------------------


def test_forward_produces_the_declared_shapes():
    """Hidden and cell states are (n_a, m, T_x); outputs are (n_y, m, T_x)."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, a0, c0, parameters, _ = unpack(fixture)

    a, c, y_pred, caches = lstm_forward(x, parameters, a0, c0)

    assert a.shape == (d["n_a"], d["m"], d["T_x"])
    assert c.shape == (d["n_a"], d["m"], d["T_x"])
    assert y_pred.shape == (d["n_y"], d["m"], d["T_x"])
    assert len(caches) == d["T_x"]


def test_backward_gradients_match_their_parameter_shapes():
    """Each gradient carries the shape of the thing it differentiates."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, a0, c0, parameters, y = unpack(fixture)

    _, _, _, caches = lstm_forward(x, parameters, a0, c0)
    grads = lstm_backward(y, caches, parameters)

    for key, value in parameters.items():
        assert grads[f"d{key}"].shape == value.shape
    assert grads["da0"].shape == (d["n_a"], d["m"])
    assert grads["dc0"].shape == (d["n_a"], d["m"])
    assert grads["dx"].shape == (d["n_x"], d["m"], d["T_x"])


def test_the_sequence_loop_agrees_with_stepping_the_cell_by_hand():
    """
    lstm_forward is exactly the cell applied T_x times with both states threaded.

    Pinning this keeps the unrolled loop honest: a state dropped or carried from
    the wrong step would still produce arrays of the right shape.
    """
    x, a0, c0, parameters, _ = unpack(make_fixture())
    a, c, y_pred, _ = lstm_forward(x, parameters, a0, c0)

    a_prev, c_prev = a0, c0
    for t in range(x.shape[2]):
        a_next, c_next, y_t, _ = lstm_cell_forward(
            x[:, :, t], a_prev, c_prev, parameters
        )
        np.testing.assert_array_equal(a[:, :, t], a_next)
        np.testing.assert_array_equal(c[:, :, t], c_next)
        np.testing.assert_array_equal(y_pred[:, :, t], y_t)
        a_prev, c_prev = a_next, c_next


def test_the_gates_are_probabilities_and_the_readout_is_a_distribution():
    """Every gate lies strictly in (0, 1); every output column sums to one."""
    x, a0, c0, parameters, _ = unpack(make_fixture())
    _, _, y_pred, caches = lstm_forward(x, parameters, a0, c0)

    for cache in caches:
        for gate in ("gamma_f", "gamma_u", "gamma_o"):
            assert np.all(cache[gate] > 0.0) and np.all(cache[gate] < 1.0)
        assert np.all(np.abs(cache["c_tilde"]) < 1.0)  # tanh, not sigmoid

    np.testing.assert_allclose(np.sum(y_pred, axis=0), 1.0, atol=1e-12)


def test_the_default_initial_states_are_the_zero_vectors():
    """Omitting a0 and c0 matches passing explicit zeros (a^{<0>} = c^{<0>} = 0)."""
    fixture = make_fixture()
    d = fixture["dims"]
    x, parameters = fixture["x"], fixture["parameters"]
    zeros = np.zeros((d["n_a"], d["m"]))

    a_default, c_default, y_default, _ = lstm_forward(x, parameters)
    a_zeros, c_zeros, y_zeros, _ = lstm_forward(x, parameters, zeros, zeros.copy())

    np.testing.assert_array_equal(a_default, a_zeros)
    np.testing.assert_array_equal(c_default, c_zeros)
    np.testing.assert_array_equal(y_default, y_zeros)


def test_a_single_timestep_sequence_works():
    """
    T_x = 1 is the degenerate edge: neither recurrence has a successor, so both
    da_next and dc_next start and stay zero. Shapes and gradient agreement must
    still hold.
    """
    fixture = make_fixture()
    d = fixture["dims"]
    x1, y1 = fixture["x"][:, :, :1], fixture["y"][:, :, :1]
    a0, c0, parameters = fixture["a0"], fixture["c0"], fixture["parameters"]

    a, c, y_pred, caches = lstm_forward(x1, parameters, a0, c0)
    assert a.shape == (d["n_a"], d["m"], 1)
    assert c.shape == (d["n_a"], d["m"], 1)
    assert y_pred.shape == (d["n_y"], d["m"], 1)

    analytic = lstm_backward(y1, caches, parameters)
    numeric = numeric_gradient(
        loss_closure(x1, parameters, a0, c0, y1), parameters["Wf"], FD_EPSILON
    )
    assert relative_error(analytic["dWf"], numeric) < FD_TOLERANCE


def test_lstm_forward_rejects_a_two_dimensional_input():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        lstm_forward(fixture["x"][:, :, 0], fixture["parameters"])


def test_lstm_forward_rejects_a_mismatched_initial_hidden_state():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        lstm_forward(fixture["x"], fixture["parameters"], np.zeros((99, 10)))


def test_lstm_forward_rejects_a_mismatched_initial_cell_state():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        lstm_forward(
            fixture["x"], fixture["parameters"], fixture["a0"], np.zeros((99, 10))
        )
