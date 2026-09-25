"""
LSTM and GRU TensorFlow Parity Tests
====================================

The hub ships two gated cells from scratch in NumPy -- an LSTM and a GRU, both
with hand-derived backpropagation through time -- and one TensorFlow port built
on ``tf.keras.layers.LSTM``, ``tf.keras.layers.GRU``, and ``tf.GradientTape``.
This module is the proof they agree. The method is the topic, so a numeric
disagreement is a bug in one of them, not a difference of style; that is what
makes this a parity port rather than an idiom track.

On the dossier's shared float64 fixtures it checks two contracts per cell. The
forward pass must reproduce the reference's hidden states, output distributions,
and final states to ``atol=1e-8``. Then -- the contract that matters --
GradientTape's gradients must match the reference's analytic BPTT to a relative
error below ``1e-6`` for every gate weight and bias, including the initial
states and the input. A mis-packed gate, a dropped ``1/m``, or the GRU's
inverted update gate would push that error far above the tolerance rather than
fail silently at training time.

The suite also pins the three Keras conventions the port has to reconcile,
because each is a fact a practitioner needs rather than an implementation
detail: the GRU's ``reset_after`` default is a *different cell* and reproduces
nothing; the GRU's update gate is stored negated; and the LSTM's
``unit_forget_bias`` default is not the paper's initialization.

Parity is checked on fixture forward and gradient values, never on a training
curve, which drifts for reasons unrelated to correctness.

This test needs TensorFlow. In the framework-free hub CI job it is skipped at
collection; the cross-framework parity job installs TensorFlow and runs it for
real, where a skip would read as the failure it is.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

# Skip the whole module when TensorFlow is absent so the framework-free test job
# can collect it. The parity CI job installs TensorFlow as an explicit step, so
# there the import succeeds and every assertion below runs.
tf = pytest.importorskip("tensorflow")

from tensorflow import keras  # noqa: E402  (after the gate)

from dlhub.nn.sequence._common import split_gate_matrix  # noqa: E402
from dlhub.nn.sequence._gradient_check import relative_error  # noqa: E402
from dlhub.nn.sequence.gru import gru_backward, gru_forward  # noqa: E402
from dlhub.nn.sequence.gru import make_fixture as make_gru_fixture  # noqa: E402
from dlhub.nn.sequence.lstm import lstm_backward, lstm_forward  # noqa: E402
from dlhub.nn.sequence.lstm import make_fixture as make_lstm_fixture  # noqa: E402
from dlhub.nn.sequence.rnn import compute_loss  # noqa: E402
from dlhub.tensorflow.sequence.lstm_gru import (  # noqa: E402
    GRUSequenceModel,
    LSTMSequenceModel,
)

# The dossier's parity contract.
FORWARD_ATOL = 1e-8
GRAD_RELATIVE_TOLERANCE = 1e-6

LSTM_GRADIENT_KEYS = [
    "dWf",
    "dWu",
    "dWc",
    "dWo",
    "dbf",
    "dbu",
    "dbc",
    "dbo",
    "dWy",
    "dby",
    "da0",
    "dc0",
    "dx",
]
GRU_GRADIENT_KEYS = [
    "dWr",
    "dWu",
    "dWc",
    "dbr",
    "dbu",
    "dbc",
    "dWy",
    "dby",
    "da0",
    "dx",
]

# Measured on this fixture with the reference weights loaded and only the
# reset_after flag changed: the reset-after cell's hidden states differ from the
# reference's by 1.787 in max absolute error. The assertion threshold is set an
# order of magnitude below that and eight orders above FORWARD_ATOL, so it
# cannot be met by a cell that merely rounds differently.
RESET_AFTER_MISMATCH = 1.787
RESET_AFTER_FLOOR = 0.5


@pytest.fixture(scope="module")
def lstm_parity():
    """
    Run the NumPy LSTM and the Keras LSTM once on the shared fixture.

    Both are float64 -- the reference fixture is float64 and the gradient
    tolerance is far finer than float32 can carry -- so the port's layers are
    constructed with ``dtype="float64"`` rather than the process-wide
    ``set_floatx``, which would leak into every other model in the session.
    """
    fixture = make_lstm_fixture()
    x, a0, c0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["c0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    a_ref, c_ref, y_ref, caches = lstm_forward(x, parameters, a0, c0)
    ref_grads = lstm_backward(y, caches, parameters)
    ref_loss = compute_loss(y_ref, y)

    model = LSTMSequenceModel(dims["n_x"], dims["n_a"], dims["n_y"])
    model.load_reference_parameters(parameters)

    return {
        "dims": dims,
        "parameters": parameters,
        "x": x,
        "y": y,
        "a0": a0,
        "c0": c0,
        "a_ref": a_ref,
        "c_ref": c_ref,
        "y_ref": y_ref,
        "ref_grads": ref_grads,
        "ref_loss": ref_loss,
        "model": model,
        "forward": model.forward(x, a0, c0),
        "port": model.bptt_gradients(x, y, a0, c0),
    }


@pytest.fixture(scope="module")
def gru_parity():
    """
    Run the NumPy GRU and the Keras GRU once on the shared fixture.

    The port is built with ``reset_after=False``, which is *not* Keras's default
    and is the only setting under which the layer computes the Cho/Ng cell the
    reference derives.
    """
    fixture = make_gru_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    a_ref, y_ref, caches = gru_forward(x, parameters, a0)
    ref_grads = gru_backward(y, caches, parameters)
    ref_loss = compute_loss(y_ref, y)

    model = GRUSequenceModel(dims["n_x"], dims["n_a"], dims["n_y"])
    model.load_reference_parameters(parameters)

    return {
        "dims": dims,
        "parameters": parameters,
        "x": x,
        "y": y,
        "a0": a0,
        "a_ref": a_ref,
        "y_ref": y_ref,
        "ref_grads": ref_grads,
        "ref_loss": ref_loss,
        "model": model,
        "forward": model.forward(x, a0),
        "port": model.bptt_gradients(x, y, a0),
    }


# --- Forward parity ------------------------------------------------------


def test_lstm_forward_activations_match_the_reference(lstm_parity):
    """Hidden states from tf.keras.layers.LSTM reproduce the reference."""
    np.testing.assert_allclose(
        lstm_parity["forward"]["a"], lstm_parity["a_ref"], atol=FORWARD_ATOL
    )


def test_lstm_forward_outputs_match_the_reference(lstm_parity):
    """The softmax read-out reproduces the reference distributions."""
    np.testing.assert_allclose(
        lstm_parity["forward"]["y_pred"], lstm_parity["y_ref"], atol=FORWARD_ATOL
    )


def test_lstm_final_cell_state_matches_the_reference(lstm_parity):
    """
    The one cell state Keras exposes agrees with the reference's last one.

    ``return_state=True`` yields ``a^{<T_x>}`` and ``c^{<T_x>}`` and nothing
    else: the sequence of cell states the reference returns has no Keras
    equivalent, because Keras treats the additive cell path as internal. So the
    cell-state contract is checked where the framework lets it be checked. This
    would fail if the four gates were packed in the wrong order, since only the
    cell state distinguishes the forget gate from the input gate.
    """
    np.testing.assert_allclose(
        lstm_parity["forward"]["c_final"],
        lstm_parity["c_ref"][:, :, -1],
        atol=FORWARD_ATOL,
    )
    np.testing.assert_allclose(
        lstm_parity["forward"]["a_final"],
        lstm_parity["a_ref"][:, :, -1],
        atol=FORWARD_ATOL,
    )


def test_gru_forward_activations_match_the_reference(lstm_parity, gru_parity):
    """Hidden states from tf.keras.layers.GRU(reset_after=False) reproduce it."""
    np.testing.assert_allclose(
        gru_parity["forward"]["a"], gru_parity["a_ref"], atol=FORWARD_ATOL
    )


def test_gru_forward_outputs_match_the_reference(gru_parity):
    """The softmax read-out reproduces the reference distributions."""
    np.testing.assert_allclose(
        gru_parity["forward"]["y_pred"], gru_parity["y_ref"], atol=FORWARD_ATOL
    )


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_the_total_loss_matches_the_reference(cell, lstm_parity, gru_parity):
    """
    Each port forms the reference's summed-over-time, batch-mean cross-entropy.

    Matching the loss is the precondition for matching gradients: the tape
    differentiates whatever scalar it is handed, so a differently scaled loss
    would give proportionally wrong gradients that no weight mapping could fix.
    """
    parity = lstm_parity if cell == "lstm" else gru_parity
    assert parity["port"]["loss"] == pytest.approx(parity["ref_loss"], abs=1e-10)


# --- Gradient parity (the contract this port exists to prove) ------------


@pytest.mark.parametrize("key", LSTM_GRADIENT_KEYS)
def test_lstm_tape_gradient_matches_the_reference_bptt(lstm_parity, key):
    """
    Every GradientTape gradient agrees with the reference's analytic BPTT.

    This is half of what the port exists to pass. The tolerance is the
    dossier's ``1e-6`` relative error; a mis-packed gate, a dropped ``1/m``, or
    a forgotten transpose drives the error orders of magnitude above it.
    """
    error = relative_error(lstm_parity["port"][key], lstm_parity["ref_grads"][key])
    assert error < GRAD_RELATIVE_TOLERANCE, f"{key}: relative error {error:.2e}"


@pytest.mark.parametrize("key", GRU_GRADIENT_KEYS)
def test_gru_tape_gradient_matches_the_reference_bptt(gru_parity, key):
    """
    Every GradientTape gradient agrees with the reference's analytic BPTT.

    ``dWu`` and ``dbu`` are the interesting entries: Keras differentiates its
    own ``z = 1 - Gamma_u``, so the port has to negate those gradients to return
    the reference's. Dropping that flip leaves the forward pass correct and the
    update-gate gradient exactly wrong.
    """
    error = relative_error(gru_parity["port"][key], gru_parity["ref_grads"][key])
    assert error < GRAD_RELATIVE_TOLERANCE, f"{key}: relative error {error:.2e}"


@pytest.mark.parametrize(
    ("cell", "key"), [("lstm", "dWf"), ("gru", "dWu")], ids=["lstm", "gru"]
)
def test_a_flipped_sign_would_be_rejected(cell, key, lstm_parity, gru_parity):
    """
    The parity metric discriminates rather than passing everything.

    Negating a gate-weight gradient collapses its agreement with the reference
    to order one, confirming a real disagreement could not slip under the
    tolerance above. The GRU's ``dWu`` is chosen deliberately: it is the one
    gradient the port itself negates, so this is the test that would catch a
    double flip.
    """
    parity = lstm_parity if cell == "lstm" else gru_parity
    error = relative_error(-parity["port"][key], parity["ref_grads"][key])
    assert error > 0.9


# --- The documented convention divergences -------------------------------


def test_reset_after_true_does_not_reproduce_the_reference(gru_parity):
    """
    Keras's *default* GRU is a different cell, and the hub says so out loud.

    ``reset_after=True`` applies the reset gate after the recurrent transform,
    which is not algebraically equal to the Cho/Ng form the reference derives.
    Loading the identical reference weights and changing only that flag moves
    the hidden states by 1.787 in max absolute error -- against the 1e-8 the
    reset-after=False layer meets. The floor is 0.5, far below the measurement
    and far above anything rounding could explain, so this fails if a future
    Keras ever made the two forms agree (which would itself be news worth
    failing over).
    """
    dims, parameters = gru_parity["dims"], gru_parity["parameters"]
    reset_after = GRUSequenceModel(
        dims["n_x"], dims["n_a"], dims["n_y"], reset_after=True
    )
    reset_after.load_reference_parameters(parameters)

    diverged = reset_after.forward(gru_parity["x"], gru_parity["a0"])["a"]
    mismatch = float(np.max(np.abs(diverged - gru_parity["a_ref"])))

    assert mismatch > RESET_AFTER_FLOOR, f"measured mismatch {mismatch:.4f}"
    assert mismatch == pytest.approx(RESET_AFTER_MISMATCH, abs=1e-3)


def test_reset_after_true_refuses_to_report_reference_gradients(gru_parity):
    """
    A cell with no reference cannot claim a reference gradient.

    The port raises rather than returning numbers that look comparable, which is
    the failure mode this whole divergence invites.
    """
    dims, parameters = gru_parity["dims"], gru_parity["parameters"]
    reset_after = GRUSequenceModel(
        dims["n_x"], dims["n_a"], dims["n_y"], reset_after=True
    )
    reset_after.load_reference_parameters(parameters)

    with pytest.raises(ValueError, match="reset_after=True"):
        reset_after.bptt_gradients(gru_parity["x"], gru_parity["y"], gru_parity["a0"])


def test_the_gru_update_gate_is_stored_negated(gru_parity):
    """
    Keras puts its gate on the old state, so ``z = 1 - Gamma_u``.

    Both are logistic, so the reconciliation is a negated pre-activation:
    ``W_z = -W_u`` and ``b_z = -b_u``, in the first of the three packed blocks.
    Pinning the loaded weights documents the mapping at the one place a reader
    would otherwise have to infer it, and fails if the minus sign is ever
    "corrected" away.
    """
    model, parameters = gru_parity["model"], gru_parity["parameters"]
    n_a = gru_parity["dims"]["n_a"]

    recurrent_half, input_half = split_gate_matrix(parameters["Wu"], n_a)
    # Keras 3 keeps the weights on the layer's inner ``cell``, not the layer.
    cell = model.recurrent.cell
    kernel = np.asarray(cell.kernel)
    recurrent_kernel = np.asarray(cell.recurrent_kernel)
    bias = np.asarray(cell.bias)

    np.testing.assert_allclose(kernel[:, :n_a], -input_half.T, atol=1e-15)
    np.testing.assert_allclose(recurrent_kernel[:, :n_a], -recurrent_half.T, atol=1e-15)
    np.testing.assert_allclose(bias[:n_a], -parameters["bu"].reshape(-1), atol=1e-15)


def test_the_port_declines_the_keras_forget_bias_default(lstm_parity):
    """
    A freshly built Keras LSTM is not the paper's initialization.

    ``unit_forget_bias`` defaults to ``True``, which writes ones into the
    forget-gate slice of the bias so the cell starts out remembering
    (Jozefowicz, Zaremba & Sutskever, 2015). Good for training, invisible to
    anyone comparing against an equation. The port turns it off, so its
    untrained layer is the plain cell the reference describes. Both halves are
    asserted: the default's ones, and the port's zeros.
    """
    dims = lstm_parity["dims"]
    n_a = dims["n_a"]

    default = keras.layers.LSTM(n_a, dtype="float64")
    default.build((None, None, dims["n_x"]))
    default_bias = np.asarray(default.cell.bias)
    # Keras packs [i, f, c, o]; the forget block is the second of the four.
    np.testing.assert_allclose(default_bias[n_a : 2 * n_a], np.ones(n_a))
    np.testing.assert_allclose(default_bias[:n_a], np.zeros(n_a))

    port = LSTMSequenceModel(dims["n_x"], n_a, dims["n_y"])
    np.testing.assert_allclose(np.asarray(port.recurrent.cell.bias), np.zeros(4 * n_a))


# --- The framework idiom the port leans on -------------------------------


def test_graph_mode_changes_no_number(gru_parity):
    """
    ``loss_and_gradients`` is a ``tf.function``, and that is a free choice.

    The module claims graph compilation is where a TensorFlow gradient step
    belongs and that it changes no number here. Running the undecorated Python
    body against the traced one checks the second half of that claim, so the
    decorator cannot quietly start costing accuracy.
    """
    model = gru_parity["model"]
    x_tensor, state_tensors = model._to_keras(gru_parity["x"], [gru_parity["a0"]])
    y_tensor = tf.convert_to_tensor(
        np.transpose(gru_parity["y"], (1, 2, 0)), dtype=model.recurrent.compute_dtype
    )

    eager = model.loss_and_gradients.python_function(x_tensor, y_tensor, state_tensors)
    graph = model.loss_and_gradients(x_tensor, y_tensor, state_tensors)

    assert float(eager[0]) == pytest.approx(float(graph[0]), abs=1e-12)
    for eager_grad, graph_grad in zip(eager[4], graph[4], strict=True):
        np.testing.assert_allclose(
            np.asarray(eager_grad), np.asarray(graph_grad), atol=1e-12
        )
