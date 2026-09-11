"""
Vanilla RNN PyTorch Parity Tests
================================

The hub ships two RNNs -- a from-scratch NumPy reference with hand-derived
backpropagation through time, and a PyTorch port built on ``torch.nn.RNN`` and
autograd. This module is the proof they agree. The method is the topic, so a
numeric disagreement is a bug in one of them, not a difference of style; that is
what makes this a parity port rather than an idiom track.

On the dossier's shared float64 fixture it checks two contracts. The forward
pass must reproduce the reference's hidden states and output distributions to
``atol=1e-8``. Then -- the contract that matters -- PyTorch's autograd gradients
must match the reference's analytic BPTT to a relative error below ``1e-6`` for
every parameter, including the initial state and the input. A dropped ``1/m``, a
mis-mapped weight, or the two-bias wiring gotcha would push that error far above
the tolerance rather than fail silently at training time.

Parity is checked on fixture forward and gradient values, never on a training
curve, which drifts for reasons unrelated to correctness.

This test needs PyTorch. In the framework-free hub CI job it is skipped at
collection; the cross-framework parity job installs torch and runs it for real,
where a skip would read as the failure it is.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

# Skip the whole module when torch is absent so the framework-free test job can
# collect it. The parity CI job installs torch as an explicit step, so there the
# import succeeds and every assertion below runs.
torch = pytest.importorskip("torch")

from dlhub.nn.sequence.rnn import (  # noqa: E402  (after the torch gate)
    compute_loss,
    make_fixture,
    rnn_backward,
    rnn_forward,
)
from dlhub.pytorch.sequence.rnn import (  # noqa: E402  (after the torch gate)
    VanillaRNN,
    bptt_gradients,
    load_reference_parameters,
)

# The dossier's parity contract.
FORWARD_ATOL = 1e-8
GRAD_RELATIVE_TOLERANCE = 1e-6

# The reference's analytic gradients, each keyed as both modules name it.
GRADIENT_KEYS = ["dWax", "dWaa", "dWya", "dba", "dby", "da0", "dx"]


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Norm-based relative difference, the metric the hub's gradient check uses."""
    numerator = np.linalg.norm(analytic - numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    return 0.0 if denominator == 0 else float(numerator / denominator)


@pytest.fixture(scope="module")
def parity():
    """
    Run both implementations once on the shared fixture and collect their results.

    The port is moved to float64 before its parameters are loaded, because the
    reference fixture is float64 and the gradient tolerance is far finer than
    float32 can carry. ``bptt_gradients`` populates the module's ``.grad`` fields,
    so the whole comparison is computed here once and shared across the tests.
    """
    fixture = make_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    a_ref, y_ref, caches = rnn_forward(x, parameters, a0)
    ref_grads = rnn_backward(y, caches, parameters)
    ref_loss = compute_loss(y_ref, y)

    model = VanillaRNN(dims["n_x"], dims["n_a"], dims["n_y"]).double()
    load_reference_parameters(model, parameters)
    port = bptt_gradients(model, x, y, a0)

    return {
        "dims": dims,
        "parameters": parameters,
        "a_ref": a_ref,
        "y_ref": y_ref,
        "ref_grads": ref_grads,
        "ref_loss": ref_loss,
        "model": model,
        "port": port,
    }


# --- Forward parity ------------------------------------------------------


def test_forward_activations_match_the_reference(parity):
    """Hidden states from torch.nn.RNN reproduce the reference to atol 1e-8."""
    np.testing.assert_allclose(parity["port"]["a"], parity["a_ref"], atol=FORWARD_ATOL)


def test_forward_outputs_match_the_reference(parity):
    """The softmax read-out reproduces the reference distributions to atol 1e-8."""
    np.testing.assert_allclose(
        parity["port"]["y_pred"], parity["y_ref"], atol=FORWARD_ATOL
    )


def test_the_total_loss_matches_the_reference(parity):
    """
    The port forms the reference's summed-over-time, batch-mean cross-entropy.

    Matching the loss is the precondition for matching gradients: autograd
    differentiates whatever scalar it is handed, so a differently scaled loss
    would give proportionally wrong gradients that no weight mapping could fix.
    """
    assert parity["port"]["loss"] == pytest.approx(parity["ref_loss"], abs=1e-10)


# --- Gradient parity (the contract this port exists to prove) ------------


@pytest.mark.parametrize("key", GRADIENT_KEYS)
def test_autograd_gradient_matches_the_reference_bptt(parity, key):
    """
    Every autograd gradient agrees with the reference's analytic BPTT.

    This is the test the port exists to pass. The tolerance is the dossier's
    ``1e-6`` relative error; a mis-mapped weight, a dropped ``1/m``, or the
    two-bias wiring gotcha drives the error orders of magnitude above it.
    """
    error = relative_error(parity["port"][key], parity["ref_grads"][key])
    assert error < GRAD_RELATIVE_TOLERANCE, f"{key}: relative error {error:.2e}"


def test_a_flipped_sign_would_be_rejected(parity):
    """
    The parity metric discriminates rather than passing everything.

    Negating the port's recurrent-weight gradient collapses its agreement with
    the reference to order one, confirming a real disagreement could not slip
    under the tolerance above.
    """
    error = relative_error(-parity["port"]["dWaa"], parity["ref_grads"]["dWaa"])
    assert error > 0.9


# --- The documented convention divergence --------------------------------


def test_the_single_bias_is_folded_into_the_input_term(parity):
    """
    nn.RNN carries two bias vectors where Ng's cell carries one.

    The parity harness maps ``b_a`` onto ``bias_ih`` and zeros ``bias_hh``; the
    forward matches because the two only ever appear as their sum. Pinning this
    documents the divergence a first PyTorch RNN gets wrong.
    """
    model, dims = parity["model"], parity["dims"]
    b_a = parity["parameters"]["ba"].reshape(-1)

    np.testing.assert_allclose(model.rnn.bias_ih_l0.detach().numpy(), b_a, atol=1e-12)
    np.testing.assert_array_equal(
        model.rnn.bias_hh_l0.detach().numpy(), np.zeros(dims["n_a"])
    )
