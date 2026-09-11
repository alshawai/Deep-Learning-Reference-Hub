"""
Recurrent Neural Network (Vanilla RNN) -- PyTorch parity port
=============================================================

The hub's from-scratch RNN, rewritten the way it is really written in PyTorch: a
``torch.nn.RNN`` tanh cell for the recurrence and an ``nn.Linear`` + softmax
read-out on top, with autograd standing in for the hand-derived backpropagation
through time of the reference module.

This is a **parity port**. It exists to show the correct wiring and to prove, on
the topic's shared fixture, that PyTorch's autograd reproduces the reference's
hand-computed gradients to a relative error below ``1e-6``. The from-scratch
derivation, the notation, and the shared fixture live in
:mod:`dlhub.nn.sequence.rnn`, which this port imports and never contradicts.

Three conventions of ``torch.nn.RNN`` diverge from the hub reference (Andrew
Ng's Course 5 notation). Each is the wiring a first PyTorch RNN gets wrong, so
each is called out where it bites:

1. **Two bias vectors.** ``nn.RNN`` computes
   ``h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`` -- one bias on the input
   term and one on the recurrent term -- whereas Ng's cell carries the single
   ``b_a``. :func:`load_reference_parameters` folds ``b_a`` into ``b_ih`` and
   zeros ``b_hh``; the forward is reproduced exactly because the two biases only
   ever appear as their sum.
2. **No read-out.** ``nn.RNN`` emits hidden states only. The softmax classifier
   ``y_hat`` is a separate ``nn.Linear`` the caller attaches, carrying ``W_ya``
   and ``b_y``.
3. **Shape order.** ``nn.RNN`` speaks ``(T_x, m, features)`` -- sequence, batch,
   feature -- while the hub reference speaks ``(features, m, T_x)``. This
   module's :meth:`VanillaRNN.forward` uses PyTorch's native order;
   :func:`bptt_gradients` bridges to the reference order so the parity test can
   compare against the reference arrays.

References
----------
- Elman, J. L. (1990). Finding Structure in Time. *Cognitive Science*, 14(2),
  179-211.
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 1. https://www.coursera.org/specializations/deep-learning
- PyTorch. torch.nn.RNN.
  https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Parity is on gradients, in float64.** float32 is too coarse for the
  reference's finite-difference contract, so the parity harness loads the fixture
  as ``float64`` and moves the module to double precision (``.double()``) before
  copying parameters in.
- **Softmax then log, not** ``log_softmax``. The read-out returns an explicit
  probability distribution ``y_hat`` (the topic wants ``y_hat`` visible), so the
  loss is ``-(y * log(y_hat)).sum() / m`` to mirror the reference's
  ``compute_loss`` exactly. PyTorch's ``softmax`` is already max-shifted, so this
  is stable at the fixture's scale; a training port would prefer
  ``cross_entropy`` on the raw logits instead.
"""

import numpy as np
import torch
from torch import nn


class VanillaRNN(nn.Module):
    """
    The hub's vanilla RNN, written the PyTorch way.

    A ``torch.nn.RNN`` (tanh nonlinearity) carries the recurrence
    ``h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)``, and an ``nn.Linear``
    followed by ``softmax`` forms the per-timestep read-out ``y_hat``. See the
    module docstring for the three conventions in which ``nn.RNN`` diverges from
    the hub reference's notation.

    Parameters
    ----------
    n_x : int
        Input dimension (features per timestep).
    n_a : int
        Hidden-state dimension.
    n_y : int
        Output dimension (number of classes).

    Attributes
    ----------
    rnn : torch.nn.RNN
        The tanh recurrence. Its ``weight_ih_l0`` maps to ``W_ax``,
        ``weight_hh_l0`` to ``W_aa``, and ``bias_ih_l0`` to ``b_a`` (with
        ``bias_hh_l0`` held at zero).
    readout : torch.nn.Linear
        The softmax classifier's affine map, carrying ``W_ya`` and ``b_y``.
    """

    def __init__(self, n_x: int, n_a: int, n_y: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=n_x, hidden_size=n_a, nonlinearity="tanh")
        self.readout = nn.Linear(n_a, n_y)

    def forward(
        self, x: torch.Tensor, a0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation through time, in PyTorch's native shape order.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(T_x, m, n_x)`` -- sequence, batch, feature
            (PyTorch's default, ``batch_first=False``).
        a0 : torch.Tensor or None, optional
            Initial hidden state ``a^{<0>}`` of shape ``(1, m, n_a)``. Defaults to
            ``None``, which ``nn.RNN`` treats as the zero state, matching the
            reference convention ``a^{<0>} = 0``.

        Returns
        -------
        y_hat : torch.Tensor
            Per-timestep output distributions, shape ``(T_x, m, n_y)``.
        a : torch.Tensor
            All hidden states, shape ``(T_x, m, n_a)``.
        """
        a, _ = self.rnn(x, a0)
        logits = self.readout(a)
        y_hat = torch.softmax(logits, dim=-1)
        return y_hat, a


def load_reference_parameters(
    model: VanillaRNN, parameters: dict[str, np.ndarray]
) -> None:
    """
    Copy the hub reference's weights into the module, reconciling the biases.

    Maps Ng's notation onto ``nn.RNN`` and ``nn.Linear`` in place:
    ``W_ax -> weight_ih_l0``, ``W_aa -> weight_hh_l0``, ``W_ya -> readout.weight``,
    ``b_y -> readout.bias``. The single ``b_a`` is folded into ``bias_ih_l0`` and
    ``bias_hh_l0`` is zeroed -- the two-bias divergence (see module docstring).
    Each array is cast to the module's own dtype and device, so calling
    ``model.double()`` first is enough to run the parity check in float64.

    Parameters
    ----------
    model : VanillaRNN
        The module to load into. Modified in place.
    parameters : dict[str, np.ndarray]
        The reference weights ``{"Wax", "Waa", "Wya", "ba", "by"}`` in the hub's
        shape convention (weights ``(out, in)``, biases ``(out, 1)``).
    """

    def like(array: np.ndarray, target: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(array), dtype=target.dtype, device=target.device
        )

    rnn = model.rnn
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(like(parameters["Wax"], rnn.weight_ih_l0))
        rnn.weight_hh_l0.copy_(like(parameters["Waa"], rnn.weight_hh_l0))
        # b_a on the input term; the recurrent-term bias is held at zero so the
        # two torch biases sum to Ng's single b_a.
        rnn.bias_ih_l0.copy_(like(parameters["ba"].reshape(-1), rnn.bias_ih_l0))
        rnn.bias_hh_l0.zero_()
        model.readout.weight.copy_(like(parameters["Wya"], model.readout.weight))
        model.readout.bias.copy_(like(parameters["by"].reshape(-1), model.readout.bias))


def bptt_gradients(
    model: VanillaRNN,
    x: np.ndarray,
    y: np.ndarray,
    a0: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Autograd BPTT, returned in the hub reference's shape convention.

    Runs the forward pass, forms the reference's total loss
    ``-(y * log(y_hat)).sum() / m`` (summed over timesteps, mean over the batch),
    and backpropagates. The inputs are given feature-first, exactly as the
    reference and its fixture produce them; this function bridges to and from
    PyTorch's ``(T_x, m, feature)`` order so the parity test compares like with
    like.

    Parameters
    ----------
    model : VanillaRNN
        A module whose parameters are already loaded (see
        :func:`load_reference_parameters`) and, for a float64 parity check, moved
        to double precision.
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    y : np.ndarray
        One-hot targets, shape ``(n_y, m, T_x)``.
    a0 : np.ndarray
        Initial hidden state ``a^{<0>}``, shape ``(n_a, m)``.

    Returns
    -------
    dict[str, np.ndarray]
        The autograd gradients keyed as the reference names them -- ``dWax``,
        ``dWaa``, ``dWya``, ``dba``, ``dby``, ``da0``, ``dx`` -- each in the hub's
        shape convention, together with the scalar ``loss`` and the forward
        ``a`` ``(n_a, m, T_x)`` and ``y_pred`` ``(n_y, m, T_x)`` for the
        forward-parity check.
    """
    dtype = model.readout.weight.dtype
    _, m, _ = x.shape

    # Bridge feature-first (reference) to sequence-first (PyTorch).
    x_t = torch.tensor(
        np.ascontiguousarray(np.transpose(x, (2, 1, 0))),
        dtype=dtype,
        requires_grad=True,
    )
    a0_t = torch.tensor(
        np.ascontiguousarray(a0.T)[None], dtype=dtype, requires_grad=True
    )
    y_t = torch.tensor(np.ascontiguousarray(np.transpose(y, (2, 1, 0))), dtype=dtype)

    y_hat, a = model(x_t, a0_t)
    # The reference's total loss: cross-entropy summed over timesteps, averaged
    # over the batch. Matching it exactly is what makes the gradients comparable.
    loss = -(y_t * torch.log(y_hat)).sum() / m
    loss.backward()

    return {
        "dWax": model.rnn.weight_ih_l0.grad.detach().numpy(),
        "dWaa": model.rnn.weight_hh_l0.grad.detach().numpy(),
        "dWya": model.readout.weight.grad.detach().numpy(),
        # b_ih carries the whole pre-activation gradient, which is Ng's db_a.
        "dba": model.rnn.bias_ih_l0.grad.detach().numpy().reshape(-1, 1),
        "dby": model.readout.bias.grad.detach().numpy().reshape(-1, 1),
        "da0": a0_t.grad.detach().numpy()[0].T,
        "dx": np.transpose(x_t.grad.detach().numpy(), (2, 1, 0)),
        "loss": float(loss.detach()),
        "a": np.transpose(a.detach().numpy(), (2, 1, 0)),
        "y_pred": np.transpose(y_hat.detach().numpy(), (2, 1, 0)),
    }


def main() -> None:
    """Load the shared fixture, run the port, and print a parity report."""
    from dlhub.nn.sequence.rnn import (
        compute_loss,
        make_fixture,
        rnn_backward,
        rnn_forward,
    )

    fixture = make_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    # Reference: hand-derived forward and BPTT.
    _, y_ref, caches = rnn_forward(x, parameters, a0)
    ref_grads = rnn_backward(y, caches, parameters)
    ref_loss = compute_loss(y_ref, y)

    # Port: torch.nn.RNN + autograd, in double precision for the gradient check.
    model = VanillaRNN(dims["n_x"], dims["n_a"], dims["n_y"]).double()
    load_reference_parameters(model, parameters)
    port = bptt_gradients(model, x, y, a0)

    print(f"loss (reference / port): {ref_loss:.10f} / {port['loss']:.10f}")
    for key in ("dWax", "dWaa", "dWya", "dba", "dby", "da0", "dx"):
        analytic, autograd = ref_grads[key], port[key]
        num = np.linalg.norm(analytic - autograd)
        den = np.linalg.norm(analytic) + np.linalg.norm(autograd)
        rel = 0.0 if den == 0 else num / den
        print(f"{key:>5}: relative error {rel:.2e}")


if __name__ == "__main__":
    main()
