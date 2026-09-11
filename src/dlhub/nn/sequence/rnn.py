"""
Recurrent Neural Network (Vanilla RNN)
======================================

A from-scratch, teaching reference for the simple recurrent network: the
recurrence relation, forward propagation through time, and backpropagation
through time (BPTT). This module is the explanation the accompanying
documents point at, so the forward pass appears in the shapes the topic
declares and the gradients are computed term by term rather than collapsed
into a single fused expression.

Notation follows Andrew Ng's Deep Learning Specialization, Course 5, Week 1,
verbatim, so the sibling sequence topics (LSTM/GRU, language modeling) extend
the same symbols. For ``t = 1 ... T_x``::

    a^{<t>}    = tanh(W_aa a^{<t-1>} + W_ax x^{<t>} + b_a)
    y_hat^{<t>} = softmax(W_ya a^{<t>} + b_y)

with ``a^{<0>}`` the initial hidden state (the zero vector by default). The
total loss is summed over timesteps, ``L = sum_t L^{<t>}``, with each ``L^{<t>}``
the softmax cross-entropy of ``y_hat^{<t>}`` against the target ``y^{<t>}``.

This implements the **many-to-many, equal-length** case (one output per input
step). The other input/output cardinalities -- one-to-one, one-to-many,
many-to-one, many-to-many with unequal lengths -- reuse this exact cell and
differ only in which steps receive an input and which produce a read-out; they
need no new cell mathematics. Bidirectional and deep (stacked) RNNs are
provided below as *compositions* of this cell, shown as forward-pass shape
walkthroughs rather than re-derived with their own BPTT. Gated cells (GRU,
LSTM), the answer to the vanishing-gradient problem stated here, live in the
``lstm-and-gru`` topic.

References
----------
- Elman, J. L. (1990). Finding Structure in Time. *Cognitive Science*, 14(2),
  179-211.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
  representations by back-propagating errors. *Nature*, 323, 533-536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 10.
  MIT Press. https://www.deeplearningbook.org/
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training
  Recurrent Neural Networks. ICML. https://arxiv.org/abs/1211.5063
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural
  Networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 1. https://www.coursera.org/specializations/deep-learning

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Stable softmax.** The textbook softmax ``exp(z) / sum(exp(z))`` overflows
  for large logits. This module subtracts the per-column maximum before
  exponentiating, which is algebraically identical (the shift cancels in the
  ratio) but keeps every exponent ``<= 0``. Because the shifted denominator is
  ``>= 1``, the output is strictly positive, so the cross-entropy ``log`` is
  taken without any epsilon guard -- and the analytic gradient
  ``(y_hat - y) / m`` stays exact for the finite-difference check.
- **The tanh backward** uses ``1 - tanh(z)^2 = 1 - (a^{<t>})^2``: since
  ``a^{<t>}`` is already the tanh output, its derivative is read straight off
  the cached activation with no second tanh evaluation.
- **Stacked weight equivalence.** Ng also writes the recurrence with a single
  weight ``W_a = [W_aa | W_ax]`` acting on the concatenation
  ``[a^{<t-1>}; x^{<t>}]``. This module keeps ``W_aa`` and ``W_ax`` separate so
  that the two weight-gradient accumulations -- the recurrent path ``dW_aa`` and
  the input path ``dW_ax`` -- appear as distinct terms, which is the BPTT lesson.
  The two forms are numerically identical, and PyTorch's ``nn.RNN`` likewise
  keeps ``weight_hh`` and ``weight_ih`` separate.
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Column-wise softmax with the max-subtraction stability shift.

    Normalizes over ``axis=0`` (the class/feature axis), so each column is a
    probability distribution that sums to one. See the module ``Notes`` for why
    the maximum is subtracted first.

    Parameters
    ----------
    z : np.ndarray
        Logits of shape ``(n_y, m)``.

    Returns
    -------
    np.ndarray
        Probabilities of shape ``(n_y, m)``, strictly positive, columns summing
        to one.
    """
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def rnn_cell_forward(
    x_t: np.ndarray, a_prev: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    One timestep of the forward recurrence.

    Computes the hidden-state update and the softmax read-out for a single step::

        a^{<t>}     = tanh(W_aa a^{<t-1>} + W_ax x^{<t>} + b_a)
        y_hat^{<t>} = softmax(W_ya a^{<t>} + b_y)

    Parameters
    ----------
    x_t : np.ndarray
        Input at this timestep, shape ``(n_x, m)``.
    a_prev : np.ndarray
        Hidden state carried in from the previous step, ``a^{<t-1>}``, shape
        ``(n_a, m)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases: ``Wax`` ``(n_a, n_x)``, ``Waa``
        ``(n_a, n_a)``, ``Wya`` ``(n_y, n_a)``, ``ba`` ``(n_a, 1)``, ``by``
        ``(n_y, 1)``.

    Returns
    -------
    a_next : np.ndarray
        Updated hidden state ``a^{<t>}``, shape ``(n_a, m)``.
    y_pred_t : np.ndarray
        Output distribution ``y_hat^{<t>}``, shape ``(n_y, m)``.
    cache : tuple
        ``(a_next, a_prev, x_t, y_pred_t)`` -- everything the backward step for
        this timestep needs.
    """
    Wax, Waa, Wya = parameters["Wax"], parameters["Waa"], parameters["Wya"]
    ba, by = parameters["ba"], parameters["by"]

    # Hidden-state update: the recurrent term W_aa a^{<t-1>} and the input term
    # W_ax x^{<t>} are added, then squashed by tanh.
    a_next = np.tanh(Waa @ a_prev + Wax @ x_t + ba)

    # Read-out: an affine map of the hidden state through softmax.
    y_pred_t = softmax(Wya @ a_next + by)

    cache = (a_next, a_prev, x_t, y_pred_t)
    return a_next, y_pred_t, cache


def rnn_forward(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Forward propagation through time across a whole sequence.

    Unrolls :func:`rnn_cell_forward` over ``T_x`` timesteps, threading the
    hidden state from one step to the next.

    Parameters
    ----------
    x : np.ndarray
        Input sequence of shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`rnn_cell_forward`).
    a0 : np.ndarray or None, optional
        Initial hidden state ``a^{<0>}`` of shape ``(n_a, m)``. Defaults to the
        zero vector, matching the standard convention ``a^{<0>} = 0``.

    Returns
    -------
    a : np.ndarray
        All hidden states stacked over time, shape ``(n_a, m, T_x)``.
    y_pred : np.ndarray
        All output distributions, shape ``(n_y, m, T_x)``.
    caches : list[tuple]
        Per-timestep caches, in order, for :func:`rnn_backward`.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional, or if a supplied ``a0`` does not have
        shape ``(n_a, m)``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (n_x, m, T_x); got ndim {x.ndim}")

    n_x, m, T_x = x.shape
    n_a = parameters["Waa"].shape[0]
    n_y = parameters["Wya"].shape[0]

    if a0 is None:
        a0 = np.zeros((n_a, m))
    elif a0.shape != (n_a, m):
        raise ValueError(f"a0 must have shape {(n_a, m)}; got {a0.shape}")

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    caches: list[tuple] = []

    a_prev = a0
    for t in range(T_x):
        a_next, y_pred_t, cache = rnn_cell_forward(x[:, :, t], a_prev, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = y_pred_t
        caches.append(cache)
        a_prev = a_next  # carry the state forward to t+1

    return a, y_pred, caches


def compute_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    Total softmax cross-entropy loss, summed over timesteps.

    Implements ``L = sum_t L^{<t>}`` with each per-step loss the batch-mean
    cross-entropy ``L^{<t>} = -(1/m) sum_{i,c} y^{<t>}_{c,i} log y_hat^{<t>}_{c,i}``.
    No epsilon guards the ``log``: the stable softmax output is strictly
    positive (see module ``Notes``).

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted distributions, shape ``(n_y, m, T_x)``.
    y : np.ndarray
        One-hot targets, same shape ``(n_y, m, T_x)``.

    Returns
    -------
    float
        The scalar loss summed over all timesteps.
    """
    m = y_pred.shape[1]
    return float(-np.sum(y * np.log(y_pred)) / m)


def rnn_cell_backward(
    y_t: np.ndarray,
    da_next: np.ndarray,
    cache: tuple,
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Backward pass for a single timestep.

    Combines the two gradient sources that reach ``a^{<t>}`` -- the output at
    this step and the recurrence carried back from ``t+1`` (``da_next``) -- and
    propagates them through the softmax read-out and the tanh recurrence.

    Parameters
    ----------
    y_t : np.ndarray
        One-hot target at this timestep, shape ``(n_y, m)``.
    da_next : np.ndarray
        Gradient of the loss w.r.t. this step's hidden state arriving from the
        recurrence at ``t+1``, shape ``(n_a, m)``. Zero for the final step.
    cache : tuple
        ``(a_next, a_prev, x_t, y_pred_t)`` from :func:`rnn_cell_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`rnn_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        This step's gradient contributions: ``dWax``, ``dWaa``, ``dWya``,
        ``dba``, ``dby``, together with ``dxt`` (into the input) and ``da_prev``
        (the recurrence handed back to ``t-1``).
    """
    a_next, a_prev, x_t, y_pred_t = cache
    Wax, Waa, Wya = parameters["Wax"], parameters["Waa"], parameters["Wya"]
    m = x_t.shape[1]

    # --- Output path: softmax cross-entropy -------------------------------
    # For softmax + cross-entropy the logit gradient collapses to (y_hat - y),
    # scaled by the 1/m batch mean used in compute_loss.
    dz_y = (y_pred_t - y_t) / m  # (n_y, m)
    dWya = dz_y @ a_next.T  # (n_y, n_a)
    dby = np.sum(dz_y, axis=1, keepdims=True)  # (n_y, 1)
    da_from_output = Wya.T @ dz_y  # (n_a, m)

    # --- Merge the two gradient sources at a^{<t>} ------------------------
    da_t = da_from_output + da_next  # (n_a, m)

    # --- Recurrence path: tanh backward -----------------------------------
    # d/dz tanh(z) = 1 - tanh(z)^2 = 1 - (a^{<t>})^2, read off the activation.
    dtanh = (1 - a_next**2) * da_t  # (n_a, m)
    dba = np.sum(dtanh, axis=1, keepdims=True)  # (n_a, 1)
    dWax = dtanh @ x_t.T  # (n_a, n_x)  -- input-path weight gradient
    dWaa = dtanh @ a_prev.T  # (n_a, n_a)  -- recurrent-path weight gradient
    dxt = Wax.T @ dtanh  # (n_x, m)    -- gradient into the input
    da_prev = Waa.T @ dtanh  # (n_a, m)    -- gradient back to t-1

    return {
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
        "dxt": dxt,
        "da_prev": da_prev,
    }


def rnn_backward(
    y: np.ndarray, caches: list[tuple], parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Backpropagation through time (BPTT).

    Walks the unrolled graph from ``t = T_x`` back to ``t = 1``. The shared
    weights receive **accumulated** gradients over every timestep, and the
    hidden-state gradient is threaded backward so that each step adds its
    recurrence contribution (``da_prev``) to the next step's incoming gradient.

    Parameters
    ----------
    y : np.ndarray
        One-hot targets, shape ``(n_y, m, T_x)``.
    caches : list[tuple]
        Per-timestep caches from :func:`rnn_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`rnn_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        Accumulated gradients ``dWax``, ``dWaa``, ``dWya``, ``dba``, ``dby``;
        the gradient ``da0`` w.r.t. the initial hidden state; and ``dx``, the
        gradient w.r.t. the whole input sequence, shape ``(n_x, m, T_x)``.
    """
    T_x = len(caches)
    _, m, _ = y.shape
    n_x = caches[0][2].shape[0]
    n_a = parameters["Waa"].shape[0]

    dWax = np.zeros_like(parameters["Wax"])
    dWaa = np.zeros_like(parameters["Waa"])
    dWya = np.zeros_like(parameters["Wya"])
    dba = np.zeros_like(parameters["ba"])
    dby = np.zeros_like(parameters["by"])
    dx = np.zeros((n_x, m, T_x))

    # The recurrence contribution into the current step, carried back one step
    # at a time. Zero at the final timestep, which has no successor.
    da_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        grads_t = rnn_cell_backward(y[:, :, t], da_next, caches[t], parameters)

        # Shared weights accumulate across all timesteps.
        dWax += grads_t["dWax"]
        dWaa += grads_t["dWaa"]
        dWya += grads_t["dWya"]
        dba += grads_t["dba"]
        dby += grads_t["dby"]

        dx[:, :, t] = grads_t["dxt"]
        da_next = grads_t["da_prev"]  # hand the recurrence back to t-1

    # After the loop, the recurrence gradient has reached a^{<0>}.
    da0 = da_next

    return {
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
        "da0": da0,
        "dx": dx,
    }


def clip_gradients(
    gradients: dict[str, np.ndarray], max_norm: float
) -> dict[str, np.ndarray]:
    """
    Global-norm gradient clipping, the exploding-gradient remedy.

    Computes the single L2 norm of all gradients viewed as one long vector and,
    if it exceeds ``max_norm``, rescales every gradient by the same factor. This
    preserves the gradient *direction* while capping its magnitude, exactly as
    prescribed by Pascanu, Mikolov & Bengio (2013).

    Parameters
    ----------
    gradients : dict[str, np.ndarray]
        Gradients to clip, typically the parameter gradients ``dWax``,
        ``dWaa``, ``dWya``, ``dba``, ``dby``.
    max_norm : float
        The maximum allowed global L2 norm.

    Returns
    -------
    dict[str, np.ndarray]
        A new dictionary of clipped gradients. The input is not modified.
    """
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {key: g * scale for key, g in gradients.items()}

    return {key: g.copy() for key, g in gradients.items()}


def update_parameters(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    """
    One step of gradient descent on the shared weights.

    Applies ``theta <- theta - learning_rate * dtheta`` to each of ``Wax``,
    ``Waa``, ``Wya``, ``ba``, ``by``.

    Parameters
    ----------
    parameters : dict[str, np.ndarray]
        The current weights and biases.
    gradients : dict[str, np.ndarray]
        Their gradients, keyed ``dWax``, ``dWaa``, ``dWya``, ``dba``, ``dby``.
    learning_rate : float
        The step size.

    Returns
    -------
    dict[str, np.ndarray]
        A new dictionary of updated parameters. The input is not modified.
    """
    return {
        key: parameters[key] - learning_rate * gradients[f"d{key}"]
        for key in ("Wax", "Waa", "Wya", "ba", "by")
    }


def bidirectional_rnn_forward(
    x: np.ndarray,
    parameters_forward: dict[str, np.ndarray],
    parameters_backward: dict[str, np.ndarray],
    a0_forward: np.ndarray | None = None,
    a0_backward: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Bidirectional RNN forward pass, as a composition of the vanilla cell.

    Runs two independent recurrences over the sequence -- one left-to-right
    (``a->``) and one right-to-left (``a<-``) -- and concatenates their hidden
    states at each timestep as ``[a->^{<t>}; a<-^{<t>}]``. There is no new cell
    mathematics: this is :func:`rnn_forward` applied twice, once to the
    time-reversed input. A bidirectional model attaches its own output layer,
    mapping the ``2 n_a``-dimensional concatenation to ``n_y``; the per-direction
    read-outs computed by the reused forward pass are not used here.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters_forward, parameters_backward : dict[str, np.ndarray]
        Independent weight sets for the two directions.
    a0_forward, a0_backward : np.ndarray or None, optional
        Initial hidden states for each direction, each shape ``(n_a, m)``.
        Default to zero.

    Returns
    -------
    a_bidirectional : np.ndarray
        Concatenated hidden states, shape ``(2 * n_a, m, T_x)``.
    parts : dict[str, np.ndarray]
        The two direction sequences, ``{"a_forward": ..., "a_backward": ...}``,
        each ``(n_a, m, T_x)`` and both in forward time order.
    """
    a_forward, _, _ = rnn_forward(x, parameters_forward, a0_forward)

    # Run the second recurrence over the reversed sequence, then realign its
    # activations back to forward time order so step t of both directions lines
    # up before concatenation.
    a_backward_reversed, _, _ = rnn_forward(
        x[:, :, ::-1], parameters_backward, a0_backward
    )
    a_backward = a_backward_reversed[:, :, ::-1]

    a_bidirectional = np.concatenate([a_forward, a_backward], axis=0)
    return a_bidirectional, {"a_forward": a_forward, "a_backward": a_backward}


def deep_rnn_forward(
    x: np.ndarray,
    parameters_per_layer: list[dict[str, np.ndarray]],
    a0_per_layer: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Deep (stacked) RNN forward pass, as a composition of the vanilla cell.

    Each layer is itself a full RNN over the sequence produced by the layer
    below: layer ``l`` consumes the activation sequence ``a^{[l-1]}`` in place of
    the input ``x``. There is no new cell mathematics -- this is
    :func:`rnn_forward` applied once per layer. Only the top layer's read-out is
    a model output; the intermediate read-outs computed by the reused forward
    pass are not used.

    Parameters
    ----------
    x : np.ndarray
        Input sequence to the first layer, shape ``(n_x, m, T_x)``.
    parameters_per_layer : list[dict[str, np.ndarray]]
        One weight set per layer, bottom to top. Layer ``l``'s ``Wax`` must
        accept the activation width of the layer below (``n_x`` for layer 0).
    a0_per_layer : list[np.ndarray] or None, optional
        Per-layer initial hidden states. Defaults to zero for every layer.

    Returns
    -------
    activations : list[np.ndarray]
        Each layer's hidden-state sequence, bottom to top, each
        ``(n_a^{[l]}, m, T_x)``.
    y_pred_top : np.ndarray
        The top layer's output distributions, shape ``(n_y, m, T_x)``.

    Raises
    ------
    ValueError
        If ``parameters_per_layer`` is empty.
    """
    if not parameters_per_layer:
        raise ValueError("a deep RNN needs at least one layer")

    if a0_per_layer is None:
        a0_per_layer = [None] * len(parameters_per_layer)

    activations: list[np.ndarray] = []
    layer_input = x
    y_pred_top = None
    for layer_params, a0 in zip(parameters_per_layer, a0_per_layer, strict=True):
        a_layer, y_pred_top, _ = rnn_forward(layer_input, layer_params, a0)
        activations.append(a_layer)
        layer_input = a_layer  # the layer above reads these activations

    return activations, y_pred_top


def make_fixture() -> dict:
    """
    Build the deterministic fixture every implementation and test shares.

    Draws are recorded in exactly the order the topic dossier pins, under
    ``np.random.seed(1)`` and ``float64`` (the default of ``randn``), so the
    seven parameter arrays reproduce bit-for-bit across implementations. The
    one-hot targets are drawn *after* those seven so that adding them cannot
    disturb the arrays a forward-parity check compares.

    Returns
    -------
    dict
        ``{"x", "a0", "parameters", "y", "dims"}`` where ``parameters`` is the
        dict ``{"Wax", "Waa", "Wya", "ba", "by"}``, ``y`` is one-hot of shape
        ``(n_y, m, T_x)``, and ``dims`` records ``n_x, n_a, n_y, m, T_x``.
    """
    np.random.seed(1)
    n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4

    x = np.random.randn(n_x, m, T_x)  # 1
    a0 = np.random.randn(n_a, m)  # 2
    Wax = np.random.randn(n_a, n_x)  # 3
    Waa = np.random.randn(n_a, n_a)  # 4
    Wya = np.random.randn(n_y, n_a)  # 5
    ba = np.random.randn(n_a, 1)  # 6
    by = np.random.randn(n_y, 1)  # 7

    # Targets, drawn after the seven parameter arrays above.
    labels = np.random.randint(0, n_y, size=(m, T_x))
    y = np.zeros((n_y, m, T_x))
    for i in range(m):
        for t in range(T_x):
            y[labels[i, t], i, t] = 1.0  # one-hot encode the class at (i, t)

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    dims = {"n_x": n_x, "n_a": n_a, "n_y": n_y, "m": m, "T_x": T_x}
    return {"x": x, "a0": a0, "parameters": parameters, "y": y, "dims": dims}


def main() -> None:
    """Run the shared fixture end to end and print shapes, loss, and grad norms."""
    fixture = make_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )

    a, y_pred, caches = rnn_forward(x, parameters, a0)
    loss = compute_loss(y_pred, y)
    gradients = rnn_backward(y, caches, parameters)

    print(f"hidden states a: {a.shape}")
    print(f"outputs y_pred:  {y_pred.shape}")
    print(f"total loss:      {loss:.6f}")
    for key in ("dWax", "dWaa", "dWya", "dba", "dby", "da0"):
        print(f"||{key}|| = {np.linalg.norm(gradients[key]):.6f}")


if __name__ == "__main__":
    main()
