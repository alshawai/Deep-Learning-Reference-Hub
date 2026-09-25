"""
Gated Recurrent Unit (GRU)
==========================

A from-scratch, teaching reference for the GRU: the two gates, the candidate
activation, the blend that carries state forward, and the hand-derived
backpropagation through time. This module is the explanation the accompanying
documents point at, so the forward pass appears in the shapes the topic declares
and every gradient is computed term by term rather than fused into one
expression.

Notation extends the vanilla RNN reference in :mod:`dlhub.nn.sequence.rnn` --
same ``a^{<t>}``, same shape conventions, same read-out and loss -- with the
gate vectors written as Andrew Ng's Course 5 writes them. For the GRU the hidden
state and the cell state are the same object, ``a^{<t>} = c^{<t>}``, so there is
no separate ``c`` to thread. For ``t = 1 ... T_x``::

    Gamma_r^{<t>} = sigma(W_r [a^{<t-1>}; x^{<t>}] + b_r)
    Gamma_u^{<t>} = sigma(W_u [a^{<t-1>}; x^{<t>}] + b_u)
    a_tilde^{<t>} = tanh(W_c [Gamma_r^{<t>} * a^{<t-1>}; x^{<t>}] + b_c)
    a^{<t>}       = Gamma_u^{<t>} * a_tilde^{<t>} + (1 - Gamma_u^{<t>}) * a^{<t-1>}
    y_hat^{<t>}   = softmax(W_y a^{<t>} + b_y)

(``*`` is elementwise.) The update gate ``Gamma_u`` interpolates between keeping
the previous state and taking the candidate; the reset gate ``Gamma_r`` decides
how much of the previous state the candidate is allowed to see. The
``(1 - Gamma_u)`` term is the GRU's answer to the vanishing gradient: when the
gate is near zero the state is copied forward, and its gradient with it, instead
of being pushed through another ``tanh`` and another weight matrix. The vanilla
RNN in this package has no such path, which is the comparison the topic exists
to make. The LSTM's version of the same idea, with a separate cell state, is in
:mod:`dlhub.nn.sequence.lstm`.

References
----------
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation. EMNLP.
  https://arxiv.org/abs/1406.1078
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of
  Gated Recurrent Neural Networks on Sequence Modeling.
  https://arxiv.org/abs/1412.3555
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 10.
  MIT Press. https://www.deeplearningbook.org/
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
- **The reset gate is applied before the hidden transform.** The candidate reads
  ``[Gamma_r * a^{<t-1>}; x^{<t>}]``, so the gate acts on the state *inside* the
  concatenation ``W_c`` multiplies. This is the Cho/Ng form and it is canonical
  for this hub. Several frameworks implement the reset *after* the recurrent
  transform instead -- ``tanh(W_x x + r * (W_h a + b_h))``, which is what
  ``torch.nn.GRU`` and ``tf.keras.layers.GRU(reset_after=True)`` compute. The two
  are not algebraically equal; the same layer name denotes two different
  functions across the ecosystem. The hub's TensorFlow port reproduces this
  module with ``reset_after=False``.
- **Stable sigmoid.** Gates use :func:`dlhub.nn.sequence._common.sigmoid`, which
  never exponentiates a positive number; see that module's ``Notes``. The softmax
  read-out and the loss are imported from :mod:`dlhub.nn.sequence.rnn`, the
  canonical home of the max-subtraction stability shift, rather than restated.
- **Derivatives are read off the cached activations.** ``sigma'(z) =
  Gamma (1 - Gamma)`` and ``tanh'(z) = 1 - a_tilde^2`` both reuse the value the
  forward pass already computed, so the backward pass evaluates no new sigmoid
  and no new tanh -- the same trick the vanilla RNN uses for its single tanh.
- **Concatenated weight form.** One matrix per gate acts on
  ``[a^{<t-1>}; x^{<t>}]``, so ``W_g`` is ``(n_a, n_a + n_x)``. The vanilla RNN
  deliberately keeps ``W_aa`` and ``W_ax`` separate to expose the two BPTT paths;
  that lesson is taught there, and repeating it across three gates is clutter.
  The two halves are still nameable: :func:`split_gate_matrix` cuts any ``W_g``
  or ``dW_g`` at column ``n_a`` into its recurrent and input parts, which is what
  :func:`main` reports.
- **The cache is a dict, not a tuple.** The vanilla RNN's four-element positional
  cache is readable; a gated cell's is not, so every cached array is named.
"""

import numpy as np

from dlhub.nn.sequence._common import sigmoid, split_gate_matrix
from dlhub.nn.sequence.rnn import compute_loss, softmax

__all__ = [
    "gru_backward",
    "gru_cell_backward",
    "gru_cell_forward",
    "gru_forward",
    "make_fixture",
]

# Every weight and bias below is shared across all timesteps, so BPTT
# accumulates one gradient per name over the whole sequence.
PARAMETER_KEYS = ("Wr", "Wu", "Wc", "Wy", "br", "bu", "bc", "by")


def gru_cell_forward(
    x_t: np.ndarray, a_prev: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    One timestep of the GRU forward recurrence.

    Computes the two gates, the candidate activation, the blended state, and the
    softmax read-out for a single step, in that order.

    Parameters
    ----------
    x_t : np.ndarray
        Input at this timestep, shape ``(n_x, m)``.
    a_prev : np.ndarray
        Hidden state carried in from the previous step, ``a^{<t-1>}``, shape
        ``(n_a, m)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases: ``Wr``, ``Wu``, ``Wc`` each
        ``(n_a, n_a + n_x)``; ``br``, ``bu``, ``bc`` each ``(n_a, 1)``; ``Wy``
        ``(n_y, n_a)``; ``by`` ``(n_y, 1)``.

    Returns
    -------
    a_next : np.ndarray
        Updated hidden state ``a^{<t>}``, shape ``(n_a, m)``.
    y_pred_t : np.ndarray
        Output distribution ``y_hat^{<t>}``, shape ``(n_y, m)``.
    cache : dict[str, np.ndarray]
        Everything this step's backward pass needs, each array named.
    """
    Wr, Wu, Wc = parameters["Wr"], parameters["Wu"], parameters["Wc"]
    br, bu, bc = parameters["br"], parameters["bu"], parameters["bc"]
    Wy, by = parameters["Wy"], parameters["by"]

    # The concatenated form: one stacked vector [a^{<t-1>}; x^{<t>}], shape
    # (n_a + n_x, m), that every gate weight multiplies.
    concat = np.concatenate([a_prev, x_t], axis=0)

    # --- Gates ------------------------------------------------------------
    gamma_r = sigmoid(Wr @ concat + br)  # reset: how much state the candidate sees
    gamma_u = sigmoid(Wu @ concat + bu)  # update: keep the old state, or take the new

    # --- Candidate --------------------------------------------------------
    # The reset gate is applied *before* the hidden transform: it scales
    # a^{<t-1>} inside the concatenation W_c acts on. See the module Notes.
    concat_reset = np.concatenate([gamma_r * a_prev, x_t], axis=0)
    a_tilde = np.tanh(Wc @ concat_reset + bc)

    # --- Blend ------------------------------------------------------------
    # A convex combination, elementwise: gate near 1 takes the candidate, gate
    # near 0 copies the previous state forward unchanged.
    a_next = gamma_u * a_tilde + (1 - gamma_u) * a_prev

    # --- Read-out ---------------------------------------------------------
    y_pred_t = softmax(Wy @ a_next + by)

    cache = {
        "a_next": a_next,
        "a_prev": a_prev,
        "gamma_r": gamma_r,
        "gamma_u": gamma_u,
        "a_tilde": a_tilde,
        "concat": concat,
        "concat_reset": concat_reset,
        "x_t": x_t,
        "y_pred_t": y_pred_t,
    }
    return a_next, y_pred_t, cache


def gru_forward(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    """
    Forward propagation through time across a whole sequence.

    Unrolls :func:`gru_cell_forward` over ``T_x`` timesteps, threading the hidden
    state from one step to the next.

    Parameters
    ----------
    x : np.ndarray
        Input sequence of shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`gru_cell_forward`).
    a0 : np.ndarray or None, optional
        Initial hidden state ``a^{<0>}`` of shape ``(n_a, m)``. Defaults to the
        zero vector, matching the standard convention ``a^{<0>} = 0``.

    Returns
    -------
    a : np.ndarray
        All hidden states stacked over time, shape ``(n_a, m, T_x)``.
    y_pred : np.ndarray
        All output distributions, shape ``(n_y, m, T_x)``.
    caches : list[dict[str, np.ndarray]]
        Per-timestep caches, in order, for :func:`gru_backward`.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional, or if a supplied ``a0`` does not have
        shape ``(n_a, m)``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (n_x, m, T_x); got ndim {x.ndim}")

    n_x, m, T_x = x.shape
    n_a = parameters["Wu"].shape[0]
    n_y = parameters["Wy"].shape[0]

    if a0 is None:
        a0 = np.zeros((n_a, m))
    elif a0.shape != (n_a, m):
        raise ValueError(f"a0 must have shape {(n_a, m)}; got {a0.shape}")

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    caches: list[dict[str, np.ndarray]] = []

    a_prev = a0
    for t in range(T_x):
        a_next, y_pred_t, cache = gru_cell_forward(x[:, :, t], a_prev, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = y_pred_t
        caches.append(cache)
        a_prev = a_next  # carry the state forward to t+1

    return a, y_pred, caches


def gru_cell_backward(
    y_t: np.ndarray,
    da_next: np.ndarray,
    cache: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Backward pass for a single GRU timestep.

    Walks the forward pass backwards, one equation at a time. Three separate
    paths deposit gradient into ``a^{<t-1>}``, and the whole point of the cell is
    that the first of them is a plain elementwise factor:

    1. the **blend**, through ``(1 - Gamma_u)`` -- no weight matrix, no
       saturating nonlinearity, so gradient crosses the step essentially undamped
       when the update gate is near zero;
    2. the **candidate**, through ``W_c`` and then the reset gate;
    3. the **gates** themselves, through ``W_r`` and ``W_u``.

    Parameters
    ----------
    y_t : np.ndarray
        One-hot target at this timestep, shape ``(n_y, m)``.
    da_next : np.ndarray
        Gradient of the loss w.r.t. this step's hidden state arriving from the
        recurrence at ``t+1``, shape ``(n_a, m)``. Zero for the final step.
    cache : dict[str, np.ndarray]
        This step's cache from :func:`gru_cell_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`gru_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        This step's gradient contributions: ``dWr``, ``dWu``, ``dWc``, ``dbr``,
        ``dbu``, ``dbc``, ``dWy``, ``dby``, together with ``dxt`` (into the
        input) and ``da_prev`` (the recurrence handed back to ``t-1``).
    """
    a_next, a_prev = cache["a_next"], cache["a_prev"]
    gamma_r, gamma_u = cache["gamma_r"], cache["gamma_u"]
    a_tilde = cache["a_tilde"]
    concat, concat_reset = cache["concat"], cache["concat_reset"]
    y_pred_t = cache["y_pred_t"]

    Wr, Wu, Wc, Wy = (
        parameters["Wr"],
        parameters["Wu"],
        parameters["Wc"],
        parameters["Wy"],
    )
    n_a, m = a_next.shape

    # --- Output path: softmax cross-entropy -------------------------------
    # For softmax + cross-entropy the logit gradient collapses to (y_hat - y),
    # scaled by the 1/m batch mean used in compute_loss.
    dz_y = (y_pred_t - y_t) / m  # (n_y, m)
    dWy = dz_y @ a_next.T  # (n_y, n_a)
    dby = np.sum(dz_y, axis=1, keepdims=True)  # (n_y, 1)
    da_from_output = Wy.T @ dz_y  # (n_a, m)

    # --- Merge the two gradient sources at a^{<t>} ------------------------
    da_t = da_from_output + da_next  # (n_a, m)

    # --- Blend: a^{<t>} = Gamma_u * a_tilde + (1 - Gamma_u) * a^{<t-1>} ---
    # Differentiate the convex combination in its three arguments.
    dgamma_u = da_t * (a_tilde - a_prev)  # (n_a, m)
    da_tilde = da_t * gamma_u  # (n_a, m)
    da_prev = da_t * (1 - gamma_u)  # path 1: the undamped carry

    # --- Candidate: a_tilde = tanh(W_c [Gamma_r * a^{<t-1>}; x^{<t>}] + b_c)
    dz_c = da_tilde * (1 - a_tilde**2)  # tanh', read off the activation
    dWc = dz_c @ concat_reset.T  # (n_a, n_a + n_x)
    dbc = np.sum(dz_c, axis=1, keepdims=True)  # (n_a, 1)
    dconcat_reset = Wc.T @ dz_c  # (n_a + n_x, m)

    # The top n_a rows are the gradient w.r.t. the *reset* state, which the
    # product rule then splits between the gate and the state itself.
    d_reset_state = dconcat_reset[:n_a, :]  # d/d(Gamma_r * a^{<t-1>})
    dgamma_r = d_reset_state * a_prev
    da_prev = da_prev + d_reset_state * gamma_r  # path 2: through the candidate
    dxt = dconcat_reset[n_a:, :]  # (n_x, m)

    # --- Gates: sigma'(z) = Gamma (1 - Gamma), read off the cached gate ----
    dz_r = dgamma_r * gamma_r * (1 - gamma_r)
    dWr = dz_r @ concat.T
    dbr = np.sum(dz_r, axis=1, keepdims=True)
    dconcat_r = Wr.T @ dz_r

    dz_u = dgamma_u * gamma_u * (1 - gamma_u)
    dWu = dz_u @ concat.T
    dbu = np.sum(dz_u, axis=1, keepdims=True)
    dconcat_u = Wu.T @ dz_u

    # Both gates read the same stacked vector, so each contributes to both the
    # recurrent gradient and the input gradient. Path 3.
    da_prev = da_prev + dconcat_r[:n_a, :] + dconcat_u[:n_a, :]
    dxt = dxt + dconcat_r[n_a:, :] + dconcat_u[n_a:, :]

    return {
        "dWr": dWr,
        "dWu": dWu,
        "dWc": dWc,
        "dbr": dbr,
        "dbu": dbu,
        "dbc": dbc,
        "dWy": dWy,
        "dby": dby,
        "dxt": dxt,
        "da_prev": da_prev,
    }


def gru_backward(
    y: np.ndarray,
    caches: list[dict[str, np.ndarray]],
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Backpropagation through time (BPTT) for the GRU.

    Walks the unrolled graph from ``t = T_x`` back to ``t = 1``. Every gate
    weight is shared across time, so each receives an **accumulated** gradient
    summed over all timesteps, and the hidden-state gradient is threaded
    backward so each step adds its recurrence contribution (``da_prev``) to the
    next step's incoming gradient.

    Parameters
    ----------
    y : np.ndarray
        One-hot targets, shape ``(n_y, m, T_x)``.
    caches : list[dict[str, np.ndarray]]
        Per-timestep caches from :func:`gru_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`gru_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        Accumulated gradients ``dWr``, ``dWu``, ``dWc``, ``dbr``, ``dbu``,
        ``dbc``, ``dWy``, ``dby``; the gradient ``da0`` w.r.t. the initial hidden
        state; and ``dx``, the gradient w.r.t. the whole input sequence, shape
        ``(n_x, m, T_x)``.
    """
    T_x = len(caches)
    _, m, _ = y.shape
    n_x = caches[0]["x_t"].shape[0]
    n_a = parameters["Wu"].shape[0]

    accumulated = {f"d{key}": np.zeros_like(parameters[key]) for key in PARAMETER_KEYS}
    dx = np.zeros((n_x, m, T_x))

    # The recurrence contribution into the current step, carried back one step
    # at a time. Zero at the final timestep, which has no successor.
    da_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        grads_t = gru_cell_backward(y[:, :, t], da_next, caches[t], parameters)

        # The shared weights accumulate: every timestep used the same matrix,
        # so every timestep adds a term to its gradient.
        for key in PARAMETER_KEYS:
            accumulated[f"d{key}"] += grads_t[f"d{key}"]

        dx[:, :, t] = grads_t["dxt"]
        da_next = grads_t["da_prev"]  # hand the recurrence back to t-1

    # After the loop, the recurrence gradient has reached a^{<0>}.
    return {**accumulated, "da0": da_next, "dx": dx}


def make_fixture() -> dict:
    """
    Build the deterministic fixture every GRU implementation and test shares.

    Draws are recorded in exactly the order the topic dossier pins, under
    ``np.random.seed(1)`` and ``float64`` (the default of ``randn``), so the
    arrays reproduce bit-for-bit across implementations. The one-hot targets are
    drawn *last* so that adding them cannot disturb the arrays a forward-parity
    check compares. Dimensions match the vanilla RNN fixture, so the notebook can
    contrast the two cells on identical shapes.

    Returns
    -------
    dict
        ``{"x", "a0", "parameters", "y", "dims"}`` where ``parameters`` is the
        dict ``{"Wr", "Wu", "Wc", "Wy", "br", "bu", "bc", "by"}``, ``y`` is
        one-hot of shape ``(n_y, m, T_x)``, and ``dims`` records
        ``n_x, n_a, n_y, m, T_x``.
    """
    np.random.seed(1)
    n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4

    x = np.random.randn(n_x, m, T_x)  # 1
    a0 = np.random.randn(n_a, m)  # 2
    Wr = np.random.randn(n_a, n_a + n_x)  # 3
    Wu = np.random.randn(n_a, n_a + n_x)  # 4
    Wc = np.random.randn(n_a, n_a + n_x)  # 5
    Wy = np.random.randn(n_y, n_a)  # 6
    br = np.random.randn(n_a, 1)  # 7
    bu = np.random.randn(n_a, 1)  # 8
    bc = np.random.randn(n_a, 1)  # 9
    by = np.random.randn(n_y, 1)  # 10

    # Targets, drawn after the ten parameter arrays above.
    labels = np.random.randint(0, n_y, size=(m, T_x))
    y = np.zeros((n_y, m, T_x))
    for i in range(m):
        for t in range(T_x):
            y[labels[i, t], i, t] = 1.0  # one-hot encode the class at (i, t)

    parameters = {
        "Wr": Wr,
        "Wu": Wu,
        "Wc": Wc,
        "Wy": Wy,
        "br": br,
        "bu": bu,
        "bc": bc,
        "by": by,
    }
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
    n_a = fixture["dims"]["n_a"]

    a, y_pred, caches = gru_forward(x, parameters, a0)
    loss = compute_loss(y_pred, y)
    gradients = gru_backward(y, caches, parameters)

    print(f"hidden states a: {a.shape}")
    print(f"outputs y_pred:  {y_pred.shape}")
    print(f"total loss:      {loss:.6f}")
    for key in ("dWr", "dWu", "dWc", "dWy", "dbr", "dbu", "dbc", "dby", "da0"):
        print(f"||{key}|| = {np.linalg.norm(gradients[key]):.6f}")

    # The concatenated gate gradient still names its two halves: the recurrent
    # path dW_ga and the input path dW_gx, split at column n_a.
    for key in ("dWr", "dWu", "dWc"):
        recurrent, input_ = split_gate_matrix(gradients[key], n_a)
        print(
            f"{key} = [recurrent | input] -> "
            f"||{key}a|| = {np.linalg.norm(recurrent):.6f}, "
            f"||{key}x|| = {np.linalg.norm(input_):.6f}"
        )


if __name__ == "__main__":
    main()
