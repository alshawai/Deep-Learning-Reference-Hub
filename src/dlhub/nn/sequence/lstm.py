"""
Long Short-Term Memory (LSTM)
=============================

A from-scratch, teaching reference for the LSTM: the three gates, the cell
state, the additive path that carries gradient across time, and the
hand-derived backpropagation through time. This module is the explanation the
accompanying documents point at, so the forward pass appears in the shapes the
topic declares and every gradient is computed term by term rather than fused
into one expression.

Notation extends the vanilla RNN reference in :mod:`dlhub.nn.sequence.rnn` --
same ``a^{<t>}``, same shape conventions, same read-out and loss -- by adding a
**separate cell state** ``c^{<t>}`` and the gate vectors, written as Andrew Ng's
Course 5 writes them. For ``t = 1 ... T_x``::

    Gamma_f^{<t>} = sigma(W_f [a^{<t-1>}; x^{<t>}] + b_f)
    Gamma_u^{<t>} = sigma(W_u [a^{<t-1>}; x^{<t>}] + b_u)
    c_tilde^{<t>} = tanh (W_c [a^{<t-1>}; x^{<t>}] + b_c)
    Gamma_o^{<t>} = sigma(W_o [a^{<t-1>}; x^{<t>}] + b_o)
    c^{<t>}       = Gamma_u^{<t>} * c_tilde^{<t>} + Gamma_f^{<t>} * c^{<t-1>}
    a^{<t>}       = Gamma_o^{<t>} * tanh(c^{<t>})
    y_hat^{<t>}   = softmax(W_y a^{<t>} + b_y)

(``*`` is elementwise.) Gates are computed and reported in the order **forget,
update, candidate, output** throughout this module. The *update* gate is
Hochreiter & Schmidhuber's *input* gate ``i``; Ng writes it ``Gamma_u``, which
this hub follows so that its correspondence with the GRU's update gate is
visible at a glance.

The cell-state line is the reason the architecture exists. ``c^{<t>}`` is built
by **addition**, and holding the gates fixed its derivative with respect to
``c^{<t-1>}`` is the elementwise factor ``Gamma_f^{<t>}`` -- not a repeated
multiplication by ``W_aa^T`` through a ``tanh`` derivative, as in the vanilla
RNN, whose product decays geometrically. A forget gate near 1 therefore
transports gradient across many steps essentially undamped. The GRU in
:mod:`dlhub.nn.sequence.gru` achieves the same thing with one gate fewer and no
separate cell state.

References
----------
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural
  Computation*, 9(8), 1735-1780.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget:
  Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.
  (The forget gate, which the 1997 cell lacked and every modern LSTM has.)
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
- **Stable sigmoid.** Gates use :func:`dlhub.nn.sequence._common.sigmoid`, which
  never exponentiates a positive number; see that module's ``Notes``. The softmax
  read-out and the loss are imported from :mod:`dlhub.nn.sequence.rnn`, the
  canonical home of the max-subtraction stability shift, rather than restated.
- **Derivatives are read off the cached activations.** ``sigma'(z) =
  Gamma (1 - Gamma)`` and ``tanh'(z) = 1 - tanh(z)^2`` both reuse values the
  forward pass already computed, so the backward pass evaluates no new sigmoid
  and no new tanh. ``tanh(c^{<t>})`` is cached for the same reason: it appears in
  the output-gate gradient and again in the cell-state gradient.
- **Concatenated weight form.** One matrix per gate acts on
  ``[a^{<t-1>}; x^{<t>}]``, so ``W_g`` is ``(n_a, n_a + n_x)``. The vanilla RNN
  deliberately keeps ``W_aa`` and ``W_ax`` separate to expose the two BPTT paths;
  that lesson is taught there, and repeating it across four gates is clutter.
  The two halves are still nameable: :func:`split_gate_matrix` cuts any ``W_g``
  or ``dW_g`` at column ``n_a`` into its recurrent and input parts, which is what
  :func:`main` reports.
- **The cache is a dict, not a tuple.** The vanilla RNN's four-element positional
  cache is readable; a gated cell's eleven-element one is not, so every cached
  array is named.
- **No epsilon anywhere.** Nothing in the cell divides by a learned quantity, and
  the stable softmax output is strictly positive, so the cross-entropy ``log``
  needs no guard. An epsilon added "for safety" would shift the loss and break
  the finite-difference gradient check at the topic's ``1e-7`` tolerance.
"""

import numpy as np

from dlhub.nn.sequence._common import sigmoid, split_gate_matrix
from dlhub.nn.sequence.rnn import compute_loss, softmax

__all__ = [
    "lstm_backward",
    "lstm_cell_backward",
    "lstm_cell_forward",
    "lstm_forward",
    "make_fixture",
]

# Every weight and bias below is shared across all timesteps, so BPTT
# accumulates one gradient per name over the whole sequence. The gate order is
# the topic's: forget, update, candidate, output, then the read-out.
PARAMETER_KEYS = ("Wf", "Wu", "Wc", "Wo", "Wy", "bf", "bu", "bc", "bo", "by")


def lstm_cell_forward(
    x_t: np.ndarray,
    a_prev: np.ndarray,
    c_prev: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    One timestep of the LSTM forward recurrence.

    Computes the four gate quantities in the topic's order (forget, update,
    candidate, output), then the cell state, the hidden state, and the softmax
    read-out.

    Parameters
    ----------
    x_t : np.ndarray
        Input at this timestep, shape ``(n_x, m)``.
    a_prev : np.ndarray
        Hidden state carried in from the previous step, ``a^{<t-1>}``, shape
        ``(n_a, m)``.
    c_prev : np.ndarray
        Cell state carried in from the previous step, ``c^{<t-1>}``, shape
        ``(n_a, m)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases: ``Wf``, ``Wu``, ``Wc``, ``Wo`` each
        ``(n_a, n_a + n_x)``; ``bf``, ``bu``, ``bc``, ``bo`` each ``(n_a, 1)``;
        ``Wy`` ``(n_y, n_a)``; ``by`` ``(n_y, 1)``.

    Returns
    -------
    a_next : np.ndarray
        Updated hidden state ``a^{<t>}``, shape ``(n_a, m)``.
    c_next : np.ndarray
        Updated cell state ``c^{<t>}``, shape ``(n_a, m)``.
    y_pred_t : np.ndarray
        Output distribution ``y_hat^{<t>}``, shape ``(n_y, m)``.
    cache : dict[str, np.ndarray]
        Everything this step's backward pass needs, each array named.
    """
    Wf, Wu, Wc, Wo = (
        parameters["Wf"],
        parameters["Wu"],
        parameters["Wc"],
        parameters["Wo"],
    )
    bf, bu, bc, bo = (
        parameters["bf"],
        parameters["bu"],
        parameters["bc"],
        parameters["bo"],
    )
    Wy, by = parameters["Wy"], parameters["by"]

    # The concatenated form: one stacked vector [a^{<t-1>}; x^{<t>}], shape
    # (n_a + n_x, m), that all four gate weights multiply. Unlike the GRU, no
    # gate is applied inside this concatenation -- the LSTM has no reset gate.
    concat = np.concatenate([a_prev, x_t], axis=0)

    # --- The four gate quantities, in the topic's order -------------------
    gamma_f = sigmoid(Wf @ concat + bf)  # forget: how much of c^{<t-1>} survives
    gamma_u = sigmoid(Wu @ concat + bu)  # update ("input"): how much candidate enters
    c_tilde = np.tanh(Wc @ concat + bc)  # candidate: what could be written
    gamma_o = sigmoid(Wo @ concat + bo)  # output: how much of the cell is exposed

    # --- Cell state: the additive path ------------------------------------
    # Two gated terms are *added*. Nothing multiplies c^{<t-1>} but an
    # elementwise gate, which is the whole architectural point (module Notes).
    c_next = gamma_u * c_tilde + gamma_f * c_prev

    # --- Hidden state: a squashed, gated view of the cell -----------------
    tanh_c_next = np.tanh(c_next)
    a_next = gamma_o * tanh_c_next

    # --- Read-out ---------------------------------------------------------
    y_pred_t = softmax(Wy @ a_next + by)

    cache = {
        "a_next": a_next,
        "c_next": c_next,
        "a_prev": a_prev,
        "c_prev": c_prev,
        "gamma_f": gamma_f,
        "gamma_u": gamma_u,
        "c_tilde": c_tilde,
        "gamma_o": gamma_o,
        "tanh_c_next": tanh_c_next,
        "concat": concat,
        "x_t": x_t,
        "y_pred_t": y_pred_t,
    }
    return a_next, c_next, y_pred_t, cache


def lstm_forward(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
    c0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    """
    Forward propagation through time across a whole sequence.

    Unrolls :func:`lstm_cell_forward` over ``T_x`` timesteps, threading **two**
    states from one step to the next: the hidden state ``a`` and the cell state
    ``c``.

    Parameters
    ----------
    x : np.ndarray
        Input sequence of shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`lstm_cell_forward`).
    a0 : np.ndarray or None, optional
        Initial hidden state ``a^{<0>}`` of shape ``(n_a, m)``. Defaults to the
        zero vector.
    c0 : np.ndarray or None, optional
        Initial cell state ``c^{<0>}`` of shape ``(n_a, m)``. Defaults to the
        zero vector.

    Returns
    -------
    a : np.ndarray
        All hidden states stacked over time, shape ``(n_a, m, T_x)``.
    c : np.ndarray
        All cell states stacked over time, shape ``(n_a, m, T_x)``.
    y_pred : np.ndarray
        All output distributions, shape ``(n_y, m, T_x)``.
    caches : list[dict[str, np.ndarray]]
        Per-timestep caches, in order, for :func:`lstm_backward`.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional, or if a supplied ``a0`` or ``c0`` does
        not have shape ``(n_a, m)``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (n_x, m, T_x); got ndim {x.ndim}")

    n_x, m, T_x = x.shape
    n_a = parameters["Wf"].shape[0]
    n_y = parameters["Wy"].shape[0]

    if a0 is None:
        a0 = np.zeros((n_a, m))
    elif a0.shape != (n_a, m):
        raise ValueError(f"a0 must have shape {(n_a, m)}; got {a0.shape}")

    if c0 is None:
        c0 = np.zeros((n_a, m))
    elif c0.shape != (n_a, m):
        raise ValueError(f"c0 must have shape {(n_a, m)}; got {c0.shape}")

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    caches: list[dict[str, np.ndarray]] = []

    a_prev, c_prev = a0, c0
    for t in range(T_x):
        a_next, c_next, y_pred_t, cache = lstm_cell_forward(
            x[:, :, t], a_prev, c_prev, parameters
        )
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y_pred[:, :, t] = y_pred_t
        caches.append(cache)
        a_prev, c_prev = a_next, c_next  # carry both states forward to t+1

    return a, c, y_pred, caches


def lstm_cell_backward(
    y_t: np.ndarray,
    da_next: np.ndarray,
    dc_next: np.ndarray,
    cache: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Backward pass for a single LSTM timestep.

    Walks the forward pass backwards, one equation at a time. Two gradients
    arrive from the future, not one: ``da_next`` along the hidden state and
    ``dc_next`` along the cell state. They meet at ``c^{<t>}``, because the
    hidden state is a function of the cell state at the same step.

    The line worth reading twice is ``dc_prev = dc_t * Gamma_f``. That is the
    whole thesis of the architecture as one statement: the gradient crossing
    from ``t`` to ``t-1`` along the cell path is scaled elementwise by the forget
    gate, with no weight matrix and no saturating nonlinearity in the way.

    Parameters
    ----------
    y_t : np.ndarray
        One-hot target at this timestep, shape ``(n_y, m)``.
    da_next : np.ndarray
        Gradient w.r.t. this step's hidden state arriving from the recurrence at
        ``t+1``, shape ``(n_a, m)``. Zero for the final step.
    dc_next : np.ndarray
        Gradient w.r.t. this step's cell state arriving from the recurrence at
        ``t+1``, shape ``(n_a, m)``. Zero for the final step.
    cache : dict[str, np.ndarray]
        This step's cache from :func:`lstm_cell_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`lstm_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        This step's gradient contributions: ``dWf``, ``dWu``, ``dWc``, ``dWo``,
        ``dbf``, ``dbu``, ``dbc``, ``dbo``, ``dWy``, ``dby``, together with
        ``dxt`` (into the input) and ``da_prev``, ``dc_prev`` (the two
        recurrences handed back to ``t-1``).
    """
    a_next, c_prev = cache["a_next"], cache["c_prev"]
    gamma_f, gamma_u = cache["gamma_f"], cache["gamma_u"]
    c_tilde, gamma_o = cache["c_tilde"], cache["gamma_o"]
    tanh_c_next, concat = cache["tanh_c_next"], cache["concat"]
    y_pred_t = cache["y_pred_t"]

    Wf, Wu, Wc, Wo, Wy = (
        parameters["Wf"],
        parameters["Wu"],
        parameters["Wc"],
        parameters["Wo"],
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

    # --- Hidden state: a^{<t>} = Gamma_o * tanh(c^{<t>}) ------------------
    dgamma_o = da_t * tanh_c_next
    # The hidden state feeds the cell state's gradient: differentiating the
    # tanh gives 1 - tanh(c)^2, read off the cached tanh. This joins the
    # gradient that arrived along the cell path from t+1.
    dc_t = dc_next + da_t * gamma_o * (1 - tanh_c_next**2)  # (n_a, m)

    # --- Cell state: c^{<t>} = Gamma_u * c_tilde + Gamma_f * c^{<t-1>} ----
    # An addition of two products, so each factor's gradient is the other.
    dgamma_u = dc_t * c_tilde
    dc_tilde = dc_t * gamma_u
    dgamma_f = dc_t * c_prev
    dc_prev = dc_t * gamma_f  # the undamped carry: d c^{<t>} / d c^{<t-1>}

    # --- Pre-activations, in the topic's gate order -----------------------
    # sigma'(z) = Gamma (1 - Gamma) and tanh'(z) = 1 - c_tilde^2, both read off
    # the cached activation rather than recomputed.
    dz_f = dgamma_f * gamma_f * (1 - gamma_f)
    dz_u = dgamma_u * gamma_u * (1 - gamma_u)
    dz_c = dc_tilde * (1 - c_tilde**2)
    dz_o = dgamma_o * gamma_o * (1 - gamma_o)

    # --- Gate weights and biases ------------------------------------------
    # Each gate read the same stacked vector [a^{<t-1>}; x^{<t>}], so each gate
    # weight's gradient is its pre-activation gradient times that vector.
    dWf, dbf = dz_f @ concat.T, np.sum(dz_f, axis=1, keepdims=True)
    dWu, dbu = dz_u @ concat.T, np.sum(dz_u, axis=1, keepdims=True)
    dWc, dbc = dz_c @ concat.T, np.sum(dz_c, axis=1, keepdims=True)
    dWo, dbo = dz_o @ concat.T, np.sum(dz_o, axis=1, keepdims=True)

    # --- Back through the concatenation -----------------------------------
    # All four gates read the same vector, so their contributions add; the top
    # n_a rows are the recurrence into a^{<t-1>} and the rest is the input.
    dconcat = Wf.T @ dz_f + Wu.T @ dz_u + Wc.T @ dz_c + Wo.T @ dz_o
    da_prev = dconcat[:n_a, :]  # (n_a, m)
    dxt = dconcat[n_a:, :]  # (n_x, m)

    return {
        "dWf": dWf,
        "dWu": dWu,
        "dWc": dWc,
        "dWo": dWo,
        "dbf": dbf,
        "dbu": dbu,
        "dbc": dbc,
        "dbo": dbo,
        "dWy": dWy,
        "dby": dby,
        "dxt": dxt,
        "da_prev": da_prev,
        "dc_prev": dc_prev,
    }


def lstm_backward(
    y: np.ndarray,
    caches: list[dict[str, np.ndarray]],
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Backpropagation through time (BPTT) for the LSTM.

    Walks the unrolled graph from ``t = T_x`` back to ``t = 1``, threading
    **two** recurrent gradients backward -- one along the hidden state and one
    along the cell state. Every gate weight is shared across time, so each
    receives an accumulated gradient summed over all timesteps.

    Parameters
    ----------
    y : np.ndarray
        One-hot targets, shape ``(n_y, m, T_x)``.
    caches : list[dict[str, np.ndarray]]
        Per-timestep caches from :func:`lstm_forward`.
    parameters : dict[str, np.ndarray]
        The shared weights and biases (see :func:`lstm_cell_forward`).

    Returns
    -------
    dict[str, np.ndarray]
        Accumulated gradients ``dWf``, ``dWu``, ``dWc``, ``dWo``, ``dbf``,
        ``dbu``, ``dbc``, ``dbo``, ``dWy``, ``dby``; the gradients ``da0`` and
        ``dc0`` w.r.t. the two initial states; and ``dx``, the gradient w.r.t.
        the whole input sequence, shape ``(n_x, m, T_x)``.
    """
    T_x = len(caches)
    _, m, _ = y.shape
    n_x = caches[0]["x_t"].shape[0]
    n_a = parameters["Wf"].shape[0]

    accumulated = {f"d{key}": np.zeros_like(parameters[key]) for key in PARAMETER_KEYS}
    dx = np.zeros((n_x, m, T_x))

    # The two recurrence contributions into the current step, carried back one
    # step at a time. Both zero at the final timestep, which has no successor.
    da_next = np.zeros((n_a, m))
    dc_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        grads_t = lstm_cell_backward(
            y[:, :, t], da_next, dc_next, caches[t], parameters
        )

        # The shared weights accumulate: every timestep used the same matrix,
        # so every timestep adds a term to its gradient.
        for key in PARAMETER_KEYS:
            accumulated[f"d{key}"] += grads_t[f"d{key}"]

        dx[:, :, t] = grads_t["dxt"]
        da_next = grads_t["da_prev"]  # hand both recurrences back to t-1
        dc_next = grads_t["dc_prev"]

    # After the loop, the two recurrence gradients have reached the initial
    # states a^{<0>} and c^{<0>}.
    return {**accumulated, "da0": da_next, "dc0": dc_next, "dx": dx}


def make_fixture() -> dict:
    """
    Build the deterministic fixture every LSTM implementation and test shares.

    Draws are recorded in exactly the order the topic dossier pins, under
    ``np.random.seed(1)`` and ``float64`` (the default of ``randn``), so the
    arrays reproduce bit-for-bit across implementations. The one-hot targets are
    drawn *last* so that adding them cannot disturb the arrays a forward-parity
    check compares. Dimensions match the vanilla RNN fixture, so the notebook can
    contrast the two cells on identical shapes.

    Returns
    -------
    dict
        ``{"x", "a0", "c0", "parameters", "y", "dims"}`` where ``parameters`` is
        the dict ``{"Wf", "Wu", "Wc", "Wo", "Wy", "bf", "bu", "bc", "bo",
        "by"}``, ``y`` is one-hot of shape ``(n_y, m, T_x)``, and ``dims``
        records ``n_x, n_a, n_y, m, T_x``.
    """
    np.random.seed(1)
    n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4

    x = np.random.randn(n_x, m, T_x)  # 1
    a0 = np.random.randn(n_a, m)  # 2
    c0 = np.random.randn(n_a, m)  # 3
    Wf = np.random.randn(n_a, n_a + n_x)  # 4
    Wu = np.random.randn(n_a, n_a + n_x)  # 5
    Wc = np.random.randn(n_a, n_a + n_x)  # 6
    Wo = np.random.randn(n_a, n_a + n_x)  # 7
    Wy = np.random.randn(n_y, n_a)  # 8
    bf = np.random.randn(n_a, 1)  # 9
    bu = np.random.randn(n_a, 1)  # 10
    bc = np.random.randn(n_a, 1)  # 11
    bo = np.random.randn(n_a, 1)  # 12
    by = np.random.randn(n_y, 1)  # 13

    # Targets, drawn after the thirteen arrays above.
    labels = np.random.randint(0, n_y, size=(m, T_x))
    y = np.zeros((n_y, m, T_x))
    for i in range(m):
        for t in range(T_x):
            y[labels[i, t], i, t] = 1.0  # one-hot encode the class at (i, t)

    parameters = {
        "Wf": Wf,
        "Wu": Wu,
        "Wc": Wc,
        "Wo": Wo,
        "Wy": Wy,
        "bf": bf,
        "bu": bu,
        "bc": bc,
        "bo": bo,
        "by": by,
    }
    dims = {"n_x": n_x, "n_a": n_a, "n_y": n_y, "m": m, "T_x": T_x}
    return {"x": x, "a0": a0, "c0": c0, "parameters": parameters, "y": y, "dims": dims}


def main() -> None:
    """Run the shared fixture end to end and print shapes, loss, and grad norms."""
    fixture = make_fixture()
    x, a0, c0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["c0"],
        fixture["parameters"],
        fixture["y"],
    )
    n_a = fixture["dims"]["n_a"]

    a, c, y_pred, caches = lstm_forward(x, parameters, a0, c0)
    loss = compute_loss(y_pred, y)
    gradients = lstm_backward(y, caches, parameters)

    print(f"hidden states a: {a.shape}")
    print(f"cell states c:   {c.shape}")
    print(f"outputs y_pred:  {y_pred.shape}")
    print(f"total loss:      {loss:.6f}")
    for key in ("dWf", "dWu", "dWc", "dWo", "dWy", "da0", "dc0"):
        print(f"||{key}|| = {np.linalg.norm(gradients[key]):.6f}")

    # The concatenated gate gradient still names its two halves: the recurrent
    # path dW_ga and the input path dW_gx, split at column n_a.
    for key in ("dWf", "dWu", "dWc", "dWo"):
        recurrent, input_ = split_gate_matrix(gradients[key], n_a)
        print(
            f"{key} = [recurrent | input] -> "
            f"||{key}a|| = {np.linalg.norm(recurrent):.6f}, "
            f"||{key}x|| = {np.linalg.norm(input_):.6f}"
        )


if __name__ == "__main__":
    main()
