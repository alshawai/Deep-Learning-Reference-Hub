"""
Minimal Gated Recurrent Cells (minGRU and minLSTM)
==================================================

A from-scratch, teaching reference for the two *minimal* gated cells of "Were
RNNs All We Needed?" (Feng et al., 2024). Section numbers quoted below are that
paper's, so any line here can be checked against it without reading it whole.

The reduction that makes these cells parallelisable is a single deletion: **the
gates stop depending on** ``a^{<t-1>}`` and read ``x^{<t>}`` alone (Sec. 3.1.1,
Sec. 3.2.1). For ``t = 1 ... T_x``::

    minGRU (Sec. 3.1.3)
        Gamma_u^{<t>} = sigma(W_u x^{<t>})
        a_tilde^{<t>} = W_c x^{<t>}
        a^{<t>} = (1 - Gamma_u^{<t>}) * a^{<t-1>} + Gamma_u^{<t>} * a_tilde^{<t>}

    minLSTM (Sec. 3.2.4)
        Gamma_f^{<t>}  = sigma(W_f x^{<t>})
        Gamma_u^{<t>}  = sigma(W_u x^{<t>})
        a_tilde^{<t>}  = W_c x^{<t>}
        Gamma_f'^{<t>} = Gamma_f^{<t>} / (Gamma_f^{<t>} + Gamma_u^{<t>})
        Gamma_u'^{<t>} = Gamma_u^{<t>} / (Gamma_f^{<t>} + Gamma_u^{<t>})
        a^{<t>} = Gamma_f'^{<t>} * a^{<t-1>} + Gamma_u'^{<t>} * a_tilde^{<t>}

Neither cell has bias terms, and neither has a recurrent weight half: every
weight here is ``(n_a, n_x)``. Both cells are then the *same* first-order
linear recurrence (Sec. 2.3)::

    a^{<t>} = alpha^{<t>} * a^{<t-1>} + beta^{<t>}

with ``alpha = 1 - Gamma_u`` and ``beta = Gamma_u * a_tilde`` for minGRU, and
``alpha = Gamma_f'`` and ``beta = Gamma_u' * a_tilde`` for minLSTM. Because
``alpha`` and ``beta`` depend on ``x`` alone, they are computable for every
timestep at once, so the sequence resolves by an **associative (parallel)
scan** rather than a serial loop. This module ships all three forms:
:func:`sequential_scan` is the definition, :func:`parallel_scan` is the same
function evaluated in a different association order, and :func:`log_space_scan`
is the numerically stable version of the parallel one (Sec. B.1).

What the reduction throws away -- the three details a reader arriving from the
classical cells is most likely to "fix" back:

- minGRU has **no reset gate** (Sec. 3.1.1). The candidate no longer reads the
  previous state, so there is nothing left to reset.
- minLSTM has **no output gate** (Sec. 3.2.3), and because ``a^{<t>} = c^{<t>}``
  the separate cell state collapses with it. The two remaining gates are
  normalised to sum to one, which gives the state a time-independent scale.
- **The candidate is a plain linear map: there is no tanh** (Sec. 3.1.2). See
  :func:`identity_candidate`.

Parameter counts follow from the deletions: minGRU is ``O(2 d_h d_x)`` against
the GRU's ``O(3 d_h (d_x + d_h))``, and minLSTM ``O(3 d_h d_x)`` against the
LSTM's ``O(4 d_h (d_x + d_h))``.

Shapes, in the sequence topic's convention (time last)::

    x(n_x, m, T_x)
    W_f, W_u, W_c(n_a, n_x)
    a0(n_a, m)
    alpha, beta, a(n_a, m, T_x)

Scope: this module derives **no BPTT**. The classical GRU and LSTM references
in this package carry the hand-derived, gradient-checked backward passes; what
is new here is the reduction and the scan, and a fourth backward derivation
would only repeat a lesson those two already teach.

References
----------
- Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H. (2024).
  Were RNNs All We Needed? arXiv:2410.01201.
  https://arxiv.org/abs/2410.01201
- Heinsen, F. A. (2023). Efficient Parallelization of a Ubiquitous Sequential
  Computation. arXiv:2311.06281. https://arxiv.org/abs/2311.06281
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation. EMNLP.
  https://arxiv.org/abs/1406.1078
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural
  Computation*, 9(8), 1735-1780.

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Softplus carries the gates in log space.** ``sigmoid`` is the shared,
  overflow-free helper the classical cells in this package use; ``softplus`` is
  added here, computed as ``log1p(exp(-|z|)) + max(z, 0)`` so it never
  exponentiates a positive number. The two are the same function seen twice,
  since ``sigma(z) = exp(-softplus(-z))``, which is what makes the log-space
  gates a rearrangement rather than a second derivation: ``log Gamma_u =
  -softplus(-k)`` and ``log(1 - Gamma_u) = -softplus(k)``, with no gate ever
  formed and then logged.
- **Why a log-space scan at all.** Resolving the recurrence needs the product
  of ``alpha`` across a prefix, and every ``alpha`` lies strictly inside
  ``(0, 1)``, so that product decays exponentially with the prefix: over 512
  steps of a nearly-closed gate it underflows to exactly zero even in float64.
  :func:`parallel_scan` never forms it as a bare number -- it composes the
  affine steps, so an underflowed decay merely zeroes a contribution that has
  genuinely vanished -- but any formulation that recovers ``beta`` by dividing
  through the running product divides by that zero, and in float32 the decay
  bottoms out far sooner. Sec. B.1 sidesteps the question by keeping every
  product as a *sum of logs* (Heinsen, 2023), where a 512-step decay is an
  ordinary negative number, and that is the form the paper ships. The price is
  the round trip through ``log`` and ``exp``: measured here, the two scans agree
  to about ``4e-15`` on the fixture and ``2e-12`` over 512 steps, which is why
  the equivalence tolerance is ``1e-10`` rather than machine epsilon.
- **The log-space form needs a positive state.** Logs of ``beta`` and of the
  initial state only exist when both are non-negative, which is why Sec. B.2.1
  replaces the candidate's identity with :func:`g`, a map that is strictly
  positive everywhere and still linear (slope 1) on ``x >= 0``. The two
  candidates give different cells, so an equivalence check compares the
  log-space scan against the *same* candidate, not against the linear one.
- **No epsilon in minLSTM's normaliser.** ``Gamma_f + Gamma_u`` is a sum of two
  strictly positive sigmoids, so the division in
  :func:`min_lstm_coefficients` is safe without a guard. It underflows only
  when both pre-activations fall below about ``-745``, where both gates round to
  zero -- precisely the regime :func:`min_lstm_log_coefficients` handles exactly,
  by taking the normaliser with ``logaddexp`` instead of a division.
- **No power-of-two assumption.** Both :func:`parallel_scan` and
  :func:`logcumsumexp` double their stride with ``while stride < T_x``, so a
  sequence of any length gets the ``ceil(log2 T_x)`` rounds it needs. The
  fixture's ``T_x = 7`` is odd and not a power of two on purpose: a scan that
  quietly assumes padding to 8 passes on a length of 4 and fails here.
"""

from collections.abc import Callable

import numpy as np

from dlhub.nn.sequence._common import sigmoid

# --- Elementwise primitives ---------------------------------------------


def softplus(z: np.ndarray) -> np.ndarray:
    """
    Stable ``log(1 + exp(z))``.

    Rewritten as ``log1p(exp(-|z|)) + max(z, 0)``, which is algebraically the
    same function but only ever exponentiates a non-positive number. See the
    module ``Notes``.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation of any shape.

    Returns
    -------
    np.ndarray
        ``log(1 + exp(z))``, elementwise, same shape as ``z``.
    """
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def identity_candidate(k: np.ndarray) -> np.ndarray:
    """
    The minimal cells' candidate activation: the identity (Sec. 3.1.2).

    The classical GRU and LSTM squash their candidate with ``tanh``. The
    minimal cells **do not**: the candidate is the plain linear map
    ``a_tilde^{<t>} = W_c x^{<t>}``. This function exists so the deletion has a
    name and a docstring, rather than being an absence a later reader restores
    by accident.

    Parameters
    ----------
    k : np.ndarray
        The candidate pre-activation ``W_c x``.

    Returns
    -------
    np.ndarray
        ``k`` itself, unchanged.
    """
    return k


def g(k: np.ndarray) -> np.ndarray:
    """
    The strictly positive candidate activation of Sec. B.2.1.

    ::

        g(k) = k + 0.5      for k >= 0
             = sigma(k)     for k <  0

    Continuous at zero (both branches give ``0.5``), positive everywhere, and
    still slope-1 linear on the positive half -- which is what lets the
    log-space scan take ``log(beta)`` at all.

    Parameters
    ----------
    k : np.ndarray
        The candidate pre-activation ``W_c x``.

    Returns
    -------
    np.ndarray
        Strictly positive values, same shape as ``k``.
    """
    return np.where(k >= 0.0, k + 0.5, sigmoid(k))


def log_g(k: np.ndarray) -> np.ndarray:
    """
    ``log(g(k))``, computed without forming ``g(k)`` first (Sec. B.2.1).

    ::

        log g(k) = log(k + 0.5)     for k >= 0
                 = -softplus(-k)    for k <  0

    The negative branch reuses the identity ``log sigma(k) = -softplus(-k)``,
    so no logarithm of an underflowed sigmoid is ever taken.

    Parameters
    ----------
    k : np.ndarray
        The candidate pre-activation ``W_c x``.

    Returns
    -------
    np.ndarray
        ``log(g(k))``, same shape as ``k``.
    """
    # The inner `where` keeps `log` away from the negative branch's inputs, so
    # the unused half of the expression cannot emit an invalid-value warning.
    positive_branch = np.where(k >= 0.0, k + 0.5, 1.0)
    return np.where(k >= 0.0, np.log(positive_branch), -softplus(-k))


# --- Gate pre-activations, for every timestep at once --------------------


def project_over_time(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply one weight matrix to every timestep in a single contraction.

    Computes ``W x^{<t>}`` for all ``t``, which is exactly the freedom the
    reduction buys: with no dependence on ``a^{<t-1>}``, nothing in a gate has
    to wait for the step before it.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape ``(n_a, n_x)``.
    x : np.ndarray
        Input sequence of shape ``(n_x, m, T_x)``.

    Returns
    -------
    np.ndarray
        Pre-activations of shape ``(n_a, m, T_x)``, whose slice at ``t`` equals
        ``W @ x[:, :, t]``.

    Raises
    ------
    ValueError
        If ``x`` is not three-dimensional.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (n_x, m, T_x); got ndim {x.ndim}")
    return np.tensordot(W, x, axes=([1], [0]))


def min_gru_coefficients(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The minGRU recurrence coefficients ``(alpha, beta)`` (Sec. 3.1.3).

    ::

        Gamma_u = sigma(W_u x)
        a_tilde = candidate(W_c x)
        alpha   = 1 - Gamma_u
        beta    = Gamma_u * a_tilde

    so that ``a^{<t>} = alpha^{<t>} * a^{<t-1>} + beta^{<t>}`` reproduces the
    cell's blend. There is no reset gate: the candidate never reads
    ``a^{<t-1>}``.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wu`` and ``Wc``, each ``(n_a, n_x)``.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation applied to ``W_c x``. Defaults to
        :func:`identity_candidate`, the paper's minimal form. Pass :func:`g` to
        match :func:`log_space_scan`'s positivity requirement, or ``np.tanh``
        to recover the classical GRU's candidate.

    Returns
    -------
    alpha : np.ndarray
        Per-step decay ``1 - Gamma_u``, shape ``(n_a, m, T_x)``.
    beta : np.ndarray
        Per-step injection ``Gamma_u * a_tilde``, shape ``(n_a, m, T_x)``.
    """
    k_u = project_over_time(parameters["Wu"], x)  # (n_a, m, T_x)
    k_c = project_over_time(parameters["Wc"], x)  # (n_a, m, T_x)

    gamma_u = sigmoid(k_u)  # update gate
    a_tilde = candidate(k_c)  # candidate state -- linear by default

    alpha = 1.0 - gamma_u  # weight on a^{<t-1>}
    beta = gamma_u * a_tilde  # weight on the candidate
    return alpha, beta


def min_lstm_coefficients(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The minLSTM recurrence coefficients ``(alpha, beta)`` (Sec. 3.2.4).

    ::

        Gamma_f  = sigma(W_f x)
        Gamma_u  = sigma(W_u x)
        a_tilde  = candidate(W_c x)
        Gamma_f' = Gamma_f / (Gamma_f + Gamma_u)
        Gamma_u' = Gamma_u / (Gamma_f + Gamma_u)
        alpha    = Gamma_f'
        beta     = Gamma_u' * a_tilde

    The normalisation is what replaces the missing output gate: because
    ``Gamma_f' + Gamma_u' = 1`` exactly, the state is a convex blend at every
    step and so keeps a time-independent scale.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wf``, ``Wu``, and ``Wc``, each ``(n_a, n_x)``.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation applied to ``W_c x``. Defaults to
        :func:`identity_candidate`.

    Returns
    -------
    alpha : np.ndarray
        The normalised forget gate ``Gamma_f'``, shape ``(n_a, m, T_x)``.
    beta : np.ndarray
        ``Gamma_u' * a_tilde``, shape ``(n_a, m, T_x)``.
    """
    k_f = project_over_time(parameters["Wf"], x)  # (n_a, m, T_x)
    k_u = project_over_time(parameters["Wu"], x)  # (n_a, m, T_x)
    k_c = project_over_time(parameters["Wc"], x)  # (n_a, m, T_x)

    gamma_f = sigmoid(k_f)  # forget gate
    gamma_u = sigmoid(k_u)  # update (input) gate
    a_tilde = candidate(k_c)  # candidate state -- linear by default

    # Normalise the two gates so they sum to one. No epsilon: the denominator
    # is a sum of two strictly positive sigmoids (see the module Notes).
    normaliser = gamma_f + gamma_u
    gamma_f_prime = gamma_f / normaliser
    gamma_u_prime = gamma_u / normaliser

    alpha = gamma_f_prime
    beta = gamma_u_prime * a_tilde
    return alpha, beta


def min_gru_log_coefficients(
    x: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    The minGRU coefficients in log space (Sec. B.1), using :func:`g`.

    Takes the gate logarithms straight from the pre-activation
    ``k_u = W_u x``, never forming the gate itself::

        log alpha = log(1 - Gamma_u) = -softplus(k_u)
        log beta  = log Gamma_u + log g(k_c) = -softplus(-k_u) + log_g(k_c)

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wu`` and ``Wc``, each ``(n_a, n_x)``.

    Returns
    -------
    log_alpha : np.ndarray
        ``log(alpha)``, shape ``(n_a, m, T_x)``.
    log_beta : np.ndarray
        ``log(beta)``, shape ``(n_a, m, T_x)``.
    """
    k_u = project_over_time(parameters["Wu"], x)
    k_c = project_over_time(parameters["Wc"], x)

    log_alpha = -softplus(k_u)  # log(1 - Gamma_u)
    log_beta = -softplus(-k_u) + log_g(k_c)  # log Gamma_u + log g(k_c)
    return log_alpha, log_beta


def min_lstm_log_coefficients(
    x: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    The minLSTM coefficients in log space (Sec. B.1), using :func:`g`.

    The gate normalisation becomes a subtraction::

        log Gamma_f = -softplus(-k_f)
        log Gamma_u = -softplus(-k_u)
        log(Gamma_f + Gamma_u) = logaddexp(log Gamma_f, log Gamma_u)
        log alpha = log Gamma_f - log(Gamma_f + Gamma_u)
        log beta  = log Gamma_u - log(Gamma_f + Gamma_u) + log_g(k_c)

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wf``, ``Wu``, and ``Wc``, each ``(n_a, n_x)``.

    Returns
    -------
    log_alpha : np.ndarray
        ``log(alpha)``, shape ``(n_a, m, T_x)``.
    log_beta : np.ndarray
        ``log(beta)``, shape ``(n_a, m, T_x)``.
    """
    k_f = project_over_time(parameters["Wf"], x)
    k_u = project_over_time(parameters["Wu"], x)
    k_c = project_over_time(parameters["Wc"], x)

    log_gamma_f = -softplus(-k_f)
    log_gamma_u = -softplus(-k_u)
    log_normaliser = np.logaddexp(log_gamma_f, log_gamma_u)

    log_alpha = log_gamma_f - log_normaliser
    log_beta = log_gamma_u - log_normaliser + log_g(k_c)
    return log_alpha, log_beta


# --- The three ways to resolve the recurrence ----------------------------


def _resolve_initial_state(a0: np.ndarray | None, n_a: int, m: int) -> np.ndarray:
    """
    Return ``a0``, defaulting to the zero state and checking its shape.

    Parameters
    ----------
    a0 : np.ndarray or None
        The caller's initial state, or None for the convention ``a^{<0>} = 0``.
    n_a, m : int
        Hidden width and batch size the coefficients were built with.

    Returns
    -------
    np.ndarray
        An ``(n_a, m)`` initial state.

    Raises
    ------
    ValueError
        If ``a0`` is supplied with the wrong shape.
    """
    if a0 is None:
        return np.zeros((n_a, m))
    if a0.shape != (n_a, m):
        raise ValueError(f"a0 must have shape {(n_a, m)}; got {a0.shape}")
    return a0


def sequential_scan(
    alpha: np.ndarray, beta: np.ndarray, a0: np.ndarray | None = None
) -> np.ndarray:
    """
    Resolve ``a^{<t>} = alpha^{<t>} a^{<t-1>} + beta^{<t>}`` one step at a time.

    This is the definition, written as the loop the recurrence describes. It is
    the reference the two scans below are checked against.

    Parameters
    ----------
    alpha, beta : np.ndarray
        Per-step coefficients, each ``(n_a, m, T_x)``.
    a0 : np.ndarray or None, optional
        Initial state ``a^{<0>}``, shape ``(n_a, m)``. Defaults to zero.

    Returns
    -------
    np.ndarray
        The hidden states, shape ``(n_a, m, T_x)``.

    Raises
    ------
    ValueError
        If ``alpha`` and ``beta`` disagree in shape, or ``a0`` is misshapen.
    """
    if alpha.shape != beta.shape:
        raise ValueError(f"alpha {alpha.shape} and beta {beta.shape} must match")

    n_a, m, T_x = alpha.shape
    a_prev = _resolve_initial_state(a0, n_a, m)

    a = np.zeros((n_a, m, T_x))
    for t in range(T_x):
        a_t = alpha[:, :, t] * a_prev + beta[:, :, t]
        a[:, :, t] = a_t
        a_prev = a_t  # carry the state forward to t+1
    return a


def parallel_scan(
    alpha: np.ndarray, beta: np.ndarray, a0: np.ndarray | None = None
) -> np.ndarray:
    """
    Resolve the same recurrence by an associative scan, in plain space.

    Each timestep is an affine map ``a -> alpha * a + beta``, and composing two
    of them -- first ``(alpha_1, beta_1)``, then ``(alpha_2, beta_2)`` -- gives
    another affine map::

        (alpha_1, beta_1) . (alpha_2, beta_2)
            = (alpha_2 * alpha_1,  alpha_2 * beta_1 + beta_2)

    Composition is associative, so the prefix compositions can be built by
    doubling the stride (the Hillis-Steele inclusive scan) in
    ``ceil(log2 T_x)`` rounds instead of ``T_x`` serial steps. After the
    doubling, entry ``t`` holds the single affine map from ``a^{<0>}`` to
    ``a^{<t>}``, which is then applied to the initial state.

    Parameters
    ----------
    alpha, beta : np.ndarray
        Per-step coefficients, each ``(n_a, m, T_x)``.
    a0 : np.ndarray or None, optional
        Initial state ``a^{<0>}``, shape ``(n_a, m)``. Defaults to zero.

    Returns
    -------
    np.ndarray
        The hidden states, shape ``(n_a, m, T_x)``, equal to
        :func:`sequential_scan`'s output up to floating-point association.

    Raises
    ------
    ValueError
        If ``alpha`` and ``beta`` disagree in shape, or ``a0`` is misshapen.
    """
    if alpha.shape != beta.shape:
        raise ValueError(f"alpha {alpha.shape} and beta {beta.shape} must match")

    n_a, m, T_x = alpha.shape
    a_start = _resolve_initial_state(a0, n_a, m)

    # prefix[t] is the composition of steps [t - span + 1 .. t]; it starts as
    # the single step t and its span doubles each round.
    prefix_alpha, prefix_beta = alpha.copy(), beta.copy()

    stride = 1
    while stride < T_x:
        # The earlier half of each pair: what steps [t - 2*stride + 1 .. t - stride]
        # already compose to. The later half is the entry being updated.
        left_alpha, left_beta = (
            prefix_alpha[:, :, :-stride],
            prefix_beta[:, :, :-stride],
        )
        right_alpha = prefix_alpha[:, :, stride:]
        right_beta = prefix_beta[:, :, stride:]

        composed_alpha = right_alpha * left_alpha
        composed_beta = right_alpha * left_beta + right_beta

        # Entries before `stride` already span all the history there is, so they
        # are carried through untouched.
        prefix_alpha = np.concatenate(
            [prefix_alpha[:, :, :stride], composed_alpha], axis=2
        )
        prefix_beta = np.concatenate(
            [prefix_beta[:, :, :stride], composed_beta], axis=2
        )
        stride *= 2

    return prefix_alpha * a_start[:, :, np.newaxis] + prefix_beta


def logcumsumexp(z: np.ndarray) -> np.ndarray:
    """
    Cumulative ``log(sum(exp(.)))`` along the last (time) axis.

    Built with the same stride-doubling as :func:`parallel_scan`, with
    ``np.logaddexp`` -- itself associative -- in place of affine composition.
    NumPy has no built-in for this, and the naive ``log(cumsum(exp(z)))``
    overflows exactly where a log-space scan is worth having.

    Parameters
    ----------
    z : np.ndarray
        Log-domain values, shape ``(..., T)``.

    Returns
    -------
    np.ndarray
        ``out[..., t] = log(sum_{i <= t} exp(z[..., i]))``, same shape as ``z``.
    """
    T = z.shape[-1]
    out = z.copy()

    stride = 1
    while stride < T:
        combined = np.logaddexp(out[..., stride:], out[..., :-stride])
        out = np.concatenate([out[..., :stride], combined], axis=-1)
        stride *= 2
    return out


def log_space_scan(
    log_alpha: np.ndarray, log_beta: np.ndarray, a0: np.ndarray | None = None
) -> np.ndarray:
    """
    Resolve the recurrence in log space -- the stable scan of Sec. B.1.

    Writing ``A_t = sum_{i <= t} log alpha_i`` for the cumulative log-decay, the
    closed form of the recurrence is::

        a^{<t>} = exp( A_t + log( sum_{j <= t} exp(log beta_j - A_j) ) )

    with the initial state entering as the ``j = 0`` term (coefficient one,
    hence ``log alpha_0 = 0``). Every product has become a sum and every sum a
    ``logaddexp``, so nothing underflows on the way. This is Heinsen's (2023)
    formulation, which the paper adopts.

    Parameters
    ----------
    log_alpha, log_beta : np.ndarray
        Logs of the per-step coefficients, each ``(n_a, m, T_x)``. Both require
        positive coefficients, which is why the candidate becomes :func:`g`.
    a0 : np.ndarray or None, optional
        Initial state ``a^{<0>}``, shape ``(n_a, m)``, and **non-negative**:
        the log of a negative state does not exist. Defaults to zero.

    Returns
    -------
    np.ndarray
        The hidden states, shape ``(n_a, m, T_x)``, all strictly positive.

    Raises
    ------
    ValueError
        If the two coefficient arrays disagree in shape, if ``a0`` is misshapen,
        or if any entry of ``a0`` is negative.
    """
    if log_alpha.shape != log_beta.shape:
        raise ValueError(
            f"log_alpha {log_alpha.shape} and log_beta {log_beta.shape} must match"
        )

    n_a, m, T_x = log_alpha.shape
    a_start = _resolve_initial_state(a0, n_a, m)
    if np.any(a_start < 0):
        raise ValueError("log_space_scan needs a non-negative a0; see module Notes")

    # Prepend the initial state as step 0: it is injected with coefficient one,
    # so its log-coefficient is zero and its log-injection is log(a0).
    zero_log_alpha = np.zeros((n_a, m, 1))
    with np.errstate(divide="ignore"):  # a0 == 0 is the default, and log 0 = -inf
        log_a_start = np.log(a_start)[:, :, np.newaxis]

    padded_log_alpha = np.concatenate([zero_log_alpha, log_alpha], axis=2)
    padded_log_beta = np.concatenate([log_a_start, log_beta], axis=2)

    cumulative_log_alpha = np.cumsum(padded_log_alpha, axis=2)  # A_t
    log_a = cumulative_log_alpha + logcumsumexp(padded_log_beta - cumulative_log_alpha)

    return np.exp(log_a[:, :, 1:])  # drop the prepended step 0


# --- Whole-cell forward passes -------------------------------------------


def min_gru_forward_sequential(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> np.ndarray:
    """
    Forward pass for minGRU, resolved by the serial recurrence.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wu`` and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Initial state, shape ``(n_a, m)``. Defaults to zero.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation, default :func:`identity_candidate`.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``.
    """
    alpha, beta = min_gru_coefficients(x, parameters, candidate)
    return sequential_scan(alpha, beta, a0)


def min_gru_forward_parallel(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> np.ndarray:
    """
    Forward pass for minGRU, resolved by the plain-space associative scan.

    Same function as :func:`min_gru_forward_sequential`, evaluated in a
    different association order.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wu`` and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Initial state, shape ``(n_a, m)``. Defaults to zero.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation, default :func:`identity_candidate`.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``.
    """
    alpha, beta = min_gru_coefficients(x, parameters, candidate)
    return parallel_scan(alpha, beta, a0)


def min_gru_forward_log(
    x: np.ndarray, parameters: dict[str, np.ndarray], a0: np.ndarray | None = None
) -> np.ndarray:
    """
    Forward pass for minGRU through the log-space scan (Sec. B.1).

    This is the *positive-candidate* cell: the candidate is :func:`g`, not the
    identity, because a log-space scan needs ``beta > 0``. It therefore matches
    :func:`min_gru_forward_sequential` called with ``candidate=g``, and not the
    default linear one.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wu`` and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Non-negative initial state, shape ``(n_a, m)``. Defaults to zero.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``, strictly positive.
    """
    log_alpha, log_beta = min_gru_log_coefficients(x, parameters)
    return log_space_scan(log_alpha, log_beta, a0)


def min_lstm_forward_sequential(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> np.ndarray:
    """
    Forward pass for minLSTM, resolved by the serial recurrence.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wf``, ``Wu``, and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Initial state, shape ``(n_a, m)``. Defaults to zero.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation, default :func:`identity_candidate`.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``.
    """
    alpha, beta = min_lstm_coefficients(x, parameters, candidate)
    return sequential_scan(alpha, beta, a0)


def min_lstm_forward_parallel(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    a0: np.ndarray | None = None,
    candidate: Callable[[np.ndarray], np.ndarray] = identity_candidate,
) -> np.ndarray:
    """
    Forward pass for minLSTM, resolved by the plain-space associative scan.

    Same function as :func:`min_lstm_forward_sequential`, evaluated in a
    different association order.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wf``, ``Wu``, and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Initial state, shape ``(n_a, m)``. Defaults to zero.
    candidate : Callable[[np.ndarray], np.ndarray], optional
        Candidate activation, default :func:`identity_candidate`.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``.
    """
    alpha, beta = min_lstm_coefficients(x, parameters, candidate)
    return parallel_scan(alpha, beta, a0)


def min_lstm_forward_log(
    x: np.ndarray, parameters: dict[str, np.ndarray], a0: np.ndarray | None = None
) -> np.ndarray:
    """
    Forward pass for minLSTM through the log-space scan (Sec. B.1).

    As with :func:`min_gru_forward_log`, the candidate is :func:`g`, so this
    matches :func:`min_lstm_forward_sequential` called with ``candidate=g``.

    Parameters
    ----------
    x : np.ndarray
        Input sequence, shape ``(n_x, m, T_x)``.
    parameters : dict[str, np.ndarray]
        ``Wf``, ``Wu``, and ``Wc``, each ``(n_a, n_x)``.
    a0 : np.ndarray or None, optional
        Non-negative initial state, shape ``(n_a, m)``. Defaults to zero.

    Returns
    -------
    np.ndarray
        Hidden states ``a``, shape ``(n_a, m, T_x)``, strictly positive.
    """
    log_alpha, log_beta = min_lstm_log_coefficients(x, parameters)
    return log_space_scan(log_alpha, log_beta, a0)


# --- Fixture and demo ----------------------------------------------------


def make_fixture() -> dict:
    """
    Build the deterministic fixture this module's tests and documents share.

    Draws follow the order the topic dossier pins, under ``np.random.seed(1)``
    and ``float64``: ``x``, ``a0``, then minGRU's ``Wu``, ``Wc`` and minLSTM's
    ``Wf``, ``Wu``, ``Wc``. The dimensions match the sibling cell modules except
    for ``T_x = 7``, which is deliberately odd and not a power of two -- a scan
    that assumes a padded length passes at ``T_x = 4`` and fails here.

    Returns
    -------
    dict
        ``{"x", "a0", "min_gru", "min_lstm", "dims"}``, where ``min_gru`` is
        ``{"Wu", "Wc"}``, ``min_lstm`` is ``{"Wf", "Wu", "Wc"}``, and ``dims``
        records ``n_x, n_a, m, T_x``.
    """
    np.random.seed(1)
    n_x, n_a, m, T_x = 3, 5, 10, 7

    x = np.random.randn(n_x, m, T_x)  # 1
    a0 = np.random.randn(n_a, m)  # 2
    gru_Wu = np.random.randn(n_a, n_x)  # 3
    gru_Wc = np.random.randn(n_a, n_x)  # 4
    lstm_Wf = np.random.randn(n_a, n_x)  # 5
    lstm_Wu = np.random.randn(n_a, n_x)  # 6
    lstm_Wc = np.random.randn(n_a, n_x)  # 7

    dims = {"n_x": n_x, "n_a": n_a, "m": m, "T_x": T_x}
    return {
        "x": x,
        "a0": a0,
        "min_gru": {"Wu": gru_Wu, "Wc": gru_Wc},
        "min_lstm": {"Wf": lstm_Wf, "Wu": lstm_Wu, "Wc": lstm_Wc},
        "dims": dims,
    }


def main() -> None:
    """Run the fixture through all three forms and print the disagreements."""
    fixture = make_fixture()
    x, a0, dims = fixture["x"], fixture["a0"], fixture["dims"]
    n_a, n_x = dims["n_a"], dims["n_x"]
    positive_a0 = np.abs(a0)  # the log-space scan needs a non-negative state

    cells = (
        (
            "minGRU",
            fixture["min_gru"],
            min_gru_forward_sequential,
            min_gru_forward_parallel,
            min_gru_forward_log,
            2 * n_a * n_x,
        ),
        (
            "minLSTM",
            fixture["min_lstm"],
            min_lstm_forward_sequential,
            min_lstm_forward_parallel,
            min_lstm_forward_log,
            3 * n_a * n_x,
        ),
    )

    print(f"x: {x.shape}   a0: {a0.shape}   T_x = {dims['T_x']} (odd, not 2^k)")
    for name, parameters, sequential, parallel, log_space, count in cells:
        a_sequential = sequential(x, parameters, a0)
        a_parallel = parallel(x, parameters, a0)
        a_log = log_space(x, parameters, positive_a0)
        a_sequential_g = sequential(x, parameters, positive_a0, candidate=g)

        print(f"\n{name}: a {a_sequential.shape}, {count} parameters")
        print(
            "  max |sequential - parallel| = "
            f"{np.max(np.abs(a_sequential - a_parallel)):.3e}"
        )
        print(
            "  max |sequential(g) - log space| = "
            f"{np.max(np.abs(a_sequential_g - a_log)):.3e}"
        )


if __name__ == "__main__":
    main()
