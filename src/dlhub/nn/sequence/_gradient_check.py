"""
Per-Tensor Gradient Check
=========================

The finite-difference machinery the sequence suites check their hand-derived
backpropagation with: a central-difference gradient for one tensor, the
norm-based relative error that compares it against the analytic gradient, and
the global norm used to report a whole gradient set.

This is a test utility, not a deep learning implementation. It lives in the
package rather than beside the tests because three suites now share it -- the
vanilla RNN, the LSTM, and the GRU -- and a copy per suite is a copy that can
drift into disagreeing about what "agrees" means.

It is deliberately distinct from :mod:`dlhub.training.gradient_checking`, which
checks a *whole network* by flattening a parameter dictionary into one vector
and printing a verdict. This module is per-tensor and closure-driven: the caller
hands it the tensor to perturb and a zero-argument loss that reads that tensor
in place, which is what a recurrent forward pass over a fixture needs.

References
----------
- Ng, A. (2018). Deep Learning Specialization, Course 2 (Improving Deep Neural
  Networks), Week 1. https://www.coursera.org/specializations/deep-learning

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Central, not forward, differences.** ``(L(t+eps) - L(t-eps)) / (2 eps)`` has
  error ``O(eps^2)`` where the one-sided ``(L(t+eps) - L(t)) / eps`` has error
  ``O(eps)``. At the ``1e-7`` tolerance the sequence topics ask for, the forward
  difference would fail on a correct implementation.
- **Choosing ``eps`` is a trade, and the default is not always the right end of
  it.** Truncation error falls as ``O(eps^2)``, while the cancellation error of
  subtracting two nearly equal float64 losses grows: the resulting floor on the
  *relative* error of a gradient ``g`` is roughly ``ulp(L) / (2 eps ||g||)``.
  That floor is what a small gradient runs into first. At ``eps = 1e-7`` with a
  loss of order 5, a gradient of norm 0.06 -- the GRU's reset-gate gradient on
  its fixture -- cannot score better than about ``1.2e-7`` however correct it
  is, which is above the tolerance those topics declare. The vanilla RNN suite
  uses the default; the gated-cell suites pass ``eps = 1e-6``, which lowers the
  floor tenfold and leaves truncation error near ``1e-12``, far below either.
- **The relative error is norm-based**, not elementwise: the whole tensor's
  difference is divided by the sum of the two norms. That is the standard
  gradient-check metric and it is bounded by 1, so a flipped sign scores close
  to 1 rather than infinity.
- **This harness stays private -- a recorded decision.** It keeps its leading
  underscore and is not exported from any package ``__all__``. The taught, public
  gradient check is :mod:`dlhub.training.gradient_checking`, with its own
  reference, how-to, and explanation pages; this is a test fixture that happens
  to live in the package so three suites can share one copy. Promoting it to a
  second public ``gradient_check`` surface -- same name, different API, different
  audience -- would recreate the confusion that splitting the two apart was meant
  to end. If a caller outside the test suites genuinely needs a per-tensor
  finite-difference primitive, revisit this note rather than quietly widening the
  import.
"""

from collections.abc import Callable

import numpy as np

# The finite-difference step every sequence suite perturbs by. See ``Notes``.
FD_EPSILON = 1e-7


def numeric_gradient(
    loss: Callable[[], float], theta: np.ndarray, eps: float = FD_EPSILON
) -> np.ndarray:
    """
    Central finite-difference gradient of ``loss`` with respect to ``theta``.

    ``loss`` must read ``theta`` in place (it is a closure over the same array),
    so each entry is perturbed by +/- eps, the loss re-evaluated, and the entry
    restored. The array is unchanged on return.

    Parameters
    ----------
    loss : Callable[[], float]
        Zero-argument scalar loss, closing over ``theta``.
    theta : np.ndarray
        The tensor to differentiate with respect to. Perturbed entry by entry
        and restored, so the caller's array is left exactly as it was.
    eps : float, optional
        The perturbation step. Defaults to :data:`FD_EPSILON`.

    Returns
    -------
    np.ndarray
        The approximate gradient, the same shape as ``theta``.
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
    """
    Norm-based relative difference, the metric the hub's gradient check uses.

    Parameters
    ----------
    analytic : np.ndarray
        The hand-derived (or autograd) gradient.
    numeric : np.ndarray
        The finite-difference approximation of the same gradient.

    Returns
    -------
    float
        ``||a - n|| / (||a|| + ||n||)``, which is ``0.0`` when both are zero and
        close to ``1.0`` when one is the negation of the other.
    """
    numerator = np.linalg.norm(analytic - numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    return 0.0 if denominator == 0 else float(numerator / denominator)


def global_norm(gradients: dict[str, np.ndarray]) -> float:
    """
    L2 norm of all gradients viewed as one concatenated vector.

    Parameters
    ----------
    gradients : dict[str, np.ndarray]
        Any gradient set, of any mix of shapes.

    Returns
    -------
    float
        The single global L2 norm.
    """
    return float(np.sqrt(sum(np.sum(g**2) for g in gradients.values())))
