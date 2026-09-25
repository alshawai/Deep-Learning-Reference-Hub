"""
Shared Sequence-Cell Helpers
============================

The small pieces the gated cells in this package share: the logistic sigmoid
every gate is squashed by, and the column split that recovers the recurrent and
input halves of a concatenated gate matrix.

References
----------
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
- **Stable sigmoid.** The textbook form ``1 / (1 + exp(-z))`` overflows for very
  negative ``z``: ``exp(-z)`` becomes ``inf`` and NumPy issues a runtime
  warning before returning the (correct) ``0.0``. The implementation below
  exponentiates ``-|z|`` instead, which is always in ``(0, 1]``, and selects the
  algebraically equal form for each sign: ``1 / (1 + e^{-|z|})`` where
  ``z >= 0`` and ``e^{-|z|} / (1 + e^{-|z|})`` where ``z < 0``. Both branches are
  finite for every input, so no clipping and no epsilon are needed, and the
  result stays exact enough for the finite-difference gradient check.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Element-wise logistic sigmoid, in the overflow-free form.

    Computes ``sigma(z) = 1 / (1 + exp(-z))`` without ever exponentiating a
    positive number. See the module ``Notes`` for why the two branches are
    written this way.

    Parameters
    ----------
    z : np.ndarray
        Pre-activations of any shape.

    Returns
    -------
    np.ndarray
        Values in ``(0, 1)``, the same shape as ``z``.
    """
    exp_neg_abs = np.exp(-np.abs(z))  # in (0, 1] for every finite z
    return np.where(
        z >= 0,
        1.0 / (1.0 + exp_neg_abs),  # z >= 0: divide through by e^{z}
        exp_neg_abs / (1.0 + exp_neg_abs),  # z <  0: the unshifted form
    )


def split_gate_matrix(matrix: np.ndarray, n_a: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a concatenated gate matrix into its recurrent and input halves.

    The gated cells use one weight per gate acting on the stacked vector
    ``[a^{<t-1>}; x^{<t>}]``, so ``W_g = [W_ga | W_gx]``. The two halves are
    never needed separately by the forward or backward pass -- the point of the
    concatenated form is that they are not -- but they are worth naming when
    reporting a gradient, because ``dW_ga`` is the recurrent path and ``dW_gx``
    the input path, which is the split the vanilla RNN keeps explicit.

    Parameters
    ----------
    matrix : np.ndarray
        A gate weight or gate weight gradient, shape ``(n_a, n_a + n_x)``.
    n_a : int
        The hidden width, which is the column the split falls on.

    Returns
    -------
    recurrent : np.ndarray
        ``W_ga``, the first ``n_a`` columns, shape ``(n_a, n_a)``.
    input_ : np.ndarray
        ``W_gx``, the remaining columns, shape ``(n_a, n_x)``.

    Raises
    ------
    ValueError
        If ``matrix`` has no more than ``n_a`` columns, which would leave the
        input half empty.
    """
    if matrix.ndim != 2 or matrix.shape[1] <= n_a:
        raise ValueError(
            f"expected a 2-D matrix with more than {n_a} columns; got {matrix.shape}"
        )
    return matrix[:, :n_a], matrix[:, n_a:]
