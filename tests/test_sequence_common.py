"""
Sequence Helper Tests
=====================

The gated cells share two small helpers, and both fail quietly if they are
wrong. A sigmoid that overflows returns the right number with a warning on one
NumPy build and a `nan` on the next; a column split off by one silently reports
the recurrent gradient as the input gradient. Neither shows up in a shape
assertion, so each is pinned here.

The stability test is the point of the module: it feeds magnitudes large enough
that the textbook `1 / (1 + exp(-z))` overflows, and asserts the result is
finite, in range, and still exact where the true value is known.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.nn.sequence._common import sigmoid, split_gate_matrix

# Where the textbook form breaks: exp(710) overflows a float64.
OVERFLOW_SCALE = 1000.0


# --- The stable sigmoid --------------------------------------------------


def test_sigmoid_matches_the_textbook_form_in_the_safe_range():
    """Where `1 / (1 + exp(-z))` is computable, the stable form equals it."""
    z = np.linspace(-30.0, 30.0, 601)
    np.testing.assert_allclose(sigmoid(z), 1.0 / (1.0 + np.exp(-z)), atol=1e-15)


def test_sigmoid_survives_magnitudes_that_overflow_the_textbook_form():
    """
    At |z| = 1000 the naive form evaluates exp(1000) and overflows to inf.

    The stable form exponentiates -|z| instead, so both tails are computed from
    a number in (0, 1] and the result saturates to exactly 0 and 1 without a
    warning, an inf, or a nan.
    """
    z = np.array([-OVERFLOW_SCALE, 0.0, OVERFLOW_SCALE])

    with np.errstate(over="raise"):  # a warning here would become an error
        out = sigmoid(z)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-15)


def test_sigmoid_is_symmetric_about_a_half():
    """sigma(-z) = 1 - sigma(z), the identity the two branches must agree on."""
    z = np.array([[-8.0, -0.25, 0.0], [0.25, 3.5, 40.0]])
    np.testing.assert_allclose(sigmoid(-z), 1.0 - sigmoid(z), atol=1e-15)


def test_sigmoid_preserves_shape():
    """Gates are computed on (n_a, m) blocks and must come back that way."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 10))
    assert sigmoid(z).shape == (5, 10)


# --- The concatenated-gate split ----------------------------------------


def test_the_split_falls_on_column_n_a():
    """
    W_g = [W_ga | W_gx] splits at n_a: the recurrent half first, input second.

    An off-by-one here would mislabel a column of the gradient report without
    changing any norm enough to look wrong.
    """
    n_a, n_x = 5, 3
    matrix = np.arange(n_a * (n_a + n_x), dtype=float).reshape(n_a, n_a + n_x)

    recurrent, input_ = split_gate_matrix(matrix, n_a)

    assert recurrent.shape == (n_a, n_a)
    assert input_.shape == (n_a, n_x)
    np.testing.assert_array_equal(recurrent, matrix[:, :n_a])
    np.testing.assert_array_equal(input_, matrix[:, n_a:])
    np.testing.assert_array_equal(np.concatenate([recurrent, input_], axis=1), matrix)


def test_the_split_rejects_a_matrix_with_no_input_half():
    """A square (n_a, n_a) matrix has no input columns, so the split is a bug."""
    with pytest.raises(ValueError):
        split_gate_matrix(np.zeros((5, 5)), 5)


def test_the_split_rejects_a_one_dimensional_array():
    """A bias vector is not a gate matrix; splitting one is a caller error."""
    with pytest.raises(ValueError):
        split_gate_matrix(np.zeros(8), 5)
