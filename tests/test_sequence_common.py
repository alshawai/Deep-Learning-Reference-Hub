"""
Sequence Helper Tests
=====================

The sequence cells share four small helpers, and the quiet ones fail quietly. A
sigmoid that overflows returns the right number with a warning on one NumPy
build and a `nan` on the next; a column split off by one silently reports the
recurrent gradient as the input gradient. Neither shows up in a shape assertion,
so each is pinned here.

The stability test is the point of the sigmoid half of the module: it feeds
magnitudes large enough that the textbook `1 / (1 + exp(-z))` overflows, and
asserts the result is finite, in range, and still exact where the true value is
known. The softmax read-out and the cross-entropy loss are pinned the same way.

The last contract is structural rather than numeric: `_common` is the *single*
home for these helpers, so the vanilla RNN and the gated cells must all read the
same `softmax` and `compute_loss` object, and the per-tensor finite-difference
harness must stay out of the public API. Those are asserted so a future
reorganization cannot quietly reintroduce a second copy or promote a private
test tool.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.nn.sequence._common import (
    compute_loss,
    sigmoid,
    softmax,
    split_gate_matrix,
)

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


# --- The softmax read-out ------------------------------------------------


def test_softmax_columns_are_probability_distributions():
    """Each column sums to one and is strictly positive, so log is epsilon-free."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 7))

    probs = softmax(z)

    assert probs.shape == (4, 7)
    assert np.all(probs > 0.0)
    np.testing.assert_allclose(probs.sum(axis=0), np.ones(7), atol=1e-15)


def test_softmax_subtracts_the_max_so_large_logits_do_not_overflow():
    """
    The max-subtraction shift is the point: raw `exp` of these logits is `inf`.

    The output must stay a finite distribution, and shifting every logit in a
    column by a constant must not change the result (the shift cancels in the
    ratio), which is the invariant the stability trick rests on.
    """
    z = np.array([[1000.0, -1000.0], [1000.0, 0.0]])

    with np.errstate(over="raise"):  # a warning here would become an error
        probs = softmax(z)

    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=0), np.ones(2), atol=1e-15)
    np.testing.assert_allclose(softmax(z + 50.0), probs, atol=1e-15)


# --- The cross-entropy loss ----------------------------------------------


def test_compute_loss_matches_the_hand_computed_cross_entropy():
    """One example, one step: the loss is just -log of the true class's prob."""
    y_pred = np.array([[0.25], [0.75]]).reshape(2, 1, 1)
    y = np.array([[0.0], [1.0]]).reshape(2, 1, 1)

    assert compute_loss(y_pred, y) == pytest.approx(-np.log(0.75))


def test_compute_loss_averages_over_the_batch_and_sums_over_time():
    """With m identical columns the batch mean is one column's loss; T_x of them sum."""
    m, T_x = 4, 3
    y_pred = np.full((2, m, T_x), 0.5)
    y = np.zeros((2, m, T_x))
    y[0] = 1.0  # the first class is the target everywhere

    # Each (example, step) contributes -log(0.5); the 1/m mean leaves T_x of them.
    assert compute_loss(y_pred, y) == pytest.approx(T_x * -np.log(0.5))


# --- The single-home contract --------------------------------------------


def test_the_rnn_and_the_gated_cells_share_one_softmax_and_loss():
    """
    `_common` is the single home: every sequence module reads the same object.

    This is the structural point of the tidy-up. If a future edit reintroduces a
    `softmax` or `compute_loss` defined inside `rnn`, `lstm`, or `gru`, these
    identities break even though every numeric test still passes.
    """
    from dlhub.nn.sequence import _common, gru, lstm, rnn

    for module in (rnn, lstm, gru):
        assert module.softmax is _common.softmax
        assert module.compute_loss is _common.compute_loss


def test_the_public_names_stay_importable_from_the_rnn_module():
    """The read-out and loss must still import from where readers already reach."""
    from dlhub.nn.sequence import compute_loss as pkg_loss
    from dlhub.nn.sequence import softmax as pkg_softmax
    from dlhub.nn.sequence.rnn import compute_loss as rnn_loss
    from dlhub.nn.sequence.rnn import softmax as rnn_softmax

    assert rnn_softmax is softmax
    assert rnn_loss is compute_loss
    assert pkg_softmax is softmax
    assert pkg_loss is compute_loss


def test_the_per_tensor_gradient_check_stays_private():
    """
    The recorded decision: the sequence finite-difference harness stays private.

    The taught, public gradient check is `dlhub.training.gradient_checking`.
    Promoting this per-tensor harness to the package's public surface would
    recreate the two-things-one-name confusion the tidy-up removed, so the public
    API must not carry its names.
    """
    import dlhub.nn.sequence as seq

    harness_names = ("numeric_gradient", "relative_error", "global_norm", "FD_EPSILON")
    for name in harness_names:
        assert name not in seq.__all__
        assert not hasattr(seq, name)
