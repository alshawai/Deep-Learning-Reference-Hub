"""
Word Embeddings Reference Tests
===============================

The correctness proof for the hub's static word-embedding reference. This topic
has **no learned parameters and no gradients** -- the vectors are a planted toy
matrix, not a training run -- so there is no finite-difference gradient check to
run. The role a gradient check plays elsewhere (catching a confidently-wrong
sign) is played here by *exact* anchors: a hand-built fixture whose every cosine,
neighbor, and analogy is a rational number computed by hand in the dossier. A
flipped sign in the analogy target, or a transposed embedding matrix, moves one
of those answers off its known value.

Three contracts, matching the hub's reference-module discipline:

- **Known answers.** The lookup identity ``E @ one_hot(i) == E[:, i]``; the
  rational cosine anchors ``cos(king, queen) = 0.6`` and ``cos(man, woman) =
  -1``; the full nearest-neighbor table; and both exact analogies, with the input
  words excluded.
- **Determinism.** ``make_fixture`` is a literal (the same matrix every call) and
  the PCA projection is byte-identical across repeated calls thanks to its sign
  convention.
- **Shape and edge contracts.** The shapes the dossier promises, plus the
  zero-vector cosine guard and the input-range guards.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.embeddings.word_embeddings import (
    PCAProjection,
    analogy,
    cosine_similarity,
    lookup,
    make_fixture,
    nearest_neighbors,
    one_hot,
    pca_project_2d,
)

# The dossier's declared tolerances.
COSINE_ATOL = 1e-12
PCA_ATOL = 1e-10


@pytest.fixture
def toy():
    """The planted toy embedding matrix and its vocabulary, per the dossier."""
    return make_fixture()


# --- The fixture itself --------------------------------------------------


def test_fixture_is_the_planted_matrix(toy):
    """
    Pin the fixture to the exact matrix the dossier publishes. If this drifts,
    every hand-computed anchor below is checking the wrong numbers.
    """
    E = toy["E"]
    assert E.shape == (4, 6)
    assert E.dtype == np.float64
    assert toy["vocab"] == ["king", "man", "prince", "princess", "queen", "woman"]
    assert toy["dims"] == {"d": 4, "V": 6}
    expected = np.array(
        [
            [2, 0, 1, 1, 2, 0],  # royalty
            [1, 1, 1, -1, -1, -1],  # gender
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(E, expected)


def test_make_fixture_is_deterministic():
    """The literal matrix is identical across calls (no seed, no random draw)."""
    assert np.array_equal(make_fixture()["E"], make_fixture()["E"])


# --- Known answers: the lookup identity ----------------------------------


def test_lookup_identity(toy):
    """
    e_w = E o_w = E[:, i]: the one-hot matrix product and the column gather agree
    exactly, for every word. This is the module's spine -- an embedding layer is a
    bias-free linear layer applied to a one-hot input.
    """
    E, V = toy["E"], toy["dims"]["V"]
    for i in range(V):
        o = one_hot(i, V)
        assert o.shape == (V,)
        assert o.sum() == 1.0
        assert o[i] == 1.0
        assert np.array_equal(E @ o, lookup(E, i))
        assert np.array_equal(lookup(E, i), E[:, i])


# --- Known answers: cosine anchors ---------------------------------------


def test_cosine_rational_anchors(toy):
    """
    The two eyeball anchors the dossier fixes: a king is closer to a queen than to
    a man (royalty dominates gender), and man/woman are exact opposites on the
    gender axis.
    """
    E, ix = toy["E"], toy["word_to_ix"]
    cos_kq = cosine_similarity(E[:, ix["king"]], E[:, ix["queen"]])
    cos_mw = cosine_similarity(E[:, ix["man"]], E[:, ix["woman"]])
    assert cos_kq == pytest.approx(0.6, abs=COSINE_ATOL)
    assert cos_mw == pytest.approx(-1.0, abs=COSINE_ATOL)


def test_cosine_is_symmetric_self_is_one_and_bounded(toy):
    """Cosine is symmetric, a self-similarity is 1, and every value is in [-1, 1]."""
    E, V = toy["E"], toy["dims"]["V"]
    for i in range(V):
        assert cosine_similarity(E[:, i], E[:, i]) == pytest.approx(
            1.0, abs=COSINE_ATOL
        )
        for j in range(V):
            forward = cosine_similarity(E[:, i], E[:, j])
            backward = cosine_similarity(E[:, j], E[:, i])
            assert forward == pytest.approx(backward, abs=COSINE_ATOL)
            assert -1.0 - COSINE_ATOL <= forward <= 1.0 + COSINE_ATOL


# --- Known answers: the nearest-neighbor table ---------------------------

# The full top-1 table from the dossier (self excluded; all strict, no ties).
NEAREST = {
    "king": "prince",
    "queen": "princess",
    "prince": "king",
    "princess": "queen",
    "man": "prince",
    "woman": "princess",
}


def test_nearest_neighbor_table(toy):
    """
    Every top-1 row from the dossier. Each is strict (no ties), so a wrong
    neighbor means a wrong similarity -- the table is the anchor.
    """
    E, ix, vocab = toy["E"], toy["word_to_ix"], toy["vocab"]
    for word, expected in NEAREST.items():
        neighbors = nearest_neighbors(E, ix[word], k=1)
        assert len(neighbors) == 1
        j, _score = neighbors[0]
        assert vocab[j] == expected


def test_nearest_neighbors_excludes_query_and_ranks_descending(toy):
    """The query is never its own neighbor, and the scores come back sorted."""
    E, V = toy["E"], toy["dims"]["V"]
    for i in range(V):
        neighbors = nearest_neighbors(E, i, k=V - 1)
        indices = [j for j, _ in neighbors]
        scores = [s for _, s in neighbors]
        assert i not in indices
        assert len(indices) == V - 1
        assert scores == sorted(scores, reverse=True)


def test_nearest_neighbors_breaks_ties_by_ascending_index():
    """
    When two candidates tie on cosine, the lower index wins. Built so the query
    points in the same direction as two candidates at once -- an exact tie.
    """
    # Query at column 0; columns 1 and 2 are identical to it; column 3 opposes it.
    E = np.array([[1.0, 1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    neighbors = nearest_neighbors(E, 0, k=3)
    assert [j for j, _ in neighbors] == [1, 2, 3]


# --- Known answers: analogies --------------------------------------------


def test_analogy_man_woman_king_is_queen(toy):
    """
    The analogy man : woman :: king : ? -> woman - man + king = [2, -1, 0, 0] =
    queen, with the three input words excluded (else king wins on the cancellation).
    """
    E, ix, vocab = toy["E"], toy["word_to_ix"], toy["vocab"]
    answer = analogy(E, a=ix["man"], b=ix["woman"], c=ix["king"])
    assert vocab[answer] == "queen"


def test_analogy_prince_princess_king_is_queen(toy):
    """The second dossier instance: prince : princess :: king : ? -> also queen."""
    E, ix, vocab = toy["E"], toy["word_to_ix"], toy["vocab"]
    answer = analogy(E, a=ix["prince"], b=ix["princess"], c=ix["king"])
    assert vocab[answer] == "queen"


def test_analogy_excludes_its_inputs(toy):
    """The answer is never one of the three input words."""
    E, ix = toy["E"], toy["word_to_ix"]
    inputs = {ix["man"], ix["woman"], ix["king"]}
    answer = analogy(E, a=ix["man"], b=ix["woman"], c=ix["king"])
    assert answer not in inputs


# --- PCA: shapes, determinism, loss-less reconstruction, isometry --------


def test_pca_shapes_and_fields(toy):
    """The projection returns the shapes the dossier declares."""
    E, V, d = toy["E"], toy["dims"]["V"], toy["dims"]["d"]
    proj = pca_project_2d(E)
    assert isinstance(proj, PCAProjection)
    assert proj.coords.shape == (V, 2)
    assert proj.components.shape == (2, d)
    assert proj.mean.shape == (d,)


def test_pca_is_byte_identical_across_calls(toy):
    """
    The sign convention removes the SVD's sign ambiguity, so repeated calls
    produce byte-for-byte identical coordinates -- not merely close ones.
    """
    E = toy["E"]
    first = pca_project_2d(E).coords
    second = pca_project_2d(E).coords
    assert first.tobytes() == second.tobytes()


def test_pca_reconstruction_is_lossless(toy):
    """
    The toy data is rank-2, so projecting to 2-D and back recovers it exactly (to
    float tolerance): coords @ components + mean == E.T.
    """
    E = toy["E"]
    proj = pca_project_2d(E)
    reconstructed = proj.coords @ proj.components + proj.mean
    assert np.allclose(reconstructed, E.T, atol=PCA_ATOL)


def test_pca_preserves_pairwise_distances(toy):
    """
    An orthonormal projection onto the data's own plane is an isometry: every
    pairwise distance in 2-D equals the distance in the original 4-D space.
    """
    E = toy["E"]
    coords = pca_project_2d(E).coords
    X = E.T

    def pairwise(matrix):
        diff = matrix[:, None, :] - matrix[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    assert np.allclose(pairwise(coords), pairwise(X), atol=PCA_ATOL)


def test_pca_axes_are_gender_then_royalty(toy):
    """
    Gender is the higher-variance axis (sum of squares 6 vs royalty's 4), so PC1
    is the gender direction and PC2 the royalty direction. The sign convention
    points both along their positive axes.
    """
    proj = pca_project_2d(toy["E"])
    assert np.allclose(proj.components[0], [0, 1, 0, 0], atol=PCA_ATOL)
    assert np.allclose(proj.components[1], [1, 0, 0, 0], atol=PCA_ATOL)
    # Singular values squared are the explained variances: 6 then 4.
    assert proj.singular_values[0] ** 2 == pytest.approx(6.0, abs=PCA_ATOL)
    assert proj.singular_values[1] ** 2 == pytest.approx(4.0, abs=PCA_ATOL)


# --- Edge and shape contracts --------------------------------------------


def test_cosine_zero_vector_guard(toy):
    """
    Cosine is undefined for a zero vector; the guard returns 0.0 rather than a nan
    so a zero vector cannot poison a ranking.
    """
    E, d = toy["E"], toy["dims"]["d"]
    zero = np.zeros(d)
    assert cosine_similarity(zero, E[:, 0]) == 0.0
    assert cosine_similarity(E[:, 0], zero) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


def test_one_hot_shape_dtype_and_value():
    """one_hot is a (V,) float64 indicator with a single 1."""
    o = one_hot(2, 6)
    assert o.shape == (6,)
    assert o.dtype == np.float64
    assert np.array_equal(o, [0, 0, 1, 0, 0, 0])


def test_one_hot_rejects_bad_index_and_size():
    """An out-of-range index or empty vocabulary is a ValueError, not a silent array."""
    with pytest.raises(ValueError):
        one_hot(6, 6)
    with pytest.raises(ValueError):
        one_hot(-1, 6)
    with pytest.raises(ValueError):
        one_hot(0, 0)


def test_lookup_shape(toy):
    """A looked-up embedding is a (d,) column."""
    E, d = toy["E"], toy["dims"]["d"]
    assert lookup(E, 0).shape == (d,)


def test_nearest_neighbors_rejects_bad_args(toy):
    """An out-of-range query or k < 1 is a ValueError."""
    E, V = toy["E"], toy["dims"]["V"]
    with pytest.raises(ValueError):
        nearest_neighbors(E, V, k=1)
    with pytest.raises(ValueError):
        nearest_neighbors(E, 0, k=0)


def test_nearest_neighbors_clips_k_to_available(toy):
    """Asking for more neighbors than exist returns every other word, not an error."""
    E, V = toy["E"], toy["dims"]["V"]
    neighbors = nearest_neighbors(E, 0, k=100)
    assert len(neighbors) == V - 1


def test_analogy_rejects_bad_index(toy):
    """An out-of-range input index is a ValueError."""
    E, V = toy["E"], toy["dims"]["V"]
    with pytest.raises(ValueError):
        analogy(E, a=0, b=1, c=V)


def test_pca_rejects_non_2d():
    """PCA needs a (d, V) matrix, not a 1-D array."""
    with pytest.raises(ValueError):
        pca_project_2d(np.zeros((3,)))
