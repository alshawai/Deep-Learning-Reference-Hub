"""
Debiasing Word Embeddings Reference Tests
=========================================

The correctness proof for the hub's bias-geometry reference. Like its sibling
:mod:`tests.test_word_embeddings` -- and unlike the learned
:mod:`tests.test_word2vec_and_glove` -- this topic has **no learned parameters
and no gradients**: the vectors are a planted toy matrix, so there is no
finite-difference check to run. The role a gradient check plays elsewhere
(catching a confidently-wrong sign) is played here by the dossier's *exact*
known answers, hand-validated against the fixture: the gender direction is
``+e0``, the direct-bias projections are ``1/sqrt(19)``, ``1/sqrt(14)``,
``1/sqrt(10)``, WEAT reports ``d = 1.9830182615`` / statistic ``3.2516189672``
/ exact ``p = 0.05`` before debiasing and ``d = 0`` / ``p = 1.0`` after, the
neutral-neighbour map is identical before and after (the Gonen-Goldberg
climax), and equalize maps ``{grandfather, grandmother}`` onto the closed form
``(+/- sqrt(1 - nu^2), 0, 0, nu)`` with ``nu = (1/sqrt(5) + 1/sqrt(2)) / 2``.
A flipped sign rule, a transposed layout, or a wrong equalize formula moves one
of those answers off its known value.

Four contracts, matching the hub's reference-module discipline:

- **Known answers.** Everything above, at the dossier's declared tolerances:
  ``1e-12`` for the exact anchors, ``1e-9`` for WEAT ``d``/statistic before
  debiasing and the equalize output vectors.
- **Determinism.** ``make_fixture`` is a literal (the same matrix every call),
  and the whole pipeline -- ``gender_direction`` through ``hard_debias`` -- is
  byte-identical across two runs, sign rule included.
- **Shape and edge contracts.** The ``(V, d)`` layout, unit-norm rows out of
  every operation, and the guards the module names: neutralize's zero
  residual, equalize's ``||nu|| > 1`` and zero-direction raises, WEAT's
  unequal-target raise, and the ``d`` zero-spread guard.
- **The thesis.** Direct bias vanishes and WEAT says "fixed", yet the
  neighbour map survives bit-identical -- both halves asserted, because a hub
  that only tested the debiasing would teach the opposite of its point.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.embeddings.debiasing_word_embeddings import (
    equalize,
    gender_direction,
    hard_debias,
    make_fixture,
    neutral_neighbors,
    neutralize,
    weat_association,
    weat_effect_size,
    weat_pvalue,
    weat_statistic,
)

# The dossier's declared tolerances.
EXACT_ATOL = 1e-12  # g, projections, d_after, p, neighbour map, equalize symmetry
WEAT_ATOL = 1e-9  # d_before and statistic_before
EQUALIZE_ATOL = 1e-9  # the equalize output vectors

# The dossier's WEAT known answers (validated numerically 2026-10-14).
D_BEFORE = 1.9830182615
STAT_BEFORE = 3.2516189672
P_BEFORE = 0.05  # = 1/20: the true split uniquely maximizes the statistic

# The raw integer rows the fixture plants, in the sorted-vocabulary order;
# make_fixture L2-normalizes each to unit length.
RAW = np.array(
    [
        [1, 0, 0, 0],  # boy
        [1, 3, 3, 0],  # engineer
        [-1, 0, 0, 0],  # girl
        [2, 0, 0, 1],  # grandfather
        [-1, 0, 0, 1],  # grandmother
        [1, 0, 0, 0],  # he
        [1, 0, 0, 0],  # man
        [1, 3, 0, 0],  # mechanic
        [-1, -3, 0, 0],  # nanny
        [-1, -3, 3, 0],  # nurse
        [1, 3, 2, 0],  # programmer
        [-1, -3, 2, 0],  # receptionist
        [-1, 0, 0, 0],  # she
        [-1, 0, 0, 0],  # woman
    ],
    dtype=np.float64,
)

# The dossier's surviving-bias map: the gendered grouping, intact.
NEIGHBOUR_MAP = {
    "engineer": "programmer",
    "programmer": "engineer",
    "mechanic": "programmer",
    "nurse": "receptionist",
    "receptionist": "nurse",
    "nanny": "receptionist",
}


def rows_of(matrix, toy, words):
    """Gather the named words' rows out of a (V, d) matrix."""
    ix = toy["word_to_ix"]
    return matrix[[ix[word] for word in words]]


def weat_matrices(matrix, toy):
    """The fixture's four WEAT sets as row matrices taken from ``matrix``."""
    sets = toy["weat_sets"]
    return tuple(rows_of(matrix, toy, sets[name]) for name in ("A", "B", "X", "Y"))


@pytest.fixture
def toy():
    """The planted bias fixture: matrix, vocabulary, and role sets."""
    return make_fixture()


@pytest.fixture
def g(toy):
    """The fixture's gender direction, via the module's own pipeline."""
    ix = toy["word_to_ix"]
    pairs = [(ix[male], ix[female]) for male, female in toy["definitional_pairs"]]
    return gender_direction(toy["E"], pairs)


@pytest.fixture
def debiased(toy, g):
    """The matrix ``hard_debias`` produces from the fixture."""
    ix = toy["word_to_ix"]
    neutral = [ix[word] for word in toy["neutral_words"]]
    equality_sets = [[ix[word] for word in s] for s in toy["equality_sets"]]
    return hard_debias(toy["E"], g, neutral, equality_sets)


# --- The fixture itself --------------------------------------------------


def test_fixture_is_the_planted_matrix(toy):
    """
    Pin the fixture to the exact matrix and role sets the dossier publishes.
    If this drifts, every hand-validated anchor below checks the wrong numbers.
    """
    E = toy["E"]
    assert E.shape == (14, 4)
    assert E.dtype == np.float64
    assert toy["vocab"] == [
        "boy",
        "engineer",
        "girl",
        "grandfather",
        "grandmother",
        "he",
        "man",
        "mechanic",
        "nanny",
        "nurse",
        "programmer",
        "receptionist",
        "she",
        "woman",
    ]
    assert toy["dims"] == {"V": 14, "d": 4}
    assert toy["definitional_pairs"] == [
        ("he", "she"),
        ("man", "woman"),
        ("boy", "girl"),
    ]
    assert toy["equality_sets"] == [["grandfather", "grandmother"]]
    assert toy["weat_sets"] == {
        "A": ["he", "man", "boy"],
        "B": ["she", "woman", "girl"],
        "X": ["engineer", "programmer", "mechanic"],
        "Y": ["nurse", "receptionist", "nanny"],
    }
    assert set(toy["neutral_words"]) == set(toy["weat_sets"]["X"]) | set(
        toy["weat_sets"]["Y"]
    )
    expected = RAW / np.linalg.norm(RAW, axis=1, keepdims=True)
    assert np.allclose(E, expected, rtol=0.0, atol=EXACT_ATOL)


def test_fixture_rows_are_unit_norm(toy):
    """Normalize-first: every row is on the unit sphere, as equalize requires."""
    norms = np.linalg.norm(toy["E"], axis=1)
    assert np.allclose(norms, 1.0, rtol=0.0, atol=EXACT_ATOL)


def test_make_fixture_is_deterministic():
    """The literal matrix is identical across calls (no seed, no training)."""
    assert np.array_equal(make_fixture()["E"], make_fixture()["E"])


# --- Known answers: the gender direction ----------------------------------


def test_gender_direction_is_male_positive_first_basis_vector(toy, g):
    """The dossier's g == +e0 exactly, with the male-positive sign rule holding."""
    assert g.shape == (4,)
    assert np.allclose(g, [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=EXACT_ATOL)
    E, ix = toy["E"], toy["word_to_ix"]
    assert np.dot(g, E[ix["he"]] - E[ix["she"]]) > 0.0


def test_definitional_deviations_are_rank_one(toy):
    """
    The deviation spectrum is [sqrt(6), 0, 0, 0] (dossier: [2.449, 0, 0, 0]):
    the second singular value is ~0, so a single direction (k = 1) captures the
    whole gender subspace on this fixture.
    """
    E, ix = toy["E"], toy["word_to_ix"]
    deviations = []
    for male, female in toy["definitional_pairs"]:
        mu = (E[ix[male]] + E[ix[female]]) / 2.0
        deviations.append(E[ix[male]] - mu)
        deviations.append(E[ix[female]] - mu)
    singular_values = np.linalg.svd(np.array(deviations), compute_uv=False)
    assert singular_values[0] == pytest.approx(np.sqrt(6.0), abs=EXACT_ATOL)
    assert np.allclose(singular_values[1:], 0.0, rtol=0.0, atol=EXACT_ATOL)


def test_gender_direction_is_deterministic(toy):
    """Two runs give a byte-identical g -- the sign rule removes the SVD ambiguity."""
    ix = toy["word_to_ix"]
    pairs = [(ix[male], ix[female]) for male, female in toy["definitional_pairs"]]
    first = gender_direction(toy["E"], pairs)
    second = gender_direction(toy["E"], pairs)
    assert first.tobytes() == second.tobytes()


# --- Known answers: direct bias -------------------------------------------


def test_direct_bias_projections(toy, g):
    """
    The exact direct-bias anchors: the male occupations project on g with
    1/sqrt(19), 1/sqrt(14), 1/sqrt(10) (norms of their raw integer rows), and
    the female occupations are the exact negatives.
    """
    E, ix = toy["E"], toy["word_to_ix"]
    expected = {
        "engineer": 1.0 / np.sqrt(19.0),
        "programmer": 1.0 / np.sqrt(14.0),
        "mechanic": 1.0 / np.sqrt(10.0),
        "nurse": -1.0 / np.sqrt(19.0),
        "receptionist": -1.0 / np.sqrt(14.0),
        "nanny": -1.0 / np.sqrt(10.0),
    }
    for word, value in expected.items():
        assert E[ix[word]] @ g == pytest.approx(value, abs=EXACT_ATOL)


# --- Known answers: WEAT before debiasing ----------------------------------


def test_weat_association_anchors(toy):
    """
    s(w, A, B) = 2 * (w . g) on the fixture, because its attributes are the
    pure-gender archetypes +/-e0: cos with A is w . g and cos with B its negative.
    """
    A, B, _, _ = weat_matrices(toy["E"], toy)
    engineer = rows_of(toy["E"], toy, ["engineer"])[0]
    nurse = rows_of(toy["E"], toy, ["nurse"])[0]
    assert weat_association(engineer, A, B) == pytest.approx(
        2.0 / np.sqrt(19.0), abs=EXACT_ATOL
    )
    assert weat_association(nurse, A, B) == pytest.approx(
        -2.0 / np.sqrt(19.0), abs=EXACT_ATOL
    )


def test_weat_statistic_before(toy):
    """The dossier's statistic anchor: 3.2516189672."""
    A, B, X, Y = weat_matrices(toy["E"], toy)
    assert weat_statistic(X, Y, A, B) == pytest.approx(STAT_BEFORE, abs=WEAT_ATOL)


def test_weat_effect_size_before(toy):
    """The dossier's hard known answer: d = 1.9830182615 (population std)."""
    A, B, X, Y = weat_matrices(toy["E"], toy)
    assert weat_effect_size(X, Y, A, B) == pytest.approx(D_BEFORE, abs=WEAT_ATOL)


def test_weat_pvalue_before_is_exact(toy):
    """
    Exact p: 1/20 = 0.05 -- the 20 equal-size partitions are enumerated, and
    the true split uniquely maximizes the statistic, so only it counts.
    """
    A, B, X, Y = weat_matrices(toy["E"], toy)
    assert weat_pvalue(X, Y, A, B) == P_BEFORE


# --- Known answers: after hard_debias --------------------------------------


def test_neutral_projections_vanish_after_hard_debias(toy, g, debiased):
    """
    Every neutral word's projection on g is 0: the direct bias is gone. (The
    definitional words keep theirs by design, and the equalized pair carries
    +/- the shared g-component -- that is what equalize is for.)
    """
    ix = toy["word_to_ix"]
    for word in toy["neutral_words"]:
        assert debiased[ix[word]] @ g == pytest.approx(0.0, abs=EXACT_ATOL)


def test_hard_debias_preserves_unit_norms(toy, debiased):
    """Every debiased row is back on the unit sphere (normalize-then-debias)."""
    norms = np.linalg.norm(debiased, axis=1)
    assert np.allclose(norms, 1.0, rtol=0.0, atol=EXACT_ATOL)


def test_hard_debias_leaves_definitional_words_alone(toy, debiased):
    """The words that defined g are untouched -- neutralizing he would annihilate it."""
    ix = toy["word_to_ix"]
    definitional = [word for pair in toy["definitional_pairs"] for word in pair]
    for word in definitional:
        assert np.array_equal(debiased[ix[word]], toy["E"][ix[word]])


def test_hard_debias_does_not_mutate_its_input(toy, g):
    """hard_debias copies: the fixture matrix is byte-identical afterwards."""
    before = toy["E"].copy()
    ix = toy["word_to_ix"]
    neutral = [ix[word] for word in toy["neutral_words"]]
    equality_sets = [[ix[word] for word in s] for s in toy["equality_sets"]]
    hard_debias(toy["E"], g, neutral, equality_sets)
    assert np.array_equal(toy["E"], before)


def test_hard_debias_applies_equalize_to_the_equality_sets(toy, g, debiased):
    """The equality-set rows of the debiased matrix are equalize's own output."""
    ix = toy["word_to_ix"]
    for equality_set in toy["equality_sets"]:
        vectors = rows_of(toy["E"], toy, equality_set)
        assert np.array_equal(
            debiased[[ix[word] for word in equality_set]], equalize(vectors, g)
        )


def test_weat_after_hard_debias_is_exactly_unbiased(toy, debiased):
    """
    Debiasing silences WEAT exactly: d = 0 (the zero-spread guard -- every
    score is 0, so the pooled std is 0 and d is 0.0 by definition, not 0/0),
    and p = 1.0 (all 20 partitions tie at the observed statistic of 0).
    """
    A, B = weat_matrices(toy["E"], toy)[:2]
    X_after = rows_of(debiased, toy, toy["weat_sets"]["X"])
    Y_after = rows_of(debiased, toy, toy["weat_sets"]["Y"])
    assert weat_statistic(X_after, Y_after, A, B) == pytest.approx(0.0, abs=EXACT_ATOL)
    assert weat_effect_size(X_after, Y_after, A, B) == 0.0
    assert weat_pvalue(X_after, Y_after, A, B) == 1.0


# --- The thesis: the neighbour map survives --------------------------------


def test_neutral_neighbor_map_is_the_dossier_map(toy):
    """The exact map the dossier publishes, computed on the *biased* fixture."""
    ix, vocab = toy["word_to_ix"], toy["vocab"]
    neutral = [ix[word] for word in toy["neutral_words"]]
    mapping = neutral_neighbors(toy["E"], neutral)
    assert {vocab[i]: vocab[j] for i, j in mapping.items()} == NEIGHBOUR_MAP


def test_neutral_neighbor_map_survives_hard_debias(toy, debiased):
    """
    The Gonen-Goldberg climax: the map is *identical* before and after. Every
    direct projection is 0, WEAT says "fixed" -- and the words still group by
    gender through the cluster axis the debiasing never touched.
    """
    ix = toy["word_to_ix"]
    neutral = [ix[word] for word in toy["neutral_words"]]
    assert neutral_neighbors(toy["E"], neutral) == neutral_neighbors(debiased, neutral)


# --- Known answers: equalize ------------------------------------------------


def test_equalize_known_answer(toy, g):
    """
    The dossier's equalize anchor. Its printed (+/-0.81663, 0, 0, 0.57716) are
    five-decimal renderings of the closed form asserted here at 1e-9: the
    shared neutral part is nu = (0, 0, 0, nu3) with nu3 = (1/sqrt(5) +
    1/sqrt(2)) / 2 -- the pair's mean, whose g-component is removed -- and the
    g-components are +/- sqrt(1 - nu3^2).
    """
    vectors = rows_of(toy["E"], toy, ["grandfather", "grandmother"])
    out = equalize(vectors, g)

    nu3 = (1.0 / np.sqrt(5.0) + 1.0 / np.sqrt(2.0)) / 2.0
    gender_coord = np.sqrt(1.0 - nu3**2)
    assert out.shape == (2, 4)
    assert np.allclose(
        out[0], [gender_coord, 0.0, 0.0, nu3], rtol=0.0, atol=EQUALIZE_ATOL
    )
    assert np.allclose(
        out[1], [-gender_coord, 0.0, 0.0, nu3], rtol=0.0, atol=EQUALIZE_ATOL
    )
    # The dossier's printed five-decimal values.
    assert gender_coord == pytest.approx(0.81663, abs=1e-5)
    assert nu3 == pytest.approx(0.57716, abs=1e-5)


def test_equalize_is_unit_norm_and_symmetric_about_g(toy, g, debiased):
    """
    The equalized pair is unit-norm, with equal-and-opposite g components and
    a shared neutral part (the midpoint sits exactly off g).
    """
    ix = toy["word_to_ix"]
    grandfather = debiased[ix["grandfather"]]
    grandmother = debiased[ix["grandmother"]]

    assert np.linalg.norm(grandfather) == pytest.approx(1.0, abs=EXACT_ATOL)
    assert np.linalg.norm(grandmother) == pytest.approx(1.0, abs=EXACT_ATOL)
    assert grandfather @ g == pytest.approx(-(grandmother @ g), abs=EXACT_ATOL)
    midpoint = (grandfather + grandmother) / 2.0
    assert midpoint @ g == pytest.approx(0.0, abs=EXACT_ATOL)


def test_equalize_is_equidistant_from_every_neutralized_word(toy, debiased, g):
    """
    Both members sit sqrt(2) from every neutralized word: for unit vectors with
    w' . u = 0 (the equalized g-parts meet a vector orthogonal to g, and the
    shared nu is orthogonal to the occupations' cluster plane), ||w' - u||^2 =
    2 - 2 w' . u = 2.
    """
    ix = toy["word_to_ix"]
    for member in ("grandfather", "grandmother"):
        w = debiased[ix[member]]
        for word in toy["neutral_words"]:
            u = debiased[ix[word]]
            distance = np.linalg.norm(w - u)
            assert distance == pytest.approx(np.sqrt(2.0), abs=EXACT_ATOL)


# --- Guards and edge contracts ----------------------------------------------


def test_neutralize_output_is_unit_and_orthogonal_to_g(toy, g):
    """Every neutralized word satisfies w' . g = 0 and ||w'|| = 1."""
    ix = toy["word_to_ix"]
    for word in toy["neutral_words"]:
        w = neutralize(toy["E"][ix[word]], g)
        assert w.shape == (4,)
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=EXACT_ATOL)
        assert w @ g == pytest.approx(0.0, abs=EXACT_ATOL)


def test_neutralize_pure_gender_word_guard(g):
    """
    A word entirely along g has no neutral part: the zero-residual guard
    returns the zero vector rather than dividing by zero (the head's cosine
    zero-guard keeps it out of rankings).
    """
    assert np.array_equal(neutralize(g, g), np.zeros(4))


def test_weat_effect_size_zero_spread_guard():
    """
    The d zero-spread guard on its own terms: every pooled score identical ->
    std = 0 -> d = 0.0 by definition, not a 0/0 nan.
    """
    A = np.array([[1.0, 0.0]])
    B = np.array([[-1.0, 0.0]])
    X = np.array([[0.0, 1.0]])
    Y = np.array([[0.0, 1.0]])
    assert weat_association(X[0], A, B) == 0.0
    assert weat_effect_size(X, Y, A, B) == 0.0


def test_gender_direction_rejects_empty_pairs(toy):
    """No definitional pairs means no direction to identify."""
    with pytest.raises(ValueError):
        gender_direction(toy["E"], [])


def test_equalize_rejects_non_unit_vectors(g):
    """
    Off the unit sphere, ||nu|| can exceed 1 and sqrt(1 - ||nu||^2) is not
    real: equalize raises instead of returning nan (normalize-first).
    """
    vectors = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        equalize(vectors, g)


def test_equalize_rejects_zero_direction(g):
    """A member whose gender component equals the set mean's has no direction to scale."""
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        equalize(vectors, g)


def test_hard_debias_rejects_overlapping_sets(toy, g):
    """A word cannot be both neutralized and equalized; the overlap is caught."""
    ix = toy["word_to_ix"]
    with pytest.raises(ValueError):
        hard_debias(
            toy["E"],
            g,
            [ix["engineer"]],
            [[ix["engineer"], ix["grandmother"]]],
        )


def test_weat_pvalue_rejects_unequal_target_sets():
    """The permutation test redistributes into equal-size partitions only."""
    A = np.array([[1.0, 0.0]])
    B = np.array([[-1.0, 0.0]])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError):
        weat_pvalue(X, Y, A, B)


def test_neutral_neighbors_rejects_single_word(toy):
    """One neutral word has no neighbour within the set."""
    ix = toy["word_to_ix"]
    with pytest.raises(ValueError):
        neutral_neighbors(toy["E"], [ix["engineer"]])


# --- Determinism of the whole pipeline ---------------------------------------


def test_pipeline_is_deterministic(toy):
    """Two runs of the full pipeline are byte-identical: g and the debiased matrix."""

    def run():
        ix = toy["word_to_ix"]
        pairs = [(ix[male], ix[female]) for male, female in toy["definitional_pairs"]]
        direction = gender_direction(toy["E"], pairs)
        neutral = [ix[word] for word in toy["neutral_words"]]
        equality_sets = [[ix[word] for word in s] for s in toy["equality_sets"]]
        return direction, hard_debias(toy["E"], direction, neutral, equality_sets)

    first_g, first_debiased = run()
    second_g, second_debiased = run()
    assert first_g.tobytes() == second_g.tobytes()
    assert first_debiased.tobytes() == second_debiased.tobytes()
