"""
Word2Vec and GloVe Reference Tests
==================================

The correctness proof for the hub's *learned* word-embedding reference. Unlike
its sibling :mod:`tests.test_word_embeddings` -- where the vectors are a planted
matrix with no gradients -- this topic trains three real objectives, so the
central anchor is the one a training module most needs: every analytic gradient
is checked against a finite-difference approximation. A flipped sign in a
teaching gradient teaches confidently, so each check is paired with a
flipped-sign control that must be *rejected*.

Four contracts, matching the hub's reference-module discipline:

- **Known answers.** Finite-difference gradient checks for SGNS, CBOW, and GloVe
  (via the public :func:`dlhub.training.gradient_checking.gradient_check`), each
  with its flipped-sign rejection; the exact ``unigram^0.75`` sampler table; the
  integer-exact co-occurrence matrix; and the seeded ranking -- the pets family
  is recovered, ``cat``'s nearest neighbour is ``dog``.
- **Determinism.** ``make_fixture`` is a literal (same corpus, vocab, and pair
  list every call) and ``train_sgns(seed=0)`` is byte-identical across two runs.
- **Shape and edge contracts.** The ``(V, d)`` tables, the ``training_pairs``
  count and order, the unigram vector summing to 1, and the GloVe ``X_ij > 0``
  guard (a zero count contributes no term).
- **Learning.** The SGNS training loss strictly decreases every epoch, and the
  seeded ranking holds -- "these vectors are learned" is the topic's claim.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.embeddings.word2vec_and_glove import (
    build_vocabulary,
    cbow_gradients,
    cbow_loss,
    cooccurrence_matrix,
    glove_gradients,
    glove_loss,
    glove_weight,
    make_fixture,
    nearest_neighbor,
    sample_negatives,
    sgns_gradients,
    sgns_loss,
    sigmoid,
    train_glove,
    train_sgns,
    training_pairs,
    unigram_probabilities,
)
from dlhub.training.gradient_checking import gradient_check

# The dossier's declared tolerances.
GRAD_CHECK_RTOL = 1e-6  # relative error a correct analytic gradient clears
FLIPPED_SIGN_FLOOR = 0.9  # relative error a flipped-sign gradient must exceed
SAMPLER_ATOL = 1e-12  # the closed-form unigram table is exact
MIN_LOSS_DROP = 1.0  # a total SGNS loss drop no float noise could fake


@pytest.fixture
def fixture():
    """The planted toy corpus and its derived structures, per the dossier."""
    return make_fixture()


@pytest.fixture
def tables():
    """
    Seeded random tables at a generic point for the gradient checks.

    A gradient check is strongest away from any special point (where terms might
    vanish and hide a bug), so these are drawn fresh rather than taken from a
    trained run. ``V`` matches the fixture; ``d`` is deliberately not the
    training ``d`` to show the gradients do not depend on it.
    """
    rng = np.random.default_rng(2024)
    V, d = 16, 4
    return {
        "W": rng.normal(0.0, 0.4, size=(V, d)),
        "C": rng.normal(0.0, 0.4, size=(V, d)),
        "Wt": rng.normal(0.0, 0.4, size=(V, d)),
        "b": rng.normal(0.0, 0.4, size=V),
        "bt": rng.normal(0.0, 0.4, size=V),
        "V": V,
        "d": d,
    }


# A pair/context/negatives drawn by hand; the exact indices do not matter to a
# gradient check, only that every touched row is exercised.
_CENTRE, _CONTEXT = 2, 6  # cat, dog
_NEGATIVES = [1, 13]  # bread, pond -- a word from two other families
_CONTEXT_WORDS = [6, 10, 0]  # dog, fish, bird -- cat's co-occurring pets
_UNUSED = np.empty(0)  # gradient_check passes X, Y straight to the cost closure


# --- Known answers: gradient checks and their flipped-sign controls ------


def test_sgns_gradient_matches_finite_differences(tables):
    """
    The SGNS analytic gradient agrees with central differences of the loss.

    This is the module's core claim: the hand-derived ``dl/dv``, ``dl/du`` and
    ``dl/du_k`` are the true gradient of ``sgns_loss``. Agreement to the dossier
    tolerance means no sign, factor, or missing term is wrong.
    """
    params = {"W": tables["W"], "C": tables["C"]}
    grads = sgns_gradients(tables["W"], tables["C"], _CENTRE, _CONTEXT, _NEGATIVES)

    def cost(X, Y, p):
        return sgns_loss(p["W"], p["C"], _CENTRE, _CONTEXT, _NEGATIVES)

    difference = gradient_check(params, grads, _UNUSED, _UNUSED, cost)
    assert difference < GRAD_CHECK_RTOL


def test_sgns_flipped_gradient_is_rejected(tables):
    """
    A sign-flipped SGNS gradient fails the check -- the control that gives the
    check its teeth. A gradient check that passes a wrong gradient does nothing.
    """
    params = {"W": tables["W"], "C": tables["C"]}
    grads = sgns_gradients(tables["W"], tables["C"], _CENTRE, _CONTEXT, _NEGATIVES)
    flipped = {name: -value for name, value in grads.items()}

    def cost(X, Y, p):
        return sgns_loss(p["W"], p["C"], _CENTRE, _CONTEXT, _NEGATIVES)

    difference = gradient_check(params, flipped, _UNUSED, _UNUSED, cost)
    assert difference > FLIPPED_SIGN_FLOOR


def test_cbow_gradient_matches_finite_differences(tables):
    """
    The CBOW analytic gradient agrees with central differences, including the
    ``1 / M`` split of the gradient on the averaged context vector back to each
    context row.
    """
    params = {"W": tables["W"], "C": tables["C"]}
    grads = cbow_gradients(
        tables["W"], tables["C"], _CENTRE, _CONTEXT_WORDS, _NEGATIVES
    )

    def cost(X, Y, p):
        return cbow_loss(p["W"], p["C"], _CENTRE, _CONTEXT_WORDS, _NEGATIVES)

    difference = gradient_check(params, grads, _UNUSED, _UNUSED, cost)
    assert difference < GRAD_CHECK_RTOL


def test_cbow_flipped_gradient_is_rejected(tables):
    """A sign-flipped CBOW gradient fails the check."""
    params = {"W": tables["W"], "C": tables["C"]}
    grads = cbow_gradients(
        tables["W"], tables["C"], _CENTRE, _CONTEXT_WORDS, _NEGATIVES
    )
    flipped = {name: -value for name, value in grads.items()}

    def cost(X, Y, p):
        return cbow_loss(p["W"], p["C"], _CENTRE, _CONTEXT_WORDS, _NEGATIVES)

    difference = gradient_check(params, flipped, _UNUSED, _UNUSED, cost)
    assert difference > FLIPPED_SIGN_FLOOR


def test_glove_gradient_matches_finite_differences(fixture, tables):
    """
    The GloVe analytic gradient -- over both tables and both bias vectors --
    agrees with central differences on the fixture's co-occurrence matrix.
    """
    X = cooccurrence_matrix(fixture["sentences"], tables["V"])
    params = {
        "W": tables["W"],
        "Wt": tables["Wt"],
        "b": tables["b"],
        "bt": tables["bt"],
    }
    grads = glove_gradients(tables["W"], tables["Wt"], tables["b"], tables["bt"], X)

    def cost(Xa, Ya, p):
        return glove_loss(p["W"], p["Wt"], p["b"], p["bt"], X)

    difference = gradient_check(params, grads, _UNUSED, _UNUSED, cost)
    assert difference < GRAD_CHECK_RTOL


def test_glove_flipped_gradient_is_rejected(fixture, tables):
    """A sign-flipped GloVe gradient fails the check."""
    X = cooccurrence_matrix(fixture["sentences"], tables["V"])
    params = {
        "W": tables["W"],
        "Wt": tables["Wt"],
        "b": tables["b"],
        "bt": tables["bt"],
    }
    grads = glove_gradients(tables["W"], tables["Wt"], tables["b"], tables["bt"], X)
    flipped = {name: -value for name, value in grads.items()}

    def cost(Xa, Ya, p):
        return glove_loss(p["W"], p["Wt"], p["b"], p["bt"], X)

    difference = gradient_check(params, flipped, _UNUSED, _UNUSED, cost)
    assert difference > FLIPPED_SIGN_FLOOR


# --- Known answers: the sampler, co-occurrence, and the learned ranking --


def test_unigram_probabilities_match_the_closed_form(fixture):
    """
    The ``unigram^0.75`` table equals the closed-form one from the known
    frequencies. Every word's corpus frequency is fixed by construction (the hub
    words appear 3 times, the rest twice), so ``P`` is a hand-checkable table --
    this catches a miscount, a dropped exponent, or a missing normalisation.
    """
    sentences, V = fixture["sentences"], fixture["dims"]["V"]
    probabilities = unigram_probabilities(sentences, V)

    counts = np.zeros(V)
    for sentence in sentences:
        for token in sentence:
            counts[token] += 1
    expected = counts**0.75 / (counts**0.75).sum()

    assert probabilities == pytest.approx(expected, abs=SAMPLER_ATOL)
    assert probabilities.sum() == pytest.approx(1.0, abs=SAMPLER_ATOL)
    # The hub words (frequency 3) are sampled more than the rest (frequency 2).
    dog, cat = fixture["word_to_ix"]["dog"], fixture["word_to_ix"]["cat"]
    assert probabilities[dog] > probabilities[cat]


def test_exponent_flattens_the_distribution(fixture):
    """
    The ``3/4`` power pulls the distribution toward uniform: the ratio between a
    frequent and a rare word shrinks relative to the raw-frequency ratio. This is
    *why* the sampler oversamples rare words as negatives.
    """
    sentences, V = fixture["sentences"], fixture["dims"]["V"]
    dog, cat = fixture["word_to_ix"]["dog"], fixture["word_to_ix"]["cat"]

    raw = unigram_probabilities(sentences, V, exponent=1.0)
    smoothed = unigram_probabilities(sentences, V, exponent=0.75)

    assert smoothed[dog] / smoothed[cat] < raw[dog] / raw[cat]


def test_sample_negatives_draws_from_the_distribution(fixture):
    """
    ``sample_negatives`` returns ``k`` valid indices and threads a single
    generator, so repeated draws advance deterministically rather than repeat.
    """
    V = fixture["dims"]["V"]
    probabilities = unigram_probabilities(fixture["sentences"], V)
    rng = np.random.default_rng(0)

    first = sample_negatives(rng, probabilities, 2)
    second = sample_negatives(rng, probabilities, 2)

    assert first.shape == (2,)
    assert np.all((0 <= first) & (first < V))
    # A fresh generator reproduces the same first draw byte for byte.
    assert np.array_equal(
        first, sample_negatives(np.random.default_rng(0), probabilities, 2)
    )
    # The two draws came off one advancing generator, not a reset one.
    assert second.shape == (2,)


def test_cooccurrence_matrix_is_integer_exact(fixture):
    """
    ``X`` equals the hand-computed within-family counts exactly.

    Because the window (2) covers a whole 3-word sentence, ``X[i, j]`` is simply
    the number of sentences containing both words -- an independent formula the
    module's windowing must reproduce. Cross-family entries and the diagonal are
    zero, which is the planted structure GloVe fits.
    """
    sentences, V = fixture["sentences"], fixture["dims"]["V"]
    X = cooccurrence_matrix(sentences, V)

    expected = np.zeros((V, V), dtype=np.int64)
    for sentence in sentences:
        unique = set(sentence)
        for i in unique:
            for j in unique:
                if i != j:
                    expected[i, j] += 1

    assert np.array_equal(X, expected)
    assert X.dtype == np.int64
    assert np.trace(X) == 0  # no word co-occurs with itself
    assert np.array_equal(X, X.T)  # symmetric on this fixture

    # Cross-family blocks are exactly zero.
    family_indices = {
        name: [fixture["word_to_ix"][w] for w in words]
        for name, words in fixture["families"].items()
    }
    for name_i, rows in family_indices.items():
        for name_j, cols in family_indices.items():
            if name_i != name_j:
                assert X[np.ix_(rows, cols)].sum() == 0


def test_seeded_training_recovers_the_pets_family(fixture):
    """
    The deliberate novelty: after a seeded run the geometry is *learned*, not
    planted. With the dossier's hyperparameters the nearest neighbour of ``cat``
    is ``dog`` and its top-3 are exactly the three other pets.

    Asserted as a rank, never a cosine value, and only for the pets (the plant
    is symmetric across families). The finite-difference checks have already
    proven the gradient, so a failure here localises to the training loop or the
    sampler rather than the math.
    """
    sentences, V = fixture["sentences"], fixture["dims"]["V"]
    ix = fixture["word_to_ix"]

    result = train_sgns(sentences, V, d=8, window=2, k=2, lr=0.05, epochs=200, seed=0)
    ranked = nearest_neighbor(result["W"], ix["cat"], k=3)

    assert ranked[0] == ix["dog"]
    assert set(ranked) == {ix["dog"], ix["fish"], ix["bird"]}


# --- Determinism ---------------------------------------------------------


def test_make_fixture_is_a_literal():
    """
    The corpus is a committed literal: the same 12 three-word sentences, vocab,
    and index map on every call, so every hand-computed anchor stays pinned.
    """
    first, second = make_fixture(), make_fixture()
    assert first["corpus"] == second["corpus"]
    assert first["vocab"] == second["vocab"]
    assert first["word_to_ix"] == second["word_to_ix"]
    assert first["sentences"] == second["sentences"]
    assert len(first["corpus"]) == 12
    assert all(len(sentence) == 3 for sentence in first["corpus"])
    assert first["dims"]["V"] == 16


def test_build_vocabulary_sorts_the_tokens():
    """The vocabulary is the sorted unique tokens, so an index is fixed by the
    corpus alone and not by the order words first appear.
    """
    vocab, word_to_ix = build_vocabulary([["b", "a"], ["c", "a"]])
    assert vocab == ["a", "b", "c"]
    assert word_to_ix == {"a": 0, "b": 1, "c": 2}


def test_train_sgns_is_byte_identical_across_runs(fixture):
    """
    A single seeded generator drives init and sampling, so two runs with the
    same seed produce byte-identical tables and loss history.
    """
    V = fixture["dims"]["V"]
    first = train_sgns(fixture["sentences"], V, epochs=20, seed=0)
    second = train_sgns(fixture["sentences"], V, epochs=20, seed=0)

    assert np.array_equal(first["W"], second["W"])
    assert np.array_equal(first["C"], second["C"])
    assert first["history"] == second["history"]


def test_train_glove_is_byte_identical_across_runs(fixture):
    """GloVe's seeded run is reproducible to the bit as well."""
    V = fixture["dims"]["V"]
    X = cooccurrence_matrix(fixture["sentences"], V)
    first = train_glove(X, V, epochs=20, seed=0)
    second = train_glove(X, V, epochs=20, seed=0)

    assert np.array_equal(first["vectors"], second["vectors"])


# --- Shape and edge contracts --------------------------------------------


def test_training_pairs_order_and_count():
    """
    Pairs come out by centre position then offset ascending, and the count
    matches the window arithmetic. Pinning the order pins the sequence SGD and
    the negative sampler see.
    """
    assert training_pairs([10, 20, 30], window=2) == [
        (10, 20),
        (10, 30),
        (20, 10),
        (20, 30),
        (30, 10),
        (30, 20),
    ]
    # Length-5 stream, window 2: 2 + 3 + 4 + 3 + 2 valid offsets.
    assert len(training_pairs([0, 1, 2, 3, 4], window=2)) == 14
    with pytest.raises(ValueError):
        training_pairs([0, 1, 2], window=0)


def test_fixture_yields_the_promised_pair_count(fixture):
    """Each 3-word sentence yields all 6 ordered pairs; 12 sentences give 72."""
    pairs = [
        p for sentence in fixture["sentences"] for p in training_pairs(sentence, 2)
    ]
    assert len(pairs) == 72


def test_trained_tables_have_the_promised_shapes(fixture):
    """``W`` and ``C`` are ``(V, d)`` rows; GloVe adds two ``(V,)`` biases and a
    summed ``(V, d)`` vector table.
    """
    V = fixture["dims"]["V"]
    sgns = train_sgns(fixture["sentences"], V, d=8, epochs=2)
    assert sgns["W"].shape == (V, 8)
    assert sgns["C"].shape == (V, 8)

    glove = train_glove(cooccurrence_matrix(fixture["sentences"], V), V, d=8, epochs=2)
    assert glove["W"].shape == (V, 8)
    assert glove["Wt"].shape == (V, 8)
    assert glove["b"].shape == (V,)
    assert glove["bt"].shape == (V,)
    assert glove["vectors"].shape == (V, 8)


def test_sigmoid_is_stable_at_the_extremes():
    """The sigmoid is finite and correct at magnitudes that overflow the naive
    ``1 / (1 + exp(-z))`` spelling, and returns a float for scalar input.
    """
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    assert isinstance(sigmoid(0.3), float)

    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)


def test_glove_weight_ramps_then_caps():
    """``f(x)`` grows below ``x_max`` and is capped at 1 at and above it."""
    assert glove_weight(1.0) == pytest.approx((1.0 / 100.0) ** 0.75)
    assert 0.0 < glove_weight(1.0) < 1.0
    assert glove_weight(100.0) == pytest.approx(1.0)
    assert glove_weight(250.0) == 1.0


def test_glove_skips_zero_counts():
    """
    A zero co-occurrence contributes no term -- ``log X`` is never evaluated at
    0 -- so a word that never co-occurs gets exactly zero gradient and the loss
    is the sum over observed entries alone.
    """
    V, d = 3, 2
    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 0.5, size=(V, d))
    Wt = rng.normal(0.0, 0.5, size=(V, d))
    b = rng.normal(0.0, 0.5, size=V)
    bt = rng.normal(0.0, 0.5, size=V)
    X = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int64)  # word 2 absent

    loss = glove_loss(W, Wt, b, bt, X)
    assert np.isfinite(loss)

    expected = sum(
        glove_weight(X[i, j]) * (W[i] @ Wt[j] + b[i] + bt[j] - np.log(X[i, j])) ** 2
        for i, j in [(0, 1), (1, 0)]
    )
    assert loss == pytest.approx(expected)

    grads = glove_gradients(W, Wt, b, bt, X)
    assert np.array_equal(grads["W"][2], np.zeros(d))
    assert np.array_equal(grads["Wt"][2], np.zeros(d))
    assert grads["b"][2] == 0.0
    assert grads["bt"][2] == 0.0


# --- Learning ------------------------------------------------------------


def test_sgns_training_loss_strictly_decreases(fixture):
    """
    Full-batch descent on the fixed objective drives the loss down every epoch.

    Strict monotonicity, not a curve match: a broken backward pass or a wrong
    loss sign would flatten or raise the curve. The total drop clears a floor no
    float noise could produce.
    """
    V = fixture["dims"]["V"]
    history = train_sgns(fixture["sentences"], V, seed=0)["history"]
    steps = [history[i + 1] - history[i] for i in range(len(history) - 1)]

    assert all(delta < 0 for delta in steps), f"largest step delta {max(steps):+.2e}"
    assert history[0] - history[-1] > MIN_LOSS_DROP


def test_glove_training_loss_strictly_decreases(fixture):
    """GloVe's weighted least-squares loss also descends every epoch."""
    V = fixture["dims"]["V"]
    X = cooccurrence_matrix(fixture["sentences"], V)
    history = train_glove(X, V, seed=0)["history"]
    steps = [history[i + 1] - history[i] for i in range(len(history) - 1)]

    assert all(delta < 0 for delta in steps)


# --- Registration --------------------------------------------------------


def test_public_names_are_exported_from_the_package():
    """
    The dossier's public surface is importable from the owning subpackage, so a
    reader imports an installed package rather than a file path.

    Every name lives on the ``word2vec_and_glove`` module. All of them except
    ``make_fixture`` are also re-exported at the ``dlhub.embeddings`` package
    level; ``make_fixture`` is not, because the head topic already owns that bare
    name there (both topics ship their own fixture), so the learned fixture is
    reached as ``dlhub.embeddings.word2vec_and_glove.make_fixture``.
    """
    import dlhub.embeddings as pkg
    from dlhub.embeddings import word2vec_and_glove as module

    surface = [
        "make_fixture",
        "build_vocabulary",
        "training_pairs",
        "unigram_probabilities",
        "sample_negatives",
        "sigmoid",
        "sgns_loss",
        "sgns_gradients",
        "cbow_loss",
        "cbow_gradients",
        "cooccurrence_matrix",
        "glove_loss",
        "glove_gradients",
        "train_sgns",
        "train_glove",
        "nearest_neighbor",
    ]
    for name in surface:
        assert hasattr(module, name), f"{name} missing from the module"

    for name in surface:
        if name == "make_fixture":
            continue
        assert hasattr(pkg, name), f"{name} is not re-exported from dlhub.embeddings"
