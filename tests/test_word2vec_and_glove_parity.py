"""
Word2Vec SGNS -- TensorFlow parity tests
========================================

The hub ships skip-gram with negative sampling from scratch in NumPy and one
TensorFlow port built on ``tf.keras.layers.Embedding``, ``tf.nn.nce_loss``, and
``tf.GradientTape``. This module is the proof they agree. The method is the
topic, so a numeric disagreement is a bug in one of them, not a difference of
style; that is what makes this a parity port rather than an idiom track.

On the dossier's shared fixture it loads the reference's seeded tables into the
port, feeds both the same *fixed sampled batch* -- every training pair and one
shared negative draw, because ``nce_loss`` shares its negatives across the batch
-- and checks the contract: the summed loss matches to ``1e-8``, and both
gradient tables match the reference's analytic ``sgns_gradients`` to a relative
error below ``1e-6``. A flipped-sign gradient is rejected, so the metric
discriminates rather than passing everything, and a few plain-SGD steps are run
in lockstep, because a gradient wrong only in some rows could match on step one
and still drift apart.

The suite also pins the conventions the port reconciles, because each is a fact
a practitioner needs: NCE's ``log Q`` correction is what the unit expected counts
disable; the output bias is a frozen zero; a negative equal to the context is
kept, not masked; the embedding gradients arrive as ``tf.IndexedSlices``; and
``tf.random.fixed_unigram_candidate_sampler(distortion=0.75)`` is TensorFlow's
own ``unigram^0.75``.

Parity is checked on fixture loss and gradient values, never on a training
curve, which drifts for reasons unrelated to correctness.

This test needs TensorFlow. In the framework-free hub CI job it is skipped at
collection; the cross-framework parity job installs TensorFlow and runs it for
real, where a skip would read as the failure it is.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

# Skip the whole module when TensorFlow is absent so the framework-free test job
# can collect it. The parity CI job installs TensorFlow, so there every
# assertion below runs for real -- a skip there would read as the failure it is.
tf = pytest.importorskip("tensorflow")

from dlhub.embeddings.word2vec_and_glove import (  # noqa: E402  (after the gate)
    make_fixture,
    sample_negatives,
    sgns_gradients,
    sgns_loss,
    train_sgns,
    training_pairs,
    unigram_probabilities,
)
from dlhub.tensorflow.embeddings.word2vec_and_glove import (  # noqa: E402
    SkipGramNegativeSampling,
)

# The fixture's declared training knobs (dossier Fixture / Semantics).
D = 8
WINDOW = 2
K = 2
SEED = 0
# A dedicated seed for the one shared negative draw the fixed batch uses. It is
# distinct from the training seed so the batch does not accidentally reuse the
# init stream, and fixed so the batch is identical every run.
NEG_SEED = 123

# Both implementations are float64 and compute the same formula, so agreement is
# to machine epsilon. The loss (magnitude ~150) matches to ~3e-14 and the
# gradients to ~1e-16; the thresholds sit orders above that and orders below any
# real bug. The gradient tolerance is the dossier's stated 1e-6 relative error.
LOSS_ATOL = 1e-8
GRAD_RELATIVE_TOLERANCE = 1e-6


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric relative error ``||a - b|| / (||a|| + ||b||)``, the hub metric."""
    denominator = np.linalg.norm(a) + np.linalg.norm(b)
    return 0.0 if denominator == 0 else float(np.linalg.norm(a - b) / denominator)


@pytest.fixture(scope="module")
def parity():
    """
    Load the reference's seeded tables into the port and run both on one batch.

    The batch is the fixed sampled batch the whole suite shares: every training
    pair in the fixture, and a single shared negative draw, because ``nce_loss``
    uses one set of negatives for the whole batch (divergence 3). The reference's
    loss and gradients are summed over the batch with those same negatives, so
    the comparison is apples-to-apples.

    The initial tables come from ``train_sgns(..., epochs=0)`` -- the public way
    to read the exact ``(W, C)`` the reference's training loop starts from,
    without reaching into a private initialiser.
    """
    fixture = make_fixture()
    sentences = fixture["sentences"]
    vocab_size = fixture["dims"]["V"]

    initial = train_sgns(
        sentences, vocab_size, d=D, window=WINDOW, k=K, epochs=0, seed=SEED
    )
    W0, C0 = initial["W"], initial["C"]

    pairs = [pair for s in sentences for pair in training_pairs(s, WINDOW)]
    centers = np.array([c for c, _ in pairs], dtype=np.int64)
    contexts = np.array([x for _, x in pairs], dtype=np.int64)

    probabilities = unigram_probabilities(sentences, vocab_size)
    negatives = np.asarray(
        sample_negatives(np.random.default_rng(NEG_SEED), probabilities, K),
        dtype=np.int64,
    )

    ref_loss = sum(
        sgns_loss(W0, C0, c, x, negatives) for c, x in zip(centers, contexts)
    )
    ref_dW = np.zeros((vocab_size, D))
    ref_dC = np.zeros((vocab_size, D))
    for c, x in zip(centers, contexts):
        grads = sgns_gradients(W0, C0, c, x, negatives)
        ref_dW += grads["W"]
        ref_dC += grads["C"]

    model = SkipGramNegativeSampling(vocab_size, D)
    model.load_reference_tables(W0, C0)
    port = model.reference_loss_and_gradients(centers, contexts, negatives)

    return {
        "sentences": sentences,
        "V": vocab_size,
        "counts": np.bincount(np.concatenate(sentences), minlength=vocab_size),
        "W0": W0,
        "C0": C0,
        "centers": centers,
        "contexts": contexts,
        "negatives": negatives,
        "ref_loss": ref_loss,
        "ref_dW": ref_dW,
        "ref_dC": ref_dC,
        "model": model,
        "port": port,
    }


# --- The parity contract (loss and gradients on the fixed batch) ---------


def test_the_loss_matches_the_reference(parity):
    """
    ``nce_loss`` summed over the batch forms the reference's total SGNS loss.

    Matching the loss is the precondition for matching gradients: the tape
    differentiates whatever scalar it is handed, so a differently scaled or
    log-Q-corrected loss would give proportionally wrong gradients that no
    weight mapping could repair.
    """
    assert parity["port"]["loss"] == pytest.approx(parity["ref_loss"], abs=LOSS_ATOL)


def test_the_center_table_gradient_matches_the_reference(parity):
    """GradientTape's gradient on ``W`` agrees with analytic ``sgns_gradients``."""
    error = relative_error(parity["port"]["W"], parity["ref_dW"])
    assert error < GRAD_RELATIVE_TOLERANCE, f"dW: relative error {error:.2e}"


def test_the_context_table_gradient_matches_the_reference(parity):
    """GradientTape's gradient on ``C`` agrees with analytic ``sgns_gradients``."""
    error = relative_error(parity["port"]["C"], parity["ref_dC"])
    assert error < GRAD_RELATIVE_TOLERANCE, f"dC: relative error {error:.2e}"


def test_a_flipped_sign_gradient_would_be_rejected(parity):
    """
    The parity metric discriminates rather than passing everything.

    Negating the centre-table gradient collapses its agreement with the
    reference to order one, confirming a real disagreement could not slip under
    the tolerance above -- the point of a gradient check is that it fails for a
    wrong gradient.
    """
    error = relative_error(-parity["port"]["W"], parity["ref_dW"])
    assert error > 0.9


def test_a_few_sgd_steps_stay_in_lockstep(parity):
    """
    Plain SGD on the port tracks the reference step for step.

    Single-step gradient parity is necessary but not sufficient: a gradient
    wrong only in untouched rows reads as zero there and matches on step one,
    then diverges once a later step touches them. Running the reference's own
    update (``theta -= lr * grad``, no optimiser state) on both for a few steps,
    with the same shared negatives each step, catches that. A fresh port is used
    so the shared module fixture's tables stay pristine.
    """
    lr, steps = 0.05, 3
    centers, contexts = parity["centers"], parity["contexts"]
    negatives = parity["negatives"]

    W, C = parity["W0"].copy(), parity["C0"].copy()
    model = SkipGramNegativeSampling(parity["V"], D)
    model.load_reference_tables(parity["W0"], parity["C0"])

    for _ in range(steps):
        dW = np.zeros_like(W)
        dC = np.zeros_like(C)
        for c, x in zip(centers, contexts):
            grads = sgns_gradients(W, C, c, x, negatives)
            dW += grads["W"]
            dC += grads["C"]
        W -= lr * dW
        C -= lr * dC

        _, port_dW, port_dC = model.loss_and_gradients(
            tf.constant(centers), tf.constant(contexts), tf.constant(negatives)
        )
        model.center.embeddings.assign_sub(lr * port_dW)
        model.context.embeddings.assign_sub(lr * port_dC)

    np.testing.assert_allclose(model.center_table, W, atol=1e-10)
    np.testing.assert_allclose(model.context_table, C, atol=1e-10)


# --- The documented convention divergences -------------------------------


def test_the_log_q_correction_is_what_the_unit_counts_disable(parity):
    """
    SGNS is NCE with the ``log Q`` correction dropped (divergence 1).

    ``nce_loss`` subtracts ``log(expected_count)`` from every logit. The port
    passes expected counts of ``1`` so ``log 1 = 0`` and the loss is the plain
    SGNS loss (the equality re-checked here). Feeding the same batch and
    negatives but a non-unit expected count -- here ``K / V``, the value a
    uniform sampler would carry -- turns the correction back on and moves the
    loss well off the reference, which is exactly the NCE/SGNS distinction.
    """
    model = parity["model"]
    centers, contexts, negatives, V = (
        parity["centers"],
        parity["contexts"],
        parity["negatives"],
        parity["V"],
    )
    labels = tf.reshape(tf.constant(contexts, dtype=tf.int64), (-1, 1))
    q = tf.cast(K / V, tf.float64)
    corrected = float(
        tf.reduce_sum(
            tf.nn.nce_loss(
                weights=model.context.embeddings,
                biases=model.output_bias,
                labels=labels,
                inputs=model.center(tf.constant(centers, dtype=tf.int64)),
                num_sampled=K,
                num_classes=V,
                num_true=1,
                sampled_values=(
                    tf.constant(negatives, dtype=tf.int64),
                    tf.ones((len(centers), 1), tf.float64) * q,
                    tf.ones((K,), tf.float64) * q,
                ),
                remove_accidental_hits=False,
            )
        )
    )
    assert parity["port"]["loss"] == pytest.approx(parity["ref_loss"], abs=LOSS_ATOL)
    assert abs(corrected - parity["ref_loss"]) > 1.0


def test_the_output_bias_is_a_frozen_zero(parity):
    """
    SGNS has no output bias; ``nce_loss`` requires one (divergence 2).

    The port supplies a constant zero vector, so the loss scores a bare inner
    product and the optimiser has only the two tables to move -- the bias can
    never drift off zero because it is not a trainable weight at all.
    """
    model = parity["model"]
    np.testing.assert_array_equal(np.asarray(model.output_bias), np.zeros(parity["V"]))
    assert len(model.trainable_variables) == 2
    assert {tuple(v.shape) for v in model.trainable_variables} == {(parity["V"], D)}


def test_a_negative_equal_to_the_context_is_kept_not_masked(parity):
    """
    ``remove_accidental_hits`` stays ``False`` (divergence 5).

    When a sampled negative coincides with the true context, the reference adds
    its term anyway; the port, with the Keras default ``False``, agrees. Setting
    it ``True`` masks that negative -- a different loss -- which is why the port
    pins the flag rather than leaving it to a reader's assumption.
    """
    model = parity["model"]
    center, context = 0, 1
    negatives = np.array([context, 5], dtype=np.int64)  # context planted as a negative

    reference = sgns_loss(parity["W0"], parity["C0"], center, context, negatives)
    kept = model.reference_loss_and_gradients(
        np.array([center]), np.array([context]), negatives
    )["loss"]
    assert kept == pytest.approx(reference, abs=LOSS_ATOL)

    labels = tf.constant([[context]], dtype=tf.int64)
    masked = float(
        tf.reduce_sum(
            tf.nn.nce_loss(
                weights=model.context.embeddings,
                biases=model.output_bias,
                labels=labels,
                inputs=model.center(tf.constant([center], dtype=tf.int64)),
                num_sampled=2,
                num_classes=parity["V"],
                num_true=1,
                sampled_values=(
                    tf.constant(negatives, dtype=tf.int64),
                    tf.ones((1, 1), tf.float64),
                    tf.ones((2,), tf.float64),
                ),
                remove_accidental_hits=True,
            )
        )
    )
    assert abs(masked - reference) > 0.1


def test_the_embedding_gradients_arrive_as_indexedslices(parity):
    """
    The tape returns sparse ``tf.IndexedSlices`` for both tables (divergence 6).

    Only the rows a batch touches get a gradient, so autograd hands back slices,
    not a dense ``(V, d)`` array. The port densifies them (summing duplicate
    indices) to match the reference's dense tables; this pins the shape the port
    has to convert from.
    """
    model = parity["model"]
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(
            model.nce_loss_terms(
                tf.constant(parity["centers"], dtype=tf.int64),
                tf.constant(parity["contexts"], dtype=tf.int64),
                tf.constant(parity["negatives"], dtype=tf.int64),
            )
        )
    raw = tape.gradient(loss, [model.center.embeddings, model.context.embeddings])
    assert all(isinstance(gradient, tf.IndexedSlices) for gradient in raw)


def test_the_builtin_sampler_is_the_reference_unigram_distribution(parity):
    """
    ``fixed_unigram_candidate_sampler(distortion=0.75)`` is ``unigram^0.75``.

    TensorFlow ships the reference's smoothed-unigram sampler (divergence 4).
    Drawing many samples from it (with replacement, fixed seed) recovers the
    reference's ``unigram_probabilities``. The tolerance is coarse by design: it
    is a finite-sample frequency, not a closed form -- with 200k draws the
    measured gap is ~1.1e-3, so ``5e-3`` sits above the sampling noise and well
    below any gap a different exponent would open.
    """
    reference_probabilities = unigram_probabilities(parity["sentences"], parity["V"])
    draws = np.asarray(
        SkipGramNegativeSampling.unigram_sampler(parity["counts"], 200_000, seed=7)
    )
    empirical = np.bincount(draws, minlength=parity["V"]) / draws.size
    np.testing.assert_allclose(empirical, reference_probabilities, atol=5e-3)


def test_the_tables_are_float64(parity):
    """
    The port is ``float64`` (the numerical convention, not ``nce_loss``'s).

    Keras defaults to ``float32``, whose ~1e-7 epsilon would swamp the ``1e-6``
    gradient tolerance. Pinning the dtype guards the parity thresholds against a
    future that quietly builds the tables in single precision.
    """
    model = parity["model"]
    assert model.center.embeddings.dtype == tf.float64
    assert model.context.embeddings.dtype == tf.float64


def test_graph_mode_changes_no_number(parity):
    """
    ``loss_and_gradients`` is a ``tf.function``, and that is a free choice.

    Running the undecorated Python body against the traced one shows graph
    compilation -- where a TensorFlow step belongs -- costs no accuracy here, so
    the decorator cannot quietly start changing numbers.
    """
    model = parity["model"]
    args = (
        tf.constant(parity["centers"], dtype=tf.int64),
        tf.constant(parity["contexts"], dtype=tf.int64),
        tf.constant(parity["negatives"], dtype=tf.int64),
    )
    graph = model.loss_and_gradients(*args)
    eager = model.loss_and_gradients.python_function(*args)

    assert float(graph[0]) == pytest.approx(float(eager[0]), abs=1e-12)
    np.testing.assert_allclose(np.asarray(graph[1]), np.asarray(eager[1]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(graph[2]), np.asarray(eager[2]), atol=1e-12)
