"""
Word2Vec SGNS -- TensorFlow/Keras parity port
=============================================

Skip-gram with negative sampling, the objective of
:mod:`dlhub.embeddings.word2vec_and_glove`, rewritten the way it is really
written in TensorFlow: two ``tf.keras.layers.Embedding`` tables for the centre
and context vectors, the loss formed by ``tf.nn.nce_loss`` -- the framework's
own negative-sampling primitive -- and the gradients taken by
``tf.GradientTape``. The method is the topic, so this is a *parity port*: it is
checked against the NumPy reference's loss and gradients on the shared fixture,
and a numeric disagreement is a bug rather than a difference of style.

``tf.nn.nce_loss`` is the reason this topic ports to TensorFlow rather than
PyTorch (which has no equivalent built-in): it is the exact surface a
practitioner reaches for, and the whole lesson here is *how to drive it so it
computes the reference's objective*. That takes reconciling six conventions, and
each is a thing a first ``nce_loss`` gets wrong:

1. **NCE subtracts ``log Q(y)``; negative sampling does not.** ``nce_loss``
   implements noise-contrastive estimation, which corrects each logit by the log
   of the sampler's expected count so the loss estimates the full softmax
   (Gutmann & Hyvarinen, 2010). Mikolov's negative sampling -- the reference --
   drops that correction (Dyer, 2014). The port disables it by passing a
   ``sampled_values`` whose expected counts are all ``1``, so ``log 1 = 0`` and
   the two agree. This single argument is the difference between NCE and SGNS.
2. **``nce_loss`` requires output biases; SGNS has none.** The ``biases``
   argument is per-class and mandatory. The reference scores a bare inner
   product, so the port passes a frozen zero vector -- never a trainable weight.
3. **One negative draw is shared across the whole batch.** ``sampled_values``
   carries a single ``(num_sampled,)`` vector of candidates, used for every row;
   the reference draws ``K`` negatives *per pair*. Parity is therefore asserted
   on a *fixed sampled batch*: the test feeds both implementations the same
   shared draw, which is apples-to-apples by construction.
4. **TensorFlow ships the ``unigram^0.75`` sampler too.**
   ``tf.random.fixed_unigram_candidate_sampler(distortion=0.75,
   unigrams=counts)`` raises each frequency to ``3/4`` and normalises -- exactly
   the reference's
   :func:`~dlhub.embeddings.word2vec_and_glove.unigram_probabilities`. The
   idiomatic training path (:meth:`SkipGramNegativeSampling.train`) uses it; the
   parity path injects the reference's own draw instead, because the two RNGs
   stream differently over the same distribution and only identical negatives
   give bit-exact agreement.
5. **``remove_accidental_hits`` stays ``False``.** When a sampled negative
   happens to equal the true context, the reference still adds its term; so does
   ``nce_loss`` with the default ``remove_accidental_hits=False``. Flipping it to
   ``True`` masks that negative and computes a different loss.
6. **``embedding_lookup`` gradients are sparse.** The tape returns
   ``tf.IndexedSlices`` for both tables -- only the rows a batch touches --
   which the port densifies (summing duplicate indices) into the full ``(V, d)``
   tables the reference's gradients are shaped as.

One more convention is numerical rather than ``nce_loss``'s own: Keras defaults
to ``float32``, and the parity tolerance is far finer than ``float32`` carries,
so every table and tensor here is ``float64``.

References
----------
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
  Distributed Representations of Words and Phrases and their Compositionality.
  NeurIPS. https://arxiv.org/abs/1310.4546 (Skip-gram with negative sampling.)
- Gutmann, M., & Hyvarinen, A. (2010). Noise-contrastive estimation: A new
  estimation principle for unnormalized statistical models. AISTATS.
- Dyer, C. (2014). Notes on Noise Contrastive Estimation and Negative Sampling.
  https://arxiv.org/abs/1410.8251 (SGNS is NCE with the log-Q term dropped.)
- TensorFlow. tf.nn.nce_loss.
  https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss
- TensorFlow. tf.random.fixed_unigram_candidate_sampler.
  https://www.tensorflow.org/api_docs/python/tf/random/fixed_unigram_candidate_sampler

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Why not ``tf.nn.sampled_softmax_loss``.** TensorFlow's other sampled
  primitive forms a *softmax* over the true class and the negatives -- one
  normalised distribution -- not the independent per-negative sigmoids SGNS
  sums. It is the sampled-softmax objective, a different loss with no counterpart
  in the reference, so it is named here but never used for parity.
- **The parity path never trains a curve.** Agreement is checked on the loss and
  the gradients of a fixed batch; :meth:`SkipGramNegativeSampling.train` exists
  only to show the idiomatic optimiser loop, and its loss history is
  demonstrated, never asserted, because training curves drift for reasons
  unrelated to correctness.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

__all__ = ["SkipGramNegativeSampling"]


class SkipGramNegativeSampling(keras.Model):
    """
    Skip-gram with negative sampling, written the TensorFlow way.

    Two ``tf.keras.layers.Embedding`` tables hold the centre and context
    vectors; :meth:`loss_and_gradients` forms the loss with ``tf.nn.nce_loss``
    and differentiates it with ``tf.GradientTape``. The six conventions the
    module docstring enumerates are all applied in :meth:`nce_loss_terms`, which
    is the one place ``nce_loss`` is called.

    The tables are stored in the reference's ``(V, d)`` row convention -- a
    word's vector is a row -- so :meth:`load_reference_tables` is a direct copy
    with no transpose, and the gradients come back in the same shape the
    reference's :func:`~dlhub.embeddings.word2vec_and_glove.sgns_gradients`
    returns.

    Parameters
    ----------
    vocab_size : int
        The vocabulary size ``V`` -- the number of rows in each table.
    embedding_dim : int
        The embedding dimension ``d``.
    dtype : str, optional
        Compute dtype for both tables. Defaults to ``"float64"``, which the
        parity tolerance requires (see the module docstring).

    Attributes
    ----------
    center : keras.layers.Embedding
        The centre (input) table ``W``, shape ``(V, d)``; ``center(w)`` gathers
        centre word ``w``'s vector ``v_w``.
    context : keras.layers.Embedding
        The context (output) table ``C``, shape ``(V, d)``; its ``embeddings``
        variable is passed to ``nce_loss`` as ``weights``.
    output_bias : tf.Tensor
        A frozen ``(V,)`` zero vector. ``nce_loss`` requires per-class biases;
        SGNS has none, so this is a constant and never a trainable weight
        (divergence 2).
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, *, dtype: str = "float64"
    ) -> None:
        super().__init__(dtype=dtype)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center = keras.layers.Embedding(
            vocab_size, embedding_dim, name="center", dtype=dtype
        )
        self.context = keras.layers.Embedding(
            vocab_size, embedding_dim, name="context", dtype=dtype
        )
        # Built here, not on first call, so reference tables can be loaded into a
        # fresh model without a forward pass first.
        self.center.build((None,))
        self.context.build((None,))
        # SGNS has no output bias; nce_loss insists on one. A constant zero keeps
        # it out of trainable_weights, so an optimiser never moves it off zero.
        self.output_bias = tf.zeros([vocab_size], dtype=self.compute_dtype)

    # --- Weight loading ------------------------------------------------

    def load_reference_tables(self, W: np.ndarray, C: np.ndarray) -> None:
        """
        Copy the reference's seeded ``(V, d)`` tables into the two layers.

        No transpose: the port keeps the reference's row convention, so ``W``
        goes straight into :attr:`center` and ``C`` into :attr:`context`. Loading
        the identical seeded tables is what lets the parity comparison start both
        implementations from the same weights.

        Parameters
        ----------
        W : np.ndarray
            The centre table, shape ``(V, d)``.
        C : np.ndarray
            The context table, shape ``(V, d)``.

        Raises
        ------
        ValueError
            If either table's shape is not ``(vocab_size, embedding_dim)``.
        """
        expected = (self.vocab_size, self.embedding_dim)
        for name, table in (("W", W), ("C", C)):
            if table.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; got {table.shape}"
                )
        self.center.set_weights([np.asarray(W, dtype=self.center.compute_dtype)])
        self.context.set_weights([np.asarray(C, dtype=self.context.compute_dtype)])

    @property
    def center_table(self) -> np.ndarray:
        """The centre table ``W`` as a ``(V, d)`` NumPy array."""
        return np.asarray(self.center.embeddings)

    @property
    def context_table(self) -> np.ndarray:
        """The context table ``C`` as a ``(V, d)`` NumPy array."""
        return np.asarray(self.context.embeddings)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Gather centre vectors ``v_w = W[w]`` -- the model's forward pass.

        Parameters
        ----------
        inputs : tf.Tensor
            Centre word indices, shape ``(batch,)``.

        Returns
        -------
        tf.Tensor
            The centre vectors, shape ``(batch, d)``.
        """
        return self.center(inputs)

    # --- The one call to tf.nn.nce_loss --------------------------------

    def nce_loss_terms(
        self, centers: tf.Tensor, contexts: tf.Tensor, negatives: tf.Tensor
    ) -> tf.Tensor:
        """
        Per-pair SGNS loss via ``tf.nn.nce_loss``, wired to match the reference.

        This is the whole port in one method; every divergence in the module
        docstring is an argument here. The expected counts of ``1`` zero out
        NCE's ``log Q`` correction (divergence 1), :attr:`output_bias` is the
        frozen zero (divergence 2), ``negatives`` is the single shared draw fed
        through ``sampled_values`` (divergences 3 and 4), and
        ``remove_accidental_hits=False`` keeps a negative that coincides with the
        context (divergence 5). With those, ``nce_loss`` returns exactly
        ``-log sigma(v_w . u_c) - sum_k log sigma(-v_w . u_{n_k})`` per pair.

        Parameters
        ----------
        centers : tf.Tensor
            Centre word indices, shape ``(batch,)``.
        contexts : tf.Tensor
            True context word indices, shape ``(batch,)``.
        negatives : tf.Tensor
            The ``K`` negative indices shared across the batch, shape ``(K,)``.

        Returns
        -------
        tf.Tensor
            The per-pair losses, shape ``(batch,)``.
        """
        labels = tf.reshape(tf.cast(contexts, tf.int64), (-1, 1))
        negatives = tf.cast(negatives, tf.int64)
        dtype = self.compute_dtype
        # Expected counts of 1 => log(1) = 0 => no log-Q correction => SGNS, not
        # NCE. This triple is the sampled_values contract: candidates, then the
        # true and sampled expected counts the correction would subtract.
        sampled_values = (
            negatives,
            tf.ones((tf.shape(labels)[0], 1), dtype=dtype),
            tf.ones((tf.shape(negatives)[0],), dtype=dtype),
        )
        return tf.nn.nce_loss(
            weights=self.context.embeddings,  # the (V, d) context table C
            biases=self.output_bias,  # frozen zeros (divergence 2)
            labels=labels,  # (batch, 1) true context
            inputs=self.center(tf.cast(centers, tf.int64)),  # (batch, d) v_w
            num_sampled=tf.shape(negatives)[0],
            num_classes=self.vocab_size,
            num_true=1,
            sampled_values=sampled_values,  # inject the reference's draw
            remove_accidental_hits=False,  # keep accidental hits (divergence 5)
        )

    # --- Loss and gradients --------------------------------------------

    @tf.function
    def loss_and_gradients(
        self, centers: tf.Tensor, contexts: tf.Tensor, negatives: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One graph-compiled forward-and-backward pass over a batch of pairs.

        Sums :meth:`nce_loss_terms` into the total loss the reference forms
        (sum over pairs) and differentiates it with ``tf.GradientTape``. The
        tape returns ``tf.IndexedSlices`` for both tables because the lookup is
        sparse; :func:`tf.convert_to_tensor` densifies them, summing the
        contributions of any row a batch touches more than once (divergence 6).

        Parameters
        ----------
        centers : tf.Tensor
            Centre word indices, shape ``(batch,)``.
        contexts : tf.Tensor
            True context word indices, shape ``(batch,)``.
        negatives : tf.Tensor
            The shared negative indices, shape ``(K,)``.

        Returns
        -------
        loss : tf.Tensor
            The scalar total loss, summed over the batch.
        d_center : tf.Tensor
            Gradient w.r.t. the centre table, dense ``(V, d)``.
        d_context : tf.Tensor
            Gradient w.r.t. the context table, dense ``(V, d)``.
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(self.nce_loss_terms(centers, contexts, negatives))
        tables = [self.center.embeddings, self.context.embeddings]
        grads = tape.gradient(loss, tables)
        return loss, tf.convert_to_tensor(grads[0]), tf.convert_to_tensor(grads[1])

    def reference_loss_and_gradients(
        self, centers: np.ndarray, contexts: np.ndarray, negatives: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Loss and gradients in the reference's names and shapes, as NumPy.

        The parity-facing entry point: runs :meth:`loss_and_gradients` on the
        batch and returns the result keyed exactly as the reference's
        :func:`~dlhub.embeddings.word2vec_and_glove.sgns_gradients` keys it, so
        the parity test compares like with like.

        Parameters
        ----------
        centers : np.ndarray
            Centre word indices, shape ``(batch,)``.
        contexts : np.ndarray
            True context word indices, shape ``(batch,)``.
        negatives : np.ndarray
            The shared negative indices, shape ``(K,)``.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"loss": float, "W": dW, "C": dC}`` -- the scalar loss and the two
            dense ``(V, d)`` gradient tables.
        """
        loss, d_center, d_context = self.loss_and_gradients(
            tf.convert_to_tensor(np.asarray(centers, dtype=np.int64)),
            tf.convert_to_tensor(np.asarray(contexts, dtype=np.int64)),
            tf.convert_to_tensor(np.asarray(negatives, dtype=np.int64)),
        )
        return {
            "loss": float(loss),
            "W": np.asarray(d_center),
            "C": np.asarray(d_context),
        }

    # --- The idiomatic training path (demonstrated, not asserted) ------

    @staticmethod
    def unigram_sampler(
        counts: np.ndarray, num_sampled: int, *, distortion: float = 0.75, seed=None
    ) -> tf.Tensor:
        """
        TensorFlow's own smoothed-unigram negative sampler.

        Wraps ``tf.random.fixed_unigram_candidate_sampler``, which raises each
        class frequency to ``distortion`` and normalises -- exactly the
        reference's
        :func:`~dlhub.embeddings.word2vec_and_glove.unigram_probabilities` with
        ``exponent = 3/4``. The parity path does not use this (it injects the
        reference's own draw for bit-exact agreement, divergence 4); :meth:`train`
        does, because this is the sampler a practitioner reaches for.

        Parameters
        ----------
        counts : np.ndarray
            The ``(V,)`` raw corpus frequencies ``f(w)``.
        num_sampled : int
            How many negatives to draw.
        distortion : float, optional
            The smoothing exponent, ``0.75`` for Word2Vec.
        seed : int or None, optional
            Seeds the draw for a reproducible demonstration.

        Returns
        -------
        tf.Tensor
            The ``(num_sampled,)`` sampled negative indices.
        """
        sampled = tf.random.fixed_unigram_candidate_sampler(
            true_classes=tf.zeros((1, 1), dtype=tf.int64),
            num_true=1,
            num_sampled=num_sampled,
            unique=False,
            range_max=len(counts),
            distortion=distortion,
            unigrams=[float(c) for c in counts],
            seed=seed,
        )
        return sampled.sampled_candidates

    def train(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        counts: np.ndarray,
        *,
        num_negatives: int = 2,
        lr: float = 0.05,
        epochs: int = 200,
        seed: int = 0,
    ) -> list[float]:
        """
        Train SGNS the idiomatic way: TF's sampler + a Keras optimiser.

        A ``keras.optimizers.SGD`` steps on the ``nce_loss`` gradients, and a
        fresh batch of negatives is drawn from :meth:`unigram_sampler` each epoch
        (classic word2vec resamples; the reference instead fixes the draw to make
        its loss a clean descent curve). This is *not* a parity path: the loss
        history is returned to be shown, never asserted, because training curves
        drift for reasons unrelated to correctness.

        Parameters
        ----------
        centers : np.ndarray
            Centre word indices for every training pair, shape ``(num_pairs,)``.
        contexts : np.ndarray
            The matching context indices, shape ``(num_pairs,)``.
        counts : np.ndarray
            The ``(V,)`` raw corpus frequencies feeding the sampler.
        num_negatives : int, optional
            Negatives per epoch (shared across the batch, as ``nce_loss`` draws).
        lr : float, optional
            The SGD step size.
        epochs : int, optional
            Number of full-batch passes.
        seed : int, optional
            Base seed; epoch ``e`` draws negatives with ``seed + e``.

        Returns
        -------
        list[float]
            The per-epoch total loss.
        """
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        centers_t = tf.convert_to_tensor(np.asarray(centers, dtype=np.int64))
        contexts_t = tf.convert_to_tensor(np.asarray(contexts, dtype=np.int64))
        tables = [self.center.embeddings, self.context.embeddings]

        history: list[float] = []
        for epoch in range(epochs):
            negatives = self.unigram_sampler(counts, num_negatives, seed=seed + epoch)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(
                    self.nce_loss_terms(centers_t, contexts_t, negatives)
                )
            optimizer.apply_gradients(zip(tape.gradient(loss, tables), tables))
            history.append(float(loss))
        return history


def main() -> None:
    """Report SGNS loss/gradient parity, then train idiomatically and show
    the pets family's recovered neighbours.
    """
    from dlhub.embeddings.word2vec_and_glove import (
        make_fixture,
        nearest_neighbor,
        sgns_gradients,
        sgns_loss,
        train_sgns,
        training_pairs,
        unigram_probabilities,
    )

    tf.keras.backend.set_floatx("float64")
    fixture = make_fixture()
    sentences, vocab, word_to_ix = (
        fixture["sentences"],
        fixture["vocab"],
        fixture["word_to_ix"],
    )
    V = fixture["dims"]["V"]
    d, window, k, seed = 8, 2, 2, 0

    # The seeded tables the reference's train loop starts from: epochs=0 returns
    # the initial W and C, so the port loads exactly those (public, no privates).
    initial = train_sgns(sentences, V, d=d, window=window, k=k, epochs=0, seed=seed)

    pairs = [p for s in sentences for p in training_pairs(s, window)]
    centers = np.array([c for c, _ in pairs], dtype=np.int64)
    contexts = np.array([x for _, x in pairs], dtype=np.int64)
    probabilities = unigram_probabilities(sentences, V)
    negatives = np.random.default_rng(123).choice(V, size=k, p=probabilities)

    ref_loss = sum(
        sgns_loss(initial["W"], initial["C"], c, x, negatives)
        for c, x in zip(centers, contexts)
    )
    ref = {"W": np.zeros((V, d)), "C": np.zeros((V, d))}
    for c, x in zip(centers, contexts):
        grads = sgns_gradients(initial["W"], initial["C"], c, x, negatives)
        ref["W"] += grads["W"]
        ref["C"] += grads["C"]

    model = SkipGramNegativeSampling(V, d)
    model.load_reference_tables(initial["W"], initial["C"])
    port = model.reference_loss_and_gradients(centers, contexts, negatives)

    def rel(a, b):
        denom = np.linalg.norm(a) + np.linalg.norm(b)
        return 0.0 if denom == 0 else np.linalg.norm(a - b) / denom

    print("SGNS parity on the fixed sampled batch (nce_loss vs the reference)")
    print(f"  loss: reference {ref_loss:.10f} / port {port['loss']:.10f}")
    print(f"  dW:   relative error {rel(ref['W'], port['W']):.2e}")
    print(f"  dC:   relative error {rel(ref['C'], port['C']):.2e}")

    # The idiomatic path: TF's own unigram^0.75 sampler + SGD. Not asserted.
    counts = np.bincount(np.concatenate(sentences), minlength=V)
    trained = SkipGramNegativeSampling(V, d)
    trained.load_reference_tables(initial["W"], initial["C"])
    history = trained.train(centers, contexts, counts, num_negatives=k, epochs=200)
    print(f"\nidiomatic training: loss {history[0]:.3f} -> {history[-1]:.3f}")
    cat = word_to_ix["cat"]
    neighbours = nearest_neighbor(trained.center_table, cat, k=3)
    print(f"nearest to 'cat': {[vocab[j] for j in neighbours]}")


if __name__ == "__main__":
    main()
