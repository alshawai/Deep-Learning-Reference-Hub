"""
Word Embeddings (TensorFlow)
============================

TensorFlow/Keras parity port of the learned-embedding reference in
:mod:`dlhub.embeddings.word2vec_and_glove`. Where the NumPy module spells out the
skip-gram negative-sampling objective and its gradient term by term, this port
shows how it is really written in TensorFlow: two ``tf.keras.layers.Embedding``
tables, the loss formed by ``tf.nn.nce_loss`` -- the framework's own
negative-sampling primitive -- and the gradients taken by ``tf.GradientTape``.
It is checked against the NumPy reference on the shared fixture, so a numeric
disagreement is a bug rather than a difference of opinion.

The port needs ``tensorflow``, but importing *this package* does not. The
re-exported names resolve lazily (PEP 562): ``tensorflow`` is pulled in only when
one is first accessed -- ``from dlhub.tensorflow.embeddings import
SkipGramNegativeSampling``, or any attribute access on the package. That keeps
the package importable in an environment without TensorFlow (the docs/style CI
job installs no framework), so it is still discovered and counted, while
``import dlhub.tensorflow.embeddings.word2vec_and_glove`` and every real use of
the port continue to require ``tensorflow``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

# Each public name maps to the submodule that defines it, so ``__getattr__`` can
# import only the one a caller reaches for -- and only then pull in tensorflow.
_WORD2VEC_AND_GLOVE_EXPORTS = frozenset({"SkipGramNegativeSampling"})

__all__ = sorted(_WORD2VEC_AND_GLOVE_EXPORTS)


def __getattr__(name):
    """Resolve a re-exported port name lazily, so importing this package does not
    import ``tensorflow`` (PEP 562). ``tensorflow`` is pulled in here, on first
    access, by importing the submodule that owns the name.
    """
    if name in _WORD2VEC_AND_GLOVE_EXPORTS:
        from dlhub.tensorflow.embeddings import word2vec_and_glove

        return getattr(word2vec_and_glove, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
