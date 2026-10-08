"""
Word Embeddings (PyTorch)
=========================

PyTorch idiom port of the static word-embedding reference in
:mod:`dlhub.embeddings.word_embeddings`. Where the NumPy module spells the lookup
identity out as a matrix product ``e_w = E o_w`` to make the mathematics visible,
this port shows the shape a practitioner actually ships: a ``torch.nn.Embedding``
whose weight *is* the reference matrix transposed to PyTorch's ``(V, d)`` row
convention, and a gather by index in place of the one-hot product.

The port needs ``torch``, but importing *this package* does not. The re-exported
names resolve lazily (PEP 562): ``torch`` is pulled in only when one is first
accessed -- ``from dlhub.pytorch.embeddings import lookup``, or any attribute
access on the package. That keeps the package importable in an environment
without PyTorch (the docs/style CI job installs no framework), so it is still
discovered and counted, while ``import dlhub.pytorch.embeddings.word_embeddings``
and every real use of the port continue to require ``torch``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

# Each public name maps to the submodule that defines it, so ``__getattr__`` can
# import only the one a caller reaches for -- and only then pull in torch.
_WORD_EMBEDDINGS_EXPORTS = frozenset(
    {"embedding_from_reference", "lookup", "tied_readout"}
)

__all__ = sorted(_WORD_EMBEDDINGS_EXPORTS)


def __getattr__(name):
    """Resolve a re-exported port name lazily, so importing this package does not
    import ``torch`` (PEP 562). ``torch`` is pulled in here, on first access, by
    importing the submodule that owns the name.
    """
    if name in _WORD_EMBEDDINGS_EXPORTS:
        from dlhub.pytorch.embeddings import word_embeddings

        return getattr(word_embeddings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
