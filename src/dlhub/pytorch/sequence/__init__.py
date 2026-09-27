"""
Sequence Models (PyTorch)
=========================

PyTorch parity ports of the sequence-model references in
:mod:`dlhub.nn.sequence`. The vanilla RNN here is the same recurrence, rewritten
with ``torch.nn.RNN`` and autograd, and checked against the from-scratch NumPy
module on the topic's shared fixture. The character-level language model
(:mod:`dlhub.pytorch.sequence.language_model`) is this package's one *canonical*
member -- the language-modeling topic has no NumPy reference, so the module is
checked on its own correctness rather than for parity.

The ports themselves need ``torch``, but importing *this package* does not. The
re-exported names resolve lazily (PEP 562): ``torch`` is pulled in only when one
is first accessed -- ``from dlhub.pytorch.sequence import VanillaRNN``, or any
attribute access on the package. That keeps the package importable in an
environment without PyTorch (the docs/style CI job installs no framework), so it
is still discovered and counted, while ``import dlhub.pytorch.sequence.rnn`` and
every real use of a port continue to require ``torch``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

# Each public name maps to the submodule that defines it, so ``__getattr__`` can
# import only the one a caller actually reaches for -- and only then pull in torch.
_RNN_EXPORTS = frozenset({"VanillaRNN", "bptt_gradients", "load_reference_parameters"})
_LM_EXPORTS = frozenset(
    {
        "CharRNN",
        "build_model",
        "build_vocabulary",
        "encode_word",
        "corpus_cross_entropy",
        "perplexity",
        "train",
        "sample",
    }
)

__all__ = sorted(_RNN_EXPORTS | _LM_EXPORTS)


def __getattr__(name):
    """Resolve a re-exported port name lazily, so importing this package does
    not import ``torch`` (PEP 562). ``torch`` is pulled in here, on first
    access, by importing the submodule that owns the name.
    """
    if name in _RNN_EXPORTS:
        from dlhub.pytorch.sequence import rnn

        return getattr(rnn, name)
    if name in _LM_EXPORTS:
        from dlhub.pytorch.sequence import language_model

        return getattr(language_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
