"""
Sequence Models (PyTorch)
=========================

PyTorch parity ports of the sequence-model references in
:mod:`dlhub.nn.sequence`. The vanilla RNN here is the same recurrence, rewritten
with ``torch.nn.RNN`` and autograd, and checked against the from-scratch NumPy
module on the topic's shared fixture.

The port itself needs ``torch``, but importing *this package* does not. The
re-exported names resolve lazily (PEP 562): ``torch`` is pulled in only when one
is first accessed -- ``from dlhub.pytorch.sequence import VanillaRNN``, or any
attribute access on the package. That keeps the package importable in an
environment without PyTorch (the docs/style CI job installs no framework), so it
is still discovered and counted, while ``import dlhub.pytorch.sequence.rnn`` and
every real use of the port continue to require ``torch``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

__all__ = [
    "VanillaRNN",
    "bptt_gradients",
    "load_reference_parameters",
]


def __getattr__(name):
    """Resolve a re-exported port name lazily, so importing this package does
    not import ``torch`` (PEP 562). ``torch`` is pulled in here, on first
    access, by importing the port module.
    """
    if name in __all__:
        from dlhub.pytorch.sequence import rnn

        return getattr(rnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
