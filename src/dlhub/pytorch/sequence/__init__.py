"""
Sequence Models (PyTorch)
=========================

PyTorch parity ports of the sequence-model references in
:mod:`dlhub.nn.sequence`. The vanilla RNN here is the same recurrence, rewritten
with ``torch.nn.RNN`` and autograd, and checked against the from-scratch NumPy
module on the topic's shared fixture.

Re-exporting the port here also means importing this subpackage pulls in
``torch``; an environment without PyTorch skips the whole subtree rather than
failing an unrelated import.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.pytorch.sequence.rnn import (
    VanillaRNN,
    bptt_gradients,
    load_reference_parameters,
)

__all__ = [
    "VanillaRNN",
    "bptt_gradients",
    "load_reference_parameters",
]
