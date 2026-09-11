"""
Sequence Models
===============

Recurrent architectures for sequential data, written to be read alongside the
documents that derive them. The vanilla RNN here is the from-scratch reference
that the framework parity port is checked against; its notation follows Andrew
Ng's Course 5 so the gated cells (LSTM, GRU) can extend the same symbols.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.nn.sequence.rnn import (
    bidirectional_rnn_forward,
    clip_gradients,
    compute_loss,
    deep_rnn_forward,
    make_fixture,
    rnn_backward,
    rnn_cell_backward,
    rnn_cell_forward,
    rnn_forward,
    softmax,
    update_parameters,
)

__all__ = [
    "bidirectional_rnn_forward",
    "clip_gradients",
    "compute_loss",
    "deep_rnn_forward",
    "make_fixture",
    "rnn_backward",
    "rnn_cell_backward",
    "rnn_cell_forward",
    "rnn_forward",
    "softmax",
    "update_parameters",
]
