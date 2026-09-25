"""
Sequence Models
===============

Recurrent architectures for sequential data, written to be read alongside the
documents that derive them. The vanilla RNN here is the from-scratch reference
that the framework parity port is checked against; its notation follows Andrew
Ng's Course 5 so the gated cells (LSTM, GRU) extend the same symbols.

Each module carries its own ``make_fixture``. The name re-exported here is the
vanilla RNN's, because a package-level name can only mean one thing; import the
gated ones from their modules, ``from dlhub.nn.sequence.lstm import
make_fixture``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.nn.sequence._common import sigmoid, split_gate_matrix
from dlhub.nn.sequence.gru import (
    gru_backward,
    gru_cell_backward,
    gru_cell_forward,
    gru_forward,
)
from dlhub.nn.sequence.lstm import (
    lstm_backward,
    lstm_cell_backward,
    lstm_cell_forward,
    lstm_forward,
)
from dlhub.nn.sequence.min_gated import (
    g,
    identity_candidate,
    log_g,
    log_space_scan,
    logcumsumexp,
    min_gru_coefficients,
    min_gru_forward_log,
    min_gru_forward_parallel,
    min_gru_forward_sequential,
    min_gru_log_coefficients,
    min_lstm_coefficients,
    min_lstm_forward_log,
    min_lstm_forward_parallel,
    min_lstm_forward_sequential,
    min_lstm_log_coefficients,
    parallel_scan,
    project_over_time,
    sequential_scan,
    softplus,
)
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
    "g",
    "gru_backward",
    "gru_cell_backward",
    "gru_cell_forward",
    "gru_forward",
    "identity_candidate",
    "log_g",
    "log_space_scan",
    "logcumsumexp",
    "lstm_backward",
    "lstm_cell_backward",
    "lstm_cell_forward",
    "lstm_forward",
    "make_fixture",
    "min_gru_coefficients",
    "min_gru_forward_log",
    "min_gru_forward_parallel",
    "min_gru_forward_sequential",
    "min_gru_log_coefficients",
    "min_lstm_coefficients",
    "min_lstm_forward_log",
    "min_lstm_forward_parallel",
    "min_lstm_forward_sequential",
    "min_lstm_log_coefficients",
    "parallel_scan",
    "project_over_time",
    "rnn_backward",
    "rnn_cell_backward",
    "rnn_cell_forward",
    "rnn_forward",
    "sequential_scan",
    "sigmoid",
    "softmax",
    "softplus",
    "split_gate_matrix",
    "update_parameters",
]
