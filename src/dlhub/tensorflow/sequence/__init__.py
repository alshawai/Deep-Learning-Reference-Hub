"""
Sequence Models (TensorFlow)
============================

TensorFlow/Keras parity ports of the sequence-model references in
:mod:`dlhub.nn.sequence`. The gated cells here are the same recurrences,
rewritten with ``tf.keras.layers.LSTM`` and ``tf.keras.layers.GRU`` and
differentiated by ``tf.GradientTape``, and checked against the from-scratch
NumPy modules on the topic's shared fixture.

The port itself needs ``tensorflow``, but importing *this package* does not. The
re-exported names resolve lazily (PEP 562): ``tensorflow`` is pulled in only
when one is first accessed -- ``from dlhub.tensorflow.sequence import
LSTMSequenceModel``, or any attribute access on the package. That keeps the
package importable in an environment without TensorFlow (the docs/style CI job
installs no framework), so it is still discovered and counted, while
``import dlhub.tensorflow.sequence.lstm_gru`` and every real use of the port
continue to require ``tensorflow``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

__all__ = [
    "GRUSequenceModel",
    "GatedSequenceModel",
    "LSTMSequenceModel",
]


def __getattr__(name):
    """Resolve a re-exported port name lazily, so importing this package does
    not import ``tensorflow`` (PEP 562). ``tensorflow`` is pulled in here, on
    first access, by importing the port module.
    """
    if name in __all__:
        from dlhub.tensorflow.sequence import lstm_gru

        return getattr(lstm_gru, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
