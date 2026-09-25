"""
TensorFlow Ports
================

Framework ports of the hub's from-scratch references, written the way a
practitioner writes them with TensorFlow and the Keras API it bundles. Each port
is a *parity port*: it is checked against the NumPy reference on the topic's
shared fixture, so a numeric disagreement means a bug rather than a difference
of opinion.

The ports depend on ``tensorflow``, which is an optional dependency (``pip
install -e '.[frameworks]'``, or the CPU wheel the parity CI job installs).
Importing the port *packages* themselves is TensorFlow-free -- their re-exports
resolve lazily -- so they are discovered in an environment without TensorFlow.
Importing a port *module* (e.g. ``dlhub.tensorflow.sequence.lstm_gru``) or using
a port requires ``tensorflow``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""
