"""
PyTorch Ports
=============

Framework ports of the hub's from-scratch references, written the way a
practitioner writes them in PyTorch. Each port is a *parity port*: it is checked
against the NumPy reference on the topic's shared fixture, so a numeric
disagreement means a bug rather than a difference of opinion.

The ports depend on ``torch``, which is an optional dependency (``pip install -e
'.[frameworks]'``, or the CPU wheel the parity CI job installs). Importing the
port *packages* themselves is torch-free -- their re-exports resolve lazily --
so they are discovered in an environment without PyTorch. Importing a port
*module* (e.g. ``dlhub.pytorch.sequence.rnn``) or using a port requires
``torch``.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""
