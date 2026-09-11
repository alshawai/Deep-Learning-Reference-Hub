"""
PyTorch Ports
=============

Framework ports of the hub's from-scratch references, written the way a
practitioner writes them in PyTorch. Each port is a *parity port*: it is checked
against the NumPy reference on the topic's shared fixture, so a numeric
disagreement means a bug rather than a difference of opinion.

The ports depend on ``torch``, which is an optional dependency (``pip install -e
'.[frameworks]'``, or the CPU wheel the parity CI job installs). Importing this
subpackage therefore requires PyTorch to be installed.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""
