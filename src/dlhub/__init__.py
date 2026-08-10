"""
Deep Learning Reference Hub
===========================

From-scratch implementations of core deep learning methods, written to be read.

Every module in this package exposes the mathematics rather than hiding it. The
implementations are teaching artifacts first and working code second, though they
are held to both standards: each is covered by tests that assert against
hand-computed values, closed-form results, or finite-difference gradient checks.

Subpackages
-----------
optimizers
    Gradient descent and its adaptive descendants.
tuning
    Hyperparameter search strategies.
nn
    Network construction and the training loop.
training
    Techniques that wrap training rather than perform it.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

__version__ = "0.1.0"
