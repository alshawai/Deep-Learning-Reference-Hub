"""
Test Harness
============================

Shared fixtures every test file uses.

Modules are imported directly from the installed `dlhub` package. This file held
a loader through the migration, resolving a module from the package when it had
moved and from its old file when it had not; the last module has moved, so the
loader and the parallel import system it maintained are gone.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """A seeded generator. Every test drawing random data draws it from here."""
    return np.random.default_rng(0)


@pytest.fixture
def params(rng):
    """
    A two-layer parameter dictionary in the shape convention the hub's
    documents use: weights are (out, in) and biases are (out, 1).
    """
    return {
        "W1": rng.standard_normal((4, 3)) * 0.1,
        "b1": np.zeros((4, 1)),
        "W2": rng.standard_normal((1, 4)) * 0.1,
        "b2": np.zeros((1, 1)),
    }


@pytest.fixture
def grads(params, rng):
    """Gradients matching `params`, non-zero so an update is observable."""
    return {k: rng.standard_normal(v.shape) * 0.01 for k, v in params.items()}


@pytest.fixture
def binary_fixture(rng):
    """
    A small separable binary problem: X is (features, samples), Y is (1, samples).

    Separable so that a correct implementation drives the cost down, which lets a
    test assert learning happened without asserting a specific loss value.
    """
    n_features, n_per_class = 3, 16
    positive = rng.standard_normal((n_features, n_per_class)) * 0.3 + 1.0
    negative = rng.standard_normal((n_features, n_per_class)) * 0.3 - 1.0
    X = np.hstack([positive, negative])
    Y = np.hstack([np.ones((1, n_per_class)), np.zeros((1, n_per_class))])
    return X, Y
