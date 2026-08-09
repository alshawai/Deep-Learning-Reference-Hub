"""
Test Harness
============================

Shared fixtures and the module loader every test file uses.

The hub's modules are teaching artifacts published under `code-examples/`, not an
installed package. Two facts about that tree decide how they get imported here:
`code-examples` is not a valid identifier, and `code-examples/numpy/` would shadow
the real NumPy if its parent ever landed on `sys.path`. So modules are loaded from
their file paths, and the package directory is never added to the path.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMPY_EXAMPLES = REPO_ROOT / "code-examples" / "numpy"

_CACHE: dict[str, object] = {}


def load(relative_path: str):
    """
    Load a hub module from its path under `code-examples/numpy/`.

    Parameters
    ----------
    relative_path : str
        Path relative to `code-examples/numpy`, e.g.
        `"optimization_algorithms/adam_optimizer.py"`

    Returns
    -------
    module
        The executed module object.

    Raises
    ------
    FileNotFoundError
        If no file exists at that path, so a renamed module fails loudly here
        rather than as a confusing collection error.
    """
    if relative_path in _CACHE:
        return _CACHE[relative_path]

    path = NUMPY_EXAMPLES / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"No hub module at {path}")

    name = f"hub_{relative_path.replace('/', '_').removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _CACHE[relative_path] = module
    return module


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
