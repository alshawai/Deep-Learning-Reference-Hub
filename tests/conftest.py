"""
Test Harness
============================

Shared fixtures and the module loader every test file uses.

The hub's modules are moving from standalone files under `code-examples/` into
the installed `dlhub` package. While that is underway, `load` resolves a module
from the package when it has moved and from its file when it has not, so the
suite passes at every point in the migration rather than only at the end.

This loader is temporary. It is deleted once the last module has moved, along
with the file-path machinery that only ever existed because `code-examples` is
not a valid identifier and a directory named `numpy` would shadow the real one.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMPY_EXAMPLES = REPO_ROOT / "code-examples" / "numpy"

# Where each module lands in the package. This table is the migration plan
# written down: a module that has moved resolves through its dotted name, and one
# that has not falls back to its file. Both work, which is what lets the modules
# move one commit at a time instead of in one unreviewable change.
#
# An entry here is a claim about the future, not the present. Nothing breaks
# while the target does not exist yet.
PACKAGE_MODULE = {
    "optimization_algorithms/adam_optimizer.py": "dlhub.optimizers.adam",
    "optimization_algorithms/momentum_optimizer.py": "dlhub.optimizers.momentum",
    "optimization_algorithms/rmsprop_optimizer.py": "dlhub.optimizers.rmsprop",
    "optimization_algorithms/mini_batch_gradient_descent.py": "dlhub.optimizers.mini_batch",
    "optimization_algorithms/exponential_weighted_averages.py": "dlhub.optimizers.exponential_weighted_averages",
    "optimization_algorithms/optimization_comparison.py": "dlhub.optimizers.comparison",
    "learning_rate_schedulers.py": "dlhub.optimizers.schedules",
    "hyperparameter_tuning/random_search.py": "dlhub.tuning.random_search",
    "hyperparameter_tuning/bayesian_optimization.py": "dlhub.tuning.bayesian",
    "hyperparameter_tuning/multifidelity_optimization.py": "dlhub.tuning.multifidelity",
    "hyperparameter_tuning/population_based_training.py": "dlhub.tuning.population_based",
    "hyperparameter_tuning/complete_tuning_framework.py": "dlhub.tuning.framework",
    "learning_rate_finder.py": "dlhub.tuning.learning_rate_finder",
    "fully_connected_nn_with_regularization.py": "dlhub.nn.fully_connected",
    "gradient_checking.py": "dlhub.training.gradient_checking",
    "early_stopping.py": "dlhub.training.early_stopping",
}

_CACHE: dict[str, object] = {}


def _load_from_package(relative_path: str):
    """
    Import a module from the `dlhub` package, if it has moved there already.

    Parameters
    ----------
    relative_path : str
        Key into `PACKAGE_MODULE`.

    Returns
    -------
    module or None
        The imported module, or None if it has no package home yet or has not
        moved into the one it is assigned.
    """
    dotted = PACKAGE_MODULE.get(relative_path)
    if dotted is None:
        return None
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError:
        return None


def _load_from_file(relative_path: str):
    """
    Execute a module from its path under `code-examples/numpy/`.

    The package directory is never added to `sys.path`, because
    `code-examples/numpy/` would shadow the real NumPy from there.

    Parameters
    ----------
    relative_path : str
        Path relative to `code-examples/numpy`.

    Returns
    -------
    module or None
        The executed module, or None if no file exists at that path.
    """
    path = NUMPY_EXAMPLES / relative_path
    if not path.is_file():
        return None

    name = f"hub_{relative_path.replace('/', '_').removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(relative_path: str):
    """
    Load a hub module, from the package if it has moved and from its file if not.

    Parameters
    ----------
    relative_path : str
        Path relative to `code-examples/numpy`, e.g.
        `"optimization_algorithms/adam_optimizer.py"`. This stays the address
        even after the module moves, so a test file is rewritten to a direct
        import when its module moves rather than being edited twice.

    Returns
    -------
    module
        The executed module object.

    Raises
    ------
    FileNotFoundError
        If the module resolves neither way, so a renamed module fails loudly
        here rather than as a confusing collection error.
    """
    if relative_path in _CACHE:
        return _CACHE[relative_path]

    module = _load_from_package(relative_path)
    if module is None:
        module = _load_from_file(relative_path)
    if module is None:
        raise FileNotFoundError(
            f"No hub module for {relative_path!r}: not importable as "
            f"{PACKAGE_MODULE.get(relative_path, '(no package home assigned)')} "
            f"and no file at {NUMPY_EXAMPLES / relative_path}"
        )

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
