"""
Package Import Tests
====================

Every module the hub publishes must import. CI once proved this separately, by
walking the tree and exec'ing each file through `importlib` -- a step that made
sense while modules lived outside the package and were not all covered by tests.
They are in the package now and the suite imports all of them, so the check
belongs here, where it runs the same way for a contributor as for CI.

What this adds over the suite importing them incidentally: it fails for a module
that no test file imports. That module would otherwise be published with a syntax
error, a bad top-level call, or a circular import, and nothing would say so.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import importlib
import pkgutil

import pytest

import dlhub


def published_modules():
    """
    Dotted names of every importable module under `dlhub`, subpackages included.

    Walked rather than listed, so a new module is covered by the tests below the
    moment it is added and without an entry to remember to write.
    """
    return sorted(
        module.name
        for module in pkgutil.walk_packages(dlhub.__path__, prefix="dlhub.")
        if not module.ispkg
    )


def test_the_walk_finds_the_modules():
    """
    Guards the parametrised test below from passing vacuously. An empty walk --
    from a renamed package directory or a missing `__init__` -- would collect
    zero cases and report success.
    """
    found = published_modules()
    assert len(found) > 10, found


@pytest.mark.parametrize("name", published_modules())
def test_every_published_module_imports(name):
    """
    Imports each module on its own. A module is published material: it has to be
    importable by a reader who pip-installs the package and reaches for it
    directly, whether or not a test file happens to exercise it.
    """
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", published_modules())
def test_importing_a_module_does_not_draw_a_plot(name):
    """
    Several modules ship a `main` that plots, and matplotlib's pyplot opens a
    window on import in some backends. A figure created at import time would hang
    a headless CI run rather than fail it, which is the worse failure.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    importlib.import_module(name)
    assert not pyplot.get_fignums(), f"{name} created a figure at import time"
