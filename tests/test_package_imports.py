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

A framework port is the one exception the check bends for. It depends on an
optional framework -- `torch` for the PyTorch ports -- that the docs/style CI job
does not install, so importing it there raises `ModuleNotFoundError` for the
framework itself. That is the optional dependency being absent, not the module
being broken, so the module is skipped where its framework is missing and
imported for real in the parity job, which installs it. Any other missing import
still fails, so a genuinely broken module is still caught.

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

# The frameworks the hub ports to. Each is an optional dependency, installed only
# in the parity CI job; the docs/style job runs without them.
OPTIONAL_FRAMEWORKS = {"torch", "tensorflow"}


def import_or_skip_optional(name):
    """
    Import `name`, but skip if it fails only for a missing optional framework.

    A `ModuleNotFoundError` naming one of `OPTIONAL_FRAMEWORKS` means the port's
    framework is not installed here; that is expected in the docs/style job and
    is not a defect in the module. Every other import error -- including a
    `ModuleNotFoundError` for anything else, such as a mistyped stdlib import --
    propagates, so the smoke test still catches a broken module.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in OPTIONAL_FRAMEWORKS:
            pytest.skip(f"{name} needs optional framework {missing!r}, absent here")
        raise


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
    directly, whether or not a test file happens to exercise it. A framework port
    is imported where its framework is installed and skipped where it is not.
    """
    assert import_or_skip_optional(name) is not None


@pytest.mark.parametrize("name", published_modules())
def test_importing_a_module_does_not_draw_a_plot(name):
    """
    Several modules ship a `main` that plots, and matplotlib's pyplot opens a
    window on import in some backends. A figure created at import time would hang
    a headless CI run rather than fail it, which is the worse failure.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    import_or_skip_optional(name)
    assert not pyplot.get_fignums(), f"{name} created a figure at import time"
