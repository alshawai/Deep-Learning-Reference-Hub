"""
Loader Seam Tests
============================

Covers the temporary loader in `conftest` that resolves a module either from the
`dlhub` package or from its old file, whichever exists.

This file is deleted with the seam it covers. It is here because a seam that
silently only ever took the fallback path would look exactly like a working one:
the suite would pass, every module would load from its file, and the migration
would appear finished without a single module having moved. These tests fail in
that case.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import conftest
import pytest
from conftest import NUMPY_EXAMPLES, PACKAGE_MODULE, load


def test_a_module_that_has_not_moved_loads_from_its_file():
    """
    The fallback path. Asserted against a module's real content rather than its
    mere existence, so an empty or half-executed module fails here.
    """
    adam = load("optimization_algorithms/adam_optimizer.py")
    assert hasattr(adam, "AdamOptimizer")


def test_a_module_that_has_moved_loads_from_the_package():
    """
    The package path, exercised against `dlhub` itself rather than a hub module,
    so this holds from the first commit of the migration to the last. Without it
    the suite could pass with the package import silently broken for every
    module, and every load quietly taking the fallback.
    """
    PACKAGE_MODULE["_seam_probe.py"] = "dlhub"
    try:
        module = load("_seam_probe.py")
    finally:
        PACKAGE_MODULE.pop("_seam_probe.py", None)
        conftest._CACHE.pop("_seam_probe.py", None)

    assert module.__name__ == "dlhub"
    assert hasattr(module, "__version__")


def test_a_module_that_resolves_neither_way_fails_loudly():
    """
    A typo in a path, or a module renamed without its table entry updated, must
    stop the run at the call. Returning None would surface as an
    `AttributeError` several frames away, in a test that looks unrelated.
    """
    with pytest.raises(FileNotFoundError, match="no-such-module"):
        load("no-such-module.py")


def test_every_module_in_the_table_still_resolves():
    """
    The migration's own guard rail. Each entry resolves through the package or
    the file, so a module moved without its table entry updated -- or moved to a
    different name than the table claims -- fails here rather than in whichever
    test file happens to load it.
    """
    for relative_path in PACKAGE_MODULE:
        assert load(relative_path) is not None, relative_path


def test_every_unmoved_module_has_a_package_home_assigned():
    """
    The table has to name a destination for every module still in the old tree,
    or Phase 1 finishes with a module stranded there and nothing reports it. A
    new module added to the old tree during the migration fails here until its
    destination is decided, which is the moment to decide it.
    """
    stranded = {
        str(path.relative_to(NUMPY_EXAMPLES))
        for path in NUMPY_EXAMPLES.rglob("*.py")
        if not path.name.startswith("__")
    } - set(PACKAGE_MODULE)
    assert not stranded, f"no package home assigned for: {sorted(stranded)}"
