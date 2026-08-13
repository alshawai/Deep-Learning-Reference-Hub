"""
Hub Check Tests
============================

Covers how `tools/hubcheck.py` decides what counts as a published
implementation -- the set the README's "Code Examples: N" claim is checked
against.

That decision used to be made by subtraction: every published `.py` except the
ones under `tools/` and `tests/`. It now asks whether a file sits in an
importable package. The count is the same either way today, so a test is the
only thing that distinguishes the two definitions, and the only thing that will
notice if a later edit drifts back toward naming directories.

The central assertion is agreement with `pkgutil`: whatever the package really
publishes is what the README is held to. A checker that counts a different set
than the one a reader can import is wrong even when its number is right.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import importlib.util
import pkgutil
import sys
from pathlib import Path

import pytest

import dlhub

REPO_ROOT = Path(__file__).resolve().parent.parent
HUBCHECK_PATH = REPO_ROOT / "tools" / "hubcheck.py"


def load_hubcheck():
    """
    Import `tools/hubcheck.py` by path.

    The tool deliberately is not part of the `dlhub` package -- it is repo
    machinery, and it stays dependency-free so it runs in a bare clone -- so
    there is no import path to reach it by. Loading it from its own location is
    the cost of that, and it keeps the test honest about testing the file CI
    actually runs.
    """
    spec = importlib.util.spec_from_file_location("hubcheck", HUBCHECK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hubcheck"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hubcheck():
    """The tool, loaded once. Its file scan is cached, so reloading is waste."""
    return load_hubcheck()


def dotted_name(path, root):
    """Turn `src/dlhub/optimizers/adam.py` into `dlhub.optimizers.adam`."""
    return ".".join(path.relative_to(root / "src").with_suffix("").parts)


class TestCodeModules:
    """The set the README's implementation count is verified against."""

    def test_the_scan_finds_modules_at_all(self, hubcheck):
        """
        Guards every assertion below. An empty scan would satisfy a disjointness
        or exclusion claim trivially, and would make the README's count check
        pass by comparing against nothing.
        """
        assert len(hubcheck.code_modules()) > 10

    def test_it_counts_exactly_what_the_package_publishes(self, hubcheck):
        """
        The contract. `pkgutil` reports what a reader importing `dlhub` can
        actually reach; the README's claim is checked against hubcheck's list.
        Those two agreeing is the whole point -- if they diverge, the claim is
        being verified against a set nobody can import.
        """
        published = {
            module.name
            for module in pkgutil.walk_packages(dlhub.__path__, prefix="dlhub.")
            if not module.ispkg
        }
        counted = {dotted_name(path, hubcheck.ROOT) for path in hubcheck.code_modules()}
        assert counted == published

    def test_it_does_not_count_the_checker_or_the_tests(self, hubcheck):
        """
        Neither is an implementation the README publishes. Under the current
        definition this falls out of `tools/` and `tests/` having no
        `__init__.py`, which is exactly why it is worth asserting: the property
        is now emergent rather than spelled out, so nothing else would catch a
        stray `__init__.py` quietly inflating the count.
        """
        counted = hubcheck.code_modules()
        assert not [p for p in counted if p.name == "hubcheck.py"]
        assert not [p for p in counted if (hubcheck.ROOT / "tests") in p.parents]

    def test_implementations_and_tests_are_disjoint(self, hubcheck):
        """
        The README states the two counts separately, so no file may answer to
        both. This holds even if tests are one day moved inside the package.
        """
        assert not set(hubcheck.code_modules()) & set(hubcheck.test_modules())

    def test_package_wiring_files_are_not_implementations(self, hubcheck):
        """
        `__init__.py` re-exports; it does not implement a method. Counting the
        five of them would overstate the hub by five.
        """
        assert not [p for p in hubcheck.code_modules() if p.name.startswith("__")]


class TestPublishedDocs:
    """The set the README's document count is verified against."""

    def test_the_scan_finds_documents_at_all(self, hubcheck):
        """
        Guards the exclusion below. A scan returning nothing would satisfy any
        "is not counted" claim trivially, and would make the README's document
        count pass by comparing against an empty set.
        """
        assert len(hubcheck.published_docs()) > 1

    def test_signpost_pages_are_not_counted_as_documents(self, hubcheck):
        """
        The site landing page and each section home are named `index.md`. They
        route a reader to documents rather than being documents, so counting
        them would inflate the README's claim once per signpost -- and the
        tutorial home, whose subject is that its section is empty, would be
        counted as a document about nothing.

        The first assertion is what keeps the second honest: it fails if the
        documentation tree stops containing signposts, which is the only way
        the second could pass without the exclusion doing any work.
        """
        assert [p for p in hubcheck.md_files() if p.name.lower() in hubcheck.SIGNPOSTS]
        assert not [
            p for p in hubcheck.published_docs() if p.name.lower() in hubcheck.SIGNPOSTS
        ]

    def test_generated_pages_are_not_counted_as_documents(self, hubcheck):
        """
        An API page's body is a docstring-extraction directive, so every word a
        reader sees on it comes from a module the implementation count already
        counts. Counting it as a document too would report one piece of work
        twice, and would make the README's document count grow whenever a
        subpackage was added.

        The first assertion keeps the second honest: it fails if no generated
        page is in the tree, which is the only way the second could pass with
        the exclusion doing nothing.
        """
        generated = [
            p
            for p in hubcheck.md_files()
            if (hubcheck.read_text(p) or "")
            and hubcheck.GENERATED_RE.search(hubcheck.read_text(p) or "")
        ]
        assert generated
        assert not set(generated) & set(hubcheck.published_docs())


class TestReadmeCheck:
    """The check that keeps the README's self-description true."""

    def test_the_readme_as_written_passes(self, hubcheck):
        """
        An end-to-end assertion on the real README. Phase 3 changed both the
        tree and the definition of what is counted, and a contributor reading a
        stale number is the failure this check exists to prevent.
        """
        assert hubcheck.check_readme() == []

    def test_it_reports_what_it_verified(self, hubcheck):
        """
        The tool promises `ok` never means `covered nothing`. The README check
        earns that by recording a claim count, which a rewording that hides the
        statistics section from the parser would drop to zero.
        """
        hubcheck.check_readme()
        assert "claim(s) verified" in hubcheck.COVERAGE["readme"]
        assert not hubcheck.COVERAGE["readme"].startswith("0 ")
