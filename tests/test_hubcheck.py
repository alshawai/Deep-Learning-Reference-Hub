"""
Hub Check Tests
============================

Covers how `tools/hubcheck.py` decides what it is looking at: which files count
as published implementations, which count as documents, and which Markdown is
its own to check rather than the site build's.

The implementation count used to be decided by subtraction: every published
`.py` except the ones under `tools/` and `tests/`. It now asks whether a file
sits in an importable package. The count is the same either way today, so a test
is the only thing that distinguishes the two definitions, and the only thing
that will notice if a later edit drifts back toward naming directories.

The central assertion there is agreement with `pkgutil`: whatever the package
really publishes is what the README is held to. A checker that counts a
different set than the one a reader can import is wrong even when its number is
right.

The domain tests carry the same weight for links and anchors. Those checks were
scoped to the tree `mkdocs build --strict` does not build, rather than retired
into it, and scoping is only safe while the two halves genuinely partition the
published Markdown. Nothing but a test says they still do.

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


class TestCheckDomains:
    """The division of labour between this tool and the site build."""

    def test_the_two_domains_partition_the_published_markdown(self, hubcheck):
        """
        The contract that makes reassigning the link checks safe rather than a
        quiet loss of coverage. `mkdocs build --strict` resolves references
        inside its own source tree; this tool takes everything else. If the two
        sets overlapped, work would be done twice; if they did not cover
        `md_files()` between them, some published page would be checked by
        nobody -- and a checker going silent after a reorganization is the
        failure this file exists to prevent.
        """
        built = set(hubcheck.built_md_files())
        unbuilt = set(hubcheck.unbuilt_md_files())
        assert built | unbuilt == set(hubcheck.md_files())
        assert not built & unbuilt

    def test_both_sides_of_the_partition_are_populated(self, hubcheck):
        """
        Guards the partition above, which an empty set would satisfy for free:
        with nothing built, the union and disjointness claims still hold while
        the split does no work at all.

        Both sides being non-empty is also the true state of the repository --
        there is a site, and there is Markdown outside it -- so this failing
        means either the site config went missing or the last document outside
        it was filed away. Both change what the checks cover, and both deserve
        to be noticed rather than absorbed.
        """
        assert hubcheck.built_md_files()
        assert hubcheck.unbuilt_md_files()

    def test_the_repository_prose_stays_on_this_tool(self, hubcheck):
        """
        README and CONTRIBUTING can never be inside the site build -- MkDocs
        renders `docs_dir` and nothing above it -- so they are the part of the
        partition that will still be here after every document has moved. They
        are named explicitly because they are the reason the checks were scoped
        instead of deleted.
        """
        unbuilt = {p.name for p in hubcheck.unbuilt_md_files()}
        assert "README.md" in unbuilt
        assert "CONTRIBUTING.md" in unbuilt

    def test_the_configured_docs_dir_is_what_sets_the_boundary(
        self, hubcheck, monkeypatch, tmp_path
    ):
        """
        The split follows `docs_dir` rather than a hardcoded directory name, so
        moving the documentation tree moves the boundary with it.

        Pointed at a config of this test's own writing, because the obvious
        version of this assertion is vacuous: `built_md_files` and
        `unbuilt_md_files` both call `site_source_dir`, so they agree with each
        other whatever it returns -- including when it ignores the config and
        falls back to `docs`. Only an independently written config distinguishes
        a scanner that reads the key from one that merely guesses correctly.
        """
        config = tmp_path / "mkdocs.yml"
        config.write_text("site_name: Hub\ndocs_dir: elsewhere\n", encoding="utf-8")
        monkeypatch.setattr(hubcheck, "MKDOCS_CONFIG", config)
        assert hubcheck.site_source_dir() == (hubcheck.ROOT / "elsewhere").resolve()

    def test_only_a_top_level_key_moves_the_boundary(
        self, hubcheck, monkeypatch, tmp_path
    ):
        """
        `docs_dir` indented under a plugin is that plugin's own setting and says
        nothing about where the site is built from. A line scan that matched it
        anywhere on the line would divide the tree along a directory MkDocs
        never heard of, and the checks would go quiet over whatever fell outside
        it.
        """
        config = tmp_path / "mkdocs.yml"
        config.write_text(
            "site_name: Hub\nplugins:\n  - somewhere:\n      docs_dir: trap\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(hubcheck, "MKDOCS_CONFIG", config)
        assert hubcheck.site_source_dir() == (hubcheck.ROOT / "docs").resolve()

    def test_no_site_config_leaves_the_whole_tree_to_this_tool(
        self, hubcheck, monkeypatch, tmp_path
    ):
        """
        The fail-closed reading of a missing `mkdocs.yml`. No config means no
        build is resolving anything, so the scoped checks must take everything
        back rather than assume a `docs/` that may not exist -- otherwise
        deleting the config would silently retire two checks.
        """
        monkeypatch.setattr(hubcheck, "MKDOCS_CONFIG", tmp_path / "absent.yml")
        assert hubcheck.site_source_dir() is None
        assert hubcheck.built_md_files() == []
        assert hubcheck.unbuilt_md_files() == hubcheck.md_files()

    def test_this_repository_is_split_where_its_config_says(self, hubcheck):
        """
        The same claim against the real config, read here by a plain string scan
        so that a rewrite of :func:`site_source_dir` is checked rather than
        trusted.
        """
        lines = hubcheck.MKDOCS_CONFIG.read_text(encoding="utf-8").splitlines()
        declared = [ln for ln in lines if ln.startswith("docs_dir:")]
        expected = declared[0].split(":", 1)[1].strip() if declared else "docs"
        site = (hubcheck.ROOT / expected).resolve()
        assert hubcheck.site_source_dir() == site
        assert site.is_dir()
        assert all(site in p.parents for p in hubcheck.built_md_files())
        assert not any(site in p.parents for p in hubcheck.unbuilt_md_files())

    def test_a_site_page_is_refused_rather_than_passed(self, hubcheck):
        """
        `--file` names one document, and a document inside the site build is
        outside these checks' domain. Reporting that is what keeps the scoping
        honest: the alternative is a scan of zero documents printing `ok`, which
        would tell a contributor their page was checked when nothing looked at
        it.
        """
        page = hubcheck.built_md_files()[0]
        for check in (hubcheck.check_links, hubcheck.check_anchors):
            problems = check(page)
            assert len(problems) == 1
            assert "mkdocs build --strict" in problems[0]

    def test_the_checks_that_are_not_divided_still_see_everything(self, hubcheck):
        """
        No site build compiles the Python in a fence or counts what the README
        claims, so those two checks keep the whole tree. Scoping them along with
        the others would drop every fence inside `docs/` -- the exact
        weakening that scoping rather than deleting was meant to avoid.
        """
        hubcheck.check_fences()
        built = len(hubcheck.built_md_files())
        assert built > 0
        assert (
            f"across {len(hubcheck.md_files())} document(s)"
            in (hubcheck.COVERAGE["fences"])
        )


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
