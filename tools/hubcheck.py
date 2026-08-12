#!/usr/bin/env python3
"""
Hub Check
============================

Mechanical integrity checks over the reference hub's published material. Four
checks, one file, because they all need the same Markdown parsing: relative
links resolve, in-document anchors resolve, Python shown in prose actually
compiles, and the README's self-description matches the tree.

This is a utility, not a deep learning implementation. It has no third-party
dependencies so that it runs in any environment the hub is cloned into.

A fence that is deliberately pseudo-code opts out with an HTML comment on the
line immediately above it:

    <!-- hubcheck: skip -->

Notes
-----
Every check is built to fail closed. A check that cannot find the thing it
verifies reports that fact rather than passing quietly, because a checker that
goes silent after a reorganization is worse than no checker at all.

Published material is whatever git would ship: tracked files plus untracked
files that ``.gitignore`` does not exclude. That definition survives any
reorganization of the tree, which a hardcoded directory list does not.

Anchor slugs follow GitHub's rules. Retargeting the hub at another renderer
means revisiting :func:`slugify`.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import unquote


def _find_root(start: Path) -> Path:
    """Locate the repository root from this file's position.

    Parameters
    ----------
    start : Path
        Directory to search upward from.

    Returns
    -------
    Path
        The directory holding ``.git``, or ``start``'s parent when none is found.
    """
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start.parent


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = _find_root(TOOLS_DIR)

# Consulted only when git is unavailable. With git present, .gitignore decides.
FALLBACK_SKIP_DIRS = {".git", ".claude", "__pycache__", ".ruff_cache", ".vscode"}

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
LINK_RE = re.compile(r"(?<!!)\[(?P<text>[^\]\n]*)\]\((?P<href>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
SKIP_MARK = "hubcheck: skip"
SIGN_RE = re.compile(r"[-+*/]=\s*[-+](?![-+=])")

# A statistics line: an optionally decorated label, a separator, then a value.
# Table rows, bolded list items, and plain `Label: value` all match; ordinary
# prose does not, which keeps the claim scan from inventing findings.
CLAIM_RE = re.compile(
    r"^[\s>\-*+|]*\**\s*(?P<label>[A-Za-z][A-Za-z ./+-]{2,40}?)\s*\**\s*[:|]\s*(?P<value>.+?)\s*\|?\s*$"
)

# Headings that promise verifiable numbers. One of these with no parseable
# claim under it means the README was rephrased out from under this check.
STATS_HEADING_RE = re.compile(
    r"^#{1,6}\s.*\b(statistics|by the numbers|at a glance)\b", re.IGNORECASE
)

# `8+`, `over 8`, `more than 8` state a floor rather than an exact count, so
# only an undershoot is a false claim.
AT_LEAST_RE = re.compile(r"\d+\s*\+|\b(?:over|more than|at least)\s+\d+", re.IGNORECASE)

REPO_META = {
    "readme.md",
    "contributing.md",
    "license.md",
    "changelog.md",
    "code_of_conduct.md",
    "security.md",
}

PLACEHOLDERS = (
    "yourusername",
    "your-username",
    "your_username",
    "yourrepo",
    "your-repo",
    "changeme",
    "path/to/",
    "lorem ipsum",
    "<your ",
)

FRAMEWORK_ALIASES = {
    "numpy": {"numpy"},
    "pytorch": {"torch"},
    "torch": {"torch"},
    "tensorflow": {"tensorflow"},
    "keras": {"keras", "tensorflow"},
    "tensorflow/keras": {"tensorflow", "keras"},
    "jax": {"jax"},
    "flax": {"flax"},
    "scikit-learn": {"sklearn"},
    "sklearn": {"sklearn"},
    "pandas": {"pandas"},
    "matplotlib": {"matplotlib"},
}

_REPO_FILES: Optional[List[Path]] = None

# One line per check describing what it actually inspected. A check that
# verified nothing says so here, so `ok` is never mistaken for `covered`.
COVERAGE: Dict[str, str] = {}


def repo_files() -> List[Path]:
    """Return every file that counts as published material.

    Asks git for tracked files plus untracked files that are not ignored, so
    a document written moments ago is checked and a private directory is not.
    Falls back to a directory walk when git is unavailable.

    Returns
    -------
    list of Path
        Sorted absolute paths.
    """
    global _REPO_FILES
    if _REPO_FILES is not None:
        return _REPO_FILES

    listing = None
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if done.returncode == 0:
            listing = done.stdout
    except (OSError, subprocess.SubprocessError):
        listing = None

    if listing is not None:
        found = {ROOT / name for name in listing.split("\0") if name}
        _REPO_FILES = sorted(p for p in found if p.is_file())
        return _REPO_FILES

    out = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts[:-1]
        if any(part in FALLBACK_SKIP_DIRS for part in parts):
            continue
        out.append(path)
    _REPO_FILES = sorted(out)
    return _REPO_FILES


def md_files() -> List[Path]:
    """Return every published Markdown file.

    Returns
    -------
    list of Path
        Sorted absolute paths.
    """
    return [p for p in repo_files() if p.suffix.lower() == ".md"]


def published_docs() -> List[Path]:
    """Return Markdown files that count as documents rather than repo meta.

    Returns
    -------
    list of Path
        Every published Markdown file except README, LICENSE, and their kin.
    """
    return [p for p in md_files() if p.name.lower() not in REPO_META]


def code_modules() -> List[Path]:
    """Return published Python modules that count as code examples.

    A module counts when it sits inside an importable package, which is to say
    when its own directory carries an ``__init__.py``. The test is positive
    rather than subtractive: it finds the implementations wherever the package
    tree lives, and it leaves out this tool, the test suite, and any scratch
    script at the repo root by virtue of their not being packages, without
    naming a single directory. Dunder files are excluded because they wire a
    package together rather than implement anything, and ``test_*.py`` because
    :func:`test_modules` counts those -- the two sets stay disjoint even if
    tests are one day moved inside the package.

    Returns
    -------
    list of Path
        Sorted absolute paths.
    """
    return [
        p
        for p in repo_files()
        if p.suffix == ".py"
        and (p.parent / "__init__.py").exists()
        and not p.name.startswith("test_")
        and not p.name.startswith("__")
    ]


def test_modules() -> List[Path]:
    """Return published test modules.

    Returns
    -------
    list of Path
        Sorted absolute paths.
    """
    return [p for p in repo_files() if p.suffix == ".py" and p.name.startswith("test_")]


def notebooks() -> List[Path]:
    """Return published Jupyter notebooks.

    Returns
    -------
    list of Path
        Sorted absolute paths.
    """
    return [p for p in repo_files() if p.suffix == ".ipynb"]


def read_text(path: Path) -> Optional[str]:
    """Read a file as UTF-8, without masking a decoding failure.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str or None
        The text, or None when the file is unreadable or not valid UTF-8.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def frameworks_present() -> Set[str]:
    """Return the top-level packages that published Python actually imports.

    Returns
    -------
    set of str
        Lowercased package names.
    """
    names: Set[str] = set()
    for path in code_modules():
        text = read_text(path)
        if text is None:
            continue
        for match in IMPORT_RE.finditer(text):
            names.add(match.group(1).split(".")[0].lower())
    return names


def strip_fences(text: str) -> List[Tuple[int, str]]:
    """Return numbered lines that sit outside fenced code blocks.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    list of tuple
        ``(line_number, line)`` pairs, 1-indexed.
    """
    out, closer = [], None
    for i, line in enumerate(text.splitlines(), 1):
        match = FENCE_RE.match(line)
        if closer is None and match:
            closer = match.group(2)[0] * len(match.group(2))
            continue
        if closer is not None:
            if match and match.group(2).startswith(closer):
                closer = None
            continue
        out.append((i, line))
    return out


def slugify(heading: str) -> str:
    """Convert heading text to a GitHub anchor.

    GitHub lowercases, deletes every character that is not alphanumeric,
    space, hyphen, or underscore, then maps each remaining space to one
    hyphen without collapsing runs.

    Parameters
    ----------
    heading : str
        Raw heading text, without the leading hashes.

    Returns
    -------
    str
        The anchor, without its leading ``#``.
    """
    text = heading.strip().lower()
    text = LINK_RE.sub(lambda m: m.group("text"), text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = "".join(c for c in text if c.isalnum() or c in " -_")
    return text.replace(" ", "-")


def anchors_of(text: str) -> Dict[str, int]:
    """Map every anchor a document emits to the count of headings claiming it.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    dict
        Anchor string to number of occurrences.
    """
    counts: Dict[str, int] = {}
    emitted: Dict[str, int] = {}
    for _, line in strip_fences(text):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slugify(match.group(2))
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        emitted[base if seen == 0 else f"{base}-{seen}"] = 1
    return emitted


def normalize_fragment(fragment: str) -> str:
    """Apply the heading character filter to a link's ``#fragment``.

    Authors paste anchors that carry an emoji's invisible variation selector
    while the heading's own slug drops it. Both sides go through the same
    filter so that difference stops reading as a broken anchor.

    Parameters
    ----------
    fragment : str
        The part of a link after ``#``, already URL-decoded.

    Returns
    -------
    str
        The comparable anchor.
    """
    text = fragment.strip().lower()
    return "".join(c for c in text if c.isalnum() or c in "-_")


def check_links(only: Optional[Path] = None) -> List[str]:
    """Verify every relative Markdown link resolves on disk.

    Parameters
    ----------
    only : Path, optional
        Restrict the check to a single document.

    Returns
    -------
    list of str
        One human-readable failure per broken link.
    """
    fails = []
    seen = links = 0
    for path in md_files():
        if only and path != only:
            continue
        rel = path.relative_to(ROOT)
        text = read_text(path)
        if text is None:
            fails.append(f"{rel}: unreadable, or not valid UTF-8")
            continue
        seen += 1
        for line_no, line in strip_fences(text):
            for match in LINK_RE.finditer(line):
                href = match.group("href")
                if href.startswith(("http://", "https://", "mailto:", "#", "<")):
                    continue
                target, _, frag = href.partition("#")
                if not target:
                    continue
                links += 1
                resolved = (path.parent / unquote(target)).resolve()
                if not resolved.exists():
                    fails.append(f"{rel}:{line_no}: dead link -> {target}")
                elif frag and resolved.suffix == ".md":
                    other = read_text(resolved)
                    if other is None:
                        fails.append(f"{rel}:{line_no}: target unreadable -> {target}")
                        continue
                    known = {normalize_fragment(a) for a in anchors_of(other)}
                    if normalize_fragment(unquote(frag)) not in known:
                        fails.append(f"{rel}:{line_no}: dead anchor -> {target}#{frag}")
    COVERAGE["links"] = f"{links} relative link(s) across {seen} document(s)"
    return fails


def check_anchors(only: Optional[Path] = None) -> List[str]:
    """Verify every in-document anchor points at a heading that exists.

    Catches the stale table-of-contents row: a heading renamed or removed
    while its own TOC entry stayed behind.

    Parameters
    ----------
    only : Path, optional
        Restrict the check to a single document.

    Returns
    -------
    list of str
        One human-readable failure per unresolved anchor.
    """
    fails = []
    seen = found = 0
    for path in md_files():
        if only and path != only:
            continue
        rel = path.relative_to(ROOT)
        text = read_text(path)
        if text is None:
            fails.append(f"{rel}: unreadable, or not valid UTF-8")
            continue
        seen += 1
        emitted = {normalize_fragment(a) for a in anchors_of(text)}
        for line_no, line in strip_fences(text):
            for match in LINK_RE.finditer(line):
                href = match.group("href")
                if not href.startswith("#"):
                    continue
                anchor = normalize_fragment(unquote(href[1:]))
                if not anchor:
                    continue
                found += 1
                if anchor not in emitted:
                    fails.append(f"{rel}:{line_no}: anchor has no heading -> #{anchor}")
    COVERAGE["anchors"] = f"{found} in-document anchor(s) across {seen} document(s)"
    return fails


def python_fences(text: str) -> List[Tuple[int, str]]:
    """Extract Python code blocks from a document.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    list of tuple
        ``(start_line, source)`` for each non-skipped Python fence.
    """
    out: List[Tuple[int, str]] = []
    lines = text.splitlines()
    closer, start, buf, lang, pad = None, 0, [], "", 0
    for i, line in enumerate(lines, 1):
        match = FENCE_RE.match(line)
        if closer is None:
            if match:
                closer = match.group(2)[0] * len(match.group(2))
                lang, start, buf = match.group(3).lower(), i, []
                pad = len(match.group(1))
            continue
        if match and match.group(2).startswith(closer):
            skip = start >= 2 and SKIP_MARK in lines[start - 2]
            if lang in {"python", "py", "python3"} and not skip:
                out.append((start, "\n".join(ln[pad:] for ln in buf)))
            closer = None
            continue
        buf.append(line)
    return out


def check_fences(only: Optional[Path] = None) -> List[str]:
    """Verify Python shown in prose compiles, and carries no flipped sign.

    Prose code is published code. It is read, copied, and trusted, and no
    linter reaches it, which is how ``theta -=- learning_rate * gradient``
    can ship gradient ascent underneath its own descent equation.

    Parameters
    ----------
    only : Path, optional
        Restrict the check to a single document.

    Returns
    -------
    list of str
        One human-readable failure per bad fence.
    """
    fails = []
    seen = fences = 0
    for path in md_files():
        if only and path != only:
            continue
        rel = path.relative_to(ROOT)
        text = read_text(path)
        if text is None:
            fails.append(f"{rel}: unreadable, or not valid UTF-8")
            continue
        seen += 1
        for start, src in python_fences(text):
            fences += 1
            for offset, line in enumerate(src.splitlines()):
                match = SIGN_RE.search(line)
                if match:
                    fails.append(
                        f"{rel}:{start + 1 + offset}: sign flip "
                        f"`{match.group(0).strip()}` -> double negation "
                        f"reverses the update"
                    )
            try:
                compile(src, str(rel), "exec")
            except SyntaxError as exc:
                at = start + (exc.lineno or 0)
                fails.append(f"{rel}:{at}: fence does not compile -> {exc.msg}")
    COVERAGE["fences"] = f"{fences} Python fence(s) across {seen} document(s)"
    return fails


def readme_claims(text: str) -> Iterator[Tuple[int, str, str]]:
    """Yield every label-and-value statistic line in the README.

    Parameters
    ----------
    text : str
        Full README text.

    Yields
    ------
    tuple
        ``(line_number, lowercased_label, value)``.
    """
    for line_no, line in strip_fences(text):
        match = CLAIM_RE.match(line)
        if match:
            yield line_no, match.group("label").strip().lower(), match.group("value").strip()


COUNTABLE = (
    (("document", "doc", "guide", "article"), "document(s)", published_docs),
    (("code example", "implementation", "module", "script"), "code example(s)", code_modules),
    (("notebook",), "notebook(s)", notebooks),
    (("test",), "test(s)", test_modules),
)


def check_readme(only: Optional[Path] = None) -> List[str]:
    """Verify the README's self-description matches the tree.

    Counts are matched by label keyword rather than by exact phrasing, and
    framework coverage is checked against the imports published code actually
    makes. A statistics heading whose numbers this check cannot parse is
    itself a failure, so rewording the README cannot silence the check.

    Parameters
    ----------
    only : Path, optional
        Unused; present so every check shares one signature.

    Returns
    -------
    list of str
        One human-readable failure per false or unverifiable claim.
    """
    fails: List[str] = []
    readme = ROOT / "README.md"
    if not readme.exists():
        COVERAGE["readme"] = "no README.md found"
        return ["README.md: missing"]
    text = read_text(readme)
    if text is None:
        COVERAGE["readme"] = "README.md unreadable"
        return ["README.md: unreadable, or not valid UTF-8"]

    verified = 0
    present = frameworks_present()

    for line_no, label, value in readme_claims(text):
        number = re.search(r"\d+", value)

        matched = False
        for keywords, noun, counter in COUNTABLE:
            if not any(word in label for word in keywords):
                continue
            matched = True
            if not number:
                break
            claimed, actual = int(number.group(0)), len(counter())
            verified += 1
            if AT_LEAST_RE.search(value):
                if actual < claimed:
                    fails.append(
                        f"README.md:{line_no}: claims at least {claimed} {noun}, "
                        f"tree has {actual}"
                    )
            elif claimed != actual:
                fails.append(
                    f"README.md:{line_no}: claims {claimed} {noun}, tree has {actual}"
                )
            break

        if matched or "framework" not in label:
            continue

        for raw in re.split(r"[,/|]|\band\b", value):
            name = raw.strip().strip("*`_ ").lower()
            if not name or name.isdigit():
                continue
            verified += 1
            if not (FRAMEWORK_ALIASES.get(name, {name}) & present):
                fails.append(
                    f"README.md:{line_no}: claims {name} coverage, "
                    f"no published module imports it"
                )

    lowered = text.lower()
    for placeholder in PLACEHOLDERS:
        if placeholder in lowered:
            fails.append(f"README.md: unfilled placeholder `{placeholder.strip()}`")

    has_stats_heading = any(
        STATS_HEADING_RE.match(line) for _, line in strip_fences(text)
    )
    if has_stats_heading and not verified:
        fails.append(
            "README.md: a statistics section exists but no claim in it could be "
            "parsed -- reword the section or teach hubcheck its shape"
        )

    COVERAGE["readme"] = f"{verified} claim(s) verified"
    return fails


CHECKS = {
    "links": check_links,
    "anchors": check_anchors,
    "fences": check_fences,
    "readme": check_readme,
}


def main() -> int:
    """Run the requested checks and report.

    Returns
    -------
    int
        Process exit status: 0 when every requested check passed.
    """
    parser = argparse.ArgumentParser(description="Hub integrity checks.")
    parser.add_argument("check", choices=[*CHECKS, "all"])
    parser.add_argument("--file", help="restrict to one document")
    args = parser.parse_args()

    only = (ROOT / args.file).resolve() if args.file else None
    if only and not only.exists():
        print(f"no such file: {args.file}", file=sys.stderr)
        return 2

    names = list(CHECKS) if args.check == "all" else [args.check]
    total = 0
    for name in names:
        fails = CHECKS[name](only)
        total += len(fails)
        status = "ok" if not fails else f"{len(fails)} problem(s)"
        print(f"== {name}: {status}  [{COVERAGE.get(name, 'nothing inspected')}]")
        for line in fails:
            print(f"   {line}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
