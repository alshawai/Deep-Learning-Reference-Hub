#!/usr/bin/env python3
"""
Release Check
=============

One mechanical check that the hub's release bookkeeping is internally
consistent: the version the package declares, the latest release tag, and
``CHANGELOG.md`` must tell the same story. It is the CI backstop behind the
release ritual described in ``docs/explanation/versioning-and-releases.md``.

This is a utility, not a deep learning implementation. Like ``hubcheck.py`` it
has no third-party dependencies, so it runs in any environment the hub is cloned
into, and it is built to fail closed: a thing it cannot find is reported, never
passed over quietly.

Two severities, because two different mistakes are possible:

* **Problems** fail the build (exit non-zero). These are states that must never
  reach a reader: a version that is unreadable, a declared version with no
  changelog entry, or a version that has regressed below a tag already cut.
* **Nudges** warn and pass (exit zero), surfaced as GitHub Actions annotations.
  These are editorial reminders the tool cannot decide on its own -- a release
  staged but not yet tagged, or a feature that landed with no release prepared.
  The bump itself is a human's editorial call, so the tool prompts rather than
  acts.

A fresh repository with no release tags yet is a nudge, not a problem: the
version history simply has not been bootstrapped.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)
CHANGELOG_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
FEAT_RE = re.compile(r"^feat(\([^)]*\))?!?:", re.IGNORECASE)

Semver = Tuple[int, int, int]


def _find_root(start: Path) -> Path:
    """Locate the repository root from this file's position.

    Parameters
    ----------
    start : Path
        Directory to search upward from.

    Returns
    -------
    Path
        The directory holding ``.git``, or ``start`` itself when none is found.
    """
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return start


def _git(root: Path, *args: str) -> Optional[str]:
    """Run a git command, returning stripped stdout, or ``None`` on failure.

    A failure here is never fatal on its own: a shallow clone missing a tag, or
    a tag absent locally, degrades to a nudge rather than a crash.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def read_version(root: Path) -> Optional[str]:
    """Read ``__version__`` from the package without importing it."""
    source = root / "src" / "dlhub" / "__init__.py"
    if not source.exists():
        return None
    match = VERSION_RE.search(source.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def parse_semver(text: str) -> Optional[Semver]:
    """Parse ``"X.Y.Z"`` into a comparable tuple, or ``None`` if it is not that."""
    parts = text.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def changelog_versions(root: Path) -> List[str]:
    """Every ``## [X.Y.Z]`` version heading in ``CHANGELOG.md`` (not Unreleased)."""
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return []
    return CHANGELOG_RE.findall(changelog.read_text(encoding="utf-8"))


def release_tags(root: Path) -> List[Tuple[Semver, str]]:
    """All ``vX.Y.Z`` tags, as ``(semver_tuple, tag_name)``, sorted ascending."""
    out = _git(root, "tag", "--list", "v*")
    if not out:
        return []
    found = []
    for name in out.splitlines():
        match = TAG_RE.match(name.strip())
        if match:
            found.append(((int(match[1]), int(match[2]), int(match[3])), name.strip()))
    return sorted(found)


def feat_commits_since(root: Path, tag: str) -> List[str]:
    """Conventional ``feat`` commit subjects on HEAD since ``tag``, newest first."""
    out = _git(root, "log", f"{tag}..HEAD", "--no-merges", "--format=%s")
    if not out:
        return []
    return [line for line in out.splitlines() if FEAT_RE.match(line)]


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _summary(lines: List[str]) -> None:
    """Append a short report to the GitHub Actions step summary, when present."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        pass


def main() -> int:
    """Check version/tag/changelog agreement; return a process exit code."""
    root = _find_root(Path(__file__).resolve().parent)

    problems: List[str] = []
    nudges: List[str] = []

    version = read_version(root)
    if version is None:
        problems.append("could not read __version__ from src/dlhub/__init__.py")
        return _report(problems, nudges)

    semver = parse_semver(version)
    if semver is None:
        problems.append(f"__version__ {version!r} is not an X.Y.Z release version")

    if version not in changelog_versions(root):
        problems.append(
            f"CHANGELOG.md has no '## [{version}]' section for the declared version"
        )

    tags = release_tags(root)
    if not tags:
        nudges.append(
            "no release tags (vX.Y.Z) found yet -- bootstrap the version history "
            "by tagging the releases"
        )
    elif semver is not None:
        latest_semver, latest_tag = tags[-1]
        if semver < latest_semver:
            problems.append(
                f"__version__ {version} is behind the latest tag {latest_tag} -- "
                "the version must not regress"
            )
        elif semver > latest_semver:
            nudges.append(
                f"__version__ {version} is ahead of the latest tag {latest_tag} -- a "
                f"release is staged; tag the merge commit v{version} and push it"
            )
        else:
            feats = feat_commits_since(root, latest_tag)
            if feats:
                shown = "\n    ".join(feats[:10])
                more = "" if len(feats) <= 10 else f"\n    ... and {len(feats) - 10} more"
                nudges.append(
                    f"{len(feats)} feat commit(s) since {latest_tag} with no version "
                    f"bump -- a minor release may be due:\n    {shown}{more}"
                )

    return _report(problems, nudges)


def _report(problems: List[str], nudges: List[str]) -> int:
    """Emit nudges and problems at the right severity; return the exit code."""
    summary = ["### Release hygiene"]

    for nudge in nudges:
        if _in_actions():
            print(f"::warning title=Release hygiene::{nudge}")
        else:
            print(f"nudge: {nudge}", file=sys.stderr)
        summary.append(f"- ⚠️ {nudge}")

    for problem in problems:
        if _in_actions():
            print(f"::error title=Release hygiene::{problem}")
        else:
            print(f"problem: {problem}", file=sys.stderr)
        summary.append(f"- ❌ {problem}")

    if problems:
        print(f"release check: {len(problems)} problem(s)")
    else:
        tail = f"  [{len(nudges)} nudge(s)]" if nudges else ""
        print(f"release check: ok{tail}")

    _summary(summary)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
