"""Fail fast on submission-guide regressions before publishing the repository.

This tool is intentionally outside the evaluator path. It checks the repository
layout and shell contract that a unit test cannot reliably exercise on Windows.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import stat
import subprocess
import sys

from src.output_adapter import FORECAST_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ROOT_FILES = ("run.sh", "requirements.txt", "README.md", ".python-version", "pickle/model.pkl")
REQUIRED_DATA_FILES = (
    "google_ads_campaign_stats.csv",
    "bing_campaign_stats.csv",
    "meta_ads_campaign_stats.csv",
)
NETWORK_MODULES = {"http", "httpx", "requests", "socket", "urllib", "websocket", "openai", "aiohttp"}


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _git_mode(path: str) -> str | None:
    completed = subprocess.run(
        ["git", "ls-files", "--stage", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return completed.stdout.split(maxsplit=1)[0]


def _protected_imports() -> set[str]:
    modules: set[str] = set()
    for path in (ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".", 1)[0])
    return modules


def check() -> list[str]:
    failures: list[str] = []
    for relative in REQUIRED_ROOT_FILES:
        _check((ROOT / relative).is_file(), f"Missing required root file: {relative}", failures)
    for filename in REQUIRED_DATA_FILES:
        _check((ROOT / "data" / filename).is_file(), f"Missing required sample data file: data/{filename}", failures)

    runner = (ROOT / "run.sh")
    if runner.is_file():
        source = runner.read_text(encoding="utf-8")
        for expected in ('DATA_DIR="${1:-./data}"', 'MODEL_PATH="${2:-./pickle/model.pkl}"', 'OUTPUT_PATH="${3:-./output/predictions.csv}"'):
            _check(expected in source, f"run.sh lacks required default: {expected}", failures)
        _check("set -euo pipefail" in source, "run.sh must fail loudly with set -euo pipefail", failures)
        git_mode = _git_mode("run.sh")
        executable = bool(runner.stat().st_mode & stat.S_IXUSR) or git_mode == "100755"
        _check(executable, "run.sh is not executable in filesystem or Git index", failures)

    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip() if (ROOT / ".python-version").is_file() else ""
    _check(re.fullmatch(r"3\.1[1-9](?:\.\d+)?", python_version) is not None, ".python-version must declare Python 3.11+", failures)
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    _check("numpy==" in requirements and "pandas==" in requirements, "requirements.txt must pin NumPy and Pandas", failures)

    imported = _protected_imports()
    forbidden = sorted(imported & NETWORK_MODULES)
    _check(not forbidden, f"Protected src imports network modules: {forbidden}", failures)
    _check("product" not in imported, "Protected src imports the optional product layer", failures)

    fixture = (ROOT / "product" / "tests" / "fixtures" / "horizon_v1_header.csv")
    _check(fixture.is_file(), "Missing versioned output-header fixture", failures)
    if fixture.is_file():
        _check(fixture.read_text(encoding="utf-8").strip().split(",") == FORECAST_COLUMNS, "Output-header fixture differs from adapter contract", failures)
    return failures


def upstream_sync_failure() -> str | None:
    """Return a release-blocking message when local HEAD is not publishable.

    The organizer clones the submitted remote URL, so a clean working tree is
    not sufficient: the branch must also contain the reviewed commits.
    """
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if upstream.returncode != 0:
        return "No upstream branch is configured; cannot verify that the submitted repository contains HEAD"
    ahead = subprocess.run(
        ["git", "rev-list", "--count", "@{u}..HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ahead.returncode != 0:
        return "Unable to compare local HEAD with its upstream branch"
    count = int(ahead.stdout.strip() or "0")
    if count:
        return f"Local HEAD is {count} commit(s) ahead of {upstream.stdout.strip()}; push before submitting"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate repository-controlled submission-guide requirements")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero status when any release gate fails")
    parser.add_argument("--require-upstream-sync", action="store_true", help="Fail when local HEAD has not been pushed to its configured upstream")
    args = parser.parse_args()
    failures = check()
    if args.require_upstream_sync:
        sync_failure = upstream_sync_failure()
        if sync_failure:
            failures.append(sync_failure)
    if failures:
        print("Release check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        if args.strict:
            raise SystemExit(1)
        return
    print("Release check passed: evaluator layout, defaults, executable mode, dependency pins, protected imports, and output fixture are valid.")


if __name__ == "__main__":
    main()
