"""Clean-room rehearsal for the hackathon's offline evaluator contract.

This is not used by ``run.sh``. It copies the supplied data and model into a
temporary directory, executes the same runner that organizers invoke, then
validates the emitted CSV and verifies the model artifact was not modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd

from src.output_adapter import validate_submission_schema


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"
ALLOWED_THIRD_PARTY = {"numpy", "pandas"}
NETWORK_MODULES = {"http", "httpx", "requests", "socket", "urllib", "websocket", "openai", "aiohttp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_evaluator_dependencies(source_root: Path = ROOT / "src", requirements_path: Path = ROOT / "requirements.txt") -> dict[str, Any]:
    """Statically verify the evaluator import surface is offline and declared."""
    standard = set(sys.stdlib_module_names) | {"__future__"}
    imported: dict[str, list[str]] = {}
    network_imports: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".", 1)[0])
        imported[path.name] = sorted(modules)
        network_imports.extend(f"{path.name}:{module}" for module in modules if module in NETWORK_MODULES)
    third_party = sorted({module for modules in imported.values() for module in modules if module not in standard and module != "src"})
    declared = {
        line.split("==", 1)[0].strip().lower()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    undeclared = sorted(set(third_party) - declared)
    if network_imports:
        raise RuntimeError(f"Offline audit found network imports in evaluator source: {network_imports}")
    if undeclared:
        raise RuntimeError(f"Dependency audit found undeclared evaluator packages: {undeclared}")
    predict_source = (source_root / "predict.py").read_text(encoding="utf-8")
    if ".fit(" in predict_source or "src.train" in predict_source:
        raise RuntimeError("Inference entry point appears to invoke training")
    return {
        "third_party_packages": third_party,
        "all_declared_in_requirements": True,
        "network_imports": [],
        "training_calls_in_predict": False,
    }


def _bash_path(bash: str, path: Path) -> str:
    """Convert a Windows path to a Git-Bash path without depending on shell interpolation."""
    converted = subprocess.run(
        [bash, "-lc", 'cygpath -u "$1"', "horizon-rehearsal", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if converted.returncode != 0 or not converted.stdout.strip():
        raise RuntimeError(f"Unable to convert Windows path for Bash: {path}")
    return converted.stdout.strip()


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _runner_command(runner: Path, data_dir: Path, model_path: Path, output_path: Path, bash: str | None) -> tuple[list[str], str | None]:
    if os.name == "nt":
        if not bash:
            raise RuntimeError("On Windows, pass --bash with a Bash executable to rehearse ./run.sh")
        root_path = _bash_path(bash, ROOT)
        command = "cd {root} && ./run.sh {data} {model} {output}".format(
            root=_shell_quote(root_path),
            data=_shell_quote(_bash_path(bash, data_dir)),
            model=_shell_quote(_bash_path(bash, model_path)),
            output=_shell_quote(_bash_path(bash, output_path)),
        )
        return [bash, "-lc", command], _bash_path(bash, Path(sys.executable))
    return [str(runner), str(data_dir), str(model_path), str(output_path)], sys.executable


def rehearse(data_dir: Path, model_path: Path, runner: Path, bash: str | None, temporary_root: Path | None = None) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    dependency_audit = audit_evaluator_dependencies()
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="horizon-evaluator-", dir=temporary_root) as temporary:
        root = Path(temporary)
        evaluator_data = root / "data"
        evaluator_model = root / "model.pkl"
        output_path = root / "predictions.csv"
        shutil.copytree(data_dir, evaluator_data)
        shutil.copy2(model_path, evaluator_model)
        model_before = _sha256(evaluator_model)
        command, python_bin = _runner_command(runner, evaluator_data, evaluator_model, output_path, bash)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHON_BIN": python_bin or sys.executable}
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Evaluator rehearsal failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}")
        if not output_path.is_file():
            raise RuntimeError("Evaluator rehearsal completed without predictions.csv")
        output = pd.read_csv(output_path)
        validate_submission_schema(output)
        model_after = _sha256(evaluator_model)
        if model_after != model_before:
            raise RuntimeError("Runner modified the pre-trained model artifact")
        # A persisted rehearsal report is release evidence, not a record of a
        # machine-specific temporary directory. Keep it comparable across runs.
        runner_messages = [
            line.replace(str(root), "<isolated-workspace>")
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        return {
            "status": "passed",
            "rows": int(len(output)),
            "columns": list(output.columns),
            "horizons": sorted(int(value) for value in output["horizon_days"].unique()),
            "model_unchanged": True,
            "dependency_audit": dependency_audit,
            "runner_messages": runner_messages,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse Horizon in an evaluator-like temporary workspace")
    parser.add_argument("--data-dir", type=Path, default=PRODUCT_ROOT / "demo_data")
    parser.add_argument("--model", type=Path, default=ROOT / "pickle" / "model.pkl")
    parser.add_argument("--runner", type=Path, default=ROOT / "run.sh")
    parser.add_argument("--bash", help="Bash executable; required for a Windows rehearsal")
    parser.add_argument("--temporary-root", type=Path, help="Optional writable location for the isolated evaluator copy")
    parser.add_argument("--report", type=Path, default=PRODUCT_ROOT / "models" / "submission_rehearsal.json")
    args = parser.parse_args()
    report = rehearse(args.data_dir, args.model, args.runner, args.bash, args.temporary_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "rows": report["rows"], "horizons": report["horizons"]}))


if __name__ == "__main__":
    main()
