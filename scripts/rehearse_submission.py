"""Clean-room rehearsal for the hackathon's offline evaluator contract.

This is not used by ``run.sh``. It copies the supplied data and model into a
temporary directory, executes the same runner that organizers invoke, then
validates the emitted CSV and verifies the model artifact was not modified.
"""
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner_command(runner: Path, data_dir: Path, model_path: Path, output_path: Path, bash: str | None) -> list[str]:
    if os.name == "nt":
        if not bash:
            raise RuntimeError("On Windows, pass --bash with a Bash executable to rehearse ./run.sh")
        return [bash, "-lc", f'cd "{ROOT.as_posix()}" && ./run.sh "{data_dir.as_posix()}" "{model_path.as_posix()}" "{output_path.as_posix()}"']
    return [str(runner), str(data_dir), str(model_path), str(output_path)]


def rehearse(data_dir: Path, model_path: Path, runner: Path, bash: str | None) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    with tempfile.TemporaryDirectory(prefix="horizon-evaluator-") as temporary:
        root = Path(temporary)
        evaluator_data = root / "data"
        evaluator_model = root / "model.pkl"
        output_path = root / "predictions.csv"
        shutil.copytree(data_dir, evaluator_data)
        shutil.copy2(model_path, evaluator_model)
        model_before = _sha256(evaluator_model)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHON_BIN": sys.executable}
        completed = subprocess.run(
            _runner_command(runner, evaluator_data, evaluator_model, output_path, bash),
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
        return {
            "status": "passed",
            "rows": int(len(output)),
            "columns": list(output.columns),
            "horizons": sorted(int(value) for value in output["horizon_days"].unique()),
            "model_unchanged": True,
            "runner_stdout": completed.stdout.strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse Horizon in an evaluator-like temporary workspace")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "pickle" / "model.pkl")
    parser.add_argument("--runner", type=Path, default=ROOT / "run.sh")
    parser.add_argument("--bash", help="Bash executable; required for a Windows rehearsal")
    parser.add_argument("--report", type=Path, default=ROOT / "models" / "submission_rehearsal.json")
    args = parser.parse_args()
    report = rehearse(args.data_dir, args.model, args.runner, args.bash)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"], "horizons": report["horizons"]}))


if __name__ == "__main__":
    main()
