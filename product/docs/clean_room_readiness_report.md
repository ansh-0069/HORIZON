# Clean Room Readiness Report

**Audit date:** 2026-07-18
**Scope:** Hackathon evaluator path only: `run.sh`, `src/`, `pickle/model.pkl`, `requirements.txt`, and supplied campaign CSVs. The optional planner UI and `product/app/evidence.py` are explicitly outside this scope.

## Result

**Submission Readiness Score: 90/100**

The repository is ready for an offline evaluator rehearsal. There are no code-level blockers in the evaluator path. The outstanding score deduction is for the organizer's final `predictions.csv` scorer schema, which is not specified in the supplied submission guide and therefore cannot yet be validated exactly.

**Latest local verification (2026-07-18):** `product.scripts.release_check --strict` passed, and the exact Git-Bash rehearsal produced `156` rows spanning the `30`, `60`, and `90` day horizons with the `horizon-direct-ridge-v2` artifact. The model is loaded read-only and its SHA-256 is verified unchanged by the rehearsal.

## Evidence

Command executed against an isolated temporary copy of the supplied data and `pickle/model.pkl`:

```powershell
python -m product.scripts.rehearse_submission `
  --data-dir .\product\supplied_data `
  --model .\pickle\model.pkl `
  --bash "C:\Program Files\Git\bin\bash.exe" `
  --temporary-root .\product\output
```

Observed result:

- Exit status: success
- Output rows: `156`
- Horizons: `30`, `60`, `90`
- Output: one `overall` record per horizon plus campaign, campaign-type, and channel records
- Model artifact SHA-256 before and after inference: unchanged
- Evaluator dependencies: `numpy`, `pandas`, both pinned in `requirements.txt`
- Network-client imports in `src/`: none
- Training calls in `src/predict.py`: none

## ✅ Passes

| Requirement | Result | Evidence |
|---|---|---|
| `./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH` | Pass | Rehearsal invokes the exact runner from a temporary data/model copy and produces `predictions.csv`. |
| No training at evaluation time | Pass | `src.predict` loads `pickle/model.pkl`; its import/call audit finds no `.fit(...)` call and the copied model hash is unchanged. |
| No OpenAI or external API calls | Pass | `run.sh` imports only `src.predict`; `src/` contains no OpenAI, HTTP, socket, request, or URL client imports. Optional `app/` code is not imported. |
| No network required | Pass | The evaluator path is file-based and only imports standard-library modules plus NumPy/Pandas. No credentials are read. |
| No hidden Python dependencies | Pass | Static dependency audit found only `numpy` and `pandas`, both pinned in `requirements.txt`. |
| `predictions.csv` is generated | Pass | The verified rehearsal generated 156 valid rows. Output writing is atomic, so an existing valid file is not replaced by a partial write. |
| Failures are meaningful | Pass | Missing data directory, missing model, and empty/corrupt pickle each exit with status `2`, print an `ERROR: offline prediction failed: ...` message, and leave no output CSV. |

## ⚠ Warnings

1. **Final scorer schema is not in the supplied guide.** The current output contains 21 documented columns. When organizers publish exact required column names, hierarchy expectations, IDs, or row granularity, update only `src/contracts.py` and `src/output_adapter.py`, then rerun this rehearsal. This is the main remaining submission risk.
2. **Python runtime must be 3.11+.** This is documented in `README.md`. The evaluator must install the pinned requirements under a compatible Python runtime before being taken offline.
3. **Source-data quality warnings are intentional.** The supplied files emit warnings for 21 missing configured-budget rows and Meta conversion semantics/taxonomy review. These do not stop output generation, but they should be transparently mentioned in the demo.
4. **Bash is part of the stated evaluator contract.** `run.sh` assumes a POSIX shell with standard `mkdir` and `dirname`, consistent with the guide's `./run.sh` command. Windows rehearsal needs Git Bash or WSL; this is not a runtime dependency for a Linux evaluator.

## ❌ Blocking Issues

**None in the current offline evaluator path.**

Do not submit until the final organizer output schema is checked against the adapter. If that schema differs from the current 21-column format, it becomes a blocking submission task, not a modeling task.

## Fixes Applied During This Audit

| Finding | Affected files | Production-quality resolution |
|---|---|---|
| Blank `quality_flags` values became null when an evaluator re-read CSV output. | `src/output_adapter.py` | Optional text fields now serialize as explicit sentinels such as `none`, preventing ambiguous empty CSV cells. |
| Prediction failures produced raw tracebacks and output writes were not fully cleanup-safe. | `src/predict.py` | Added expected-error handling with exit code `2` and `ERROR:` messages; added atomic write/replace with temporary-file cleanup. |
| Windows clean-room rehearsal could not pass temporary paths through Git Bash. | `product/scripts/rehearse_submission.py` | Added safe path conversion, a writable temporary-root option, model hash verification, output contract validation, and static dependency/training/network audits. |

## Release Gate

Before publishing the submission repository:

1. Install `requirements.txt` on the evaluator's Python 3.11+ runtime while internet access is available.
2. Run `python -m unittest discover -s product/tests -q`.
3. Run `product.scripts.rehearse_submission` against the exact model and sample data being submitted.
4. Replace or adapt only the centralized output adapter once the organizer releases the final CSV contract.
5. Run the rehearsal again and attach `product/models/submission_rehearsal.json` to internal release notes.
