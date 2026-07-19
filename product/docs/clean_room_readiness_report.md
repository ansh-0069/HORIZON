# Clean Room Readiness Report

**Audit scope:** Hackathon evaluator path only: `run.sh`, `src/`, `pickle/model.pkl`, `pickle/model_manifest.json`, `requirements.txt`, and supplied campaign CSVs. The optional planner UI and all `product/app/` code are outside this scope.

## Current status

**Status: Passed final post-v5 clean-room verification (2026-07-19).**

The repository now seals the following evaluator artifact in `pickle/model_manifest.json`:

| Field | Sealed value |
| --- | --- |
| Artifact SHA-256 | `7f88607a703d529564432a0fa43c99cf2b9669b93948fef67ffffcf5795d83fa` |
| Model version | `horizon-direct-ridge-v5-oof-factor-copula` |
| Artifact build runtime | `3.12` |
| Minimum supported runtime | `3.11` |
| Training-data fingerprint | `e2778df8a45675c6d182173d5400e98339ca6cbf3f8554292faa27dff20a01f4` |
| Feature-schema fingerprint | `f51e8603f9cc1710fd9f7c873163912b52f399afdda92c33a1aa0fce5d520a32` |

Earlier rehearsal evidence used the v4 artifact and is intentionally not used as evidence for v5. The final v5 rehearsal was rerun through Git Bash against an isolated copy of `./data`, the sealed pickle, and its copied sibling manifest. The persisted evidence is `product/models/submission_rehearsal.json`.

**Recorded result:** success; `162` rows; all locked `horizon-v1` columns; horizons `30`, `60`, and `90`; model SHA unchanged; `model_manifest_enforced=true`; only NumPy/Pandas imports; no network imports or prediction-time training calls. The root sample data emitted only the documented unreviewed metadata warnings. The same sealed artifact also produced a byte-identical CSV under local Python 3.11 and 3.12 runs (`SHA-256 17940DAB2FE35A79E7904770C21B29DF61098057D54147DF5CD193D4E6CA8324`); the adapter fixes numeric text at six decimal places and LF endings for portable serialization.

## Static protected-path evidence

The protected path satisfies the evaluator contract as follows; the recorded rehearsal verifies these behaviors end to end.

| Requirement | Protected-path implementation |
| --- | --- |
| `./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH` | `run.sh` resolves defaults/arguments, selects a local Python executable, and invokes `python -m src.predict`. |
| Inference only | `src.predict` loads the sealed `HorizonModel`; no protected-path training entry point is invoked. |
| No API key or external service | `run.sh` and `src/` use local files, Python standard library, NumPy, and Pandas only. |
| No runtime download | Package installation is separate from execution; the runner does not install packages or access a network. |
| Artifact integrity | The manifest pins SHA-256, model version, and model provenance; mismatch fails before inference. |
| Output generation | `src/output_adapter.py` is the sole writer and atomically writes a non-empty UTF-8 `predictions.csv` only after schema validation. |
| Meaningful failures | Expected input, artifact, validation, and output errors are surfaced as non-zero `ERROR: offline prediction failed: ...` results. |

## Recorded final post-v5 verification

Run these checks from a fresh checkout after dependencies have been installed from the pinned requirements (or a pre-provisioned offline wheelhouse):

```bash
python -m unittest discover -s product/tests -q
python -m product.scripts.release_check --strict
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
python -m product.scripts.verify_evaluator_contract \
  --predictions ./output/predictions.csv
```

Then run an isolated Git-Bash or Linux rehearsal against a temporary copy of the exact submitted data and sealed artifact:

```powershell
python -m product.scripts.rehearse_submission `
  --data-dir .\data `
  --model .\pickle\model.pkl `
  --bash "C:\Program Files\Git\bin\bash.exe" `
  --temporary-root .\product\output
```

The recorded command exited successfully. Its report records the output row count, horizons, exact output header, model immutability, manifest enforcement, dependency audit, and quality warnings. The hash remained `7f88607a703d529564432a0fa43c99cf2b9669b93948fef67ffffcf5795d83fa` before and after the rehearsal.

## Remaining release risks

1. **Organizer scorer schema remains unpublished in the supplied guide.** `horizon-v1` is locked locally through `src/output_adapter.py` and its fixture. If organizers publish a different required header, row granularity, or field semantics, adapt the centralized adapter and rerun every final gate.

## Known environment and data warnings

1. **Python runtime:** the pinned NumPy/Pandas versions require Python 3.11+.
2. **Bash runtime:** `run.sh` assumes a POSIX shell with standard `mkdir` and `dirname`, consistent with the stated evaluator command. Windows rehearsal needs Git Bash or WSL; Linux evaluation does not need either product tool.
3. **Input quality:** absent/unreviewed semantic metadata, unreviewed Meta taxonomy mappings, and missing configured-budget rows are surfaced as warnings; corrupted fields, blank IDs, invalid numeric values, or duplicate campaign-day rows fail closed. Microsoft `CampaignType` presentation variants are deterministically normalized (`Audience` to `DISPLAY`, `PerformanceMax` to `PERFORMANCE_MAX`, `Search` to `SEARCH`, and `Shopping` to `SHOPPING`) before model feature lookup; unrecognized non-blank types remain disclosed upper-snake labels.
4. **Forecast use:** v5 includes empirical OOF residual calibration and deterministic factor-copula aggregation. It is not conformal prediction and must not be represented as a guaranteed confidence interval.

## Recorded sign-off evidence

The following are recorded for the sealed v5 artifact:

- `release_check --strict` succeeds;
- the exact `run.sh` command succeeds from a clean temporary working copy;
- `predictions.csv` is non-empty, valid against the locked adapter contract, and includes 30/60/90-day outputs;
- the model SHA-256 is unchanged before and after inference;
- negative-path checks return actionable errors without leaving a partial output; and
- the exact repository revision must still be pushed and referenced for submission.
