# Heldout Diagnostic Workspace

This repository keeps the R041-R045 heldout work as diagnostic audit material only. None of these runs are full QA, full-data generalization, or official MMLongBench scores.

## Retained Results

- `outputs/heldout/r041_r040_identical_score_attribution/`: final R041 aggregate/record-level attribution reports.
- `outputs/heldout/r042_r040_manual_attribution/`: final R042 manual attribution reports and cases.
- `outputs/heldout/r043_contrastive_prompt_exposure/`: R043 gate, manifest, and compact prompt exposure index. The large prompt preview JSONL files are intentionally not retained; rerun R043 before rerunning R044 provider calls.
- `outputs/heldout/r044_small_contrastive_provider/`: final 22-record provider diagnostic report, gate, and predictions.
- `outputs/heldout/r045_support_rubric/`: post-hoc support/citation-aware rubric report and cases.

## Retained Source

- `mdocnexus/` is core library and test code and is not treated as temporary experiment output.
- `scripts/run_heldout_diagnostic_audits.py` is the unified entrypoint for R043-R045.
- `scripts/run_r043_contrastive_prompt_exposure.py`, `scripts/run_r044_small_contrastive_provider.py`, and `scripts/run_r045_support_rubric.py` remain the focused reproducible runners.

Example reruns:

```bash
python3 scripts/run_heldout_diagnostic_audits.py r043 -- --execute
python3 scripts/run_heldout_diagnostic_audits.py r044 -- --execute --model deepseek-ai/DeepSeek-V3 --provider-note "Qwen/Qwen3-8B timed out in prior R044 run"
python3 scripts/run_heldout_diagnostic_audits.py r045 -- --execute
```

## Cleaned Material

- Retained `refine-logs/EXPERIMENT_PLAN.md` and `refine-logs/EXPERIMENT_TRACKER.md` for the experiment-plan workflow. Removed only transient `refine-logs/*.log` run logs.
- Removed Hydra date-run logs under `outputs/2026-*`; they are run logs, not final diagnostic artifacts.
- Removed R043 large prompt preview JSONLs after retaining the compact exposure index and hashes.
- Removed R041/R042 one-off scripts after retaining their final reports.
- Removed local Python bytecode caches.

## Protected Results

The original-paper reproduction results are intentionally preserved:

- `results/MMLongBench/mmlb-MDocAgent/`
- `results/MMLongBench/mmlb-MDocAgent-top4/`

## Current Diagnostic Conclusion

R045 supersedes the simple R044 matcher for transition-case interpretation: record 569 was undercounted as a supported refusal, record 384 shows artifact snippets can introduce unsupported false-positive risk, and record 508 remains the clearest positive diagnostic for refusing AR03. The next useful step is to improve refusal/support rubric and question-aware artifact selection before any full-data run.
