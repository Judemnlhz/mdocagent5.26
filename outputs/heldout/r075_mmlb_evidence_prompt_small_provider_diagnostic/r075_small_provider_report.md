# R075 MMLB Evidence Prompt Small Provider Diagnostic

Decision: `r075_small_provider_diagnostic_complete`

## Boundary
- Small text-provider diagnostic only; not full MDocAgent multi-agent QA.
- Not full MMLB and not an official score.
- Uses existing top-4 baseline correctness only for sampled help/hurt comparison.

## Summary
- selected cases: 66
- provider predictions: 66
- evaluations: 66
- provider failures: 21 (0.318182)
- evaluation failures: 21 (0.318182)
- sample accuracy not official: 0.212121
- baseline sample accuracy reference not official: 0.560606
- changed_to_right_minus_wrong: -23

## Outcomes
- `changed_to_right`: 6
- `changed_to_wrong`: 29
- `kept_right`: 8
- `kept_wrong`: 23

## Selection Buckets
- `baseline_correct_no_selected_artifact_risk`: 29
- `baseline_correct_stable_candidate`: 8
- `baseline_wrong_capsule_supported_candidate`: 8
- `baseline_wrong_guarded_or_page_routed_candidate`: 21

## Recommended Next
- Provider stability is a blocker: 21 sampled rows timed out or failed and were conservatively scored as incorrect.
- Observed help-hurt delta is negative (-23); do not launch full MMLB QA from the current R074 prompt.
- First rerun a smaller balanced diagnostic with a stable provider or longer timeout, then reduce baseline-correct no-selected-artifact prompt intervention strength.
