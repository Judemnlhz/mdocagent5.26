# R075 MMLB Evidence Prompt Small Provider Diagnostic

Decision: `r075_small_provider_diagnostic_complete`

## Boundary
- Small text-provider diagnostic only; not full MDocAgent multi-agent QA.
- Not full MMLB and not an official score.
- Uses existing top-4 baseline correctness only for sampled help/hurt comparison.

## Summary
- selected cases: 20
- provider predictions: 20
- evaluations: 20
- provider failures: 0 (0.0)
- evaluation failures: 0 (0.0)
- sample accuracy not official: 0.15
- baseline sample accuracy reference not official: 0.6
- changed_to_right_minus_wrong: -9
- paired original predictions/evaluations: 20/20
- paired original sample accuracy not official: 0.1
- paired changed_to_right_minus_wrong: 1

## Outcomes
- `changed_to_wrong`: 9
- `kept_right`: 3
- `kept_wrong`: 8

## Paired Original vs Evidence Outcomes
- `changed_to_right`: 1
- `kept_right`: 2
- `kept_wrong`: 17

## Selection Buckets
- `baseline_correct_no_selected_artifact_risk`: 8
- `baseline_correct_stable_candidate`: 4
- `baseline_wrong_capsule_supported_candidate`: 8

## Recommended Next
- Paired original-vs-evidence diagnostic is positive (1); inspect changed-to-wrong cases before expanding.
- Next run can expand the paired diagnostic under parallel_workers<=3, still not full MMLB QA.
