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
- provider failures: 7 (0.106061)
- evaluation failures: 7 (0.106061)
- sample accuracy not official: 0.242424
- baseline sample accuracy reference not official: 0.560606
- changed_to_right_minus_wrong: -21
- paired original predictions/evaluations: 66/66
- paired original sample accuracy not official: 0.227273
- paired changed_to_right_minus_wrong: 1

## Outcomes
- `changed_to_right`: 4
- `changed_to_wrong`: 25
- `kept_right`: 12
- `kept_wrong`: 25

## Paired Original vs Evidence Outcomes
- `changed_to_right`: 2
- `changed_to_wrong`: 1
- `kept_right`: 14
- `kept_wrong`: 49

## Selection Buckets
- `baseline_correct_no_selected_artifact_risk`: 29
- `baseline_correct_stable_candidate`: 8
- `baseline_wrong_capsule_supported_candidate`: 8
- `baseline_wrong_guarded_or_page_routed_candidate`: 21

## Recommended Next
- Paired original-vs-evidence diagnostic is positive (1); inspect changed-to-wrong cases before expanding.
- Next run can expand the paired diagnostic under parallel_workers<=3, still not full MMLB QA.
