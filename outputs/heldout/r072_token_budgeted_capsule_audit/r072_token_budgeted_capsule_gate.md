# R072 Token-Budgeted Capsule Gate

Decision: `r072_token_budgeted_capsule_audit_complete`
Gate passed: True

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Deterministic capsule/token audit only.
- Not an official score.

## Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `no_full_qa`: True
- `not_official_score`: True
- `records_scanned_positive`: True
- `capsule_with_guard_mean_lower_than_raw`: True
- `capsule_plain_mean_not_more_than_guarded`: True
- `capsule_with_guard_mean_ratio_below_one`: True
- `capsule_lower_than_raw_for_majority`: True
- `no_gold_fields_in_public_outputs`: True
- `artifact_store_bound_to_r038d_union_atomic_store`: True
