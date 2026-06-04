# R065 Parser Code-Type Gate

Decision: `r065_parser_code_type_gate_pass`
Gate passed: True

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Regression for parser post-normalization only.
- Checks 508 code/table routing and 384 metadata control.

## Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `no_full_qa`: True
- `target_records_exactly_508_384`: True
- `record_508_passed`: True
- `record_384_control_passed`: True
- `no_gold_fields_in_outputs`: True
- `does_not_claim_artifact_lift`: True
- `not_official_score`: True
- `artifact_store_bound_to_r038d_union_atomic_store`: True
