# R064 Parser/Artifact Mismatch Gate

Decision: `r064_parser_artifact_mismatch_gate_pass`
Gate passed: True

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Reads R063 parser/selector outputs and public artifact/page context only.
- Attributes mismatch causes; does not report a score or artifact-lift claim.

## Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `no_full_qa`: True
- `target_records_match_r063_small_set`: True
- `r063_gate_was_passed`: True
- `all_records_have_root_cause`: True
- `all_records_have_coverage_matrix`: True
- `no_gold_fields_in_audits`: True
- `does_not_claim_artifact_lift`: True
- `not_official_score`: True
- `artifact_store_bound_to_r038d_union_atomic_store`: True
