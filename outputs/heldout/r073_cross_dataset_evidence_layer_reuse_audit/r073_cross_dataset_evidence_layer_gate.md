# R073 Cross-Dataset Evidence Layer Gate

Decision: `r073_cross_dataset_reuse_audit_complete_with_input_gaps`
Gate passed: True

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Does not use answers, evidence pages, or official scoring.
- Missing retrieval/artifact bindings are reported as blocked inputs, not silently substituted.

## Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `no_full_qa`: True
- `not_official_score`: True
- `all_target_datasets_reported`: True
- `registry_contract_valid`: True
- `registry_has_no_dataset_specific_skill_names`: True
- `mmlb_full_reuse_audit_available`: True
- `mmlb_capsule_mean_ratio_below_one`: True
- `cross_dataset_question_skill_activation_recorded`: True
- `blocked_inputs_are_explicit_not_silent`: True
- `no_gold_fields_in_public_outputs`: True

## Blocked Inputs
- FETA
- LDU
- PTAB
- PTEXT
