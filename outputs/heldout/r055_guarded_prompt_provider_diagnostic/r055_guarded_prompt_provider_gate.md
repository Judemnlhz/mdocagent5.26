# R055 Guarded Prompt Provider Gate

Decision: `r055_guarded_prompt_provider_gate_pass`
Gate passed: True

## Boundary
- Records 384, 508, and 569 only.
- Provider diagnostic on R054 guarded prompt previews only.
- Tests refusal/noise-avoidance behavior only.
- Does not prove artifact positive lift, retrieval improvement, full QA, or official score.

## Checks
- `target_records_exactly_384_508_569`: True
- `provider_predictions_exactly_3`: True
- `prompt_hashes_match_r054`: True
- `uses_r054_zero_artifact_guarded_prompts`: True
- `prediction_records_have_no_gold_fields`: True
- `all_guard_behaviors_passed`: True
- `scope_limited_to_guard_prompt_refusal_noise_avoidance`: True
- `does_not_claim_artifact_positive_lift`: True
- `not_full_qa`: True
- `not_official_score`: True
