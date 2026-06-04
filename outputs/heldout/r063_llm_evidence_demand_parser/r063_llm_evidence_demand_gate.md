# R063 LLM Evidence-Demand Parser Gate

Decision: `r063_llm_evidence_demand_parser_gate_pass`
Gate passed: True

## Boundary
- Qwen3-VL-8B-Instruct is used only as a question-only evidence-demand parser.
- The LLM does not answer questions and does not select artifacts directly.
- Deterministic guarded selector still performs artifact scoring and selection.
- No prediction, no evaluation, no full QA, no official score, and no artifact-lift claim.

## Checks
- `target_records_match_requested_small_set`: True
- `provider_outputs_exactly_target_count`: True
- `all_parser_outputs_parseable`: True
- `parser_inputs_question_only`: True
- `parser_outputs_have_no_gold_fields`: True
- `selector_previews_have_no_gold_fields`: True
- `deterministic_selector_still_used`: True
- `llm_does_not_select_artifacts_directly`: True
- `scope_limited_to_evidence_demand_parser_diagnostic`: True
- `no_answer_generation`: True
- `no_prediction_or_eval`: True
- `no_full_qa`: True
- `not_official_score`: True
- `does_not_claim_artifact_lift`: True
- `artifact_store_bound_to_r038d_union_atomic_store`: True
