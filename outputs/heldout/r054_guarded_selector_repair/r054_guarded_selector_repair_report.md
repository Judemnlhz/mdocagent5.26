# R054 Guarded Selector Repair

Decision: `r054_guarded_selector_repair_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Converts manual R053 feedback for records 384, 508, and 569 into hard selector/prompt guards.
- Not an official score.

## Summary
- cases: 8
- rubric labels: `{"all_conditions_miss_requires_error_analysis": 5, "artifact_injection_introduces_false_positive_risk": 1, "diagnostic_gold_match_undercounts_supported_refusal": 1, "rerank_and_artifact_context_help_unanswerable_refusal": 1}`
- guard decisions: `{"document_metadata_refusal_guard": 1, "exact_code_absence_guard": 1, "no_relevant_artifact_guard": 1, "operand_completeness_guard": 1, "token_key_value_selection": 4}`
- guard reasons: `{"document_metadata_lookup_uses_page_text_not_numeric_artifacts": 1, "missing_operands:children": 1, "no_artifact_contains_exact_question_code": 1, "no_question_overlapping_artifacts": 1, "numeric_artifacts_rejected_without_exact_code_key": 1, "operand_completeness_failed": 1, "selected_question_overlapping_artifacts": 4, "visible_page_metadata_present": 1}`

## Manual Repair Summary
- 384: document metadata/refusal route; numeric/table artifacts rejected
- 508: exact AR03 key-value selection required; otherwise use page evidence for unsupported/refusal
- 569: operand-completeness guard blocks calculation from partial snippets

## Recommended Next
- Do not run full QA from R054.
- Manually inspect R054 prompt previews for 384, 508, and 569 against the recorded guard decisions.
- If accepted, run only a tiny provider diagnostic on the guarded prompts; treat it as diagnostic attribution, not an official score.
- If rejected, repair selector guards again before any provider call.
