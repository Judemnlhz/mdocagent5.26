# R053 Question-Aware Artifact Selection Scaffold

Decision: `r053_question_aware_scaffold_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Uses R045 cases to design a question-aware artifact selector and citation prompt preview.
- Not an official score.

## Summary
- cases: 8
- rubric labels: `{"all_conditions_miss_requires_error_analysis": 5, "artifact_injection_introduces_false_positive_risk": 1, "diagnostic_gold_match_undercounts_supported_refusal": 1, "rerank_and_artifact_context_help_unanswerable_refusal": 1}`
- transition labels: `{"all_conditions_miss": 5, "artifact_injection_gain_vs_original": 1, "artifact_injection_loss_vs_original": 1, "page_rerank_gain_vs_original": 2, "snippet_only_sufficient": 1}`
- selection reasons: `{"artifact_reranked_page": 60, "atomic_artifact_priority": 60, "metric_label_overlap": 30, "numeric_table_type_priority": 60, "original_candidate_page": 60, "question_token_overlap": 45, "value_match:18": 24, "value_match:2002": 8, "value_match:4": 8, "value_match:65": 5, "value_match:8": 16}`

## Recommended Next
- Manually inspect R053 prompt previews for records 384, 508, and 569 before any provider run.
- If selected artifacts still lack necessary evidence, improve artifact selection with stronger question/value/metric matching.
- If selected artifacts contain evidence but prompts remain ambiguous, run a small prompt-template diagnostic with citation-required output only.
- Do not run full QA until this scaffold is manually accepted and recorded in the tracker.
