# R044 Small Contrastive Execution Gate

Decision: `r044_small_contrastive_gate_pass`
Gate passed: True

## Boundary
- 22 R042 focus records only.
- Four R043 conditions only.
- Diagnostic attribution only; not full QA and not an official score.

## Checks
- `r043_condition_mapping_confirmed`: True
- `target_records_exactly_22`: True
- `four_conditions_present`: True
- `predictions_complete_4x22`: True
- `prompt_hashes_match_r043`: True
- `prediction_records_have_no_gold_fields`: True
- `not_full_qa`: True
- `not_official_score`: True

## R043 Mapping Confirmation
- `original_pages_only`: expected retrieval `top4_original_only`, expected page_text=True, expected artifacts=False; mapping checks retrieval=True, page_text=True, artifacts=True, focus_rows=22
- `page_rerank_only`: expected retrieval `top4_artifact_only`, expected page_text=True, expected artifacts=False; mapping checks retrieval=True, page_text=True, artifacts=True, focus_rows=22
- `original_pages_plus_artifact_snippets`: expected retrieval `top4_original_only`, expected page_text=True, expected artifacts=True; mapping checks retrieval=True, page_text=True, artifacts=True, focus_rows=22
- `artifact_snippets_only`: expected retrieval `top4_artifact_only`, expected page_text=False, expected artifacts=True; mapping checks retrieval=True, page_text=True, artifacts=True, focus_rows=22
