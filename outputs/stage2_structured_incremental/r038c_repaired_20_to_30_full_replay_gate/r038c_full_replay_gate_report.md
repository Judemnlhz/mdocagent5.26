# R038c Repaired 20 -> 30 Full Replay Gate

Decision: `stop_before_heldout_review_replay_gate`

## Scope
- Full replay of the original R028 20 -> 30 ten-page delta.
- Temporary cumulative20 plus repaired-delta store for diagnostic activation only.
- No final artifact-store merge, QA, graph, effectiveness claim, or rerank tuning.
- Model key remains environment-only; no config or key file is written by this runner.

## Provider Quality
- Provider success/fail: 10 / 0
- JSON parse success: 10
- Parse failures: 0
- Valid/discarded artifacts: 155 / 4

## Delta Artifact Quality
- Total artifacts: 155
- Atomic strong eligible: 135
- Type counts: `{"caption": 3, "figure": 3, "numeric_fact": 68, "section_header": 1, "table": 2, "table_cell": 67, "text_span": 11}`
- Quality counts: `{"atomic_numeric_ok": 135, "schema_valid_but_semantically_weak": 1, "weak_locator": 1}`
- R038a expected artifacts: 134
- Artifact growth vs R038a: 1.156716

## Temporary Activation
- Activated records: 19
- Eligible for held-out: 19
- Changed records, artifact_only: 29
- Changed records, original_plus_artifact: 15
- Held-out available: `False`
- Max doc share: 0.3158
- Max page share: 0.2105

## Checks
- `provider_success_all_pages`: True
- `parse_failure_zero`: True
- `mock_or_placeholder_zero`: True
- `full_page_only_locator_zero`: False
- `broad_table_only_zero_or_discarded`: True
- `atomic_artifact_present`: True
- `artifact_growth_vs_r038a_bounded`: True
- `eligible_for_heldout_at_least_r037`: False
- `max_doc_share_acceptable`: True
- `max_page_share_acceptable`: True
- `no_qa`: True
- `not_merged_into_final_cumulative_artifacts`: True

## Next Step
Review R038c gate failures before constructing held-out subset or running any QA/effectiveness gate.
