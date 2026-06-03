# R028 Repaired Artifact Quality Inspection

Decision: quality_inspect_prompt_repair_needed

## Scope
- No Stage 2 page expansion.
- No QA, graph, rerank tuning, or full ablation.
- Replay artifacts were not merged into cumulative artifacts.

## Metrics
- total_artifacts: 6
- strong_eligible_artifacts: 5
- eligible_pages: 2
- mock_or_placeholder_content: 0
- full_page_only_locator: 1
- missing_strong_locator: 0
- artifact_type_counts: {'figure': 1, 'table': 5}
- atomic_artifact_count: 0
- whole_table_blob_count: 4
- content_length_avg: 353.2

Interpretation: eligibility is nonzero, but atomic structured evidence is still insufficient; use a bounded prompt-only replay on the same pages before any activation scan.
