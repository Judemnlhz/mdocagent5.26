# R069 Dataset Artifact Health Audit

Decision: `r069_dataset_artifact_health_audit_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Uses public questions, public retrieved page text, and public artifacts only.
- Does not use answers, evidence pages, official scoring, or artifact-lift claims.

## Summary
- records scanned: 1073
- top-k per modality: 4
- code-like literal records: 64
- exact-code lookup records: 4
- exact-code lookup public text literal present: 4
- exact-code lookup current artifact literal present: 1
- exact-code lookup replay artifact literal present: 3
- exact-code lookup current selector selected: 0
- exact-code lookup replay selector selected: 3

## Failure Buckets
- `artifact_dimension_support_guard`: 17
- `artifact_extraction_missing_code_like_literal`: 2
- `code_like_temporal_metric_literal_triggered_exact_code_guard`: 56
- `current_store_selected_artifacts`: 16
- `document_metadata_refusal_guard`: 2
- `no_relevant_artifact_guard`: 31
- `operand_completeness_guard`: 2
- `replay_code_name_code_like_literal_selected`: 3
- `retrieval_or_public_text_missing_code_like_literal`: 3
- `retrieval_or_public_text_missing_literals`: 666
- `retrieved_text_has_literals_but_no_artifacts`: 272
- `uncategorized_artifact_gap`: 3

## Recommended Next
- Treat code-like literal rows as a broad health bucket; inspect exact-code lookup separately from years, quarters, and metric names.
- Use R069 exact-code lookup rows to separate retrieval/text absence from artifact extraction and selector failures before any provider run.
- Rebuild a bounded Stage 2 artifact store with code/name extraction before testing exact-code positive provider cases.
- Do not run full QA until artifact health shows positive cases with visible text, generated artifacts, and selector-selected support.
