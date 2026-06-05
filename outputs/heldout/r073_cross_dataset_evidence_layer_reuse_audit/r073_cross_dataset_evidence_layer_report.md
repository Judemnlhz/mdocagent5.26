# R073 Cross-Dataset Evidence Layer Reuse Audit

Decision: `r073_cross_dataset_reuse_audit_complete_with_input_gaps`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Public questions and page-text availability are used; answer/evidence fields are excluded from outputs.
- Non-MMLB datasets without public retrieval/artifact bindings are marked blocked rather than scored.

## Dataset Status
- `MMLB`: full_public_retrieval_artifact_capsule_audited; records scanned=1073; full capsule audit=True
- `LDU`: blocked_missing_public_retrieval_or_artifacts; records scanned=2325; full capsule audit=False
- `FETA`: blocked_missing_public_retrieval_or_artifacts; records scanned=1016; full capsule audit=False
- `PTEXT`: blocked_missing_public_retrieval_or_artifacts; records scanned=2804; full capsule audit=False
- `PTAB`: blocked_missing_public_retrieval_or_artifacts; records scanned=393; full capsule audit=False

## MMLB Token Reuse
- mean guarded capsule/raw ratio: 0.222101

## Cross-Dataset Skill Activation
- `exact_code_lookup`: 84
- `figure_caption_grounding`: 488
- `key_value_lookup`: 2810
- `numeric_computation`: 25
- `table_numeric_lookup`: 2904
- `text_span_grounding`: 1670

## Recommended Next
- Keep the same Evidence Skill Registry and capsule renderer; MMLB guarded capsule/raw ratio remains 0.222101 under the cross-dataset audit.
- Do not claim cross-dataset token or citation gains until public retrieval/artifact bindings are built for blocked datasets: ['FETA', 'LDU', 'PTAB', 'PTEXT'].
- Next engineering step should be a small reusable public retrieval-to-artifact binding adapter, not new dataset-named skills or a large graph tree.
