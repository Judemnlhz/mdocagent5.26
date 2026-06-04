# R067 Source/OCR Code-List Extraction Audit

Decision: `r067_source_ocr_code_list_audit_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Single-case diagnostic for record 508 / page 7 / AR03.
- No official score and no artifact-positive lift claim.

## Summary
- target code present in OCR text: False
- AR codes in OCR text: `['AR01', 'AR02']`
- existing page-7 artifact count: 0
- extracted code/name pairs: 30
- extracted AR pairs: `[{"artifact_id": "code_name_pair_001", "artifact_type": "text_span", "content": "AR01: Little Rock", "element_locatable": true, "eps_code": "AR01", "geographic_market_name": "Little Rock", "group_label": "Arkansas", "page_index": 7, "source_anchored": true}, {"artifact_id": "code_name_pair_002", "artifact_type": "text_span", "content": "AR02: Northern Arkansas", "element_locatable": true, "eps_code": "AR02", "geographic_market_name": "Northern Arkansas", "group_label": "Arkansas", "page_index": 7, "source_anchored": true}]`
- target code extracted: False
- selector guard with extracted artifacts: `exact_code_absence_guard`

## Recommended Next
- Keep exact-code matching strict; the repaired code-list extractor must not infer AR03 from AR01/AR02 or Arkansas context.
- Manually inspect the page image for AR03 only if the benchmark expects an answer; current OCR/text evidence does not support it.
- Integrate the code-name list extractor into Stage 2 for EPS-like text lists, then rerun no-provider coverage before any provider diagnostic.
- For record 508, route to page-cited absence/refusal unless source/OCR repair reveals exact AR03.
