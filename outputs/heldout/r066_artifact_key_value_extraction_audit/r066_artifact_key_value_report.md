# R066 Artifact Key/Value Extraction Audit

Decision: `r066_artifact_key_value_audit_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Single-case diagnostic for record 508 / AR03 after R065 parser normalization.
- No official score and no artifact-positive lift claim.

## Summary
- selector guard: `exact_code_absence_guard`
- candidate artifacts: 16 with page counts `{"2": 0, "5": 0, "6": 0, "7": 0, "8": 16, "9": 0}`
- artifact exact code present: False
- retrieved page exact code present: False
- retrieved page state marker present: True
- whole-document extracted text exact code present: False
- primary root cause: `extracted_document_text_missing_required_code`
- all categories: `["artifact_normalization_over_numeric_tables_not_eps_key_values", "artifact_store_missing_exact_code_key_value", "extracted_document_text_missing_required_code", "selector_guard_correct_for_current_public_evidence", "visible_page_supports_code_absence_not_answer"]`

## Recommended Next
- Do not relax exact-code selector matching for AR03; current public evidence does not contain the requested code.
- Keep the exact-code absence guard and route this case to page-cited refusal/absence handling.
- Audit the source PDF/OCR for whether AR03 is visually present but missing from extracted text; if absent in the source, treat as not answerable under visible evidence.
- Repair EPS/table-list artifact extraction for page text with code/name lists; page 7 has no artifacts while it contains the Arkansas EPS neighborhood.
- After extraction repair, rerun a no-provider exact-code coverage audit before any provider diagnostic.
