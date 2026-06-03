# R031 Fallback Boundary Audit

Decision: `pass_public_safe_ocr_only`

Uses page_text: `True`
Requires document_generic: `True`
Uses layout locator: `True`
Reads question/answer/gold/evidence: `False`
Forbidden term hits: `{}`

## Notes
- Fallback is gated to document_generic mode.
- Fallback reads OCR page_text and page layout locator only.
- selected_page is used for doc_id/page_index identity, not question or answer fields.
- No activation, QA, graph, or rerank tuning is performed by this audit.
