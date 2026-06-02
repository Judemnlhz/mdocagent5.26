# Strong Artifact Method Gate Summary

Artifact path: `outputs/experiments/matrix/retrieval_topk_scope_topk4_hybrid_full_hybrid_page_neighborhood/stage2_doc_coverage/artifacts.jsonl`

## Strong Eligibility Policy
- allowed types: `numeric_fact`, `table`, `table_cell`, `figure`, `caption`
- requires non-mock content
- requires valid source anchor
- requires non-full-page strong locator
- rejects unstructured text/span/full-page/mock candidates

## Quality Audit
- total artifacts: 450
- eligible artifacts: 0
- eligible rate: 0.000000

## Rejection Reasons
- `mock_or_placeholder_content`: 250
- `unstructured_artifact_type`: 200

## Activation Scan
- total records: 1073
- activated records: 0
- changed records: 0

## Gate Decision
- held-out activation-rich subset available: `False`
- QA/eval run: `False`
- reason: strong eligibility produced zero activated records, so an effectiveness run would only test fallback safety.

## Conclusion
Current Stage 2 outputs are not sufficient for a positive artifact-aware retrieval method because structured candidates are mock/placeholders and all strong-eligible activation is zero.

## Next Required Work
- Replace dry-run/mock Stage 2 artifact compiler with real structured extraction for table_cell, numeric_fact, table, figure, and caption artifacts.
- Emit non-mock normalized_content with table/cell/metric fields where available.
- Require locators such as table_cell, bbox, figure_region, caption_block, or text_offset tied to source anchors.
- Re-run eligibility audit until eligible_artifacts > 0 and held-out activated records are available.
