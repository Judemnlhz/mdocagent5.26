# R042 Manual Attribution of R040 Aggregate Tie

Decision: `r042_manual_attribution_complete`

## Boundary
- Deterministic post-hoc attribution only.
- No provider calls, no new prediction, no new evaluation, no full QA.
- Gold answers/evidence pages are used only for diagnosis.
- Not full-data generalization and not an official MMLongBench result.

## Main Finding
The R040 aggregate tie comes from artifact_only losing two answer-value/unanswerable cases and gaining two numeric cases; at least one gain is not attributable to changed selected pages.

## Divergent Records
- cases: 4
- artifact_only gains: 2
- artifact_only losses: 2

| record_id | pattern | label | confidence |
| ---: | --- | --- | --- |
| 223 | 110 | artifact_only_loss_answer_value_shift | high |
| 508 | 110 | artifact_only_loss_unanswerable_false_positive | high |
| 1034 | 001 | artifact_only_gain_order_sensitive_numeric | medium |
| 1035 | 001 | artifact_only_gain_same_pages_generation_variance | medium |

## Text-Different Binary-Same Records
- cases: 18
- pattern counts: `{"000": 13, "111": 5}`

| label | count |
| --- | ---: |
| binary_same_all_correct_surface_variation | 5 |
| binary_same_all_wrong_failure_variation | 13 |

## Interpretation
- The next experiment should separate page reranking, page order, explicit artifact-snippet injection, and judge bucket effects before any full-data QA.
- `top4_artifact_only` losses are not the same failure mode as its gains; the aggregate tie hides both directions.
- The current R040 setup is page-rerank/page-order diagnosis, not prompt-visible artifact-context diagnosis.

## Recommended R043 Design
- Keep the 37-record targeted subset frozen for contrastive diagnosis.
- Run a no-new-retrieval prompt contrast only after R042 review: original page text, original page text plus explicit artifact snippets, artifact snippets only, and page-rerank-only.
- Log artifact exposure per record: artifact ids, source pages, snippet tokens, prompt inclusion flag, and whether the gold/evidence page is included.
- Add a support-aware/citation-aware manual rubric for the 4 divergent and 18 text-diff/binary-same records before scaling.
- Do not run full QA until the prompt-visible artifact condition is implemented and separated from page-order effects.
