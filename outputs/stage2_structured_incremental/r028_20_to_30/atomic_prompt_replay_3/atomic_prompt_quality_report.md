# R028 Atomic Prompt Quality Probe

Scope: same 3 failed pages only; no Stage 2 expansion, no activation scan, no QA, no graph, no rerank tuning. Replay outputs are probes and are not merged into cumulative artifacts.

## Metrics

| Metric | Parse-repair replay | Atomic prompt replay | Delta |
|---|---:|---:|---:|
| parse_failure_count | 0 | 0 | +0 |
| json_parse_success_count | 3 | 3 | +0 |
| valid_artifacts | 6 | 10 | +4 |
| discarded_artifacts | 0 | 0 | +0 |
| strong_eligible_artifacts | 5 | 9 | +4 |
| eligible_pages | 2 | 2 | +0 |
| mock_or_placeholder_content | 0 | 0 | +0 |
| full_page_only_locator | 1 | 0 | -1 |
| table_cell_artifacts | 0 | 6 | +6 |
| numeric_fact_artifacts | 0 | 0 | +0 |

## Artifact Shape

- Parse-repair replay artifact types: `{'figure': 1, 'table': 5}`
- Atomic prompt replay artifact types: `{'figure': 1, 'table': 3, 'table_cell': 6}`
- Atomic prompt created six `table_cell` artifacts on `2401.18059v1.pdf#p006`, including retrieval-method percentage values.
- `936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003` and `BESTBUY_2023_10K.pdf#p026` still emitted broad table summaries rather than atomic numeric facts.

## Decision

Atomic prompt repair improves artifact structure but is not stable enough to justify activation or QA. Continue bounded prompt/parser repair on the same failed-page probe, with emphasis on `numeric_fact` extraction and multi-page consistency. Keep raw provider responses private; this report contains only parsed summaries and hashes/metrics.
