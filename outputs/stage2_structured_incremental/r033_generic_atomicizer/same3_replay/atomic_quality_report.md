# R030 Atomic Quality Replay

Scope: same 3 R028 failed pages only; no expansion, no activation scan, no QA, no graph, no rerank tuning. Replay outputs are probes and are not merged into cumulative artifacts.

Decision: `ready_for_activation_scan_review`

## Metrics

| Metric | R029 atomic prompt | R030 atomic quality | Delta |
|---|---:|---:|---:|
| `parse_failure_count` | 0 | 0 | +0 |
| `json_parse_success_count` | 3 | 3 | +0 |
| `valid_artifacts` | 10 | 83 | +73 |
| `discarded_artifacts` | 0 | 3 | +3 |
| `strong_eligible_artifacts` | 9 | 81 | +72 |
| `atomic_strong_eligible_artifacts` | 6 | 78 | +72 |
| `eligible_pages` | 2 | 3 | +1 |
| `eligible_pages_with_atomic_artifact` | 1 | 3 | +2 |
| `mock_or_placeholder_content` | 0 | 0 | +0 |
| `full_page_only_locator` | 0 | 0 | +0 |
| `table_cell_count` | 6 | 42 | +36 |
| `numeric_fact_count` | 0 | 36 | +36 |
| `broad_table_only_count` | 2 | 0 | -2 |

## Checks
- `parse_failure_still_zero`: True
- `mock_still_zero`: True
- `full_page_only_still_zero`: True
- `table_cell_kept_or_increased`: True
- `numeric_fact_appeared`: True
- `broad_table_only_declined`: True
- `eligible_pages_not_decreased`: True

## Scope Guard
- Uses only Stage 2 output quality taxonomy; no gold fields.
- Broad/table-title-only schema-valid artifacts are excluded from atomic strong eligibility.
- Activation scan remains blocked unless atomic artifacts are stable on this bounded replay.
