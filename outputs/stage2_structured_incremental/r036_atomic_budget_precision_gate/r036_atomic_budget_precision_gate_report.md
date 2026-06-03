# R036 Atomic Budget/Precision Gate

## Scope

R036 adds budgeted ranking to the generic Stage 2 table/numeric atomicizer and replays the same R035 10-page subset. It does not expand pages, run QA, run effectiveness evaluation, build graphs, or tune reranking.

## Change

- Default atomicizer budget reduced from 12 cells/page to 8 cells/page.
- Candidate cells are ranked before truncation.
- Ranking favors real column labels, year columns, percentages, normalizable numeric values, short row labels, compact source text, and earlier columns.
- Added a unit test that enforces the default per-page budget.

## Same-Subset Replay

Subset: `outputs/stage2_structured_incremental/r035_atomic_coverage_probe/subset_r035_atomic_coverage.jsonl`

| Metric | R035 unbudgeted | R036 budgeted |
|---|---:|---:|
| `num_pages_attempted` | 10 | 10 |
| `parse_failure_count` | 0 | 0 |
| `json_parse_success_count` | 10 | 10 |
| `num_valid_artifacts` | 300 | 218 |
| `num_discarded_artifacts` | 4 | 3 |
| `table_cell_count` | 149 | 100 |
| `numeric_fact_count` | 120 | 80 |
| `atomic_strong_eligible_artifacts` | 259 | 171 |
| `eligible_pages_with_atomic_artifact` | 10 | 10 |
| `broad_table_only_count` | 0 | 0 |

## Diagnostic Activation

| Metric | R035 atomic-only | R036 atomic-only |
|---|---:|---:|
| `activated_count` | 32 | 32 |
| `eligible_for_heldout_count` | 22 | 22 |
| `changed_count` | 39 | 39 |
| `original_plus_changed_count` | 27 | 27 |
| `strong_eligible_page_count` | 10 | 10 |
| `max_doc_share` | 0.3125 | 0.3125 |
| `max_page_share` | 0.3125 | 0.3125 |

## Decision

`budget_passed_activation_blocked`

The budgeted atomicizer reduces artifact volume by about 27% while preserving current atomic-only activation and page coverage. This supports continuing with budgeted coverage broadening. However, `eligible_for_heldout_count=22` remains below the gate for repaired 20 -> 30 expansion. Do not run repaired 20 -> 30 or effectiveness yet.
