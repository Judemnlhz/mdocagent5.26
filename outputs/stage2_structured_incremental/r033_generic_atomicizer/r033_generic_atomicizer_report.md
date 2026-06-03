# R033 Generic Table/Numeric Atomicizer Report

## Scope

R033 implements a generic Stage 2 table/numeric atomicizer before any activation or QA scan. It is bounded to the same 3 R028/R032 failed pages for this replay. No expansion, activation, graph, rerank tuning, or QA was run.

## Implementation

- Added `mdocnexus/stage2/table_numeric_atomicizer.py` as a separate module instead of adding more rules to `scripts/stage2.py`.
- The atomicizer reads only page-local Stage 2 inputs: `page_text`, `layout_blocks`, selected page identity for artifact ids, and existing provider artifacts for deduplication.
- It does not inspect question, answer, gold fields, evidence pages, dataset names, document names, or task labels for extraction decisions.
- It extracts conservative table-like OCR structures with row labels, column labels, concrete values, units, source text, `text_offset`, `source_block`, and `table_cell` locators.
- It emits paired `table_cell` and `numeric_fact` artifacts only when row/column/value context can be constructed.
- It remains distinct from the removed R030 deterministic fallback and contains no BestBuy/IPMS/MMLongBench-specific extraction functions.

## Static Genericity Audit

Checked core files:

- `scripts/stage2.py`
- `mdocnexus/stage2/table_numeric_atomicizer.py`
- `mdocnexus/stage2/provider.py`
- `mdocnexus/stage2/artifact_quality.py`

Forbidden probe-specific patterns checked:

- BestBuy / bestbuy
- deterministic_page_text_numeric_fallback
- _extract_bestbuy_numeric_facts
- _extract_performance_table_numeric_facts
- selected financial data
- Performance Information Table

Result: no matches in checked core files.

## Verification

- `python3 -m py_compile scripts/stage2.py mdocnexus/stage2/table_numeric_atomicizer.py mdocnexus/stage2/artifact_quality.py`
- `python3 -m unittest mdocnexus.stage2.tests.test_table_numeric_atomicizer mdocnexus.stage2.tests.test_artifact_schema_validation`
- `git diff --check`

All completed successfully.

## Same-3 Replay Metrics

Replay path: `outputs/stage2_structured_incremental/r033_generic_atomicizer/same3_replay/atomic_quality_report.md`

| Metric | Value |
|---|---:|
| `parse_failure_count` | 0 |
| `json_parse_success_count` | 3 |
| `valid_artifacts` | 83 |
| `discarded_artifacts` | 3 |
| `strong_eligible_artifacts` | 81 |
| `atomic_strong_eligible_artifacts` | 78 |
| `eligible_pages` | 3 |
| `eligible_pages_with_atomic_artifact` | 3 |
| `mock_or_placeholder_content` | 0 |
| `full_page_only_locator` | 0 |
| `table_cell_count` | 42 |
| `numeric_fact_count` | 36 |
| `broad_table_only_count` | 0 |
| `broad_table_only_discarded_count` | 3 |

## Decision

R033 passes the bounded same-page atomicity gate. The next step may be a diagnostic-only bounded activation scan using R033 outputs, but not QA or effectiveness evaluation yet. If activation is sufficiently broad and not over-concentrated, run a repaired 20 -> 30 expansion gate before any limited effectiveness gate.
