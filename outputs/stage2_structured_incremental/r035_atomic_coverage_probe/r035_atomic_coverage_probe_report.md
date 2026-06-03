# R035 Atomic Coverage Broaden Probe

## Scope

R035 broadens atomic Stage 2 coverage before any repaired 20 -> 30 expansion gate. It is not QA, not effectiveness evaluation, not graph expansion, and not rerank tuning.

## Subset Construction

A new bounded subset was selected with `scripts/build_stage2_r035_atomic_coverage_subset.py` using only retrieval top-k page ids and page-local OCR/layout text. The selector excludes cumulative20 pages and the R033 same-3 probe pages, then ranks pages by whether the generic atomicizer can derive table/numeric structures offline.

- Records: `data/MMLongBench/sample-with-retrieval-results.json`
- Exclusions: `outputs/stage2_structured_incremental/r028_10_to_20/cumulative/subset_cumulative_20.jsonl`, `outputs/stage2_structured_incremental/r028_20_to_30/parse_repair_replay_3/subset_failed_pages.jsonl`
- Selected: 10 pages / 6 docs
- No answer, gold answer, evidence pages, evidence sources, binary correctness, or `gold_*` fields are used for selection.

## Stage 2 Coverage Results

| Metric | Value |
|---|---:|
| `num_pages_attempted` | 10 |
| `num_selected_docs` | 6 |
| `parse_failure_count` | 0 |
| `json_parse_success_count` | 10 |
| `num_valid_artifacts` | 300 |
| `num_discarded_artifacts` | 4 |
| `table_cell_count` | 149 |
| `numeric_fact_count` | 120 |
| `atomic_strong_eligible_artifacts` | 259 |
| `eligible_pages_with_atomic_artifact` | 10 |
| `broad_table_only_count` | 0 |

Coverage decision: passed. The generic atomicizer is no longer limited to the R033 same-3 pages.

## Diagnostic Activation Results

Temporary artifact store: cumulative20 + R035 coverage artifacts.

| Metric | Merged all | Atomic only |
|---|---:|---:|
| `activated_count` | 60 | 32 |
| `eligible_for_heldout_count` | 50 | 22 |
| `changed_count` | 69 | 39 |
| `original_plus_changed_count` | 48 | 27 |
| `strong_eligible_page_count` | 19 | 10 |
| `max_doc_share` | 0.1667 | 0.3125 |
| `max_page_share` | 0.1667 | 0.3125 |
| `effective_num_docs` | 8.6538 | 4.5714 |
| `effective_num_pages` | 13.1574 | 6.3333 |

Activation decision: blocked. Atomic-only activation improved from R034's 7 to 32 activated records and is less concentrated, but `eligible_for_heldout_count=22` is below the gate threshold for moving into repaired 20 -> 30 expansion.

## Decision

`continue_atomic_coverage_broadening_no_qa`

Do not run repaired 20 -> 30 expansion gate yet. Continue broadening atomic coverage across additional pages/documents, then rerun a bounded activation scan. Only after atomic-only activation is sufficiently broad and held-out eligible should the workflow proceed to repaired 20 -> 30. Effectiveness/QA remains blocked.
