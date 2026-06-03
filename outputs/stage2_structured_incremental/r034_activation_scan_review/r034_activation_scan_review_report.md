# R034 Activation Scan Review After Generic Atomicizer

## Scope

R034 is diagnostic only. It temporarily combines cumulative20 artifacts with R033 same-3 generic atomicizer artifacts, then compares merged-all activation with atomic-only activation. It does not run QA, graph expansion, rerank tuning, or effectiveness evaluation.

## Inputs

- Base artifacts: `outputs/stage2_structured_incremental/r028_10_to_20/cumulative/artifacts.jsonl`
- R033 artifacts: `outputs/stage2_structured_incremental/r033_generic_atomicizer/same3_replay/stage2_delta/artifacts.jsonl`
- Temporary merged store: `outputs/stage2_structured_incremental/r034_activation_scan_review/merged_cumulative20_plus_r030/artifacts.jsonl`
- Temporary atomic-only store: `outputs/stage2_structured_incremental/r034_activation_scan_review/atomic_only/artifacts.jsonl`

## Results

| Metric | Merged all | Atomic only |
|---|---:|---:|
| `activated_count` | 37 | 7 |
| `eligible_for_heldout_count` | 37 | 7 |
| `changed_count` | 47 | 11 |
| `original_plus_changed_count` | 30 | 5 |
| `strong_eligible_page_count` | 12 | 3 |
| `max_doc_share` | 0.2162 | 0.4286 |
| `max_page_share` | 0.1892 | 0.4286 |
| `effective_num_docs` | 7.1675 | 2.5789 |
| `effective_num_pages` | 9.0865 | 2.5789 |

## Interpretation

R033 improved same-page atomic artifact quality, but R034 shows the usable atomic activation range is still too small. Atomic-only activation reaches only 7 records and remains tied to the same 3 probe pages. The merged-all view reaches 37 records, but that view includes non-atomic cumulative20 artifacts and should not be treated as proof that R033 atomic coverage is broad enough.

## Decision

`continue_stage2_coverage_quality_no_qa`

Do not run the repaired 20 -> 30 expansion gate yet. The next step should broaden Stage 2 atomic coverage across more pages/documents, then rerun this bounded activation scan. Only if atomic-only activation is sufficiently broad and not concentrated should the process proceed to repaired 20 -> 30 expansion gate. Effectiveness/QA remains blocked.
