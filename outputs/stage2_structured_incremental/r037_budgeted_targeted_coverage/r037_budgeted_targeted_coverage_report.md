# R037 Budgeted Targeted Coverage

Decision: `activation_gate_passed_proceed_to_repaired_20_to_30_expansion_gate`

## Scope
- Diagnostic activation scan only.
- No QA, no effectiveness, no graph, no rerank tuning.
- Page selection uses retrieval candidate pages plus page-local OCR/table-text atomicizer signals; no answer/gold/evidence fields.
- Atomicizer budget stays at 8 cells/page.

## Targeted Subset
- Target record ids after excluding activated/prior top30: 1021
- Candidates: 1353
- Selected pages/docs: 10 pages / 7 docs

## Stage 2 Delta Quality
- Pages attempted: 10
- Parse failures: 0
- Provider success/fail: 10 / 0
- Valid/discarded artifacts: 138 / 3
- table_cell / numeric_fact: 66 / 69

## Delta Eligibility
- Atomic strong eligible: 135
- Eligible pages with atomic artifact: 10
- Broad table only: 0
- Mock / full_page_only: 0 / 0

## Merged Atomic Activation
- Strong eligible pages: 20
- Activated records: 113
- Eligible for held-out: 103
- Changed records artifact_only / original_plus: 88 / 66
- Max doc share: 0.1239
- Max page share: 0.1150
- Effective docs/pages: 10.867234 / 15.121763

## Next Step
Run the repaired 20 -> 30 expansion gate. Do not run effectiveness until that generalization gate passes.
