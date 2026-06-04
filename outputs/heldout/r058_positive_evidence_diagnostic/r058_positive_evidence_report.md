# R058 Positive-Evidence Diagnostic

Decision: `r058_positive_evidence_needs_selector_fix`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Audits R056 positive-signal records for visible, citable, answer-supporting artifact evidence.
- Positive signal is not treated as support; selected artifacts must cover public question dimensions.
- Not an official score and not evidence of artifact positive lift.

## Summary
- cases: 4
- guard decisions: `{"token_key_value_selection": 4}`
- support classes: `{"artifact_positive_signal_only_insufficient": 4}`
- artifact support sufficient records: `[]`
- artifact support insufficient records: `[69, 223, 224, 227]`

## Per-Record Audit
### Record 69
- guard decision: `token_key_value_selection`
- selected artifacts: 8 `['atomicizer_numeric_fact_002', 'atomicizer_numeric_fact_006', 'atomicizer_numeric_fact_008', 'atomicizer_numeric_fact_010', 'atomicizer_numeric_fact_012', 'atomicizer_numeric_fact_014', 'atomicizer_numeric_fact_016', 'atomicizer_table_cell_001']`
- artifact support sufficient: False
- visible support sufficient: False
- missing artifact dimensions: `['figure_4', 'retrieved_nodes', 'both_questions']`
- failure reasons: `['artifact_missing_dimensions:figure_4,retrieved_nodes,both_questions', 'visible_context_missing_dimensions:both_questions']`
### Record 223
- guard decision: `token_key_value_selection`
- selected artifacts: 8 `['atomicizer_numeric_fact_008', 'atomicizer_table_cell_007', 'atomicizer_numeric_fact_010', 'atomicizer_numeric_fact_012', 'atomicizer_table_cell_009', 'atomicizer_table_cell_011', 'atomicizer_numeric_fact_004', 'atomicizer_numeric_fact_006']`
- artifact support sufficient: False
- visible support sufficient: True
- missing artifact dimensions: `['higher_income_seniors', 'go_online', 'tablet_computer', 'year_2013']`
- failure reasons: `['artifact_missing_dimensions:higher_income_seniors,go_online,tablet_computer,year_2013']`
### Record 224
- guard decision: `token_key_value_selection`
- selected artifacts: 8 `['atomicizer_numeric_fact_008', 'atomicizer_table_cell_007', 'atomicizer_numeric_fact_010', 'atomicizer_numeric_fact_012', 'atomicizer_table_cell_009', 'atomicizer_table_cell_011', 'atomicizer_numeric_fact_004', 'atomicizer_numeric_fact_006']`
- artifact support sufficient: False
- visible support sufficient: False
- missing artifact dimensions: `['higher_income_seniors', 'go_online', 'tablet_computer', 'year_2022']`
- failure reasons: `['artifact_missing_dimensions:higher_income_seniors,go_online,tablet_computer,year_2022', 'visible_context_missing_dimensions:year_2022']`
### Record 227
- guard decision: `token_key_value_selection`
- selected artifacts: 8 `['atomicizer_numeric_fact_006', 'atomicizer_table_cell_005', 'atomicizer_numeric_fact_008', 'atomicizer_numeric_fact_010', 'atomicizer_numeric_fact_012', 'atomicizer_table_cell_007', 'atomicizer_table_cell_009', 'atomicizer_table_cell_011']`
- artifact support sufficient: False
- visible support sufficient: True
- missing artifact dimensions: `['age_65_plus', 'college_graduate', 'tablet_computer', 'gap_operation', 'year_2013']`
- failure reasons: `['artifact_missing_dimensions:age_65_plus,college_graduate,tablet_computer,gap_operation,year_2013']`

## Recommended Next
- Do not run provider QA on R058 positives yet.
- Repair guarded selector ranking so positive-signal artifacts must cover question dimensions, not just token overlap.
- Add table/key-value dimension matching for demographic, time, metric, and operand constraints.
- Rerun R058 before wiring guarded selector into full QA.
