# R071 Evidence Skill Graph Registry Gate

Decision: `r071_evidence_skill_registry_gate_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Uses public questions, public retrieved page text, and public artifacts only.
- Does not use answers, evidence pages, official scoring, or artifact-lift claims.

## Summary
- records scanned: 1073
- registry skills: 6 `['exact_code_lookup', 'figure_caption_grounding', 'key_value_lookup', 'numeric_computation', 'table_numeric_lookup', 'text_span_grounding']`
- evidence unit types: 6 `['text_span', 'table_cell', 'numeric_fact', 'key_value', 'caption', 'code_name_pair']`
- document edge types: 8 `['contains', 'same_page', 'same_table', 'row_of', 'column_of', 'caption_of', 'nearby', 'code_maps_to']`
- contract failures: `[]`
- control activation passed: True

## Activated Skill Counts
- `exact_code_lookup`: 8
- `figure_caption_grounding`: 165
- `key_value_lookup`: 206
- `numeric_computation`: 10
- `table_numeric_lookup`: 625
- `text_span_grounding`: 163

## Guard Decision Counts
- `artifact_dimension_support_guard`: 30
- `document_metadata_refusal_guard`: 3
- `exact_code_absence_guard`: 8
- `no_relevant_artifact_guard`: 1007
- `operand_complete_selection`: 1
- `operand_completeness_guard`: 9
- `token_key_value_selection`: 15

## Recommended Next
- Keep the registry bounded and dataset-agnostic; do not add dataset-named skills or a large skill tree.
- Proceed to R072 token-budgeted capsule renderer using this registry as the only skill dispatch interface.
- Do not run provider QA until R072/R073 no-provider capsule and cross-dataset audits pass.
