# R072 Token-Budgeted Evidence Capsule Audit

Decision: `r072_token_budgeted_capsule_audit_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Uses public questions, public retrieved page text, and public artifacts only.
- Does not use answers, evidence pages, official scoring, or artifact-lift claims.

## Summary
- records scanned: 1073
- mean raw page tokens: 1607.252563
- mean flat artifact tokens: 41.38863
- mean capsule tokens without trace: 38.671948
- mean capsule tokens with guard trace: 57.319664
- mean guarded capsule/raw ratio: 0.222101
- guarded capsule lower than raw rate: 0.976701

## Activated Skill Counts
- `exact_code_lookup`: 8
- `figure_caption_grounding`: 165
- `key_value_lookup`: 206
- `numeric_computation`: 10
- `table_numeric_lookup`: 625
- `text_span_grounding`: 163

## Recommended Next
- Use the R071 registry renderer for R073 cross-dataset reuse; current guarded capsule mean/raw token ratio is 0.222101.
- Keep capsule rendering deterministic and bounded; do not add a second capsule abstraction.
- Do not run provider QA until R073 confirms cross-dataset schema and token behavior.
