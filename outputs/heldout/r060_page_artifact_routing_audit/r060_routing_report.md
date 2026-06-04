# R060 Page/Artifact Routing Audit

Decision: `r060_routing_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Audits prompts where page evidence is sufficient but artifact evidence is rejected.
- Not an official score and not evidence of artifact positive lift.

## Summary
- cases: 2
- guard decisions: `{"artifact_dimension_support_guard": 2}`
- page-routed records: `[223, 227]`

## Per-Record Routing
### Record 223
- guard decision: `artifact_dimension_support_guard`
- answer policy: `use_page_evidence_or_refuse`
- selected artifacts: 0
- visible page support sufficient: True
- artifact support sufficient: False
- routing passed: True
- routing failures: `[]`
### Record 227
- guard decision: `artifact_dimension_support_guard`
- answer policy: `use_page_evidence_or_refuse`
- selected artifacts: 0
- visible page support sufficient: True
- artifact support sufficient: False
- routing passed: True
- routing failures: `[]`

## Recommended Next
- Do not run full QA from R060.
- If manually accepted, the next bounded step can be a tiny provider diagnostic on page-routed prompts only.
- Keep claims limited: R060 validates prompt routing behavior, not artifact-aware retrieval lift.
