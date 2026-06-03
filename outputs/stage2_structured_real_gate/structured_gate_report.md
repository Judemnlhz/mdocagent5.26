# Stage 2 Real Structured Artifact Gate

Decision: `stage2_structured_gate_pass`
Status: `executed`

## Scope
- No rerank tuning.
- No QA effectiveness run.
- No full artifact ablation.

## Subset
- Selected docs: 9
- Selected pages: 10
- Subset file: `outputs/subsets/stage2_structured_real_gate_subset.jsonl`

## Stage 2 Quality
- Real provider calls: 5 success / 5 failed
- Parsed artifacts: 26
- Parse failures: 5

## Eligibility
- Strong eligible artifacts: 4
- Eligible pages: 2
- Eligible docs: 2
- Mock or placeholder content: 0
- Full-page-only locator: 2
- Eligible artifact types: `{'table': 4}`

## Next Step
Build a new held-out activation-rich subset from records activated by these strong eligible artifacts, excluding prior policy-tuning top30.
