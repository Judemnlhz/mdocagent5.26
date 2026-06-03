# R031 Bounded Activation Scan Review

Decision: `continue_stage2_coverage_quality_no_qa`
Reason: activation remains insufficient or too concentrated; do not run QA/effectiveness gate

## Scope
- Diagnostic only.
- Temporary `cumulative20 + R030 repaired 3 pages` artifact store.
- Activation scan uses atomic-only artifacts.
- No QA, no graph, no rerank tuning, no effectiveness claim.

## Fallback Boundary
- Boundary decision: `pass_fallback_removed_from_default_stage2_path`
- Reads question/answer/gold/evidence: `False`

## Merged-All Eligibility
- Total artifacts: 282
- Eligible artifacts: 202
- Atomic strong eligible artifacts: 171
- Numeric facts: 80
- Table cells: 100
- Broad table only: 11

## Atomic-Only Eligibility
- Atomic artifacts: 171
- Atomic strong eligible artifacts: 171
- Numeric facts: 80
- Table cells: 91
- Eligible pages with atomic artifact: 10
- Broad table only: 0

## Merged-All Activation
- Activated records: 60
- Eligible for held-out: 50
- Changed records, artifact_only: 70
- Changed records, original_plus_artifact: 48
- Held-out available: `True`

## Atomic-Only Activation
- Activated records: 32
- Eligible for held-out: 22
- Changed records, artifact_only: 40
- Changed records, original_plus_artifact: 27
- Held-out available: `False`

## Merged-All Concentration
- Max doc share: 0.1667
- Max page share: 0.1667
- Effective docs: 8.653846
- Effective pages: 13.15736
- Activated by doc: `{'05-03-18-political-release.pdf': 10, '2005.12872v3.pdf': 2, '2023.acl-long.386.pdf': 9, '2023.findings-emnlp.248.pdf': 2, '2210.02442v1.pdf': 4, '2303.05039v2.pdf': 8, '2307.09288v2.pdf': 1, '3M_2018_10K.pdf': 1, 'NETFLIX_2015_10K.pdf': 6, 'NIKE_2021_10K.pdf': 6, 'PG_2021.03.04_US-Views-on-China_FINAL.pdf': 3, 'STEPBACK.pdf': 8}`
- Activated by page: `{'05-03-18-political-release.pdf#p003': 10, '2005.12872v3.pdf#p008': 2, '2005.12872v3.pdf#p012': 2, '2023.acl-long.386.pdf#p001': 3, '2023.acl-long.386.pdf#p003': 5, '2023.acl-long.386.pdf#p007': 5, '2023.findings-emnlp.248.pdf#p002': 1, '2023.findings-emnlp.248.pdf#p013': 1, '2210.02442v1.pdf#p003': 4, '2210.02442v1.pdf#p004': 2, '2303.05039v2.pdf#p003': 8, '2307.09288v2.pdf#p053': 1, '3M_2018_10K.pdf#p020': 1, 'NETFLIX_2015_10K.pdf#p020': 2, 'NETFLIX_2015_10K.pdf#p023': 4, 'NIKE_2021_10K.pdf#p033': 6, 'PG_2021.03.04_US-Views-on-China_FINAL.pdf#p014': 3, 'STEPBACK.pdf#p004': 7, 'STEPBACK.pdf#p005': 5}`

## Atomic-Only Concentration
- Max doc share: 0.3125
- Max page share: 0.3125
- Effective docs: 4.571429
- Effective pages: 6.333333
- Activated by doc: `{'05-03-18-political-release.pdf': 10, '2005.12872v3.pdf': 2, '2023.acl-long.386.pdf': 6, '2023.findings-emnlp.248.pdf': 2, '2210.02442v1.pdf': 4, '2303.05039v2.pdf': 8}`
- Activated by page: `{'05-03-18-political-release.pdf#p003': 10, '2005.12872v3.pdf#p008': 2, '2005.12872v3.pdf#p012': 2, '2023.acl-long.386.pdf#p001': 3, '2023.acl-long.386.pdf#p003': 5, '2023.findings-emnlp.248.pdf#p002': 1, '2023.findings-emnlp.248.pdf#p013': 1, '2210.02442v1.pdf#p003': 4, '2210.02442v1.pdf#p004': 2, '2303.05039v2.pdf#p003': 8}`
