# R031 Bounded Activation Scan Review

Decision: `continue_stage2_coverage_quality_no_qa`
Reason: activation remains insufficient or too concentrated; do not run QA/effectiveness gate

## Scope
- Diagnostic only.
- Temporary `cumulative20 + R030 repaired 3 pages` artifact store.
- Activation scan uses atomic-only artifacts.
- No QA, no graph, no rerank tuning, no effectiveness claim.

## Fallback Boundary
- Boundary decision: `pass_public_safe_ocr_only`
- Reads question/answer/gold/evidence: `False`

## Merged-All Eligibility
- Total artifacts: 97
- Eligible artifacts: 53
- Atomic strong eligible artifacts: 32
- Numeric facts: 16
- Table cells: 16
- Broad table only: 3

## Atomic-Only Eligibility
- Atomic artifacts: 32
- Atomic strong eligible artifacts: 32
- Numeric facts: 16
- Table cells: 16
- Eligible pages with atomic artifact: 2
- Broad table only: 0

## Merged-All Activation
- Activated records: 36
- Eligible for held-out: 36
- Changed records, artifact_only: 42
- Changed records, original_plus_artifact: 29
- Held-out available: `False`

## Atomic-Only Activation
- Activated records: 6
- Eligible for held-out: 6
- Changed records, artifact_only: 6
- Changed records, original_plus_artifact: 4
- Held-out available: `False`

## Merged-All Concentration
- Max doc share: 0.2222
- Max page share: 0.1944
- Effective docs: 6.821053
- Effective pages: 8.695652
- Activated by doc: `{'2023.acl-long.386.pdf': 5, '2307.09288v2.pdf': 1, '3M_2018_10K.pdf': 1, '936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf': 3, 'BESTBUY_2023_10K.pdf': 3, 'NETFLIX_2015_10K.pdf': 6, 'NIKE_2021_10K.pdf': 6, 'PG_2021.03.04_US-Views-on-China_FINAL.pdf': 3, 'STEPBACK.pdf': 8}`
- Activated by page: `{'2023.acl-long.386.pdf#p007': 5, '2307.09288v2.pdf#p053': 1, '3M_2018_10K.pdf#p020': 1, '936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003': 3, 'BESTBUY_2023_10K.pdf#p026': 3, 'NETFLIX_2015_10K.pdf#p020': 2, 'NETFLIX_2015_10K.pdf#p023': 4, 'NIKE_2021_10K.pdf#p033': 6, 'PG_2021.03.04_US-Views-on-China_FINAL.pdf#p014': 3, 'STEPBACK.pdf#p004': 7, 'STEPBACK.pdf#p005': 5}`

## Atomic-Only Concentration
- Max doc share: 0.5000
- Max page share: 0.5000
- Effective docs: 2.0
- Effective pages: 2.0
- Activated by doc: `{'936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf': 3, 'BESTBUY_2023_10K.pdf': 3}`
- Activated by page: `{'936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003': 3, 'BESTBUY_2023_10K.pdf#p026': 3}`
