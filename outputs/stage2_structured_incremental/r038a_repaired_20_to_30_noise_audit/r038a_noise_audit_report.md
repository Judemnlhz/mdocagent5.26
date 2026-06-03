# R038a Repaired 20 -> 30 Noise Audit

Decision: `proceed_to_small_repaired_provider_gate`

## Scope
- Offline atomicizer noise audit only.
- No provider calls, no Stage 2 compile, no artifact-store merge.
- No activation scan, QA, graph, effectiveness claim, or rerank tuning.
- No model config or API key is used by this audit.

## Summary
- Pages/docs audited: 10 / 9
- Total artifacts: 134
- Artifacts/page: 13.4
- R037 delta artifacts/page: 13.8
- Artifact growth vs R037: 0.971014
- Eligible atomic artifacts: 134
- Duplicate atomic keys: 0
- Noise failure pages: 0
- Type counts: `{"numeric_fact": 67, "table_cell": 67}`
- Quality counts: `{"atomic_numeric_ok": 134}`

## Checks
- `no_provider_calls`: True
- `not_merged_into_cumulative_artifacts`: True
- `no_artifacts_jsonl_written`: True
- `artifact_budget_per_page_respected`: True
- `no_quality_noise_flags`: True
- `artifact_growth_vs_r037_within_limit`: True
- `all_pages_have_text`: True

## Page Results
- 2310.09158v1.pdf#p007: artifacts=16, eligible=16, flags=[]
- 2401.18059v1.pdf#p006: artifacts=16, eligible=16, flags=[]
- 936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003: artifacts=16, eligible=16, flags=[]
- BESTBUY_2023_10K.pdf#p026: artifacts=16, eligible=16, flags=[]
- BESTBUY_2023_10K.pdf#p027: artifacts=14, eligible=14, flags=[]
- PIP_Seniors-and-Tech-Use_040314.pdf#p008: artifacts=16, eligible=16, flags=[]
- PS_2018.01.09_STEM_FINAL.pdf#p028: artifacts=16, eligible=16, flags=[]
- STEPBACK.pdf#p007: artifacts=16, eligible=16, flags=[]
- afe620b9beac86c1027b96d31d396407.pdf#p003: artifacts=0, eligible=0, flags=[]
- q1-2023-bilibili-inc-investor-presentation.pdf#p023: artifacts=8, eligible=8, flags=[]

## Next Step
Run a very small repaired provider gate on 2-3 R028 pages before any full 20 -> 30 replay.
