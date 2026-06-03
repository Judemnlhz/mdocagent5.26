# R038b Small Repaired Provider Gate

Decision: `proceed_to_repaired_20_to_30_full_replay_gate`

## Scope
- Tiny real-provider Stage 2 compile on 3 selected R028 failed pages.
- No artifact-store merge, activation scan, QA, graph, effectiveness claim, or rerank tuning.
- Model key remains environment-only; no config or key file is written by this runner.

## Selected Pages
- 2401.18059v1.pdf#p006
- 936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003
- BESTBUY_2023_10K.pdf#p026

## Provider Quality
- Provider success/fail: 3 / 0
- JSON parse success: 3
- Parse failures: 0
- Valid/discarded artifacts: 59 / 2

## Artifact Quality
- Total artifacts: 59
- Atomic strong eligible: 48
- Type counts: `{"figure": 1, "numeric_fact": 24, "table": 2, "table_cell": 24, "text_span": 8}`
- Quality counts: `{"atomic_numeric_ok": 48}`
- Page counts: `{"2401.18059v1.pdf#p006": 17, "936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003": 16, "BESTBUY_2023_10K.pdf#p026": 26}`
- R038a expected artifacts for selected pages: 48
- Artifact growth vs R038a: 1.229167

## Checks
- `provider_success_all_pages`: True
- `parse_failure_zero`: True
- `mock_or_placeholder_zero`: True
- `full_page_only_locator_zero`: True
- `broad_table_only_zero`: True
- `atomic_artifact_present`: True
- `artifact_growth_vs_r038a_bounded`: True
- `no_activation_scan`: True
- `not_merged_into_cumulative_artifacts`: True

## Next Step
Run the repaired 20 -> 30 full replay gate only; still do not run QA/effectiveness until that gate passes.
