
## 2026-06-02 Stage 2 real structured artifact gate

- ARIS project records (`refine-logs/EXPERIMENT_PLAN.md`, `docs/mdocagent_integration_plan.md`) confirm the current bottleneck is artifact coverage/quality, not rerank tuning or full ablation.
- Existing 450-artifact coverage output is fake/mock; eligibility audit correctly rejects it (`mock_or_placeholder_content`), so it must not drive QA claims.
- Added a structured real small-sample subset at `outputs/subsets/stage2_structured_real_subset.jsonl` selecting 10 table/chart/numeric-heavy pages without answer/evidence fields.
- Strengthened document-generic Stage 2 prompt to prefer usable `table_cell`, `numeric_fact`, `figure`, `caption`, and `table` artifacts with structured `normalized_content` and non-mock content.
- Added eligibility audit gate fields: `strong_eligible_artifacts`, `mock_or_placeholder_content`, `full_page_only_locator`, `missing_strong_locator`.
- Remote shell currently has no `SILICONFLOW_API_KEY`, so real-provider compile was not run. Fake smoke at `outputs/stage2_structured_real_smoke_fake/` verifies the pipeline and confirms fake artifacts remain ineligible.
- Next action: run real `doc-compile` on the fixed 10-page subset, then eligibility audit. Only if `strong_eligible_artifacts > 0` and `mock_or_placeholder_content = 0`, build the activation-rich held-out subset and run the top-4 effectiveness gate.
