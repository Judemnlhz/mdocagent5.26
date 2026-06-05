# Experiment Plan

**Problem**: Connect MDocNexus artifact-aware retrieval back into the original MDocAgent top-4 QA path without changing answer generation, prompts, models, or page budget.
**Method Thesis**: A document-generic artifact store can improve MDocAgent by reranking or selecting the same top-4 page budget using typed, provenance-preserving evidence artifacts; if graph context adds no QA gain, the failure should be reported as an evidence-coverage and graph-structure limitation rather than hidden by larger budgets.
**Date**: 2026-06-01

## Source Notes Incorporated

This plan incorporates the two local analysis notes provided by the user:

- `F:\网页下载\ChatGPT-Mdocagent方向分析与评估 (2).md`
- `F:\网页下载\ChatGPT-代码改进评估与建议 (2).md`

Important constraints from those notes and the current repository:

- Do not frame the paper as "MDocAgent + GraphRAG + refusal". The near-term testable claim is a controlled MDocAgent module improvement.
- The main QA path must return to `scripts/predict.py -> BaseDataset -> MDocAgent.predict_dataset() -> MDocAgent.predict() -> scripts/eval.py`.
- Official reproduction is top-1 and top-4 only. Top-8/top-10/top-20 are additional budget diagnostics, not official reproduction.
- MDocNexus Stage 2/3/4 is currently a provenance-safe prototype. It is not yet enough to claim proof-oriented reasoning, calibrated refusal, or final evidence-graph reasoning.
- Current weaknesses to audit explicitly: low artifact coverage, incomplete real multimodal validation, possible gold-field exposure if public inputs are not isolated, weak graph edge semantics, and graph expansion showing little or no retrieval gain.

## Current State Snapshot

Historical MDocAgent results already present in the repo:

| Setting | Result file | Average Binary Correctness | Role |
|---|---|---:|---|
| MDocAgent top-1-like default | `results/MMLongBench/mmlb-MDocAgent/results.txt` | 0.459 | Reference only until rerun on current commit |
| MDocAgent top-4 | `results/MMLongBench/mmlb-MDocAgent-top4/results.txt` | 0.493 | Reference only until rerun on current commit |

Current MDocNexus artifacts and diagnostics:

| Artifact | Current observation | Implication |
|---|---|---|
| `outputs/stage2_doc/quality_report.json` | 20 artifacts from 5 docs / 10 pages in default doc compile output | Too small for main QA claims |
| `outputs/stage3_doc_artifact_retrieval/quality_report.json` | 33 / 1073 queries have doc artifacts; 1040 zero-hit queries | Artifact coverage is the first gate |
| `outputs/experiments/matrix/summary_matrix.md` | Capped matrix reaches 450 artifacts but still has many zero-hit queries | Good diagnostic, not a final QA result |
| `outputs/eval/stage4_graph_expansion_eval/report.json` | graph delta recall@5 and coverage@5 are 0 in default eval | Graph context may be a failure-analysis module unless improved |
| `outputs/experiments/mdocagent_module_ablation/summary.md` | top-4 adapter runs are prepared, not executed | The next paper-relevant experiment is QA execution |

Environment risks observed:

- `python3 scripts/predict.py --help` with system Python fails on missing `PIL`; run QA through `/opt/conda` and the `mdocagent` conda environment:

```bash
source /opt/conda/bin/activate mdocagent
```

- `nvidia-smi` reports NVML initialization failure on the remote host. The current MDocAgent path uses API models, but extraction/retrieval/runtime checks should still record GPU status.
- All API keys must remain in environment variables or private local config, never in manifests or logs.

## 2026-06-02 Pivot: Stage 2 Real Structured Artifact Extraction

The next paper-relevant bottleneck is no longer rerank policy or full ablation. The current strong-artifact gate reports:

- `strong eligible artifacts = 0 / 450` from `outputs/experiments/mdocagent_module_ablation/run_tags/strong_artifact_eligibility_scan_full/strong_artifact_method_gate_summary.json`.
- `activated records = 0 / 1073`; therefore no held-out activation-rich subset can be built yet.
- Existing fake structured smoke output still has `mock_or_placeholder_content > 0`, so it cannot support a method claim.

Current priority order is frozen as:

1. Build a 5-10 page real Stage 2 structured subset biased toward tables, charts, numeric facts, percentages, and paper/brochure/report pages.
2. Compile only real, non-mock structured artifacts: `table_cell`, `numeric_fact`, `figure`, `caption`, and `table`, with `normalized_content` and element locators such as `table_cell`, `text_offset`, `bbox`, `caption_block`, or `figure_region`.
3. Run eligibility audit only, not QA: require `eligible_artifacts > 0`, `eligible_pages > 0`, `mock_or_placeholder_content = 0`, and low `full_page_only_locator`.
4. Only after strong eligible artifacts activate retrieval records, build a new held-out activation-rich subset of 30-50 records. Do not reuse the previous policy-tuning top30 as the main validation set.
5. Only then run the limited effectiveness gate: `top4_original_only`, `top4_original_plus_artifact`, and `artifact_only` diagnostic. Do not run rerank tuning, graph/full ablation, or full QA before Stage 2 passes.

Implemented gate artifacts:

- Runner: `scripts/run_stage2_real_structured_gate.py`.
- Unified CLI: `python3 scripts/mdocnexus.py stage2-real-structured-gate ...`.
- Selected subset: `outputs/subsets/stage2_structured_real_gate_subset.jsonl`.
- Gate report: `outputs/stage2_structured_real_gate/structured_gate_report.json`.

Latest gate result: `stage2_structured_gate_pass`. The gate selected 10 structured pages from 9 documents, made 10 real provider calls, parsed 26 real artifacts, and eligibility audit found 4 strong eligible table artifacts across 2 pages with `mock_or_placeholder_content = 0`.

Decision rule:

- If the rerun still gives `eligible_artifacts = 0`, downgrade the paper direction to a failure-informed framework around artifact coverage and structured extraction limits.
- If it produces stable strong eligible artifacts, proceed to held-out activation-rich subset construction and only then the limited effectiveness gate.


### 2026-06-02 Activation Scan After Real Structured Gate

The real Stage 2 structured gate passed the minimum artifact-quality check, but it does not yet support a 30-50 record held-out effectiveness subset.

Activation scan output:

- Report: `outputs/experiments/mdocagent_module_ablation/run_tags/real_structured_activation_scan/real_structured_activation_scan_report.json`.
- Strong eligible artifact pages: 2.
- Activated records: 6 / 1073.
- Eligible for held-out after excluding prior policy top30: 6.
- `artifact_only` changed records: 6.
- `original_plus_artifact` changed records: 6.
- Concentration: 5 / 6 activated records are from `2023.acl-long.386.pdf#p007`; 1 / 6 is from `3M_2018_10K.pdf#p020`.

Decision: do not run the effectiveness gate yet. The held-out activation-rich subset is unavailable because activated records are below the 30-record minimum and are too concentrated. The next work item is to expand real Stage 2 structured coverage and reduce parse failures before returning to held-out subset construction.

Immediate Stage 2 targets:

- Increase real compiled structured pages beyond the current 10-page smoke, still as a bounded quality run rather than full ablation.
- Prioritize documents/pages likely to produce `table_cell`, `numeric_fact`, caption, and table artifacts across more documents.
- Fix provider parse failures observed on 5 / 10 pages, or add a robust JSON repair/parser path that remains public-safe and does not invent artifacts.
- Re-run eligibility and activation scan; only build held-out subset if `activated_count >= 30` and concentration is acceptable.

### 2026-06-02 R028 Bounded Expansion Outcome

Bounded expansion follows `10 -> 20 -> 30`, and only continues when the stop/go metrics remain clean:

- `parse_success_rate` does not degrade.
- `strong_eligible_artifacts` increases.
- `eligible_pages` increases.
- `activated_count` increases.
- `mock_or_placeholder_content` remains 0.

R028 `10 -> 20` passed: parse success improved from 0.5 to 0.7, strong eligible artifacts increased from 4 to 21, eligible pages increased from 2 to 9, activated records increased from 6 to 30, and mock/placeholder content stayed 0.

R028 `20 -> 30` was stopped early after 6 / 10 delta pages. The partial run had 2 successes and 4 parse failures, so the current partial parse success rate was 0.333. Even if the 4 remaining pages all succeeded, the final rate could only reach 0.600, below the previous increment's 0.700. Report: `outputs/stage2_structured_incremental/r028_20_to_30/partial_stop_report.md`.

Decision: do not continue expansion, do not construct held-out subset from the partial 20 -> 30 run, and do not run the effectiveness gate. The next action is parse-failure auditing and repair. Concentration metrics remain external-validity diagnostics only and are not used for reranking, scoring, or gold-field selection.

Parse-failure repair outcome: Stage 2 now records provider parse-failure taxonomy without public raw responses: failure type, raw response length/hash, JSON-like block presence, and schema missing fields when inferable. The provider prompt was tightened to require strict JSON only, the parser now extracts fenced JSON and JSON objects surrounded by prose, and the compiler conservatively wraps a single returned artifact object into a PageArtifactOutput container before normal validation. No artifact content is invented; invalid or incomplete artifacts still go through deterministic discard.

Bounded replay on exactly 3 previously failed pages passed the parser repair check: provider_call_success_count=3, json_parse_success_count=3, parse_failure_count=0, num_valid_artifacts=6, num_discarded_artifacts=0. Report: `outputs/stage2_structured_incremental/r028_20_to_30/parse_repair_replay_3_wrapped/parse_repair_replay_report.md`. This replay is not a Stage 2 expansion, is not merged into cumulative artifacts, and is not a QA/held-out/effectiveness result.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Artifact-aware top-4 page reranking improves MDocAgent under the same model and page budget. | This is the narrow, defensible contribution that connects MDocNexus back to the original MDocAgent QA system. | `top4_original_plus_artifact` beats current-commit MDocAgent top-4 and `top4_original_only`; all use the same Qwen/Qwen3 and Qwen3-VL answer models, the same top-4 page budget, and the same evaluation model. | B1, B2, B3 |
| C2: Any gain is not explained by stronger models, larger page budgets, gold leakage, or hand-written question heuristics. | This is the main reviewer attack. | Manifests show same model configs, top-k=4, no debug/semantic edges, no gold fields in retrieval inputs, and `stage3_artifact_answer_smoke.py` is excluded from QA. Optional top-8/top-10 diagnostics are labeled budget diagnostics only. | B1, B2, B4 |
| C3: Graph-guided context selection is useful only if formal rule-only edges change page selection in ways that improve QA or retrieval coverage. | The current graph module is clean but weak; it should not be oversold. | `top4_graph_context` improves over `top4_original_plus_artifact` or supplies a clear negative result with edge-ablation and failure categories. | B3, B4, B5 |

Anti-claims to rule out:

- The gain comes from reading more pages.
- The gain comes from a stronger answer or evaluation model.
- The gain comes from gold answers, gold evidence pages, or debug edges.
- The method is only a small smoke test with too few artifact-covered queries.
- The graph is just a bookkeeping graph and not useful for context selection.

## Paper Storyline

Main paper must prove:

- MDocNexus can be connected to MDocAgent as a controlled retrieval-record adapter, not as a separate answer-generation system.
- Artifact-aware reranking changes top-4 page selection while preserving the original MDocAgent answer path.
- The resulting QA accuracy and failure patterns justify artifact-aware retrieval as a real module improvement.

Appendix can support:

- Larger top-k budget diagnostics.
- Lambda-weight sweeps for `original_plus_artifact`.
- Retrieval-only Stage 3/4 reports and graph edge ablations.
- Cost and artifact coverage reports.

Experiments intentionally cut from this round:

- Proof trace generation.
- Calibrated refusal or selective prediction.
- Artifact context augmentation inside prompts.
- LLM-generated semantic graph edges.
- Hand-crafted `artifact_answer_smoke.py` QA accuracy.

## Experiment Blocks

### Block 1: Reproducibility and Leakage Gate

- Claim tested: C2.
- Why this block exists: It prevents the main result from being invalidated by environment drift, model mismatch, or gold leakage.
- Dataset / split / task: MMLongBench, current repository data in `data/MMLongBench`.
- Compared systems: No model comparison; this is an audit gate.
- Metrics:
  - `audit --all` pass/fail.
  - model role status pass/fail.
  - no-gold leakage pass/fail.
  - reproducibility manifest completeness.
  - adapter manifest hash and retrieval-record hash recorded.
- Setup details:
  - Activate the project environment before any QA run:

```bash
source /opt/conda/bin/activate mdocagent
```

  - Use `SILICONFLOW_API_KEY` from environment.
  - Run:

```bash
cd /home/lhz/MDocAgent
python3 scripts/mdocnexus.py audit --all
python3 scripts/run_mdocagent_module_ablation.py
```

- Success criterion:
  - Audits pass.
  - The prepared adapter runs remain `prepared_not_run`.
  - Manifests record `same_model_as_baseline=true`, `same_page_budget_as_baseline=true`, `no_gold_fields_used=true`, `used_debug_edges=false`, `used_semantic_edges=false`.
- Failure interpretation:
  - Any audit failure blocks QA execution.
  - Missing Python packages or env mismatch must be fixed before runs, not hidden in the paper.
- Table / figure target: Appendix reproducibility table.
- Priority: MUST-RUN.

### Block 2: Artifact Coverage Gate

- Claim tested: C1, C2.
- Why this block exists: Artifact-aware top-4 QA cannot be meaningful if most candidate pages have no artifacts.
- Dataset / split / task: MMLongBench top-4/top-10 candidate retrieval pages from `data/MMLongBench/sample-with-retrieval-results.json`.
- Compared systems:
  - Current default artifact store.
  - Capped coverage artifact store.
  - Full or expanded top-4 candidate-page artifact store if budget allows.
- Metrics:
  - query coverage: percentage of QA records with at least one artifact in candidate pages.
  - page coverage: percentage of unique candidate pages with artifacts.
  - artifact type distribution.
  - visual artifact rate and image payload audit status.
  - element locator coverage.
  - zero-hit query count.
- Setup details:
  - Use document-generic Stage 2 outputs only.
  - Public query inputs must exclude answer/evidence fields.
  - Real-provider smoke must remain small before any large API run.
  - If using fake provider outputs, label them as pipeline diagnostics only.
- Success criterion:
  - Minimum for QA pilot: at least 20 percent query coverage in the selected pilot subset.
  - Minimum for paper main run: target 60 percent or higher query coverage over the official evaluation set or a clearly declared fixed subset.
  - Real multimodal payload audit passes on a small smoke run.
- Failure interpretation:
  - If coverage stays below 20 percent, do not run expensive full QA; write failure analysis around artifact coverage and compile scope.
  - If visual artifact rate remains low, claim only text/layout artifact reranking, not multimodal evidence reasoning.
- Table / figure target:
  - Table A: artifact coverage and type distribution.
  - Figure A: coverage by query type / document type if available.
- Priority: MUST-RUN before full QA.

### Block 3: Official Top-4 QA Reproduction and Adapter Sanity

- Claim tested: C1, C2.
- Why this block exists: The adapter must preserve MDocAgent behavior when run in `original_only` mode.
- Dataset / split / task: MMLongBench QA.
- Compared systems:
  - `mdocagent_top1_official_reproduction`.
  - `mdocagent_top4_official_reproduction`.
  - `top4_original_only`.
- Metrics:
  - Average Binary Correctness from `scripts/eval.py`.
  - prediction completion rate.
  - API error/retry count.
  - page overlap between official top-4 and adapter original-only top-4.
- Setup details:
  - Run in screen or another durable remote session.
  - Use current commit, current model configs, and `dataset.top_k=4`.
  - Historical top-4 score 0.493 is reference only; the main baseline must be current-commit.
- Success criterion:
  - `top4_original_only` matches official top-4 retrieval records exactly or near-exactly.
  - QA score is within normal API stochastic variance of current-commit top-4.
- Failure interpretation:
  - If original-only adapter diverges, fix adapter before testing artifact variants.
  - If official reproduction is far below 0.493, diagnose environment/API/model drift before claiming method failure.
- Table / figure target: Main Table 1, rows 1-3.
- Priority: MUST-RUN.

### Block 4: Top-4 Module Ablation QA

- Claim tested: C1, C3.
- Why this block exists: This is the core paper experiment.
- Dataset / split / task: MMLongBench QA, same top-4 page budget.
- Compared systems:
  - `top4_original_only`.
  - `top4_artifact_only`.
  - `top4_original_plus_artifact` with lambda=0.5.
  - `top4_graph_context` with `page_neighborhood`.
- Metrics:
  - Average Binary Correctness.
  - delta vs `top4_original_only`.
  - delta vs current-commit official top-4.
  - changed-page rate.
  - artifact-covered changed-page rate.
  - graph-added-page rate for `top4_graph_context`.
  - same model/page-budget flags from manifests.
- Setup details:
  - Use:

```bash
cd /home/lhz/MDocAgent
python3 scripts/run_mdocagent_module_ablation.py --execute
```

  - If the full script is too expensive, run prepared manifest commands in this order:
    1. top-4 official reproduction.
    2. `top4_original_only`.
    3. `top4_original_plus_artifact`.
    4. `top4_artifact_only`.
    5. `top4_graph_context`.
- Success criterion:
  - Main success: `top4_original_plus_artifact` improves over `top4_original_only` by at least +0.02 absolute binary correctness or shows a statistically credible positive difference on the executed set.
  - Secondary success: `top4_artifact_only` is competitive enough to show artifacts contain useful signal, but it need not beat the combined method.
  - Graph success: `top4_graph_context` improves over flat artifact selection or produces a clear failure diagnosis.
- Failure interpretation:
  - If artifact-only hurts and original-plus-artifact helps, artifacts are useful only as a prior, not a replacement for original retrieval.
  - If all artifact variants are flat or worse, diagnose coverage, artifact quality, and reranking calibration before moving to proof/refusal.
  - If graph context is flat, report graph edges as clean but insufficient for QA selection, and keep graph as a future module.
- Table / figure target:
  - Main Table 1: QA results.
  - Main Figure 2: top-4 page selection changes and artifact coverage.
- Priority: MUST-RUN.

### Block 5: Failure Analysis and Reviewer Counterfactuals

- Claim tested: C2, C3.
- Why this block exists: A paper can still be credible if it explains where artifacts help and where they fail.
- Dataset / split / task: All completed QA outputs from Block 3 and Block 4.
- Compared systems:
  - baseline correct -> method wrong.
  - baseline wrong -> method correct.
  - both wrong.
  - graph-context changed vs unchanged.
- Metrics:
  - error bucket counts.
  - no-artifact failure rate.
  - wrong-artifact failure rate.
  - retrieval changed-to-wrong rate.
  - retrieval changed-to-right rate.
  - visual/table/numeric question performance if labels can be derived without gold leakage.
  - evaluator disagreement or unparseable answer count.
- Setup details:
  - Do not use gold fields to select examples before evaluation.
  - After evaluation, use gold only for analysis labels.
  - Sample examples from predefined buckets, not hand-picked wins.
- Success criterion:
  - At least one robust positive bucket is identified, or the plan records why artifact-aware retrieval is not yet publishable.
  - Failure categories directly map to next engineering tasks: coverage, locator, visual payload, lambda calibration, graph edge types.
- Failure interpretation:
  - If no positive bucket exists, do not write the paper as a performance paper; write a prototype/diagnostic report or return to Stage 2 coverage.
- Table / figure target:
  - Table 2: failure taxonomy.
  - Figure 3: changed-page wins/losses.
- Priority: MUST-RUN.

## Nice-to-Have Diagnostics

These are not official reproduction rows.

| Diagnostic | Purpose | Run only if |
|---|---|---|
| MDocAgent top-8/top-10/top-20 QA | Rule out "just read more pages" | top-4 artifact result is positive and budget allows |
| lambda sweep: 0.25 / 0.5 / 0.75 | Check whether artifact score is over-weighted | `original_plus_artifact` changes many pages |
| graph expansion edge ablation | Explain why graph context helps or fails | `top4_graph_context` changes pages |
| second dataset small run: PaperTab or FetaTab | Check transfer beyond MMLongBench | main MMLongBench result is positive |

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Environment and audit gate | `audit --all`, dry-run module ablation | All audits pass; Python env supports `PIL`, `pymupdf`, Hydra dependencies | Low | Env drift blocks QA |
| M1 | Artifact coverage gate | Stage 2/3 coverage reports, real smoke audit if needed | Query coverage sufficient for pilot or full run | Low to medium API cost | Low coverage invalidates QA |
| M2 | Official reproduction | top-1/top-4 official, top-4 original-only adapter | Adapter sanity passes | Medium API cost | API stochasticity or previous baseline drift |
| M3 | Main module ablation | artifact-only, original-plus-artifact, graph-context | At least one artifact variant gives useful signal | Medium to high API cost | No positive signal |
| M4 | Reviewer counterfactuals | budget diagnostics, lambda sweep, graph ablation | Anti-claims ruled out | Optional high cost | Too many diagnostics dilute story |
| M5 | Failure analysis package | bucketed errors, coverage/cost tables | Paper has defensible narrative | Low | Cherry-picking risk |

## Compute and Data Budget

- Primary data: MMLongBench from the existing `data/MMLongBench` directory.
- Main QA runs: 5 to 6 MDocAgent runs if using the prepared module-ablation set.
- Main API dependency: SiliconFlow API for Qwen3/Qwen3-VL and DeepSeek-V3 evaluation.
- GPU dependency: not central for API inference, but GPU status should be recorded because retrieval/extraction tooling may use CUDA.
- Largest likely cost: repeated MDocAgent API inference, not deterministic Stage 3/4 diagnostics.
- Stop rule: do not launch all module QA runs if Block 1 or Block 2 fails.

## Risks and Mitigations

- Risk: artifact coverage is too low.
  - Mitigation: expand document-generic compile scope over top-4/top-10 candidate pages before QA, or restrict the first QA pilot to a fixed coverage subset and label it as pilot.
- Risk: visual artifacts are not truly multimodal.
  - Mitigation: run real-provider smoke audit and report visual/table/caption artifact rates separately from generic anchoring.
- Risk: adapter original-only differs from official top-4.
  - Mitigation: block artifact QA until retrieval-record schema parity is fixed.
- Risk: graph context has no gain.
  - Mitigation: treat graph as failure analysis unless edge ablation shows a useful formal-edge family.
- Risk: reviewer says gains come from extra pages.
  - Mitigation: keep top-4 fixed for main comparisons; label top-8/top-10/top-20 as diagnostics only.
- Risk: reviewer says gains come from stronger models.
  - Mitigation: use identical Qwen/Qwen3-VL models and DeepSeek evaluator across rows; record config hashes.
- Risk: reviewer suspects leakage.
  - Mitigation: public query inputs exclude answer/evidence fields; audit manifests show no gold fields, no debug edges, no semantic edges.

## Final Checklist

- [ ] Main paper tables are covered by current-commit top-4 QA results.
- [ ] Adapter original-only sanity is passed.
- [ ] Artifact-aware reranking is evaluated under the same top-4 budget.
- [ ] Graph-guided selection is evaluated honestly, including negative outcome.
- [ ] Gold leakage and debug-edge leakage audits pass.
- [ ] Stage 3 answer smoke is excluded from formal QA.
- [ ] Larger top-k runs are labeled diagnostic only.
- [ ] Failure analysis has predefined buckets and no cherry-picked examples.

## R028 Parse Repair and Atomic Artifact Quality Update

R028 bounded expansion remains stopped at the 20 -> 30 partial run. The stop reason is unchanged: after 6 / 10 delta pages, parse success was 2 / 6 and the best possible final success rate could not remain non-decreasing relative to the 10 -> 20 run. Do not expand Stage 2 pages from this branch.

Post-stop repair work is limited to the same failed-page probe and is not merged into cumulative artifacts:

- Parser repair replay on 3 failed pages: parse failures = 0, valid artifacts = 6, discarded artifacts = 0, strong eligible artifacts = 5, eligible pages = 2, mock content = 0. Quality inspection found zero atomic artifacts and mostly broad table/figure blobs.
- Atomic prompt replay on the same 3 pages: parse failures = 0, valid artifacts = 10, discarded artifacts = 0, strong eligible artifacts = 9, eligible pages = 2, mock content = 0, full-page-only locators = 0, table_cell artifacts = 6, numeric_fact artifacts = 0.

Decision: atomic prompting improves artifact structure but is not stable enough for activation scan or QA. Continue bounded prompt/parser repair on the same probe pages, focusing on numeric_fact extraction and consistent atomic values across financial/table pages. Raw provider responses remain private; public reports contain parsed summaries and hash/length style diagnostics only.

Current reports:

- `outputs/stage2_structured_incremental/r028_20_to_30/parse_repair_replay_3_wrapped/parse_repair_replay_report.md`
- `outputs/stage2_structured_incremental/r028_20_to_30/parse_repair_replay_3_wrapped/artifact_quality_inspection.md`
- `outputs/stage2_structured_incremental/r028_20_to_30/atomic_prompt_replay_3/atomic_prompt_quality_report.md`

### 2026-06-03 R030 Same-Page Atomic Quality Repair

R030 keeps the ARIS/refine constraint from R028/R029: bounded repair on the same 3 failed pages only, no Stage 2 expansion, no activation scan, no QA, no graph, no rerank tuning, and replay outputs are not merged into cumulative artifacts.

Implemented repair:

- Added a Stage 2-only artifact quality taxonomy: `atomic_numeric_ok`, `broad_table_only`, `missing_numeric_fact`, `weak_locator`, `caption_or_table_title_only`, and `schema_valid_but_semantically_weak`.
- Split locator eligibility from atomic evidence eligibility in the eligibility audit. The audit now reports `atomic_strong_eligible_artifacts`, `numeric_fact_count`, `table_cell_count`, `broad_table_only_count`, and `eligible_pages_with_atomic_artifact`.
- Tightened the document-generic Stage 2 prompt to require numeric/table extraction before compact table descriptors, especially for financial/performance/percentage pages.
- Added deterministic OCR-text numeric fallback for explicit financial/performance table lines, producing paired `numeric_fact` and `table_cell` artifacts with row label, column label, value, unit, normalized value when possible, and `source_block`/`text_offset` locators.
- Broad/table-title-only schema-valid artifacts are discarded from the Stage 2 artifact store or excluded from atomic strong eligibility.

Bounded replay result on the same 3 pages:

- parse failures = 0
- JSON parse successes = 3
- mock/placeholder content = 0
- full-page-only locators = 0
- broad table only = 0, down from 2 in R029
- table_cell artifacts = 16, up from 6 in R029
- numeric_fact artifacts = 16, up from 0 in R029
- strong eligible artifacts = 32
- atomic strong eligible artifacts = 32, up from 6 under the new taxonomy baseline
- eligible pages = 2, not decreased
- eligible pages with atomic artifact = 2, up from 1

Decision: R030 passes the same-page bounded artifact quality gate and upgrades strong eligibility from "can be located" toward "can be used as structured numeric evidence" on this probe. This is still not an activation or QA result. The next step may be an activation scan only after reviewing whether the deterministic OCR fallback should remain enabled for the broader Stage 2 run.

Current R030 reports:

- `outputs/stage2_structured_incremental/r028_20_to_30/r030_atomic_quality_replay_3/atomic_quality_report.md`
- `outputs/stage2_structured_incremental/r028_20_to_30/r030_atomic_quality_replay_3/eligibility_audit.md`

### 2026-06-03 R031 Bounded Activation Scan Review

R031 is diagnostic only. It does not run QA, graph expansion, rerank tuning, or an effectiveness gate. It temporarily constructs `cumulative20 + R030 repaired 3 pages` and keeps the result outside cumulative artifacts.

Boundary audit:

- R030 deterministic OCR numeric fallback passes the public-safe boundary check.
- It is gated to document-generic Stage 2 mode.
- It reads `page_text` OCR and layout locators only.
- It uses `selected_page` only for `doc_id` / `page_index` identity.
- It does not read question, answer, gold fields, evidence pages, evidence sources, or binary correctness.

R031 ran two activation-scan views:

1. `merged_all`: all artifacts from the temporary `cumulative20 + R030` store.
2. `atomic_only`: only artifacts passing the new atomic strong eligibility taxonomy.

Results:

| View | Activated | Eligible held-out | Changed artifact_only | Changed original_plus | Strong pages | Max doc share | Max page share |
|---|---:|---:|---:|---:|---:|---:|---:|
| merged_all | 36 | 36 | 42 | 29 | 11 | 0.2222 | 0.1944 |
| atomic_only | 6 | 6 | 6 | 4 | 2 | 0.5000 | 0.5000 |

Interpretation:

- The merged-all store appears to reach the nominal activation threshold, but it still includes non-atomic eligible artifacts such as broad/table descriptor artifacts from the earlier cumulative run.
- The stricter atomic-only view shows that genuinely structured numeric evidence is still limited to 2 pages and only activates 6 records.
- Therefore R031 does not justify repaired 20 -> 30 expansion, activation-rich held-out construction, or QA/effectiveness evaluation.

Decision: continue Stage 2 coverage/quality repair. The next repair should expand atomic numeric coverage across more pages/documents before rerunning activation diagnostics. Do not run R032 limited effectiveness gate yet.

Current R031 reports:

- `outputs/stage2_structured_incremental/r031_activation_scan_review/r031_activation_scan_review_report.md`
- `outputs/stage2_structured_incremental/r031_activation_scan_review/fallback_boundary_audit.md`
- `outputs/stage2_structured_incremental/r031_activation_scan_review/atomic_only/eligibility_audit.md`
- `outputs/stage2_structured_incremental/r031_activation_scan_review/activation_scan_atomic/real_structured_activation_scan_report.md`


## ARIS Skill Execution Protocol for Future Experiments

This section is the binding workflow for future MDocAgent/MDocNexus experiments. It is derived from the project-local ARIS skills under `.agents/skills/`, especially `experiment-plan`, `run-experiment`, `monitor-experiment`, and `experiment-audit`. Do not treat `.agents/skills` as temporary output; the symlinks are the skill installation.

### Required Skill Chain

1. `experiment-plan`: freeze the claim, anti-claim, run order, stop/go gates, and tracker rows before any new experiment is launched.
2. `run-experiment`: when the user explicitly says to run, execute pre-flight checks first, then launch with `screen` on the target host.
3. `monitor-experiment`: after launch, inspect `screen`, tee logs, result files, and raw numbers before interpretation.
4. `experiment-audit`: after results are complete and before paper claims, audit result provenance, scope, metric computation, and file existence.
5. `result-to-claim` or paper-writing skills may only use results that passed the tracker boundary and are labeled with scope: official, diagnostic, or pilot.

### Current Paper Claims to Defend

| Claim | Status | Evidence Needed Before Claiming |
|---|---|---|
| C1: Artifact-aware top-4 retrieval improves MDocAgent under the same model and page budget. | Not yet established on full MMLongBench. | Current-commit `top4_original_only`, `top4_original_plus_artifact`, and preferably `top4_artifact_only` results with identical answer/eval models and fixed top-4 budget. |
| C2: Graph-guided retrieval is useful or honestly negative. | Not yet established. | `top4_graph_context` under the same top-4 budget, plus graph-added-page and edge/failure analysis. |
| C3: Any gain is not leakage, budget, or model drift. | Must be audited each time. | ARIS pre-flight audit, no-gold public inputs, model config hashes, adapter sanity, and result existence audit. |

### Run-Experiment Gate

A future full MMLongBench top-4 run must not start until all checks below are recorded in a run note or tracker update:

- `AGENTS.md` read and target confirmed: `mdoc-remote`, `/home/lhz/MDocAgent`, conda env `mdocagent`.
- GPU status checked with `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`; record failure as environment context even if API inference does not need GPU.
- Conda environment activated via `eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate mdocagent`.
- API key presence checked without printing values, especially `SILICONFLOW_API_KEY`.
- Dry-run/import checks pass before screen launch: command help/import, model config audit, and adapter dry-run/audit relevant to the run.
- Launch uses `screen -dmS <run_id> bash -lc '... 2>&1 | tee <log_file>'` and writes logs outside deleted transient Hydra directories.
- The exact command, screen session, log path, output path, model names, top-k, and scope label are recorded before reporting success.

### Execution Order After This Planning Update

1. Do not launch a full QA run automatically. Wait for explicit user confirmation such as "now run R047".
2. First executable run should be an ARIS pre-flight bundle for current-commit MMLongBench top-4 reproduction and adapter sanity.
3. If pre-flight passes, launch official top-4 reproduction or adapter-original-only sanity as the first screen job, not all ablations at once.
4. After each completed run, monitor logs/results, update `EXPERIMENT_TRACKER.md`, run integrity audit when claims will be made, then commit and push results.
5. Only after baseline/original sanity is credible should artifact-aware and graph-guided rows be launched.


### 2026-06-04 R053 Question-Aware Scaffold Plan

R045 completed the manual support/citation-aware rubric over R044 transition cases and found that the next issue is input design, not experiment scale. R053 is therefore a no-provider design gate.

Claim tested: C1/C3 input validity before any new model call. The question is whether R045's fixes can be encoded as a visible-context prompt scaffold without gold leakage.

Compared systems: no score comparison. R053 compares the previous fixed first-N artifact exposure against a question-aware selector design only through prompt/selection previews.

Scope and boundary:

- R045 cases only: transition cases plus the fixed all-miss sample.
- No provider calls, no predictions, no evaluation, no full QA, no official score.
- Artifact selection uses question tokens, code/value/metric matching, artifact type priority, and retrieved candidate pages only.
- Prompt template must separate page evidence and artifact evidence, require page/artifact citations, and include an unsupported-answer guard.
- Gold answers/evidence pages must not appear in public previews or selection inputs.

Success criterion: R053 gate passes and the prompt previews for records 384, 508, and 569 are manually acceptable enough to justify a later tiny provider diagnostic. If selected artifacts still lack necessary evidence, improve artifact selection first. If selected artifacts contain evidence but the prompt remains ambiguous, improve prompt template first. Do not run full QA from R053.


### 2026-06-04 R053 Gate Result

R053 ran as a no-provider scaffold and passed.

Outputs:

- `outputs/heldout/r053_question_aware_scaffold/r053_question_aware_scaffold_report.md`
- `outputs/heldout/r053_question_aware_scaffold/r053_question_aware_scaffold_gate.md`
- `outputs/heldout/r053_question_aware_scaffold/r053_question_aware_compact_index.jsonl`
- `outputs/heldout/r053_question_aware_scaffold/r053_question_aware_prompt_previews.jsonl`

Gate checks passed: no provider calls, no prediction/eval/full QA, target cases match R045, prompt previews require artifact/page citations, page evidence and artifact evidence are separated, unsupported-answer guard is present, selected artifact budget is respected, no public gold fields are present, and the artifact store remains bound to the R038d union atomic store.

Selected artifact counts by key transition record:

- 384: 4 selected artifacts on page 10; R045 warned that artifact snippets may be misleading for the May 2018 question.
- 508: 8 selected artifacts on page 8; R045 marked this as the clearest positive unanswerable-refusal diagnostic.
- 569: 8 selected artifacts on page 28; R045 marked snippet-only refusal as supported insufficient-data behavior.

Decision: do not run full QA. The next step is manual inspection of R053 prompt previews for records 384, 508, and 569. If selected artifacts still do not contain necessary evidence, improve question-aware artifact selection. If selected artifacts contain evidence but the prompt remains underspecified, improve the citation prompt template and then run only a tiny diagnostic provider check.


### 2026-06-04 R054 Guarded Selector Repair

R053 manual inspection found that the question-aware selector was still too permissive on the key transition cases. R054 therefore encodes the user's manual review into a no-provider guard repair rather than launching another experiment.

Manual corrections applied:

- 384: route document metadata/date/producer questions to page-text metadata/refusal; reject irrelevant numeric/table artifacts.
- 508: require exact `AR03` code/key-value support; if no exact artifact exists, use page evidence for absence/unsupported refusal and reject numeric noise.
- 569: require operand completeness before any calculation; if snippets do not cover all operands, force an insufficient-data/refusal guard.

Scope and boundary:

- R045 cases only; no new provider calls, no prediction, no evaluation, no full QA, no official score.
- R054 is a selector/prompt guard gate, not a performance result.
- Public previews remain no-gold: no answer, gold evidence page, or official correctness fields are exposed.

Outputs:

- `outputs/heldout/r054_guarded_selector_repair/r054_guarded_selector_repair_report.md`
- `outputs/heldout/r054_guarded_selector_repair/r054_guarded_selector_repair_gate.md`
- `outputs/heldout/r054_guarded_selector_repair/r054_guarded_compact_index.jsonl`
- `outputs/heldout/r054_guarded_selector_repair/r054_guarded_prompt_previews.jsonl`

Gate result: passed. The hard checks confirmed `r384_metadata_refusal_guard`, `r384_no_numeric_table_artifacts_selected`, `r508_exact_code_or_absence_guard`, `r508_no_artifact_without_exact_ar03`, and `r569_operand_completeness_guard`.

Key repaired decisions:

- 384: `document_metadata_refusal_guard`, selected artifacts = 0.
- 508: `exact_code_absence_guard`, selected artifacts = 0.
- 569: `operand_completeness_guard`, selected artifacts = 0.

Decision: do not run full QA. The next valid step is manual acceptance of R054 prompt previews for 384/508/569. If accepted, run only a tiny provider diagnostic on guarded prompts and label it diagnostic attribution, not an official MMLongBench result.


### 2026-06-04 R055 Guarded Prompt Provider Diagnostic

R055 ran only after manual acceptance of the R054 prompt previews for records 384, 508, and 569. The purpose is deliberately narrow: test whether the R054 guard prompt makes the provider refuse unsupported answers and avoid artifact-noise traps. R055 cannot prove artifact-aware retrieval has positive lift because it has no retrieval-condition comparison and only uses three guarded prompts.

Scope and boundary:

- Target records only: 384, 508, 569.
- Provider calls: 3, model `Qwen/Qwen3-8B` through SiliconFlow-compatible API.
- Uses R054 guarded prompt previews and zero selected artifacts for all three cases.
- Not full QA, not an official MMLongBench score, and not evidence of artifact positive improvement.
- GPU/NVML check failed (`Failed to initialize NVML: Unknown Error`), but this API-only diagnostic did not require GPU.
- `screen` was unavailable on the remote host, so the run used foreground execution with `tee` logging.

Outputs:

- `outputs/heldout/r055_guarded_prompt_provider_diagnostic/r055_guarded_prompt_provider_report.md`
- `outputs/heldout/r055_guarded_prompt_provider_diagnostic/r055_guarded_prompt_provider_gate.md`
- `outputs/heldout/r055_guarded_prompt_provider_diagnostic/predictions/r055_predictions.jsonl`
- `outputs/heldout/r055_guarded_prompt_provider_diagnostic/logs/r055_screen.log`

Gate result: passed. The diagnostic guard behavior checks passed for 3 / 3 cases, reported as a diagnostic count, not a score.

Per-case behavior:

- 384: metadata/refusal guard passed; model did not use the misleading numeric/table artifact path.
- 508: exact-code absence guard passed; model returned Not answerable for missing exact `AR03` support.
- 569: operand-completeness guard passed; model refused the calculation because the children operand was missing.

Decision: R055 supports keeping the R054 guard prompt as a candidate prompt-control component. It does not support any claim that artifacts improve retrieval or QA. The next decision is whether to integrate these guards into the selector/prompt path or run another tiny contrastive diagnostic with explicit positive-evidence cases before broader QA.


### 2026-06-04 R056 Guarded Selector/Prompt Scaffold Audit

R056 does not run an experiment. It extracts the R054/R055 refusal guards into a reusable, auditable scaffold at `mdocnexus.integration.guarded_prompt` and verifies the scaffold with no provider calls.

Purpose:

- Make metadata/refusal, exact-code absence, and operand-completeness guards reusable outside one-off R054 scripts.
- Keep public prompt previews no-gold and provenance-safe.
- Check the guard is not a trivial all-refusal/all-clear strategy by requiring non-refusal positive-signal cases to retain artifacts.

Scope and boundary:

- R045 diagnostic cases only.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Not evidence that artifacts improve QA or retrieval.
- Artifact store remains the R038d union atomic store.

Outputs:

- `mdocnexus/integration/guarded_prompt.py`
- `mdocnexus/integration/tests/test_guarded_prompt.py`
- `scripts/run_r056_guarded_scaffold_audit.py`
- `outputs/heldout/r056_guarded_scaffold_audit/r056_guarded_scaffold_report.md`
- `outputs/heldout/r056_guarded_scaffold_audit/r056_guarded_scaffold_gate.md`
- `outputs/heldout/r056_guarded_scaffold_audit/r056_guarded_compact_index.jsonl`
- `outputs/heldout/r056_guarded_scaffold_audit/r056_guarded_prompt_previews.jsonl`

Gate result: passed.

Key checks:

- 384: `document_metadata_refusal_guard`, selected artifacts = 0.
- 508: `exact_code_absence_guard`, selected artifacts = 0.
- 569: `operand_completeness_guard`, selected artifacts = 0.
- Positive-signal non-refusal records: `[69, 223, 224, 227]`.
- Positive-signal cases cleared: `[]`.
- Public preview forbidden gold fields: 0.

Decision: R056 makes the guard reusable and audit-ready. The next decision is whether to wire `mdocnexus.integration.guarded_prompt` into an actual adapter/prompt path behind an explicit config flag, or first run a tiny positive-evidence diagnostic. Do not run full QA from R056.


### 2026-06-04 R057 Guarded Integration Design Gate

R057 does not run an experiment. It turns the R056 guarded selector/prompt scaffold into an opt-in integration contract with `enable_guarded_prompt_scaffold=false` by default.

Purpose:

- Define how `mdocnexus.integration.guarded_prompt` can be called later without changing the current adapter path by default.
- Prove the default-disabled path leaves adapter records unchanged and generates no prompt previews.
- Prove the enabled design gate only emits prompt previews and a manifest; it still does not call providers, predict, evaluate, or report a score.
- Keep the contract explicit that this is not evidence of artifact-aware retrieval lift.

Scope and boundary:

- R045 diagnostic cases only.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Default config remains disabled: `enable_guarded_prompt_scaffold=false`.
- Enabled mode creates prompt previews and integration manifest only; record hashes remain unchanged.
- Public previews and manifest contain no gold/secret values.

Outputs:

- `mdocnexus/integration/guarded_integration.py`
- `mdocnexus/integration/tests/test_guarded_integration.py`
- `scripts/run_r057_guarded_integration_gate.py`
- `outputs/heldout/r057_guarded_integration_design_gate/r057_guarded_integration_report.md`
- `outputs/heldout/r057_guarded_integration_design_gate/r057_guarded_integration_gate.md`
- `outputs/heldout/r057_guarded_integration_design_gate/r057_integration_contract.json`
- `outputs/heldout/r057_guarded_integration_design_gate/integration_outputs/guarded_prompt_integration_manifest.json`
- `outputs/heldout/r057_guarded_integration_design_gate/integration_outputs/guarded_prompt_integration_previews.jsonl`

Gate result: passed.

Key checks:

- Contract default disabled: true.
- Config flag: `enable_guarded_prompt_scaffold`.
- Disabled records unchanged: true.
- Disabled prompt previews: 0.
- Enabled records unchanged: true.
- Enabled prompt previews: 8.
- Enabled public previews no-gold: true.
- Not artifact-lift claim: true.

Decision: R057 establishes a safe, default-off integration contract. The next step should be either R058 tiny positive-evidence diagnostic, or explicit wiring behind this disabled-by-default flag after review. Do not run full QA from R057.


### 2026-06-04 R058 Positive-Evidence Diagnostic

R058 is a no-provider diagnostic, not a scoring run. Its purpose was to test the missing positive side of R054-R057: whether the guarded selector preserves artifact evidence that is not merely non-empty, but visibly relevant, citable by artifact id, and capable of supporting the public question.

Purpose:

- Audit R056 positive-signal records `[69, 223, 224, 227]`.
- Separate `positive_selection_signal` from actual `artifact_support_sufficient`.
- Require selected artifacts to cover public question dimensions such as entity, metric, time, value, and computation operands where applicable.
- Keep the prompt previews no-gold and public-input-only.

Scope and boundary:

- Target records only: 69, 223, 224, 227.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Does not prove artifact-aware retrieval lift.
- A failed R058 gate means selector repair is needed before any positive-evidence provider run.

Outputs:

- `scripts/run_r058_positive_evidence_diagnostic.py`
- `outputs/heldout/r058_positive_evidence_diagnostic/r058_positive_evidence_report.md`
- `outputs/heldout/r058_positive_evidence_diagnostic/r058_positive_evidence_gate.md`
- `outputs/heldout/r058_positive_evidence_diagnostic/r058_positive_evidence_compact_index.jsonl`
- `outputs/heldout/r058_positive_evidence_diagnostic/r058_positive_evidence_prompt_previews.jsonl`

Decision rule:

- Pass only if all four positive cases have non-empty, citable selected artifacts and the artifact evidence covers the question dimensions needed for answer support.
- If only page evidence covers the question, or artifacts only match loose tokens, do not pass.
- If R058 fails, next step is guarded selector repair: stronger table/key-value dimension matching for demographic, time, metric, value, and operand constraints.

Gate result: failed by design-critical audit.

Key checks:

- Target positive-signal records matched: `[69, 223, 224, 227]`.
- All four cases had non-empty selected artifacts.
- All selected artifacts were citable by artifact id and page id.
- Public previews had no forbidden gold fields.
- Artifact store remained the R038d union atomic store.
- `all_positive_cases_have_supporting_artifact_evidence`: false.

Per-record result:

- 69: selected artifacts mention RAPTOR metrics, but do not cover Figure 4, retrieved nodes, or both-question support.
- 223: selected artifacts cover some smartphone/tablet-adjacent tokens, but miss Higher-income seniors, go online, tablet computer, and year 2013 as artifact evidence.
- 224: same artifact pattern as 223, and the visible context also does not support year 2022.
- 227: selected artifacts include cell-phone/tablet-adjacent values, but miss 65+, College graduate, explicit gap operation, tablet computer, and year 2013 as artifact evidence.

Decision: do not run a provider diagnostic or full QA from these positives yet. R058 shows the selector preserves positive signals, but not answer-supporting artifact evidence. Next step is selector repair, specifically stronger dimension/key-value matching for demographic group, time constraint, metric label, value coverage, and operation/operand completeness.


### 2026-06-04 R059 Selector Dimension-Support Repair

R059 is a no-provider selector repair gate. It addresses the R058 finding that `token_key_value_selection` could retain artifacts with loose token overlap even when the artifacts did not cover the question dimensions needed for answer support.

Purpose:

- Move question-dimension support auditing into `mdocnexus.integration.guarded_prompt`.
- Add `artifact_dimension_support_guard` after token/key-value candidate selection.
- Guard artifacts that do not cover public question dimensions such as figure/entity, demographic group, time constraint, metric label, value coverage, and operation/operand constraints.
- Verify the repair is not an all-refusal strategy by retaining synthetic positive controls with full dimension coverage.

Scope and boundary:

- Target records only: 69, 223, 224, 227.
- Synthetic positive controls only for proving the selector can still retain fully supporting artifact evidence.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Does not prove artifact-aware retrieval lift.

Outputs:

- `mdocnexus/integration/guarded_prompt.py`
- `mdocnexus/integration/tests/test_guarded_prompt.py`
- `scripts/run_r059_selector_dimension_repair.py`
- `outputs/heldout/r059_selector_dimension_repair/r059_selector_repair_report.md`
- `outputs/heldout/r059_selector_dimension_repair/r059_selector_repair_gate.md`
- `outputs/heldout/r059_selector_dimension_repair/r059_selector_repair_compact_index.jsonl`
- `outputs/heldout/r059_selector_dimension_repair/r059_selector_repair_previews.jsonl`
- `outputs/heldout/r059_selector_dimension_repair/r059_positive_control_previews.jsonl`

Gate result: passed.

Key checks:

- Records 69, 223, 224, 227 all had positive candidates before the guard.
- All four records were rejected by `artifact_dimension_support_guard`.
- All four records selected zero artifacts after the dimension guard.
- Positive controls retained: `raptor_full_dimension_control`, `higher_income_2013_full_dimension_control`.
- Public previews had no forbidden gold fields.
- Artifact store remained the R038d union atomic store.
- No provider, prediction, evaluation, full QA, or official score.

Decision: R059 repairs selector safety for the R058 failure mode. It does not create evidence of artifact positive lift. The next reasonable step is a no-provider routing audit: when page evidence is sufficient but artifact evidence is insufficient, confirm prompts route the model to page evidence and do not present artifact noise as support.


### 2026-06-04 R060 Page/Artifact Routing Audit

R060 is a no-provider prompt-routing audit. It checks the post-R059 case where artifact evidence is rejected by `artifact_dimension_support_guard`, but visible page evidence is sufficient. The goal is to ensure the prompt routes the model to page evidence or refusal, rather than presenting rejected artifact snippets as support.

Purpose:

- Audit records 223 and 227, where page evidence is sufficient and artifact evidence is insufficient.
- Confirm selected artifacts are zero after the dimension guard.
- Confirm prompt previews separate page evidence from artifact evidence.
- Confirm the prompt explicitly says not to cite rejected artifact ids.
- Confirm the prompt says to answer from cited page ids only when page evidence is sufficient, otherwise use `Not answerable`.

Scope and boundary:

- Target records only: 223 and 227.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Does not prove artifact-aware retrieval lift.

Outputs:

- `scripts/run_r060_page_artifact_routing_audit.py`
- `outputs/heldout/r060_page_artifact_routing_audit/r060_routing_report.md`
- `outputs/heldout/r060_page_artifact_routing_audit/r060_routing_gate.md`
- `outputs/heldout/r060_page_artifact_routing_audit/r060_routing_compact_index.jsonl`
- `outputs/heldout/r060_page_artifact_routing_audit/r060_routing_prompt_previews.jsonl`

Gate result: passed.

Key checks:

- Records 223 and 227 matched the page-sufficient / artifact-insufficient target set.
- Both records used `artifact_dimension_support_guard`.
- Both records selected zero artifacts.
- Both records had visible page support sufficient.
- Prompts blocked rejected artifact citation.
- Prompts routed answer generation to cited page ids only when sufficient.
- Public previews had no forbidden gold fields.
- Artifact store remained the R038d union atomic store.
- No provider, prediction, evaluation, full QA, or official score.

Decision: R060 validates prompt routing behavior for page-sufficient/artifact-insufficient cases. It still does not prove artifact-aware retrieval lift. The next bounded step, only after manual acceptance, can be a tiny provider diagnostic on page-routed prompts; do not run full QA from R060.


### 2026-06-04 R061 Page-Routed Provider Diagnostic

R061 is a tiny provider diagnostic for the R060 page-routing prompts. It tests whether the provider follows the page-only route when artifacts are rejected by `artifact_dimension_support_guard`.

Purpose:

- Run only records 223 and 227.
- Use compact provider prompts derived from R060 public prompt previews because full R060 previews are long and the first full-preview provider attempt timed out before writing predictions.
- Confirm the provider does not cite rejected artifact ids.
- Confirm artifact evidence is treated as none.
- Confirm the provider either answers from page evidence or refuses without using artifacts.

Scope and boundary:

- Provider calls: 2 predictions total.
- Model: `Qwen/Qwen3-8B`.
- Provider prompt mode: `r060_derived_compact_page_routing_prompt`.
- No prediction/eval pipeline, no full QA, no official score.
- Does not prove artifact-aware retrieval lift or retrieval improvement.

Outputs:

- `scripts/run_r061_page_routed_provider.py`
- `outputs/heldout/r061_page_routed_provider_diagnostic/r061_page_routed_provider_report.md`
- `outputs/heldout/r061_page_routed_provider_diagnostic/r061_page_routed_provider_gate.md`
- `outputs/heldout/r061_page_routed_provider_diagnostic/predictions/r061_predictions.jsonl`

Gate result: passed.

Key checks:

- Target records exactly 223 and 227.
- Provider predictions exactly 2.
- Predictions bind to the R060 prompt preview hashes.
- R060 prompts are page-routed, zero-artifact, `artifact_dimension_support_guard` prompts.
- Prediction records contain no forbidden gold fields.
- Page-only routing behavior passed for both records.
- No provider result is interpreted as full QA or official score.

Decision: R061 supports keeping the page/artifact routing scaffold as a candidate provider-facing prompt control. It still does not support any artifact-lift or retrieval-improvement claim. The next step should be either manual review of the two R061 outputs or a no-provider integration decision about where this scaffold should sit in the adapter path; do not launch full QA from R061.

### 2026-06-04 R062 Guarded Routing Integration Decision Gate

R062 is a no-provider integration decision gate for the guarded selector and compact page-routing scaffold from R059/R060/R061.

Purpose:

- Decide whether the guarded selector and compact page-routing scaffold should remain as a default-off adapter control surface.
- Verify disabled mode leaves adapter records unchanged by hash.
- Verify enabled mode emits prompt previews and manifest only.
- Bind the compact scaffold provenance to R060 public page-routed previews and R061 compact prompt hashes.
- Repeat the boundary that this is not full QA, not an official score, and not artifact-lift evidence.

Scope and boundary:

- Target records only: 223 and 227.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Does not prove artifact-aware retrieval lift or retrieval improvement.

Outputs:

- `scripts/run_r062_guarded_routing_integration_decision.py`
- `outputs/heldout/r062_guarded_routing_integration_decision/r062_guarded_routing_integration_report.md`
- `outputs/heldout/r062_guarded_routing_integration_decision/r062_guarded_routing_integration_gate.md`
- `outputs/heldout/r062_guarded_routing_integration_decision/r062_integration_decision_manifest.json`
- `outputs/heldout/r062_guarded_routing_integration_decision/r062_compact_routing_scaffolds.jsonl`

Gate result: passed.

Key checks:

- Default-disabled integration leaves records unchanged and emits zero prompt previews.
- Enabled integration leaves records unchanged and emits two audit previews plus a manifest.
- Enabled previews for 223 and 227 use `artifact_dimension_support_guard`, select zero artifacts, and route to page evidence/refusal.
- Compact scaffold hashes match R061 provider prompt hashes for records 223 and 227.
- Compact scaffold is marked optional provider-facing only, not the default full prompt.
- Public previews/scaffolds contain no forbidden gold fields.
- Artifact store remains the R038d union atomic store.

Decision: keep the guarded selector and compact page-routing scaffold default-off behind `enable_guarded_prompt_scaffold`; expose previews/manifest for audit before any future provider run. Do not run full QA from R062.

### 2026-06-04 R063 LLM Evidence-Demand Parser Diagnostic

R063 tests whether Qwen3-VL-8B-Instruct can serve as a question-only evidence-demand parser for the guarded selector. The model parses required entities, metrics, values/codes, operands, and evidence dimensions; it does not answer questions or select artifacts directly.

Purpose:

- Add a default-off reusable evidence-demand parser scaffold.
- Use Qwen3-VL-8B-Instruct only to parse public question text into structured evidence requirements.
- Compare rule-only profiles against LLM-parsed profiles through the deterministic guarded selector.
- Check whether parser-assisted profiles reduce artifact noise or preserve truly supporting artifacts.
- Keep the result diagnostic-only: no prediction, no evaluation, no full QA, no official score.

Scope and boundary:

- Target records only: 384, 508, 569, 69, 223, 224, and 227.
- Provider calls: 7 parser calls total, all question-only.
- Model: `Qwen/Qwen3-VL-8B-Instruct`.
- The LLM is not allowed to answer or choose artifacts; deterministic selector still performs scoring and selection.
- Does not prove artifact-aware retrieval lift or retrieval improvement.

Outputs:

- `mdocnexus/integration/evidence_demand_parser.py`
- `mdocnexus/integration/tests/test_evidence_demand_parser.py`
- `scripts/run_r063_llm_evidence_demand_parser.py`
- `outputs/heldout/r063_llm_evidence_demand_parser/r063_llm_evidence_demand_report.md`
- `outputs/heldout/r063_llm_evidence_demand_parser/r063_llm_evidence_demand_gate.md`
- `outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl`
- `outputs/heldout/r063_llm_evidence_demand_parser/provider/r063_evidence_demand_parser_outputs.jsonl`

Gate result: passed.

Key checks:

- All 7 parser outputs were parseable JSON under the R063 evidence-demand schema.
- Parser inputs were question-only and contained no gold fields.
- Selector previews contained no forbidden gold fields.
- Deterministic selector remained responsible for artifact scoring/selection.
- No prediction, evaluation, full QA, official score, or artifact-lift claim was made.
- Artifact store remained the R038d union atomic store.

Diagnostic outcome:

- LLM guard decisions: 4 `artifact_dimension_support_guard`, 2 `document_metadata_refusal_guard`, 1 `operand_completeness_guard`.
- LLM-selected positive records: none.
- LLM artifact-supporting records: none.
- Interpretation: Qwen3-VL-8B-Instruct can parse evidence demands and tighten/confirm rejection, but this run does not show that parser-assisted profiles recover supporting artifacts from the current store.

Decision: keep the LLM evidence-demand parser as a default-off scaffold, but do not proceed to full QA from R063. The next step should manually inspect parser dimensions and artifact coverage; if useful dimensions are missing from artifacts, repair artifact extraction/coverage rather than scaling QA.

### 2026-06-04 R064 Parser/Artifact Mismatch Audit

R064 is a no-provider attribution audit for the R063 negative signal. It asks where the parser requirements and current artifact/page content disconnect, rather than running another model or QA experiment.

Purpose:

- Rebuild the same public candidate artifacts and page contexts used by R063.
- Compare Qwen3-VL evidence dimensions, required values/codes, and operands against artifact snippets and visible page text.
- Attribute each mismatch to a concrete repair bucket: parser answer-type error, artifact key/value gap, operand gap, artifact dimension gap, selector/alias threshold issue, or retrieval/context/parser constraint gap.
- Keep the result diagnostic-only: no provider calls, no prediction, no evaluation, no full QA, no official score.

Scope and boundary:

- Target records only: 384, 508, 569, 69, 223, 224, and 227.
- Inputs: R063 selector comparisons, R063 gate, R040/R039 retrieval records, R038d union atomic artifact store, and public page context.
- No model/API calls were made in R064.
- Does not prove artifact-aware retrieval lift or retrieval improvement.

Outputs:

- `scripts/run_r064_parser_artifact_mismatch_audit.py`
- `outputs/heldout/r064_parser_artifact_mismatch_audit/r064_parser_artifact_mismatch_report.md`
- `outputs/heldout/r064_parser_artifact_mismatch_audit/r064_parser_artifact_mismatch_gate.md`
- `outputs/heldout/r064_parser_artifact_mismatch_audit/r064_mismatch_audits.jsonl`
- `outputs/heldout/r064_parser_artifact_mismatch_audit/r064_mismatch_compact_index.jsonl`

Gate result: passed.

Root-cause distribution:

- `retrieval_context_or_parser_constraint_gap`: 3 records, `[223, 224, 384]`
- `parser_answer_type_misclassified`: 1 record, `[508]`
- `artifact_operand_missing`: 1 record, `[569]`
- `artifact_store_missing_required_dimensions`: 1 record, `[69]`
- `selector_support_threshold_or_alias_gap`: 1 record, `[227]`

Per-record summary:

- 384: metadata parser found revision/producer needs, but artifact snippets cover neither and page text only partially covers them; inspect parser constraint vs retrieved context before changing selector.
- 508: parser mislabeled an EPS code/table lookup as metadata; fix parser/post-normalization so code patterns force table/code lookup.
- 569: computation operands remain incomplete in artifacts; keep operand-completeness guard and repair extraction before QA.
- 69: page text covers all parser dimensions, but artifact snippets miss figure/node dimensions; repair artifact extraction/normalization.
- 223/224: parser requirements are missing from both artifacts and current page context; inspect overconstraint vs retrieval coverage.
- 227: some artifact aliases match, but support remains insufficient; manually inspect whether this is a real alias bridge or lexical noise.

Decision: do not run more models or full QA from R064. First repair parser code-type normalization for 508, then inspect/repair artifact extraction or retrieval coverage for page-visible and operand/key-value gaps. Rerun no-provider coverage audit before any provider QA experiment.

### 2026-06-04 R065 Parser Code-Type Normalization Regression

R065 fixes the most local and certain R064 issue: Qwen parser outputs containing literal EPS/code patterns must be normalized to table/code lookup, not document metadata lookup.

Purpose:

- Add parser post-normalization so code-like values/entities such as `AR03` force `answer_type=table_lookup`.
- Set `requires_exact_code_selection=True` and clear `is_document_metadata_lookup` for code/table lookups.
- Verify the repair on record 508.
- Verify the metadata control record 384 is not damaged.
- Keep the result no-provider and diagnostic-only.

Scope and boundary:

- Target records only: 508 and 384.
- Inputs: cached R063 parser outputs and public artifact/page context.
- No provider calls, no prediction, no evaluation, no full QA, no official score.
- Does not prove artifact-aware retrieval lift or retrieval improvement.

Outputs:

- `mdocnexus/integration/evidence_demand_parser.py`
- `mdocnexus/integration/tests/test_evidence_demand_parser.py`
- `scripts/run_r065_parser_code_type_regression.py`
- `outputs/heldout/r065_parser_code_type_regression/r065_parser_code_type_report.md`
- `outputs/heldout/r065_parser_code_type_regression/r065_parser_code_type_gate.md`
- `outputs/heldout/r065_parser_code_type_regression/r065_parser_code_type_regressions.jsonl`

Gate result: passed.

Key checks:

- 508 changed from `metadata_lookup` to `table_lookup` after normalization.
- 508 now has `requires_exact_code_selection=True`, `is_document_metadata_lookup=False`, and selector replay routes to `exact_code_absence_guard` with zero selected artifacts because AR03 is absent from artifact snippets.
- 384 stays `metadata_lookup`, does not force exact-code selection, and selector replay keeps `document_metadata_refusal_guard`.
- Public outputs contain no forbidden gold fields.
- No provider, prediction, evaluation, full QA, official score, or artifact-lift claim.

Decision: keep R065 parser code-type normalization in the default-off parser scaffold. Next should inspect or repair artifact key/value extraction for missing AR03 evidence, then rerun no-provider coverage/mismatch audit before any provider QA.


### 2026-06-04 R066 Artifact Key/Value Extraction Audit

R066 audits the remaining record 508 / `AR03` break after R065 normalized the parser route from metadata lookup to table/code lookup.

Purpose:

- Inspect whether `AR03` exists in candidate artifacts, retrieved page text, or whole-document extracted text.
- Replay the deterministic exact-code selector guard.
- Attribute the failure without provider calls, prediction, evaluation, full QA, official score, or artifact-lift claim.

Scope and boundary:

- Target record only: 508.
- Inputs: cached R063 parser output, R065 gate, R040/R039 retrieval records, R038d union atomic artifact store, and public extracted page text.
- No model/API calls were made in R066.

Outputs:

- `scripts/run_r066_artifact_key_value_extraction_audit.py`
- `outputs/heldout/r066_artifact_key_value_extraction_audit/r066_artifact_key_value_report.md`
- `outputs/heldout/r066_artifact_key_value_extraction_audit/r066_artifact_key_value_gate.md`
- `outputs/heldout/r066_artifact_key_value_extraction_audit/r066_artifact_key_value_audit.json`
- `outputs/heldout/r066_artifact_key_value_extraction_audit/r066_key_value_compact_index.jsonl`

Gate result: passed.

Key findings:

- Selector replay is `exact_code_absence_guard`.
- Candidate artifact count is 16; all are on page 8. Candidate page counts are `{"2": 0, "5": 0, "6": 0, "7": 0, "8": 16, "9": 0}`.
- No candidate artifact raw or normalized text contains exact `AR03`.
- Retrieved page text has Arkansas/EPS context, but no exact `AR03`.
- Whole-document extracted text has AR-family codes but no exact `AR03`.
- Primary root cause is `extracted_document_text_missing_required_code`, with secondary categories for artifact key/value absence and numeric-table-only artifact normalization.

Decision: do not relax exact-code matching for AR03. Keep the exact-code absence guard and route this case to page-cited refusal/absence handling. Before any provider diagnostic, audit whether the source PDF visually contains AR03 but OCR/extraction dropped it; if the source also lacks AR03, treat this as not answerable under visible evidence. Separately, repair EPS/table-list artifact extraction for page text such as page 7, where the Arkansas EPS neighborhood is visible but no artifacts are generated.

### 2026-06-04 R067 Source/OCR Code-List Extraction Audit

R067 follows R066 by separating two issues for record 508 / page 7 / `AR03`: whether current OCR/source text supports the requested exact code, and whether Stage 2 can extract code/name list artifacts from visible EPS-like text.

Purpose:

- Check that page text and page image exist for the target source page.
- Confirm whether OCR/extracted text contains exact `AR03`.
- Add a deterministic, page-local code/name list extractor for EPS-like lists.
- Verify the extractor recovers visible code/name pairs without inventing missing codes.
- Replay exact-code selection against the extracted artifacts.
- Keep the result no-provider and diagnostic-only.

Scope and boundary:

- Target record only: 508.
- Target page only: 7.
- Target code: `AR03`.
- Inputs: R063 cached parser output, R066 gate, R040/R039 retrieval records, R038d union atomic artifact store, and public page image/text under `tmp/MMLongBench`.
- No provider calls, no prediction, no evaluation, no full QA, no official score, and no artifact-lift claim.

Outputs:

- `mdocnexus/stage2/code_name_list_extractor.py`
- `mdocnexus/stage2/tests/test_code_name_list_extractor.py`
- `scripts/run_r067_source_ocr_code_list_extraction_audit.py`
- `outputs/heldout/r067_source_ocr_code_list_extraction_audit/r067_source_ocr_code_list_report.md`
- `outputs/heldout/r067_source_ocr_code_list_extraction_audit/r067_source_ocr_code_list_gate.md`
- `outputs/heldout/r067_source_ocr_code_list_extraction_audit/r067_source_ocr_code_list_audit.json`
- `outputs/heldout/r067_source_ocr_code_list_extraction_audit/r067_code_list_compact_index.jsonl`

Gate result: passed.

Key findings:

- Page text and image both exist for page 7.
- OCR text contains `AR01` and `AR02`, but not exact `AR03`.
- Existing artifact count for page 7 is 0.
- The code/name extractor recovers 30 visible EPS-like pairs from page 7.
- The extractor recovers `AR01: Little Rock` and `AR02: Northern Arkansas` and does not invent `AR03`.
- Replaying the exact-code selector with extracted artifacts still gives `exact_code_absence_guard` for record 508.

Decision: integrate the code/name list extractor into Stage 2 for EPS-like public text lists, because it repairs the page-7 artifact coverage gap. Do not relax exact-code matching and do not answer record 508 from `AR01`/`AR02` or Arkansas context. Record 508 should remain page-cited absence/refusal unless source-image or OCR repair reveals exact `AR03`.

### 2026-06-04 R068 Code-List Stage2 Integration Audit

R068 turns the R067 extractor result into a Stage 2 integration gate without running providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Verify that `extract_code_name_list_artifacts` is wired into the `scripts/stage2.py` document-generic final-store postprocess branch.
- Replay the deterministic integration path on record 508 / page 7 / `AR03`.
- Confirm that visible EPS-like code/name artifacts are produced while exact-code absence/refusal behavior remains strict.

Scope and boundary:

- Target record only: 508.
- Target page only: 7.
- Target code: `AR03`.
- Inputs: R067 gate, R063 cached parser comparison, R040/R039 retrieval records, R038d union atomic artifact store, and public page image/text under `tmp/MMLongBench`.
- No model/API calls were made in R068.

Outputs:

- `scripts/stage2.py`
- `scripts/run_r068_code_list_stage2_integration_audit.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r068_code_list_stage2_integration_audit/r068_code_list_stage2_integration_report.md`
- `outputs/heldout/r068_code_list_stage2_integration_audit/r068_code_list_stage2_integration_gate.md`
- `outputs/heldout/r068_code_list_stage2_integration_audit/r068_code_list_stage2_integration_audit.json`
- `outputs/heldout/r068_code_list_stage2_integration_audit/r068_code_list_stage2_compact_index.jsonl`

Gate result: passed.

Key findings:

- Stage 2 now imports and calls the code/name extractor inside the document-generic final-store postprocess branch after the numeric atomicizer.
- Existing page-7 artifact count before integration is 0.
- Integration replay generates 30 final code/name artifacts from public page text.
- Integrated artifacts include `AR01` and `AR02`, pass final quality filtering, and are locatable.
- Integrated artifacts do not include `AR03`.
- Selector replay remains `exact_code_absence_guard`, so record 508 is still routed to exact-code absence/refusal rather than inferred from nearby Arkansas/EPS context.

Decision: keep the Stage 2 code/name extractor integration for public EPS-like lists. Do not relax exact-code matching or run QA for record 508; next work should look for source/OCR evidence for missing exact codes or broaden no-provider coverage checks on positive code/name cases.

### 2026-06-04 R069 Dataset Artifact Health Audit

R069 moves from single-case OCR/source diagnosis to a dataset-level public retrieval and artifact health audit. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Scan MMLongBench public questions and top-4 public retrieved page text.
- Compare current R038d artifact store coverage with deterministic R068 code/name replay coverage.
- Separate broad code-like literals from actionable exact-code lookup cases.
- Attribute failures to retrieval/public text absence, artifact extraction gaps, selector/guard rejection, or selected support availability.

Scope and boundary:

- Records scanned: 1073.
- Inputs: `data/MMLongBench/sample-with-retrieval-results.json`, R038d union atomic artifact store, and public page text under `tmp/MMLongBench`.
- The audit intentionally does not use `answer`, `evidence_pages`, prediction correctness, or provider outputs.

Outputs:

- `scripts/run_r069_dataset_artifact_health_audit.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r069_dataset_artifact_health_audit/r069_dataset_artifact_health_report.md`
- `outputs/heldout/r069_dataset_artifact_health_audit/r069_dataset_artifact_health_gate.md`
- `outputs/heldout/r069_dataset_artifact_health_audit/r069_dataset_artifact_health_summary.json`
- `outputs/heldout/r069_dataset_artifact_health_audit/r069_dataset_artifact_health_records.jsonl`

Gate result: passed.

Key findings:

- Code-like literal records: 64.
- Actionable exact-code lookup records: 4.
- Exact-code lookup public retrieved text contains the actionable literal in 4/4 cases.
- Current R038d artifacts contain exact-code lookup literals in 1/4 cases, but current selector selects 0/4.
- R068 code/name replay contains exact-code lookup literals in 3/4 cases and selector selects 3/4: `AR01`, `CA03`, and `CA19`.
- `AR03` remains unsupported because public retrieved text does not contain exact `AR03`.
- The broad code-like bucket contains 56 temporal/metric literals (`FY/Q/AP/F1` style) that can trigger exact-code guard behavior and should be normalized separately from actionable code lookup.
- Broader artifact health remains weak: many records have public text literals but no selected artifacts, so full QA remains premature.

Decision: R068 code/name extraction is worth carrying into a bounded Stage 2 artifact-store rebuild for actionable code/name questions, but selector/parser normalization must also distinguish actionable codes from fiscal years, quarters, and metric labels. Do not run full QA yet; first repair code-like literal guard normalization and rebuild a bounded artifact store for positive exact-code lookup cases.

### 2026-06-05 R070 Code-Like Literal Guard Normalization

R070 repairs and audits the selector/parser normalization gap found by R069. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Keep actionable exact codes (`AR01`, `CA03`, `CA19`, `AR03`) on strict exact-code selection/absence behavior.
- Normalize temporal/metric code-like literals (`FY2015`, `FY2018`, `Q3`, `AP50`, `F1`) so they do not trigger `exact_code_absence_guard`.
- Verify the rule-profile and LLM evidence-demand parser merge paths use the same actionable-code semantics.

Scope and boundary:

- Records scanned: 1073.
- Inputs: `data/MMLongBench/sample-with-retrieval-results.json`, R038d union atomic artifact store, and public page text under `tmp/MMLongBench`.
- The audit intentionally does not use `answer`, `evidence_pages`, prediction correctness, or provider outputs.

Outputs:

- `mdocnexus/integration/guarded_prompt.py`
- `mdocnexus/integration/evidence_demand_parser.py`
- `mdocnexus/integration/tests/test_guarded_prompt.py`
- `mdocnexus/integration/tests/test_evidence_demand_parser.py`
- `scripts/run_r070_code_like_literal_guard_normalization.py`
- `scripts/run_r069_dataset_artifact_health_audit.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r070_code_like_literal_guard_normalization/r070_code_like_literal_guard_report.md`
- `outputs/heldout/r070_code_like_literal_guard_normalization/r070_code_like_literal_guard_gate.md`
- `outputs/heldout/r070_code_like_literal_guard_normalization/r070_code_like_literal_guard_summary.json`
- `outputs/heldout/r070_code_like_literal_guard_normalization/r070_code_like_literal_guard_records.jsonl`

Gate result: passed.

Key findings:

- Code-like records: 64.
- Temporal/metric records: 56.
- Temporal/metric exact-code guard count: 0.
- Actionable exact-code records: 8.
- Actionable strict-guard records: 8.
- Target coverage confirms `FY2015`, `FY2018`, `Q3`, `AP50`, and `F1` are temporal/metric literals, while `AR01`, `AR03`, `CA03`, and `CA19` remain actionable exact codes.

Decision: keep the R070 selector/parser normalization repair. Do not relax exact-code matching; do not run full QA yet. Next no-provider step should rebuild or replay a bounded Stage 2 artifact store for positive actionable code/name cases, then run a small guarded provider diagnostic only after artifact support is visible and selector-selected.

### 2026-06-05 Paper Experiment Roadmap: Lightweight Evidence Skill Graph Layer

This roadmap uses the ARIS `experiment-plan` framing for the paper-facing method after R070. It is a plan only: no provider run, no QA run, no official score, and no new experiment launch is authorized by this planning update.

Problem anchor:

- Original MDocAgent retrieves multimodal text/image contexts and lets specialized agents read large raw contexts.
- The proposed extension should stay lightweight, dataset-agnostic, and verifiable: a unified evidence layer rather than a dataset-specific artifact stack, heavy GraphRAG, or large skill tree.

Method thesis:

- Add a lightweight Evidence Skill Graph layer between MDocAgent retrieval and multi-agent answering. Retrieved pages are compiled into locatable evidence units, dispatched through a small Evidence Skill Registry, compressed into token-budgeted evidence capsules, and checked by guarded answerability before generation.

Claim map:

| Claim | Why it matters | Minimum convincing evidence | Linked runs |
|-------|----------------|-----------------------------|-------------|
| C1: unified evidence layer | Shows the method is not benchmark-specific artifact engineering | Same schema/registry works across MMLB, LDU, PTAB, PTEXT, and FETA with no dataset-specific artifact types | R071, R073 |
| C2: token-budgeted capsule | Differentiates the method from original MDocAgent raw top-k context consumption | Capsule prompt tokens are substantially lower than original top-k context while preserving required evidence fields | R072, R073, R075 |
| C3: guarded answerability | Shows the method improves reliability, not just retrieval volume | Fewer unsupported answers, fewer rejected-evidence citations, and exact-code/operand/dimension guards trigger correctly | R071, R074, R075 |
| C4: QA utility | Prevents the method from being only an audit wrapper | At least partial QA improvement or no meaningful drop under lower token cost on bounded splits | R075, R078 |

Anti-claims to rule out:

- The gain comes only from concatenating more artifacts.
- The system is a dataset-specific collection of rules for MMLB/LDU/PTAB/PTEXT/FETA.
- The graph component is decorative and equivalent to flat artifact reranking.
- The guard only improves refusal cases while hurting answerable cases.

Paper storyline:

- Main paper must prove: unified evidence layer reuse, token reduction, citation/unsupported-answer reliability, and at least partial QA utility against original MDocAgent.
- Appendix can support: more failure buckets, per-skill traces, edge sensitivity, and dataset-specific qualitative examples.
- Experiments intentionally cut: large-scale skill-tree distillation, heavy global GraphRAG/entity graph construction, and full agentic planning DAGs.

Experiment blocks:

#### Block A: Evidence Skill Graph Design Gate

- Claim tested: C1, C3.
- Why this block exists: It freezes a lightweight, auditable method interface before any QA, preventing artifact rules from sprawling into dataset-specific engineering.
- Dataset / split / task: public retrieved pages and existing artifact stores; no answer/evidence-page fields.
- Compared systems: current guarded selector vs registry-backed selector trace; no provider.
- Metrics: schema validity, allowed evidence unit types, allowed edge types, skill activation trace coverage, guard trace completeness, no-gold audit.
- Setup details: max 6 evidence unit families and max 8 document-native edge families; skills limited to exact-code lookup, key-value lookup, table/numeric lookup, numeric computation, figure/caption grounding, and text-span grounding.
- Success criterion: all skills define applies-if, accepted unit types, required fields, guard rule, capsule render policy, and audit trace; no dataset-specific skill names.
- Failure interpretation: if the registry needs dataset names or many special cases, the method is not paper-ready.
- Table / figure target: method schematic and registry table.
- Priority: MUST-RUN.

#### Block B: Token-Budgeted Evidence Capsule Audit

- Claim tested: C2.
- Why this block exists: The main efficiency claim needs a no-provider measurement before expensive QA.
- Dataset / split / task: MMLB plus available LDU/PTAB/PTEXT/FETA public retrieval contexts or held-out subsets.
- Compared systems: original top-k raw context, flat artifact concat, evidence capsule, evidence capsule plus guard trace.
- Metrics: prompt token count, evidence unit count, capsule compression ratio, retained required literal/entity/metric coverage, locator/citation availability.
- Setup details: token budgets at small/medium/default settings; deterministic renderer only.
- Success criterion: meaningful token reduction versus original MDocAgent top-k context while preserving required evidence dimensions on answerable cases.
- Failure interpretation: if capsule loses required fields, reduce compression or revise registry render policy before provider runs.
- Table / figure target: main efficiency table plus capsule example figure.
- Priority: MUST-RUN.

#### Block C: Cross-Dataset Reuse and Verifiability Audit

- Claim tested: C1, C2, C3.
- Why this block exists: Reviewers will challenge whether artifacts are benchmark-specific.
- Dataset / split / task: MMLB, LDU, PTAB, PTEXT, FETA; public inputs only.
- Compared systems: same evidence layer config across datasets; no dataset-specific artifact type additions.
- Metrics: schema coverage, rejected/selected evidence trace availability, guard trigger distribution, unsupported-risk buckets, token reduction by dataset.
- Setup details: reuse identical registry, unit schema, edge schema, and capsule renderer.
- Success criterion: same evidence layer runs on all target datasets with interpretable traces and no dataset-named skill rules.
- Failure interpretation: if a dataset needs custom schema, move it to appendix or narrow the claim.
- Table / figure target: cross-dataset audit table.
- Priority: MUST-RUN.

#### Block D: Bounded Provider Diagnostic

- Claim tested: C3, C4.
- Why this block exists: Before full QA, verify that agents can consume capsules and respect guards.
- Dataset / split / task: small balanced diagnostic set covering answerable and unsupported cases across evidence skills.
- Compared systems: original MDocAgent prompt, flat artifact prompt, evidence capsule prompt, evidence capsule plus guard.
- Metrics: unsupported answer rate, rejected-evidence citation rate, guard compliance, token count, diagnostic answer correctness.
- Setup details: small provider run only after R071-R073 pass and user explicitly authorizes launch.
- Success criterion: capsule plus guard reduces unsupported answers and citation violations with no obvious answerable-case collapse.
- Failure interpretation: if agents ignore guard traces, revise capsule format before official QA.
- Table / figure target: diagnostic reliability table.
- Priority: MUST-RUN.

#### Block E: Bounded QA Comparison Against Original MDocAgent

- Claim tested: C2, C3, C4.
- Why this block exists: The paper still needs task-level utility, not only audits.
- Dataset / split / task: bounded reproducible splits; start with MMLB held-out, then extend to LDU/PTAB/PTEXT/FETA if prior gates pass.
- Compared systems: original MDocAgent top-k, original plus flat artifacts, evidence capsule without guard, evidence capsule plus guard.
- Metrics: binary correctness or dataset-native QA score, prompt tokens, answerability/refusal correctness, citation faithfulness, changed-answer buckets.
- Setup details: same model/backend as MDocAgent reproduction; same retrieval top-k budget before evidence layer.
- Success criterion: lower token cost, better citation/unsupported behavior, and at least partial QA improvement or no meaningful score drop under lower cost.
- Failure interpretation: if QA drops despite reliability gains, claim should be downgraded to efficiency/faithfulness rather than accuracy.
- Table / figure target: main result table.
- Priority: MUST-RUN.

#### Block F: Ablation and Simplicity Defense

- Claim tested: anti-claims.
- Why this block exists: It proves the method is not a bloated combination of arbitrary components.
- Dataset / split / task: same bounded QA and no-provider audit subsets as Blocks C-E.
- Compared systems: no graph edges, no registry boundaries, no token budget, no guard, flat artifact concat, heavier graph expansion if feasible.
- Metrics: token count, selected evidence sufficiency, unsupported answer rate, QA score, citation faithfulness.
- Setup details: prioritize config-only deletions before code-heavy variants.
- Success criterion: final method is better balanced than flat concat and overbuilt graph variants; each retained component has a measurable role.
- Failure interpretation: remove or demote components that do not change reviewer belief.
- Table / figure target: ablation table.
- Priority: MUST-RUN after main diagnostic passes.

Run order and milestones:

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Freeze method interface | R071 | Registry/schema passes no-gold and no-dataset-specific checks | Low CPU | Scope creep into too many skills |
| M1 | Prove low-token evidence packaging | R072, R073 | Cross-dataset audit passes; token reduction does not remove required evidence | Low CPU | Capsule too lossy |
| M2 | Wire method and verify provider behavior before QA | R074, next provider diagnostic | Prompt integration passes no-gold gate, then guard/citation behavior improves on a small diagnostic set | No-provider then small provider | Agents may ignore guard text |
| M3 | Compare against original MDocAgent | R075 | Lower tokens plus partial QA/citation/unsupported gains | Moderate provider/QA | Original reproduction not stable |
| M4 | Defend novelty and simplicity | R076, R077 | Ablations show graph/registry/guard/token budget each matter or are cut | Moderate | Components look decorative |
| M5 | Paper claim audit | R078 | ARIS audit clears provenance, metrics, scope, and claim labels | Low CPU | Claims need downgrade |

Planned run queue:

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R071 | M0 | Evidence Skill Graph registry design gate | no-provider registry/schema/trace scaffold | public retrieved pages + existing artifacts | schema validity, skill boundary coverage, no-gold audit | MUST | TODO | Freeze lightweight skills, evidence unit types, edge types, required fields, guard rules, capsule render policies. |
| R072 | M1 | Token-budgeted capsule renderer audit | deterministic capsule renderer | MMLB heldout public contexts first | token count, compression ratio, retained evidence requirements, locator coverage | MUST | TODO/BLOCKED-UNTIL-R071 | No provider. Compare raw top-k, flat artifact concat, capsule, capsule+guard trace. |
| R073 | M1 | Cross-dataset evidence-layer reuse audit | same registry across MMLB/LDU/PTAB/PTEXT/FETA | public inputs only | schema coverage, guard trace distribution, token reduction by dataset, no dataset-specific rules | MUST | TODO/BLOCKED-UNTIL-R072 | Must prove this is not MMLB-only artifact engineering. |
| R074 | M2 | MMLB baseline-aligned evidence prompt integration gate | default-off page+capsule+guard prompt variant | full MMLB top-4 public records | prompt plumbing, original-question preservation, bucket plan, no-gold audit | MUST | DONE/GATE-PASSED | No-provider gate. Prepared runnable MDocAgent-compatible input; provider launch remains a separate explicit step. |
| R075 | M3 | Bounded QA comparison against original MDocAgent | original MDocAgent vs evidence-layer variants | bounded reproducible split | QA score, token cost, citation faithfulness, unsupported answer rate | MUST | TODO/BLOCKED-UNTIL-R074 | Main paper utility table candidate. |
| R076 | M4 | Component ablation | remove graph edges / registry / token budget / guard | same bounded split | QA, tokens, citation, unsupported answer, evidence sufficiency | MUST | TODO/BLOCKED-UNTIL-R075 | Defends novelty against concat/rerank criticism. |
| R077 | M4 | Simplicity and overbuilt-graph comparison | lightweight evidence graph vs heavier expansion | no-provider plus optional small provider | token growth, evidence sufficiency, QA/citation if provider authorized | SHOULD | TODO/BLOCKED-UNTIL-R076 | Negative result is acceptable if it justifies lightweight design. |
| R078 | M5 | ARIS paper-claim integrity audit | completed R071-R077 outputs | all completed result dirs | provenance, metric validity, claim scope, file existence | MUST | TODO/BLOCKED-UNTIL-RESULTS | Required before paper-ready claims. |

Compute and data budget:

- R071-R073 and R078 are CPU/no-provider audits.
- R074 is the no-provider integration gate; the first provider diagnostic after R074 must remain small.
- R075-R077 are only launched after the original MDocAgent baseline/reproduction path is stable and the user authorizes provider/QA work.
- Biggest bottleneck: credible original MDocAgent reproduction plus consistent cross-dataset public input plumbing.

Risks and mitigations:

- Risk: the method looks like artifact engineering. Mitigation: freeze a registry with dataset-agnostic evidence unit and skill names, then audit all datasets with the same config.
- Risk: graph edges are decorative. Mitigation: R076 removes edges and compares against flat artifact concat.
- Risk: token compression loses necessary evidence. Mitigation: R072/R073 require retained evidence-dimension coverage before provider runs.
- Risk: guards improve refusal but hurt answerable QA. Mitigation: R074/R075 use balanced answerable/unsupported cases and report changed-answer buckets.
- Risk: official QA is premature. Mitigation: ARIS gates require no-provider audits and provider diagnostics before bounded QA, then R078 audits all claims.

Final checklist:

- [ ] Main paper tables are covered by R073/R075/R076.
- [ ] Novelty is isolated by R076.
- [ ] Simplicity is defended by R077 or cut if unnecessary.
- [ ] Frontier/agent contribution is justified as a lightweight evidence layer for MDocAgent, not a new large agent system.
- [ ] Nice-to-have graph expansion runs are separated from must-run evidence-layer runs.

### 2026-06-05 R071 Evidence Skill Graph Registry Gate

R071 freezes the lightweight Evidence Skill Graph registry interface before capsule, provider, or QA work. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Define a bounded, dataset-agnostic Evidence Skill Registry rather than a large skill tree or global graph.
- Register each evidence skill with applies-if logic, accepted evidence unit types, required fields, guard rule, capsule render policy, and audit trace fields.
- Verify deterministic skill activation and trace generation over public records/artifacts.

Scope and boundary:

- Records scanned: 1073.
- Inputs: `data/MMLongBench/sample-with-retrieval-results.json`, R038d union atomic artifact store, and public page text under `tmp/MMLongBench`.
- The audit intentionally does not use `answer`, `evidence_pages`, prediction correctness, or provider outputs.

Outputs:

- `mdocnexus/integration/evidence_skill_registry.py`
- `mdocnexus/integration/tests/test_evidence_skill_registry.py`
- `scripts/run_r071_evidence_skill_graph_registry_gate.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r071_evidence_skill_graph_registry_gate/r071_evidence_skill_registry_report.md`
- `outputs/heldout/r071_evidence_skill_graph_registry_gate/r071_evidence_skill_registry_gate.md`
- `outputs/heldout/r071_evidence_skill_graph_registry_gate/r071_evidence_skill_registry_summary.json`
- `outputs/heldout/r071_evidence_skill_graph_registry_gate/r071_evidence_skill_registry_records.jsonl`

Gate result: passed.

Key findings:

- Registry skills: 6 `['exact_code_lookup', 'figure_caption_grounding', 'key_value_lookup', 'numeric_computation', 'table_numeric_lookup', 'text_span_grounding']`.
- Evidence unit types: 6 `['text_span', 'table_cell', 'numeric_fact', 'key_value', 'caption', 'code_name_pair']`.
- Document edge types: 8 `['contains', 'same_page', 'same_table', 'row_of', 'column_of', 'caption_of', 'nearby', 'code_maps_to']`.
- Contract failures: `[]`.
- Control activation passed: True.
- Dataset records activated all registered skills: `['exact_code_lookup', 'figure_caption_grounding', 'key_value_lookup', 'numeric_computation', 'table_numeric_lookup', 'text_span_grounding']`.

Decision: keep the registry bounded and dataset-agnostic. R072 should use this registry as the only skill dispatch interface for token-budgeted capsule rendering; do not add dataset-named skills or launch provider QA before R072/R073 no-provider gates pass.

### 2026-06-05 R072 Token-Budgeted Evidence Capsule Audit

R072 validates the first efficiency claim for the lightweight evidence layer by rendering deterministic evidence capsules from the existing R071 registry. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Reuse the R071 Evidence Skill Registry as the only dispatch interface.
- Compare token estimates for raw retrieved page context, flat artifact context, evidence capsule without trace, and evidence capsule with guard trace.
- Verify that capsule rendering is bounded and deterministic before any cross-dataset or provider run.

Scope and boundary:

- Records scanned: 1073.
- Inputs: `data/MMLongBench/sample-with-retrieval-results.json`, R038d union atomic artifact store, and public page text under `tmp/MMLongBench`.
- The audit intentionally does not use `answer`, `evidence_pages`, prediction correctness, or provider outputs.

Outputs:

- `mdocnexus/integration/evidence_skill_registry.py`
- `mdocnexus/integration/tests/test_evidence_skill_registry.py`
- `scripts/run_r072_token_budgeted_capsule_audit.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r072_token_budgeted_capsule_audit/r072_token_budgeted_capsule_report.md`
- `outputs/heldout/r072_token_budgeted_capsule_audit/r072_token_budgeted_capsule_gate.md`
- `outputs/heldout/r072_token_budgeted_capsule_audit/r072_token_budgeted_capsule_summary.json`
- `outputs/heldout/r072_token_budgeted_capsule_audit/r072_token_budgeted_capsule_records.jsonl`

Gate result: passed.

Key findings:

- Mean raw page tokens: 1607.252563.
- Mean flat artifact tokens: 41.38863.
- Mean capsule tokens without guard trace: 38.671948.
- Mean capsule tokens with guard trace: 57.319664.
- Mean guarded capsule/raw token ratio: 0.222101.
- Guarded capsule lower than raw rate: 0.976701.

Decision: keep capsule rendering inside the existing Evidence Skill Registry module rather than adding another abstraction. Proceed to R073 cross-dataset reuse/token audit before any provider QA.

### 2026-06-05 R073 Cross-Dataset Evidence Layer Reuse Audit

R073 audits whether the lightweight evidence layer can be reused across MMLB, LDU, PTAB, PTEXT, and FETA public inputs without adding dataset-named skills or a large graph/skill tree. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Reuse the existing R071 Evidence Skill Registry and R072 capsule renderer as the only evidence-layer interface.
- Report cross-dataset public input availability and question-only skill activation with the same registry.
- Run full token/capsule reuse only where public retrieval pages and artifact bindings already exist.

Scope and boundary:

- Datasets reported: MMLB, LDU, FETA, PTAB, PTEXT.
- MMLB full capsule audit records scanned: 1073.
- LDU/FETA/PTAB/PTEXT are marked `blocked_missing_public_retrieval_or_artifacts` because public samples and page text exist, but no equivalent public retrieval-page list plus artifact store binding is present.
- The audit intentionally does not use `answer`, `evidence_pages`, prediction correctness, provider outputs, or gold pages as substitute retrieval.

Outputs:

- `scripts/run_r073_cross_dataset_evidence_layer_reuse_audit.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r073_cross_dataset_evidence_layer_reuse_audit/r073_cross_dataset_evidence_layer_report.md`
- `outputs/heldout/r073_cross_dataset_evidence_layer_reuse_audit/r073_cross_dataset_evidence_layer_gate.md`
- `outputs/heldout/r073_cross_dataset_evidence_layer_reuse_audit/r073_cross_dataset_evidence_layer_summary.json`
- `outputs/heldout/r073_cross_dataset_evidence_layer_reuse_audit/r073_cross_dataset_evidence_layer_records.jsonl`

Gate result: passed with explicit input gaps.

Key findings:

- MMLB mean guarded capsule/raw ratio: 0.222101.
- Cross-dataset skill activation used all registry skills: `exact_code_lookup`, `figure_caption_grounding`, `key_value_lookup`, `numeric_computation`, `table_numeric_lookup`, `text_span_grounding`.
- Blocked datasets: `['FETA', 'LDU', 'PTAB', 'PTEXT']` due to missing public retrieval/artifact bindings.
- No forbidden gold fields were found in public outputs.

Decision: keep the evidence layer lightweight and shared. The next step should be a small reusable public retrieval-to-artifact binding adapter for blocked datasets, not new dataset-specific skills or a heavy GraphRAG tree. Do not claim cross-dataset token/citation gains until those bindings exist and pass a follow-up audit.

### 2026-06-05 R074 MMLB Baseline-Aligned Evidence Prompt Integration Gate

R074 turns the R071-R073 lightweight evidence layer into a runnable, default-off MDocAgent prompt variant before any provider or full QA run. It does not run providers, predictions, evaluation, full QA, official scoring, or artifact-lift claims.

Purpose:

- Preserve the original MMLB `question` field for evaluation while storing the evidence-layer prompt in `_nexus_prompt_question`.
- Reuse the existing top-4 MDocAgent baseline records and R038d artifact store to build page+capsule+guard prompts under the same retrieval budget.
- Prepare an explicit `predict.py` command that activates the prompt variant only with `+dataset.prompt_question_key=_nexus_prompt_question`.

Scope and boundary:

- Records scanned: 1073.
- Baseline top-4 score reference: 0.49301 from `results/MMLongBench/mmlb-MDocAgent-top4/2026-05-19-14-19_results.json`.
- No provider calls, no predictions, no evaluation, no full QA, and no official score.
- Public retrieval output contains no `answer`, `evidence_pages`, or `binary_correctness` fields.

Outputs:

- `mydatasets/base_dataset.py`
- `mdocnexus/integration/tests/test_mdocagent_adapter.py`
- `scripts/run_r074_mmlb_evidence_prompt_integration_gate.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_layer_top4_retrieval.json`
- `outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_prompt_report.md`
- `outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_prompt_gate.md`
- `outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_prompt_summary.json`
- `outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_prompt_records.jsonl`

Gate result: passed.

Key findings:

- Original `question` is preserved for eval; `_nexus_prompt_question` is present on all 1073 records.
- Comparison buckets: baseline-wrong help candidates=29; baseline-correct no-selected-artifact risk=521; baseline-wrong stable=515.
- Mean prompt/original question token ratio: 11.306894. This is expected for prompt augmentation and must be tested against QA/citation behavior before full runs.
- Recommended command: `python3 scripts/predict.py --config-name mmlb run-name=mmlb-MDocAgent-r074-evidence-layer-top4 dataset.top_k=4 dataset.sample_with_retrieval_path=outputs/heldout/r074_mmlb_evidence_prompt_integration_gate/r074_mmlb_evidence_layer_top4_retrieval.json +dataset.prompt_question_key=_nexus_prompt_question`.

Decision: the method is now wired as a runnable MDocAgent variant without changing baseline defaults. The next run should be a small provider diagnostic over balanced help/risk buckets; do not launch full MMLB QA until help > hurt is observed.

### 2026-06-05 R075 MMLB Evidence Prompt Small Provider Diagnostic

R075 is a bounded provider diagnostic over R074 help/risk/stable buckets. It does not run full MDocAgent multi-agent QA, full MMLB, or official scoring.

Purpose:

- Test whether the R074 evidence-layer prompt helps more than it hurts before any full QA launch.
- Reuse existing MDocAgent top-4 baseline correctness only as sampled comparison metadata.
- Add resumable, cached, parallel provider/evaluator execution so slow provider calls do not block the diagnostic.

Scope and boundary:

- Selected cases: 66 from R074 buckets (`29` help candidates, `29` baseline-correct risk cases, `8` stable cases).
- Provider model: `Qwen/Qwen3-8B`; evaluator model: `deepseek-ai/DeepSeek-V3`.
- Parallel workers: 6; request timeout: 20 seconds; max retries: 1 for the completed diagnostic run.
- No official score, no full MMLB claim, and no full MDocAgent QA claim.

Outputs:

- `scripts/run_r075_mmlb_evidence_prompt_small_provider_diagnostic.py`
- `scripts/run_heldout_diagnostic_audits.py`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/r075_selected_cases.jsonl`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/predictions/r075_predictions.jsonl`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/predictions/r075_evaluations.jsonl`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/r075_small_provider_summary.json`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/r075_small_provider_gate.md`
- `outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic/r075_small_provider_report.md`

Gate result: completed, but blocks full-run expansion.

Key findings:

- Provider predictions/evaluations: 66/66.
- Provider failures: 21 (`0.318182`), conservatively scored as incorrect.
- Evaluation failures: 21 (`0.318182`), corresponding to provider failures.
- Sample accuracy, not official: `0.212121`; baseline sample reference, not official: `0.560606`.
- Outcomes: `changed_to_right=6`, `changed_to_wrong=29`, `kept_right=8`, `kept_wrong=23`; help-hurt delta = `-23`.
- Bucket diagnosis: baseline-correct no-selected-artifact risk cases produced 22 changed-to-wrong rows, and baseline-correct stable cases produced 7 changed-to-wrong rows.

Decision: do not launch full MMLB QA from the current R074 evidence prompt. The next step should be a smaller balanced rerun with a stable provider or longer timeout, plus a prompt/guard repair that keeps baseline-correct/no-artifact cases close to the original question instead of forcing evidence-layer refusal or page routing.

