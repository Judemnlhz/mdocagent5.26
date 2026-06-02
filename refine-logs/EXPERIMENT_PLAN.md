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
