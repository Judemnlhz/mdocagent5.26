# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Environment check | `source /opt/conda/bin/activate mdocagent`, then verify `python3 scripts/predict.py` imports | Remote `/home/lhz/MDocAgent` | import pass/fail, package versions | MUST | TODO | System Python currently misses `PIL`; use `/opt/conda/envs/mdocagent` |
| R002 | M0 | Model/config audit | `python3 scripts/mdocnexus.py audit --all` | Repo configs and outputs | audit pass/fail | MUST | TODO | Blocks all QA if failing |
| R003 | M0 | Adapter dry-run | `python3 scripts/run_mdocagent_module_ablation.py` | MMLongBench | manifests created, hashes, no-gold flags | MUST | DONE/PREPARED | Existing summary says `prepared_not_run`; rerun after any artifact update |
| R004 | M1 | Current artifact coverage audit | Read `outputs/stage2_doc`, `outputs/stage3_doc_artifact_retrieval` reports | MMLongBench | query coverage, zero-hit count, artifact types | MUST | TODO | Current default coverage is too low for final QA |
| R005 | M1 | Expanded coverage matrix | `python3 scripts/mdocnexus.py run-matrix` or targeted `run_coverage_experiment.py` | Fixed public subset | coverage@5, recall@5, zero-hit count | MUST | TODO | Use public inputs only; no answer/evidence fields |
| R006 | M1 | Real multimodal smoke | `python3 scripts/mdocnexus.py run-real-smoke-small` after dry-run | Tiny fixed subset | image payload audit, visual artifact rate | MUST | TODO | Do not scale real API until smoke passes |
| R007 | M2 | Official top-1 reproduction | `mdocagent_top1_official_reproduction` | MMLongBench | binary correctness | MUST | TODO | Official reproduction row; current historical reference is 0.459 |
| R008 | M2 | Official top-4 reproduction | `mdocagent_top4_official_reproduction` | MMLongBench | binary correctness | MUST | TODO | Official reproduction row; current historical reference is 0.493 |
| R009 | M2 | Adapter sanity | `top4_original_only` | MMLongBench | binary correctness, page overlap | MUST | TODO | Must match official top-4 behavior before other adapter rows |
| R010 | M3 | Artifact-only ablation | `top4_artifact_only` | MMLongBench | binary correctness, changed-page rate | MUST | TODO | Expected to be aggressive; useful for isolation |
| R011 | M3 | Main artifact-aware row | `top4_original_plus_artifact`, lambda=0.5 | MMLongBench | binary correctness, delta vs R009 | MUST | TODO | Primary paper row |
| R012 | M3 | Graph context ablation | `top4_graph_context`, page_neighborhood | MMLongBench | binary correctness, graph-added-page rate | MUST | TODO | Negative result is acceptable if analyzed |
| R013 | M4 | Lambda sweep low | `original_plus_artifact`, lambda=0.25 | MMLongBench or fixed subset | binary correctness, page changes | NICE | TODO | Run only if R011 is sensitive |
| R014 | M4 | Lambda sweep high | `original_plus_artifact`, lambda=0.75 | MMLongBench or fixed subset | binary correctness, page changes | NICE | TODO | Run only if R011 is sensitive |
| R015 | M4 | Budget diagnostic top-8 | MDocAgent original top-8 | MMLongBench or subset | binary correctness, pages read | NICE | TODO | Not official reproduction; anti-claim diagnostic |
| R016 | M4 | Budget diagnostic top-10/top-20 | MDocAgent original larger budget | MMLongBench or subset | binary correctness, cost | NICE | TODO | Run only if paper needs budget counterfactual |
| R017 | M4 | Graph edge ablation | `eval_stage4_graph_expansion.py --edge-ablation` | Retrieval/evidence eval | delta recall/coverage by edge family | NICE | TODO | Useful if R012 changes pages |
| R018 | M5 | Win/loss bucket analysis | Compare R009/R011/R012 predictions | Completed QA outputs | changed-to-right, changed-to-wrong, no-artifact failures | MUST | TODO | Predefine buckets before reading examples |
| R019 | M5 | Cost and coverage table | Parse manifests, call logs, result dirs | Completed runs | API calls, compile pages, pages read, coverage | MUST | TODO | Needed for "not just more cost" argument |
| R020 | M5 | Paper-ready claim extraction | `result-to-claim` style summary from results | Completed runs | supported / weak / unsupported claims | MUST | TODO | Decides whether to proceed to paper-plan |
