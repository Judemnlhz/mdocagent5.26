# Next Ablation Plan

Run the small artifact ablation only; graph_context remains gated on positive artifact signal.

Do not run these automatically.

Recommended next runs:
- top4_artifact_only
- top4_original_plus_artifact

Blocked until positive signal:
- top4_graph_context

Commands:
- `python3 scripts/run_mdocagent_module_ablation.py --execute-predict --runs top4_artifact_only,top4_original_plus_artifact --record-slice 30:60 --run-tag small_artifact --confirm-run-api`
- `python3 scripts/run_mdocagent_module_ablation.py --execute-eval --runs top4_artifact_only,top4_original_plus_artifact --record-slice 30:60 --run-tag small_artifact --confirm-run-eval`

Keep the same top-4 page budget and the same model configuration.
