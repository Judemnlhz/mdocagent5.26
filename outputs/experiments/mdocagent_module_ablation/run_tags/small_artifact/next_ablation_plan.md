# Next Ablation Plan

Run the small graph_context diagnostic after original_plus_artifact showed positive signal.

Do not run these automatically.

Recommended next runs:
- top4_graph_context

Blocked until positive signal:
- none

Commands:
- `python3 scripts/run_mdocagent_module_ablation.py --execute-predict --runs top4_graph_context --record-slice 0:30 --run-tag small_graph --confirm-run-api`
- `python3 scripts/run_mdocagent_module_ablation.py --execute-eval --runs top4_graph_context --record-slice 0:30 --run-tag small_graph --confirm-run-eval`

Keep the same top-4 page budget and the same model configuration.
