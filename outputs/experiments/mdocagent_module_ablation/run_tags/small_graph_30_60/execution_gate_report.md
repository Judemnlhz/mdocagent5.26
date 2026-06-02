# MDocAgent Execution Gate Report

Status: pass
Phase: phase_3_small_graph
Gate passed: True
Recommended next phase: mark_graph_context_diagnostic_only
Hard failures: 0
Soft warnings: 0

| check | pass |
| --- | --- |
| prediction_outputs_exist | True |
| evaluation_outputs_exist | True |
| retrieval_input_equivalent | True |
| top_k_consistent | True |
| model_config_hash_consistent | True |
| page_budget_consistent | True |
| evaluation_judge_consistent | True |
| prediction_command_shape_consistent | True |
| evaluation_command_shape_consistent | True |
| binary_correctness_delta_recorded | {'delta': None, 'max_allowed_delta': 0.05, 'api_nondeterminism_note': 'prediction text differences are not treated as hard failures'} |
| record_slice_consistent | True |
| max_records_consistent | True |
| adapter_manifest_policy | True |
| public_leakage | True |
| run_name_resume_path_pollution | {'detected': False} |
| api_nondeterminism_possible | True |

## Runs

| run_name | execution_run_name | prediction | evaluation | binary_correctness |
| --- | --- | --- | --- | ---: |
| top4_graph_context | top4_graph_context__small_graph_30_60 | results/MMLongBench/top4_graph_context__small_graph_30_60/2026-06-01-13-13.json | results/MMLongBench/top4_graph_context__small_graph_30_60/2026-06-01-13-13_results.json | 0.266667 |
