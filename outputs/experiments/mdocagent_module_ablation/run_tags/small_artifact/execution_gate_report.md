# MDocAgent Execution Gate Report

Status: pass
Phase: phase_2_small_artifact
Gate passed: True
Recommended next phase: phase_3_small_graph
Hard failures: 0
Soft warnings: 2

| check | pass |
| --- | --- |
| prediction_outputs_exist | True |
| evaluation_outputs_exist | True |
| retrieval_input_equivalent | True |
| top_k_consistent | True |
| model_config_hash_consistent | True |
| page_budget_consistent | True |
| evaluation_judge_consistent | True |
| prediction_command_shape_consistent | False |
| evaluation_command_shape_consistent | True |
| binary_correctness_delta_recorded | {'delta': 0.43333333333333335, 'max_allowed_delta': 0.05, 'api_nondeterminism_note': 'prediction text differences are not treated as hard failures'} |
| record_slice_consistent | True |
| max_records_consistent | True |
| adapter_manifest_policy | True |
| public_leakage | True |
| run_name_resume_path_pollution | {'detected': False} |
| api_nondeterminism_possible | True |

## Runs

| run_name | execution_run_name | prediction | evaluation | binary_correctness |
| --- | --- | --- | --- | ---: |
| top4_artifact_only | top4_artifact_only__small_artifact | results/MMLongBench/top4_artifact_only__small_artifact/2026-06-01-07-03.json | results/MMLongBench/top4_artifact_only__small_artifact/2026-06-01-07-03_results.json | 0.166667 |
| top4_original_plus_artifact | top4_original_plus_artifact__small_artifact | results/MMLongBench/top4_original_plus_artifact__small_artifact/2026-06-01-07-56.json | results/MMLongBench/top4_original_plus_artifact__small_artifact/2026-06-01-07-56_results.json | 0.600000 |

## Soft Warnings

- answer_text_diff_binary_same: Answer text differs while binary correctness is unchanged.
- binary_correctness_delta: Binary correctness differs while retrieval/model/top_k/eval configuration may still be consistent.
