# MDocAgent Execution Gate Report

Status: fail
Phase: phase_2_small_artifact
Gate passed: False
Recommended next phase: stop
Hard failures: 1
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
| prediction_command_shape_consistent | True |
| evaluation_command_shape_consistent | True |
| binary_correctness_delta_recorded | {'delta': 0.13333333333333336, 'max_allowed_delta': 0.05, 'api_nondeterminism_note': 'prediction text differences are not treated as hard failures'} |
| record_slice_consistent | True |
| max_records_consistent | True |
| adapter_manifest_policy | True |
| public_leakage | False |
| run_name_resume_path_pollution | {'detected': False} |
| api_nondeterminism_possible | True |

## Runs

| run_name | execution_run_name | prediction | evaluation | binary_correctness |
| --- | --- | --- | --- | ---: |
| top4_artifact_only | top4_artifact_only__small_artifact_60_90 | results/MMLongBench/top4_artifact_only__small_artifact_60_90/2026-06-02-03-21.json | results/MMLongBench/top4_artifact_only__small_artifact_60_90/2026-06-02-03-21_results.json | 0.266667 |
| top4_original_plus_artifact | top4_original_plus_artifact__small_artifact_60_90 | results/MMLongBench/top4_original_plus_artifact__small_artifact_60_90/2026-06-02-03-35.json | results/MMLongBench/top4_original_plus_artifact__small_artifact_60_90/2026-06-02-03-35_results.json | 0.400000 |

## Hard Failures

- public_leakage

## Soft Warnings

- answer_text_diff_binary_same: Answer text differs while binary correctness is unchanged.
- binary_correctness_delta: Binary correctness differs while retrieval/model/top_k/eval configuration may still be consistent.
