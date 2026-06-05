# R074 MMLB Baseline-Aligned Evidence Prompt Integration

Decision: `r074_mmlb_evidence_prompt_integration_ready_for_provider_diagnostic`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Builds a default-off MDocAgent prompt variant for later explicit provider runs.
- Keeps original `question` for evaluation and stores the evidence prompt in `_nexus_prompt_question`.

## Summary
- records scanned: 1073
- baseline top-4 score reference: 0.49301
- selected artifact record rate: 0.014911
- mean prompt/original question token ratio: 1.421143
- prompt modes: {'original_question_passthrough_no_artifact': 1040, 'page_plus_capsule_plus_guard_prompt_question': 33}

## Comparison Buckets
- `baseline_correct_no_selected_artifact_risk`: 521
- `baseline_correct_stable_candidate`: 8
- `baseline_wrong_capsule_supported_candidate`: 8
- `baseline_wrong_guarded_or_page_routed_candidate`: 21
- `baseline_wrong_stable_candidate`: 515

## Recommended Commands
```bash
python3 scripts/predict.py --config-name mmlb run-name=mmlb-MDocAgent-r074-evidence-layer-top4 dataset.top_k=4 dataset.sample_with_retrieval_path=outputs/heldout/r076_no_artifact_passthrough_prompt_repair/r074_mmlb_evidence_layer_top4_retrieval.json +dataset.prompt_question_key=_nexus_prompt_question
python3 scripts/eval.py --config-name mmlb run-name=mmlb-MDocAgent-r074-evidence-layer-top4
```

## Recommended Next
- Run a small provider diagnostic before full QA, sampling baseline-wrong help candidates (29) and baseline-correct risk candidates (521).
- Use the generated retrieval JSON with dataset.prompt_question_key set to _nexus_prompt_question; do not overwrite the original question field.
- If the diagnostic shows help <= hurt, revise prompt format or use a page+capsule hybrid with weaker guard wording before full MMLB.
