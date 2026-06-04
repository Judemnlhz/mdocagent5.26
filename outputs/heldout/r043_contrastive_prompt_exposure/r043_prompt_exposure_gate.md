# R043 Prompt Exposure Gate

Decision: `r043_prompt_exposure_gate_pass`
Gate passed: True
Recommended next phase: `small_provider_contrastive_run`

## Boundary
- No provider calls, no new prediction, no new evaluation, no full QA.
- Prompt previews only.
- Gold answer/evidence fields are excluded from previews.

## Conditions
| condition | records | page-text records | artifact records | total artifact snippets | missing page-text records |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_pages_only | 37 | 37 | 0 | 0 | 0 |
| page_rerank_only | 37 | 37 | 0 | 0 | 0 |
| original_pages_plus_artifact_snippets | 37 | 37 | 37 | 264 | 0 |
| artifact_snippets_only | 37 | 0 | 37 | 276 | 0 |

## Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `four_conditions_present`: True
- `all_conditions_have_37_records`: True
- `no_gold_fields_in_previews`: True
- `page_rerank_only_has_no_artifact_snippets`: True
- `artifact_snippets_only_has_no_page_text`: True
- `plus_condition_has_artifact_exposure`: True
- `snippet_only_has_artifact_exposure`: True
- `original_pages_only_has_no_artifact_snippets`: True
- `artifact_store_bound_to_r038d_union_atomic_store`: True
- `prompt_hashes_unique_by_condition`: True
