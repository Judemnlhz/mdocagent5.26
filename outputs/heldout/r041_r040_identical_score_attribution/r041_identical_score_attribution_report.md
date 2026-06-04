# R041 R040 Identical Score Attribution Audit

Decision: `r041_identical_aggregate_score_attribution_complete`

## Boundary
- Post-hoc attribution audit only.
- No provider calls, no new prediction, no new evaluation, no full QA.
- Uses only frozen R039/R040 targeted 37-record outputs.
- Not full-data generalization and not an official MMLongBench result.

## Score Pattern
| run | correct / scored | binary_correctness |
| --- | ---: | ---: |
| top4_original_only | 18 / 37 | 0.486486 |
| top4_original_plus_artifact | 18 / 37 | 0.486486 |
| top4_artifact_only | 18 / 37 | 0.486486 |

## Outcome Counts
- all correct across all three runs: 16
- all wrong across all three runs: 17
- binary divergent records: 4
- answer text differs while binary correctness stays same: 18
- cancellation: top4_artifact_only gained 2 records and lost 2 records relative to the other two runs, producing an aggregate tie.

## Binary Divergence
| record_id | pattern | note |
| ---: | --- | --- |
| 223 | 110 | Among the Higher-income seniors, what are the percentage of them go online, has smartphone phone, and own a tablet compu |
| 508 | 110 | According to this document, what's the geographic market name for EPS Code AR03? |
| 1034 | 001 | what is Amazon's FY2017 Operating Profit Margin Before Depreciation? round your answer to three decimal |
| 1035 | 001 | what is Amazon's FY2017 return on asset ? round your answer to three decimal |

## Retrieval Change vs Original
| run | any branch list changed | any branch set changed | combined set changed | mean combined page Jaccard |
| --- | ---: | ---: | ---: | ---: |
| top4_original_plus_artifact | 17 | 12 | 7 | 0.961358 |
| top4_artifact_only | 21 | 15 | 7 | 0.952831 |

## Artifact Exposure
| run | records with artifacts on selected pages | total artifacts on selected pages | records with positive artifact scores |
| --- | ---: | ---: | ---: |
| top4_original_only | 37 | 638 | n/a |
| top4_original_plus_artifact | 37 | 638 | 37 |
| top4_artifact_only | 37 | 663 | 37 |

## Key Checks
- `no_provider_calls`: True
- `no_prediction_or_eval_invoked`: True
- `targeted_37_only`: True
- `three_requested_runs_only`: True
- `aggregate_scores_equal`: True
- `binary_correctness_vectors_identical`: False
- `scores_match_r040_summary`: True
- `prediction_commands_have_no_artifact_context_arg`: True
- `artifact_modes_candidate_pool_constrained_for_all_records`: True

## Attribution
- All three aggregate scores are exactly 18/37; the tie is not a rounding artifact.
- The record-level binary vectors are not identical: 4 records diverge, and gains/losses cancel out in the aggregate.
- top4_artifact_only gained 2 records and lost 2 records relative to the other two runs, producing an aggregate tie.
- The artifact-labeled R040 runs changed page-ranking inputs only; prediction commands did not pass artifact content as prompt-visible context.
- Artifact-mode selected pages stayed within the original top-10 retrieval candidate pools for every audited record.
- Retrieval/order changed for some records, but those changes did not cross the binary correctness decision boundary.
- Artifact scores were present on selected pages, so the audit does not reduce to a completely inert artifact store; the observed effect is binary-insensitive under this setup.

Bottom line: R040's identical aggregate scores are best attributed to small record-level cancellation under a page-reranking-only, original-candidate-pool-constrained setup, not to identical per-record behavior and not to evidence that artifact context improves or fails on full data.

## Recommended Next Steps
- Do manual error analysis on records where answer text changed but binary correctness stayed the same.
- If running a later experiment, make it contrastive: separate page-reranking effects from explicit artifact-context injection.
- Add exposure accounting to future manifests so artifact score, selected page, and prompt-visible evidence are distinct fields.
- Do not launch full QA from R040 alone; R041 supports only a targeted diagnostic conclusion.
