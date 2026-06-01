# MDocAgent Module Ablation

Status: prepared_not_run

| run_name | group | top_k | module | status |
| --- | --- | ---: | --- | --- |
| mdocagent_top1_official_reproduction | official_reproduction | 1 | none_original_mdocagent | prepared_not_run |
| mdocagent_top4_official_reproduction | official_reproduction | 4 | none_original_mdocagent | prepared_not_run |
| top4_original_only | adapter_sanity_check | 4 | adapter_original_only | prepared_not_run |
| top4_artifact_only | module_ablation | 4 | artifact_reranking | prepared_not_run |
| top4_original_plus_artifact | module_ablation | 4 | original_plus_artifact_reranking | prepared_not_run |
| top4_graph_context | module_ablation | 4 | graph_guided_page_selection | prepared_not_run |

Official reproduction is limited to top-1 and top-4. Larger budgets are additional diagnostics only.
Prepared runs do not generate answers or call evaluation APIs.
