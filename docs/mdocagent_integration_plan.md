# MDocAgent Integration Plan

The current MDocNexus Stage 2/3/4 matrix is diagnostic. It validates document-generic artifact compilation, deterministic lexical or hybrid artifact retrieval, rule-only document-native graph construction, retrieval-only evaluation, and graph-expansion-only evaluation. It is not the final QA main result.

The final paper comparison must return to the original MDocAgent top-1/top-4 inference path:

`scripts/predict.py -> BaseDataset -> MDocAgent.predict_dataset() -> MDocAgent.predict() -> scripts/eval.py`

Official reproduction is limited to top-1 and top-4. Top-8, top-10, and top-20 are additional budget diagnostics only and must not be labeled as official reproduction.

## Retrieval Adapter

The adapter writes a new `sample-with-retrieval-results-nexus.json` file that keeps the schema expected by `BaseDataset.load_sample_retrieval_data`: `doc_id`, `question`, and the MDocAgent retrieval page fields such as `text-top-10-question` and `image-top-10-question`.

Artifact-aware reranking replaces only the retrieval page records:

- `original_only` keeps the original retrieval order.
- `artifact_only` ranks pages by the maximum deterministic BM25 lexical score between the question and artifacts on that page.
- `original_plus_artifact` uses `final_score = lambda_weight * original_score + (1 - lambda_weight) * artifact_score` with fixed `lambda_weight`.

Graph-guided page selection reads only formal `edges.jsonl` edges. It never reads `debug_edges.jsonl`, never uses semantic edges, and only changes which pages fit inside the same final `top_k` budget. It does not add extra context pages beyond the MDocAgent page budget.

## Controlled Comparison

All module ablations use the same MDocAgent models and the same top-k/page budget as the baseline. This prevents a stronger model or larger page budget from being the reason for any QA gain.

Model roles are fixed:

- DeepSeek-V3 (`deepseek-ai/DeepSeek-V3`) is evaluation-only.
- Qwen3-8B (`Qwen/Qwen3-8B`) is used for text-only processing.
- Qwen3-VL-8B-Instruct (`Qwen/Qwen3-VL-8B-Instruct`) is used for multimodal/VLM processing.

The adapter itself is deterministic and has `model_role=none_deterministic`. DeepSeek-V3 is not used in Stage 2/3/4, reranking, graph selection, or context selection.

## Out Of Scope In This Round

This integration round does not modify `MDocAgent.predict()`, agent prompts, the summarizing agent, or final answer generation. It also does not implement proof trace, refusal, or artifact context augmentation. The only connection point is replacing `dataset.sample_with_retrieval_path` with an adapter-produced retrieval-record file.
