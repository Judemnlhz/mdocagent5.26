# R044 Small Contrastive Diagnostic Attribution

Decision: `r044_small_contrastive_diagnostic_complete`

## Boundary
- 22 selected R042 focus records only.
- Diagnostic attribution only.
- Not full QA and not an official MMLongBench result.
- Counts below are diagnostic gold-match counts, not official scores.

## Diagnostic Counts
- model: `deepseek-ai/DeepSeek-V3`
- temperature: `0.0`
- max_tokens: `256`
- provider note: Qwen/Qwen3-8B timed out on the first full R043 prompt; DeepSeek-V3 was used for this diagnostic provider run.
- predictions: 88
- target records: 22
- condition diagnostic match counts: `{"artifact_snippets_only": 1, "original_pages_only": 1, "original_pages_plus_artifact_snippets": 1, "page_rerank_only": 3}`
- transition counts: `{"all_conditions_miss": 19, "artifact_injection_gain_vs_original": 1, "artifact_injection_loss_vs_original": 1, "page_rerank_gain_vs_original": 2, "snippet_only_sufficient": 1}`

## Interpretation
- R044 is a 22-case diagnostic contrast over prompt-visible exposure; counts are not official scores.
- artifact injection gains vs original: 1
- artifact injection losses vs original: 1
- page rerank gains vs original: 2
- page rerank losses vs original: 0
- snippet-only sufficient cases: 1

## Recommended Next
- Manually inspect records where artifact snippets improve over original pages and where snippet-only fails despite plus succeeding.
- Do not promote these counts to official scores; they are prompt-exposure diagnostics on 22 selected cases.
- If the diagnostic pattern is coherent, implement a controlled R045 with fixed prompt templates and explicit support/citation rubric before any full-data run.
