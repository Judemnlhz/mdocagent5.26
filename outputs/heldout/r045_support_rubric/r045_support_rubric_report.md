# R045 Support/Citation-Aware Rubric

Decision: `r045_support_rubric_complete`

## Boundary
- Post-hoc support rubric only.
- No provider calls, no new prediction, no new evaluation, no full QA.
- Not an official score.

## Key Findings
- R044's simple diagnostic matcher undercounted at least one supported refusal: record 569 snippet-only is a supported insufficient-data answer.
- Record 384 shows artifact snippets can introduce unsupported false-positive risk on not-answerable questions.
- Record 508 is the clearest positive diagnostic: reranked pages and artifact snippets support refusing AR03 rather than hallucinating a market.
- The next iteration should improve refusal/support rubric and artifact selection before any full-data run.

## Cases
| record_id | case_type | rubric_label | support summary |
| ---: | --- | --- | --- |
| 384 | transition_case | artifact_injection_introduces_false_positive_risk | Page text supports a refusal because the visible revision is May 2016, not May 2018. Artifact snippets mention Strategic Planning Services Team on page 10 but do not support the requested producer/date relation. |
| 508 | transition_case | rerank_and_artifact_context_help_unanswerable_refusal | The visible Arkansas codes include AR01 and AR02 but not AR03. Original pages led to a false positive, while reranked pages, page+artifact snippets, and snippet-only conditions all support refusal. |
| 569 | transition_case | diagnostic_gold_match_undercounts_supported_refusal | The task asks about children with STEM degrees, but visible contexts do not provide that data. Page-rerank and snippet-only predictions explicitly refuse; the simple R044 matcher counted only page-rerank as a match. |
| 69 | all_conditions_miss_sample | all_conditions_miss_requires_error_analysis | All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring. |
| 223 | all_conditions_miss_sample | all_conditions_miss_requires_error_analysis | All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring. |
| 224 | all_conditions_miss_sample | all_conditions_miss_requires_error_analysis | All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring. |
| 227 | all_conditions_miss_sample | all_conditions_miss_requires_error_analysis | All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring. |
| 367 | all_conditions_miss_sample | all_conditions_miss_requires_error_analysis | All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring. |

## Recommended Next
- Implement question-aware artifact selection rather than fixed first-N artifacts per page.
- Add explicit unsupported-answer/refusal instructions for Not answerable cases.
- Require cited page ids or artifact ids in future diagnostic prompts before reporting any broader score.
- Treat R044 counts as preliminary; R045 support labels supersede simple gold-match labels for transition cases.
