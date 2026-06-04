# R055 Guarded Prompt Provider Diagnostic

Decision: `r055_guarded_prompt_provider_complete`

## Boundary
- 3 records only: 384, 508, 569.
- Provider diagnostic on R054 guarded prompts only.
- Can only show whether guard prompts make the model refuse/avoid artifact noise.
- Cannot show artifact positive lift, retrieval improvement, full QA, or an official MMLongBench score.

## Diagnostic Counts
- model: `Qwen/Qwen3-8B`
- predictions: 3
- guard behavior pass count, not score: 3 / 3
- diagnostic counts, not scores: `{"pass": 3}`

## Per Record
- 384: guard=`document_metadata_refusal_guard`, passed=True, failures=`[]`
- 508: guard=`exact_code_absence_guard`, passed=True, failures=`[]`
- 569: guard=`operand_completeness_guard`, passed=True, failures=`[]`

## Interpretation
- bottom_line: R055 only tests whether the R054 guard prompts make the provider refuse or avoid artifact-noise traps on 3 manually accepted cases.
- artifact_lift_claim: unsupported_by_this_run
- retrieval_improvement_claim: unsupported_by_this_run
- official_score_claim: unsupported_by_this_run

## Recommended Next
- If R055 passes, keep the guard as a candidate prompt-control component for later controlled diagnostics.
- Do not claim artifact-aware retrieval improves from R055; it has zero retrieval-condition comparison and only three guarded prompts.
- Before any broader run, decide whether to integrate these guards into the selector/prompt path or run another tiny contrastive diagnostic with explicit positive evidence cases.
