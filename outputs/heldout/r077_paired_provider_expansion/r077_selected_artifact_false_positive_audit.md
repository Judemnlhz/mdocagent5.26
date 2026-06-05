# R077 Selected-Artifact False-Positive Audit

Decision: `selected_artifact_false_positive_review_needed`

## Summary
- selected_cases: 66
- selected_artifact_cases: 16
- strict_guard_cases: 10
- paired_outcome_counts: {'kept_wrong': 49, 'kept_right': 14, 'changed_to_right': 2, 'changed_to_wrong': 1}
- selected_artifact_paired_outcome_counts: {'kept_wrong': 13, 'changed_to_right': 2, 'kept_right': 1}
- selected_artifact_guard_counts: {'operand_complete_selection': 1, 'token_key_value_selection': 15}
- paired_changed_to_wrong_ids: [1035]
- paired_changed_to_right_ids: [1017, 1060]
- provider_failure_ids: [10, 224, 227, 306, 568, 641, 807, 1002, 1003, 1034, 1058]

## Paired Outcome By Bucket
- `baseline_correct_no_selected_artifact_risk`: {'kept_wrong': 20, 'kept_right': 9}
- `baseline_correct_stable_candidate`: {'kept_wrong': 5, 'changed_to_right': 2, 'kept_right': 1}
- `baseline_wrong_capsule_supported_candidate`: {'kept_wrong': 8}
- `baseline_wrong_guarded_or_page_routed_candidate`: {'kept_wrong': 16, 'kept_right': 4, 'changed_to_wrong': 1}

## Paired Changed To Wrong
### record 1035 / AMAZON_2017_10K.pdf
- bucket: `baseline_wrong_guarded_or_page_routed_candidate`
- prompt_mode: `page_plus_capsule_plus_guard_prompt_question`
- guard: `operand_completeness_guard`, selected_artifact_count: 0
- question: what is Amazon's FY2017 return on asset ? round your answer to three decimal
- original prediction: Amazon's FY2017 return on assets (ROA) is calculated as Net Income divided by Total Assets. From the provided data: - **Net Income (2017)** = $3,033 million - **Total Assets (2017)** = $131,310 million ROA = 3,033 / 131,310 ≈ **0.02309** (rounded to three decimal places: **0.023**). **Answer:** 0.023
- evidence prediction: Not answerable. The evidence capsule indicates missing operands for net income and total assets, but the retrieved page text provides both values for FY2017. However, the guard policy prevents using inferred values when operands are incomplete. The capsule's missing support conflicts with visible page evidence, so the answer cannot be computed as per the guard's instruction. **Final Answer** Not answerable
- evidence prompt preview: [MDocAgent Evidence Layer - page plus capsule plus guard] Use the normal MDocAgent retrieved page text and images as the primary evidence. Use the evidence capsule below as a compact checklist of selected evidence, missing requirements, and guard policy. If capsule evidence conflicts with visible page evidence, rely on visible page evidence and mention the missing support. Do not infer an exact code, numeric value, or citation from partial evidence when the guard says it is missing. Return the final answer in the dataset's expected concise format. [Original Question] what is Amazon's FY2017 return on asset ? round your answer to three decimal [Evidence Capsule] Skills: numeric_computation Guard: operand_completeness_guard; policy: not_answerable_due_to_incomplete_operands Evidence: - id=atomicizer_numeric_fact_016 | page=36 | type=numeric_fact | fields=row_label=CASH AND CASH EQUIVALENTS

## Paired Changed To Right IDs
- [1017, 1060]
