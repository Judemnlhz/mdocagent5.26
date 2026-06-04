# R063 LLM Evidence-Demand Parser Diagnostic

Decision: `r063_llm_evidence_demand_parser_complete`

## Boundary
- Uses Qwen3-VL-8B-Instruct only to parse question evidence requirements.
- Does not ask the model to answer, select artifacts, evaluate, or run QA.
- Does not prove artifact positive lift, retrieval improvement, or official MMLongBench performance.

## Summary
- model: `Qwen/Qwen3-VL-8B-Instruct`
- records: 7
- parseable outputs: 7
- guard decision counts: `{"artifact_dimension_support_guard": 4, "document_metadata_refusal_guard": 2, "operand_completeness_guard": 1}`
- interpretation counts: `{"llm_requirements_no_positive_artifact_lift_claim": 3, "llm_requirements_tighten_or_confirm_artifact_rejection": 4}`
- LLM-selected positive records: `[]`
- LLM artifact-supporting records: `[]`

## Per Record
- 384: answer_type=`metadata_lookup`, rule_guard=`document_metadata_refusal_guard`, llm_guard=`document_metadata_refusal_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_no_positive_artifact_lift_claim`
- 508: answer_type=`metadata_lookup`, rule_guard=`exact_code_absence_guard`, llm_guard=`document_metadata_refusal_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_no_positive_artifact_lift_claim`
- 569: answer_type=`computation`, rule_guard=`operand_completeness_guard`, llm_guard=`operand_completeness_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_no_positive_artifact_lift_claim`
- 69: answer_type=`visual_caption`, rule_guard=`artifact_dimension_support_guard`, llm_guard=`artifact_dimension_support_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_tighten_or_confirm_artifact_rejection`
- 223: answer_type=`table_lookup`, rule_guard=`artifact_dimension_support_guard`, llm_guard=`artifact_dimension_support_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_tighten_or_confirm_artifact_rejection`
- 224: answer_type=`numeric_comparison`, rule_guard=`artifact_dimension_support_guard`, llm_guard=`artifact_dimension_support_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_tighten_or_confirm_artifact_rejection`
- 227: answer_type=`numeric_comparison`, rule_guard=`artifact_dimension_support_guard`, llm_guard=`artifact_dimension_support_guard`, rule_selected=0, llm_selected=0, artifact_support=False, interpretation=`llm_requirements_tighten_or_confirm_artifact_rejection`

## Recommended Next
- Manually inspect R063 selector comparisons before changing the adapter path.
- If the parser adds useful dimensions but still selects no supporting artifacts, improve artifact store/coverage rather than running full QA.
- If the parser preserves or improves supporting artifact retention on positive cases, add a default-off integration gate for parser-assisted profiles before any provider QA run.
