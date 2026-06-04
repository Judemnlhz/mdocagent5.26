# R064 Parser/Artifact Mismatch Audit

Decision: `r064_parser_artifact_mismatch_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- No official score and no artifact-lift claim.
- Audits where R063 parser requirements fail to connect to artifact snippets/page context.

## Summary
- records: 7
- root cause counts: `{"artifact_operand_missing": 1, "artifact_store_missing_required_dimensions": 1, "parser_answer_type_misclassified": 1, "retrieval_context_or_parser_constraint_gap": 3, "selector_support_threshold_or_alias_gap": 1}`
- records by root cause: `{"artifact_operand_missing": [569], "artifact_store_missing_required_dimensions": [69], "parser_answer_type_misclassified": [508], "retrieval_context_or_parser_constraint_gap": [223, 224, 384], "selector_support_threshold_or_alias_gap": [227]}`

## Per Record
- 384: root=`retrieval_context_or_parser_constraint_gap`, answer_type=`metadata_lookup`, artifact_dims=0/2, page_dims=1/2, missing_artifact=`['revision_date', 'producer']`, missing_page=`['revision_date']`
- 508: root=`parser_answer_type_misclassified`, answer_type=`metadata_lookup`, artifact_dims=0/2, page_dims=1/2, missing_artifact=`['eps_code', 'geographic_market_name']`, missing_page=`['eps_code']`
- 569: root=`artifact_operand_missing`, answer_type=`computation`, artifact_dims=0/5, page_dims=2/5, missing_artifact=`['survey_date', 'organization', 'age_group', 'education_status', 'employment_status']`, missing_page=`['survey_date', 'age_group', 'employment_status']`
- 69: root=`artifact_store_missing_required_dimensions`, answer_type=`visual_caption`, artifact_dims=1/3, page_dims=3/3, missing_artifact=`['figure_number', 'node']`, missing_page=`[]`
- 223: root=`retrieval_context_or_parser_constraint_gap`, answer_type=`table_lookup`, artifact_dims=0/2, page_dims=0/2, missing_artifact=`['population_group', 'device_usage']`, missing_page=`['population_group', 'device_usage']`
- 224: root=`retrieval_context_or_parser_constraint_gap`, answer_type=`numeric_comparison`, artifact_dims=0/3, page_dims=1/3, missing_artifact=`['population_group', 'survey_source', 'survey_period']`, missing_page=`['population_group', 'survey_period']`
- 227: root=`selector_support_threshold_or_alias_gap`, answer_type=`numeric_comparison`, artifact_dims=1/4, page_dims=3/4, missing_artifact=`['age_group', 'education_level', 'survey_period']`, missing_page=`['age_group']`

## Recommended Next
- Do not run more models or full QA from R064.
- First fix parser/code-type normalization for parser_answer_type_misclassified cases.
- Then repair artifact extraction/normalization for page-visible dimensions and exact key/value or operand gaps.
- Rerun a no-provider coverage audit before any provider QA experiment.
