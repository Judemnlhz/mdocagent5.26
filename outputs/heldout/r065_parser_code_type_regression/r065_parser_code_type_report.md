# R065 Parser Code-Type Normalization Regression

Decision: `r065_parser_code_type_regression_complete`

## Boundary
- No provider calls, no prediction, no evaluation, no full QA.
- Does not prove artifact lift or official MMLongBench performance.

## Per Record
- 508: before=`metadata_lookup`, after=`table_lookup`, guard=`exact_code_absence_guard`, passed=True, failed=`[]`
- 384: before=`metadata_lookup`, after=`metadata_lookup`, guard=`document_metadata_refusal_guard`, passed=True, failed=`[]`

## Recommended Next
- Keep the R065 parser code-type normalization in the default-off parser scaffold.
- Do not run full QA from R065; next inspect or repair artifact key/value extraction for the missing AR03 evidence.
- Rerun no-provider mismatch/coverage audit after artifact extraction changes.
