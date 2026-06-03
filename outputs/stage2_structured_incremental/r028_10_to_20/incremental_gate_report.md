# Stage 2 Structured Incremental Gate

Range: 10 -> 20 pages
Decision: `continue_to_next_increment`

## Stop/Go Metrics
- parse_success_rate: 0.5 -> 0.7
- strong_eligible_artifacts: 4 -> 21
- eligible_pages: 2 -> 9
- activated_count: 6 -> 30
- mock_or_placeholder_content: 0 -> 0

## Checks
- `parse_success_rate_non_decreasing`: True
- `strong_eligible_artifacts_increased`: True
- `eligible_pages_increased`: True
- `activated_count_increased`: True
- `mock_or_placeholder_content_still_zero`: True

## Scope
- Metrics are stop/go diagnostics only.
- Concentration metrics are external-validity diagnostics only.
- No gold fields, no QA, no rerank tuning, no full ablation.
