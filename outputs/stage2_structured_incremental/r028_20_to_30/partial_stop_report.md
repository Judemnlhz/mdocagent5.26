# R028 20 -> 30 Partial Stop

Decision: stop_and_inspect_parse_noise

## Parse Gate
- Completed delta pages: 6 / 10
- Success / failure: 2 / 4
- Current partial parse_success_rate: 0.333
- Previous increment parse_success_rate: 0.700
- Best possible final parse_success_rate if remaining pages all succeed: 0.600
- parse_success_rate_non_decreasing_still_possible: False

## Scope
- No QA was run.
- No rerank tuning was run.
- No full ablation was run.
- Concentration metrics remain external-validity diagnostics only and are not used for reranking or scoring.

## Completed Pages
- 2310.09158v1.pdf#p007: success=True, failure_type=None, parsed_artifact_count=5
- 2401.18059v1.pdf#p006: success=False, failure_type=parse_failure, parsed_artifact_count=0
- 936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003: success=False, failure_type=parse_failure, parsed_artifact_count=0
- BESTBUY_2023_10K.pdf#p026: success=False, failure_type=parse_failure, parsed_artifact_count=0
- BESTBUY_2023_10K.pdf#p027: success=False, failure_type=parse_failure, parsed_artifact_count=0
- PIP_Seniors-and-Tech-Use_040314.pdf#p008: success=True, failure_type=None, parsed_artifact_count=6

Next action: inspect parse failures/provider response format before any further expansion.
