# R028 Parse Repair Replay

Decision: parse_repair_probe_passed_bounded_replay_only

## Scope
- Replayed exactly 3 previously failed pages.
- No Stage 2 page expansion.
- No QA, graph, rerank tuning, or full ablation.
- Replay artifacts were not merged into cumulative artifacts.
- Raw provider responses were not written to public outputs; public logs contain hashes/summary fields only when failures occur.

## Metrics
- provider_call_success_count: 3
- provider_call_failed_count: 0
- json_parse_success_count: 3
- parse_failure_count: 0
- num_valid_artifacts: 6
- num_discarded_artifacts: 0
- schema_valid_rate: 1.0
- artifact_type_counts: {'figure': 1, 'table': 5}

## Page Results
- 2401.18059v1.pdf#p006: success=True, failure_type=None, parsed=2, discarded=0
- 936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf#p003: success=True, failure_type=None, parsed=1, discarded=0
- BESTBUY_2023_10K.pdf#p026: success=True, failure_type=None, parsed=3, discarded=0

Interpretation: parser repair fixed the provider-format failure mode in this bounded replay. It is not a QA result and not an expansion gate pass by itself.
