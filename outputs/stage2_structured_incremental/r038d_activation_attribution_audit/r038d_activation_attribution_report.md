# R038d Activation Attribution Audit

Decision: `freeze_r037_based_heldout_with_targeted_coverage_caveat`

## Scope
- No-provider attribution audit over existing artifacts only.
- Temporary atomic-only stores and diagnostic activation scans.
- No Stage 2 compile, final artifact merge, QA, graph, effectiveness claim, or rerank tuning.

## Variant Metrics
### cumulative20_plus_r037
- Activated records: 81
- Eligible for held-out: 81
- Held-out available: `False` (25 records)
- Changed artifact_only / original_plus: 48 / 39
- Strong eligible pages: 10
- Max doc/page share: 0.1728 / 0.1605

### cumulative20_plus_r038c
- Activated records: 19
- Eligible for held-out: 19
- Held-out available: `False` (17 records)
- Changed artifact_only / original_plus: 29 / 15
- Strong eligible pages: 9
- Max doc/page share: 0.3158 / 0.2105

### cumulative20_plus_r037_plus_r038c
- Activated records: 96
- Eligible for held-out: 96
- Held-out available: `True` (37 records)
- Changed artifact_only / original_plus: 73 / 50
- Strong eligible pages: 19
- Max doc/page share: 0.1562 / 0.1354

## Overlap
- R037 activated: 81
- R038c activated: 19
- Union activated: 96
- Activated overlap: 4
- R037 unique activated: 77
- R038c unique activated: 15
- Union new over R037: 15
- Union held-out records: 37
- Union new held-out over R037: 13
- Union reliance on R037: 0.84375

## Checks
- `no_provider_calls`: True
- `no_stage2_compile`: True
- `no_qa`: True
- `union_heldout_available`: True
- `union_eligible_for_heldout_at_least_30`: True
- `r038c_adds_activation_over_r037`: True
- `r038c_standalone_below_heldout_gate`: True
- `r037_remains_primary_activation_source`: True

## Next Step
Freeze a held-out subset using union attribution, but explicitly label targeted coverage as necessary; still run no QA until subset IDs are committed.
