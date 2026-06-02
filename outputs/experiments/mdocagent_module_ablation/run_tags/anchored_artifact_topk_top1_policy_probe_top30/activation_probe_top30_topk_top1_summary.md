# Anchored Artifact Top-k + Top-1 Policy Probe

Run tag: `anchored_artifact_topk_top1_policy_probe_top30`
Policy: original_plus_artifact reranks only original branch top-k and preserves each branch top-1
Probe records: 30
Probe activated: 27/30
Probe changed pages: 25/30

## Accuracy
- `top4_original_only`: 0.500
- `top4_artifact_only`: 0.400
- `top4_original_plus_artifact`: 0.467
- delta `original_plus_artifact - original_only`: -0.033
- delta `artifact_only - original_only`: -0.100

## Help / Hurt / Neutral
- `original_plus_artifact` `help`: 1
- `original_plus_artifact` `hurt`: 2
- `original_plus_artifact` `neutral`: 27
- `artifact_only_help_vs_original`: 1
- `artifact_only_hurt_vs_original`: 4
- `artifact_only_neutral_vs_original`: 25

## Retrieval-Attributable Outcome
- retrieval-attributable `help`: 0
- retrieval-attributable `hurt`: 2
- retrieval-attributable `neutral`: 27
- model/eval variance `help`: 1
- model/eval variance `hurt`: 0

Conclusion: top-k/top-1 constrained policy reduces harms but still does not pass the effectiveness gate; no retrieval-attributable help remains on this top30 probe.

## Gate
- effectiveness gate passed: `False`

## Per Record
- record `14`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `19`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `99`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `100`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `103`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `186`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `187`: original=1, original_plus=1, artifact_only=0, outcome=`neutral`
- record `188`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `189`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `190`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `191`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `192`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `206`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `207`: original=1, original_plus=0, artifact_only=0, outcome=`hurt`
- record `208`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `209`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `210`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `211`: original=1, original_plus=1, artifact_only=0, outcome=`neutral`
- record `212`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `213`: original=1, original_plus=0, artifact_only=0, outcome=`hurt`
- record `214`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `215`: original=0, original_plus=1, artifact_only=1, outcome=`help`
- record `216`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `217`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `230`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `231`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `232`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `233`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `234`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `236`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
