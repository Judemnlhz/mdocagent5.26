# Anchored Artifact Activation Probe Top30

Run tag: `anchored_artifact_activation_probe_top30_matrix`
Artifact source: `outputs/experiments/matrix/retrieval_topk_scope_topk4_hybrid_full_hybrid_page_neighborhood/stage2_doc_coverage/artifacts.jsonl`
Probe records: 30
Full-scan activated records for this artifact source: 99/1073
Probe activated: 30/30
Probe changed pages: 30/30

## Accuracy
- `top4_original_only`: 0.500
- `top4_original_plus_artifact`: 0.433
- `top4_artifact_only`: 0.400
- delta `original_plus_artifact - original_only`: -0.067
- delta `artifact_only - original_only`: -0.100

## Help / Hurt / Neutral
- `original_plus_artifact` `help`: 2
- `original_plus_artifact` `hurt`: 4
- `original_plus_artifact` `neutral`: 24
- `artifact_only_help_vs_original`: 1
- `artifact_only_hurt_vs_original`: 4
- `artifact_only_neutral_vs_original`: 25

## Gate
- effectiveness gate passed: `False`
- criterion: original_plus_artifact must improve activated-subset accuracy over original_only without introducing hurts

## Per Record
- record `14`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `19`: original=0, original_plus=1, artifact_only=0, outcome=`help`
- record `99`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `100`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `103`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `186`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `187`: original=1, original_plus=0, artifact_only=0, outcome=`hurt`
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
- record `211`: original=1, original_plus=0, artifact_only=0, outcome=`hurt`
- record `212`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `213`: original=1, original_plus=1, artifact_only=0, outcome=`neutral`
- record `214`: original=1, original_plus=0, artifact_only=1, outcome=`hurt`
- record `215`: original=0, original_plus=0, artifact_only=1, outcome=`neutral`
- record `216`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `217`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `230`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `231`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
- record `232`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `233`: original=0, original_plus=1, artifact_only=0, outcome=`help`
- record `234`: original=1, original_plus=1, artifact_only=1, outcome=`neutral`
- record `236`: original=0, original_plus=0, artifact_only=0, outcome=`neutral`
