# Anchored Artifact Activation Probe

Run tag: `anchored_artifact_activation_probe_n2`
Probe records: 2 (`[14, 19]`)
Full-scan activation: 2/1073 = 0.001864
Activated + changed pages: 2/1073

## Accuracy
- `top4_original_only`: 1.000
- `top4_original_plus_artifact`: 1.000
- `top4_artifact_only`: 0.500

## Outcome vs Original
- `help`: 0
- `hurt`: 0
- `neutral`: 2

## Interpretation
- The 60:90 fallback result is a safety check, not an artifact effectiveness check.
- The activation probe is too small for an effectiveness claim because only 2/1073 records activate anchored artifact reranking.
- Do not proceed to paper-facing full artifact ablation as a positive-effectiveness claim until artifact coverage/locator quality is improved or a larger activated subset is available.

## Per Record
- record `14`: original=1, original_plus_artifact=1, artifact_only=1, outcome=`neutral`
- record `19`: original=1, original_plus_artifact=1, artifact_only=0, outcome=`neutral`
