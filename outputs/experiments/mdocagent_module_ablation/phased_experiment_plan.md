# MDocAgent Phased Experiment Plan

Run phases sequentially. Do not run artifact or graph ablations until the preceding gate passes.

## phase_1_small_gate
- runs: mdocagent_top4_official_reproduction, top4_original_only
- record_slice: 0:30
- purpose: adapter consistency under real QA
- allowed_if: adapter gate has passed

## phase_2_small_artifact
- runs: top4_artifact_only, top4_original_plus_artifact
- record_slice: 0:30
- purpose: test artifact-aware reranking signal under same page budget
- allowed_if: phase_1_small_gate passes

## phase_3_small_graph
- runs: top4_graph_context
- record_slice: 0:30
- purpose: test graph_context only after artifact signal is positive
- allowed_if: top4_original_plus_artifact has positive signal

## phase_4_full_ablation
- runs: mdocagent_top4_official_reproduction, top4_original_only, top4_artifact_only, top4_original_plus_artifact, top4_graph_context
- record_slice: None
- purpose: paper-facing full top-4 QA ablation
- allowed_if: small runs pass sanity
