# Stage 2 Runtime Boundary Manifest

This repository currently contains runtime-level modifications in:

- `scripts/predict.py`
- `agents/multi_agent_system.py`

These modifications are treated as baseline/runtime utilities, including:

- `resume_path` support
- skipping already completed samples
- optional parallel prediction
- retry on failure/OOM
- model caching if present

These modifications are not Stage 2 method contributions.

The Stage 2 method implementation is isolated in:

- `mdocnexus/stage2/`
- `scripts/stage2_augment_retrieval_results.py`
- `scripts/stage2_select_trial_candidate.py`
- `scripts/stage2_compile_small_batch.py`
- `scripts/stage2_audit_small_batch.py`
- later `scripts/stage2_compile_crossdoc_batch.py`

Stage 2 does not require editing:

- `scripts/extract.py`
- `scripts/retrieve.py`
- `scripts/predict.py`
- `scripts/eval.py`
- `agents/multi_agent_system.py`
- `models/siliconflow.py`

All Stage 2 experiments must record:

- git commit hash
- whether runtime resume/parallel/retry was used for baseline prediction
- stage2 script name
- model config path
- provider
- model_name
- max_pages / max_docs limits
- whether real API was called
- raw_output_log path
- discard_log path
- artifact_store output path

For paper writing:

- runtime patches must be described as engineering utilities;
- they must not be counted as algorithmic contributions;
- main method contribution remains provenance-preserving evidence artifact compilation.
