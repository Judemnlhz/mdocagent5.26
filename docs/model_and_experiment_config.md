# Model And Experiment Configuration

`config/model/*.yaml` is the only public model configuration source for this
repository. Do not add `model_roles.yaml`.

Model roles:

- `deepseek-ai/DeepSeek-V3` is reserved for evaluation or judge-only workflows.
  It must not be used by Stage 2 artifact compilation, Stage 3 retrieval, or
  Stage 4 graph construction.
- `Qwen/Qwen3-8B` is the text-only processing model. It may be used for text
  extraction, metadata normalization, and query text processing. It must not be
  used for image payloads or multimodal extraction.
- `Qwen/Qwen3-VL-8B-Instruct` is the multimodal/VLM model for image artifact
  extraction and the bounded real smoke workflow. It must not be used as an
  evaluation judge.

Stage 2/3/4 main-flow manifests must record deterministic or fake model roles
when no real model is used. Evaluation manifests may record an evaluator model
only when they are marked with both `evaluation_only=true` and
`not_consumed_by_stage2_stage3_stage4=true`.

API keys must be supplied through environment variables or private local
configuration. Public manifests, logs, summaries, and model config files must
not contain real API key values, raw provider responses, base64 payloads,
`data:image`, local absolute paths, or file URLs.

Recommended entrypoint:

```bash
python3 scripts/mdocnexus.py --help
python3 scripts/mdocnexus.py run-matrix --dry-run
python3 scripts/mdocnexus.py run-real-smoke-small --dry-run
python3 scripts/mdocnexus.py audit --model-configs
```

Older scripts remain available for compatibility, but new experiment runs should
prefer the unified CLI.
