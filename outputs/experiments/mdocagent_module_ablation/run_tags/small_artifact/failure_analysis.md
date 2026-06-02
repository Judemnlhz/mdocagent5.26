# MDocAgent Execution Gate Failure Analysis

Status: fail
Hard failures: 1
Soft warnings: 2

- retrieval input consistent: False
- top_k consistent: True
- model config consistent: True
- prediction command shape consistent: False
- eval command shape consistent: True
- run_name / resume_path pollution: False
- API nondeterminism may affect text: True

The artifact/graph ablation runs must remain stopped until this gate passes.
