# ARIS Experiment Protocol for MDocAgent/MDocNexus

This protocol binds future experiment work in this repository to the project-local ARIS Codex skills installed under `.agents/skills/`.

## Skills Used

- `experiment-plan`: claim-driven experiment roadmap and tracker updates.
- `run-experiment`: remote pre-flight checks and `screen` launch procedure.
- `monitor-experiment`: screen/log/result monitoring after launch.
- `experiment-audit`: integrity audit before any paper claim.

The `.agents/skills/*` entries are symlinks managed by ARIS. Do not edit, delete, or clean them as generated output.

## Planning Rule

Every new run must have a tracker row before launch. The row must state:

- claim or anti-claim tested
- system variant and split
- metric and scope label
- pre-flight gate
- stop/go dependency
- whether it is official, diagnostic, or pilot

## Run-Experiment Rule

When the user explicitly authorizes an experiment launch, follow `run-experiment`:

1. Read `AGENTS.md` and confirm target: `mdoc-remote`, `/home/lhz/MDocAgent`, conda env `mdocagent`.
2. Check GPU with `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`.
3. Activate env with `eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate mdocagent`.
4. Check API key presence without printing secret values.
5. Run dry-run/import/audit checks.
6. Launch exactly one approved job per `screen` session, using `tee` for logs.
7. Report screen name, command, log path, output path, and expected monitoring command.

## Monitor Rule

After launch, use `monitor-experiment` before interpreting:

- `screen -ls`
- screen hardcopy or tee log tail
- latest JSON/result files
- raw metric table before explanation
- crash or partial-run status if incomplete

## Audit Rule

Before paper claims, use `experiment-audit` on completed result directories and scripts. Claims must be downgraded or labeled if the audit finds scope, provenance, metric, or file-existence issues.

## Current Next Runs

The current planned sequence has two gated branches:

### Baseline/Reproduction Branch

1. R047: ARIS pre-flight for MMLongBench top-4.
2. R048: current-commit official top-4 reproduction.
3. R049: adapter original-only sanity.
4. R052: monitor plus integrity audit package for completed baseline outputs.

### Lightweight Evidence Layer Paper Branch

1. R071: Evidence Skill Graph registry design gate, no-provider.
2. R072: token-budgeted capsule renderer audit, no-provider.
3. R073: cross-dataset evidence-layer reuse audit, no-provider.
4. R074: small guarded capsule provider diagnostic, user-authorized only.
5. R075: bounded QA comparison against original MDocAgent, user-authorized only.
6. R076: component ablation.
7. R077: lightweight-vs-overbuilt graph comparison.
8. R078: ARIS paper-claim integrity audit.

Do not start provider or QA runs until the preceding no-provider gates pass and the user explicitly authorizes execution. R050/R051 are superseded as paper-claim candidates by the more focused R071-R078 evidence-layer branch unless revived explicitly.
