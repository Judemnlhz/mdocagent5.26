#!/usr/bin/env python3
"""Audit existing config/model/*.yaml role usage and public model metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.common.model_config import audit_model_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit model config roles and public output model metadata.")
    parser.add_argument("--config", action="append", dest="config_paths", help="Model config YAML to check. Repeatable.")
    parser.add_argument("--stage2-dir", action="append", dest="stage2_dirs")
    parser.add_argument("--stage3-dir", action="append", dest="stage3_dirs")
    parser.add_argument("--stage4-dir", action="append", dest="stage4_dirs")
    parser.add_argument("--evaluation-dir", action="append", dest="evaluation_dirs")
    parser.add_argument("--experiment-dir", action="append", dest="experiment_dirs")
    parser.add_argument("--output", default="outputs/audits/model_config_audit_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_model_configs(
        config_paths=args.config_paths or ("config/model/deepseekv3.yaml", "config/model/qwen3.yaml", "config/model/qwen3vl.yaml"),
        stage2_dirs=args.stage2_dirs or ("outputs/stage2_doc",),
        stage3_dirs=args.stage3_dirs or ("outputs/stage3_doc_artifact_retrieval",),
        stage4_dirs=args.stage4_dirs or ("outputs/stage4/evidence_graph",),
        evaluation_dirs=args.evaluation_dirs or ("outputs/eval",),
        experiment_dirs=args.experiment_dirs or ("outputs/experiments/matrix",),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
