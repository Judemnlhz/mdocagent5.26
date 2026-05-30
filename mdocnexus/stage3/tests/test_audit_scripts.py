"""Tests for public leakage and reproducibility audit scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


class AuditScriptsTest(unittest.TestCase):
    def test_no_gold_leakage_audit_allows_safe_sentinel_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "stage3"
            output_dir.mkdir()
            write_jsonl(
                output_dir / "retrieval.jsonl",
                [
                    {
                        "record_id": "r1",
                        "retrieved_artifact_ids": ["a1"],
                        "no_answer_generation": True,
                        "no_gold_fields_used": True,
                    }
                ],
            )
            (output_dir / "quality_report.json").write_text(json.dumps({"num_outputs_with_answer_field": 0}), encoding="utf-8")

            completed = run_script("scripts/audit_no_gold_leakage.py", "--scan-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('"status": "pass"', completed.stdout)

    def test_no_gold_leakage_audit_fails_on_injected_answer_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "stage3"
            output_dir.mkdir()
            write_jsonl(output_dir / "retrieval.jsonl", [{"record_id": "r1", "answer": "leaked"}])

            completed = run_script("scripts/audit_no_gold_leakage.py", "--scan-dir", str(output_dir))

        self.assertEqual(completed.returncode, 1)
        self.assertIn("forbidden_field:answer", completed.stdout)

    def test_reproducibility_audit_accepts_canonical_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage3_dir = root / "stage3"
            stage3_dir.mkdir()
            retrieval_rows = [{"record_id": "r1", "retrieved_artifact_ids": ["a1"], "retrieval_scores": [1.0]}]
            quality = {"num_queries": 1, "retrieval_method": "deterministic_lexical"}
            write_jsonl(stage3_dir / "retrieval.jsonl", retrieval_rows)
            (stage3_dir / "quality_report.json").write_text(json.dumps(quality, sort_keys=True), encoding="utf-8")
            manifest = {
                "retrieval_hash": canonical_json_hash(retrieval_rows),
                "quality_report_hash": canonical_json_hash(quality),
            }
            (stage3_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = run_script(
                "scripts/audit_reproducibility.py",
                "--stage2-dir",
                str(root / "missing_stage2"),
                "--stage3-dir",
                str(stage3_dir),
                "--stage4-dir",
                str(root / "missing_stage4"),
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('"status": "pass"', completed.stdout)

    def test_reproducibility_audit_fails_on_changed_retrieval_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage3_dir = root / "stage3"
            stage3_dir.mkdir()
            original_rows = [{"record_id": "r1", "retrieved_artifact_ids": ["a1"], "retrieval_scores": [1.0]}]
            changed_rows = [{"record_id": "r1", "retrieved_artifact_ids": ["a2"], "retrieval_scores": [1.0]}]
            quality = {"num_queries": 1}
            write_jsonl(stage3_dir / "retrieval.jsonl", changed_rows)
            (stage3_dir / "quality_report.json").write_text(json.dumps(quality, sort_keys=True), encoding="utf-8")
            manifest = {
                "retrieval_hash": canonical_json_hash(original_rows),
                "quality_report_hash": canonical_json_hash(quality),
            }
            (stage3_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = run_script(
                "scripts/audit_reproducibility.py",
                "--stage2-dir",
                str(root / "missing_stage2"),
                "--stage3-dir",
                str(stage3_dir),
                "--stage4-dir",
                str(root / "missing_stage4"),
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("retrieval_hash", completed.stdout)


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = payload.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
