"""Tests for model configuration role audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.common.model_config import audit_model_configs


class ModelConfigAuditTest(unittest.TestCase):
    def test_model_config_audit_accepts_fixed_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = write_model_configs(root)
            stage2 = write_json(root / "stage2" / "manifest.json", {"model_id": None, "model_role": "none_or_fake", "evaluator_model_used": False})
            stage3 = write_json(root / "stage3" / "manifest.json", {"model_id": None, "model_role": "none_deterministic", "evaluator_model_used": False})
            stage4 = write_json(root / "stage4" / "manifest.json", {"model_id": None, "model_role": "none_rule_only", "evaluator_model_used": False})
            evaluation = write_json(
                root / "eval" / "manifest.json",
                {
                    "model_id": "deepseek-ai/DeepSeek-V3",
                    "model_role": "evaluation_judge",
                    "evaluator_model_used": True,
                    "evaluation_only": True,
                    "not_consumed_by_stage2_stage3_stage4": True,
                },
            )

            report = audit_model_configs(
                config_paths=configs,
                stage2_dirs=[stage2.parent],
                stage3_dirs=[stage3.parent],
                stage4_dirs=[stage4.parent],
                evaluation_dirs=[evaluation.parent],
                experiment_dirs=[],
            )

        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["stage2_model_violations"])
        self.assertFalse(report["evaluation_model_violations"])

    def test_stage_main_flow_deepseek_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = write_model_configs(root)
            stage2 = write_json(root / "stage2" / "manifest.json", {"model_id": "deepseek-ai/DeepSeek-V3", "model_role": "evaluation_judge"})

            report = audit_model_configs(
                config_paths=configs,
                stage2_dirs=[stage2.parent],
                stage3_dirs=[],
                stage4_dirs=[],
                evaluation_dirs=[],
                experiment_dirs=[],
            )

        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["stage2_model_violations"])

    def test_evaluation_deepseek_requires_evaluation_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = write_model_configs(root)
            evaluation = write_json(root / "eval" / "manifest.json", {"model_id": "deepseek-ai/DeepSeek-V3", "model_role": "evaluation_judge"})

            report = audit_model_configs(
                config_paths=configs,
                stage2_dirs=[],
                stage3_dirs=[],
                stage4_dirs=[],
                evaluation_dirs=[evaluation.parent],
                experiment_dirs=[],
            )

        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["evaluation_model_violations"])

    def test_nonempty_api_key_fails_audit_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = write_model_configs(root, qwen3_api_key="SECRET_VALUE")

            report = audit_model_configs(
                config_paths=configs,
                stage2_dirs=[],
                stage3_dirs=[],
                stage4_dirs=[],
                evaluation_dirs=[],
                experiment_dirs=[],
            )
            public_text = json.dumps(report, sort_keys=True)

        self.assertEqual(report["status"], "fail")
        self.assertIn("non_empty_api_key_in_model_config", public_text)
        self.assertNotIn("SECRET_VALUE", public_text)


def write_model_configs(root: Path, qwen3_api_key: str = "") -> list[Path]:
    config_root = root / "config" / "model"
    config_root.mkdir(parents=True)
    deepseek = config_root / "deepseekv3.yaml"
    qwen3 = config_root / "qwen3.yaml"
    qwen3vl = config_root / "qwen3vl.yaml"
    deepseek.write_text(
        "model_id: deepseek-ai/DeepSeek-V3\nmodel: deepseek-ai/DeepSeek-V3\napi_key:\napi_key_env: SILICONFLOW_API_KEY\nclass_name: SiliconFlowTextModel\nmodule_name: models.siliconflow\n",
        encoding="utf-8",
    )
    qwen3.write_text(
        f"model_id: Qwen/Qwen3-8B\nmodel: Qwen/Qwen3-8B\napi_key: {qwen3_api_key}\napi_key_env: SILICONFLOW_API_KEY\nclass_name: SiliconFlowTextModel\nmodule_name: models.siliconflow\n",
        encoding="utf-8",
    )
    qwen3vl.write_text(
        "model_id: Qwen/Qwen3-VL-8B-Instruct\nmodel: Qwen/Qwen3-VL-8B-Instruct\napi_key:\napi_key_env: SILICONFLOW_API_KEY\nclass_name: SiliconFlowVisionModel\nmodule_name: models.siliconflow\n",
        encoding="utf-8",
    )
    return [deepseek, qwen3, qwen3vl]


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
