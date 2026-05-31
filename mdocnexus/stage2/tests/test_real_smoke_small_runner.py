"""Tests for the bounded real smoke runner safety gates."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from scripts.run_real_smoke_small import build_parser, main, validate_args


class RealSmokeSmallRunnerTest(unittest.TestCase):
    def test_default_dry_run_does_not_execute_real_api(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([])
        payload = json.loads(buffer.getvalue())

        self.assertFalse(payload["will_execute"])
        self.assertTrue(payload["requires_execute"])
        self.assertTrue(payload["requires_enable_real_api"])
        self.assertTrue(payload["requires_run_real_trial"])

    def test_execute_requires_double_confirmation(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--execute"])

        with self.assertRaises(RuntimeError):
            validate_args(args)

    def test_max_pages_total_above_cap_fails(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--max-pages-total", "6"])

        with self.assertRaises(RuntimeError):
            validate_args(args)

    def test_public_dry_run_output_has_no_raw_or_path_leakage(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["--max-pages-total", "3"])
        text = buffer.getvalue()

        for forbidden in ("raw_response", "raw_output", "data:image", "file:///home", "api_key", "secret", "image_path"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
