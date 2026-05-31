"""Smoke tests for the unified CLI dry-run commands."""

from __future__ import annotations

import json
import subprocess
import unittest


class UnifiedCliTest(unittest.TestCase):
    def test_unified_cli_help_and_dry_runs(self) -> None:
        commands = [
            ["python3", "scripts/mdocnexus.py", "--help"],
            ["python3", "scripts/mdocnexus.py", "audit", "--help"],
            ["python3", "scripts/mdocnexus.py", "run-matrix", "--dry-run"],
            ["python3", "scripts/mdocnexus.py", "run-real-smoke-small", "--dry-run"],
        ]
        for command in commands:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        matrix = subprocess.run(commands[2], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        payload = json.loads(matrix.stdout)
        self.assertFalse(payload["will_execute"])
        self.assertGreaterEqual(payload["num_runs"], 5)


if __name__ == "__main__":
    unittest.main()
