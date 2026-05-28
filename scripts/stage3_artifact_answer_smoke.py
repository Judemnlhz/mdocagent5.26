#!/usr/bin/env python3
"""Command wrapper for Stage 3B artifact-only answer smoke."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage3.artifact_answer_smoke import main

main()
