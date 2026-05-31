#!/usr/bin/env python3
"""Command wrapper for Stage 3A artifact retrieval dry run."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage3.artifact_retrieval_dryrun import main

main()
