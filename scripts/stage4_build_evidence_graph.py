#!/usr/bin/env python3
"""Command wrapper for Stage 4A minimal structural evidence graph."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage4.evidence_graph import main

main()
