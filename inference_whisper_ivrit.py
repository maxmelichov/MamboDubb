#!/usr/bin/env python3
"""Compatibility wrapper. Prefer: python3 inference/transcribe.py"""

import runpy
import sys
from pathlib import Path

print(
    "Note: use `python3 inference/transcribe.py` (this wrapper remains for compatibility).",
    file=sys.stderr,
)
runpy.run_path(str(Path(__file__).resolve().parent / "inference" / "transcribe.py"), run_name="__main__")
