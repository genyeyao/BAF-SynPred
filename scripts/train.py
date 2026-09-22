"""Repository-level training entry point."""

import os
import runpy
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

runpy.run_module("bafsynpred.train", run_name="__main__")
