import os
import sys
from pathlib import Path

os.environ["BACKEND"] = "local"  # before any bearlm import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
