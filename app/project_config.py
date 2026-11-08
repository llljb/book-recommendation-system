"""Portable paths for the published copy. Runtime data is not committed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BOOK_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
MODEL_DIR = Path(os.environ.get("BOOK_MODEL_DIR", str(ROOT / "saved_models"))).expanduser().resolve()
RUNTIME_DIR = ROOT / "runtime"
for directory in (DATA_DIR, MODEL_DIR, RUNTIME_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def data_path(name):
    return str(DATA_DIR / name)

def model_path(name):
    return str(MODEL_DIR / name)

def runtime_path(name):
    return str(RUNTIME_DIR / name)
