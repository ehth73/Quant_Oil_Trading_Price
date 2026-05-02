import os, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def getenv(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
