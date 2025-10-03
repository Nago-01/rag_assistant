import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "code")

APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")
