import os
from pathlib import Path

# Root directory of your project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Environment file
ENV_FPATH = ROOT_DIR / ".env"

# Code and config paths
CODE_DIR = ROOT_DIR / "code"
APP_CONFIG_FPATH = CODE_DIR / "config" / "config.yaml"
PROMPT_CONFIG_FPATH = CODE_DIR / "config" / "prompt_config.yaml"

# Output directory
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data directory
DATA_DIR = ROOT_DIR / "data"

# Get all publication files
PUBLICATION_FPATH = list(DATA_DIR.glob("*.*"))

# Filter to only common publication types
PUBLICATION_FILES = [
    f for f in PUBLICATION_FPATH
    if f.suffix.lower() in [".md", ".txt", ".pdf", ".doc", ".docx"]
]
