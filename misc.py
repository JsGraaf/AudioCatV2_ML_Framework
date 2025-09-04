import logging
from pathlib import Path
from typing import Optional

import yaml


def load_config(path: Path = Path("config.yaml")) -> Optional[dict]:
    """
    Loads the yaml config
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error(f"Failed to load config: {exc}")
            return None
