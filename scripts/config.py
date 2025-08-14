import yaml
from pathlib import Path
from typing import Dict

def load_config(config_path: str) -> Dict:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config