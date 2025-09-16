import yaml
from pathlib import Path

# Directory of this package (src/hallucination_pipeline)
BASE_DIR = Path(__file__).parent


def load_yaml(filename: str):
    with open(BASE_DIR / filename, "r") as f:
        return yaml.safe_load(f)


# Load YAMLs co-located with this module
PROMPTS = load_yaml("../templates/hallucination.yaml")
CONFIG = load_yaml("config.yaml")
