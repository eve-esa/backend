"""
Utility for loading and using YAML templates.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Directory of the templates folder
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def load_yaml_template(filename: str) -> Dict[str, Any]:
    """
    Load a YAML template file.

    Args:
        filename: Name of the YAML file in the templates directory

    Returns:
        Dictionary containing the template data

    Raises:
        FileNotFoundError: If the template file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    template_path = TEMPLATES_DIR / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_template(template_name: str, filename: str = "prompts.yaml") -> str:
    """
    Get a specific template from a YAML file.

    Args:
        template_name: Name of the template within the YAML file
        filename: Name of the YAML file in the templates directory

    Returns:
        Template string

    Raises:
        KeyError: If the template name doesn't exist in the file
    """
    templates = load_yaml_template(filename)
    if template_name not in templates:
        raise KeyError(f"Template '{template_name}' not found in {filename}")

    return templates[template_name]


def format_template(template_name: str, **kwargs) -> str:
    """
    Load and format a template with the given parameters.

    Args:
        template_name: Name of the template within the prompts.yaml file
        **kwargs: Parameters to format into the template

    Returns:
        Formatted template string
    """
    template = get_template(template_name)
    return template.format(**kwargs)
