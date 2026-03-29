"""Persona engine: load bias personas from YAML config, render Jinja2 prompts.

Adding a new persona requires only:
1. A new entry in prompts/personas.yaml
2. (Optional) a custom Jinja2 template -- defaults to system.j2
No Python changes needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from ensemble.models import EventSnapshot

# Default paths relative to project root
DEFAULT_PERSONAS_PATH = Path("prompts/personas.yaml")
DEFAULT_TEMPLATES_DIR = Path("prompts/templates")


class PersonaConfig(BaseModel):
    """A single persona definition loaded from YAML."""

    name: str
    id: str
    bias_type: str
    description: str
    traits: list[str]
    system_template: str = "system.j2"
    user_template: str = "user.j2"


def load_personas(path: Path = DEFAULT_PERSONAS_PATH) -> list[PersonaConfig]:
    """Load all persona definitions from a YAML file.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the YAML structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Personas config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "personas" not in raw:
        raise ValueError(f"Expected top-level 'personas' key in {path}")

    personas = []
    for i, entry in enumerate(raw["personas"]):
        try:
            personas.append(PersonaConfig.model_validate(entry))
        except Exception as e:
            raise ValueError(f"Invalid persona at index {i}: {e}") from e

    return personas


def get_persona(persona_id: str, path: Path = DEFAULT_PERSONAS_PATH) -> PersonaConfig:
    """Load a specific persona by ID.

    Raises KeyError if the persona ID is not found.
    """
    personas = load_personas(path)
    for p in personas:
        if p.id == persona_id:
            return p
    available = [p.id for p in personas]
    raise KeyError(f"Persona '{persona_id}' not found. Available: {available}")


def _get_jinja_env(templates_dir: Path = DEFAULT_TEMPLATES_DIR) -> Environment:
    """Create a Jinja2 environment with strict undefined variable checking."""
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_system_prompt(
    persona: PersonaConfig,
    balance: float = 100.0,
    templates_dir: Path = DEFAULT_TEMPLATES_DIR,
) -> str:
    """Render the system prompt for a persona.

    Args:
        persona: Persona configuration
        balance: Current portfolio balance in dollars
        templates_dir: Directory containing Jinja2 templates
    """
    env = _get_jinja_env(templates_dir)
    template = env.get_template(persona.system_template)
    return template.render(persona=persona, balance=balance)


def render_user_prompt(
    persona: PersonaConfig,
    snapshot: EventSnapshot,
    balance: float = 100.0,
    templates_dir: Path = DEFAULT_TEMPLATES_DIR,
) -> str:
    """Render the user prompt for a persona + market snapshot.

    Args:
        persona: Persona configuration
        snapshot: LLM-safe market snapshot (no outcome data)
        balance: Current portfolio balance in dollars
        templates_dir: Directory containing Jinja2 templates
    """
    env = _get_jinja_env(templates_dir)
    template = env.get_template(persona.user_template)
    return template.render(persona=persona, snapshot=snapshot, balance=balance)
