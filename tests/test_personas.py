"""Tests for the Persona Engine (Phase 2).

Validates:
- YAML config loading and validation
- 6 personas with required fields
- Jinja2 template rendering
- No outcome leakage in rendered prompts
- Config-driven extensibility (no Python changes needed)
"""

from pathlib import Path

import pytest

from ensemble.models import EventSnapshot, TimeWindowLabel
from ensemble.personas import (
    PersonaConfig,
    get_persona,
    load_personas,
    render_system_prompt,
    render_user_prompt,
)

PERSONAS_PATH = Path("prompts/personas.yaml")
TEMPLATES_DIR = Path("prompts/templates")


@pytest.fixture
def personas():
    """Load all personas from the project config."""
    if not PERSONAS_PATH.exists():
        pytest.skip("Personas config not yet created")
    return load_personas(PERSONAS_PATH)


@pytest.fixture
def sample_snapshot():
    """Create a sample EventSnapshot for template rendering tests."""
    from datetime import datetime, timezone

    return EventSnapshot(
        event_ticker="TEST-001",
        market_ticker="TEST-001-YES",
        title="Will it rain tomorrow?",
        question="Will it rain in New York City tomorrow?",
        description="Weather prediction market for NYC precipitation.",
        category="weather",
        yes_price_cents=65,
        no_price_cents=35,
        window=TimeWindowLabel.T_1D,
        snapshot_timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
    )


class TestPersonaLoading:
    def test_loads_six_personas(self, personas) -> None:
        """PERS-01: System includes exactly 6 behavioral bias personas."""
        assert len(personas) == 6, f"Expected 6 personas, got {len(personas)}"

    def test_all_personas_have_required_fields(self, personas) -> None:
        """Every persona has name, id, bias_type, description, traits."""
        for p in personas:
            assert p.name, f"Persona missing name"
            assert p.id, f"Persona {p.name} missing id"
            assert p.bias_type, f"Persona {p.name} missing bias_type"
            assert p.description, f"Persona {p.name} missing description"
            assert len(p.traits) >= 3, (
                f"Persona {p.name} has {len(p.traits)} traits, need at least 3"
            )

    def test_persona_ids_are_unique(self, personas) -> None:
        """No duplicate persona IDs."""
        ids = [p.id for p in personas]
        assert len(ids) == len(set(ids)), f"Duplicate persona IDs: {ids}"

    def test_expected_bias_types_present(self, personas) -> None:
        """All 6 required bias types are represented."""
        bias_types = {p.bias_type for p in personas}
        expected = {
            "overconfidence",
            "risk_aversion",
            "recency_bias",
            "base_rate_focus",
            "contrarian_bias",
            "anchoring_bias",
        }
        assert expected == bias_types, f"Bias types mismatch: expected {expected}, got {bias_types}"

    def test_get_persona_by_id(self) -> None:
        """Can retrieve a specific persona by ID."""
        persona = get_persona("contrarian")
        assert persona.id == "contrarian"
        assert persona.bias_type == "contrarian_bias"

    def test_get_persona_invalid_id_raises(self) -> None:
        """KeyError raised for nonexistent persona ID."""
        with pytest.raises(KeyError, match="not found"):
            get_persona("nonexistent_persona")


class TestPromptRendering:
    def test_system_prompt_contains_persona_name(self, personas) -> None:
        """System prompt includes the persona's name."""
        for p in personas:
            prompt = render_system_prompt(p, balance=100.0, templates_dir=TEMPLATES_DIR)
            assert p.name in prompt, f"System prompt missing persona name '{p.name}'"

    def test_system_prompt_contains_all_traits(self, personas) -> None:
        """System prompt includes every trait from the persona config."""
        for p in personas:
            prompt = render_system_prompt(p, balance=100.0, templates_dir=TEMPLATES_DIR)
            for trait in p.traits:
                assert trait in prompt, (
                    f"Persona {p.name}: trait '{trait}' missing from system prompt"
                )

    def test_system_prompt_contains_balance(self, personas) -> None:
        """System prompt shows the current portfolio balance."""
        persona = personas[0]
        prompt = render_system_prompt(persona, balance=73.50, templates_dir=TEMPLATES_DIR)
        assert "73.50" in prompt

    def test_user_prompt_contains_market_data(self, personas, sample_snapshot) -> None:
        """User prompt includes event title, question, and prices."""
        persona = personas[0]
        prompt = render_user_prompt(
            persona, sample_snapshot, balance=100.0, templates_dir=TEMPLATES_DIR
        )
        assert sample_snapshot.title in prompt
        assert sample_snapshot.question in prompt
        assert str(sample_snapshot.yes_price_cents) in prompt
        assert str(sample_snapshot.no_price_cents) in prompt

    def test_user_prompt_contains_window_label(self, personas, sample_snapshot) -> None:
        """User prompt shows which time window this snapshot is from."""
        persona = personas[0]
        prompt = render_user_prompt(
            persona, sample_snapshot, balance=100.0, templates_dir=TEMPLATES_DIR
        )
        assert sample_snapshot.window.value in prompt

    def test_rendered_prompts_exclude_outcome_words(self, personas, sample_snapshot) -> None:
        """CRITICAL: No rendered prompt should contain outcome/result/settlement."""
        forbidden = {"outcome", "result", "settlement", "resolution"}
        for p in personas:
            sys_prompt = render_system_prompt(p, balance=100.0, templates_dir=TEMPLATES_DIR)
            usr_prompt = render_user_prompt(
                p, sample_snapshot, balance=100.0, templates_dir=TEMPLATES_DIR
            )
            combined = (sys_prompt + usr_prompt).lower()
            leaked = {w for w in forbidden if w in combined}
            assert leaked == set(), (
                f"Persona {p.name}: prompt contains forbidden words: {leaked}"
            )


class TestConfigExtensibility:
    def test_persona_config_accepts_custom_template(self) -> None:
        """PERS-03: PersonaConfig supports custom template paths."""
        config = PersonaConfig(
            name="Custom",
            id="custom",
            bias_type="test_bias",
            description="A test persona",
            traits=["trait1", "trait2", "trait3"],
            system_template="custom_system.j2",
            user_template="custom_user.j2",
        )
        assert config.system_template == "custom_system.j2"
        assert config.user_template == "custom_user.j2"

    def test_invalid_yaml_raises_value_error(self, tmp_path) -> None:
        """Loading invalid YAML structure raises ValueError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("just_a_string: true\n")
        with pytest.raises(ValueError, match="Expected top-level"):
            load_personas(bad_yaml)

    def test_missing_file_raises_file_not_found(self) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_personas(Path("/nonexistent/personas.yaml"))
