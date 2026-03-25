"""Tests for contamination check scoring logic and prompt construction.

NOTE: Does NOT test actual API calls (requires real API key).
Only tests scoring logic and prompt safety.
"""

from ensemble.contamination import (
    CONTAMINATION_CHECK_SYSTEM,
    CONTAMINATION_CHECK_USER,
    ContaminationResult,
    score_contamination,
)


class TestScoreContamination:
    """Verify scoring logic classifies verdicts correctly."""

    def test_score_contamination_exclude(self) -> None:
        """knows_outcome=True + confidence='high' -> EXCLUDE."""
        result = ContaminationResult(
            knows_outcome=True,
            confidence="high",
            stated_outcome="yes",
            reasoning="I know this happened.",
        )
        assert score_contamination(result) == "EXCLUDE"

    def test_score_contamination_flag(self) -> None:
        """knows_outcome=True + confidence='medium' -> FLAG."""
        result = ContaminationResult(
            knows_outcome=True,
            confidence="medium",
            stated_outcome="no",
            reasoning="I think I recall this.",
        )
        assert score_contamination(result) == "FLAG"

    def test_score_contamination_include_low(self) -> None:
        """knows_outcome=True + confidence='low' -> INCLUDE."""
        result = ContaminationResult(
            knows_outcome=True,
            confidence="low",
            stated_outcome=None,
            reasoning="Very vague impression.",
        )
        assert score_contamination(result) == "INCLUDE"

    def test_score_contamination_include_false(self) -> None:
        """knows_outcome=False (regardless of confidence) -> INCLUDE."""
        result = ContaminationResult(
            knows_outcome=False,
            confidence="high",
            stated_outcome=None,
            reasoning="I don't know this event.",
        )
        assert score_contamination(result) == "INCLUDE"

    def test_score_contamination_include_none(self) -> None:
        """confidence='none' -> INCLUDE."""
        result = ContaminationResult(
            knows_outcome=False,
            confidence="none",
            stated_outcome=None,
            reasoning="No knowledge.",
        )
        assert score_contamination(result) == "INCLUDE"


class TestPromptSafety:
    """Verify prompts never reveal outcome data to the model."""

    def test_contamination_prompt_excludes_outcome(self) -> None:
        """User prompt template must NOT contain outcome-revealing words."""
        forbidden = ["outcome", "result", "settlement"]
        template_lower = CONTAMINATION_CHECK_USER.lower()
        for word in forbidden:
            assert word not in template_lower, (
                f"CONTAMINATION_CHECK_USER contains forbidden word '{word}'. "
                "The prompt must never reveal outcome data to the model."
            )

    def test_contamination_prompt_includes_required_fields(self) -> None:
        """Template must contain all required placeholders."""
        required_placeholders = ["{title}", "{question}", "{category}", "{close_date}"]
        for placeholder in required_placeholders:
            assert placeholder in CONTAMINATION_CHECK_USER, (
                f"CONTAMINATION_CHECK_USER missing required placeholder '{placeholder}'"
            )

    def test_system_prompt_emphasizes_honesty(self) -> None:
        """System prompt should emphasize honest reporting."""
        assert "honest" in CONTAMINATION_CHECK_SYSTEM.lower()
        assert "scientific" in CONTAMINATION_CHECK_SYSTEM.lower()
