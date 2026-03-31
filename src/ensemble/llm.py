"""Async OpenAI LLM caller with structured output for persona decisions.

Uses gpt-5-nano with Pydantic v2 structured outputs to guarantee
parseable PersonaDecision responses. No web search tools are provided.

Note: gpt-5-nano does not support `temperature`. Persona divergence is
achieved entirely through system prompt engineering — each persona has
explicit decision rules, stake ranges, and skip thresholds.
"""

from __future__ import annotations

from openai import AsyncOpenAI, BadRequestError, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ensemble.models import Action, PersonaDecision
from ensemble.personas import PersonaConfig, render_system_prompt, render_user_prompt
from ensemble.models import EventSnapshot


RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
)
async def call_persona(
    client: AsyncOpenAI,
    persona: PersonaConfig,
    snapshot: EventSnapshot,
    balance: float,
    model: str = "gpt-5-nano",
) -> PersonaDecision:
    """Call the LLM as a specific persona and return a structured decision.

    Args:
        client: Async OpenAI client (shared across calls)
        persona: Which bias persona to use
        snapshot: LLM-safe market data (no outcome)
        balance: Current portfolio balance for this persona
        model: Model ID to use

    Returns:
        PersonaDecision with action, stake_dollars, and reasoning
    """
    system_prompt = render_system_prompt(persona, balance=balance)
    user_prompt = render_user_prompt(persona, snapshot, balance=balance)

    response = await client.responses.parse(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        text_format=PersonaDecision,
    )

    decision = response.output_parsed

    # Enforce stake constraints
    if decision.action == Action.SKIP:
        decision.stake_dollars = 0.0
    decision.stake_dollars = min(decision.stake_dollars, balance)
    decision.stake_dollars = max(decision.stake_dollars, 0.0)

    return decision
