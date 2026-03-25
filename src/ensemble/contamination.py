"""Contamination detection: probes whether gpt-5-nano knows event outcomes from training data.

If the model already knows an outcome, the event is scientifically invalid for
the ensemble prediction experiment. This module checks each event and classifies
it as INCLUDE, FLAG, or EXCLUDE.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from ensemble.models import Event

load_dotenv()


CONTAMINATION_CHECK_SYSTEM = """You are a research assistant helping detect training data contamination in a prediction market experiment.

Your task: determine honestly whether you already know the outcome of the event described below. Scientific integrity depends on your honesty.

Rules:
- If you know or strongly suspect the outcome, say so clearly
- If you're uncertain, say so
- If you have no knowledge of this specific event's outcome, say so
- Do NOT guess or reason about what the outcome might be — only report what you already know from your training data
- Be honest — a false negative (claiming ignorance when you know) invalidates the experiment"""

CONTAMINATION_CHECK_USER = """Event: {title}
Question: {question}
Category: {category}
Expected resolution date: {close_date}

Do you already know how this event resolved from your training data? Report honestly."""


class ContaminationResult(BaseModel):
    """Result of a contamination check for a single event."""

    knows_outcome: bool
    confidence: Literal["high", "medium", "low", "none"]
    stated_outcome: str | None = None
    reasoning: str


def score_contamination(result: ContaminationResult) -> Literal["INCLUDE", "FLAG", "EXCLUDE"]:
    """Score a contamination result into a verdict.

    - EXCLUDE: model knows the outcome with high confidence
    - FLAG: model knows with medium confidence (review manually)
    - INCLUDE: model doesn't know or has low/no confidence
    """
    if not result.knows_outcome:
        return "INCLUDE"
    if result.confidence == "high":
        return "EXCLUDE"
    if result.confidence == "medium":
        return "FLAG"
    return "INCLUDE"


async def check_contamination(
    event: Event,
    model: str = "gpt-5-nano",
) -> ContaminationResult:
    """Check whether the LLM knows the outcome of an event.

    Calls the OpenAI API with a contamination check prompt. Never reveals
    the actual outcome to the model.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_message = CONTAMINATION_CHECK_USER.format(
        title=event.title,
        question=event.question,
        category=event.category,
        close_date=event.close_time.strftime("%Y-%m-%d"),
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": CONTAMINATION_CHECK_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_format=ContaminationResult,
            temperature=0,
        )
        result = response.choices[0].message.parsed
        assert result is not None
        return result
    except Exception as e:
        return ContaminationResult(
            knows_outcome=False,
            confidence="none",
            stated_outcome=None,
            reasoning=f"API call failed: {e}",
        )


async def check_all_events(
    events: list[Event],
    model: str = "gpt-5-nano",
) -> list[tuple[Event, ContaminationResult]]:
    """Run contamination checks sequentially for all events.

    Sequential (not parallel) to be rate-limit friendly. Adds 1-second
    delay between calls.
    """
    results: list[tuple[Event, ContaminationResult]] = []
    for event in events:
        result = await check_contamination(event, model)
        results.append((event, result))
        await asyncio.sleep(1)
    return results
