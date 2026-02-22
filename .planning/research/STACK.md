# Stack Research

**Domain:** LLM multi-agent ensemble / prediction market simulation (Python CLI)
**Researched:** 2026-02-22
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11+ | Runtime | Native asyncio, strong typing, dominant in ML/research |
| openai SDK | >=1.60 | LLM calls to gpt-5-nano | Official SDK, supports structured outputs via Pydantic, async-native |
| pydantic | v2 | Decision schema validation, structured outputs | OpenAI SDK structured outputs require Pydantic v2 models; zero-parse overhead |
| pandas | 2.x | Market data manipulation, results analysis | Standard for tabular research data; CSV/JSON in/out |
| asyncio | stdlib | Parallel persona LLM calls | Fire all persona calls concurrently per time window |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | >=8.0 | Retry logic for API failures | Wrap all OpenAI calls; handles rate limits and transient errors |
| rich | >=13.0 | CLI tables, progress bars, colored output | Displaying persona decisions and portfolio summaries in terminal |
| typer | >=0.12 | CLI interface | Simple, type-annotated CLI commands with --help out of the box |
| python-dotenv | >=1.0 | OPENAI_API_KEY management | Load API keys from .env without hardcoding |
| jinja2 | >=3.0 | Persona prompt templating | Separate prompt templates from Python code; easy to iterate on personas |
| jsonlines | >=4.0 | Streaming result storage | Append-safe output for long runs; survives interruption |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Package management + virtualenv | Faster than pip; `uv sync` installs from pyproject.toml |
| ruff | Linting + formatting | Replaces black + flake8; single tool |
| pytest + pytest-asyncio | Testing | Required for testing async LLM orchestration logic |
| pyproject.toml | Project config | Defines deps, scripts, tool config in one file |

## Installation

```bash
# Create project with uv
uv init llm-prediction-ensemble
cd llm-prediction-ensemble

# Core dependencies
uv add openai pydantic pandas tenacity rich typer python-dotenv jinja2 jsonlines

# Dev dependencies
uv add --dev pytest pytest-asyncio ruff
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| openai SDK (direct) | LangChain | LangChain if you need agent orchestration framework with memory — overkill for this research tool |
| asyncio (native) | ThreadPoolExecutor | Threads if LLM calls were CPU-bound; they're IO-bound so async is correct |
| pandas | polars | Polars for larger datasets (>1M rows); pandas is simpler for 15-event research scale |
| typer | click | Click for more complex CLI routing; typer is sufficient and less boilerplate |
| jinja2 templates | f-strings | f-strings for simple prompts; jinja2 when persona prompts become multi-section with conditionals |
| pydantic structured outputs | manual JSON parsing | Manual parsing if model doesn't support structured outputs; gpt-5-nano does |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| LangChain | Abstracts away OpenAI call details, hides token usage, adds unnecessary complexity for a focused research script | openai SDK directly |
| requests/httpx directly | OpenAI SDK handles auth, retry, rate limits, streaming — reinventing it is error-prone | openai SDK |
| GPT-5-nano web_search tool | Enabling web search in API calls contaminates the experiment — model can look up outcomes | Never pass `tools=[{"type":"web_search"}]` |
| SQLite for state | Overkill for 15-event simulation; adds schema migration complexity | JSON/CSV files + pandas |
| Celery/task queues | Designed for distributed workers; asyncio handles parallel persona calls within a single process | asyncio.gather() |

## Stack Patterns by Variant

**For structured decision output (YES/NO/SKIP + stake):**
- Use OpenAI structured outputs with `response_format=PersonaDecision` (Pydantic model)
- Ensures decisions are always parseable, never free-text
- Set `temperature=0` for reproducibility

**For parallel persona calls per time window:**
- Use `asyncio.gather(*[call_persona(p, data) for p in personas])`
- All personas see the same data simultaneously — true blind phase

**For experiment reproducibility:**
- Save full config snapshot (personas, event IDs, time windows, model, temperature) alongside results
- Use `datetime` timestamp in output filenames

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| openai >=1.60 | pydantic v2 | openai SDK requires pydantic v2 for structured outputs |
| pytest-asyncio >=0.23 | pytest >=7.0 | Must set `asyncio_mode = "auto"` in pytest.ini |
| pandas 2.x | Python 3.11+ | pandas 2.0 dropped Python 3.8 support |

## Sources

- OpenAI Python SDK docs — structured outputs, async usage
- gpt-5-nano model page — confirms structured outputs supported, no fine-tuning
- Standard Python research tooling conventions (2025/2026)

---
*Stack research for: LLM ensemble prediction market simulation*
*Researched: 2026-02-22*
