# ⚡ ThunderEval

Automated evaluation framework for Claude. Give it a system prompt, and it generates a full test suite, benchmarks responses across three models in parallel, and scores everything with an LLM-as-judge — all in a Streamlit dashboard.

---

## How it works

| Step | Description | Model |
|------|-------------|-------|
| 1. Generate | Analyzes your system prompt to create golden, edge, and adversarial test cases plus a grading rubric | Claude Sonnet 4.5 + extended thinking |
| 2. Benchmark | Fires all test cases against all models in parallel with prompt caching | Haiku 4.5 / Sonnet 4.5 / Opus 4.5 |
| 3. Judge | Grades each response against the rubric using blind LLM-as-judge evaluation | Claude Sonnet 4.5 |
| 4. Analyze | Side-by-side comparison of scores, cost, and latency with exportable report | — |

---

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Or copy `.env.example` to `.env` and add your key there.

---

## Project structure

```
thundereval/
├── app.py                  # Streamlit app
├── core/
│   ├── generator.py        # Eval generation with extended thinking
│   ├── runner.py           # Async parallel benchmark runner
│   ├── judge.py            # LLM-as-judge scoring
│   └── models.py           # Pydantic data models
├── ui/
│   ├── components.py       # Streamlit components
│   └── charts.py           # Plotly charts
├── utils/
│   ├── cost.py             # Token cost calculation
│   └── export.py           # Markdown/JSON report export
└── examples/               # Pre-built system prompts to try
```

---

## Design notes

**Two-pass generation** — The eval generator uses extended thinking on pass 1 to deeply reason about the system prompt, then a forced tool-use call on pass 2 to extract structured JSON. This split is necessary because the Anthropic API doesn't allow extended thinking and `tool_choice` in the same request.

**Parallel execution** — The benchmark runner uses `asyncio.gather()` with a semaphore to fire all API calls concurrently while staying within rate limits. Wall-clock time ≈ slowest single call, not the sum.

**Prompt caching** — `cache_control: {type: "ephemeral"}` is applied to the system prompt on every model call, since it's identical across all test cases for a given run.

**Blind judging** — The judge prompt never reveals which model produced a response. Only the response text is shown, preventing any bias based on model reputation.

---

## Examples

Three pre-built system prompts are included to get started quickly:

- **Contract Analysis** — Legal AI that identifies risks, obligations, and key terms
- **Customer Support** — SaaS support agent for billing, features, and troubleshooting
- **Code Review** — Senior engineer reviewing PRs for bugs and security issues

---

## Cost

| Model | Input | Output |
|-------|-------|--------|
| Claude Haiku 4.5 | $0.80 / MTok | $4.00 / MTok |
| Claude Sonnet 4.5 | $3.00 / MTok | $15.00 / MTok |
| Claude Opus 4.5 | $5.00 / MTok | $25.00 / MTok |

A typical full benchmark run costs $0.05–0.50 depending on test case count.

---

## Requirements

- Python 3.11+
- Anthropic API key with access to the Claude 4.5 model family
