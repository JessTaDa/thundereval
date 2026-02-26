"""Multi-model eval runner using asyncio for parallel execution."""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

import anthropic

from core.models import EvalSuite, ModelResult
from utils.cost import BENCHMARK_MODELS, calculate_cost

# Max concurrent API connections — stays well under Anthropic's limit
MAX_CONCURRENT = 5


async def _run_single(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    test_case_input: str,
    test_case_id: str,
    semaphore: asyncio.Semaphore,
) -> ModelResult:
    """Run one model on one test case, gated by a semaphore."""
    async with semaphore:
        start = time.monotonic()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                # Cache the system prompt — same content repeated across all test cases
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": test_case_input}],
            )

            latency_ms = (time.monotonic() - start) * 1000
            text_parts = [b.text for b in response.content if hasattr(b, "text")]

            return ModelResult(
                model=model,
                test_case_id=test_case_id,
                response="\n".join(text_parts),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                cost=calculate_cost(
                    model, response.usage.input_tokens, response.usage.output_tokens
                ),
            )

        except anthropic.RateLimitError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return ModelResult(
                model=model,
                test_case_id=test_case_id,
                response="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cost=0.0,
                error=f"Rate limit: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000
            return ModelResult(
                model=model,
                test_case_id=test_case_id,
                response="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cost=0.0,
                error=str(exc),
            )


async def run_benchmark(
    eval_suite: EvalSuite,
    api_key: Optional[str] = None,
) -> List[ModelResult]:
    """Run all benchmark models against all test cases in parallel.

    A semaphore caps concurrent connections at MAX_CONCURRENT to avoid
    hitting Anthropic's concurrent-connection rate limit, while still
    running all tasks as a pool (much faster than purely sequential).

    Args:
        eval_suite: The suite to run (uses its system_prompt and test_cases).
        api_key: Optional API key override.

    Returns:
        List of ModelResult objects (one per model × test_case combination).
    """
    client = (
        anthropic.AsyncAnthropic(api_key=api_key)
        if api_key
        else anthropic.AsyncAnthropic()
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _run_single(
            client=client,
            model=model,
            system_prompt=eval_suite.system_prompt,
            test_case_input=tc.input,
            test_case_id=tc.id,
            semaphore=semaphore,
        )
        for model in BENCHMARK_MODELS
        for tc in eval_suite.test_cases
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in raw_results if isinstance(r, ModelResult)]
