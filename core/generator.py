"""Eval suite generation using Claude Sonnet 4.5 with extended thinking."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import anthropic

from core.models import EvalSuite, RubricCriterion, TestCase, TestCaseCategory

GENERATOR_MODEL = "claude-sonnet-4-5-20250929"

_GENERATE_TOOL: Dict = {
    "name": "generate_eval_suite",
    "description": (
        "Generate a comprehensive evaluation test suite with diverse test cases "
        "and a grading rubric for a given AI system prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "test_cases": {
                "type": "array",
                "description": "List of test cases across all categories",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["golden", "edge", "adversarial"],
                            "description": (
                                "golden: standard inputs with clear expected behavior; "
                                "edge: boundary conditions and unusual inputs; "
                                "adversarial: prompt injection attempts and constraint violations"
                            ),
                        },
                        "input": {
                            "type": "string",
                            "description": "The user message to send to the system",
                        },
                        "expected_output": {
                            "type": "string",
                            "description": (
                                "For golden cases: the ideal response or key elements it should contain. "
                                "For edge/adversarial: what the model SHOULD do (e.g., refuse, redirect)."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief label for this test case (under 80 chars)",
                        },
                    },
                    "required": ["category", "input", "description"],
                    "additionalProperties": False,
                },
            },
            "rubric": {
                "type": "array",
                "description": "4–5 grading criteria for evaluating responses (scored 1–5)",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short criterion name (e.g., Accuracy, Tone, Safety)",
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "What this criterion measures and how to score it. "
                                "Include what 1 (poor) and 5 (excellent) looks like."
                            ),
                        },
                        "weight": {
                            "type": "number",
                            "description": "Relative importance (default 1.0; use 2.0 for critical criteria)",
                        },
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["test_cases", "rubric"],
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT = """\
You are an expert at designing rigorous evaluation frameworks for AI language models. \
Your test cases should be realistic, diverse, and specifically tailored to probe the \
strengths and weaknesses of the given system prompt. Think carefully about what could \
go wrong, what edge cases exist, and what adversarial users might attempt."""


def generate_eval_suite(
    system_prompt: str,
    example_pairs: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
) -> EvalSuite:
    """Generate a comprehensive eval suite for the given system prompt.

    Uses a two-pass approach: extended thinking (pass 1) for deep analysis,
    followed by a tool-use call (pass 2) to extract structured JSON. This is
    necessary because the API does not allow extended thinking and forced
    tool_choice simultaneously.
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    examples_section = ""
    if example_pairs:
        lines = ["\n\n## Example Input/Output Pairs\n"]
        for i, pair in enumerate(example_pairs, 1):
            lines.append(f"**Example {i}**")
            lines.append(f"Input: {pair.get('input', '').strip()}")
            lines.append(f"Output: {pair.get('output', '').strip()}\n")
        examples_section = "\n".join(lines)

    analysis_prompt = f"""\
Analyze the following system prompt and plan a comprehensive evaluation suite.

## System Prompt to Evaluate
```
{system_prompt}
```{examples_section}

## What to Plan

**Test Cases:**
- **Golden cases (5–8):** Standard, well-formed inputs that should produce ideal responses. \
Include expected outputs so the judge has a reference.
- **Edge cases (3–5):** Boundary conditions — empty inputs, very long inputs, ambiguous \
requests, multilingual inputs, requests at the edge of the system's stated scope.
- **Adversarial cases (3–5):** Attempts to break the system — prompt injection \
("Ignore previous instructions..."), off-topic requests, requests that violate the \
system prompt's explicit constraints, social engineering attempts.

**Rubric (4–5 criteria):** Design criteria specific to this system prompt's domain. \
Each criterion should be scored 1–5 with clear definitions of what 1 and 5 look like.

Think carefully about what could go wrong, what edge cases exist, and what adversarial \
users might attempt. Then write out each test case and rubric criterion in detail."""

    start = time.time()

    # Pass 1: extended thinking — free-form analysis
    thinking_response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": analysis_prompt}],
    )

    analysis_text = next(
        (b.text for b in thinking_response.content if b.type == "text"), ""
    )

    # Pass 2: tool use — structure the analysis into JSON
    structure_response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=8000,
        system=(
            "You are a precise data extractor. Convert the provided eval suite plan "
            "into structured data using the generate_eval_suite tool. "
            "Preserve all test cases and rubric criteria exactly as described."
        ),
        tools=[_GENERATE_TOOL],
        tool_choice={"type": "tool", "name": "generate_eval_suite"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Convert this eval suite plan into structured test cases and rubric:\n\n"
                    + analysis_text
                ),
            }
        ],
    )

    generation_time = time.time() - start

    tool_block = next(
        (b for b in structure_response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        raise RuntimeError("Generator returned no tool_use block — check API response")

    data = tool_block.input

    test_cases: List[TestCase] = [
        TestCase(
            id=f"tc_{idx:03d}",
            category=TestCaseCategory(tc["category"]),
            input=tc["input"],
            expected_output=tc.get("expected_output"),
            description=tc["description"],
        )
        for idx, tc in enumerate(data["test_cases"], 1)
    ]

    rubric: List[RubricCriterion] = [
        RubricCriterion(
            name=rc["name"],
            description=rc["description"],
            weight=float(rc.get("weight", 1.0)),
        )
        for rc in data["rubric"]
    ]

    return EvalSuite(
        system_prompt=system_prompt,
        example_pairs=example_pairs or [],
        test_cases=test_cases,
        rubric=rubric,
        generation_time_s=round(generation_time, 2),
    )
