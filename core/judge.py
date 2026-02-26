"""LLM-as-judge scoring using Claude Sonnet 4.5 — blind evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import anthropic

from core.models import CriterionScore, EvalSuite, JudgeScore, ModelResult

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

_GRADE_TOOL: Dict = {
    "name": "grade_response",
    "description": "Score an AI response against each rubric criterion",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": "One score object per rubric criterion",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {
                            "type": "string",
                            "description": "Criterion name exactly as given in the rubric",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score from 1 (very poor) to 5 (excellent)",
                        },
                        "justification": {
                            "type": "string",
                            "description": "1–2 sentence explanation citing specific evidence from the response",
                        },
                    },
                    "required": ["criterion", "score", "justification"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

_JUDGE_SYSTEM = """\
You are an expert AI evaluator conducting a blind evaluation. You do NOT know which AI \
model produced the response you are grading. Evaluate solely on response quality against \
the provided rubric. Be fair, consistent, and specific — cite evidence from the response \
in your justifications. Use the full 1–5 scale; reserve 5 for genuinely excellent responses."""


def _judge_single(
    client: anthropic.Anthropic,
    eval_suite: EvalSuite,
    result: ModelResult,
) -> JudgeScore:
    tc = next((t for t in eval_suite.test_cases if t.id == result.test_case_id), None)
    if tc is None:
        raise ValueError(f"Test case {result.test_case_id!r} not found in suite")

    rubric_text = "\n".join(
        f"- **{rc.name}** (weight {rc.weight:.1f}): {rc.description}"
        for rc in eval_suite.rubric
    )

    expected_section = ""
    if tc.expected_output:
        expected_section = (
            f"\n**Reference / Expected Output** (use as a quality benchmark):\n"
            f"{tc.expected_output}\n"
        )

    user_message = f"""\
## Evaluation Task

**System Prompt the AI was following:**
{eval_suite.system_prompt}

---

**Test Case** (category: `{tc.category}`):
{tc.input}
{expected_section}
---

**Response to Evaluate:**
{result.response if result.response else "(No response — model returned an error)"}

---

## Grading Rubric
Score each criterion **1–5** (1 = very poor, 3 = acceptable, 5 = excellent):

{rubric_text}

Use the `grade_response` tool to provide structured scores with justifications."""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        system=_JUDGE_SYSTEM,
        tools=[_GRADE_TOOL],
        tool_choice={"type": "tool", "name": "grade_response"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise RuntimeError(f"Judge returned no tool_use block for {result.test_case_id}")

    data = tool_block.input
    criterion_scores: List[CriterionScore] = []
    for s in data["scores"]:
        criterion_scores.append(
            CriterionScore(
                criterion=s["criterion"],
                score=max(1, min(5, int(s["score"]))),
                justification=s["justification"],
            )
        )

    criterion_weights: Dict[str, float] = {rc.name.lower(): rc.weight for rc in eval_suite.rubric}
    total_weight = sum(criterion_weights.values()) or 1.0
    weighted_sum = sum(cs.score * criterion_weights.get(cs.criterion.lower(), 1.0) for cs in criterion_scores)

    return JudgeScore(
        model=result.model,
        test_case_id=result.test_case_id,
        criterion_scores=criterion_scores,
        overall_score=round(weighted_sum / total_weight, 2),
    )


def judge_all_results(
    eval_suite: EvalSuite,
    model_results: List[ModelResult],
    api_key: Optional[str] = None,
    progress_callback=None,
) -> List[JudgeScore]:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    gradeable = [r for r in model_results if not r.error]
    scores: List[JudgeScore] = []

    for idx, result in enumerate(gradeable):
        try:
            scores.append(_judge_single(client, eval_suite, result))
        except Exception as exc:  # noqa: BLE001
            print(f"[judge] {result.model}/{result.test_case_id}: {exc}")

        if progress_callback:
            progress_callback(idx + 1, len(gradeable))

    return scores
