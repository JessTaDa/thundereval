"""Markdown report export for benchmark results."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from core.models import BenchmarkResult, JudgeScore, ModelResult
from utils.cost import MODEL_DISPLAY_NAMES, format_cost, format_latency


def _model_summary(model: str, results: List[ModelResult], scores: List[JudgeScore]) -> Dict:
    model_results = [r for r in results if r.model == model]
    model_scores = [s for s in scores if s.model == model]

    avg_score = sum(s.overall_score for s in model_scores) / len(model_scores) if model_scores else 0.0
    avg_latency = sum(r.latency_ms for r in model_results) / len(model_results) if model_results else 0.0
    total_cost = sum(r.cost for r in model_results)

    return {
        "model": model,
        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
        "avg_score": round(avg_score, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "total_cost": total_cost,
        "total_input_tokens": sum(r.input_tokens for r in model_results),
        "total_output_tokens": sum(r.output_tokens for r in model_results),
        "num_cases": len(model_results),
    }


def generate_markdown_report(result: BenchmarkResult) -> str:
    suite = result.eval_suite
    models_used = list(dict.fromkeys(r.model for r in result.model_results))
    summaries = {m: _model_summary(m, result.model_results, result.judge_scores) for m in models_used}
    best_model = max(summaries.values(), key=lambda s: s["avg_score"], default=None)

    lines: List[str] = []

    lines += [
        "# ThunderEval Benchmark Report",
        "",
        f"**Generated:** {datetime.fromtimestamp(result.completed_at).strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Models:** {', '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in models_used)}",
        f"**Test Cases:** {len(suite.test_cases)}",
        f"**Total API Calls:** {len(result.model_results)}",
        "",
        "---",
        "",
    ]

    lines += [
        "## System Prompt",
        "",
        "```",
        suite.system_prompt,
        "```",
        "",
        "---",
        "",
    ]

    lines += [
        "## Grading Rubric",
        "",
        "| Criterion | Weight | Description |",
        "|-----------|--------|-------------|",
    ]
    for rc in suite.rubric:
        lines.append(f"| {rc.name} | {rc.weight:.1f} | {rc.description} |")
    lines += ["", "---", ""]

    lines += [
        "## Model Comparison",
        "",
        "| Model | Avg Score | Avg Latency | Total Cost | Input Tokens | Output Tokens |",
        "|-------|-----------|-------------|------------|--------------|---------------|",
    ]
    for s in summaries.values():
        lines.append(
            f"| **{s['display_name']}** "
            f"| {s['avg_score']:.2f} / 5.0 "
            f"| {format_latency(s['avg_latency_ms'])} "
            f"| {format_cost(s['total_cost'])} "
            f"| {s['total_input_tokens']:,} "
            f"| {s['total_output_tokens']:,} |"
        )
    lines += ["", "---", ""]

    criteria = [rc.name for rc in suite.rubric]
    lines += [
        "## Scores by Criterion",
        "",
        "| Model | " + " | ".join(criteria) + " |",
        "|-------|" + "|".join(["---"] * len(criteria)) + "|",
    ]
    for model in models_used:
        model_scores = [s for s in result.judge_scores if s.model == model]
        criterion_avgs = []
        for c in criteria:
            vals = [
                cs.score
                for js in model_scores
                for cs in js.criterion_scores
                if cs.criterion.lower() == c.lower()
            ]
            criterion_avgs.append(f"{sum(vals) / len(vals):.2f}" if vals else "0.00")
        lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model, model)} | " + " | ".join(criterion_avgs) + " |"
        )
    lines += ["", "---", ""]

    lines += ["## Detailed Results", ""]
    for tc in suite.test_cases:
        category_emoji = {"golden": "🟢", "edge": "🟡", "adversarial": "🔴"}.get(tc.category, "")
        lines += [
            f"### {category_emoji} {tc.id}: {tc.description}",
            "",
            f"**Category:** `{tc.category}`",
            "",
            "**Input:**",
            "```",
            tc.input,
            "```",
            "",
        ]

        if tc.expected_output:
            lines += ["**Expected Output:**", tc.expected_output, ""]

        for model in models_used:
            display = MODEL_DISPLAY_NAMES.get(model, model)
            mr = next(
                (r for r in result.model_results if r.model == model and r.test_case_id == tc.id),
                None,
            )
            js = next(
                (s for s in result.judge_scores if s.model == model and s.test_case_id == tc.id),
                None,
            )
            lines += [f"#### {display}"]

            if mr and mr.error:
                lines += [f"> ⚠️ Error: {mr.error}", ""]
                continue

            if mr:
                lines += [
                    f"**Response:** ({format_latency(mr.latency_ms)}, "
                    f"{mr.input_tokens}+{mr.output_tokens} tokens, "
                    f"{format_cost(mr.cost)})",
                    "",
                    mr.response if mr.response else "_No response_",
                    "",
                ]

            if js:
                lines += [
                    f"**Overall Score: {js.overall_score:.2f} / 5.0**",
                    "",
                    "| Criterion | Score | Justification |",
                    "|-----------|-------|---------------|",
                ]
                for cs in js.criterion_scores:
                    emoji = "🟢" if cs.score >= 4 else ("🟡" if cs.score == 3 else "🔴")
                    lines.append(f"| {cs.criterion} | {emoji} {cs.score}/5 | {cs.justification} |")
                lines.append("")

        lines += ["---", ""]

    if best_model:
        lines += [
            "## Recommendation",
            "",
            f"Based on this benchmark, **{best_model['display_name']}** achieved the "
            f"highest average score of **{best_model['avg_score']:.2f}/5.0**.",
            "",
        ]

        sorted_by_score = sorted(summaries.values(), key=lambda s: s["avg_score"], reverse=True)
        if len(sorted_by_score) >= 2:
            best = sorted_by_score[0]
            runner_up = sorted_by_score[1]
            score_diff = best["avg_score"] - runner_up["avg_score"]
            cost_ratio = (
                best["total_cost"] / runner_up["total_cost"]
                if runner_up["total_cost"] > 0
                else float("inf")
            )
            lines += [
                f"The score difference between {best['display_name']} and "
                f"{runner_up['display_name']} is {score_diff:.2f} points, while "
                f"{best['display_name']} costs {cost_ratio:.1f}× more per run.",
                "",
                "_Consider your latency and cost constraints when choosing a model for production._",
                "",
            ]

    lines += ["---", "", "_Generated by ThunderEval_"]

    return "\n".join(lines)
