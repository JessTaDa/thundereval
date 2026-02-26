"""Reusable Streamlit UI components for ThunderEval."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from core.models import (
    BenchmarkResult,
    EvalSuite,
    JudgeScore,
    ModelResult,
    TestCase,
    TestCaseCategory,
)
from utils.cost import MODEL_DISPLAY_NAMES, format_cost, format_latency


def score_color(score: float) -> str:
    """Return a CSS color string for a 1–5 score."""
    if score >= 4:
        return "#34D399"  # green
    if score >= 3:
        return "#FBBF24"  # yellow
    return "#F87171"  # red


def score_emoji(score: float) -> str:
    if score >= 4:
        return "🟢"
    if score >= 3:
        return "🟡"
    return "🔴"


def render_test_case_card(tc: TestCase, key_prefix: str = "") -> None:
    """Render a single test case as an expandable card."""
    cat_emoji = {
        TestCaseCategory.GOLDEN: "🟢",
        TestCaseCategory.EDGE: "🟡",
        TestCaseCategory.ADVERSARIAL: "🔴",
    }.get(tc.category, "")

    with st.expander(f"{cat_emoji} **{tc.id}** — {tc.description}", expanded=False):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("**Input:**")
            st.code(tc.input, language=None)
            if tc.expected_output:
                st.markdown("**Expected Output:**")
                st.markdown(tc.expected_output)
        with col_b:
            st.caption(f"Category: `{tc.category}`")
            st.caption(f"ID: `{tc.id}`")


def render_rubric_table(suite: EvalSuite) -> None:
    """Render the grading rubric as a clean DataFrame."""
    data = [
        {
            "Criterion": rc.name,
            "Weight": f"{rc.weight:.1f}",
            "Description": rc.description,
        }
        for rc in suite.rubric
    ]
    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Criterion": st.column_config.TextColumn(width="small"),
            "Weight": st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
        },
    )


def render_model_summary_table(benchmark: BenchmarkResult) -> pd.DataFrame:
    """Build summary DataFrame with per-model aggregate stats."""
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))
    rows = []
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        mrs = [r for r in benchmark.model_results if r.model == model]
        jss = [s for s in benchmark.judge_scores if s.model == model]

        avg_score = sum(s.overall_score for s in jss) / len(jss) if jss else 0.0
        avg_latency = sum(r.latency_ms for r in mrs) / len(mrs) if mrs else 0.0
        total_cost = sum(r.cost for r in mrs)
        cost_per_case = total_cost / len(mrs) if mrs else 0.0

        rows.append(
            {
                "Model": display,
                "Avg Score": round(avg_score, 2),
                "Avg Latency": format_latency(avg_latency),
                "Total Cost": format_cost(total_cost),
                "Cost/Case": format_cost(cost_per_case),
                "Test Cases": len(mrs),
                "_score_raw": avg_score,
            }
        )

    return pd.DataFrame(rows)


def render_per_case_results(benchmark: BenchmarkResult) -> None:
    """Render expandable rows for each test case showing all model responses."""
    suite = benchmark.eval_suite
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))

    for tc in suite.test_cases:
        cat_emoji = {"golden": "🟢", "edge": "🟡", "adversarial": "🔴"}.get(
            tc.category, ""
        )

        # Get scores for this test case across models
        tc_scores = {
            s.model: s.overall_score
            for s in benchmark.judge_scores
            if s.test_case_id == tc.id
        }
        score_summary = " | ".join(
            f"{MODEL_DISPLAY_NAMES.get(m, m)}: {score_emoji(tc_scores.get(m, 0))} {tc_scores.get(m, 0):.1f}"
            for m in models
        )

        with st.expander(
            f"{cat_emoji} **{tc.id}** — {tc.description}  \n{score_summary}",
            expanded=False,
        ):
            st.markdown(f"**Input:**\n```\n{tc.input}\n```")
            if tc.expected_output:
                st.markdown(f"**Expected:**\n{tc.expected_output}")

            st.divider()
            cols = st.columns(len(models))
            for col, model in zip(cols, models):
                display = MODEL_DISPLAY_NAMES.get(model, model)
                mr = next(
                    (
                        r
                        for r in benchmark.model_results
                        if r.model == model and r.test_case_id == tc.id
                    ),
                    None,
                )
                js = next(
                    (
                        s
                        for s in benchmark.judge_scores
                        if s.model == model and s.test_case_id == tc.id
                    ),
                    None,
                )

                with col:
                    st.markdown(f"**{display}**")
                    if mr and mr.error:
                        st.error(f"Error: {mr.error}")
                        continue

                    if mr:
                        st.caption(
                            f"{format_latency(mr.latency_ms)} · "
                            f"{mr.input_tokens + mr.output_tokens} tok · "
                            f"{format_cost(mr.cost)}"
                        )
                        # Score badge
                        if js:
                            overall = js.overall_score
                            color = score_color(overall)
                            st.markdown(
                                f"<span style='background:{color};color:#000;padding:2px 8px;"
                                f"border-radius:4px;font-weight:bold'>"
                                f"Score: {overall:.1f}/5</span>",
                                unsafe_allow_html=True,
                            )

                        # Response text (truncated for readability)
                        response_preview = mr.response[:500] + (
                            "..." if len(mr.response) > 500 else ""
                        )
                        st.markdown(response_preview)

                    if js:
                        with st.expander("Criterion scores"):
                            for cs in js.criterion_scores:
                                e = score_emoji(cs.score)
                                st.markdown(
                                    f"**{cs.criterion}**: {e} {cs.score}/5 — {cs.justification}"
                                )


def render_generation_stats(suite: EvalSuite) -> None:
    """Show generation stats as metrics row."""
    golden = sum(1 for tc in suite.test_cases if tc.category == "golden")
    edge = sum(1 for tc in suite.test_cases if tc.category == "edge")
    adv = sum(1 for tc in suite.test_cases if tc.category == "adversarial")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Test Cases", len(suite.test_cases))
    col2.metric("🟢 Golden", golden)
    col3.metric("🟡 Edge", edge)
    col4.metric("🔴 Adversarial", adv)

    if suite.generation_time_s:
        st.caption(
            f"Generated {len(suite.test_cases)} test cases in {suite.generation_time_s:.1f}s "
            f"using Sonnet 4.5 with extended thinking · "
            f"{len(suite.rubric)} rubric criteria"
        )
