"""Plotly chart builders for the Analysis tab."""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go

from core.models import BenchmarkResult
from utils.cost import MODEL_COLORS, MODEL_DISPLAY_NAMES, PRICING


def _model_criterion_scores(benchmark: BenchmarkResult) -> Dict[str, Dict[str, float]]:
    criteria = [rc.name for rc in benchmark.eval_suite.rubric]
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))
    result: Dict[str, Dict[str, float]] = {}

    for model in models:
        model_scores = [s for s in benchmark.judge_scores if s.model == model]
        result[model] = {}
        for c in criteria:
            vals = [
                cs.score
                for js in model_scores
                for cs in js.criterion_scores
                if cs.criterion.lower() == c.lower()
            ]
            result[model][c] = round(sum(vals) / len(vals), 2) if vals else 0.0

    return result


def plot_avg_score_by_criterion(benchmark: BenchmarkResult) -> go.Figure:
    criterion_data = _model_criterion_scores(benchmark)
    criteria = [rc.name for rc in benchmark.eval_suite.rubric]
    models = list(criterion_data.keys())

    fig = go.Figure()
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#888")
        scores = [criterion_data[model].get(c, 0.0) for c in criteria]
        fig.add_trace(
            go.Bar(
                name=display,
                x=criteria,
                y=scores,
                marker_color=color,
                text=[f"{s:.2f}" for s in scores],
                textposition="outside",
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title="Average Score by Rubric Criterion",
        xaxis_title="Criterion",
        yaxis_title="Avg Score (1–5)",
        yaxis=dict(range=[0, 5.5]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        margin=dict(t=60, b=40, l=40, r=20),
        height=380,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_quality_vs_cost(benchmark: BenchmarkResult) -> go.Figure:
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))

    avg_scores: Dict[str, float] = {}
    total_costs: Dict[str, float] = {}
    avg_latencies: Dict[str, float] = {}

    for model in models:
        mrs = [r for r in benchmark.model_results if r.model == model]
        jss = [s for s in benchmark.judge_scores if s.model == model]
        avg_scores[model] = sum(s.overall_score for s in jss) / len(jss) if jss else 0.0
        total_costs[model] = sum(r.cost for r in mrs)
        avg_latencies[model] = sum(r.latency_ms for r in mrs) / len(mrs) if mrs else 0.0

    fig = go.Figure()
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#888")
        lat = avg_latencies[model]
        marker_size = max(20, min(60, lat / 80))

        fig.add_trace(
            go.Scatter(
                x=[total_costs[model]],
                y=[avg_scores[model]],
                mode="markers+text",
                name=display,
                text=[display],
                textposition="top center",
                marker=dict(
                    size=marker_size,
                    color=color,
                    opacity=0.85,
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    f"Avg Score: {avg_scores[model]:.2f}<br>"
                    f"Total Cost: ${total_costs[model]:.4f}<br>"
                    f"Avg Latency: {lat:.0f}ms<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Quality vs. Cost (bubble size = avg latency)",
        xaxis_title="Total Cost (USD)",
        yaxis_title="Avg Score (1–5)",
        yaxis=dict(range=[0, 5.5]),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        margin=dict(t=60, b=40, l=50, r=20),
        height=380,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", tickformat="$.4f")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_cost_breakdown(benchmark: BenchmarkResult) -> go.Figure:
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    input_costs = []
    output_costs = []
    for model in models:
        mrs = [r for r in benchmark.model_results if r.model == model]
        p = PRICING.get(model, {"input": 0, "output": 0})
        input_costs.append(sum(r.input_tokens * p["input"] for r in mrs))
        output_costs.append(sum(r.output_tokens * p["output"] for r in mrs))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Input tokens",
            x=display_names,
            y=input_costs,
            marker_color="#6366F1",
            text=[f"${c:.4f}" for c in input_costs],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Output tokens",
            x=display_names,
            y=output_costs,
            marker_color="#A78BFA",
            text=[f"${c:.4f}" for c in output_costs],
            textposition="inside",
        )
    )

    totals = [i + o for i, o in zip(input_costs, output_costs)]
    for name, total in zip(display_names, totals):
        fig.add_annotation(
            x=name,
            y=total,
            text=f"<b>${total:.4f}</b>",
            showarrow=False,
            yshift=10,
            font=dict(color="#F1F5F9", size=12),
        )

    fig.update_layout(
        title="Cost Breakdown by Model",
        xaxis_title="Model",
        yaxis_title="Cost (USD)",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        margin=dict(t=60, b=40, l=60, r=20),
        height=350,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", tickformat="$.4f")
    return fig


def plot_latency_comparison(benchmark: BenchmarkResult) -> go.Figure:
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))

    fig = go.Figure()
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#888")
        latencies = [r.latency_ms for r in benchmark.model_results if r.model == model]
        fig.add_trace(
            go.Box(
                y=latencies,
                name=display,
                marker_color=color,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )

    fig.update_layout(
        title="Latency Distribution by Model",
        yaxis_title="Latency (ms)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
        height=350,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig
