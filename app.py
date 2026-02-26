"""ThunderEval — Automated Evaluation Framework for Claude."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
import nest_asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

st.set_page_config(
    page_title="ThunderEval",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.generator import generate_eval_suite
from core.judge import judge_all_results
from core.models import BenchmarkResult, EvalSuite
from core.runner import run_benchmark
from ui.charts import (
    plot_avg_score_by_criterion,
    plot_cost_breakdown,
    plot_latency_comparison,
    plot_quality_vs_cost,
)
from ui.components import (
    render_generation_stats,
    render_model_summary_table,
    render_per_case_results,
    render_rubric_table,
    render_test_case_card,
    score_color,
    score_emoji,
)
from utils.cost import BENCHMARK_MODELS, MODEL_DISPLAY_NAMES, format_cost
from utils.export import generate_markdown_report

for key, default in [
    ("eval_suite", None),
    ("benchmark_result", None),
    ("generation_error", None),
    ("benchmark_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

EXAMPLES_DIR = Path(__file__).parent / "examples"


def load_example(name: str) -> Dict:
    with open(EXAMPLES_DIR / f"{name}.json") as f:
        return json.load(f)


def parse_example_pairs(text: str) -> List[Dict[str, str]]:
    pairs = []
    if not text.strip():
        return pairs
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        inp = next((l.replace("Input:", "").strip() for l in lines if l.startswith("Input:")), None)
        out = next((l.replace("Output:", "").strip() for l in lines if l.startswith("Output:")), None)
        if inp:
            pairs.append({"input": inp, "output": out or ""})
    return pairs


def get_api_key() -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        key = st.session_state.get("_api_key_input", "")
    return key or None


def ai_recommendation(benchmark: BenchmarkResult, api_key: Optional[str]) -> str:
    models = list(dict.fromkeys(r.model for r in benchmark.model_results))
    summaries = []
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        jss = [s for s in benchmark.judge_scores if s.model == model]
        mrs = [r for r in benchmark.model_results if r.model == model]
        avg_score = sum(s.overall_score for s in jss) / len(jss) if jss else 0.0
        avg_lat = sum(r.latency_ms for r in mrs) / len(mrs) if mrs else 0.0
        total_cost = sum(r.cost for r in mrs)
        summaries.append(
            f"{display}: avg_score={avg_score:.2f}, avg_latency={avg_lat:.0f}ms, total_cost=${total_cost:.4f}"
        )

    prompt = (
        "Based on this benchmark summary, write a 2–3 sentence recommendation "
        "for which model to use, considering the cost/quality/latency tradeoffs. "
        "Be specific and mention concrete numbers.\n\n"
        + "\n".join(summaries)
    )

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        return f"_(Could not generate recommendation: {exc})_"


# Sidebar
with st.sidebar:
    st.markdown("## ⚡ ThunderEval")
    st.caption("Automated Evaluation Framework for Claude")

    env_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not env_key:
        st.session_state["_api_key_input"] = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Set ANTHROPIC_API_KEY env var to skip this.",
        )
        if not st.session_state.get("_api_key_input"):
            st.warning("⚠️ API key required", icon="🔑")
    else:
        st.success("✅ API key loaded from environment", icon="🔑")

    st.divider()

    example_choice = st.selectbox(
        "Load Example",
        options=["(none)", "contract_analysis", "customer_support", "code_review"],
        format_func=lambda x: {
            "(none)": "— Select an example —",
            "contract_analysis": "📄 Contract Analysis",
            "customer_support": "💬 Customer Support",
            "code_review": "🔍 Code Review",
        }.get(x, x),
        index=0,
    )

    if example_choice != "(none)":
        _ex = load_example(example_choice)
        _sp_val = _ex["system_prompt"]
        _ex_val = "\n\n".join(
            f"Input: {p['input']}\nOutput: {p['output']}"
            for p in _ex.get("examples", [])
        )
    else:
        _sp_val = ""
        _ex_val = ""

    system_prompt_input = st.text_area(
        "System Prompt *",
        value=_sp_val,
        height=220,
        placeholder="Paste your system prompt here…",
        help="The system prompt you want to evaluate.",
    )

    examples_input = st.text_area(
        "Example I/O Pairs (optional)",
        value=_ex_val,
        height=120,
        placeholder="Input: Tell me about X\nOutput: X is …\n\nInput: …\nOutput: …",
        help="Helps the generator create more targeted test cases.",
    )

    st.divider()

    gen_btn = st.button(
        "🔬 Generate Evals",
        use_container_width=True,
        type="primary",
        disabled=not system_prompt_input.strip(),
        help="Generate test cases + rubric using Claude Sonnet 4.5 with extended thinking",
    )

    run_btn = False
    if st.session_state.eval_suite is not None:
        run_btn = st.button(
            "🚀 Run Benchmark",
            use_container_width=True,
            type="secondary",
            help=f"Run all {len(st.session_state.eval_suite.test_cases)} test cases across "
            f"Haiku 4.5, Sonnet 4.5, and Opus 4.5 in parallel",
        )

    # If raw results exist but packaging failed, offer recovery without re-running
    has_raw = (
        st.session_state.get("_raw_model_results")
        and st.session_state.get("_raw_judge_scores")
        and st.session_state.benchmark_result is None
    )
    if has_raw:
        st.warning("⚠️ Previous benchmark data recovered.")
        if st.button("♻️ Recover Results", use_container_width=True):
            try:
                recovered = BenchmarkResult(
                    eval_suite=st.session_state._raw_suite,
                    model_results=st.session_state._raw_model_results,
                    judge_scores=st.session_state._raw_judge_scores,
                )
                st.session_state.benchmark_result = recovered
                st.rerun()
            except Exception as exc:
                st.error(f"Recovery failed: {exc}")

    if st.session_state.eval_suite or st.session_state.benchmark_result:
        if st.button("🗑 Reset", use_container_width=True):
            st.session_state.eval_suite = None
            st.session_state.benchmark_result = None
            st.session_state.generation_error = None
            st.session_state.benchmark_error = None
            for k in ("_raw_model_results", "_raw_judge_scores", "_raw_suite"):
                st.session_state.pop(k, None)
            st.rerun()


# Generate
if gen_btn:
    api_key = get_api_key()
    if not api_key:
        st.error("Please set your Anthropic API key.")
    elif not system_prompt_input.strip():
        st.error("System prompt cannot be empty.")
    else:
        example_pairs = parse_example_pairs(examples_input)
        st.session_state.generation_error = None
        st.session_state.eval_suite = None
        st.session_state.benchmark_result = None

        with st.status("🧠 Generating eval suite with extended thinking…", expanded=True) as gen_status:
            st.write("Analyzing system prompt…")
            t0 = time.time()
            try:
                suite = generate_eval_suite(
                    system_prompt=system_prompt_input,
                    example_pairs=example_pairs if example_pairs else None,
                    api_key=api_key,
                )
                elapsed = time.time() - t0
                st.write(
                    f"✅ Generated **{len(suite.test_cases)} test cases** "
                    f"({sum(1 for tc in suite.test_cases if tc.category=='golden')} golden, "
                    f"{sum(1 for tc in suite.test_cases if tc.category=='edge')} edge, "
                    f"{sum(1 for tc in suite.test_cases if tc.category=='adversarial')} adversarial) "
                    f"and **{len(suite.rubric)} rubric criteria** in {elapsed:.1f}s"
                )
                st.session_state.eval_suite = suite
                gen_status.update(
                    label=f"✅ Eval suite ready — {len(suite.test_cases)} test cases generated in {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
            except anthropic.AuthenticationError:
                err = "Authentication failed — check your API key."
                st.session_state.generation_error = err
                gen_status.update(label=f"❌ {err}", state="error")
            except Exception as exc:
                err = str(exc)
                st.session_state.generation_error = err
                gen_status.update(label=f"❌ Error: {err}", state="error")

        if st.session_state.generation_error:
            st.error(st.session_state.generation_error)
        elif st.session_state.eval_suite is not None:
            st.rerun()


# Run benchmark
if run_btn and st.session_state.eval_suite:
    api_key = get_api_key()
    if not api_key:
        st.error("Please set your Anthropic API key.")
    else:
        suite: EvalSuite = st.session_state.eval_suite
        total_calls = len(BENCHMARK_MODELS) * len(suite.test_cases)
        st.session_state.benchmark_error = None
        st.session_state.benchmark_result = None

        with st.status("🚀 Running benchmark…", expanded=True) as run_status:
            st.write(
                f"⚡ Firing **{total_calls} parallel API calls** across "
                f"{len(BENCHMARK_MODELS)} models × {len(suite.test_cases)} test cases…"
            )
            t1 = time.time()
            try:
                model_results = asyncio.run(run_benchmark(suite, api_key=api_key))
                elapsed1 = time.time() - t1
                successes = sum(1 for r in model_results if not r.error)
                errors = sum(1 for r in model_results if r.error)
                st.write(
                    f"✅ Collected **{successes} responses** in {elapsed1:.1f}s "
                    + (f"({errors} errors)" if errors else "")
                )

                # Persist raw results before packaging so a crash never loses this data
                st.session_state._raw_model_results = [r.model_dump() for r in model_results]
                st.session_state._raw_suite = suite.model_dump()

                total_to_judge = successes
                judged = [0]
                st.write(f"🧑‍⚖️ Scoring **{total_to_judge} responses** with LLM-as-judge (Claude Sonnet 4.5)…")
                judge_placeholder = st.empty()

                def judge_progress(completed, total):
                    judged[0] = completed
                    judge_placeholder.caption(f"  Scored {completed}/{total} responses…")

                t2 = time.time()
                judge_scores = judge_all_results(
                    suite,
                    model_results,
                    api_key=api_key,
                    progress_callback=judge_progress,
                )
                elapsed2 = time.time() - t2
                judge_placeholder.empty()
                st.write(f"✅ Scored **{len(judge_scores)} responses** in {elapsed2:.1f}s")

                st.session_state._raw_judge_scores = [s.model_dump() for s in judge_scores]

                # Use model_dump() dicts to avoid Pydantic class identity issues
                # across Streamlit hot-reloads
                benchmark = BenchmarkResult(
                    eval_suite=st.session_state._raw_suite,
                    model_results=st.session_state._raw_model_results,
                    judge_scores=st.session_state._raw_judge_scores,
                )
                st.session_state.benchmark_result = benchmark

                total_cost = sum(r.cost for r in model_results)
                run_status.update(
                    label=(
                        f"✅ Benchmark complete — "
                        f"{successes} responses, {len(judge_scores)} scores, "
                        f"total cost {format_cost(total_cost)}"
                    ),
                    state="complete",
                    expanded=False,
                )

            except anthropic.AuthenticationError:
                err = "Authentication failed — check your API key."
                st.session_state.benchmark_error = err
                run_status.update(label=f"❌ {err}", state="error")
            except Exception as exc:
                err = str(exc)
                st.session_state.benchmark_error = err
                run_status.update(label=f"❌ Error: {err}", state="error")

        if st.session_state.benchmark_error:
            st.error(st.session_state.benchmark_error)
        elif st.session_state.benchmark_result is not None:
            st.rerun()


# Main content
st.markdown(
    """
    <h1 style="margin-bottom:0">⚡ ThunderEval</h1>
    <p style="color:#94A3B8;margin-top:4px;font-size:1.05rem">
    Automated Evaluation Framework for Claude &mdash;
    generate test cases, benchmark Haiku / Sonnet / Opus, score with LLM-as-judge
    </p>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔬 Eval Suite", "📊 Results", "📈 Analysis", "📤 Export"]
)

with tab1:
    if st.session_state.eval_suite is None:
        st.markdown(
            """
### How ThunderEval Works

1. **Paste** a system prompt in the sidebar (or load one of 3 examples)
2. **Generate Evals** — Claude Sonnet 4.5 uses extended thinking to create golden, edge, and adversarial test cases plus a grading rubric
3. **Run Benchmark** — All test cases fire in parallel against Haiku 4.5, Sonnet 4.5, and Opus 4.5 (with prompt caching)
4. **Review Results** — LLM-as-judge scores every response blind; compare cost, quality, and latency
            """
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**📄 Contract Analysis**\nLegal AI that identifies risks, obligations, and key terms in contracts.")
        with col2:
            st.info("**💬 Customer Support**\nSaaS support agent answering billing, features, and troubleshooting.")
        with col3:
            st.info("**🔍 Code Review**\nSenior engineer reviewing PRs for bugs, security issues, and style.")
        st.caption("← Load an example from the sidebar to get started in seconds.")
    else:
        suite = st.session_state.eval_suite

        render_generation_stats(suite)
        st.divider()

        for category, label, emoji in [
            ("golden", "Golden Cases", "🟢"),
            ("edge", "Edge Cases", "🟡"),
            ("adversarial", "Adversarial Cases", "🔴"),
        ]:
            cases = [tc for tc in suite.test_cases if tc.category == category]
            if not cases:
                continue
            st.subheader(f"{emoji} {label} ({len(cases)})")
            st.caption(
                {
                    "golden": "Standard inputs that should produce ideal responses — includes expected outputs as reference.",
                    "edge": "Boundary conditions, ambiguous inputs, and unusual requests.",
                    "adversarial": "Prompt injection attempts, off-topic requests, and constraint violations.",
                }[category]
            )
            for tc in cases:
                render_test_case_card(tc)
            st.divider()

        st.subheader("📋 Grading Rubric")
        render_rubric_table(suite)

        if st.session_state.benchmark_result is None:
            st.info("👈 Click **Run Benchmark** in the sidebar when you're ready to score all models.")


with tab2:
    if st.session_state.benchmark_result is None:
        if st.session_state.eval_suite:
            st.info("👈 Click **Run Benchmark** in the sidebar to compare models.")
        else:
            st.info("Generate an eval suite first, then run the benchmark.")
    else:
        benchmark: BenchmarkResult = st.session_state.benchmark_result
        models_run = list(dict.fromkeys(r.model for r in benchmark.model_results))

        st.subheader("Model Comparison")

        summary_df = render_model_summary_table(benchmark)
        best_score = summary_df["_score_raw"].max()

        display_df = summary_df.drop(columns=["_score_raw"])
        styled = display_df.style.apply(
            lambda row: [
                "background-color: rgba(99,102,241,0.2)" if summary_df.loc[row.name, "_score_raw"] == best_score else ""
                for _ in row
            ],
            axis=1,
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn(width="small"),
                "Avg Score": st.column_config.NumberColumn(
                    format="%.2f",
                    width="small",
                ),
            },
        )

        st.subheader("Scores by Criterion")
        criteria = [rc.name for rc in benchmark.eval_suite.rubric]
        rows = []
        for model in models_run:
            display = MODEL_DISPLAY_NAMES.get(model, model)
            row: Dict = {"Model": display}
            for c in criteria:
                vals = [
                    cs.score
                    for js in benchmark.judge_scores
                    if js.model == model
                    for cs in js.criterion_scores
                    if cs.criterion.lower() == c.lower()
                ]
                row[c] = round(sum(vals) / len(vals), 2) if vals else 0.0
            rows.append(row)

        crit_df = pd.DataFrame(rows)

        def color_score_cell(val):
            if isinstance(val, float):
                c = score_color(val)
                return f"background-color: {c}20; color: {c}"
            return ""

        styled_crit = crit_df.style.map(color_score_cell, subset=criteria)
        st.dataframe(styled_crit, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Per-Test-Case Results")
        render_per_case_results(benchmark)


with tab3:
    if st.session_state.benchmark_result is None:
        st.info("Run the benchmark to see analysis.")
    else:
        benchmark = st.session_state.benchmark_result
        models_run = list(dict.fromkeys(r.model for r in benchmark.model_results))

        cols = st.columns(len(models_run))
        for col, model in zip(cols, models_run):
            display = MODEL_DISPLAY_NAMES.get(model, model)
            jss = [s for s in benchmark.judge_scores if s.model == model]
            avg = sum(s.overall_score for s in jss) / len(jss) if jss else 0.0
            total_cost = sum(r.cost for r in benchmark.model_results if r.model == model)
            col.metric(
                label=display,
                value=f"{avg:.2f} / 5.0",
                delta=format_cost(total_cost),
                delta_color="off",
                help="Avg LLM-judge score / Total benchmark cost",
            )

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_avg_score_by_criterion(benchmark), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_quality_vs_cost(benchmark), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(plot_cost_breakdown(benchmark), use_container_width=True)
        with col_d:
            st.plotly_chart(plot_latency_comparison(benchmark), use_container_width=True)

        st.divider()

        st.subheader("💰 Cost Breakdown")
        cost_rows = []
        for model in models_run:
            mrs = [r for r in benchmark.model_results if r.model == model]
            cost_rows.append(
                {
                    "Model": MODEL_DISPLAY_NAMES.get(model, model),
                    "Input Tokens": sum(r.input_tokens for r in mrs),
                    "Output Tokens": sum(r.output_tokens for r in mrs),
                    "Total Tokens": sum(r.input_tokens + r.output_tokens for r in mrs),
                    "Total Cost": format_cost(sum(r.cost for r in mrs)),
                    "Cost / Test Case": format_cost(
                        sum(r.cost for r in mrs) / len(mrs) if mrs else 0
                    ),
                }
            )
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("🤖 AI Recommendation")
        api_key = get_api_key()
        if "recommendation" not in st.session_state or st.button("Regenerate", key="regen_rec"):
            with st.spinner("Generating recommendation…"):
                st.session_state["recommendation"] = ai_recommendation(benchmark, api_key)

        rec = st.session_state.get("recommendation", "")
        if rec:
            st.info(rec, icon="💡")


with tab4:
    if st.session_state.benchmark_result is None:
        if st.session_state.eval_suite:
            suite = st.session_state.eval_suite

            st.subheader("📋 Eval Suite Export")
            st.caption("Run the benchmark to export full results.")

            suite_json = json.dumps(
                {
                    "system_prompt": suite.system_prompt,
                    "rubric": [
                        {"name": rc.name, "description": rc.description, "weight": rc.weight}
                        for rc in suite.rubric
                    ],
                    "test_cases": [
                        {
                            "id": tc.id,
                            "category": tc.category,
                            "description": tc.description,
                            "input": tc.input,
                            "expected_output": tc.expected_output,
                        }
                        for tc in suite.test_cases
                    ],
                },
                indent=2,
            )
            st.download_button(
                "⬇️ Download Eval Suite (JSON)",
                data=suite_json,
                file_name="thundereval_suite.json",
                mime="application/json",
                use_container_width=True,
            )
            st.code(suite_json, language="json")
        else:
            st.info("Generate an eval suite to export.")
    else:
        benchmark = st.session_state.benchmark_result

        st.subheader("📤 Export Benchmark Report")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Full Markdown Report**")
            report_md = generate_markdown_report(benchmark)
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=report_md,
                file_name="thundereval_report.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary",
            )

        with col2:
            st.markdown("**Raw Results (JSON)**")
            results_json = benchmark.model_dump_json(indent=2)
            st.download_button(
                "⬇️ Download Results (JSON)",
                data=results_json,
                file_name="thundereval_results.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        st.subheader("📋 Copy System Prompt + Rubric")
        combined = f"## System Prompt\n\n{benchmark.eval_suite.system_prompt}\n\n"
        combined += "## Grading Rubric\n\n"
        for rc in benchmark.eval_suite.rubric:
            combined += f"- **{rc.name}** (weight {rc.weight:.1f}): {rc.description}\n"
        st.code(combined, language="markdown")

        st.divider()

        st.subheader("Report Preview")
        st.markdown(report_md)
