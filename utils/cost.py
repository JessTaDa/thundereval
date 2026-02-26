"""Token counting and cost calculation for benchmark models."""

from __future__ import annotations

from typing import Dict

# Pricing per token (input / output) — as specified for these 4.5 model versions
PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 0.80 / 1_000_000,   # $0.80 / MTok
        "output": 4.00 / 1_000_000,  # $4.00 / MTok
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00 / 1_000_000,   # $3.00 / MTok
        "output": 15.00 / 1_000_000, # $15.00 / MTok
    },
    "claude-opus-4-5-20251101": {
        "input": 5.00 / 1_000_000,   # $5.00 / MTok
        "output": 25.00 / 1_000_000, # $25.00 / MTok
    },
}

# Human-readable display names
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "claude-opus-4-5-20251101": "Opus 4.5",
}

# Ordered list of benchmark models (fast → powerful)
BENCHMARK_MODELS: list[str] = list(PRICING.keys())

# Model colors for charts
MODEL_COLORS: Dict[str, str] = {
    "claude-haiku-4-5-20251001": "#34D399",   # green
    "claude-sonnet-4-5-20250929": "#60A5FA",  # blue
    "claude-opus-4-5-20251101": "#A78BFA",    # purple
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for a model call."""
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return input_tokens * p["input"] + output_tokens * p["output"]


def format_cost(usd: float) -> str:
    """Format cost for display."""
    if usd < 0.0001:
        return f"${usd * 1000:.4f}m"  # milli-dollars
    return f"${usd:.4f}"


def format_latency(ms: float) -> str:
    """Format latency for display."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"
