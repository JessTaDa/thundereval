"""Pydantic data models for ThunderEval."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TestCaseCategory(str, Enum):
    GOLDEN = "golden"
    EDGE = "edge"
    ADVERSARIAL = "adversarial"


_cfg = ConfigDict(revalidate_instances="never")


class TestCase(BaseModel):
    model_config = _cfg

    id: str = Field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:6]}")
    category: TestCaseCategory
    input: str
    expected_output: Optional[str] = None
    description: str


class RubricCriterion(BaseModel):
    model_config = _cfg

    name: str
    description: str
    weight: float = 1.0


class EvalSuite(BaseModel):
    model_config = _cfg

    system_prompt: str
    example_pairs: List[Dict[str, str]] = Field(default_factory=list)
    test_cases: List[TestCase]
    rubric: List[RubricCriterion]
    generated_at: float = Field(default_factory=time.time)
    generation_time_s: Optional[float] = None


class ModelResult(BaseModel):
    model_config = _cfg

    model: str
    test_case_id: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    error: Optional[str] = None


class CriterionScore(BaseModel):
    model_config = _cfg

    criterion: str
    score: int  # 1–5
    justification: str


class JudgeScore(BaseModel):
    model_config = _cfg

    model: str
    test_case_id: str
    criterion_scores: List[CriterionScore]
    overall_score: float  # weighted average


class BenchmarkResult(BaseModel):
    model_config = _cfg

    eval_suite: EvalSuite
    model_results: List[ModelResult]
    judge_scores: List[JudgeScore]
    completed_at: float = Field(default_factory=time.time)
