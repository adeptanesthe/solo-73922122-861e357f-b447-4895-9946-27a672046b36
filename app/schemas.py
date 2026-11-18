"""请求/响应模型与输入校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

MAX_TOTAL_TOKENS = 512
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16
MAX_SEED = 2**63 - 1

# /winner-stability 的独立约束
MAX_STABILITY_TOKENS = 64
MAX_QUERY_POSITIONS = 8
MAX_EPSILON = 0.5


def _validate_doc_lengths(value: list[int], max_total: int) -> list[int]:
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError("doc_lengths 必须为正整数列表")
    total = sum(value)
    if total == 0 or total > max_total:
        raise ValueError(f"文档总长度必须在 1..{max_total} 之间，当前为 {total}")
    return value


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(
        ..., ge=0, le=MAX_SEED, description="随机种子（非负整数）"
    )
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        return _validate_doc_lengths(value, MAX_TOTAL_TOKENS)

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest


# ---------------------------------------------------------------------------
# /winner-stability
# ---------------------------------------------------------------------------


class WinnerStabilityRequest(BaseModel):
    """/winner-stability 请求体。

    Q/K 由 ``seed`` 与 ``doc_lengths`` 按与 ``/verify`` 完全相同的方式生成。
    """

    seed: int = Field(
        ..., ge=0, le=MAX_SEED, description="随机种子；与 /verify 使用同一生成序列"
    )
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 64"
    )
    chunk_size: int = Field(
        ..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE, description="分块长度"
    )
    query_positions: list[StrictInt] = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_POSITIONS,
        description="要分析的全局 token 位置，1~8 个且互不相同",
    )
    epsilon: float = Field(
        ...,
        ge=0.0,
        le=MAX_EPSILON,
        description="Q/K 每个分量的独立误差上限 ε，0 ≤ ε ≤ 0.5",
    )

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        return _validate_doc_lengths(value, MAX_STABILITY_TOKENS)

    @field_validator("epsilon")
    @classmethod
    def _check_epsilon(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("epsilon 必须为有限值")
        return value

    @model_validator(mode="after")
    def _check_query_positions(self) -> "WinnerStabilityRequest":
        total = sum(self.doc_lengths)
        positions = self.query_positions
        for p in positions:
            if isinstance(p, bool) or not isinstance(p, int):
                raise ValueError("query_positions 必须为整数位置列表")
            if p < 0 or p >= total:
                raise ValueError(
                    f"query_positions 中的位置 {p} 超出范围 0..{total - 1}"
                )
        if len(set(positions)) != len(positions):
            raise ValueError("query_positions 不得重复")
        return self


class QKChange(BaseModel):
    """一处 Q 或 K 分量的改动。"""

    position: int = Field(..., description="全局 token 位置")
    local_position: int = Field(..., description="该 token 在所属文档内的位置")
    component: int = Field(..., ge=0, lt=HEAD_DIM, description="16 维分量下标")
    old_value: float
    new_value: float
    delta: float


class Witness(BaseModel):
    """使原赢家改变的一组界内扰动。"""

    competitor_position: int = Field(..., description="压过原赢家的 key 的全局位置")
    competitor_local_position: int
    q_changes: list[QKChange] = Field(..., description="对查询 Q 的改动")
    k_changes: list[QKChange] = Field(..., description="对 K 的改动")
    recomputed_winner_position: int
    recomputed_winner_local_position: int
    recomputed_winner_score: float
    changed: bool = Field(..., description="重算赢家是否确实不同于原赢家")
    within_bounds: bool = Field(..., description="每处改动是否都在 ±ε 内")
    verified_exact: bool = Field(..., description="精确有理数复核是否通过")
    verified_float64: bool = Field(..., description="float64 重算复核是否通过")
    note: str


class WinnerStabilityEntry(BaseModel):
    query_position: int
    query_local_position: int
    doc_id: int
    winner_position: int | None = Field(
        ..., description="原赢家的全局位置；并列时取最早"
    )
    winner_local_position: int | None = None
    winner_score: float | None = None
    valid_key_count: int
    tightest_competitor_position: int | None = None
    tightest_competitor_local_position: int | None = None
    tightest_margin: float | None = Field(
        default=None, description="最紧竞争者的优势下界（有理数精确值转 float）"
    )
    tightest_margin_fraction: str | None = Field(
        default=None, description="最紧竞争者优势下界的精确有理数字符串"
    )
    status: Literal["stable", "unstable", "undetermined"]
    advantage_lower_bound: float | None = Field(
        default=None,
        description=(
            "稳定时返回：对每一竞争者 j 精确计算其独立最坏扰动的优势下确界 m_j，"
            "返回 min_j m_j。这是对所有允许扰动一致成立的 QK 点积优势下界"
            "（偏向保守，因 min_j inf 与 inf min 一般不相等）；不稳定或仅自身"
            "可读时为 null。该 float 值向下舍入，保证不超过精确下界"
        ),
    )
    advantage_lower_bound_fraction: str | None = Field(
        default=None, description="优势下界的精确有理数字符串（仅稳定且有竞争者时）"
    )
    reason: str
    witness: Witness | None = None


class WinnerStabilityResponse(BaseModel):
    seed: int
    doc_lengths: list[int]
    chunk_size: int
    epsilon: float
    total_tokens: int
    num_docs: int
    all_stable: bool
    any_undetermined: bool
    undetermined_count: int
    perturbation_model: str
    decision_method: str
    queries: list[WinnerStabilityEntry]
