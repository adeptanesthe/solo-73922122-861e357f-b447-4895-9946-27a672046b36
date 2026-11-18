"""赢家稳定性分析：Q/K 分量独立扰动下，每个查询的原赢家是否保持不变。

问题定义
--------
对查询 token ``i``，有效 key 为同一篇文档中位置不晚于 ``i`` 的 token（与
``/verify`` 的掩码一致）。记原始 QK 点积分数 ``s_j = Q_i · K_j``，原赢家
``w`` 为分数最大者；并列时取位置最早者。

Q 的每个分量与每个 K 行的每个分量都可以**独立**取任意偏移
``δ ∈ [-ε, +ε]``。逐查询判断：在所有允许扰动下，按“最早并列”规则选出的
赢家是否仍然都是 ``w``。

三种结论
--------
* ``stable``：所有允许扰动下赢家不变；返回优势下界
  ``min_j inf_δ (s_w~(δ) - s_j~(δ))``。若只剩自身可读，没有竞争者，
  该下界为 ``null``。
* ``unstable``：存在界内扰动改变赢家。返回一组具体的 Q/K 分量改动
  （位置、旧值、新值）与重算赢家；见证同时通过精确有理数和 float64
  两种复核，保证确实改变赢家。
* ``undetermined``：精确判定已成立，但把见证扰动写成 float64 后无法在
  float64 下可靠复现赢家改变（数值边界）；明确说明未确定，不返回
  不可靠见证。

精确性
------
所有判定与下界推导都对本就由 float64 生成的 Q/K 做 :class:`fractions.Fraction`
精确有理数运算，避免浮点舍入造成误判。仅“见证能否用 float64 复现”一项
依赖浮点复核。

优势下界推导
------------
固定 Q 的扰动取值 ``q~``，对竞争者 ``j`` 令 ``d_t = K_{w,t} - K_{j,t}``。
由于 ``K_w`` 与 ``K_j`` 可独立扰动，K 侧可把差距压到

    s_w~ - s_j~ = q~·d - 2ε·||q~||_1.

再对每个分量独立最小化 ``g_t(x) = d_t·x - 2ε·|x|``，其中
``x ∈ [q_t-ε, q_t+ε]``。``g_t`` 是凸的分段线性函数，最小值必在区间端点
或 ``x=0``（当 0 落在区间内）取得。记该精确下确界为 ``m_j``。

注意 ``min_j inf_δ`` 与 ``inf_δ min_j`` 一般**不可交换**：前者可能更保守
（不同竞争者的最坏扰动方向可能相反，无法由同一组扰动同时达到）。因此

* 稳定性判定对每个竞争者分别检查 ``m_j``，是**精确且紧**的：原赢家稳定
  当且仅当每个竞争者的 ``m_j`` 满足上述方向条件。
* 返回的 ``advantage_lower_bound = min_j m_j`` 是对**所有**允许扰动一致
  成立的合法下界（因为对每个 ``j`` 都有 ``min_j m_j ≤ m_j ≤ 任意扰动的
  实际优势``），但未必是 ``inf_δ min_j`` 的精确值；它偏向保守。

判定规则
--------
* 更早的竞争者 ``j < w``：必须严格落后，``bound_j > 0`` 才稳定；
  若下界为 0，扰动可使其并列而由更早位置获胜 → 不稳定。
* 更晚的竞争者 ``j > w``：``bound_j ≥ 0`` 即稳定（并列时原赢家仍最早）。

按块计算
--------
分数按 ``chunk_size`` 的 query/key 块遍历，只为被查询 token 收集有效
位置对；不构造任何 N×N 矩阵，峰值额外内存为 O(查询数 × N)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .attention import HEAD_DIM
from .schemas import (
    QKChange,
    WinnerStabilityEntry,
    WinnerStabilityRequest,
    WinnerStabilityResponse,
    Witness,
)
from .verification import generate_qk

ZERO = Fraction(0)


@dataclass(frozen=True)
class _DocLayout:
    doc_ids: list[int]
    local_positions: list[int]
    doc_starts: list[int]


@dataclass
class _QueryAnalysis:
    query_position: int
    winner_position: int
    winner_score: Fraction
    scores: dict[int, Fraction]
    margins: dict[int, Fraction]
    violations: list[int]


def _split(num: int, chunk: int) -> list[tuple[int, int]]:
    return [(b, min(b + chunk, num)) for b in range(0, num, chunk)]


def _layout(doc_lengths: list[int]) -> _DocLayout:
    doc_ids: list[int] = []
    local_positions: list[int] = []
    doc_starts: list[int] = []
    cursor = 0
    for doc_id, length in enumerate(doc_lengths):
        doc_starts.append(cursor)
        doc_ids.extend([doc_id] * length)
        local_positions.extend(range(length))
        cursor += length
    return _DocLayout(doc_ids, local_positions, doc_starts)


def _to_fraction_rows(tensor) -> list[list[Fraction]]:
    """float64 张量逐元素精确转有理数（Fraction.from_float 无损）。"""
    return [[Fraction.from_float(v) for v in row] for row in tensor.tolist()]


def _dot_exact(a: list[Fraction], b: list[Fraction]) -> Fraction:
    total = ZERO
    for t in range(HEAD_DIM):
        total += a[t] * b[t]
    return total


def _collect_block_scores(
    queries: list[int],
    q_rows: list[list[Fraction]],
    k_rows: list[list[Fraction]],
    layout: _DocLayout,
    chunk_size: int,
) -> dict[int, dict[int, Fraction]]:
    """按 query/key 块收集每个查询对其有效 key 的精确点积分数。"""
    n = len(layout.doc_ids)
    result: dict[int, dict[int, Fraction]] = {qi: {} for qi in queries}
    for q0, q1 in _split(n, chunk_size):
        active = [qi for qi in queries if q0 <= qi < q1]
        if not active:
            continue
        for k0, k1 in _split(n, chunk_size):
            for qi in active:
                doc = layout.doc_ids[qi]
                start = layout.doc_starts[doc]
                lo = max(k0, start)
                hi = min(k1, qi + 1)
                for j in range(lo, hi):
                    if layout.doc_ids[j] == doc:
                        result[qi][j] = _dot_exact(q_rows[qi], k_rows[j])
    return result


def _earliest_max_exact(scores: dict[int, Fraction]) -> int:
    best_key = -1
    best_score = None
    for key in sorted(scores):
        if best_score is None or scores[key] > best_score:
            best_key, best_score = key, scores[key]
    return best_key


def _earliest_max_float(scores: dict[int, float]) -> int:
    best_key = -1
    best_score = None
    for key in sorted(scores):
        if best_score is None or scores[key] > best_score:
            best_key, best_score = key, scores[key]
    return best_key


def _component_candidates(qt: Fraction, eps: Fraction) -> list[Fraction]:
    candidates = [qt - eps, qt + eps]
    if qt - eps <= ZERO <= qt + eps:
        candidates.append(ZERO)
    return sorted(set(candidates))


def _min_component_gap(delta: Fraction, qt: Fraction, eps: Fraction) -> Fraction:
    """``min_{x∈[qt-ε,qt+ε]} delta*x - 2ε|x|`` 的精确值。"""
    return min(
        delta * x - 2 * eps * abs(x) for x in _component_candidates(qt, eps)
    )


def _min_advantage(
    q_row: list[Fraction],
    k_winner: list[Fraction],
    k_other: list[Fraction],
    eps: Fraction,
) -> Fraction:
    total = ZERO
    for t in range(HEAD_DIM):
        delta = k_winner[t] - k_other[t]
        total += _min_component_gap(delta, q_row[t], eps)
    return total


def _choose_q_values(
    q_row: list[Fraction],
    k_winner: list[Fraction],
    k_other: list[Fraction],
    eps: Fraction,
) -> list[Fraction]:
    """逐分量选达到下确界的 Q 扰动后取值（确定性 tie-break）。"""
    chosen: list[Fraction] = []
    for t in range(HEAD_DIM):
        delta = k_winner[t] - k_other[t]
        qt = q_row[t]

        def score(x: Fraction) -> tuple[Fraction, Fraction, Fraction]:
            # 目标最小，其次改动最小，最后数值最小，保证确定性。
            return (delta * x - 2 * eps * abs(x), abs(x - qt), x)

        chosen.append(min(_component_candidates(qt, eps), key=score))
    return chosen


def _analyze_query(
    qi: int,
    scores: dict[int, Fraction],
    q_rows: list[list[Fraction]],
    k_rows: list[list[Fraction]],
    eps: Fraction,
) -> _QueryAnalysis:
    winner = _earliest_max_exact(scores)
    margins: dict[int, Fraction] = {}
    violations: list[int] = []
    for j in sorted(scores):
        if j == winner:
            continue
        margin = _min_advantage(q_rows[qi], k_rows[winner], k_rows[j], eps)
        margins[j] = margin
        if (j < winner and margin <= ZERO) or (j > winner and margin < ZERO):
            violations.append(j)
    return _QueryAnalysis(
        query_position=qi,
        winner_position=winner,
        winner_score=scores[winner],
        scores=scores,
        margins=margins,
        violations=violations,
    )


def _round_down(value: Fraction) -> float:
    """把有理数转为不超过其精确值的 float64（真正的下界）。"""
    f = float(value)
    while Fraction.from_float(f) > value:
        f = math.nextafter(f, -math.inf)
    return f


def _fraction_str(value: Fraction) -> str:
    """精确有理数的可读字符串（小数近似 + 精确分数）。"""
    return f"{float(value):.17g} (~{value.numerator}/{value.denominator})"


def _float_exact(value: Fraction) -> float:
    """精确有理数的 float64 最近值；过大/过小时安全退化。"""
    try:
        return float(value)
    except OverflowError:
        return math.copysign(math.inf, value.numerator)


def _clamp_to_float(value: Fraction, lo: Fraction, hi: Fraction) -> float | None:
    """有理数转 float64 并保证落在 [lo, hi]；区间过窄时返回 None。"""
    f = float(value)
    if Fraction.from_float(f) < lo:
        f = math.nextafter(f, math.inf)
    elif Fraction.from_float(f) > hi:
        f = math.nextafter(f, -math.inf)
    if not (lo <= Fraction.from_float(f) <= hi):
        return None
    return f


def _float_score(q_row: list[float], k_row: list[float]) -> float:
    total = 0.0
    for t in range(HEAD_DIM):
        total += q_row[t] * k_row[t]
    return total


def _build_witness(
    analysis: _QueryAnalysis,
    competitor: int,
    q_row: list[Fraction],
    k_rows: list[list[Fraction]],
    layout: _DocLayout,
    eps: Fraction,
) -> Witness | None:
    """构造并复核见证；无法用 float64 复现时返回 None。"""
    w = analysis.winner_position
    chosen = _choose_q_values(q_row, k_rows[w], k_rows[competitor], eps)

    # --- Q 侧改动 ---
    q_new: list[float] = []
    q_changes: list[QKChange] = []
    for t in range(HEAD_DIM):
        old = q_row[t]
        new = _clamp_to_float(chosen[t], old - eps, old + eps)
        if new is None:
            return None
        q_new.append(new)
        if Fraction.from_float(new) != old:
            q_changes.append(
                QKChange(
                    position=analysis.query_position,
                    local_position=layout.local_positions[analysis.query_position],
                    component=t,
                    old_value=float(old),
                    new_value=new,
                    delta=new - float(old),
                )
            )

    # --- K 侧改动：q~_t > 0 时压低 K_w、抬高 K_j；反之亦然 ---
    k_new: dict[int, list[float]] = {
        w: [float(v) for v in k_rows[w]],
        competitor: [float(v) for v in k_rows[competitor]],
    }
    k_changes: list[QKChange] = []
    for t in range(HEAD_DIM):
        x = chosen[t]
        if x > ZERO:
            targets = ((w, -eps), (competitor, eps))
        elif x < ZERO:
            targets = ((w, eps), (competitor, -eps))
        else:
            continue  # 该分量的 Q 扰动后为 0，K 改动对点积差无贡献
        for row_pos, direction in targets:
            old = k_rows[row_pos][t]
            new = _clamp_to_float(old + direction, old - eps, old + eps)
            if new is None:
                return None
            k_new[row_pos][t] = new
            if Fraction.from_float(new) != old:
                k_changes.append(
                    QKChange(
                        position=row_pos,
                        local_position=layout.local_positions[row_pos],
                        component=t,
                        old_value=float(old),
                        new_value=new,
                        delta=new - float(old),
                    )
                )

    keys = sorted(analysis.scores)

    # --- 1) 精确有理数复核：扰动必须在界内、且赢家确实改变 ---
    q_new_exact = [Fraction.from_float(v) for v in q_new]
    exact_scores: dict[int, Fraction] = {}
    for j in keys:
        row = (
            [Fraction.from_float(v) for v in k_new[j]]
            if j in k_new
            else k_rows[j]
        )
        exact_scores[j] = _dot_exact(q_new_exact, row)
    exact_winner = _earliest_max_exact(exact_scores)
    within_bounds = True
    for change in (*q_changes, *k_changes):
        old = Fraction.from_float(change.old_value)
        new = Fraction.from_float(change.new_value)
        if abs(new - old) > eps:
            within_bounds = False
    exact_changed = (
        within_bounds and exact_winner != analysis.winner_position
    )

    # --- 2) float64 复核：用户按同样方式重算也应看到赢家改变 ---
    k_float = {
        j: (k_new[j] if j in k_new else [float(v) for v in k_rows[j]]) for j in keys
    }
    float_scores = {j: _float_score(q_new, k_float[j]) for j in keys}
    float_winner = _earliest_max_float(float_scores)
    float_changed = float_winner != analysis.winner_position
    if not (exact_changed and float_changed):
        return None

    return Witness(
        competitor_position=competitor,
        competitor_local_position=layout.local_positions[competitor],
        q_changes=q_changes,
        k_changes=k_changes,
        recomputed_winner_position=float_winner,
        recomputed_winner_local_position=layout.local_positions[float_winner],
        recomputed_winner_score=float_scores[float_winner],
        changed=True,
        within_bounds=True,
        verified_exact=True,
        verified_float64=True,
        note=(
            "改动位置为全局 token 位置；旧值取自种子生成的 float64 Q/K，"
            "新值在 [旧值-ε, 旧值+ε] 内；重算赢家按同一文档内因果掩码的"
            "float64 QK 点积与最早并列规则得到"
        ),
    )


def analyze_winner_stability(req: WinnerStabilityRequest) -> WinnerStabilityResponse:
    """对请求中的每个查询位置做稳定性分析。"""
    q, k = generate_qk(req.seed, req.doc_lengths)
    q_rows = _to_fraction_rows(q)
    k_rows = _to_fraction_rows(k)
    layout = _layout(req.doc_lengths)
    eps = Fraction.from_float(float(req.epsilon))

    scores_by_query = _collect_block_scores(
        sorted(req.query_positions), q_rows, k_rows, layout, req.chunk_size
    )

    entries: list[WinnerStabilityEntry] = []
    for qi in req.query_positions:
        analysis = _analyze_query(
            qi, scores_by_query[qi], q_rows, k_rows, eps
        )
        w = analysis.winner_position
        tightest = (
            min(analysis.margins, key=lambda j: (analysis.margins[j], j))
            if analysis.margins
            else None
        )
        common = dict(
            query_position=qi,
            query_local_position=layout.local_positions[qi],
            doc_id=layout.doc_ids[qi],
            winner_position=w,
            winner_local_position=layout.local_positions[w],
            winner_score=_float_exact(analysis.winner_score),
            valid_key_count=len(analysis.scores),
            tightest_competitor_position=tightest,
            tightest_competitor_local_position=(
                layout.local_positions[tightest] if tightest is not None else None
            ),
            tightest_margin=(
                _round_down(analysis.margins[tightest])
                if tightest is not None
                else None
            ),
            tightest_margin_fraction=(
                _fraction_str(analysis.margins[tightest])
                if tightest is not None
                else None
            ),
        )

        # 只有自身可读：稳定，优势下界为空。
        if not analysis.margins:
            entries.append(
                WinnerStabilityEntry(
                    **common,
                    status="stable",
                    advantage_lower_bound=None,
                    reason="有效 key 只有查询自身，不存在竞争者；任何扰动下赢家都不变",
                )
            )
            continue

        if analysis.violations:
            # 按竞争者优势下界从紧到松依次尝试，取第一个能可靠复现的见证。
            ordered = sorted(
                analysis.violations, key=lambda j: (analysis.margins[j], j)
            )
            witness = None
            for competitor in ordered:
                witness = _build_witness(
                    analysis, competitor, q_rows[qi], k_rows, layout, eps
                )
                if witness is not None:
                    break
            if witness is None:
                competitor = ordered[0]
                entries.append(
                    WinnerStabilityEntry(
                        **common,
                        status="undetermined",
                        advantage_lower_bound=None,
                        reason=(
                            f"精确判定原赢家不稳定（竞争者位置 {competitor} 的下界 "
                            f"{_float_exact(analysis.margins[competitor]):.6e} ≤ 0），但界内扰动"
                            "写成 float64 后无法可靠复现赢家改变，属数值边界，故未确定；"
                            "不返回不可靠见证"
                        ),
                    )
                )
                continue
            assert witness is not None
            entries.append(
                WinnerStabilityEntry(
                    **common,
                    status="unstable",
                    advantage_lower_bound=None,
                    reason=(
                        f"存在界内扰动使原赢家改变：竞争者位置 "
                        f"{witness.competitor_position} 的优势下界 "
                        f"{_float_exact(analysis.margins[witness.competitor_position]):.6e} ≤ 0，"
                        "见 witness"
                    ),
                    witness=witness,
                )
            )
            continue

        bound = min(analysis.margins.values())
        entries.append(
            WinnerStabilityEntry(
                **common,
                status="stable",
                advantage_lower_bound=_round_down(bound),
                advantage_lower_bound_fraction=_fraction_str(bound),
                reason=(
                    "所有允许扰动下，原赢家都保持为最早的最大分数位置"
                    if bound > ZERO
                    else "下界为 0：更晚的竞争者可在扰动下与原赢家并列，"
                    "但并列取最早位置，原赢家仍不变"
                ),
            )
        )

    return WinnerStabilityResponse(
        seed=req.seed,
        doc_lengths=list(req.doc_lengths),
        chunk_size=req.chunk_size,
        epsilon=req.epsilon,
        total_tokens=sum(req.doc_lengths),
        num_docs=len(req.doc_lengths),
        all_stable=all(e.status == "stable" for e in entries),
        any_undetermined=any(e.status == "undetermined" for e in entries),
        undetermined_count=sum(1 for e in entries if e.status == "undetermined"),
        perturbation_model=(
            "Q 与每个 K 行的每个分量可独立取任意偏移 δ，|δ| ≤ epsilon；"
            "比较对象为原始 QK 点积（正缩放不改变赢家）；并列取最早位置"
        ),
        decision_method=(
            "exact rational arithmetic on the float64 inputs; witness replayed "
            "in exact rational and float64 arithmetic"
        ),
        queries=entries,
    )
