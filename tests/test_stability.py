"""赢家稳定性端点测试：语义、下界、见证、数值边界与输入校验。"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch
from fastapi.testclient import TestClient

from app import stability
from app.main import app
from app.schemas import WinnerStabilityRequest
from app.stability import analyze_winner_stability
from app.verification import generate_inputs, generate_qk

client = TestClient(app)

BASE = {
    "seed": 42,
    "doc_lengths": [7, 11, 5],
    "chunk_size": 8,
    "query_positions": [0, 1, 3, 6, 7, 17, 22],
    "epsilon": 0.01,
}


def _valid_keys(doc_lengths, qi):
    layout = stability._layout(doc_lengths)
    doc = layout.doc_ids[qi]
    start = layout.doc_starts[doc]
    return [j for j in range(start, qi + 1) if layout.doc_ids[j] == doc]


def _float_scores(q_row, k_rows, keys):
    return {j: sum(q_row[t] * k_rows[j][t] for t in range(16)) for j in keys}


def _argmax_earliest(scores):
    best = None
    for j in sorted(scores):
        if best is None or scores[j] > scores[best]:
            best = j
    return best


def test_qk_generation_matches_verify():
    """同 seed/长度下，/winner-stability 使用的 Q/K 与 /verify 逐位一致。"""
    data = generate_inputs(
        type("Req", (), {"seed": 42, "doc_lengths": [7, 11, 5]})()
    )
    q, k = generate_qk(42, [7, 11, 5])
    assert torch.equal(q, data.q)
    assert torch.equal(k, data.k)


def test_response_shape_and_sorted_semantics():
    r = analyze_winner_stability(WinnerStabilityRequest(**BASE))
    assert r.total_tokens == 23
    assert [e.query_position for e in r.queries] == BASE["query_positions"]
    for e in r.queries:
        keys = _valid_keys(BASE["doc_lengths"], e.query_position)
        assert e.valid_key_count == len(keys)
        assert e.winner_position in keys
        assert e.winner_local_position == keys.index(e.winner_position)
        assert e.status in ("stable", "unstable", "undetermined")


def test_only_self_readable_is_stable_with_empty_bound():
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=7,
            doc_lengths=[1, 4],
            chunk_size=3,
            query_positions=[0, 1],
            epsilon=0.5,
        )
    )
    single = r.queries[0]
    assert single.winner_position == 0
    assert single.valid_key_count == 1
    assert single.status == "stable"
    assert single.advantage_lower_bound is None
    assert single.witness is None
    assert "自身" in single.reason


@pytest.mark.parametrize(
    "seed,doc_lengths,eps,chunk,queries",
    [
        (42, [7, 11, 5], 0.01, 8, [0, 1, 3, 6, 7, 17, 22]),
        (42, [7, 11, 5], 0.5, 3, [6, 7, 17]),
        (3, [10], 0.2, 4, [0, 5, 9]),
        (9, [64], 0.05, 7, [1, 63]),
    ],
)
def test_stable_bound_holds_under_random_perturbations(
    seed, doc_lengths, eps, chunk, queries
):
    """稳定结论用随机采样复核；优势下界不得高于任何采样点的真实差距。"""
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=seed,
            doc_lengths=doc_lengths,
            chunk_size=chunk,
            query_positions=queries,
            epsilon=eps,
        )
    )
    q, k = generate_qk(seed, doc_lengths)
    qf, kf = q.tolist(), k.tolist()
    rng = random.Random(seed)
    for e in r.queries:
        keys = _valid_keys(doc_lengths, e.query_position)
        qi = e.query_position
        if e.status == "stable":
            # 原赢家在所有采样扰动下不变
            for _ in range(200):
                qn = [v + rng.uniform(-eps, eps) for v in qf[qi]]
                kn = {
                    j: [v + rng.uniform(-eps, eps) for v in kf[j]] for j in keys
                }
                assert _argmax_earliest(_float_scores(qn, kn, keys)) == e.winner_position
            if e.advantage_lower_bound is not None:
                # 下界不超过任何允许扰动下的真实最小差距
                for _ in range(50):
                    qn = [v + rng.uniform(-eps, eps) for v in qf[qi]]
                    kn = {
                        j: [v + rng.uniform(-eps, eps) for v in kf[j]]
                        for j in keys
                    }
                    sc = _float_scores(qn, kn, keys)
                    real_gap = sc[e.winner_position] - max(
                        sc[j] for j in keys if j != e.winner_position
                    )
                    assert e.advantage_lower_bound <= real_gap + 1e-12
        else:
            assert e.status == "unstable"
            w = e.witness
            assert w is not None and w.changed
            assert w.verified_exact and w.verified_float64
            assert w.recomputed_winner_position != e.winner_position


@pytest.mark.parametrize(
    "seed,doc_lengths,eps,chunk,queries",
    [
        (42, [7, 11, 5], 0.01, 8, [0, 1, 3, 6, 7, 17, 22]),
        (3, [10], 0.2, 4, [0, 5, 9]),
        (9, [64], 0.05, 7, [1, 63]),
    ],
)
def test_witness_is_within_bounds_and_replays(seed, doc_lengths, eps, chunk, queries):
    """见证每处改动都在 ±ε 内，且按返回的新值独立重算确实改变赢家。"""
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=seed,
            doc_lengths=doc_lengths,
            chunk_size=chunk,
            query_positions=queries,
            epsilon=eps,
        )
    )
    q, k = generate_qk(seed, doc_lengths)
    qf, kf = q.tolist(), k.tolist()
    eps_fraction = Fraction.from_float(eps)
    for e in r.queries:
        if e.status != "unstable":
            continue
        w = e.witness
        assert w is not None
        # 改动在界内（精确有理数比较），旧值来自原始 Q/K
        for change in (*w.q_changes, *w.k_changes):
            old = Fraction.from_float(change.old_value)
            new = Fraction.from_float(change.new_value)
            assert abs(new - old) <= eps_fraction
            assert change.old_value != change.new_value
        # 独立按全局位置重放
        qi = e.query_position
        qn = list(qf[qi])
        for change in w.q_changes:
            assert change.position == qi
            assert abs(change.old_value - qf[qi][change.component]) < 1e-12
            qn[change.component] = change.new_value
        keys = _valid_keys(doc_lengths, qi)
        kn = {j: list(kf[j]) for j in keys}
        for change in w.k_changes:
            assert change.position in keys
            assert abs(change.old_value - kf[change.position][change.component]) < 1e-12
            kn[change.position][change.component] = change.new_value
        sc = _float_scores(qn, kn, keys)
        assert _argmax_earliest(sc) == w.recomputed_winner_position
        assert w.recomputed_winner_position != e.winner_position


def test_unstable_when_earlier_competitor_can_tie():
    """较早竞争者若能在界内追平，则并列取最早位置 → 原赢家不稳定。"""
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=1,
            doc_lengths=[16],
            chunk_size=4,
            query_positions=[12],
            epsilon=0.01,
        )
    )
    e = r.queries[0]
    assert e.winner_position == 11  # 赢家不是最早可读位置 0
    assert e.status == "unstable"
    assert e.witness.competitor_position < e.winner_position
    assert e.witness.recomputed_winner_position == e.witness.competitor_position


def test_stable_bound_zero_with_later_tie_still_stable():
    """更晚竞争者最多追平；并列取最早，因此下界恰为 0 时仍稳定。

    构造 ``q1 = ε``、``K_0 = 2ε``、``K_1 = 0``（逐分量）：竞争者 1 的
    下确界恰为 ``d·x - 2ε|x| = 0``，即界内最好只能追平；因它更晚，
    并列时原赢家 0 仍胜出 → stable 且下界为 0。
    """
    eps = 0.05
    q = torch.zeros(2, 16, dtype=torch.float64)
    q[1] = eps
    k = torch.stack(
        [
            torch.full((16,), 2 * eps, dtype=torch.float64),
            torch.zeros(16, dtype=torch.float64),
        ]
    )

    import app.stability as stability_module

    original = stability_module.generate_qk
    stability_module.generate_qk = lambda seed, docs: (q, k)
    try:
        r = analyze_winner_stability(
            WinnerStabilityRequest(
                seed=0,
                doc_lengths=[2],
                chunk_size=2,
                query_positions=[1],
                epsilon=eps,
            )
        )
    finally:
        stability_module.generate_qk = original
    e = r.queries[0]
    assert e.winner_position == 0
    assert e.status == "stable"
    assert e.advantage_lower_bound == 0.0
    assert "并列" in e.reason


def test_epsilon_zero_bound_equals_exact_gap():
    """ε=0 时优势下界等于真实最小分差。"""
    for seed in (1, 2, 7):
        r = analyze_winner_stability(
            WinnerStabilityRequest(
                seed=seed,
                doc_lengths=[12, 8],
                chunk_size=5,
                query_positions=[11, 19],
                epsilon=0.0,
            )
        )
        q, k = generate_qk(seed, [12, 8])
        qf, kf = q.tolist(), k.tolist()
        for e in r.queries:
            keys = _valid_keys([12, 8], e.query_position)
            sc = _float_scores(qf[e.query_position], kf, keys)
            gap = sc[e.winner_position] - max(
                sc[j] for j in keys if j != e.winner_position
            )
            assert e.status == "stable"
            assert e.advantage_lower_bound == pytest.approx(gap, rel=1e-12, abs=1e-15)


def test_chunk_size_does_not_change_conclusions():
    def run(chunk):
        return analyze_winner_stability(
            WinnerStabilityRequest(
                seed=2026,
                doc_lengths=[40, 24],
                chunk_size=chunk,
                query_positions=[2, 23, 39, 63],
                epsilon=0.05,
            )
        )

    base = run(1)
    for chunk in (7, 16, 32):
        other = run(chunk)
        for a, b in zip(base.queries, other.queries):
            assert a.status == b.status
            assert a.winner_position == b.winner_position
            assert a.advantage_lower_bound == b.advantage_lower_bound
            if a.witness is not None:
                assert a.witness.recomputed_winner_position == (
                    b.witness.recomputed_winner_position
                )


def test_no_full_length_square_matrix(monkeypatch):
    """按块计算审计：不得出现 N×N 规模张量，也不得调用 torch.softmax。"""
    import inspect

    source = inspect.getsource(stability)
    assert "torch.softmax" not in source
    assert "scaled_dot_product_attention" not in source

    seen: list[tuple[int, ...]] = []
    original_matmul = torch.Tensor.__matmul__

    def spy(self, other):
        seen.append(tuple(self.shape))
        return original_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)
    n = 64

    def fake_generate_qk(seed, doc_lengths):
        g = torch.Generator().manual_seed(seed)
        return (
            torch.randn(n, 16, generator=g, dtype=torch.float64),
            torch.randn(n, 16, generator=g, dtype=torch.float64),
        )

    monkeypatch.setattr(stability, "generate_qk", fake_generate_qk)
    analyze_winner_stability(
        WinnerStabilityRequest(
            seed=0,
            doc_lengths=[n],
            chunk_size=8,
            query_positions=list(range(n - 8, n)),
            epsilon=0.01,
        )
    )
    for shape in seen:
        assert shape[0] * shape[1] <= n * 16, f"出现过大张量乘法 {shape}"


def test_eight_query_positions_accepted():
    resp = client.post(
        "/winner-stability",
        json={
            "seed": 5,
            "doc_lengths": [64],
            "chunk_size": 8,
            "query_positions": [0, 9, 18, 27, 36, 45, 54, 63],
            "epsilon": 0.0,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["queries"]) == 8


def test_epsilon_zero_matches_plain_argmax():
    for seed in (1, 2, 3):
        r = analyze_winner_stability(
            WinnerStabilityRequest(
                seed=seed,
                doc_lengths=[20],
                chunk_size=6,
                query_positions=[19],
                epsilon=0.0,
            )
        )
        e = r.queries[0]
        assert e.status == "stable"
        assert e.advantage_lower_bound is not None
        assert e.advantage_lower_bound > 0.0
        q, k = generate_qk(seed, [20])
        keys = _valid_keys([20], 19)
        sc = _float_scores(q.tolist()[19], k.tolist(), keys)
        assert _argmax_earliest(sc) == e.winner_position


def test_undetermined_when_float64_cannot_represent_witness(monkeypatch):
    """数值边界：精确判定不稳定，但所需扰动小于 float64 可表示间隔。

    构造同一文档内 K 行完全相同的数据：更晚竞争者可在极小扰动下追平，
    但 ε 远小于 1.0 附近的 ulp，float64 无法表示该改动 → 未确定。
    """
    n, d = 4, 16
    q = torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    k = torch.ones(n, d, dtype=torch.float64)  # 所有 K 行相同 → 分数全并列

    def fake_generate_qk(seed, doc_lengths):
        return q, k

    monkeypatch.setattr(stability, "generate_qk", fake_generate_qk)
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=0,
            doc_lengths=[4],
            chunk_size=2,
            query_positions=[3],
            epsilon=1e-20,
        )
    )
    e = r.queries[0]
    assert e.winner_position == 0  # 并列取最早
    assert e.status == "undetermined"
    assert e.witness is None
    assert e.advantage_lower_bound is None
    assert "未确定" in e.reason
    assert r.any_undetermined and r.undetermined_count == 1
    assert r.all_stable is False


def test_exact_tie_witness_rounding_is_verified(monkeypatch):
    """可精确表示的并列构造应给出 unstable 且见证通过双重复核。"""
    n, d = 3, 16
    eps = 2.0 ** -20
    winner = torch.zeros(d, dtype=torch.float64)
    competitor = torch.full((d,), 2 * eps, dtype=torch.float64)
    q = torch.empty(n, d, dtype=torch.float64)
    q[0] = 0.0
    q[1] = torch.full((d,), eps, dtype=torch.float64)
    q[2] = torch.full((d,), eps, dtype=torch.float64)
    k = torch.stack([winner, competitor, competitor])

    monkeypatch.setattr(stability, "generate_qk", lambda seed, docs: (q, k))
    r = analyze_winner_stability(
        WinnerStabilityRequest(
            seed=0,
            doc_lengths=[3],
            chunk_size=2,
            query_positions=[2],
            epsilon=eps,
        )
    )
    e = r.queries[0]
    # 原赢家为位置 1（分数 32ε²），位置 2 并列；更早的 0 分差为 0？
    # 位置 1 与 2 分数相同，最早为 1；位置 1 的分数高于位置 0。
    assert e.winner_position == 1
    # 更晚的竞争者 2 在界内可反超（下界为负），见证必须改变赢家。
    assert e.status == "unstable"
    assert e.witness is not None
    assert e.witness.changed
    assert e.witness.verified_exact and e.witness.verified_float64
    assert e.witness.recomputed_winner_position != e.winner_position


# ---------------------------------------------------------------------------
# API 层
# ---------------------------------------------------------------------------


def test_openapi_includes_winner_stability():
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/winner-stability" in paths
    assert "/verify" in paths


def test_api_winner_stability_ok():
    resp = client.post("/winner-stability", json=BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_tokens"] == 23
    assert body["epsilon"] == 0.01
    assert len(body["queries"]) == len(BASE["query_positions"])
    assert body["all_stable"] is False
    unstable = [q for q in body["queries"] if q["status"] == "unstable"]
    assert unstable
    w = unstable[0]["witness"]
    assert w["changed"] is True
    assert w["verified_exact"] and w["verified_float64"]
    assert w["recomputed_winner_position"] != unstable[0]["winner_position"]
    assert w["q_changes"] or w["k_changes"]
    for change in w["q_changes"] + w["k_changes"]:
        assert abs(change["new_value"] - change["old_value"]) <= BASE["epsilon"] + 1e-15


def test_api_stable_query_has_bound():
    resp = client.post(
        "/winner-stability",
        json={
            "seed": 42,
            "doc_lengths": [7, 11, 5],
            "chunk_size": 8,
            "query_positions": [3],
            "epsilon": 0.001,
        },
    )
    body = resp.json()
    entry = body["queries"][0]
    assert entry["status"] == "stable"
    assert entry["advantage_lower_bound"] is not None
    assert entry["advantage_lower_bound"] > 0
    assert entry["advantage_lower_bound_fraction"]
    assert entry["witness"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"doc_lengths": [40, 40]},  # 总长 80 > 64
        {"query_positions": [0, 0]},  # 重复
        {"query_positions": [23]},  # 越界
        {"query_positions": [-1]},
        {"query_positions": list(range(9))},  # 9 个位置
        {"query_positions": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
        {"chunk_size": 0},
        {"chunk_size": 33},
        {"epsilon": -0.01},
        {"epsilon": 0.51},
        {"seed": -1},
        {"seed": 2**63},
        {"query_positions": [True]},
    ],
)
def test_api_validation_errors(mutation):
    payload = {**BASE, **mutation}
    if "query_positions" not in mutation:
        payload["query_positions"] = [0]
    resp = client.post("/winner-stability", json=payload)
    assert resp.status_code == 422, (mutation, resp.json())


def test_api_verify_still_works():
    resp = client.post(
        "/verify?include_tensors=false",
        json={"seed": 42, "doc_lengths": [7, 11, 5], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 200
    assert resp.json()["passed"] is True
