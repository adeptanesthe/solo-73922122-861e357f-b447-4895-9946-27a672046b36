"""FastAPI 入口：分块注意力核对服务。

端点
----
* ``GET  /health``           健康检查
* ``GET  /scenarios``        内置边界情形样例列表
* ``POST /verify``           单次核对（自定义分块算子 vs 独立稠密参考）
* ``POST /winner-stability`` 低精度误差预算下各查询原赢家 key 的稳定性
* ``POST /scenarios/run``    运行全部或单个内置样例
* ``GET  /docs``             Swagger UI（自动生成）
"""

from __future__ import annotations

import math

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .scenarios import SCENARIOS
from .schemas import (
    VerifyRequest,
    VerifyResponse,
    WinnerStabilityRequest,
    WinnerStabilityResponse,
)
from .stability import analyze_winner_stability
from .verification import TOLERANCE, run_verification

app = FastAPI(
    title="分块文档内因果注意力核对 API",
    version="1.0.0",
    description=(
        "CPU + float64 环境下，对自定义分块（文档内因果）注意力 autograd 算子"
        "与独立稠密参考实现进行前向输出和 Q/K/V 梯度核对，容差为 1e-8。"
        "工作张量与跨前后向保存量规模为 O(N + 块长²)。"
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _json_safe(value):
    """把 NaN/Inf 等非 JSON 合规值递归替换为字符串，避免 422 响应序列化失败。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


@app.get("/scenarios")
def list_scenarios() -> dict[str, object]:
    return {
        "tolerance": TOLERANCE,
        "scenarios": [
            {"name": name, "description": s.description, "request": s.request.model_dump()}
            for name, s in SCENARIOS.items()
        ],
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(
    req: VerifyRequest,
    include_tensors: bool = Query(
        default=True, description="是否在响应中回传完整输出/梯度张量"
    ),
) -> VerifyResponse:
    result = run_verification(req, include_tensors=include_tensors)
    # 核对失败属于业务结果而非服务错误：返回 200 并在体内明确 passed=false。
    return result


@app.post(
    "/winner-stability",
    response_model=WinnerStabilityResponse,
    summary="低精度误差预算下的注意力赢家稳定性",
    description=(
        "沿用 /verify 的 Q/K 生成方式（同 seed 与文档长度得到逐位相同的 float64 "
        "Q/K）。给定块长、1~8 个互不相同的查询位置与每分量误差上限 ε，逐查询判断"
        "仅缩小到自身文档内不晚于自己的 key 时，原 QK 点积赢家在所有允许扰动下"
        "是否不变。稳定时返回优势下界；不稳定时返回界内改动位置、新值与重算赢家；"
        "float64 无法可靠复现时明确标注未确定。"
    ),
)
def winner_stability(req: WinnerStabilityRequest) -> WinnerStabilityResponse:
    return analyze_winner_stability(req)


@app.post("/scenarios/run")
def run_scenarios(
    name: str | None = Query(default=None, description="只运行指定样例；缺省运行全部"),
    include_tensors: bool = Query(default=False),
) -> JSONResponse:
    if name is not None and name not in SCENARIOS:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"未知样例 {name!r}",
                "available": list(SCENARIOS.keys()),
            },
        )
    chosen = [name] if name else list(SCENARIOS.keys())
    results = {}
    all_passed = True
    for key in chosen:
        resp = run_verification(SCENARIOS[key].request, include_tensors=include_tensors)
        all_passed = all_passed and resp.passed
        results[key] = resp.model_dump()
    return JSONResponse(
        content={"all_passed": all_passed, "tolerance": TOLERANCE, "results": results}
    )
