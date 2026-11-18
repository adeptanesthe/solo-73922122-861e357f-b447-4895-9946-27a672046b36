# 分块文档内因果注意力核对 API

把多篇文档拼入同一批次（总长 N ≤ 512），在 **CPU + float64** 下核对一个
自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度。

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax 原语（online softmax 手写）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：输出与 Q/K/V 梯度的最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因

## 目录结构

```
app/
  attention.py    # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py    # 独立稠密参考（完整 N×N 掩码与注意力矩阵，不受内存约束）
  verification.py # 数据生成、双路计算与误差核对
  stability.py    # /winner-stability：Q/K 扰动下的赢家稳定性（精确有理数判定）
  schemas.py      # 请求/响应模型与输入校验
  scenarios.py    # 内置边界情形样例
  main.py         # FastAPI 入口
tests/            # pytest：正确性、gradcheck、边界情形、内存与源码审计、稳定性、API
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

`/docs` 中可直接试调 `/verify`、`/winner-stability` 与场景端点。

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

### `POST /winner-stability`

把 Q/K 降到低精度前，判断每个查询最关注的 key（QK 点积最大者）在误差预算内
是否仍是同一个。Q/K 沿用 `/verify` 的生成方式：同一 `seed` 与 `doc_lengths`
得到逐位相同的 float64 Q/K。

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | 0 ≤ seed ≤ 2⁶³−1，与 `/verify` 共用生成序列 |
| `doc_lengths` | int[] | 正整数列表，总长 1..64 |
| `chunk_size` | int | 1..32 |
| `query_positions` | int[] | 1~8 个**互不相同**的全局 token 位置，须在 `0..总长-1` |
| `epsilon` | float | 0 ≤ ε ≤ 0.5，Q/K 每个分量的独立误差上限 |

语义：

- 每个查询只与**同一篇文档中位置不晚于自己**的 key 比较（与 `/verify` 掩码一致）。
- 分数为原始 QK 点积；`/verify` 的 softmax 缩放为正，不改变赢家与顺序。
- 并列取**最早位置**；只有查询自身可读时视为稳定，优势下界为空。
- Q 的每个分量与每个 K 行的每个分量都可独立偏移任意 `δ ∈ [-ε, ε]`。
- 判定为每个查询**独立**给出，互不影响。

响应核心字段（`queries[]` 每项）：

| 字段 | 含义 |
| --- | --- |
| `status` | `stable` / `unstable` / `undetermined` |
| `winner_position`, `winner_score` | 原赢家的全局位置与原始点积分数 |
| `advantage_lower_bound` | 稳定时：对所有允许扰动一致成立的 QK 点积优势下界；`null` 表示不稳定或仅自身可读 |
| `advantage_lower_bound_fraction` | 该下界的精确有理数表示 |
| `tightest_competitor_position`, `tightest_margin` | 最紧竞争者及其下界 |
| `witness` | 不稳定时的改动见证（见下） |

`witness` 包含 `q_changes` / `k_changes`（每项的全局位置、文档内位置、分量下标、
旧值、新值、增量）、`recomputed_winner_position`、`changed=true`、`within_bounds=true`，
以及 `verified_exact` 与 `verified_float64` 双重复核标记。所有新值都在
`[旧值-ε, 旧值+ε]` 内，按这些值重算确实会改变赢家。

`undetermined`：精确有理数判定已完成，但把所需扰动写成 float64 后无法在
float64 下可靠复现赢家改变（数值边界）。此时 `reason` 明确说明未确定，
不返回不可靠的 `witness`；响应顶层 `any_undetermined` / `undetermined_count`
给出汇总。

curl 示例：

```bash
curl -X POST "http://localhost:8000/winner-stability" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,
       "query_positions":[0,1,3,6,7,17,22],"epsilon":0.01}'
```

实现要点：

1. **精确判定**：对本就由 float64 生成的 Q/K 用 `fractions.Fraction` 做精确
   有理数运算，避免浮点舍入造成误判。
2. **下界推导**：固定 Q 扰动后，K 侧可把 `s_w − s_j` 压到
   `q~·d − 2ε‖q~‖₁`；再逐分量最小化凸的分段线性函数
   `d_t·x − 2ε|x|`（最小值在区间端点或 0 处），得到竞争者 `j` 的精确
   下确界 `m_j`。更早的竞争者需 `m_j > 0`（可追平即由更早位置获胜），
   更晚的竞争者 `m_j ≥ 0` 即稳定（并列仍取原赢家）。返回的
   `min_j m_j` 是对所有扰动一致成立的下界；因 `min_j inf` 与 `inf min`
   一般不可交换，它可能比 `inf min` 更保守。
3. **按块计算**：分数按 `chunk_size` 的 query/key 块遍历，只为被查询 token
   收集有效位置对，不构造任何 N×N 矩阵；峰值额外内存 O(查询数 × N)。
4. **见证双重复核**：先精确有理数复核（改动均在 ±ε、赢家确实改变），再用
   float64 重算复核；任一不通过即降级为 `undetermined`。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。

## 自测结果

`pytest -q` 共 63 项全部通过，包含：11 组参数化前向+梯度核对（误差均 ≤ 1e-8，
实测在 1e-12 ~ 1e-16 量级）、`torch.autograd.gradcheck` 有限差分核对、
块长不变性、确定性、掩码语义、大分数有限性、保存张量线性规模审计、
禁用算子源码审计、全部 API 端点与参数校验测试。

`/winner-stability` 另有专项测试：Q/K 与 `/verify` 逐位一致；随机采样复核
稳定结论与下界；见证的界内性、独立重放与双重复核；`ε=0` 退化为普通 argmax；
块长不影响结论；较早/较晚竞争者的并列语义；`undetermined` 数值边界路径；
块计算审计（无 N×N 张量与内置 softmax）；8 个查询位置与 API 校验。
