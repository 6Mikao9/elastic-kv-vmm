# 已知未来请求规划器设计

本文面向当前 `feature/sglang-framework` 分支，定义“已知未来请求”如何影响
SGLang 的准入顺序和 VMM 驻留决策。第一阶段以正确、可离线验证为目标；
GPU 结果出来前，不把规划收益写成已证实的性能收益。

当前实现状态：`adaptive_sparse/future_plan.py` 已提供不依赖 SGLang/CUDA 的
确定性事件模拟，比较 `fifo`、`longest`、`mode-aware` 和 `future` 四种策略；
`adaptive_sparse/sglang_future_hook.py` 已实现默认关闭的 scheduler 排序 hook。
在带原生 `kv_hints` 的 SGLang revision 上它直接复用上游传输；在 0.5.2 上则
由 `adaptive_sparse/kv_hints_compat.py` 在 HTTP 请求类、tokenizer 和 scheduler
之间补上字段传递。两条路径都保持 `PrefillAdder` 为最终准入权威。原生传输路径
已完成一次真实 scheduler smoke；0.5.2 兼容路径也已在真实 A100 的 V5 eager
和 CUDA Graph 服务中通过 VMM+future 集成 smoke。当前证据只证明链路和回退行为，
尚不能当成服务端吞吐或延迟收益。

在驻留准入侧，`adaptive_sparse/future_residency.py` 已把同一份 hint 中的
`residence` 与预计输出长度变成一个确定性的“是否占用完整 GPU KV 槽”决策，
并由 `adaptive_sparse/request_residency.py` 仅在
`EKV_RESIDENT_HINT_POLICY=adaptive` 时启用。默认 `off` 保持此前的长请求默认
策略，避免改变已有实验。`adaptive_sparse/future_prefill.py` 进一步把 horizon
内等待请求的完整 KV、HiSparse 热集和可选 activation 预测聚合为有界预算，
在 eager prefill boundary 上通过 `dry_run` 或 `enforce` 消费。运行中抢占仍是
已有 `SchedulerBudget` 的独立动作；两者共用同一预算观测，但尚未求在线最优解。

## 1. 要解决的问题

输入不是传统的未知到达流量，而是一段可提前看到的请求集合。每个请求至少可提供：

- 预计开始时间或相对释放时间；
- 截止时间；
- 实际 prompt 长度；
- 预计输出 token 数；
- 请求类型，例如 `tool`、`chat`、`background` 或 `rollout`；
- 期望驻留模式：`auto`、`resident` 或 `host_backed`；
- 业务权重或硬优先级。

规划器回答两个问题：

1. 当前应让哪些等待请求先进入 prefill；
2. 已运行请求应在什么时候降级、升级或保持驻留。

规划器不能代替 SGLang 的准入检查。真正的 KV、请求槽、LoRA、grammar、
Mamba 和 paged allocator 约束仍由原生 `PrefillAdder` 决定。规划器只提供顺序、
期限和有界预留建议；建议失效时必须退回原生行为。

## 2. 已确认的 SGLang 接入点

审计基线为本地 SGLang `9f6678a`。

### 2.1 请求提示已能端到端传递

`GenerateReqInput` 已有 `kv_hints` 字段，tokenizer manager 不解析其业务内容，
而是把 `KvHintsEnvelope` 原样传给 scheduler，并最终挂到 `Req.kv_hints`。
每个 action 自带类型和版本，未实现的 action 可以忽略。

第一阶段使用新的 action：

```json
{
  "action_id": "request-42",
  "action_type": "elastic.future",
  "action_version": "1",
  "payload": {
    "release_after_ms": 250,
    "deadline_after_ms": 1800,
    "expected_output_tokens": 192,
    "kind": "tool",
    "residence": "auto",
    "weight": 1.0
  }
}
```

时间必须相对于同一个请求事件，推荐相对于 `Req.time_stats.created_time`。
不要混用 Wall clock 与 `time.perf_counter()`。所有数字都要做有限值、非负值和
上限校验。`kv_hints` 是受信任 orchestrator 的内部提示通道，不应当成任意应用
客户端的安全控制接口。

### 2.2 原生排序发生在 prefill 准入内部

`Scheduler._get_new_batch_prefill_raw()` 在以下顺序中工作：

1. 处理 grammars，检查 running batch 和请求槽；
2. 调用 `self.policy.calc_priority(waiting_queue, running_batch, ...)`；
3. 构造 `PrefillAdder`；
4. 从队首逐个调用 `add_one_req()`；
5. 用 `ScheduleBatch.init_new()` 和 `prepare_for_extend()` 形成真实 prefill batch。

因此，未来排序应接在原生 `calc_priority()` 之后。若只在外部包装
`get_new_batch_prefill()` 前重排等待队列，下一步原生排序会覆盖结果。当前不修改
上游代码的最小兼容方案是：调用原生 `SchedulePolicy.calc_priority()` 后，再调用
规划器做一次稳定重排。直接 fork SGLang 时，应把同一调用点放进 scheduler 源码，
而不是长期依赖 monkeypatch。

迁移动作不能在排序函数中执行。当前 VMM 的 `SchedulerBudget.boundary()` 已经证明
迁移应发生在 eager、所有 rank 可共识的 pre-forward 边界。规划器可以包装
`Scheduler.get_new_batch_prefill()` 和 `Scheduler.update_running_batch()`，在这些边界
执行预检，但排序本身必须是无 GPU I/O 的纯 CPU 决策。

## 3. 模块边界

`adaptive_sparse/future_plan.py` 只包含纯 Python 决策；GPU 与 SGLang
适配放在独立的 hook 中。当前离线 profile 已支持 `auto`、`resident` 和
`host_backed` 三种驻留偏好，并用其约束模拟器可选的 residency mode；
`auto_resident_output_threshold` 可让离线回放使用与在线相同的短输出阈值。

```text
KvHintsEnvelope
      |
      v
FutureRequestSpec.parse() ---- invalid hint -> ignore / count metric
      |
      v
CapacitySnapshot <---- allocator, VMM tracker, activation telemetry
      |
      v
FuturePlanner.plan(now, waiting, running)
      |             |
      |             +--> residency transitions (advisory)
      +--------------------> deterministic waiting order
                              |
                              v
                     native PrefillAdder is authoritative
```

核心类型应保持可序列化，避免规划器持有 `Req` 作为长期状态键：

```python
@dataclass(frozen=True)
class FutureRequestSpec:
    rid: str
    release_at: float
    deadline: float | None
    expected_output_tokens: int
    residence: Literal["auto", "resident", "host_backed"]
    weight: float

@dataclass(frozen=True)
class CapacitySnapshot:
    now: float
    logical_kv_bytes: int
    physical_kv_bytes: int
    activation_headroom_bytes: int
    prefill_peak_bytes: int

@dataclass(frozen=True)
class PlanEntry:
    rid: str
    order_key: tuple
    wanted_residence: str
    reserve_until: float | None
    reason: str
```

`rid` 是唯一跨 CPU 状态键。重试和 retraction 可能创建新的 `Req` 对象，不能依赖对象
地址。重复 `rid` 的冲突 hint 要拒绝或以后到者覆盖，策略必须固定并测试。

## 4. 第一阶段排序策略

MVP 不求解任意加权优化问题，而是使用可解释的字典序，避免隐藏常数和不可复现的
打分：

1. 保留 SGLang 已有的硬 priority 语义和 FCFS 平局规则；
2. 未到 `release_at` 的请求放到已释放请求之后；这是软排序提示，不是硬准入
   门槛，`PrefillAdder` 的遍历仍可能尝试后面的未释放请求；
3. 已释放请求中优先 `latest_start` 更早者：

   ```text
   latest_start = deadline - estimated_prefill_time - estimated_decode_time
   ```

4. `latest_start` 相同，优先预计完成更早、释放内存更快的请求；
5. 无 hint 的请求维持原生顺序，不能因为规划关闭而被移到队尾；
6. hook 不跨越 SGLang 原生 priority tier；启用 priority scheduling 时只在
   相同 priority 的连续队列段内重排，无法取得有效 priority 时整体回退；
7. `chunked_req` 不参与重排，当前 chunk 必须先完成；
8. 只改变合法候选的相对顺序，不跳过原生 `PrefillAdder` 检查。

服务时间先用可观测的 EWMA 估计：

```text
prefill_time = prompt_tokens / prefill_tokens_per_s
decode_time  = expected_output_tokens / decode_tokens_per_s
```

初值来自配置，之后只用真实完成样本更新。estimator 必须带置信区间或保守倍率；
没有样本时宁可使用宽裕估计，也不要声称能精确满足 deadline。

### 4.1 受控 rollout 的可执行计划

`plan_rollout()` 把离线策略结果转换成 orchestrator 可执行的动作。每个 action
包含原始与计划到达时间、预计完成时间、请求类型，以及可直接放入
`elastic.future` payload 的字段。控制器在 `submit_at_ms` 才实际提交请求；
hint 使用 `release_after_ms=0`，并把绝对 deadline 换算成相对 scheduler
接收时刻的 `deadline_after_ms`。

```bash
python scripts/future_plan_compare.py \
  benchmarks/workloads/known-future-rollout.json \
  --emit-policy future \
  --output results/future-planning/known-future-rollout.json
```

当有限 horizon 内的高价值请求即将到来、当前长 prefill 会让它错过硬期限，而
短暂空转能保住期限时，`future` 策略允许 idle 到该请求到达。这里控制的是
orchestrator 的真实提交时刻；runtime hook 中的 `release_after_ms` 仍只是软排序
提示，不能被误写成 `PrefillAdder` 的硬准入门槛。

仓库内合成样例只用于回归 harness。该样例中 FIFO、mode-aware、future 分别记录
3、2、0 个 deadline miss，future 的模拟 makespan 为 1,988 ms，峰值预算为
4,008,192 bytes。它是固定输入上的策略行为证据，不是模型、GPU 或生产负载上的
性能结论；真实效果必须用同一模型、权重、请求体和 TP 配置重新测量。

## 5. VMM 驻留计划

驻留规划使用和现有 `SchedulerBudget` 相同的保守容量账本：

```text
required =
    mapped_physical_KV
  + pending_promotions
  + pending_prefill_backing
  + activation_headroom
```

只有所有 TP rank 的 `required <= min(common_capacity)` 时，计划才可执行。

MVP 动作优先级：

1. 不为远期请求提前映射完整 KV；只对当前 prefill 和有限 horizon 内的请求做
   有 TTL 的软预留；
2. 当截止时间临近且收益为正时，允许把低未来价值的 host-backed 请求升级；
3. 在必须降级时，优先迁移“未来时段不活跃、传输代价小、可整块释放”的请求；
4. 已预测但未发生的收益不能抵扣已经占用的字节；
5. 预留到期立即失效，并在每次真实 allocation 前重新核对，不做乐观续期。

扩展现有 `Candidate.remaining_benefit` 时，应区分三类价值：

- deadline value：满足硬期限的价值；
- residency value：驻留能减少的预计迁移或重算时间；
- reuse value：共享前缀或后续请求能复用的价值。

第一阶段只输出建议并记录“若执行”的结果。只有在离线回放和 GPU smoke 证明
决策正确后，才让建议触发真实迁移。

### 5.1 已实现的请求级驻留准入切片

当前先实现最容易独立验证的一步：请求已完成 prefill、准备进入 HiSparse
staging 时，决定它使用原生有界热池，还是占用一个完整 GPU KV 驻留槽。策略是纯
CPU、确定性的，并且每个 TP rank 使用相同的 hint；真正的 admission 仍受
`EKV_RESIDENT_FREE_RESERVE` 和所有 rank 的最小可用物理容量约束。

| `residence` | 当前行为 |
|---|---|
| `host_backed` | 放弃完整驻留槽，使用原生 HiSparse 热池 |
| `resident` | 有槽和物理 reserve 时请求完整驻留 |
| `auto` | 预计输出不少于阈值时驻留，否则使用原生热池 |
| 缺失 hint | 保持旧行为：超过原生热池容量的长请求可驻留 |

`resident` 是带容量的软偏好：槽不足、物理 reserve 不足，或请求本身已能完整
放入原生热池时，在线路径会回退；离线模拟器也记录
`hint-resident-falls-back`，不会为了遵守提示而制造超预算或死锁。`host_backed`
仍可在未使用额外完整驻留槽时选择原生热池。

运行器新增两个显式参数：`--resident-hint-policy adaptive` 和
`--resident-min-hint-output-tokens`。前者必须与 `--future-policy deadline`、
`--resident-limit` 同时启用；后者的运行器默认值为 128，因此当前
`mixed12k_unequal` 中 12K/128 的长请求可以申请驻留，12K/64 的长请求会
退到原生热池。统计写入每 rank 的 `request_residency` telemetry，至少观察
`future_hint_claim_attempts`、`future_hint_declines`、`future_hint_missing`、
`future_hint_invalid` 和 `future_decision_reasons`。

这一步的验收目标是“占用槽的请求确实是策略希望保护的请求”，而不是仅凭驻留
请求数宣称加速。性能结论仍需和静态驻留、原生 HiSparse 在同一独占窗口配对比较。

### 5.2 驻留增长预算账本

请求槽数量小于物理 KV 容量上限。一个完整驻留请求即使当前能通过
`available_size() >= free_reserve`，其后续 decode 仍可能继续申请页面；若只看
槽位，框架会把同一份未来 headroom 承诺给多个请求。
[`residency_broker.py`](../adaptive_sparse/residency_broker.py) 因此维护一个
CPU-only、确定性的增长账本：

```text
required = fixed_reserve
         + sum(active pending growth reservations)
         + candidate predicted growth
```

候选增长优先使用 hint 的 `expected_output_tokens`；没有该字段时使用显式的
lookahead 保守值。预测和消费都按 allocator 的 `page_size` 取整：例如 128-token
页上预测当前 128 个 token 后再生成 130 个 token，共享预算必须记入 256 个新增
slot，而不是只记 130 个 token 的字节。请求跨过页边界时消费对应整页承诺；完成
或中止时释放剩余承诺。`available_size()` 已经包含实际占用的 KV，因此这里只
跟踪 allocator 尚未观察到的未来增长，不能把已消费部分再次重复计数。

运行器提供两种模式：

| 模式 | 行为 |
|---|---|
| `dry_run` | 记录是否超出共同预算，但保留原有驻留选择；用于校准预测 |
| `enforce` | 超预算时回退原生 HiSparse hot-set，并在 telemetry 中记录原因 |

两种模式都要求启用请求驻留。各 TP rank 仍使用所有 rank 的最小可用容量做决定，
账本输入和 hint 相同，因此拒绝决定保持一致。它不申请 CUDA 内存，不替代原生
allocator，也不能把预测变成硬 OOM 保证。动态 demotion 在物理提交完成后释放
对应承诺；promotion 在迁移前把“当前完整历史所需页面 + 剩余输出增长”一起计入
同一账本，提交成功后才只保留尚未发生的未来增长承诺。hint 的输出长度是请求初始
总预测，动态升级时会扣除从 staging 基线到当前长度已经生成的 token。升级迁移在
清理 host hot buffer 时只释放当前 reservation，保留 baseline，使后续
demote/promote 仍按同一生成进度扣减；只有真实请求完成或 abort 才删除 baseline。
请求完成与中止则无条件清理残留承诺。加入 prefill reservation 回归后，当前在
SGLang 0.5.17 镜像内 157 项测试通过、3 项因无 GPU 跳过；在线迁移开销、预测误差和端到端
收益仍需真实 serving 配对实验。

### 5.3 已知未来请求的 prefill 预留

`adaptive_sparse/future_prefill.py` 把 waiting queue 中的 hint 聚合为一次有界
预留。它只计算字节和建议，不分配 GPU 内存，也不会替代 `PrefillAdder`。每个
prefill boundary 都从当前 queue 重新计算；没有跨 tick 的陈旧 reservation。因此
请求离开队列或越过 `horizon` 后，旧建议自然失效。现有
`EKV_ADAPTIVE_DWELL_SECONDS` 继续负责迁移防抖。

SGLang 0.5.17 源码审计确认 `get_new_batch_prefill()` 在进入
`_get_new_batch_prefill_raw()` 后依次执行原生 priority、构造 `PrefillAdder`、
入队检查和会改变请求长度的 `add_one_req()`。因此 hook 在
`get_new_batch_prefill()` 外层执行时仍在原生排序和准入之前；它只用 hint 估算
headroom，随后仍调用原方法。当前允许的 `scheduler.py` SHA256 为
`1d4a7683…1619b`（0.5.17）和 `fc96ed4b…afe60`（MiniMax 固定开发镜像）。

单个请求的保守估计为：

```text
prompt = 未命中的 prompt token
auto  -> expected_output >= threshold ? resident : host_backed

resident kv = (prompt + expected_output) * kv_bytes_per_token
host_backed kv = min(prompt + expected_output, native_hot_tokens)
                 * kv_bytes_per_token

activation_rate = kind_override.get(kind, activation_bytes_per_token)
reservation = kv + prompt * activation_rate
```

`resident` 缺少输出长度预测时不进入预留；显式 `host_backed` 仍可按
`min(prompt, native_hot_tokens)` 预留有界热集。所有完整条目按 waiting queue
顺序装入 `max_reserve_bytes`，装不下的请求整条跳过，不记录会造成误导的部分
字节。TP 各 rank 交换 plan signature；任一 rank 不一致时所有 rank 同时回退为
零预留。

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `EKV_FUTURE_PREFILL_MODE` | `off` | `off`、`dry_run` 或 `enforce` |
| `EKV_FUTURE_PREFILL_HORIZON_MS` | `2000` | 只考虑该时间窗内将释放的请求 |
| `EKV_FUTURE_PREFILL_MAX_BYTES` | `256 MiB` | 单次聚合预留上限 |
| `EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN` | `0` | 实测校准的 prefill 激活预算 |
| `EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON` | `{}` | 按请求 `kind` 覆盖 activation bytes/token，例如 `{"tool":1536,"chat":1024}` |
| `EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS` | `128` | `auto` 选择完整驻留的输出阈值 |

`dry_run` 只写 telemetry；`enforce` 把 reservation 作为预填充边界的有效
headroom，且要求 `EKV_ADAPTIVE_BUDGET_BYTES > 0`。有效 headroom 为

```text
预测 KV bytes
+ max(固定 prefill headroom, 请求级 activation bytes)
```

固定值与请求级 activation 近似覆盖同一 scratch 空间，因此不直接相加；预测 KV
是独立分配，必须完整计入。这里的“预测性扩容”是提前触发既有迁移，为
即将发生的原生分配腾出空间，并没有在 hint 到来时直接 `cuMemCreate` 整个预测
块。activation 默认为零，必须在目标模型和 chunk 大小上标定后才能用于容量结论。

每 rank telemetry 新增 `future_prefill_ticks`、`future_prefill_reserve_bytes`、
`future_prefill_selected`、`future_prefill_effective_headroom_bytes`，以及
deferred/missing/invalid/unestimated/capped/TP-fallback 和 reservation-pressure
计数。当前只有 CPU 单元测试与 SGLang 源码接入证据；真实 GPU 上仍须比较 native、
`dry_run`、`enforce` 三组，并同时观察 TTFT、TPOT、吞吐、OOM/retraction、迁移
字节和整卡 NVML。

## 6. 状态机与失败回退

```text
HINTED -> RELEASED -> ADMITTED -> RUNNING -> DONE
   |         |           |          |
   |         |           +-> RETRACTED
   |         +-> EXPIRED
   +-> INVALID_HINT
```

`RETRACTED` 请求保留原 `rid` 的 hint，但清除 reserve；重新进入 waiting 时重新计算。

以下情况必须无副作用地退回原生顺序：

- hint 缺失、过期或校验失败；
- `release_at` 在未来超过 horizon；
- 时间估计没有置信样本；
- 任一 rank 的容量观测不一致；
- 模型或运行配置不在已验证范围；
- chunked prefill 正在占用当前 batch；
- SGLang 原生 priority、LoRA、grammar、pooling mode 或 Mamba 约束发生冲突。

规划器不能依赖排序来保证 OOM 安全。`PrefillAdder` 的 `NO_TOKEN`、decode 的
`check_decode_mem()` 和 retraction 路径仍然必须有效。

## 7. 运行范围

第一轮验证沿用当前安全范围：

- eager、无 overlap；
- 单模型、同构 full-attention KV；
- 不做 PD disaggregation；
- 无 speculation；
- chunked prefill 只允许最大的一个请求继续；
- CUDA Graph, prefix caching, HiCache 和混合 Full/SWA/Mamba 后置。

未来规划与这些功能并非永久互斥，但每增加一种内存池或调度状态，都需要重新审计
请求槽预留、峰值激活和迁移边界。

## 8. 可观测指标

每次 planner tick 至少记录：

- `tick_reason`、`horizon_ms`、`waiting_count`、`running_count`；
- `eligible_count`、`deferred_not_released`、`invalid_hints`；
- `logical_kv_bytes`、`physical_kv_bytes`、`activation_headroom_bytes`；
- `reserve_bytes`、`reserve_expired_bytes`、`reserve_hit/miss`；
- `promote/demote_requested`、`promote/demote_committed`；
- `deadline_feasible_before/after`、`native_order_changed`；
- 每请求的 `rid_hash`、`reason`、`predicted_slack_ms`。

当前 hook 可把 tick、重排、队列规模、释放/未释放数量、各类回退和 telemetry
错误写入 `EKV_FUTURE_STATS_PATH`。该文件与 VMM adapter 的 `EKV_STATS_PATH`
分开，避免两个组件相互覆盖。

可选设置 `EKV_FUTURE_ORDERING_HORIZON_MS` 后，队列中任一请求的
`release_after_ms` 超出该窗口都会让本轮排序整体回退，并累加
`fallback_horizon`。带 deadline 但缺少 `expected_output_tokens` 的请求也会
整体回退；未知工作量不再被当作 1 个输出 token 做乐观估计。该 JSON 是单个
调度进程的最后 writer 快照；多 TP rank 实验应按 rank 分目录保存后再聚合。

实验至少比较 native、EDF/SPT 排序、排序 + 驻留建议、完整执行四组；报告 TTFT p50/p95、
端到端延迟、deadline hit rate、吞吐、OOM/retraction 次数、迁移字节和浪费预留。

## 9. 测试顺序

1. 纯 CPU 单测：hint 解析、稳定排序、无 hint 回退、重复 rid、过期 hint、时钟边界；
2. 离线 trace 回放：证明相同输入产生相同顺序，并统计 oracle gap；
3. SGLang hook 单测：确认原生 `calc_priority()` 先执行，planner 只做稳定重排；
4. 容量账本单测：pending prefill、pending promotion 和 activation headroom 不可漏算；
5. A100 eager smoke：短请求、长请求、共享前缀和 deadline burst；
6. 性能 A/B：只有在输出正确、无 OOM、无 retraction 增加后才报告性能；
7. Git 提交：把设计、实现和实验分别提交，避免把未验证建议混成“已实现收益”。

离线运行示例：

```bash
python3 scripts/future_plan_compare.py \
  examples/future_planning/controlled_rollout.json \
  --output results/future-planning/controlled-rollout.json
```

该人工负载包含一个先到的长请求和 50 ms 后到达的五个短高优先级请求。CPU
事件模拟中，FIFO 发生 3 个期限 miss，加权完成时间 875.5 ms；未来感知策略
把长请求切到有界 hot-set，使五个短请求零 miss，加权完成时间降至 591.5 ms。
长请求本身的完成时间从 3501 ms 增至 4376 ms，因此这是一个延迟分配决策，
不是“所有指标同时更好”的性能结论。

## 10. 明确不做的事

- 不用预测结果绕过 `PrefillAdder`；
- 不在 CPU 排序阶段做 CUDA synchronize、mapping 或 D2H/H2D 传输；
- 不为整个未来窗口永久预留 KV；
- 不把未完成实验写成性能结论；
- 不改变应用可见的采样语义；
- 不把无 hint 流量的行为变成新策略。
