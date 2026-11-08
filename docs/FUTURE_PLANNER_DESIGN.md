# 已知未来请求规划器设计

本文面向当前 `feature/sglang-framework` 分支，定义“已知未来请求”如何影响
SGLang 的准入顺序和 VMM 驻留决策。第一阶段以正确、可离线验证为目标；
GPU 结果出来前，不把规划收益写成已证实的性能收益。

当前实现状态：`adaptive_sparse/future_plan.py` 已提供不依赖 SGLang/CUDA 的
确定性事件模拟，比较 `fifo`、`longest`、`mode-aware` 和 `future` 四种策略；
`adaptive_sparse/sglang_future_hook.py` 已实现默认关闭的 scheduler 排序 hook。
该 hook 只在支持 `kv_hints` 的 SGLang revision 上启用，并保持 `PrefillAdder`
为最终准入权威。它尚未在带 `kv_hints` 的上游 revision 上完成真实服务 A/B，
也不能把离线结果当成服务端吞吐或延迟结论。

## 1. 要解决的问题

输入不是传统的未知到达流量，而是一段可提前看到的请求集合。每个请求至少可提供：

- 预计开始时间或相对释放时间；
- 截止时间；
- 实际 prompt 长度；
- 预计输出 token 数；
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

建议新增 `adaptive_sparse/future_plan.py`，只包含纯 Python 决策；GPU 与 SGLang
适配放在独立的 `install_future_planner()` 中。

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

当前 hook 可把 tick、重排、队列规模、释放/未释放数量、三类回退和 telemetry
错误写入 `EKV_FUTURE_STATS_PATH`。该文件与 VMM adapter 的 `EKV_STATS_PATH`
分开，避免两个组件相互覆盖。

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
