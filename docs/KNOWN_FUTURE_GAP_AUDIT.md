# 已知未来规划器缺口审计

审计日期：2026-11-09
代码基线：`feature/sglang-framework`，`9419a78776addb2035fcba4c5ce44fc9f1daefff`

## 结论

离线规划、hint 校验、请求排序和 CPU 增长账本的边界基本清楚，222 项相关
CPU 测试全部通过。2026-11-09 的加固把调度钩子改为 fail-open、补齐全空闲
TTL tick、修正多 plane 量纲并在预算动作后预热；相关 CPU 回归随后扩展到 418
项（含 131 个子测试）。其中一项使用真实 SGLang 0.5.17 `Scheduler` 类安装
包装层，并注入 VMM 预热异常，验证原生 prefill 仍能接管。`enforce` 仍是显式
opt-in，且真实 serving A/B 尚缺，
因此还不能写入默认配置或声称性能收益。初始审计确认的缺口如下：

1. 预热异常没有隔离，可能从可选的预测路径直接破坏一次正常 admission；
2. 服务完全空闲后没有周期 tick 保证 TTL 回收，预测映射可能无限期保留；
3. 多平面 VMM 的“需求字节”到“逻辑槽”换算重复计入平面数，实测会按需求的
   2 倍映射；
4. 单元测试和 CUDA 冒烟验证了显式生命周期，没有验证调度器空闲、异常回退和
   真实命中的端到端组合。

因此，`dry_run` 仍适合收集候选和校准误差；真实 CUDA 比例、同 trace serving
A/B 和处置 OOM 尚未完成前，不应把 `enforce` 设为默认做共享或对外服务，更不应
据此宣称 TTFT、吞吐或显存收益。

当前状态：上述 P0/P1 的代码路径与 CPU 回归已修复，真实 scheduler 包装层的
异常回退也有兼容回归；CUDA 命中/过量和同 trace serving A/B 仍属于发布门槛。
新的 fail-open 行为不会修复已经 poisoned 的原生
VMM backing，它只保证预测路径不再额外阻断原生 admission，并停止后续预测。

## 缺口矩阵

下表记录 2026-11-09 加固前的证据；当前修复状态见结论和验证记录。

| 优先级 | 缺口 | 证据 | 影响 |
|---|---|---|---|
| P0 | 预测性预热的分配异常会穿透 scheduler hot path | [`scheduler_budget.py`](../adaptive_sparse/scheduler_budget.py#L1293) 直接调用 `prewarm(reservation.kv_bytes)`；[`adapter.py`](../elastic_kv/adapter.py#L386) 直接调用 `arena.precommit()`，两条路径都没有失败隔离或 fallback 计数 | 可选的 hint 可能因 CUDA VMM、全局 broker 或初始化失败，让原本可由原生 `PrefillAdder` 继续处理的请求整体失败。这与“建议失效时退回原生行为”的契约不一致。 |
| P1 | TTL 不是真正的时间上限 | `expire_predictive_prewarm()` 只在调度 `run()` 开始、`admit()` 和 `admit_ranges()` 中调用；[`on_idle()`](../adaptive_sparse/scheduler_budget.py#L1337) 只 publish telemetry，不执行回收。服务没有 running/waiting batch 时通常不会再调用这些路径 | 请求被取消、队列清空或服务转闲置时，过期 lease 及其 broker/GPU backing 可能一直存活到下一次有请求，破坏“有界 TTL”的资源上界。 |
| P1 | 预热槽位换算把平面字节重复计入 | [`predictive_prewarm()`](../elastic_kv/adapter.py#L340) 用 `target_bytes / slot_bytes` 选择槽；但 `demand` 是 K/V 与全部层的总字节，而一个逻辑槽会同时映射所有 plane | 二层投影的最小样例中，4 MiB 需求映射了 8 MiB backing。层数越多，过度预热越明显，既削弱收益，也更早触发 OOM 或全局 broker 拒绝。 |
| P1 | prewarm 在容量控制器腾挪之前发生 | [`scheduler_budget.py`](../adaptive_sparse/scheduler_budget.py#L1293) 先预热，随后才调用 `controller.boundary(...)` 执行既有迁移策略 | 内存紧张时先尝试 reserve，而不是先降级 running request 后利用释放出的空间，导致可避免的分配失败或无用 lease。`enforce` 的命中率和净收益会被系统性低估。 |
| P1 | horizon 聚合按字节求和，而不是按时间重叠求峰值 | [`future_prefill.py`](../adaptive_sparse/future_prefill.py#L279) 把 horizon 中所有可估算条目的 KV 和 activation 直接相加 | 两个先后到达、互不重叠的请求仍按并发峰值计费，可能造成无必要的降级。共享前缀元数据也没有在这一层抵扣，当前实现偏保守但会污染性能结论。 |
| P1 | 真实 scheduler 接入失败回退已补回归 | [`test_scheduler_budget_sglang_integration.py`](../tests/test_scheduler_budget_sglang_integration.py) 在真实 SGLang 0.5.17 `Scheduler` 类上安装包装层，让 tracker 的 `predictive_prewarm()` 抛出异常 | 测试断言异常被记录、预测路径被失效且原生 prefill 的返回值保持不变。真实 CUDA hit/miss 和 serving A/B 仍不能由这条 CPU 回归替代。 |
| P2 | 预热 telemetry 没有失败和拒绝归因 | counters 只有 tick、candidate、mapped、hit、expired，没有 `prewarm_error`、`broker_denied`、`requested_bytes` 与物理 mapped bytes 的独立字段 | 实验里看见 `candidate_chunks` 无法区分“没有触发”“被容量拒绝”“调用失败”，也无法核对过度映射。 |
| P2 | 没有真实 serving A/B 证据 | 设计文档已明确只有 CPU 与显式 CUDA 冒烟；没有 native、forecast-disable、`dry_run`、`enforce` 的同 trace 配对结果 | 不能证明预测性扩容改善 TTFT、减少 OOM 或提高吞吐，也不能排除迁移抖动和 broker 争用造成的负收益。 |

## 可复现证据

### 多平面过度映射

隔离测试使用一个 `planes=2` 的 Dense VMM arena，先确认输入需求为 4 MiB，再调用
`predictive_prewarm(4 MiB)`。结果如下：

```json
{
  "demand_bytes": 4194304,
  "selected_logical_chunks": 2,
  "physical_backing_growth_bytes": 8388608,
  "overmapping_ratio": 2.0,
  "planes": 2
}
```

每个逻辑 chunk 会同时提交所有 K/V plane，因此需求字节应先除以 plane 数，才能
换算为每个 plane 的地址空间槽数。当前 CUDA 冒烟中的
`predictive_prewarm(2 * SLOT)` 还断言会选中 2 个 chunk；这恰好把错误的单位当成了
预期契约。

### 失败没有回退

调度路径是：

```text
Scheduler wrapper
  -> tracker.predictive_prewarm(reservation.kv_bytes)
  -> arena.precommit(missing)
  -> Backing.precommit()
```

`Backing.precommit()` 可因全局 broker 拒绝或 CUDA VMM 错误抛出异常；调用栈中没有
`try/except`、降级计数或“回到原生 admission”的分支。异常发生在外层
`get_new_batch_prefill()` 调用原生 scheduler 之前，因此预测失败会阻止原生路径。

### TTL 只在下一次活动时推进

仓库中 `expire_predictive_prewarm()` 的调用点只有：

- `run()` 开始的 prefill/decode 边界；
- `admit()`；
- `admit_ranges()`；
- `predictive_prewarm()` 在 `enforce` 下的下一次重算。

`on_idle()` 没有调用它，也没有后台 timer。当前接线只能保证“下一次有调度活动时
会清理”，不能保证 wall-clock TTL。

## 必须修改

1. 把 `expire_predictive_prewarm()` 和 `predictive_prewarm()` 在 scheduler hook
   内做成 best-effort 契约。容量拒绝必须无副作用地返回；任何已发生的物理状态
   不确定都要明确使该路径失效并记录原因，不能让 hint 失败替代原生 admission。
2. 在 `on_idle()` 或一个受控的 period tick 中执行 TTL 回收；增加测试证明 queue
   删除、abort 和完全空闲三种路径都会释放预测 backing 与 broker lease。
3. 用“总需求字节 / plane 总数”计算逻辑覆盖范围，或直接以每个 plane 的 token
   stride 作为预热量纲；cap、selected chunks 与 mapped bytes 必须分别记录单位。
4. 先让 `SchedulerBudget.boundary()` 完成当前预算动作，再在仍有安全余量时尝试
   prewarm；不要把预测分配放在能够腾挪空间的控制动作之前。
5. 为 horizon reservation 增加按 release/start/finish 区间求并发峰值，并复用同一
   prefix group 的抵扣规则。保守模式可以保留，但必须作为显式策略，而不是默认
   把所有条目当成同时活跃。
6. telemetry 至少增加 `prewarm_requested_bytes`、`prewarm_candidate_bytes`、
   `prewarm_mapped_bytes`、`prewarm_broker_denied`、`prewarm_exception`，并分别记录
   dry-run、成功 map、native hit、TTL expire。

## 必需回归

| 测试 | 断言 |
|---|---|
| 多平面需求换算 | 2/8 个 plane、不同 `slot_bytes` 下，逻辑覆盖等价于总需求，物理增长不超过一个 chunk 粒度的上界 |
| scheduler 中 broker denial | `prewarm` 返回 fallback，原生 prefill 仍被调用，不抛出到请求层 |
| scheduler 中 VMM exception | 记录错误并使预测租约失效；测试明确验证不会把预测失败伪装成成功 |
| 空闲 TTL | 创建 lease 后删除 waiting request、停止 running batch，推进时钟，断言 backing 和 broker 回到初始值 |
| 时间重叠账本 | 两个不重叠的未来请求不按并发相加；真正重叠时按峰值相加 |
| 共享前缀 | 同一 group 同时活跃时只计一次共享前缀，超出成员仍各自计费 |
| CUDA hit/miss | 冒烟分别制造真实命中和过期 miss，核对 mapped、hit、expired bytes 与原生 `ev_backed` |
| serving A/B | 同一独占 GPU、同一 trace，至少比较 forecast-off、`dry_run`、`enforce`；只有正确性、OOM 和无额外 retraction 同时通过后才报告延迟与吞吐 |

## 验证记录

- 加固前相关 CPU 测试：222 passed，84 subtests passed。
- 加固后完整 CPU 回归：416 passed，3 skipped，131 subtests passed。
- 加固前隔离多平面核算：4 MiB demand -> 8 MiB backing，比例 2.0。
- 加固后需求按 `slot_bytes *` 全部 arena 的 plane 数换算逻辑槽；物理字节遥测
  单独记录 requested、candidate 和 mapped bytes。
- SGLang 0.5.17 真实 scheduler wrapper 异常注入回归通过；完整容器套件为
  `418 passed, 2 skipped, 131 subtests passed`。
- CUDA hit/miss 和同 trace serving A/B 仍未运行；本审计只
  确认代码契约、CPU 策略与资源记账，不把 CPU 证据外推成真实服务收益或正确性
  结论。
