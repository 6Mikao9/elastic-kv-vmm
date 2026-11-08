# 已知未来请求规划器设计

本文面向当前 `feature/sglang-framework` 分支，定义“已知未来请求”如何影响
SGLang 的准入顺序和 VMM 驻留决策。第一阶段以正确、可离线验证为目标；
GPU 结果出来前，不把规划收益写成已证实的性能收益。

当前实现状态：`adaptive_sparse/future_plan.py` 已提供不依赖 SGLang/CUDA 的
确定性事件模拟，比较 `fifo`、`longest`、`mode-aware`、`future` 和有界
`oracle` 五种策略；
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
  "action_version": "2",
  "payload": {
    "release_after_ms": 250,
    "deadline_after_ms": 1800,
    "deadline_clock": "perf_counter",
    "deadline_absolute_ms": 9123456800,
    "expected_output_tokens": 192,
    "kind": "tool",
    "residence": "auto",
    "weight": 1.0
  }
}
```

action version 2 在 HTTP 实际发送时同时写入 `deadline_after_ms` 和同机
`perf_counter` 时钟域下的 `deadline_absolute_ms`。scheduler 接收请求后使用绝对
值计算剩余时间，因此网络、tokenization 和跨进程排队不会被误算成额外 slack。
旧 action version 1 仍按 scheduler 相对时间解释，以保持兼容。执行器默认发送
version 2；启用实测服务时间反馈后，取得样本的后续请求升级为 version 3，并
额外携带 `prefill_tokens_per_s` 与 `decode_tokens_per_s`。该绝对时钟假定
orchestrator 与 SGLang 位于同一主机。跨机部署需要
改用经过校准的 Unix clock 或显式时钟偏移。所有数字都要做有限值、非负值和上限
校验。`kv_hints` 是受信任 orchestrator 的内部提示通道，不应当成任意应用客户端
的安全控制接口。

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

服务时间使用可观测的闭环 EWMA 估计：

```text
prefill_time = prompt_tokens / prefill_tokens_per_s
decode_time  = expected_output_tokens / decode_tokens_per_s
```

`adaptive_sparse/service_time.py` 从已经写回 rollout 结果的 `ttft_s` 与 `tpot_s`
更新两条独立状态：

```text
observed_prefill_rate = prompt_tokens / ttft_s
observed_decode_rate  = 1 / tpot_s
```

每个方向分别使用 EWMA，默认 `alpha=0.2`。首个有效样本之前不伪造测量值，
调度器继续使用保守的 `8,000 token/s` prefill 与 `20,000 token/s` decode 默认值；
取得样本后，执行器把两项估计写入新请求的 `elastic.future` action v3。action
v1/v2 的载荷、排序和兼容行为保持不变，只有 v3 可以携带
`prefill_tokens_per_s` 与 `decode_tokens_per_s`。

速率限制在 `[0.001, 1e9] token/s`。异常高/低样本会被截断并分别记录；非有限、
非正延迟或非正 token 数不会进入 EWMA，而是计入 `rejected_samples`。TTFT 包含
请求在真实服务端的排队和调度时间，因此这里得到的是“有效服务速率”，不是
GPU prefill 峰值算力。实验启用入口为 `--service-time-feedback`
（MiniMax runner 为 `--future-service-time-feedback`），结果摘要会输出两条
EWMA、样本数、截断数和拒绝数。该反馈默认关闭；开启后由执行器在已完成的
in-flight 请求被回收后，为后续派发请求附带最新速率。
每个结果行同时记录 `service_time_prefill_tokens_per_s_hint` 与
`service_time_decode_tokens_per_s_hint`，从而可以核对某个请求实际收到的是
冷启动默认值还是某次 EWMA 快照。

没有足够样本、样本持续越界或 TP 侧观测不一致时仍应使用宽裕默认值，不能仅凭
短窗口 EWMA 声称 deadline 可满足。真实收益必须比较关闭/开启反馈的两组相同
GPU trace。

### 4.0 已知未来的有界 oracle

`oracle` 用于离线研究和在线调度器的上界参照。它在每个可决策边界枚举：

- 已到达请求及其可行的 `resident` / `hi-sparse` 驻留模式；
- 为等待 active 请求释放预算或保护即将到达的硬期限请求而主动 idle；
- 下一次到达、prefill 完成和 active 完成等事件推进。

目标按字典序比较：期限 miss 数、加权完成时间、makespan、HiSparse 请求数，
最后按 request ID 和开始时间确定稳定平局。完全相同的请求在资源与期限字段上
属于对称节点；只展开其中 ID 最小的一个，能够去掉工具批次的阶乘重复，同时
保持上述确定性平局结果。

搜索受 `oracle_max_expansions` 限制。实现先把现有 `future` 计划作为 incumbent，
因此达到上限或搜索失败时不会产生比启发式更差的计划；此时输出
`search_truncated=true`、`search_exact=false`、`fallback_policy="future"`。
只有搜索完整且产生了搜索 incumbent 时才报告 `search_exact=true`。这套 bounded
oracle 只验证离线决策质量，不会在 scheduler 热路径运行，也不能绕过
`PrefillAdder`。请求的 `priority` 是历史 JSON 字段名，表示离线目标中的软业务
权重；`plan_rollout()` 只把它写成 `FutureHint.weight`，不会赋值给 SGLang 的
`request.priority`，因此原生硬优先级层级仍由 SGLang 所有。

### 4.0.1 已实现：共享前缀的规划容量抵扣

`prefix_group` 与 `shared_prefix_tokens` 现在进入规划器和运行时 hint。对于同一
prefix group 的多个同时驻留请求，规划器只计算一份不可变共享前缀 KV：

```text
resident aggregate
  = sum(每请求完整或 HiSparse hot-set)
  - sum_group((resident成员数 - 1) * 组内最小 shared_prefix_tokens
              * kv_bytes_per_token)
```

只有 `resident` 成员参与抵扣；HiSparse 的 Top-K 选择不保证包含前缀，因此仍按
完整有界热集计费。同一 group 内不同请求若声明不同前缀长度，规划入口直接拒绝，
避免把不同内容误算成共享。已知未来准入会同时抵扣多个 imminent resident 请求
之间的共享前缀，防止预留空间被重复计算。

运行时 demotion 排序进一步区分“带前缀元数据”和“确有消费者”。只有
`EKV_SHARED_PREFIX_REUSE=direct` 时，控制器才把以下证据计入 donor 的驻留价值：
其他活跃 host-backed 请求通过完整 fingerprint 精确解析到该 donor；或 waiting
queue 中存在同一 `prefix_group` 和相同 `shared_prefix_tokens` 的潜在 host-backed /
auto 请求。后者只利用轻量几何匹配作为排序预测，不声称 Top-K 一定命中，也不抵扣
规划预算；真正复用仍必须通过 token、adapter 与前缀 fingerprint 校验。因此一个请求
单独携带 `shared_prefix_tokens` 不再自动获得高驻留分，没有消费者时保持中性。
预测只覆盖已进入 SGLang waiting queue 的请求；尚未提交到服务端的远期计划仍需后续
控制面通道，不能从当前队列状态推断。

`elastic.future` payload 已能无损携带 `prefix_group` 和
`shared_prefix_tokens`，但当前 SGLang 实跑范围仍关闭 radix cache，也没有把
第二个请求直接绑定到第一个请求的完整驻留 KV。因此这一步已经完成的是确定性容量
模型、计划输出、hint 校验与传输；真实共享 backing 或 Top-K 跨请求命中仍需单独
实现引用、写保护、模型/权重版本和 lifetime 校验，不能把规划节省直接写成已测
运行时显存收益。

当前回归 fixture 使用两条各 2,000-token 的请求、1,500-token 相同前缀和
2,600-token 预算。启用共享后两条请求都能 resident，规划峰值 2,500 token，
共享抵扣 1,500 token；去掉共享元数据后，模拟器把第二条降为 200-token
HiSparse hot-set，峰值反而只有 2,200 token。固定速率下，前者模拟 makespan
为 254 ms，后者为 1,004 ms。这个例子说明共享前缀用少量额外驻留字节换取了更少
的 HiSparse 模式，也说明不能只比较峰值显存；它是确定性策略回放，不是模型实测。

### 4.1 受控 rollout 的可执行计划

`plan_rollout()` 把离线策略结果转换成 orchestrator 可执行的动作。每个 action
包含原始与计划到达时间、预计完成时间、请求类型，以及可直接放入
`elastic.future` payload 的字段。控制器在 `submit_at_ms` 才实际提交请求；
hint 使用 `release_after_ms=0`，并把绝对 deadline 换算成相对 scheduler
接收时刻的 `deadline_after_ms`。

```bash
python3 scripts/future_plan_compare.py \
  benchmarks/workloads/known-future-rollout.json \
  --policies fifo future oracle \
  --emit-policy oracle \
  --output results/future-planning/oracle-comparison.json
```

`adaptive_sparse/rollout_executor.py` 是这个计划与真实 SGLang HTTP 服务之间的
执行边界。它按 `submit_at_ms` 单调唤醒，发送确定性的 tokenized prompt，并把
同一份计划任务包成 `kv_hints`。每条执行记录同时保存计划/实际提交与完成时间、
提交抖动、deadline 目标、deadline 命中、TTFT、TPOT 和输出 token 数。计划中的
`deadline_ms` 保留计划器时间线中的原始期限；每次实际发送前，runtime hint 的
`deadline_after_ms` 会按本次真实提交时刻重新计算，同时写入同机
`perf_counter` 绝对 deadline。旧版本可以使用相对字段，新 scheduler 优先使用
绝对字段并分别验证两者的一致性。

执行器在发出任何 HTTP 请求之前拒绝以下不一致：重复 ID、到达时间晚于提交时间、
预估完成早于提交、hint 的 `kind` 或输出长度与 submission 不同、控制器填充了
非零 `release_after_ms`、相对 deadline 与原始绝对 deadline 不一致，以及
`max_parallel=0`。实际请求启动超过允许抖动、响应 token 数不足或 TP rank 对同一
请求的驻留结论不一致，都会使整轮实验失败。应用 deadline miss 则作为结果保留，
不会伪装成传输失败。

吞吐使用“最早计划到达至最晚实际完成”的 makespan，包含策略刻意引入的空转；
另记录从最早实际提交开始的服务窗口吞吐，便于区分调度收益和请求处理速度。

执行器还提供一个可选的在线准入层。`--max-inflight` 用真实的尚未完成请求数
限制进入服务的请求；`--queue-policy deadline` 会在同一时刻已到达的竞争者中先
选择绝对 deadline 最紧且剩余 slack 最小的请求。计划时间在这种模式下是“最早
可提交时间”，实际容量等待会写入 `capacity_delay_ms`，不会伪装成网络提交抖动。

`--elastic-residency-budget` 从计划中的 `planner_config` 恢复每 token KV 字节数、
热 token 数和总预算。执行器维护当前在飞请求的预测字节账本：低负载且完整 KV
能装入预算时保留 planner 选择的 resident；高负载下只把尚未提交的长请求降为
`host_backed`，已经在飞请求不迁移。每个请求记录 `planned_residence`、
`runtime_residence`、`elastic_admission_reason`、准入前后字节数和是否发生预算
超限。若连热集也装不下，控制器默认等待现有请求完成；硬 deadline 已经等不起时
才允许按配置超限提交。该账本用于控制器准入，SGLang 的原生容量检查和请求级
驻留 broker 仍然是最终安全门。

real-serving 对照必须使用不同输出目录交错运行同一请求集合：

```bash
python3 scripts/run_future_smoke.py \
  --model-path /model \
  --output-dir results/future-planning/fifo-no-hints \
  --workload benchmarks/workloads/known-future-rollout.json \
  --planning-policy fifo --future-policy off --no-hints

python3 scripts/run_future_smoke.py \
  --model-path /model \
  --output-dir results/future-planning/future-hints \
  --workload benchmarks/workloads/known-future-rollout.json \
  --planning-policy future --future-policy deadline \
  --max-inflight 4 --queue-policy deadline --elastic-residency-budget
```

对 MiniMax HiSparse 运行，额外传入 `--residency-stats-dir` 和
`--require-residency-evidence`。执行器会读取所有 `tp-*.json`，逐个请求核对
`planned_residence` 与 `actual_residence`，并要求各 TP rank 对实际模式一致。
只有这一证据存在时，才能把“计划让哪些请求驻留”与端到端性能一起分析。

`run_hisparse_smoke.py` 也已接入同一个执行器。计划模式会自动读取
`vmm_stats/tp-*.json`，并在 VMM 和请求驻留同时启用时要求完整的逐请求、逐 rank
证据后方可完成：

```bash
bash scripts/container_hisparse_smoke.sh \
  0,2,3,4 /path/to/MiniMax-M3.1-Flash native \
  results/hisparse-plan/future-hints \
  --sm80-bf16-projections --vmm-backing --resident-limit 2 \
  --resident-free-reserve 4096 --resident-hint-policy adaptive \
  --future-workload benchmarks/workloads/known-future-rollout.json \
  --future-planning-policy future --future-policy deadline \
  --future-max-inflight 4 --future-queue-policy deadline \
  --future-elastic-residency-budget
```

不带 hint 的对照使用 `--future-policy off --future-no-hints`；规划时间轴仍按同一
workload 生成，但服务端执行原生排队。两次运行必须写到不同目录。

当有限 horizon 内的高价值请求即将到来、当前长 prefill 会让它错过硬期限，而
短暂空转能保住期限时，`future` 策略允许 idle 到该请求到达。这里控制的是
orchestrator 的真实提交时刻；runtime hook 中的 `release_after_ms` 仍只是软排序
提示，不能被误写成 `PrefillAdder` 的硬准入门槛。

仓库内合成样例只用于回归 harness。该样例中 FIFO、mode-aware、future 分别记录
3、2、0 个 deadline miss，future 的模拟 makespan 为 1,988 ms，峰值预算为
4,008,192 bytes。默认 100,000 次展开的有界 oracle 在该七请求 trace 上触发上限，
并按契约保留 future incumbent；这不表示真实服务应在线运行 oracle。构造性小
样例会在单元测试中证明 oracle 能找到 FIFO 漏掉的期限可行方案，并验证主动
idle 和两种驻留模式都会被搜索。以上都是固定输入上的策略行为证据，不是模型、
GPU 或生产负载上的性能结论；真实效果必须用同一模型、权重、请求体和 TP 配置
重新测量。

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
请求完成与中止则无条件清理残留承诺。加入 prefill reservation 和 0.5.17
IPC transport 回归后，当前在 SGLang 0.5.17 镜像内 167 项测试通过、2 项因无
GPU 跳过；在线迁移开销、预测误差和端到端
收益仍需真实 serving 配对实验。

### 5.3 已知未来请求的 prefill 预留

`adaptive_sparse/future_prefill.py` 把 waiting queue 中的 hint 聚合为一次有界
预留。它只计算字节和建议，不分配 GPU 内存，也不会替代 `PrefillAdder`。每个
prefill boundary 都从当前 queue 重新计算；decode 则在每
`EKV_ADAPTIVE_OBSERVE_STEPS` 个真实运行步刷新一次，使即将到来的 prefill 不用
等到主 KV 已经越线才触发降级。每次刷新都替换上一份建议；请求离开队列或越过
`horizon` 后，旧建议自然失效。现有 `EKV_ADAPTIVE_DWELL_SECONDS` 继续负责迁移
防抖。

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
字节。TP 各 rank 交换 plan signature；signature 分别包含每个条目的 KV 与
activation 字节，不仅比较两者之和，因为有效 headroom 对这两类字节采用不同
合并规则。任一 rank 的拆分或总量不一致时，所有 rank 同时回退为零预留。

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `EKV_FUTURE_PREFILL_MODE` | `off` | `off`、`dry_run` 或 `enforce` |
| `EKV_FUTURE_PREFILL_HORIZON_MS` | `2000` | 只考虑该时间窗内将释放的请求 |
| `EKV_FUTURE_PREFILL_MAX_BYTES` | `256 MiB` | 单次聚合预留上限 |
| `EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN` | `0` | 实测校准的 prefill 激活预算 |
| `EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON` | `{}` | 按请求 `kind` 覆盖 activation bytes/token，例如 `{"tool":1536,"chat":1024}` |
| `EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS` | `128` | `auto` 选择完整驻留的输出阈值 |

`dry_run` 只写 telemetry；`enforce` 把 reservation 作为 prefill 与周期性
decode 边界的有效 headroom，且要求 `EKV_ADAPTIVE_BUDGET_BYTES > 0`。prefill
边界的有效 headroom 为

```text
预测 KV bytes
+ max(固定 prefill headroom, 请求级 activation bytes)
```

固定值与请求级 activation 近似覆盖同一 scratch 空间，因此不直接相加；预测 KV
是独立分配，必须完整计入。这里的“预测性扩容”是提前触发既有迁移，为
即将发生的原生分配腾出空间，并没有在 hint 到来时直接 `cuMemCreate` 整个预测
块。activation 默认为零，必须在目标模型和 chunk 大小上标定后才能用于容量结论。

decode 只使用 reservation 产生的 KV 与 activation 余量；固定 prefill scratch
headroom 不会被永久加到 decode 高水位。每 rank telemetry 记录
`future_plan_ticks`、`last_future_plan_boundary`、`future_prefill_ticks`、
`future_prefill_reserve_bytes`、`future_prefill_selected`、
`future_prefill_effective_headroom_bytes`，以及 decode/prefill pressure、
deferred/missing/invalid/unestimated/capped/TP-fallback 计数。当前只有 CPU
单元测试与 SGLang 源码接入证据；真实 GPU 上仍须比较 native、`dry_run`、
`enforce` 三组，并同时观察 TTFT、TPOT、吞吐、OOM/retraction、迁移字节和整卡
NVML。

### 5.4 请求准入的最短期未来预留

滚动执行器现在会在决定一个请求能否占用完整 KV 前，查看同一已知 plan 中有限
horizon 内的后续请求。预留只覆盖同时满足以下条件的请求：

1. 可能在该请求预计完成前启动；
2. 位于 `lookahead_ms` 内；
3. 带 deadline 且等待当前请求可能耗尽可行 slack，或 soft business weight
   严格高于当前请求。

默认 `lookahead_ms=0`，因此不传未来队列时行为与原来的实时 active-byte 准入
相同。启用后最多计入 `max_future_reservations` 条近期请求；未来请求若是短请求
或显式 `host_backed`，只预留其有界 HiSparse hot-set，不会为所有未来请求永久
映射完整 KV。deadline 在等待当前请求结束后仍有正 slack 时，仅 deadline 信号不
触发当前请求降级；更高业务权重仍独立保护该未来请求。估计即使立即启动也无法在
期限内完成的请求不会以 deadline 为由挤占当前请求，但高权重原因仍可触发预留。

预留按当前请求存活窗口内各未来请求的实际到达区间取**并发峰值**，不把 horizon
内先后出现、互不重叠的请求字节直接相加。若请求因排队已越过原预计完成时刻，仍
从其实际可启动时刻重新计算一个预测服务时长，避免生成 `finish < start` 的倒挂
区间。共享前缀只对同一时刻共同活跃的 resident 请求抵扣一次。

当前请求仍在预算内时，若 `active + current + imminent_future` 超限，它会在
请求级选择上退为 `host_backed`，原因记为 `imminent-future-demotion`。当前请求
自身的 prefill workspace 使峰值越界时记为 `prefill-headroom-demotion`；未来请求的
KV 与 prefill workspace 共同使峰值越界时记为
`imminent-future-prefill-demotion`。这个决定
只发生在请求准入，随后的整个请求生命周期保持稳定；运行中的动态迁移继续由
`SchedulerBudget` 的 dwell、low-water stable interval、joint demotion、每轮
字节上限、可选的迁移时间额度和 in-flight 限制防抖。时间额度通过近期成功事务
的 demotion/promotion EWMA 带宽换算；没有真实样本时只使用配置的保守启动带宽，
各 rank 仍经 collective 选择公共候选。rollout 结果逐请求记录
`elastic_future_reserved_bytes`、`elastic_future_prefill_bytes`、
`elastic_future_reserved_requests`、
`elastic_future_effective_after_bytes`、`elastic_future_over_budget`、
`elastic_prefill_workspace_bytes`、`elastic_prefill_peak_bytes`、
`elastic_future_effective_peak_bytes`、`elastic_prefill_over_budget` 和
`elastic_future_peak_over_budget`。
其中 `future_reserved_bytes` 只记录未来持久 KV，未来 activation workspace 单独记录为
`future_prefill_bytes`。`future_effective_after_bytes` 是保留兼容的“当前驻留 KV +
未来持久 KV”字段，
但不含当前请求自己的 prefill workspace；`future_effective_peak_bytes` 才是把
当前 prefill 也加入后的完整容量判定账本。`future_over_budget` 与
`prefill_over_budget` 分别保留未来预留和当前 prefill 的归因，
`future_peak_over_budget` 才汇总两者。执行器在无法观察 SGLang 精确 prefill
完成边界时，会把每个在途请求的 workspace 保守计到请求结束，避免多个并发
prefill 漏记。汇总记录
`future_reservation_admissions`、`future_reservation_demotions` 与
`prefill_headroom_demotions`。

这是已知未来控制面的容量策略，不是对未知线上请求的预测。共享前缀或跨请求
共享物理 chunk 仍应使用实际 owner index 抵扣；当前 CPU 测试只验证确定性预算
和接入，不能据此宣称端到端吞吐、TTFT 或 OOM 收益。

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

当比较集合包含 `oracle` 时，每个策略的 JSON 现在还包含 `oracle_gap`：
加权完成时间、makespan、deadline miss 和峰值显存的绝对差及百分比，并标明
oracle 是否在 expansion cap 前完成精确搜索。`oracle` 的目标顺序首先是 deadline
miss，其次加权完成时间，再其次 makespan；因此某个策略的 makespan 差值可以为负，
不能把所有负值都解释成脚本错误。反过来，也不应因为 `future` 在一组 trace 上
接近 oracle 就宣称它在所有 rollout 上最优。

## 10. 明确不做的事

- 不用预测结果绕过 `PrefillAdder`；
- 不在 CPU 排序阶段做 CUDA synchronize、mapping 或 D2H/H2D 传输；
- 不为整个未来窗口永久预留 KV；
- 不把未完成实验写成性能结论；
- 不改变应用可见的采样语义；
- 不把无 hint 流量的行为变成新策略。
