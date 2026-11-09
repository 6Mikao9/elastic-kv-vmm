# 请求级 HiSparse 与 VMM 弹性显存研究报告

2026-11-09 的代码审查、共享前缀换入修正与 profiling 接入见[本轮进展](HISPARSE_PROGRESS_20261109.md)。新增完整模型性能结果仍待四卡实验窗口；下表中的历史数据未改写。

本阶段在 MiniMax-M3.1-Flash 的真实稀疏注意力推理路径中，实现了请求级的完整 GPU 驻留、固定热池 HiSparse，以及运行中的双向切换。切换保留模型原有的 Top-K 选择和注意力计算。VMM 负责按实际使用映射物理显存，并在无存活引用后归还给驱动。

目前最有研究价值的实证是**逻辑请求与物理回收单位之间的冲突**：两个长请求分别降级时，都可能释放不了一个完整 VMM chunk；联合规划保留页和释放页后，可以回收真实物理显存。自动降级与恢复已经通过完整历史 K/V 的逐位校验。端到端速度是否受益，取决于省下的 HiSparse 换入成本是否足以抵消新增寻址、预算观察和迁移成本，不能从回收字节数直接推导。

## 1. 研究问题和实现范围

原生 HiSparse 通过 CPU 历史 KV 和固定 GPU 热池支持较大的逻辑上下文。短请求能装入热池时已有快路径；长请求即使暂时没有显存压力，也仍受热池容量约束。本文研究在显存充足时让部分长请求保留完整 GPU 历史，在压力出现时把活跃请求转换为原生 HiSparse，再在压力下降后恢复。

“完整驻留”指历史 KV 的存放位置，**不表示改用 Full Attention**。完整驻留与 HiSparse 都执行模型原有的稀疏 Top-K 算法。热池容量 T 也不同于每步选中的 Top-K 数量：本实验选择 16 个 128-token 块，最多包含 2,048 个位置，短请求与尾块可能更少；T 为 4,096 token，另有一页 128 token 的最新写入保留区。

本阶段仍保留 SGLang 的软件页、请求表和逻辑到设备槽位的映射。取消软件分页属于下一阶段；先验证驻留策略，可以避免同时修改注意力算法、寻址接口和显存回收机制。

| 项目 | 当前覆盖 |
|---|---|
| 弹性物理 backing | 稀疏注意力的主 K/V 平面 |
| 请求状态 | 全 GPU 驻留、CPU 历史加固定 GPU 热池 |
| 触发 | 主 K/V backing 高水位、低水位稳定窗口 |
| 迁移 | 安全调度边界下的 prepare、TP 投票、commit |
| 防抖 | 高低水位间隔、驻留时间、全局冷却、低水位稳定时间、单次传输上限 |
| 未纳入统一预算的部分 | 权重、indexer KV、其他 KV、PyTorch 激活与 workspace、Graph private pool |
| 执行边界 | TP4，eager，关闭 overlap、CUDA Graph 和 radix cache |

## 2. VMM 的作用与显存口径

`cuMemAddressReserve` 只预留虚拟地址范围；真正占用物理显存的是 `cuMemCreate` 创建的存储，以及后续映射。释放时必须完成 GPU 使用、解除映射并释放物理 handle。本项目没有把未使用的物理 handle 留在隐藏缓存中。[CUDA VMM 文档](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)

因此，预留一个很大的虚拟 KV 空间，并不等于启动时占满同样大小的物理显存。但内存至少要按驱动支持的粒度提交；仍有数据占用的 chunk 不能部分释放。本实现还保留 sink 的映射，所以“没有活跃请求”也不等于主 K/V 物理占用为零。

实验必须区分以下口径。

| 口径 | 含义 | 能回答的问题 |
|---|---|---|
| 逻辑容量 | 请求可使用的 token 槽位上限 | 是否在相同上下文容量下比较 |
| VMM committed backing | 已映射的主 K/V 物理字节 | 机制实际向驱动申请、归还了多少 |
| 可回收 slack | 已映射但无存活页引用的 chunk | 是否只是尚未执行回收 |
| PyTorch allocated/reserved | PyTorch 的活跃分配与缓存池 | 其他分配是否抵消了 VMM 节约 |
| NVML used | 整卡显存占用，包含其他运行时开销 | 同卡其他工作能实际得到多少空间 |

这里的 800 MiB 高水位是主 K/V 的**软目标**，不是整卡硬上限。调度观察前可以出现 960 MiB 峰值，候选迁移也可能因共享 chunk、临时目标空间或传输上限而无法执行。prepare 失败可以撤销临时资源，但当前机制尚未提供 OOM 后恢复调度保证、跨实例强制抢占或保证成功的 admission control。

## 3. 接入 SGLang 时修改了什么

适配器在隔离的 launcher 中安装带源码校验的 hooks，不修改模型文件或原有 SGLang 安装。主要接入点如下，详细数据流见[架构说明](HISPARSE_ARCHITECTURE.md)。

| 位置 | 修改目的 |
|---|---|
| MiniMax 主 K/V pool 创建 | 用稳定 VA 的 BF16 张量视图替代一次性物理预分配 |
| 物理页分配与释放 | 在写入前提交对应 chunk，维护引用，识别可归还的 backing |
| HiSparse staging/coordinator | 让完整驻留请求保留原 prefill KV，原生请求保持 host-backed 流程 |
| 稀疏 decode 的 swap-in 入口 | 在一个 batch 中同时处理驻留行与原生 HiSparse 行 |
| 驻留 Top-K 解析 | 将原始选中块转换为已驻留的物理位置，保持选择语义不变 |
| Scheduler 的 prefill/decode 安全边界 | 观察字节预算、选择候选并发起迁移 |
| 迁移事务 | 备份完整历史、转换热池所有权、释放页；或恢复历史后发布完整驻留状态 |

降级先将完整历史 K/V 按逻辑顺序拷贝至 pinned host memory，再选出保留热页，建立原生 HiSparse 元数据，最后释放其他页。提升先分配完整目标、恢复 CPU 历史，并处理可能仍位于设备保留槽的最新 token。源状态保持存活，直到目标准备好。TP ranks 对准备状态投票；commit 失败采用 fail-stop，不继续生成可能不一致的结果。

这段实现依赖 eager 调度暂停期间的安全边界。某 rank 在 collective 前发生异常仍可能让其他 rank 等待 watchdog；它不是具有故障恢复能力的生产级分布式内存系统。

## 4. 物理 chunk 共享决定能否回收

当前布局有 60 层、每层 K/V 两个平面。TP4 下每个平面的 token 行为 256 字节；软件页包含 128 token，即 32 KiB；VMM chunk 为 2 MiB，因此一个平面的 chunk 可以覆盖 64 个软件页。即使单个请求不需要软件页中的大部分数据，只要同一 chunk 还被其他请求使用，驱动就不能收回它。

在记录下来的双请求布局中，每个长请求有 97 个物理页；降级保留 33 页、释放 64 页。分别计算两个请求的候选，实际可回收 chunk 都为零。将它们的释放集合合并后，120 个平面 chunk 同时变空，实际回收 **240 MiB/rank**。[布局快照](../results/vmm-pair12k-shadow/adaptive_stats/plans-rank00.json)

| 候选 | 原有软件页 | 保留软件页 | 释放软件页 | 实际回收 |
|---|---:|---:|---:|---:|
| 请求 A | 97 | 33 | 64 | 0 |
| 请求 B | 97 | 33 | 64 | 0 |
| A 与 B 联合 | 194 | 66 | 128 | 120 chunks，240 MiB/rank |

两个实现选择帮助解决这个问题。首先，保留页规划优先集中到共享或已经必须保留的 chunk，减少新保护的 chunk。其次，当所有单请求候选都无法释放物理空间时，策略可以检查小规模请求组合的释放并集。当前控制面允许 2–4 个请求，以 256 个组合为默认上限，并要求同一组合在每个 TP rank 都能实际释放整块；候选和传输仍受 dwell、byte cap 与防抖约束。三请求共享 chunk 的 CPU 回归已经覆盖，GPU 上的迁移成本、组合规模收益与端到端加速仍需在预算窗口内配对实测。

这个例子说明，按请求长度排序只能估计逻辑 KV 的大小；面向实际回收，需要同时考虑 chunk 所有权、保留页选择和迁移成本。它为进一步研究稀疏注意力的显存管理提供了具体问题，而不只是把原生热池换成 VMM。

## 5. 已做的优化与失败尝试

| 尝试 | 原因与结果 |
|---|---|
| Torch Top-K 展开改为 Triton 解析 | 把展开、边界判断和槽位查询合成一个 kernel。独立微测中约 211–214 µs 降至约 30 µs，包含调用开销；不能按这个比例推导整模型速度。 |
| decode 无新页快路径 | 从 CPU 序列长度判断 append 仍在已有页内，省去 GPU ID 回传和重复引用跟踪；跨页时仍走完整提交。 |
| full-batch swap 与原位覆盖 | 原生 swap 对驻留行使用零长度；驻留 resolver 原位填入其有效位置，减少每层切片与 scatter。 |
| 200 ms 空闲宽限 | 相邻请求可以复用尚未回收的映射；持续空闲后归还。压力回收不等待宽限。 |
| 单请求按长度降级 | 两个 12k 长请求的共享 chunk 下，没有可回收物理候选；134 次压力观察未发生迁移。 |
| chunk 感知保留与双请求规划 | 真实压力触发联合降级并回收物理 backing，见上一节。 |
| 低水位先回收无主 slack | 防止仅仅因为缓存映射尚未归还，就一直不开始低水位稳定计时；已触发自动提升。 |

微测和校验运行用于解释机制。正式性能对照关闭 logical shadow、强制切换和布局抓取，避免把诊断成本当作正常运行成本。

## 6. 正确性证据

原生路径在随机 token 输入下也出现过相同 greedy 请求生成 ID 不一致。追踪显示，部分 prefill 历史 KV 已有微小数值差异，差异随后在 decode 中放大；检查到的 Top-K 集合与顺序相同。没有观察到输入 Q/K/V 逐位相同而 attention 输出不同的反例。当前不能把跨运行生成 ID 的严格相等当作唯一的迁移正确性判据，也不能把数值差异直接归因于 VMM。

独立 logical shadow 在生产者写入时，按逻辑槽记录 BF16 K/V，并对实际被稀疏 decode 读取的物理槽进行逐位比较。迁移还检查降级前全部已提交源历史，以及提升后全部目标历史，而不只检查本步选中的 Top-K。

自动往返实验在每个 TP rank 上完成了 3 次请求降级和 1 次自动提升。完整迁移核验覆盖 **1,511,946,240 字节/rank**；选中 decode KV 核验覆盖 **25,272,944,640 字节/rank**，所有非法位置、未初始化读取、数据错配与格式错误计数均为零。完整运行见[原始结果](../results/vmm-auto-roundtrip-shadow/results.json)。这些是累计检查字节，不是显存峰值。

该运行累计降级释放 480 MiB/rank，提升增加 480 MiB/rank，另一次低水位无主 slack 回收累计 480 MiB/rank。三者属于不同事件，不能相加后宣称“每卡节省 1.4 GiB”。主 K/V 峰值为 960 MiB/rank，空闲回收后为 240 MiB/rank。

logical shadow 复用原有请求表和 Top-K 输出，因此主要证明 KV 存储、迁移和解析保持数据，不独立证明模型的选块算法或数值精度。四道自然语言题只作语义烟测，不替代长上下文质量评估。完整证据范围见[验证记录](HISPARSE_VALIDATION.md)。

## 7. 正式性能对照

原生 HiSparse 的正式三轮已完成。随后静态组在启动前触发 TP 显存不均衡检查，未进入计时；之后的设备快照显示其他服务占用全部 GPU。该快照不能证明先前报错的原因，但后续配对实验需要重新取得独占窗口。失败运行不进入性能汇总。[启动失败记录](../results/final-static-greedy/failure.json)

所有比较采用同一 MiniMax 模型、TP4、BF16 KV、相同逻辑容量和公共 SM80 适配。原镜像的相关投影路径要求更新的 GPU 架构；公共适配在加载后把非专家 MXFP8 投影转为 BF16，MoE 保持原有 NVFP4 Marlin 路径。这增加了公共权重占用，不是原版镜像直接支持 A100 的证据，也不是 VMM 单独的成本。精确镜像、软件版本与 checkpoint 索引摘要见[环境记录](../adaptive_environment.json)，完整参数见[复现说明](HISPARSE_REPRODUCE.md)。

端到端测量从本机 HTTP 请求提交至最后一个生成 token；不含模型加载、编译或 prompt 文本分词。随机 token 输入控制长度，但不代表真实业务质量与专家路由分布。每轮包含四个请求，三轮提供 12 个请求；p95 是这些请求的描述性统计，不能视作生产服务 SLO 结论。

### 原生 HiSparse 的三轮基线

下表吞吐为三轮中位数，TTFT 与 TPOT 为 12 个测量请求合并后的中位数和 p95。整卡显存为四张 GPU 在同一次 NVML 采样中的合计峰值，不把不同时间的各卡峰值相加。[完整测量](../results/final-native-greedy/performance.json)

| 工作负载 | output token/s | TTFT 中位 / p95，ms | TPOT 中位 / p95，ms | 四卡 NVML 峰值，GiB |
|---|---:|---:|---:|---:|
| 短串行，128→64 | 8.39 | 134.9 / 152.2 | 119.0 / 119.4 | 283.63 |
| 短并发，128→64 | 31.73 | 265.4 / 271.3 | 123.8 / 125.9 | 283.63 |
| 长串行，8192→128 | 7.81 | 1218.6 / 1232.3 | 118.9 / 121.8 | 283.63 |
| 长并发，8192→128 | 24.93 | 2995.7 / 4778.7 | 136.5 / 151.4 | 283.63 |
| 两短两长混合 | 20.88 | 1505.8 / 2584.3 | 135.9 / 161.0 | 283.63 |

原生基线的四道语义烟测全部通过，均以正常 stop 结束。[语义记录](../results/final-native-greedy/semantic_smoke.json)

### 单轮探索对照

以下三组各只有一轮，采用相同请求规格与公共运行时。它们用于评估优化方向，不与上一张三轮表直接配对，也不作为稳定增益结论。static-fast 启用 Triton resolver 和无新页快路径；full-batch 进一步减少混合行切片与 scatter。两种静态组均未使用后续的 200 ms 空闲宽限。

| 工作负载 | 原生 HiSparse | static-fast | static-full-batch |
|---|---:|---:|---:|
| 长串行 output token/s | 7.76 | 7.92 | 7.93 |
| 长并发 output token/s | 24.61 | 23.24 | 24.10 |
| 两短两长 output token/s | 20.38 | 19.92 | 20.90 |

[原生 pilot](../results/native-perf-pilot/performance.json) · [fast pilot](../results/static-fast-perf-pilot/performance.json) · [full-batch pilot](../results/static-fullbatch-perf-pilot/performance.json)

这个范围内，静态完整驻留没有显示出普遍的大幅加速。完整驻留避免了该请求的换入，但仍需解析 Top-K 位置，而且模型的其他计算没有减少。单轮波动与生成路径的数值差异也限制了小百分比结果的解释。

静态组空闲时四卡 NVML 合计约 281.94 GiB，相对原生约 283.63 GiB 少 1.69 GiB，约 0.6%；并发峰值则达到 284.75 GiB，高于原生。主 K/V backing 空闲回到 240 MiB/rank，其他 allocator、缓存和运行时占用会影响整卡结果。当前结果支持“释放了物理主 KV，并降低了空闲整卡占用”，不支持“所有情况下峰值更低”。生命周期内的 PyTorch max counters 与单个工作负载的峰值也不可混用。

把原生三轮运行和 full-batch pilot 的空闲记录展开，可以核对这 1.69 GiB 的组成。每 rank 的 PyTorch `reserved` 少了 672 MiB，另计的 VMM backing 增加 240 MiB，净少 **432 MiB/rank**，四卡共 **1.6875 GiB**。长并发和混合负载的 NVML 空闲差额逐卡吻合；长串行还多出 4 MiB/rank 的差额。[逐卡对账与来源摘要](../results/memory-accounting/accounting.json)

| 空闲账目，MiB/rank | 原生三轮运行 | full-batch pilot | 静态减原生 |
|---|---:|---:|---:|
| PyTorch allocated | 64,412.57 | 63,447.58 | −964.99 |
| PyTorch reserved | 65,412 | 64,740 | −672 |
| 另计 VMM backing | 0 | 240 | +240 |
| reserved 与 VMM 合计差额 | — | — | −432 |

`reserved − allocated` 的差额反而增加约 293 MiB/rank。这部分是 PyTorch 池内未计入活跃分配的空间，不能直接假设都能由 `empty_cache()` 释放。两组的重复和预热协议不同，NVML 与 scheduler 空闲快照也未同步，因此这是观测对账，不是逐项因果归因。它明确了后续优化方向：主 KV 释放量之外，还要测量并协调 PyTorch 池的实际归还，才能提高整卡可用余量。

full-batch pilot 的四道题原始输出均正确，但初版评分器未处理 MiniMax 的思考标签而误判。保留原始失败评分，并另存带原始文件摘要的[修正评分](../results/static-fullbatch-perf-pilot/semantic_rescored.json)，没有改写原始输出。

### 尚待补齐的配对

相同三轮协议下的静态混合和完全驻留对照仍待独占窗口。12k 压力下原生、静态、单请求和双请求策略的脚本已准备好；本阶段新增的 future-aware 联合运行完成了 off/deadline 与自动双请求迁移的最小矩阵，见下一小节。压力组固定提交规格，但实际到达与调度次序仍可能变化，因此需要结合每轮实际迁移和页布局解释结果。

### VMM、请求级 HiSparse 与 future 排队的联合运行

联合实验在同一 TP4 服务进程组中同时启用原生 HiSparse、VMM backing、请求级完整驻留、自动迁移和 future-aware 排队。非自适应组只切换 future policy，每个 arm 跑一次，执行顺序与机器预热状态不同：

| Run | mixed 12K token/s | short concurrent token/s | long concurrent token/s | mixed TTFT 中位，ms | mixed TPOT 中位，ms |
|---|---:|---:|---:|---:|---:|
| future off r2，较冷 | 14.069 | 27.893 | 22.414 | 3087.9 | 160.5 |
| future deadline r2 | 16.032 | 31.357 | 23.636 | 2211.8 | 157.6 |
| future off r3，预热后 | 16.743 | 32.403 | 24.522 | 2934.8 | 130.2 |

deadline r2 比 off r2 快，但预热后的 off r3 又高于 deadline r2，说明这组顺序敏感的单轮结果不能证明稳定吞吐增益。可以确认的是 future 队列真实执行：deadline r2 有 84 个 scheduler tick 和 1 次重排，异常、非法 hint 与 telemetry error 均为零；对应 short-concurrent TTFT 中位从 off r2 的约 366.8 ms 降至 272.5 ms。

自动压力组在每 rank 触发 3 次降级，其中一次是双请求联合动作，**回收 240 MiB/rank**，累计释放 480 MiB/rank。off 轮随后自动提升一次；deadline 轮没有在本轮触发提升。

| future policy | throughput token/s | TTFT 中位，ms | TPOT 中位，ms | promotion |
|---|---:|---:|---:|---:|
| off | 15.789 | 2206.6 | 150.3 | 1 |
| deadline | 15.870 | 1429 | 172.0 | 0 |

deadline 轮 TTFT 降低而 TPOT 增加，吞吐基本持平。这更像等待顺序和尾延迟的重新分配，不能包装成免费加速。两组结束时 VMM backing 都回到 240 MiB/rank，`owned_pages=0`，说明未使用的物理 handle 已归还；240 MiB 是仍有 sink 等映射的稳态，不是零占用。完整表格、边界和证据链接见[MiniMax 联合集成报告](MINIMAX_FUTURE_INTEGRATION.md)。

收口版本的 97 项 CPU 测试通过，3 项依赖 CUDA 或真实 SGLang 安装的测试跳过。完整迁移的 GPU 证据来自第 6 节的真实模型运行。可控生成 token replay 目前只通过了 CPU 处理器测试，尚不纳入测量结论。

## 8. 如何转化为真实场景收益

显存节约首先带来可使用的余量，只有系统将余量转化成更多有效工作，才会变成用户收益。

在同卡部署中，空闲模型按需归还 backing，可以减少另一个模型启动时被迫缩小 KV 容量的问题。上一阶段 dense Qwen 实验已测到这种容量效应：空闲 0.6B 实例与 40 个长 4B 请求共卡时，吞吐从 484 提高到 614 output token/s，TTFT p95 从 29.96 秒下降到 15.46 秒；人工调小静态池可达到 637 token/s。这个结果证明自动归还闲置容量的使用价值，并不证明本阶段 MiniMax 有同样增益。[上一阶段报告](RESULTS_PHASE2.md)

对于单个稀疏模型，更直接的目标是把“固定选择一种驻留模式”改为负载驱动的模式选择。低压力时减少长请求换入；高压力时回收部分历史，使更多请求或更大的 prefill 工作区获得空间。但如果模型计算远比换入昂贵，或驻留寻址本身代价较高，低压力模式也可能没有速度收益。若新增请求尚未进入调度，回收显存本身不会自动提高吞吐。

下一步应在固定显存约束下比较最大可承受并发、拒绝或重算次数、TTFT/TPOT SLO 达成率，并单独测量模式转换产生的尾延迟。当前主 K/V 软预算实验只建立了这条路径中的物理回收与迁移能力。

## 9. Prefill 激活与 CUDA Graph

prefill 激活能否参与统一预算，关键在于实际谁分配它、何时存活和能否归还。把一个 headroom 数字加入预算，不会自动控制 PyTorch、attention workspace 或 Graph private pool。当前代码可在 prefill 边界观察预算并迁移请求，但 headroom 是启发式参数；本阶段压力对照将其设为零，避免声称已经实现激活与 KV 的物理借还。

较稳妥的后续接入是先测量特定 batch/chunk 的激活峰值，在 prefill 入场前预留余量并等待迁移完成；确认可释放的 PyTorch 缓存归还后再提交 KV。若无法获得足够空间，需要 scheduler 延后入场，而不是继续分配后等待 OOM。更完整的统一系统还需要处理 Graph private pool、融合算子的 workspace 和异步生命周期。

稳定虚拟地址有利于 CUDA Graph，但不能单独保证 Graph 安全：replay 前所有真实访问和 padding/sink 地址都必须有有效 backing；模式分支和元数据张量也需要稳定接口。可探索在捕获前固定两种路径或以设备侧模式表路由，把 map/unmap 放在 replay 外安全边界。当前 MiniMax 路径尚未完成 Graph 验证；上一阶段 dense V5 的 exact/padded Graph 测试不能替代它。

## 10. 与已有工作的关系及后续研究

vAttention 已经讨论后台提前映射、异步分配和延迟回收；V5 优化了本原型的映射控制路径，但不应把这些通用思想当作首次提出。MorphServe 研究压力驱动的 KV 容量调整，Llumnix 研究活跃请求跨实例迁移。相关工作的机制对照见[文献与主张边界](HISPARSE_RELATED_WORK.md)。

本阶段可辩护的贡献是一个可复现的机制组合与实验发现：在原生稀疏模型上实现请求级混合驻留和双向转换；用实际物理 chunk 引用而非请求长度估计回收；在单请求无收益时联合规划保留页与释放页；并用真实推理及逐位历史 KV 校验检验结果。是否形成论文层面的新贡献，还需要更多布局、压力轨迹、模型和相关系统对照。

完成当前版本后，取消软件分页可以围绕三个具体问题继续。第一，给每个请求提供稳定连续 VA 后，Top-K 块到 KV 地址的解析能否减少一次软件间接寻址。第二，离散访问的局部性、GPU TLB 与映射粒度如何相互影响，是否出现“省去了软件表却增加硬件地址翻译成本”的情况。第三，前缀共享、稀疏选块和可回收物理 chunk 如何解耦，避免为了释放物理存储而约束逻辑树节点边界。模型算法的选块粒度和显存分配页不是同一个概念，应分别改变和评估。
