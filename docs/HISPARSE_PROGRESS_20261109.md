# 2026-11-09：共享前缀换入修正与真实模型 profiling 入口

本轮继续完善当前请求级 HiSparse，不改变原生稀疏 Top-K 语义，也没有开展软件页表移除。
开发和验证在 A100 的持久容器工作区中执行。以下 CPU 测试数量对应包含原有未提交
改动的工作树，不能直接当作某个公开提交的测试数量。

## 部分前缀命中时，真正跳过对应换入

审查发现：此前部分命中的请求仍把完整 Top-K 交给原生 swap，之后才用 donor 的
GPU 槽位覆盖命中位置。输出引用了共享 KV，但不能据此声称那些位置已免去原生 staging。

现在在调用原生 swap 前，将可直接读取的块标记为 `-1`，保留 Top-K 的形状和顺序，
原生处理剩余位置后再合并结果。全命中行不调用 swap；最新 token 所在块仍由原生
保留槽处理。固定版本 CUDA 内核对负位置在哈希表、LRU 和 miss-copy 前进行排除；
启用直接复用时同时检查该内核与 Python 包装器的原始 SHA256。

新增测试实际调用共享换入 helper，检查送给原生函数的块、最终输出、全命中、部分
命中、最新 token、无效块和无 donor 的情况。统计项
`shared_prefix_masked_native_blocks` 表示部分行中从原生调用排除的块数，
不等于实际 H2D 字节：原生热池中的命中本来就可能不发生数据传输。
GPU 正确性、实际流量与 TPOT 仍需单独验证。详见[共享前缀实现](SHARED_PREFIX_RUNTIME.md)。

## 真实模型 profiler 的执行入口

新增 `--profile-workload`，在指定 workload 预热后开启固定镜像自带的 PyTorch
Profiler，捕获相同请求，再停止并等待四个 TP rank 的 trace 写出。默认每请求最多
生成 16 token，可设置 1–32；关闭 stack 和 shape 记录以限制诊断规模。

分析器要求每个 rank 都有 CPU 算子和 CUDA kernel；统计各内核耗时、GPU 活动区间
的并集与耗时总和、同步 API 的 CPU 调用时间、可见的显式 memcpy。并发 GPU 活动
不能简单相加当作端到端时间；stream-wait API 的 CPU 耗时也不等于 GPU 等待时间。
HiSparse 可在内核里读取 mapped host memory，因此显式 memcpy 字节不是完整 H2D 流量。

诊断运行与正式性能测量互斥，manifest 带诊断标记，汇总器拒绝将其纳入吞吐对照。
当前已完成控制接口、异常清理、四 rank trace 覆盖、区间统计和过滤规则的 CPU 验证。
**完整 MiniMax 的 GPU trace 尚未采集**。现有真实 GPU trace 来自单卡 VMM/Graph
映射测试，二者不能混同。[使用说明](HISPARSE_PROFILING.md) ·
[已完成的单卡验证](CONTAINER_VALIDATION_20261109.md)

## 验证与正式对照状态

在开发容器的 Python 3.10 / Torch 2.8.0+cu128 中：针对性测试 65 passed；完整测试
598 passed、3 skipped，另有 184 subtests passed。日志位于容器持久目录
`/workspace/elastic-kv-vmm/migration/research-full-tests.log`。

历史原生五负载三轮结果保留，不重复。共同的投影适配、遥测、语义探针与容器包装器
字节一致，80 个历史输入 hash 均可重建匹配。当前处理组采用更新后的驻留实现，
与历史原生结果属于负载匹配的跨运行比较；12K 压力组仍需自己的原生对照。
用户已允许改用 GPU 0、1、3、4，拓扑均为 NV12。详见[配对审查](HISPARSE_PAIRING_20261109.md)。

首次正式实验启动前另一个程序占用了全部八张卡，门禁阻止了模型加载。随后服务器
上的单个控制任务先冻结源码并准备缓存，在四卡通过连续空闲检查后启动静态混合组，
其余组顺序执行；结果完成前不进入性能表。它通过本机查询检查资源，本地复用同一
SSH 连接跟进。旧应用定时检查经核实已暂停。
等待任务的状态文件只是定位信息，接续时必须核实进程、容器和 GPU，不能重复启动。

### 统一入口的 spawn 缺陷

静态组首次恢复启动后，在模型加载前被 SM90 最低能力检查拒绝。根因是
`python -m elastic_kv` 的入口是包的 `__main__`；Python multiprocessing spawn
不会重建这种模块，因此 worker 没有执行原有的 SM80 投影适配。这个问题也可能影响
依赖同一导入入口的其他 worker hooks，CPU 中直接调用安装函数的测试不能覆盖它。

已将实验 runner 改为 `python -m elastic_kv.launch`，包入口也通过 `runpy` 暴露
实际启动模块。新增测试启动真实 spawn 子进程，以每个进程自身 PID 核对安装行为：
旧包入口可复现失败，两个修正入口均通过。容器完整 CPU 测试更新为 599 passed、
3 skipped，另有 186 subtests passed。失败结果保留，后续采用新目录重试；仍须由
真实模型启动和推理验证这些 hooks，而不能只凭这项 CPU 回归宣称已恢复性能对照。
