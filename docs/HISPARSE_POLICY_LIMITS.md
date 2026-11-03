# HiSparse 动态预算的当前边界

调度器在 prefill 分配前检查预算，decode 默认每 16 步观测一次。每次检查以 TP rank 中最大的稀疏主 KV backing 加上适用的 prefill 余量来判定压力。每次边界最多提交一个迁移事务，默认一个请求。启用 EKV_ADAPTIVE_MAX_GROUP=2 后，可在无单请求可回收候选时选择一个双请求降级事务。实际释放量按 rank 记录。这个预算覆盖主 KV backing，权重、其他 KV 池、CUDA workspace 和整卡压力需另行核算。

## 候选与迁移成本

Demotion 按各 rank 最小可回收字节除以最大传输字节排序，请求 ID 只用于平局。预测基于独占物理 chunk，迁移后另记录实际净释放。双请求候选按两者释放页的并集重新计算 chunk 引用，且要求每个 rank 的净释放预测为正。单请求或双请求事务的传输上限默认每 rank 512 MiB，可用 EKV_ADAPTIVE_MIGRATION_BYTE_CAP 调整。当前缺少滚动带宽和耗时额度。后续可用实测拷贝速率做启动前筛选。

## 防抖与 prefill 余量

常规迁移同时受全局冷却和逐请求状态驻留时间限制，从初次观察或最近一次状态变化计时。Promotion 还要求低水位持续稳定。采样间隔内的短时越线仍可能漏过。诊断强制迁移绕过驻留时间和字节额度，分别记录 cap 和 dwell 拒绝次数。

当前 prefill headroom 是环境变量提供的固定数，调度器未从真实 activation telemetry 自动估算。可按模型、prefill chunk、有效 batch 形状校准预留，独立记录 PyTorch allocated 峰值、驱动余量和整卡 NVML。固定余量适合可复现实验，其触发结果只说明配置条件下的策略行为。

## 真实预算压力实验

实测压力对照采用主 KV 高水位 800 MiB、低水位 720 MiB。固定步 round trip 验证事务和内容，关闭强制开关后观察 prefill_pressure、decode_pressure 或其 group 变体，才能确认配置驱动了降级。记录逐 rank 实际净释放、迁移时长、请求完成数、TTFT、TPOT 和峰值。Prefill headroom 仍需独立校准。

## 已验证的共享几何与自动回迁

双长请求快照中，两个 resident 请求各有 97 个物理页，分别保留 33 页并释放 64 页。两者单独释放均无法回收完整 chunk，联合释放可回收 120 个 2 MiB chunk。另有 5 个存活页不属于这两个候选。双请求事务跨 TP 确认准备和提交，实测联合净释放每 rank 240 MiB。包含一次单请求降级后，累计降级 3 个请求，净释放 480 MiB/rank。见 [页几何](../results/vmm-pair12k-shadow/adaptive_stats/plans-rank00.json)及 [逐 rank 统计](../results/vmm-pair12k-shadow/adaptive_stats/rank00.json)。

自动往返用 800 MiB 高水位和 720 MiB 低水位，主 KV backing 峰值达到 960 MiB。水位用于周期边界上的软控制。四个 rank 各记录一次单请求和一次双请求降级，共 480 MiB 净释放。低水位路径另回收 480 MiB 无主 slack，按回收后实际 backing 开始稳定计时，随后自动提升一个仍在 decode 的请求。提升净增长 480 MiB/rank。每 rank 的迁移内容校验覆盖 1,511,946,240 字节，完整目的历史和 decode 选中 KV 均无错配。该运行开启 shadow，仅用于内容正确性与触发验证。见 [自动往返结果](../results/vmm-auto-roundtrip-shadow/results.json)。

## 正式压力对照

用同一固定源码、权重、预留、模型参数和并发提交规格，只改变 EKV_ADAPTIVE_MAX_GROUP 的 1 与 2。两组均开启相同的 chunk-aware retention 和低水位 slack 清理，关闭 shadow、页计划诊断和强制迁移。先以相同负载预热，再各测三轮。建议输入为两个并发 12,288-token 长请求，输出各 64、128 tokens，再加两个相同短请求。两组固定请求规格及客户端提交顺序。并发线程的实际到达和服务端排队顺序可能波动，进而改变页布局，需结合每轮事件判断。较长 decode 请求在另一个请求结束后仍保留，以覆盖回迁窗口。

参考设置为 resident limit 2、slot reserve 4,096、高水位 838860800 bytes、低水位 754974720 bytes、prefill headroom 0、迁移上限 1073741824 bytes/rank。两组固定相同的 dwell、低水位稳定时间和观测步频。512 MiB 默认迁移上限可能拒绝两个 12K 请求的联合拷贝。至少确认两组都出现真实压力；MAX_GROUP=2 要出现 group 事件才证明所测工作负载执行了联合路径。按 arm 比较四卡 NVML 峰值和主 KV backing、净释放、迁移次数与时间、吞吐和逐请求 TTFT/TPOT。若需要单独归因回迁开销，可另跑两组均关闭低水位 promotion 的补充对照。
