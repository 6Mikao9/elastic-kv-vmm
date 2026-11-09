# 容器迁移与 VMM/Graph 验证（2026-11-09）

当前代码、Git 历史及七个历史实验目录已整理到开发容器的持久挂载目录
`/workspace/elastic-kv-vmm`。工作目录为 `project/`，旧结果在 `legacy/` 中只读保存。
迁移采用复制、逐文件 SHA256 核验；原目录保留，已有模型和运行环境未修改。
后续编辑和实验由容器内执行，避免本地频繁新建 SSH 连接。

## 验证结果

| 项目 | 实测结果 |
|---|---|
| 代码基点 | `709f2ca1c05a89ffc4482c43ce6e6805fa6385a7`，包含迁移时工作树中的未提交修改 |
| 初始工作树快照 | 601 个非忽略文件，逐文件哈希核验 |
| 历史数据 | 七个目录；源文件复制前后与副本哈希一致，清单保留于 `migration/legacy-manifest.json` |
| 开发环境 | 独立 venv，复用 Torch 2.8.0+cu128、SGLang 0.5.2；离线 editable 安装 |
| 原生库 | 在开发容器内编译 `native/libelastic_vmm.so` 成功 |
| CPU 测试 | 593 passed、3 skipped，另有 184 subtests passed |
| CUDA 测试 | `BackingTest`、`GraphBackingTest` 共 2 passed |
| GPU | 一张 A100-SXM4-80GB，启动前两次空闲检查间隔约 61 秒 |
| 结束状态 | 该卡无计算进程，NVML 显存 0 MiB、利用率 0% |

初次 CPU 验证发现：已有 scheduler fence 时，代码跳过了设备同步，却仍把分支耗时
计入同步耗时。已修正 `boundary_sync_ns` 和 `request_affine_reclaim_sync_ns`，
仅实际同步时计时。修正后完整 CPU 测试通过。首次 GPU 测试入口使用了不可导入的
`tests.*` 模块路径，尚未执行 CUDA 测试便失败；改用 pytest 文件入口后两项通过。
所有失败日志保留。

## 真实映射与 Graph

`scripts/profile_vmm_handoff.py` 保留 64 MiB 虚拟地址，以 16 MiB 为物理映射粒度：

| 阶段 | 已映射 chunk | VMM backing |
|---|---|---:|
| 仅保留地址 | 无 | 0 MiB |
| 首次映射 | 0、1 | 32 MiB |
| 扩容 | 0、1、2 | 48 MiB |
| 回收中间块 | 0、2 | 32 MiB |
| 重新映射中间块 | 0、1、2 | 48 MiB |

虚拟基址始终不变。保留块的数据逐字节检查通过；重新映射后，先前捕获的 Graph
成功向原虚拟地址写入，并验证目标以外的数据未改变。独立 Graph 测试还覆盖了
中间块缺页期间仅访问保留块的重放。不能让 Graph 读取尚未重新映射的地址。

PyTorch Profiler trace 有 18 个 CUDA kernel 事件、1 次 `cudaGraphLaunch`；
Graph replay 标记区间内包含该 launch 和 1 个 kernel，确有设备执行证据。
trace 大小为 131,877 bytes，SHA256：
`be6f7da8955dd379e6b53c7db87f8c85a60c3adce649dc6ccc8ba69cd079c90f`。

VMM backing 是本 allocator 的实际映射量，不是整卡 NVML 占用。此次只采集了整卡
开始和结束的空闲状态，没有逐阶段 NVML 曲线。Profiler 中的区间耗时包括同步、
初始化和 profiler 开销，不用于证明推理加速。也不要求 Kineto 展示每一次 `cuMem*`
驱动调用；原生返回检查、映射状态和数据正确性共同验证映射操作。

## 证据与接续

- `migration/snapshot.json`：迁移代码清单与初始哈希。
- `migration/setup-result.json`：首次操作与失败记录。
- `migration/validation-r2.json`、`cpu-tests-r2.log`：修正后的 CPU 结果。
- `migration/validation-r3.json`、`gpu-tests-r3.log`：实际 CUDA 结果与空闲记录。
- `migration/vmm-profile-r3/{result.json,trace.json,trace-evidence.json}`：功能检查与 trace。

以上路径均相对于容器 `/workspace/elastic-kv-vmm`。本轮只验证容器环境和模型无关的
映射生命周期，没有运行四卡 MiniMax HiSparse 对照。后者继续使用固定 MiniMax 镜像、
只读权重和原基线参数，待约定四卡空闲后按 `HISPARSE_REPRODUCE.md` 继续。
开发容器的 SGLang 0.5.2 不能直接代替该固定镜像。
