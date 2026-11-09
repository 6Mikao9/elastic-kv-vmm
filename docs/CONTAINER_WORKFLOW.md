# A100 容器开发与实验入口

2026-11-09 用户要求：尽量在容器内工作；本地频繁建立 SSH 连接曾触发服务器风控。
日常编辑、编译、CPU/CUDA 测试和 profiling 在服务器容器内执行。使用已有容器 Codex
会话或单条长连接，不以循环启动新 SSH 的方式监控实验。此约定不改变 GPU 空闲检查。

## 持久目录

本次整理目标是现有 `training-sandbox` 的 `/workspace/elastic-kv-vmm`：

- `project/`：Windows 工作树的完整非忽略文件快照，包含未提交修改及 Git 历史。
- `legacy/`：本项目七个宿主机历史实验目录的校验副本，内容只读；原目录保留。
- `venv/`：独立 Python 环境，复用镜像依赖，不修改原容器的包。
- `migration/`：传输清单、原文件哈希、验证日志、GPU smoke 和 profiler trace。
- `CONTAINER_HANDOFF.md`：容器内接续工作说明。

这是 bind mount 中的数据，不依赖容器可写层寿命。已有 `/workspace/ekv-integration-dev`
保留；新的工作树包含更新的本地改动。`migration/setup-result.json` 保留首次操作记录，
其中首次 CPU 测试失败；修正后的 CPU 结果见 `migration/validation-r2.json`，
GPU 复测结果见 `migration/validation-r3.json`。每轮失败记录保留，不能只根据目录存在
或旧记录中的状态判断最终结果。具体宿主机路径只记在工作区内部台账。
本次已完成的检查及范围见 [容器验证记录](CONTAINER_VALIDATION_20261109.md)。

## 在已连接的容器终端使用

```bash
cd /workspace/elastic-kv-vmm/project
source ../venv/bin/activate
source scripts/container_session.sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m elastic_kv.build_vmm
```

`container_session.sh` 把新缓存和临时文件留在项目 `.container-work/` 中，默认启用
Hugging Face 离线模式。它不分配 GPU，也不预先选卡。构建 VMM 库不需要计算 GPU。
运行前仍需检查整机计算进程、显存、利用率及其他任务的启动活动。

原生 MiniMax 的正式对照继续采用固定
`lmsysorg/sglang:dev-minimax-m31-flash` 镜像，ImageID 为
`sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87`。
它与开发容器内 SGLang 0.5.2 不同，不能混用来宣称与原基线公平。
Docker 的创建/挂载属于宿主机控制操作；实际编译与推理在目标容器中执行。
现有开发容器不挂载 Docker socket，不需要为迁移扩大其权限。

模型始终只读挂载，MiniMax 权重不复制进工作树。历史结果、编译缓存和数据源路径
保持原始记录；旧 broker SQLite/WAL 不能作为新运行状态继续使用。
多容器共享 broker 时须保证进程 PID 可见性一致，不能把独立 PID namespace 中的
PID 直接视为全局身份。

## Profiling

`scripts/profile_vmm_handoff.py` 提供模型无关的 CUDA VMM/Graph 验证：
仅接受一个显式 GPU UUID，需要 `--ack-idle-gpu`，输出 `result.json` 和 PyTorch
Profiler 的 Chrome trace。分别标记 map、grow、drop、remap、Graph replay。
其数据只证明映射生命周期与 Graph 地址复用，不能当作 HiSparse 端到端加速。

```bash
# 先完成空闲核查，再填写真实 UUID；输出目录必须为新目录。
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  python -m scripts.profile_vmm_handoff --ack-idle-gpu \
  --output-dir .container-work/profiles/vmm-handoff-r1
```

完整模型的 profiler 运行应与无 profiler 计时分开，记录源文件哈希、GPU UUID、
镜像、模型和参数。四卡 MiniMax 对照仍按 `HISPARSE_REPRODUCE.md`，不重复已完成的
原生五负载三轮基线；静态、完全驻留与压力策略对照尚待完成。
