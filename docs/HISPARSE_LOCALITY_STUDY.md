# 真实负载下的 HiSparse 驻留容量与访问局部性

2026-11-10 用户授权的新实验。旧开发暂停不再阻止本实验；不自动恢复旧 r9 profiling 队列。2026-11-11 已完成原生 2048/4096/8192 三种驻留容量，每种六请求，采集覆盖与容量核验均通过，服务已退出。最新数据与解释见 [首轮结果](HISPARSE_LOCALITY_RESULTS_20261111.md)，下方失败记录保留实验历史。在线动态热集和无插桩端到端收益尚未验证。

## 要回答的问题

1. 聊天、工具交互轨迹和长文问答之间，是否存在请求级访问局部性差异？
2. 同一请求在连续解码窗口内，是否有高/低 miss 阶段？
3. 扩容能否显著降低高 miss 请求的换入，低 miss 请求能否缩容而仅小幅增加 miss？
4. 在相同总物理显存预算下，容量重新分配是否优于固定驻留容量？这是后续验证目标，不是当前已证实收益。

## 资源与运行边界

- 2026-11-10 12:55:31（北京时间）实时查询：八张 A100-SXM4-80GB 均使用 77,875 MiB，均有外部 VLLM::EngineCore 进程。本项目没有启动 GPU 实验。
- 13:20:01 八卡短暂空闲且无计算进程；13:21:08 复查时八卡已被新 VLLM 进程占用（前七卡各 58,819 MiB、第八卡 61,349 MiB），稳定空闲门槛未通过，仍未启动。拓扑各卡间 NV12。
- 已有 `a100-hisparse` 资源检查改为每 30 分钟。允许任选四张合适的空闲 A100，记录卡号、UUID、拓扑。同四卡两次间隔至少 60 秒无计算进程、显存 0 MiB、利用率 0%，且无外部启动争用，才启动。
- 复用 SSH 会话；否则每轮最多一次连接，通过现有 5090 ProxyJump。不要密集重连。只查询 GPU 和必要进程元数据，不读取他人文件。
- 实验使用独立容器、固定镜像 `lmsysorg/sglang:dev-minimax-m31-flash`（ID 见 STATUS.json）、只读 MiniMax-M3.1-Flash、TP4、既有 SM80 BF16 投影适配。既有模型与原环境不修改。禁止访问 `09_个人信息`。
- 自有开发容器 `training-sandbox`；项目 `/workspace/elastic-kv-vmm/project`。原始数据只放 `.container-work/locality-study/`，不提交公开仓库。依赖缺少时装进本实验的 `pythondeps`，不要改原环境。
- 资源检查仅等待并启动本实验；确认开始后停用。当前运行的实验在同一工作中跟进，不重复启动，也不将资源繁忙认定为研究完成。

## 数据来源与解释范围

| 类型 | 数据集与固定版本 | 用法 |
|---|---|---|
| Chatbot | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)，`7d6490e462285cf85d91eabea0f9a954fbddcd1f`，ODC-BY | 第一分片中的真实多轮上下文；保留不同语言与连续检查点。不保留 IP 等无关字段。 |
| Agent | [SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)，`68195a1450865274106246d0d0296a1d6807b88e`，CC-BY-4.0 | 第一分片，按真实轨迹的多个观察边界构造前缀。只回放文本，不执行工具；不是闭环 SWE-bench 解题成绩。 |
| 长文/不同领域 | [LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)，`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，Apache-2.0 | 按原始领域/子领域分层，保留原文、问题和选项。要求有依据地回答属于访问诊断，不作为官方 benchmark 分数。 |

下载脚本 `scripts/prepare_hisparse_locality_data.py` 记录 revision、文件、大小、SHA256 和抽样参数。先小样本：每类四个独立来源，聊天/轨迹可含多个检查点。第一分片试验是方便抽样，不宣称代表整个数据集。来源数和请求数分别统计，不能把同一轨迹多个检查点当独立来源扩大样本数。

先按字符数筛候选，再用原模型 tokenizer/chat template 核查真实长度。主分析匹配 4K–12K 输入长度；超限直接记为排除，不拼接无关文本、不复制填充、不静默截断。保留自然 EOS，不强制生成满 512 token；短输出单独分层，不能因为不够长就隐去。不同任务种类、话题和长度互相混杂，类型差异需在长度匹配后解释。

## 采集与实验顺序

1. CPU 准备固定样本和 manifest；检查来源、模板、输入长度与排除原因。
2. 独占 TP4 窗口内先跑 1–2 个真实样本作诊断，确认真实 request ID、slot 复用代数、rank、稀疏层、decode step 对齐。检查原生配置与实际容量。TP rank 的选择轨迹先比较后再决定是否只采 rank0。
3. 模型选择 Top-K 固定为 2048。采集 MiniMax 原生 `swap_in_selected_blocks` 的逻辑 block ID，按真实 sparse_block_size 展开并裁掉 padding/越界；不能把词表 token ID、GPU slot、host slot 当逻辑 KV 身份。
4. 用同一条真实选择轨迹回放 2048/4096/8192 驻留容量。这是**模拟批量 token-LRU**，冷启动、最新 token、prefill 初始热集与原生替换策略差异必须标明。不能把模拟 miss 称为原生 H2D 次数或端到端性能。
5. 采集原生调用前热缓存成员用于独立核对。短请求完整驻留、最新 token 保留槽、无效选项等需按原生语义处理。派生成员缺失数不是 kernel 原生 miss counter，需小样本验证后使用。
6. 再做原生固定容量 2048/4096/8192 的独立服务对照，记录输出轨迹是否一致。若输出不同，同轨迹回放负责隔离容量效应；自由生成跑次仅作端到端观察。
7. 诊断采集允许显式同步，但一律不能用于性能收益。吞吐/TTFT/TPOT 使用关闭捕获的独立对照，并记录整卡 NVML、实际热池与 VMM backing（如开启）。有外部争用的跑次单独保留并排除正式计时。

## 分析与报告

- 按请求、rank、层分别统计 16/32 步窗口；展示原始窗口分母、有效选择数和 miss 数，不只平均所有请求的百分比。采样断点不可当相邻步骤。
- 三张主要图：请求 × 容量的 miss 热图；代表请求随 decode step 的 miss 曲线；容量增加/减少对应的 miss 增量。代表请求按事先规则选，兼顾有效/无效案例。
- 每请求报告 Top-K turnover、跨步复用、各容量 miss 曲线；预热窗口与稳态分开。不同层可能比请求类别更能解释差异，层间差异不可混掉。
- 阶段变化先用同一容量、连续步骤辨认；再看同窗口其它容量反事实。随后设计仅用历史窗口的决策，与知道未来的离线最优上界分开。
- 缩容容忍阈值事先设为增加 1、2、5 个百分点的敏感性分析；不在看完结果后只选最有利阈值。
- 先验证“是否存在可重新分配的容量”，再讨论在线控制器；记录最小保留容量、调整粒度、防抖、原生 JIT/Graph 限制及实际可回收物理块。VMM 不自动使 native 热缓存形状动态化。

## 准备状态

已实现数据准备、独立容器入口、可选的原生诊断捕获、窗口容量回放。真实模型配置已只读核查：60 层均启用稀疏注意力，16 个选中 block × 128 token；首轮采第 5、15、25 层及 rank0/1。采集文件含请求代次，避免 slot 复用混淆。运行器跳过启动预热，防止无关请求污染样本。

本机通过 capture/replay/runtime 的 13 项 CPU 测试；A100 开发容器通过 capture 5 项、replay 5 项、runner 1 项以及 Bash 语法检查。固定 MiniMax 镜像实际导入采集器、校验三份原生源码哈希并安装 hook 成功，确认 CUDA 未初始化；不构成 GPU 运行证明。数据已在本机从官方源下载，逐一核对固定版本 LFS SHA256；由于 A100 到 Hugging Face 网络不可达，使用单次批量 SCP 搬入容器挂载工作区，传输已完成。原始数据不推送公开仓库。

真实容器构建产生 24 条候选。固定镜像内、只读真实 MiniMax tokenizer 的 CPU 准备完成，运行目录 `.container-work/locality-study/runs/prepare/20261110T051919Z-02b93fa3`。12 条通过 4096–12288 token 筛选：chatbot 3、agent 6、长文 3；其余为 chatbot 5 条过短、agent 3 条过短/3 条过长、长文 1 条过长。没有截断或人工填充。主试验应报告这种筛选偏差，并保留全部原始候选。

系统盘曾仅余 2–3 GiB，因此入口增加磁盘门槛：已有共享编译缓存时至少余 2 GiB，新建缓存时至少余 6 GiB。将本项目早期的 120 MiB 已验证缓存复制到当前容器工作区后复用，旧缓存只读、不修改。Docker 带唯一名称和 `elastic-kv-vmm.locality=1` 标签。当前只做小批量诊断，正式计时还需独立的争用监测与缓存处理一致性。

```bash
# 在自有开发容器 /workspace/elastic-kv-vmm/project 内构建；依赖单独放置。
PYTHONPATH=.container-work/locality-study/pythondeps \
  /workspace/elastic-kv-vmm/venv/bin/python \
  scripts/prepare_hisparse_locality_data.py build

# 宿主机只负责 docker 管理；数据、分词和推理均在容器中。
# root=/home/liuyuntao/ai-sandbox/workspace/elastic-kv-vmm/project
MODEL=/mnt/adminserver-nfsrdma/Model/zys-m31/MiniMax-M3.1-Flash
DATA="$PWD/.container-work/locality-study/datasets/workloads.jsonl"
bash scripts/container_hisparse_locality.sh none "$MODEL" "$DATA" \
  .container-work/locality-study/runs/prepare --prepare-only

# 仅在两次实时资源检查满足门槛后执行；GPUS 替换为当时实际空闲四卡。
# 第一次用 --limit-per-workload 2，确认捕获完整后再用默认上限8。
bash scripts/container_hisparse_locality.sh "$GPUS" "$MODEL" "$DATA" \
  .container-work/locality-study/runs/pilot \
  --capture --hot-buffer-tokens 4096 --sm80-bf16-projections \
  --limit-per-workload 2

# 同一原生轨迹的模拟，初始状态为每段冷启动；不称原生实测。
python benchmarks/hisparse_locality_replay.py \
  .container-work/locality-study/runs/pilot/RUN_ID/access/*.jsonl \
  --capacities 2048 4096 8192 --window-size 16
```

每次运行产生新的时间戳目录。`prepared_partial`/`partial` 明确表示类型覆盖、层/rank/步覆盖或截断存在问题，不可当完整三类结果。输出包含可恢复原始提示的 token IDs，因此所有原始运行产物也只放 `.container-work`。未来对照的吞吐和真实物理占用采集仍需补接；当前运行器不输出插桩条件下的速度收益。


## 2026-11-10 首次真实启动与重试状态

14:00:32 与 14:02:23（北京时间）两次查询，GPU 2–7 均为 0 MiB / 0%，无计算进程；选择 NV12 互联的 GPU 4、5、6、7。系统盘剩余约 430 GiB。满足门槛后删除原资源检查，在独立容器启动每类 2 条、Top-K 2048、驻留 4096、最多自然生成 512 token 的诊断。

真实目录为 `.container-work/locality-study/runs/pilot-20261110/20261110T060256Z-2a543f0f`。48 个权重分片加载完成后，14:19:28 在采集器配置检查中失败：`HiSparse access trace requires eager colocated CUDA without overlap, radix or spec`。运行器记录 `failed_step=startup`，**完成请求 0、轨迹字节 0**。因此尚不能回答容量是否改变 miss，也没有性能结果。失败输出与当时采集器源码均保留；旧源码位于同级 `failed-capture-source/`。

初始参数日志显示 eager、无 overlap、无 radix、无 speculative、无 disaggregation，但尚未确定运行时究竟哪个谓词不满足。采集器现改用 SGLang canonical runtime getters、规范化 CUDA 设备名、输出具体配置谓词，并把同样的检查前移至 `ModelRunner.load_model` 之前，避免不兼容配置在加载权重后才暴露。新版 6 项 CPU 测试通过；**真实启动修复尚未验证**，下一次须保留检查并核查具体值，不能直接移除保护。

14:26:03 八卡短暂空闲，14:28:00 复查时八卡均被外部 VLLM 服务占用，各 58,819 MiB，重试没有启动。原资源检查已删除后，重新建立每 30 分钟的 `a100-hisparse`，仅等待下一次稳定四卡窗口；任务仍未完成。下一次使用新输出目录，不覆盖失败跑次，不恢复旧 r9。

新增 `scripts/summarize_hisparse_locality.py` 用于后续真实轨迹的匿名汇总：按请求/层分开，主 rank 与交叉核对 rank 分开，16/32 步窗口及首 16 步预热单列，同时区分原生 pre-state 成员推导缺失率与同轨迹 LRU 模拟。其 3 项 CPU 测试通过，但尚无真实轨迹可供分析。公开输出不含原始对话、token IDs 或原始请求标识。


## 2026-11-11 凌晨重试：定位运行时配置读取错误

00:00:09 与 00:01:37（北京时间）两次实时查询八卡均空闲，无计算进程；再次选择 GPU 4–7。按约定删除资源检查，启动独立容器 `ekv-locality-20261110T160217-2058921`。拓扑、卡号和 UUID 保存在 `.container-work/locality-study/runs/pilot-retry-20261111/`，实际运行目录为其下的 `20261110T160218Z-d6275554`。

00:03:43，新加的提前检查在权重加载前捕获 `torch.device(None)`：实际 `get_server_args().device` 为 `None`。本次完成请求仍为 0，trace 为 0；不是模型推理失败，也没有 miss/速度结果。旧检查源码与测试已保存在该输出根的 `failed-capture-source/`，失败输出保留。

只读核查固定镜像源码后，将设备取值改为 `get_device().device`（原生 ModelRunner 同样使用），Radix Cache 改为 `get_memory().disable_radix_cache`（原生 KV cache builder 同样使用），图配置改为 `get_exec().graph.cuda_graph_config`。其余安全断言与加载前检查均保留，没有绕过检查。新增测试通过安装入口调用 `ModelRunner.load_model` 的采集器包装器（底层执行器和配置使用测试替身），模拟 legacy 参数为 None，验证有效 resolved 配置进入原加载器，无效配置在加载前拒绝。本机及 A100 自有开发容器均通过 7 项测试，修正已同步；**完整 GPU 启动仍未验证**。

00:06:44 复查八卡均有外部 VLLM 进程，各使用 77,875 MiB；本项目容器已退出。没有启动新的 GPU 重试，重新建立每 30 分钟的 `a100-hisparse` 资源检查。下一窗口使用新目录验证这次修正，通过启动后再做真实请求采集与覆盖核验。任务尚未完成。
