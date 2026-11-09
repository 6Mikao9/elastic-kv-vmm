# SGLang 框架接入 smoke

日期：2026-11-08。目的不是重新做完整性能矩阵，而是验证新的
`feature/sglang-framework` 在真实的 SGLang 0.5.2 服务中：

- 启动器能在父进程和 multiprocessing worker 中安装同一个适配器；
- VMM 适配器确实替代了固定物理 KV pool；
- eager 与 graph-pad 路径都能完成流式请求和固定 greedy 正确性探针；
- eager、Graph、VMM backing 和整卡 NVML 指标能够同时落盘。

## 条件

单张空闲 A100-SXM4-80GB、Qwen3-0.6B、SGLang 0.5.2、
`training-sandbox:torch280-cu128`。详细版本见 `environment.json`。服务使用
TP1、Triton attention、page size 1，关闭 overlap 和 radix cache，逻辑 KV
容量 32,768 slots。

负载为 `short_serial`：2 轮，每轮 6 个串行请求，每个请求 128 input / 64
output token。每轮有独立 warmup；每个模式另跑 12 个固定 greedy 探针。
这只能验证接入和观察数量级，不能替代已有三轮矩阵或统计显著性检验。

## 结果

| 执行方式 | arm | 输出 token/s | TTFT 中位 | TPOT 中位 | 整卡峰值 | V5 最终 backing |
|---|---|---:|---:|---:|---:|---:|
| eager | baseline | 62.983 | 23.98 ms | 15.76 ms | 6.23 GiB | — |
| eager | V5 | 63.722 | 23.90 ms | 15.58 ms | 2.73 GiB | 112 MiB |
| graph-pad | baseline | 353.771 | 23.11 ms | 2.493 ms | 6.29 GiB | — |
| graph-pad | V5 | 345.731 | 23.59 ms | 2.529 ms | 2.79 GiB | 896 MiB |

相对同执行方式 baseline：

- eager V5 的整卡峰值降低 56.2%；吞吐差为 +1.2%，TTFT 差为 -0.3%，
  该小样本不能证明加速，只说明没有明显退化。
- graph-pad V5-keep8 的整卡峰值降低 55.7%；吞吐为 -2.3%，TTFT 为 +2.1%，
  TPOT 为 +1.4%。这里观察到约 2% 的 Graph 路径成本。
- eager 与 graph-pad 的 12/12 固定探针输出哈希均与各自 baseline 一致。
- graph-pad V5 记录了 1 次 capture、3,040 次实际 Graph replay 和 381 次 padding
  replay，说明不是只跑到了 eager fallback。
- 容器内 60 项 CPU/CUDA 单元测试全部通过；其中 VMM backing 和 Graph 地址复用
  两项随后在空闲 GPU2 上实际运行。
- 在无网络容器内以 `--no-build-isolation --no-deps` 成功构建并安装 wheel，
  `elastic_kv` 与 `adaptive_sparse` 都进入安装包。

机器生成的汇总与逐请求结果见
[`results/sglang-framework-smoke`](../results/sglang-framework-smoke/)。

## 2026-11-08 SGLang 0.5.17 接入复核

5090 新增了 0.5.17 的 tokenizer-to-scheduler transport。0.5.17 的
`TokenizedGenerateReqInput` 是基于 `msgspec.Struct` 的二进制 IPC schema，普通
Python 动态属性不会进入 msgpack。适配器在 worker 启动前重新生成带
`kv_hints` 字段的同名 struct，tokenizer 发送规范化的 mapping，scheduler 再恢复
为不可变 envelope。对应请求类和 scheduler 排队 wrapper 均保持原生
`PrefillPlan`、`PrefillAdder` 和容量检查不变。

同一镜像内全量结果为 167 passed、2 skipped；跳过项都要求 CUDA device。
随后尝试在 5090 GPU0 启动 `Qwen2.5-3B`。服务已越过请求 transport 并进入 SGLang
启动流程。GPU0、GPU1、GPU3 的容器都能执行 `nvidia-smi` 并看到 2 MiB 空闲显存，
但 PyTorch 在 `torch.cuda` 延迟初始化时对三张卡都报 `CUDA unknown error`，最终在
`ServerArgs._handle_missing_default_values()` 报 `No accelerator`。宿主
`nvidia-container-cli info` 同时报 `nvml error: unknown error`，而 GPU2 在宿主
NVML 中本身为 `Unknown Error`。因此没有生成 serving 延迟或显存结果，也没有把
启动失败记成算法收益。机器可读的三卡尝试、原始日志哈希和服务错误摘录保存在
[`results/future-smoke/5090-gpu-blocker`](../results/future-smoke/5090-gpu-blocker/)。

随后新增的模型无关预测预热冒烟
（`scripts/container_predictive_smoke.sh`）也在同一台 5090 GPU0 上复现了
`CUDA unknown error`。Linux/SGLang 容器中的全套 CPU 测试为 400 passed、
2 skipped，因此该轮只完成了 CPU 部分的验证，
没有把未执行的 CUDA 映射生命周期记成通过。

恢复 GPU runtime 后，先复跑 baseline，再以相同模型和输出目录版本运行 VMM：

```bash
FUTURE_SMOKE_IMAGE=lmsysorg/sglang:latest \
  bash scripts/container_future_smoke.sh 0 /data/model/Qwen2.5-3B \
  results/future-smoke/5090-qwen25-3b-baseline-r4

FUTURE_SMOKE_IMAGE=lmsysorg/sglang:latest \
  bash scripts/container_future_smoke.sh 0 /data/model/Qwen2.5-3B \
  results/future-smoke/5090-qwen25-3b-v5-r1 \
  --elastic-mode v5 --elastic-retain-chunks 1
```

runner 现在会落盘逐请求 TTFT、TPOT、总延迟、输出 token/s、整卡 NVML、future hook
计数和 VMM logical/backed bytes；只有 baseline 和 V5 都完成后才能比较内存与效率。

计划 rollout 的结果现在还包含 `decode_occupancy`。它按 request ID 把计划中的
`decode_batch_size` 与 SGLang hook 观测到的真实 batch-size histogram 对齐，并导出
最近 4096 个 decode batch 的有界事件序列。`--require-decode-occupancy` 要求运行
至少出现一次真实 decode 观测，适合接入回归；实际 batch size 与计划不同会记录为
结构证据，不会直接改写成吞吐结论。相邻事件之间的单调时钟差会用于计算每个请求在
真实 batch size 上的驻留时长和时间加权 exact fraction；最后一个未闭合事件明确
不计入时长。

`adaptive_sparse/profile_calibration.py` 进一步把同一 `results.json` 中的逐请求
TTFT/TPOT 和 `future_stats` 转成规划器可直接读取的 `decode_batch_profiles` 与
`prefill_batch_profiles`。它只采用宽度唯一的 decode 观测和完整原生 prefill cohort，
排除 mixed-width/incomplete-cohort 样本；样本数不足的组合只保留为证据。推荐用
`scripts/calibrate_future_profiles.py --min-samples 3 --require-available` 生成下一轮
workload 的 profile，避免把一次偶然命中写成稳定吞吐。decode profile 会保留
`(residence, kind, batch size)` 的嵌套覆盖；某类型样本不足时自动使用同 batch 的
aggregate 值，避免把少量观测固化成服务差异。

## 本次修复

第一次 V5 smoke 没有进入 `ElasticPool`：服务日志仍显示原生固定 pool。原因是新启动器
只在父进程调用 `install()`，Python spawn 出来的 worker 会重新导入 launch 模块，
但不会重新执行 `main()`。现在父进程先完成配置校验，再设置仅供本次进程树使用的
安装 sentinel；worker 导入 launch 模块时据此安装相同适配器。普通 import 和
`--elastic-dry-run` 仍不会导入 SGLang 或触碰 CUDA。

同时修正了 baseline 基准脚本错误地要求读取 VMM 专有
`adapter_stats.json` 的问题。baseline 现在使用整卡 NVML；V5 使用 NVML 加适配器
逻辑/物理 backing telemetry。

## 2026-11-09 request-affine VMM 集成复核

在 5090 宿主机新增 clean snapshot
`/home/liuyuntao/experiments/elastic-kv-vmm-5090-20261108/src-r23-request-affine-final`，
对应提交 `39ea49fa5850d7ea0f44b22b794d6eb18f2bd677`。使用既有
`lmsysorg/sglang:latest` 镜像（SGLang 0.5.17），以 `--network none`、不授予 GPU、
源码只读挂载的隔离容器执行完整测试。快照为 171,167,267 bytes，结果为
**461 passed, 2 skipped, 146 subtests passed**；两个 skip 都需要真实 CUDA device。
日志为 `tests-0.5.17-r23-request-affine-final.log`，SHA256
`cb5bf517f71498ca2a89a91338e68fd024b0ee1ca78ba8fbb383681477d266d3`。

本轮验证了 0.5.17 `alloc_for_extend` / `alloc_for_decode` 请求身份注入、
request-private chunk 增长与释放、池尾 chunk、`FreeSlots` 守恒、physical chunk
预检回退，以及 SGLang 实际 import/install 路径。另以 0.5.2 源码形状验证
`ScheduleBatch.alloc_token_slots` extend/decode 包装。它仍不是 CUDA smoke：
GPU0/1/3 在容器退出后为 2 MiB / 0%，GPU2 仍为 `Unknown Error`，宿主 CUDA
runtime 也未恢复。因此本轮不报告 serving 吞吐、TTFT、TPOT、整卡显存或 mapping
延迟收益。

## 2026-11-09 measured batch cohort 复核

提交 `3c1920b7a371862bb670d84d5f8f27402de24d30` 新增了按实测 prefill/decode
profile 形成已知未来 cohort 的 `batched` 策略、联合 resident/HiSparse placement、
deadline guard，以及执行器的 cohort 元数据校验。在 5090 宿主机使用既有
`lmsysorg/sglang:latest`（SGLang 0.5.17）镜像，以 `--network none`、不授予 GPU、
源码只读挂载运行完整测试，结果为 **487 passed, 2 skipped, 157 subtests passed**。
两个 skip 都需要真实 CUDA。日志 SHA256 为
`8cb3f3920df1e23b3cbeab5e41d530e9078629b2f45f899129577f15cd3a708a`。

这轮证明新控制面能在 SGLang 0.5.17 源码环境中导入、序列化并通过现有接入回归；
它没有证明 GPU 上实际形成了目标 cohort，也没有产生 TTFT、TPOT、吞吐或显存收益。
GPU0/1/3 当时整卡为 2 MiB、0%，GPU2 仍为 NVML `Unknown Error`，宿主 CUDA
runtime 尚未恢复，因此真实 paired serving 继续留给可用加速器窗口。
