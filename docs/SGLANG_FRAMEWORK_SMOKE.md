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

## 本次修复

第一次 V5 smoke 没有进入 `ElasticPool`：服务日志仍显示原生固定 pool。原因是新启动器
只在父进程调用 `install()`，Python spawn 出来的 worker 会重新导入 launch 模块，
但不会重新执行 `main()`。现在父进程先完成配置校验，再设置仅供本次进程树使用的
安装 sentinel；worker 导入 launch 模块时据此安装相同适配器。普通 import 和
`--elastic-dry-run` 仍不会导入 SGLang 或触碰 CUDA。

同时修正了 baseline 基准脚本错误地要求读取 VMM 专有
`adapter_stats.json` 的问题。baseline 现在使用整卡 NVML；V5 使用 NVML 加适配器
逻辑/物理 backing telemetry。
