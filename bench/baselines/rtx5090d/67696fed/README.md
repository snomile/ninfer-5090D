# deepthink RTX 5090 D 优化前基线

在现役 NInfer `67696fed9f761fa8613869df50e0917e9c0c0b63` 上实测。模型为
Qwen3.8-27B NVFP4，服务使用 FP8 KV、FP32 线性注意力状态、固定 DFlash2 K7、
1024 token Prefill 块、两路上限。补测复用现役镜像和模型，独立启动五轮；
各请求均为零前缀命中。正式补测用回环端口，完成后恢复现役服务并通过健康检查。
完整镜像、模型和测试版本以及资源监控见各结果文件。

| Serve 场景 | 首字延迟中位数 | 输出与校验 |
| --- | ---: | --- |
| 2K / 8K 单路 | 0.259 / 0.674 s | 每路 404 token，全部通过 |
| 2K / 8K 双路 | 0.365–0.395 / 1.333–1.364 s | 每路 404 token，全部通过 |
| 32K / 64K 单路 | 3.082 / 7.639 s | 每路 404 token，全部通过 |
| 32K / 64K 等长双路 | 6.226–6.258 / 15.201–15.231 s | 每路 404 token，全部通过 |
| 32K + 64K 不等长双路 | 6.416 / 12.131 s | 每路 404 token，5/5 轮精确匹配；两路完成中位数 13.039 s |
| 32K / 64K 检索 | 3.106 / 7.642 s | 两组均 5/5 轮答对全部 3 条；接受率 42.9% / 27.6%，输出 48 / 41 token |
| 低接受率长解码 | 0.767 s | 每轮 1,568 token，5/5 轮答对全部 12 条；接受率 26.79%，首字后 176.11 token/s |

等长和普通 2K 短回答来自同版本的五轮发布归档；二者与原发布的真实 Host KV
压力恢复一并收录在 [matched-serve.json](matched-serve.json)。压力恢复实测 GPU
下放 1.13 GiB，回读 50.39 MiB。归档中的 `baseline_pending` 是补测前的状态；
不等长及检索补测完成的数据见 [supplemental-serve.json](supplemental-serve.json)。

长解码使用 `free-decode-retrieval`：9,438 token 输入，按指定次序回传散布在上下文
里的 12 条独立合成编码，不使用 JSON Schema 约束；必须完整答对并自然结束。正式记录见
[low-acceptance-serve.json](low-acceptance-serve.json)，覆盖条件为至少 512 输出
token，固定 K7 的 `accepted_tokens / drafted_tokens` 不高于 70%。五轮均满足条件，
生成序列哈希一致；容器启动到健康的中位数为 5.74 s，单次请求完成中位数为 9.666 s。
最早的 `free-decode-ledger` 也保留在补测数据中：生成了 1,271 token，五轮均未输出正确
JSON 字段，接受率 95.1%，不计入低接受率覆盖。

[ReplaySSM CSV](replayssm.csv) 记录 6 种 CUDA Graph 负载的冷、热数据；
每种预热 10 次、采样 100 次。冷数据下，单路/双路 record 的 GPU 延迟中位数为
10.24 / 14.34 µs；48 层 fold 单路提交 1 / 8 token 为 188.42 / 198.66 µs，
双路提交 1+8 / 8+8 token 为 407.55 / 415.74 µs。这些是独立算子时间。

[困惑度报告](perplexity.json) 使用 `ninfer-ppl-1m-v1` 的 quick 组 4 条流，
窗口 65,536、步长 32,768，FP32 状态、FP8 KV。共评分 261,167 token，
PPL **4.059396**，稳态评分耗时 33.97 s；模型加载另计 2.74 s。
[首次补测来源及校验和](supplemental-provenance.json) 和
[测试参数](supplemental-plan.json) 随数据保留。

后续比较先把候选版本固定为 K7 + FP32，沿用同一请求、同一模型和服务参数。
然后在同一候选版本内单独切换自适应草稿长度，并将 BF16 状态另做对照，使用同一
评分语料复核质量。每组跑五轮新服务实例，保留首字延迟、逐轮输出校验、草稿计数和
输出序列哈希。`free-decode-retrieval` 独立跑，不与其他请求混在同一轮：

```bash
python3 -B tools/bench/run_optimization_ab.py run --case free-decode-retrieval \
  --base-url http://127.0.0.1:11403 --request-log "$REQUEST_LOG" \
  --revision "$NINFER_REVISION" --model-sha256 "$MODEL_SHA256" > "$ROUND_REPORT"
python3 -B tools/bench/run_optimization_ab.py compare \
  --before bench/baselines/rtx5090d/67696fed/low-acceptance-serve.json \
  --after "$ROUND1" "$ROUND2" "$ROUND3" "$ROUND4" "$ROUND5"
```

`compare --case` 可从历史补测文件选出不等长双路或单个检索 Case；`supplemental`
仍表示原先包含 ledger 的整套负载。本归档只记录优化前基线，没有估算候选版本的收益。
