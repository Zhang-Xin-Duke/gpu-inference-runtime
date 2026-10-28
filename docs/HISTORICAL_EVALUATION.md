# Historical GPU and CPU evaluations

These earlier measurements are retained with their original conditions and raw artifacts. Commands run from the repository root. The current L40 multi-GPU comparison is documented in [Multi-GPU evaluation](MULTI_GPU_EVALUATION.md).

## Single-GPU measurements

**Measured on one local RTX 5090, Windows WDDM, FP32:**

| Check | Result |
|---|---|
| Installed-wheel correctness | **31 passed, 1 skipped**; all three CUDA cases passed; Unix SIGTERM drain test skipped on Windows |
| Three-seed steady/burst comparison | Two-slot batching completed **18,000 / 18,000** requests within 100 ms across both bundled models |
| Five-minute soaks | **300,291 / 300,291** requests on time for each model; request reservations released after drain |
| Highest independent-process load cell | Both pipeline models completed **80,063 / 80,063** requests in 10 s at nominal 8,000 requests/s |
| Large FP32 compute stress | Approximately 2.15B parameters; observed **99% GPU utilization**, 577.2 W peak, 74°C maximum at stock settings |

![RTX 5090 deadline goodput across offered load](../results/rtx5090/load-sweep.svg)

The 91 extended cells include load, batch/slot, model-size, sequence-padding and sustained tests. The highest tested rate is not a calibrated maximum capacity. Local profiler traces contain real kernels/copies on distinct streams but show **zero copy/compute overlap**; higher application goodput does not establish physical overlap. [Full RTX 5090 results, conditions and negative outcomes](../docs/RTX5090_EVALUATION.md) · [Raw data and checksums](../results/rtx5090/).

![Recorded process failover on one RTX 5090](../results/rtx5090/local-process-replicas/process-failover.gif)

Two local worker processes share one physical GPU in this recording. A controlled stop/restart exposed stale HTTP connections across RPC threads; connection invalidation allows the restarted worker to rejoin. Both before/after recordings and deadline losses are retained. The newer [L40 evaluation](../docs/MULTI_GPU_EVALUATION.md) measures one versus two allocated devices and three-replica process recovery separately.

The earlier Thunder L40 measurement is retained in [historical application-level results](https://github.com/Zhang-Xin-Duke/gpu-inference-runtime/blob/01e4b04/README.md#gpu-application-level-results).


### GPU application-level results

The following table is the original Thunder L40 experiment. The newer local workstation evaluation is reported separately in [RTX 5090 results](../docs/RTX5090_EVALUATION.md).

![Original L40 deadline completion](../docs/gpu-deadline-completion.svg)

The GPU run used one Thunder Compute **NVIDIA L40 (48 GB class)** instance, six vCPUs, Python 3.12.13, PyTorch 2.12.1+cu130 and one PyTorch CPU thread. Both dense and sequence outputs passed the direct-model comparisons; all **25 tests passed**, including the CUDA tests. [Raw results and SHA-256 checks](../results/gpu/) accompany the run.

Each steady/burst configuration used 1,500 requests, three seeds (7, 17, 27), a nominal 1,000 requests/s arrival process and a 100 ms deadline. The table aggregates on-time counts; latency, goodput and generator lag are medians of the three per-run summaries. These are application-level results on a virtualized GPU, including host scheduling and framework overhead.

| Traffic / model | Mode | On-time / 4,500 | Goodput (req/s) | P99 (ms) | Generator lag P99 (ms) |
|---|---|---:|---:|---:|---:|
| steady / dense | unbatched | 37 | 7.7 | 108.56 | 6.21 |
| steady / dense | sync | 4,446 | 997.6 | 99.29 | 74.46 |
| steady / dense | pipeline | 4,500 | 1022.1 | 27.87 | 6.03 |
| steady / sequence | unbatched | 54 | 11.6 | 106.86 | 2.79 |
| steady / sequence | sync | 4,305 | 958.4 | 127.22 | 81.32 |
| steady / sequence | pipeline | 4,500 | 992.5 | 82.18 | 6.77 |
| burst / dense | unbatched | 251 | 54.5 | 106.88 | 6.03 |
| burst / dense | sync | 4,268 | 1034.0 | 99.68 | 71.54 |
| burst / dense | pipeline | 4,500 | 1049.9 | 41.39 | 6.70 |
| burst / sequence | unbatched | 406 | 86.0 | 105.81 | 8.45 |
| burst / sequence | sync | 4,096 | 955.4 | 103.62 | 72.88 |
| burst / sequence | pipeline | 4,466 | 1012.0 | 99.98 | 26.73 |

Batching substantially increased the number of requests meeting this deadline compared with executing one request at a time. Two in-flight slots completed all 4,500 steady-traffic requests for each model; burst sequence traffic completed 4,466 / 4,500. The shared-process generator stalled more in several one-slot runs, so these comparisons include host-side scheduling effects and do not isolate a CUDA-overlap benefit.

**Workload-dependent outcomes:** the single-run sequence ablation reduced padded token positions from 384,000 to 125,728 with buckets, while observed P99 increased from 33.47 to 79.36 ms. Reducing padding did not guarantee a latency improvement in this workload. In the larger dense exploratory run (4 samples/request, 4,096 features, hidden width 2,048, nominal 2,000 requests/s), two slots produced 431.5 deadline-goodput requests/s versus 525.3 with one slot; its generator lag P99 reached 123.25 ms. That run cannot establish the runtime's maximum sustainable throughput.

**Profiling boundary:** Thunder [documents GPU virtualization](https://www.thundercompute.com/docs/legacy/prototyping-and-production-modes) and lists [GPU profilers as incompatible](https://www.thundercompute.com/docs/compatibility). The exported trace contains generic kernel names, stream ID 0 and zero-byte copy metadata. The event-based overlap counter reported 0 ms. This run does **not** establish physical transfer/compute overlap; validate that separately on a standard CUDA environment. The repository retains the trace as a diagnostic artifact.

<details>
<summary>Local CPU smoke results</summary>



The checked-in [raw CPU smoke run](../results/cpu-smoke.json) records 300 requests per model/mode, a nominal 500 requests/s arrival process, a 100 ms deadline and one PyTorch CPU thread. This is a functional and instrumentation check; it does not measure maximum sustainable throughput or GPU performance. Arrival randomness and generator scheduling affect this short run.

| Model | Mode | Successful / offered | P50 (ms) | P99 (ms) |
|---|---|---:|---:|---:|
| dense | unbatched | 300 / 300 | 0.70 | 2.49 |
| dense | sync | 300 / 300 | 2.92 | 5.75 |
| dense | pipeline | 300 / 300 | 3.14 | 34.79 |
| sequence | unbatched | 300 / 300 | 0.76 | 3.15 |
| sequence | sync | 300 / 300 | 2.98 | 4.86 |
| sequence | pipeline | 300 / 300 | 2.97 | 4.93 |

The deterministic padding test uses four length-5 sequences and a configured maximum length of 128. Without buckets it prepares 512 token positions; bucket 8 prepares 32, with the same 20 useful tokens and matching outputs.

</details>

