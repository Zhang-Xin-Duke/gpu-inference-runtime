# GPU evaluation

Evaluate correctness first, then compare latency and deadline goodput under matched inputs and offered load. The bundled models exercise real inference operations with random weights; these benchmarks measure runtime behavior rather than prediction quality.

## Setup and correctness

Use Python 3.11+, a C++20 compiler and CUDA-enabled PyTorch appropriate for the GPU and driver. Follow PyTorch's [installation selector](https://pytorch.org/get-started/locally/). Build the extension for the selected Python environment.

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install '.[test]'
python -m pytest -q
```

All CUDA-marked tests must execute on a GPU host. They compare dense/variable-length outputs, reuse staging slots and switch sequence buckets. Resolve correctness failures before measuring performance. On Windows, the Unix SIGTERM drain test is skipped; `scripts/build_windows_zig.py` supplies a local compiler fallback when MSVC is unavailable.

## Start with the demo and a smoke test

```bash
python examples/quick_demo.py --device cuda
python benchmarks/extended.py --device cuda --duration 10 --rps 1000 \
    --output results/smoke.json
```

Inspect generator lag, transport delay, admission time, output checks and terminal accounting. The extended harness uses a separate arrival-generator process, a bounded descriptor queue and fixed-memory latency histograms. Timing uses monotonic, high-resolution `perf_counter`; hardware metadata records the selected clock and FP32/TF32 settings.

## Workload matrix

```bash
python scripts/extended_matrix.py --device cuda --kind dense \
    --output-dir results/experiment/dense --cell-seconds 10 --soak-seconds 300
python scripts/extended_matrix.py --device cuda --kind sequence \
    --output-dir results/experiment/sequence --cell-seconds 10 --soak-seconds 300
```

These matrices vary offered load, execution mode, batch limit, in-flight slots, model/sample size and sequence-length distribution. Sequence runs include bucketed/unbucketed padding. Both models include a five-minute soak with resource observations.

`scripts/compute_saturation.py --output-dir results/experiment` adds three-seed workloads using an approximately 2.15B-parameter FP32 MLP. It targets a 32 GB GPU and deliberately overloads admission. Report rejection and expiration alongside goodput; the offered rate is not a sustainable-capacity estimate.

The smaller `scripts/gpu_matrix.sh` runs the original steady/burst, policy and padding comparisons. Use multiple seeds and repetitions before making stable comparative claims. Preserve unfavorable outcomes and distinguish saturated throughput from latency below saturation.

## CUDA trace

```bash
python benchmarks/profile_cuda.py --output results/experiment/trace.json
```

Open the Chrome trace in a compatible viewer. Check real kernel names, stream IDs and copy sizes. Separate streams and queued operations alone do not prove physical overlap; inspect copy/kernel time intersections. The runtime's CUDA-event timeline complements the profiler.

The local [RTX 5090 evaluation](RTX5090_EVALUATION.md) contains genuine CUDA trace metadata but measured zero copy/compute overlap. The earlier Thunder L40 provider lists GPU profilers as incompatible, so its trace remains diagnostic only. Neither result establishes verified physical overlap.

## Reproduction and comparisons

Save raw JSON, hardware/software, configuration, seed, source revision, test output and checksums. Record generator lag, rejection, expiry and drain duration alongside goodput and completion latency. Report actual GPU count and transport conditions when testing replicas; two processes sharing one card do not measure two-GPU scaling.

[RTX 5090 raw data](../results/rtx5090/) includes both five-minute soaks, larger workloads, recorded telemetry and same-GPU process failure/recovery. It also contains wheel build commands and validation logs. [Distributed evaluation](DISTRIBUTED.md) describes replica-count sweeps and controlled stop/restart tests.
