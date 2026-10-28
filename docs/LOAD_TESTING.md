# Load and stress testing

This evaluation separates input-packing efficiency, single-GPU service behavior
and request-level replica scaling. A fast packing kernel can coexist with a
CPU-bound service, so each layer has its own baseline and measurements.

## Workload matrix

| Layer | Controlled comparison | Recorded outputs |
|---|---|---|
| Packing | CPU expansion/upload versus compact upload/fused GPU expansion; repeated device copies versus one kernel | Synchronized wall time, stream intervals, captured device paths, transferred bytes |
| Feature service | CPU versus fused packing for the same feature schema, model and arrivals | Deadline goodput, latency, rejection, expiry, errors, queueing and stage costs |
| Recovery | Continuous normal, overload, then normal arrivals through the same runtime | Outcomes by arrival phase, completion timeline, pending work and memory |
| Stability | Five minutes of steady arrivals after warmup | Memory progression, outcomes, latency and resource counters |
| Replicas | One versus two GPUs on the same host with the same model and arrival seeds | Goodput, latency, router/worker accounting and device telemetry |

The extended packing matrix contains 23 configurations per seed. It varies
request counts (1–128), samples per request (1–128), feature width and shared
feature ratio (0/8 through 7/8), including unequal request sizes. Both CPU and
GPU paths use reusable storage. The device-copy comparison additionally uses
CUDA Graph capture to reduce repeated Python submission gaps; it is distinct
from the end-to-end packing measurement.

Feature load uses eight 128-column features, alternating shared/sample inputs,
32 samples per request, a maximum batch of 128 samples and a 1,000 ms deadline.
The default model is `DenseModel(1024, 32768, 32)`: 1,108,410,400 float32
parameters. Replica load uses `DenseModel(256, 32768, 32)` with 1,083,244,576
parameters. These are seeded validation models. Results from these two
workloads are reported separately.

## Arrival generation and metric definitions

`extended.py` generates Poisson arrivals in a process separate from execution.
Identical seeds and durations produce identical offered arrivals for each
matched comparison. CPU/fused and one/two-replica order reverse on the second
seed. Model loading, extension compilation and warmup precede the arrival window.

- **Offered:** all scheduled arrivals, including generator/IPC drops.
- **Generator stale / IPC full:** load-generator limitations, recorded separately
  from runtime rejection. Generator P99 lag makes these limitations visible.
- **Deadline goodput:** requests whose results complete by the original arrival
  deadline, divided by the configured arrival duration.
- **Latency P95/P99:** admitted terminal completions, including expiry and errors;
  immediate admission rejections are counted separately.
- **Transport delay:** IPC or network-path delay before backend submission.
- **Admission time:** backend submission overhead, distinct from model execution.
- **Drain:** time needed to bring all emitted requests to terminal outcomes after
  the final arrival. It is recorded separately from the arrival window.

Overload is expected to produce rejection. The checks require every emitted
request to reach a terminal outcome and every request reservation to be released.
They do not require an overloaded service to accept every offered request.

## Continuous overload and recovery

The default three-phase run offers 150 requests/s for 30 seconds, 1,800/s for
30 seconds, then 150/s for 60 seconds. It uses the same runtime and model for
all three phases. Phase outcomes are grouped by request arrival time; a request
can complete after its phase ends. The timeline instead shows completion rate
at the observation time. This distinction avoids attributing drained overload
requests to the recovery phase.

The stability run offers 150 requests/s for 300 seconds. Observations record
process RSS, process CPU time, queue and staging counts, and Torch allocated and
reserved CUDA memory. GPU-wide telemetry separately records utilization, memory,
power and clocks. Process CPU deltas are core-seconds per elapsed second, not a
percentage normalized to the host's full CPU count. Allocated/reserved CUDA
memory and total GPU-reported memory have different ownership scopes.

Compare memory after warmup and across later windows. Allocator caching and
capped trace storage can increase RSS without proving a leak. A five-minute run
is evidence for its recorded duration, not a long-term production guarantee.

## Replica scaling

Both configurations use the same dual-GPU host and local binary HTTP. The second
worker can remain loaded but receives no benchmark requests in the one-replica
case. Each GPU holds a complete model. The actual router RPC pool has 16 threads
for one worker and 32 for two, so the comparison includes RPC concurrency as
well as the additional GPU. Report it as deployment goodput scaling.

GPU-wide counters, worker completion totals and model fingerprints verify the
tested configuration. Provider-supplied device visibility is not evidence of
exclusive bare-metal hardware. CPU quota and software versions accompany the
results. CPU affinity is not pinned.

## Reproduce

Use a host with two visible NVIDIA GPUs and a CUDA compiler compatible with
PyTorch. The script uses existing devices and never rents or purchases resources.

```bash
pip install '.[test,bench,cuda-build]'
MAX_JOBS=2 python -m pytest -q
python benchmarks/stress_suite.py --cell-seconds 20 --soak-seconds 300 \
  --seeds 7 17 --output-dir results/stress
python benchmarks/summarize_stress.py --root results/stress
```

Source hashes, exact commands, timestamps and partial output files are saved
during execution. The default suite contains 46 packing configurations across
two seeds and 35 load runs: 16 feature cells, two continuous recovery runs, one
stability run and 16 replica cells. Its configured load windows total 1,180
seconds, excluding compilation, model loading, warmup and packing microbenchmarks.
