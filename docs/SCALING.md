# Scaling request-level GPU inference

Each worker holds a complete model. Scaling adds independent execution capacity
and model copies; there is no gradient synchronization, model sharding or
cross-worker attention state. A CPU router distributes requests to workers,
which form their own local batches.

## Where additional replicas can stop helping

| Constraint | Mechanism | Evidence to collect |
|---|---|---|
| CPU routing and transport | Serialization, connection waits and RPC threads can saturate before the GPUs | Router CPU, generator lag, active calls, connection counts and bytes |
| Smaller local batches | Splitting a fixed arrival stream across more replicas reduces arrivals per worker | Per-worker batch-size distributions, wait time and GPU utilization |
| Unequal request costs | Equal request counts can hide different sample counts and model costs | Outstanding samples, stage times, queue wait and per-worker completions |
| Network transfer | Cross-host input/output transfer can dominate a short forward pass | Payload size, transport latency, local versus cross-host comparison |
| Overload and retries | Repeatedly sending work to full workers can amplify pressure | Offered/accepted/rejected/expired counts, attempt counts and cooldowns |
| Fixed concurrency budgets | A router cap can prevent additional workers receiving sufficient concurrent work | RPC pool size and per-worker active-call limits |

The current least-work policy tracks outstanding sample counts, not just request
counts. Selection and reservation occur under one lock. Each replica has a
bounded shared HTTP connection pool. Admission bounds pending requests, and
each accepted request has one end-to-end deadline. A transport failure can cause
one attempt on another replica within the remaining budget. This can duplicate
execution if the first server received the original request.

Connections also have a lifetime policy: idle sockets retire before the bundled
server's keep-alive timeout, and a reused socket closed by its peer does not
automatically cool down the whole GPU worker. Low-load replica tests exposed
that distinction: old idle sockets could otherwise turn a healthy worker into
an unavailable routing target. New-connection failures still receive cooldown.

One current limit is explicit in `distributed.py`: actual RPC threads are
`min(requested_concurrency, 32, replica_count * 16)`. With default limits, two
replicas can already consume the 32-thread router budget. Adding a third GPU
does not automatically raise it. A larger configurable budget must be evaluated
with CPU and network load before increasing the cap.

## What the existing L40 results establish

The extended same-host one/two-GPU comparison increases on-time request goodput
from 204.25 to 480.13 requests/s at nominal 1,000 offered requests/s for the
recorded 1.08B-parameter float32 MLP. Each configuration uses two 20-second
arrival seeds. Both GPU count and actual RPC concurrency increase. The 2.35x
result includes deadline and overload effects; it is not pure compute scaling
or a universal capacity ratio. At nominal 400/s, both two-GPU seeds complete
all offered requests. See the [extended conditions and failures](STRESS_EVALUATION.md).

The earlier [three-replica cross-host recording](MULTI_GPU_EVALUATION.md) establishes progress on remaining
workers during a process restart, with recorded failures. It does not measure a
three-GPU compute speedup. Local GPU scaling and cross-host scaling need separate
matched comparisons because they stress different paths.

## Shape bucketing and scaling are separate controls

Shape buckets reduce padding or the number of execution shapes within a worker.
They can improve single-replica efficiency, but they do not balance load across
workers. More buckets also split the arrival stream and may reduce batch size.
The runtime currently buckets sequence lengths; it does not integrate an XLA
compiler or claim compiler-derived acceleration.

## Evaluation matrix

The executable [load-testing suite](LOAD_TESTING.md) records these measurements
and includes continuous overload/recovery and stability phases.

Use one, two and three GPUs with the same model, inputs, host allocation,
transport, deadlines and controlled concurrency budgets. Sweep request rate,
samples per request and input width. Run both fixed total arrival rates and
rates proportional to replica count. Record per-worker batch sizes, utilization,
stage times and completion counts as well as aggregate goodput and latency.
Keep generator timing independent, warm up first, rotate configuration order
and use multiple seeds. Isolate controlled stop/restart tests from steady-state
scaling measurements.
