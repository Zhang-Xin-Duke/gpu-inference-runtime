# Execution design

## Boundary

The unit of scheduling is an inference request containing CPU tensors, a registered model version and a relative deadline. The unit of GPU execution is a compatible batch. Each request returns one future whose value is an owned CPU tensor.

The C++ scheduler contains metadata and no framework tensors. This keeps queue operations independent of PyTorch and lets the blocking condition-variable wait release Python's GIL. Python retains model and tensor ownership so CUDA integration does not require binding the full LibTorch C++ distribution into the scheduler extension.

## 1. Admission before packing

`Runtime.submit` validates shape/dtype and derives a compatibility key. `Scheduler.offer` atomically checks the accepted request count and reserved-byte limit under a mutex. A request is either queued with a reservation or rejected before a staging copy is made.

For a dense request with `B` samples, `F` features and `O` outputs, the logical reservation is `B × (2 × F × 4 + O × 4)` bytes. A sequence request substitutes padded tokens and lengths for the dense feature bytes. This estimates two input staging copies and a returned output. It excludes user-owned input memory already allocated before submission.

The pool independently checks full-slot allocation against `staging_bytes`. A request reservation does not guarantee a slot will fit: a configured staging budget smaller than a full slot causes an explicit terminal `MemoryError`. Keeping these two budgets separate makes the distinction between queued work and reusable physical staging visible.

The limits apply to queued plus executing work. Cancelling an executing request leaves its reservation in place until the backend finishes; releasing early could allow new work to reuse storage still accessed by CUDA.

## 2. Compatible grouping and flush

A key has the form `model-version:shape-bucket`. The public adapters use fixed dtypes; each registered version also fixes its input/output contract. A model version cannot be replaced in place, so batches cannot mix generations accidentally.

A single scan accumulates readiness information for all groups, preserving first-seen order for ties. A group becomes eligible when it can fill the sample limit, reaches its oldest request's maximum wait, or approaches a member's deadline. The deadline policy subtracts an EWMA of observed pack-to-completion service time; its cold estimate is 10 ms. There is no hard real-time guarantee: contention, allocator work and model variability can exceed this estimate.

The scheduler prefers the earliest eligible flush time. It never splits a request across batches. Within a compatible group, a smaller request can fit the remaining capacity even if a larger queued request cannot. This improves filling at the cost of strict request FIFO; age and deadline rules still bound the wait before reconsideration.

The wait releases the GIL and sleeps on a condition variable. Offers, cancellations, completion and shutdown wake it. Queue expiration is checked even while all execution slots are occupied.

## 3. Variable-length sequences

For sequence width `L`, the bucket is `min(next_power_of_two(L), max_length)`. Requests in a bucket pack into a common rectangular token tensor; lengths drive the model's attention mask. This reduces wasted padding relative to a global maximum, while generating more groups that may fill less efficiently.

The implementation favors a small explicit adapter over arbitrary ragged tensor graphs. It supports one token tensor, one length vector and one dense output. Additional input schemas should define their own compatibility, packing and output-splitting contracts.

## 4. Staging slots

A slot contains pinned host inputs, device inputs, a pinned output and completion events. It is allocated at the configured maximum batch size for a shape signature, then reused for smaller batches through slices. Only the used sample range is initialized and transferred.

Completed matching slots are reused first. When a new shape needs capacity, completed idle slots can be evicted. Executing slots are never evicted. The number of slots is capped by `max_inflight`; byte limits may reject a shape before the count limit is reached.

Fixed-capacity slots simplify ownership and reduce repeated allocation. They reserve more memory for sparse traffic and can cause allocation churn when many shape buckets alternate. A larger size-class cache is an extension to evaluate with a workload trace; it would need its own eviction and byte accounting.

## 5. CUDA dependency and ownership chain

```mermaid
sequenceDiagram
    participant D as Dispatcher / CPU
    participant C as H2D stream
    participant G as Compute stream
    participant O as D2H stream
    participant F as Completion thread
    D->>D: Lease slot; pack request slices
    D->>C: Enqueue pinned-to-device copies
    C->>G: Record ready event; compute waits
    G->>G: Execute model forward
    G->>O: Record compute event; output waits
    O->>O: Copy output into pinned host slot
    O->>F: Record final event
    D->>D: Pack / launch next batch if slot available
    F->>F: Query final event without blocking dispatcher
    F->>F: Clone per-request results
    F->>D: Return completed slot and reservations
```

Explicit stream waits encode dependencies. The slot stays alive through its last output event. The temporary model output is recorded on the output stream so the CUDA allocator cannot recycle it before the output copy finishes. Model weights and input records remain referenced by each batch.

The completion thread polls the final event with a short sleep. This keeps the dispatcher available but adds polling and host scheduling overhead, particularly for very small models. The implementation uses one compute stream and one output stream; compute kernels from different batches remain ordered, while transfers and CPU packing may overlap them.

Setting `pipeline=False` limits execution to one in-flight batch. It preserves the same copy and completion logic for an ablation, rather than changing both the model implementation and scheduling policy simultaneously.

## 6. Result ownership and failures

Offsets recorded during packing determine output slices. Each returned result is cloned into independent CPU storage before its staging slot is recycled. Returning owned data costs a host copy, but lets callers keep results without pinning a shared slot indefinitely.

- Queued cancellation removes metadata and releases admission immediately.
- Running cancellation marks the future cancelled while retaining execution resources.
- Queued expiration returns a timeout before model execution.
- Execution expiration returns a timeout after safe backend completion.
- A model exception fails all requests in that batch and returns reservations.
- A failed CUDA context closes admission and quarantines uncertain storage. Restart is required; recovering a poisoned device context in place is outside the process contract.

Completion accounting validates the complete native batch before releasing capacity. Partial, duplicate and unknown completion calls fail without modifying reservations. User future callbacks run on the completion thread; they should remain short and should not call blocking shutdown or wait on another request from the same runtime.

## 7. Process and cluster lifecycle

The service process owns its queues and model generations. Registration completes model placement before readiness becomes true. A registered model and submitted input tensors must remain immutable while requests use them.

On SIGTERM, admission closes and queued work becomes immediately eligible for flushing. The process waits for its accepted work, then exits. A cluster grace period can still expire before a hung model returns; the cluster manager can kill the process, and clients must handle lost in-memory work.

The supplied Kubernetes manifest is a deployment example. It requests one GPU and exposes readiness, liveness and startup probes. Testing the HTTP process on a local CPU verifies the application contract, while validating an actual GPU device plugin, image and cluster deployment requires a GPU-enabled cluster. Google Borg integration likewise requires its real job specification and execution environment.

## 8. Measurement boundaries

CPU smoke runs verify the scheduler, result mapping and metric collection. They do not validate pinned-memory behavior, CUDA dependencies, overlap or GPU speedup. CUDA tests and benchmark scripts are kept separately runnable.

The load generator schedules Poisson arrivals against a fixed monotonic origin and records scheduling lag. Goodput counts only successful completions before the planned arrival's deadline. Latency percentiles include accepted terminal requests, including timeouts and failures; rejected requests are counted separately. A shared-process generator can become the bottleneck, which is visible through its lag and `generator_late` count.

CUDA events provide stage durations and relative transfer/compute intervals. The overlap metric compares different batches only. For causally attributing a speedup, compare repeated runs at matched arrival rates and input distributions, inspect saturation and tail latency, and use a profiler trace to verify the measured stream behavior.
