# Distributed inference and extended evaluation

**Goal:** Evaluate longer and larger workloads, add two-host inference replicas with bounded routing, and put measured behavior at the front of the README.

**Architecture:** Keep the C++/PyTorch runtime local to each GPU worker. Add a bounded binary HTTP transport and a CPU request router that sends complete inference requests to identical model replicas. Benchmark one versus two remote workers through the same router, and retain a separate-process local load generator for single-worker saturation/soak runs.

**Tech stack:** Existing C++20/PyTorch runtime, Python standard HTTP/concurrency/multiprocessing libraries, NumPy tensor framing, pytest, SSH forwarding, Matplotlib.

## Decisions

- Use request-level data parallelism: each GPU has a complete model and independent batching queue. This fits the stateless inference boundary and permits independent failure. Splitting one model across GPUs would require a different execution/data-transfer contract and is outside this change.
- Use binary tensor payloads to avoid serializing thousands of floats as JSON during throughput tests. Keep the JSON demonstration endpoint compatible.
- Route by outstanding sample count, cap pending work, verify model fingerprints, retry once on a different ready replica when deadline budget permits. Remote work may finish after client timeout; this is not an exactly-once distributed protocol.
- Bind workers to loopback and reach them through SSH forwards. No public inference listener is needed for the experiment.
- Generate timed arrivals in a separate process for local runtime sweeps; count transport queue overflow and generator lag explicitly. Keep histogram memory bounded and collect periodic resource observations for soak tests.

## Implementation and checks

1. `wire.py`, service extensions: validate framing, tensor shapes, model identity and binary output; test malformed bodies and CPU parity.
2. `distributed.py`: bounded routing, version validation, deadlines, failover and counters; test with two local CPU workers and a failed node.
3. `benchmarks/extended.py`: duration-based independent-process generator, histograms, input catalog, resource samples and exact terminal accounting; CPU smoke test before rental.
4. `benchmarks/distributed.py`: one/two-node load sweeps and failure/recovery recording; compare matched model, request shape, offered load and deadline.
5. Prepare all remote commands before provisioning. Attempt two single-L40 hosts within the existing USD 20 total budget; do not add payment methods or bypass account limits. Record instance ownership, price and control-plane cleanup deadline.
6. Run longer saturation, burst, sequence-length and batch-size sweeps; run a sustained soak; save raw results and checksums. Stop or delete all owned instances after artifact retrieval.
7. Put measured summary plots and a recorded telemetry animation before implementation details in README. Keep original results, disclose generator/network effects, and publish the new code/results after verification.
