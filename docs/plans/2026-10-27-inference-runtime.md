# GPU Inference Runtime Implementation Plan

**Goal:** Build and publish an independently runnable GPU inference execution subsystem, with bounded admission, compatible request batching, safe asynchronous execution, and measured GPU results.

**Architecture:** A C++20 scheduler owns admission, grouping, queue deadlines and completion accounting. A Python/PyTorch backend owns models, tensors and CUDA events. A process-level service contract supports deployment under a cluster manager; public Kubernetes manifests exercise that contract.

**Tech Stack:** C++20, pybind11, CMake, Python 3.12, PyTorch, pytest, CUDA streams/events, Docker and Kubernetes.

## 1. Native scheduler
- Create `cpp/scheduler.cpp`, `CMakeLists.txt`, and `pyproject.toml`.
- Implement bounded request/byte admission before packing; compatibility groups; size/age/deadline flush; queued expiration and cancellation; in-flight reservations; drain on shutdown.
- Verify overflow, incompatible groups, deadlines, cancellation, and accounting in `tests/test_scheduler.py` with `python -m pytest`.

## 2. Execution backend
- Create `src/gpu_runtime/runtime.py`, `models.py`, and package exports.
- Execute dense and variable-length sequence models. Retain per-request offsets and generation ownership.
- Pool staging slots within explicit byte/slot limits; use separate copy, compute and output CUDA streams with events. Recycle slots only after output completion.
- Verify against direct model execution, mixed batch sizes, cancellation, exceptions and shutdown using CPU tests and CUDA-marked tests.

## 3. Service integration
- Create `src/gpu_runtime/service.py`, `Dockerfile`, `deploy/kubernetes.yaml`.
- Expose inference, liveness, readiness and metrics; reject new work on drain, finish accepted work on SIGTERM.
- Test the process lifecycle locally. State clearly that Borg deployment itself is unverified and requires access to Google's environment.

## 4. Reproducible evaluation
- Create `benchmarks/run.py` with unbatched, synchronous batching and pipelined configurations; dense and variable-length workloads; open-loop arrival timing, deadline goodput and latency percentiles.
- Record hardware/software, exact commands, correctness checks, raw results and CUDA timeline evidence.
- Provision one temporary GPU only after CPU checks pass. Spend at most USD 25; retrieve outputs and remove the owned instance after testing.

## 5. Documentation and publication
- Write English `README.md` with data-flow/lifecycle diagrams, API examples, design choices, limitations, actual results and references.
- Use the PolyForm Noncommercial license for the independent implementation.
- Initialize Git only inside `engine/`; exclude environments, credentials and all sibling private source/document archives. Audit tracked files, create the public repository, push and verify remote visibility and revision.

## Acceptance checks
- Every accepted request reaches one terminal state, including failures and shutdown.
- Resources for cancelled running work remain reserved until completion.
- Outputs match direct model execution within stated precision tolerances.
- No performance gain is claimed without a reproducible measured result.
- CPU-only and CUDA tests are reported separately; unavailable GPU validation remains explicitly pending.
