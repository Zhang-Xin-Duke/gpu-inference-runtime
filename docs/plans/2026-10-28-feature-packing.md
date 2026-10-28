# Fused GPU Feature Packing Implementation Plan

**Goal:** Combine variable-size requests containing per-sample and per-request
features into one model-ready GPU tensor with a single packing kernel.

**Architecture:** A feature schema describes column widths and shared features.
Requests retain compact shared values until upload. Reusable bounded staging
holds compact data, segment descriptors and the expanded output. One CUDA kernel
copies sample features and broadcasts shared features into row-major `[B, F]`
model input on the existing upload stream. The current batching scheduler,
deadlines, completion events and result splitting remain responsible for request
lifecycle. A CPU packing backend supplies an executable comparison.

**Tech Stack:** C++20 scheduler, Python/PyTorch adapters, optional JIT-compiled
C++/CUDA extension, pytest, CUDA event timing.

## Choices

- Use one descriptor per request/feature rather than many device copy calls.
- Emit row-major model input to avoid a subsequent feature concatenation.
- Start with float32 and explicit schemas, without sparse lookup or compression.
- Load the optional CUDA extension when registering a fused feature model,
  before accepting its requests; ordinary dense/sequence use needs no NVCC.
- Keep the original CPU expansion path selectable for matched measurements.
- Account for compact staging, descriptor storage and expanded model input;
  this budget does not include weights or intermediate model activations.

## Tasks

1. Add feature schemas, validation, a pure PyTorch reference and layout tests.
2. Add independently written CUDA packing kernel and checked bindings.
3. Integrate feature models into admission, reusable slots, stream dependencies,
   byte accounting and output splitting; preserve existing model APIs.
4. Test uneven request sizes, shared/sample features, shape validation, memory
   budgets, cancellation, stream use and output parity against direct models.
5. Add packing-only and end-to-end matched comparisons; report CPU staging,
   device execution and synchronized wall time separately, not just launch time.
6. Run local regression tests, then CUDA tests and the comparison on an available
   GPU within the existing rental budget; delete any rented instance afterward.
7. Document supported scope and measured scaling bottlenecks. Keep source-system
   incident research and private artifacts outside the public repository.

## Scaling questions

Measure offered load versus on-time throughput, reject/expiry counts, per-worker
batch sizes and outstanding samples, CPU/router saturation and transport cost.
Distinguish local multi-GPU compute scaling from cross-host network scaling.
XLA/shape bucketing addresses execution shapes; it does not establish multi-GPU
scaling by itself.
