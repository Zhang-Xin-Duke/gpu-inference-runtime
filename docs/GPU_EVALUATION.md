# GPU evaluation procedure

## Completed local RTX 5090 evaluation

The local Windows WDDM run completed the dense/sequence matrices, both five-minute soaks, three-seed large FP32 workloads and a same-card two-process stop/restart experiment. See [results and limitations](RTX5090_EVALUATION.md) and [raw artifacts](../results/rtx5090/). The local traces contain real stream/kernel/copy metadata but show zero measured copy/compute overlap. Two-host/two-GPU scaling remains pending.

The actual matrix used 10-second cells and 300-second soaks. Install CUDA-enabled PyTorch using the official selector, then build the C++20 extension for the local Python; do not reuse another machine's binary. `scripts/build_windows_zig.py` supplies a project-local compiler fallback when MSVC is unavailable. Normal CMake wheel builds can select Ninja and Zig explicitly. All three CUDA cases executed; the Unix SIGTERM drain test is skipped on Windows. Full wheel commands and validation logs accompany the results.

Measurements use high-resolution monotonic `perf_counter`, since this Python build's Windows `monotonic()` implementation has 15.625 ms resolution. Record the selected clock and FP32/TF32 settings with hardware/software metadata. A deadline-flush unit test now seeds a known service estimate instead of depending on OS wake-up jitter being smaller than the 10 ms cold-start estimate.

`scripts/compute_saturation.py` exercises a roughly 2.15B-parameter FP32 MLP without changing precision or driver power limits. It deliberately overloads admission; report rejections and expirations alongside goodput rather than calling the offered rate sustainable capacity.

## Prepare before starting a paid instance

1. Run CPU tests and build the package locally.
2. Prepare a source archive containing only this repository, excluding `.venv`, `.git`, build outputs and credentials.
3. Confirm the instance quote and remaining budget. Use one GPU and the included CPU/disk allocation.
4. Prepare remote setup, correctness, benchmark and artifact download commands. Keep package installation time inside the rental budget.
5. Set an external cleanup deadline. A guest OS shutdown alone may not stop provider billing; confirm deletion or stop in the provider control plane.

## First GPU checks

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install '.[test]'
python -m pytest -q
```

All CUDA-marked tests must execute rather than skip. They compare dense and variable-length model outputs, reuse slots and switch sequence buckets. If dependencies or correctness checks fail, stop performance runs until the cause is understood.

## Measurements

The checked-in `scripts/gpu_matrix.sh` runs the prepared matrix within a wall-clock timeout: steady traffic for both models, burst traffic, deadline-policy ablation and sequence-padding ablation. It saves raw JSON, software/device information, pytest output and a profiler trace under `results/gpu/`.

```bash
bash scripts/gpu_matrix.sh
```

On Thunder Compute, skip GPU-profiler interpretation: the provider lists GPU profilers as incompatible. The application benchmark remains useful, but verify physical stream overlap on a standard CUDA environment. The saved Thunder trace is diagnostic only.

The first pass uses small runs to identify load-generator or model limits. Increase offered load only while recording generator lag, rejection and expiry. A throughput increase under a saturated workload is different from a latency change below saturation. Use multiple seeds and repetitions before drawing a stable comparative conclusion.

## Trace

`benchmarks/profile_cuda.py` captures CPU launch activity and CUDA kernels/copies using PyTorch Profiler. Open the exported Chrome trace in a compatible local trace viewer. The runtime's event timeline complements the profiler; queued stream operations alone do not establish overlap.

## Cleanup

Download results, verify the files and their checksums locally, then stop or delete only the temporary instance created for this evaluation. Confirm no running instance remains and record billed usage separately from the original price estimate.
