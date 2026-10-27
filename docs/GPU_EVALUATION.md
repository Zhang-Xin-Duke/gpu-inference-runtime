# GPU evaluation procedure

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

The first pass uses small runs to identify load-generator or model limits. Increase offered load only while recording generator lag, rejection and expiry. A throughput increase under a saturated workload is different from a latency change below saturation. Use multiple seeds and repetitions before drawing a stable comparative conclusion.

## Trace

`benchmarks/profile_cuda.py` captures CPU launch activity and CUDA kernels/copies using PyTorch Profiler. Open the exported Chrome trace in a compatible local trace viewer. The runtime's event timeline complements the profiler; queued stream operations alone do not establish overlap.

## Cleanup

Download results, verify the files and their checksums locally, then stop or delete only the temporary instance created for this evaluation. Confirm no running instance remains and record billed usage separately from the original price estimate.
