# Fused GPU feature packing

The feature adapter batches requests that contain both per-sample features and
features shared by every sample in a request. It produces the row-major
`[total_samples, sum(feature_widths)]` tensor consumed by a dense model.

For example, a request with five candidate samples can carry one `[1, 8]` shared
feature and one `[5, 16]` sample feature. The model needs `[5, 24]`. A second
request with two samples adds two rows, producing a combined `[7, 24]` input.
Shared values remain compact during upload and expand on the GPU.

```mermaid
flowchart LR
    R[Requests: shared rows + sample rows] --> C[Compact CPU staging]
    C --> U[Bulk payload upload + descriptor upload]
    U --> K[One CUDA packing kernel]
    K --> M[Contiguous model input]
    M --> F[Model forward]
    F --> O[Split output by original sample ranges]
```

## Layout and execution

`FeatureSchema.widths` defines column widths and `shared` identifies one-row
features. All requests for a model version use the same schema. Sample counts
may differ between requests. Inputs are CPU float32 tensors; strided inputs are
normalized while staging. At least one sample-level feature determines the
request's sample count.

CPU preparation copies each input feature into a compact pinned payload and
builds one descriptor per request/feature rectangle. A descriptor holds source
offset, first destination row, sample count, first destination column, feature
width and whether its source row is shared. Prefix row offsets preserve request
order. The compact payload and descriptor table each use one bulk H2D copy.

The CUDA grid assigns a segment to one or more thread blocks. Adjacent lanes
process adjacent columns. Sample features advance the source row; shared
features repeatedly read their sole source row. Both write the final row-major
destination directly. This avoids a later feature-concatenation pass. All
segments execute in one kernel launch on the runtime's upload stream.

The existing input-ready event makes the compute stream wait for both uploads
and packing. A slot owns compact host/device data, descriptors and expanded
model input until completion. Slot byte accounting includes these extra
buffers. Fused packing uses additional device storage and a descriptor upload;
small batches can be faster with CPU packing. It does not reduce model weights
or intermediate activation memory.

## Usage

```python
from gpu_runtime import FeatureSchema, ModelSpec, Runtime
from gpu_runtime.models import DenseModel

schema = FeatureSchema((8, 16), (True, False))
with Runtime(device="cuda", feature_packing="cuda") as runtime:
    runtime.register("v1", ModelSpec(DenseModel(24, 64, 4), "features", 24, 4,
                                     feature_schema=schema))
    # submit("v1", shared_float32_tensor[1, 8], sample_tensor[B, 16])
```

Run `python examples/feature_inference.py --device cuda` for an executable
example. Set `feature_packing="cpu"` on the same CUDA runtime for CPU expansion
followed by one expanded-input upload. CPU runtimes also support feature models.

The optional extension compiles when a CUDA feature model is registered. Install
a CUDA toolkit compatible with PyTorch, a C++ compiler and Ninja (`pip install
ninja`). `MAX_JOBS=2` bounds compilation parallelism. Ordinary dense and sequence
models do not require this extension. Feature schemas currently use the
in-process Runtime API; the existing HTTP dense/sequence protocol is unchanged.

## Measurement

The extended dual-L40-host evaluation records 23 layouts with two input seeds,
three rounds per method and 20 repetitions per round. The following wall times
include CPU preparation, upload and completed packing; each value is the median
of the two seed-level medians. Only one GPU runs the packing comparison.

| Input layout | CPU expand/upload | Compact upload/GPU pack | CPU / GPU time |
|---|---:|---:|---:|
| Small: 4 requests × 4 samples, widths 16/32 | 0.063 ms | 0.199 ms | 0.32× |
| 32 requests × 64 samples, 8 × 256 columns, half shared | 7.021 ms | 4.595 ms | 1.53× |
| Same dimensions, 6 of 8 features shared | 6.656 ms | 3.028 ms | 2.20× |

In the last layout, compact feature payload is 4,390,912 bytes rather than
16,777,216 expanded bytes, plus 12,288 descriptor bytes. Shared rows travel once
per request and expand on the GPU. The two seeds give 2.09× and 2.30× packing
speedups. Small requests pay more for descriptors, an extra upload and a kernel.
Device staging additionally retains the compact source alongside expanded model
input; lower transfer volume does not imply lower peak GPU memory.

These are packing results. The [load methodology](LOAD_TESTING.md) separately
measures model inference and overload behavior. Raw seed results:
[7](../results/stress-2026-10-28/results/stress/micro/seed-7.json) and
[17](../results/stress-2026-10-28/results/stress/micro/seed-17.json).

```bash
python -m pytest tests/test_feature_pack.py -q
python benchmarks/feature_pack.py --iterations 30 --rounds 3 \
  --output results/feature-packing.json
```

The benchmark compares repeated per-feature device copies with one fused device
kernel, and CPU expansion/upload with compact upload/GPU expansion. All paths
use matched inputs and reusable storage. Report the latter end-to-end packing
comparison when discussing deployment benefit: fewer device launches alone
does not establish faster request processing. Compilation and warmup are
excluded. Wall measurements synchronize completion. CUDA stream intervals can
include idle time while Python submits work, and are not isolated kernel time.

The test suite covers unequal sample counts, first-position shared features,
multiple shared columns, strided inputs, non-default streams, repeated slot use,
direct-model output parity and byte-budget accounting.

For a broader load evaluation, `benchmarks/stress_suite.py` runs 23 packing
layouts per seed, open-loop feature-model arrival sweeps with CPU/fused packing,
continuous normal/overload/recovery phases, a five-minute steady-load run and
same-host one/two-GPU replica sweeps. The arrival generator is a separate
process. It records generator lag and IPC drops separately from service
rejection, deadlines and errors. `benchmarks/summarize_stress.py` validates and
plots those artifacts. Install the plotting dependency with `pip install
'.[bench]'`.

```bash
python benchmarks/stress_suite.py --cell-seconds 20 --soak-seconds 300 \
  --output-dir results/stress
python benchmarks/summarize_stress.py --root results/stress
```
