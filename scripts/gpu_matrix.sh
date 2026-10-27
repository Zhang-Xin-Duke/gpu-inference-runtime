#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. The parent timeout bounds GPU computation, not billing.
mkdir -p results/gpu
nvidia-smi > results/gpu/nvidia-smi.txt
python -m pytest -q | tee results/gpu/tests.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: stop before benchmarking'
PY
for seed in 7 17 27; do
  timeout 180 python benchmarks/run.py --device cuda --requests 1500 --rps 1000 --seed "$seed" \
      --deadline-ms 100 --output "results/gpu/steady-${seed}.json"
  timeout 180 python benchmarks/run.py --device cuda --requests 1500 --rps 1000 --seed "$seed" \
      --deadline-ms 100 --burst --output "results/gpu/burst-${seed}.json"
done
timeout 180 python benchmarks/run.py --device cuda --requests 1500 --rps 1000 --policy age \
    --deadline-ms 100 --output results/gpu/age-policy.json
timeout 180 python benchmarks/run.py --device cuda --requests 1500 --rps 1000 --no-buckets \
    --deadline-ms 100 --kind sequence --output results/gpu/no-buckets.json
# A larger dense workload complements small-model launch/dispatch overhead tests.
timeout 180 python benchmarks/run.py --device cuda --requests 2000 --rps 2000 \
    --features 4096 --hidden 2048 --samples 4 --kind dense --deadline-ms 100 \
    --output results/gpu/large-dense.json
timeout 120 python benchmarks/profile_cuda.py --output results/gpu/trace.json
python - <<'PY'
from pathlib import Path
import hashlib
files = sorted(p for p in Path('results/gpu').rglob('*') if p.is_file())
Path('results/gpu/SHA256SUMS').write_text(''.join(
    hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p) + '\n' for p in files))
PY
