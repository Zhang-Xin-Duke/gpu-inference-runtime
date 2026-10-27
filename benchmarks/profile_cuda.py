import argparse
from pathlib import Path
import torch
from gpu_runtime import Runtime, ModelSpec
from gpu_runtime.models import DenseModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available(): p.error("CUDA is required")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    with Runtime(device="cuda", max_batch=64, max_inflight=2, delay_ms=1) as r:
        r.register("v1", ModelSpec(DenseModel(4096, 2048, 32), "dense", 4096, 32))
        x = torch.randn(8, 4096)
        for _ in range(10): r.submit("v1", x, timeout_ms=10000).result(15)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            handles = [r.submit("v1", x, timeout_ms=10000) for _ in range(96)]
            for h in handles: h.result(15)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.output))


if __name__ == "__main__": main()
