import argparse
from pathlib import Path
import torch
from gpu_runtime import Runtime, ModelSpec
from gpu_runtime.models import DenseModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--features', type=int, default=4096)
    p.add_argument('--hidden', type=int, default=2048)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--max-batch', type=int, default=64)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--requests', type=int, default=96)
    args = p.parse_args()
    if not torch.cuda.is_available(): p.error("CUDA is required")
    if min(args.features, args.hidden, args.samples, args.max_batch, args.slots, args.requests) < 1:
        p.error('sizes must be positive')
    if args.samples > args.max_batch:
        p.error('samples must fit max-batch')
    torch.set_num_threads(1)
    torch.manual_seed(7)
    with Runtime(device="cuda", max_batch=args.max_batch, max_inflight=args.slots,
                 max_requests=max(256, args.requests), delay_ms=1) as r:
        r.register("v1", ModelSpec(DenseModel(args.features, args.hidden, 32), "dense", args.features, 32))
        x = torch.randn(args.samples, args.features)
        for _ in range(10): r.submit("v1", x, timeout_ms=10000).result(15)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            handles = [r.submit("v1", x, timeout_ms=10000) for _ in range(args.requests)]
            for h in handles: h.result(15)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.output))


if __name__ == "__main__": main()
