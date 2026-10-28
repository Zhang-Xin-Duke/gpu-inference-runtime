"""Compare feature packing with matched inputs and reusable storage. No rentals."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from gpu_runtime.feature_pack import FeatureSchema, FeatureStaging, reference_pack, load_cuda_packer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--suite", choices=["basic", "extended"], default="basic")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("a CUDA GPU is required")
    if min(args.iterations, args.rounds) <= 0: raise ValueError("positive repeat counts required")
    torch.set_num_threads(1)
    packer = load_cuda_packer()  # compilation excluded
    generator = torch.Generator().manual_seed(args.seed)
    cells = []
    cases = [("small", 4, 4, (16, 32)),
             ("many_features", 32, 16, (16, 32, 64, 32, 16, 64, 32, 16)),
             ("wide_features", 32, 64, (256, 256, 128, 128, 256, 256)),
             ("uneven_requests", 48, None, (32, 64, 128, 32, 64, 128))]
    cases = [(*case, None) for case in cases]
    if args.suite == "extended":
        cases += [(f"requests_{n}", n, 16, (64,)*8, None) for n in (1,4,16,64,128)]
        cases += [(f"samples_{n}", 32, n, (64,)*8, None) for n in (1,4,16,64,128)]
        cases += [(f"width_{w}", 32, 64, (w,)*8, None) for w in (32,128,256,512)]
        cases += [(f"shared_{n}_of_8", 32, 64, (256,)*8,
                   tuple(i < n for i in range(8))) for n in (0,2,4,6,7)]
    for name, count, rows, widths, sharing in cases:
        schema = FeatureSchema(widths, sharing or tuple(i % 2 == 0 for i in range(len(widths))))
        sizes = [rows or (1 + i % 31) for i in range(count)]
        data = [tuple(torch.randn(1 if shared else n, d, generator=generator)
                      for d, shared in zip(schema.widths, schema.shared)) for n in sizes]
        total = sum(sizes)
        expected = reference_pack(schema, data)
        staging = FeatureStaging(schema, total, torch.device("cuda"))
        staging.prepare(data)
        staging.enqueue(total)
        torch.cuda.synchronize()
        torch.testing.assert_close(staging.output.cpu(), expected, rtol=0, atol=0)
        descriptors = staging.meta[:staging.segments].tolist()
        baseline = torch.empty_like(staging.output)
        host = torch.empty_like(expected, pin_memory=True)

        def many_device_copies():
            for src, row, n, col, d, shared in descriptors:
                value = staging.source[src:src + (d if shared else n*d)]
                baseline[row:row+n, col:col+d].copy_(value.view(1 if shared else n, d))

        def fused_device():
            packer.pack(staging.source, staging.device_meta[:staging.segments],
                        staging.output, staging.max_elements)

        def cpu_expand_upload():
            row = 0
            for inputs, n in zip(data, sizes):
                col = 0
                for value, d in zip(inputs, widths):
                    host[row:row+n, col:col+d].copy_(value)
                    col += d
                row += n
            baseline.copy_(host, non_blocking=True)

        def compact_upload_fused():
            staging.prepare(data)
            staging.enqueue(total)

        many_device_copies()
        torch.cuda.synchronize()
        torch.testing.assert_close(baseline.cpu(), expected, rtol=0, atol=0)
        measurements = {}
        methods = {"many_device_copies": many_device_copies, "fused_device": fused_device,
                   "cpu_expand_upload": cpu_expand_upload, "compact_upload_fused": compact_upload_fused}
        for method in methods.values():
            for _ in range(3): method()
        torch.cuda.synchronize()
        # Reverse method order on alternate rounds to reduce simple order bias.
        raw = {key: [] for key in methods}
        for trial in range(args.rounds):
            ordered = list(methods.items())
            if trial % 2: ordered.reverse()
            for key, method in ordered:
                torch.cuda.synchronize()
                before = time.perf_counter()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations): method()
                end.record()
                end.synchronize()
                wall = (time.perf_counter()-before)*1000/args.iterations
                raw[key].append(dict(wall_ms=wall, stream_interval_ms=start.elapsed_time(end)/args.iterations))
        cpu_expand_upload()
        torch.cuda.synchronize()
        torch.testing.assert_close(baseline.cpu(), expected, rtol=0, atol=0)
        for key, values in raw.items():
            measurements[key] = dict(median_wall_ms=statistics.median(v["wall_ms"] for v in values),
                                     median_stream_interval_ms=statistics.median(v["stream_interval_ms"] for v in values),
                                     rounds=values)
        # Capture only device-resident paths to remove repeated Python submission
        # gaps. Report separately from normal dispatch and end-to-end packing.
        graph_timings = {}
        for key, method in (("many_device_copies", many_device_copies), ("fused_device", fused_device)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): method()
            graph.replay()
            torch.cuda.synchronize()
            values = []
            for _ in range(args.rounds):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations): graph.replay()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) / args.iterations)
            graph_timings[key] = dict(median_ms=statistics.median(values), rounds_ms=values)
        cell = dict(name=name, requests=count, samples=total, widths=widths,
                    shared=schema.shared, rows=sizes, segments=staging.segments,
                    expanded_payload_bytes=expected.numel()*4,
                    compact_payload_bytes=staging.used*4,
                    descriptor_bytes=staging.segments*6*8, parity=True, measurements=measurements,
                    captured_device_paths=graph_timings)
        cells.append(cell)
        print(name, json.dumps({k: round(v["median_wall_ms"],4) for k,v in measurements.items()}), flush=True)
    source = Path(__file__).parents[1] / "src/gpu_runtime"
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  cpu_threads=torch.get_num_threads(), iterations=args.iterations, rounds=args.rounds,
                  suite=args.suite, seed=args.seed,
                  timing_scope="stream intervals include GPU idle time caused by host submission; wall times synchronize completion",
                  sources={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob("*") if p.is_file() and p.suffix in (".py", ".cpp", ".cu")},
                  cells=cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__": main()
