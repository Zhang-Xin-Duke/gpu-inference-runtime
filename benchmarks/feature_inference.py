"""Matched end-to-end feature inference with CPU versus fused GPU packing."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from gpu_runtime import Runtime, ModelSpec, FeatureSchema
from gpu_runtime.feature_pack import reference_pack
from gpu_runtime.models import DenseModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    if min(args.iterations, args.rounds) <= 0: raise ValueError("positive repetitions required")
    torch.set_num_threads(1)
    cells = []
    for name, count, samples, widths in [
            ("small", 4, 4, (16, 32)),
            ("many_features", 32, 16, (16, 32, 64, 32, 16, 64, 32, 16)),
            ("wide_features", 32, 64, (256, 256, 128, 128, 256, 256))]:
        schema = FeatureSchema(widths, tuple(i % 2 == 0 for i in range(len(widths))))
        generator = torch.Generator().manual_seed(31)
        data = [tuple(torch.randn(1 if shared else samples, d, generator=generator)
                      for d, shared in zip(widths, schema.shared)) for _ in range(count)]
        trials = {"cpu": [], "cuda": []}
        for trial in range(args.rounds):
            order = ("cpu", "cuda") if trial % 2 == 0 else ("cuda", "cpu")
            for backend in order:
                torch.manual_seed(19)
                model = DenseModel(schema.width, 256, 32)
                with Runtime(device="cuda", max_batch=count*samples, delay_ms=10,
                             max_requests=count*2, feature_packing=backend,
                             staging_bytes=256 << 20) as runtime:
                    runtime.register("features", ModelSpec(model, "features", schema.width, 32,
                                                           feature_schema=schema))
                    def execute():
                        handles = [runtime.submit("features", *x, timeout_ms=10000) for x in data]
                        return [h.result(15) for h in handles]
                    outputs = execute()
                    with torch.inference_mode():
                        for inputs, output in zip(data, outputs):
                            expected = model(reference_pack(schema, [inputs]).cuda()).cpu()
                            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
                    for _ in range(3): execute()
                    before = runtime.stats()
                    begin = time.perf_counter()
                    for _ in range(args.iterations): execute()
                    seconds = time.perf_counter()-begin
                    after = runtime.stats()
                    batches = after["batches"]-before["batches"]
                    trials[backend].append(dict(seconds=seconds,
                        requests_per_second=count*args.iterations/seconds,
                        mean_execution_samples=count*samples*args.iterations/batches,
                        stages_ms={k:(after[k]-before[k])/batches for k in
                                   ("pack_ms", "h2d_ms", "gpu_pack_ms", "compute_ms", "d2h_ms")}))
                    assert after["reserved_bytes"] == 0
        result = dict(name=name, requests_per_group=count, samples_per_request=samples,
                      widths=widths, shared=schema.shared, parity=True, trials=trials,
                      median_requests_per_second={key:statistics.median(t["requests_per_second"] for t in value)
                                                  for key,value in trials.items()})
        cells.append(result)
        print(name, result["median_requests_per_second"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        iterations=args.iterations, rounds=args.rounds, hidden=256, outputs=32,
        cpu_threads=1, scope="sequential groups through Runtime; not an arrival-rate capacity sweep",
        cells=cells), indent=2)+"\n")


if __name__ == "__main__": main()
