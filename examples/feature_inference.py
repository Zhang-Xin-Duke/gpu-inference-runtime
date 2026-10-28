"""A shared user feature plus per-sample features form one GPU model input."""
import argparse
import torch
from gpu_runtime import Runtime, ModelSpec, FeatureSchema
from gpu_runtime.models import DenseModel
from gpu_runtime.feature_pack import reference_pack

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
schema = FeatureSchema((8, 16), (True, False))
model = DenseModel(schema.width, 32, 4)
requests = [(torch.randn(1, 8), torch.randn(n, 16)) for n in (2, 5, 3)]
with Runtime(device=args.device, max_batch=16, delay_ms=10) as runtime:
    runtime.register("v1", ModelSpec(model, "features", schema.width, 4, feature_schema=schema))
    handles = [runtime.submit("v1", *inputs, timeout_ms=10000) for inputs in requests]
    for inputs, handle in zip(requests, handles):
        with torch.inference_mode():
            expected = model(reference_pack(schema, [inputs]).to(args.device)).cpu()
        torch.testing.assert_close(handle.result(15), expected, rtol=1e-4, atol=1e-5)
    print("feature inference parity passed", runtime.stats())
