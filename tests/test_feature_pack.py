import pytest
import torch

from gpu_runtime.feature_pack import FeatureSchema, FeatureStaging, reference_pack, load_cuda_packer
from gpu_runtime.models import DenseModel
from gpu_runtime.runtime import Runtime, ModelSpec


def requests(schema, rows=(1, 3, 2)):
    generator = torch.Generator().manual_seed(71)
    # Strided tensors test normalization of non-contiguous inputs as well.
    return [tuple(torch.randn(1 if shared else n, dim * 2, generator=generator)[:, ::2]
                  for dim, shared in zip(schema.widths, schema.shared)) for n in rows]


def test_feature_schema_and_reference():
    schema = FeatureSchema([2, 3, 1], [True, False, True])
    data = requests(schema)
    actual = reference_pack(schema, data)
    assert actual.shape == (6, 6)
    assert schema.widths == (2, 3, 1)
    torch.testing.assert_close(actual[1:4, :2], data[1][0].expand(3, -1))
    torch.testing.assert_close(actual[1:4, 2:5], data[1][1])
    with pytest.raises(ValueError): FeatureSchema((2,), (True,))
    with pytest.raises(ValueError): FeatureSchema((0,), (False,))
    with pytest.raises(ValueError): schema.validate(data[0][:-1])
    with pytest.raises(ValueError): schema.validate((torch.ones(2, 2), *data[0][1:]))


@pytest.mark.parametrize("device,backend", [("cpu", "cpu"),
    pytest.param("cuda", "cpu", marks=pytest.mark.cuda),
    pytest.param("cuda", "cuda", marks=pytest.mark.cuda)])
def test_feature_runtime_matches_direct_model_and_reuses_slots(device, backend):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    schema = FeatureSchema((5, 17, 3), (True, False, True))
    torch.manual_seed(17)
    model = DenseModel(schema.width, 32, 4).eval().to(device)
    data = requests(schema)
    with Runtime(device=device, max_batch=12, delay_ms=20, feature_packing=backend) as runtime:
        runtime.register("features", ModelSpec(model, "features", schema.width, 4,
                                               feature_schema=schema))
        for _ in range(3):
            handles = [runtime.submit("features", *x, timeout_ms=10000) for x in data]
            for x, handle in zip(data, handles):
                with torch.inference_mode():
                    expected = model(reference_pack(schema, [x]).to(device)).cpu()
                torch.testing.assert_close(handle.result(15), expected, rtol=1e-4, atol=1e-5)
        assert runtime.stats()["reserved_bytes"] == 0
        assert runtime.slot_count <= runtime.max_inflight
        assert runtime.pool_bytes <= runtime.staging_bytes
    assert runtime.stats()["staging_bytes"] == 0


def test_feature_validation_and_budget_failures_release_reservation():
    schema = FeatureSchema((3, 5), (False, True))
    with Runtime(max_batch=8, staging_bytes=1, delay_ms=0) as runtime:
        with pytest.raises(ValueError):
            runtime.register("wrong", ModelSpec(DenseModel(), "features", 9, 2,
                                                feature_schema=schema))
        runtime.register("f", ModelSpec(DenseModel(8, 16, 2), "features", 8, 2,
                                        feature_schema=schema))
        with pytest.raises(ValueError):
            runtime.submit("f", torch.ones(2, 3), torch.ones(2, 5))
        assert runtime.stats()["accepted"] == 0
        with pytest.raises(MemoryError):
            runtime.submit("f", *requests(schema, (2,))[0]).result(3)
        assert runtime.stats()["reserved_bytes"] == 0


@pytest.mark.cuda
@pytest.mark.parametrize("widths,shared,rows", [
    ((1, 3), (False, True), (1,)),
    ((5, 257, 3), (True, False, True), (2, 17, 3)),
    ((129, 3, 11), (False, False, False), (33, 1, 7)),
])
def test_fused_kernel_nondefault_stream_and_uneven_requests(widths, shared, rows):
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    load_cuda_packer()
    schema = FeatureSchema(widths, shared)
    staging = FeatureStaging(schema, sum(rows) + 3, torch.device("cuda"))
    data = requests(schema, rows)
    stream = torch.cuda.Stream()
    for _ in range(2):
        n = staging.prepare(data)
        with torch.cuda.stream(stream):
            staging.enqueue(n)
            owned = staging.output[:n].clone()
        stream.synchronize()
        torch.testing.assert_close(owned.cpu(), reference_pack(schema, data), rtol=0, atol=0)
    assert staging.used <= sum(rows) * schema.width
    actual_bytes = sum(t.numel() * t.element_size() for t in
                       (staging.host, staging.source, staging.meta,
                        staging.device_meta, staging.output))
    assert actual_bytes == FeatureStaging.storage_bytes(schema, staging.capacity, True)
