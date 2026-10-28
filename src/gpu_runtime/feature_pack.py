"""Compact feature staging and one-launch GPU expansion into model input."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


@dataclass(frozen=True)
class FeatureSchema:
    """Feature widths; shared features have one row per request, not sample."""

    widths: tuple[int, ...]
    shared: tuple[bool, ...]

    def __post_init__(self):
        object.__setattr__(self, "widths", tuple(self.widths))
        object.__setattr__(self, "shared", tuple(self.shared))
        if (not self.widths or len(self.widths) != len(self.shared)
                or any(type(d) is not int or d <= 0 for d in self.widths)
                or any(type(s) is not bool for s in self.shared)
                or all(self.shared)):
            raise ValueError("positive feature widths and at least one sample feature required")

    @property
    def width(self):
        return sum(self.widths)

    def validate(self, inputs):
        if len(inputs) != len(self.widths):
            raise ValueError("feature count does not match schema")
        rows = None
        for x, dim, shared in zip(inputs, self.widths, self.shared):
            if (x.device.type != "cpu" or x.requires_grad or x.dtype != torch.float32
                    or x.ndim != 2 or x.shape[1] != dim or x.shape[0] <= 0):
                raise ValueError("features require CPU float32 [samples, width] tensors")
            if shared:
                if x.shape[0] != 1:
                    raise ValueError("shared features require exactly one row per request")
            elif rows is None:
                rows = x.shape[0]
            elif rows != x.shape[0]:
                raise ValueError("sample feature row counts must match")
        return rows


def reference_pack(schema, requests):
    """Independent eager reference: expand shared features, concatenate columns."""
    batches = []
    for inputs in requests:
        n = schema.validate(inputs)
        batches.append(torch.cat([x.expand(n, -1) if shared else x
                                  for x, shared in zip(inputs, schema.shared)], dim=1))
    if not batches:
        raise ValueError("at least one request required")
    return torch.cat(batches, dim=0)


@lru_cache(maxsize=1)
def load_cuda_packer():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / "cuda"
    return load(name="gpu_runtime_feature_pack_v1",
                sources=[str(root / "feature_pack.cpp"), str(root / "feature_pack.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


class FeatureStaging:
    """Reusable compact payload, descriptor table and expanded destination.

    Storage is owned by the runtime slot until its final completion event.
    Each descriptor is (source offset, first output row, rows, first output
    column, feature width, shared flag). At least one sample per request bounds
    descriptor capacity by max_batch * feature_count.
    """

    @staticmethod
    def storage_bytes(schema, capacity, fused):
        data = capacity * schema.width * 4
        return 3 * data + 2 * capacity * len(schema.widths) * 6 * 8 if fused else 2 * data

    def __init__(self, schema, capacity, device):
        self.schema, self.capacity = schema, capacity
        self.host = torch.empty(capacity * schema.width, dtype=torch.float32, pin_memory=True)
        self.source = torch.empty_like(self.host, device=device)
        self.meta = torch.empty((capacity * len(schema.widths), 6),
                                dtype=torch.int64, pin_memory=True)
        self.device_meta = torch.empty_like(self.meta, device=device)
        self.output = torch.empty((capacity, schema.width), dtype=torch.float32, device=device)
        self.upload_done = torch.cuda.Event(enable_timing=True)
        self.used = self.segments = self.max_elements = 0

    def prepare(self, requests):
        descriptors, src, row, largest = [], 0, 0, 0
        for inputs in requests:
            n = self.schema.validate(inputs)
            if row + n > self.capacity:
                raise ValueError("feature batch exceeds staging capacity")
            col = 0
            for x, dim, shared in zip(inputs, self.schema.widths, self.schema.shared):
                count = x.numel()
                self.host[src:src + count].copy_(x.reshape(-1))
                descriptors.append((src, row, n, col, dim, int(shared)))
                src += count
                col += dim
                largest = max(largest, n * dim)
            row += n
        if not descriptors:
            raise ValueError("at least one request required")
        self.segments, self.used, self.max_elements = len(descriptors), src, largest
        self.meta[:self.segments].copy_(torch.tensor(descriptors, dtype=torch.int64))
        return row

    def enqueue(self, samples):
        """Enqueue two bulk uploads and one kernel on the current CUDA stream."""
        self.source[:self.used].copy_(self.host[:self.used], non_blocking=True)
        self.device_meta[:self.segments].copy_(self.meta[:self.segments], non_blocking=True)
        self.upload_done.record()
        load_cuda_packer().pack(self.source, self.device_meta[:self.segments],
                               self.output[:samples], self.max_elements)
