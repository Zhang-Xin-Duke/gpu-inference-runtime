"""Length-prefixed metadata plus contiguous little-endian tensor payloads."""
import json
import math
import struct
import numpy as np
import torch

MAX_BODY = 4 << 20
MAX_HEADER = 4096
DTYPES = {"float32": np.dtype("<f4"), "int64": np.dtype("<i8")}


def encode(tensors, **metadata):
    if not 1 <= len(tensors) <= 2: raise ValueError("invalid tensor count")
    descriptions, payloads = [], []
    for tensor in tensors:
        if tensor.device.type != "cpu" or tensor.dtype not in (torch.float32, torch.int64):
            raise ValueError("wire tensors must be CPU float32 or int64")
        dtype = "float32" if tensor.dtype == torch.float32 else "int64"
        array = tensor.detach().contiguous().numpy().astype(DTYPES[dtype], copy=False)
        descriptions.append(dict(shape=list(array.shape), dtype=dtype))
        payloads.append(array.tobytes())
    header = json.dumps(dict(version=1, tensors=descriptions, **metadata), allow_nan=False).encode()
    body = struct.pack("!I", len(header)) + header + b"".join(payloads)
    if len(header) > MAX_HEADER or len(body) > MAX_BODY: raise ValueError("wire size limit exceeded")
    return body


def decode(body):
    if not 4 <= len(body) <= MAX_BODY: raise ValueError("invalid wire body length")
    length = struct.unpack("!I", body[:4])[0]
    if not 0 < length <= MAX_HEADER or 4 + length > len(body): raise ValueError("invalid wire header length")
    header = json.loads(body[4:4+length])
    if not isinstance(header, dict) or header.get("version") != 1: raise ValueError("unsupported wire version")
    descriptors = header.get("tensors")
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= 2: raise ValueError("invalid tensor count")
    tensors, offset = [], 4 + length
    for desc in descriptors:
        if not isinstance(desc,dict): raise ValueError("invalid tensor descriptor")
        shape, dtype = desc.get("shape"), desc.get("dtype")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 2 or
            any(type(n) is not int or n <= 0 for n in shape) or dtype not in DTYPES):
            raise ValueError("invalid tensor schema")
        count = math.prod(shape)
        size = count * DTYPES[dtype].itemsize
        if size > MAX_BODY or offset + size > len(body): raise ValueError("truncated tensor payload")
        array = np.frombuffer(body, dtype=DTYPES[dtype], count=count, offset=offset).reshape(shape).copy()
        tensors.append(torch.from_numpy(array))
        offset += size
    if offset != len(body): raise ValueError("unexpected trailing payload")
    return tuple(tensors), header
