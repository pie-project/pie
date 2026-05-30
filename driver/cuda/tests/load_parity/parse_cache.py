"""Reader for the materialized-weight artifact cache (PIEWSTOR format, see
driver/cuda/src/loader/weight_artifact_cache.hpp).

Parses a `<key>.weights` file into the FINAL materialized form of every tensor
the loader produced — i.e. after TP slicing, fusion joins, quant/dequant and
packing. Used by the absolute parity checks to compare the device final form
against an independent numpy expectation computed from the synthetic source.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# dtype metadata lives in one place (dtypes.py); these are derived views:
#   _DTYPE: tag -> (numpy view dtype, element bytes)   DTYPE_NAME: tag -> display name
from dtypes import VIEW_BY_TAG as _DTYPE
from dtypes import NAME_BY_TAG as DTYPE_NAME
from dtypes import TAG_BY_NAME


@dataclass
class Tensor:
    name: str
    dtype: int
    shape: list[int]
    raw: bytes  # exact materialized bytes

    @property
    def dtype_name(self) -> str:
        return DTYPE_NAME[self.dtype]

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def as_f32(self) -> np.ndarray:
        """Decode raw bytes to float32 for numeric comparison (BF16/F32 only)."""
        if self.dtype == 0:  # BF16 -> f32 (bf16 in high 16 bits)
            u16 = np.frombuffer(self.raw, dtype="<u2").astype(np.uint32)
            return (u16 << 16).view(np.float32)
        if self.dtype == 2:
            return np.frombuffer(self.raw, dtype="<f4").astype(np.float32)
        raise ValueError(f"as_f32 unsupported for dtype {self.dtype_name}")


class _Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.p = 0

    def take(self, n: int) -> bytes:
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u32(self) -> int:
        return struct.unpack_from("<I", self.b, self._adv(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.b, self._adv(8))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.b, self._adv(4))[0]

    def i64(self) -> int:
        return struct.unpack_from("<q", self.b, self._adv(8))[0]

    def u8(self) -> int:
        return self.b[self._adv(1)]

    def s(self) -> str:
        n = self.u32()
        return self.take(n).decode("utf-8", "replace")

    def _adv(self, n: int) -> int:
        p = self.p
        self.p += n
        return p


def _read_spec(r: _Reader):
    name = r.s()
    dtype = r.u8()
    rank = r.u32()
    shape = [r.i64() for _ in range(rank)]
    r.u8(); r.u8(); r.u8()                 # layout, ownership, parallel
    r.u8(); r.u8(); r.i32(); r.i32()       # quant: format, gran, group, channel
    r.s(); r.s()                           # quant scale_tensor, zero_point_tensor
    r.s()                                  # backing_tensor
    r.i32(); r.i64(); r.i64()              # view_axis, view_start, view_length
    return name, dtype, shape


def read_cache(path: str) -> dict[str, Tensor]:
    """Parse a .weights file -> {name: Tensor} of final materialized tensors."""
    r = _Reader(open(path, "rb").read())
    magic = r.take(8)
    assert magic == b"PIEWSTOR", f"bad magic {magic!r}"
    r.u32()                 # format_version
    r.s()                   # cache_key
    n_owned = r.u64()
    n_views = r.u64()
    n_quant = r.u64()
    r.u64()                 # blob_section_bytes

    owned = []  # (name, dtype, shape, nbytes, blob_offset)
    for _ in range(n_owned):
        name, dtype, shape = _read_spec(r)
        nbytes = r.u64()
        blob_offset = r.u64()
        r.u64()             # checksum
        owned.append((name, dtype, shape, nbytes, blob_offset))
    views = []  # (name, dtype, shape, root_index, byte_offset)
    for _ in range(n_views):
        name, dtype, shape = _read_spec(r)
        root_index = r.u64()
        byte_offset = r.u64()
        views.append((name, dtype, shape, root_index, byte_offset))
    for _ in range(n_quant):
        r.s(); r.u8(); r.s(); r.s(); r.i32(); r.i32()

    blob_base = r.p
    root_bytes: list[bytes] = []
    out: dict[str, Tensor] = {}
    for name, dtype, shape, nbytes, blob_offset in owned:
        start = blob_base + blob_offset
        raw = r.b[start:start + nbytes]
        root_bytes.append(raw)
        out[name] = Tensor(name, dtype, shape, raw)
    for name, dtype, shape, root_index, byte_offset in views:
        elem = _DTYPE[dtype][1]
        n = 1
        for d in shape:
            n *= d
        nbytes = n * elem
        UINT64_MAX = (1 << 64) - 1
        if root_index == UINT64_MAX or root_index >= len(root_bytes):
            raw = b""
        else:
            raw = root_bytes[root_index][byte_offset:byte_offset + nbytes]
        out[name] = Tensor(name, dtype, shape, raw)
    return out


def read_safetensors(path: str) -> dict[str, Tensor]:
    """Parse a .safetensors file -> {name: Tensor} (the synthetic source)."""
    buf = open(path, "rb").read()
    hlen = struct.unpack_from("<Q", buf, 0)[0]
    import json
    header = json.loads(buf[8:8 + hlen])
    base = 8 + hlen
    out: dict[str, Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = Tensor(name, TAG_BY_NAME[meta["dtype"]], list(meta["shape"]),
                           buf[base + s:base + e])
    return out


if __name__ == "__main__":
    import sys
    tensors = read_cache(sys.argv[1])
    print(f"{len(tensors)} final tensors:")
    for name in sorted(tensors):
        t = tensors[name]
        print(f"  {t.dtype_name:8} {t.shape}  {name}")
