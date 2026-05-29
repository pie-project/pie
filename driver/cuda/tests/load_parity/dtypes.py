"""Single source of truth for the dtype metadata the harness shares across the
write side (gen.py), the read side (parse_cache.py) and the compare side
(oracle.py). Keyed by the cache dtype tag (the driver DType enum / PIEWCAC3
order). Everything else derives from the one DTYPES table below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    tag: int          # cache dtype tag (driver DType enum order)
    name: str         # display name == safetensors tag for the writable ones
    np: str           # numpy view dtype for raw bytes
    elem: int         # bytes per element (packed forms expose raw bytes -> 1)
    packed: bool      # sub-byte packed (no meaningful per-element interleave)
    st: bool          # appears in safetensors checkpoints (gen writes these)


# Order == the driver DType enum / PIEWCAC3 tag values. Add a row to extend.
DTYPES: list[DType] = [
    DType(0, "BF16", "bf16", 2, False, True),
    DType(1, "F16", "f16", 2, False, True),
    DType(2, "F32", "<f4", 4, False, True),
    DType(3, "I8", "i1", 1, False, True),
    DType(4, "I32", "<i4", 4, False, True),
    DType(5, "I64", "<i8", 8, False, True),
    DType(6, "U8", "u1", 1, False, True),
    DType(7, "F8_E4M3", "u1", 1, False, True),
    DType(8, "F8_E5M2", "u1", 1, False, True),
    DType(9, "INT4P", "u1", 1, True, False),
    DType(10, "MXFP4P", "u1", 1, True, False),
]

# Derived views (one definition each; consumers import these).
VIEW_BY_TAG = {d.tag: (d.np, d.elem) for d in DTYPES}          # parse_cache raw views
NAME_BY_TAG = {d.tag: d.name for d in DTYPES}                  # tag -> display name
TAG_BY_NAME = {d.name: d.tag for d in DTYPES if d.st}          # safetensors name -> tag
BYTES_BY_NAME = {d.name: d.elem for d in DTYPES if d.st}       # gen: name -> element bytes
# Logical element bytes for shard reassembly; None for packed (can't interleave).
ELEM_BY_TAG = {d.tag: (None if d.packed else d.elem) for d in DTYPES}
