"""Synthetic-checkpoint generator for end-to-end weight-loading parity tests.

Builds tiny checkpoints whose *layout* (tensor names, dtypes, the per-arch
transforms the loader applies) mirrors the real supported models, but with
small randomly-filled tensors. The loader materializes them; the harness then
checks parity (see run_parity.py) — catching ABI / materialize / dequant /
transcode / fusion / packing / TP bugs without multi-hundred-GB downloads.

No torch / safetensors dependency: we write the safetensors container by hand
(8-byte LE header length + JSON header + concatenated tensor bytes) with numpy.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dtypes import BYTES_BY_NAME as _DTYPE_BYTES  # safetensors name -> element bytes


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str  # safetensors tag

    def nbytes(self) -> int:
        n = _DTYPE_BYTES[self.dtype]
        for d in self.shape:
            n *= d
        return n


@dataclass
class Template:
    """A synthetic model: a HF config dict + the tensors the loader expects."""
    name: str          # template id (for fixtures / test ids)
    model_type: str    # config.json model_type (drives the loader's arch path)
    config: dict
    tensors: list[TensorSpec] = field(default_factory=list)
    # runtime_quant to pass to the driver for the transcode path ("" or "mxfp4").
    runtime_quant: str = ""


def _fill(spec: TensorSpec, rng: np.random.Generator) -> bytes:
    """Deterministic random bytes for a tensor, valid for its dtype."""
    n_elems = 1
    for d in spec.shape:
        n_elems *= d
    if spec.dtype in ("F8_E4M3", "F8_E5M2"):
        b = rng.integers(0, 256, size=n_elems, dtype=np.uint8)
        # Avoid E4M3 NaN encodings (S.1111.111 => 0x7f / 0xff) so dequant is sane.
        nan_mask = (b & 0x7F) == 0x7F
        b[nan_mask] &= 0x7E
        return b.tobytes()
    if spec.dtype in ("BF16", "F16"):
        b = rng.integers(0, 0x10000, size=n_elems, dtype=np.uint16)
        if spec.dtype == "BF16":
            # Avoid inf/NaN exponent (all-ones) for clean dequant/quant.
            exp_all_ones = ((b >> 7) & 0xFF) == 0xFF
            b[exp_all_ones] &= 0xBFFF
        return b.tobytes()
    if spec.dtype == "F32":
        # Block-scale tensors (_scale_inv): positive, modest magnitude.
        v = rng.uniform(0.25, 3.0, size=n_elems).astype("<f4")
        return v.tobytes()
    if spec.dtype == "I8":
        return rng.integers(-128, 128, size=n_elems, dtype=np.int8).tobytes()
    if spec.dtype == "U8":
        return rng.integers(0, 256, size=n_elems, dtype=np.uint8).tobytes()
    if spec.dtype == "I32":
        return rng.integers(-1000, 1000, size=n_elems, dtype="<i4").tobytes()
    if spec.dtype == "I64":
        return rng.integers(-1000, 1000, size=n_elems, dtype="<i8").tobytes()
    raise ValueError(f"unhandled dtype {spec.dtype}")


def write_checkpoint(tmpl: Template, out_dir: Path, seed: int = 0) -> Path:
    """Write config.json + a single-shard safetensors + index into out_dir.

    Returns out_dir. Tensor data is deterministic given (template, seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- safetensors container (one shard) ---
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for spec in tmpl.tensors:
        data = _fill(spec, rng)
        assert len(data) == spec.nbytes(), (spec.name, len(data), spec.nbytes())
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": spec.shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors wants 8-byte alignment of the data section (pad the header).
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad

    shard = "model-00001-of-00001.safetensors"
    with open(out_dir / shard, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)

    # --- index ---
    weight_map = {spec.name: shard for spec in tmpl.tensors}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": offset}, "weight_map": weight_map})
    )

    # --- config.json ---
    (out_dir / "config.json").write_text(json.dumps(tmpl.config, indent=1))

    return out_dir
