"""KV cache format metadata and portable quantize-dequantize append policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


VALID_KV_CACHE_DTYPES = {
    "auto",
    "bf16",
    "bfloat16",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8_per_token_head",
    "fp8_per_token_head",
    "fp4_e2m1",
    "nvfp4",
}


@dataclass(frozen=True)
class KvCacheFormat:
    name: str = "bf16"
    storage_dtype: str = "bf16"
    scheme: str = "native"
    scale_layout: str = "none"
    block_size: int = 0

    @property
    def is_native(self) -> bool:
        return self.scheme == "native"

    @property
    def has_side_scales(self) -> bool:
        return self.scale_layout != "none"

    def kv_bytes_per_page(self, page_size: int, num_kv_heads: int, head_dim: int) -> int:
        storage_head_dim = (head_dim + 1) // 2 if self.scheme == "fp4_block" else head_dim
        elem = {
            "bf16": 2,
            "fp8_e4m3": 1,
            "fp8_e5m2": 1,
            "int8": 1,
            "uint8": 1,
        }[self.storage_dtype]
        return page_size * num_kv_heads * storage_head_dim * elem

    def scale_bytes_per_page(self, page_size: int, num_kv_heads: int, head_dim: int) -> int:
        if self.scale_layout == "none":
            return 0
        scales_per_head = 1
        if self.scale_layout == "per_token_head_block":
            block = self.block_size or 16
            scales_per_head = (head_dim + block - 1) // block
        return page_size * num_kv_heads * scales_per_head * 4

    def total_bytes_per_page(self, page_size: int, num_kv_heads: int, head_dim: int) -> int:
        return 2 * self.kv_bytes_per_page(page_size, num_kv_heads, head_dim) + (
            2 * self.scale_bytes_per_page(page_size, num_kv_heads, head_dim)
        )


def parse_kv_cache_dtype(value: str, activation_dtype: torch.dtype | None = None) -> KvCacheFormat:
    name = (value or "auto").lower()
    if name not in VALID_KV_CACHE_DTYPES:
        raise ValueError(
            f"Invalid kv_cache_dtype: {value!r}. "
            f"Expected one of: {sorted(VALID_KV_CACHE_DTYPES)}"
        )
    if name in {"auto", "bf16", "bfloat16"}:
        return KvCacheFormat()
    if name == "fp8_e4m3":
        return KvCacheFormat(name=name, storage_dtype="fp8_e4m3", scheme="fp8_per_tensor")
    if name == "fp8_e5m2":
        return KvCacheFormat(name=name, storage_dtype="fp8_e5m2", scheme="fp8_per_tensor")
    if name == "int8_per_token_head":
        return KvCacheFormat(
            name=name,
            storage_dtype="int8",
            scheme="int8_per_token_head",
            scale_layout="per_token_head",
        )
    if name == "fp8_per_token_head":
        return KvCacheFormat(
            name=name,
            storage_dtype="fp8_e4m3",
            scheme="fp8_per_token_head",
            scale_layout="per_token_head",
        )
    return KvCacheFormat(
        name=name,
        storage_dtype="uint8",
        scheme="fp4_block",
        scale_layout="per_token_head_block",
        block_size=16,
    )


@dataclass
class KvCacheHandle:
    layers: list[torch.Tensor]
    format: KvCacheFormat

    def page_tensors(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        # Portable fallback stores dequantized native pages only. The method is
        # intentionally shaped for future side buffers so swap code has one
        # place to ask what belongs to a page.
        return (self.layers[layer_idx],)


_REGISTRY: dict[int, KvCacheFormat] = {}


def register_kv_cache(layers: Iterable[torch.Tensor], fmt: KvCacheFormat) -> KvCacheHandle:
    layer_list = list(layers)
    for layer in layer_list:
        _REGISTRY[id(layer)] = fmt
    return KvCacheHandle(layer_list, fmt)


def format_for_tensor(tensor: torch.Tensor) -> KvCacheFormat:
    return _REGISTRY.get(id(tensor), KvCacheFormat())


def quantize_dequantize_for_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    fmt: KvCacheFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fmt.is_native:
        return k, v
    if fmt.scheme == "int8_per_token_head":
        return _int8_per_token_head_qdq(k), _int8_per_token_head_qdq(v)
    if fmt.scheme == "fp8_per_token_head":
        return _fp8_per_token_head_qdq(k, exponent_bits=4, mantissa_bits=3, max_value=448.0), _fp8_per_token_head_qdq(
            v, exponent_bits=4, mantissa_bits=3, max_value=448.0
        )
    if fmt.scheme == "fp8_per_tensor":
        if fmt.storage_dtype == "fp8_e5m2":
            return _fake_fp_qdq(k, 5, 2, 57344.0), _fake_fp_qdq(v, 5, 2, 57344.0)
        return _fake_fp_qdq(k, 4, 3, 448.0), _fake_fp_qdq(v, 4, 3, 448.0)
    if fmt.scheme == "fp4_block":
        return _fp4_block_qdq(k, fmt.block_size or 16), _fp4_block_qdq(v, fmt.block_size or 16)
    return k, v


def _int8_per_token_head_qdq(x: torch.Tensor) -> torch.Tensor:
    out_dtype = x.dtype
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(xf / scale).clamp(-128, 127)
    return (q * scale).to(out_dtype)


def _fp8_per_token_head_qdq(
    x: torch.Tensor,
    *,
    exponent_bits: int,
    mantissa_bits: int,
    max_value: float,
) -> torch.Tensor:
    out_dtype = x.dtype
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True) / max_value
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (_fake_fp_qdq(xf / scale, exponent_bits, mantissa_bits, max_value) * scale).to(out_dtype)


def _fake_fp_qdq(
    x: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    max_value: float,
) -> torch.Tensor:
    out_dtype = x.dtype
    xf = x.float().clamp(-max_value, max_value)
    ax = xf.abs()
    nonzero = ax > 0
    safe = torch.where(nonzero, ax, torch.ones_like(ax))
    exp = torch.floor(torch.log2(safe))
    step = torch.pow(torch.full_like(exp, 2.0), exp - mantissa_bits)
    q = torch.round(ax / step) * step
    q = torch.where(nonzero, q, torch.zeros_like(q)).clamp(0, max_value)
    return (torch.sign(xf) * q).to(out_dtype)


def _fp4_block_qdq(x: torch.Tensor, block_size: int) -> torch.Tensor:
    out_dtype = x.dtype
    xf = x.float()
    orig_shape = xf.shape
    d = orig_shape[-1]
    pad = (block_size - d % block_size) % block_size
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    blocks = xf.reshape(*xf.shape[:-1], -1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 6.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    y = blocks / scale
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=y.dtype,
        device=y.device,
    )
    idx = (y.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q = levels[idx] * torch.sign(y)
    dq = (q * scale).reshape(*xf.shape)
    if pad:
        dq = dq[..., :d]
    return dq.reshape(orig_shape).to(out_dtype)
