"""pie_bridge — Python access to the pie-bridge wire format.

The wire schema lives in Rust (`driver/bridge/src/schema.rs`); this
module is a thin facade over the PyO3 `_native` extension built by
maturin from that crate.

Read path (server side, parsing requests from the runtime):

    import pie_bridge as pb

    f = pb.Frame.parse(payload)
    if f.payload.kind == pb.REQUEST_FORWARD:
        fr = f.payload.as_forward()
        token_ids = fr.token_ids          # zero-copy numpy view
        for i in range(fr.samplers_len):
            s = fr.samplers_at(i)
            ...

Write path (server side, building responses):

    out = pb.build_status_response(driver_id=7, status=0)
    out = pb.build_forward_response(driver_id=7, num_requests=1,
                                    tokens_indptr=..., tokens=..., ...)

The `build_*` helpers use ctypes to construct `Pie<T>Desc` POD
descriptors and call `pie_build_<type>` directly — there's no PyO3
builder API yet, but the ctypes path is fine: numpy arrays passed in
get their data pointers handed through (zero-copy) and the rkyv
encode step is the same kernel either way.
"""

from __future__ import annotations

# ============================================================================
# PyO3 extension — read path + shmem ring (the bulk of the API).
# ============================================================================

from pie_bridge._native import (  # noqa: F401
    # Top-level frames + payloads
    Frame, ResponseFrame, RequestPayload, ResponsePayload,
    # Concrete request / response variants
    ForwardRequest, ForwardResponse, CopyRequest, AdapterRequest,
    StatusResponse, AdapterBinding,
    # Sampler tagged union + unit enums
    Sampler, CopyDir, CopyResource, AdapterOp,
    # Shmem ring transport
    ShmemServer, Lease,
    # Schema handshake hash (compared against the server's header).
    SCHEMA_HASH,
)

# ============================================================================
# Discriminant constants. Mirror the Rust schema enum variant order
# (see `driver/bridge/src/schema.rs`). Kept here as plain ints so
# callers can write `if kind == pb.REQUEST_FORWARD:` without reaching
# into `_native`.
# ============================================================================

REQUEST_FORWARD = 0
REQUEST_COPY = 1
REQUEST_ADAPTER = 2
REQUEST_HEALTH = 3

RESPONSE_FORWARD = 0
RESPONSE_STATUS = 1

SAMPLER_MULTINOMIAL = 0
SAMPLER_TOP_K = 1
SAMPLER_TOP_P = 2
SAMPLER_MIN_P = 3
SAMPLER_TOP_K_TOP_P = 4
SAMPLER_EMBEDDING = 5
SAMPLER_DIST = 6
SAMPLER_RAW_LOGITS = 7
SAMPLER_LOGPROB = 8
SAMPLER_LOGPROBS = 9
SAMPLER_ENTROPY = 10

COPY_D2H = 0
COPY_H2D = 1
COPY_D2D = 2
COPY_H2H = 3
COPY_RESOURCE_KV = 0
COPY_RESOURCE_RS = 1

ADAPTER_LOAD = 0
ADAPTER_SAVE = 1
ADAPTER_ZO_INIT = 2
ADAPTER_ZO_UPDATE = 3

# ============================================================================
# Write path — ctypes-based builders.
#
# Constructs `Pie<T>Desc` POD structs from Python and calls the C ABI
# `pie_build_<type>` to produce rkyv-archived bytes. Used by Python
# subprocess drivers to encode `ForwardResponse` / `StatusResponse`
# replies back to the runtime.
#
# numpy arrays are passed by their underlying data pointer (zero-copy
# into the descriptor); Python lists fall back to a one-shot copy
# into a `ctypes` array.
# ============================================================================

import ctypes as _ct
import os as _os
import sys as _sys
from pathlib import Path as _Path
from typing import Optional as _Optional

try:
    import numpy as _np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False

_NP_U32 = _np.dtype(_np.uint32) if _HAVE_NUMPY else None
_NP_F32 = _np.dtype(_np.float32) if _HAVE_NUMPY else None
_NP_U8 = _np.dtype(_np.uint8) if _HAVE_NUMPY else None


def _candidate_library_paths() -> list[_Path]:
    override = _os.environ.get("PIE_BRIDGE_LIB")
    if override:
        return [_Path(override)]
    if _sys.platform == "darwin":
        names = ["libpie_bridge.dylib"]
    elif _sys.platform == "win32":
        names = ["pie_bridge.dll", "libpie_bridge.dll"]
    else:
        names = ["libpie_bridge.so"]
    here = _Path(__file__).resolve()
    candidates: list[_Path] = []
    for ancestor in [_Path.cwd(), *here.parents]:
        for profile in ("release", "debug"):
            for name in names:
                candidates.append(ancestor / "target" / profile / name)
    return candidates


def _load_library() -> _ct.CDLL:
    last: _Optional[Exception] = None
    for path in _candidate_library_paths():
        try:
            if path.exists():
                lib = _ct.CDLL(str(path))
                if hasattr(lib, "pie_build_response_frame"):
                    return lib
        except OSError as e:
            last = e
    try:
        lib = _ct.CDLL("libpie_bridge.so")
        if hasattr(lib, "pie_build_response_frame"):
            return lib
    except OSError as e:
        last = e
    raise RuntimeError(
        "pie_bridge: could not locate libpie_bridge built with the `cabi` "
        "feature. Set PIE_BRIDGE_LIB to the absolute path. Last error: "
        f"{last}"
    )


_LIB = _load_library()
_c_size_t = _ct.c_size_t

# Slot header sizes — must match the C struct layout in
# `include/pie_bridge.h`. `tests/test_cabi.py` verifies the offsets.
class _PieAdapterBindingDesc(_ct.Structure):
    _fields_ = [
        ("adapter_id", _ct.c_int64),  # -1 = unbound
        ("seed", _ct.c_int64),        # -1 = no caller seed
    ]


class _PieSamplerDesc(_ct.Structure):
    _fields_ = [
        ("kind", _ct.c_uint8),
        ("temperature", _ct.c_float),
        ("seed", _ct.c_uint32),       # 0 = use fresh per-fire seed
        ("k", _ct.c_uint32),
        ("p", _ct.c_float),
        ("num_tokens", _ct.c_uint32),
        ("token_id", _ct.c_uint32),
        ("token_ids_ptr", _ct.POINTER(_ct.c_uint32)),
        ("token_ids_len", _c_size_t),
    ]


class _PieBrleDesc(_ct.Structure):
    _fields_ = [
        ("buffer_ptr", _ct.POINTER(_ct.c_uint32)), ("buffer_len", _c_size_t),
        ("total_size", _ct.c_uint64),
    ]


class _PieForwardRequestDesc(_ct.Structure):
    # Empty default-constructed instance is sufficient for the
    # builders below (we only fill it for non-Forward Frame variants).
    _fields_ = [
        ("token_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("token_ids_len", _c_size_t),
        ("position_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("position_ids_len", _c_size_t),
        ("kv_page_indices_ptr", _ct.POINTER(_ct.c_uint32)), ("kv_page_indices_len", _c_size_t),
        ("kv_page_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("kv_page_indptr_len", _c_size_t),
        ("kv_last_page_lens_ptr", _ct.POINTER(_ct.c_uint32)), ("kv_last_page_lens_len", _c_size_t),
        ("qo_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("qo_indptr_len", _c_size_t),
        ("rs_slot_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("rs_slot_ids_len", _c_size_t),
        ("rs_slot_flags_ptr", _ct.POINTER(_ct.c_uint8)), ("rs_slot_flags_len", _c_size_t),
        ("masks_ptr", _ct.POINTER(_PieBrleDesc)), ("masks_len", _c_size_t),
        ("mask_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("mask_indptr_len", _c_size_t),
        ("logit_masks_ptr", _ct.POINTER(_PieBrleDesc)), ("logit_masks_len", _c_size_t),
        ("logit_mask_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("logit_mask_indptr_len", _c_size_t),
        ("sampling_indices_ptr", _ct.POINTER(_ct.c_uint32)), ("sampling_indices_len", _c_size_t),
        ("sampling_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("sampling_indptr_len", _c_size_t),
        ("samplers_ptr", _ct.POINTER(_PieSamplerDesc)), ("samplers_len", _c_size_t),
        ("sampler_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("sampler_indptr_len", _c_size_t),
        ("adapter_bindings_ptr", _ct.POINTER(_PieAdapterBindingDesc)), ("adapter_bindings_len", _c_size_t),
        ("spec_token_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_token_ids_len", _c_size_t),
        ("spec_position_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_position_ids_len", _c_size_t),
        ("spec_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_indptr_len", _c_size_t),
        ("output_spec_flags_ptr", _ct.POINTER(_ct.c_uint8)), ("output_spec_flags_len", _c_size_t),
        ("context_ids_ptr", _ct.POINTER(_ct.c_uint64)), ("context_ids_len", _c_size_t),
        ("single_token_mode", _ct.c_uint8),
        ("has_user_mask", _ct.c_uint8),
    ]


class _PieForwardResponseDesc(_ct.Structure):
    _fields_ = [
        ("num_requests", _ct.c_uint32),
        ("tokens_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("tokens_indptr_len", _c_size_t),
        ("tokens_ptr", _ct.POINTER(_ct.c_uint32)), ("tokens_len", _c_size_t),
        ("dists_req_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("dists_req_indptr_len", _c_size_t),
        ("dists_kv_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("dists_kv_indptr_len", _c_size_t),
        ("dists_ids_ptr", _ct.POINTER(_ct.c_uint32)), ("dists_ids_len", _c_size_t),
        ("dists_probs_ptr", _ct.POINTER(_ct.c_float)), ("dists_probs_len", _c_size_t),
        ("logits_req_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("logits_req_indptr_len", _c_size_t),
        ("logits_byte_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("logits_byte_indptr_len", _c_size_t),
        ("logits_bytes_ptr", _ct.POINTER(_ct.c_uint8)), ("logits_bytes_len", _c_size_t),
        ("logprobs_req_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("logprobs_req_indptr_len", _c_size_t),
        ("logprobs_val_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("logprobs_val_indptr_len", _c_size_t),
        ("logprobs_values_ptr", _ct.POINTER(_ct.c_float)), ("logprobs_values_len", _c_size_t),
        ("entropies_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("entropies_indptr_len", _c_size_t),
        ("entropies_ptr", _ct.POINTER(_ct.c_float)), ("entropies_len", _c_size_t),
        ("spec_indptr_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_indptr_len", _c_size_t),
        ("spec_tokens_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_tokens_len", _c_size_t),
        ("spec_positions_ptr", _ct.POINTER(_ct.c_uint32)), ("spec_positions_len", _c_size_t),
    ]


class _PieCopyRequestDesc(_ct.Structure):
    _fields_ = [
        ("dir", _ct.c_uint8),
        ("srcs_ptr", _ct.POINTER(_ct.c_uint32)), ("srcs_len", _c_size_t),
        ("dsts_ptr", _ct.POINTER(_ct.c_uint32)), ("dsts_len", _c_size_t),
        ("resource", _ct.c_uint8),
    ]


class _PieAdapterRequestDesc(_ct.Structure):
    _fields_ = [
        ("op", _ct.c_uint8),
        ("adapter_id", _ct.c_uint64),
        ("path_ptr", _ct.POINTER(_ct.c_uint8)), ("path_len", _c_size_t),
    ]


class _PieStatusResponseDesc(_ct.Structure):
    _fields_ = [("status", _ct.c_int32)]


class _PieRequestPayloadDesc(_ct.Structure):
    _fields_ = [
        ("kind", _ct.c_uint8),
        ("forward", _PieForwardRequestDesc),
        ("copy", _PieCopyRequestDesc),
        ("adapter", _PieAdapterRequestDesc),
    ]


class _PieResponsePayloadDesc(_ct.Structure):
    _fields_ = [
        ("kind", _ct.c_uint8),
        ("forward", _PieForwardResponseDesc),
        ("status", _PieStatusResponseDesc),
    ]


class _PieFrameDesc(_ct.Structure):
    _fields_ = [
        ("driver_id", _ct.c_uint32),
        ("payload", _PieRequestPayloadDesc),
    ]


class _PieResponseFrameDesc(_ct.Structure):
    _fields_ = [
        ("driver_id", _ct.c_uint32),
        ("aborted", _ct.c_uint8),
        ("payload", _PieResponsePayloadDesc),
    ]


_pie_build_frame = _LIB.pie_build_frame
_pie_build_frame.restype = _c_size_t
_pie_build_frame.argtypes = [_ct.POINTER(_PieFrameDesc), _ct.POINTER(_ct.c_uint8), _c_size_t]

_pie_build_response_frame = _LIB.pie_build_response_frame
_pie_build_response_frame.restype = _c_size_t
_pie_build_response_frame.argtypes = [_ct.POINTER(_PieResponseFrameDesc), _ct.POINTER(_ct.c_uint8), _c_size_t]


def _array_ptr(buf, c_type, np_dtype):
    ptr_type = _ct.POINTER(c_type)
    if buf is None:
        return ptr_type(), None, 0
    if _HAVE_NUMPY and isinstance(buf, _np.ndarray):
        arr = buf
        if arr.dtype != np_dtype or not arr.flags.c_contiguous:
            arr = _np.ascontiguousarray(arr, dtype=np_dtype)
        if arr.size == 0:
            return ptr_type(), arr, 0
        return arr.ctypes.data_as(ptr_type), arr, int(arr.size)

    n = len(buf)
    if n == 0:
        return ptr_type(), None, 0
    arr = (c_type * n)(*buf)
    return _ct.cast(arr, ptr_type), arr, n


def _u32_ptr(buf):
    return _array_ptr(buf, _ct.c_uint32, _NP_U32)


def _f32_ptr(buf):
    return _array_ptr(buf, _ct.c_float, _NP_F32)


def _u8_ptr(buf):
    return _array_ptr(buf, _ct.c_uint8, _NP_U8)


def _try_build(build_fn, desc_ptr, initial_cap=256, max_cap=16 * 1024 * 1024):
    cap = initial_cap
    while True:
        out = (_ct.c_uint8 * cap)()
        n = build_fn(desc_ptr, out, cap)
        if n > 0:
            return bytes(out[:n])
        cap *= 4
        if cap > max_cap:
            raise RuntimeError(f"build failed at {max_cap // 1024}KiB buffer")


def build_status_response(driver_id: int, status: int, aborted: bool = False) -> bytes:
    """Build a `ResponseFrame { driver_id, aborted, payload:
    StatusResponse { status } }` rkyv-archived buffer.
    """
    desc = _PieResponseFrameDesc()
    desc.driver_id = driver_id
    desc.aborted = 1 if aborted else 0
    desc.payload.kind = RESPONSE_STATUS
    desc.payload.status.status = status
    return _try_build(_pie_build_response_frame, _ct.byref(desc))


def build_health_request(driver_id: int) -> bytes:
    """Build a `Frame { driver_id, payload: Health }` rkyv-archived
    buffer. The runtime uses this for the periodic liveness probe.
    """
    desc = _PieFrameDesc()
    desc.driver_id = driver_id
    desc.payload.kind = REQUEST_HEALTH
    return _try_build(_pie_build_frame, _ct.byref(desc))


def build_forward_response(
    driver_id: int,
    num_requests: int,
    *,
    tokens_indptr=None, tokens=None,
    dists_req_indptr=None, dists_kv_indptr=None, dists_ids=None, dists_probs=None,
    logits_req_indptr=None, logits_byte_indptr=None, logits_bytes=None,
    logprobs_req_indptr=None, logprobs_val_indptr=None, logprobs_values=None,
    entropies_indptr=None, entropies=None,
    spec_indptr=None, spec_tokens=None, spec_positions=None,
) -> bytes:
    """Build a `ResponseFrame { driver_id, aborted: false, payload:
    Forward(ForwardResponse) }` from per-field arrays. Each array
    may be `None` (interpreted as empty), a numpy array (zero-copy
    pointer hand-off), or a plain Python list.
    """
    desc = _PieResponseFrameDesc()
    desc.driver_id = driver_id
    desc.aborted = 0
    desc.payload.kind = RESPONSE_FORWARD
    desc.payload.status.status = 0  # unused for this variant

    fwd = desc.payload.forward
    fwd.num_requests = num_requests
    owners = []

    def _pair(name, buf, ptr_fn):
        ptr, owner, n = ptr_fn(buf)
        if owner is not None:
            owners.append(owner)
        setattr(fwd, name + "_ptr", ptr)
        setattr(fwd, name + "_len", n)

    _pair("tokens_indptr",       tokens_indptr,       _u32_ptr)
    _pair("tokens",              tokens,              _u32_ptr)
    _pair("dists_req_indptr",    dists_req_indptr,    _u32_ptr)
    _pair("dists_kv_indptr",     dists_kv_indptr,     _u32_ptr)
    _pair("dists_ids",           dists_ids,           _u32_ptr)
    _pair("dists_probs",         dists_probs,         _f32_ptr)
    _pair("logits_req_indptr",   logits_req_indptr,   _u32_ptr)
    _pair("logits_byte_indptr",  logits_byte_indptr,  _u32_ptr)
    _pair("logits_bytes",        logits_bytes,        _u8_ptr)
    _pair("logprobs_req_indptr", logprobs_req_indptr, _u32_ptr)
    _pair("logprobs_val_indptr", logprobs_val_indptr, _u32_ptr)
    _pair("logprobs_values",     logprobs_values,     _f32_ptr)
    _pair("entropies_indptr",    entropies_indptr,    _u32_ptr)
    _pair("entropies",           entropies,           _f32_ptr)
    _pair("spec_indptr",         spec_indptr,         _u32_ptr)
    _pair("spec_tokens",         spec_tokens,         _u32_ptr)
    _pair("spec_positions",      spec_positions,      _u32_ptr)

    return _try_build(_pie_build_response_frame, _ct.byref(desc), initial_cap=4096)


__all__ = [
    # Re-exports from _native
    "Frame", "ResponseFrame", "RequestPayload", "ResponsePayload",
    "ForwardRequest", "ForwardResponse", "CopyRequest", "AdapterRequest",
    "StatusResponse", "AdapterBinding",
    "Sampler", "CopyDir", "CopyResource", "AdapterOp",
    "ShmemServer", "Lease", "SCHEMA_HASH",
    # Discriminant constants
    "REQUEST_FORWARD", "REQUEST_COPY", "REQUEST_ADAPTER", "REQUEST_HEALTH",
    "RESPONSE_FORWARD", "RESPONSE_STATUS",
    "SAMPLER_MULTINOMIAL", "SAMPLER_TOP_K", "SAMPLER_TOP_P", "SAMPLER_MIN_P",
    "SAMPLER_TOP_K_TOP_P", "SAMPLER_EMBEDDING", "SAMPLER_DIST",
    "SAMPLER_RAW_LOGITS", "SAMPLER_LOGPROB", "SAMPLER_LOGPROBS", "SAMPLER_ENTROPY",
    "COPY_D2H", "COPY_H2D", "COPY_D2D", "COPY_H2H",
    "COPY_RESOURCE_KV", "COPY_RESOURCE_RS",
    "ADAPTER_LOAD", "ADAPTER_SAVE", "ADAPTER_ZO_INIT", "ADAPTER_ZO_UPDATE",
    # Build helpers
    "build_status_response", "build_health_request", "build_forward_response",
]
