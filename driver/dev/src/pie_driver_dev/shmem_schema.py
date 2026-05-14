"""Zero-copy flat schema for the shmem fast path.

Iteration 2 implemented `parse_request` (Rust → Python).
Iteration 3 adds `write_response` (Python → Rust) — the common-case
token-only response is laid out as a tiny flat header + concatenated
tokens, eliminating the `create_responses` Python loop and msgpack
serialization on the worker side.

Response wire format:

    [Header: 16 bytes]
      0:  u32 magic    = 0x42504953 ('BPIS')
      4:  u32 mode     0 = flat token-only schema
                       1 = msgpack-fallback (advanced path)
      8:  u32 num_requests
      12: u32 total_tokens

    [Mode 0 body]
      u32[num_requests] : per-request token count
      u32[total_tokens] : concatenated tokens

    [Mode 1 body]
      msgpack-encoded {"results": [...]}

Request layout matches `runtime/src/shmem_schema.rs`. We build numpy views
into the shmem buffer (no copy) and assemble a kwargs dict that
`_handle_fire_batch` already expects (msgpack-decoded bytes are reinterpreted
via np.frombuffer in `Batch.__init__`; we provide compatible bytes-typed
values).
"""

from __future__ import annotations

import numpy as np

MAGIC = 0x42504951  # 'BPIQ'
# v2 added A_PREDICT_FLAGS (u8 per request) for pass-level speculative
# execution. See SPECULATIVE_EXECUTION_DESIGN.md.
SCHEMA_VERSION = 2

# Response schema constants
RESP_MAGIC = 0x42504953  # 'BPIS'
RESP_HEADER_SIZE = 16
RESP_MODE_FLAT = 0
RESP_MODE_MSGPACK = 1
HEADER_SIZE = 512
NUM_ARRAYS = 29
FIXED_HEADER = 32

# Array indices (must match Rust).
A_TOKEN_IDS = 0
A_POSITION_IDS = 1
A_KV_PAGE_INDICES = 2
A_KV_PAGE_INDPTR = 3
A_KV_LAST_PAGE_LENS = 4
A_QO_INDPTR = 5
A_FLATTENED_MASKS = 6
A_MASK_INDPTR = 7
A_LOGIT_MASKS = 8
A_LOGIT_MASK_INDPTR = 9
A_SAMPLING_INDICES = 10
A_SAMPLING_INDPTR = 11
A_SAMPLER_TEMPERATURES = 12
A_SAMPLER_TOP_K = 13
A_SAMPLER_TOP_P = 14
A_SAMPLER_MIN_P = 15
A_SAMPLER_TYPES = 16
A_SAMPLER_SEEDS = 17
A_REQUEST_NUM_SAMPLERS = 18
A_SAMPLER_LABEL_IDS = 19
A_SAMPLER_LABEL_INDPTR = 20
A_ADAPTER_INDICES = 21  # i64
A_ADAPTER_SEEDS = 22  # i64
A_SPEC_TOKEN_IDS = 23
A_SPEC_POSITION_IDS = 24
A_SPEC_INDPTR = 25
A_OUTPUT_SPEC_FLAGS = 26  # u8
A_CONTEXT_IDS = 27  # u64
A_PREDICT_FLAGS = 28  # u8 (v2)

ELEM_SIZE = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 1, 8, 1,
]


def parse_request(view: memoryview) -> dict:
    """Parse a serialized BatchedForwardPassRequest from the given memoryview.

    Returns a dict whose keys match what `_handle_fire_batch` expects (i.e.
    the same keys msgpack would produce from the existing wire format).

    Byte buffers in the returned dict are *views* into the shmem region —
    Python must not retain them past the time the slot is released, but
    they are consumed by `Batch(args, ...)` which copies what it needs into
    numpy arrays before returning from the handler.
    """
    if len(view) < HEADER_SIZE:
        raise ValueError(f"buffer too small ({len(view)} < {HEADER_SIZE})")

    raw = view if isinstance(view, memoryview) else memoryview(view)
    # numpy view of the header to read u32s/u64s
    header = np.frombuffer(raw, dtype=np.uint8, count=HEADER_SIZE)

    magic = int(header[0:4].view(np.uint32)[0])
    if magic != MAGIC:
        raise ValueError(f"shmem schema magic mismatch: 0x{magic:08x} != 0x{MAGIC:08x}")
    schema_version = int(header[4:8].view(np.uint32)[0])
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"shmem schema version mismatch: {schema_version} != {SCHEMA_VERSION}")
    device_id = int(header[8:12].view(np.uint32)[0])
    flags = int(header[12:16].view(np.uint32)[0])
    num_arrays = int(header[16:20].view(np.uint32)[0])
    if num_arrays != NUM_ARRAYS:
        raise ValueError(f"shmem num_arrays {num_arrays} != {NUM_ARRAYS}")

    single_token_mode = bool(flags & 1)

    # Read offset/len table.
    table = header[FIXED_HEADER:FIXED_HEADER + NUM_ARRAYS * 8].view(np.uint32)
    offsets = table[0::2].copy()  # small (<256B) — copy is fine
    lens = table[1::2].copy()

    def slice_bytes(idx: int) -> bytes:
        off = int(offsets[idx])
        n = int(lens[idx]) * ELEM_SIZE[idx]
        # Return as bytes (immutable copy) — mirrors what msgpack delivers.
        # The Batch constructor calls np.frombuffer(data, dtype=...) which
        # works with bytes or buffer-protocol types. To keep zero-copy, we
        # return a memoryview slice; Batch's np.frombuffer(memoryview) is
        # zero-copy.
        return raw[off:off + n]

    _I64_MIN = np.iinfo(np.int64).min

    def slice_adapter_seeds(idx: int) -> list:
        off = int(offsets[idx])
        n = int(lens[idx])
        if n == 0:
            return []
        arr = np.frombuffer(raw, dtype=np.int64, count=n, offset=off)
        # Common case: all sentinel ⇒ all None.
        if (arr == _I64_MIN).all():
            return [None] * n
        # Otherwise convert via tolist() (one batched call into numpy C).
        py = arr.tolist()
        return [None if v == _I64_MIN else v for v in py]

    def slice_bool_array(idx: int) -> list:
        off = int(offsets[idx])
        n = int(lens[idx])
        if n == 0:
            return []
        arr = np.frombuffer(raw, dtype=np.uint8, count=n, offset=off)
        # Fast paths for all-False / all-True.
        if not arr.any():
            return [False] * n
        if arr.all():
            return [True] * n
        return arr.astype(bool).tolist()

    def slice_u64_array(idx: int) -> list:
        off = int(offsets[idx])
        n = int(lens[idx])
        if n == 0:
            return []
        arr = np.frombuffer(raw, dtype=np.uint64, count=n, offset=off)
        return arr.tolist()  # batched

    # Adapter indices: -1 sentinel.
    def slice_adapter_indices(idx: int) -> list:
        off = int(offsets[idx])
        n = int(lens[idx])
        if n == 0:
            return []
        arr = np.frombuffer(raw, dtype=np.int64, count=n, offset=off)
        if (arr == -1).all():
            return [None] * n
        py = arr.tolist()
        return [None if v == -1 else v for v in py]

    # Match the kwargs that _handle_fire_batch / Batch expects.
    args = {
        "token_ids": slice_bytes(A_TOKEN_IDS),
        "position_ids": slice_bytes(A_POSITION_IDS),
        "kv_page_indices": slice_bytes(A_KV_PAGE_INDICES),
        "kv_page_indptr": slice_bytes(A_KV_PAGE_INDPTR),
        "kv_last_page_lens": slice_bytes(A_KV_LAST_PAGE_LENS),
        "qo_indptr": slice_bytes(A_QO_INDPTR),
        "flattened_masks": slice_bytes(A_FLATTENED_MASKS),
        "mask_indptr": slice_bytes(A_MASK_INDPTR),
        "logit_masks": slice_bytes(A_LOGIT_MASKS),
        "logit_mask_indptr": slice_bytes(A_LOGIT_MASK_INDPTR),
        "sampling_indices": slice_bytes(A_SAMPLING_INDICES),
        "sampling_indptr": slice_bytes(A_SAMPLING_INDPTR),
        "sampler_temperatures": slice_bytes(A_SAMPLER_TEMPERATURES),
        "sampler_top_k": slice_bytes(A_SAMPLER_TOP_K),
        "sampler_top_p": slice_bytes(A_SAMPLER_TOP_P),
        "sampler_min_p": slice_bytes(A_SAMPLER_MIN_P),
        "sampler_types": slice_bytes(A_SAMPLER_TYPES),
        "sampler_seeds": slice_bytes(A_SAMPLER_SEEDS),
        "request_num_samplers": slice_bytes(A_REQUEST_NUM_SAMPLERS),
        "sampler_label_ids": slice_bytes(A_SAMPLER_LABEL_IDS),
        "sampler_label_indptr": slice_bytes(A_SAMPLER_LABEL_INDPTR),
        "adapter_indices": slice_adapter_indices(A_ADAPTER_INDICES),
        "adapter_seeds": slice_adapter_seeds(A_ADAPTER_SEEDS),
        "spec_token_ids": slice_bytes(A_SPEC_TOKEN_IDS),
        "spec_position_ids": slice_bytes(A_SPEC_POSITION_IDS),
        "spec_indptr": slice_bytes(A_SPEC_INDPTR),
        "output_spec_flags": slice_bool_array(A_OUTPUT_SPEC_FLAGS),
        "context_ids": slice_u64_array(A_CONTEXT_IDS),
        "predict_flags": slice_bool_array(A_PREDICT_FLAGS),  # v2
        "single_token_mode": single_token_mode,
        "device_id": device_id,
    }
    return args


_struct_header_pack = None  # lazy init

# Sampler types that produce non-token outputs (dist / logits / logprobs /
# entropies). Presence of any of these in `batch.sampler_types` forces the
# msgpack-fallback path because the flat token-only schema can't carry
# their payloads.
_SPECIAL_SAMPLERS = frozenset({0, 7, 8, 9, 10})


def write_response_v2(sampling_results, batch, dst_buf, msgpack_module=None) -> int:
    """Direct shmem encode bypassing `Batch.create_responses`.

    Detects the fast path (no special-typed samplers, no spec-verify, no
    spec-out drafts) and writes counts + tokens straight into the shmem
    response buffer. Falls back to building a result dict via
    `create_responses` and msgpack-encoding it for the advanced path.

    Saves the per-request Python `ForwardPassResponse` and dict allocations
    that `create_responses` + the calling site otherwise pay for every
    iteration.
    """
    global _struct_header_pack
    if _struct_header_pack is None:
        import struct as _struct
        _struct_header_pack = _struct.Struct("<IIII").pack_into

    sampler_types = batch.sampler_types  # list[int]
    spec_accepted = sampling_results.get("spec_accepted_tokens")
    spec_tokens_all = sampling_results.get("spec_tokens")

    fast_path = (
        spec_accepted is None
        and (spec_tokens_all is None
             or not any(t is not None for t in spec_tokens_all))
        and not any(t in _SPECIAL_SAMPLERS for t in sampler_types)
    )

    if not fast_path:
        # Slow path: build the dict via create_responses, msgpack-encode it.
        responses = batch.create_responses(sampling_results)
        results = [
            {
                "tokens": resp.tokens,
                "dists": resp.dists,
                "logits": getattr(resp, "logits", []) or [],
                "logprobs": getattr(resp, "logprobs", []) or [],
                "entropies": getattr(resp, "entropies", []) or [],
                "spec_tokens": getattr(resp, "spec_tokens", []) or [],
                "spec_positions": getattr(resp, "spec_positions", []) or [],
            }
            for resp in responses
        ]
        return write_response(
            {"results": results},
            dst_buf,
            msgpack_module,
        )

    # Fast path: counts come straight from `request_output_counts`, tokens
    # from `sampling_results['tokens']` (already in flat order).
    counts = batch.request_output_counts
    tokens = sampling_results["tokens"]

    if not isinstance(counts, np.ndarray) or counts.dtype != np.uint32:
        counts = np.asarray(counts, dtype=np.uint32)
    if not isinstance(tokens, np.ndarray) or tokens.dtype != np.uint32:
        tokens = np.asarray(tokens, dtype=np.uint32)

    n_req = counts.size
    n_tokens = tokens.size
    body_size = (n_req + n_tokens) * 4
    total = RESP_HEADER_SIZE + body_size
    dst_capacity = len(dst_buf)
    if total > dst_capacity:
        raise ValueError(
            f"shmem response (flat-v2) {total} > dst {dst_capacity}; "
            f"raise the SHMEM_RESP_BUF constant in worker.py / device.rs"
        )

    _struct_header_pack(
        dst_buf, 0, RESP_MAGIC, RESP_MODE_FLAT, n_req, n_tokens
    )
    counts_off = RESP_HEADER_SIZE
    tokens_off = counts_off + n_req * 4
    if n_req > 0:
        dst_buf[counts_off:tokens_off] = counts.tobytes()
    if n_tokens > 0:
        dst_buf[tokens_off:total] = tokens.tobytes()

    return total


def write_response(result: dict, dst_buf, msgpack_module=None) -> int:
    """Write a worker `result` dict directly into the shmem response buffer.

    `result` is the return value of `_handle_fire_batch`, shaped as
    `{"results": [{"tokens": [...], "dists": [...], ...}, ...]}`.
    `dst_buf` is a writable buffer-protocol object (e.g.
    `(c_uint8 * N).from_address(...)`).

    Returns: number of bytes written.
    """
    global _struct_header_pack
    if _struct_header_pack is None:
        import struct as _struct
        _struct_header_pack = _struct.Struct("<IIII").pack_into

    results = result["results"]
    n_req = len(results)

    # Single-pass advanced detection + counts + flat token list.
    # Cheaper than two passes (one to detect, one to count). The truthy
    # checks short-circuit on common-case empty advanced fields.
    counts = [0] * n_req
    flat_tokens: list = []
    flat_extend = flat_tokens.extend
    advanced = False
    for i, r in enumerate(results):
        if (r["dists"] or r["logits"] or r["logprobs"] or r["entropies"]
                or r["spec_tokens"] or r["spec_positions"]):
            advanced = True
            break
        t = r["tokens"]
        counts[i] = len(t)
        flat_extend(t)

    dst_capacity = len(dst_buf)

    if advanced:
        # Fallback path. Defer to msgpack of the full dict.
        if msgpack_module is None:
            import msgpack as msgpack_module
        body = msgpack_module.packb(result)
        body_len = len(body)
        total = RESP_HEADER_SIZE + body_len
        if total > dst_capacity:
            raise ValueError(
                f"shmem response (msgpack-fallback) {total} > dst {dst_capacity}; "
                f"raise the SHMEM_RESP_BUF constant in worker.py / device.rs"
            )
        _struct_header_pack(dst_buf, 0, RESP_MAGIC, RESP_MODE_MSGPACK, n_req, 0)
        dst_buf[RESP_HEADER_SIZE:total] = body
        return total

    # Token-only flat path.
    total_tokens = len(flat_tokens)
    body_size = (n_req + total_tokens) * 4
    total = RESP_HEADER_SIZE + body_size
    if total > dst_capacity:
        raise ValueError(
            f"shmem response (flat) {total} > dst {dst_capacity}; "
            f"raise the SHMEM_RESP_BUF constant in worker.py / device.rs"
        )

    _struct_header_pack(
        dst_buf, 0, RESP_MAGIC, RESP_MODE_FLAT, n_req, total_tokens
    )
    counts_off = RESP_HEADER_SIZE
    tokens_off = counts_off + n_req * 4

    # Single numpy array build + frombuffer assignment for the body. We
    # avoid creating a numpy view of dst_buf (which has noticeable overhead
    # per call); ctypes Arrays support slice-assignment from a bytes-like.
    if n_req > 0:
        counts_bytes = np.array(counts, dtype=np.uint32).tobytes()
        dst_buf[counts_off:tokens_off] = counts_bytes
    if total_tokens > 0:
        tokens_bytes = np.array(flat_tokens, dtype=np.uint32).tobytes()
        dst_buf[tokens_off:total] = tokens_bytes

    return total
