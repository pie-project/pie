"""Shmem wire-format façade — translates the new rkyv schema to the
legacy SoA dict shape that `batching.Batch` expects.

The pie-bridge crate now uses an rkyv-derived schema (Rust struct =
wire format), and samplers are encoded as a `Vec<Sampler>` tagged union
(AoS). The dev driver's `Batch` was written against the older flat-SoA
shape (separate `sampler_temperatures`, `sampler_top_k`, … arrays). This
module bridges those shapes:

  * `parse_request(payload_bytes)` → dict[str, np.ndarray] in the old
    SoA layout, including a `method_tag` and the legacy sampler-type IDs.
  * `ResponseBuilder` accumulates per-request response data (tokens,
    entropies, logprobs, logits, dists) and emits wire bytes via
    `pie_bridge.build_forward_response`.

The sampler-type ID space differs between old and new bridge (the rkyv
enum's variant order is intentionally MULTINOMIAL-first; the legacy
table put DISTRIBUTION first). `_NEW_TO_OLD_SAMPLER` remaps so callers
that compare against old `sampler_type` constants keep working.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import pie_bridge as _pb

# Sampler types that produce non-token outputs (dist / logits / logprobs /
# entropies). Set in legacy IDs so callers can compare against
# `batch.sampler_types`.
_SPECIAL_SAMPLERS = frozenset({
    0,   # DISTRIBUTION (legacy)
    7,   # RAW_LOGITS
    8,   # LOGPROB
    9,   # LOGPROBS
    10,  # ENTROPY
})

# Map new bridge sampler kind (index in the rkyv `Sampler` enum) to the
# legacy `sampler_type` u32 ID space.
#
# New enum order (in `driver/bridge/src/schema.rs`):
#   0 Multinomial, 1 TopK, 2 TopP, 3 MinP, 4 TopKTopP, 5 Embedding,
#   6 Dist, 7 RawLogits, 8 Logprob, 9 Logprobs, 10 Entropy
#
# Legacy IDs (from `pie_bridge.sampler_type.*`):
#   0 DISTRIBUTION, 1 MULTINOMIAL, 2 TOP_K, 3 TOP_P, 4 MIN_P,
#   5 TOP_K_TOP_P, 6 EMBEDDING, 7 RAW_LOGITS, 8 LOGPROB,
#   9 LOGPROBS, 10 ENTROPY
_NEW_TO_OLD_SAMPLER = [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# Parse — new Frame → legacy SoA dict
# ---------------------------------------------------------------------------

def peek_method_tag(payload: bytes) -> int:
    """Return the legacy method tag for `payload` without copying the
    bulk of the request. Used by the worker's polling loop to dispatch
    before the (more expensive) full parse.
    """
    f = _pb.Frame.parse(payload)
    payload_obj = f.payload
    kind = payload_obj.kind
    if kind == _pb.REQUEST_FORWARD:
        return 0  # FORWARD
    if kind == _pb.REQUEST_COPY:
        cr = payload_obj.as_copy()
        if cr is None:
            raise ValueError("REQUEST_COPY frame missing payload")
        # Legacy: COPY_D2H=1, COPY_H2D=2, COPY_D2D=3, COPY_H2H=4 — the
        # CopyDir variant order matches: D2H=0, H2D=1, D2D=2, H2H=3.
        return cr.dir.value + 1
    if kind == _pb.REQUEST_ADAPTER:
        ar = payload_obj.as_adapter()
        if ar is None:
            raise ValueError("REQUEST_ADAPTER frame missing payload")
        # Legacy: LOAD=5, SAVE=6, ZO_INIT=7, ZO_UPDATE=8 — AdapterOp
        # variant order: Load=0, Save=1, ZoInit=2, ZoUpdate=3.
        return ar.op.value + 5
    if kind == _pb.REQUEST_HEALTH:
        return 9  # synthetic HEALTH
    raise ValueError(f"unknown payload_kind {kind}")


def parse_request(payload: bytes | memoryview) -> dict[str, Any]:
    """Convert a forward-request wire payload to the legacy dict shape.

    Raises if the frame isn't a forward request — copy/adapter requests
    use their own decode path (see `worker._handle_copy` / `_handle_load_adapter`).
    """
    if isinstance(payload, memoryview):
        payload = bytes(payload)
    f = _pb.Frame.parse(payload)
    payload_obj = f.payload
    if payload_obj.kind != _pb.REQUEST_FORWARD:
        raise ValueError(
            f"parse_request: expected REQUEST_FORWARD, got payload_kind={payload_obj.kind}"
        )
    fr = payload_obj.as_forward()
    if fr is None:
        raise ValueError("REQUEST_FORWARD frame missing payload")
    return _forward_to_dict(fr, driver_id=f.driver_id)


def _forward_to_dict(fr, *, driver_id: int) -> dict[str, Any]:
    """Walk the rkyv ForwardRequest and produce a SoA dict."""
    # Sampler arrays — walk the tagged-union Vec and dispatch on kind.
    n_samp = fr.samplers_len
    sampler_types: list[int] = []
    temperatures: list[float] = []
    top_k: list[int] = []
    top_p: list[float] = []
    min_p: list[float] = []
    seeds: list[int] = []
    label_ids: list[int] = []
    label_indptr: list[int] = [0]

    for i in range(n_samp):
        s = fr.samplers_at(i)
        new_kind = s.kind
        sampler_types.append(_NEW_TO_OLD_SAMPLER[new_kind])
        # Temperature (defaults to 0.0 for non-temperature samplers).
        # PyO3 exposes scalar fields as #[getter] — accessed without parens.
        temperatures.append(float(s.temperature))
        # k field — only meaningful for TopK / TopKTopP, else 0.
        top_k.append(int(s.k))
        # p field — meaningful for TopP / MinP / TopKTopP. For MinP
        # specifically, legacy stored it in min_p; we duplicate so both
        # legacy arrays carry it.
        p = float(s.p)
        if new_kind == 3:  # MinP
            min_p.append(p)
            top_p.append(0.0)
        else:
            min_p.append(0.0)
            top_p.append(p)
        # Seed — only meaningful for Multinomial. The schema sentinel
        # is `seed == 0` → use a fresh per-fire random seed (matches
        # the legacy "no seed" convention).
        seeds.append(int(s.seed))
        # Logprobs.token_ids — populate label_ids per request.
        if new_kind == 9:  # LOGPROBS
            ids = s.token_ids
            label_ids.extend(int(x) for x in ids)
        label_indptr.append(len(label_ids))

    # Adapter bindings (Vec<AdapterBinding> { adapter_id: i64, seed:
    # i64 }). Schema sentinel: `-1` = unbound for either field. The
    # legacy SoA shape in `batching.py` expects `None` for unbound
    # entries (it does `if adapter_idx is not None`), so translate the
    # sentinel here at the wire boundary.
    n_bind = fr.adapter_bindings_len
    adapter_indices: list[int | None] = []
    adapter_seeds: list[int | None] = []
    for i in range(n_bind):
        b = fr.adapter_bindings_at(i)
        aid = int(b.adapter_id)
        sd = int(b.seed)
        adapter_indices.append(None if aid < 0 else aid)
        adapter_seeds.append(None if sd < 0 else sd)

    # Reconstruct the legacy flat (run buffer + per-row/per-request byte
    # offset) shape from the bridge's Vec<Brle> for both attention and
    # logit masks.
    flat_attn, attn_byte_indptr = _flatten_brle_vec(fr, "masks")
    flat_logit, logit_byte_indptr = _flatten_logit_masks(fr)

    return {
        "method_tag": 0,  # FORWARD
        "driver_id": driver_id,
        "token_ids": np.asarray(fr.token_ids, dtype=np.uint32),
        "position_ids": np.asarray(fr.position_ids, dtype=np.uint32),
        "kv_page_indices": np.asarray(fr.kv_page_indices, dtype=np.uint32),
        "kv_page_indptr": np.asarray(fr.kv_page_indptr, dtype=np.uint32),
        "kv_last_page_lens": np.asarray(fr.kv_last_page_lens, dtype=np.uint32),
        "qo_indptr": np.asarray(fr.qo_indptr, dtype=np.uint32),
        "flattened_masks": flat_attn,
        "mask_indptr": attn_byte_indptr,
        "logit_masks": flat_logit,
        "logit_mask_indptr": logit_byte_indptr,
        "sampling_indices": np.asarray(fr.sampling_indices, dtype=np.uint32),
        "sampling_indptr": np.asarray(fr.sampling_indptr, dtype=np.uint32),
        "spec_token_ids": np.asarray(fr.spec_token_ids, dtype=np.uint32),
        "spec_position_ids": np.asarray(fr.spec_position_ids, dtype=np.uint32),
        "spec_indptr": np.asarray(fr.spec_indptr, dtype=np.uint32),
        "output_spec_flags": list(fr.output_spec_flags),
        "single_token_mode": bool(fr.single_token_mode),
        # Sampler SoA — legacy schema.
        "sampler_types": np.asarray(sampler_types, dtype=np.uint32),
        "sampler_temperatures": np.asarray(temperatures, dtype=np.float32),
        "sampler_top_k": np.asarray(top_k, dtype=np.uint32),
        "sampler_top_p": np.asarray(top_p, dtype=np.float32),
        "sampler_min_p": np.asarray(min_p, dtype=np.float32),
        "sampler_seeds": np.asarray(seeds, dtype=np.uint32),
        "sampler_label_ids": np.asarray(label_ids, dtype=np.uint32),
        "sampler_label_indptr": np.asarray(label_indptr, dtype=np.uint32),
        "request_num_samplers": _per_request_counts(fr.sampler_indptr),
        # Adapter bindings.
        "adapter_indices": adapter_indices,
        "adapter_seeds": adapter_seeds,
    }


def _flatten_brle_vec(fr, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Walk a Vec<Brle> field on `fr` (using `<field>_len` / `<field>_at`)
    and rebuild the legacy `(flattened_buffer, per_row_byte_indptr)` shape.

    The new wire schema stores attention masks as `Vec<Brle>`, one Brle
    per query row. Downstream legacy code still wants the row buffers
    concatenated into one u32 array with per-row byte offsets, so this
    helper performs that translation in Python."""
    n = getattr(fr, f"{field}_len")
    at = getattr(fr, f"{field}_at")
    bufs: list[np.ndarray] = []
    indptr = [0]
    total = 0
    for i in range(n):
        buf = np.asarray(at(i).buffer, dtype=np.uint32)
        bufs.append(buf)
        total += buf.size
        indptr.append(total)
    flat = (
        np.concatenate(bufs)
        if bufs
        else np.empty(0, dtype=np.uint32)
    )
    return flat, np.asarray(indptr, dtype=np.uint32)


def _flatten_logit_masks(fr) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the legacy per-request `(logit_masks, logit_mask_indptr)`
    byte-offset shape from the new wire format's `Vec<Brle>` +
    per-request `logit_mask_indptr`. Each request contributes 0 or 1
    Brle entries."""
    req_indptr = np.asarray(fr.logit_mask_indptr, dtype=np.uint32)
    bufs: list[np.ndarray] = []
    out_indptr = [0]
    total = 0
    if req_indptr.size >= 1:
        for r in range(req_indptr.size - 1):
            lo = int(req_indptr[r])
            hi = int(req_indptr[r + 1])
            for i in range(lo, hi):
                buf = np.asarray(fr.logit_masks_at(i).buffer, dtype=np.uint32)
                bufs.append(buf)
                total += buf.size
            out_indptr.append(total)
    flat = (
        np.concatenate(bufs)
        if bufs
        else np.empty(0, dtype=np.uint32)
    )
    return flat, np.asarray(out_indptr, dtype=np.uint32)


def _per_request_counts(indptr) -> np.ndarray:
    """Legacy `request_num_samplers` was a count-per-request array; new
    wire uses CSR `sampler_indptr`. Recover the counts via differences."""
    a = np.asarray(indptr, dtype=np.uint32)
    if a.size <= 1:
        return np.zeros(0, dtype=np.uint32)
    return np.diff(a).astype(np.uint32)


# ---------------------------------------------------------------------------
# ResponseBuilder — accumulates per-request results, emits bytes
# ---------------------------------------------------------------------------

class ResponseBuilder:
    """Build a forward-pass response payload.

    Mirrors the legacy interface (`reset` / `add_request` / `build`)
    so callers in `worker.py` keep working unchanged. Internally collects
    per-request results, then on `build` flattens into the SoA arrays
    `pie_bridge.build_forward_response` expects.

    Reuse one instance across iterations; `reset` clears state.
    """

    __slots__ = ("_requests",)

    def __init__(self) -> None:
        self._requests: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._requests.clear()

    def add_request(
        self,
        *,
        tokens=None,
        entropies=None,
        logprobs=None,
        logits=None,
        dists=None,
    ) -> None:
        """Append one request's outputs."""
        self._requests.append({
            "tokens": list(tokens) if tokens is not None else [],
            "entropies": list(entropies) if entropies else [],
            "logprobs": [list(v) for v in (logprobs or [])],
            "logits": [bytes(blob) if not isinstance(blob, bytes) else blob
                       for blob in (logits or [])],
            "dists": [(list(ids), list(probs)) for ids, probs in (dists or [])],
        })

    # ----- High-level API (preserves the legacy build() signature) -----

    def build_from_batch(
        self,
        sampling_results: dict,
        batch,
        driver_id: int,
        dst_buf=None,
    ) -> int | bytes:
        """Walk `sampling_results` + `batch`, accumulate per-request
        outputs, and serialize a ResponseFrame. Mirrors the legacy
        `build(sampling_results, batch, dst_buf)` semantics but adds
        `driver_id` (the new wire format embeds it in ResponseFrame).
        """
        sampler_types = batch.sampler_types  # list[int] in legacy IDs
        spec_accepted = sampling_results.get("spec_accepted_tokens")
        spec_tokens_all = sampling_results.get("spec_tokens")

        # Fast path: no special samplers, no spec output.
        fast_path = (
            spec_accepted is None
            and (spec_tokens_all is None
                 or not any(t is not None for t in spec_tokens_all))
            and not any(t in _SPECIAL_SAMPLERS for t in sampler_types)
        )

        self.reset()
        if fast_path:
            self._fill_token_only(
                batch.request_output_counts,
                sampling_results["tokens"],
            )
        else:
            self._fill_full(batch.create_responses(sampling_results))

        return self.build(driver_id, dst_buf=dst_buf)

    def _fill_token_only(self, counts, tokens) -> None:
        """Fast path: one `add_request` per request with just tokens."""
        cursor = 0
        for n in counts:
            n = int(n)
            req_tokens = list(tokens[cursor:cursor + n])
            self.add_request(tokens=req_tokens)
            cursor += n

    def _fill_full(self, responses) -> None:
        """Slow path: walk per-request ForwardPassResponse-like objects."""
        for resp in responses:
            tokens = list(resp.tokens) if not isinstance(resp.tokens, list) else resp.tokens
            entropies = list(getattr(resp, "entropies", None) or [])
            logprobs = [list(v) for v in (getattr(resp, "logprobs", None) or [])]
            logits = [
                bytes(blob) if not isinstance(blob, bytes) else blob
                for blob in (getattr(resp, "logits", None) or [])
            ]
            dists = [
                (list(ids), list(probs))
                for ids, probs in (getattr(resp, "dists", None) or [])
            ]
            self.add_request(
                tokens=tokens,
                entropies=entropies,
                logprobs=logprobs,
                logits=logits,
                dists=dists,
            )

    # ----- Low-level API: flatten + serialize -----

    def build(self, driver_id: int, dst_buf=None) -> int | bytes:
        """Flatten the accumulated requests into a ResponseFrame +
        ForwardResponse and return wire bytes. If `dst_buf` is provided
        (a `memoryview` or `bytearray`), copy into it and return bytes
        written; otherwise just return the bytes.
        """
        num_requests = len(self._requests)

        # Tokens — concatenate, build indptr.
        tokens_indptr: list[int] = [0]
        tokens: list[int] = []
        for req in self._requests:
            tokens.extend(int(t) for t in req["tokens"])
            tokens_indptr.append(len(tokens))

        # Entropies.
        entropies_indptr: list[int] = [0]
        entropies: list[float] = []
        for req in self._requests:
            entropies.extend(float(e) for e in req["entropies"])
            entropies_indptr.append(len(entropies))

        # Logprobs — flatten per-request: req_indptr (req → chunks),
        # val_indptr (chunk → values), values.
        logprobs_req_indptr: list[int] = [0]
        logprobs_val_indptr: list[int] = [0]
        logprobs_values: list[float] = []
        for req in self._requests:
            for chunk in req["logprobs"]:
                logprobs_values.extend(float(v) for v in chunk)
                logprobs_val_indptr.append(len(logprobs_values))
            logprobs_req_indptr.append(len(logprobs_val_indptr) - 1)

        # Logits — opaque bytes blobs.
        logits_req_indptr: list[int] = [0]
        logits_byte_indptr: list[int] = [0]
        logits_bytes_buf = bytearray()
        for req in self._requests:
            for blob in req["logits"]:
                logits_bytes_buf.extend(blob)
                logits_byte_indptr.append(len(logits_bytes_buf))
            logits_req_indptr.append(len(logits_byte_indptr) - 1)

        # Dists — pairs of (ids, probs) per chunk.
        dists_req_indptr: list[int] = [0]
        dists_kv_indptr: list[int] = [0]
        dists_ids: list[int] = []
        dists_probs: list[float] = []
        for req in self._requests:
            for ids, probs in req["dists"]:
                dists_ids.extend(int(x) for x in ids)
                dists_probs.extend(float(p) for p in probs)
                dists_kv_indptr.append(len(dists_ids))
            dists_req_indptr.append(len(dists_kv_indptr) - 1)

        result = _pb.build_forward_response(
            driver_id=driver_id,
            num_requests=num_requests,
            tokens_indptr=np.asarray(tokens_indptr, dtype=np.uint32),
            tokens=np.asarray(tokens, dtype=np.uint32),
            dists_req_indptr=np.asarray(dists_req_indptr, dtype=np.uint32),
            dists_kv_indptr=np.asarray(dists_kv_indptr, dtype=np.uint32),
            dists_ids=np.asarray(dists_ids, dtype=np.uint32),
            dists_probs=np.asarray(dists_probs, dtype=np.float32),
            logits_req_indptr=np.asarray(logits_req_indptr, dtype=np.uint32),
            logits_byte_indptr=np.asarray(logits_byte_indptr, dtype=np.uint32),
            logits_bytes=bytes(logits_bytes_buf),
            logprobs_req_indptr=np.asarray(logprobs_req_indptr, dtype=np.uint32),
            logprobs_val_indptr=np.asarray(logprobs_val_indptr, dtype=np.uint32),
            logprobs_values=np.asarray(logprobs_values, dtype=np.float32),
            entropies_indptr=np.asarray(entropies_indptr, dtype=np.uint32),
            entropies=np.asarray(entropies, dtype=np.float32),
        )
        if dst_buf is not None:
            mv = memoryview(dst_buf)
            mv[: len(result)] = result
            return len(result)
        return result
