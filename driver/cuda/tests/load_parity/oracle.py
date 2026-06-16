"""Generic byte-reconstruction oracle for the load-parity harness.

Given the materialized final tensors (read from the artifact cache) and the
source tensors, derive whether each output is correct WITHOUT any per-model
fusion/shard map — purely by reconstructing it from the source bytes:

  * direct  — output bytes == source bytes (same name, same dtype)
  * fusion  — output == ordered concat of consumed source siblings (the inputs
              that disappeared from the output set), in source-declaration order
  * split   — output is a contiguous slice of a consumed source, AND the family
              of slices from that source exactly tiles it (catches wrong offsets)
  * expert-stack — fused 3-D MoE expert tensor (experts.gate_up_proj /
              experts.down_proj) rebuilt from the per-expert source weights
              (a non-contiguous gather that `fusion`'s contiguous-run match can't express)
  * skip-quant — output dtype differs from source / is packed (FP8/MXFP4): can't
              be reconstructed without reimplementing the quant kernel; covered
              by the differential runs + test_transcode_fused.cu instead

For TP (>1 rank) the per-rank shards are first reassembled on their shard axis
into the full form (shape-driven, using the source shape to disambiguate the
axis for direct tensors; column-parallel byte-concat for derived ones), then the
SAME classify() runs against the source. So tp=1 and tp=N share one checker, and
arbitrary DSL compositions are checkable with no hardcoded knowledge.
"""

from __future__ import annotations

from dtypes import DTYPES
from dtypes import ELEM_BY_TAG as ELEM   # tag -> logical element bytes (None if packed)
from parse_cache import Tensor

ARENA = "__pie.storage_arena.0"
PACKED = {d.tag for d in DTYPES if d.packed}
# Derived quant scale/metadata outputs (a NEW tensor the quantizer emits, e.g.
# MXFP4's E8M0 `.weight_scale` when the source had `.weight_scale_inv`) can't be
# reconstructed from the source without the quant kernel -> skip-quant.
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_inv", ".scale", ".scales",
                  ".scale_inv", ".weight_scale_2")


def _internal(name: str) -> bool:
    return name.startswith("__pie.")   # storage arena / row-shard bank / etc.


def _strip(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def _src_index(src: dict, prefix: str):
    """Source tensors keyed in the OUTPUT namespace (prefix stripped), plus the
    source-declaration order (fusion concat order)."""
    by_name, order = {}, []
    for name, t in src.items():
        s = _strip(name, prefix)
        by_name[s] = t
        order.append(s)
    return by_name, order


def _match_fusion(target: bytes, parts: list[bytes]) -> bool:
    """True if some contiguous run of `parts` (in source order) concatenates to
    `target`. The inner loop stops as soon as the accumulated length reaches or
    exceeds the target, so each start position is O(run length) — and a real
    fusion's parts are a handful (q/k/v, gate/up), keeping this cheap in practice
    even though the worst case is quadratic in len(parts)."""
    n = len(parts)
    for i in range(n):
        acc = bytearray()
        for j in range(i, n):
            acc += parts[j]
            if len(acc) == len(target):
                if bytes(acc) == target:
                    return True
                break
            if len(acc) > len(target):
                break
    return False


def _match_split(target: bytes, sources: list[tuple[str, Tensor]]):
    """If target is a contiguous slice of exactly one source, return (name, off)."""
    for name, t in sources:
        if len(t.raw) > len(target):
            off = t.raw.find(target)
            if off >= 0:
                return name, off
    return None


# Stacked-MoE-expert fusions: per-expert `experts.{e}.{gate,up,down}_proj.weight`
# get stacked into 3-D `experts.gate_up_proj` ([E, 2*inter, hidden], slice e =
# gate[e] ++ up[e]) and `experts.down_proj` ([E, hidden, inter], slice e = down[e]).
# This is a *non-contiguous* gather of the source set (all gate/up, then all down),
# so _match_fusion (contiguous runs only) can't express it — reconstruct directly.
_EXPERT_STACKS = (
    (".experts.gate_up_proj", ("gate_proj.weight", "up_proj.weight")),
    (".experts.down_proj", ("down_proj.weight",)),
)


def _match_expert_stack(name: str, by_name: dict) -> bytes | None:
    """If `name` is a stacked-expert tensor, return the expected bytes rebuilt from
    the per-expert source weights (in expert-major order), else None."""
    for suffix, members in _EXPERT_STACKS:
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)] + ".experts."   # "....mlp.experts."
        acc, e = bytearray(), 0
        while all((f"{base}{e}.{m}") in by_name for m in members):
            for m in members:
                acc += by_name[f"{base}{e}.{m}"].raw
            e += 1
        return bytes(acc) if e > 0 else None         # e==0: pre-fused source, not this pattern
    return None


def classify(materialized: dict, src: dict, prefix: str = "") -> dict:
    """name -> (kind, ok). kind in {direct, fusion, split, skip-quant, unexplained}."""
    by_name, order = _src_index(src, prefix)
    out = {n: t for n, t in materialized.items() if not _internal(n)}
    consumed = [(s, by_name[s]) for s in order if s not in out]   # source order

    results: dict[str, tuple[str, bool]] = {}
    splits: dict[str, list[tuple[int, int, str]]] = {}   # src -> [(off, len, out)]
    for name, o in out.items():
        if name in by_name:
            s = by_name[name]
            results[name] = ("direct", o.raw == s.raw) if o.dtype == s.dtype \
                else ("skip-quant", True)
            continue
        if o.dtype in PACKED or name.endswith(SCALE_SUFFIXES):   # quant weight / scale artifact
            results[name] = ("skip-quant", True)
            continue
        stacked = _match_expert_stack(name, by_name)             # fused MoE expert stack
        if stacked is not None:
            results[name] = ("expert-stack", o.raw == stacked)
            continue
        same = [(s, t) for (s, t) in consumed if t.dtype == o.dtype]
        if _match_fusion(o.raw, [t.raw for _, t in same]):
            results[name] = ("fusion", True)
            continue
        hit = _match_split(o.raw, same)
        if hit is not None:
            sname, off = hit
            splits.setdefault(sname, []).append((off, len(o.raw), name))
            results[name] = ("split", True)         # confirmed by tiling pass below
            continue
        results[name] = ("unexplained", False)

    for sname, parts in splits.items():             # slices must tile the source exactly
        parts.sort()
        cur, ok = 0, True
        for off, ln, _ in parts:
            if off != cur:
                ok = False
                break
            cur += ln
        ok = ok and cur == len(by_name[sname].raw)
        if not ok:
            for _, _, nm in parts:
                results[nm] = ("split", False)
    return results


def _reassemble_raw(ra: Tensor, rb: Tensor, full_shape):
    """Reassemble two rank shards into the full byte string on their shard axis,
    or None if this rank ordering can't produce a clean full tensor."""
    s0, s1 = list(ra.shape), list(rb.shape)
    if full_shape is not None:
        sf = list(full_shape)
        if s0 == sf and s1 == sf:                   # replicated: ranks must agree
            return ra.raw if ra.raw == rb.raw else None
        if len(s0) != len(sf) or len(s1) != len(sf):
            return None
        diff = [i for i in range(len(sf)) if s0[i] != sf[i] or s1[i] != sf[i]]
        if len(diff) != 1:
            return None
        ax = diff[0]
        if s0[ax] + s1[ax] != sf[ax] or any(s0[i] != sf[i] for i in range(len(sf)) if i != ax):
            return None
        if ax == 0:                                 # column-parallel: byte concat
            return ra.raw + rb.raw
        elem = ELEM.get(ra.dtype)                   # row-parallel: interleave by row
        if elem is None:
            return None
        inner, outer = elem, 1
        for x in sf[ax + 1:]:
            inner *= x
        for x in sf[:ax]:
            outer *= x
        b0, b1 = s0[ax] * inner, s1[ax] * inner
        if len(ra.raw) != outer * b0 or len(rb.raw) != outer * b1:
            return None
        out = bytearray()
        for k in range(outer):
            out += ra.raw[k * b0:(k + 1) * b0]
            out += rb.raw[k * b1:(k + 1) * b1]
        return bytes(out)
    # derived output (not in source): replicated, else column-parallel concat
    return ra.raw if ra.raw == rb.raw else ra.raw + rb.raw


def _reassemble_all(a: dict, b: dict, by_name: dict):
    """Reassemble every output across two ranks. Returns (full_dict, n_unreassembled,
    n_sharded) — n_sharded counts tensors whose rank shard shape differs from the
    source full shape (the loader actually split them, vs replicated/derived)."""
    full, bad, sharded = {}, 0, 0
    for name, ra in a.items():
        if _internal(name):
            continue
        rb = b.get(name)
        if rb is None:
            bad += 1
            continue
        # Use the source shape to pick the shard axis only for a genuine copy/shard
        # (same dtype). Quant outputs (packed weight / derived scale) have a
        # different shape than their source; reassemble them generically and let
        # classify() mark them skip-quant.
        match = name in by_name and ra.dtype == by_name[name].dtype
        if match and list(ra.shape) != list(by_name[name].shape):
            sharded += 1
        raw = _reassemble_raw(ra, rb, by_name[name].shape if match else None)
        if raw is None:
            bad += 1
            continue
        full[name] = Tensor(name, ra.dtype, list(ra.shape), raw)
    return full, bad, sharded


def verify_tp(rank_a: dict, rank_b: dict, src: dict, prefix: str = ""):
    """Reassemble the two rank caches (picking the global order that reassembles
    cleanly) and classify the result against the source. The rank order is opaque
    in the cache key, so we try both and keep the one with no failures. Returns
    (results, n_sharded) — n_sharded == 0 flags a vacuous (nothing-sharded) run."""
    by_name, _ = _src_index(src, prefix)
    best = None
    for a, b in ((rank_a, rank_b), (rank_b, rank_a)):
        full, bad, sharded = _reassemble_all(a, b, by_name)
        res = classify(full, src, prefix)
        fails = bad + sum(1 for _, ok in res.values() if not ok)
        if fails == 0:
            return res, sharded
        if best is None or fails < best[0]:
            best = (fails, res, bad, sharded)
    res = best[1]
    if best[2]:                                     # some tensors couldn't reassemble
        res["<reassembly>"] = ("tp-shard", False)
    return res, best[3]


def summarize(results: dict) -> tuple[int, dict, list[str]]:
    """(n_checked_ok, kind_counts, failures)."""
    kinds: dict[str, int] = {}
    fails: list[str] = []
    for name, (kind, ok) in results.items():
        kinds[kind] = kinds.get(kind, 0) + 1
        if not ok:
            fails.append(f"{name}: {kind}")
    return sum(1 for _, ok in results.values() if ok), kinds, fails
