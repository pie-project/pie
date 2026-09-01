#!/usr/bin/env python3
"""Build a runnable, single-GPU-sized checkpoint out of a frontier-scale one.

The frontier MoE models we want to bring up (GLM-5.2, DeepSeek-V4-Pro,
DeepSeek-V4-Flash, Kimi-K2.6) are hundreds of gigabytes — far past one B200.
But *kernel* bring-up and pie-vs-vLLM parity only need the real tensor
*shapes*, dtypes and layer wiring, not the full depth or the full expert bank.

So this tool builds a scaled-down checkpoint that is byte-for-byte a real
checkpoint of the same architecture:

  * keep every "width" dimension exactly (hidden_size, head counts, MLA ranks,
    moe_intermediate_size, vocab) so every kernel sees production shapes;
  * keep only a handful of *source layers*, chosen so that the per-layer
    pattern (dense/sparse MLP, DSA indexer full/shared, V4 compress ratios,
    hash-routed layers) is fully covered;
  * keep only the first `--experts` routed experts per MoE layer, slicing the
    router rows to match;
  * optionally `--repeat` the selected layer block to get a deeper model at no
    extra download cost — repeated layers alias the same source weights.

Only the tensors that survive are downloaded, via HTTP range requests against
the safetensors shards, so a ~700 GB repo costs a few GB.

The result loads unmodified in both pie and vLLM, which is what makes it a
parity fixture rather than a mock.

Usage:
    python benches/shrink_checkpoint.py --repo zai-org/GLM-5.2 \
        --out ~/models/glm5.2-mini --experts 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import urllib.error
import urllib.request

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# safetensors dtype tag -> element size in bytes
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F8_E4M3FN": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

MAX_SHARD_BYTES = 4 * 1024**3
# Fetch spans separated by less than this are merged into one request; the
# wasted bytes are cheaper than another round trip.
COALESCE_GAP = 8 * 1024**2
CHUNK = 8 * 1024**2


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _token() -> str | None:
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    p = Path.home() / ".cache/huggingface/token"
    if p.is_file():
        t = p.read_text().strip()
        if t:
            return t
    return None


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    h = dict(headers or {})
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    h.setdefault("User-Agent", "pie-shrink-checkpoint/1.0")
    return urllib.request.Request(url, headers=h)


def _urlopen(url: str, headers: dict[str, str] | None = None, retries: int = 6):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(_request(url, headers), timeout=120)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def file_url(repo: str, revision: str, name: str) -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{name}"


def fetch_bytes(repo: str, revision: str, name: str,
                start: int | None = None, end: int | None = None) -> bytes:
    """Fetch [start, end) of a repo file; whole file when start is None."""
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    with _urlopen(file_url(repo, revision, name), headers) as resp:
        return resp.read()


def fetch_to_file(repo: str, revision: str, name: str, dest: Path,
                  start: int, end: int) -> None:
    headers = {"Range": f"bytes={start}-{end - 1}"}
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _urlopen(file_url(repo, revision, name), headers) as resp, tmp.open("wb") as out:
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            out.write(buf)
    got, want = tmp.stat().st_size, end - start
    if got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}[{start}:{end}] short read: {got} != {want}")
    tmp.rename(dest)


def list_repo_files(repo: str, revision: str) -> list[dict[str, Any]]:
    url = f"{HF_ENDPOINT}/api/models/{repo}/revision/{revision}"
    with _urlopen(url) as resp:
        info = json.load(resp)
    return info.get("siblings", [])


# --------------------------------------------------------------------------- #
# safetensors
# --------------------------------------------------------------------------- #
@dataclass
class SrcTensor:
    """A tensor living in a remote safetensors shard."""
    name: str
    shard: str
    dtype: str
    shape: list[int]
    start: int          # absolute byte offset in the shard file
    end: int
    # The dim-0 ROW WINDOW that survives, when only part of the tensor does.
    # An expert bank stacked on dim 0 keeps a prefix (`keep_from` 0); the PLE
    # carve keeps an interior window of one n-gram shard, so the window needs
    # an origin as well as a length. Either way it is ONE contiguous range, so
    # it is one range request.
    keep_from: int = 0
    keep_rows: int | None = None

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.shape[0] if self.shape and self.shape[0] else self.nbytes

    @property
    def fetch_start(self) -> int:
        return self.start + self.keep_from * self.row_bytes

    @property
    def keep_bytes(self) -> int:
        if self.keep_rows is None or not self.shape:
            return self.nbytes
        if self.keep_from == 0 and self.shape[0] <= self.keep_rows:
            return self.nbytes
        return self.row_bytes * self.keep_rows

    @property
    def keep_end(self) -> int:
        return self.fetch_start + self.keep_bytes


def read_shard_header(repo: str, revision: str, shard: str) -> dict[str, SrcTensor]:
    raw = fetch_bytes(repo, revision, shard, 0, 8)
    hdr_len = int.from_bytes(raw, "little")
    hdr = json.loads(fetch_bytes(repo, revision, shard, 8, 8 + hdr_len))
    data_start = 8 + hdr_len
    out: dict[str, SrcTensor] = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = SrcTensor(name, shard, meta["dtype"], list(meta["shape"]),
                              data_start + s, data_start + e)
    return out


def write_safetensors(path: Path, entries: list[tuple[str, str, list[int], Callable[[], Iterable[bytes]]]]) -> None:
    """entries: (name, dtype, shape, byte-chunk generator factory)."""
    header: dict[str, Any] = {}
    offset = 0
    for name, dtype, shape, _ in entries:
        n = DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with path.open("wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for name, dtype, shape, gen in entries:
            want = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            got = 0
            for chunk in gen():
                f.write(chunk)
                got += len(chunk)
            if got != want:
                raise RuntimeError(f"{name}: wrote {got} bytes, header says {want}")


# --------------------------------------------------------------------------- #
# Layer plans: which source layers to keep, and how to rewrite the config
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    """How to rebuild a smaller model from a big one."""
    family: str
    layer_prefix: str            # e.g. "model.layers." or "layers."
    src_layers: list[int]        # source layer indices, in output order
    experts: int | None          # routed experts to keep (None = keep all)
    # tensor-name globs (regex) that are per-expert, with the expert index group
    expert_re: re.Pattern | None
    # names (relative to a layer prefix) whose leading dim indexes experts
    router_suffixes: tuple[str, ...] = ()
    # A weight the checkpoint stores as a fixed number of side-by-side pieces
    # rather than one tensor -- Qwen4's PLE n-gram table ships as
    # `split_ngram_parts` shards of one row space. `part_re` names the piece
    # index the way `expert_re` names the expert; the output keeps `parts` of
    # them, and the config's own part count and row budget are scaled to match.
    # For a HASHED table the pieces cannot simply be truncated (see
    # `PleCarve`), so the family also names the config keys the hashing is
    # derived from.
    part_re: re.Pattern | None = None
    parts: int | None = None
    ple_carve: "PleCarve | None" = None
    # extra top-level tensors to keep
    globals_keep: tuple[str, ...] = ()
    drop_res: tuple[re.Pattern, ...] = ()
    config_rewrite: Callable[[dict, "Plan"], None] | None = None
    # sub-config holding the text model (Kimi wraps it in "text_config")
    text_cfg_key: str | None = None
    # AttnRes residual-block stride, scaled with the depth (Kimi-K3)
    attn_res_block_size: int | None = None
    # drop the vision tower and flatten `text_cfg_key` to the top level
    text_only: bool = False


def _text_cfg(cfg: dict, plan: Plan) -> dict:
    return cfg[plan.text_cfg_key] if plan.text_cfg_key else cfg


def _rewrite_common(cfg: dict, plan: Plan) -> None:
    tc = _text_cfg(cfg, plan)
    tc["num_hidden_layers"] = len(plan.src_layers)
    if plan.experts is not None:
        for key in ("n_routed_experts", "num_experts", "num_local_experts"):
            if key in tc:
                tc[key] = plan.experts
        for key in ("num_experts_per_tok", "num_experts_per_token", "moe_topk"):
            if key in tc and isinstance(tc[key], int):
                tc[key] = min(tc[key], plan.experts)
        # noaux_tc group routing must stay divisible
        if tc.get("n_group"):
            tc["n_group"] = min(tc["n_group"], plan.experts)
            tc["topk_group"] = min(tc.get("topk_group", 1), tc["n_group"])
    # MTP layers are a separate bring-up; drop them from the shrunken model.
    if "num_nextn_predict_layers" in tc:
        tc["num_nextn_predict_layers"] = 0


def _slice_list(cfg_list: list, src_layers: list[int]) -> list:
    return [cfg_list[i] for i in src_layers]


def _rewrite_glm(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    for key in ("mlp_layer_types", "indexer_types", "layer_types"):
        if isinstance(cfg.get(key), list):
            cfg[key] = _slice_list(cfg[key], plan.src_layers)
    if isinstance(cfg.get("mlp_layer_types"), list):
        dense = sum(1 for t in cfg["mlp_layer_types"] if t == "dense")
        cfg["first_k_dense_replace"] = dense
    # An indexer layer marked "shared" reuses the previous "full" layer's
    # indexer weights, so the block must start on a "full" layer.
    it = cfg.get("indexer_types")
    if isinstance(it, list) and it and it[0] != "full":
        raise SystemExit("glm: selected layer block must start on a 'full' indexer layer")


def _rewrite_dsv4(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    if isinstance(cfg.get("compress_ratios"), list):
        cfg["compress_ratios"] = _slice_list(cfg["compress_ratios"], plan.src_layers)
    if "num_hash_layers" in cfg:
        # The hash-routed layers are the leading ones, and the config can only
        # say "the first N" -- so they have to stay leading after the cut.
        cfg["num_hash_layers"] = _prefix_count(
            plan, "num_hash_layers", cfg["num_hash_layers"])


def _rewrite_kimi(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    tc = _text_cfg(cfg, plan)
    if "first_k_dense_replace" in tc:
        tc["first_k_dense_replace"] = sum(
            1 for i in plan.src_layers if i < tc["first_k_dense_replace"])


def _prefix_count(plan: Plan, key: str, bound: int) -> int:
    """Length of the leading run of selected layers with `src_idx < bound`.

    Config keys like `first_k_dense_replace` and `num_hash_layers` name a
    *prefix* of the output layers, so a selection that puts one of those source
    layers anywhere but at the front (or repeats it) cannot be described by the
    config at all -- and silently renumbering it would move the marked layers
    onto layers that are not of that kind.
    """
    marked = [i < bound for i in plan.src_layers]
    n = 0
    while n < len(marked) and marked[n]:
        n += 1
    if any(marked[n:]):
        raise SystemExit(
            f"{plan.family}: the first {bound} source layers are the ones "
            f"`{key}` names, and it can only name a *prefix* of the output "
            f"layers -- so they must be selected once and up front; got "
            f"--layers {plan.src_layers}")
    return n


def _dense_prefix_len(plan: Plan, first_k_dense_replace: int) -> int:
    return _prefix_count(plan, "first_k_dense_replace", first_k_dense_replace)


TEXT_TOWER_PREFIX = "language_model."


def _strip_text_tower(name: str) -> str:
    return name[len(TEXT_TOWER_PREFIX):] if name.startswith(TEXT_TOWER_PREFIX) else name


def _flatten_text_only(cfg: dict, plan: Plan) -> None:
    """Turn a multimodal config into a text-only one of the same decoder.

    The text sub-config moves to the top level, the vision half goes away, and
    the architecture is renamed so nothing claims to still take images. The
    decoder itself is untouched -- same widths, same layer types, same weights
    -- which is the whole point: a text-only K3 is still a K3 text tower, and
    it is the only half a text-generation parity run compares.
    """
    tc = cfg.pop(plan.text_cfg_key)
    for key in ("architectures", "auto_map", "_name_or_path", "model_type",
                "torch_dtype", "dtype"):
        tc.pop(key, None)
    for key in ("vision_config", "auto_map", "image_placeholder",
                "media_placeholder_token_id", "ignore_index"):
        cfg.pop(key, None)
    cfg.update(tc)
    cfg["model_type"] = "kimi_k3"
    cfg["architectures"] = ["KimiK3ForCausalLM"]


def _rewrite_kimi_k3(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    tc = _text_cfg(cfg, plan)
    if "first_k_dense_replace" in tc:
        tc["first_k_dense_replace"] = _dense_prefix_len(
            plan, tc["first_k_dense_replace"])

    # `kda_layers` / `full_attn_layers` are **1-indexed**: layer `i` is KDA
    # when `i + 1` is listed (`KimiLinearConfig.is_kda_layer`). Rebuild both
    # from the source layers' types so the hybrid pattern survives the cut.
    lac = tc.get("linear_attn_config")
    if isinstance(lac, dict):
        src_kda = set(lac.get("kda_layers") or ())
        kda: list[int] = []
        full: list[int] = []
        for out_idx, src_idx in enumerate(plan.src_layers):
            (kda if (src_idx + 1) in src_kda else full).append(out_idx + 1)
        if not kda or not full:
            raise SystemExit(
                "kimi_k3: the selection must keep at least one KDA layer and one "
                "full-attention (MLA) layer; source MLA layers are "
                f"{sorted(i - 1 for i in (lac.get('full_attn_layers') or ()))[:8]}...")
        lac["kda_layers"] = kda
        lac["full_attn_layers"] = full

    # AttnRes opens a new residual block every `attn_res_block_size` layers and
    # every layer then blends over the blocks opened so far. Left at the
    # production 12 a shrunken model would only ever hold one block -- the one
    # width the blend does not actually have to mix -- so scale the stride with
    # the depth unless the caller pinned it.
    if "attn_res_block_size" in tc:
        want = plan.attn_res_block_size
        if want is None:
            want = max(2, len(plan.src_layers) // 3)
        tc["attn_res_block_size"] = min(int(want), tc["attn_res_block_size"])

    if plan.text_only:
        _flatten_text_only(cfg, plan)


def _rewrite_gpt_oss(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    # `layer_types` is what makes a layer sliding vs full attention.
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = _slice_list(cfg["layer_types"], plan.src_layers)
    # gpt-oss carries the top-k under two names and HF reads both.
    if plan.experts is not None and isinstance(cfg.get("experts_per_token"), int):
        cfg["experts_per_token"] = min(cfg["experts_per_token"], plan.experts)


def _rewrite_qwen3_5_moe(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    # Everything below lives in `text_config`, not at the top level: the
    # checkpoint is a `Qwen3_5MoeForConditionalGeneration` wrapper around the
    # text tower, and reading `cfg` directly silently rewrites nothing.
    tc = _text_cfg(cfg, plan)
    # `layer_types` is what makes a layer linear vs full attention, and the
    # pattern has period `full_attention_interval`; a block that is a whole
    # number of periods keeps the ratio the model was trained with.
    if isinstance(tc.get("layer_types"), list):
        tc["layer_types"] = _slice_list(tc["layer_types"], plan.src_layers)


def _rewrite_qwen4_exp(cfg: dict, plan: Plan) -> None:
    """Qwen3.8-Flash-Next (`qwen4_exp`), as `mlx_lm` converts it.

    Four things beyond the common rewrite, and each one is a *declaration* the
    shrunken artifact would otherwise keep lying about:

    * `layer_types` -- linear vs full attention, period
      `full_attention_interval`. A whole number of periods keeps the ratio the
      model was trained with.
    * the PLE. The hashed n-gram table ships as `split_ngram_parts` pieces of
      one row space; the output keeps `--parts` of them, so
      `split_ngram_parts` and the row budget `ngram_vocab_size_base` both
      scale by the same fraction. The padding quantum
      `make_ngram_vocab_size_divisible_by` does NOT scale -- it is the table's
      row alignment, not its size. `ple_layer_ids` is **one-indexed**
      (`Ple::layer`'s doc: the shipped `[2]` names layer 1), so it is remapped
      through the selection rather than sliced.

      **AND THE TABLE ITSELF IS RE-CARVED, NOT TRUNCATED.** The table's row
      count is a *derived* number, not a stored one: sixteen primes past
      `ngram_vocab_size_base`, summed, rounded up to
      `make_ngram_vocab_size_divisible_by`. At the shipped base that is
      320_001_536 rows == 128 x 2_500_012, and the shards divide it exactly --
      but 2_500_012 is not itself a multiple of 128, so `parts` shards kept
      VERBATIM hold a row count that is not a multiple of the padding quantum
      and therefore cannot be the output of the derivation for ANY base. At
      `--parts 8` the base scales to 1_250_000, whose sixteen primes pad to
      20_001_536 while eight verbatim shards would hold 20_000_096: a
      1_440-row overstatement, and sixteen head offsets running to 300_001_275
      into a table a sixteenth that tall. That is a checkpoint no reader can
      load and no hasher can index.

      So the shards are not a slice of the source's row space; they are a
      re-carve of it. `PleCarve` derives BOTH hashings, takes each miniature
      head's rows from the head's own segment of the original table, and
      rewrites the two published buffers that describe them --
      `ngram_heads_vocab_sizes` and `ngram_heads_offsets` -- to the
      miniature's. (`layer_multipliers` is derived from the seed and the
      vocabulary, neither of which a shrink touches, so it is carried
      unchanged and asserted equal.) Config, tensors and a reader that derives
      its own constants then all say the same thing.
    * the MTP head. `mtp.*` is a separate bring-up, and none of its planes are
      kept, so the count that promises them goes to zero and the sub-config
      that shapes them goes away.
    * the vision tower. Its 333 planes are not kept either, and this config
      carries the switch that names the text-only reading of the artifact.
    """
    _rewrite_common(cfg, plan)
    tc = _text_cfg(cfg, plan)
    if isinstance(tc.get("layer_types"), list):
        tc["layer_types"] = _slice_list(tc["layer_types"], plan.src_layers)

    src_parts = tc.get("split_ngram_parts")
    if plan.parts is not None and isinstance(src_parts, int):
        if plan.parts > src_parts:
            raise SystemExit(f"qwen4_exp: --parts {plan.parts} exceeds the "
                             f"checkpoint's {src_parts} n-gram shards")
        tc["split_ngram_parts"] = plan.parts
        base = tc.get("ngram_vocab_size_base")
        if isinstance(base, int):
            tc["ngram_vocab_size_base"] = ple_base_for(base, src_parts, plan.parts)
        # The config is the only place the row count is written down, and the
        # carve is the only place it was ACTED on. If they were derived from
        # different numbers the artifact would ship the old disagreement under
        # a new name, so they are held together here rather than trusted.
        carve = plan.ple_carve
        if carve is not None and carve.dst_base != tc.get("ngram_vocab_size_base"):
            raise SystemExit(
                f"qwen4_exp: the carve cut the table for "
                f"ngram_vocab_size_base {carve.dst_base} and the config rewrite "
                f"declares {tc.get('ngram_vocab_size_base')}")

    ids = tc.get("ple_layer_ids")
    if isinstance(ids, list) and ids:
        out_of = {src: out for out, src in enumerate(plan.src_layers)}
        kept = [out_of[i - 1] + 1 for i in ids if (i - 1) in out_of]
        if not kept:
            raise SystemExit(
                f"qwen4_exp: the selection drops every PLE layer (the config's "
                f"one-indexed ple_layer_ids {ids} name layer(s) "
                f"{[i - 1 for i in ids]}), and a Flash-Next without its n-gram "
                f"memory is a different model; got --layers {plan.src_layers}")
        tc["ple_layer_ids"] = kept

    if "mtp_num_hidden_layers" in tc:
        tc["mtp_num_hidden_layers"] = 0
    tc.pop("mtp", None)

    if "language_model_only" in cfg:
        cfg["language_model_only"] = True


# --------------------------------------------------------------------------- #
# The PLE n-gram table: a re-carve, not a slice
# --------------------------------------------------------------------------- #
def ple_base_for(src_base: int, src_parts: int, dst_parts: int) -> int:
    """The miniature's `ngram_vocab_size_base`: the same fraction as the parts."""
    return src_base * dst_parts // src_parts


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True


def ngram_hash_geometry(base: int, heads: int, divisible_by: int
                        ) -> tuple[list[int], list[int], int]:
    """`_find_nth_prime_after`'s arithmetic, restated: `heads` consecutive
    primes at or past `base`, their prefix sums, and the sum rounded up to
    `divisible_by`.

    This is the SAME derivation `model::qwen_4::model::hash_constants` runs and
    the reference's `modular_qwen4_exp.py` runs, and it is the only thing that
    decides how tall a qwen4's n-gram table is. Held against the checkpoint's
    own published buffers by `PleCarve.checks`, so a mis-port of it fails the
    build rather than shipping a table nobody can index.
    """
    primes: list[int] = []
    offsets: list[int] = []
    total = 0
    prime = base - 1
    for _ in range(heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        primes.append(prime)
        offsets.append(total)
        total += prime
    return primes, offsets, -(-total // divisible_by) * divisible_by


@dataclass
class Piece:
    """`rows` rows of one source tensor, starting at its row `row0`."""
    src: SrcTensor
    row0: int
    rows: int


@dataclass
class PleCarve:
    """**HOW THE MINIATURE'S HASHED TABLE IS CUT OUT OF THE ORIGINAL'S.**

    A qwen4's PLE table is not one embedding: it is `heads` embeddings stacked
    into one row space, head `h` occupying `offsets[h] .. offsets[h] +
    primes[h]`. The hash is `mixed % primes[h] + offsets[h]`, so a row's
    MEANING is its position, and both numbers come out of the derivation above
    -- from `ngram_vocab_size_base` and nothing else.

    Which is why keeping the first `parts` stored shards produces a checkpoint
    that cannot be loaded OR indexed. The stored shards are equal slices of the
    padded row space, and that space is cut by the SHARD seam, not by the HEAD
    seam: at `--parts 8` of 128 the eight kept shards are the whole of head 0's
    segment and ninety-three rows of head 1's. Every other head's rows are
    gone, and the offsets beside them still name where those rows used to be.

    So the table is re-cut by head. Head `h` of the miniature takes the first
    `dst_primes[h]` rows of head `h` of the ORIGINAL -- real rows of the real
    table, at the real head they belong to, from the original's own
    `src_offsets[h]` -- and the sixteen segments are concatenated in order and
    re-chopped into `dst_parts` equal shards. The rows the padding quantum adds
    past the last head come off the last head's own segment too, so every row
    of the miniature is a row of the original and no row is invented.

    Two consequences worth naming:

    * a shard boundary of the OUTPUT falls wherever the concatenation puts it,
      which is not where any source shard ends -- so an output shard is a
      CONCATENATION of source row windows, and that is what `Piece` is for;
    * the quantisation is affine over the LAST axis (`(4, 32)` here: 20 u32
      code words and 5 `bf16` scale/bias groups per row), so every plane is
      row-aligned however the rows are cut. The carve is stated once in rows
      and applied to the code, scale and bias planes alike.
    """
    heads: int
    divisible_by: int
    src_base: int
    dst_base: int
    src_parts: int
    dst_parts: int
    src_primes: list[int]
    src_offsets: list[int]
    src_padded: int
    dst_primes: list[int]
    dst_offsets: list[int]
    dst_padded: int
    # dst shard -> the source (part, row-in-part, rows) it is made of
    layout: list[list[tuple[int, int, int]]]
    # source part -> the one contiguous row window that part contributes
    windows: dict[int, tuple[int, int]]

    @property
    def src_rows(self) -> int:
        return self.src_padded // self.src_parts

    @property
    def dst_rows(self) -> int:
        return self.dst_padded // self.dst_parts


def plan_ple_carve(cfg: dict, plan: Plan) -> PleCarve | None:
    """Derive both hashings and cut one out of the other, in rows.

    Runs off the config alone, before a byte is fetched, because WHICH source
    shards the carve reads is what decides the download.
    """
    if plan.parts is None or plan.part_re is None:
        return None
    tc = _text_cfg(cfg, plan)
    keys = ("ngram_vocab_size_base", "split_ngram_parts", "ngram_size",
            "heads_per_ngram", "make_ngram_vocab_size_divisible_by")
    if any(not isinstance(tc.get(k), int) for k in keys):
        return None
    src_base, src_parts, ngram, per_ngram, divisible_by = (tc[k] for k in keys)
    if plan.parts == src_parts:
        return None
    heads = (ngram - 1) * per_ngram
    dst_base = ple_base_for(src_base, src_parts, plan.parts)

    src_primes, src_offsets, src_padded = ngram_hash_geometry(src_base, heads, divisible_by)
    dst_primes, dst_offsets, dst_padded = ngram_hash_geometry(dst_base, heads, divisible_by)
    if src_padded % src_parts:
        raise SystemExit(
            f"qwen4_exp: the source's {src_padded} padded n-gram rows do not "
            f"divide into its {src_parts} stored shards")
    if dst_padded % plan.parts:
        raise SystemExit(
            f"qwen4_exp: `ngram_vocab_size_base {dst_base}` derives {dst_padded} "
            f"padded n-gram rows, which do not divide into --parts {plan.parts} "
            f"equal shards. Pick a --parts that divides it, or the table cannot "
            f"be stored the way this family stores tables.")

    # Every miniature head reads its own head's segment of the original, and
    # the padding rows come off the last one -- so nothing is invented and
    # nothing is read past a head seam.
    spans = [(src_offsets[h], dst_primes[h]) for h in range(heads)]
    spans[-1] = (spans[-1][0], spans[-1][1] + dst_padded - sum(dst_primes))
    for h, (_, rows) in enumerate(spans):
        if rows > src_primes[h]:
            raise SystemExit(
                f"qwen4_exp: miniature head {h} wants {rows} rows and the "
                f"source's head {h} has {src_primes[h]}: this is a shrink, and "
                f"that would be a growth")

    src_rows = src_padded // src_parts
    dst_rows = dst_padded // plan.parts
    flat: list[tuple[int, int, int]] = []
    for row0, rows in spans:
        at, left = row0, rows
        while left:
            part, local = divmod(at, src_rows)
            take = min(left, src_rows - local)
            flat.append((part, local, take))
            at += take
            left -= take

    layout: list[list[tuple[int, int, int]]] = [[] for _ in range(plan.parts)]
    windows: dict[int, tuple[int, int]] = {}
    at = 0
    for part, local, rows in flat:
        lo, hi = windows.get(part, (local, local + rows))
        windows[part] = (min(lo, local), max(hi, local + rows))
        while rows:
            shard, off = divmod(at, dst_rows)
            take = min(rows, dst_rows - off)
            layout[shard].append((part, local, take))
            local += take
            at += take
            rows -= take
    if at != dst_padded:
        raise SystemExit(f"qwen4_exp: the carve laid {at} rows, not {dst_padded}")

    return PleCarve(
        heads=heads, divisible_by=divisible_by,
        src_base=src_base, dst_base=dst_base,
        src_parts=src_parts, dst_parts=plan.parts,
        src_primes=src_primes, src_offsets=src_offsets, src_padded=src_padded,
        dst_primes=dst_primes, dst_offsets=dst_offsets, dst_padded=dst_padded,
        layout=layout, windows=windows,
    )


def carve_source_names(pairs: list[tuple[str, str]], plan: Plan) -> list[str]:
    """The source shard tensor names the carve reads that `pairs` does not.

    `pairs` maps output shard `k` to source shard `k` -- which is the right
    NAME mapping (it is what carries the per-shard quantisation override) and
    the wrong BYTE mapping. The rows come from wherever the head segments are.
    """
    carve = plan.ple_carve
    if carve is None:
        return []
    extra: set[str] = set()
    for _, src_name in pairs:
        if plan.part_re.search(src_name) is None:
            continue
        for part in carve.windows:
            extra.add(_part_name(src_name, plan, part))
    return sorted(extra)


def _part_name(src_name: str, plan: Plan, part: int) -> str:
    """`...shard_3.scales` with its part index rewritten to `part`."""
    m = plan.part_re.search(src_name)
    lo, hi = m.span(1)
    return f"{src_name[:lo]}{part}{src_name[hi:]}"


def apply_ple_carve(outs: list[OutTensor], src: dict[str, SrcTensor],
                    plan: Plan) -> list[tuple[str, SrcTensor, bytes]]:
    """Turn the table's placeholder outputs into concatenations, mark the row
    windows the download should fetch, and restate the published head buffers.

    Returns the checks to run once the bytes are down: what the source's own
    `ngram_heads_*` must say if [`ngram_hash_geometry`] is the derivation the
    conversion used.
    """
    carve = plan.ple_carve
    if carve is None:
        return []
    i64 = lambda v: b"".join(int(x).to_bytes(8, "little", signed=True) for x in v)  # noqa: E731

    seen = 0
    checks: list[tuple[str, SrcTensor, bytes]] = []
    for ot in outs:
        if plan.part_re.search(ot.src.name) is not None:
            out_part = int(plan.part_re.search(ot.name).group(1))
            rows_each = ot.src.shape[0]
            if rows_each != carve.src_rows:
                raise SystemExit(
                    f"{ot.src.name} stores {rows_each} rows and the config's "
                    f"hashing derives {carve.src_rows} per shard: the table in "
                    f"the file is not the table the config describes, and this "
                    f"tool will not guess which is right")
            ot.pieces = [
                Piece(src[_part_name(ot.src.name, plan, part)], row0, rows)
                for part, row0, rows in carve.layout[out_part]
            ]
            ot.rows = carve.dst_rows
            seen += 1
            # The row window this plane reads out of each source shard: one
            # contiguous range per shard, because the head segments are twenty
            # million rows apart and a shard is two and a half.
            for part, (lo, hi) in carve.windows.items():
                piece = src[_part_name(ot.src.name, plan, part)]
                piece.keep_from, piece.keep_rows = lo, hi - lo
            continue
        tail = ot.src.name.rsplit(".", 1)[-1]
        if tail == "ngram_heads_vocab_sizes":
            ot.literal = i64(carve.dst_primes)
            checks.append((ot.src.name, ot.src, i64(carve.src_primes)))
        elif tail == "ngram_heads_offsets":
            ot.literal = i64(carve.dst_offsets)
            checks.append((ot.src.name, ot.src, i64(carve.src_offsets)))
    print(f"      the n-gram table is re-carved by head: {carve.src_padded} rows "
          f"in {carve.src_parts} shards -> {carve.dst_padded} in {carve.dst_parts}, "
          f"{seen} planes from {len(carve.windows)} source shards")
    print(f"      heads past {carve.dst_base}: primes {carve.dst_primes[0]}"
          f"..{carve.dst_primes[-1]}, offsets to {carve.dst_offsets[-1]}, "
          f"padded to {carve.dst_padded} = {carve.dst_parts} x {carve.dst_rows}")
    return checks


# --------------------------------------------------------------------------- #
# Per-tensor quantisation overrides
# --------------------------------------------------------------------------- #
def carry_quant_overrides(cfg: dict, pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Move the per-tensor quant overrides onto the names that survived.

    A mixed-precision MLX conversion does not carry one recipe. `quantization`
    (and its `quantization_config` twin) holds the default bit width and group
    size as scalars, plus one entry per *module* that departs from it --
    Qwen4's 128 four-bit/group-32 PLE shards and its two-bit/group-128
    `switch_mlp` banks, DeepSeek-V4's two-bit split where every routed
    `gate_proj` is group 32 except layer 42's, which is group 64.

    Those keys name modules by their **source** name. Dropping layers and
    renumbering the survivors invalidates every one of them at once: keys that
    name a dropped layer become stale claims about tensors that are not there,
    and the surviving tensors lose the override that describes them, silently
    falling back to the file's scalar default -- which for both of these
    artifacts is the *wrong* width. So the map is rebuilt from the output
    names, not edited in place.

    A key is kept exactly when some tensor under it survived, and it is renamed
    by the same edit that renamed that tensor. Returns a per-dict count of the
    entries that survived, for the caller to report.
    """
    kept: dict[str, int] = {}
    for key in ("quantization", "quantization_config"):
        table = cfg.get(key)
        if not isinstance(table, dict):
            continue
        modules = {k: v for k, v in table.items() if isinstance(v, dict)}
        rebuilt = {k: v for k, v in table.items() if not isinstance(v, dict)}
        for module, spec in modules.items():
            head = module + "."
            for out_name, src_name in pairs:
                if not src_name.startswith(head):
                    continue
                # The rename is a prefix edit, so the module's new name is the
                # output tensor's name minus the same trailing plane.
                tail = len(src_name) - len(module)
                rebuilt[out_name[:-tail]] = spec
        cfg[key] = rebuilt
        kept[key] = len(rebuilt) - (len(table) - len(modules))
    return kept


FAMILIES: dict[str, dict[str, Any]] = {
    "qwen3_5_moe": dict(
        layer_prefix="model.language_model.layers.",
        # Qwen3.5 ships the routed bank already stacked on dim 0 -- one
        # `gate_up_proj` and one `down_proj` per layer, no per-expert name --
        # so `router_suffixes` does all the slicing, as it does for gpt-oss.
        expert_re=None,
        router_suffixes=(
            "mlp.gate.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ),
        globals_keep=("model.language_model.embed_tokens.weight",
                      "model.language_model.norm.weight",
                      "lm_head.weight"),
        # Period 4: three `linear_attention` layers then one `full_attention`.
        # Eight layers is two whole periods, so both kernels and both cache
        # kinds are exercised.
        default_layers="0-7",
        config_rewrite=_rewrite_qwen3_5_moe,
        text_cfg_key="text_config",
        keep_res=(re.compile(r"^model\.visual\."),),
    ),
    "glm_moe_dsa": dict(
        layer_prefix="model.layers.",
        expert_re=re.compile(r"\.mlp\.experts\.(\d+)\."),
        router_suffixes=("mlp.gate.weight", "mlp.gate.e_score_correction_bias"),
        globals_keep=("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"),
        default_layers="0-6",
        config_rewrite=_rewrite_glm,
    ),
    "deepseek_v4": dict(
        layer_prefix="layers.",
        expert_re=re.compile(r"\.ffn\.experts\.(\d+)\."),
        router_suffixes=("ffn.gate.weight", "ffn.gate.bias"),
        globals_keep=("embed.weight", "norm.weight", "head.weight",
                      "hc_head_fn", "hc_head_scale", "hc_head_base"),
        default_layers="0-5",
        config_rewrite=_rewrite_dsv4,
    ),
    # The same architecture as it comes out of `mlx_lm.convert`, which is a
    # different *checkpoint*: the trunk is re-rooted under `model.`, the routed
    # bank is stacked on dim 0 into one `switch_mlp` triple instead of 256
    # per-expert names, and every quantised plane is a
    # `weight`/`scales`/`biases` trio. Two entries share one `model_type`, so
    # the family is picked by which layer prefix the index actually uses.
    "deepseek_v4_mlx": dict(
        model_type="deepseek_v4",
        layer_prefix="model.layers.",
        expert_re=None,
        router_suffixes=(
            "ffn.gate.weight",
            "ffn.gate.e_score_correction_bias",
            "ffn.switch_mlp.gate_proj.weight",
            "ffn.switch_mlp.gate_proj.scales",
            "ffn.switch_mlp.gate_proj.biases",
            "ffn.switch_mlp.up_proj.weight",
            "ffn.switch_mlp.up_proj.scales",
            "ffn.switch_mlp.up_proj.biases",
            "ffn.switch_mlp.down_proj.weight",
            "ffn.switch_mlp.down_proj.scales",
            "ffn.switch_mlp.down_proj.biases",
        ),
        globals_keep=("model.embed_tokens.weight", "model.embed_tokens.scales",
                      "model.embed_tokens.biases", "model.norm.weight",
                      "lm_head.weight", "lm_head.scales", "lm_head.biases",
                      "model.hc_head.base", "model.hc_head.fn",
                      "model.hc_head.scale"),
        # Five layers that between them hold every per-layer kind this
        # architecture has: 0 and 1 are `compress_ratio` 0 and hash-routed
        # (`ffn.gate.tid2eid`), 2 is ratio 4 (compressor *and* indexer), 3 is
        # ratio 128 (compressor only, and the first `e_score_correction_bias`),
        # and 42 is the one layer in the artifact whose routed `gate_proj` is
        # quantised at group 64 where all forty-two of its siblings are group
        # 32 -- the per-tensor quant override has nothing to discriminate
        # against unless that layer is in the miniature.
        default_layers="0-3,42",
        config_rewrite=_rewrite_dsv4,
    ),
    "qwen4_exp": dict(
        layer_prefix="language_model.model.layers.",
        # Stacked on dim 0, like gpt-oss and Qwen3.5: one `switch_mlp` triple
        # per layer, three planes each, no per-expert name. The 2-bit packing
        # is on the LAST axis, so the leading axis is still whole experts and
        # a dim-0 prefix is still a valid bank.
        expert_re=None,
        router_suffixes=(
            "mlp.gate.weight",
            "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.gate_proj.scales",
            "mlp.switch_mlp.gate_proj.biases",
            "mlp.switch_mlp.up_proj.weight",
            "mlp.switch_mlp.up_proj.scales",
            "mlp.switch_mlp.up_proj.biases",
            "mlp.switch_mlp.down_proj.weight",
            "mlp.switch_mlp.down_proj.scales",
            "mlp.switch_mlp.down_proj.biases",
        ),
        # The PLE n-gram table is `split_ngram_parts` pieces of one row space.
        part_re=re.compile(r"\.ngram_embedding\.shard_(\d+)\."),
        default_parts=8,
        globals_keep=("language_model.model.embed_tokens.weight",
                      "language_model.lm_head.weight",
                      "language_model.model.hyper_connection_mixer.hc_norm.weight",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_down.weight",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_down.scales",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_down.biases",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_up.weight",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_up.scales",
                      "language_model.model.hyper_connection_mixer."
                      "input_mix_weight_up.biases"),
        # There is no `language_model.model.norm.weight` in this artifact: the
        # trunk's output norm IS the residual mixer's own
        # `hyper_connection_mixer.hc_norm.weight`, which is also the only
        # trunk-level norm `model::qwen_4::import` reads.
        # `full_attention_interval` is 4: three `linear_attention` layers then
        # one `full_attention`. Layers 0-3 are one whole period -- and layer 1
        # is the PLE layer, so the n-gram memory is in the block too.
        default_layers="0-3",
        config_rewrite=_rewrite_qwen4_exp,
        text_cfg_key="text_config",
    ),
    "kimi_k25": dict(
        layer_prefix="language_model.model.layers.",
        expert_re=re.compile(r"\.mlp\.experts\.(\d+)\."),
        router_suffixes=("mlp.gate.weight", "mlp.gate.e_score_correction_bias"),
        globals_keep=("language_model.model.embed_tokens.weight",
                      "language_model.model.norm.weight",
                      "language_model.lm_head.weight"),
        default_layers="0-3",
        config_rewrite=_rewrite_kimi,
        text_cfg_key="text_config",
        keep_res=(re.compile(r"^vision_tower\."), re.compile(r"^mm_projector\.")),
    ),
    "kimi_k3": dict(
        layer_prefix="language_model.model.layers.",
        expert_re=re.compile(r"\.block_sparse_moe\.experts\.(\d+)\."),
        router_suffixes=("block_sparse_moe.gate.weight",
                         "block_sparse_moe.gate.e_score_correction_bias"),
        globals_keep=("language_model.model.embed_tokens.weight",
                      "language_model.model.norm.weight",
                      "language_model.lm_head.weight",
                      # AttnRes blends the whole block stack once more at the end
                      "language_model.model.output_attn_res_norm.weight",
                      "language_model.model.output_attn_res_proj.weight"),
        # Layer 0 is the only dense MLP and the attention pattern has period 4
        # (three KDA layers then one MLA layer), so 0-7 covers every layer kind
        # twice and stays a whole number of periods.
        default_layers="0-7",
        config_rewrite=_rewrite_kimi_k3,
        text_cfg_key="text_config",
        keep_res=(re.compile(r"^vision_tower\."), re.compile(r"^mm_projector\.")),
    ),
    "gpt_oss": dict(
        layer_prefix="model.layers.",
        # gpt-oss stacks the whole expert bank on dim 0 of a handful of
        # tensors instead of giving each expert its own name, so there is no
        # per-expert name to filter -- `router_suffixes` does all the slicing.
        expert_re=None,
        router_suffixes=(
            "mlp.router.weight",
            "mlp.router.bias",
            "mlp.experts.gate_up_proj_blocks",
            "mlp.experts.gate_up_proj_scales",
            "mlp.experts.gate_up_proj_bias",
            "mlp.experts.down_proj_blocks",
            "mlp.experts.down_proj_scales",
            "mlp.experts.down_proj_bias",
        ),
        globals_keep=("model.embed_tokens.weight", "model.norm.weight",
                      "lm_head.weight"),
        # `layer_types` alternates sliding/full, so an even block starting at 0
        # covers both.
        default_layers="0-3",
        config_rewrite=_rewrite_gpt_oss,
    ),
}


def resolve_family(cfg: dict, weight_map: dict[str, str],
                   forced: str | None) -> str:
    """Pick the FAMILIES entry that describes this checkpoint.

    `model_type` names the *architecture*, and one architecture can ship in two
    checkpoint layouts -- DeepSeek-V4 upstream roots its trunk at `layers.` and
    gives every expert its own name, while `mlx_lm.convert`'s output of the
    same model roots at `model.layers.` and stacks the bank. So the model_type
    narrows the candidates and the index itself decides between them: only one
    layer prefix is actually a prefix of the names in the file.
    """
    if forced is not None:
        if forced not in FAMILIES:
            raise SystemExit(f"unknown --family {forced!r}; "
                             f"known: {sorted(FAMILIES)}")
        return forced
    model_type = cfg.get("model_type", "")
    named = [k for k, v in FAMILIES.items() if v.get("model_type", k) == model_type]
    if not named:
        raise SystemExit(f"unsupported model_type {model_type!r}; known: "
                         f"{sorted({v.get('model_type', k) for k, v in FAMILIES.items()})}")
    if len(named) == 1:
        return named[0]
    fits = [k for k in named
            if any(n.startswith(FAMILIES[k]["layer_prefix"]) for n in weight_map)]
    if len(fits) != 1:
        raise SystemExit(
            f"model_type {model_type!r} is described by {named}, and the index "
            f"{'matches all of' if fits else 'matches none of'} their layer "
            f"prefixes {[FAMILIES[k]['layer_prefix'] for k in named]}; "
            f"pass --family")
    return fits[0]


def parse_layer_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
@dataclass
class OutTensor:
    name: str
    src: SrcTensor
    # row slice applied to dim 0 (router rows when the expert bank shrinks;
    # the carved row count when `pieces` is set)
    rows: int | None = None
    # remap int64 expert ids into the reduced bank
    mod: int | None = None
    # The rows of OTHER tensors this one is a concatenation of -- the PLE
    # re-carve, whose output shard seams fall nowhere near the source's. `src`
    # stays the representative plane it takes its name, dtype and row width
    # from.
    pieces: list[Piece] | None = None
    # Content supplied by the build rather than by the checkpoint: the two
    # published n-gram head buffers, which describe a hashing the carve just
    # changed.
    literal: bytes | None = None

    @property
    def out_bytes(self) -> int:
        if self.literal is not None:
            return len(self.literal)
        if self.rows is None or not self.src.shape:
            return self.src.nbytes
        return self.src.row_bytes * self.rows


def layer_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def build_out_tensors(index: dict[str, str], plan: Plan,
                      keep_res: tuple[re.Pattern, ...]) -> list[tuple[str, str]]:
    """Return (out_name, src_name) pairs."""
    pairs: list[tuple[str, str]] = []
    for g in plan.globals_keep:
        if g in index:
            pairs.append((g, g))
    for kr in keep_res:
        for name in index:
            if kr.match(name):
                pairs.append((name, name))

    by_layer: dict[int, list[str]] = {}
    for name in index:
        li = layer_index(name, plan.layer_prefix)
        if li is not None:
            by_layer.setdefault(li, []).append(name)

    for out_idx, src_idx in enumerate(plan.src_layers):
        if src_idx not in by_layer:
            raise SystemExit(f"source layer {src_idx} not present in the checkpoint")
        src_pfx = f"{plan.layer_prefix}{src_idx}."
        out_pfx = f"{plan.layer_prefix}{out_idx}."
        for name in sorted(by_layer[src_idx]):
            suffix = name[len(src_pfx):]
            if plan.experts is not None and plan.expert_re is not None:
                m = plan.expert_re.search(name)
                if m and int(m.group(1)) >= plan.experts:
                    continue
            if plan.parts is not None and plan.part_re is not None:
                m = plan.part_re.search(name)
                if m and int(m.group(1)) >= plan.parts:
                    continue
            pairs.append((out_pfx + suffix, name))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. zai-org/GLM-5.2")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True, help="output snapshot directory")
    ap.add_argument("--layers", default=None,
                    help="source layer indices, e.g. '0-6' or '0,1,2,60' "
                         "(default: per-family block covering every layer type)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the selected layer block N times (weights are "
                         "reused, so this costs no extra download)")
    ap.add_argument("--experts", type=int, default=8,
                    help="routed experts to keep per MoE layer (0 = all)")
    ap.add_argument("--parts", type=int, default=None,
                    help="pieces to keep of a weight the checkpoint splits "
                         "into a fixed number of them -- Qwen4's PLE n-gram "
                         "table (default: the family's, 0 = all)")
    ap.add_argument("--family", default=None,
                    help="force a FAMILIES entry instead of picking one from "
                         "config.json's model_type (two entries can describe "
                         "one model_type: upstream naming vs an mlx_lm "
                         f"conversion of it). Known: {sorted(FAMILIES)}")
    ap.add_argument("--attn-res-block-size", type=int, default=None,
                    help="Kimi-K3 AttnRes residual-block stride (default: "
                         "scaled down with the kept depth so the blend sees "
                         "more than one block)")
    ap.add_argument("--text-only", action="store_true",
                    help="drop the vision tower and flatten the text sub-config "
                         "to the top level, producing a text-only checkpoint of "
                         "the same decoder (Kimi-K3)")
    ap.add_argument("--cache", default=None,
                    help="raw tensor cache dir (default: <out>/.srccache)")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the download plan and exit")
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(os.path.expanduser(args.cache)) if args.cache else out_dir / ".srccache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] reading {args.repo} config + index")
    cfg = json.loads(fetch_bytes(args.repo, args.revision, "config.json"))
    index_json = json.loads(fetch_bytes(args.repo, args.revision,
                                        "model.safetensors.index.json"))
    weight_map: dict[str, str] = index_json["weight_map"]

    family = resolve_family(cfg, weight_map, args.family)
    fam = FAMILIES[family]

    src_layers = parse_layer_spec(args.layers or fam["default_layers"])
    if args.repeat > 1:
        src_layers = src_layers * args.repeat
    parts = fam.get("default_parts") if args.parts is None else args.parts
    plan = Plan(
        family=family,
        layer_prefix=fam["layer_prefix"],
        src_layers=src_layers,
        experts=args.experts or None,
        expert_re=fam["expert_re"],
        router_suffixes=fam["router_suffixes"],
        globals_keep=fam["globals_keep"],
        config_rewrite=fam["config_rewrite"],
        text_cfg_key=fam.get("text_cfg_key"),
        attn_res_block_size=args.attn_res_block_size,
        text_only=args.text_only,
        part_re=fam.get("part_re"),
        parts=parts or None,
    )
    # Which source rows the hashed table is re-cut from is pure arithmetic over
    # the config, and it decides which shards get read at all -- so it is
    # settled before the pairs are.
    plan.ple_carve = plan_ple_carve(cfg, plan)

    pairs = build_out_tensors(weight_map, plan,
                              () if args.text_only else fam.get("keep_res", ()))
    if args.text_only and plan.layer_prefix.startswith(TEXT_TOWER_PREFIX):
        # A text-only checkpoint has no second tower to disambiguate, and the
        # text-only architecture reads its own weights at the top level -- so
        # the wrapper prefix has to come off the tensor names too, not just
        # out of the config.
        pairs = [(_strip_text_tower(out), src) for out, src in pairs]
    src_names = sorted(set(carve_source_names(pairs, plan)) | {s for _, s in pairs})
    missing = [n for n in src_names if n not in weight_map]
    if missing:
        raise SystemExit(f"the n-gram carve reads {len(missing)} tensor(s) the "
                         f"index does not have, first {missing[0]}")
    print(f"      {len(weight_map)} source tensors -> {len(pairs)} output tensors "
          f"({len(src_names)} distinct downloads)")

    print("[2/6] reading safetensors headers")
    shards = sorted({weight_map[s] for s in src_names})
    headers: dict[str, dict[str, SrcTensor]] = {}
    for i, shard in enumerate(shards, 1):
        headers[shard] = read_shard_header(args.repo, args.revision, shard)
        print(f"      [{i}/{len(shards)}] {shard}", end="\r", flush=True)
    print(" " * 78, end="\r")

    src: dict[str, SrcTensor] = {}
    for name in src_names:
        shard = weight_map[name]
        t = headers[shard].get(name)
        if t is None:
            raise SystemExit(f"{name} missing from {shard} header")
        src[name] = t

    print("[3/6] applying expert-bank slicing")
    outs = assign_rows(pairs, src, plan)
    checks = apply_ple_carve(outs, src, plan)
    raw_total = sum(src[n].nbytes for n in src_names)
    total = sum(src[n].keep_bytes for n in src_names)
    if total < raw_total:
        print(f"      expert banks stacked on dim 0 keep only their prefix: "
              f"{total / 1024**3:.2f} GiB instead of {raw_total / 1024**3:.2f}")
    print(f"      download size: {total / 1024**3:.2f} GiB from {len(shards)} shards")
    if args.dry_run:
        for n in src_names[:20]:
            t = src[n]
            print(f"      {n}  {t.dtype}{t.shape}  {t.keep_bytes/1024**2:.1f} MiB")
        return 0

    print("[4/6] downloading tensors")
    by_shard: dict[str, list[SrcTensor]] = {}
    for n in src_names:
        by_shard.setdefault(src[n].shard, []).append(src[n])
    done_bytes = 0
    t0 = time.time()
    for shard, tensors in by_shard.items():
        tensors.sort(key=lambda t: t.fetch_start)
        groups: list[list[SrcTensor]] = []
        for t in tensors:
            if groups and t.fetch_start - groups[-1][-1].keep_end <= COALESCE_GAP:
                groups[-1].append(t)
            else:
                groups.append([t])
        for grp in groups:
            missing = [t for t in grp if not (cache / cache_key(t)).exists()]
            if not missing:
                done_bytes += sum(t.keep_bytes for t in grp)
                continue
            lo = grp[0].fetch_start
            hi = max(t.keep_end for t in grp)
            blob_path = cache / f"span-{shard.replace('/', '_')}-{lo}-{hi}.bin"
            fetch_to_file(args.repo, args.revision, shard, blob_path, lo, hi)
            with blob_path.open("rb") as bf:
                for t in grp:
                    dest = cache / cache_key(t)
                    if dest.exists():
                        continue
                    bf.seek(t.fetch_start - lo)
                    data = bf.read(t.keep_bytes)
                    tmp = dest.with_suffix(".part")
                    tmp.write_bytes(data)
                    tmp.rename(dest)
            blob_path.unlink(missing_ok=True)
            done_bytes += sum(t.keep_bytes for t in grp)
            el = time.time() - t0
            print(f"      {done_bytes/1024**3:7.2f}/{total/1024**3:.2f} GiB "
                  f"({done_bytes/1024**2/max(el,1e-3):.0f} MiB/s)", end="\r", flush=True)
    print(" " * 78, end="\r")

    # **THE DERIVATION, HELD AGAINST THE CHECKPOINT'S OWN BUFFERS.** The carve
    # cut the table for the sixteen primes `ngram_hash_geometry` derives from
    # the SOURCE base; if the conversion hashed with anything else then those
    # are not the head seams and the carve read the wrong rows. The file says
    # what it hashed with, so it is asked.
    for name, tensor, want in checks:
        got = (cache / cache_key(tensor)).read_bytes()
        if got != want:
            wide = lambda b: [int.from_bytes(b[i:i + 8], "little", signed=True)  # noqa: E731
                              for i in range(0, len(b), 8)]
            raise SystemExit(
                f"{name}: this tool derives {wide(want)} and the checkpoint "
                f"publishes {wide(got)}. The n-gram carve reads the head "
                f"segments the derivation names, so it must not run against a "
                f"hashing it cannot reproduce.")
    if checks:
        print(f"      the source's own n-gram head buffers are the derivation's "
              f"({len(checks)} of them)")

    print("[5/6] writing shards")
    shard_files = write_output(out_dir, cache, outs)

    print("[6/6] writing config + tokenizer")
    plan.config_rewrite(cfg, plan)
    for key, n in carry_quant_overrides(cfg, pairs).items():
        print(f"      {key}: {n} per-tensor overrides carried onto the "
              f"renumbered names")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    copy_aux_files(args.repo, args.revision, out_dir)
    write_index(out_dir, shard_files)

    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)

    size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))
    print(f"done: {out_dir}  ({size / 1024**3:.2f} GiB, "
          f"{len(outs)} tensors, {len(plan.src_layers)} layers, "
          f"{plan.experts or 'all'} experts)")
    return 0


def cache_key(t: SrcTensor) -> str:
    safe = t.name.replace("/", "_")
    # The cached blob is a *prefix* when the expert bank is sliced, so the row
    # count has to be part of the key or a full blob from an earlier run with a
    # different `--experts` would be reused as if it were the prefix.
    rows = "" if t.keep_rows is None else f".rows{t.keep_from}+{t.keep_rows}"
    return f"{safe}{rows}.bin"


def assign_rows(pairs: list[tuple[str, str]], src: dict[str, SrcTensor],
                plan: Plan) -> list["OutTensor"]:
    """Decide the dim-0 slice for every output tensor.

    Runs *before* the download so that stacked expert banks (gpt-oss keeps all
    128 experts in one tensor) only fetch the prefix that survives.
    """
    outs: list[OutTensor] = []
    router_rows = plan.experts
    full: set[str] = set()
    for out_name, src_name in pairs:
        t = src[src_name]
        ot = OutTensor(out_name, t)
        if router_rows is not None:
            tail = out_name.split(plan.layer_prefix, 1)[-1]
            tail = tail.split(".", 1)[-1] if "." in tail else tail
            if tail in plan.router_suffixes and t.shape and t.shape[0] > router_rows:
                ot.rows = router_rows
            if tail.endswith("gate.tid2eid"):
                ot.mod = router_rows
        if ot.rows is None:
            full.add(src_name)
        outs.append(ot)
    for ot in outs:
        # A source tensor reused by two outputs with different slices has to be
        # fetched whole; only mark the prefix when every use agrees.
        if ot.rows is not None and ot.src.name not in full:
            ot.src.keep_rows = ot.rows
    return outs


def transform_bytes(ot: OutTensor, raw: bytes) -> tuple[bytes, list[int]]:
    shape = list(ot.src.shape)
    if ot.rows is not None and shape and shape[0] > ot.rows:
        # `raw` may already be the prefix (see `assign_rows`), so size the row
        # from the full tensor rather than from what was downloaded.
        row = ot.src.nbytes // shape[0]
        raw = raw[: ot.rows * row]
        shape[0] = ot.rows
    if ot.mod:
        import numpy as np
        a = np.frombuffer(raw, dtype=np.int64).copy()
        a %= ot.mod
        raw = a.tobytes()
    return raw, shape


def _rows_from(path: Path, row_bytes: int, skip: int, rows: int) -> Iterable[bytes]:
    """`rows` rows of a cached blob, `skip` rows in, a chunk at a time."""
    left = rows * row_bytes
    with path.open("rb") as f:
        f.seek(skip * row_bytes)
        while left:
            buf = f.read(min(CHUNK, left))
            if not buf:
                raise RuntimeError(f"{path}: ran out {left} bytes early")
            left -= len(buf)
            yield buf


def output_entry(cache: Path, ot: OutTensor
                 ) -> tuple[str, str, list[int], Callable[[], Iterable[bytes]]]:
    """One `write_safetensors` entry, as a LAZY generator.

    Streamed rather than materialised: a carved n-gram shard is two hundred
    megabytes and an output file holds up to four gigabytes of them, and this
    tool has to run beside nothing else on a 32 GiB box.
    """
    shape = list(ot.src.shape)
    if ot.literal is not None:
        data = ot.literal
        return ot.name, ot.src.dtype, shape, (lambda: iter((data,)))
    if ot.pieces is not None:
        shape[0] = ot.rows

        def joined() -> Iterable[bytes]:
            for piece in ot.pieces:
                yield from _rows_from(cache / cache_key(piece.src),
                                      piece.src.row_bytes,
                                      piece.row0 - piece.src.keep_from, piece.rows)

        return ot.name, ot.src.dtype, shape, joined
    if ot.mod:
        # Tiny, and the only entry that needs its bytes in hand.
        raw, shape = transform_bytes(ot, (cache / cache_key(ot.src)).read_bytes())
        return ot.name, ot.src.dtype, shape, (lambda r=raw: iter((r,)))
    rows = ot.src.shape[0] if shape else 1
    if ot.rows is not None and shape and shape[0] > ot.rows:
        rows, shape[0] = ot.rows, ot.rows
    path = cache / cache_key(ot.src)
    if not shape:
        return ot.name, ot.src.dtype, shape, (lambda: iter((path.read_bytes(),)))
    return ot.name, ot.src.dtype, shape, (
        lambda: _rows_from(path, ot.src.row_bytes, 0, rows))


def write_output(out_dir: Path, cache: Path, outs: list[OutTensor]) -> list[tuple[str, list[str]]]:
    for f in out_dir.glob("*.safetensors"):
        f.unlink()
    shards: list[tuple[str, list[str]]] = []
    batch: list[OutTensor] = []
    acc = 0

    def flush(idx: int) -> None:
        nonlocal batch, acc
        if not batch:
            return
        name = f"model-{idx:05d}.safetensors"
        write_safetensors(out_dir / name, [output_entry(cache, ot) for ot in batch])
        shards.append((name, [ot.name for ot in batch]))
        batch, acc = [], 0

    idx = 1
    for ot in outs:
        n = ot.out_bytes
        if acc + n > MAX_SHARD_BYTES and batch:
            flush(idx)
            idx += 1
        batch.append(ot)
        acc += n
    flush(idx)
    return shards


def write_index(out_dir: Path, shards: list[tuple[str, list[str]]]) -> None:
    total = sum((out_dir / n).stat().st_size for n, _ in shards)
    renamed = []
    for i, (name, names) in enumerate(shards, 1):
        new = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        (out_dir / name).rename(out_dir / new)
        renamed.append((new, names))
    weight_map = {t: n for n, names in renamed for t in names}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")


AUX_MAX = 64 * 1024**2
AUX_SKIP_SUFFIX = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")


def copy_aux_files(repo: str, revision: str, out_dir: Path) -> None:
    for sib in list_repo_files(repo, revision):
        name = sib.get("rfilename", "")
        if not name or "/" in name:
            continue
        if name.endswith(AUX_SKIP_SUFFIX) or name == "model.safetensors.index.json":
            continue
        if name == "config.json":
            continue
        try:
            data = fetch_bytes(repo, revision, name)
        except Exception as exc:  # noqa: BLE001 - aux files are best effort
            print(f"      skip {name}: {exc}")
            continue
        if len(data) > AUX_MAX:
            continue
        (out_dir / name).write_bytes(data)


if __name__ == "__main__":
    sys.exit(main())
