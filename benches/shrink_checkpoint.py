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
    # When the expert bank is stacked on dim 0, only the leading `keep_rows`
    # rows survive -- and they are a contiguous prefix, so only that prefix
    # needs to be fetched.
    keep_rows: int | None = None

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def keep_bytes(self) -> int:
        if self.keep_rows is None or not self.shape or self.shape[0] <= self.keep_rows:
            return self.nbytes
        return self.nbytes // self.shape[0] * self.keep_rows

    @property
    def keep_end(self) -> int:
        return self.start + self.keep_bytes


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
        cfg["num_hash_layers"] = sum(1 for i in plan.src_layers
                                     if i < cfg["num_hash_layers"])


def _rewrite_kimi(cfg: dict, plan: Plan) -> None:
    _rewrite_common(cfg, plan)
    tc = _text_cfg(cfg, plan)
    if "first_k_dense_replace" in tc:
        tc["first_k_dense_replace"] = sum(
            1 for i in plan.src_layers if i < tc["first_k_dense_replace"])


def _dense_prefix_len(plan: Plan, first_k_dense_replace: int) -> int:
    """Length of the leading run of selected layers that are dense.

    `first_k_dense_replace` names a *prefix* of the output layers, so a
    selection that puts a dense source layer anywhere but at the front (or
    repeats it) cannot be described by the config at all.
    """
    is_dense = [i < first_k_dense_replace for i in plan.src_layers]
    n = 0
    while n < len(is_dense) and is_dense[n]:
        n += 1
    if any(is_dense[n:]):
        raise SystemExit(
            f"{plan.family}: the dense (pre-MoE) source layers must be selected "
            f"once and up front, because `first_k_dense_replace` can only name a "
            f"prefix; got --layers {plan.src_layers}")
    return n


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
    # row slice applied to dim 0 (router rows when the expert bank shrinks)
    rows: int | None = None
    # remap int64 expert ids into the reduced bank
    mod: int | None = None


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
    model_type = cfg.get("model_type", "")
    if model_type not in FAMILIES:
        raise SystemExit(f"unsupported model_type {model_type!r}; "
                         f"known: {sorted(FAMILIES)}")
    fam = FAMILIES[model_type]

    src_layers = parse_layer_spec(args.layers or fam["default_layers"])
    if args.repeat > 1:
        src_layers = src_layers * args.repeat
    plan = Plan(
        family=model_type,
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
    )

    index_json = json.loads(fetch_bytes(args.repo, args.revision,
                                        "model.safetensors.index.json"))
    weight_map: dict[str, str] = index_json["weight_map"]

    pairs = build_out_tensors(weight_map, plan,
                              () if args.text_only else fam.get("keep_res", ()))
    if args.text_only and plan.layer_prefix.startswith(TEXT_TOWER_PREFIX):
        # A text-only checkpoint has no second tower to disambiguate, and the
        # text-only architecture reads its own weights at the top level -- so
        # the wrapper prefix has to come off the tensor names too, not just
        # out of the config.
        pairs = [(_strip_text_tower(out), src) for out, src in pairs]
    src_names = sorted({s for _, s in pairs})
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
        tensors.sort(key=lambda t: t.start)
        groups: list[list[SrcTensor]] = []
        for t in tensors:
            if groups and t.start - groups[-1][-1].keep_end <= COALESCE_GAP:
                groups[-1].append(t)
            else:
                groups.append([t])
        for grp in groups:
            missing = [t for t in grp if not (cache / cache_key(t)).exists()]
            if not missing:
                done_bytes += sum(t.keep_bytes for t in grp)
                continue
            lo = grp[0].start
            hi = max(t.keep_end for t in grp)
            blob_path = cache / f"span-{shard.replace('/', '_')}-{lo}-{hi}.bin"
            fetch_to_file(args.repo, args.revision, shard, blob_path, lo, hi)
            with blob_path.open("rb") as bf:
                for t in grp:
                    dest = cache / cache_key(t)
                    if dest.exists():
                        continue
                    bf.seek(t.start - lo)
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

    print("[5/6] writing shards")
    shard_files = write_output(out_dir, cache, outs)

    print("[6/6] writing config + tokenizer")
    plan.config_rewrite(cfg, plan)
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
    rows = "" if t.keep_rows is None else f".rows{t.keep_rows}"
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
        entries = []
        for ot in batch:
            raw, shape = transform_bytes(ot, (cache / cache_key(ot.src)).read_bytes())
            entries.append((ot.name, ot.src.dtype, shape, (lambda r=raw: iter((r,)))))
        write_safetensors(out_dir / name, entries)
        shards.append((name, [ot.name for ot in batch]))
        batch, acc = [], 0

    idx = 1
    for ot in outs:
        n = ot.src.nbytes if ot.rows is None else ot.src.nbytes * ot.rows // max(ot.src.shape[0], 1)
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
