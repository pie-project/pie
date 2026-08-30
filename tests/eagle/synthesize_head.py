#!/usr/bin/env python3
"""Derive a SYNTHETIC EAGLE draft head from a base qwen35 checkpoint.

**WHY A DERIVED HEAD AND NOT A TRAINED ONE** (campaign §3: "a REAL EAGLE head
for gemma4 is out of 100%; M-4 gates the MECHANISM with a synthetic head").
The identity gate — a greedy speculative run answering the non-speculative
run's tokens byte for byte — is true for ANY draft head, because verification
discards a wrong draft and keeps the target's own argmax. So what a head is
FOR in this gate is exercising the mechanism: the overlay import, the `aux.*`
binding, the drafts window, `mtp_logits`, and the verify loop. A trained head
would make the acceptance rate interesting and would not make the gate any
more provable.

**THE CONSTRUCTION, AND WHY IT IS THE ONE THAT SAYS SOMETHING.** The head is
built out of the base model's own planes:

    fc      = [ 0 | I ]      so the fusion answers the trunk's hidden, unchanged
    block   = a COPY of the base's last full-attention layer
    readout = the base `lm_head`  (the text reads it out of `embed`, tied)

which makes the draft "run one more decoder layer on top of the final hidden
and read it out" — a real, if crude, one-step lookahead. It proposes tokens a
trained head might propose, so the acceptance statistics the gate reports are
a number and not a zero; and every plane in it is a tensor the base already
ships, so the file is derived rather than invented.

    fc_embed = 0 is not a degenerate accident either: it is what makes the
    draft depend on the trunk's hidden state alone, which is the half of
    EAGLE's input the mechanism has to carry across the fire boundary.

Writes the twelve tensors `model::qwen_3`'s `Recipe::Eagle` binds, in the
family's OWN block spelling, so that `pie model import <base> --aux <this>`
prefixes them to `aux.*` and the text finds them there.

**TWO FAMILIES, ONE CONSTRUCTION.** qwen35 was the first rig and gemma4 is
where the identity gate can actually close (multimodal §17: qwen35 is a
HYBRID, and a rejected draft row folds into a gated-delta state no mask can
cut; gemma attends and does not recur). The derivation is the same sentence in
both — zero-and-identity fusion, then a copy of a full-attention layer — and
what differs is which layer is full-attention and what its tensors are called.

Usage:
    python tests/eagle/synthesize_head.py <base-snapshot-dir> <out.safetensors>
                                          [--family qwen35|gemma4]

The family is inferred from the snapshot's own tensor names when not stated.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

# The one SKU this gate rig serves. Read off `Model::d0_8b_dims` and
# `config.json`'s `text_config`, and asserted against the base's own shapes
# below rather than trusted.
HIDDEN = 1024
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
INTER = 3584
# `layer_types` alternates three linear_attention to one full_attention, so the
# full layers are `l % 4 == 3` and the last of twenty-four is 23. The head's
# block has a full-attention layer's shapes and only a full-attention layer's,
# so this is the layer it can be copied from.
DONOR_LAYER = 23


def read_index(snapshot: Path):
    """Every tensor of a safetensors snapshot: name -> (file, dtype, shape, offsets)."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({name for name in weight_map.values()})
    else:
        files = [p.name for p in snapshot.glob("*.safetensors")]
        if not files:
            raise SystemExit(f"{snapshot} holds no .safetensors")
    table = {}
    for file in files:
        path = snapshot / file
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            table[name] = (path, 8 + n, meta)
    return table


def load(table, name) -> np.ndarray:
    """One tensor, as a numpy array of its stored dtype (bf16 read as u16)."""
    if name not in table:
        raise SystemExit(f"the base checkpoint publishes no `{name}`")
    path, base, meta = table[name]
    start, end = meta["data_offsets"]
    with path.open("rb") as fh:
        fh.seek(base + start)
        raw = fh.read(end - start)
    if meta["dtype"] != "BF16":
        raise SystemExit(f"`{name}` is {meta['dtype']}, and this rig reads bf16")
    return np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])


def bf16_zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint16)


def bf16_identity(n: int) -> np.ndarray:
    """`I`, in bf16 bit patterns. 1.0 is 0x3F80 and every other entry is zero."""
    out = np.zeros((n, n), dtype=np.uint16)
    np.fill_diagonal(out, 0x3F80)
    return out


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name])
        blob = array.tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(encoded)) % 8
    encoded += b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)


#: gemma-4-E4B's own numbers, off `text_config`. The donor must be a layer
#: that is BOTH full-attention (`l % 6 == 5`) and owns its banks (below the
#: 18-layer shared tail, so `l < 24`): 23 is the last such layer.
GEMMA = {
    "hidden": 2560,
    "q_w": 8 * 512,
    "kv_w": 2 * 512,
    "head_dim": 512,
    "inter": 10240,
    "donor": 23,
    "prefix": "model.language_model.layers.{l}",
}


def gemma_head(table) -> dict[str, np.ndarray]:
    """The same construction as qwen's, in gemma's four-norm block spelling."""
    layer = GEMMA["prefix"].format(l=GEMMA["donor"])
    h = GEMMA["hidden"]

    def donor(suffix: str, want: tuple) -> np.ndarray:
        array = load(table, f"{layer}.{suffix}")
        if tuple(array.shape) != want:
            raise SystemExit(
                f"`{layer}.{suffix}` is {tuple(array.shape)} and this rig expects {want}; "
                "the donor is not the full-attention layer it was read for"
            )
        return array

    return {
        # `[fc_embed | fc_hidden]` = `[0 | I]`, so the block below reads the
        # trunk's own final-normed hidden and nothing of the embedding. The
        # text slices this bank at column `hidden`, embedding half first.
        "fc.weight": np.concatenate(
            [bf16_zeros((h, h)), bf16_identity(h)], axis=1
        ),
        "layers.0.input_layernorm.weight": donor("input_layernorm.weight", (h,)),
        "layers.0.post_attention_layernorm.weight": donor(
            "post_attention_layernorm.weight", (h,)
        ),
        "layers.0.pre_feedforward_layernorm.weight": donor(
            "pre_feedforward_layernorm.weight", (h,)
        ),
        "layers.0.post_feedforward_layernorm.weight": donor(
            "post_feedforward_layernorm.weight", (h,)
        ),
        "layers.0.self_attn.q_proj.weight": donor(
            "self_attn.q_proj.weight", (GEMMA["q_w"], h)
        ),
        "layers.0.self_attn.k_proj.weight": donor(
            "self_attn.k_proj.weight", (GEMMA["kv_w"], h)
        ),
        "layers.0.self_attn.v_proj.weight": donor(
            "self_attn.v_proj.weight", (GEMMA["kv_w"], h)
        ),
        "layers.0.self_attn.o_proj.weight": donor(
            "self_attn.o_proj.weight", (h, GEMMA["q_w"])
        ),
        "layers.0.self_attn.q_norm.weight": donor(
            "self_attn.q_norm.weight", (GEMMA["head_dim"],)
        ),
        "layers.0.self_attn.k_norm.weight": donor(
            "self_attn.k_norm.weight", (GEMMA["head_dim"],)
        ),
        "layers.0.mlp.gate_proj.weight": donor("mlp.gate_proj.weight", (GEMMA["inter"], h)),
        "layers.0.mlp.up_proj.weight": donor("mlp.up_proj.weight", (GEMMA["inter"], h)),
        "layers.0.mlp.down_proj.weight": donor("mlp.down_proj.weight", (h, GEMMA["inter"])),
    }


def main() -> None:
    if len(sys.argv) not in (3, 5):
        raise SystemExit(__doc__)
    snapshot, out = Path(sys.argv[1]), Path(sys.argv[2])
    table = read_index(snapshot)
    family = None
    if len(sys.argv) == 5 and sys.argv[3] == "--family":
        family = sys.argv[4]
    if family is None:
        # The snapshot names itself: gemma publishes a vision tower under
        # `model.vision_tower.*` and qwen under `model.visual.*`.
        family = "gemma4" if any("vision_tower" in k for k in table) else "qwen35"
    if family not in ("qwen35", "gemma4"):
        raise SystemExit(f"unknown family {family!r}")

    if family == "gemma4":
        tensors = gemma_head(table)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_safetensors(out, tensors)
        total = sum(t.size * 2 for t in tensors.values())
        print(f"wrote {len(tensors)} gemma4 tensors, {total / 1024 / 1024:.1f} MiB -> {out}")
        return

    layer = f"model.language_model.layers.{DONOR_LAYER}"

    def donor(suffix: str, want: tuple) -> np.ndarray:
        array = load(table, f"{layer}.{suffix}")
        if tuple(array.shape) != want:
            raise SystemExit(
                f"`{layer}.{suffix}` is {tuple(array.shape)} and this rig expects {want}; "
                "the donor layer is not the full-attention layer it was read for"
            )
        return array

    q_w = Q_HEADS * HEAD_DIM
    kv_w = KV_HEADS * HEAD_DIM
    tensors = {
        # The fusion: `[fc_embed | fc_hidden]` in the order the text slices it
        # (embedding first, hidden second). Zero and the identity, so the block
        # below reads the trunk's own final-normed hidden.
        "fc.weight": np.concatenate(
            [bf16_zeros((HIDDEN, HIDDEN)), bf16_identity(HIDDEN)], axis=1
        ),
        "layers.0.input_layernorm.weight": donor("input_layernorm.weight", (HIDDEN,)),
        "layers.0.self_attn.q_proj.weight": donor(
            "self_attn.q_proj.weight", (2 * q_w, HIDDEN)
        ),
        "layers.0.self_attn.k_proj.weight": donor("self_attn.k_proj.weight", (kv_w, HIDDEN)),
        "layers.0.self_attn.v_proj.weight": donor("self_attn.v_proj.weight", (kv_w, HIDDEN)),
        "layers.0.self_attn.o_proj.weight": donor("self_attn.o_proj.weight", (HIDDEN, q_w)),
        "layers.0.self_attn.q_norm.weight": donor("self_attn.q_norm.weight", (HEAD_DIM,)),
        "layers.0.self_attn.k_norm.weight": donor("self_attn.k_norm.weight", (HEAD_DIM,)),
        "layers.0.post_attention_layernorm.weight": donor(
            "post_attention_layernorm.weight", (HIDDEN,)
        ),
        "layers.0.mlp.gate_proj.weight": donor("mlp.gate_proj.weight", (INTER, HIDDEN)),
        "layers.0.mlp.up_proj.weight": donor("mlp.up_proj.weight", (INTER, HIDDEN)),
        "layers.0.mlp.down_proj.weight": donor("mlp.down_proj.weight", (HIDDEN, INTER)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(out, tensors)
    total = sum(t.size * 2 for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1024 / 1024:.1f} MiB -> {out}")


if __name__ == "__main__":
    main()
