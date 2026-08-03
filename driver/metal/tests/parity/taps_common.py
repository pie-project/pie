"""The part every `*_mlx_taps.py` shares: read the arguments, load the model,
and hand back a `dump` that writes what `cosine_bisect.py` expects to read.

Each family's reference dumper differs only in the forward pass it re-runs
inline. Everything around that -- the argument order, the token-id parsing, the
f32 widening, the `(rows, -1)` shape and the `<name>.npy` file naming -- is a
property of the COMPARISON, not of the model, and a dumper that got any of it
subtly different would be diffed against the others' output as if it had not.
"""
import os
import sys

try:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
except ImportError as missing:  # pragma: no cover - a setup error, not a code path
    sys.exit(
        f"{missing}\n"
        "These scripts need the pinned mlx-lm that produced every reference in\n"
        "this tree. See tests/parity/requirements.txt:\n"
        "  python3.14 -m venv ~/.cache/pie-metal/mlxenv\n"
        "  ~/.cache/pie-metal/mlxenv/bin/pip install -r requirements.txt\n"
        "then run these with ~/.cache/pie-metal/mlxenv/bin/python."
    )


def requantize_mxfp4(model):
    """Re-express every MXFP4 module as the affine-U4 g64 the DRIVER holds.

    gpt-oss is the one family whose weights this driver does not read as
    shipped. Every published mlx conversion keeps the MoE experts in openai's
    MXFP4 -- E2M1 nibbles in blocks of 32 under E8M0 exponents -- and no kernel
    here reads that, so the loader decodes it and re-encodes it affine-U4 g64.

    mlx-lm does not take that step. So an oracle run on the checkpoint as
    shipped is an oracle for a model with DIFFERENT expert weights, and the
    difference shows up as a smooth cosine decay with depth -- not a collapse,
    which is what a defect looks like, and the two are easy to confuse when
    only the tokens are compared.

    This is applied unconditionally rather than behind a flag: the driver always
    re-quantizes, so re-quantizing is not an option the caller has, it is what
    the comparison IS. Returns how many modules were changed.
    """
    changed = 0
    for _, module in model.named_modules():
        if getattr(module, "mode", None) != "mxfp4":
            continue
        values = mx.dequantize(
            module.weight,
            module.scales,
            getattr(module, "biases", None),
            group_size=module.group_size,
            bits=module.bits,
            mode="mxfp4",
        )
        w, s, b = mx.quantize(values, group_size=64, bits=4, mode="affine")
        module.weight, module.scales, module.biases = w, s, b
        module.group_size, module.bits, module.mode = 64, 4, "affine"
        changed += 1
    if changed:
        mx.eval(model.parameters())
        print(f"re-quantized {changed} MXFP4 modules to affine g64/b4", file=sys.stderr)
    return changed


def open_taps(argv):
    """`<model_path> <comma_token_ids> <out_dir>` -> (model, ids, dump, out_dir).

    The model returned is mlx-lm's TOP-level object; each family reaches for
    its own body (`top.model`, or `top.language_model.model` where there is a
    multimodal wrapper) because that path is one of the things that differs.
    """
    model_path, ids_csv, out_dir = argv[1], argv[2], argv[3]
    ids = [int(x) for x in ids_csv.strip().split(",") if x]
    os.makedirs(out_dir, exist_ok=True)
    top, _ = load(model_path)

    def dump(name, t):
        a = np.array(mx.astype(t, mx.float32)).reshape(len(ids), -1)
        np.save(os.path.join(out_dir, name + ".npy"), a)

    return top, ids, dump, out_dir
