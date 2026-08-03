#!/usr/bin/env python3
"""The greedy continuation mlx-lm produces from the weights THIS DRIVER HOLDS.

gpt-oss is the one family whose weights the driver re-quantizes on the way in.
Every published mlx conversion keeps the MoE experts in the MXFP4 that openai
shipped -- E2M1 nibbles in blocks of 32 under E8M0 exponents -- and no kernel
here reads that, so the loader decodes it and re-encodes it as the affine-U4
group-64 the matvecs do read.

That is a lossy step, and mlx-lm does not take it: it reads the MXFP4 directly.
So a token-exact gate against stock mlx-lm asks the driver to agree with a model
that has different expert weights, and it diverges around the third token --
smoothly, with every tap's cosine decaying from 1.0 and no tap collapsing, which
is the signature of drift rather than of a defect.

The fix is not to loosen the gate. It is to hand the oracle the same weights:
dequantize each MXFP4 expert bank and re-quantize it affine g64/b4, in memory,
exactly as the loader does. Then an exact token comparison means what it means
everywhere else in this file's neighbours.

Usage: gptoss_requantized_ref.py <checkpoint> <comma-separated ids> [n]
"""

import sys

# Importing it IS the guard: `taps_common` exits with the pinned-oracle recipe
# rather than an ImportError if mlx-lm is missing, and every script here wants
# that same message.
from taps_common import requantize_mxfp4

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402


def main() -> int:
    ckpt, ids_arg = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    ids = [int(x) for x in ids_arg.split(",")]

    model, tokenizer = load(ckpt)
    if requantize_mxfp4(model) == 0:
        print("no MXFP4 modules: this checkpoint needs no re-quantization", file=sys.stderr)

    cur, out = list(ids), []
    for _ in range(count):
        logits = model(mx.array([cur]))[0, -1]
        token = int(mx.argmax(logits).item())
        out.append(token)
        cur.append(token)
    print(" ".join(str(t) for t in out))
    print(repr(tokenizer.decode(out)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
