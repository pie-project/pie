"""MiniMax-M3 reference logits, from transformers' own `minimax_m3_vl`, for a
miniature snapshot (`scripts/bench/shrink_checkpoint.py --repo
MiniMaxAI/MiniMax-M3 --layers 0-5 --experts 8`), read in float32 on the CPU
so pie's bf16 run is graded against the model rather than against another
bf16 rounding.

    /root/venvs/vllm/bin/python scripts/minimax_m3_parity_ref.py \
        <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Beside the short prompts it writes two long ones, past sixteen 128-key
blocks, so the indexer's selection drops blocks rather than keeping every
one. Writes `OUT/probes.json` (a stated file is used instead) and, per probe
NAME:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl as reference

PROMPTS = [
    *parity_ref.PROMPTS,
    ("chat", "]~b]user\nWhat is 17 times 23?[e~[\n]~b]ai\n"),
]

# Long probes: the reference's own source, and its docstrings joined into
# prose, cut at a length past sixteen blocks of 128.
LONG = [("long_code", 3000), ("long_prose", 2400)]


def long_texts():
    source = open(reference.__file__).read()
    prose = " ".join(
        line.strip().strip('"').strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", "class ", "import ", "from ", "@", "return", "self."))
    )
    return {"long_code": source, "long_prose": prose}


def main():
    ap = parity_ref.parser()
    ap.add_argument("--long-steps", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma-separated probe names")
    ap.add_argument("--dtype", default="float32", help="float32, or bfloat16 for the bf16 floor")
    args = ap.parse_args()
    torch.set_num_threads(min(64, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForImageTextToText.from_pretrained(
        args.snapshot, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).eval()
    texts = long_texts()
    more = []
    for name, length in LONG:
        ids = tokenizer.encode(texts[name])
        assert len(ids) >= length, f"{name}: {len(ids)} tokens, {length} wanted"
        more.append({"name": name, "ids": ids[:length]})
    probes = parity_ref.probes(args, tokenizer.encode, PROMPTS, more if not args.probes else ())
    if args.only:
        probes = [p for p in probes if p["name"] in set(args.only.split(","))]

    @torch.no_grad()
    def logits(ids):
        return model(input_ids=torch.tensor([ids])).logits[0].float().numpy()

    for probe in probes:
        started = time.time()
        steps = args.long_steps if len(probe["ids"]) > 2048 else args.steps
        tf, rows, gen = parity_ref.greedy(logits, probe["ids"], steps)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, tokenizer.decode)


if __name__ == "__main__":
    main()
