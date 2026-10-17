"""Hy4-preview reference logits, from transformers' own `hy_v4`, for the
miniature snapshot (`scripts/bench/shrink_checkpoint.py --repo
tencent/Hy4-preview --experts 16`: layers 0-5, sixteen experts, an indexer
that keeps sixteen keys), read in float32 on the CPU so pie's bf16 run is
graded against the model rather than against another bf16 rounding.

transformers 5.17 turns the rope dims of both the attention and the indexer
as halves (`rotate_half`); the checkpoint, vLLM and sglang turn them in
interleaved pairs, and so does pie. The reference is patched to the pairs, and
its indexer's key norm to the eps vLLM and sglang use (1e-6, not
`rms_norm_eps`).

    /root/venvs/vllm/bin/python scripts/hy4_parity_ref.py <snapshot-dir> OUT \
        [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (a stated file is used instead) and, per probe NAME:

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import apply_rotary_pos_emb_interleave
from transformers.models.hy_v4 import modeling_hy_v4

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river, there lived an old fisherman who "
              "every morning rowed out before dawn and every evening came home with a story about the fish "
              "that got away."),
    ("chat", "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>"
             "reasoning_effort:no_think<｜hy_end:opensource｜><｜hy_start:opensource｜>user"
             "<｜hy_middle:opensource｜>What is 17 times 23?<｜hy_end:opensource｜>"
             "<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>"
             "</think:opensource>"),
]


def interleaved(q, k, cos, sin, unsqueeze_dim=1):
    return apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def main():
    ap = parity_ref.parser()
    ap.add_argument("--dtype", default="float32", help="float32, or bfloat16 for the bf16 floor")
    args = ap.parse_args()
    modeling_hy_v4.apply_rotary_pos_emb = interleaved
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=getattr(torch, args.dtype)).eval()
    for layer in model.model.layers:
        if layer.self_attn.indexer is not None:
            layer.self_attn.indexer.k_norm.eps = 1e-6

    @torch.no_grad()
    def logits(ids):
        return model(input_ids=torch.tensor([ids])).logits[0].float().numpy()

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    for probe in parity_ref.probes(args, encode, PROMPTS):
        started = time.time()
        tf, rows, gen = parity_ref.greedy(logits, probe["ids"], args.steps)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, tokenizer.decode)


if __name__ == "__main__":
    main()
