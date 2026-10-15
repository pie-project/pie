"""Inkling reference logits, from `transformers`' own `inkling` (5.15+), so a
pie reading of the same snapshot can be read against the official
implementation organ by organ.

    python3 scripts/inkling_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

The logits are the model's readout as `InklingForCausalLM` states it: the
hidden state over `logits_mup_width_multiplier`, the head, then the row cut to
`unpadded_vocab_size` (200 058 of the 201 024 the head stores). Inkling writes
no `<|begin_of_text|>` at the head of a conversation, so none is prepended.

The text model is driven directly: `InklingModel.forward` norms the embedding
with `embed_norm` and hands it to the text model, which norms it again, so the
multimodal wrapper reads a twice-normed embedding no checkpoint was trained on.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, InklingForConditionalGeneration

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
    (
        "chat",
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>What is 17 times 23?<|end_message|><|message_model|>",
    ),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    # A checkpoint states its routed experts' width as `intermediate_size`
    # beside `dense_intermediate_size`; `InklingConfig` takes the dense one
    # and leaves `moe_intermediate_size` at Inkling's 3072, which is not
    # Inkling-Small's 2048.
    config = AutoConfig.from_pretrained(args.snapshot)
    stated = json.load(open(os.path.join(args.snapshot, "config.json")))["text_config"]
    if "dense_intermediate_size" in stated:
        config.text_config.moe_intermediate_size = stated["intermediate_size"]
    model = InklingForConditionalGeneration.from_pretrained(
        args.snapshot, config=config, dtype=torch.bfloat16
    )
    model.to(args.device).eval()

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = []
        for name, text in PROMPTS:
            ids = tokenizer.encode(text, add_special_tokens=False)
            probes.append({"name": name, "ids": ids})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    text = model.model.language_model
    stated = config.text_config

    def read(input_ids, past):
        out = text(input_ids=input_ids, past_key_values=past, use_cache=True)
        logits = model.lm_head(out.last_hidden_state / stated.logits_mup_width_multiplier)
        if stated.unpadded_vocab_size is not None:
            logits = logits[..., : stated.unpadded_vocab_size]
        return logits[0].float().cpu().numpy(), out.past_key_values

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        input_ids = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            tf, past = read(input_ids, None)
            gen_rows = [tf[-1].copy()]
            produced = []
            last = int(tf[-1].argmax())
            for _ in range(args.steps):
                produced.append(last)
                step, past = read(torch.tensor([[last]], device=args.device), past)
                row = step[-1]
                gen_rows.append(row)
                last = int(row.argmax())
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.ref.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, vocab {vocab}, gen={produced[:12]}  ({time.time() - started:.1f}s)")
        print(f"      {tokenizer.decode(produced)!r}")

if __name__ == "__main__":
    main()
