"""Google's Gemma 4 assistant drafter, run through mlx-vlm's own MTP round
loop over one prompt, with every round traced — the reference a pie
`gemma4-*-mtp` row's `mtp-speculative-decoding` run is read against:

    python3 scripts/gemma4_assistant_ref.py --model mlx-community/gemma-4-26b-a4b-it-4bit \
        --draft mlx-community/gemma-4-26B-A4B-it-assistant-bf16 [--k 2] [--max-tokens 64] \
        [--prompt "..."] [--out OUT.json]

Prints, per round, the drafts proposed and what the target answered at each
verified row, then the committed tokens. The pie loop's `proposed_trace` /
`truth_trace` / `commits_trace` say the same things in the same order; on the
same greedy path the two must agree round for round, up to bf16 ties.

The prompt is wrapped in the chat template with the same prefix pie's
`chat::prefix()` + `encode` produce (`<bos>` + raw text) — pass `--raw` for
that; default is the instruct template as mlx-vlm applies it.
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_vlm.utils import load
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative import mtp as mtp_mod
from mlx_vlm.models import cache as cache_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--k", type=int, default=2, help="drafts per round (block size - 1)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--raw", action="store_true", help="<bos> + raw text, no chat template")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, processor = load(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    drafter, _kind = load_drafter(args.draft, kind="mtp")
    lm = model.language_model if hasattr(model, "language_model") else model

    if args.raw:
        ids = tokenizer.encode(args.prompt)
        if not ids or ids[0] != tokenizer.bos_token_id:
            ids = [tokenizer.bos_token_id] + ids
    else:
        msgs = [{"role": "user", "content": args.prompt}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    ids = mx.array([ids])

    prompt_cache = cache_mod.make_prompt_cache(lm)
    out = lm(ids, cache=prompt_cache, return_hidden=True, return_shared_kv=True)
    bonus = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    # The recorded hidden is the last layer's output before the final norm;
    # the drafter is fed it normed (`speculative_draft_hidden`).
    hidden = mtp_mod._mtp_draft_hidden(lm, out.hidden_states[-1][:, -1:, :])
    shared = out.shared_kv_states

    drafter.reset(model)
    kv_offset = mtp_mod._mtp_cache_offset_max(prompt_cache)
    drafter.set_shared_kv(shared, kv_offset, position=mtp_mod._mtp_draft_position(kv_offset), kv_valid_len=kv_offset)

    greedy = lambda x: mx.argmax(x, axis=-1)
    committed = [bonus]
    rounds = []
    started = time.time()
    while len(committed) < args.max_tokens:
        bs = args.k + 1
        drafts = drafter.draft_block(committed[-1], hidden, None, bs, greedy, mx.int32, greedy=True)
        verify_input = mx.concatenate([mx.array([[committed[-1]]], dtype=mx.int32), drafts], axis=1)
        verify = mtp_mod._mtp_verify_target(lm, verify_input, prompt_cache, greedy, sample_target_tokens=True)
        if verify.target_tokens is not None:
            truth = [int(t) for t in verify.target_tokens.reshape(-1).tolist()]
        else:
            truth = [int(t) for t in mx.argmax(lm.speculative_logits_from_hidden(verify.hidden)[0], axis=-1).tolist()]
        proposed = [int(t) for t in drafts[0].tolist()]
        accepted = 0
        while accepted < len(proposed) and proposed[accepted] == truth[accepted]:
            accepted += 1
        new_tokens = proposed[:accepted] + [truth[accepted]]
        rounds.append({"proposed": proposed, "truth": truth, "accepted": accepted, "committed": new_tokens})
        committed.extend(new_tokens)
        hidden = mtp_mod._mtp_draft_hidden(lm, verify.hidden[:, accepted:accepted + 1, :])
        if accepted < bs - 1:
            lm.rollback_speculative_cache(prompt_cache, None, accepted, bs)
        next_shared = mtp_mod._slice_shared_kv_after_reject(verify.shared_kv_states, bs - (accepted + 1))
        kv_offset += accepted + 1
        drafter.set_shared_kv(next_shared, kv_offset, position=mtp_mod._mtp_draft_position(kv_offset), kv_valid_len=kv_offset)
        if tokenizer.eos_token_id in new_tokens:
            break
    took = time.time() - started
    drafted = sum(len(r["proposed"]) for r in rounds)
    acc = sum(r["accepted"] for r in rounds)
    text = tokenizer.decode(committed)
    print(f"{len(committed)} tokens over {len(rounds)} rounds in {took:.2f}s: accepted {acc}/{drafted}")
    for i, r in enumerate(rounds[:12]):
        print(f"  round {i}: proposed {r['proposed']} truth {r['truth']} -> +{r['accepted']}")
    print(repr(text[:200]))
    if args.out:
        json.dump({"ids": ids[0].tolist(), "tokens": committed, "rounds": rounds, "text": text}, open(args.out, "w"))


if __name__ == "__main__":
    main()
