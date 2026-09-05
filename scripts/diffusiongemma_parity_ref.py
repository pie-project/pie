"""DiffusionGemma reference, from transformers' own `DiffusionGemmaForBlockDiffusion`,
so a pie artifact of the same snapshot can be read against the reference
denoiser step by step.

    python3 scripts/diffusiongemma_parity_ref.py ref <snapshot-dir> OUT [--prompt ...] [--seed N]
        [--temperature 0.8] [--taps 64] [--generate]
    python3 scripts/diffusiongemma_parity_ref.py compare OUT PIE.json

`ref` writes to OUT:

    case.json       what the pie side is handed: prompt ids, canvas0/canvas1, the
                    temperature, the top-`taps` taps of step 0 (ids, probs)
    ref.json        the reference's answers: prefill last-row argmax/top8, step-0 and
                    step-1 argmax/entropy/top8 per canvas row, tap coverage stats,
                    and (with --generate) the reference sampler's block
    step0.f32       float32 [256, vocab] step-0 logits (raw, before temperature)
    step1.f32       float32 [256, vocab] step-1 logits (self-conditioned on step 0)

Step 0 is the denoiser over a uniform-random canvas with NO self-conditioning;
step 1 is a second random canvas with the reference's exact self-conditioning
(`softmax(step0 / T) · E`). The pie side reproduces step 1 with the top-`taps`
truncation of that distribution, so `compare` reports the truncation's own
covered mass beside the agreement numbers.

`compare` reads `PIE.json` — the `diffusion-parity` inferlet's output — and
prints argmax agreement, top-8 overlap and entropy correlation for the prefill
row, step 0 and step 1.
"""

import argparse
import json
import math
import os
import time

import numpy as np


def xorshift_canvas(seed: int, length: int, vocab: int):
    """The `diffusion-baseline` inferlet's host noise, bit for bit."""
    x = (seed | 1) & 0xFFFFFFFF
    out = []
    for _ in range(length):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out.append(x % vocab)
    return out


def softcap(x, cap):
    return cap * np.tanh(x / cap)


def row_stats(logits, temperature, k=8):
    """Per-row argmax, entropy of softmax(logits / T), top-k ids and probs."""
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    p = np.exp(scaled)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-45, None))).sum(axis=-1)
    top = np.argsort(-p, axis=-1)[:, :k]
    topp = np.take_along_axis(p, top, axis=-1)
    return {
        "argmax": logits.argmax(axis=-1).astype(int).tolist(),
        "entropy": ent.astype(float).tolist(),
        "top8_ids": top.astype(int).ravel().tolist(),
        "top8_probs": topp.astype(float).ravel().tolist(),
    }


def ref(args):
    import torch
    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.snapshot)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.snapshot,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": args.cpu_mem},
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    cfg = model.config
    vocab = cfg.text_config.vocab_size
    canvas = cfg.canvas_length
    cap = cfg.text_config.final_logit_softcapping

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = [int(t) for t in ids]
    device = model.get_input_embeddings().weight.device
    ids_t = torch.tensor([ids], device=device)

    canvas0 = xorshift_canvas(args.seed ^ 0, canvas, vocab)
    canvas1 = xorshift_canvas(args.seed ^ 0x9E3779B9, canvas, vocab)
    c0 = torch.tensor([canvas0], device=device)
    c1 = torch.tensor([canvas1], device=device)

    with torch.no_grad():
        # Prefill + step 0 in one forward: the encoder writes the cache, the
        # decoder reads it with no self-conditioning.
        out0 = model(input_ids=ids_t, decoder_input_ids=c0)
        pkv = out0.past_key_values
        enc_hidden = out0.encoder_last_hidden_state[0, -1]
        prefill_logits = softcap(
            model.lm_head(enc_hidden.to(model.lm_head.weight.device)).float().cpu().numpy(), cap
        )
        step0 = out0.logits[0].float().cpu().numpy()
        # The reference feeds step 1 the TEMPERATURE-SCALED step-0 logits, in
        # the embedding dtype, exactly as `_denoising_step` does.
        scaled0 = (out0.logits / args.temperature).to(model.model.decoder.embed_tokens.weight.dtype)
        out1 = model(past_key_values=pkv, decoder_input_ids=c1, self_conditioning_logits=scaled0)
        step1 = out1.logits[0].float().cpu().numpy()
        # The same step with the distribution TRUNCATED to the top-`taps`
        # (everything else -inf): the reference renormalizes those taps to
        # one where pie feeds them unnormalized, but the block's pre-norm is
        # an RMSNorm and scale-invariant, so the two agree up to rounding.
        # This is the exact target for pie's step 1; `step1` above carries
        # the truncation's own error on top.
        top = torch.topk(scaled0.float(), args.taps, dim=-1).indices
        truncated = torch.full_like(scaled0, float("-inf"))
        truncated.scatter_(-1, top, scaled0.gather(-1, top))
        out1t = model(past_key_values=pkv, decoder_input_ids=c1, self_conditioning_logits=truncated)
        step1_trunc = out1t.logits[0].float().cpu().numpy()
        # Diagnostics that isolate the trunk from the denoiser: the ENCODER
        # (causal, no post-norm) over the prompt at every position, and over
        # `[prompt | canvas0]` at the canvas positions. A pie encode pass over
        # the same rows must agree with these up to quantization; the gap
        # between that agreement and the decoder's is the denoise path's own.
        prefill_all = softcap(
            model.lm_head(out0.encoder_last_hidden_state[0].to(model.lm_head.weight.device))
            .float()
            .cpu()
            .numpy(),
            cap,
        )
        enc_all = model.model.encoder(input_ids=torch.cat([ids_t, c0], dim=1))
        canvas_hidden = enc_all.last_hidden_state[0, len(ids):]
        encoder_canvas = softcap(
            model.lm_head(canvas_hidden.to(model.lm_head.weight.device)).float().cpu().numpy(), cap
        )
    print(f"prefill + two steps in {time.time() - t0:.0f}s", flush=True)

    # Taps for the pie side: top-`taps` of softmax(step0 / T), unnormalized.
    scaled = step0.astype(np.float64) / args.temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    p = np.exp(scaled)
    p /= p.sum(axis=-1, keepdims=True)
    order = np.argsort(-p, axis=-1)[:, : args.taps]
    tap_probs = np.take_along_axis(p, order, axis=-1)
    covered = tap_probs.sum(axis=-1)

    step0.astype(np.float32).tofile(os.path.join(args.out, "step0.f32"))
    step1.astype(np.float32).tofile(os.path.join(args.out, "step1.f32"))
    step1_trunc.astype(np.float32).tofile(os.path.join(args.out, "step1_trunc.f32"))
    case = {
        "prompt": args.prompt,
        "prompt_ids": ids,
        "canvas0": canvas0,
        "canvas1": canvas1,
        "temperature": args.temperature,
        "taps": args.taps,
        "taps_ids": order.astype(int).ravel().tolist(),
        "taps_weights": tap_probs.astype(float).ravel().tolist(),
        "vocab": vocab,
        "canvas_length": canvas,
    }
    ref_out = {
        "prefill": row_stats(prefill_logits[None, :], 1.0),
        "prefill_rows": row_stats(prefill_all, 1.0),
        "encoder_canvas": row_stats(encoder_canvas, args.temperature),
        "step0": row_stats(step0, args.temperature),
        "step1": row_stats(step1, args.temperature),
        "step1_trunc": row_stats(step1_trunc, args.temperature),
        "tap_coverage": {
            "min": float(covered.min()),
            "mean": float(covered.mean()),
            "p10": float(np.percentile(covered, 10)),
        },
    }
    if args.generate:
        torch.manual_seed(args.seed)
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(input_ids=ids_t, max_new_tokens=canvas)
        seq = gen.sequences[0, len(ids):].tolist() if hasattr(gen, "sequences") else gen[0, len(ids):].tolist()
        ref_out["generate"] = {
            "tokens": [int(t) for t in seq],
            "text": tok.decode([int(t) for t in seq], skip_special_tokens=False),
            "tokens_per_forward": float(gen.tokens_per_forward.float().mean())
            if hasattr(gen, "tokens_per_forward")
            else None,
            "seconds": time.time() - t1,
        }
        print(f"generate in {time.time() - t1:.0f}s: {ref_out['generate']['text'][:200]!r}", flush=True)
    json.dump(case, open(os.path.join(args.out, "case.json"), "w"))
    json.dump(ref_out, open(os.path.join(args.out, "ref.json"), "w"))
    print(
        f"tap coverage over {args.taps}: min {covered.min():.4f} mean {covered.mean():.4f}; "
        f"step0 mean entropy {np.mean(ref_out['step0']['entropy']):.3f}; wrote {args.out}",
        flush=True,
    )


def agreement(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float((a == b).mean())


def top8_overlap(a, b, k=8):
    a = np.asarray(a).reshape(-1, k)
    b = np.asarray(b).reshape(-1, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def compare(args):
    ref_out = json.load(open(os.path.join(args.out, "ref.json")))
    # `pie run` prints a human header before the document; take the JSON line.
    lines = [line for line in open(args.pie).read().splitlines() if line.startswith("{")]
    if not lines:
        raise SystemExit(f"{args.pie}: no JSON document (did the run fail?)")
    pie = json.loads(lines[-1])
    if "result" in pie and isinstance(pie["result"], (dict, str)):
        pie = pie["result"]
    if isinstance(pie, str):
        pie = json.loads(pie)
    rows = []
    r = ref_out["prefill"]
    rows.append(("prefill last row", "argmax", agreement(r["argmax"], [pie["prefill_argmax"]]),
                 top8_overlap(r["top8_ids"], pie["prefill_top8_ids"]), None))
    for name, key, pie_key in (
        ("prefill all rows", "prefill_rows", "prefill_rows"),
        ("encode canvas0", "encoder_canvas", "encode_canvas"),
        ("step 0 (no SC)", "step0", "step0"),
        ("step 1 (exact SC)", "step1", "step1"),
        ("step 1 (top-K SC)", "step1_trunc", "step1"),
    ):
        if pie_key not in pie or key not in ref_out:
            continue
        r = ref_out[key]
        p = pie[pie_key]
        corr = float(np.corrcoef(np.asarray(r["entropy"]), np.asarray(p["entropy"]))[0, 1])
        rows.append((name, "argmax", agreement(r["argmax"], p["argmax"]),
                     top8_overlap(r["top8_ids"], p["top8_ids"]), corr))
    print(f"{'stage':18} {'argmax agree':>13} {'top8 overlap':>13} {'entropy corr':>13}")
    for name, _, agree, overlap, corr in rows:
        c = "" if corr is None else f"{corr:13.4f}"
        print(f"{name:18} {agree:13.4f} {overlap:13.4f} {c}")
    cov = ref_out["tap_coverage"]
    print(f"tap coverage: min {cov['min']:.4f} p10 {cov['p10']:.4f} mean {cov['mean']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("snapshot")
    r.add_argument("out")
    r.add_argument("--prompt", default="Why is the sky blue?")
    r.add_argument("--seed", type=int, default=0x12345678)
    r.add_argument("--temperature", type=float, default=0.8)
    r.add_argument("--taps", type=int, default=64)
    r.add_argument("--generate", action="store_true")
    r.add_argument("--gpu-mem", default="38GiB")
    r.add_argument("--cpu-mem", default="700GiB")
    r.set_defaults(fn=ref)
    c = sub.add_parser("compare")
    c.add_argument("out")
    c.add_argument("pie")
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
