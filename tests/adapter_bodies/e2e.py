#!/usr/bin/env python3
import argparse, asyncio, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--model", default="Qwen/Qwen3-0.6B")
a.add_argument("--groups", type=int, default=32)
a.add_argument("--decode", type=int, default=256)
a.add_argument("--rank", type=int, default=8)
a.add_argument("--scale", type=float, default=1e-4)
args = a.parse_args()

import torch
from transformers import AutoModelForCausalLM
from pie_client import PieClient

dev = "cuda" if torch.cuda.is_available() else "cpu"
hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
L, H, R = hf.config.num_hidden_layers, hf.config.hidden_size, args.rank


def plane(shape, salt):
    n = torch.arange(L * R * H, dtype=torch.int64) + salt
    return ((n * 2654435761 % 1000 - 500).float() * args.scale).bfloat16().float().reshape(shape).to(dev)


A, B = plane([L, R, H], 0), plane([L, H, R], L * R * H)
for l, layer in enumerate(hf.model.layers):
    layer.self_attn.register_forward_hook(
        lambda m, _, kw, out, l=l: (out[0] + kw["hidden_states"] @ A[l].T @ B[l].T, *out[1:]), with_kwargs=True)


@torch.no_grad()
def ref(prompt, toks):
    lp = torch.log_softmax(hf(torch.tensor([prompt + toks], device=dev)).logits[0].float(), -1)
    return [float(lp[len(prompt) - 1 + i, t]) for i, t in enumerate(toks)]


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program_bytes((HERE / "inferlet.py").read_bytes(), "adapter-bodies.py")
    rng = random.Random(0)
    prompts = [[rng.randrange(1000, 30000) for _ in range(rng.randrange(16, 64))] for _ in range(args.groups)]
    inp = dict(layers=L, rank=R, hidden=H, scale=args.scale, decode=args.decode, prompts=prompts)

    async def run(adapter):
        out = await (await c.launch_process(prog, input=dict(inp, adapter=adapter))).result()
        return json.loads(out) if isinstance(out, str) else out

    outs = {}
    for adapter in (False, True):
        await run(adapter)
        outs[adapter] = await run(adapter)
    got = outs[True]
    moved = sum(x != y for x, y in zip(got["tokens"], outs[False]["tokens"]))
    gaps = [abs(x - y) for p, t, lp in zip(prompts, got["tokens"], got["logprobs"]) for x, y in zip(lp, ref(p, t))]
    bad = sum(g > 0.25 for g in gaps)
    print(f"plain {outs[False]['step_ms']:.2f} ms/step, adapter {got['step_ms']:.2f} ms/step "
          f"({args.groups} groups, rank {args.rank})")
    print(f"{'FAIL' if bad else 'PASS'} adapter vs reference: {bad}/{len(gaps)} draws off, max gap {max(gaps):.3f}, "
          f"{moved}/{args.groups} groups moved by the adapter")
    await c.close()
    sys.exit(1 if bad else 0)


asyncio.run(main())
