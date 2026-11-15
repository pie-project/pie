
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from pie_client import PieClient

PROMPT = "The capital of France is"
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CASES = [(4, ALL), (16, ["q_proj", "v_proj"])]
STEPS = 8


def adapted(model_id: str, rank: int, targets: list[str], out: Path):
    torch.manual_seed(rank)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets))
    for name, p in model.named_parameters():
        if "lora_" in name:
            torch.nn.init.normal_(p, std=0.03)
            p.data = p.data.bfloat16().float()
    model.peft_config["default"].save_pretrained(out)
    state = {k: v.bfloat16().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    save_file(state, out / "adapter_model.safetensors")
    return model.eval()


def reference(model, ids):
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
        tokens = model.generate(ids, max_new_tokens=STEPS, do_sample=False)[0, ids.shape[1] :]
    return logits, tokens.tolist()


async def served(url: str, wasm: str, files: dict[str, bytes]):
    async with PieClient(url) as client:
        await client.authenticate("peft")
        name = await client.install_program(wasm)
        process = await client.launch_process(name, {"prompt": PROMPT, "max_tokens": STEPS}, files=files)
        return await process.result()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wasm", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    ids = AutoTokenizer.from_pretrained(args.model)(PROMPT, return_tensors="pt").input_ids
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    base_logits, base_tokens = reference(base, ids)
    pie_base = asyncio.run(served(args.base_url, args.wasm, {}))
    failed = 0
    for rank, targets in CASES:
        with tempfile.TemporaryDirectory() as out:
            model = adapted(args.model, rank, targets, Path(out))
            logits, tokens = reference(model, ids)
            files = {f: (Path(out) / f).read_bytes() for f in ("adapter_config.json", "adapter_model.safetensors")}
        got = asyncio.run(served(args.base_url, args.wasm, files))
        top = [t for t, _ in got["top"]]
        pie_logits = torch.tensor([v for _, v in got["top"]])
        pie_base_at = dict(map(tuple, pie_base["top"]))
        error = (pie_logits - logits[top]).abs().max().item()
        shift = logits[top] - base_logits[top]
        pie_shift = torch.tensor([v - pie_base_at.get(t, float("nan")) for t, v in got["top"]])
        shared = ~pie_shift.isnan()
        cos = torch.nn.functional.cosine_similarity(pie_shift[shared], shift[shared], dim=0).item()
        ok = got["tokens"] == tokens != base_tokens and top[0] == int(logits.argmax()) and error < 0.25 and cos > 0.99
        failed += not ok
        sites = ",".join(t.split("_")[0] for t in targets)
        print(f"{'ok' if ok else 'FAIL'} r={rank} {sites}: tokens {got['tokens']} vs {tokens}, "
              f"max |logit error| {error:.3f}, adapter shift {shift.abs().max():.2f} (cosine {cos:.4f})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
