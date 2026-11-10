"""Checks pie's latent steps against Hugging Face transformers: the same prompt and the same steps as
the authors' code takes them (last hidden state, rescaled to the mean embedding norm, fed back as
`inputs_embeds` on the KV cache), comparing every latent input.

  python parity.py --model Qwen/Qwen3-4B --steps 40 --uri ws://127.0.0.1:8080

A cosine near 1 after the prompt says the `hidden` reading is right; staying near 1 across the steps
says each latent row is fed in and written to the KV cache right. Both sides run bf16, so the vectors
drift apart as slowly as two bf16 runs of the same model do.
"""

import argparse
import asyncio
import json
from pathlib import Path

HERE = Path(__file__).parent
QUESTION = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?")


async def pie_inputs(a, tokens):
    from pie_client import Event, PieClient

    client = PieClient(a.uri)
    await client.connect()
    await client.authenticate("parity")
    program = await client.install_program_bytes((HERE / "main.py").read_bytes(), "latentmas-parity.py", "0.1.0", force_overwrite=True)
    process = await client.launch_process(program, {"op": "probe", "tokens": tokens, "steps": a.steps, "scale": a.scale})
    inputs = []
    while True:
        event, value = await process.recv()
        if event != Event.Message:
            raise SystemExit(value)
        message = json.loads(value)
        if "input" not in message:
            break
        inputs.append(message["input"])
    await process.signal("quit")
    await client.close()
    return inputs


def hf_inputs(a, tokens):
    import torch
    from transformers import AutoModelForCausalLM

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()
    inputs = []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tokens], device=device), use_cache=True, output_hidden_states=True)
        for step in range(a.steps + 1):
            h = out.hidden_states[-1][:, -1].float()
            e = h * (a.scale / h.norm(dim=-1, keepdim=True))
            inputs.append(e[0].cpu())
            if step == a.steps:
                break
            out = model(inputs_embeds=e.to(torch.bfloat16)[:, None], past_key_values=out.past_key_values,
                        use_cache=True, output_hidden_states=True)
    return inputs


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--scale", type=float, default=1.0974, help="Qwen3-4B's mean input-embedding norm")
    p.add_argument("--uri", default="ws://127.0.0.1:8080")
    p.add_argument("--device", default="", help="where Hugging Face runs (default: the GPU, else the CPU)")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.\n\nQuestion: {QUESTION}\n"}]
    tokens = tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    pie = asyncio.run(pie_inputs(a, tokens))
    hf = hf_inputs(a, tokens)
    rows = []
    for i, (x, y) in enumerate(zip(pie, hf)):
        x = torch.tensor(x).flatten()
        rows.append({"step": i, "cosine": round(torch.nn.functional.cosine_similarity(x, y, dim=0).item(), 4)})
    print(json.dumps({"prompt_tokens": len(tokens), "steps": a.steps, "rows": rows}))
    print(f"cosine after the prompt {rows[0]['cosine']}, lowest over {a.steps} steps {min(r['cosine'] for r in rows)}")


if __name__ == "__main__":
    main()
