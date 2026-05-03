"""Smoke test for raw-completion inferlet (no chat template).

Boots a one-shot Pie server with the portable driver, installs the
raw-completion inferlet, and runs a single prompt against the model
named on the CLI.
"""
import argparse
import asyncio
import sys
import tomllib
from pathlib import Path

from pie_client import Event


async def run(args):
    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, AuthConfig, RuntimeConfig, SchedulerConfig,
        ServerConfig, TelemetryConfig, DriverConfig,
    )

    repo_root = Path(__file__).parent.parent
    inf = repo_root / "inferlets" / "raw-completion"
    wasm = inf / "target" / "wasm32-wasip2" / "release" / "raw_completion.wasm"
    manifest = inf / "Pie.toml"
    if not wasm.exists():
        print(f"missing wasm: {wasm}", file=sys.stderr)
        sys.exit(1)

    pkg = tomllib.loads(manifest.read_text())["package"]
    inferlet_name = f"{pkg['name']}@{pkg['version']}"

    cfg = Config(
        server=ServerConfig(port=0),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        runtime=RuntimeConfig(wasm_max_instances=64),
        models=[ModelConfig(
            name="default",
            hf_repo=args.model,
            scheduler=SchedulerConfig(
                batch_policy="adaptive",
                default_token_limit=2000,
                default_endowment_pages=64,
                admission_oversubscription_factor=1000.0,
            ),
            driver=DriverConfig(
                type="portable", device=["cuda:0"], tensor_parallel_size=1,
                options={"max_batch_size": 64,
                         "max_num_kv_pages": args.kv_pages,
                         "n_gpu_layers": -1},
            ),
        )],
    )
    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm, manifest, force_overwrite=True)
        proc = await client.launch_process(inferlet_name, input={
            "prompt": args.prompt, "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        })
        text = []
        while True:
            ev, msg = await proc.recv()
            if ev == Event.Stdout: text.append(msg)
            elif ev == Event.Return:
                text.append(msg)
                break
            elif ev == Event.Error:
                print(f"ERROR: {msg}", file=sys.stderr); sys.exit(1)
        print("".join(text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The quick brown fox jumps over the")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--kv-pages", type=int, default=2048)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
