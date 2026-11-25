"""Serving: pie running the mounted program, and the client that talks to it.

`serve` writes the PIE_HOME config (the listen address, the checkpoint as
the model source, weight reload allowed), starts `pie serve`, waits until it
is ready, and stops it on exit. `Pie` installs the two stanza programs and
runs them: `stanza-chat` generates greedily from token ids, and
`stanza-score` returns the log-prob and the argmax of each next token. An
adapter's bundle bytes follow its launch by the chunked file transfer, which
carries bundles of any size.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from pie_client import Event, PieClient

from .data import ends, prompt

USER = "stanza"
"""The tenant the stanza programs are installed for."""
READY = "Server ready"
"""What `pie serve` prints once it accepts clients."""
PROGRAMS = {"chat": "stanza_chat.wasm", "score": "stanza_score.wasm"}
"""The two inferlets, by the file their build writes."""
BOOT_SECONDS = 300
"""How long `pie serve` may take to become ready."""


def programs(model) -> dict[str, Path]:
    """The two inferlets, in the wasm target directory beside the pie binary's."""
    release = model.pie_bin.parent.parent / "wasm32-wasip3" / "release"
    return {kind: release / file for kind, file in PROGRAMS.items()}


def configure(model, port: int) -> None:
    """Write the PIE_HOME config pie serves this model from."""
    if not (model.home / "config.toml").exists():
        model.pie("config", "init")
    model.pie("config", "set", "server.listen", f"127.0.0.1:{port}")
    model.pie("config", "set", "model.source", str(model.weights))
    model.pie("config", "set", "server.allow_weight_reload", "true")


@contextmanager
def serve(model, port: int):
    """Run `pie serve` for `model` on `port` while the block runs, and give a
    `Pie` for it. The server's output goes to `<pie_home>/stanza/serve.log`."""
    if model.weights is None:
        raise ValueError(f"{model.id}: serving needs weights, a Hugging Face checkpoint directory")
    if model.models != model.home / "models":
        raise ValueError(f"pie serves {model.home / 'models'}, not the mounted {model.models}")
    configure(model, port)
    log = model.home / "stanza" / "serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        server = subprocess.Popen([str(model.pie_bin), "serve"], env=model.env(), stdout=out,
                                  stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + BOOT_SECONDS
        while READY not in log.read_text(errors="replace"):
            if server.poll() is not None or time.monotonic() > deadline:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
                raise RuntimeError(f"pie serve did not become ready (exit {server.poll()}):\n{tail}")
            time.sleep(0.1)
        yield Pie(f"ws://127.0.0.1:{port}", model)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


class Pie:
    """A client of one pie server for one model: chat, batched chat and
    scoring, each on the base model or with an adapter."""

    def __init__(self, url: str, model):
        self.url = url
        """The server's websocket address."""
        self.model = model
        """The model it serves, whose tokenizer renders and decodes."""
        self.programs = asyncio.run(self._install())
        """The installed `name@hash` of each inferlet, by kind."""

    def chat(self, messages: list[dict], adapter=None, max_tokens: int = 128) -> str:
        """The greedy reply to `messages`."""
        return self.chat_many([(messages, adapter)], max_tokens=max_tokens)[0]

    def chat_many(self, requests: list[tuple[list[dict], object]], max_tokens: int = 128) -> list[str]:
        """The greedy reply to each `(messages, adapter or None)`, launched
        together so the engine runs them in the same batches."""
        tok = self.model.tokenizer
        launches = [("chat", {"tokens": prompt(tok, m), "max_tokens": max_tokens, "stop": sorted(ends(tok))}, a)
                    for m, a in requests]
        outs = asyncio.run(self._launch(launches))
        return [tok.decode(o["tokens"], skip_special_tokens=True).strip() for o in outs]

    def score(self, tokens: list[int], adapter=None) -> tuple[list[float], list[int]]:
        """The log-prob pie gives each next token of `tokens`, and its argmax:
        entry `i` is about `tokens[i + 1]`."""
        out = asyncio.run(self._launch([("score", {"tokens": tokens}, adapter)]))[0]
        return out["logprobs"], out["argmax"]

    async def _install(self) -> dict[str, str]:
        async with PieClient(self.url) as client:
            await client.authenticate(USER)
            return {kind: await client.install_program(path) for kind, path in programs(self.model).items()}

    async def _launch(self, launches: list[tuple[str, dict, object]]) -> list[dict]:
        async with PieClient(self.url) as client:
            await client.authenticate(USER)

            async def one(kind, input, adapter):
                process = await client.launch_process(self.programs[kind], input | {"adapter": adapter is not None})
                if adapter is not None:
                    # The inferlet says "ready" once it waits for the bundle; a file sent earlier is dropped.
                    event, value = await process.recv()
                    if (event, value) != (Event.Message, "ready"):
                        raise RuntimeError(f"stanza-{kind} did not ask for the adapter bundle: {event.value} {value}")
                    await process.transfer_file(adapter.bytes)
                return await process.result()

            return await asyncio.gather(*(one(*launch) for launch in launches))
