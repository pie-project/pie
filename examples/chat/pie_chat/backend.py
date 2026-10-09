import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import AsyncIterator

from .config import MODEL


class EngineBackend:
    def __init__(self, url: str) -> None:
        self.endpoint = url.rstrip("/") + "/v1/chat/completions"
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history.clear()

    async def reply(self, text: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": text})
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        pieces: list[str] = []

        def pump() -> None:
            body = json.dumps({
                "model": MODEL,
                "messages": self.history,
                "stream": True,
                "temperature": 0.6,
                "top_p": 0.95,
            }).encode()
            request = urllib.request.Request(
                self.endpoint, data=body, headers={"content-type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request) as response:
                    for raw in response:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        choice = json.loads(payload)["choices"][0]
                        delta = choice.get("delta", {}).get("content") or ""
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", delta))
            except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(error)))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=pump, daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                self.history.pop()
                raise RuntimeError(value)
            pieces.append(value)
            yield value
        self.history.append({"role": "assistant", "content": "".join(pieces)})


class PlaceholderBackend:
    REPLY = (
        "This is a placeholder reply. The engine is not connected, so every answer "
        "is the same text. Run without --placeholder to talk to the engine."
    )

    def reset(self) -> None:
        pass

    async def reply(self, _text: str) -> AsyncIterator[str]:
        for word in self.REPLY.split(" "):
            await asyncio.sleep(0.04)
            yield word + " "
