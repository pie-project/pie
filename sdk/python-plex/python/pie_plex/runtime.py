import json
from collections.abc import Callable, Iterable
from typing import Any

from ._native import NativeAsyncRuntime, NativeRuntime


class Runtime:
    def __init__(
        self,
        policy: str,
        query: Callable[[str, dict[str, Any]], Any] | None = None,
        mechanics: Iterable[str] = (),
    ) -> None:
        self._native = NativeRuntime(policy, query, list(mechanics))

    def invoke(self, event: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.invoke_json(json.dumps(event)))

    def invoke_json(self, event_json: str) -> str:
        return self._native.invoke_json(event_json)


class AsyncRuntime:
    def __init__(
        self,
        policy: str,
        query: Callable[[str, dict[str, Any]], Any] | None = None,
        mechanics: Iterable[str] = (),
        queue_capacity: int = 64,
    ) -> None:
        self._native = NativeAsyncRuntime(
            policy,
            query,
            list(mechanics),
            queue_capacity,
        )

    def try_submit(self, channel: str, epoch: int, event: dict[str, Any]) -> bool:
        return self.try_submit_json(channel, epoch, json.dumps(event))

    def try_submit_json(self, channel: str, epoch: int, event_json: str) -> bool:
        return self._native.try_submit_json(channel, epoch, event_json)

    def try_submit_bytes(self, channel: str, epoch: int, event_json: bytes) -> bool:
        return self._native.try_submit_bytes(channel, epoch, event_json)

    def try_submit_batch(
        self, channel: str, epoch: int, events: list[dict[str, Any]]
    ) -> bool:
        return self._native.try_submit_batch_json(
            channel,
            epoch,
            json.dumps(events),
        )

    def latest(self, channel: str, after_epoch: int = 0) -> tuple[int, dict] | None:
        result = self.latest_json(channel, after_epoch)
        if result is None:
            return None
        epoch, outcome_json = result
        return epoch, json.loads(outcome_json)

    def latest_json(self, channel: str, after_epoch: int = 0) -> tuple[int, str] | None:
        return self._native.latest_json(channel, after_epoch)

    def stats(self) -> tuple[int, int, int]:
        return self._native.stats()

    def shutdown(self) -> None:
        self._native.shutdown()
