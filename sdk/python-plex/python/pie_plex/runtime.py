import json
from collections.abc import Callable, Iterable
from typing import Any

from ._native import NativeRuntime


class Runtime:
    def __init__(
        self,
        policy: str,
        query: Callable[[str, dict[str, Any]], Any] | None = None,
        actions: Iterable[str] = (),
    ) -> None:
        self._native = NativeRuntime(policy, query, list(actions))

    def invoke(self, event: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.invoke_json(json.dumps(event)))

    def invoke_json(self, event_json: str) -> str:
        return self._native.invoke_json(event_json)
