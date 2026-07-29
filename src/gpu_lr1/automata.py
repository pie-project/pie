from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True)
class Fragment:
    start: int
    end: int


class NFABuilder:
    def __init__(self) -> None:
        self.epsilon_edges: list[set[int]] = []
        self.byte_edges: list[dict[int, set[int]]] = []

    def new_state(self) -> int:
        state = len(self.epsilon_edges)
        self.epsilon_edges.append(set())
        self.byte_edges.append({})
        return state

    def add_epsilon(self, source: int, target: int) -> None:
        self.epsilon_edges[source].add(target)

    def add_byte(self, source: int, value: int, target: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError(f"byte value out of range: {value}")
        self.byte_edges[source].setdefault(value, set()).add(target)

    def empty(self) -> Fragment:
        start = self.new_state()
        end = self.new_state()
        self.add_epsilon(start, end)
        return Fragment(start, end)

    def literal(self, value: bytes) -> Fragment:
        start = self.new_state()
        current = start
        for byte in value:
            target = self.new_state()
            self.add_byte(current, byte, target)
            current = target
        return Fragment(start, current)

    def charset(self, values: Iterable[int]) -> Fragment:
        start = self.new_state()
        end = self.new_state()
        for value in values:
            self.add_byte(start, value, end)
        return Fragment(start, end)

    def concat(self, fragments: Iterable[Fragment]) -> Fragment:
        parts = list(fragments)
        if not parts:
            return self.empty()
        for left, right in zip(parts, parts[1:], strict=False):
            self.add_epsilon(left.end, right.start)
        return Fragment(parts[0].start, parts[-1].end)

    def alternate(self, fragments: Iterable[Fragment]) -> Fragment:
        parts = list(fragments)
        if not parts:
            start = self.new_state()
            end = self.new_state()
            return Fragment(start, end)
        start = self.new_state()
        end = self.new_state()
        for part in parts:
            self.add_epsilon(start, part.start)
            self.add_epsilon(part.end, end)
        return Fragment(start, end)

    def optional(self, fragment: Fragment) -> Fragment:
        return self.alternate([self.empty(), fragment])

    def star(self, fragment: Fragment) -> Fragment:
        start = self.new_state()
        end = self.new_state()
        self.add_epsilon(start, end)
        self.add_epsilon(start, fragment.start)
        self.add_epsilon(fragment.end, fragment.start)
        self.add_epsilon(fragment.end, end)
        return Fragment(start, end)


@dataclass(frozen=True)
class ByteDFA:
    transitions: np.ndarray
    accepting: np.ndarray
    start_state: int
    dead_state: int = 0

    def __post_init__(self) -> None:
        if self.transitions.dtype != np.int32:
            raise TypeError("transitions must use int32")
        if self.transitions.ndim != 2 or self.transitions.shape[1] != 256:
            raise ValueError("transitions must have shape [states, 256]")
        if self.accepting.shape != (self.transitions.shape[0],):
            raise ValueError("accepting must have one entry per state")

    @property
    def num_states(self) -> int:
        return int(self.transitions.shape[0])

    def advance(self, state: int, value: bytes) -> int:
        current = state
        for byte in value:
            current = int(self.transitions[current, byte])
        return current

    def accepts(self, value: bytes | str) -> bool:
        data = value.encode("utf-8") if isinstance(value, str) else value
        state = self.advance(self.start_state, data)
        return bool(self.accepting[state])


def determinize(builder: NFABuilder, fragment: Fragment) -> ByteDFA:
    closure_cache: dict[frozenset[int], frozenset[int]] = {}

    def epsilon_closure(states: Iterable[int]) -> frozenset[int]:
        key = frozenset(states)
        cached = closure_cache.get(key)
        if cached is not None:
            return cached
        closure = set(key)
        pending = list(key)
        while pending:
            state = pending.pop()
            for target in builder.epsilon_edges[state]:
                if target not in closure:
                    closure.add(target)
                    pending.append(target)
        result = frozenset(closure)
        closure_cache[key] = result
        return result

    start_subset = epsilon_closure([fragment.start])
    subsets: list[frozenset[int]] = [frozenset(), start_subset]
    subset_ids = {start_subset: 1}
    rows: list[np.ndarray] = [np.zeros(256, dtype=np.int32)]
    accepting = [False]
    queue: deque[frozenset[int]] = deque([start_subset])

    while queue:
        subset = queue.popleft()
        state_id = subset_ids[subset]
        while len(rows) <= state_id:
            rows.append(np.zeros(256, dtype=np.int32))
            accepting.append(False)

        accepting[state_id] = fragment.end in subset
        destinations: dict[int, set[int]] = {}
        for nfa_state in subset:
            for byte, targets in builder.byte_edges[nfa_state].items():
                destinations.setdefault(byte, set()).update(targets)

        row = rows[state_id]
        for byte, targets in destinations.items():
            target_subset = epsilon_closure(targets)
            target_id = subset_ids.get(target_subset)
            if target_id is None:
                target_id = len(subsets)
                subsets.append(target_subset)
                subset_ids[target_subset] = target_id
                queue.append(target_subset)
            row[byte] = target_id

    transitions = np.stack(rows).astype(np.int32, copy=False)
    return ByteDFA(
        transitions=transitions,
        accepting=np.asarray(accepting, dtype=np.bool_),
        start_state=1,
    )

