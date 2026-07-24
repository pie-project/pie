from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


ACTION_ERROR = np.int32(0)
ACTION_ACCEPT = np.int32(np.iinfo(np.int32).min)


def encode_shift(state: int) -> np.int32:
    if state < 0 or state >= np.iinfo(np.int32).max:
        raise ValueError(f"shift state is out of range: {state}")
    return np.int32(state + 1)


def decode_shift(action: int) -> int:
    if action <= 0:
        raise ValueError(f"action is not a shift: {action}")
    return action - 1


def encode_reduce(production: int) -> np.int32:
    if production < 0 or production >= np.iinfo(np.int32).max - 1:
        raise ValueError(f"production is out of range: {production}")
    return np.int32(-(production + 1))


def decode_reduce(action: int) -> int:
    if action >= 0 or action == int(ACTION_ACCEPT):
        raise ValueError(f"action is not a reduction: {action}")
    return -action - 1


class LR1ConflictError(ValueError):
    pass


class LR1StepStatus(IntEnum):
    SHIFTED = 0
    ACCEPTED = 1
    ERROR = 2
    OVERFLOW = 3
    REDUCTION_LIMIT = 4


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.lhs:
            raise ValueError("production lhs must not be empty")
        if any(not symbol for symbol in self.rhs):
            raise ValueError("production symbols must not be empty")


@dataclass(frozen=True)
class Grammar:
    name: str
    start: str
    productions: tuple[Production, ...]
    terminals: tuple[str, ...]
    eof: str = "$"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("grammar name must not be empty")
        if not self.productions:
            raise ValueError("grammar requires at least one production")
        if len(set(self.terminals)) != len(self.terminals):
            raise ValueError("terminal names must be unique")

        nonterminals = {production.lhs for production in self.productions}
        if self.start not in nonterminals:
            raise ValueError(f"start symbol has no production: {self.start}")
        if self.eof in nonterminals or self.eof in self.terminals:
            raise ValueError("EOF symbol must be distinct from grammar symbols")
        overlap = nonterminals.intersection(self.terminals)
        if overlap:
            raise ValueError(
                "symbols cannot be both terminal and nonterminal: "
                + ", ".join(sorted(overlap))
            )

        known = nonterminals.union(self.terminals)
        unknown = {
            symbol
            for production in self.productions
            for symbol in production.rhs
            if symbol not in known
        }
        if unknown:
            raise ValueError(
                "unknown production symbols: " + ", ".join(sorted(unknown))
            )

    @classmethod
    def from_rules(
        cls,
        name: str,
        start: str,
        rules: Mapping[str, Iterable[Sequence[str]]],
        *,
        terminals: Iterable[str] | None = None,
        eof: str = "$",
    ) -> "Grammar":
        productions: list[Production] = []
        nonterminals = tuple(rules)
        nonterminal_set = set(nonterminals)
        inferred_terminals: list[str] = []
        seen_terminals: set[str] = set()

        for lhs, alternatives in rules.items():
            for alternative in alternatives:
                if isinstance(alternative, str):
                    raise TypeError(
                        "production alternatives must be symbol sequences, "
                        "not strings"
                    )
                rhs = tuple(alternative)
                productions.append(Production(lhs, rhs))
                if terminals is None:
                    for symbol in rhs:
                        if (
                            symbol not in nonterminal_set
                            and symbol not in seen_terminals
                        ):
                            inferred_terminals.append(symbol)
                            seen_terminals.add(symbol)

        terminal_names = (
            tuple(terminals)
            if terminals is not None
            else tuple(inferred_terminals)
        )
        return cls(
            name=name,
            start=start,
            productions=tuple(productions),
            terminals=terminal_names,
            eof=eof,
        )


@dataclass(frozen=True, order=True)
class LR1Item:
    production: int
    dot: int
    lookahead: int


@dataclass(frozen=True)
class CompiledLR1:
    grammar: Grammar
    terminal_names: tuple[str, ...]
    nonterminal_names: tuple[str, ...]
    action_indptr: np.ndarray
    action_symbols: np.ndarray
    action_values: np.ndarray
    goto_indptr: np.ndarray
    goto_symbols: np.ndarray
    goto_targets: np.ndarray
    production_lhs: np.ndarray
    production_rhs_len: np.ndarray
    start_state: int
    eof_terminal: int
    state_items: tuple[tuple[LR1Item, ...], ...]

    def __post_init__(self) -> None:
        _validate_csr(
            self.action_indptr,
            self.action_symbols,
            self.action_values,
            len(self.state_items),
            "ACTION",
        )
        _validate_csr(
            self.goto_indptr,
            self.goto_symbols,
            self.goto_targets,
            len(self.state_items),
            "GOTO",
        )
        if self.production_lhs.shape != self.production_rhs_len.shape:
            raise ValueError("production metadata shapes do not match")

    @property
    def num_states(self) -> int:
        return len(self.state_items)

    @property
    def num_terminals(self) -> int:
        return len(self.terminal_names)

    @property
    def num_nonterminals(self) -> int:
        return len(self.nonterminal_names)

    @property
    def num_productions(self) -> int:
        return int(self.production_lhs.size)

    @property
    def action_row_nnz(self) -> np.ndarray:
        return np.diff(self.action_indptr)

    def terminal_id(self, name: str) -> int:
        try:
            return self.terminal_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown terminal: {name}") from error

    def action(self, state: int, terminal: int) -> int:
        return _csr_lookup(
            self.action_indptr,
            self.action_symbols,
            self.action_values,
            state,
            terminal,
        )

    def goto(self, state: int, nonterminal: int) -> int:
        return _csr_lookup(
            self.goto_indptr,
            self.goto_symbols,
            self.goto_targets,
            state,
            nonterminal,
            default=-1,
        )

    def allowed_terminals(self, state: int) -> np.ndarray:
        start = int(self.action_indptr[state])
        end = int(self.action_indptr[state + 1])
        return self.action_symbols[start:end]

    def accepts(self, symbols: Iterable[str | int]) -> bool:
        stack = [self.start_state]
        terminal_ids = [
            self.terminal_id(symbol) if isinstance(symbol, str) else int(symbol)
            for symbol in symbols
        ]
        terminal_ids.append(self.eof_terminal)

        for terminal in terminal_ids:
            reductions = 0
            while True:
                action = self.action(stack[-1], terminal)
                if action == int(ACTION_ERROR):
                    return False
                if action == int(ACTION_ACCEPT):
                    return terminal == self.eof_terminal
                if action > 0:
                    stack.append(decode_shift(action))
                    break

                production = decode_reduce(action)
                pop_count = int(self.production_rhs_len[production])
                if pop_count >= len(stack):
                    return False
                if pop_count:
                    del stack[-pop_count:]
                lhs = int(self.production_lhs[production])
                target = self.goto(stack[-1], lhs)
                if target < 0:
                    return False
                stack.append(target)
                reductions += 1
                if reductions > 1_000_000:
                    raise RuntimeError("reduction cycle did not consume input")
        return False


class CanonicalLR1Compiler:
    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar

    def compile(self) -> CompiledLR1:
        grammar = self.grammar
        terminal_names = grammar.terminals + (grammar.eof,)
        terminal_ids = {
            terminal: index for index, terminal in enumerate(terminal_names)
        }

        nonterminal_names = tuple(
            dict.fromkeys(production.lhs for production in grammar.productions)
        )
        nonterminal_ids = {
            nonterminal: index
            for index, nonterminal in enumerate(nonterminal_names)
        }
        augmented_start = _augmented_start_name(
            grammar.start,
            set(terminal_names).union(nonterminal_names),
        )
        internal_productions = (
            Production(augmented_start, (grammar.start,)),
        ) + grammar.productions
        productions_by_lhs: dict[str, list[int]] = {}
        for index, production in enumerate(internal_productions):
            productions_by_lhs.setdefault(production.lhs, []).append(index)

        nullable, first = _first_sets(
            grammar.productions,
            terminal_ids,
            nonterminal_names,
        )
        eof_terminal = terminal_ids[grammar.eof]

        def first_with_lookahead(
            sequence: Sequence[str],
            lookahead: int,
        ) -> set[int]:
            output: set[int] = set()
            for symbol in sequence:
                terminal = terminal_ids.get(symbol)
                if terminal is not None:
                    output.add(terminal)
                    return output
                output.update(first[symbol])
                if symbol not in nullable:
                    return output
            output.add(lookahead)
            return output

        def closure(seed: Iterable[LR1Item]) -> frozenset[LR1Item]:
            items = set(seed)
            pending = list(items)
            while pending:
                item = pending.pop()
                production = internal_productions[item.production]
                if item.dot >= len(production.rhs):
                    continue
                symbol = production.rhs[item.dot]
                referenced = productions_by_lhs.get(symbol)
                if referenced is None:
                    continue
                lookaheads = first_with_lookahead(
                    production.rhs[item.dot + 1 :],
                    item.lookahead,
                )
                for production_id in referenced:
                    for lookahead in lookaheads:
                        candidate = LR1Item(production_id, 0, lookahead)
                        if candidate not in items:
                            items.add(candidate)
                            pending.append(candidate)
            return frozenset(items)

        def advance(
            items: frozenset[LR1Item],
            symbol: str,
        ) -> frozenset[LR1Item]:
            shifted = [
                LR1Item(item.production, item.dot + 1, item.lookahead)
                for item in items
                if (
                    item.dot < len(internal_productions[item.production].rhs)
                    and internal_productions[item.production].rhs[item.dot]
                    == symbol
                )
            ]
            return closure(shifted) if shifted else frozenset()

        start_items = closure([LR1Item(0, 0, eof_terminal)])
        states = [start_items]
        state_ids = {start_items: 0}
        transitions: dict[tuple[int, str], int] = {}
        queue = deque([0])
        symbol_order = {
            symbol: index
            for index, symbol in enumerate(terminal_names + nonterminal_names)
        }

        while queue:
            state = queue.popleft()
            symbols = {
                internal_productions[item.production].rhs[item.dot]
                for item in states[state]
                if item.dot < len(internal_productions[item.production].rhs)
            }
            for symbol in sorted(symbols, key=symbol_order.__getitem__):
                target_items = advance(states[state], symbol)
                target = state_ids.get(target_items)
                if target is None:
                    target = len(states)
                    states.append(target_items)
                    state_ids[target_items] = target
                    queue.append(target)
                transitions[state, symbol] = target

        action_rows: list[dict[int, int]] = [dict() for _ in states]
        goto_rows: list[dict[int, int]] = [dict() for _ in states]

        for state, items in enumerate(states):
            for item in items:
                production = internal_productions[item.production]
                if item.dot < len(production.rhs):
                    symbol = production.rhs[item.dot]
                    target = transitions[state, symbol]
                    terminal = terminal_ids.get(symbol)
                    if terminal is not None:
                        _set_action(
                            grammar,
                            state,
                            terminal_names[terminal],
                            action_rows[state],
                            terminal,
                            int(encode_shift(target)),
                            items,
                            internal_productions,
                        )
                    elif symbol != augmented_start:
                        goto_rows[state][nonterminal_ids[symbol]] = target
                    continue

                if item.production == 0:
                    if item.lookahead == eof_terminal:
                        _set_action(
                            grammar,
                            state,
                            grammar.eof,
                            action_rows[state],
                            eof_terminal,
                            int(ACTION_ACCEPT),
                            items,
                            internal_productions,
                        )
                    continue

                _set_action(
                    grammar,
                    state,
                    terminal_names[item.lookahead],
                    action_rows[state],
                    item.lookahead,
                    int(encode_reduce(item.production - 1)),
                    items,
                    internal_productions,
                )

        action_indptr, action_symbols, action_values = _dict_rows_to_csr(
            action_rows
        )
        goto_indptr, goto_symbols, goto_targets = _dict_rows_to_csr(goto_rows)
        production_lhs = np.asarray(
            [nonterminal_ids[production.lhs] for production in grammar.productions],
            dtype=np.int32,
        )
        production_rhs_len = np.asarray(
            [len(production.rhs) for production in grammar.productions],
            dtype=np.int32,
        )

        return CompiledLR1(
            grammar=grammar,
            terminal_names=terminal_names,
            nonterminal_names=nonterminal_names,
            action_indptr=action_indptr,
            action_symbols=action_symbols,
            action_values=action_values,
            goto_indptr=goto_indptr,
            goto_symbols=goto_symbols,
            goto_targets=goto_targets,
            production_lhs=production_lhs,
            production_rhs_len=production_rhs_len,
            start_state=0,
            eof_terminal=eof_terminal,
            state_items=tuple(
                tuple(sorted(state_items)) for state_items in states
            ),
        )


@dataclass
class PackedLR1Tables:
    grammar_names: tuple[str, ...]
    terminal_names: tuple[str, ...]
    state_offsets: np.ndarray
    nonterminal_offsets: np.ndarray
    production_offsets: np.ndarray
    start_states: np.ndarray
    eof_terminals: np.ndarray
    action_indptr: np.ndarray
    action_symbols: np.ndarray
    action_values: np.ndarray
    goto_indptr: np.ndarray
    goto_symbols: np.ndarray
    goto_targets: np.ndarray
    production_lhs: np.ndarray
    production_rhs_len: np.ndarray

    @property
    def num_grammars(self) -> int:
        return len(self.grammar_names)

    @property
    def num_states(self) -> int:
        return int(self.state_offsets[-1])

    @property
    def num_terminals(self) -> int:
        return len(self.terminal_names)

    @property
    def num_productions(self) -> int:
        return int(self.production_offsets[-1])

    @property
    def action_row_nnz(self) -> np.ndarray:
        return np.diff(self.action_indptr)

    @property
    def goto_row_nnz(self) -> np.ndarray:
        return np.diff(self.goto_indptr)

    @property
    def max_action_row_nnz(self) -> int:
        return int(self.action_row_nnz.max(initial=0))

    @property
    def max_goto_row_nnz(self) -> int:
        return int(self.goto_row_nnz.max(initial=0))

    def memory_bytes(self) -> dict[str, int]:
        return {
            "action": int(
                self.action_indptr.nbytes
                + self.action_symbols.nbytes
                + self.action_values.nbytes
            ),
            "goto": int(
                self.goto_indptr.nbytes
                + self.goto_symbols.nbytes
                + self.goto_targets.nbytes
            ),
            "productions": int(
                self.production_lhs.nbytes + self.production_rhs_len.nbytes
            ),
            "offsets": int(
                self.state_offsets.nbytes
                + self.nonterminal_offsets.nbytes
                + self.production_offsets.nbytes
                + self.start_states.nbytes
                + self.eof_terminals.nbytes
            ),
        }

    def torch_tensors(
        self,
        device: torch.device | str = "cuda",
    ) -> "TorchPackedLR1Tables":
        target = torch.device(device)
        return TorchPackedLR1Tables(
            state_offsets=torch.from_numpy(self.state_offsets).to(target),
            nonterminal_offsets=torch.from_numpy(
                self.nonterminal_offsets
            ).to(target),
            production_offsets=torch.from_numpy(self.production_offsets).to(
                target
            ),
            start_states=torch.from_numpy(self.start_states).to(target),
            eof_terminals=torch.from_numpy(self.eof_terminals).to(target),
            action_indptr=torch.from_numpy(self.action_indptr).to(target),
            action_symbols=torch.from_numpy(self.action_symbols).to(target),
            action_values=torch.from_numpy(self.action_values).to(target),
            goto_indptr=torch.from_numpy(self.goto_indptr).to(target),
            goto_symbols=torch.from_numpy(self.goto_symbols).to(target),
            goto_targets=torch.from_numpy(self.goto_targets).to(target),
            production_lhs=torch.from_numpy(self.production_lhs).to(target),
            production_rhs_len=torch.from_numpy(
                self.production_rhs_len
            ).to(target),
            num_terminals=self.num_terminals,
            max_action_row_nnz=self.max_action_row_nnz,
            max_goto_row_nnz=self.max_goto_row_nnz,
        )


@dataclass
class TorchPackedLR1Tables:
    state_offsets: torch.Tensor
    nonterminal_offsets: torch.Tensor
    production_offsets: torch.Tensor
    start_states: torch.Tensor
    eof_terminals: torch.Tensor
    action_indptr: torch.Tensor
    action_symbols: torch.Tensor
    action_values: torch.Tensor
    goto_indptr: torch.Tensor
    goto_symbols: torch.Tensor
    goto_targets: torch.Tensor
    production_lhs: torch.Tensor
    production_rhs_len: torch.Tensor
    num_terminals: int
    max_action_row_nnz: int
    max_goto_row_nnz: int


def pack_lr1_tables(compiled: Iterable[CompiledLR1]) -> PackedLR1Tables:
    grammars = list(compiled)
    if not grammars:
        raise ValueError("at least one LR(1) grammar is required")

    terminal_names = tuple(
        dict.fromkeys(
            terminal
            for grammar in grammars
            for terminal in grammar.terminal_names
        )
    )
    terminal_ids = {
        terminal: index for index, terminal in enumerate(terminal_names)
    }
    state_offsets = _prefix_offsets(grammar.num_states for grammar in grammars)
    nonterminal_offsets = _prefix_offsets(
        grammar.num_nonterminals for grammar in grammars
    )
    production_offsets = _prefix_offsets(
        grammar.num_productions for grammar in grammars
    )

    action_rows: list[dict[int, int]] = []
    goto_rows: list[dict[int, int]] = []
    production_lhs: list[np.ndarray] = []
    production_rhs_len: list[np.ndarray] = []

    for grammar_id, grammar in enumerate(grammars):
        state_offset = int(state_offsets[grammar_id])
        nonterminal_offset = int(nonterminal_offsets[grammar_id])
        production_offset = int(production_offsets[grammar_id])
        local_to_global_terminal = np.asarray(
            [terminal_ids[name] for name in grammar.terminal_names],
            dtype=np.int32,
        )

        for state in range(grammar.num_states):
            row: dict[int, int] = {}
            start = int(grammar.action_indptr[state])
            end = int(grammar.action_indptr[state + 1])
            for index in range(start, end):
                terminal = int(
                    local_to_global_terminal[grammar.action_symbols[index]]
                )
                action = int(grammar.action_values[index])
                if action > 0:
                    action = int(
                        encode_shift(state_offset + decode_shift(action))
                    )
                elif action != int(ACTION_ACCEPT):
                    action = int(
                        encode_reduce(
                            production_offset + decode_reduce(action)
                        )
                    )
                row[terminal] = action
            action_rows.append(row)

            goto_row: dict[int, int] = {}
            start = int(grammar.goto_indptr[state])
            end = int(grammar.goto_indptr[state + 1])
            for index in range(start, end):
                nonterminal = (
                    nonterminal_offset + int(grammar.goto_symbols[index])
                )
                target = state_offset + int(grammar.goto_targets[index])
                goto_row[nonterminal] = target
            goto_rows.append(goto_row)

        production_lhs.append(
            grammar.production_lhs + np.int32(nonterminal_offset)
        )
        production_rhs_len.append(grammar.production_rhs_len)

    action_indptr, action_symbols, action_values = _dict_rows_to_csr(action_rows)
    goto_indptr, goto_symbols, goto_targets = _dict_rows_to_csr(goto_rows)
    return PackedLR1Tables(
        grammar_names=tuple(grammar.grammar.name for grammar in grammars),
        terminal_names=terminal_names,
        state_offsets=state_offsets,
        nonterminal_offsets=nonterminal_offsets,
        production_offsets=production_offsets,
        start_states=np.asarray(
            [
                int(state_offsets[index]) + grammar.start_state
                for index, grammar in enumerate(grammars)
            ],
            dtype=np.int32,
        ),
        eof_terminals=np.asarray(
            [
                terminal_ids[grammar.terminal_names[grammar.eof_terminal]]
                for grammar in grammars
            ],
            dtype=np.int32,
        ),
        action_indptr=action_indptr,
        action_symbols=action_symbols,
        action_values=action_values,
        goto_indptr=goto_indptr,
        goto_symbols=goto_symbols,
        goto_targets=goto_targets,
        production_lhs=np.concatenate(production_lhs).astype(
            np.int32,
            copy=False,
        ),
        production_rhs_len=np.concatenate(production_rhs_len).astype(
            np.int32,
            copy=False,
        ),
    )


@dataclass
class RaggedLR1Stacks:
    values: np.ndarray
    offsets: np.ndarray
    pointers: np.ndarray

    @classmethod
    def initialize(
        cls,
        start_states: Sequence[int] | np.ndarray,
        capacities: int | Sequence[int] | np.ndarray,
    ) -> "RaggedLR1Stacks":
        starts = np.asarray(start_states, dtype=np.int32)
        if starts.ndim != 1:
            raise ValueError("start states must be one-dimensional")
        if np.isscalar(capacities):
            capacity_values = np.full(
                starts.size,
                int(capacities),
                dtype=np.int32,
            )
        else:
            capacity_values = np.asarray(capacities, dtype=np.int32)
        if capacity_values.shape != starts.shape:
            raise ValueError("one stack capacity is required per sequence")
        if np.any(capacity_values < 1):
            raise ValueError("stack capacities must be positive")

        offsets = np.empty(starts.size + 1, dtype=np.int32)
        offsets[0] = 0
        cumulative = np.cumsum(capacity_values, dtype=np.int64)
        if cumulative[-1] > np.iinfo(np.int32).max:
            raise ValueError("ragged stack pool exceeds int32 indexing")
        offsets[1:] = cumulative.astype(np.int32)
        values = np.full(int(offsets[-1]), -1, dtype=np.int32)
        values[offsets[:-1]] = starts
        return cls(
            values=values,
            offsets=offsets,
            pointers=np.ones(starts.size, dtype=np.int32),
        )

    @property
    def batch_size(self) -> int:
        return int(self.pointers.size)

    @property
    def capacities(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def top_states(self) -> np.ndarray:
        return self.values[self.offsets[:-1] + self.pointers - 1]

    def clone(self) -> "RaggedLR1Stacks":
        return RaggedLR1Stacks(
            values=self.values.copy(),
            offsets=self.offsets.copy(),
            pointers=self.pointers.copy(),
        )

    def torch_tensors(
        self,
        device: torch.device | str = "cuda",
    ) -> "TorchRaggedLR1Stacks":
        target = torch.device(device)
        return TorchRaggedLR1Stacks(
            values=torch.from_numpy(self.values).to(target),
            offsets=torch.from_numpy(self.offsets).to(target),
            pointers=torch.from_numpy(self.pointers).to(target),
        )


@dataclass
class TorchRaggedLR1Stacks:
    values: torch.Tensor
    offsets: torch.Tensor
    pointers: torch.Tensor

    def clone(self) -> "TorchRaggedLR1Stacks":
        return TorchRaggedLR1Stacks(
            values=self.values.clone(),
            offsets=self.offsets.clone(),
            pointers=self.pointers.clone(),
        )


@dataclass(frozen=True)
class LR1StepResult:
    terminals: np.ndarray
    statuses: np.ndarray
    reductions: np.ndarray


def select_lr1_terminals_cpu(
    logits: np.ndarray,
    tables: PackedLR1Tables,
    stacks: RaggedLR1Stacks,
) -> tuple[np.ndarray, np.ndarray]:
    if logits.shape != (stacks.batch_size, tables.num_terminals):
        raise ValueError("logits shape does not match LR(1) batch")
    terminals = np.full(stacks.batch_size, -1, dtype=np.int32)
    actions = np.zeros(stacks.batch_size, dtype=np.int32)
    for batch_id, state in enumerate(stacks.top_states):
        start = int(tables.action_indptr[state])
        end = int(tables.action_indptr[state + 1])
        if start == end:
            continue
        candidates = tables.action_symbols[start:end]
        selected_offset = int(np.argmax(logits[batch_id, candidates]))
        terminals[batch_id] = int(candidates[selected_offset])
        actions[batch_id] = int(tables.action_values[start + selected_offset])
    return terminals, actions


def step_lr1_terminals_cpu(
    tables: PackedLR1Tables,
    stacks: RaggedLR1Stacks,
    terminals: Sequence[int] | np.ndarray,
    *,
    max_reductions: int = 128,
    initial_actions: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    terminal_ids = np.asarray(terminals, dtype=np.int32)
    if terminal_ids.shape != stacks.pointers.shape:
        raise ValueError("one terminal is required per sequence")
    if max_reductions < 0:
        raise ValueError("max_reductions must be non-negative")
    actions = (
        np.asarray(initial_actions, dtype=np.int32)
        if initial_actions is not None
        else None
    )
    if actions is not None and actions.shape != terminal_ids.shape:
        raise ValueError("initial actions must match terminal shape")

    statuses = np.full(
        stacks.batch_size,
        int(LR1StepStatus.ERROR),
        dtype=np.int32,
    )
    reductions = np.zeros(stacks.batch_size, dtype=np.int32)

    for batch_id, terminal in enumerate(terminal_ids):
        base = int(stacks.offsets[batch_id])
        capacity = int(stacks.offsets[batch_id + 1] - base)
        pointer = int(stacks.pointers[batch_id])
        action = (
            int(actions[batch_id])
            if actions is not None
            else _csr_lookup(
                tables.action_indptr,
                tables.action_symbols,
                tables.action_values,
                int(stacks.values[base + pointer - 1]),
                int(terminal),
            )
        )

        while True:
            if action == int(ACTION_ERROR):
                statuses[batch_id] = int(LR1StepStatus.ERROR)
                break
            if action == int(ACTION_ACCEPT):
                statuses[batch_id] = int(LR1StepStatus.ACCEPTED)
                break
            if action > 0:
                if pointer >= capacity:
                    statuses[batch_id] = int(LR1StepStatus.OVERFLOW)
                    break
                stacks.values[base + pointer] = decode_shift(action)
                pointer += 1
                stacks.pointers[batch_id] = pointer
                statuses[batch_id] = int(LR1StepStatus.SHIFTED)
                break

            if reductions[batch_id] >= max_reductions:
                statuses[batch_id] = int(LR1StepStatus.REDUCTION_LIMIT)
                break
            production = decode_reduce(action)
            pop_count = int(tables.production_rhs_len[production])
            if pop_count >= pointer:
                statuses[batch_id] = int(LR1StepStatus.ERROR)
                break
            pointer -= pop_count
            exposed = int(stacks.values[base + pointer - 1])
            lhs = int(tables.production_lhs[production])
            target = _csr_lookup(
                tables.goto_indptr,
                tables.goto_symbols,
                tables.goto_targets,
                exposed,
                lhs,
                default=-1,
            )
            if target < 0:
                statuses[batch_id] = int(LR1StepStatus.ERROR)
                break
            if pointer >= capacity:
                statuses[batch_id] = int(LR1StepStatus.OVERFLOW)
                break
            stacks.values[base + pointer] = target
            pointer += 1
            stacks.pointers[batch_id] = pointer
            reductions[batch_id] += 1
            action = _csr_lookup(
                tables.action_indptr,
                tables.action_symbols,
                tables.action_values,
                target,
                int(terminal),
            )

    return statuses, reductions


def select_and_step_lr1_cpu(
    logits: np.ndarray,
    tables: PackedLR1Tables,
    stacks: RaggedLR1Stacks,
    *,
    max_reductions: int = 128,
) -> LR1StepResult:
    terminals, actions = select_lr1_terminals_cpu(logits, tables, stacks)
    statuses, reductions = step_lr1_terminals_cpu(
        tables,
        stacks,
        terminals,
        max_reductions=max_reductions,
        initial_actions=actions,
    )
    return LR1StepResult(terminals, statuses, reductions)


def _first_sets(
    productions: Sequence[Production],
    terminal_ids: Mapping[str, int],
    nonterminals: Sequence[str],
) -> tuple[set[str], dict[str, set[int]]]:
    nullable: set[str] = set()
    first = {nonterminal: set() for nonterminal in nonterminals}
    changed = True
    while changed:
        changed = False
        for production in productions:
            if not production.rhs:
                if production.lhs not in nullable:
                    nullable.add(production.lhs)
                    changed = True
                continue

            all_nullable = True
            for symbol in production.rhs:
                terminal = terminal_ids.get(symbol)
                if terminal is not None:
                    if terminal not in first[production.lhs]:
                        first[production.lhs].add(terminal)
                        changed = True
                    all_nullable = False
                    break
                previous_size = len(first[production.lhs])
                first[production.lhs].update(first[symbol])
                changed |= len(first[production.lhs]) != previous_size
                if symbol not in nullable:
                    all_nullable = False
                    break
            if all_nullable and production.lhs not in nullable:
                nullable.add(production.lhs)
                changed = True
    return nullable, first


def _set_action(
    grammar: Grammar,
    state: int,
    terminal_name: str,
    row: dict[int, int],
    terminal: int,
    action: int,
    items: frozenset[LR1Item],
    productions: Sequence[Production],
) -> None:
    existing = row.get(terminal)
    if existing is None or existing == action:
        row[terminal] = action
        return
    item_text = ", ".join(
        _format_item(item, productions, grammar.eof) for item in sorted(items)
    )
    raise LR1ConflictError(
        f"grammar {grammar.name!r} is not LR(1): state {state}, "
        f"lookahead {terminal_name!r}, actions "
        f"{_format_action(existing)} and {_format_action(action)}; "
        f"items: {item_text}"
    )


def _format_item(
    item: LR1Item,
    productions: Sequence[Production],
    eof: str,
) -> str:
    production = productions[item.production]
    rhs = list(production.rhs)
    rhs.insert(item.dot, "·")
    lookahead = eof if item.lookahead < 0 else str(item.lookahead)
    return f"{production.lhs} -> {' '.join(rhs)}, {lookahead}"


def _format_action(action: int) -> str:
    if action == int(ACTION_ERROR):
        return "error"
    if action == int(ACTION_ACCEPT):
        return "accept"
    if action > 0:
        return f"shift {decode_shift(action)}"
    return f"reduce {decode_reduce(action)}"


def _dict_rows_to_csr(
    rows: Sequence[Mapping[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.empty(len(rows) + 1, dtype=np.int32)
    indptr[0] = 0
    columns: list[int] = []
    values: list[int] = []
    for row_index, row in enumerate(rows):
        for column, value in sorted(row.items()):
            columns.append(column)
            values.append(value)
        if len(columns) > np.iinfo(np.int32).max:
            raise ValueError("sparse LR(1) table exceeds int32 indexing")
        indptr[row_index + 1] = len(columns)
    return (
        indptr,
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.int32),
    )


def _csr_lookup(
    indptr: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    row: int,
    column: int,
    *,
    default: int = 0,
) -> int:
    if row < 0 or row + 1 >= indptr.size:
        return default
    start = int(indptr[row])
    end = int(indptr[row + 1])
    offset = int(np.searchsorted(columns[start:end], column))
    index = start + offset
    if index < end and int(columns[index]) == column:
        return int(values[index])
    return default


def _validate_csr(
    indptr: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    rows: int,
    name: str,
) -> None:
    if (
        indptr.dtype != np.int32
        or columns.dtype != np.int32
        or values.dtype != np.int32
    ):
        raise TypeError(f"{name} arrays must use int32")
    if indptr.shape != (rows + 1,):
        raise ValueError(f"{name} indptr shape is invalid")
    if columns.shape != values.shape:
        raise ValueError(f"{name} column and value shapes do not match")
    if int(indptr[-1]) != columns.size:
        raise ValueError(f"{name} indptr does not cover all entries")


def _prefix_offsets(sizes: Iterable[int]) -> np.ndarray:
    values = list(sizes)
    offsets = np.empty(len(values) + 1, dtype=np.int32)
    offsets[0] = 0
    cumulative = np.cumsum(values, dtype=np.int64)
    if cumulative.size and cumulative[-1] > np.iinfo(np.int32).max:
        raise ValueError("packed LR(1) table exceeds int32 indexing")
    offsets[1:] = cumulative.astype(np.int32)
    return offsets


def _augmented_start_name(start: str, symbols: set[str]) -> str:
    candidate = f"{start}'"
    while candidate in symbols:
        candidate += "'"
    return candidate
