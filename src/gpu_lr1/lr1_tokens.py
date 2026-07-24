from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from gpu_lr1.kernels import triton_csr_argmax_advance
from gpu_lr1.lr1 import (
    ACTION_ACCEPT,
    ACTION_ERROR,
    CompiledLR1,
    LR1StepStatus,
    decode_reduce,
    decode_shift,
)
from gpu_lr1.vocab import Vocabulary


class LR1ConfigurationLimitError(RuntimeError):
    pass


class LR1TokenCompileTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class LR1TokenVocabulary:
    vocabulary: Vocabulary
    terminal_sequences: tuple[tuple[int, ...] | None, ...]
    name: str

    def __post_init__(self) -> None:
        if len(self.terminal_sequences) != self.vocabulary.size:
            raise ValueError("one terminal sequence is required per token")
        eos_sequence = self.terminal_sequences[self.vocabulary.eos_token_id]
        if eos_sequence != ():
            raise ValueError("EOS must use an empty terminal sequence")
        for token_id, sequence in enumerate(self.terminal_sequences):
            if (
                token_id != self.vocabulary.eos_token_id
                and sequence is not None
                and not sequence
            ):
                raise ValueError("non-EOS tokens cannot use empty sequences")

    @property
    def size(self) -> int:
        return self.vocabulary.size

    @property
    def eos_token_id(self) -> int:
        return self.vocabulary.eos_token_id

    @property
    def representable_tokens(self) -> int:
        return sum(sequence is not None for sequence in self.terminal_sequences)

    @property
    def max_terminals_per_token(self) -> int:
        return max(
            (
                len(sequence)
                for sequence in self.terminal_sequences
                if sequence is not None
            ),
            default=0,
        )

    @classmethod
    def from_symbol_sequences(
        cls,
        compiled: CompiledLR1,
        vocabulary: Vocabulary,
        sequences: Sequence[Sequence[str | int] | None],
        *,
        name: str | None = None,
    ) -> "LR1TokenVocabulary":
        if len(sequences) != vocabulary.size:
            raise ValueError("one symbol sequence is required per token")
        terminal_ids = {
            terminal: index
            for index, terminal in enumerate(compiled.terminal_names)
        }
        converted: list[tuple[int, ...] | None] = []
        for token_id, sequence in enumerate(sequences):
            if sequence is None:
                converted.append(None)
                continue
            if token_id == vocabulary.eos_token_id:
                if len(sequence) != 0:
                    raise ValueError("EOS symbol sequence must be empty")
                converted.append(())
                continue
            converted_sequence = tuple(
                terminal_ids[symbol]
                if isinstance(symbol, str)
                else int(symbol)
                for symbol in sequence
            )
            if any(
                terminal < 0 or terminal >= compiled.num_terminals
                for terminal in converted_sequence
            ):
                raise ValueError("token sequence contains an invalid terminal id")
            if compiled.eof_terminal in converted_sequence:
                raise ValueError("non-EOS tokens cannot contain the EOF terminal")
            converted.append(converted_sequence)
        return cls(
            vocabulary=vocabulary,
            terminal_sequences=tuple(converted),
            name=name or f"{compiled.grammar.name}-{vocabulary.name}",
        )

    @classmethod
    def from_byte_vocabulary(
        cls,
        compiled: CompiledLR1,
        vocabulary: Vocabulary,
        byte_terminals: Mapping[int, str],
        *,
        name: str | None = None,
    ) -> "LR1TokenVocabulary":
        terminal_ids = {
            terminal: index
            for index, terminal in enumerate(compiled.terminal_names)
        }
        byte_ids: dict[int, int] = {}
        for byte, terminal in byte_terminals.items():
            if not 0 <= byte <= 255:
                raise ValueError(f"byte mapping is out of range: {byte}")
            if terminal not in terminal_ids:
                raise ValueError(
                    f"byte mapping references unknown terminal: {terminal}"
                )
            terminal_id = terminal_ids[terminal]
            if terminal_id == compiled.eof_terminal:
                raise ValueError("byte mappings cannot target the EOF terminal")
            byte_ids[byte] = terminal_id

        sequences: list[tuple[int, ...] | None] = []
        for token_id, token in enumerate(vocabulary.tokens):
            if token_id == vocabulary.eos_token_id:
                sequences.append(())
                continue
            if not token or any(byte not in byte_ids for byte in token):
                sequences.append(None)
                continue
            sequences.append(tuple(byte_ids[byte] for byte in token))
        return cls(
            vocabulary=vocabulary,
            terminal_sequences=tuple(sequences),
            name=name or f"{compiled.grammar.name}-{vocabulary.name}-bytes",
        )


@dataclass(frozen=True)
class BoundedLR1TokenAutomaton:
    grammar_name: str
    token_vocabulary: LR1TokenVocabulary
    max_stack_depth: int
    config_stacks: tuple[tuple[int, ...], ...]
    accepting: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_next_state: np.ndarray
    csr_reductions: np.ndarray
    overflow_edges: int
    compile_seconds: float

    @property
    def num_states(self) -> int:
        return len(self.config_stacks)

    @property
    def vocab_size(self) -> int:
        return self.token_vocabulary.size

    @property
    def start_state(self) -> int:
        return 0

    @property
    def row_nnz(self) -> np.ndarray:
        return np.diff(self.csr_indptr)

    @property
    def config_depths(self) -> np.ndarray:
        return np.asarray(
            [len(stack) for stack in self.config_stacks],
            dtype=np.int32,
        )

    def next_state(self, state: int, token: int) -> int:
        start = int(self.csr_indptr[state])
        end = int(self.csr_indptr[state + 1])
        offset = int(np.searchsorted(self.csr_indices[start:end], token))
        index = start + offset
        if index < end and int(self.csr_indices[index]) == token:
            return int(self.csr_next_state[index])
        return -1


def compile_bounded_lr1_token_automaton(
    compiled: CompiledLR1,
    token_vocabulary: LR1TokenVocabulary,
    *,
    max_stack_depth: int,
    max_configurations: int = 100_000,
    max_reductions_per_terminal: int = 256,
    max_compile_seconds: float | None = None,
) -> BoundedLR1TokenAutomaton:
    if max_stack_depth < 1:
        raise ValueError("max_stack_depth must be positive")
    if max_configurations < 1:
        raise ValueError("max_configurations must be positive")
    if max_reductions_per_terminal < 0:
        raise ValueError("max_reductions_per_terminal must be non-negative")
    if max_compile_seconds is not None and max_compile_seconds < 0:
        raise ValueError("max_compile_seconds must be non-negative")

    started = time.perf_counter()
    trie = _TerminalTrie(token_vocabulary)
    start_stack = (compiled.start_state,)
    configs = [start_stack]
    config_ids = {start_stack: 0}
    queue = deque([0])
    rows: list[dict[int, int]] = []
    reduction_rows: list[dict[int, int]] = []
    accepting: list[bool] = []
    overflow_edges = 0

    while queue:
        if (
            max_compile_seconds is not None
            and time.perf_counter() - started > max_compile_seconds
        ):
            raise LR1TokenCompileTimeoutError(
                f"grammar {compiled.grammar.name!r} exceeded "
                f"{max_compile_seconds:.3f}s bounded token compile limit"
            )
        state = queue.popleft()
        stack = configs[state]
        row: dict[int, int] = {}
        reduction_row: dict[int, int] = {}

        eos_status, _, eos_reductions = _advance_terminal(
            compiled,
            stack,
            compiled.eof_terminal,
            max_stack_depth=max_stack_depth,
            max_reductions=max_reductions_per_terminal,
        )
        is_accepting = eos_status == LR1StepStatus.ACCEPTED
        accepting.append(is_accepting)
        if is_accepting:
            row[token_vocabulary.eos_token_id] = state
            reduction_row[token_vocabulary.eos_token_id] = eos_reductions

        pending: list[tuple[int, tuple[int, ...], int]] = [(0, stack, 0)]
        while pending:
            trie_node, current_stack, reductions = pending.pop()
            for terminal, target_node in trie.children[trie_node].items():
                status, next_stack, step_reductions = _advance_terminal(
                    compiled,
                    current_stack,
                    terminal,
                    max_stack_depth=max_stack_depth,
                    max_reductions=max_reductions_per_terminal,
                )
                if status == LR1StepStatus.OVERFLOW:
                    overflow_edges += 1
                    continue
                if status != LR1StepStatus.SHIFTED:
                    continue
                total_reductions = reductions + step_reductions
                for token_id in trie.token_ids[target_node]:
                    next_state = config_ids.get(next_stack)
                    if next_state is None:
                        if len(configs) >= max_configurations:
                            raise LR1ConfigurationLimitError(
                                f"grammar {compiled.grammar.name!r} exceeded "
                                f"{max_configurations} bounded stack configurations"
                            )
                        next_state = len(configs)
                        configs.append(next_stack)
                        config_ids[next_stack] = next_state
                        queue.append(next_state)
                    row[token_id] = next_state
                    reduction_row[token_id] = total_reductions
                pending.append((target_node, next_stack, total_reductions))

        rows.append(row)
        reduction_rows.append(reduction_row)

    csr_indptr, csr_indices, csr_next_state = _dict_rows_to_csr(rows)
    _, reduction_indices, csr_reductions = _dict_rows_to_csr(reduction_rows)
    if not np.array_equal(csr_indices, reduction_indices):
        raise AssertionError("token and reduction CSR rows diverged")
    return BoundedLR1TokenAutomaton(
        grammar_name=compiled.grammar.name,
        token_vocabulary=token_vocabulary,
        max_stack_depth=max_stack_depth,
        config_stacks=tuple(configs),
        accepting=np.asarray(accepting, dtype=np.bool_),
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        csr_next_state=csr_next_state,
        csr_reductions=csr_reductions,
        overflow_edges=overflow_edges,
        compile_seconds=time.perf_counter() - started,
    )


@dataclass
class PackedLR1TokenTables:
    vocabulary: Vocabulary
    grammar_names: tuple[str, ...]
    state_offsets: np.ndarray
    start_states: np.ndarray
    accepting: np.ndarray
    config_depths: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_next_state: np.ndarray
    csr_reductions: np.ndarray
    compile_seconds: float
    overflow_edges: int

    @property
    def num_grammars(self) -> int:
        return len(self.grammar_names)

    @property
    def num_states(self) -> int:
        return int(self.state_offsets[-1])

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size

    @property
    def row_nnz(self) -> np.ndarray:
        return np.diff(self.csr_indptr)

    @property
    def max_row_nnz(self) -> int:
        return int(self.row_nnz.max(initial=0))

    def memory_bytes(self) -> dict[str, int]:
        return {
            "csr_tokens": int(
                self.csr_indptr.nbytes + self.csr_indices.nbytes
            ),
            "csr_next_state": int(self.csr_next_state.nbytes),
            "csr_reductions": int(self.csr_reductions.nbytes),
            "state_metadata": int(
                self.state_offsets.nbytes
                + self.start_states.nbytes
                + self.accepting.nbytes
                + self.config_depths.nbytes
            ),
        }

    def torch_tensors(
        self,
        device: torch.device | str = "cuda",
    ) -> "TorchPackedLR1TokenTables":
        target = torch.device(device)
        return TorchPackedLR1TokenTables(
            state_offsets=torch.from_numpy(self.state_offsets).to(target),
            start_states=torch.from_numpy(self.start_states).to(target),
            accepting=torch.from_numpy(self.accepting).to(target),
            config_depths=torch.from_numpy(self.config_depths).to(target),
            csr_indptr=torch.from_numpy(self.csr_indptr).to(target),
            csr_indices=torch.from_numpy(self.csr_indices).to(target),
            csr_next_state=torch.from_numpy(self.csr_next_state).to(target),
            csr_reductions=torch.from_numpy(self.csr_reductions).to(target),
            vocab_size=self.vocab_size,
            max_row_nnz=self.max_row_nnz,
        )


@dataclass
class TorchPackedLR1TokenTables:
    state_offsets: torch.Tensor
    start_states: torch.Tensor
    accepting: torch.Tensor
    config_depths: torch.Tensor
    csr_indptr: torch.Tensor
    csr_indices: torch.Tensor
    csr_next_state: torch.Tensor
    csr_reductions: torch.Tensor
    vocab_size: int
    max_row_nnz: int


def pack_bounded_lr1_token_automata(
    automata: Iterable[BoundedLR1TokenAutomaton],
) -> PackedLR1TokenTables:
    compiled = list(automata)
    if not compiled:
        raise ValueError("at least one bounded LR(1) token automaton is required")
    vocabulary = compiled[0].token_vocabulary.vocabulary
    for automaton in compiled[1:]:
        candidate = automaton.token_vocabulary.vocabulary
        if (
            candidate.tokens != vocabulary.tokens
            or candidate.eos_token_id != vocabulary.eos_token_id
        ):
            raise ValueError("bounded LR(1) automata must share token IDs")

    offsets = np.empty(len(compiled) + 1, dtype=np.int32)
    offsets[0] = 0
    cumulative = np.cumsum(
        [automaton.num_states for automaton in compiled],
        dtype=np.int64,
    )
    if cumulative[-1] > np.iinfo(np.int32).max:
        raise ValueError("packed LR(1) token states exceed int32 indexing")
    offsets[1:] = cumulative.astype(np.int32)

    indptr = np.empty(int(offsets[-1]) + 1, dtype=np.int32)
    indptr[0] = 0
    indices: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    reductions: list[np.ndarray] = []
    entry_offset = 0
    row_offset = 0
    accepting_parts = []
    depth_parts = []

    for grammar_id, automaton in enumerate(compiled):
        state_offset = int(offsets[grammar_id])
        counts = automaton.row_nnz.astype(np.int64)
        cumulative_entries = entry_offset + np.cumsum(counts, dtype=np.int64)
        if cumulative_entries.size and (
            cumulative_entries[-1] > np.iinfo(np.int32).max
        ):
            raise ValueError("packed LR(1) token edges exceed int32 indexing")
        indptr[row_offset + 1 : row_offset + automaton.num_states + 1] = (
            cumulative_entries.astype(np.int32)
        )
        entry_offset = int(cumulative_entries[-1]) if counts.size else entry_offset
        row_offset += automaton.num_states
        indices.append(automaton.csr_indices)
        next_states.append(
            automaton.csr_next_state + np.int32(state_offset)
        )
        reductions.append(automaton.csr_reductions)
        accepting_parts.append(automaton.accepting)
        depth_parts.append(automaton.config_depths)

    return PackedLR1TokenTables(
        vocabulary=vocabulary,
        grammar_names=tuple(
            automaton.grammar_name for automaton in compiled
        ),
        state_offsets=offsets,
        start_states=offsets[:-1].copy(),
        accepting=np.concatenate(accepting_parts).astype(np.bool_, copy=False),
        config_depths=np.concatenate(depth_parts).astype(np.int32, copy=False),
        csr_indptr=indptr,
        csr_indices=np.concatenate(indices).astype(np.int32, copy=False),
        csr_next_state=np.concatenate(next_states).astype(
            np.int32,
            copy=False,
        ),
        csr_reductions=np.concatenate(reductions).astype(
            np.int32,
            copy=False,
        ),
        compile_seconds=float(
            sum(automaton.compile_seconds for automaton in compiled)
        ),
        overflow_edges=sum(automaton.overflow_edges for automaton in compiled),
    )


def select_and_advance_bounded_lr1_cpu(
    logits: np.ndarray,
    tables: PackedLR1TokenTables,
    states: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_ids = np.asarray(states, dtype=np.int32)
    if logits.shape != (state_ids.size, tables.vocab_size):
        raise ValueError("logits shape does not match bounded LR(1) batch")
    tokens = np.full(state_ids.size, -1, dtype=np.int32)
    next_states = np.full(state_ids.size, -1, dtype=np.int32)
    for batch_id, state in enumerate(state_ids):
        start = int(tables.csr_indptr[state])
        end = int(tables.csr_indptr[state + 1])
        if start == end:
            continue
        candidates = tables.csr_indices[start:end]
        offset = int(np.argmax(logits[batch_id, candidates]))
        tokens[batch_id] = int(candidates[offset])
        next_states[batch_id] = int(tables.csr_next_state[start + offset])
    return tokens, next_states


def triton_bounded_lr1_step(
    logits: torch.Tensor,
    tables: TorchPackedLR1TokenTables,
    states: torch.Tensor,
    *,
    output_tokens: torch.Tensor | None = None,
    output_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or states.shape != (logits.shape[0],):
        raise ValueError("states must contain one bounded LR state per row")
    if logits.shape[1] < tables.vocab_size:
        raise ValueError("logits do not cover the bounded LR token vocabulary")
    return triton_csr_argmax_advance(
        logits,
        tables.csr_indptr,
        tables.csr_indices,
        tables.csr_next_state,
        states,
        max_row_nnz=tables.max_row_nnz,
        output_tokens=output_tokens,
        output_states=output_states,
    )


@dataclass
class _TerminalTrie:
    children: list[dict[int, int]]
    token_ids: list[list[int]]

    def __init__(self, vocabulary: LR1TokenVocabulary) -> None:
        self.children = [{}]
        self.token_ids = [[]]
        for token_id, sequence in enumerate(vocabulary.terminal_sequences):
            if token_id == vocabulary.eos_token_id or not sequence:
                continue
            node = 0
            for terminal in sequence:
                child = self.children[node].get(terminal)
                if child is None:
                    child = len(self.children)
                    self.children[node][terminal] = child
                    self.children.append({})
                    self.token_ids.append([])
                node = child
            self.token_ids[node].append(token_id)


def _advance_terminal(
    compiled: CompiledLR1,
    stack: tuple[int, ...],
    terminal: int,
    *,
    max_stack_depth: int,
    max_reductions: int,
) -> tuple[LR1StepStatus, tuple[int, ...], int]:
    values = list(stack)
    reductions = 0
    while True:
        action = compiled.action(values[-1], terminal)
        if action == int(ACTION_ERROR):
            return LR1StepStatus.ERROR, stack, reductions
        if action == int(ACTION_ACCEPT):
            return LR1StepStatus.ACCEPTED, tuple(values), reductions
        if action > 0:
            if len(values) >= max_stack_depth:
                return LR1StepStatus.OVERFLOW, stack, reductions
            values.append(decode_shift(action))
            return LR1StepStatus.SHIFTED, tuple(values), reductions
        if reductions >= max_reductions:
            return LR1StepStatus.REDUCTION_LIMIT, stack, reductions

        production = decode_reduce(action)
        pop_count = int(compiled.production_rhs_len[production])
        if pop_count >= len(values):
            return LR1StepStatus.ERROR, stack, reductions
        if pop_count:
            del values[-pop_count:]
        lhs = int(compiled.production_lhs[production])
        target = compiled.goto(values[-1], lhs)
        if target < 0:
            return LR1StepStatus.ERROR, stack, reductions
        if len(values) >= max_stack_depth:
            return LR1StepStatus.OVERFLOW, stack, reductions
        values.append(target)
        reductions += 1


def _dict_rows_to_csr(
    rows: Sequence[Mapping[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.empty(len(rows) + 1, dtype=np.int32)
    indptr[0] = 0
    columns: list[int] = []
    values: list[int] = []
    for row_id, row in enumerate(rows):
        for column, value in sorted(row.items()):
            columns.append(column)
            values.append(value)
        if len(columns) > np.iinfo(np.int32).max:
            raise ValueError("bounded LR(1) token table exceeds int32 indexing")
        indptr[row_id + 1] = len(columns)
    return (
        indptr,
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.int32),
    )
