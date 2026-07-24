from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from gpu_lr1.automata import ByteDFA
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import NamedSchema


@dataclass(frozen=True)
class CompiledSchema:
    name: str
    family: str
    dfa: ByteDFA


@dataclass
class PackedTables:
    vocabulary: Vocabulary
    schema_names: tuple[str, ...]
    schema_families: tuple[str, ...]
    state_offsets: np.ndarray
    start_states: np.ndarray
    accepting: np.ndarray
    byte_transitions: np.ndarray
    dense_mask: np.ndarray
    bitset_mask: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_next_state: np.ndarray
    next_state: np.ndarray | None
    compile_seconds: float

    @property
    def num_states(self) -> int:
        return int(self.byte_transitions.shape[0])

    @property
    def num_schemas(self) -> int:
        return len(self.schema_names)

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size

    @property
    def row_nnz(self) -> np.ndarray:
        return np.diff(self.csr_indptr)

    def schema_state_range(self, schema_id: int) -> range:
        return range(
            int(self.state_offsets[schema_id]),
            int(self.state_offsets[schema_id + 1]),
        )

    def memory_bytes(self) -> dict[str, int]:
        token_storage = self.vocabulary.size * (
            self.vocabulary.max_token_bytes + np.dtype(np.int32).itemsize
        )
        values = {
            "byte_transitions": int(self.byte_transitions.nbytes),
            "token_bytes_and_lengths": int(token_storage),
            "dense_mask": int(self.dense_mask.nbytes),
            "bitset_mask": int(self.bitset_mask.nbytes),
            "csr_token_ids": int(
                self.csr_indptr.nbytes + self.csr_indices.nbytes
            ),
            "csr_next_state": int(self.csr_next_state.nbytes),
        }
        if self.next_state is not None:
            values["dense_next_state"] = int(self.next_state.nbytes)
        return values

    def torch_tensors(self, device: torch.device | str = "cuda") -> "TorchTables":
        token_bytes, token_lengths = self.vocabulary.padded_bytes()
        target = torch.device(device)
        return TorchTables(
            state_offsets=torch.from_numpy(self.state_offsets).to(target),
            start_states=torch.from_numpy(self.start_states).to(target),
            accepting=torch.from_numpy(self.accepting).to(target),
            byte_transitions=torch.from_numpy(self.byte_transitions).to(target),
            dense_mask=torch.from_numpy(self.dense_mask).to(target),
            bitset_mask=torch.from_numpy(self.bitset_mask).to(target),
            csr_indptr=torch.from_numpy(self.csr_indptr).to(target),
            csr_indices=torch.from_numpy(self.csr_indices).to(target),
            csr_next_state=torch.from_numpy(self.csr_next_state).to(target),
            next_state=(
                torch.from_numpy(self.next_state).to(target)
                if self.next_state is not None
                else None
            ),
            token_bytes=torch.from_numpy(token_bytes).to(target),
            token_lengths=torch.from_numpy(token_lengths).to(target),
        )


@dataclass
class TorchTables:
    state_offsets: torch.Tensor
    start_states: torch.Tensor
    accepting: torch.Tensor
    byte_transitions: torch.Tensor
    dense_mask: torch.Tensor
    bitset_mask: torch.Tensor
    csr_indptr: torch.Tensor
    csr_indices: torch.Tensor
    csr_next_state: torch.Tensor
    next_state: torch.Tensor | None
    token_bytes: torch.Tensor
    token_lengths: torch.Tensor


def compile_packed_tables(
    schemas: Iterable[CompiledSchema],
    vocabulary: Vocabulary,
    *,
    device: torch.device | str = "cuda",
    include_next_state: bool = True,
    state_chunk_size: int = 128,
) -> PackedTables:
    compiled = list(schemas)
    if not compiled:
        raise ValueError("at least one schema is required")
    if state_chunk_size <= 0:
        raise ValueError("state_chunk_size must be positive")

    started = time.perf_counter()
    target = torch.device(device)
    token_bytes_np, token_lengths_np = vocabulary.padded_bytes()
    token_bytes = torch.from_numpy(token_bytes_np).to(target)
    token_lengths = torch.from_numpy(token_lengths_np).to(target)

    state_offsets = [0]
    start_states = []
    accepting_parts = []
    byte_transition_parts = []
    dense_parts = []
    next_parts = []

    for schema in compiled:
        offset = state_offsets[-1]
        dfa = schema.dfa
        local_dense, local_next = _compile_local_token_table(
            dfa,
            vocabulary,
            token_bytes,
            token_lengths,
            target,
            state_chunk_size,
        )
        dense_parts.append(local_dense)
        next_parts.append(local_next + np.int32(offset))

        global_byte_transitions = dfa.transitions + np.int32(offset)
        byte_transition_parts.append(global_byte_transitions)
        accepting_parts.append(dfa.accepting)
        start_states.append(offset + dfa.start_state)
        state_offsets.append(offset + dfa.num_states)

    dense_mask = np.concatenate(dense_parts, axis=0)
    bitset_mask = pack_bitset32(dense_mask)
    global_next_state = np.concatenate(next_parts, axis=0).astype(
        np.int32,
        copy=False,
    )
    csr_indptr, csr_indices, csr_next_state = dense_to_csr(
        dense_mask,
        global_next_state,
    )

    return PackedTables(
        vocabulary=vocabulary,
        schema_names=tuple(schema.name for schema in compiled),
        schema_families=tuple(schema.family for schema in compiled),
        state_offsets=np.asarray(state_offsets, dtype=np.int32),
        start_states=np.asarray(start_states, dtype=np.int32),
        accepting=np.concatenate(accepting_parts).astype(np.bool_, copy=False),
        byte_transitions=np.concatenate(byte_transition_parts, axis=0).astype(
            np.int32,
            copy=False,
        ),
        dense_mask=dense_mask,
        bitset_mask=bitset_mask,
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        csr_next_state=csr_next_state,
        next_state=(
            global_next_state
            if include_next_state
            else None
        ),
        compile_seconds=time.perf_counter() - started,
    )


def _compile_local_token_table(
    dfa: ByteDFA,
    vocabulary: Vocabulary,
    token_bytes: torch.Tensor,
    token_lengths: torch.Tensor,
    device: torch.device,
    state_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    transitions = torch.from_numpy(dfa.transitions).to(device)
    flat_transitions = transitions.reshape(-1)
    vocab_size = vocabulary.size
    dense_parts = []
    next_parts = []

    for state_start in range(0, dfa.num_states, state_chunk_size):
        state_end = min(state_start + state_chunk_size, dfa.num_states)
        states = (
            torch.arange(
                state_start,
                state_end,
                dtype=torch.int32,
                device=device,
            )[:, None]
            .expand(-1, vocab_size)
            .clone()
        )
        for byte_index in range(vocabulary.max_token_bytes):
            active_ids = torch.nonzero(
                token_lengths > byte_index,
                as_tuple=False,
            ).flatten()
            if active_ids.numel() == 0:
                break
            active_states = states[:, active_ids]
            byte_values = token_bytes[active_ids, byte_index].to(torch.int32)
            states[:, active_ids] = flat_transitions[
                active_states * np.int32(256) + byte_values[None, :]
            ]

        allowed = states != np.int32(dfa.dead_state)
        eos = vocabulary.eos_token_id
        initial_states = torch.arange(
            state_start,
            state_end,
            dtype=torch.int32,
            device=device,
        )
        accepting = torch.from_numpy(
            dfa.accepting[state_start:state_end]
        ).to(device)
        allowed[:, eos] = accepting
        states[:, eos] = initial_states

        dense_parts.append(allowed.to(torch.uint8).cpu().numpy())
        next_parts.append(states.cpu().numpy().astype(np.int32, copy=False))

    return np.concatenate(dense_parts), np.concatenate(next_parts)


def pack_bitset32(dense_mask: np.ndarray) -> np.ndarray:
    if dense_mask.ndim != 2:
        raise ValueError("dense mask must be two-dimensional")
    rows, vocab_size = dense_mask.shape
    words = (vocab_size + 31) // 32
    padded_size = words * 32
    if padded_size != vocab_size:
        padded = np.zeros((rows, padded_size), dtype=np.uint8)
        padded[:, :vocab_size] = dense_mask
    else:
        padded = np.ascontiguousarray(dense_mask, dtype=np.uint8)
    packed_bytes = np.packbits(padded, axis=1, bitorder="little")
    packed_words = np.ascontiguousarray(packed_bytes).view(np.int32)
    return packed_words.reshape(rows, words)


def unpack_bitset32(bitset: np.ndarray, vocab_size: int) -> np.ndarray:
    packed_bytes = np.ascontiguousarray(bitset).view(np.uint8)
    unpacked = np.unpackbits(packed_bytes, axis=1, bitorder="little")
    return unpacked[:, :vocab_size].astype(np.uint8, copy=False)


def dense_to_csr(
    dense_mask: np.ndarray,
    next_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = dense_mask.sum(axis=1, dtype=np.int64)
    indptr = np.empty(dense_mask.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    rows, columns = np.nonzero(dense_mask)
    if indptr[-1] > np.iinfo(np.int32).max:
        raise ValueError("CSR table exceeds int32 indexing")
    packed_indptr = indptr.astype(np.int32)
    packed_columns = columns.astype(np.int32, copy=False)
    if next_state is None:
        return packed_indptr, packed_columns
    packed_next = next_state[rows, columns].astype(np.int32, copy=False)
    return packed_indptr, packed_columns, packed_next


def table_summary(tables: PackedTables) -> dict[str, Any]:
    row_nnz = tables.row_nnz
    memory = tables.memory_bytes()
    byte_runtime = (
        memory["bitset_mask"]
        + memory["byte_transitions"]
        + memory["token_bytes_and_lengths"]
    )
    csr_runtime = (
        memory["csr_token_ids"]
        + memory["byte_transitions"]
        + memory["token_bytes_and_lengths"]
    )
    csr_fused_runtime = memory["csr_token_ids"] + memory["csr_next_state"]
    dense_next = memory.get("dense_next_state", 0)
    return {
        "schemas": tables.num_schemas,
        "states": tables.num_states,
        "vocab_size": tables.vocab_size,
        "compile_seconds": tables.compile_seconds,
        "allowed_tokens": {
            "min": int(row_nnz.min()),
            "median": float(np.median(row_nnz)),
            "p95": float(np.percentile(row_nnz, 95)),
            "max": int(row_nnz.max()),
            "mean_density": float(row_nnz.mean() / tables.vocab_size),
        },
        "memory_bytes": memory,
        "runtime_strategy_bytes": {
            "bitset_plus_byte_dfa": int(byte_runtime),
            "csr_plus_byte_dfa": int(csr_runtime),
            "csr_token_and_next": int(csr_fused_runtime),
            "bitset_plus_dense_next": int(memory["bitset_mask"] + dense_next),
            "dense_mask_plus_dense_next": int(
                memory["dense_mask"] + dense_next
            ),
        },
        "memory_total_bytes": int(sum(memory.values())),
    }


def compile_named_schemas(
    named_schemas: Iterable[NamedSchema],
) -> list[CompiledSchema]:
    from gpu_lr1.schema import CanonicalJSONSchemaCompiler

    return [
        CompiledSchema(
            name=item.name,
            family=item.family,
            dfa=CanonicalJSONSchemaCompiler(item.schema).compile(),
        )
        for item in named_schemas
    ]
