"""
The DSL channel and the neutral trace builder — ports of
`eta-dsl/src/channel.rs` and `eta-dsl/src/builder.rs` (+ `lint.rs`).

`DslChannel` is the trace-side half of a channel (declaration + endpoint
claims). `Builder` takes descriptor-port bindings and stage closures, traces
the closures once into the canonical `TraceContainer`, derives every
channel's host role, and runs the span lints. It knows nothing of WIT; the
`bridge` module wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ir import (
    ChannelDecl,
    HostRole,
    Port,
    PortBinding,
    SinkScope,
    Stage,
    StageProgram,
    TraceContainer,
    container_hash,
    numel,
    shape_of,
)
from .trace import (
    ChannelState,
    SinkCall,
    StageResult,
    TraceError,
    intern_channel,
    is_tracing,
    next_gid,
    record_channel_put,
    record_channel_read,
    trace_stage,
    with_constants,
    with_session,
)
from .value import ConstData, Tensor, const_data, materialize, reshape_id_to


class DslChannel:
    """A handle to a channel's trace state. Inside a traced stage,
    `take`/`read`/`put` record the IR's channel ops."""

    __slots__ = ("state",)

    def __init__(self, state: ChannelState) -> None:
        self.state = state

    @staticmethod
    def _build(shape, dtype, capacity: int, seed: ConstData | None, seeded: bool) -> "DslChannel":
        gid = next_gid()
        state = ChannelState(
            gid=gid,
            name=f"ch{gid}",
            shape=shape_of(shape),
            dtype=dtype,
            capacity=capacity,
            seeded=seeded or seed is not None,
            has_seed=seed is not None,
        )
        return DslChannel(state)

    @staticmethod
    def new(shape, dtype) -> "DslChannel":
        return DslChannel._build(shape, dtype, 1, None, False)

    @staticmethod
    def from_const(data: ConstData) -> "DslChannel":
        return DslChannel._build(data.shape, data.dtype, 1, data, True)

    @staticmethod
    def seeded(shape, dtype) -> "DslChannel":
        return DslChannel._build(shape, dtype, 1, None, True)

    # -- declaration ------------------------------------------------------

    def capacity(self, n: int) -> "DslChannel":
        self.state.capacity = n
        return self

    def named(self, name: str) -> "DslChannel":
        self.state.name = name
        return self

    @property
    def gid(self) -> int:
        return self.state.gid

    @property
    def name(self) -> str:
        return self.state.name

    @property
    def shape(self):
        return self.state.shape

    @property
    def dtype(self):
        return self.state.dtype

    @property
    def is_seeded(self) -> bool:
        return self.state.seeded

    # -- endpoint claims ----------------------------------------------------

    def note_host_put(self) -> None:
        self.state.host_puts += 1

    def note_host_take(self) -> None:
        self.state.host_takes += 1

    def note_host_read(self) -> None:
        self.state.host_reads += 1

    def note_desc_claim(self, consumes: bool) -> None:
        if consumes:
            self.state.desc_takes += 1
        else:
            self.state.desc_reads += 1

    # -- in-program ops ------------------------------------------------------

    def take(self) -> Tensor:
        if not is_tracing():
            raise TraceError(
                f"channel {self.name}: take() outside a stage body is a host operation; "
                "use `await ch.take_host()`"
            )
        vid, ty = record_channel_read(self.state, True)
        return Tensor.node(vid, ty)

    def read(self) -> Tensor:
        if not is_tracing():
            raise TraceError(
                f"channel {self.name}: read() outside a stage body is a host operation; "
                "use `await ch.read_host()`"
            )
        vid, ty = record_channel_read(self.state, False)
        return Tensor.node(vid, ty)

    def put_tensor(self, t: Tensor) -> None:
        if not is_tracing():
            raise TraceError(f"channel {self.name}: put(Tensor) outside a traced stage")
        vid, ty = materialize(t)
        fitted = reshape_id_to(vid, ty, self.state.shape)
        record_channel_put(self.state, fitted)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class Traced:
    """A traced, linted forward pass: the canonical container plus the
    channels it references, in container (dense) order."""

    container: TraceContainer
    channels: list[ChannelState]

    def encode(self) -> bytes:
        return self.container.encode()

    def identity_hash(self) -> int:
        return container_hash(self.encode())


class Builder:
    def __init__(self, vocab: int, page_size: int) -> None:
        self.ports: list[tuple[Port, DslChannel]] = []
        self.stages: list[tuple[Stage, Callable[[], None]]] = []
        self.vocab = vocab
        self.page_size = page_size

    def bind_port(self, port: Port, source: DslChannel) -> None:
        """Bind a descriptor port to a channel, recording the port's endpoint
        claim per its consumption discipline."""
        source.note_desc_claim(port.consumes)
        self.ports.append((port, source))

    def stage(self, stage: Stage, body: Callable[[], None]) -> None:
        for i, (s, _) in enumerate(self.stages):
            if s == stage:
                self.stages[i] = (stage, body)
                return
        self.stages.append((stage, body))

    def _channel_port(self, port: Port) -> DslChannel | None:
        for p, ch in self.ports:
            if p == port:
                return ch
        return None

    def _rows(self) -> int:
        ch = self._channel_port(Port.READOUT)
        if ch is not None:
            return max(min(numel(ch.shape), 0xFFFF_FFFF), 1)
        ch = self._channel_port(Port.EMBED_INDPTR)
        if ch is not None:
            return max(min(numel(ch.shape), 0xFFFF_FFFF) - 1, 1)
        return 1

    def _record(self, rows: int) -> tuple[list[StageResult], list[tuple[Port, int]]]:
        ports: list[tuple[Port, int]] = []
        for port, source in self.ports:
            ports.append((port, intern_channel(source.state)))
        results: list[StageResult] = []
        for stage in Stage:
            body = None
            for s, b in self.stages:
                if s == stage:
                    body = b
            if body is None:
                continue
            results.append(trace_stage(stage, rows, body))
        return results, ports

    def build(self) -> Traced:
        rows = self._rows()
        (stage_results, ports), channels, names = with_constants(
            self.vocab, self.page_size, lambda: with_session(lambda: self._record(rows))
        )

        # Re-key the container to gid (declaration) order.
        order = sorted(range(len(channels)), key=lambda i: channels[i].gid)
        remap = [0] * len(channels)
        for new_idx, old_idx in enumerate(order):
            remap[old_idx] = new_idx
        channels = [channels[i] for i in order]
        for r in stage_results:
            for op in r.ops:
                if op.chan >= 0:
                    op.chan = remap[op.chan]
        ports = [(p, remap[ci]) for p, ci in ports]

        # Name table: strictly sorted and unique.
        name_order = sorted(range(len(names)), key=lambda i: names[i])
        name_remap = [0] * len(names)
        for new_idx, old_idx in enumerate(name_order):
            name_remap[old_idx] = new_idx
        names = [names[i] for i in name_order]
        for r in stage_results:
            for op in r.ops:
                if op.tag in (0xA1, 0xA2):
                    op.name_idx = name_remap[op.name_idx]

        sinks: list[tuple[Stage, SinkCall]] = [(r.stage, s) for r in stage_results for s in r.sinks]

        decls: list[ChannelDecl] = []
        for st in channels:
            has_prog_put = bool(st.prog_puts)
            has_prog_consume = bool(st.prog_takes) or bool(st.prog_reads)
            has_desc_use = st.desc_takes > 0 or st.desc_reads > 0
            has_host_put = st.host_puts > 0
            host_consumes = st.host_takes > 0 or st.host_reads > 0
            is_terminal_output = (
                has_prog_put
                and not has_prog_consume
                and not has_desc_use
                and not has_host_put
                and not st.seeded
                and not st.has_seed
            )
            seeded_descriptor_writer = st.seeded and has_desc_use and not has_prog_put
            if (has_host_put or seeded_descriptor_writer) and not has_prog_put:
                host_role = HostRole.WRITER
            elif host_consumes and (bool(st.prog_takes) or has_prog_put):
                host_role = HostRole.READER
            elif is_terminal_output:
                host_role = HostRole.READER
            else:
                host_role = HostRole.NONE
            seeded = st.seeded or (has_host_put and has_prog_put)
            decls.append(ChannelDecl(st.shape, st.dtype, st.capacity, host_role, seeded))

        stages = [StageProgram(r.stage, r.ops) for r in stage_results]
        port_bindings = [PortBinding(p, ci) for p, ci in sorted(ports, key=lambda x: int(x[0]))]
        container = TraceContainer(names=names, channels=decls, ports=port_bindings, stages=stages)

        _lint(channels, sinks)

        return Traced(container=container, channels=channels)


def _lint(channels: list[ChannelState], sinks: list[tuple[Stage, SinkCall]]) -> None:
    errs: list[str] = []
    for st in channels:
        host_writes = st.host_puts > 0
        host_consumes = st.host_takes > 0 or st.host_reads > 0
        stage_puts = bool(st.prog_puts)
        stage_consumes = bool(st.prog_takes) or st.desc_takes > 0
        if host_writes and host_consumes:
            errs.append(
                f"channel `{st.name}` has two host endpoints (host writes and host consumes); "
                "SPSC needs one pass endpoint"
            )
        produced = stage_puts or host_writes or st.seeded or st.has_seed
        consumed = stage_consumes or bool(st.prog_reads) or st.desc_reads > 0 or host_consumes
        if consumed and not produced:
            errs.append(f"channel `{st.name}` is consumed but never produced or seeded")
    for stage, s in sinks:
        if s.scope == SinkScope.PASS_WIDE:
            ok = stage == Stage.PROLOGUE
        else:
            ok = stage in (Stage.PROLOGUE, Stage.ON_ATTN_PROJ)
        if not ok:
            errs.append(f"sink `{s.name}` is misplaced in stage `{stage.wire_name}`")
    if errs:
        raise TraceError("; ".join(errs))
