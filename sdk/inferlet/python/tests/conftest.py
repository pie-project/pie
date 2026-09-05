"""
Mock WIT bindings for unit-testing the inferlet SDK outside a componentize-py
build.

The real `wit_world` package only exists inside such a build, so importing
`inferlet` on a plain interpreter fails. These stubs stand in for the imports
the hand-written layer actually uses, and nothing else.

SCOPE: the non-forward interfaces only, which is the whole of the SDK's
hand-written layer today. The forward-pass surface has no Python counterpart
yet -- see `inferlet/__init__.py` and `scripts/check-sdk-interfaces.sh`. When
it lands, the channel / working-set / pipeline / forward* stubs belong here.

These are stubs, not a simulator: enough to prove the wrapper layer calls the
right binding and marshals the result, and no more. That the SDK matches the
REAL world is proven by actually building a component --

    componentize-py -d interface/inferlet -w inferlet componentize \\
        -p . -p sdk/python/src -p sdk/python/src/inferlet/bindings app

-- which is the check `release-pypi.yml` notes it cannot run. A stub can be
made to agree with a world that no longer exists; that is how the previous
version of this file kept passing against `pie:core/inference`.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# pie:inferlet/model
# ---------------------------------------------------------------------------


class ForwardKind(Enum):
    ATTENTION = 0
    RECURRENT = 1
    HYBRID = 2
    DIFFUSION = 3


class DiffusionMode(Enum):
    ENCODE = 0
    DENOISE = 1


def _make_model() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.model")
    m.ForwardKind = ForwardKind
    m.name = lambda: "mock-model"
    m.architecture = lambda: "qwen3_5"
    m.default_system_speculation = lambda: False
    m.mtp_depth = lambda: 0
    m.submit_deadline_us = lambda: 50_000
    m.pass_kind = lambda: ForwardKind.HYBRID
    m.output_vocab_size = lambda: 151936
    m.kv_page_size = lambda: 16
    m.frame_size = lambda: 1
    m.channel_capacity = lambda: 8
    m.max_embed_length = lambda: 2048
    m.rs_state_size = lambda: 4096
    m.rs_buffer_page_size = lambda: 64
    m.rs_fold_granularity = lambda: 1
    m.arena_block_size = lambda: 8192
    m.run_ahead_window = lambda: 4
    m.BlockDrafter = BlockDrafterStub
    m.CanvasShape = CanvasShapeStub
    m.draft_block = lambda: None
    m.canvas = lambda: CanvasShapeStub(32, 2560, 4)
    return m


@dataclass
class BlockDrafterStub:
    rows: int
    mask_token: int
    bidirectional: bool
    proposals_from: int


@dataclass
class CanvasShapeStub:
    length: int
    hidden: int
    self_cond_taps: int


# ---------------------------------------------------------------------------
# pie:inferlet/tokenizer
# ---------------------------------------------------------------------------


@dataclass
class _Token:
    id: int
    bytes: bytes


def _make_tokenizer() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.tokenizer")
    # Byte-identity codec: keeps assertions readable while still exercising
    # the list()/str marshalling the wrapper does.
    m.encode = lambda text: [ord(c) for c in text]
    m.decode = lambda tokens: "".join(chr(t) for t in tokens)
    m.Token = _Token
    m.vocabs = lambda: [_Token(0, b"a"), _Token(1, b"b")]
    m.split_regex = lambda: r"\w+"
    m.special_tokens = lambda: [_Token(2, b"<eos>")]
    m.token_bytes = lambda tokens: [bytes([t]) for t in tokens]
    m.tokens_with_prefix = lambda prefix: list(prefix)
    return m


# ---------------------------------------------------------------------------
# pie:inferlet/session
# ---------------------------------------------------------------------------


class SessionSpy:
    """Records what the session wrapper sent; scripts what it receives."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.sent_files: list[bytes] = []
        self.to_receive: list[str | None] = []
        self.files_to_receive: list[bytes | None] = []

    def reset(self) -> None:
        self.sent.clear()
        self.sent_files.clear()
        self.to_receive.clear()
        self.files_to_receive.clear()


SESSION_SPY = SessionSpy()


def _make_session(spy: SessionSpy) -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.session")

    def send(message: str) -> None:
        spy.sent.append(message)

    async def receive() -> str | None:
        return spy.to_receive.pop(0) if spy.to_receive else None

    def send_file(data: bytes) -> None:
        spy.sent_files.append(data)

    async def receive_file() -> bytes | None:
        return spy.files_to_receive.pop(0) if spy.files_to_receive else None

    m.send = send
    m.receive = receive
    m.send_file = send_file
    m.receive_file = receive_file
    return m


# ---------------------------------------------------------------------------
# pie:inferlet/chat and pie:inferlet/reasoning
#
# Both are a `variant event` plus a `decoder` resource. componentize-py lowers
# a variant to one dataclass per case, named `Event_<Case>`, and the wrappers
# dispatch with `isinstance` -- so the stub has to keep that shape exactly.
# ---------------------------------------------------------------------------


@dataclass
class ChatDelta:
    value: str


@dataclass
class ChatInterrupt:
    value: int


@dataclass
class ChatDone:
    value: str


class _ScriptedDecoder:
    """Replays a class-level script, one event per `feed` call.

    Tests set `script` before constructing the SDK's wrapper, because the
    wrapper builds its inner decoder in `__init__`.
    """

    script: list[object] = []

    def __init__(self) -> None:
        self.fed: list[list[int]] = []
        self.resets = 0
        self._queue = list(type(self).script)

    def feed(self, tokens: list[int]) -> object:
        self.fed.append(list(tokens))
        # WIT declares `feed: func(...) -> result<event, error>`, so there is
        # no empty return: a run off the end of the script is a test bug, not
        # an idle event.
        if not self._queue:
            raise AssertionError("decoder stub: script exhausted")
        return self._queue.pop(0)

    def reset(self) -> None:
        self.resets += 1
        self._queue = list(type(self).script)


class ChatDecoderStub(_ScriptedDecoder):
    script: list[object] = []


def _make_chat() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.chat")
    m.Event_Delta = ChatDelta
    m.Event_Interrupt = ChatInterrupt
    m.Event_Done = ChatDone
    m.Decoder = ChatDecoderStub
    m.prefix = lambda: [0]
    m.system = lambda message: [1, *(ord(c) for c in message)]
    m.first_user = lambda message: [2, *(ord(c) for c in message)]
    m.system_user = lambda system, user: [4, *(ord(c) for c in system), *(ord(c) for c in user)]
    m.first_user = lambda message: [2, *(ord(c) for c in message)]
    m.user = lambda message: [3, *(ord(c) for c in message)]
    m.system_user = lambda system, user: [4]
    m.assistant = lambda message: [5, *(ord(c) for c in message)]
    m.cue = lambda: [6]
    m.seal = lambda: [7]
    m.stop_tokens = lambda: [8, 9]
    m.create_decoder = ChatDecoderStub
    return m


@dataclass
class ReasoningStart:
    pass


@dataclass
class ReasoningDelta:
    value: str


@dataclass
class ReasoningComplete:
    value: str


class ReasoningDecoderStub(_ScriptedDecoder):
    script: list[object] = []


def _make_reasoning() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.reasoning")
    m.Event_Start = ReasoningStart
    m.Event_Delta = ReasoningDelta
    m.Event_Complete = ReasoningComplete
    m.Decoder = ReasoningDecoderStub
    m.create_decoder = ReasoningDecoderStub
    return m




# ---------------------------------------------------------------------------
# The forward-pass surface: pie:inferlet/{types, channel, pipeline,
# working-set, forward, forward-recurrent, forward-hybrid} + the
# componentize-py `Err` type. Enough for `inferlet.eta.bridge` to import and
# for its host-side plumbing to be exercised without a device: a `Channel`
# remembers what was put; `take()` hands it back.
# ---------------------------------------------------------------------------


class Dtype(Enum):
    F32 = 0
    I32 = 1
    U32 = 2
    BOOL = 3


class WitErr(Exception):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.value = value


def _make_componentize_py_types() -> types.ModuleType:
    m = types.ModuleType("componentize_py_types")
    m.Err = WitErr
    m.Ok = type("Ok", (), {"__init__": lambda self, value: setattr(self, "value", value)})
    m.Result = object
    m.Some = type("Some", (), {"__init__": lambda self, value: setattr(self, "value", value)})
    return m


def _make_types() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.pie_inferlet_types")
    m.Dtype = Dtype
    return m


class ChannelStub:
    created: list["ChannelStub"] = []

    def __init__(self, shape, dtype, capacity) -> None:
        self.shape, self.dtype, self.capacity = list(shape), dtype, capacity
        self.cells: list[bytes] = []
        ChannelStub.created.append(self)

    def put(self, value: bytes) -> None:
        self.cells.append(bytes(value))

    def set(self, value: bytes) -> None:
        if not self.cells:
            raise WitErr("set on an empty channel")
        self.cells[0] = bytes(value)

    async def take(self) -> bytes:
        if not self.cells:
            raise WitErr("take on an empty channel (stub)")
        return self.cells.pop(0)

    async def read(self) -> bytes:
        if not self.cells:
            raise WitErr("read on an empty channel (stub)")
        return self.cells[0]


def _make_channel() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.channel")
    m.Channel = ChannelStub
    return m


class PipelineStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_pipeline() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.pipeline")
    m.Pipeline = PipelineStub
    return m


@dataclass
class PageRange:
    start: int
    len: int


@dataclass
class PageSpan:
    start: int
    end: int | None


class KvWorkingSetStub:
    def __init__(self) -> None:
        self.pages = 0

    def page_len(self) -> int:
        return self.pages

    def reserve(self, pages: int) -> PageRange:
        r = PageRange(self.pages, pages)
        self.pages += pages
        return r

    def fork(self, on):
        return KvWorkingSetStub()


class RsWorkingSetStub:
    def __init__(self) -> None:
        pass

    def buffer_size(self) -> int:
        return 0


def _make_working_set() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.working_set")
    m.PageRange = PageRange
    m.PageSpan = PageSpan
    m.KvWorkingSet = KvWorkingSetStub
    m.RsWorkingSet = RsWorkingSetStub
    return m


@dataclass
class KvGeometry:
    readable_pages: PageSpan
    writable_pages: PageSpan
    kv_len: object
    pages: object
    page_indptr: object
    w_slot: object
    w_off: object
    positions: object
    mask: object


@dataclass
class RsGeometry:
    fold_len: object
    buffer: PageSpan


@dataclass
class KvBinding:
    working_set: object
    geometry: KvGeometry


class ForwardPassStub:
    """Records what a pass was given; `program` keeps the container bytes."""

    submitted: list[tuple[object, list]] = []

    def __init__(self) -> None:
        self.embedded = None
        self.attention_args = None
        self.program_bytes: bytes | None = None
        self.program_channels: list | None = None

    def embed(self, tokens, indptr) -> None:
        self.embedded = (tokens, indptr)

    def readout(self, indices) -> None:
        self.readout_ch = indices

    def attention(self, *args) -> None:
        self.attention_args = args

    def set_max_layers(self, n: int) -> None:
        self.max_layers = n

    def set_drafting_block(self, on: bool) -> None:
        self.drafting = on

    def media(self, spans) -> None:
        self.spans = spans

    def canvas(self, mode) -> None:
        self.mode = mode

    def self_conditioning(self, rows, weights) -> None:
        self.self_cond = (list(rows), list(weights))

    def program(self, container_bytes: bytes, channels) -> None:
        self.program_bytes = bytes(container_bytes)
        self.program_channels = list(channels)


def _make_forward(name: str) -> types.ModuleType:
    m = types.ModuleType(f"wit_world.imports.{name}")
    m.ForwardPass = ForwardPassStub
    m.KvGeometry = KvGeometry
    m.RsGeometry = RsGeometry
    m.KvBinding = KvBinding
    m.MediaSpan_Image = lambda v: ("image", v)
    m.MediaSpan_Audio = lambda v: ("audio", v)
    m.Mode = DiffusionMode

    def submit(on, slots) -> None:
        ForwardPassStub.submitted.append((on, list(slots)))

    m.submit = submit
    m.park = lambda on: None
    return m


# ---------------------------------------------------------------------------
# pie:inferlet/grammar, tools, media — enough for the wrappers to import and
# marshal; a grammar "matches" everything and terminates after 3 tokens.
# ---------------------------------------------------------------------------


class GrammarStub:
    def __init__(self, source: str) -> None:
        self.source = source

    @classmethod
    def from_json_schema(cls, schema: str):
        if schema == "bad":
            raise WitErr("not a schema")
        return cls(schema)

    @classmethod
    def json(cls):
        return cls("json")

    @classmethod
    def from_regex(cls, pattern: str):
        return cls(pattern)

    @classmethod
    def from_ebnf(cls, ebnf: str):
        return cls(ebnf)

    def to_string(self) -> str:
        return self.source


class MatcherStub:
    def __init__(self, grammar) -> None:
        self.grammar = grammar
        self.accepted: list[int] = []

    def accept_tokens(self, ids) -> None:
        if 999 in ids:
            raise WitErr("token 999 is not allowed")
        self.accepted.extend(ids)

    def mask(self):
        return [0b101]

    def is_terminated(self) -> bool:
        return len(self.accepted) >= 3

    def reset(self) -> None:
        self.accepted.clear()

    def fork(self):
        m = MatcherStub(self.grammar)
        m.accepted = list(self.accepted)
        return m

    def rollback(self, n: int) -> None:
        del self.accepted[len(self.accepted) - n :]

    def rollback_capacity(self) -> int:
        return len(self.accepted)


def _make_grammar() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.grammar")
    m.Grammar = GrammarStub
    m.Matcher = MatcherStub
    return m


@dataclass
class ToolCallStub:
    name: str
    arguments_json: str


@dataclass
class ToolsEventStart:
    pass


@dataclass
class ToolsEventCall:
    value: ToolCallStub


class ToolsDecoderStub:
    def __init__(self) -> None:
        self.n = 0

    def feed(self, tokens):
        self.n += 1
        if self.n == 1:
            return ToolsEventStart()
        return ToolsEventCall(ToolCallStub("lookup", '{"q": 1}'))

    def reset(self) -> None:
        self.n = 0


def _make_tools() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.tools")
    m.Event_Start = ToolsEventStart
    m.Event_Call = ToolsEventCall
    m.ToolCall = ToolCallStub
    m.Decoder = ToolsDecoderStub
    m.equip = lambda tools: [10, len(tools)]
    m.answer = lambda name, value: [11, len(name), len(value)]
    m.format = lambda tools: GrammarStub("tools") if tools else None
    m.create_matcher = lambda tools: MatcherStub(GrammarStub("tools"))
    return m


@dataclass
class MergedGridStub:
    t: int
    h: int
    w: int


class ImageStub:
    @classmethod
    def from_bytes(cls, data: bytes):
        if not data:
            raise WitErr("empty image")
        return cls()

    def tokens(self):
        return [5, 6, 7]

    def digest(self):
        return b"\x01\x02"

    def token_count(self):
        return 3

    def position_span(self):
        return 1

    def grid(self):
        return MergedGridStub(1, 2, 3)

    def prefix_tokens(self):
        return [5]

    def suffix_tokens(self):
        return [7]


def _make_media() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.media")
    m.MergedGrid = MergedGridStub
    m.Image = ImageStub
    m.Audio = ImageStub
    m.Video = ImageStub
    return m


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install() -> None:
    """Register the stub `wit_world` package in `sys.modules`.

    Import-time, not fixture-time: `inferlet`'s modules bind their WIT imports
    at module scope, so the stubs must be in place before the first
    `import inferlet` anywhere in the session.
    """
    if "wit_world" in sys.modules:
        return

    wit_world = types.ModuleType("wit_world")
    imports = types.ModuleType("wit_world.imports")
    wit_world.imports = imports

    submodules = {
        "model": _make_model(),
        "tokenizer": _make_tokenizer(),
        "session": _make_session(SESSION_SPY),
        "chat": _make_chat(),
        "reasoning": _make_reasoning(),
        "pie_inferlet_types": _make_types(),
        "channel": _make_channel(),
        "pipeline": _make_pipeline(),
        "working_set": _make_working_set(),
        "forward": _make_forward("forward"),
        "forward_recurrent": _make_forward("forward_recurrent"),
        "forward_hybrid": _make_forward("forward_hybrid"),
        "forward_diffusion": _make_forward("forward_diffusion"),
        "grammar": _make_grammar(),
        "tools": _make_tools(),
        "media": _make_media(),
    }
    sys.modules.setdefault("componentize_py_types", _make_componentize_py_types())

    sys.modules["wit_world"] = wit_world
    sys.modules["wit_world.imports"] = imports
    for name, module in submodules.items():
        setattr(imports, name, module)
        sys.modules[f"wit_world.imports.{name}"] = module


install()
