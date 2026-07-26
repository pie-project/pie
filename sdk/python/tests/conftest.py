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


def _make_model() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.model")
    m.ForwardKind = ForwardKind
    m.name = lambda: "mock-model"
    m.architecture = lambda: "qwen3_5"
    m.default_system_speculation = lambda: False
    m.is_linear = lambda: True
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
    return m


# ---------------------------------------------------------------------------
# pie:inferlet/tokenizer
# ---------------------------------------------------------------------------


def _make_tokenizer() -> types.ModuleType:
    m = types.ModuleType("wit_world.imports.tokenizer")
    # Byte-identity codec: keeps assertions readable while still exercising
    # the list()/str marshalling the wrapper does.
    m.encode = lambda text: [ord(c) for c in text]
    m.decode = lambda tokens: "".join(chr(t) for t in tokens)
    m.vocabs = lambda: ([0, 1], [b"a", b"b"])
    m.split_regex = lambda: r"\w+"
    m.special_tokens = lambda: ([2], [b"<eos>"])
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
    m.system = lambda message: [1, *(ord(c) for c in message)]
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
    }

    sys.modules["wit_world"] = wit_world
    sys.modules["wit_world.imports"] = imports
    for name, module in submodules.items():
        setattr(imports, name, module)
        sys.modules[f"wit_world.imports.{name}"] = module


install()
