"""
Mock WIT bindings for unit testing the inferlet SDK outside the Pie runtime.

Installs mock ``wit_world.imports.*`` modules before any inferlet code imports.

Reflects the WASI-Preview-3 single-model + working-set bindings the SDK now
targets:

* ``model`` exposes GLOBAL functions over the one bound model (no
  ``model`` / ``tokenizer`` resource handles) plus the working-set / arena
  capability getters.
* the opaque ``context`` resource is gone — replaced by ``kv-working-set`` /
  ``rs-working-set`` plus explicit ``kv-context`` / ``kv-output`` forward-pass
  descriptors (``inference``).
* ``forward-pass.execute`` (and ``messaging.pull`` / ``session.receive*``) are
  component-model-async — ``async def`` returning values directly, with no
  pollable ``future-output``.
* ``media`` (image / video / audio) is the multimodal splice surface.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

# =============================================================================
# Fake resources & types
# =============================================================================


class FakePollable:
    def block(self) -> None:
        pass


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if 0 <= t < 128)

    def vocabs(self):
        return (list(range(256)), [bytes([i]) for i in range(256)])

    def split_regex(self) -> str:
        return r"."

    def special_tokens(self):
        return ([0, 1, 2], [b"<pad>", b"<bos>", b"<eos>"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# --- Working set (replaces the retired `context` resource) ---


@dataclass
class PageRange:
    start: int
    len: int


class FakeKvWorkingSet:
    """Dense ordered KV page array; structural mutators bump `generation`."""

    _PAGE_SIZE = 64

    def __init__(self):
        self._size = 0
        self._generation = 0

    def size(self) -> int:
        return self._size

    def generation(self) -> int:
        return self._generation

    def page_size(self) -> int:
        return self._PAGE_SIZE

    def alloc(self, n: int) -> PageRange:
        start = self._size
        self._size += n
        self._generation += 1
        return PageRange(start, n)

    def free(self, indices: list[int]) -> None:
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= self._size for i in indices
        ):
            raise ValueError("free: out-of-range or duplicate indices")
        self._size -= len(indices)
        self._generation += 1

    def reorder(self, perm: list[int]) -> None:
        if sorted(perm) != list(range(self._size)):
            raise ValueError("reorder: perm is not a bijection over 0..size")
        self._generation += 1

    def slice(self, start: int, length: int) -> "FakeKvWorkingSet":
        if start < 0 or start + length > self._size:
            raise ValueError("slice: out-of-range span")
        new = FakeKvWorkingSet()
        new._size = length
        return new

    def append(self, other: "FakeKvWorkingSet") -> None:
        self._size += other._size
        self._generation += 1

    def fork(self) -> "FakeKvWorkingSet":
        new = FakeKvWorkingSet()
        new._size = self._size
        new._generation = self._generation
        return new

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeRsWorkingSet:
    """Buffered recurrent-state working set (hybrid / linear-attention)."""

    _PAGE_SIZE = 64

    def __init__(self):
        self._buffer = 0
        self._generation = 0

    def state_size(self) -> int:
        return 0

    def buffer_size(self) -> int:
        return self._buffer

    def buffer_page_size(self) -> int:
        return self._PAGE_SIZE

    def alloc_buffer(self, n: int) -> PageRange:
        start = self._buffer
        self._buffer += n
        self._generation += 1
        return PageRange(start, n)

    def free_buffer(self, indices: list[int]) -> None:
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= self._buffer for i in indices
        ):
            raise ValueError("free-buffer: out-of-range or duplicate indices")
        self._buffer -= len(indices)
        self._generation += 1

    def reorder_buffer(self, perm: list[int]) -> None:
        if sorted(perm) != list(range(self._buffer)):
            raise ValueError("reorder-buffer: perm is not a bijection")
        self._generation += 1

    def fork(self) -> "FakeRsWorkingSet":
        new = FakeRsWorkingSet()
        new._buffer = self._buffer
        new._generation = self._generation
        return new

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# --- Multimodal media handles ---


class FakeImage:
    @classmethod
    def from_bytes(cls, data: bytes):
        return cls()

    def token_count(self) -> int:
        return 4

    def position_span(self) -> int:
        return 4

    def grid(self):
        return (1, 2, 2)

    def prefix_tokens(self) -> list[int]:
        return [100]

    def suffix_tokens(self) -> list[int]:
        return [101]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeVideo:
    @classmethod
    def from_bytes(cls, data: bytes, max_frames: int):
        return cls()

    def frame_count(self) -> int:
        return 1

    def frame(self, index: int) -> FakeImage:
        return FakeImage()

    def timestamp(self, index: int) -> float:
        return 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeAudio:
    @classmethod
    def from_bytes(cls, data: bytes):
        return cls()

    def token_count(self) -> int:
        return 3

    def position_span(self) -> int:
        return 3

    def prefix_tokens(self) -> list[int]:
        return []

    def suffix_tokens(self) -> list[int]:
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# --- Forward-pass memory descriptors (records) ---
@dataclass
class KvContext:
    set: object
    start: int
    len: int
    valid_tokens: int


@dataclass
class KvOutput:
    set: object
    generation: int
    indices: list
    per_page_valid_lens: list


@dataclass
class RsBufferContext:
    set: object
    start_token: int
    len_tokens: int


@dataclass
class RsBufferOutput:
    set: object
    start_token: int
    len_tokens: int


# --- Sampler variants ---
@dataclass
class Sampler_Multinomial:
    value: tuple
@dataclass
class Sampler_TopK:
    value: tuple
@dataclass
class Sampler_TopP:
    value: tuple
@dataclass
class Sampler_MinP:
    value: tuple
@dataclass
class Sampler_TopKTopP:
    value: tuple
@dataclass
class Sampler_Embedding:
    pass
@dataclass
class Sampler_Dist:
    value: tuple
@dataclass
class Sampler_RawLogits:
    pass
@dataclass
class Sampler_Logprob:
    value: int
@dataclass
class Sampler_Logprobs:
    value: list
@dataclass
class Sampler_Entropy:
    pass

# --- Slot output variants ---
@dataclass
class SlotOutput_Token:
    value: int
@dataclass
class SlotOutput_Distribution:
    value: tuple
@dataclass
class SlotOutput_Logits:
    value: bytes
@dataclass
class SlotOutput_Logprobs:
    value: list
@dataclass
class SlotOutput_Entropy:
    value: float
@dataclass
class SlotOutput_Embedding:
    value: bytes

# --- Output (record) ---
@dataclass
class Output:
    slots: list
    spec_tokens: list
    spec_positions: list


class FakeForwardPass:
    """P3 forward pass: explicit kv/rs descriptors + async `execute() -> output`."""

    _next_output = None

    def __init__(self):
        pass

    def kv_context(self, ctx): pass
    def kv_output(self, out): pass
    def rs_context(self, ctx): pass
    def rs_output(self, out): pass
    def fold_buffered(self, tokens): pass
    def input_tokens(self, tokens, positions): pass
    def input_image(self, image, anchor): pass
    def input_audio(self, audio, anchor): pass
    def input_speculative_tokens(self, tokens, positions): pass
    def output_speculative_tokens(self, flag): pass
    def pass_speculation(self, flag): pass
    def attention_mask(self, mask): pass
    def logit_mask(self, mask): pass
    def sampler(self, indices, sampler): pass
    def adapter(self, adapter): pass

    async def execute(self):
        if FakeForwardPass._next_output is not None:
            return FakeForwardPass._next_output
        # Default: a single EOS token (id 2, see special_tokens above).
        return Output(slots=[SlotOutput_Token(2)], spec_tokens=[], spec_positions=[])

    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class FakeGrammar:
    @classmethod
    def from_json_schema(cls, schema): return cls()
    @classmethod
    def json(cls): return cls()
    @classmethod
    def from_regex(cls, pattern): return cls()
    @classmethod
    def from_ebnf(cls, ebnf): return cls()
    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeMatcher:
    def __init__(self, grammar):
        self._terminated = False
    def accept_tokens(self, token_ids): pass
    def next_token_logit_mask(self): return [1] * 256
    def is_terminated(self): return self._terminated
    def reset(self): self._terminated = False
    def __enter__(self): return self
    def __exit__(self, *args): pass


# --- Chat decoder / events ---
@dataclass
class ChatEvent_Delta:
    value: str
@dataclass
class ChatEvent_Interrupt:
    value: int
@dataclass
class ChatEvent_Done:
    value: str

class FakeChatDecoder:
    def __init__(self):
        self._call_count = 0
    def feed(self, tokens):
        self._call_count += 1
        return ChatEvent_Delta("hello ")
    def reset(self): self._call_count = 0
    def __enter__(self): return self
    def __exit__(self, *args): pass

# --- Reasoning decoder / events ---
@dataclass
class ReasoningEvent_Start:
    pass
@dataclass
class ReasoningEvent_Delta:
    value: str
@dataclass
class ReasoningEvent_Complete:
    value: str

class FakeReasoningDecoder:
    def feed(self, tokens): return ReasoningEvent_Delta("thinking...")
    def reset(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

# --- Tool use decoder / events ---
@dataclass
class ToolEvent_Start:
    pass
@dataclass
class ToolEvent_Call:
    value: tuple

class FakeToolDecoder:
    def feed(self, tokens): return ToolEvent_Start()
    def reset(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


# --- Messaging ---
class FakeStreamReader:
    """Stand-in for a P3 `stream<string>` subscription handle: `async read`
    returns up to `max_count` items, then `[]` (with `writer_dropped`) once
    the writable end closes; `__exit__` drops the readable end."""

    def __init__(self, messages=None):
        self._messages = list(messages) if messages is not None else ["msg1", "msg2"]
        self.writer_dropped = False
        self.dropped = False

    async def read(self, max_count):
        if self._messages:
            out = self._messages[:max_count]
            del self._messages[:max_count]
            return out
        self.writer_dropped = True
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.dropped = True
        return None


# --- Adapter ---
class FakeAdapter:
    @classmethod
    def create(cls, name): return cls()
    @classmethod
    def open(cls, name): return None
    def fork(self, new_name): return FakeAdapter()
    def load(self, path): pass
    def save(self, path): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


# =============================================================================
# Module installation
# =============================================================================


def _build_mock_modules():
    """Build and install mock wit_world modules into sys.modules."""

    # componentize_py_types
    cpy_types = types.ModuleType("componentize_py_types")
    cpy_types.Result = type("Result", (), {})
    cpy_types.Ok = type("Ok", (), {})
    cpy_types.Err = type("Err", (), {})
    cpy_types.Some = type("Some", (), {})

    # wit_world
    wit_world = types.ModuleType("wit_world")
    wit_imports = types.ModuleType("wit_world.imports")
    wit_world.imports = wit_imports

    # poll
    poll_mod = types.ModuleType("wit_world.imports.poll")
    poll_mod.Pollable = FakePollable

    # model — the engine serves one bound model; global functions only,
    # including the working-set / arena capability getters.
    _tok = FakeTokenizer()
    model_mod = types.ModuleType("wit_world.imports.model")
    model_mod.name = lambda: "mock-model"
    model_mod.architecture = lambda: "mock-arch"
    model_mod.default_system_speculation = lambda: False
    model_mod.encode = lambda text: _tok.encode(text)
    model_mod.decode = lambda tokens: _tok.decode(tokens)
    model_mod.vocabs = lambda: _tok.vocabs()
    model_mod.split_regex = lambda: _tok.split_regex()
    model_mod.special_tokens = lambda: _tok.special_tokens()
    model_mod.rs_state_size = lambda: 0
    model_mod.rs_buffer_page_size = lambda: 0
    model_mod.rs_fold_granularity = lambda: 1
    model_mod.arena_block_size = lambda: 64

    # working_set — replaces the retired `context` resource.
    ws_mod = types.ModuleType("wit_world.imports.working_set")
    ws_mod.KvWorkingSet = FakeKvWorkingSet
    ws_mod.RsWorkingSet = FakeRsWorkingSet
    ws_mod.PageRange = PageRange

    # media — multimodal splice surface.
    media_mod = types.ModuleType("wit_world.imports.media")
    media_mod.Image = FakeImage
    media_mod.Video = FakeVideo
    media_mod.Audio = FakeAudio

    # inference
    inf_mod = types.ModuleType("wit_world.imports.inference")
    for cls in [
        KvContext, KvOutput, RsBufferContext, RsBufferOutput,
        Sampler_Multinomial, Sampler_TopK, Sampler_TopP, Sampler_MinP,
        Sampler_TopKTopP, Sampler_Embedding, Sampler_Dist, Sampler_RawLogits,
        Sampler_Logprob, Sampler_Logprobs, Sampler_Entropy,
        SlotOutput_Token, SlotOutput_Distribution, SlotOutput_Logits,
        SlotOutput_Logprobs, SlotOutput_Entropy, SlotOutput_Embedding,
        Output,
    ]:
        setattr(inf_mod, cls.__name__, cls)
    inf_mod.Sampler = (
        Sampler_Multinomial | Sampler_TopK | Sampler_TopP | Sampler_MinP
        | Sampler_TopKTopP | Sampler_Embedding | Sampler_Dist
        | Sampler_RawLogits | Sampler_Logprob | Sampler_Logprobs
        | Sampler_Entropy
    )
    inf_mod.SlotOutput = (
        SlotOutput_Token | SlotOutput_Distribution | SlotOutput_Logits
        | SlotOutput_Logprobs | SlotOutput_Entropy | SlotOutput_Embedding
    )
    inf_mod.ForwardPass = FakeForwardPass
    inf_mod.Grammar = FakeGrammar
    inf_mod.Matcher = FakeMatcher

    # chat
    chat_mod = types.ModuleType("wit_world.imports.chat")
    chat_mod.Event_Delta = ChatEvent_Delta
    chat_mod.Event_Interrupt = ChatEvent_Interrupt
    chat_mod.Event_Done = ChatEvent_Done
    chat_mod.Decoder = FakeChatDecoder
    chat_mod.system = lambda msg: []
    chat_mod.first_user = lambda msg: [66]
    chat_mod.user = lambda msg: []
    chat_mod.system_user = lambda system, user: [67]
    chat_mod.assistant = lambda msg: []
    chat_mod.cue = lambda: [65]
    chat_mod.seal = lambda: []
    chat_mod.stop_tokens = lambda: [2]
    chat_mod.create_decoder = lambda: FakeChatDecoder()

    # reasoning
    reasoning_mod = types.ModuleType("wit_world.imports.reasoning")
    reasoning_mod.Event_Start = ReasoningEvent_Start
    reasoning_mod.Event_Delta = ReasoningEvent_Delta
    reasoning_mod.Event_Complete = ReasoningEvent_Complete
    reasoning_mod.Decoder = FakeReasoningDecoder
    reasoning_mod.create_decoder = lambda: FakeReasoningDecoder()

    # tool_use
    tool_mod = types.ModuleType("wit_world.imports.tool_use")
    tool_mod.Event_Start = ToolEvent_Start
    tool_mod.Event_Call = ToolEvent_Call
    tool_mod.Decoder = FakeToolDecoder
    tool_mod.equip = lambda tools: []
    tool_mod.answer = lambda name, value: []
    tool_mod.create_decoder = lambda: FakeToolDecoder()
    tool_mod.format = lambda tools: None
    tool_mod.create_matcher = lambda tools: FakeMatcher(None)

    # runtime
    runtime_mod = types.ModuleType("wit_world.imports.runtime")
    runtime_mod.version = lambda: "0.1.0-mock"
    runtime_mod.instance_id = lambda: "mock-instance-001"
    runtime_mod.username = lambda: "test-user"

    async def _sleep(duration_ns):
        return None
    runtime_mod.sleep = _sleep

    # messaging — `pull` is async; `subscribe` yields a stream reader.
    messaging_mod = types.ModuleType("wit_world.imports.messaging")
    messaging_mod.push = lambda topic, msg: None

    async def _pull(topic):
        return "pulled"
    messaging_mod.pull = _pull
    messaging_mod.broadcast = lambda topic, msg: None
    messaging_mod.subscribe = lambda topic: FakeStreamReader()

    # session — `receive` / `receive_file` are async.
    session_mod = types.ModuleType("wit_world.imports.session")
    session_mod.send = lambda msg: None

    async def _receive():
        return "received"
    session_mod.receive = _receive
    session_mod.send_file = lambda data: None

    async def _receive_file():
        return b"file-data"
    session_mod.receive_file = _receive_file

    # adapter
    adapter_mod = types.ModuleType("wit_world.imports.adapter")
    adapter_mod.Adapter = FakeAdapter

    # zo
    zo_mod = types.ModuleType("wit_world.imports.zo")
    zo_mod.adapter_seed = lambda fp, seed: None
    zo_mod.initialize = lambda adapter, rank, alpha, pop, mu, sigma: None
    zo_mod.update = lambda adapter, scores, seeds, max_sigma: None

    # Install all
    modules = {
        "componentize_py_types": cpy_types,
        "wit_world": wit_world,
        "wit_world.imports": wit_imports,
        "wit_world.imports.poll": poll_mod,
        "wit_world.imports.model": model_mod,
        "wit_world.imports.working_set": ws_mod,
        "wit_world.imports.media": media_mod,
        "wit_world.imports.inference": inf_mod,
        "wit_world.imports.chat": chat_mod,
        "wit_world.imports.reasoning": reasoning_mod,
        "wit_world.imports.tool_use": tool_mod,
        "wit_world.imports.runtime": runtime_mod,
        "wit_world.imports.messaging": messaging_mod,
        "wit_world.imports.session": session_mod,
        "wit_world.imports.adapter": adapter_mod,
        "wit_world.imports.zo": zo_mod,
    }

    for name, mod in modules.items():
        sys.modules[name] = mod

    for attr in [
        "poll", "model", "working_set", "media",
        "inference", "chat", "reasoning", "tool_use", "runtime",
        "messaging", "session", "adapter", "zo",
    ]:
        setattr(wit_imports, attr, sys.modules[f"wit_world.imports.{attr}"])


@pytest.fixture(autouse=True)
def mock_wit():
    """Install mock WIT bindings before each test."""
    _build_mock_modules()
    yield
    to_remove = [k for k in sys.modules if k.startswith("inferlet")]
    for k in to_remove:
        del sys.modules[k]


# Install at import time
_build_mock_modules()
