"""
Mock WIT bindings for unit testing the inferlet SDK outside the Pie runtime.

Installs mock ``wit_world.imports.*`` modules before any inferlet code imports.
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


class FakeFutureBool:
    def __init__(self, value: bool = True):
        self._value = value
    def pollable(self):
        return FakePollable()
    def get(self):
        return self._value


class FakeFutureString:
    def __init__(self, value: str = ""):
        self._value = value
    def pollable(self):
        return FakePollable()
    def get(self):
        return self._value


class FakeFutureBlob:
    def __init__(self, value: bytes = b""):
        self._value = value
    def pollable(self):
        return FakePollable()
    def get(self):
        return self._value


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


class FakeModel:
    """Deprecated stand-in kept only so old references don't explode; the
    real binding now exposes global functions (see ``model_mod`` below)."""

    def __init__(self):
        self._tokenizer = FakeTokenizer()


class FakeContext:
    def __init__(self):
        self._cursor: int = 0
        self._committed_pages: int = 0
        self._tokens_per_page_val: int = 64

    @classmethod
    def create(cls):
        return cls()

    @classmethod
    def open(cls, name):
        return cls()

    @classmethod
    def take(cls, name):
        return cls()

    @classmethod
    def delete(cls, name):
        return None

    def destroy(self):
        pass

    def save(self, name):
        pass

    def snapshot(self):
        return "snap"

    def suspend(self):
        pass

    def fork(self):
        new = FakeContext()
        new._cursor = self._cursor
        new._committed_pages = self._committed_pages
        return new

    def acquire_lock(self):
        return FakeFutureBool(True)

    def release_lock(self):
        pass

    def tokens_per_page(self):
        return self._tokens_per_page_val

    def committed_page_count(self):
        return self._committed_pages

    def working_page_count(self):
        return 0

    def commit_working_pages(self, num_pages):
        self._committed_pages += num_pages

    def reserve_working_pages(self, n):
        pass

    def release_working_pages(self, n):
        pass

    def working_page_token_count(self):
        return self._cursor

    def truncate_working_page_tokens(self, num_tokens):
        self._cursor = max(0, self._cursor - num_tokens)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


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


class FakeFutureOutput:
    def __init__(self, output=None):
        self._output = output or Output(slots=[SlotOutput_Token(42)], spec_tokens=[], spec_positions=[])
    def pollable(self):
        return FakePollable()
    def get(self):
        return self._output
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class FakeForwardPass:
    _next_output = None

    def __init__(self):
        pass

    def context(self, ctx): pass
    def input_tokens(self, tokens, positions): pass
    def input_speculative_tokens(self, tokens, positions): pass
    def output_speculative_tokens(self, flag): pass
    def attention_mask(self, mask): pass
    def logit_mask(self, mask): pass
    def sampler(self, indices, sampler): pass
    def adapter(self, adapter): pass

    def execute(self):
        if FakeForwardPass._next_output is not None:
            return FakeFutureOutput(FakeForwardPass._next_output)
        return FakeFutureOutput(Output(slots=[SlotOutput_Token(2)], spec_tokens=[], spec_positions=[]))  # EOS

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
class FakeSubscription:
    def __init__(self):
        self._messages = ["msg1", "msg2"]
        self._idx = 0
    def pollable(self): return FakePollable()
    def get(self):
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        return None
    def unsubscribe(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


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

    # pie_core_types
    core_types = types.ModuleType("wit_world.imports.pie_core_types")
    core_types.FutureBool = FakeFutureBool
    core_types.FutureString = FakeFutureString
    core_types.FutureBlob = FakeFutureBlob

    # model — the engine serves one bound model; global functions only.
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

    # context
    context_mod = types.ModuleType("wit_world.imports.context")
    context_mod.Context = FakeContext

    # inference
    inf_mod = types.ModuleType("wit_world.imports.inference")
    for cls in [
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
    inf_mod.FutureOutput = FakeFutureOutput
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
    chat_mod.user = lambda msg: []
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

    # messaging
    messaging_mod = types.ModuleType("wit_world.imports.messaging")
    messaging_mod.push = lambda topic, msg: None
    messaging_mod.pull = lambda topic: FakeFutureString("pulled")
    messaging_mod.broadcast = lambda topic, msg: None
    messaging_mod.subscribe = lambda topic: FakeSubscription()
    messaging_mod.Subscription = FakeSubscription

    # session
    session_mod = types.ModuleType("wit_world.imports.session")
    session_mod.send = lambda msg: None
    session_mod.receive = lambda: FakeFutureString("received")
    session_mod.send_file = lambda data: None
    session_mod.receive_file = lambda: FakeFutureBlob(b"file-data")

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
        "wit_world.imports.pie_core_types": core_types,
        "wit_world.imports.model": model_mod,
        "wit_world.imports.context": context_mod,
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
        "poll", "pie_core_types", "model", "context",
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
