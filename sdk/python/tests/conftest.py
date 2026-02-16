"""
Mock WIT bindings for unit testing the inferlet SDK outside the Pie runtime.

Installs mock ``wit_world.imports.*`` modules before any inferlet code imports.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Optional

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
    def __init__(self):
        self._tokenizer = FakeTokenizer()

    @classmethod
    def load(cls, name: str):
        return cls()

    def tokenizer(self):
        return self._tokenizer

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeContext:
    def __init__(self):
        self._buffered_tokens: list[int] = []
        self._cursor: int = 0
        self._last_position: Optional[int] = None
        self._committed_pages: int = 0
        self._tokens_per_page_val: int = 64

    @classmethod
    def create(cls, model, name, fill=None):
        ctx = cls()
        if fill:
            ctx._buffered_tokens = list(fill)
        return ctx

    def destroy(self):
        pass

    @classmethod
    def lookup(cls, model, name):
        return None

    def fork(self, new_name):
        new = FakeContext()
        new._cursor = self._cursor
        new._last_position = self._last_position
        new._committed_pages = self._committed_pages
        new._buffered_tokens = list(self._buffered_tokens)
        return new

    def acquire_lock(self):
        return FakeFutureBool(True)

    def release_lock(self):
        pass

    def tokens_per_page(self):
        return self._tokens_per_page_val

    def model(self):
        return FakeModel()

    def committed_page_count(self):
        return self._committed_pages

    def uncommitted_page_count(self):
        return 0

    def commit_pages(self, page_indices):
        self._committed_pages += len(page_indices)

    def reserve_pages(self, n):
        pass

    def release_pages(self, n):
        pass

    def cursor(self):
        return self._cursor

    def set_cursor(self, cursor):
        self._cursor = cursor

    def buffered_tokens(self):
        return list(self._buffered_tokens)

    def set_buffered_tokens(self, tokens):
        self._buffered_tokens = list(tokens)

    def append_buffered_tokens(self, tokens):
        self._buffered_tokens.extend(tokens)

    def last_position(self):
        return self._last_position

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

# --- Output variants ---
@dataclass
class Output_None_:
    pass
@dataclass
class Output_Tokens:
    value: list
@dataclass
class Output_TokensWithSpeculation:
    value: tuple
@dataclass
class Output_Embeddings:
    value: list
@dataclass
class Output_Distributions:
    value: list


class FakeFutureOutput:
    def __init__(self, output=None):
        self._output = output or Output_Tokens([42])
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

    def __init__(self, model):
        self._model = model

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
        return FakeFutureOutput(Output_Tokens([2]))  # EOS

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
    def __init__(self, grammar, tokenizer):
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


# --- MCP ---
@dataclass
class ContentText:
    value: str
@dataclass
class ContentImage:
    value: object
@dataclass
class ContentEmbeddedResource:
    value: object

class FakeMcpSession:
    def list_tools(self): return '{"tools": []}'
    def call_tool(self, name, args): return []
    def list_resources(self): return '{"resources": []}'
    def read_resource(self, uri): return []
    def list_prompts(self): return '{"prompts": []}'
    def get_prompt(self, name, args): return "{}"
    def __enter__(self): return self
    def __exit__(self, *args): pass

@dataclass
class McpError:
    code: int
    message: str
    data: object

# --- Adapter ---
class FakeAdapter:
    @classmethod
    def create(cls, model, name): return cls()
    @classmethod
    def lookup(cls, model, name): return None
    def clone(self, new_name): return FakeAdapter()
    def acquire_lock(self): return FakeFutureBool(True)
    def release_lock(self): pass
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

    # pie_mcp_types
    mcp_types = types.ModuleType("wit_world.imports.pie_mcp_types")
    mcp_types.Content_Text = ContentText
    mcp_types.Content_Image = ContentImage
    mcp_types.Content_EmbeddedResource = ContentEmbeddedResource
    mcp_types.Content = ContentText | ContentImage | ContentEmbeddedResource
    mcp_types.Error = McpError

    # model
    model_mod = types.ModuleType("wit_world.imports.model")
    model_mod.Tokenizer = FakeTokenizer
    model_mod.Model = FakeModel

    # context
    context_mod = types.ModuleType("wit_world.imports.context")
    context_mod.Context = FakeContext

    # inference
    inf_mod = types.ModuleType("wit_world.imports.inference")
    for cls in [
        Sampler_Multinomial, Sampler_TopK, Sampler_TopP, Sampler_MinP,
        Sampler_TopKTopP, Sampler_Embedding, Sampler_Dist,
        Output_None_, Output_Tokens, Output_TokensWithSpeculation,
        Output_Embeddings, Output_Distributions,
    ]:
        setattr(inf_mod, cls.__name__, cls)
    inf_mod.Sampler = (
        Sampler_Multinomial | Sampler_TopK | Sampler_TopP | Sampler_MinP
        | Sampler_TopKTopP | Sampler_Embedding | Sampler_Dist
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
    chat_mod.system = lambda ctx, msg: ctx.append_buffered_tokens([])
    chat_mod.user = lambda ctx, msg: ctx.append_buffered_tokens([])
    chat_mod.assistant = lambda ctx, msg: ctx.append_buffered_tokens([])
    chat_mod.cue = lambda ctx: ctx.append_buffered_tokens([65])
    chat_mod.seal = lambda ctx: None
    chat_mod.stop_tokens = lambda model: [2]
    chat_mod.create_decoder = lambda model: FakeChatDecoder()

    # reasoning
    reasoning_mod = types.ModuleType("wit_world.imports.reasoning")
    reasoning_mod.Event_Start = ReasoningEvent_Start
    reasoning_mod.Event_Delta = ReasoningEvent_Delta
    reasoning_mod.Event_Complete = ReasoningEvent_Complete
    reasoning_mod.Decoder = FakeReasoningDecoder
    reasoning_mod.create_decoder = lambda model: FakeReasoningDecoder()

    # tool_use
    tool_mod = types.ModuleType("wit_world.imports.tool_use")
    tool_mod.Event_Start = ToolEvent_Start
    tool_mod.Event_Call = ToolEvent_Call
    tool_mod.Decoder = FakeToolDecoder
    tool_mod.equip = lambda ctx, tools: None
    tool_mod.answer = lambda ctx, name, value: None
    tool_mod.create_decoder = lambda model: FakeToolDecoder()
    tool_mod.create_matcher = lambda model, tools: FakeMatcher(None, None)

    # runtime
    runtime_mod = types.ModuleType("wit_world.imports.runtime")
    runtime_mod.version = lambda: "0.1.0-mock"
    runtime_mod.instance_id = lambda: "mock-instance-001"
    runtime_mod.username = lambda: "test-user"
    runtime_mod.models = lambda: ["mock-model"]
    runtime_mod.spawn = lambda pkg, args: FakeFutureString("spawned")

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

    # client (MCP)
    client_mod = types.ModuleType("wit_world.imports.client")
    client_mod.available_servers = lambda: ["mock-server"]
    client_mod.connect = lambda name: FakeMcpSession()
    client_mod.Session = FakeMcpSession

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
        "wit_world.imports.pie_mcp_types": mcp_types,
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
        "wit_world.imports.client": client_mod,
        "wit_world.imports.zo": zo_mod,
    }

    for name, mod in modules.items():
        sys.modules[name] = mod

    for attr in [
        "poll", "pie_core_types", "pie_mcp_types", "model", "context",
        "inference", "chat", "reasoning", "tool_use", "runtime",
        "messaging", "session", "adapter", "client", "zo",
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
