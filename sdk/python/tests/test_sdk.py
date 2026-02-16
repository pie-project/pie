"""Unit tests for the inferlet SDK."""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestModel:
    def test_load(self):
        from inferlet.model import Model
        m = Model.load("test-model")
        assert m._handle is not None

    def test_tokenizer_encode_decode(self):
        from inferlet.model import Model
        m = Model.load("test-model")
        tok = m.tokenizer()
        assert tok.encode("Hi") == [72, 105]
        assert tok.decode([72, 105]) == "Hi"


class TestSampler:
    def test_greedy(self):
        from inferlet.sampler import Sampler
        s = Sampler.greedy()
        assert s._variant.value[0] == 0.0

    def test_top_p(self):
        from inferlet.sampler import Sampler
        s = Sampler.top_p(temperature=0.7, top_p=0.9)
        assert s._variant.value == (0.7, 0.9)

    def test_top_k(self):
        from inferlet.sampler import Sampler
        s = Sampler.top_k(temperature=0.6, top_k=40)
        assert s._variant.value == (0.6, 40)

    def test_min_p(self):
        from inferlet.sampler import Sampler
        s = Sampler.min_p(temperature=0.5, min_p=0.1)
        assert s._variant.value == (0.5, 0.1)

    def test_top_k_top_p(self):
        from inferlet.sampler import Sampler
        s = Sampler.top_k_top_p(temperature=0.6, top_k=50, top_p=0.95)
        assert s._variant.value == (0.6, 50, 0.95)

    def test_embedding(self):
        from inferlet.sampler import Sampler
        from wit_world.imports.inference import Sampler_Embedding
        s = Sampler.embedding()
        assert isinstance(s._variant, Sampler_Embedding)


class TestContext:
    def test_create(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        assert ctx._handle is not None

    def test_fill_tokens(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([1, 2, 3])
        assert ctx._handle.buffered_tokens() == [1, 2, 3]

    def test_system_user_cue_no_return(self):
        """Mutators return None, not self."""
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        assert ctx.system("Hello") is None
        assert ctx.user("World") is None
        assert ctx.cue() is None

    def test_fork(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([1, 2, 3])
        forked = ctx.fork("forked-ctx")
        assert forked._handle is not None
        assert forked._handle.buffered_tokens() == [1, 2, 3]

    def test_context_manager(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        with Context(m) as ctx:
            ctx.fill_tokens([42])
            assert ctx._handle.buffered_tokens() == [42]

    def test_equip_tools_no_return(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        assert ctx.equip_tools(["tool1"]) is None

    def test_answer_tool_no_return(self):
        from inferlet.model import Model
        from inferlet.context import Context
        m = Model.load("test-model")
        ctx = Context(m)
        assert ctx.answer_tool("search", '{}') is None


class TestGenerate:
    def test_token_stream_default(self):
        from inferlet.model import Model
        from inferlet.context import Context, TokenStream
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        stream = ctx.generate(Sampler.greedy())
        assert isinstance(stream, TokenStream)

    def test_event_stream_with_decode(self):
        from inferlet.model import Model
        from inferlet.context import Context, EventStream
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        stream = ctx.generate(Sampler.greedy(), decode=True)
        assert isinstance(stream, EventStream)

    def test_event_stream_with_reasoning(self):
        from inferlet.model import Model
        from inferlet.context import Context, EventStream
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        # reasoning=True implies decode
        stream = ctx.generate(Sampler.greedy(), reasoning=True)
        assert isinstance(stream, EventStream)

    def test_max_tokens_zero(self):
        from inferlet.model import Model
        from inferlet.context import Context
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        tokens = list(ctx.generate(Sampler.greedy(), max_tokens=0))
        assert tokens == []

    def test_collect_tokens(self):
        from inferlet.model import Model
        from inferlet.context import Context
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        tokens = ctx.generate(Sampler.greedy()).collect_tokens()
        assert isinstance(tokens, list)

    def test_collect_text(self):
        from inferlet.model import Model
        from inferlet.context import Context
        from inferlet.sampler import Sampler
        m = Model.load("test-model")
        ctx = Context(m)
        ctx.fill_tokens([65])
        text = ctx.generate(Sampler.greedy()).collect_text()
        assert isinstance(text, str)


class TestDecoder:
    def test_create(self):
        from inferlet.model import Model
        from inferlet.context import Decoder
        m = Model.load("test-model")
        d = Decoder(m)
        assert d._chat_dec is not None

    def test_with_reasoning(self):
        from inferlet.model import Model
        from inferlet.context import Decoder
        m = Model.load("test-model")
        d = Decoder(m, reasoning=True)
        assert d._reasoning_dec is not None

    def test_feed_returns_event(self):
        from inferlet.model import Model
        from inferlet.context import Decoder, Event
        m = Model.load("test-model")
        d = Decoder(m)
        event = d.feed([65, 66])
        assert isinstance(event, Event.Text)


class TestEvent:
    def test_text(self):
        from inferlet.context import Event
        e = Event.Text("hello")
        assert e.text == "hello"

    def test_thinking(self):
        from inferlet.context import Event
        e = Event.Thinking("hmm")
        assert e.text == "hmm"

    def test_done(self):
        from inferlet.context import Event
        e = Event.Done("final")
        assert e.text == "final"

    def test_tool_call(self):
        from inferlet.context import Event
        e = Event.ToolCall(name="search", arguments='{"q": "test"}')
        assert e.name == "search"


class TestRuntime:
    def test_version(self):
        from inferlet import runtime
        assert runtime.version() == "0.1.0-mock"

    def test_instance_id(self):
        from inferlet import runtime
        assert runtime.instance_id() == "mock-instance-001"

    def test_username(self):
        from inferlet import runtime
        assert runtime.username() == "test-user"

    def test_models(self):
        from inferlet import runtime
        assert runtime.models() == ["mock-model"]

    def test_spawn(self):
        from inferlet import runtime
        assert runtime.spawn("test:pkg@1.0.0") == "spawned"


class TestMessaging:
    def test_push(self):
        from inferlet import messaging
        messaging.push("topic", "msg")

    def test_pull(self):
        from inferlet import messaging
        assert messaging.pull("topic") == "pulled"

    def test_broadcast(self):
        from inferlet import messaging
        messaging.broadcast("topic", "msg")

    def test_subscribe_next(self):
        from inferlet import messaging
        sub = messaging.subscribe("events")
        assert sub.next() == "msg1"
        assert sub.next() == "msg2"

    def test_subscribe_iter(self):
        from inferlet import messaging
        msgs = list(messaging.subscribe("events"))
        assert msgs == ["msg1", "msg2"]

    def test_subscribe_context_manager(self):
        from inferlet import messaging
        with messaging.subscribe("events") as sub:
            assert sub.next() == "msg1"


class TestSession:
    def test_send(self):
        from inferlet import session
        session.send("hello")

    def test_receive(self):
        from inferlet import session
        assert session.receive() == "received"

    def test_send_file(self):
        from inferlet import session
        session.send_file(b"data")

    def test_receive_file(self):
        from inferlet import session
        assert session.receive_file() == b"file-data"


class TestGrammar:
    def test_from_json_schema(self):
        from inferlet.grammar import Grammar
        g = Grammar.from_json_schema('{"type": "object"}')
        assert g._handle is not None

    def test_json(self):
        from inferlet.grammar import Grammar
        assert Grammar.json()._handle is not None

    def test_from_regex(self):
        from inferlet.grammar import Grammar
        assert Grammar.from_regex(r"\d+")._handle is not None

    def test_from_ebnf(self):
        from inferlet.grammar import Grammar
        assert Grammar.from_ebnf('root ::= "hello"')._handle is not None


class TestMatcher:
    def test_create(self):
        from inferlet.grammar import Grammar, Matcher
        from inferlet.model import Model
        m = Model.load("test-model")
        matcher = Matcher(Grammar.json(), m.tokenizer())
        assert not matcher.is_terminated

    def test_logit_mask(self):
        from inferlet.grammar import Grammar, Matcher
        from inferlet.model import Model
        m = Model.load("test-model")
        mask = Matcher(Grammar.json(), m.tokenizer()).next_token_logit_mask()
        assert len(mask) == 256


class TestForwardPass:
    def test_create(self):
        from inferlet.model import Model
        from inferlet.forward import ForwardPass
        fp = ForwardPass(Model.load("test-model"))
        assert fp._handle is not None


class TestMcp:
    def test_available_servers(self):
        from inferlet import mcp
        assert mcp.available_servers() == ["mock-server"]

    def test_connect(self):
        from inferlet import mcp
        assert mcp.connect("mock-server") is not None

    def test_list_tools(self):
        from inferlet import mcp
        assert "tools" in mcp.connect("mock-server").list_tools()

    def test_call_tool(self):
        from inferlet import mcp
        result = mcp.connect("mock-server").call_tool("search", '{"q": "test"}')
        assert isinstance(result, list)


class TestAdapter:
    def test_create(self):
        from inferlet.model import Model
        from inferlet.adapter import Adapter
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        assert a._handle is not None

    def test_load_save_paths(self):
        """Adapter load/save take file paths, not bytes."""
        from inferlet.model import Model
        from inferlet.adapter import Adapter
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        a.load("/path/to/weights")
        a.save("/path/to/output")


class TestZo:
    def test_adapter_seed(self):
        from inferlet import zo
        from inferlet.model import Model
        from inferlet.forward import ForwardPass
        fp = ForwardPass(Model.load("test-model"))
        zo.adapter_seed(fp, 42)

    def test_initialize(self):
        from inferlet import zo
        from inferlet.model import Model
        from inferlet.adapter import Adapter
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        zo.initialize(a, rank=8, alpha=16.0, population_size=64, mu_fraction=0.25, initial_sigma=0.1)

    def test_update(self):
        from inferlet import zo
        from inferlet.model import Model
        from inferlet.adapter import Adapter
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        zo.update(a, scores=[1.0, 2.0], seeds=[1, 2], max_sigma=0.5)
