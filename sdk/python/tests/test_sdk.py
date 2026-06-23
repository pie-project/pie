"""Unit tests for the redesigned inferlet SDK."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# =============================================================================
# Model
# =============================================================================


class TestModel:
    def test_load(self):
        from inferlet import Model
        m = Model.load("test-model")
        assert m._handle is not None

    def test_tokenizer_encode_decode(self):
        from inferlet import Model
        tok = Model.load("test-model").tokenizer()
        assert tok.encode("Hi") == [72, 105]
        assert tok.decode([72, 105]) == "Hi"


# =============================================================================
# Sampler — token-producing
# =============================================================================


class TestSampler:
    def test_argmax(self):
        from inferlet import Sampler
        s = Sampler.argmax()
        # Argmax compiles to TopP { temperature: 0.0, p: 1.0 }
        assert s._variant.value == (0.0, 1.0)

    def test_top_p(self):
        from inferlet import Sampler
        s = Sampler.top_p(temperature=0.7, p=0.9)
        assert s._variant.value == (0.7, 0.9)

    def test_top_k(self):
        from inferlet import Sampler
        s = Sampler.top_k(temperature=0.6, k=40)
        assert s._variant.value == (0.6, 40)

    def test_min_p(self):
        from inferlet import Sampler
        s = Sampler.min_p(temperature=0.5, p=0.1)
        assert s._variant.value == (0.5, 0.1)

    def test_top_k_top_p(self):
        from inferlet import Sampler
        s = Sampler.top_k_top_p(temperature=0.6, k=50, p=0.95)
        assert s._variant.value == (0.6, 50, 0.95)

    def test_multinomial(self):
        from inferlet import Sampler
        s = Sampler.multinomial(temperature=1.0, draws=1)
        assert s._variant.value == (1.0, 1)


# =============================================================================
# Probes — distribution access (dataclasses)
# =============================================================================


class TestProbes:
    def test_logits_dataclass(self):
        from inferlet import Logits
        from wit_world.imports.inference import Sampler_RawLogits
        p = Logits()
        assert isinstance(p._to_wit(), Sampler_RawLogits)

    def test_distribution_dataclass(self):
        from inferlet import Distribution
        from wit_world.imports.inference import Sampler_Dist
        p = Distribution(temperature=1.0, k=8)
        assert p.temperature == 1.0
        assert p.k == 8
        wit = p._to_wit()
        assert isinstance(wit, Sampler_Dist)
        assert wit.value == (1.0, 8)

    def test_logprob_dataclass(self):
        from inferlet import Logprob
        from wit_world.imports.inference import Sampler_Logprob
        p = Logprob(token=42)
        assert p.token == 42
        wit = p._to_wit()
        assert isinstance(wit, Sampler_Logprob)
        assert wit.value == 42

    def test_logprobs_dataclass(self):
        from inferlet import Logprobs
        from wit_world.imports.inference import Sampler_Logprobs
        p = Logprobs(tokens=(7, 11, 13))
        wit = p._to_wit()
        assert isinstance(wit, Sampler_Logprobs)
        assert wit.value == [7, 11, 13]

    def test_logprobs_accepts_iterable(self):
        """Logprobs takes any Iterable[int] and converts to tuple."""
        from inferlet import Logprobs
        # list works
        assert Logprobs([1, 2, 3]).tokens == (1, 2, 3)
        # tuple works
        assert Logprobs((4, 5, 6)).tokens == (4, 5, 6)
        # range works
        assert Logprobs(range(3)).tokens == (0, 1, 2)
        # default = empty
        assert Logprobs().tokens == ()

    def test_entropy_dataclass(self):
        from inferlet import Entropy
        from wit_world.imports.inference import Sampler_Entropy
        p = Entropy()
        assert isinstance(p._to_wit(), Sampler_Entropy)


# =============================================================================
# Context — chat fillers + lifecycle
# =============================================================================


class TestContext:
    def test_create(self):
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        assert ctx._handle is not None

    def test_chat_fillers_chain(self):
        """system / user / cue return self for chaining."""
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        result = ctx.system("Hello").user("World").cue()
        assert result is ctx

    def test_append_raw_tokens(self):
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        ctx.append([1, 2, 3])
        assert ctx.buffer() == [1, 2, 3]

    def test_fork(self):
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        ctx.append([1, 2, 3])
        forked = ctx.fork()
        assert forked._handle is not None

    def test_release_via_context_manager(self):
        from inferlet import Context, Model
        with Context(Model.load("test-model")) as ctx:
            ctx.append([42])
            assert ctx.buffer() == [42]

    def test_no_equip_tools_method(self):
        """The old equip_tools / answer_tool methods are gone — tools live
        in the `inferlet.tools` module now."""
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        assert not hasattr(ctx, "equip_tools")
        assert not hasattr(ctx, "answer_tool")

    def test_idle_context_manager(self):
        """`with ctx.idle():` drops bid to 0, restores on exit."""
        from inferlet import Context, Model

        class _RecordingHandle:
            """Wrap the FakeContext handle, recording bid() calls."""
            def __init__(self, inner):
                self._inner = inner
                self.bids: list[float] = []

            def bid(self, value: float) -> None:
                self.bids.append(value)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        ctx = Context(Model.load("test-model"))
        ctx._handle = _RecordingHandle(ctx._handle)
        with ctx.idle():
            assert ctx._handle.bids[-1] == 0.0
        # On exit a non-zero bid is restored (saved bid >= 0).
        assert len(ctx._handle.bids) == 2
        assert ctx._handle.bids[-1] >= 0.0


# =============================================================================
# Forward primitive
# =============================================================================


class TestForward:
    def test_create_via_context(self):
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        assert fwd._ctx is ctx
        assert fwd.start_position() == ctx.seq_len

    def test_input_appends(self):
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        fwd.input([1, 2, 3])
        assert fwd._auto_inputs == [1, 2, 3]

    def test_sample_returns_handle(self):
        from inferlet import Context, Model, Sampler
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        fwd.input([1, 2, 3])
        h = fwd.sample([2], Sampler.argmax())
        assert h.slot == 0
        assert h.arity == 1

    def test_probe_returns_typed_handle(self):
        from inferlet import Context, Distribution, Logits, Model
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        h_dist = fwd.probe(0, Distribution(temperature=1.0, k=8))
        h_logits = fwd.probe(0, Logits())
        assert h_dist.kind == "distribution"
        assert h_logits.kind == "logits"
        # Slots are sequential.
        assert h_dist.slot == 0
        assert h_logits.slot == 1

    def test_execute_empty_raises(self):
        """Forward.execute() with no inputs and no slots is almost
        certainly a programming error — raise rather than silently
        returning an empty Output."""
        import asyncio
        import pytest
        from inferlet import Context, Model
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        with pytest.raises(ValueError, match="no inputs and no slots"):
            asyncio.run(fwd.execute())

    def test_multi_arity_sample_advances_slot_correctly(self):
        """A multi-arity sampler attaches len(indices) Token slots, so the
        next slot index must advance by that count — not by 1."""
        from inferlet import Context, Distribution, Model, Sampler
        ctx = Context(Model.load("test-model"))
        fwd = ctx.forward()
        h_multi = fwd.sample([0, 1, 2], Sampler.argmax())
        h_probe = fwd.probe(0, Distribution(temperature=1.0, k=0))
        assert h_multi.slot == 0
        assert h_multi.arity == 3
        # 3 Token slots come first, so the probe lands at slot 3.
        assert h_probe.slot == 3


# =============================================================================
# Generator — kwargs-driven
# =============================================================================


class TestGenerator:
    def test_generate_returns_generator(self):
        from inferlet import Context, Generator, Model, Sampler
        ctx = Context(Model.load("test-model"))
        ctx.user("Hi")
        g = ctx.generate(Sampler.argmax(), max_tokens=64, auto_flush=False)
        assert isinstance(g, Generator)

    def test_chain_method(self):
        from inferlet import Context, Generator, Model, Sampler
        ctx = Context(Model.load("test-model"))
        g = (
            ctx.generate(Sampler.argmax(), auto_flush=False)
            .max_tokens(128)
            .horizon(256)
            .rebid_each_step(False)
        )
        assert isinstance(g, Generator)
        assert g._max_tokens == 128
        assert g._horizon == 256
        assert g._rebid_each_step is False

    def test_generator_rebids_each_step_by_default(self):
        import asyncio
        from inferlet import Context, Model, Sampler

        class _RecordingHandle:
            def __init__(self, inner):
                self._inner = inner
                self.bids: list[float] = []

            def bid(self, value: float) -> None:
                self.bids.append(value)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        ctx = Context(Model.load("test-model"))
        ctx._handle = _RecordingHandle(ctx._handle)
        g = ctx.generate(Sampler.argmax(), auto_flush=False)

        assert len(ctx._handle.bids) == 1
        asyncio.run(g.__anext__())
        assert len(ctx._handle.bids) == 2

    def test_generator_can_disable_step_rebids(self):
        import asyncio
        from inferlet import Context, Model, Sampler

        class _RecordingHandle:
            def __init__(self, inner):
                self._inner = inner
                self.bids: list[float] = []

            def bid(self, value: float) -> None:
                self.bids.append(value)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        ctx = Context(Model.load("test-model"))
        ctx._handle = _RecordingHandle(ctx._handle)
        g = ctx.generate(Sampler.argmax(), auto_flush=False, rebid_each_step=False)

        assert len(ctx._handle.bids) == 1
        asyncio.run(g.__anext__())
        assert len(ctx._handle.bids) == 1

    def test_constrain_with_schema(self):
        from inferlet import (
            Context,
            JsonSchema,
            Model,
            Sampler,
        )
        ctx = Context(Model.load("test-model"))
        g = ctx.generate(
            Sampler.argmax(),
            constrain=JsonSchema('{"type":"object"}'),
            auto_flush=False,
        )
        assert len(g._constraints) == 1

    def test_constrain_with_list_composes(self):
        from inferlet import AnyJson, Context, Ebnf, Model, Sampler
        ctx = Context(Model.load("test-model"))
        g = ctx.generate(
            Sampler.argmax(),
            constrain=[AnyJson(), Ebnf('root ::= "x"')],
            auto_flush=False,
        )
        assert len(g._constraints) == 2

    def test_speculator_and_system_are_mutually_exclusive(self):
        from inferlet import Context, Model, Sampler
        import pytest
        ctx = Context(Model.load("test-model"))

        class _Drafter:
            def draft(self): return [], []
            def accept(self, _): pass
            def rollback(self, _): pass
            def reset(self): pass

        with pytest.raises(ValueError):
            ctx.generate(
                Sampler.argmax(),
                speculator=_Drafter(),
                system_speculation=True,
                auto_flush=False,
            )


# =============================================================================
# Schema — Protocol + duck-typed implementors
# =============================================================================


class TestSchema:
    def test_jsonschema(self):
        from inferlet import JsonSchema
        s = JsonSchema('{"type":"object"}')
        assert s.schema == '{"type":"object"}'

    def test_anyjson(self):
        from inferlet import AnyJson
        s = AnyJson()
        # frozen dataclass — no fields
        assert s == AnyJson()

    def test_regex(self):
        from inferlet import Regex
        s = Regex(r"\d+")
        assert s.pattern == r"\d+"

    def test_ebnf(self):
        from inferlet import Ebnf
        s = Ebnf('root ::= "x"')
        assert s.source == 'root ::= "x"'

    def test_user_schema_via_protocol(self):
        """Any class with build_constraint(model) satisfies the Schema protocol."""
        from inferlet import Model, Schema

        class MyGrammar:
            def build_constraint(self, model):
                from inferlet import Grammar, GrammarConstraint
                return GrammarConstraint.from_grammar(Grammar.json(), model)

        # Runtime-checkable Protocol — isinstance works.
        assert isinstance(MyGrammar(), Schema)
        assert MyGrammar().build_constraint(Model.load("test-model")) is not None


# =============================================================================
# Decoders — independent, with Idle event
# =============================================================================


class TestDecoders:
    def test_chat_decoder_create(self):
        from inferlet import Model, chat
        d = chat.Decoder(Model.load("test-model"))
        assert d._inner is not None

    def test_chat_event_classes(self):
        from inferlet import chat
        # Idle has no fields
        idle = chat.Event.Idle()
        delta = chat.Event.Delta(text="hi")
        done = chat.Event.Done(text="all")
        interrupt = chat.Event.Interrupt(token=42)
        assert delta.text == "hi"
        assert done.text == "all"
        assert interrupt.token == 42

    def test_reasoning_decoder_create(self):
        from inferlet import Model, reasoning
        d = reasoning.Decoder(Model.load("test-model"))
        assert d._inner is not None

    def test_reasoning_event_classes(self):
        from inferlet import reasoning
        assert reasoning.Event.Start() == reasoning.Event.Start()
        delta = reasoning.Event.Delta(text="thinking…")
        end = reasoning.Event.End(text="full")
        assert delta.text == "thinking…"
        assert end.text == "full"


# =============================================================================
# Tools — opt-in, separate module
# =============================================================================


class TestTools:
    def test_equip_prefix(self):
        from inferlet import Model, tools
        toks = tools.equip_prefix(Model.load("test-model"), ['{"name":"calc"}'])
        assert isinstance(toks, list)

    def test_answer_prefix_dict(self):
        from inferlet import Model, tools
        toks = tools.answer_prefix(Model.load("test-model"), "calc", {"result": 42})
        assert isinstance(toks, list)

    def test_decoder_create(self):
        from inferlet import Model, tools
        d = tools.Decoder(Model.load("test-model"))
        assert d._inner is not None

    def test_event_classes(self):
        from inferlet import tools
        start = tools.Event.Start()
        call = tools.Event.Call(name="search", args='{"q":"x"}')
        assert call.name == "search"
        assert call.args == '{"q":"x"}'


# =============================================================================
# Speculator Protocol
# =============================================================================


class TestSpec:
    def test_protocol_is_runtime_checkable(self):
        from inferlet import Speculator

        class MyDrafter:
            def draft(self): return ([], [])
            def accept(self, _): pass
            def rollback(self, _): pass
            def reset(self): pass

        assert isinstance(MyDrafter(), Speculator)

    def test_missing_method_fails(self):
        from inferlet import Speculator

        class Incomplete:
            def draft(self): return ([], [])
            # missing accept / rollback / reset

        # Protocol checks structural conformance.
        assert not isinstance(Incomplete(), Speculator)


# =============================================================================
# Grammar / Matcher (low-level)
# =============================================================================


class TestGrammar:
    def test_grammar_factories(self):
        from inferlet import Grammar
        assert Grammar.from_json_schema('{"type":"object"}')._handle is not None
        assert Grammar.json()._handle is not None
        assert Grammar.from_regex(r"\d+")._handle is not None
        assert Grammar.from_ebnf('root ::= "x"')._handle is not None

    def test_matcher_create(self):
        from inferlet import Grammar, Matcher, Model
        m = Model.load("test-model")
        matcher = Matcher(Grammar.json(), m.tokenizer())
        assert not matcher.is_terminated


# =============================================================================
# Runtime / messaging / session — unchanged from before
# =============================================================================


class TestRuntime:
    def test_models(self):
        from inferlet import runtime
        assert runtime.models() == ["mock-model"]

    def test_version(self):
        from inferlet import runtime
        assert runtime.version() == "0.1.0-mock"


class TestMessaging:
    def test_push(self):
        from inferlet import messaging
        messaging.push("topic", "msg")


class TestSession:
    def test_send(self):
        from inferlet import session
        session.send("hello")


# =============================================================================
# Adapter / zo (carry-over)
# =============================================================================


class TestAdapter:
    def test_create(self):
        from inferlet import Adapter, Model
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        assert a._handle is not None


class TestZo:
    def test_adapter_seed(self):
        from inferlet import Adapter, Model, zo
        m = Model.load("test-model")
        # accept either a Forward (SDK) or a raw WIT ForwardPass
        fwd = m._handle  # raw handle as fallback
        zo.adapter_seed(fwd, 42)

    def test_initialize(self):
        from inferlet import Adapter, Model, zo
        a = Adapter.create(Model.load("test-model"), "my-adapter")
        zo.initialize(
            a,
            rank=8,
            alpha=16.0,
            population_size=64,
            mu_fraction=0.25,
            initial_sigma=0.1,
        )
