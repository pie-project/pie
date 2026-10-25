"""
Unit tests for the inferlet library's hand-written layer, against the stub
`wit_world` in `conftest.py`.

The container bytes the `eta` port encodes are pinned to the Rust
encoder's by `test_eta_goldens.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from conftest import (  # noqa: E402
    SESSION_SPY,
    ChatDecoderStub,
    ChatDelta,
    ChatDone,
    ChatInterrupt,
    ReasoningComplete,
    ReasoningDecoderStub,
    ReasoningDelta,
    ReasoningStart,
)


# =============================================================================
# Package surface
# =============================================================================


class TestPackageSurface:
    def test_exports_only_the_live_modules(self):
        import inferlet

        assert set(inferlet.__all__) == {
            "chat",
            "eta",
            "grammar",
            "mask",
            "media",
            "model",
            "reasoning",
            "session",
            "tokenizer",
            "tools",
        }

    def test_forward_surface_is_absent(self):
        """The removed modules must stay removed, not come back as shims.

        A stub that raises on use would be worse than nothing: it would make
        `scripts/check-inferlet-interfaces.sh` green while the inferlet library still cannot run
        a model.
        """
        import inferlet

        for gone in (
            "Context",
            "Forward",
            "Generator",
            "Sampler",
            "Adapter",
            "Speculator",
        ):
            assert not hasattr(inferlet, gone), f"{gone} should not be re-exported"


# =============================================================================
# Model
# =============================================================================


class TestModel:
    def test_identity(self):
        from inferlet import model

        assert model.name() == "mock-model"
        assert model.architecture() == "qwen3_5"
        assert model.default_system_speculation() is False

    def test_state_and_switches(self):
        from inferlet import model

        assert model.rs_state_size() > 0
        assert model.switch("mode").values == ["encode", "denoise"]
        assert model.switch("reading") is None

    def test_memory_shaping_capabilities(self):
        from inferlet import model

        assert model.kv_page_size() == 16
        assert model.frame_size() == 1
        assert model.channel_capacity() == 8
        assert model.max_readout_rows(model.ReadoutKind.LOGITS) == 256
        assert model.max_readout_rows(model.ReadoutKind.HIDDEN) == 2048

    def test_output_vocab_may_exceed_tokenizer_vocab(self):
        """These are different numbers on purpose -- logits are padded. Use
        `output_vocab_size` for anything logits-shaped."""
        from inferlet import model, tokenizer

        ids, _ = tokenizer.vocabs()
        assert model.output_vocab_size() > len(ids)

    def test_tokenizer_surface_rides_model_too(self):
        """It lives in `tokenizer`, and `model` re-exports it so inferlet
        source reads `model.encode` the way the Rust `inferlet` crate's does."""
        from inferlet import model, tokenizer

        for name in ("encode", "decode", "vocabs", "split_regex", "special_tokens"):
            assert getattr(model, name) is getattr(tokenizer, name)


# =============================================================================
# Tokenizer
# =============================================================================


class TestTokenizer:
    def test_encode_decode_roundtrip(self):
        from inferlet import tokenizer

        assert tokenizer.encode("Hi") == [72, 105]
        assert tokenizer.decode([72, 105]) == "Hi"

    def test_encode_returns_a_real_list(self):
        """The binding hands back a host sequence; the wrapper's `list()` is
        what makes it indexable and mutable on the Python side."""
        from inferlet import tokenizer

        out = tokenizer.encode("abc")
        assert isinstance(out, list)
        out.append(0)

    def test_vocabs_and_special_tokens(self):
        from inferlet import tokenizer

        ids, byte_seqs = tokenizer.vocabs()
        assert ids == [0, 1] and byte_seqs == [b"a", b"b"]

        sids, sbytes = tokenizer.special_tokens()
        assert sids == [2] and sbytes == [b"<eos>"]

    def test_split_regex(self):
        from inferlet import tokenizer

        assert tokenizer.split_regex() == r"\w+"


# =============================================================================
# Session
# =============================================================================


class TestSession:
    def setup_method(self):
        SESSION_SPY.reset()

    def test_str_is_sent_verbatim(self):
        from inferlet import session

        session.send("plain text")
        assert SESSION_SPY.sent == ["plain text"]

    def test_non_str_is_json_encoded(self):
        from inferlet import session

        session.send({"event": "tick", "n": 3})
        session.send([1, 2, 3])
        assert json.loads(SESSION_SPY.sent[0]) == {"event": "tick", "n": 3}
        assert json.loads(SESSION_SPY.sent[1]) == [1, 2, 3]

    def test_model_dump_json_is_preferred_over_json_dumps(self):
        from inferlet import session

        class Pydanticish:
            def model_dump_json(self) -> str:
                return '{"from":"model_dump_json"}'

        session.send(Pydanticish())
        assert SESSION_SPY.sent == ['{"from":"model_dump_json"}']

    def test_unserializable_falls_back_to_str(self):
        """`default=str` is what keeps a stray object from raising inside a
        send -- worth pinning, since it silently changes the payload."""
        from inferlet import session

        class Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        session.send({"o": Opaque()})
        assert json.loads(SESSION_SPY.sent[0]) == {"o": "<opaque>"}

    def test_receive(self):
        from inferlet import session

        SESSION_SPY.to_receive.append("hello")
        assert asyncio.run(session.receive()) == "hello"

    def test_receive_raises_when_the_host_yields_none(self):
        from inferlet import session

        with pytest.raises(RuntimeError):
            asyncio.run(session.receive())

    def test_file_roundtrip(self):
        from inferlet import session

        session.send_file(b"\x00\x01")
        assert SESSION_SPY.sent_files == [b"\x00\x01"]

        SESSION_SPY.files_to_receive.append(b"\x02")
        assert asyncio.run(session.receive_file()) == b"\x02"

    def test_receive_file_raises_when_the_host_yields_none(self):
        from inferlet import session

        with pytest.raises(RuntimeError):
            asyncio.run(session.receive_file())


# =============================================================================
# Chat
# =============================================================================


class TestChatFillers:
    def test_fillers_return_lists(self):
        from inferlet import chat

        assert chat.system("s")[0] == 1
        assert chat.user("u")[0] == 3
        assert chat.assistant("a")[0] == 5
        assert chat.cue() == [6]
        assert chat.cue_without_thinking() == [6, 7]
        assert chat.cue_thinking(True) == chat.cue()
        assert chat.cue_thinking(False) == chat.cue_without_thinking()
        assert chat.seal() == [7]
        assert chat.stop_tokens() == [8, 9]
        assert isinstance(chat.system("s"), list)


class TestChatDecoder:
    def teardown_method(self):
        ChatDecoderStub.script = []

    def _decoder(self, script):
        from inferlet import chat

        ChatDecoderStub.script = script
        return chat.Decoder()

    def test_delta_and_done(self):
        from inferlet import chat

        dec = self._decoder([ChatDelta("he"), ChatDelta("llo"), ChatDone("hello")])
        assert dec.feed([1]) == chat.Event.Delta("he")
        assert dec.feed([2]) == chat.Event.Delta("llo")
        assert dec.feed([3]) == chat.Event.Done("hello")

    def test_empty_delta_becomes_idle(self):
        """So callers never need an `if text:` guard around a Delta branch."""
        from inferlet import chat

        dec = self._decoder([ChatDelta("")])
        assert dec.feed([1]) == chat.Event.Idle()

    def test_interrupt_is_surfaced_raw(self):
        from inferlet import chat

        dec = self._decoder([ChatInterrupt(42)])
        assert dec.feed([1]) == chat.Event.Interrupt(42)

    def test_unknown_variant_is_idle(self):
        """`feed` returns `result<event, error>`, so it never returns nothing --
        but the wrapper still has a trailing Idle for a case it does not know.
        That branch is what keeps a newly added WIT variant from crashing an
        older library, so it is worth pinning."""
        from inferlet import chat

        dec = self._decoder([object()])
        assert dec.feed([1]) == chat.Event.Idle()

    def test_tokens_reach_the_host_decoder(self):
        dec = self._decoder([ChatDelta("x")])
        dec.feed([7, 8])
        assert dec._inner.fed == [[7, 8]]

    def test_reset_forwards(self):
        dec = self._decoder([])
        dec.reset()
        assert dec._inner.resets == 1


# =============================================================================
# Reasoning
# =============================================================================


class TestReasoningDecoder:
    def teardown_method(self):
        ReasoningDecoderStub.script = []

    def _decoder(self, script):
        from inferlet import reasoning

        ReasoningDecoderStub.script = script
        return reasoning.Decoder()

    def test_start_delta_end(self):
        from inferlet import reasoning

        dec = self._decoder(
            [ReasoningStart(), ReasoningDelta("think"), ReasoningComplete("think")]
        )
        assert dec.feed([1]) == reasoning.Event.Start()
        assert dec.feed([2]) == reasoning.Event.Delta("think")
        assert dec.feed([3]) == reasoning.Event.End("think")

    def test_empty_delta_becomes_idle(self):
        from inferlet import reasoning

        dec = self._decoder([ReasoningDelta("")])
        assert dec.feed([1]) == reasoning.Event.Idle()

    def test_unknown_variant_is_idle(self):
        from inferlet import reasoning

        dec = self._decoder([object()])
        assert dec.feed([1]) == reasoning.Event.Idle()

    def test_complete_maps_to_end(self):
        """The WIT case is `complete`; the inferlet library calls it `End`. The rename is
        the reason this mapping is worth a test of its own."""
        from inferlet import reasoning

        dec = self._decoder([ReasoningComplete("done")])
        assert dec.feed([1]) == reasoning.Event.End("done")


class TestForwardPassState:
    def test_state_binds_lanes_and_only_the_stated_ports(self):
        from inferlet.eta import Channel, ForwardPass, Port, StateGeometry, WorkingSet, dtype

        a, b = WorkingSet(), WorkingSet()
        slots = Channel.from_([3, 4], dtype.u32)
        fwd = ForwardPass()
        fwd.state([a, b, a], StateGeometry(slots=slots))
        lanes, geom = fwd.wit.state_args
        assert lanes == [a.ws, b.ws, a.ws]
        assert geom.slots is slots._wit()
        assert (geom.positions, geom.kv_len, geom.mask, geom.fold) == (None, None, None, None)
        assert [p for p, _ in fwd._ports] == [Port.SLOTS]

    def test_fold_all_is_one_constant_channel_per_pass(self):
        from inferlet.eta import ForwardPass, InferletError, Port, StateGeometry, WorkingSet

        fwd = ForwardPass()
        fwd.state([WorkingSet()], StateGeometry.folding())
        (_, first) = fwd.wit.state_args
        assert [p for p, _ in fwd._ports] == [Port.FOLD]
        with pytest.raises(InferletError, match="fold is already bound"):
            fwd.state([WorkingSet()], StateGeometry.folding())
        fwd._program_attached = True  # a rebind once the program is attached
        fwd.state([WorkingSet()], StateGeometry.folding())
        (_, again) = fwd.wit.state_args
        assert again.fold is first.fold
        with pytest.raises(ValueError):
            ForwardPass().state([WorkingSet()], StateGeometry(fold="some"))  # type: ignore[arg-type]

    def test_checkpoints(self):
        from inferlet.eta import ForwardPass

        fwd = ForwardPass()
        fwd.checkpoints([16, 32])
        assert fwd.wit.checkpoint_slots == [16, 32]


class TestDiffusionSurface:
    def test_model_facts(self):
        from inferlet import model

        assert model.positions() is None
        assert [i.name for i in model.lane([("mode", "denoise")]).inputs] == ["self_cond_ids"]
        assert model.draft_block() is None
        assert model.run_ahead_window() == 4

    def test_pass_sets_a_model_switch(self):
        from inferlet.eta import ForwardPass

        fwd = ForwardPass()
        fwd.switch("mode", "encode")
        assert not fwd.switched()
        fwd.switch("mode", "denoise")
        assert fwd.wit.switches == [("mode", "encode"), ("mode", "denoise")]
        assert fwd.switched()

    def test_linear_temperature_is_the_reference_schedule(self):
        from inferlet.eta.diffusion import linear_temperature

        assert linear_temperature(4, 4, 1.0, 0.2) == pytest.approx(1.0)
        assert linear_temperature(1, 4, 1.0, 0.2) == pytest.approx(0.4)
