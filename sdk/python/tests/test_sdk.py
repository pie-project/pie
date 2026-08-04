"""
Unit tests for the inferlet SDK's hand-written layer.

SCOPE NOTE: these cover the non-forward interfaces only. That is not an
omission -- it is the SDK's entire surface right now. The forward-pass
interfaces (`forward` / `forward-recurrent` / `forward-hybrid` plus `channel`,
`working-set`, `pipeline`) have no Python counterpart, because the guest now
ships PTIR container bytes and the encoder that produces them exists only in
Rust. `scripts/check-sdk-interfaces.sh` fails on exactly that, deliberately,
and blocks publishing until it is fixed.

The previous version of this file tested `Sampler`, `Forward`, `Generator` and
`Context` against a mock of `pie:core/inference` -- an interface that had
already been deleted from the runtime. It passed the whole time. Keep the
stubs in `conftest.py` narrow, and treat the componentize-py build (see that
file's docstring) as the check that the world is real.
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
    ForwardKind,
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
            "model",
            "reasoning",
            "session",
            "tokenizer",
        }

    def test_forward_surface_is_absent(self):
        """The removed modules must stay removed, not come back as shims.

        A stub that raises on use would be worse than nothing: it would make
        `scripts/check-sdk-interfaces.sh` green while the SDK still cannot run
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

    def test_pass_kind_and_is_linear_agree(self):
        """`is_linear()` == (`pass_kind()` != attention) is a documented
        invariant of the WIT interface, and the reason both exist."""
        from inferlet import model

        assert model.pass_kind() is ForwardKind.HYBRID
        assert model.is_linear() == (model.pass_kind() is not ForwardKind.ATTENTION)

    def test_memory_shaping_capabilities(self):
        from inferlet import model

        assert model.kv_page_size() == 16
        assert model.frame_size() == 1
        assert model.channel_capacity() == 8
        assert model.max_embed_length() == 2048
        assert model.arena_block_size() == 8192

    def test_recurrent_capabilities(self):
        from inferlet import model

        assert model.rs_state_size() == 4096
        assert model.rs_buffer_page_size() == 64
        assert model.rs_fold_granularity() == 1

    def test_output_vocab_may_exceed_tokenizer_vocab(self):
        """These are different numbers on purpose -- logits are padded. Use
        `output_vocab_size` for anything logits-shaped."""
        from inferlet import model, tokenizer

        ids, _ = tokenizer.vocabs()
        assert model.output_vocab_size() > len(ids)

    def test_tokenizer_surface_moved_off_model(self):
        """It lives in `tokenizer` now; leaving an alias on `model` would hide
        the interface split from anyone reading the SDK."""
        from inferlet import model

        for gone in ("encode", "decode", "vocabs", "split_regex", "special_tokens"):
            assert not hasattr(model, gone)


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
        older SDK, so it is worth pinning."""
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
        """The WIT case is `complete`; the SDK calls it `End`. The rename is
        the reason this mapping is worth a test of its own."""
        from inferlet import reasoning

        dec = self._decoder([ReasoningComplete("done")])
        assert dec.feed([1]) == reasoning.Event.End("done")
