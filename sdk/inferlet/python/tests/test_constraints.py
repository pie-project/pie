"""The grammar / mask / tools / media wrappers, against the conftest stubs."""

from __future__ import annotations

import pytest

from inferlet import grammar, mask, media, tools


class TestMask:
    def test_bit_allowed_indexes_word_and_bit(self):
        m = [0b101]
        assert mask.bit_allowed(m, 0) and not mask.bit_allowed(m, 1) and mask.bit_allowed(m, 2)
        m2 = [0, 0b10]
        assert mask.bit_allowed(m2, 33) and not mask.bit_allowed(m2, 32)

    def test_refuses_tokens_past_the_mask(self):
        m = mask.pack_allowed(151_669, [7, 151_668])
        assert len(m) == 4740
        assert mask.bit_allowed(m, 7) and mask.bit_allowed(m, 151_668)
        for j in (151_680, 151_935, 1 << 40):
            assert not mask.bit_allowed(m, j)

    def test_pack_unpack_round_trip(self):
        m = mask.pack_allowed(40, [0, 2, 33, 99])
        assert len(m) == 2
        bools = mask.unpack_mask(m, 40)
        assert [i for i, b in enumerate(bools) if b] == [0, 2, 33]
        assert mask.unpack_mask([], 5) == [True] * 5
        assert mask.all_allowed(33) == [0xFFFF_FFFF, 0xFFFF_FFFF]

    def test_apply_mask_argmax(self):
        assert mask.apply_mask_argmax([0.1, 9.0, 3.0], [0b101]) == 2
        assert mask.apply_mask_argmax([0.1, 9.0, 3.0], [0]) == 0


class TestGrammar:
    def test_matcher_walks_and_terminates(self):
        g = grammar.Grammar.from_json_schema('{"type": "object"}')
        assert str(g) == '{"type": "object"}'
        m = grammar.Matcher(g)
        assert m.mask() == [0b101]
        m.accept_tokens([1, 2])
        assert not m.is_terminated()
        f = m.fork()
        f.accept_tokens([3])
        assert f.is_terminated() and not m.is_terminated()
        m.rollback(1)
        assert m.rollback_capacity() == 1

    def test_errors_are_grammar_errors(self):
        with pytest.raises(grammar.GrammarError, match="not a schema"):
            grammar.Grammar.from_json_schema("bad")
        m = grammar.Matcher(grammar.Grammar.json())
        with pytest.raises(grammar.GrammarError, match="999"):
            m.accept_tokens([999])


class TestTools:
    def test_equip_answer_format(self):
        assert tools.equip(["a", "b"]) == [10, 2]
        assert tools.answer("f", "xyz") == [11, 1, 3]
        assert tools.format([]) is None
        assert str(tools.format(["a"])) == "tools"
        assert tools.create_matcher(["a"]).mask() == [0b101]

    def test_decoder_events(self):
        d = tools.Decoder()
        assert isinstance(d.feed([1]), tools.Start)
        ev = d.feed([2])
        assert isinstance(ev, tools.Call) and ev.call == tools.ToolCall("lookup", '{"q": 1}')


class TestMedia:
    def test_image(self):
        img = media.Image.from_bytes(b"png")
        assert img.tokens() == [5, 6, 7] and img.token_count() == 3
        assert img.grid() == media.MergedGrid(1, 2, 3)
        assert img.digest() == b"\x01\x02"
        with pytest.raises(media.MediaError):
            media.Image.from_bytes(b"")

    def test_forward_pass_media_wraps_handles(self):
        from inferlet.eta import ForwardPass

        img = media.Image.from_bytes(b"png")
        aud = media.Audio.from_bytes(b"wav")
        fwd = ForwardPass()
        fwd.media([img, aud])
        # The stub lowers `MediaSpan_*` to `(tag, resource)`: the pass must
        # hand over the WIT resource, not the SDK wrapper.
        assert fwd.wit.spans == [("image", img.handle), ("audio", aud.handle)]
        with pytest.raises(TypeError):
            fwd.media([b"raw"])
