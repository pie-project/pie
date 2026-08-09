//! Gemma 2's chat template — which is gemma-3's, and therefore lives in
//! [`shared`](crate::shared).
//!
//! # What this file used to be
//!
//! A second implementation of the same format, headed "Gemma 2/3
//! instruct implementation" from when it served both. Gemma-3 was
//! given its own copy, that copy was corrected, and this one was not.
//! By the time they were compared they disagreed about two things, in
//! the direction that does not crash:
//!
//! * **No BOS.** Gemma is trained with a single `<bos>` opening the
//!   rendered chat, and this crate's tokenizer has no post-processor to
//!   add one — [`Tokenizer::encode`] returns exactly what the template
//!   asks for. Gemma-3 emits it in `first_user`; this file had no
//!   `bos_token` field at all, so every gemma-2 prompt began mid-stream.
//! * **A system turn Gemma has no room for.** `system()` returns bare
//!   text, correctly, and its doc said "the caller should embed this
//!   inside the first user turn" — but the caller is the default
//!   [`system_user`](crate::instruct::Instruct::system_user), which
//!   concatenates `system()` then `user()` and therefore placed that
//!   text *before* `<start_of_turn>`, outside any turn. Gemma-3
//!   overrides `system_user` to fold it in, as the reference template
//!   does.
//!
//! Its own `full_conversation` test asserted the broken rendering as a
//! golden string, which is how both survived: the vocabulary that test
//! builds even contains `<bos>`, and nothing emitted it.
//!
//! Neither could be fixed here without leaving two copies to diverge
//! again, so gemma-2 binds the one gemma-3 and gemma-3n already bind.
//! The tests below are what is gemma-2's own: that the template it gets
//! is the one Google published for gemma-2.
//!
//! [`Tokenizer::encode`]: tokenizer::Tokenizer::encode

pub use crate::shared::gemma_chat::Gemma3Instruct as GemmaInstruct;

#[cfg(test)]
mod tests {
    use super::GemmaInstruct;
    use crate::instruct::Instruct;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    fn gemma() -> (GemmaInstruct, Arc<Tokenizer>) {
        let vocab: Vec<String> = [
            "<start_of_turn>",
            "<end_of_turn>",
            "<eos>",
            "<bos>",
            "user",
            "model",
            "\n",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        (GemmaInstruct::new(tok.clone()), tok)
    }

    /// The published gemma-2 rendering, BOS and all.
    ///
    /// The string this replaced began `"Hello\n<start_of_turn>user"` —
    /// no BOS, and the system text outside every turn.
    #[test]
    fn a_conversation_renders_as_google_published_it() {
        let (inst, tok) = gemma();
        let mut tokens = inst.system_user("Be brief", "Hello");
        tokens.extend(inst.assistant("Hi"));
        tokens.extend(inst.user("More"));
        tokens.extend(inst.cue());
        assert_eq!(
            tok.decode(&tokens, false),
            "<bos><start_of_turn>user\nBe brief\nHello<end_of_turn>\n\
             <start_of_turn>model\nHi<end_of_turn>\n\
             <start_of_turn>user\nMore<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    /// Exactly one BOS, and only at the front.
    ///
    /// A second `<bos>` mid-conversation is the failure mode of fixing
    /// the first defect by putting the token in `user()` instead of
    /// `first_user()`.
    #[test]
    fn only_the_first_turn_carries_bos() {
        let (inst, tok) = gemma();
        let mut tokens = inst.first_user("Hello");
        tokens.extend(inst.assistant("Hi"));
        tokens.extend(inst.user("More"));
        let text = tok.decode(&tokens, false);
        assert_eq!(text.matches("<bos>").count(), 1, "{text}");
        assert!(text.starts_with("<bos>"), "{text}");
    }

    /// Gemma-2 seals on `<end_of_turn>` and `<eos>`, not `<|im_end|>`.
    #[test]
    fn it_seals_with_gemmas_tokens() {
        let (inst, tok) = gemma();
        let stop = inst.seal();
        assert_eq!(
            stop,
            vec![
                tok.token_to_id("<end_of_turn>").unwrap(),
                tok.token_to_id("<eos>").unwrap()
            ]
        );
    }

    /// Gemma-2 has no tool protocol, and says so by emitting nothing.
    #[test]
    fn it_has_no_tool_protocol() {
        let (inst, _tok) = gemma();
        assert!(inst.equip(&["{\"name\":\"f\"}".to_string()]).is_empty());
        assert!(inst.answer("f", "42").is_empty());
        assert!(inst.tool_call_grammar(&[]).is_none());
    }
}
