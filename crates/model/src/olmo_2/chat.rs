//! OLMo-2 instruct implementation.
//!
//! OLMo-2's published chat template is *not* ChatML (despite the family
//! sometimes being grouped with OLMo-3 / qwen-style templates). It uses
//! plain `<|system|>` / `<|user|>` / `<|assistant|>` role markers with
//! newline message separators, prefixed with `<|endoftext|>` as a BOS
//! and terminated with `<|endoftext|>` as the stop token. There is no
//! `<|end|>` between messages.
//!
//! Chat shape (rendered):
//!     <|endoftext|>
//!     <|system|>\n{system}\n
//!     <|user|>\n{user}\n
//!     <|assistant|>\n{assistant}<|endoftext|>
//!     ...
//!     <|assistant|>\n          ← cue
//!
//! Verified by `tokenizer.apply_chat_template` on
//! `allenai/OLMo-2-1124-7B-Instruct`.

use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct Olmo2Instruct {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    system_prefix: Vec<u32>,    // "<|system|>\n"
    user_prefix: Vec<u32>,      // "<|user|>\n"
    assistant_prefix: Vec<u32>, // "<|assistant|>\n"
    newline: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Olmo2Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = encode(role);
            v.extend(&newline);
            v
        };

        let stop_strs = ["<|endoftext|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            bos: encode("<|endoftext|>"),
            system_prefix: make_prefix("<|system|>"),
            user_prefix: make_prefix("<|user|>"),
            assistant_prefix: make_prefix("<|assistant|>"),
            newline,
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for Olmo2Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        // OLMo-2's template puts <|endoftext|> at the very start of the
        // conversation. We prepend it on the first message (system) so
        // the framing matches the tokenizer's apply_chat_template output.
        let mut v = self.bos.clone();
        v.extend(&self.system_prefix);
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.newline);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.newline);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.assistant_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        // Assistant turn closes with the EOS token.
        v.extend(&self.stop_ids);
        v
    }

    fn cue(&self) -> Vec<u32> {
        self.assistant_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }
    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruct::ChatEvent;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| (*s).to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// The three role markers and the terminator as single tokens, which
    /// is what a real `allenai/OLMo-2-1124-*-Instruct` ships.
    fn olmo() -> Olmo2Instruct {
        Olmo2Instruct::new(make_tok(&[
            "<|endoftext|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "\n",
            "Sys",
            "Hello",
            "Ok",
        ]))
    }

    /// The whole conversation, against the rendering this module's own
    /// doc records `tokenizer.apply_chat_template` producing.
    ///
    /// Whole rather than per-method because the failure worth catching
    /// is a framing one — a missing newline or a marker in the wrong
    /// place reads fine method by method and produces a prompt the model
    /// has never seen.
    #[test]
    fn a_conversation_renders_as_the_published_template_does() {
        let inst = olmo();
        let mut tokens = inst.system("Sys");
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Ok"));
        tokens.extend(inst.cue());
        assert_eq!(
            inst.tokenizer.decode(&tokens, false),
            "<|endoftext|><|system|>\nSys\n<|user|>\nHello\n\
             <|assistant|>\nOk<|endoftext|><|assistant|>\n"
        );
    }

    /// The BOS goes on the FIRST message and nowhere else.
    ///
    /// `system` prepends it; `user` and `assistant` do not. A
    /// conversation that opens on a user turn therefore has none, which
    /// is what the template does too.
    #[test]
    fn the_terminator_opens_the_conversation_and_closes_each_answer() {
        let inst = olmo();
        let eot = inst
            .tokenizer
            .token_to_id("<|endoftext|>")
            .expect("in vocab");
        assert_eq!(inst.system("Sys").first(), Some(&eot));
        assert_ne!(inst.user("Hello").first(), Some(&eot));
        assert_ne!(inst.assistant("Ok").first(), Some(&eot));
        assert_eq!(inst.assistant("Ok").last(), Some(&eot));
        assert_eq!(inst.seal(), vec![eot]);
    }

    /// There is no `<|end|>` between messages — a newline separates
    /// them. This is the difference from ChatML the module doc opens by
    /// naming, and getting it wrong is invisible until generation
    /// quality drops.
    #[test]
    fn a_user_turn_is_closed_by_a_newline_and_not_by_a_marker() {
        let inst = olmo();
        assert_eq!(
            inst.tokenizer.decode(&inst.user("Hello"), false),
            "<|user|>\nHello\n"
        );
        assert_eq!(
            inst.tokenizer.decode(&inst.system("Sys"), false),
            "<|endoftext|><|system|>\nSys\n"
        );
        assert_eq!(
            inst.tokenizer.decode(&inst.cue(), false),
            "<|assistant|>\n",
            "the cue is the prefix alone, so the model writes the answer"
        );
    }

    /// OLMo-2 has no tool protocol and no reasoning channel, and the
    /// three methods that would carry them say so by being empty rather
    /// than by emitting a marker the vocabulary does not contain.
    #[test]
    fn the_protocols_this_family_does_not_have_are_empty_rather_than_invented() {
        let inst = olmo();
        assert!(inst.equip(&["get_weather".to_string()]).is_empty());
        assert!(inst.answer("get_weather", "{\"c\":21}").is_empty());
        let mut r = inst.reasoning_decoder();
        assert!(
            matches!(r.feed(&[0]), crate::instruct::ReasoningEvent::Delta(d) if d.is_empty()),
            "no reasoning channel, so every token is an empty delta"
        );
        let mut t = inst.tool_decoder();
        assert!(
            matches!(t.feed(&[0]), crate::instruct::ToolEvent::Start),
            "no tool protocol, so the decoder never leaves its waiting state"
        );
    }

    /// The chat decoder stops on the same token `seal` names, which is
    /// the property that makes the two answers one fact.
    #[test]
    fn the_decoder_stops_on_the_token_the_seal_names() {
        let inst = olmo();
        let eot = inst
            .tokenizer
            .token_to_id("<|endoftext|>")
            .expect("in vocab");
        let ok = inst.tokenizer.token_to_id("Ok").expect("in vocab");
        let mut d = inst.chat_decoder();
        assert!(
            matches!(d.feed(&[ok]), ChatEvent::Delta(t) if t == "Ok"),
            "an ordinary token is text"
        );
        assert!(
            matches!(d.feed(&[eot]), ChatEvent::Done(t) if t == "Ok"),
            "and the terminator ends the turn, carrying what was said"
        );
        assert_eq!(inst.seal(), vec![eot], "and `seal` names that same token");
    }
}
