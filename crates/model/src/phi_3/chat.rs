//! Phi-3 instruct implementation.
//!
//! Chat shape:
//!   <|system|>\n{system}<|end|>\n
//!   <|user|>\n{user}<|end|>\n
//!   <|assistant|>\n{assistant}<|end|>\n
//!   ...
//!   <|assistant|>\n          ← cue
//!
//! Each role marker and `<|end|>` are single special tokens
//! (e.g. id 32010 / 32007 on Phi-3-mini-4k-instruct).

use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct Phi3Instruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    end_suffix: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Phi3Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<|end|>", "<|endoftext|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let newline = encode("\n");
        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = encode(role);
            v.extend(&newline);
            v
        };
        let mut end_suffix = encode("<|end|>");
        end_suffix.extend(&newline);

        Self {
            system_prefix: make_prefix("<|system|>"),
            user_prefix: make_prefix("<|user|>"),
            assistant_prefix: make_prefix("<|assistant|>"),
            end_suffix,
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for Phi3Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        let mut v = self.system_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.assistant_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
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
    use crate::instruct::{ChatEvent, ReasoningEvent, ToolEvent};

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| (*s).to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// A vocabulary that holds the four markers as single tokens, which
    /// is what a real Phi-3-instruct checkpoint ships.
    fn phi3() -> Phi3Instruct {
        Phi3Instruct::new(make_tok(&[
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<|endoftext|>",
            "\n",
            "Sys",
            "Hello",
            "Ok",
        ]))
    }

    #[test]
    fn a_turn_is_a_marker_a_newline_the_text_and_an_end() {
        let inst = phi3();
        let mut tokens = inst.system("Sys");
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Ok"));
        tokens.extend(inst.cue());
        assert_eq!(
            inst.tokenizer.decode(&tokens, false),
            "<|system|>\nSys<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nOk<|end|>\n\
             <|assistant|>\n"
        );
    }

    /// The cue is the assistant prefix and nothing else: a cue that
    /// carried an `<|end|>` would close the turn the model has not
    /// started.
    #[test]
    fn the_cue_opens_a_turn_and_does_not_close_it() {
        let inst = phi3();
        assert_eq!(inst.cue(), inst.assistant_prefix);
        assert_eq!(inst.tokenizer.decode(&inst.cue(), false), "<|assistant|>\n");
    }

    /// Both stop strings, in the order the constant lists them.
    #[test]
    fn the_seal_is_every_stop_the_vocabulary_holds() {
        let inst = phi3();
        let ids: Vec<String> = inst
            .seal()
            .iter()
            .map(|id| inst.tokenizer.decode(&[*id], false))
            .collect();
        assert_eq!(ids, vec!["<|end|>", "<|endoftext|>"]);
    }

    /// A vocabulary missing one stop keeps the other, rather than
    /// dropping both or inventing an id.
    #[test]
    fn a_missing_stop_is_dropped_and_the_rest_survive() {
        let inst = Phi3Instruct::new(make_tok(&[
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "\n",
        ]));
        assert_eq!(inst.seal().len(), 1);
        assert_eq!(
            inst.tokenizer.decode(&inst.seal(), false),
            "<|end|>",
            "the surviving stop is the one the vocabulary holds"
        );
    }

    /// Phi-3 has no tool protocol, and says so by producing nothing —
    /// which is a different thing from producing a protocol the model
    /// was never trained on.
    #[test]
    fn there_is_no_tool_protocol_here() {
        let inst = phi3();
        assert!(inst.equip(&["get_weather".to_string()]).is_empty());
        assert!(inst.answer("get_weather", "{\"t\":7}").is_empty());
        // The decoders still have to answer, and both answer with their
        // enum's "nothing to report" value — which for `ToolEvent` is
        // spelled `Start`, because the enum has no third variant.
        let mut tools = inst.tool_decoder();
        assert!(matches!(tools.feed(&[0, 1, 2]), ToolEvent::Start));
        tools.reset();
        let mut reasoning = inst.reasoning_decoder();
        match reasoning.feed(&[0, 1, 2]) {
            ReasoningEvent::Delta(delta) => assert!(
                delta.is_empty(),
                "with no reasoning channel there is no reasoning text"
            ),
            other => panic!("expected an empty delta, got {other:?}"),
        }
        reasoning.reset();
    }

    /// The decoder stops on the same ids `seal` publishes. These are two
    /// statements of one fact and they are made in two places, so a test
    /// holds them together.
    #[test]
    fn the_chat_decoder_stops_where_the_seal_says() {
        let inst = phi3();
        let mut decoder = inst.chat_decoder();
        let text = inst.tokenizer.encode("Ok");
        match decoder.feed(&text) {
            ChatEvent::Delta(delta) => assert_eq!(delta, "Ok"),
            other => panic!("expected a delta, got {other:?}"),
        }
        for stop in inst.seal() {
            let mut decoder = inst.chat_decoder();
            decoder.feed(&text);
            match decoder.feed(&[stop]) {
                ChatEvent::Done(all) => assert_eq!(all, "Ok", "the turn is what came before"),
                other => panic!("{stop} did not end the turn: {other:?}"),
            }
        }
    }
}
