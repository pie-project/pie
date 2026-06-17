//! Gemma 4 instruct implementation.
//!
//! Diverges from Gemma 2/3: Gemma 4 introduces a single-token turn
//! delimiter pair (`<|turn>` id 105 / `<turn|>` id 106) instead of the
//! multi-piece `<start_of_turn>` / `<end_of_turn>`.
//!
//! Gemma 4 also has a native system/developer block:
//!
//!   <bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n
//!
//! Gemma 4 wraps the assistant turn in a Harmony-style "thought" channel.
//! The official chat template's generation prompt is
//!
//!   <|turn>model\n<|channel>thought\n<channel|>          (enable_thinking=false, the default)
//!   <|turn>model\n                                       (enable_thinking=true)
//!
//! and an assistant turn renders as
//!
//!   <|channel>thought\n{reasoning}\n<channel|>{answer}<turn|>
//!
//! i.e. `<|channel>` (id 100) opens the thought header, `<channel|>` (id
//! 101) closes it, and the final answer follows bare until `<turn|>`. The
//! `cue()` MUST include the trained `<|channel>thought\n<channel|>` scaffold
//! (non-thinking default) — without it the model is left to invent the
//! channel structure itself and degenerates. The reasoning decoder demuxes
//! the thought channel from the visible answer.

use crate::model::instruct::decoders::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::model::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    Gemma4,
    Gemma4Text,
}

pub struct Gemma4Instruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
    /// `<|channel>thought\n<channel|>` — the non-thinking generation scaffold
    /// appended after `model_prefix` in `cue()`. Pre-closing the thought
    /// channel makes the model emit the final answer directly.
    thought_scaffold: Vec<u32>,
    /// Reasoning-block open marker `<|channel>thought` — when the model DOES
    /// emit a thought block (thinking mode, or spontaneously), the reasoning
    /// decoder enters on this and exits on `channel_close_ids`.
    thought_open_ids: Vec<u32>,
    /// `<channel|>` — the thought-channel close marker (reasoning-block end).
    channel_close_ids: Vec<u32>,
}

impl Gemma4Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::for_variant(tokenizer, Gemma4Variant::Gemma4)
    }

    pub fn for_variant(tokenizer: Arc<Tokenizer>, _variant: Gemma4Variant) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        // `<turn|>` (closing) + `<eos>` are both terminal — generation
        // stops at either. The runtime's `seal()` returns this list.
        let stop_strs = ["<turn|>", "<eos>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        // Gemma-4's tokenizer treats `<|turn>` and `<turn|>` as single
        // added tokens (ids 105 and 106 on the E2B vocab); `encode`
        // returns a 1-element vector for each. We assemble the
        // role-prefixes by token concatenation, matching how
        // GemmaInstruct does it for `<start_of_turn>`.
        let open_turn = encode("<|turn>");
        let close_turn = encode("<turn|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = open_turn.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = close_turn;
        turn_suffix.extend(&newline);

        // Thought-channel markers. `<|channel>` (100) opens, `<channel|>`
        // (101) closes. The open marker for the *reasoning* block is
        // `<|channel>thought`; the scaffold pre-closes it for non-thinking
        // generation.
        let channel_open = encode("<|channel>");
        let channel_close = encode("<channel|>");
        let mut thought_open_ids = channel_open.clone();
        thought_open_ids.extend(encode("thought"));

        let mut thought_scaffold = thought_open_ids.clone();
        thought_scaffold.extend(&newline);
        thought_scaffold.extend(&channel_close);

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("model"),
            turn_suffix,
            bos_token: encode("<bos>"),
            stop_ids,
            thought_scaffold,
            thought_open_ids,
            channel_close_ids: channel_close,
            tokenizer,
        }
    }

    fn encode_trimmed(&self, message: &str) -> Vec<u32> {
        self.tokenizer.encode(message.trim())
    }
}

impl Instruct for Gemma4Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(system));
        v.extend(&self.turn_suffix);
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(user));
        v.extend(&self.turn_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.model_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn cue(&self) -> Vec<u32> {
        // <|turn>model\n<|channel>thought\n<channel|> — the official
        // enable_thinking=false generation prompt. The trained scaffold is
        // load-bearing: omitting it leaves the model to invent the channel
        // structure and degenerate.
        let mut v = self.model_prefix.clone();
        v.extend(&self.thought_scaffold);
        v
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
        // Demux the Harmony thought channel: enter on `<|channel>thought`,
        // exit on `<channel|>`; the bare text after the close is the visible
        // answer. With the non-thinking cue the thought is pre-closed in the
        // prompt, so the decoder stays Outside and all generated text is
        // visible — but a spontaneous (or thinking-mode) thought block is
        // still demuxed instead of leaking as content.
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.thought_open_ids.clone(),
            self.channel_close_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn gemma4() -> Gemma4Instruct {
        let tok = make_tok(&[
            "<|turn>", "<turn|>", "<eos>", "<bos>", "system", "user", "model", "\n", "Sys",
            "Hello", "Ok", "<|channel>", "thought", "<channel|>", "answer", "reason",
        ]);
        Gemma4Instruct::new(tok)
    }

    #[test]
    fn system_user_uses_native_system_turn_once() {
        let inst = gemma4();
        let mut tokens = inst.system_user("Sys", "Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        // cue() emits the non-thinking generation scaffold after the model role.
        assert_eq!(
            text,
            "<bos><|turn>system\nSys<turn|>\n<|turn>user\nHello<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn first_user_starts_with_bos() {
        let inst = gemma4();
        let mut tokens = inst.first_user("Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn cue_appends_non_thinking_thought_scaffold() {
        let inst = gemma4();
        let cue = inst.cue();
        let text = inst.tokenizer.decode(&cue, false);
        // The official enable_thinking=false generation prompt: the model
        // role followed by an empty (pre-closed) thought channel.
        assert_eq!(text, "<|turn>model\n<|channel>thought\n<channel|>");
        // The scaffold must be the model role prefix + the thought scaffold,
        // and must end with the channel-close marker (id 101 on the real
        // vocab) so the model emits the answer directly.
        assert_eq!(&cue[..inst.model_prefix.len()], &inst.model_prefix[..]);
        assert_eq!(cue.last(), inst.channel_close_ids.last());
    }

    #[test]
    fn reasoning_decoder_demuxes_thought_channel_from_answer() {
        use crate::model::instruct::ReasoningEvent;
        let inst = gemma4();
        let mut dec = inst.reasoning_decoder();
        let enc = |s: &str| inst.tokenizer.encode(s);
        // A thinking-mode-style emission: <|channel>thought reason <channel|> answer.
        // Feed the decoder the instruct's own marker ids (the exact ids it was
        // built with) so the test pins demux behavior, not toy-vocab encoding.
        match dec.feed(&inst.thought_open_ids) {
            ReasoningEvent::Start => {}
            other => panic!("expected Start on thought open, got {:?}", other),
        }
        // Reasoning content accumulates.
        match dec.feed(&enc("reason")) {
            ReasoningEvent::Delta(s) => assert_eq!(s, "reason"),
            other => panic!("expected reasoning Delta, got {:?}", other),
        }
        // Channel close ends the thought block.
        match dec.feed(&inst.channel_close_ids) {
            ReasoningEvent::Complete(s) => assert_eq!(s, "reason"),
            other => panic!("expected Complete on channel close, got {:?}", other),
        }
        // The bare answer after the close is NOT reasoning (stays Outside).
        match dec.feed(&enc("answer")) {
            ReasoningEvent::Delta(s) => assert!(s.is_empty(), "answer must not be reasoning"),
            other => panic!("expected empty Delta for answer, got {:?}", other),
        }
    }

    #[test]
    fn chat_decoder_stops_on_turn_close() {
        use crate::model::instruct::ChatEvent;
        let inst = gemma4();
        // stop_ids resolves <turn|> (and <eos>) — id 1 = <turn|> on the toy vocab.
        let turn_close = inst.tokenizer.token_to_id("<turn|>").unwrap();
        assert!(inst.stop_ids.contains(&turn_close));
        let mut dec = inst.chat_decoder();
        dec.feed(&inst.tokenizer.encode("answer"));
        match dec.feed(&[turn_close]) {
            ChatEvent::Done(s) => assert_eq!(s, "answer"),
            other => panic!("expected Done on <turn|>, got {:?}", other),
        }
    }

    #[test]
    fn later_user_omits_bos() {
        let inst = gemma4();
        let tokens = inst.user("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>user\nHello<turn|>\n");
    }

    #[test]
    fn assistant_uses_model_role() {
        let inst = gemma4();
        let tokens = inst.assistant("Ok");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>model\nOk<turn|>\n");
    }

    // Pins pie's role-structure encoding against real google/gemma-4-E4B-it
    // token ids (system + user + model role prefix, through `<|turn>model\n`).
    // We extend `model_prefix` rather than `cue()` here: `cue()` additionally
    // appends the non-thinking thought scaffold whose `thought` token id is
    // variant/vocab-dependent and not introspectable without that exact
    // tokenizer. The scaffold tail is verified separately on a controlled
    // vocab by `cue_appends_non_thinking_thought_scaffold`.
    #[test]
    fn cached_gemma4_matches_hf_benchmark_role_prompt_ids() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let tokenizer_path = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots")
            .read_dir()
            .ok()
            .and_then(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path().join("tokenizer.json"))
                    .find(|path| path.exists())
            });
        let Some(tokenizer_path) = tokenizer_path else {
            return;
        };
        let Ok(tokenizer) = Tokenizer::from_file(&tokenizer_path) else {
            return;
        };
        let inst = Gemma4Instruct::new(Arc::new(tokenizer));
        let mut tokens = inst.system_user(
            "You are a helpful benchmarking assistant.",
            "Write a short story about a robot. (Request #0)",
        );
        tokens.extend(&inst.model_prefix);
        assert_eq!(
            tokens,
            vec![
                2, 105, 9731, 107, 3048, 659, 496, 11045, 141657, 16326, 236761, 106, 107, 105,
                2364, 107, 6974, 496, 2822, 3925, 1003, 496, 16775, 236761, 568, 3932, 997, 236771,
                236768, 106, 107, 105, 4368, 107,
            ]
        );
    }
}
