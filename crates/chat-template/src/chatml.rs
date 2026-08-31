//! ChatML: `<|im_start|>{role}\n{message}<|im_end|>\n`.
//!
//! The format Qwen and GLM both write. They had a file each, identical down to
//! the grammar's whitespace, differing only in which tokens end a turn — so
//! what differed is a [`ChatML`] value and what did not is this module.

use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special, specials,
};

/// What one ChatML-speaking model does differently from another.
pub struct ChatML {
    /// The model emits `<think>…</think>`.
    pub thinking: bool,
    /// Replayed assistant turns KEEP their `<think>` blocks.
    ///
    /// The knob upstream calls `preserve_thinking`, and the fact that split it
    /// out of [`thinking`](ChatML::thinking): Qwen3.5/3.6 and GLM strip a past
    /// turn's reasoning before replaying it (their templates' default), while
    /// Qwen3.8 flips the default and replays it whole — its interleaved-
    /// thinking convention, read off the two `chat_template.jinja`s one line
    /// apart (`preserve_thinking is defined and … is true` became
    /// `preserve_thinking is undefined or … is true`). Meaningless when
    /// `thinking` is false: a model that emits no `<think>` has nothing to
    /// strip or keep.
    pub preserve_thinking: bool,
    /// The model was trained on the `<tool_call>` grammar below.
    pub tools: bool,
    /// Text appended to the assistant header when cueing generation.
    pub generation_suffix: &'static str,
    /// Every token that ends a turn.
    pub stop_tokens: &'static [&'static str],
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

pub struct ChatMLInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatML,

    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,

    think_open: Vec<u32>,
    think_close: u32,

    tool_response_prefix: Vec<u32>,
    tool_response_suffix: Vec<u32>,
}

impl ChatMLInstruct {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatML) -> Self {
        let stop_ids = specials(&tokenizer, config.stop_tokens);

        let im_start = special(&tokenizer, "<|im_start|>");
        let im_end = special(&tokenizer, "<|im_end|>");
        let newline = tokenizer.encode("\n");

        let header = |role: &str| -> Vec<u32> {
            let mut tokens = vec![im_start];
            tokens.extend(tokenizer.encode(role));
            tokens.extend(&newline);
            tokens
        };

        let mut turn_suffix = vec![im_end];
        turn_suffix.extend(&newline);

        let mut tool_response_prefix = tokenizer.encode("<tool_response>");
        tool_response_prefix.extend(&newline);
        let mut tool_response_suffix = newline.clone();
        tool_response_suffix.extend(tokenizer.encode("</tool_response>"));

        let mut generation_header = header("assistant");
        generation_header.extend(tokenizer.encode(config.generation_suffix));

        Self {
            system_prefix: header("system"),
            user_prefix: header("user"),
            assistant_prefix: header("assistant"),
            generation_header,
            turn_suffix,
            stop_ids,
            think_open: vec![special(&tokenizer, THINK_OPEN)],
            think_close: special(&tokenizer, THINK_CLOSE),
            tool_response_prefix,
            tool_response_suffix,
            tokenizer,
            config,
        }
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn without_thinking(msg: &str) -> &str {
        match msg.rfind(THINK_CLOSE) {
            Some(at) => msg[at + THINK_CLOSE.len()..].trim_start_matches('\n'),
            None => msg,
        }
    }

    /// What a replayed assistant turn says: its whole message, or the message
    /// with its reasoning stripped — the one decision
    /// [`preserve_thinking`](ChatML::preserve_thinking) exists to state.
    fn replay_body<'a>(config: &ChatML, msg: &'a str) -> &'a str {
        if config.thinking && !config.preserve_thinking {
            Self::without_thinking(msg)
        } else {
            msg
        }
    }

    fn tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            " # Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>",
        );
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(tool);
        }
        prompt.push_str(
            "\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
             </tool_call>",
        );
        prompt
    }
}

/// The names a tool list declares, as grammar alternatives, or `None` when the
/// list declares none this reader can find.
pub fn tool_names(tools: &[String]) -> Option<String> {
    let mut names: Vec<String> = Vec::new();
    for tool in tools {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
            let name = parsed
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| parsed.get("name"))
                .and_then(|name| name.as_str());
            if let Some(name) = name {
                names.push(format!("\"{name}\""));
            }
        }
    }
    if names.is_empty() {
        return None;
    }
    Some(names.join(" | "))
}

/// The JSON value production every tool grammar in this crate ends with.
pub const JSON_GRAMMAR: &str = r#"json-object ::= "{" json-members? "}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#;

impl Instruct for ChatMLInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.system_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(
            &self.assistant_prefix,
            Self::replay_body(&self.config, msg),
        )
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if !self.config.tools {
            return Vec::new();
        }
        self.system(&Self::tool_system_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        if !self.config.tools {
            return Vec::new();
        }
        let mut tokens = self.user_prefix.clone();
        tokens.extend(&self.tool_response_prefix);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_response_suffix);
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if !self.config.thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        if !self.config.tools {
            return Box::new(NoopToolDecoder);
        }
        Box::new(ChatMLToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.tools || tools.is_empty() {
            return None;
        }
        let alternatives = tool_names(tools)?;
        let source = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}}"
tool-name ::= {alternatives}
{JSON_GRAMMAR}"#
        );
        Some(ToolGrammar { source })
    }
}

/// Reads `<tool_call>{json}</tool_call>` spans out of generated text.
///
/// One clock: the tokens become text and everything after is text. The scan
/// runs to the end of what has arrived, so two calls in one batch are two
/// events, and a span whose JSON does not parse is dropped where it stands
/// rather than stopping the scan.
struct ChatMLToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

impl ToolDecoder for ChatMLToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(TOOL_CALL_CLOSE) else {
                    return events;
                };
                let call = self.accumulated[..at].trim().to_string();
                self.accumulated = self.accumulated[at + TOOL_CALL_CLOSE.len()..].to_string();
                self.inside = false;
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&call)
                    && let Some(name) = parsed["name"].as_str()
                {
                    events.push(ToolEvent::Call(
                        name.to_string(),
                        parsed["arguments"].to_string(),
                    ));
                }
            } else {
                let Some(at) = self.accumulated.find(TOOL_CALL_OPEN) else {
                    return events;
                };
                self.accumulated = self.accumulated[at + TOOL_CALL_OPEN.len()..].to_string();
                self.inside = true;
                events.push(ToolEvent::Start);
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TURN: &str = "<think>\nlet me count\n</think>\n\nfour";

    fn config(thinking: bool, preserve_thinking: bool) -> ChatML {
        ChatML {
            thinking,
            preserve_thinking,
            tools: true,
            generation_suffix: "",
            stop_tokens: &["<|im_end|>"],
        }
    }

    /// Qwen3.5/3.6 and GLM: a replayed turn is its answer alone.
    #[test]
    fn a_stripping_model_replays_the_answer_without_its_reasoning() {
        assert_eq!(
            ChatMLInstruct::replay_body(&config(true, false), TURN),
            "four"
        );
    }

    /// Qwen3.8's interleaved-thinking default: the turn is replayed whole,
    /// `<think>` block and all.
    #[test]
    fn a_preserving_model_replays_the_turn_whole() {
        assert_eq!(ChatMLInstruct::replay_body(&config(true, true), TURN), TURN);
    }

    /// A model that emits no `<think>` has nothing to strip or keep — even a
    /// message that happens to contain the marker is replayed verbatim.
    #[test]
    fn a_non_thinking_model_replays_verbatim() {
        assert_eq!(ChatMLInstruct::replay_body(&config(false, false), TURN), TURN);
    }

    /// The strip keeps only what follows the LAST `</think>`, which is
    /// upstream's own `split('</think>')[-1].lstrip('\n')`.
    #[test]
    fn the_strip_reads_past_the_last_close_marker() {
        let twice = "<think>a</think>\ninterim<think>b</think>\n\ndone";
        assert_eq!(ChatMLInstruct::replay_body(&config(true, false), twice), "done");
        assert_eq!(
            ChatMLInstruct::replay_body(&config(true, false), "no reasoning here"),
            "no reasoning here"
        );
    }
}
