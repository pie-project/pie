//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

// =============================================================================
// Configuration
// =============================================================================

// The implementation below mirrors the published Qwen3 jinja chat template;
// the verbatim copy that used to sit here as a static was never read — the
// checkpoint's own `chat_template` is the reference.

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub generation_suffix: &'static str,
    /// Stop token strings (vary per sub-architecture)
    pub stop_tokens: &'static [&'static str],
}

// =============================================================================
// QwenInstruct
// =============================================================================

pub struct QwenInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatMLConfig,
    // Pre-tokenized delimiters
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    // Thinking delimiters
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool delimiters
    tool_response_prefix_tokens: Vec<u32>,
    tool_response_suffix_tokens: Vec<u32>,
}

impl QwenInstruct {
    /// Create with full config.
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatMLConfig) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = config
            .stop_tokens
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = im_start.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = im_end;
        turn_suffix.extend(&newline);

        let think_prefix = encode("<think>");
        let think_suffix = encode("</think>");

        let mut tool_resp_prefix = encode("<tool_response>");
        tool_resp_prefix.extend(&newline);
        let mut tool_resp_suffix = newline.clone();
        tool_resp_suffix.extend(encode("</tool_response>"));

        let mut generation_header = make_prefix("assistant");
        generation_header.extend(encode(config.generation_suffix));

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            generation_header,
            turn_suffix,
            stop_ids,
            think_prefix_ids: think_prefix,
            think_suffix_ids: think_suffix,
            tool_response_prefix_tokens: tool_resp_prefix,
            tool_response_suffix_tokens: tool_resp_suffix,
            tokenizer,
            config,
        }
    }

    fn role_tokens(&self, role: &str, msg: &str) -> Vec<u32> {
        let prefix = match role {
            "system" => &self.system_prefix,
            "user" => &self.user_prefix,
            "assistant" => &self.assistant_prefix,
            _ => &self.user_prefix,
        };
        let mut tokens = prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// Strips `<think>...</think>` content from an assistant message for replay.
    /// If `</think>` is present, keeps only the content after the last `</think>`,
    /// with leading newlines stripped (matching the reference template).
    fn strip_thinking(msg: &str) -> &str {
        if let Some(pos) = msg.rfind("</think>") {
            msg[pos + "</think>".len()..].trim_start_matches('\n')
        } else {
            msg
        }
    }

    /// Build the tool system prompt matching the Qwen reference format.
    /// Both Qwen3 and Qwen2.5 use identical `<tools>` XML + `<tool_call>` format.
    fn build_tool_system_prompt(tools: &[String]) -> String {
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

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let name = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| parsed.get("name"))
                    .and_then(|n| n.as_str());
                if let Some(n) = name {
                    names.push(format!("\"{}\"", n));
                }
            }
        }
        if names.is_empty() {
            return None;
        }

        let name_alt = names.join(" | ");
        let grammar = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}}"
tool-name ::= {name_alt}
json-object ::= "{{" json-members? "}}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#,
            name_alt = name_alt
        );
        Some(grammar)
    }
}

impl Instruct for QwenInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        // Strip <think>...</think> on replay (Qwen3 template does this;
        // for Qwen2 has_thinking=false so strip_thinking is a no-op on normal content)
        let stripped = if self.config.has_thinking {
            Self::strip_thinking(msg)
        } else {
            msg
        };
        self.role_tokens("assistant", stripped)
    }

    fn cue(&self) -> Vec<u32> {
        // Reference: <|im_start|>assistant\n
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        let prompt = Self::build_tool_system_prompt(tools);
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        // Reference: tool responses go in a user turn with <tool_response> wrapper
        // Format: <|im_start|>user\n<tool_response>\ncontent\n</tool_response><|im_end|>\n
        let mut tokens = self.user_prefix.clone();
        tokens.extend(&self.tool_response_prefix_tokens);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_response_suffix_tokens);
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
        if !self.config.has_thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_prefix_ids.clone(),
            self.think_suffix_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        let source = Self::build_tool_call_grammar(tools)?;
        Some(ToolGrammar { source })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

/// The markers the prompt asks for, the grammar constrains generation to,
/// and the decoder reads back. Named once so the three cannot drift.
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

struct QwenToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    has_tools: bool,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        // Looping matters: a caller is free to hand over a whole
        // generation at once, and a block that both opens and closes
        // inside a single feed has to resolve within it.
        loop {
            if !self.inside {
                let Some(pos) = self.accumulated.find(TOOL_CALL_OPEN) else {
                    return ToolEvent::Start;
                };
                self.accumulated = self.accumulated[pos + TOOL_CALL_OPEN.len()..].to_string();
                self.inside = true;
                continue;
            }
            let Some(pos) = self.accumulated.find(TOOL_CALL_CLOSE) else {
                return ToolEvent::Start;
            };
            let call_json = self.accumulated[..pos].trim().to_string();
            self.accumulated = self.accumulated[pos + TOOL_CALL_CLOSE.len()..].to_string();
            self.inside = false;
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json)
                && let Some(name) = v["name"].as_str()
            {
                // A `Call` with an empty name is one no dispatcher can
                // route, so a block naming no function reports nothing.
                let args = v["arguments"].to_string();
                return ToolEvent::Call(name.to_string(), args);
            }
            return ToolEvent::Start;
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}

// ── The named templates ──────────────────────────────────────────────
//
// A catalog row NAMES its template instead of re-spelling a struct
// literal, which is what `instruct::create`'s arms did — nine of them,
// six identical. Naming them also puts the differences where they can
// be read side by side: nemotron-h opens the assistant turn already
// inside a `<think>`, and GLM adds two role markers to the stop set.

/// Qwen 2.5 / 3 / 3.5 / 3-VL: thinking, tools, no forced prefix.
pub const QWEN_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: true,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};

/// Nemotron-H: ChatML that OPENS the assistant turn inside a `<think>`.
pub const NEMOTRON_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: false,
    generation_suffix: "<think>\n",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};

/// GLM-5.1: ChatML whose stop set also holds the two role markers.
pub const GLM_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: true,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
};

/// Plain ChatML: no thinking, no tools.
///
/// This is the shape `instruct::create`'s `_ =>` arm handed to every
/// architecture it had never heard of. It is still here because some
/// rows genuinely are plain ChatML — but a ROW has to ask for it now,
/// and a row that has not been written cannot ask.
pub const PLAIN_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: false,
    has_tools: false,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    pub(super) fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "Hello",
            " world",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<tools>",
            "</tools>",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    pub(super) fn qwen3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn qwen2() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: false,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn olmo3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>"],
            },
        )
    }

    #[test]
    fn qwen3_has_2_stop_tokens() {
        assert_eq!(qwen3().stop_ids.len(), 2);
    }

    #[test]
    fn qwen2_has_2_stop_tokens() {
        assert_eq!(qwen2().stop_ids.len(), 2);
    }

    #[test]
    fn olmo3_has_1_stop_token() {
        assert_eq!(olmo3().stop_ids.len(), 1);
    }

    #[test]
    fn qwen3_thinking_enabled() {
        assert!(qwen3().config.has_thinking);
    }

    #[test]
    fn qwen2_thinking_disabled() {
        assert!(!qwen2().config.has_thinking);
    }

    #[test]
    fn equip_noop_when_disabled() {
        let inst = olmo3();
        assert!(inst.equip(&["tool".to_string()]).is_empty());
        assert!(inst.answer("fn1", "42").is_empty());
    }

    #[test]
    fn equip_produces_tokens_when_enabled() {
        assert!(qwen3().config.has_tools);
    }

    #[test]
    fn seal_returns_stop_ids() {
        let inst = qwen3();
        assert_eq!(inst.seal(), inst.stop_ids);
    }

    #[test]
    fn generation_header_matches_cue() {
        let inst = qwen3();
        assert_eq!(inst.cue(), inst.generation_header);
    }

    #[test]
    fn strip_thinking_works() {
        assert_eq!(QwenInstruct::strip_thinking("plain text"), "plain text");
        assert_eq!(QwenInstruct::strip_thinking("<think>foo</think>bar"), "bar");
    }

    #[test]
    fn equip_format_matches_reference() {
        let prompt = QwenInstruct::build_tool_system_prompt(&["{}".to_string()]);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("</tools>"));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn answer_does_not_include_name() {
        let inst = qwen3();
        let tokens = inst.answer("get_weather", "sunny");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(!text.contains("get_weather:"));
    }

    #[test]
    fn tool_call_grammar_none_when_disabled() {
        let inst = olmo3();
        assert!(inst.tool_call_grammar(&["{}".to_string()]).is_none());
    }

    #[test]
    fn full_conversation() {
        let inst = qwen3();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>system\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn answer_format() {
        let inst = qwen3();
        let tokens = inst.answer("fn1", "Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>user\n<tool_response>\nHello\n</tool_response><|im_end|>\n"
        );
    }

    #[test]
    fn tool_decoder_parses_call() {
        // Build vocab with the JSON content as a single entry
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "Hello",
            " world",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<tools>",
            "</tools>",
            r#"{"name": "f", "arguments": {}}"#,
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = QwenInstruct::new(
            tok,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        );
        let mut dec = inst.tool_decoder();
        // Feed: <tool_call> \n JSON \n </tool_call>
        dec.feed(&[11]); // <tool_call> → enters inside, returns Start
        dec.feed(&[4]); // \n
        let event = dec.feed(&[17, 4, 12]); // JSON + \n + </tool_call>
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}

#[cfg(test)]
mod tool_tests {
    use super::tests::{make_tok, qwen3};
    use super::*;
    use crate::instruct::{ChatEvent, ReasoningEvent};
    use crate::shared::decoders::gbnf_sentence;

    const WRAPPED: &str = r#"{"type":"function","function":{"name":"get_weather"}}"#;
    const BARE: &str = r#"{"name":"get_time"}"#;

    fn toolless() -> QwenInstruct {
        QwenInstruct::new(make_tok(), PLAIN_CHATML)
    }

    fn stream(inst: &QwenInstruct, d: &mut dyn ToolDecoder, parts: &[&str]) -> Vec<ToolEvent> {
        parts
            .iter()
            .map(|p| d.feed(&inst.tokenizer.encode(p)))
            .collect()
    }

    fn last_call(events: &[ToolEvent]) -> Option<(String, String)> {
        events.iter().rev().find_map(|e| match e {
            ToolEvent::Call(n, a) => Some((n.clone(), a.clone())),
            ToolEvent::Start => None,
        })
    }

    #[test]
    fn the_test_tokenizer_round_trips_the_markers() {
        let inst = qwen3();
        for part in [
            "<tool_call>",
            "</tool_call>",
            r#"{"name":"f","arguments":{}}"#,
        ] {
            let ids = inst.tokenizer.encode(part);
            assert_eq!(inst.tokenizer.decode(&ids, false), part, "part {part:?}");
        }
    }

    // ─── Grammar ─────────────────────────────────────────────

    /// The grammar constrains generation and the decoder reads the result
    /// back; nothing but this test binds them. llama-3 shipped a rule
    /// admitting text its own decoder rejected, so the binding is checked
    /// rather than assumed.
    #[test]
    fn the_grammar_admits_exactly_what_the_decoder_reads() {
        let inst = qwen3();
        let grammar = inst
            .tool_call_grammar(&[WRAPPED.to_string()])
            .expect("a tool was offered");
        let sentence = gbnf_sentence(
            &grammar.source,
            "tool-call",
            &[("json-object", r#"{"city":"Oslo"}"#)],
        );
        assert_eq!(
            sentence,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Oslo\"}}\n</tool_call>"
        );

        let mut decoder = inst.tool_decoder();
        let events = stream(&inst, decoder.as_mut(), &[&sentence]);
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), r#"{"city":"Oslo"}"#.into())),
            "the grammar admits {sentence:?}, which the decoder did not read as that call"
        );
    }

    /// The prompt asks for the markers, the grammar constrains generation
    /// to them, and the decoder reads them back. They were three separate
    /// spellings of the same two strings.
    #[test]
    fn the_prompt_the_grammar_and_the_decoder_name_the_same_markers() {
        let inst = qwen3();
        let grammar = inst
            .tool_call_grammar(&[WRAPPED.to_string()])
            .expect("a tool was offered");
        let prompt = QwenInstruct::build_tool_system_prompt(&[WRAPPED.to_string()]);
        for marker in [TOOL_CALL_OPEN, TOOL_CALL_CLOSE] {
            assert!(grammar.source.contains(marker), "grammar lacks {marker}");
            assert!(prompt.contains(marker), "prompt lacks {marker}");
        }
    }

    #[test]
    fn the_grammar_names_every_tool_in_either_spelling() {
        let inst = qwen3();
        let grammar = inst
            .tool_call_grammar(&[
                WRAPPED.to_string(),
                BARE.to_string(),
                "not json at all".to_string(),
                r#"{"description":"nameless"}"#.to_string(),
            ])
            .expect("two tools are named");
        assert!(
            grammar
                .source
                .contains(r#"tool-name ::= "get_weather" | "get_time""#)
        );
    }

    #[test]
    fn a_grammar_needs_a_tool_that_names_itself() {
        let inst = qwen3();
        assert!(inst.tool_call_grammar(&[]).is_none());
        assert!(inst.tool_call_grammar(&["not json".to_string()]).is_none());
        assert!(
            inst.tool_call_grammar(&[r#"{"description":"x"}"#.to_string()])
                .is_none()
        );
    }

    /// A template without tools must not publish a grammar for them, even
    /// when the caller offers some.
    #[test]
    fn a_toolless_template_publishes_no_grammar() {
        assert!(
            toolless()
                .tool_call_grammar(&[WRAPPED.to_string()])
                .is_none()
        );
    }

    // ─── The prompt ──────────────────────────────────────────

    #[test]
    fn the_tool_prompt_lists_every_signature_between_the_tags() {
        let prompt =
            QwenInstruct::build_tool_system_prompt(&["sig-one".to_string(), "sig-two".to_string()]);
        // The preamble names `<tools></tools>` before the block itself
        // opens, so the block is the LAST pair.
        let inner = prompt
            .rsplit_once("</tools>")
            .and_then(|(head, _)| head.rsplit_once("<tools>"))
            .map(|(_, inner)| inner)
            .expect("the signatures sit between the tags");
        assert_eq!(inner, "\nsig-one\nsig-two\n");
        assert!(prompt.contains("<tool_call></tool_call> XML tags"));
    }

    #[test]
    fn a_toolless_template_equips_nothing_and_answers_nothing() {
        let inst = toolless();
        assert!(inst.equip(&[WRAPPED.to_string()]).is_empty());
        assert!(inst.answer("get_weather", "sunny").is_empty());
    }

    #[test]
    fn a_tool_result_comes_back_as_a_wrapped_user_turn() {
        let inst = qwen3();
        let text = inst
            .tokenizer
            .decode(&inst.answer("get_weather", "Hello"), false);
        assert_eq!(
            text,
            "<|im_start|>user\n<tool_response>\nHello\n</tool_response><|im_end|>\n"
        );
    }

    #[test]
    fn an_unknown_role_is_written_as_a_user_turn() {
        let inst = qwen3();
        assert_eq!(
            inst.role_tokens("environment", "Hello"),
            inst.role_tokens("user", "Hello")
        );
    }

    // ─── The decoder ─────────────────────────────────────────

    #[test]
    fn a_call_may_arrive_across_several_steps() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        let events = stream(
            &inst,
            decoder.as_mut(),
            &[
                "Let me check. ",
                "<tool_call>",
                "\n",
                r#"{"name": "get_weather", "arguments": {"city":"Oslo"}}"#,
                "\n",
                "</tool_call>",
            ],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), r#"{"city":"Oslo"}"#.into()))
        );
        // Opening the block is not yet a call.
        assert!(matches!(events[1], ToolEvent::Start));
    }

    /// Regression: the decoder made at most one state transition per
    /// feed, so a block that opened and closed inside one chunk consumed
    /// its opener, reported `Start`, and dropped the call. The guest
    /// chooses the chunking, so this was reachable from any caller that
    /// batched tokens.
    #[test]
    fn a_whole_block_in_one_feed_resolves_within_it() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        let events = stream(
            &inst,
            decoder.as_mut(),
            &[r#"Sure. <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>"#],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), "{}".into()))
        );
    }

    #[test]
    fn two_blocks_in_a_row_both_resolve() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        let events = stream(
            &inst,
            decoder.as_mut(),
            &[
                "<tool_call>",
                r#"{"name": "first", "arguments": {}}"#,
                "</tool_call>",
                "<tool_call>",
                r#"{"name": "second", "arguments": {}}"#,
                "</tool_call>",
            ],
        );
        let names: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Call(n, _) => Some(n.as_str()),
                ToolEvent::Start => None,
            })
            .collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn a_block_naming_no_function_is_not_a_call() {
        let inst = qwen3();
        for body in [r#"{"arguments": {}}"#, r#"{"name": 7}"#, "not json", "[]"] {
            let mut decoder = inst.tool_decoder();
            let events = stream(
                &inst,
                decoder.as_mut(),
                &["<tool_call>", body, "</tool_call>"],
            );
            assert_eq!(last_call(&events), None, "body {body:?}");
        }
    }

    #[test]
    fn a_call_with_no_arguments_key_reports_null() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        let events = stream(
            &inst,
            decoder.as_mut(),
            &["<tool_call>", BARE, "</tool_call>"],
        );
        assert_eq!(last_call(&events), Some(("get_time".into(), "null".into())));
    }

    #[test]
    fn a_toolless_template_never_reports_a_call() {
        let inst = toolless();
        let mut decoder = inst.tool_decoder();
        let events = stream(
            &inst,
            decoder.as_mut(),
            &[
                "<tool_call>",
                r#"{"name": "get_weather", "arguments": {}}"#,
                "</tool_call>",
            ],
        );
        assert_eq!(last_call(&events), None);
    }

    #[test]
    fn reset_abandons_a_block_in_progress() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        stream(&inst, decoder.as_mut(), &["<tool_call>"]);
        decoder.reset();
        let after = stream(
            &inst,
            decoder.as_mut(),
            &[r#"{"name": "f", "arguments": {}}"#, "</tool_call>"],
        );
        assert_eq!(last_call(&after), None);
    }

    #[test]
    fn reset_forgets_buffered_text() {
        let inst = qwen3();
        let mut decoder = inst.tool_decoder();
        stream(&inst, decoder.as_mut(), &["thinking about it <tool_"]);
        decoder.reset();
        let after = stream(
            &inst,
            decoder.as_mut(),
            &["call>", r#"{"name": "f", "arguments": {}}"#, "</tool_call>"],
        );
        assert_eq!(last_call(&after), None);
    }

    // ─── Reasoning ───────────────────────────────────────────

    /// A template without thinking must not treat a stray `<think>` in
    /// the answer as the start of a reasoning block.
    #[test]
    fn a_template_without_thinking_reports_no_reasoning() {
        let inst = toolless();
        let mut decoder = inst.reasoning_decoder();
        for part in ["<think>", "Hello", "</think>"] {
            assert!(matches!(
                decoder.feed(&inst.tokenizer.encode(part)),
                ReasoningEvent::Delta(t) if t.is_empty()
            ));
        }
        decoder.reset();
    }

    #[test]
    fn reasoning_waits_for_the_opening_tag() {
        let inst = qwen3();
        let mut decoder = inst.reasoning_decoder();
        let hello = inst.tokenizer.encode("Hello");
        assert!(matches!(decoder.feed(&hello), ReasoningEvent::Delta(t) if t.is_empty()));
        assert!(matches!(
            decoder.feed(&inst.tokenizer.encode("<think>")),
            ReasoningEvent::Start
        ));
        assert!(matches!(
            decoder.feed(&inst.tokenizer.encode("</think>")),
            ReasoningEvent::Complete(t) if t.is_empty()
        ));
    }

    #[test]
    fn the_chat_decoder_stops_on_the_turn_end() {
        let inst = qwen3();
        let mut decoder = inst.chat_decoder();
        let hello = inst.tokenizer.encode("Hello");
        assert!(matches!(decoder.feed(&hello), ChatEvent::Delta(t) if t == "Hello"));
        let mut tail = inst.tokenizer.encode(" world");
        tail.extend(inst.tokenizer.encode("<|im_end|>"));
        assert!(matches!(decoder.feed(&tail), ChatEvent::Done(t) if t == "Hello world"));
    }
}
