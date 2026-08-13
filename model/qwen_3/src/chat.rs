//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use pie_model_common::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use pie_model_common::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use pie_tokenizer::{Tokenizer, TokenizerDecoder};
use std::sync::Arc;

// =============================================================================
// Configuration
// =============================================================================

// The implementation below mirrors the published Qwen3 jinja chat template;
// the verbatim copy that used to sit here as a static was never read — the
// checkpoint's own `chat_template` is the reference.

/// Which tool-call surface a checkpoint's own `chat_template` teaches.
///
/// This is not a preference. The template is what the checkpoint was trained
/// and evaluated against, so a generation whose template demonstrates the XML
/// form will emit XML however it is prompted; constraining it to the other form
/// masks it into a protocol it never learned, and decoding the other form
/// silently yields no call at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCallFormat {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` — Qwen3.
    Json,
    /// `<tool_call>\n<function=NAME>\n<parameter=P>\nv\n</parameter>\n</function>\n</tool_call>`
    /// — Qwen3.5 and later, including the `qwen3_5`-architected Qwen3.6.
    Qwen35Xml,
}

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub tool_call_format: ToolCallFormat,
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
    ///
    /// The demonstrated call MUST be the same surface the grammar admits. A
    /// prompt teaching one form while the mask enforces the other puts the
    /// model's instructions and its token constraint in direct conflict, and
    /// the constraint wins silently -- so the model spends the turn being
    /// steered away from what it was just told to do.
    fn build_tool_system_prompt(tools: &[String], format: ToolCallFormat) -> String {
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
        prompt.push_str("\n</tools>\n\n");
        prompt.push_str(match format {
            ToolCallFormat::Json => {
                "For each function call, return a json object with function name and arguments \
                 within <tool_call></tool_call> XML tags:\n\
                 <tool_call>\n\
                 {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                 </tool_call>"
            }
            ToolCallFormat::Qwen35Xml => {
                "If you choose to call a function ONLY reply in the following format with NO \
                 suffix:\n\
                 <tool_call>\n\
                 <function=example_function_name>\n\
                 <parameter=example_parameter_1>\n\
                 value_1\n\
                 </parameter>\n\
                 </function>\n\
                 </tool_call>"
            }
        });
        prompt
    }

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    fn build_tool_call_grammar(
        tools: &[String],
        format: ToolCallFormat,
        has_thinking: bool,
    ) -> Option<String> {
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
        let tool_grammar = match format {
            ToolCallFormat::Json => format!(
                r#"tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
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
            ),
            ToolCallFormat::Qwen35Xml => format!(
                r#"tool-call ::= "<tool_call>\n<function=" tool-name ">\n" parameter* "</function>\n</tool_call>"
tool-name ::= {name_alt}
parameter ::= "<parameter=" parameter-name ">\n" parameter-value "\n</parameter>\n"
parameter-name ::= [A-Za-z_][A-Za-z0-9_-]*
parameter-value ::= parameter-char*
parameter-char ::= [^<]
"#,
                name_alt = name_alt
            ),
        };
        // A thinking model reaches its action THROUGH deliberation. A root that
        // admits only the call masks the reasoning block out of existence, so
        // the turn cannot contain a thought or a word of plan -- which is not a
        // constraint on the tool syntax at all, it is a constraint on the model
        // being itself. Reasoning syntax stays here in the model formatter;
        // inferlets request the native matcher and remain family-agnostic.
        let root = if has_thinking {
            r#"root ::= reasoning-block? tool-call ("\n" tool-call)*
reasoning-block ::= "<think>" reasoning-content "</think>" "\n"*
reasoning-content ::= reasoning-piece*
reasoning-piece ::= [^<] | "<" [^/] | "</" [^t] | "</t" [^h] | "</th" [^i] | "</thi" [^n] | "</thin" [^k] | "</think" [^>]
"#
        } else {
            "root ::= tool-call (\"\\n\" tool-call)*\n"
        };
        Some(format!("{root}{tool_grammar}"))
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
        let prompt = Self::build_tool_system_prompt(tools, self.config.tool_call_format);
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
            format: self.config.tool_call_format,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        let source = Self::build_tool_call_grammar(
            tools,
            self.config.tool_call_format,
            self.config.has_thinking,
        )?;
        Some(ToolGrammar { source })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

struct QwenToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    has_tools: bool,
    format: ToolCallFormat,
}

impl QwenToolDecoder {
    fn parse_json_tool_call(call: &str) -> Option<(String, String)> {
        let value = serde_json::from_str::<serde_json::Value>(call).ok()?;
        let name = value.get("name")?.as_str()?.to_string();
        if name.is_empty() {
            return None;
        }
        Some((name, value["arguments"].to_string()))
    }

    /// Parses the surface the Qwen3.5+ `chat_template` demonstrates:
    /// `<function=NAME>` wrapping `<parameter=KEY>value</parameter>` pairs.
    ///
    /// Every value arrives as a JSON string. The surface carries no types, so
    /// inferring them here would be this decoder guessing at a schema it cannot
    /// see; the tool boundary validates against the real one.
    fn parse_xml_tool_call(call: &str) -> Option<(String, String)> {
        let call = call.trim();
        let function_prefix = "<function=";
        let function_start = call.find(function_prefix)? + function_prefix.len();
        let function_name_end = call[function_start..].find('>')? + function_start;
        let name = call[function_start..function_name_end].trim().to_string();
        if name.is_empty() {
            return None;
        }
        let function_body_start = function_name_end + 1;
        let function_close = "</function>";
        let function_body_end =
            call[function_body_start..].find(function_close)? + function_body_start;
        let mut rest = &call[function_body_start..function_body_end];
        let mut args = serde_json::Map::new();

        while let Some(parameter_pos) = rest.find("<parameter=") {
            let name_start = parameter_pos + "<parameter=".len();
            let name_end = rest[name_start..].find('>')? + name_start;
            let param_name = rest[name_start..name_end].trim();
            if param_name.is_empty() {
                return None;
            }
            let value_start = name_end + 1;
            let value_close = "</parameter>";
            let value_end = rest[value_start..].find(value_close)? + value_start;
            let value = rest[value_start..value_end].trim_matches('\n').to_string();
            args.insert(param_name.to_string(), serde_json::Value::String(value));
            rest = &rest[value_end + value_close.len()..];
        }

        Some((name, serde_json::Value::Object(args).to_string()))
    }

    /// Tries the configured surface first and the other one second.
    ///
    /// The fallback is not indecision: a checkpoint's template teaches one form,
    /// but a model prompted with the other -- or replaying a history written in
    /// it -- can emit either, and dropping a call it plainly made is the worse
    /// failure. Silence here is indistinguishable from "the model said nothing",
    /// which is what made the mismatch invisible for so long.
    fn parse_tool_call(&self, call: &str) -> Option<(String, String)> {
        match self.format {
            ToolCallFormat::Json => {
                Self::parse_json_tool_call(call).or_else(|| Self::parse_xml_tool_call(call))
            }
            ToolCallFormat::Qwen35Xml => {
                Self::parse_xml_tool_call(call).or_else(|| Self::parse_json_tool_call(call))
            }
        }
    }
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        if !self.inside {
            if self.accumulated.contains("<tool_call>") {
                self.inside = true;
                if let Some(pos) = self.accumulated.find("<tool_call>") {
                    self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                }
                return ToolEvent::Start;
            }
        } else if let Some(pos) = self.accumulated.find("</tool_call>") {
            let call_json = self.accumulated[..pos].trim().to_string();
            self.accumulated = self.accumulated[pos + "</tool_call>".len()..].to_string();
            self.inside = false;
            if let Some((name, args)) = self.parse_tool_call(&call_json) {
                return ToolEvent::Call(name, args);
            }
        }
        ToolEvent::Start
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
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    /// The exact surface `Qwen/Qwen3.6-27B-FP8`'s own `chat_template` tells the
    /// model to emit. Decoding it silently yielded no call at all, which is
    /// indistinguishable from the model having said nothing.
    const QWEN36_TEMPLATE_SURFACE: &str =
        "<function=bash>\n<parameter=cmd>\nls -la\n</parameter>\n</function>";

    #[test]
    fn the_xml_surface_the_checkpoint_teaches_decodes_to_its_call() {
        assert_eq!(
            QwenToolDecoder::parse_xml_tool_call(QWEN36_TEMPLATE_SURFACE),
            Some(("bash".to_string(), r#"{"cmd":"ls -la"}"#.to_string()))
        );
    }

    /// Each format leads with its own surface and still accepts the other, so a
    /// call the model plainly made is never dropped for being in the other one.
    #[test]
    fn either_configured_format_accepts_both_surfaces() {
        let json_body = r#"{"name": "bash", "arguments": {"cmd": "ls -la"}}"#;
        let expected = Some(("bash".to_string(), r#"{"cmd":"ls -la"}"#.to_string()));

        for format in [ToolCallFormat::Json, ToolCallFormat::Qwen35Xml] {
            let decoder = QwenToolDecoder {
                decoder: make_tok().decoder(false),
                accumulated: String::new(),
                inside: false,
                has_tools: true,
                format,
            };

            assert_eq!(decoder.parse_tool_call(QWEN36_TEMPLATE_SURFACE), expected);
            assert_eq!(decoder.parse_tool_call(json_body), expected);
        }
    }

    /// A thinking model reaches its action through deliberation. A root that
    /// admits only the call masks the reasoning block out of existence, so the
    /// turn cannot hold a thought or a word of plan.
    #[test]
    fn a_thinking_model_may_reason_before_it_acts() {
        let tools = [r#"{"name": "bash"}"#.to_string()];

        let thinking =
            QwenInstruct::build_tool_call_grammar(&tools, ToolCallFormat::Qwen35Xml, true)
                .expect("a named tool yields a grammar");
        assert!(thinking.starts_with("root ::= reasoning-block? tool-call"));
        assert!(thinking.contains(r#"tool-call ::= "<tool_call>\n<function=""#));

        // A model with no reasoning channel gains no prefix it cannot fill.
        let plain = QwenInstruct::build_tool_call_grammar(&tools, ToolCallFormat::Json, false)
            .expect("a named tool yields a grammar");
        assert!(plain.starts_with("root ::= tool-call"));
        assert!(plain.contains("tool-json"));
    }

    fn make_tok() -> Arc<Tokenizer> {
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

    fn qwen3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                tool_call_format: ToolCallFormat::Qwen35Xml,
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
                tool_call_format: ToolCallFormat::Qwen35Xml,
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
                tool_call_format: ToolCallFormat::Qwen35Xml,
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
        let prompt = QwenInstruct::build_tool_system_prompt(&["{}".to_string()], ToolCallFormat::Json);
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
                tool_call_format: ToolCallFormat::Qwen35Xml,
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
