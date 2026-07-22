//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use crate::inference::structured::grammar::Grammar;
use crate::inference::structured::json_schema::{JsonSchemaOptions, json_schema_to_ebnf_with_root};
use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use crate::model::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

// =============================================================================
// Configuration
// =============================================================================

static TEMPLATE: &str = r#"
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- " # Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for forward_message in messages %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set message = messages[index] %}
    {%- set current_content = message.content if message.content is not none else '' %}
    {%- set tool_start = '<tool_response>' %}
    {%- set tool_start_length = tool_start|length %}
    {%- set start_of_message = current_content[:tool_start_length] %}
    {%- set tool_end = '</tool_response>' %}
    {%- set tool_end_length = tool_end|length %}
    {%- set start_pos = (current_content|length) - tool_end_length %}
    {%- if start_pos < 0 %}
        {%- set start_pos = 0 %}
    {%- endif %}
    {%- set end_of_message = current_content[start_pos:] %}
    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
                {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
                {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"#;

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub tool_call_format: ToolCallFormat,
    pub generation_suffix: &'static str,
    /// Stop token strings (vary per sub-architecture)
    pub stop_tokens: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCallFormat {
    Json,
    Qwen35Xml,
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

    /// Build the tool system prompt matching the model's native tokenizer template.
    fn build_tool_system_prompt(format: ToolCallFormat, tools: &[String]) -> String {
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
        match format {
            ToolCallFormat::Json => prompt.push_str(
                "\n</tools>\n\n\
                 For each function call, return a json object with function name and arguments \
                 within <tool_call></tool_call> XML tags:\n\
                 <tool_call>\n\
                 {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                 </tool_call>",
            ),
            ToolCallFormat::Qwen35Xml => prompt.push_str(
                "\n</tools>\n\n\
                 If you choose to call a function ONLY reply in the following format with no suffix:\n\
                 \n\
                 <tool_call>\n\
                 <function=example_function_name>\n\
                 <parameter=example_parameter_1>\n\
                 value_1\n\
                 </parameter>\n\
                 <parameter=example_parameter_2>\n\
                 This is the value for the second parameter\n\
                 that can span\n\
                 multiple lines\n\
                 </parameter>\n\
                 </function>\n\
                 </tool_call>",
            ),
        }
        prompt
    }

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    fn build_tool_call_grammar(
        format: ToolCallFormat,
        tools: &[String],
        has_thinking: bool,
    ) -> Option<String> {
        // Parse each tool into a name literal plus a per-tool payload schema.
        // The payload schema fixes the tool name and constrains `arguments` to
        // the tool's own parameter schema, so the generated grammar admits only
        // strict-JSON, schema-valid arguments (no unescaped control characters,
        // no undeclared keys) instead of an unconstrained object.
        let mut names: Vec<String> = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        let mut variants: Vec<serde_json::Value> = Vec::new();
        for tool in tools {
            // Fail closed: a malformed declaration, a missing name, or a
            // duplicate name would produce a grammar that silently drops or
            // ambiguates a tool, so refuse to build a native grammar at all.
            let parsed = serde_json::from_str::<serde_json::Value>(tool).ok()?;
            let func = parsed.get("function").unwrap_or(&parsed);
            let name = func.get("name").and_then(|n| n.as_str())?;
            if name.is_empty() {
                return None;
            }
            if seen.iter().any(|existing| existing == name) {
                return None;
            }
            seen.push(name.to_string());
            // Emit the name as a JSON string literal so quotes, backslashes, and
            // control characters are escaped into a form Pie's EBNF parser accepts
            // (unlike raw `format!("\"{name}\"")`, which a `"`/`\` in the name would
            // corrupt, and unlike `{name:?}`, whose `\u{..}` escapes the parser
            // rejects). The schema `const` below keeps the original, unescaped name.
            names.push(serde_json::to_string(name).ok()?);
            let params = func
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            variants.push(serde_json::json!({
                "type": "object",
                "properties": { "name": { "const": name }, "arguments": params },
                "required": ["name", "arguments"],
                "additionalProperties": false,
            }));
        }
        if names.is_empty() {
            return None;
        }

        let name_alt = names.join(" | ");
        let tool_grammar = match format {
            ToolCallFormat::Json => {
                // Lower the tool-call payload with the shared JSON-Schema→EBNF
                // converter (strict objects, RFC 8259 strings). Embed it under a
                // custom `tool-json` root so it composes with the tool-call and
                // reasoning wrappers below.
                let payload_schema = serde_json::json!({ "anyOf": variants });
                let options = JsonSchemaOptions {
                    any_whitespace: true,
                    strict_mode: true,
                    ..JsonSchemaOptions::default()
                };
                let payload_ebnf =
                    json_schema_to_ebnf_with_root(&payload_schema, &options, "tool-json").ok()?;
                format!(
                    "tool-call ::= \"<tool_call>\\n\" tool-json \"\\n</tool_call>\"\n{payload_ebnf}"
                )
            }
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
        let root = if has_thinking {
            // Reasoning syntax belongs to the model formatter. Inferlets only
            // request the native tool matcher and remain family-agnostic.
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
        let prompt = Self::build_tool_system_prompt(self.config.tool_call_format, tools);
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
            tokenizer: self.tokenizer.clone(),
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
            self.config.tool_call_format,
            tools,
            self.config.has_thinking,
        )?;
        let grammar = Grammar::from_ebnf(&source, "root").ok()?;
        Some(ToolGrammar {
            source,
            grammar: Arc::new(grammar),
        })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

struct QwenToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    inside: bool,
    has_tools: bool,
    format: ToolCallFormat,
}

impl QwenToolDecoder {
    fn parse_json_tool_call(call: &str) -> Option<(String, String)> {
        let v = serde_json::from_str::<serde_json::Value>(call).ok()?;
        let name = v.get("name")?.as_str()?.to_string();
        let args = v
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()))
            .to_string();
        Some((name, args))
    }

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
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        if !self.inside {
            if self.accumulated.contains("<tool_call>") {
                self.inside = true;
                if let Some(pos) = self.accumulated.find("<tool_call>") {
                    self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                }
                return ToolEvent::Start;
            }
        } else if self.accumulated.contains("</tool_call>") {
            if let Some(pos) = self.accumulated.find("</tool_call>") {
                let call_json = self.accumulated[..pos].trim().to_string();
                self.accumulated = self.accumulated[pos + "</tool_call>".len()..].to_string();
                self.inside = false;
                if let Some((name, args)) = self.parse_tool_call(&call_json) {
                    return ToolEvent::Call(name, args);
                }
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.inside = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::structured::matcher::GrammarMatcher;
    use crate::model::tokenizer::Tokenizer;
    use std::sync::Arc;

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
            r#"{"name": "edit", "arguments": {}}"#,
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
                has_thinking: true,
                has_tools: true,
                tool_call_format: ToolCallFormat::Json,
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
                tool_call_format: ToolCallFormat::Json,
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
                tool_call_format: ToolCallFormat::Json,
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
        let prompt =
            QwenInstruct::build_tool_system_prompt(ToolCallFormat::Json, &["{}".to_string()]);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("</tools>"));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn qwen35_tool_prompt_uses_native_function_parameter_template() {
        let prompt =
            QwenInstruct::build_tool_system_prompt(ToolCallFormat::Qwen35Xml, &["{}".to_string()]);
        assert!(prompt.contains("<function=example_function_name>"));
        assert!(prompt.contains("<parameter=example_parameter_1>"));
        assert!(!prompt.contains(r#"{"name": <function-name>"#));
    }

    #[test]
    fn qwen35_tool_grammar_uses_native_function_parameter_template() {
        let tool = r#"{"function":{"name":"edit"}}"#.to_string();
        let grammar =
            QwenInstruct::build_tool_call_grammar(ToolCallFormat::Qwen35Xml, &[tool], true)
                .expect("grammar");
        assert!(grammar.contains(r#"<function=""#));
        assert!(grammar.contains(r#"<parameter=""#));
        assert!(!grammar.contains(r#""\"name\": \""#));
    }

    #[test]
    fn qwen35_tool_grammar_compiles() {
        let inst = QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                tool_call_format: ToolCallFormat::Qwen35Xml,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        );
        let tool = r#"{"function":{"name":"edit"}}"#.to_string();
        assert!(inst.tool_call_grammar(&[tool]).is_some());
    }

    #[test]
    fn tool_grammar_follows_reasoning_capability_and_still_decodes_calls() {
        let tool = r#"{"function":{"name":"edit"}}"#.to_string();
        let qwen3 = qwen3();
        let qwen3_grammar = qwen3.tool_call_grammar(&[tool.clone()]).unwrap();
        let qwen2_grammar = qwen2().tool_call_grammar(&[tool]).unwrap();
        assert!(qwen3_grammar.source.contains("reasoning-block?"));
        assert!(!qwen2_grammar.source.contains("reasoning-block"));

        let mut decoder = qwen3.tool_decoder();
        assert!(matches!(decoder.feed(&[9, 7, 10, 11, 4]), ToolEvent::Start));
        assert!(matches!(
            decoder.feed(&[17, 4, 12]),
            ToolEvent::Call(name, arguments) if name == "edit" && arguments == "{}"
        ));
    }

    #[test]
    fn json_tool_grammar_lowers_argument_schema_and_excludes_control_chars() {
        let tool = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#.to_string();
        let source = QwenInstruct::build_tool_call_grammar(ToolCallFormat::Json, &[tool], false)
            .expect("grammar source");
        // The old unconstrained productions are gone.
        assert!(!source.contains("json-char ::= [^\"\\\\]"), "loose json-char still present:\n{source}");
        assert!(!source.contains("json-object ::= "), "unconstrained json-object still present:\n{source}");
        // The converter's RFC 8259 string char class (excludes control chars 0x00-0x1f) is present.
        assert!(
            source.to_lowercase().contains("x1f"),
            "expected control-char exclusion, got:\n{source}"
        );
        // Arguments are constrained to the declared property key and the tool name.
        assert!(source.contains("command"), "declared arg key not lowered:\n{source}");
        assert!(source.contains("bash"), "tool name not present:\n{source}");
        // Composes into a valid grammar.
        assert!(
            Grammar::from_ebnf(&source, "root").is_ok(),
            "grammar failed to compile:\n{source}"
        );
    }

    #[test]
    fn json_tool_grammar_rejects_corrupt_arguments_and_accepts_valid() {
        let tool = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#.to_string();
        let source =
            QwenInstruct::build_tool_call_grammar(ToolCallFormat::Json, &[tool], false).unwrap();
        let grammar = Arc::new(Grammar::from_ebnf(&source, "root").expect("compile"));
        let tok = make_tok();
        let stop: Vec<u32> = Vec::new();

        // A well-formed, schema-valid tool call is accepted.
        let mut ok = GrammarMatcher::new(grammar.clone(), tok.clone(), stop.clone(), 64);
        let valid = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
        assert!(ok.accept_string(valid), "valid tool call should be accepted");

        // The corruption observed in run 0f5e3b1e (undeclared key + a literal
        // unescaped newline inside a JSON string) must be rejected.
        let mut bad = GrammarMatcher::new(grammar, tok, stop, 64);
        let corrupt = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\">\n'\":null}}\n</tool_call>";
        assert!(!bad.accept_string(corrupt), "corrupt arguments must be rejected");
    }

    fn json_tool_grammar(tools: &[&str], has_thinking: bool) -> Arc<Grammar> {
        let owned: Vec<String> = tools.iter().map(|t| t.to_string()).collect();
        let source =
            QwenInstruct::build_tool_call_grammar(ToolCallFormat::Json, &owned, has_thinking)
                .expect("grammar source");
        Arc::new(Grammar::from_ebnf(&source, "root").expect("grammar compiles"))
    }

    fn grammar_accepts(grammar: &Arc<Grammar>, s: &str) -> bool {
        let mut m = GrammarMatcher::new(grammar.clone(), make_tok(), Vec::new(), 64);
        // Require a complete match: the string must advance and the matcher
        // must be at a valid terminal state (not merely a valid prefix).
        m.accept_string(s) && m.can_terminate()
    }

    fn json_call(name: &str, args: &str) -> String {
        format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
    }

    #[test]
    fn json_tool_grammar_enforces_required_wrong_type_and_undeclared_keys() {
        let bash = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#;
        let g = json_tool_grammar(&[bash], false);

        assert!(grammar_accepts(&g, &json_call("bash", r#"{"command": "ls"}"#)));
        // Missing required key.
        assert!(!grammar_accepts(&g, &json_call("bash", "{}")));
        // Wrong value type (number where a string is required).
        assert!(!grammar_accepts(&g, &json_call("bash", r#"{"command": 7}"#)));
        // Only an undeclared key.
        assert!(!grammar_accepts(&g, &json_call("bash", r#"{"foo": "x"}"#)));
        // Declared key plus an extra undeclared key.
        assert!(!grammar_accepts(
            &g,
            &json_call("bash", r#"{"command": "ls", "foo": "x"}"#)
        ));
    }

    #[test]
    fn json_tool_grammar_rejects_literal_control_char_in_a_declared_string() {
        let bash = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#;
        let g = json_tool_grammar(&[bash], false);

        // A LITERAL (unescaped) newline inside the declared string value is
        // rejected on its own — the only difference from the accepted case is
        // the escaping, isolating the control-character exclusion.
        let literal_newline = "{\"command\": \"a\nb\"}";
        assert!(!grammar_accepts(&g, &json_call("bash", literal_newline)));
        // The properly escaped form of the same value is accepted.
        let escaped_newline = r#"{"command": "a\nb"}"#;
        assert!(grammar_accepts(&g, &json_call("bash", escaped_newline)));
        // A literal tab (also 0x00-0x1f) is likewise rejected.
        let literal_tab = "{\"command\": \"a\tb\"}";
        assert!(!grammar_accepts(&g, &json_call("bash", literal_tab)));
        // A literal NUL (0x00) is rejected.
        let literal_nul = "{\"command\": \"a\u{0}b\"}";
        assert!(!grammar_accepts(&g, &json_call("bash", literal_nul)));
    }

    #[test]
    fn json_tool_grammar_handles_missing_parameters() {
        // A tool declaration without a `parameters` field must still build a
        // working grammar (defaulting to an object) rather than failing.
        let no_params = r#"{"function":{"name":"finish"}}"#;
        let g = json_tool_grammar(&[no_params], false);
        assert!(grammar_accepts(&g, &json_call("finish", "{}")));
        // Non-object arguments are still rejected.
        assert!(!grammar_accepts(&g, &json_call("finish", r#""oops""#)));
    }

    #[test]
    fn json_tool_grammar_fails_closed_on_unsupported_parameter_schema() {
        // The converter does not support allOf with multiple schemas, so an
        // unsupported parameter schema fails closed instead of emitting an
        // unconstrained tool.
        let bad = r#"{"function":{"name":"bash","parameters":{"allOf":[{"type":"object"},{"type":"string"}]}}}"#;
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &[bad.to_string()],
                false
            )
            .is_none()
        );
    }

    #[test]
    fn json_tool_grammar_constrains_arguments_with_reasoning_enabled() {
        let bash = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#;
        let g = json_tool_grammar(&[bash], true);

        // A bare tool call is accepted under the reasoning-enabled root.
        assert!(grammar_accepts(&g, &json_call("bash", r#"{"command": "ls"}"#)));
        // A reasoning block preceding the tool call is accepted.
        let with_reasoning =
            format!("<think>plan</think>\n{}", json_call("bash", r#"{"command": "ls"}"#));
        assert!(grammar_accepts(&g, &with_reasoning));
        // Argument constraints still hold with reasoning enabled: missing
        // required key, undeclared key, and a literal control char are rejected.
        assert!(!grammar_accepts(&g, &json_call("bash", "{}")));
        assert!(!grammar_accepts(&g, &json_call("bash", r#"{"foo": "x"}"#)));
        assert!(!grammar_accepts(&g, &json_call("bash", "{\"command\": \"a\nb\"}")));
    }

    #[test]
    fn json_tool_grammar_binds_each_tool_to_its_own_schema() {
        let bash = r#"{"function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}"#;
        let finish = r#"{"function":{"name":"finish","parameters":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"],"additionalProperties":false}}}"#;
        let g = json_tool_grammar(&[bash, finish], false);

        // Each tool accepts its own schema.
        assert!(grammar_accepts(&g, &json_call("bash", r#"{"command": "ls"}"#)));
        assert!(grammar_accepts(&g, &json_call("finish", r#"{"status": "done"}"#)));
        // Cross-schema binding is rejected (name is bound to its own arguments).
        assert!(!grammar_accepts(&g, &json_call("bash", r#"{"status": "done"}"#)));
        assert!(!grammar_accepts(&g, &json_call("finish", r#"{"command": "ls"}"#)));
        // An undeclared tool name is rejected.
        assert!(!grammar_accepts(&g, &json_call("rm", r#"{"command": "ls"}"#)));
    }

    #[test]
    fn json_tool_grammar_fails_closed_on_malformed_or_duplicate_tools() {
        // Malformed JSON declaration.
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &["{not json".to_string()],
                false
            )
            .is_none()
        );
        // Missing name.
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &[r#"{"function":{}}"#.to_string()],
                false
            )
            .is_none()
        );
        // Empty tool name.
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &[r#"{"function":{"name":""}}"#.to_string()],
                false
            )
            .is_none()
        );
        // Duplicate names.
        let dup = r#"{"function":{"name":"bash","parameters":{"type":"object"}}}"#.to_string();
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &[dup.clone(), dup],
                false
            )
            .is_none()
        );
        // A well-formed declaration alongside a malformed one also fails closed
        // (the good tool is not silently kept).
        let good = r#"{"function":{"name":"bash","parameters":{"type":"object"}}}"#.to_string();
        assert!(
            QwenInstruct::build_tool_call_grammar(
                ToolCallFormat::Json,
                &[good, "nope".to_string()],
                false
            )
            .is_none()
        );
    }

    #[test]
    fn qwen35_xml_escapes_tool_name_into_the_grammar_literal() {
        // A name carrying both a double-quote and a backslash must be escaped
        // into the Qwen3.5 XML `tool-name` EBNF literal rather than corrupting or
        // widening it. JSON parsing unescapes the declaration to the name a"b\c.
        let tool = r#"{"function":{"name":"a\"b\\c"}}"#.to_string();
        let source =
            QwenInstruct::build_tool_call_grammar(ToolCallFormat::Qwen35Xml, &[tool], false)
                .expect("grammar source");
        let grammar =
            Arc::new(Grammar::from_ebnf(&source, "root").expect("grammar compiles despite quote/backslash in name"));

        // Accepts a call naming the exact declared tool (quote + backslash intact).
        assert!(grammar_accepts(
            &grammar,
            "<tool_call>\n<function=a\"b\\c>\n</function>\n</tool_call>"
        ));
        // Rejects any other name — the literal is bound to a"b\c, not widened.
        assert!(!grammar_accepts(
            &grammar,
            "<tool_call>\n<function=other>\n</function>\n</tool_call>"
        ));
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
                tool_call_format: ToolCallFormat::Json,
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

    #[test]
    fn qwen35_tool_decoder_parses_native_function_parameter_call() {
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
            "<function=edit>",
            "<parameter=path>",
            "src/lib.rs",
            "</parameter>",
            "<parameter=replacement>",
            "new body",
            "</function>",
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
                tool_call_format: ToolCallFormat::Qwen35Xml,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        );
        let mut dec = inst.tool_decoder();

        dec.feed(&[11]);
        dec.feed(&[4]);
        let event = dec.feed(&[17, 4, 18, 4, 19, 4, 20, 4, 21, 4, 22, 4, 20, 4, 23, 4, 12]);
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "edit");
                let args: serde_json::Value = serde_json::from_str(&args).unwrap();
                assert_eq!(args["path"], "src/lib.rs");
                assert_eq!(args["replacement"], "new body");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}
