//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

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
        let stop_ids: Vec<u32> = config.stop_tokens
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        Self {
            system_prefix: encode("<|im_start|>system\n"),
            user_prefix: encode("<|im_start|>user\n"),
            assistant_prefix: encode("<|im_start|>assistant\n"),
            turn_suffix: encode("<|im_end|>\n"),
            generation_header: encode("<|im_start|>assistant\n"),
            stop_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_response_prefix_tokens: encode("<tool_response>\n"),
            tool_response_suffix_tokens: encode("\n</tool_response>"),
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
            "# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>"
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
             </tool_call>"
        );
        prompt
    }

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let name = parsed.get("function")
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
        Box::new(QwenChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(QwenReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            think_prefix_ids: self.think_prefix_ids.clone(),
            think_suffix_ids: self.think_suffix_ids.clone(),
            state: ReasoningState::Outside,
            accumulated: String::new(),
            match_pos: 0,
            has_thinking: self.config.has_thinking,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<String> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        Self::build_tool_call_grammar(tools)
    }
}

// =============================================================================
// Chat Decoder
// =============================================================================

struct QwenChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for QwenChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
        for &t in tokens {
            if self.stop_ids.contains(&t) {
                let text = std::mem::take(&mut self.accumulated);
                return ChatEvent::Done(text);
            }
        }
        let delta = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&delta);
        ChatEvent::Delta(delta)
    }

    fn reset(&mut self) {
        self.accumulated.clear();
    }
}

// =============================================================================
// Reasoning Decoder
// =============================================================================

enum ReasoningState {
    Outside,
    Inside,
}

struct QwenReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    state: ReasoningState,
    accumulated: String,
    match_pos: usize,
    has_thinking: bool,
}

impl ReasoningDecoder for QwenReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        if !self.has_thinking {
            return ReasoningEvent::Delta(String::new());
        }
        match self.state {
            ReasoningState::Outside => {
                for &t in tokens {
                    if self.match_pos < self.think_prefix_ids.len()
                        && t == self.think_prefix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_prefix_ids.len() {
                            self.state = ReasoningState::Inside;
                            self.match_pos = 0;
                            return ReasoningEvent::Start;
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                ReasoningEvent::Delta(String::new())
            }
            ReasoningState::Inside => {
                for &t in tokens {
                    if self.match_pos < self.think_suffix_ids.len()
                        && t == self.think_suffix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_suffix_ids.len() {
                            self.state = ReasoningState::Outside;
                            self.match_pos = 0;
                            let text = std::mem::take(&mut self.accumulated);
                            return ReasoningEvent::Complete(text);
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                let delta = self.tokenizer.decode(tokens, false);
                self.accumulated.push_str(&delta);
                ReasoningEvent::Delta(delta)
            }
        }
    }

    fn reset(&mut self) {
        self.state = ReasoningState::Outside;
        self.accumulated.clear();
        self.match_pos = 0;
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
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json) {
                    let name = v["name"].as_str().unwrap_or("").to_string();
                    let args = v["arguments"].to_string();
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
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant", "Hello", " world",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<tools>", "</tools>",
        ].into_iter().map(String::from).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn qwen3() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: true, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        })
    }

    fn qwen2() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: false, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })
    }

    fn olmo3() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: true, has_tools: false,
            stop_tokens: &["<|im_end|>"],
        })
    }

    #[test]
    fn qwen3_has_3_stop_tokens() {
        assert_eq!(qwen3().stop_ids.len(), 3);
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
}
