//! Mistral 3 instruct implementation.
//!
//! Implements Mistral V3 chat template features:
//! - [INST]...[/INST] for user messages
//! - [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system messages
//! - [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS] for tool definitions
//! - [TOOL_CALLS]name[ARGS]args for tool calls
//! - [TOOL_RESULTS]content[/TOOL_RESULTS] for tool outputs
//!
//! Reference: Mistral V3 Jinja chat template.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;
use crate::model::instruct::gemma2::NoopReasoningDecoder;

static TEMPLATE: &str = r#"
{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = 'You are Ministral-3-14B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\nThe current date is {today}.\n\nWhen you\'re not sure about some information or when the user\'s request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don\'t have the information and avoid making up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\nYou cannot read nor transcribe audio files or videos.\n\n# TOOL CALLING INSTRUCTIONS\n\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\n\n1. When the request requires up-to-date information.\n2. When the request requires specific data that you do not have in your knowledge base.\n3. When the request involves actions that you cannot perform without tools.\n\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.' %}
{#- Begin of sequence token. #}
{{- bos_token }}
{#- Handle system prompt if it exists. #}
{#- System prompt supports text content or text chunks. #}
{%- if messages[0]['role'] == 'system' %}
    {{- '[SYSTEM_PROMPT]' -}}
    {%- if messages[0]['content'] is string %}
        {{- messages[0]['content'] -}}
    {%- else %}        
        {%- for block in messages[0]['content'] %}
            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '[/SYSTEM_PROMPT]' -}}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- if default_system_message != '' %}
        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}
    {%- endif %}
{%- endif %}
{#- Tools definition #}
{%- set tools_definition = '' %}
{%- set has_tools = false %}
{%- if tools is defined and tools is not none and tools|length > 0 %}
    {%- set has_tools = true %}
    {%- set tools_definition = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}
    {{- tools_definition }}
{%- endif %}
{#- Checks for alternating user/assistant messages. #}
{%- set ns = namespace(index=0) %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}
        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}
            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}
{#- Handle conversation messages. #}
{%- for message in loop_messages %}
    {#- User messages supports text content or text and image chunks. #}
    {%- if message['role'] == 'user' %}
        {%- if message['content'] is string %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
        {%- elif message['content'] | length > 0 %}
            {{- '[INST]' }}
            {%- if message['content'] | length == 2 %}
                {%- set blocks = message['content'] | sort(attribute='type') %}
            {%- else %}
                {%- set blocks = message['content'] %}
            {%- endif %}
            {%- for block in blocks %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- elif block['type'] in ['image', 'image_url'] %}
                    {{- '[IMG]' }}
                {%- else %}
                    {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}
                {%- endif %}
            {%- endfor %}
            {{- '[/INST]' }}
        {%- else %}
            {{- raise_exception('User message must have a string or a list of chunks in content') }}
        {%- endif %}
    {#- Assistant messages supports text content or text and image chunks. #}
    {%- elif message['role'] == 'assistant' %}
        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}
            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}
        {%- endif %}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- elif message['content'] | length > 0 %}
            {%- for block in message['content'] %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- else %}
                    {{- raise_exception('Only text chunks are supported in assistant message contents.') }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        
        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}
            {%- for tool in message['tool_calls'] %}
                {%- set arguments = tool['function']['arguments'] %}
                {%- if arguments is not string %}
                    {%- set arguments = arguments|tojson|safe %}
                {%- elif arguments == '' %}
                    {%- set arguments = '{}' %}
                {%- endif %}
                {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}
            {%- endfor %}
        {%- endif %}
        {#- End of sequence token for each assistant messages. #}
        {{- eos_token }}
    {#- Tool messages only supports text content. #}
    {%- elif message['role'] == 'tool' %}
        {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}
    {#- Raise exception for unsupported roles. #}
    {%- else %}
        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}
    {%- endif %}
{%- endfor %}
"#;


// =============================================================================
// MistralInstruct
// =============================================================================

pub struct MistralInstruct {
    tokenizer: Arc<Tokenizer>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
    // Delimiters
    inst_start: Vec<u32>,
    inst_end: Vec<u32>,
    sys_start: Vec<u32>,
    sys_end: Vec<u32>,
    tools_start: Vec<u32>,
    tools_end: Vec<u32>,
    tool_calls: Vec<u32>,
    tool_args: Vec<u32>,
    tool_results_start: Vec<u32>,
    tool_results_end: Vec<u32>,
}

impl MistralInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["</s>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        // Safe encoding: ensure special tokens are encoded as themselves, not merged with spaces
        // if the tokenizer deals with them that way.
        let mut inst_start = encode("[INST]");
        inst_start.extend(encode(" "));
        
        let mut inst_end = encode(" ");
        inst_end.extend(encode("[/INST]"));

        Self {
            bos_token: encode("<s>"),
            stop_ids,
            inst_start,
            inst_end,
            sys_start: encode("[SYSTEM_PROMPT]"),
            sys_end: encode("[/SYSTEM_PROMPT]"),
            tools_start: encode("[AVAILABLE_TOOLS]"),
            tools_end: encode("[/AVAILABLE_TOOLS]"),
            tool_calls: encode("[TOOL_CALLS]"),
            tool_args: encode("[ARGS]"),
            tool_results_start: encode("[TOOL_RESULTS]"),
            tool_results_end: encode("[/TOOL_RESULTS]"),
            tokenizer,
        }
    }
}

impl Instruct for MistralInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.sys_start.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.sys_end);
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.inst_start.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.inst_end);
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(msg);
        tokens.extend(&self.stop_ids);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        Vec::new() 
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let json_list = format!("[{}]", tools.join(","));
        let mut tokens = self.tools_start.clone();
        tokens.extend(self.tokenizer.encode(&json_list));
        tokens.extend(&self.tools_end);
        tokens
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut tokens = self.tool_results_start.clone();
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_results_end);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(MistralChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(MistralToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            state: ToolState::Outside,
            stop_ids: self.stop_ids.clone(),
            current_name: None,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<String> {
        if tools.is_empty() {
            return None;
        }
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
            r#"root ::= tool-call+
tool-call ::= "[TOOL_CALLS]" tool-name "[ARGS]" json-object
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

// =============================================================================
// Chat Decoder
// =============================================================================

struct MistralChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for MistralChatDecoder {
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
// Tool Decoder
// =============================================================================

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    InsideName,
    InsideArgs,
}

struct MistralToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    state: ToolState,
    stop_ids: Vec<u32>,
    current_name: Option<String>,
}

impl ToolDecoder for MistralToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);
        
        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        self.accumulated = self.accumulated[pos + "[TOOL_CALLS]".len()..].to_string();
                        self.state = ToolState::InsideName;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        let keep = self.accumulated.len() - 50;
                        self.accumulated = self.accumulated[keep..].to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::InsideName => {
                    if let Some(pos) = self.accumulated.find("[ARGS]") {
                        let name = self.accumulated[..pos].trim().to_string();
                        self.current_name = Some(name);
                        self.accumulated = self.accumulated[pos + "[ARGS]".len()..].to_string();
                        self.state = ToolState::InsideArgs;
                        continue;
                    }
                    return ToolEvent::Start; // Wait for ARGS
                }
                ToolState::InsideArgs => {
                    // Check for next marker
                    let mut end_pos = None;
                    
                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        end_pos = Some(pos);
                    } else if let Some(pos) = self.accumulated.find("</s>") {
                         end_pos = Some(pos);
                    }

                    if let Some(pos) = end_pos {
                        let args = self.accumulated[..pos].trim().to_string();
                        // Keep marker for next iteration/state check logic if needed.
                        self.accumulated = self.accumulated[pos..].to_string(); 
                        
                        if let Some(name) = self.current_name.take() {
                            self.state = ToolState::Outside; 
                            return ToolEvent::Call(name, args);
                        }
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.state = ToolState::Outside;
        self.current_name = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn mistral() -> MistralInstruct {
        let tok = make_tok(&[
            "<s>", "</s>", " ",
            "[INST]", "[/INST]",
            "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
            "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
            "[TOOL_CALLS]", "[ARGS]",
            "[TOOL_RESULTS]", "[/TOOL_RESULTS]",
            "Hello", "world",
            "func", "{", "}", ":", "\"", "arg",
            "name", "f", "Hi", "result", ",", "[", "]",
            "INST", "SYSTEM_PROMPT", "AVAILABLE_TOOLS",
            "TOOL_CALLS", "ARGS", "TOOL_RESULTS",
            r#"[{"name":"f"}]"#, // Add complex token to handle non-splitting mock tokenizer
        ]);
        MistralInstruct::new(tok)
    }

    #[test]
    fn system_format() {
        let inst = mistral();
        let tokens = inst.system("Hi");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[SYSTEM_PROMPT]Hi[/SYSTEM_PROMPT]");
    }

    #[test]
    fn user_format() {
        let inst = mistral();
        let tokens = inst.user("Hi");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[INST] Hi [/INST]");
    }

    #[test]
    fn equip_format() {
        let inst = mistral();
        let tokens = inst.equip(&[r#"{"name":"f"}"#.to_string()]);
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, r#"[AVAILABLE_TOOLS][{"name":"f"}][/AVAILABLE_TOOLS]"#);
    }

    #[test]
    fn answer_format() {
        let inst = mistral();
        let tokens = inst.answer("f", "result");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[TOOL_RESULTS]result[/TOOL_RESULTS]");
    }
    
    #[test]
    fn grammar_generation() {
        let inst = mistral();
        let tools = vec![r#"{"function":{"name":"foo"}}"#.to_string()];
        let g = inst.tool_call_grammar(&tools).unwrap();
        assert!(g.contains("tool-call ::= \"[TOOL_CALLS]\""));
        assert!(g.contains("foo"));
    }
}
