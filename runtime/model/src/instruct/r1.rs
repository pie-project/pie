//! DeepSeek R1 instruct implementation.
//!
//! Uses fullwidth Unicode delimiters for roles and tool calls.
//! Reference: DeepSeek R1 Jinja chat template with tool-calling support.

use crate::instruct::decoders::{GenericChatDecoder, ThinkingDecoder};
use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use pie_grammar::grammar::Grammar;
use pie_tokenizer::Tokenizer;
use std::sync::Arc;

static TEMPLATE: &str = r#"
{%- if not add_generation_prompt is defined %}
    {%- set add_generation_prompt = false %}
{%- endif %}
{%- set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if ns.is_first_sp %}
            {%- set ns.system_prompt = ns.system_prompt + message['content'] %}
            {%- set ns.is_first_sp = false %}
        {%- else %}
            {%- set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if tools is defined and tools is not none %}
    {%- set tool_ns = namespace(text='You are a helpful assistant with tool calling capabilities. ' + 'When a tool call is needed, you MUST use the following format to issue the call:\n' + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n' + '```json\n{"param1": "value1", "param2": "value2"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n' + 'Make sure the JSON is valid.' + '## Tools\n\n### Function\n\nYou have the following functions available:\n\n') %}
    {%- for tool in tools %}
        {%- set tool_ns.text = tool_ns.text + '\n```json\n' + (tool | tojson) + '\n```\n' %}
    {%- endfor %}
    {%- if ns.system_prompt|length != 0 %}
        {%- set ns.system_prompt = ns.system_prompt + '\n\n' + tool_ns.text %}
    {%- else %}
        {%- set ns.system_prompt = tool_ns.text %}
    {%- endif %}
{%- endif %}
{{- bos_token }}
{{- ns.system_prompt }}
{%- set last_index = (messages|length - 1) %}
{%- for message in messages %}
    {%- set content = message['content'] %}
    {%- if message['role'] == 'user' %}
        {%- set ns.is_tool = false -%}
        {%- set ns.is_first = false -%}
        {%- set ns.is_last_user = true -%}
        {%- if loop.index0 == last_index %}
            {{- '<｜User｜>' + content }}
        {%- else %}
            {{- '<｜User｜>' + content + '<｜Assistant｜>'}}
        {%- endif %}
    {%- endif %}
    {%- if message['role'] == 'assistant' %}
        {%- if '</think>' in content %}
            {%- set content = (content.split('</think>')|last) %}
        {%- endif %}
    {%- endif %}
    {%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}
        {%- set ns.is_last_user = false -%}
        {%- if ns.is_tool %}
            {{- '<｜tool▁outputs▁end｜>'}}
        {%- endif %}
        {%- set ns.is_first = false %}
        {%- set ns.is_tool = false -%}
        {%- set ns.is_output_first = true %}
        {%- for tool in message['tool_calls'] %}
            {%- set arguments = tool['function']['arguments'] %}
            {%- if arguments is not string %}
                {%- set arguments = arguments|tojson %}
            {%- endif %}
            {%- if not ns.is_first %}
                {%- if content is none %}
                    {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + arguments + '\n' + '```' + '<｜tool▁call▁end｜>'}}
                }
                {%- else %}
                    {{- content + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + arguments + '\n' + '```' + '<｜tool▁call▁end｜>'}}
                {%- endif %}
                {%- set ns.is_first = true -%}
            {%- else %}
                {{- '\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + arguments + '\n' + '```' + '<｜tool▁call▁end｜>'}}
            {%- endif %}
        {%- endfor %}
        {{- '<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
    {%- endif %}
    {%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none) %}
        {%- set ns.is_last_user = false -%}
        {%- if ns.is_tool %}
            {{- '<｜tool▁outputs▁end｜>' + content + '<｜end▁of▁sentence｜>'}}
            {%- set ns.is_tool = false -%}
        {%- else %}
            {{- content + '<｜end▁of▁sentence｜>'}}
        {%- endif %}
    {%- endif %}
    {%- if message['role'] == 'tool' %}
        {%- set ns.is_last_user = false -%}
        {%- set ns.is_tool = true -%}
        {%- if ns.is_output_first %}
            {{- '<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}
            {%- set ns.is_output_first = false %}
        {%- else %}
            {{- '\n<｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}
        {%- endif %}
    {%- endif %}
{%- endfor -%}
{%- if ns.is_tool %}
    {{- '<｜tool▁outputs▁end｜>'}}
{%- endif %}
{#- if add_generation_prompt and not ns.is_last_user and not ns.is_tool #}
{%- if add_generation_prompt and not ns.is_tool %}
    {{- '<｜Assistant｜>'}}
{%- endif %}"#;

pub struct R1Instruct {
    tokenizer: Arc<Tokenizer>,
    bos_token: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eos_ids: Vec<u32>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool tokens
    tool_calls_begin: Vec<u32>,
    tool_calls_end: Vec<u32>,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    tool_sep: Vec<u32>,
    tool_outputs_begin: Vec<u32>,
    tool_outputs_end: Vec<u32>,
    tool_output_begin: Vec<u32>,
    tool_output_end: Vec<u32>,
}

impl R1Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<｜end▁of▁sentence｜>", "<|EOT|>"];
        let eos_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            bos_token: encode("<｜begin▁of▁sentence｜>"),
            user_prefix: encode("<｜User｜>"),
            assistant_prefix: encode("<｜Assistant｜>"),
            eos_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_calls_begin: encode("<｜tool▁calls▁begin｜>"),
            tool_calls_end: encode("<｜tool▁calls▁end｜>"),
            tool_call_begin: encode("<｜tool▁call▁begin｜>"),
            tool_call_end: encode("<｜tool▁call▁end｜>"),
            tool_sep: encode("<｜tool▁sep｜>"),
            tool_outputs_begin: encode("<｜tool▁outputs▁begin｜>"),
            tool_outputs_end: encode("<｜tool▁outputs▁end｜>"),
            tool_output_begin: encode("<｜tool▁output▁begin｜>"),
            tool_output_end: encode("<｜tool▁output▁end｜>"),
            tokenizer,
        }
    }

    /// Strips `<think>...</think>` content from an assistant message for replay,
    /// keeping only the content after the last `</think>`.
    fn strip_thinking(msg: &str) -> &str {
        if let Some(pos) = msg.rfind("</think>") {
            &msg[pos + "</think>".len()..]
        } else {
            msg
        }
    }

    /// Build the R1 tool system prompt from tool JSON schemas.
    fn build_tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "You are a helpful assistant with tool calling capabilities. \
             When a tool call is needed, you MUST use the following format to issue the call:\n\
             <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n\
             ```json\n\
             {\"param1\": \"value1\", \"param2\": \"value2\"}\n\
             ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n\
             Make sure the JSON is valid.\
             ## Tools\n\n\
             ### Function\n\n\
             You have the following functions available:\n\n",
        );
        for tool in tools {
            prompt.push_str("\n```json\n");
            prompt.push_str(tool);
            prompt.push_str("\n```\n");
        }
        prompt
    }
}

impl Instruct for R1Instruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // R1 bare-prepends system text without role wrapping.
        // bos_token + system_prompt (reference: {{- bos_token }}{{ ns.system_prompt }})
        self.tokenizer.encode(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        // Reference: <｜User｜> + content
        // (the <｜Assistant｜> separating user→assistant is emitted by assistant())
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        // Reference: content + <｜end▁of▁sentence｜>
        // Strip <think>...</think> on replay (reference template does this)
        // Prepend <｜Assistant｜> (boundary choice: user() doesn't append it)
        let stripped = Self::strip_thinking(msg);
        let mut tokens = self.assistant_prefix.clone();
        tokens.extend(self.tokenizer.encode(stripped));
        tokens.extend(&self.eos_ids[..1]); // first EOS token
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        // Reference: {{- '<｜Assistant｜>'}} — no <think> prefix
        // The model generates <think> on its own when it decides to reason.
        self.assistant_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        // R1 embeds tool definitions in system prompt with specific format
        let prompt = Self::build_tool_system_prompt(tools);
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        // Reference: <｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>content<｜tool▁output▁end｜>
        //            ... <｜tool▁outputs▁end｜> (emitted on transition to assistant)
        // Note: for multiple consecutive tool outputs, the container delimiters
        // should wrap the group. The per-message API emits them per-call.
        let mut tokens = self.tool_outputs_begin.clone();
        tokens.extend(&self.tool_output_begin);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_output_end);
        tokens.extend(&self.tool_outputs_end);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.eos_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_prefix_ids.clone(),
            self.think_suffix_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(R1ToolDecoder {
            tokenizer: self.tokenizer.clone(),
            tool_call_begin: self.tool_call_begin.clone(),
            tool_call_end: self.tool_call_end.clone(),
            accumulated: String::new(),
            inside: false,
            match_pos: 0,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if tools.is_empty() {
            return None;
        }
        // Build an EBNF grammar that constrains generation to valid R1 tool calls.
        // Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
        //         ```json\n{...}\n```<｜tool▁call▁end｜>[more calls]<｜tool▁calls▁end｜>
        //
        // Extract function names from tool JSON schemas for the name alternation.
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                if let Some(name) = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                {
                    names.push(format!("\"{}\"", name));
                }
            }
        }
        if names.is_empty() {
            return None;
        }

        let name_alt = names.join(" | ");
        let grammar = format!(
            r#"root ::= "<｜tool▁calls▁begin｜>" tool-call+ "<｜tool▁calls▁end｜>"
tool-call ::= "<｜tool▁call▁begin｜>" "function" "<｜tool▁sep｜>" tool-name "\n```json\n" json-object "\n```" "<｜tool▁call▁end｜>"
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
        let parsed = Grammar::from_ebnf(&grammar, "root").ok()?;
        Some(ToolGrammar {
            source: grammar,
            grammar: Arc::new(parsed),
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct R1ToolDecoder {
    tokenizer: Arc<Tokenizer>,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    accumulated: String,
    inside: bool,
    match_pos: usize,
}

impl ToolDecoder for R1ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        if !self.inside {
            // Match tool_call_begin token sequence
            for &t in tokens {
                if self.match_pos < self.tool_call_begin.len()
                    && t == self.tool_call_begin[self.match_pos]
                {
                    self.match_pos += 1;
                    if self.match_pos == self.tool_call_begin.len() {
                        self.inside = true;
                        self.match_pos = 0;
                        self.accumulated.clear();
                        return ToolEvent::Start;
                    }
                } else {
                    self.match_pos = 0;
                }
            }
        } else {
            // Match tool_call_end token sequence
            for &t in tokens {
                if self.match_pos < self.tool_call_end.len()
                    && t == self.tool_call_end[self.match_pos]
                {
                    self.match_pos += 1;
                    if self.match_pos == self.tool_call_end.len() {
                        self.inside = false;
                        self.match_pos = 0;
                        // Parse: type<tool_sep>name\n```json\nargs\n```
                        let content = std::mem::take(&mut self.accumulated);
                        // Extract function name and args from R1 format
                        if let Some(sep_pos) = content.find("\n```json\n") {
                            let header = &content[..sep_pos];
                            let json_start = sep_pos + "\n```json\n".len();
                            if let Some(json_end) = content[json_start..].find("\n```") {
                                let args = content[json_start..json_start + json_end].to_string();
                                // Extract name from "type<sep>name" format
                                let name = header
                                    .rsplit_once("\n")
                                    .map_or(header, |(_, n)| n)
                                    .to_string();
                                return ToolEvent::Call(name, args);
                            }
                        }
                    }
                } else {
                    self.match_pos = 0;
                }
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.inside = false;
        self.match_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn r1() -> R1Instruct {
        let tok = make_tok(&[
            "<｜end▁of▁sentence｜>",
            "<|EOT|>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "Hello",
            "\n",
            "<think>",
            "</think>",
            "<｜begin▁of▁sentence｜>",
            "<｜tool▁outputs▁begin｜>",
            "<｜tool▁outputs▁end｜>",
            "<｜tool▁output▁begin｜>",
            "<｜tool▁output▁end｜>",
        ]);
        R1Instruct::new(tok)
    }

    #[test]
    fn has_correct_stop_tokens() {
        let inst = r1();
        let stop = inst.seal();
        assert!(stop.contains(&0)); // <｜end▁of▁sentence｜>
    }

    #[test]
    fn user_starts_with_prefix() {
        let inst = r1();
        let tokens = inst.user("Hello");
        assert_eq!(tokens[..inst.user_prefix.len()], inst.user_prefix[..]);
    }

    #[test]
    fn system_is_bare_text() {
        let inst = r1();
        let sys = inst.system("Hello");
        // Bare prepend: should NOT start with user or assistant prefix
        if !inst.user_prefix.is_empty() {
            assert_ne!(
                &sys[..inst.user_prefix.len().min(sys.len())],
                &inst.user_prefix[..inst.user_prefix.len().min(sys.len())]
            );
        }
    }

    #[test]
    fn cue_is_assistant_prefix_only() {
        let inst = r1();
        let cue = inst.cue();
        // cue should be exactly <｜Assistant｜> — no <think> prefix
        assert_eq!(cue, inst.assistant_prefix);
    }

    #[test]
    fn assistant_strips_thinking() {
        assert_eq!(R1Instruct::strip_thinking("some text"), "some text");
        assert_eq!(
            R1Instruct::strip_thinking("<think>reasoning</think>actual answer"),
            "actual answer"
        );
        assert_eq!(
            R1Instruct::strip_thinking("<think>a</think><think>b</think>final"),
            "final"
        );
    }

    #[test]
    fn equip_has_reference_format() {
        let prompt = R1Instruct::build_tool_system_prompt(&["{}".to_string()]);
        assert!(prompt.contains("You are a helpful assistant with tool calling capabilities"));
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("### Function"));
        assert!(prompt.contains("```json"));
    }

    #[test]
    fn tool_call_grammar_returns_ebnf() {
        let inst = r1();
        let tools = vec![r#"{"function":{"name":"get_weather","parameters":{}}}"#.to_string()];
        let grammar = inst.tool_call_grammar(&tools);
        assert!(grammar.is_some());
        let g = grammar.unwrap();
        assert!(g.source.contains("root"));
        assert!(g.source.contains("get_weather"));
    }

    #[test]
    fn tool_call_grammar_none_for_empty() {
        let inst = r1();
        assert!(inst.tool_call_grammar(&[]).is_none());
    }

    #[test]
    fn full_conversation() {
        let inst = r1();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "Hello\
             <｜User｜>Hello\
             <｜Assistant｜>Hello<｜end▁of▁sentence｜>\
             <｜User｜>Hello\
             <｜Assistant｜>"
        );
    }

    #[test]
    fn answer_format() {
        let inst = r1();
        let tokens = inst.answer("fn", "Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<｜tool▁outputs▁begin｜>\
             <｜tool▁output▁begin｜>Hello\
             <｜tool▁output▁end｜>\
             <｜tool▁outputs▁end｜>"
        );
    }
}
