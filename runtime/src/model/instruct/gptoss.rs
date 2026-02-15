//! GPT-OSS instruct implementation.
//!
//! Uses channel-based formatting with analysis/final channels.
//! Reasoning uses the `analysis` channel, not XML tags.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder,
    Instruct,
    ReasoningDecoder,
    ToolDecoder,
};
use crate::model::instruct::decoders::{GenericChatDecoder, ThinkingDecoder, NoopToolDecoder};
use crate::model::tokenizer::Tokenizer;

static TEMPLATE: &str = r#"
{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - "builtin_tools": A list, can contain "browser" and/or "python".
  - "model_identity": A string that optionally describes the model identity.
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
 #}
{#- Tool Definition Rendering ============================================== #}
{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}
    {%- if param_spec.type == "array" -%}
        {%- if param_spec['items'] -%}
            {%- if param_spec['items']['type'] == "string" -%}
                {{- "string[]" }}
            {%- elif param_spec['items']['type'] == "number" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "integer" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "boolean" -%}
                {{- "boolean[]" }}
            {%- else -%}
                {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
                {%- if inner_type == "object | object" or inner_type|length > 50 -%}
                    {{- "any[]" }}
                {%- else -%}
                    {{- inner_type + "[]" }}
                {%- endif -%}
            {%- endif -%}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- else -%}
            {{- "any[]" }}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
        {%- if param_spec.type | length > 1 -%}
            {{- param_spec.type | join(" | ") }}
        {%- else -%}
            {{- param_spec.type[0] }}
        {%- endif -%}
    {%- elif param_spec.oneOf -%}
        {#- Handle oneOf schemas - check for complex unions and fallback to any #}
        {%- set has_object_variants = false -%}
        {%- for variant in param_spec.oneOf -%}
            {%- if variant.type == "object" -%}
                {%- set has_object_variants = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
            {{- "any" }}
        {%- else -%}
            {%- for variant in param_spec.oneOf -%}
                {{- render_typescript_type(variant, required_params) -}}
                {%- if variant.description %}
                    {{- "// " + variant.description }}
                {%- endif -%}
                {%- if variant.default is defined %}
                    {{ "// default: " + variant.default|tojson }}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- " | " }}
                {% endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif param_spec.type == "string" -%}
        {%- if param_spec.enum -%}
            {{- '"' + param_spec.enum|join('" | "') + '"' -}}
        {%- else -%}
            {{- "string" }}
            {%- if param_spec.nullable %}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type == "number" -%}
        {{- "number" }}
    {%- elif param_spec.type == "integer" -%}
        {{- "number" }}
    {%- elif param_spec.type == "boolean" -%}
        {{- "boolean" }}
    {%- elif param_spec.type == "object" -%}
        {%- if param_spec.properties -%}
            {{- "{\n" }}
            {%- for prop_name, prop_spec in param_spec.properties.items() -%}
                {{- prop_name -}}
                {%- if prop_name not in (param_spec.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{ render_typescript_type(prop_spec, param_spec.required or []) }}
                {%- if not loop.last -%}
                    {{-", " }}
                {%- endif -%}
            {%- endfor -%}
            {{- "}" }}
        {%- else -%}
            {{- "object" }}
        {%- endif -%}
    {%- else -%}
        {{- "any" }}
    {%- endif -%}
{%- endmacro -%}
{%- macro render_tool_namespace(namespace_name, tools) -%}
    {{- " ## " + namespace_name + "\n\n" }}
    {{- "namespace " + namespace_name + " {\n\n" }}
    {%- for tool in tools %}
        {%- set tool = tool.function %}
        {{- "// " + tool.description + "\n" }}
        {{- "type "+ tool.name + " = " }}
        {%- if tool.parameters and tool.parameters.properties %}
            {{- "(_: {\n" }}
            {%- for param_name, param_spec in tool.parameters.properties.items() %}
                {%- if param_spec.description %}
                    {{- "// " + param_spec.description + "\n" }}
                {%- endif %}
                {{- param_name }}
                {%- if param_name not in (tool.parameters.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
                {%- if param_spec.default is defined -%}
                    {%- if param_spec.enum %}
                        {{- ", // default: " + param_spec.default }}
                    {%- elif param_spec.oneOf %}
                        {{- "// default: " + param_spec.default }}
                    {%- else %}
                        {{- ", // default: " + param_spec.default|tojson }}
                    {%- endif -%}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- ",\n" }}
                {%- else %}
                    {{- ",\n" }}
                {%- endif -%}
            {%- endfor %}
            {{- "}) => any;\n\n" }}
        {%- else -%}
            {{- "() => any;\n\n" }}
        {%- endif -%}
    {%- endfor %}
    {{- "} // namespace " + namespace_name }}
{%- endmacro -%}
{%- macro render_builtin_tools(browser_tool, python_tool) -%}
    {%- if browser_tool %}
        {{- " ## browser\n\n" }}
        {{- "// Tool for browsing.\n" }}
        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\n" }}
        {{- "// Cite information from the tool using the following format:\n" }}
        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\n" }}
        {{- "// Do not quote more than 10 words directly from the tool output.\n" }}
        {{- "// sources=web (default: web)\n" }}
        {{- "namespace browser {\n\n" }}
        {{- "// Searches for information related to `query` and displays `topn` results.\n" }}
        {{- "type search = (_: {\n" }}
        {{- "query: string,\n" }}
        {{- "topn?: number, // default: 10\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\n" }}
        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\n" }}
        {{- "// If `cursor` is not provided, the most recent page is implied.\n" }}
        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\n" }}
        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\n" }}
        {{- "// Use this function without `id` to scroll to a new location of an opened page.\n" }}
        {{- "type open = (_: {\n" }}
        {{- "id?: number | string, // default: -1\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "loc?: number, // default: -1\n" }}
        {{- "num_lines?: number, // default: -1\n" }}
        {{- "view_source?: boolean, // default: false\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\n" }}
        {{- "type find = (_: {\n" }}
        {{- "pattern: string,\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "}) => any;\n\n" }}
        {{- "} // namespace browser\n\n" }}
    {%- endif -%}
    {%- if python_tool %}
        {{- " ## python\n\n" }}
        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\n" }}
        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\n\n" }}
    {%- endif -%}
{%- endmacro -%}
{#- System Message Construction ============================================ #}
{%- macro build_system_message() -%}
    {%- if model_identity is not defined %}
        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
    {%- endif %}
    {{- model_identity + "\n" }}
    {{- "Knowledge cutoff: 2024-06\n" }}
    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}
    {%- if reasoning_effort is not defined %}
        {%- set reasoning_effort = "medium" %}
    {%- endif %}
    {{- "Reasoning: " + reasoning_effort + "\n\n" }}
    {%- if builtin_tools %}
        {{- " # Tools\n\n" }}
        {%- set available_builtin_tools = namespace(browser=false, python=false) %}
        {%- for tool in builtin_tools %}
            {%- if tool == "browser" %}
                {%- set available_builtin_tools.browser = true %}
            {%- elif tool == "python" %}
                {%- set available_builtin_tools.python = true %}
            {%- endif %}
        {%- endfor %}
        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}
    {%- endif -%}
    {{- " # Valid channels: analysis, commentary, final. Channel must be included for every message." }}
    {%- if tools -%}
        {{- "\nCalls to these tools must go to the commentary channel: 'functions'." }}
    {%- endif -%}
{%- endmacro -%}
{#- Main Template Logic ================================================= #}
{#- Set defaults #}
{#- Render system message #}
{{- "<|start|>system<|message|>" }}
{{- build_system_message() }}
{{- "<|end|>" }}
{#- Extract developer message #}
{%- if messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}
{#- Render developer message #}
{%- if developer_message or tools %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- " # Instructions\n\n" }}
        {{- developer_message }}
        {{- "\n\n" }}
    {%- endif %}
    {%- if tools -%}
        {{- " # Tools\n\n" }}
        {{- render_tool_namespace("functions", tools) }}
    {%- endif -%}
    {{- "<|end|>" }}
{%- endif %}
{#- Render messages #}
{%- set last_tool_call = namespace(name=none) %}
{%- for message in loop_messages -%}
    {#- At this point only assistant/user/tool messages should remain #}
    {%- if message.role == 'assistant' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "thinking" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "tool_calls" in message %}
            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}
            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}
            {#- when we render CoT/analysis messages in inference. #}
            {%- set future_final_message = namespace(found=false) %}
            {%- for future_message in loop_messages[loop.index:] %}
                {%- if future_message.role == 'assistant' and "tool_calls" not in future_message %}
                    {%- set future_final_message.found = true %}
                {%- endif %}
            {%- endfor %}
            {#- We assume max 1 tool call per message, and so we infer the tool call name #}
            {#- in "tool" messages from the most recent assistant tool call name #}
            {%- set tool_call = message.tool_calls[0] %}
            {%- if tool_call.function %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {%- if message.content and message.thinking %}
                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}
            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- elif message.thinking and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant to=" }}
            {{- "functions." + tool_call.name + "<|channel|>commentary " }}
            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}
            {{- tool_call.arguments|tojson }}
            {{- "<|call|>" }}
            {%- set last_tool_call.name = tool_call.name %}
        {%- elif loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
        {%- else %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- endif %}
    {%- elif message.role == 'tool' -%}
        {%- if last_tool_call.name is none %}
            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
        {%- endif %}
        {{- "<|start|>functions." + last_tool_call.name }}
        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
    {%- elif message.role == 'user' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}
{#- Generation prompt #}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}
"#;


pub struct GptOssInstruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_final_prefix: Vec<u32>,
    assistant_analysis_prefix: Vec<u32>,
    end_token: Vec<u32>,
    stop_ids: Vec<u32>,
    // Channel tokens for reasoning decoder
    analysis_prefix_ids: Vec<u32>,
    // Generation prompt
    generation_prefix: Vec<u32>,
}

impl GptOssInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<|endoftext|>", "<|return|>", "<|call|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let start = encode("<|start|>");
        let message = encode("<|message|>");
        let channel = encode("<|channel|>");
        let end_token = encode("<|end|>");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = start.clone();
            v.extend(encode(role));
            v.extend(&message);
            v
        };

        let make_assistant_prefix = |chan: &str| -> Vec<u32> {
            let mut v = start.clone();
            v.extend(encode("assistant"));
            v.extend(&channel);
            v.extend(encode(chan));
            v.extend(&message);
            v
        };

        let mut analysis_prefix = channel.clone();
        analysis_prefix.extend(encode("analysis"));
        analysis_prefix.extend(&message);

        let mut generation_prefix = start.clone();
        generation_prefix.extend(encode("assistant"));

        Self {
            system_prefix: make_prefix("system"),
            developer_prefix: make_prefix("developer"),
            user_prefix: make_prefix("user"),
            assistant_final_prefix: make_assistant_prefix("final"),
            assistant_analysis_prefix: make_assistant_prefix("analysis"),
            end_token,
            stop_ids,
            analysis_prefix_ids: analysis_prefix,
            generation_prefix,
            tokenizer,
        }
    }

    fn wrap(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.end_token);
        tokens
    }
}

/// Render a JSON Schema type as a TypeScript-like type string (simplified).
fn render_typescript_type(spec: &serde_json::Value) -> String {
    let type_str = spec.get("type").and_then(|t| t.as_str());
    match type_str {
        Some("string") => {
            if let Some(enums) = spec.get("enum").and_then(|e| e.as_array()) {
                let parts: Vec<String> = enums
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| format!("\"{}\"", s))
                    .collect();
                parts.join(" | ")
            } else if spec.get("nullable").and_then(|n| n.as_bool()).unwrap_or(false) {
                "string | null".to_string()
            } else {
                "string".to_string()
            }
        }
        Some("number") | Some("integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("array") => {
            if let Some(items) = spec.get("items") {
                format!("{}[]", render_typescript_type(items))
            } else {
                "any[]".to_string()
            }
        }
        Some("object") => {
            if let Some(props) = spec.get("properties").and_then(|p| p.as_object()) {
                let required: Vec<&str> = spec
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                    .unwrap_or_default();
                let mut lines = Vec::new();
                for (name, pspec) in props {
                    let opt = if required.contains(&name.as_str()) { "" } else { "?" };
                    lines.push(format!("{}{}: {}", name, opt, render_typescript_type(pspec)));
                }
                format!("{{\n{}\n}}", lines.join(",\n"))
            } else {
                "object".to_string()
            }
        }
        _ => {
            if let Some(one_of) = spec.get("oneOf").and_then(|o| o.as_array()) {
                let types: Vec<String> = one_of.iter().map(render_typescript_type).collect();
                types.join(" | ")
            } else {
                "any".to_string()
            }
        }
    }
}

impl Instruct for GptOssInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // GPT-OSS uses developer role for system-like messages
        self.wrap(&self.developer_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.assistant_final_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        // Reference: tools rendered as TypeScript namespace in developer message.
        // Note: if system() is also called, they produce separate developer turns;
        // the reference template merges them into one. This is a per-message API limitation.
        let mut prompt = String::from(" # Tools\n\n");
        prompt.push_str(" ## functions\n\n");
        prompt.push_str("namespace functions {\n\n");
        for tool_json in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool_json) {
                let func = parsed.get("function").unwrap_or(&parsed);
                let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                let desc = func
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                prompt.push_str(&format!("// {}\n", desc));
                prompt.push_str(&format!("type {} = ", name));
                if let Some(props) = func
                    .get("parameters")
                    .and_then(|p| p.get("properties"))
                    .and_then(|p| p.as_object())
                {
                    let required: Vec<&str> = func
                        .get("parameters")
                        .and_then(|p| p.get("required"))
                        .and_then(|r| r.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();
                    prompt.push_str("(_: {\n");
                    for (pname, pspec) in props {
                        if let Some(pdesc) = pspec.get("description").and_then(|d| d.as_str()) {
                            prompt.push_str(&format!("// {}\n", pdesc));
                        }
                        let opt = if required.contains(&pname.as_str()) { "" } else { "?" };
                        let ptype = render_typescript_type(pspec);
                        prompt.push_str(&format!("{}{}: {},\n", pname, opt, ptype));
                    }
                    prompt.push_str("}) => any;\n\n");
                } else {
                    prompt.push_str("() => any;\n\n");
                }
            }
        }
        prompt.push_str("} // namespace functions");
        self.wrap(&self.developer_prefix, &prompt)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        // Reference: <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{content|tojson}<|end|>
        let header = format!(
            "<|start|>functions.{} to=assistant<|channel|>commentary<|message|>",
            name
        );
        let json_value =
            serde_json::to_string(value).unwrap_or_else(|_| format!("\"{}\"", value));
        let mut tokens = self.tokenizer.encode(&header);
        tokens.extend(self.tokenizer.encode(&json_value));
        tokens.extend(&self.end_token);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(self.tokenizer.clone(), self.stop_ids.clone()))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        // GPT-OSS reasoning uses channel-based detection:
        // start = <|channel|>analysis<|message|>, end = <|end|>
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.analysis_prefix_ids.clone(),
            self.end_token.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        // GPT-OSS uses <|call|> stop token for tool calling, no in-band tool decoder
        Box::new(NoopToolDecoder)
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

    fn gptoss() -> GptOssInstruct {
        let tok = make_tok(&[
            "<|start|>", "<|message|>", "<|channel|>", "<|end|>",
            "<|endoftext|>", "<|return|>", "<|call|>",
            "system", "developer", "user", "assistant", "analysis", "final",
            "\n", "Hello",
        ]);
        GptOssInstruct::new(tok)
    }

    #[test]
    fn has_correct_stop_tokens() {
        let inst = gptoss();
        let stop = inst.seal();
        assert!(stop.contains(&4)); // <|endoftext|>
    }

    #[test]
    fn system_uses_developer_prefix() {
        let inst = gptoss();
        let sys = inst.system("Hello");
        assert!(!sys.is_empty());
        assert_eq!(&sys[..inst.developer_prefix.len()], &inst.developer_prefix[..]);
    }

    #[test]
    fn user_starts_with_user_prefix() {
        let inst = gptoss();
        let usr = inst.user("Hello");
        assert!(!usr.is_empty());
        assert_eq!(&usr[..inst.user_prefix.len()], &inst.user_prefix[..]);
    }

    #[test]
    fn full_conversation() {
        let inst = gptoss();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|start|>developer<|message|>Hello<|end|>\
             <|start|>user<|message|>Hello<|end|>\
             <|start|>assistant<|channel|>final<|message|>Hello<|end|>\
             <|start|>user<|message|>Hello<|end|>\
             <|start|>assistant"
        );
    }
}
