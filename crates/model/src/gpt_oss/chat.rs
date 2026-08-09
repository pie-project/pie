//! GPT-OSS instruct implementation.
//!
//! Uses channel-based formatting with analysis/final channels.
//! Reasoning uses the `analysis` channel, not XML tags.

use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

// The implementation below mirrors the published gpt-oss (Harmony) jinja
// chat template; the verbatim copy that used to sit here as a static was
// never read — the checkpoint's own `chat_template` is the reference.

pub struct GptOssInstruct {
    tokenizer: Arc<Tokenizer>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_final_prefix: Vec<u32>,
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
            developer_prefix: make_prefix("developer"),
            user_prefix: make_prefix("user"),
            assistant_final_prefix: make_assistant_prefix("final"),
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
            } else if spec
                .get("nullable")
                .and_then(|n| n.as_bool())
                .unwrap_or(false)
            {
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
                    let opt = if required.contains(&name.as_str()) {
                        ""
                    } else {
                        "?"
                    };
                    lines.push(format!(
                        "{}{}: {}",
                        name,
                        opt,
                        render_typescript_type(pspec)
                    ));
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
                let name = func
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");
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
                        let opt = if required.contains(&pname.as_str()) {
                            ""
                        } else {
                            "?"
                        };
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
        let json_value = serde_json::to_string(value).unwrap_or_else(|_| format!("\"{}\"", value));
        let mut tokens = self.tokenizer.encode(&header);
        tokens.extend(self.tokenizer.encode(&json_value));
        tokens.extend(&self.end_token);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
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
    use crate::instruct::{ChatEvent, ReasoningEvent, ToolEvent};
    use serde_json::json;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn gptoss() -> GptOssInstruct {
        let tok = make_tok(&[
            "<|start|>",
            "<|message|>",
            "<|channel|>",
            "<|end|>",
            "<|endoftext|>",
            "<|return|>",
            "<|call|>",
            "system",
            "developer",
            "user",
            "assistant",
            "analysis",
            "final",
            "\n",
            "Hello",
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
        assert_eq!(
            &sys[..inst.developer_prefix.len()],
            &inst.developer_prefix[..]
        );
    }

    #[test]
    fn user_starts_with_user_prefix() {
        let inst = gptoss();
        let usr = inst.user("Hello");
        assert!(!usr.is_empty());
        assert_eq!(&usr[..inst.user_prefix.len()], &inst.user_prefix[..]);
    }

    /// Every branch of the JSON-Schema-to-TypeScript renderer.
    ///
    /// It is reached only through [`GptOssInstruct::equip`], and a tool
    /// schema is the one input to a chat template that a caller supplies
    /// verbatim -- so an arm that renders the wrong string is a wrong
    /// tool description handed to the model, not a crash.
    #[test]
    fn schema_types_render_as_typescript() {
        let ts = |v: serde_json::Value| render_typescript_type(&v);
        assert_eq!(ts(json!({"type": "string"})), "string");
        assert_eq!(
            ts(json!({"type": "string", "enum": ["a", "b"]})),
            "\"a\" | \"b\""
        );
        assert_eq!(
            ts(json!({"type": "string", "nullable": true})),
            "string | null"
        );
        assert_eq!(ts(json!({"type": "number"})), "number");
        assert_eq!(ts(json!({"type": "integer"})), "number");
        assert_eq!(ts(json!({"type": "boolean"})), "boolean");
        assert_eq!(
            ts(json!({"type": "array", "items": {"type": "boolean"}})),
            "boolean[]"
        );
        assert_eq!(ts(json!({"type": "array"})), "any[]");
        assert_eq!(ts(json!({"type": "object"})), "object");
        assert_eq!(
            ts(json!({"oneOf": [{"type": "string"}, {"type": "number"}]})),
            "string | number"
        );
        assert_eq!(ts(json!({})), "any");
        // A property not named in `required` is optional, which is the
        // one place the renderer changes a type's MEANING rather than
        // its spelling.
        assert_eq!(
            ts(json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
                "required": ["a"],
            })),
            "{\na: string,\nb?: number\n}"
        );
    }

    /// A nested schema recurses on both the array and the object arm.
    #[test]
    fn schema_types_nest() {
        assert_eq!(
            render_typescript_type(&json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"tag": {"type": "string", "enum": ["x"]}},
                    "required": ["tag"],
                },
            })),
            "{\ntag: \"x\"\n}[]"
        );
    }

    /// No tools is no developer turn, not an empty one.
    #[test]
    fn equipping_nothing_emits_nothing() {
        assert!(gptoss().equip(&[]).is_empty());
    }

    /// The namespace block, rendered into a developer turn.
    #[test]
    fn equip_renders_the_functions_namespace() {
        let inst = gptoss();
        let tool = json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "Which city"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city"],
                },
            },
        });
        let text = inst
            .tokenizer
            .decode(&inst.equip(&[tool.to_string()]), false);
        assert_eq!(
            text,
            "<|start|>developer<|message|> # Tools\n\n ## functions\n\n\
             namespace functions {\n\n\
             // Get the weather\n\
             type get_weather = (_: {\n\
             // Which city\n\
             city: string,\n\
             unit?: \"c\" | \"f\",\n\
             }) => any;\n\n\
             } // namespace functions<|end|>"
        );
    }

    /// A tool with no parameters is a nullary call, not a call taking an
    /// empty object -- the two render differently and mean different
    /// things to the model.
    #[test]
    fn a_tool_without_parameters_is_nullary() {
        let inst = gptoss();
        let text = inst.tokenizer.decode(
            &inst.equip(&[json!({"name": "ping", "description": "d"}).to_string()]),
            false,
        );
        assert!(text.contains("type ping = () => any;"), "{text}");
    }

    /// The schema may be the function object itself, without the
    /// OpenAI-style `{"type":"function","function":{...}}` envelope.
    #[test]
    fn a_bare_function_object_is_unwrapped_the_same_way() {
        let inst = gptoss();
        let bare = json!({"name": "f", "description": "d", "parameters":
            {"properties": {"x": {"type": "number"}}}});
        let wrapped = json!({"function": bare});
        assert_eq!(
            inst.equip(&[bare.to_string()]),
            inst.equip(&[wrapped.to_string()])
        );
    }

    /// An unparseable tool is skipped, and the ones around it are not.
    ///
    /// `equip` takes strings a caller built, so malformed JSON is a
    /// reachable input rather than a hypothetical one.
    #[test]
    fn an_unparseable_tool_does_not_take_the_others_with_it() {
        let inst = gptoss();
        let text = inst.tokenizer.decode(
            &inst.equip(&[
                "{not json".to_string(),
                json!({"name": "kept", "description": ""}).to_string(),
            ]),
            false,
        );
        assert!(text.contains("type kept = "), "{text}");
    }

    /// A tool missing its name or description still renders, because a
    /// caller's omission should not silently drop the tool.
    #[test]
    fn a_nameless_tool_renders_as_unknown() {
        let inst = gptoss();
        let text = inst
            .tokenizer
            .decode(&inst.equip(&[json!({"parameters": {}}).to_string()]), false);
        assert!(text.contains("// \ntype unknown = () => any;"), "{text}");
    }

    /// A tool result comes back on the commentary channel, addressed
    /// from the function to the assistant, with the value JSON-encoded.
    #[test]
    fn answer_returns_on_the_commentary_channel() {
        let inst = gptoss();
        let text = inst
            .tokenizer
            .decode(&inst.answer("get_weather", "12C"), false);
        assert_eq!(
            text,
            "<|start|>functions.get_weather to=assistant\
             <|channel|>commentary<|message|>\"12C\"<|end|>"
        );
    }

    /// The value is escaped as JSON, so a quote in a tool result cannot
    /// end the string early.
    #[test]
    fn answer_escapes_the_value() {
        let inst = gptoss();
        let text = inst.tokenizer.decode(&inst.answer("f", "a\"b"), false);
        assert!(text.ends_with("<|message|>\"a\\\"b\"<|end|>"), "{text}");
    }

    /// The chat decoder ends the turn on a stop token and hands back
    /// everything before it.
    #[test]
    fn the_chat_decoder_stops_on_a_seal_token() {
        let inst = gptoss();
        let mut dec = inst.chat_decoder();
        let hello = inst.tokenizer.encode("Hello");
        match dec.feed(&hello) {
            ChatEvent::Delta(d) => assert_eq!(d, "Hello"),
            other => panic!("expected a delta, got {other:?}"),
        }
        let ret = inst.tokenizer.token_to_id("<|return|>").unwrap();
        match dec.feed(&[ret]) {
            ChatEvent::Done(text) => assert_eq!(text, "Hello"),
            other => panic!("expected done, got {other:?}"),
        }
    }

    /// Reasoning is a CHANNEL, not a tag pair: the block opens on
    /// `<|channel|>analysis<|message|>` and closes on `<|end|>`.
    #[test]
    fn the_reasoning_decoder_reads_the_analysis_channel() {
        let inst = gptoss();
        let mut dec = inst.reasoning_decoder();

        // Before the channel opens there is nothing to report.
        assert!(matches!(
            dec.feed(&inst.tokenizer.encode("Hello")),
            ReasoningEvent::Delta(d) if d.is_empty()
        ));

        assert!(matches!(
            dec.feed(&inst.analysis_prefix_ids),
            ReasoningEvent::Start
        ));
        assert!(matches!(
            dec.feed(&inst.tokenizer.encode("thinking")),
            ReasoningEvent::Delta(d) if d == "thinking"
        ));
        match dec.feed(&inst.end_token) {
            ReasoningEvent::Complete(text) => assert_eq!(text, "thinking"),
            other => panic!("expected complete, got {other:?}"),
        }
    }

    /// gpt-oss calls tools by stopping on `<|call|>`, so there is no
    /// in-band tool decoder -- the noop is the statement that the stop
    /// token carries it.
    #[test]
    fn the_tool_decoder_is_a_noop_because_a_stop_token_does_the_work() {
        let inst = gptoss();
        assert!(
            inst.stop_ids
                .contains(&inst.tokenizer.token_to_id("<|call|>").unwrap())
        );
        let mut dec = inst.tool_decoder();
        assert!(matches!(dec.feed(&[1, 2, 3]), ToolEvent::Start));
        dec.reset();
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
