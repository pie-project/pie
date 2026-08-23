use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct GptOssInstruct {
    tokenizer: Arc<Tokenizer>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_final_prefix: Vec<u32>,
    end_token: Vec<u32>,
    stop_ids: Vec<u32>,

    analysis_prefix_ids: Vec<u32>,

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
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.analysis_prefix_ids.clone(),
            self.end_token.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
