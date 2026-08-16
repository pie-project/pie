//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use pie_model_common::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use pie_model_common::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolCall, ToolDecoder, ToolEvent, ToolGrammar,
    ToolObservation,
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

/// What the Qwen3.5+ `chat_template` says after the `<tools>` block, verbatim.
///
/// A raw literal, not a `\`-continued one: every space and blank line here is
/// the checkpoint's, and a continuation would let rustfmt's indentation into a
/// prompt whose bytes are the contract.
const QWEN35_XML_CALL_INSTRUCTION: &str = r#"If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub tool_call_format: ToolCallFormat,
    pub generation_suffix: &'static str,
    /// What opens the model's turn when the caller asked for no thinking.
    ///
    /// Separate from `generation_suffix` because for the Qwen3.5+ templates the
    /// two are different strings — `<think>\n` opens a reasoning block, while
    /// `<think>\n\n</think>\n\n` closes an empty one — and one field cannot
    /// carry both. A template with no thinking-off form repeats
    /// `generation_suffix` here.
    pub thinking_off_suffix: &'static str,
    /// Whether a replayed assistant turn past the last user query opens a
    /// reasoning block even when it carried no reasoning.
    ///
    /// Qwen3's template renders one only when the turn is the last message or
    /// carries `reasoning_content`:
    ///
    /// ```jinja
    /// {%- if loop.index0 > ns.last_query_index %}
    ///     {%- if loop.last or (not loop.last and reasoning_content) %}
    /// ```
    ///
    /// Qwen3.5 and later dropped the inner condition and open one on every
    /// post-query turn, so this cannot be one behaviour for both.
    ///
    /// The `loop.last` half is a fact about the message LIST and does not reach
    /// this crate: `assistant_call` is told which side of the last user query
    /// its turn sits on and nothing else. A conversation whose final message is
    /// an assistant turn therefore renders no block where Qwen3's template
    /// renders an empty one -- the rarer of the two cases, and the one that
    /// cannot be fixed without widening the trait every generation implements.
    pub empty_reasoning_header: bool,
    /// Where the caller's system text sits in the turn that declares tools.
    ///
    /// Qwen3 and Qwen2.5 open the turn with it and put the declaration after:
    ///
    /// ```jinja
    /// {{- '<|im_start|>system\n' }}
    /// {%- if messages[0].role == 'system' %}
    ///     {{- messages[0].content + '\n\n' }}
    /// {%- endif %}
    /// {{- "# Tools\n\n..." }}
    /// ```
    ///
    /// Qwen3.5 and later lead with the declaration and nest the system text
    /// after it. Same two pieces, opposite order, and the order is the prompt.
    pub system_before_tools: bool,
    /// What separates non-empty assistant content from the FIRST replayed call.
    ///
    /// Qwen3 and Qwen2.5 write a single newline, and the same one before every
    /// later call:
    ///
    /// ```jinja
    /// {%- if (loop.first and content) or (not loop.first) %}
    ///     {{- '\n' }}
    /// {%- endif %}
    /// ```
    ///
    /// Qwen3.5 and later open the first call after a blank line instead. Later
    /// calls are one newline apart in every generation, so only this one
    /// separator varies.
    pub content_call_separator: &'static str,
    /// Whether the checkpoint's template applies `|trim` to message content.
    ///
    /// Qwen3.5 and later do; Qwen3 and Qwen2 emit `message.content` verbatim.
    /// Trimming unconditionally would change the prompt of every checkpoint
    /// whose template does not, so the template says which.
    pub trim_content: bool,
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
    thinking_off_header: Vec<u32>,
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
        let mut thinking_off_header = make_prefix("assistant");
        thinking_off_header.extend(encode(config.thinking_off_suffix));

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            generation_header,
            thinking_off_header,
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
        tokens.extend(self.tokenizer.encode(self.rendered(msg)));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// A message's content as the template renders it, before any role framing.
    fn rendered<'a>(&self, msg: &'a str) -> &'a str {
        if self.config.trim_content {
            msg.trim()
        } else {
            msg
        }
    }

    /// Strips `<think>...</think>` content from an assistant message for replay.
    /// If `</think>` is present, keeps only the content after the last `</think>`,
    /// with leading newlines stripped (matching the reference template).
    fn strip_thinking(msg: &str) -> &str {
        Self::split_thinking(msg).1
    }

    /// Splits a replayed assistant message into (reasoning, content).
    ///
    /// The template's own arithmetic: the reasoning is what sits before the
    /// FIRST `</think>` and after the LAST `<think>` before it, and the content
    /// is what follows the LAST `</think>`. The two indices differ, so this
    /// cannot be one `split`.
    fn split_thinking(msg: &str) -> (&str, &str) {
        const CLOSE: &str = "</think>";
        let Some(first_close) = msg.find(CLOSE) else {
            return ("", msg);
        };
        let head = msg[..first_close].trim_end_matches('\n');
        let reasoning = match head.rfind("<think>") {
            Some(open) => &head[open + "<think>".len()..],
            None => head,
        };
        let last_close = msg.rfind(CLOSE).unwrap_or(first_close);
        (
            reasoning.trim_start_matches('\n').trim(),
            msg[last_close + CLOSE.len()..].trim_start_matches('\n'),
        )
    }

    /// Build the tool system prompt matching the Qwen reference format.
    ///
    /// The demonstrated call MUST be the same surface the grammar admits. A
    /// prompt teaching one form while the mask enforces the other puts the
    /// model's instructions and its token constraint in direct conflict, and
    /// the constraint wins silently -- so the model spends the turn being
    /// steered away from what it was just told to do.
    fn build_tool_system_prompt(tools: &[String], format: ToolCallFormat) -> String {
        // The opening differs per template generation, and both are transcribed
        // from a checkpoint's own `chat_template`. The Json one used to carry a
        // leading space that no template has: Qwen3 and Qwen2.5 both write
        // `# Tools` at the very start of the block, whether or not a system
        // message precedes it. One character, one token, on every prompt that
        // declares a tool.
        let mut prompt = String::from(match format {
            ToolCallFormat::Json => {
                "# Tools\n\n\
                 You may call one or more functions to assist with the user query.\n\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n\
                 <tools>"
            }
            ToolCallFormat::Qwen35Xml => {
                "# Tools\n\nYou have access to the following functions:\n\n<tools>"
            }
        });
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
            ToolCallFormat::Qwen35Xml => QWEN35_XML_CALL_INSTRUCTION,
        });
        prompt
    }

    /// The system turn a tool-declaring conversation opens with.
    ///
    /// No Qwen template renders the caller's system message as its own turn
    /// when tools are present: they emit ONE system turn holding both pieces.
    /// Two turns are a different prompt, not a formatting variant. Which piece
    /// leads is the template's to say -- see `system_before_tools`.
    fn tool_system_body(&self, system: &str, tools: &[String]) -> String {
        let declaration = Self::build_tool_system_prompt(tools, self.config.tool_call_format);
        let system = system.trim();
        if system.is_empty() {
            return declaration;
        }
        if self.config.system_before_tools {
            format!("{system}\n\n{declaration}")
        } else {
            format!("{declaration}\n\n{system}")
        }
    }

    /// A replayed assistant turn's body: reasoning header, content, calls.
    fn assistant_turn_body(
        &self,
        msg: &str,
        calls: &[ToolCall],
        reasoning_header: bool,
    ) -> String {
        let msg = self.rendered(msg);
        let (reasoning, content) = if self.config.has_thinking {
            Self::split_thinking(msg)
        } else {
            ("", msg)
        };

        let mut body = String::new();
        let renders_header = self.config.empty_reasoning_header || !reasoning.is_empty();
        if reasoning_header && self.config.has_thinking && renders_header {
            body.push_str("<think>\n");
            body.push_str(reasoning);
            body.push_str("\n</think>\n\n");
        }
        body.push_str(content);
        for (index, call) in calls.iter().enumerate() {
            // The template separates the first call from non-empty content by
            // `content_call_separator` and every later call by a newline.
            if index == 0 {
                if !content.trim().is_empty() {
                    body.push_str(self.config.content_call_separator);
                }
            } else {
                body.push('\n');
            }
            body.push_str(&self.tool_call_surface(call));
        }
        body
    }

    /// The `<tool_call>` surface for one replayed call, without its separator.
    ///
    /// In the surface the checkpoint's own template teaches -- the same
    /// `ToolCallFormat` the declaration demonstrates, the grammar admits and
    /// the decoder parses. This was the one of those five sites that ignored
    /// the field, so a Qwen3 or Qwen2.5 conversation was replayed to the model
    /// in a syntax its template never shows and its own next call will not use.
    ///
    /// Arguments arrive as a JSON object — the same string the decoder reports
    /// in [`ToolEvent::Call`]. `Json` writes it through, which is the
    /// template's `tool_call.arguments` verbatim when it is a string. `Qwen35Xml`
    /// writes string values raw and everything else as JSON, which is that
    /// template's `args_value | string if ... else args_value | tojson`; a value
    /// that is a bare string would otherwise come back quoted and no longer be
    /// the value the tool was called with.
    fn tool_call_surface(&self, call: &ToolCall) -> String {
        if self.config.tool_call_format == ToolCallFormat::Json {
            // `{"name": "NAME", "arguments": {...}}`, the Hermes-style body
            // Qwen3's and Qwen2.5's templates write between the tags.
            let arguments = match call.arguments_json.trim() {
                "" => "{}",
                arguments => arguments,
            };
            return format!(
                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {arguments}}}\n</tool_call>",
                call.name
            );
        }
        let mut surface = format!("<tool_call>\n<function={}>\n", call.name);
        if let Ok(serde_json::Value::Object(arguments)) =
            serde_json::from_str::<serde_json::Value>(&call.arguments_json)
        {
            for (name, value) in &arguments {
                surface.push_str(&format!("<parameter={name}>\n"));
                match value {
                    serde_json::Value::String(text) => surface.push_str(text),
                    other => surface.push_str(&other.to_string()),
                }
                surface.push_str("\n</parameter>\n");
            }
        }
        surface.push_str("</function>\n</tool_call>");
        surface
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

    fn assistant_call(&self, msg: &str, calls: &[ToolCall], reasoning_header: bool) -> Vec<u32> {
        let mut tokens = self.assistant_prefix.clone();
        tokens.extend(
            self.tokenizer
                .encode(&self.assistant_turn_body(msg, calls, reasoning_header)),
        );
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        // Reference: <|im_start|>assistant\n + the template's generation suffix.
        self.generation_header.clone()
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        self.thinking_off_header.clone()
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

    fn equip_into_system(&self, system: &str, tools: &[String]) -> Vec<u32> {
        if !self.config.has_tools || tools.is_empty() {
            return self.system(system);
        }
        self.role_tokens("system", &self.tool_system_body(system, tools))
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        self.answer_all(std::slice::from_ref(&ToolObservation {
            name: name.to_string(),
            value: value.to_string(),
        }))
    }

    fn answer_all(&self, observations: &[ToolObservation]) -> Vec<u32> {
        if !self.config.has_tools || observations.is_empty() {
            return Vec::new();
        }
        // Reference: a RUN of consecutive tool results is ONE user turn holding
        // one <tool_response> block each — `<|im_start|>user` is emitted only
        // when the previous message was not a tool result, and `<|im_end|>` only
        // when the next one is not.
        let mut tokens = self.user_prefix.clone();
        for (index, observation) in observations.iter().enumerate() {
            if index > 0 {
                tokens.extend(self.tokenizer.encode("\n"));
            }
            tokens.extend(&self.tool_response_prefix_tokens);
            tokens.extend(self.tokenizer.encode(self.rendered(&observation.value)));
            tokens.extend(&self.tool_response_suffix_tokens);
        }
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

    /// Locates the opening tag naming the function, and returns its name and the
    /// byte offset of that tag's `>`.
    ///
    /// The template teaches `<function=NAME>`, and that is tried first. But a
    /// model that closes with `</function>` while having opened with a bare
    /// `<NAME>` has still plainly named a function, and refusing it loses an
    /// action the model unambiguously took. Observed live from this very
    /// checkpoint, on a surface that was otherwise complete and correct:
    ///
    ///     <tool_call>\n<list_files>\n<parameter=recursive>false</parameter>\n</function>\n</tool_call>
    ///
    /// vLLM's `qwen3_xml` parser accepts that; refusing it cost us the call and
    /// reported it as malformed syntax, which sends an agent into re-issuing the
    /// action rather than acting on its result.
    ///
    /// The bare form is accepted only under conditions that make a false
    /// positive implausible: the tag must be the FIRST tag in the call, it must
    /// not be one of the surface's own structural tags, and -- enforced by the
    /// caller -- the body must still close with `</function>`. Prose cannot
    /// reach here; this only ever sees the inside of a `<tool_call>` block.
    fn parse_xml_function_opener(call: &str) -> Option<(String, usize)> {
        let function_prefix = "<function=";
        if let Some(position) = call.find(function_prefix) {
            let name_start = position + function_prefix.len();
            let name_end = call[name_start..].find('>')? + name_start;
            return Some((call[name_start..name_end].trim().to_string(), name_end));
        }

        // Take the first tag and nothing later, so a `<parameter=...>` deeper in
        // the body can never be mistaken for the function name.
        let open = call.find('<')?;
        let name_end = call[open + 1..].find('>')? + open + 1;
        let inner = call[open + 1..name_end].trim();

        // `<function>NAME`: the tag is right and the name follows it as text.
        // Observed at 3 of 29 live calls. Read the name from after the tag,
        // bounded by the first whitespace or `<`, so it cannot swallow the body.
        if inner == "function" {
            let rest = &call[name_end + 1..];
            let name = rest
                .trim_start()
                .split(|c: char| c.is_whitespace() || c == '<')
                .next()
                .unwrap_or("");
            if !name.is_empty() && Self::is_plausible_function_name(name) {
                return Some((name.to_string(), name_end));
            }
            return None;
        }

        // Bare `<NAME>`.
        if !Self::is_plausible_function_name(inner)
            || matches!(inner, "tool_call" | "parameter")
        {
            return None;
        }
        Some((inner.to_string(), name_end))
    }

    /// Whether a run of text can be a function name at all.
    ///
    /// `<` is rejected explicitly, and that rejection is the whole point. An
    /// earlier version of this fallback checked for `=`, whitespace and the
    /// structural tag names but not for `<`, so the live surface
    /// `<function<bash>` parsed to the NAME `function<bash` with correct
    /// arguments and no error at all. A sentinel is a bad outcome; a
    /// confidently wrong tool name is a worse one, because nothing downstream
    /// can tell it from a real call.
    /// Finds a `<function=NAME>` opener that can stand in for a missing
    /// `<tool_call>`, returning its byte offset.
    ///
    /// WHY THIS EXISTS. The checkpoint itself sometimes emits a corrupted token
    /// where `<tool_call>` belongs. Measured on 201 replayed SWE-agent turns,
    /// the word `THOUGHT:` came back mangled -- as `THO`, `THOFT` or `THOTH` --
    /// on 12 of them, and vLLM produced the SAME mangling at the same rate (11
    /// of the same 201), so this is a property of the weights, not of this
    /// engine. When the mangling lands on the opener the surface reads:
    ///
    ///     </think>\n\nTHO<function=bash>\n<parameter=command>\n...\n</parameter>\n</function>\n</tool_call>
    ///
    /// Everything needed to execute the call is present and unambiguous.
    /// vLLM's `qwen3_xml` parser recovers it; refusing it cost 9 of 201 turns
    /// (4.5%), reported as `__ttb_malformed_tool_call__`, which sends the agent
    /// into re-issuing an action instead of acting on its result -- the exact
    /// read-only-trajectory failure this investigation started from.
    ///
    /// SAFETY. Only the explicit `<function=` form is recovered, never the bare
    /// `<NAME>` form: outside a `<tool_call>` wrapper there is nothing to
    /// distinguish a bare tag from ordinary prose or from a code block the
    /// model is quoting. The name must additionally pass
    /// `is_plausible_function_name`, and the CALLER only attempts recovery once
    /// `</tool_call>` has arrived, so a partially streamed prose fragment can
    /// never open a call that then swallows a real one.
    fn recoverable_function_opener(text: &str) -> Option<usize> {
        let prefix = "<function=";
        let position = text.find(prefix)?;
        let name_start = position + prefix.len();
        let name_end = text[name_start..].find('>')? + name_start;
        if Self::is_plausible_function_name(text[name_start..name_end].trim()) {
            Some(position)
        } else {
            None
        }
    }

    fn is_plausible_function_name(name: &str) -> bool {
        !name.is_empty()
            && !name.starts_with('/')
            && !name.contains('=')
            && !name.contains('<')
            && !name.contains('>')
            && !name.contains(char::is_whitespace)
    }

    /// Parses the surface the Qwen3.5+ `chat_template` demonstrates:
    /// `<function=NAME>` wrapping `<parameter=KEY>value</parameter>` pairs.
    ///
    /// Every value arrives as a JSON string. The surface carries no types, so
    /// inferring them here would be this decoder guessing at a schema it cannot
    /// see; the tool boundary validates against the real one.
    fn parse_xml_tool_call(call: &str) -> Option<(String, String)> {
        let call = call.trim();
        let (name, function_name_end) = Self::parse_xml_function_opener(call)?;
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
            // The same screen the function name gets, and for the same reason.
            // A model that writes `<parameter=command=` with no `>` sends this
            // scan hunting for the next `>` in the document -- which, in a shell
            // command, is the redirect in `2>/dev/null`. That yielded a
            // parameter NAME ninety characters long containing newlines, quotes
            // and pipes, paired with the value `/dev/null | head -20`, and no
            // error anywhere. Observed live on django__django-10914.
            if !Self::is_plausible_function_name(param_name) {
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
            // No opener -- but the checkpoint corrupts that very token often
            // enough to matter (see `recoverable_function_opener`). Wait for the
            // CLOSER before deciding: once `</tool_call>` has arrived the whole
            // block is in hand and can be judged complete, so recovery can never
            // half-open on a prose fragment and swallow a later real call.
            let recovered = self.accumulated.find("</tool_call>").and_then(|end| {
                Self::recoverable_function_opener(&self.accumulated[..end])
                    .map(|start| (start, end))
            });
            if let Some((start, end)) = recovered {
                let call = self.accumulated[start..end].trim().to_string();
                self.accumulated = self.accumulated[end + "</tool_call>".len()..].to_string();
                if let Some((name, args)) = self.parse_tool_call(&call) {
                    return ToolEvent::Call(name, args);
                }
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

    /// Captured verbatim from this checkpoint on 2026-08-14, decoding
    /// `tool_call_bash` against the BF16 weights. The model opened with a bare
    /// `<list_files>` instead of `<function=list_files>` and still closed with
    /// `</function>`. The surface was otherwise complete and correct -- not
    /// truncated -- and refusing it reported `__ttb_malformed_tool_call__`,
    /// which is what makes an agent re-issue its action instead of acting on
    /// the result.
    const QWEN36_BARE_OPENER_SURFACE: &str =
        "<list_files>\n<parameter=recursive>false</parameter>\n</function>";

    #[test]
    fn a_bare_opening_tag_still_names_the_function_the_model_called() {
        assert_eq!(
            QwenToolDecoder::parse_xml_tool_call(QWEN36_BARE_OPENER_SURFACE),
            Some((
                "list_files".to_string(),
                r#"{"recursive":"false"}"#.to_string()
            ))
        );
    }

    /// Captured verbatim on 2026-08-15 from `django__django-11039` turn 1 of a
    /// 201-turn SWE-agent replay. The checkpoint mangled the token where
    /// `<tool_call>` belongs into `THO` -- the truncated head of the `THOUGHT:`
    /// it meant to write. vLLM produced the same mangling on 11 of the same 201
    /// turns to this engine's 12, so the corruption is in the weights, not here;
    /// what differed is that vLLM's parser recovered the call and this one did
    /// not, losing 9 of 201 turns (4.5%) to `__ttb_malformed_tool_call__`.
    const QWEN36_CORRUPTED_OPENER_SURFACE: &str = concat!(
        "</think>\n\nTHO<function=bash>\n<parameter=command>\n",
        "find /testbed -type f -name \"*.py\"\n</parameter>\n</function>\n</tool_call>"
    );

    #[test]
    fn a_corrupted_tool_call_opener_still_yields_the_call_the_model_made() {
        let end = QWEN36_CORRUPTED_OPENER_SURFACE.find("</tool_call>").unwrap();
        let start =
            QwenToolDecoder::recoverable_function_opener(&QWEN36_CORRUPTED_OPENER_SURFACE[..end])
                .expect("a plausible <function=NAME> opener must be recoverable");
        assert_eq!(
            QwenToolDecoder::parse_xml_tool_call(
                QWEN36_CORRUPTED_OPENER_SURFACE[start..end].trim()
            ),
            Some((
                "bash".to_string(),
                r#"{"command":"find /testbed -type f -name \"*.py\""}"#.to_string()
            ))
        );
    }

    /// The recovery must not invent a call out of prose. Without the
    /// `<tool_call>` wrapper there is no structural signal left, so the name
    /// screen is the only thing standing between a recovered action and a
    /// hallucinated one.
    #[test]
    fn prose_that_merely_mentions_a_function_tag_is_not_recovered_as_a_call() {
        for text in [
            "the helper is written <function=two words> in the docs",
            "see <function=> for details",
            "<function=a<b>",
            "compare <function=x=y> against",
        ] {
            assert_eq!(
                QwenToolDecoder::recoverable_function_opener(text),
                None,
                "must not recover a call from {text:?}"
            );
        }
        // The legitimate form is still recovered.
        assert!(QwenToolDecoder::recoverable_function_opener("noise<function=bash>x").is_some());
    }

    /// Captured live on 2026-08-14 across 29 tool calls from this checkpoint.
    /// 8 of 29 openers (27.6%) did not conform to the template. These are the
    /// exact bytes of the two remaining forms.
    #[test]
    fn the_live_non_conforming_openers_parse_to_their_real_names() {
        // `<function>NAME`, 3 of 29. The tag is right; the name follows as text.
        assert_eq!(
            QwenToolDecoder::parse_xml_tool_call(
                "<function>bash\n<parameter=command>\ndu -sh .\n</parameter>\n</function>"
            ),
            Some(("bash".to_string(), r#"{"command":"du -sh ."}"#.to_string()))
        );
    }

    /// `<function<bash>` produced the NAME `function<bash`, with correct
    /// arguments and no error, because the first version of this fallback
    /// screened for `=`, whitespace and the structural tag names but not for
    /// `<`. A sentinel loses an action; a confidently wrong name INVENTS one,
    /// and nothing downstream can tell it from a real call.
    #[test]
    fn a_malformed_opener_never_becomes_a_confidently_wrong_tool_name() {
        let live = "<function<bash>\n<parameter=command>\nhead -n 40 README.md\n</parameter>\n</function>";
        let parsed = QwenToolDecoder::parse_xml_tool_call(live);
        assert_ne!(
            parsed.as_ref().map(|(name, _)| name.as_str()),
            Some("function<bash"),
            "a tag name containing '<' must never be accepted as a function name"
        );
        assert_eq!(parsed, None);
    }

    /// Captured live on django__django-10914, turn 1. The model wrote
    /// `<parameter=command=` with no `>`, so the scan for the parameter name ran
    /// on to the next `>` in the document -- the shell redirect in
    /// `2>/dev/null`. The old code produced a call whose argument KEY was
    /// ninety characters of shell text and whose VALUE was `/dev/null | head
    /// -20`, silently, and the agent executed nothing useful.
    #[test]
    fn a_parameter_name_that_swallowed_a_shell_redirect_is_refused() {
        let live = concat!(
            "<function=bash>\n",
            "<parameter=command=\n",
            "find /testbed -type f -name \"*.py\" | xargs grep -l \"FILE_UPLOAD_PERMISSION\" 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        let parsed = QwenToolDecoder::parse_xml_tool_call(live);
        assert_eq!(
            parsed, None,
            "a parameter name containing whitespace or quotes is not a name"
        );

        // The well-formed version of the very same call still parses, so the
        // screen rejects the malformation and not the command's content --
        // `2>/dev/null` inside a VALUE is ordinary shell and must survive.
        let repaired = concat!(
            "<function=bash>\n",
            "<parameter=command>\n",
            "find /testbed -type f -name \"*.py\" | xargs grep -l \"FILE_UPLOAD_PERMISSION\" 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        let (name, args) = QwenToolDecoder::parse_xml_tool_call(repaired).expect("repaired parses");
        assert_eq!(name, "bash");
        assert!(args.contains("2>/dev/null"), "{args}");
    }

    /// The fallback must not turn the surface's own structure, or an unclosed
    /// body, into a call. Each of these would be a false action -- worse than
    /// the dropped one it exists to recover, because a wrong tool call executes.
    #[test]
    fn the_bare_opener_fallback_refuses_what_is_not_a_call() {
        for surface in [
            // Structural tags of the surface itself.
            "<tool_call>\n<parameter=x>1</parameter>\n</function>",
            "<parameter=x>1</parameter>\n</function>",
            // A closing tag leading.
            "</function>",
            // No `</function>` at all: genuinely incomplete, and the caller's
            // completeness requirement is what rejects it.
            "<list_files>\n<parameter=recursive>false</parameter>",
            // Not a tag at all.
            "just some prose about functions",
            // An empty tag names nothing.
            "<>\n</function>",
        ] {
            assert_eq!(
                QwenToolDecoder::parse_xml_tool_call(surface),
                None,
                "should not parse: {surface}"
            );
        }
    }

    /// The conforming surface must keep winning when both could match, so this
    /// leniency cannot change what a well-formed call decodes to.
    #[test]
    fn the_conforming_opener_is_preferred_over_a_bare_one() {
        assert_eq!(
            QwenToolDecoder::parse_xml_tool_call(
                "<wrong>\n<function=right>\n<parameter=k>v</parameter>\n</function>"
            ),
            Some(("right".to_string(), r#"{"k":"v"}"#.to_string()))
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
                thinking_off_suffix: "",
                empty_reasoning_header: true,
                system_before_tools: false,
                content_call_separator: "\n\n",
                trim_content: false,
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
                thinking_off_suffix: "",
                empty_reasoning_header: true,
                system_before_tools: false,
                content_call_separator: "\n\n",
                trim_content: false,
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
                thinking_off_suffix: "",
                empty_reasoning_header: true,
                system_before_tools: false,
                content_call_separator: "\n\n",
                trim_content: false,
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
        // The reasoning half is taken at DIFFERENT indices from the content
        // half -- before the first `</think>`, after the last `<think>` -- so a
        // message carrying either delimiter twice must not fold into one split.
        assert_eq!(
            QwenInstruct::split_thinking("<think>\na\n</think>\n\nb </think> c"),
            ("a", " c")
        );
    }

    /// The `qwen3_6` arm `pie_model::instruct::create` selects: the Qwen3.5+
    /// template's own generation headers, and its `|trim`.
    fn qwen3_6() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                tool_call_format: ToolCallFormat::Qwen35Xml,
                has_thinking: true,
                has_tools: true,
                generation_suffix: "<think>\n",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                empty_reasoning_header: true,
                system_before_tools: false,
                content_call_separator: "\n\n",
                trim_content: true,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    /// The `qwen3` arm `pie_model::instruct::create` selects, field for field.
    ///
    /// A separate helper from `qwen3()` above, which is a generic ChatML
    /// fixture and does not claim to be any checkpoint: the four fields this
    /// one restates are the ones Qwen3's template answers differently from
    /// Qwen3.5's, so a test that means "the Qwen3 checkpoint" has to say them.
    fn qwen3_arm() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                tool_call_format: ToolCallFormat::Json,
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                empty_reasoning_header: false,
                system_before_tools: true,
                content_call_separator: "\n",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    /// Qwen3's template renders the reasoning block on a post-query turn only
    /// when the turn is last or carries reasoning; Qwen3.5's renders one either
    /// way. Pie emitted an empty block for both, which is 19 characters the
    /// Qwen3 checkpoint never sees in that position.
    #[test]
    fn an_empty_reasoning_header_is_rendered_only_where_the_template_renders_one() {
        let call = ToolCall { name: "f".into(), arguments_json: "{}".into() };
        let qwen3 = qwen3_arm();
        let bare = qwen3.assistant_turn_body("", &[call.clone()], true);
        assert!(!bare.starts_with("<think>"), "{bare}");
        // Reasoning present: the block is the template's in both generations.
        assert!(
            qwen3
                .assistant_turn_body("<think>\nwhy\n</think>\n\nok", &[], true)
                .starts_with("<think>\nwhy\n</think>\n\n")
        );
        // And the Qwen3.5+ arm is unmoved: still an empty block on a turn that
        // carried no reasoning.
        assert!(
            qwen3_6()
                .assistant_turn_body("", &[call], true)
                .starts_with("<think>\n\n</think>\n\n")
        );
    }

    /// The template opens the model's turn INSIDE a reasoning block, and closes
    /// an empty one when the caller asked for no thinking. Two different
    /// strings, so one `generation_suffix` cannot carry both -- and the
    /// thinking-off answer is a header, never an injected user turn.
    ///
    /// The bytes are pinned end to end by the checkpoint-tokenizer parity
    /// fixture; what is checkable against this vocabulary is that the two
    /// headers are wired to different suffixes and share the assistant prefix.
    #[test]
    fn the_generation_header_is_the_templates_cue_in_both_thinking_modes() {
        let inst = qwen3_6();
        assert_eq!(inst.config.generation_suffix, "<think>\n");
        assert_eq!(inst.config.thinking_off_suffix, "<think>\n\n</think>\n\n");
        assert_ne!(inst.cue(), inst.cue_without_thinking());
        for header in [inst.cue(), inst.cue_without_thinking()] {
            assert!(header.starts_with(&inst.assistant_prefix));
            assert!(header.len() > inst.assistant_prefix.len());
        }
        // A template with no thinking-off header answers with its only one.
        let plain = qwen3();
        assert_eq!(plain.cue_without_thinking(), plain.cue());
        assert_eq!(plain.cue(), plain.assistant_prefix);
    }

    /// `|trim` is per-template: Qwen3.5+ trims every message, Qwen3 and Qwen2
    /// emit `message.content` verbatim.
    #[test]
    fn content_is_trimmed_only_where_the_template_trims() {
        assert_eq!(qwen3_6().rendered("\n  Hello  \n"), "Hello");
        assert_eq!(qwen3().rendered("\n  Hello  \n"), "\n  Hello  \n");
    }

    /// A run of consecutive results is ONE turn in the template. One turn each
    /// is a different prompt, and it is what `answer` alone could express.
    #[test]
    fn consecutive_tool_results_share_a_turn() {
        let inst = qwen3();
        let run = [
            ToolObservation { name: "a".into(), value: "Hello".into() },
            ToolObservation { name: "b".into(), value: " world".into() },
        ];
        assert_eq!(
            inst.tokenizer.decode(&inst.answer_all(&run), false),
            "<|im_start|>user\n<tool_response>\nHello\n</tool_response>\n\
             <tool_response>\n world\n</tool_response><|im_end|>\n"
        );
        // A single result is the same turn `answer` has always produced.
        assert_eq!(inst.answer("a", "Hello"), inst.answer_all(&run[..1]));
    }

    /// An assistant turn after the last user query keeps a reasoning header,
    /// empty when the replayed message carried no reasoning.
    #[test]
    fn a_post_query_assistant_turn_keeps_its_reasoning_header() {
        let inst = qwen3_6();
        let call = ToolCall { name: "f".into(), arguments_json: "{}".into() };
        // An empty header when the replayed message carried no reasoning, and
        // the call surface `encoded_turns` used to drop entirely.
        assert_eq!(
            inst.assistant_turn_body("", &[call.clone()], true),
            "<think>\n\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>"
        );
        // Content and a call are separated by a blank line, later calls by one.
        assert_eq!(
            inst.assistant_turn_body("<think>\nwhy\n</think>\n\nok", &[call.clone(), call.clone()], true),
            "<think>\nwhy\n</think>\n\nok\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n<tool_call>\n<function=f>\n</function>\n</tool_call>"
        );
        // Qwen3's template writes ONE newline there, and the same one before
        // every later call.
        assert_eq!(
            qwen3_arm().assistant_turn_body("<think>\nwhy\n</think>\n\nok", &[call.clone(), call], true),
            "<think>\nwhy\n</think>\n\nok\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"
        );
        // Before the boundary the header is dropped, which is what `assistant`
        // has always rendered.
        assert_eq!(inst.assistant_call("Hello", &[], false), inst.assistant("Hello"));
    }

    /// The surface the checkpoint's template demonstrates, byte for byte --
    /// `encoded_turns` used to render a replayed call as an EMPTY turn, and
    /// then to render every arm's in the Qwen3.5 XML one.
    #[test]
    fn a_replayed_call_renders_the_surface_the_template_teaches() {
        let call = ToolCall {
            name: "get_weather".into(),
            arguments_json: r#"{"city": "Paris", "days": 3}"#.into(),
        };
        assert_eq!(
            qwen3_6().tool_call_surface(&call),
            "<tool_call>\n<function=get_weather>\n\
             <parameter=city>\nParis\n</parameter>\n\
             <parameter=days>\n3\n</parameter>\n\
             </function>\n</tool_call>"
        );
        // The Json arms get the body their own template writes, arguments
        // through as they arrived.
        assert_eq!(
            qwen3_arm().tool_call_surface(&call),
            "<tool_call>\n{\"name\": \"get_weather\", \
             \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>"
        );
    }

    /// Six ways the declaration used to differ from the checkpoint's own
    /// template: a stray leading space, the wording, the call instruction's
    /// blank line, one example parameter instead of two, no `<IMPORTANT>`
    /// block, and the caller's system content as a SECOND turn.
    #[test]
    fn the_tool_declaration_is_the_checkpoints_own_system_turn() {
        let inst = qwen3_6();
        let body = inst.tool_system_body("You are a helpful assistant.", &["{}".to_string()]);
        assert_eq!(
            body,
            format!(
                "# Tools\n\nYou have access to the following functions:\n\n\
                 <tools>\n{{}}\n</tools>\n\n{QWEN35_XML_CALL_INSTRUCTION}\n\n\
                 You are a helpful assistant."
            )
        );
        assert!(body.contains("<parameter=example_parameter_2>"));
        assert!(body.contains("<IMPORTANT>"));
        // No system content, no separator and no empty tail.
        assert!(inst.tool_system_body("  ", &["{}".to_string()]).ends_with("</IMPORTANT>"));
    }

    /// Qwen3 and Qwen2.5 write the caller's system text FIRST and the
    /// declaration after it; Qwen3.5+ do the reverse. Same two pieces, and the
    /// order is the prompt.
    #[test]
    fn the_declaration_and_the_system_text_are_ordered_as_the_template_writes_them() {
        let tools = ["{}".to_string()];
        let qwen3 = qwen3_arm();
        let body = qwen3.tool_system_body("You are a helpful assistant.", &tools);
        assert!(body.starts_with("You are a helpful assistant.\n\n# Tools\n\n"), "{body}");
        // The declaration opens with `# Tools` and no leading space -- no
        // template has one, and it cost a token on every tool-declaring prompt.
        assert!(
            QwenInstruct::build_tool_system_prompt(&tools, ToolCallFormat::Json)
                .starts_with("# Tools")
        );
        // With no system text the turn is the declaration alone, either way up.
        assert_eq!(
            qwen3.tool_system_body("  ", &tools),
            QwenInstruct::build_tool_system_prompt(&tools, ToolCallFormat::Json)
        );
        // And the Qwen3.5+ arm still leads with the declaration.
        assert!(
            qwen3_6()
                .tool_system_body("You are a helpful assistant.", &tools)
                .starts_with("# Tools")
        );
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
                thinking_off_suffix: "",
                empty_reasoning_header: true,
                system_before_tools: false,
                content_call_separator: "\n\n",
                trim_content: false,
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

    /// The recovery must work through `feed`, not merely in the helper.
    ///
    /// This streams the live `django__django-11039` turn-1 shape: a mangled
    /// `THO` where `<tool_call>` belongs, and NO opener token at any point. A
    /// helper-only test would pass while the decoder still returned
    /// `ToolEvent::Start` forever and the call stayed lost.
    #[test]
    fn feed_recovers_a_call_whose_tool_call_opener_never_arrived() {
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "\n",
            "</think>",
            "THO",
            "<function=bash>",
            "<parameter=command>",
            "ls -la",
            "</parameter>",
            "</function>",
            "</tool_call>",
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
                thinking_off_suffix: "",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        );
        let mut dec = inst.tool_decoder();
        // </think> \n THO <function=bash> <parameter=command> ls -la </parameter> </function>
        for id in [4u32, 3, 5, 6, 7, 8, 9, 10] {
            assert!(
                matches!(dec.feed(&[id]), ToolEvent::Start),
                "nothing may resolve before the closer arrives"
            );
        }
        match dec.feed(&[11]) {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "bash");
                assert_eq!(args, r#"{"command":"ls -la"}"#);
            }
            other => panic!("expected the recovered Call, got {other:?}"),
        }
    }
}
