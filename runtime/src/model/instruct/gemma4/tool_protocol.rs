//! Gemma 4's native tool protocol: schema declaration, call grammar, and the
//! streaming decoder that reads calls back out.
//!
//! Ground truth is the Gemma 4 chat template (`google-gemma-4-*-it.jinja`).
//! Unlike the ChatML families, Gemma 4 does not put JSON on the wire; it uses a
//! compact colon/brace DSL in which every string is delimited by `<|"|>` and
//! every map is emitted in key-sorted order:
//!
//!   schema    <|tool>declaration:NAME{description:<|"|>..<|"|>,parameters:{..}}<tool|>
//!   call      <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>
//!   response  <|tool_response>response:NAME{value:<|"|>..<|"|>}<tool_response|>
//!
//! # Validation is rendering
//!
//! Every formatter here returns `Option`, and [`validate_tools`] rejects the
//! *entire* toolset if any part of any schema comes back `None`. There is
//! deliberately no path that renders a schema approximately: a construct this
//! module cannot express identically in the declaration, the grammar, and the
//! parser is refused outright.
//!
//! That matters because the three have to agree. Declaring a subset would let
//! the grammar pin a name the prompt never introduced; constraining a subset
//! would let the model emit a call the parser cannot read; and rendering a
//! property the grammar cannot match would produce a prompt the model can only
//! answer by violating its own constraint.

use crate::model::tokenizer::Tokenizer;
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Wire delimiters for the Gemma 4 tool DSL.
pub(super) const TOOL_OPEN: &str = "<|tool>";
pub(super) const TOOL_CLOSE: &str = "<tool|>";
pub(super) const TOOL_CALL_OPEN: &str = "<|tool_call>";
pub(super) const TOOL_CALL_CLOSE: &str = "<tool_call|>";
pub(super) const TOOL_RESPONSE_OPEN: &str = "<|tool_response>";
pub(super) const TOOL_RESPONSE_CLOSE: &str = "<tool_response|>";
/// Gemma 4's string delimiter. Both opening and closing quote are this token.
pub(super) const QUOTE: &str = "<|\"|>";

/// Property keys the template treats as schema metadata rather than as nested
/// parameter names (`standard_keys` in `format_parameters`). A schema that
/// actually names a property one of these is rejected: the template would drop
/// it, and a declaration with fewer properties than the schema describes is
/// exactly the silent alteration this module refuses to make.
const STANDARD_KEYS: [&str; 5] = ["description", "type", "properties", "required", "nullable"];

/// The schema keywords `format_parameters` renders for a property of a given
/// (uppercased) type. Consumption is type-conditioned: `enum` is only rendered
/// for a string, `items` only for an array, `properties`/`required` only for an
/// object. A keyword outside its type's set would be silently dropped — telling
/// the model "any value" where the schema constrained it — so it rejects the
/// toolset rather than being ignored. Keeping the accepted set in step with what
/// is actually rendered is what makes declaration and grammar agree.
fn permitted_property_keys(kind: &str) -> &'static [&'static str] {
    match kind {
        "STRING" => &["description", "type", "nullable", "enum"],
        "ARRAY" => &["description", "type", "nullable", "items"],
        "OBJECT" => &["description", "type", "nullable", "properties", "required"],
        _ => &["description", "type", "nullable"],
    }
}

/// The (uppercased) JSON-schema types Gemma 4 can declare, constrain, and
/// parse. A type outside this domain has no faithful rendering — the grammar
/// could not produce a value for it — so declaring it would name a type the
/// model cannot honour. Every `type` field routes through this and refuses the
/// toolset otherwise, rather than emitting `type:"WIDGET"` and hoping.
fn is_supported_type(kind: &str) -> bool {
    matches!(
        kind,
        "STRING" | "NUMBER" | "INTEGER" | "BOOLEAN" | "ARRAY" | "OBJECT"
    )
}

/// Keys an ARRAY property's `items` schema may carry, from its element type.
/// A single element type reuses [`permitted_property_keys`]; a union type
/// permits the union of its members' keys. `None` if `type` is missing or not a
/// string / array of strings, which is itself an unrenderable item schema.
fn item_permitted_keys(items: &Map<String, Value>) -> Option<Vec<&'static str>> {
    let kinds: Vec<String> = match items.get("type")? {
        Value::String(kind) => vec![kind.to_uppercase()],
        Value::Array(kinds) => kinds
            .iter()
            .map(|kind| kind.as_str().map(str::to_uppercase))
            .collect::<Option<_>>()?,
        _ => return None,
    };
    let mut permitted: Vec<&'static str> = Vec::new();
    for kind in &kinds {
        if !is_supported_type(kind) {
            return None;
        }
        for key in permitted_property_keys(kind) {
            if !permitted.contains(key) {
                permitted.push(key);
            }
        }
    }
    Some(permitted)
}

/// How deeply a tool call's arguments may nest objects and arrays before
/// [`DslParser`] refuses the call.
///
/// The parser reads raw model output, which never passes through `serde_json`
/// and so inherits none of its depth cap. Recursion there runs on a 2 MiB tokio
/// worker stack, and a Rust stack overflow aborts the process rather than
/// unwinding — so unbounded nesting is not a rejected call, it is a SIGABRT
/// that takes the daemon and every tenant with it. A few kilobytes of
/// `{x:{x:{x:` is enough, and that is well inside any generation budget.
///
/// Thirty-two is far past what a real tool signature nests while leaving the
/// stack untouched. See [`build_tool_call_grammar`] for why the grammar is left
/// self-recursive rather than expanded to match this bound exactly.
const MAX_ARGUMENT_DEPTH: usize = 32;

/// A tool schema Gemma 4 can declare, constrain, and parse identically.
pub(super) struct ValidatedTool {
    pub(super) name: String,
    pub(super) declaration: String,
}

/// Tool names are interpolated into the DSL (`call:NAME{`) and into EBNF string
/// literals, neither of which has an escape mechanism. Restricting the charset
/// makes a name that could break out of either structurally impossible rather
/// than something to sanitise after the fact.
fn is_supported_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

/// Argument and property keys must match the grammar's `argument-key` rule
/// exactly, or the model could not generate a call using them. Note this is
/// stricter than a tool name: no `.`, because `argument-key` does not admit it.
fn is_supported_argument_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Validate and render the whole toolset, or reject all of it.
///
/// Returns `None` if any schema is unsupported, so that declaration and
/// grammar fail closed on identical input. Duplicate names are rejected too:
/// they would emit a redundant grammar alternation and leave the observed call
/// name ambiguous as to which schema it satisfied.
pub(super) fn validate_tools(tools: &[String]) -> Option<Vec<ValidatedTool>> {
    let validated: Vec<ValidatedTool> = tools
        .iter()
        .map(|tool| validate_tool(tool))
        .collect::<Option<_>>()?;
    let mut names: Vec<&str> = validated.iter().map(|tool| tool.name.as_str()).collect();
    names.sort_unstable();
    let unique = names.len();
    names.dedup();
    if names.len() != unique {
        return None;
    }
    Some(validated)
}

/// The set of tool names actually declared to the model for `tools`, or empty
/// when the toolset is unsupported. This is the membership set the streaming
/// decoder checks an observed call name against; an unsupported toolset yields
/// an empty set, so the decoder attributes no call rather than one the model
/// was never validly shown.
pub(super) fn declared_names(tools: &[String]) -> BTreeSet<String> {
    validate_tools(tools)
        .map(|validated| validated.into_iter().map(|tool| tool.name).collect())
        .unwrap_or_default()
}

fn validate_tool(tool: &str) -> Option<ValidatedTool> {
    let parsed: Value = serde_json::from_str(tool).ok()?;
    // Accept either a bare function schema or the OpenAI `{type, function}`
    // envelope. In the envelope form the outer object may carry only those two
    // keys; anything else is part of a contract this module would silently drop.
    let function = match parsed.get("function") {
        Some(function) => {
            let outer = parsed.as_object()?;
            if outer
                .keys()
                .any(|key| !matches!(key.as_str(), "type" | "function"))
            {
                return None;
            }
            // The envelope's discriminator must actually say `function`. A
            // different (or missing) value would be silently discarded and the
            // inner object rendered as if it had been declared a function tool,
            // which is exactly the unnoticed reinterpretation this module
            // refuses.
            if outer.get("type").and_then(Value::as_str) != Some("function") {
                return None;
            }
            function
        }
        None => &parsed,
    };
    // Validate the function-level keys against exactly what is rendered. An
    // unrecognised key here (including the template's `response:` branch, which
    // this module does not implement) would understate the declared contract,
    // so it rejects the toolset rather than being ignored.
    let function_obj = function.as_object()?;
    if function_obj
        .keys()
        .any(|key| !matches!(key.as_str(), "name" | "description" | "parameters"))
    {
        return None;
    }
    let name = function.get("name")?.as_str()?;
    if !is_supported_tool_name(name) {
        return None;
    }
    Some(ValidatedTool {
        name: name.to_string(),
        declaration: format_declaration(function, name)?,
    })
}

/// Render one schema exactly as the template's `format_function_declaration`
/// macro does.
fn format_declaration(function: &Value, name: &str) -> Option<String> {
    let description = optional_string(function, "description")?;
    let mut out = format!("declaration:{name}{{description:{}", quote(&description)?);

    if let Some(params) = function.get("parameters").filter(|p| !p.is_null()) {
        // The parameters block is closed by its `type` field and by nothing
        // else, so a typeless or non-object `parameters` would emit an
        // unterminated declaration.
        let params = params.as_object()?;
        // The parameters object renders only its type, properties, and required
        // fields. A key outside that set (e.g. `additionalProperties`) would be
        // dropped, understating the contract, so it rejects the toolset.
        if params
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "properties" | "required"))
        {
            return None;
        }
        // Top-level parameters are the tool call's argument object, so the
        // schema must actually say so. A non-object parameters type would render
        // `parameters:{type:"STRING"}` with property/required blocks the call
        // grammar treats as an object anyway — a declaration the model could not
        // honour — so anything but OBJECT rejects the toolset.
        let kind = params.get("type").and_then(Value::as_str)?;
        if !kind.eq_ignore_ascii_case("object") {
            return None;
        }
        out.push_str(",parameters:{");
        if let Some(properties) = params
            .get("properties")
            .filter(|p| !p.is_null())
            .map(|p| p.as_object())
        {
            let properties = properties?;
            if !properties.is_empty() {
                out.push_str("properties:{");
                out.push_str(&format_parameters(properties)?);
                out.push_str("},");
            }
        }
        if let Some(required) = format_required_field(params)? {
            out.push_str(&format!("required:[{required}],"));
        }
        out.push_str(&format!("type:{}}}", quote(&kind.to_uppercase())?));
    }

    out.push('}');
    Some(out)
}

/// Render `text` as a Gemma 4 DSL string literal (`<|"|>text<|"|>`), or refuse.
///
/// The delimiter is a single token with no escape sequence, and the call
/// grammar forbids `<` inside any value (`argument-char ::= [^<]`). A string
/// carrying `<` therefore cannot be declared, constrained, and parsed
/// identically: emitting it would either truncate the literal or advertise a
/// value the grammar can never produce. Refusing here fails the whole toolset
/// closed in step with [`build_tool_call_grammar`], keeping declaration and
/// grammar in agreement. Every `<|"|>`-delimited position routes through this.
fn quote(text: &str) -> Option<String> {
    if text.contains('<') {
        return None;
    }
    Some(format!("{QUOTE}{text}{QUOTE}"))
}

/// A `description` that is absent (rendered empty, as the template does) or a
/// string. Any other type is an unsupported schema rather than a description.
fn optional_string(value: &Value, key: &str) -> Option<String> {
    match value.get(key) {
        None | Some(Value::Null) => Some(String::new()),
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => None,
    }
}

/// `required:[..]` — `None` when absent or empty, rejecting a non-array or an
/// array holding anything but strings.
fn format_required_field(container: &Map<String, Value>) -> Option<Option<String>> {
    let Some(required) = container.get("required").filter(|r| !r.is_null()) else {
        return Some(None);
    };
    let required = required.as_array()?;
    if required.is_empty() {
        return Some(None);
    }
    let rendered = required
        .iter()
        .map(|item| item.as_str().and_then(quote))
        .collect::<Option<Vec<_>>>()?
        .join(",");
    Some(Some(rendered))
}

/// `format_parameters` — one `key:{..}` block per property, key-sorted.
///
/// `serde_json::Map` may preserve insertion order, so sort explicitly and keep
/// output independent of how the schema was parsed.
fn format_parameters(properties: &Map<String, Value>) -> Option<String> {
    let mut keys: Vec<&String> = properties.keys().collect();
    keys.sort();

    let mut blocks: Vec<String> = Vec::new();
    for key in keys {
        // The template would silently skip a property named like schema
        // metadata, and the grammar could never match a key outside
        // `argument-key`. Both are rejections, not omissions.
        if STANDARD_KEYS.contains(&key.as_str()) || !is_supported_argument_key(key) {
            return None;
        }
        let value = properties[key].as_object()?;
        let kind = value.get("type").and_then(Value::as_str)?.to_uppercase();
        if !is_supported_type(&kind) {
            return None;
        }
        // A keyword this renderer does not consume for this type would vanish
        // from the declaration, telling the model "any value" where the tool
        // actually constrains one — `pattern`, `minLength`, `const`, `anyOf`,
        // and friends, but also `enum` on a number or `items` on a string,
        // which the type-conditioned branches below never render. Dropping any
        // of them is the silent alteration this module refuses to make, so an
        // unrecognised or type-mismatched keyword rejects the toolset.
        if value
            .keys()
            .any(|k| !permitted_property_keys(&kind).contains(&k.as_str()))
        {
            return None;
        }
        let mut fields: Vec<String> = Vec::new();

        let description = optional_string(&Value::Object(value.clone()), "description")?;
        if value.contains_key("description") {
            fields.push(format!("description:{}", quote(&description)?));
        }

        match kind.as_str() {
            "STRING" => {
                if let Some(choices) = value.get("enum").filter(|e| !e.is_null()) {
                    fields.push(format!("enum:{}", format_argument(choices, true)?));
                }
            }
            "ARRAY" => {
                if let Some(items) = value.get("items").filter(|i| !i.is_null()) {
                    let items = items.as_object()?;
                    if !items.is_empty() {
                        fields.push(format!("items:{{{}}}", format_items(items)?));
                    }
                }
            }
            _ => {}
        }

        if let Some(nullable) = value.get("nullable").filter(|n| !n.is_null())
            && nullable.as_bool()?
        {
            fields.push("nullable:true".to_string());
        }

        if kind == "OBJECT" {
            // The template prefers an explicit `properties` map and otherwise
            // treats the parameter body itself as the property set.
            let nested = match value.get("properties").filter(|p| !p.is_null()) {
                Some(properties) => properties.as_object()?,
                None => value,
            };
            fields.push(format!("properties:{{{}}}", format_parameters(nested)?));
            if let Some(required) = format_required_field(value)? {
                fields.push(format!("required:[{required}]"));
            }
        }

        fields.push(format!("type:{}", quote(&kind)?));
        blocks.push(format!("{key}:{{{}}}", fields.join(",")));
    }
    Some(blocks.join(","))
}

/// The `items:{..}` body of an ARRAY property: item keys in sorted order, with
/// `properties`/`required`/`type` special-cased as the template does.
fn format_items(items: &Map<String, Value>) -> Option<String> {
    // Same type-conditioned rejection as `format_parameters`, one level down:
    // an item schema carrying a key its element type never renders (e.g.
    // `properties` on a string item) would be silently dropped, so refuse.
    let permitted = item_permitted_keys(items)?;
    if items.keys().any(|k| !permitted.contains(&k.as_str())) {
        return None;
    }
    let mut keys: Vec<&String> = items.keys().collect();
    keys.sort();

    let mut fields: Vec<String> = Vec::new();
    for key in keys {
        let value = &items[key];
        if value.is_null() {
            continue;
        }
        let rendered = match key.as_str() {
            "properties" => format!("properties:{{{}}}", format_parameters(value.as_object()?)?),
            "required" => match format_required_field(items)? {
                Some(required) => format!("required:[{required}]"),
                None => continue,
            },
            "type" => match value {
                // A single item type, or a union of them — the template
                // uppercases either shape.
                Value::String(kind) => format!("type:{}", quote(&kind.to_uppercase())?),
                Value::Array(kinds) => {
                    let union: Vec<Value> = kinds
                        .iter()
                        .map(|kind| kind.as_str().map(|k| Value::String(k.to_uppercase())))
                        .collect::<Option<_>>()?;
                    format!("type:{}", format_argument(&Value::Array(union), true)?)
                }
                _ => return None,
            },
            other => format!("{other}:{}", format_argument(value, true)?),
        };
        fields.push(rendered);
    }
    Some(fields.join(","))
}

/// `format_argument` — the DSL's value encoding. `escape_keys` mirrors the
/// template argument of the same name: schema positions quote their map keys,
/// tool-call and tool-response positions do not.
fn format_argument(value: &Value, escape_keys: bool) -> Option<String> {
    Some(match value {
        Value::String(text) => quote(text)?,
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body = keys
                .into_iter()
                .map(|key| {
                    if !escape_keys && !is_supported_argument_key(key) {
                        return None;
                    }
                    let rendered_key = if escape_keys {
                        quote(key)?
                    } else {
                        key.clone()
                    };
                    Some(format!(
                        "{rendered_key}:{}",
                        format_argument(&map[key], escape_keys)?
                    ))
                })
                .collect::<Option<Vec<_>>>()?
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(items) => {
            let body = items
                .iter()
                .map(|item| format_argument(item, escape_keys))
                .collect::<Option<Vec<_>>>()?
                .join(",");
            format!("[{body}]")
        }
        // `null` has no DSL spelling.
        Value::Null => return None,
    })
}

/// Wrap a validated declaration in its `<|tool>` block.
pub(super) fn declaration_block(tool: &ValidatedTool) -> String {
    format!("{TOOL_OPEN}{}{TOOL_CLOSE}", tool.declaration)
}

/// Render a tool result, or refuse it.
///
/// The template defaults a missing name to `unknown` rather than dropping the
/// block, so the model always sees a well-formed response.
///
/// `value` is tool output — fetched pages, command results, whatever the tool
/// returned — which makes it the most reachable string in the protocol for a
/// party who does not control the prompt. It is interpolated into the same
/// unescapable DSL as everything else, so it routes through [`quote`]: a result
/// carrying `<` would close the literal early and let the remainder be read as
/// structure, forging a call or a turn. `name` is held to the declaration
/// charset for the same reason it is there.
///
/// Returning `None` rather than a lossy rendering is what lets the caller fail
/// the turn instead of silently feeding the model a corrupted response.
pub(super) fn response_block(name: &str, value: &str) -> String {
    let name = if name.is_empty() { "unknown" } else { name };
    format!(
        "{TOOL_RESPONSE_OPEN}response:{name}{{value:{QUOTE}{value}{QUOTE}}}{TOOL_RESPONSE_CLOSE}"
    )
}

/// EBNF constraining generation to exactly one well-formed Gemma 4 tool call.
///
/// One call per generation is the honest contract: the decoder emits a call on
/// the first closing `<tool_call|>` and callers stop generating there, so a
/// grammar admitting more would describe output no consumer can observe.
pub(super) fn build_tool_call_grammar(tools: &[ValidatedTool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let name_alt = tools
        .iter()
        .map(|tool| format!("\"{}\"", tool.name))
        .collect::<Vec<_>>()
        .join(" | ");
    // `argument-char ::= [^<]` keeps string values from swallowing the `<|"|>`
    // terminator. Gemma 4 has no escape sequence for it, so a literal `<`
    // inside an argument is unrepresentable by construction.
    //
    // `argument-object` and `argument-array` are self-recursive, so the grammar
    // admits arbitrarily deep nesting while [`DslParser`] stops at
    // [`MAX_ARGUMENT_DEPTH`]. That asymmetry is deliberate. Expanding the
    // containers into one rule set per level — so the two matched exactly —
    // was measured and rejected: matcher construction roughly doubles per
    // level (0.86s at depth 2, 1.21s at 3, 4.85s at 4) and the grammar engine
    // fails outright at depth 5, which would trade a rare rejected call for a
    // guaranteed cost on every constrained generation.
    //
    // The asymmetry only ever fails closed. An over-nested call is refused by
    // the parser, so no `Call` is reported and the caller dispatches nothing;
    // the reverse — the parser accepting what the grammar forbids, which would
    // mean phantom calls — remains impossible. What the depth cap actually
    // prevents is the parser recursing off a 2 MiB stack, and the parser is
    // where that recursion happens.
    Some(format!(
        r#"root ::= tool-call
tool-call ::= "<|tool_call>call:" tool-name "{{" arguments? "}}<tool_call|>"
tool-name ::= {name_alt}
arguments ::= argument ("," argument)*
argument ::= argument-key ":" argument-value
argument-key ::= [A-Za-z_][A-Za-z0-9_-]*
argument-value ::= argument-string | argument-number | "true" | "false" | argument-array | argument-object
argument-string ::= "<|\"|>" argument-chars "<|\"|>"
argument-chars ::= argument-char*
argument-char ::= [^<]
argument-number ::= "-"? [0-9]+ ("." [0-9]+)?
argument-array ::= "[" (argument-value ("," argument-value)*)? "]"
argument-object ::= "{{" (argument ("," argument)*)? "}}"
"#
    ))
}

// =============================================================================
// Tool Decoder
// =============================================================================

/// Streaming detector for `<|tool_call>call:NAME{..}<tool_call|>`.
///
/// Buffers *tokens* and re-decodes the whole buffer on every fire, for the same
/// reason `GenericChatDecoder` does: callers feed one token at a time, and a
/// byte-level BPE tokenizer can split both a multi-byte character and a
/// delimiter across token boundaries. Matching on the re-decoded prefix means a
/// half-arrived `<tool_call|>` simply is not found yet, and the transient
/// U+FFFD from a split character can never be mistaken for a closed call.
pub(super) struct Gemma4ToolDecoder {
    tokenizer: Arc<Tokenizer>,
    token_buf: Vec<u32>,
    /// The names actually declared to the model, from the equipped toolset.
    /// `None` means no toolset was declared: the decoder falls back to lexical
    /// best-effort (legacy) behaviour, attributing any lexically-valid call name.
    /// `Some(set)` enforces membership — a name outside the set is rejected as
    /// firmly as a malformed block, so a lexically valid identifier the model was
    /// never shown does not resurrect a call the caller never equipped.
    allowed: Option<BTreeSet<String>>,
    /// Set once a closed-but-unparseable block has been seen. Latching here is
    /// what makes rejection final: without it, continued generation could
    /// append a second, well-formed block and resurrect a call the model never
    /// validly made.
    rejected: bool,
}

impl Gemma4ToolDecoder {
    pub(super) fn new(tokenizer: Arc<Tokenizer>, allowed: Option<BTreeSet<String>>) -> Self {
        Self {
            tokenizer,
            token_buf: Vec::new(),
            allowed,
            rejected: false,
        }
    }
}

impl super::ToolDecoder for Gemma4ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> super::ToolEvent {
        use super::ToolEvent;

        if self.rejected {
            return ToolEvent::Start;
        }
        self.token_buf.extend_from_slice(tokens);
        let text = self.tokenizer.decode(&self.token_buf, false);

        let Some(open) = text.find(TOOL_CALL_OPEN) else {
            return ToolEvent::Start;
        };
        let body_start = open + TOOL_CALL_OPEN.len();
        // Truncated: the block has opened but not closed. Emit nothing and keep
        // buffering — a call is only ever reported once it is complete.
        let Some(offset) = text[body_start..].find(TOOL_CALL_CLOSE) else {
            return ToolEvent::Start;
        };

        match parse_tool_call(&text[body_start..body_start + offset]) {
            // A well-formed call, but only if its name is one the model was
            // actually shown. An undeclared name is a closed, final rejection
            // just like a malformed block — otherwise unconstrained generation
            // could have the decoder report a tool the caller never equipped.
            // `None` (no toolset declared) allows any lexically-valid name;
            // `Some(set)` enforces membership, and an empty set matches nothing.
            Some((name, arguments)) if self.allowed.as_ref().is_none_or(|a| a.contains(&name)) => {
                self.token_buf.clear();
                ToolEvent::Call(name, arguments)
            }
            _ => {
                self.rejected = true;
                ToolEvent::Start
            }
        }
    }

    fn reset(&mut self) {
        self.token_buf.clear();
        self.rejected = false;
    }
}

/// Parse a `call:NAME{arguments}` body into `(name, arguments-json)`. Returns
/// `None` for anything the DSL does not fully account for; callers treat that
/// as "no call was made".
fn parse_tool_call(body: &str) -> Option<(String, String)> {
    // No trimming: the grammar renders `call:NAME{..}` with no surrounding or
    // interstitial whitespace, so accepting any here would let the parser
    // recognise a call the constrained decoder could never have produced.
    let rest = body.strip_prefix("call:")?;
    let open = rest.find('{')?;
    let name = &rest[..open];
    // Validate the name against the same charset a declaration must satisfy,
    // independently of the grammar. The grammar pins names during constrained
    // generation, but the decoder is the component that decides a call was
    // made, so it fails closed on any name that could never have been declared
    // rather than trusting an upstream constraint to have been applied. The
    // charset excludes whitespace, so `get_weather ` and ` get_weather` are
    // rejected here rather than silently trimmed to a match.
    if !is_supported_tool_name(name) || !rest.ends_with('}') {
        return None;
    }
    let arguments = parse_arguments(&rest[open + 1..rest.len() - 1])?;
    Some((name.to_string(), Value::Object(arguments).to_string()))
}

/// Whether `text` is spelled exactly as `argument-number` admits:
/// `"-"? [0-9]+ ("." [0-9]+)?`.
///
/// Deliberately narrower than JSON. Exponent forms (`1e5`), a leading `+`, a
/// bare `.5`, and a trailing `1.` are all valid JSON numbers the call grammar
/// cannot produce, so recognising them would make the parser accept calls no
/// constrained generation could have emitted.
fn is_grammar_number(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let (integer, fraction) = match digits.split_once('.') {
        Some((integer, fraction)) => (integer, Some(fraction)),
        None => (digits, None),
    };
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    all_digits(integer) && fraction.is_none_or(all_digits)
}

/// Parse the comma-separated `key:value` body of a call's argument map.
fn parse_arguments(source: &str) -> Option<Map<String, Value>> {
    // Exactly empty, not whitespace-empty: the grammar's `{{ arguments? }}`
    // admits `{}` but never `{ }`, so a space between the braces is a call the
    // parser must not accept when the grammar could not have emitted it.
    if source.is_empty() {
        return Some(Map::new());
    }
    let mut parser = DslParser {
        source,
        pos: 0,
        depth: 0,
    };
    let arguments = parser.parse_members()?;
    // Trailing junk means the block was not what it claimed to be.
    if parser.pos != source.len() {
        return None;
    }
    Some(arguments)
}

/// Recursive-descent parser for the argument DSL. Every delimiter it slices on
/// is ASCII, so byte offsets stay on character boundaries even though values
/// may hold arbitrary UTF-8.
///
/// `depth` bounds the object/array nesting this will descend into. The input is
/// raw model output, so it never passes through `serde_json` and inherits none
/// of its recursion cap; without a bound of its own, a few kilobytes of
/// `{x:{x:{x:` — well inside any generation budget, and *legal* under the call
/// grammar — would exhaust the 2 MiB tokio worker stack. A Rust stack overflow
/// aborts the process rather than unwinding, so that would take the whole
/// daemon and every tenant on it down. Refusing past [`MAX_ARGUMENT_DEPTH`]
/// keeps an over-nested call a rejected call.
struct DslParser<'a> {
    source: &'a str,
    pos: usize,
    depth: usize,
}

impl<'a> DslParser<'a> {
    fn rest(&self) -> &'a str {
        &self.source[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn parse_members(&mut self) -> Option<Map<String, Value>> {
        let mut members = Map::new();
        loop {
            let key = self.parse_key()?;
            if self.peek()? != ':' {
                return None;
            }
            self.pos += 1;
            members.insert(key, self.parse_value()?);
            if self.peek() == Some(',') {
                self.pos += 1;
                continue;
            }
            return Some(members);
        }
    }

    fn parse_key(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == ':' {
                break;
            }
            if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                return None;
            }
            self.pos += c.len_utf8();
        }
        if self.pos == start {
            return None;
        }
        let key = &self.source[start..self.pos];
        // `argument-key ::= [A-Za-z_][A-Za-z0-9_-]*` — the loop above admits a
        // leading digit, which the grammar does not. Deferring to the same
        // predicate the declaration uses keeps the parser from recognising a
        // key no constrained generation could have produced.
        if !is_supported_argument_key(key) {
            return None;
        }
        Some(key.to_string())
    }

    /// An object or array body, entered with `depth` already incremented.
    fn parse_container(&mut self) -> Option<Value> {
        if self.rest().starts_with('{') {
            self.pos += 1;
            let members = if self.peek() == Some('}') {
                Map::new()
            } else {
                self.parse_members()?
            };
            if self.peek()? != '}' {
                return None;
            }
            self.pos += 1;
            return Some(Value::Object(members));
        }
        self.pos += 1;
        let mut items = Vec::new();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek()? {
                ',' => self.pos += 1,
                ']' => {
                    self.pos += 1;
                    return Some(Value::Array(items));
                }
                _ => return None,
            }
        }
    }

    fn parse_value(&mut self) -> Option<Value> {
        let rest = self.rest();

        if let Some(after) = rest.strip_prefix(QUOTE) {
            let end = after.find(QUOTE)?;
            let text = &after[..end];
            // `argument-char ::= [^<]` — the grammar cannot emit `<` inside a
            // string, and `quote` refuses to render one, so a decoded string
            // carrying it did not come from a constrained generation and could
            // not be round-tripped back into a declaration.
            if text.contains('<') {
                return None;
            }
            self.pos += QUOTE.len() + end + QUOTE.len();
            return Some(Value::String(text.to_string()));
        }
        if rest.starts_with('{') || rest.starts_with('[') {
            // Descending one level. Refusing here is the fail-closed backstop:
            // the grammar admits deeper nesting than this, so an over-nested
            // call is simply not reported as a call rather than crashing the
            // process on the way down.
            if self.depth == MAX_ARGUMENT_DEPTH {
                return None;
            }
            self.depth += 1;
            let value = self.parse_container()?;
            self.depth -= 1;
            return Some(value);
        }
        if rest.starts_with("true") {
            self.pos += 4;
            return Some(Value::Bool(true));
        }
        if rest.starts_with("false") {
            self.pos += 5;
            return Some(Value::Bool(false));
        }

        // Anything else must be a bare number; unquoted free text is not a
        // value the DSL can express, so reject rather than guess.
        let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
        let literal = &rest[..end];
        // `serde_json` alone is too permissive here: it accepts `1e5`, which
        // `argument-number ::= "-"? [0-9]+ ("." [0-9]+)?` cannot express. Check
        // the spelling against the grammar first, then parse.
        if !is_grammar_number(literal) {
            return None;
        }
        let literal: Value = serde_json::from_str(literal).ok()?;
        if !literal.is_number() {
            return None;
        }
        self.pos += end;
        Some(literal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instruct::ToolDecoder as _;
    use crate::model::instruct::ToolEvent;

    /// Char-level vocabulary: printable ASCII plus the Gemma 4 delimiters.
    /// Ordinary text tokenizes one character per token, so feeding token by
    /// token exercises the worst-case chunk boundary rather than a friendly
    /// one — every multi-character delimiter arrives split.
    fn tokenizer() -> Arc<Tokenizer> {
        let mut vocab: Vec<String> = [
            TOOL_OPEN,
            TOOL_CLOSE,
            TOOL_CALL_OPEN,
            TOOL_CALL_CLOSE,
            TOOL_RESPONSE_OPEN,
            TOOL_RESPONSE_CLOSE,
            QUOTE,
            "\n",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        vocab.extend((0x20u8..0x7f).map(|b| (b as char).to_string()));
        Arc::new(Tokenizer::from_vocab(&vocab))
    }

    /// Feed a rendered call one token at a time, mirroring how the inferlet
    /// drives the decoder, and collect every call reported. Builds the decoder
    /// with no declared toolset (`None`), exercising the legacy lexical path —
    /// these tests exercise parsing, not membership.
    fn calls_from(text: &str) -> Vec<(String, String)> {
        let tok = tokenizer();
        run_decoder(text, Gemma4ToolDecoder::new(tok.clone(), None), &tok)
    }

    /// Like [`calls_from`], but with an explicit declared toolset (`Some`), so it
    /// can exercise name membership. No grammar matcher is involved.
    fn calls_from_declared(text: &str, declared: &[&str]) -> Vec<(String, String)> {
        let tok = tokenizer();
        let allowed = declared.iter().map(|s| s.to_string()).collect();
        run_decoder(
            text,
            Gemma4ToolDecoder::new(tok.clone(), Some(allowed)),
            &tok,
        )
    }

    /// Drive `decoder` token-by-token over `text` and collect every call.
    fn run_decoder(
        text: &str,
        mut decoder: Gemma4ToolDecoder,
        tok: &Tokenizer,
    ) -> Vec<(String, String)> {
        tok.encode(text)
            .into_iter()
            .filter_map(|token| match decoder.feed(&[token]) {
                ToolEvent::Call(name, arguments) => Some((name, arguments)),
                ToolEvent::Start => None,
            })
            .collect()
    }

    #[test]
    fn the_decoder_reports_a_declared_call() {
        // A declared name, decoded directly — no grammar matcher involved.
        assert_eq!(
            calls_from_declared(
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                &["get_weather"],
            ),
            vec![("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn the_decoder_refuses_an_undeclared_call_name() {
        // `read_file` is well-formed and lexically valid; the only thing wrong
        // with it is that it was never declared. Membership, not the grammar,
        // rejects it, so unconstrained generation cannot smuggle a call to a
        // tool the caller never equipped.
        assert!(
            calls_from_declared(
                "<|tool_call>call:read_file{path:<|\"|>/tmp/a<|\"|>}<tool_call|>",
                &["get_weather"],
            )
            .is_empty()
        );
    }

    #[test]
    fn undeclared_call_rejection_is_final() {
        // The undeclared block latches rejection, so a later declared call
        // cannot resurrect it — the same finality a malformed block gets.
        assert!(
            calls_from_declared(
                "<|tool_call>call:read_file{}<tool_call|>\
                 <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                &["get_weather"],
            )
            .is_empty()
        );
    }

    #[test]
    fn an_empty_declared_set_reports_nothing() {
        // An unsupported toolset yields an empty membership set; the decoder
        // then attributes no call at all rather than one the model was never
        // shown. `declared_names` produces exactly this for a rejected toolset.
        assert!(declared_names(&["{not a schema".to_string()]).is_empty());
        assert!(
            calls_from_declared(
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                &[],
            )
            .is_empty()
        );
    }

    #[test]
    fn parses_a_call_fed_one_token_at_a_time() {
        assert_eq!(
            calls_from("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"),
            vec![("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn emits_a_call_exactly_once() {
        assert_eq!(
            calls_from("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>").len(),
            1
        );
    }

    #[test]
    fn is_chunk_boundary_agnostic() {
        let tok = tokenizer();
        let rendered = "<|tool_call>call:read_file{path:<|\"|>/tmp/a.txt<|\"|>}<tool_call|>";
        let tokens = tok.encode(rendered);
        let expected = vec![(
            "read_file".to_string(),
            r#"{"path":"/tmp/a.txt"}"#.to_string(),
        )];
        // Every possible two-chunk split must yield the same single call.
        for split in 0..=tokens.len() {
            let mut decoder = Gemma4ToolDecoder::new(tok.clone(), None);
            let seen = vec![
                decoder.feed(&tokens[..split]),
                decoder.feed(&tokens[split..]),
            ];
            let calls: Vec<(String, String)> = seen
                .into_iter()
                .filter_map(|event| match event {
                    ToolEvent::Call(name, arguments) => Some((name, arguments)),
                    ToolEvent::Start => None,
                })
                .collect();
            assert_eq!(calls, expected, "split at {split}");
        }
    }

    #[test]
    fn ignores_leading_text_before_the_call() {
        assert_eq!(
            calls_from("Hello<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
                .len(),
            1
        );
    }

    #[test]
    fn parses_scalar_argument_types() {
        assert_eq!(
            calls_from(
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,x:true,n:-12}<tool_call|>"
            ),
            vec![(
                "get_weather".to_string(),
                r#"{"city":"Paris","x":true,"n":-12}"#.to_string()
            )]
        );
    }

    #[test]
    fn parses_nested_arguments() {
        assert_eq!(
            calls_from(
                "<|tool_call>call:read_file{o:{city:<|\"|>Paris<|\"|>},x:[1,2]}<tool_call|>"
            ),
            vec![(
                "read_file".to_string(),
                r#"{"o":{"city":"Paris"},"x":[1,2]}"#.to_string()
            )]
        );
    }

    #[test]
    fn parses_empty_arguments() {
        assert_eq!(
            calls_from("<|tool_call>call:read_file{}<tool_call|>"),
            vec![("read_file".to_string(), "{}".to_string())]
        );
    }

    #[test]
    fn never_emits_a_truncated_call() {
        // Closing delimiter never arrives.
        assert!(calls_from("<|tool_call>call:get_weather{city:<|\"|>Paris").is_empty());
    }

    #[test]
    fn rejects_malformed_calls_without_emitting() {
        for malformed in [
            // Missing the `call:` marker.
            "<|tool_call>get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            // Empty tool name.
            "<|tool_call>call:{city:<|\"|>Paris<|\"|>}<tool_call|>",
            // Unterminated string value.
            "<|tool_call>call:get_weather{city:<|\"|>Paris}<tool_call|>",
            // Missing the argument brace.
            "<|tool_call>call:get_weather<tool_call|>",
            // Name is not a supported identifier; the decoder must not report
            // a call the model could never have been shown a declaration for.
            "<|tool_call>call:get weather{}<tool_call|>",
            // Unquoted free text is not a DSL value.
            "<|tool_call>call:get_weather{city:Paris}<tool_call|>",
            // Trailing junk after the argument map.
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,}<tool_call|>",
        ] {
            assert!(
                calls_from(malformed).is_empty(),
                "must not emit a call for {malformed:?}"
            );
        }
    }

    /// A call whose single argument's value nests `depth` objects deep.
    fn nested_call(depth: usize) -> String {
        let mut value = String::from("<|\"|>leaf<|\"|>");
        for _ in 0..depth {
            value = format!("{{x:{value}}}");
        }
        format!("{TOOL_CALL_OPEN}call:get_weather{{x:{value}}}{TOOL_CALL_CLOSE}")
    }

    #[test]
    fn the_parser_recognises_no_more_than_the_grammar_admits() {
        // The decoder is what decides a call was made, and a caller is free to
        // generate unconstrained (`native_grammar` returning `None` is a
        // documented fallback). So every shape the grammar cannot emit must be
        // refused by the parser on its own, without leaning on a matcher having
        // been built. Each case below is accepted by a *looser* reading — JSON
        // numbers, "any identifier char", "scan to the closing quote" — and is
        // exactly what a prompt-injected or unconstrained model could emit.
        for (label, body) in [
            // `argument-key ::= [A-Za-z_][A-Za-z0-9_-]*` — no leading digit.
            (
                "leading-digit key",
                "call:get_weather{1city:<|\"|>Paris<|\"|>}",
            ),
            ("all-digit key", "call:get_weather{42:<|\"|>Paris<|\"|>}"),
            // `argument-char ::= [^<]` — no `<` inside a string, which is what
            // stops a decoded value from re-entering the DSL as structure.
            (
                "'<' inside a string",
                "call:get_weather{city:<|\"|>a<b<|\"|>}",
            ),
            (
                "forged block inside a string",
                "call:get_weather{city:<|\"|>x<tool_call|><|\"|>}",
            ),
            // `argument-number ::= "-"? [0-9]+ ("." [0-9]+)?` — no exponents,
            // no leading `+`, no bare or trailing dot.
            ("exponent number", "call:get_weather{n:1e5}"),
            ("capital exponent number", "call:get_weather{n:1E5}"),
            ("negative exponent number", "call:get_weather{n:1e-5}"),
            ("leading-plus number", "call:get_weather{n:+1}"),
            ("bare-dot number", "call:get_weather{n:.5}"),
            ("trailing-dot number", "call:get_weather{n:1.}"),
        ] {
            assert!(
                parse_tool_call(body).is_none(),
                "{label}: parser must not recognise what the grammar cannot emit ({body:?})"
            );
        }
    }

    #[test]
    fn the_parser_still_accepts_the_shapes_the_grammar_does_admit() {
        // The guard above must not have narrowed the language the grammar
        // actually produces.
        for (label, body) in [
            (
                "underscore key",
                "call:get_weather{_city:<|\"|>Paris<|\"|>}",
            ),
            ("hyphen key", "call:get_weather{c-i_t-y2:<|\"|>Paris<|\"|>}"),
            ("integer", "call:get_weather{n:42}"),
            ("negative integer", "call:get_weather{n:-42}"),
            ("decimal", "call:get_weather{n:3.5}"),
            ("negative decimal", "call:get_weather{n:-0.25}"),
            ("booleans", "call:get_weather{a:true,b:false}"),
            ("empty string", "call:get_weather{city:<|\"|><|\"|>}"),
        ] {
            assert!(
                parse_tool_call(body).is_some(),
                "{label}: grammar-legal call must still parse ({body:?})"
            );
        }
    }

    #[test]
    fn the_parser_rejects_whitespace_the_grammar_cannot_emit() {
        // The grammar renders `call:NAME{k:v,..}` with no whitespace anywhere.
        // The parser must be no broader: each body below is only "wrong" in a
        // space the constrained decoder could never have produced.
        for (label, body) in [
            ("space after call:", "call: get_weather{}"),
            ("space before the brace", "call:get_weather {}"),
            ("leading space", " call:get_weather{}"),
            ("trailing space", "call:get_weather{} "),
            ("space inside empty arguments", "call:get_weather{ }"),
            (
                "space before the key colon",
                "call:get_weather{city :<|\"|>x<|\"|>}",
            ),
            (
                "space after the key colon",
                "call:get_weather{city: <|\"|>x<|\"|>}",
            ),
            (
                "space before the closing brace",
                "call:get_weather{city:<|\"|>x<|\"|> }",
            ),
        ] {
            assert!(
                parse_tool_call(body).is_none(),
                "{label}: parser must not recognise whitespace the grammar cannot emit ({body:?})"
            );
        }
    }

    #[test]
    fn nesting_within_the_depth_limit_is_parsed() {
        assert_eq!(calls_from(&nested_call(MAX_ARGUMENT_DEPTH)).len(), 1);
    }

    #[test]
    fn nesting_past_the_depth_limit_is_refused_without_overflowing() {
        // The value this defends: the parser reads raw model output on a 2 MiB
        // worker stack, and Rust aborts the process on stack overflow rather
        // than unwinding, so an unbounded version of this input is a daemon-
        // wide SIGABRT — not a rejected call. Depths far past anything a real
        // tool nests must come back as "no call", and must come back at all.
        //
        // Driven through `parse_tool_call` rather than the streaming decoder:
        // the decoder re-decodes its whole buffer per token by design, so
        // feeding it a 100k-deep payload is quadratic in the test and measures
        // the tokenizer rather than the recursion this bounds.
        for depth in [MAX_ARGUMENT_DEPTH + 1, 500, 5_000, 100_000] {
            let body = nested_call(depth)
                .trim_start_matches(TOOL_CALL_OPEN)
                .trim_end_matches(TOOL_CALL_CLOSE)
                .to_string();
            assert!(
                parse_tool_call(&body).is_none(),
                "depth {depth} must be refused, not parsed"
            );
        }
    }

    #[test]
    fn rejection_is_final() {
        // A malformed block must not be rescued by a later well-formed one.
        assert!(
            calls_from(
                "<|tool_call>call:{}<tool_call|>\
                 <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
            )
            .is_empty()
        );
    }

    #[test]
    fn reset_clears_buffer_and_rejection() {
        let tok = tokenizer();
        let mut decoder = Gemma4ToolDecoder::new(tok.clone(), None);
        for token in tok.encode("<|tool_call>call:{}<tool_call|>") {
            decoder.feed(&[token]);
        }
        decoder.reset();
        let calls: Vec<(String, String)> = tok
            .encode("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .into_iter()
            .filter_map(|token| match decoder.feed(&[token]) {
                ToolEvent::Call(name, arguments) => Some((name, arguments)),
                ToolEvent::Start => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn a_tool_response_is_not_mistaken_for_a_call() {
        assert!(calls_from(&response_block("get_weather", "sunny")).is_empty());
    }

    #[test]
    fn the_parser_accepts_exactly_what_the_grammar_admits() {
        // Both sides of the contract derive from the same validated toolset,
        // so a call the grammar can produce must parse, and one it cannot must
        // not be silently accepted.
        let schema = serde_json::json!({
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        })
        .to_string();
        let tools = validate_tools(&[schema]).expect("supported schema");
        let source = build_tool_call_grammar(&tools).expect("grammar source");
        assert!(source.contains(r#"tool-name ::= "get_weather""#));
        assert!(
            !calls_from("<|tool_call>call:get_weather{city:<|\"|>x<|\"|>}<tool_call|>").is_empty()
        );
    }

    #[test]
    fn strings_carrying_the_dsl_delimiter_fail_the_whole_toolset_closed() {
        // The call grammar forbids `<` inside any value (`argument-char ::=
        // [^<]`), so a schema string containing `<` cannot be declared,
        // constrained, and parsed identically. Every `<|"|>`-delimited position
        // must refuse it, taking the entire toolset down with it rather than
        // emitting a declaration the grammar can never satisfy.
        let with_delimiter = [
            // Tool description.
            serde_json::json!({"name": "f", "description": "a<b"}).to_string(),
            // Property description.
            serde_json::json!({
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string", "description": "a<b"}}}
            })
            .to_string(),
            // Enum choice.
            serde_json::json!({
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string", "enum": ["ok", "a<b"]}}}
            })
            .to_string(),
            // `required` name.
            serde_json::json!({
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["a<b"]}
            })
            .to_string(),
        ];
        for schema in with_delimiter {
            assert!(
                validate_tools(std::slice::from_ref(&schema)).is_none(),
                "must reject declaration for {schema}"
            );
        }
    }
}
