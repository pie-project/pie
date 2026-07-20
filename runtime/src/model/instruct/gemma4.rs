//! Gemma 4 instruct implementation.
//!
//! Diverges from Gemma 2/3: Gemma 4 introduces a single-token turn
//! delimiter pair (`<|turn>` id 105 / `<turn|>` id 106) instead of the
//! multi-piece `<start_of_turn>` / `<end_of_turn>`.
//!
//! Gemma 4 also has a native system/developer block:
//!
//!   <bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n
//!
//! The native tool protocol — schema DSL, call grammar, and streaming decoder
//! — lives in [`tool_protocol`]. This file covers turn structure and how the
//! two compose:
//!
//! * Declarations belong *inside* the first system turn, so
//!   [`Instruct::system_equip`] is the only way to emit them. Standalone
//!   [`Instruct::equip`] cannot reach back into a turn `system` has already
//!   closed, so this architecture declares that it requires a system turn and
//!   the host rejects the standalone call rather than returning no
//!   declarations alongside a live tool grammar.
//! * `cue` is exactly `<|turn>model\n`. The template's `add_generation_prompt`
//!   emits that and nothing more when no tool response is outstanding, and
//!   emits *nothing at all* when resuming after one, because the model turn is
//!   still open. Continuation therefore appends no cue — see
//!   [`Instruct::cue`]. Ground truth is pinned in
//!   `gemma4_matches_pinned_e2b_revision`.
//! * `answer` always takes the template's scalar branch
//!   (`response:NAME{value:..}`); the trait hands us `value: &str`, so the
//!   mapping branch is not reachable without guessing at the caller's intent.

mod tool_protocol;

use crate::inference::structured::grammar::Grammar;
use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder};
use crate::model::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;
use tool_protocol::{
    Gemma4ToolDecoder, build_tool_call_grammar, declaration_block, declared_names, response_block,
    validate_tools,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    Gemma4,
    Gemma4Text,
}

pub struct Gemma4Instruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Gemma4Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::for_variant(tokenizer, Gemma4Variant::Gemma4)
    }

    pub fn for_variant(tokenizer: Arc<Tokenizer>, _variant: Gemma4Variant) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        // `<turn|>` (closing) + `<eos>` are both terminal — generation
        // stops at either. The runtime's `seal()` returns this list.
        let stop_strs = ["<turn|>", "<eos>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        // Gemma-4's tokenizer treats `<|turn>` and `<turn|>` as single
        // added tokens (ids 105 and 106 on the E2B vocab); `encode`
        // returns a 1-element vector for each. We assemble the
        // role-prefixes by token concatenation, matching how
        // GemmaInstruct does it for `<start_of_turn>`.
        let open_turn = encode("<|turn>");
        let close_turn = encode("<turn|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = open_turn.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = close_turn;
        turn_suffix.extend(&newline);

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("model"),
            turn_suffix,
            bos_token: encode("<bos>"),
            stop_ids,
            tokenizer,
        }
    }

    fn encode_trimmed(&self, message: &str) -> Vec<u32> {
        self.tokenizer.encode(message.trim())
    }
}

impl Instruct for Gemma4Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(system));
        v.extend(&self.turn_suffix);
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(user));
        v.extend(&self.turn_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.model_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn cue(&self) -> Vec<u32> {
        // The template's generation prompt, for the state this cue models.
        //
        // `add_generation_prompt` has three arms. With no tool response
        // outstanding it emits `<|turn>model\n` and nothing else — the empty
        // thought block is never emitted anywhere in the template, at any
        // point, and `enable_thinking` defaults to false. This implementation
        // exposes no reasoning channel, so that default is always in force.
        //
        // The other two arms both emit nothing once thinking is off, and they
        // cover resuming after a tool response or a tool call. That is why
        // continuation appends no cue at all: the model turn opened here is
        // still open across call -> response -> continuation, and reopening it
        // would produce a turn structure the template never generates. See
        // `gemma4_tool_use_multi_turn` for the full loop.
        self.model_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// Declarations nest inside the first system turn, which a standalone
    /// block cannot reach once `system` has closed it.
    ///
    /// Returning an empty vector here would be actively dangerous: production
    /// callers emit `system` then `equip` then build a matcher, so silently
    /// declaring nothing while `tool_call_grammar` still succeeds would
    /// constrain the model to call tools it was never shown. Declaring the
    /// requirement instead lets the host reject the call outright.
    fn tool_declarations_require_system_turn(&self) -> bool {
        true
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn system_equip(&self, system: &str, tools: &[String]) -> Option<Vec<u32>> {
        // Refuse the turn outright. Emitting a plain system turn here would
        // hand the caller a successful result carrying zero declarations, and
        // nothing downstream can recover from that: `tool_call_grammar`
        // refusing in step only helps a caller that chose to constrain
        // generation, and the SDK documents the unconstrained path as a
        // legitimate fallback. The caller would then dispatch tools the model
        // was never shown.
        let tools = validate_tools(tools)?;
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(system));
        for tool in &tools {
            v.extend(self.tokenizer.encode(&declaration_block(tool)));
        }
        v.extend(&self.turn_suffix);
        Some(v)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        self.tokenizer.encode(&response_block(name, value))
    }

    fn try_answer(&self, name: &str, value: &str) -> Option<Vec<u32>> {
        // A `<` in the value is unrepresentable: the DSL string delimiter has no
        // escape, so rendering it naively would let the value close the literal
        // early and forge structure. Refuse rather than emit a corrupted block.
        if value.contains('<') {
            return None;
        }
        Some(self.answer(name, value))
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        // No declared toolset (`None`): the decoder falls back to legacy lexical
        // best-effort, attributing any lexically-valid call name. Callers that
        // know their toolset build the membership-enforcing decoder via
        // `tool_decoder_for_tools`.
        Box::new(Gemma4ToolDecoder::new(self.tokenizer.clone(), None))
    }

    fn tool_decoder_for_tools(&self, tools: &[String]) -> Box<dyn ToolDecoder> {
        // The declared-tool decoder: it only reports a call whose name is one
        // of `tools`, so an undeclared name is refused independently of the
        // grammar — the fail-closed contract for unconstrained generation. An
        // unsupported toolset yields an empty membership set (reports nothing).
        Box::new(Gemma4ToolDecoder::new(
            self.tokenizer.clone(),
            Some(declared_names(tools)),
        ))
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        let tools = validate_tools(tools)?;
        let source = build_tool_call_grammar(&tools)?;
        let grammar = Grammar::from_ebnf(&source, "root").ok()?;
        Some(ToolGrammar {
            source,
            grammar: Arc::new(grammar),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::structured::compiled_grammar::CompiledGrammar;
    use crate::inference::structured::matcher::GrammarMatcher;

    /// The generation prompt the template emits with thinking disabled.
    const CUE: &str = "<|turn>model\n";

    /// The exact Gemma 4 revision this implementation was reconciled against.
    ///
    /// Pinned to a commit rather than a branch on purpose: `main` moves, and a
    /// protocol reverse-engineered from a moving template is unauditable. Every
    /// wire detail asserted in this file — delimiter spellings and ids, turn
    /// structure, and the generation prompt — was read from
    /// `chat_template.jinja` and `tokenizer.json` at this revision.
    #[allow(dead_code)]
    const PINNED_E2B_REVISION: &str =
        "google/gemma-4-E2B-it@179516f0c449474fdc46f08f30ead5b11e178497";

    /// `system_user` at [`PINNED_E2B_REVISION`], for the two messages in
    /// `gemma4_matches_pinned_e2b_revision`.
    const PINNED_SYSTEM_USER_IDS: &[u32] = &[
        2, 105, 9731, 107, 3048, 659, 496, 11045, 141657, 16326, 236761, 106, 107, 105, 2364, 107,
        6974, 496, 2822, 3925, 1003, 496, 16775, 236761, 568, 3932, 997, 236771, 236768, 106, 107,
    ];

    /// `cue` at [`PINNED_E2B_REVISION`]: the bare `<|turn>model\n`.
    /// `<|turn>` + `model` + `\n`. Three tokens: had the empty thought channel
    /// belonged here, this would carry `<|channel>`/`<channel|>` too.
    const PINNED_CUE_IDS: &[u32] = &[105, 4368, 107];

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn gemma4() -> Gemma4Instruct {
        let tok = make_tok(&[
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<bos>",
            "<|channel>",
            "<channel|>",
            "thought",
            "system",
            "user",
            "model",
            "\n",
            "Sys",
            "Hello",
            "Ok",
        ]);
        Gemma4Instruct::new(tok)
    }

    #[test]
    fn system_user_uses_native_system_turn_once() {
        let inst = gemma4();
        let mut tokens = inst.system_user("Sys", "Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<bos><|turn>system\nSys<turn|>\n<|turn>user\nHello<turn|>\n".to_string() + CUE
        );
    }

    #[test]
    fn first_user_starts_with_bos() {
        let inst = gemma4();
        let mut tokens = inst.first_user("Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<bos><|turn>user\nHello<turn|>\n".to_string() + CUE);
    }

    #[test]
    fn later_user_omits_bos() {
        let inst = gemma4();
        let tokens = inst.user("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>user\nHello<turn|>\n");
    }

    #[test]
    fn assistant_uses_model_role() {
        let inst = gemma4();
        let tokens = inst.assistant("Ok");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>model\nOk<turn|>\n");
    }

    // ─── Tool turn composition ───────────────────────────────

    /// Char-level vocabulary: every printable ASCII character, newline, and
    /// the Gemma 4 delimiters.
    ///
    /// Only a whole encoded piece hits the tokenizer's fast path, so ordinary
    /// text here tokenizes one character per token. That is deliberate — it
    /// makes the decoder tests exercise the worst case, where every
    /// multi-character delimiter arrives split across many `feed` calls.
    fn gemma4_tools() -> Gemma4Instruct {
        Gemma4Instruct::new(Arc::new(Tokenizer::from_vocab(&tool_vocab())))
    }

    fn tool_vocab() -> Vec<String> {
        let mut vocab: Vec<String> = [
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<bos>",
            "<|tool>",
            "<tool|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
            "<|channel>",
            "<channel|>",
            "\n",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        vocab.extend((0x20u8..0x7f).map(|b| (b as char).to_string()));
        vocab
    }

    fn weather_schema() -> String {
        serde_json::json!({
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Look up weather"}
                },
                "required": ["city"]
            }
        })
        .to_string()
    }

    /// Schemas Gemma 4 cannot declare, constrain, and parse identically. Each
    /// must take down the whole toolset, not just itself.
    fn unsupported_schemas() -> Vec<(&'static str, String)> {
        let with_property = |property: serde_json::Value| {
            serde_json::json!({
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": property}
                }
            })
            .to_string()
        };
        vec![
            ("not json", "{not json".to_string()),
            (
                "no name",
                serde_json::json!({"description": "no name"}).to_string(),
            ),
            ("empty name", serde_json::json!({"name": ""}).to_string()),
            (
                "non-string name",
                serde_json::json!({"name": 7}).to_string(),
            ),
            (
                "name would break the DSL",
                serde_json::json!({"name": "get{weather"}).to_string(),
            ),
            (
                "name would break the EBNF",
                serde_json::json!({"name": "get\"weather"}).to_string(),
            ),
            (
                // Without `type` the parameters block never closes.
                "parameters without type",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {"properties": {"city": {"type": "string"}}}
                })
                .to_string(),
            ),
            (
                "parameters not an object",
                serde_json::json!({"name": "get_weather", "parameters": "object"}).to_string(),
            ),
            (
                // The template has a `response:` branch this module does not
                // implement, so rendering without it would understate the
                // contract the model was given.
                "response schema",
                serde_json::json!({
                    "name": "get_weather",
                    "response": {"type": "object", "description": "forecast"}
                })
                .to_string(),
            ),
            (
                // `bad.key` renders fine but can never satisfy `argument-key`,
                // so the model could not generate a call using it.
                "property key outside argument-key",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"bad.key": {"type": "string"}}
                    }
                })
                .to_string(),
            ),
            (
                // The template silently drops a property named like schema
                // metadata; a short declaration is a silent alteration.
                "property named like schema metadata",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"required": {"type": "string"}}
                    }
                })
                .to_string(),
            ),
            (
                "property without type",
                with_property(serde_json::json!({})),
            ),
            (
                "property with non-string type",
                with_property(serde_json::json!({"type": 7})),
            ),
            (
                "property not an object",
                with_property(serde_json::json!("string")),
            ),
            (
                "non-string description",
                with_property(serde_json::json!({"type": "string", "description": 7})),
            ),
            (
                "non-bool nullable",
                with_property(serde_json::json!({"type": "string", "nullable": "yes"})),
            ),
            (
                "enum holding a null",
                with_property(serde_json::json!({"type": "string", "enum": ["a", null]})),
            ),
            (
                "array items not an object",
                with_property(serde_json::json!({"type": "array", "items": "string"})),
            ),
            (
                "array item type neither string nor union",
                with_property(serde_json::json!({"type": "array", "items": {"type": 7}})),
            ),
            (
                "nested object property key outside argument-key",
                with_property(serde_json::json!({
                    "type": "object",
                    "properties": {"bad.key": {"type": "string"}}
                })),
            ),
            (
                "nested object property without type",
                with_property(serde_json::json!({
                    "type": "object",
                    "properties": {"inner": {}}
                })),
            ),
            (
                "required not an array",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": "city"
                    }
                })
                .to_string(),
            ),
            (
                "required holding a non-string",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city", 7]
                    }
                })
                .to_string(),
            ),
            (
                // A description carrying `<` cannot be rendered as a DSL string
                // literal (the grammar forbids `<` in any value), so it would
                // desync the declaration from the grammar.
                "description carrying the DSL delimiter",
                serde_json::json!({"name": "get_weather", "description": "a<b"}).to_string(),
            ),
            (
                "property description carrying the DSL delimiter",
                with_property(serde_json::json!({"type": "string", "description": "a<b"})),
            ),
            (
                "enum value carrying the DSL delimiter",
                with_property(serde_json::json!({"type": "string", "enum": ["ok", "a<b"]})),
            ),
            (
                "required name carrying the DSL delimiter",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["ci<ty"]
                    }
                })
                .to_string(),
            ),
            (
                // A function-level key this module never renders would drop
                // part of the declared contract if accepted.
                "unrecognised function-level key",
                serde_json::json!({"name": "get_weather", "strict": true}).to_string(),
            ),
            (
                // `response:` is a template branch this module does not render.
                "function response branch",
                serde_json::json!({
                    "name": "get_weather",
                    "response": {"type": "object", "description": "forecast"}
                })
                .to_string(),
            ),
            (
                // A parameters-level key outside {type, properties, required}.
                "unrecognised parameters-level key",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {"city": {"type": "string"}}
                    }
                })
                .to_string(),
            ),
            (
                // `enum` is rendered only for a string; on a number it would be
                // silently dropped.
                "enum on a non-string property",
                with_property(serde_json::json!({"type": "number", "enum": [1, 2, 3]})),
            ),
            (
                // `items` is rendered only for an array.
                "items on a non-array property",
                with_property(serde_json::json!({"type": "string", "items": {"type": "string"}})),
            ),
            (
                // `properties`/`required` are rendered only for an object.
                "properties on a non-object property",
                with_property(serde_json::json!({
                    "type": "string",
                    "properties": {"inner": {"type": "string"}}
                })),
            ),
            (
                "required on a non-object property",
                with_property(serde_json::json!({"type": "string", "required": ["inner"]})),
            ),
            (
                // Recursion: a type-mismatched keyword one level down, inside an
                // array's item schema, must also take the toolset down.
                "nested item keyword mismatched to its type",
                with_property(serde_json::json!({
                    "type": "array",
                    "items": {"type": "string", "properties": {"x": {"type": "string"}}}
                })),
            ),
            (
                // Envelope discriminator says something other than `function`;
                // accepting it would silently reinterpret the inner object.
                "envelope discriminator is not function",
                serde_json::json!({
                    "type": "not_function",
                    "function": {"name": "get_weather"}
                })
                .to_string(),
            ),
            (
                // Envelope with no discriminator at all.
                "envelope missing the discriminator",
                serde_json::json!({"function": {"name": "get_weather"}}).to_string(),
            ),
            (
                // A property type outside the domain Gemma 4 can render.
                "property type outside the supported domain",
                with_property(serde_json::json!({"type": "widget"})),
            ),
            (
                // A parameters type outside the domain.
                "parameters type outside the supported domain",
                serde_json::json!({
                    "name": "get_weather",
                    "parameters": {
                        "type": "widget",
                        "properties": {"city": {"type": "string"}}
                    }
                })
                .to_string(),
            ),
            (
                // In-domain but not an object: parameters must be the argument
                // object itself, so a scalar/array type is refused.
                "parameters type is a string, not an object",
                serde_json::json!({"name": "get_weather", "parameters": {"type": "string"}})
                    .to_string(),
            ),
            (
                "parameters type is an array, not an object",
                serde_json::json!({"name": "get_weather", "parameters": {"type": "array"}})
                    .to_string(),
            ),
            (
                // An array item type outside the domain.
                "array item type outside the supported domain",
                with_property(serde_json::json!({"type": "array", "items": {"type": "widget"}})),
            ),
        ]
    }

    #[test]
    fn standalone_equip_is_declared_unsupported_rather_than_silently_empty() {
        // The regression that matters: production emits system, then equip,
        // then builds a matcher. If equip could return empty while the grammar
        // still compiled, the model would be constrained to call tools it was
        // never shown. The capability flag is what lets the host refuse.
        let inst = gemma4_tools();
        assert!(inst.tool_declarations_require_system_turn());
        assert!(inst.equip(&[weather_schema()]).is_empty());
        assert!(
            inst.tool_call_grammar(&[weather_schema()]).is_some(),
            "grammar is live, so empty declarations must be refused at the host"
        );
    }

    #[test]
    fn system_equip_splices_declarations_into_the_first_system_turn() {
        // Exact full prompt: one system turn carrying both the system message
        // and the declarations, then the user turn and the generation cue.
        let inst = gemma4_tools();
        let mut tokens = inst
            .system_equip("You are helpful.", &[weather_schema()])
            .expect("supported toolset");
        tokens.extend(inst.user("Weather?"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<bos><|turn>system\nYou are helpful.\
             <|tool>declaration:get_weather{description:<|\"|>Look up weather<|\"|>,\
             parameters:{properties:{city:{description:<|\"|>Look up weather<|\"|>,\
             type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],\
             type:<|\"|>OBJECT<|\"|>}}<tool|>\
             <turn|>\n\
             <|turn>user\nWeather?<turn|>\n\
             <|turn>model\n"
        );
    }

    #[test]
    fn system_equip_emits_exactly_one_system_turn() {
        let inst = gemma4_tools();
        let text = inst.tokenizer.decode(
            &inst
                .system_equip("You are helpful.", &[weather_schema()])
                .expect("supported toolset"),
            false,
        );
        assert_eq!(text.matches("<|turn>system").count(), 1);
        assert_eq!(text.matches("<bos>").count(), 1);
    }

    #[test]
    fn system_equip_without_tools_matches_a_plain_system_turn() {
        let inst = gemma4_tools();
        assert_eq!(
            inst.system_equip("You are helpful.", &[]).unwrap(),
            inst.system("You are helpful.")
        );
    }

    #[test]
    fn system_equip_accepts_the_openai_function_envelope() {
        let inst = gemma4_tools();
        let flat = inst
            .tokenizer
            .decode(&inst.system_equip("S", &[weather_schema()]).unwrap(), false);
        let wrapped_schema = serde_json::json!({
            "type": "function",
            "function": serde_json::from_str::<serde_json::Value>(&weather_schema()).unwrap()
        })
        .to_string();
        let wrapped = inst
            .tokenizer
            .decode(&inst.system_equip("S", &[wrapped_schema]).unwrap(), false);
        assert_eq!(flat, wrapped);
    }

    #[test]
    fn unsupported_schema_fails_the_whole_toolset_closed() {
        let inst = gemma4_tools();
        for (label, schema) in unsupported_schemas() {
            assert!(
                inst.system_equip("S", std::slice::from_ref(&schema))
                    .is_none(),
                "{label}: must refuse the turn rather than declare nothing"
            );
            assert!(
                inst.tool_call_grammar(std::slice::from_ref(&schema))
                    .is_none(),
                "{label}: must not produce a grammar"
            );

            // Alongside a valid schema — the valid one must not survive.
            let mixed = vec![weather_schema(), schema];
            assert!(
                inst.system_equip("S", &mixed).is_none(),
                "{label}: a valid sibling must not be declared"
            );
            assert!(
                inst.tool_call_grammar(&mixed).is_none(),
                "{label}: a valid sibling must not be constrained"
            );
        }
    }

    #[test]
    fn duplicate_tool_names_are_rejected() {
        // A repeated name makes the grammar alternation redundant and leaves
        // an observed call ambiguous as to which schema it satisfied.
        let inst = gemma4_tools();
        let duplicated = vec![weather_schema(), weather_schema()];
        assert!(
            inst.system_equip("S", &duplicated).is_none(),
            "duplicate names must refuse the turn"
        );
        assert!(inst.tool_call_grammar(&duplicated).is_none());
    }

    #[test]
    fn declaration_and_grammar_agree_on_supportedness() {
        // The invariant that makes fail-closed meaningful: a tool is never
        // declared without also being constrained, or vice versa.
        let inst = gemma4_tools();
        let supported = vec![weather_schema()];
        assert!(
            inst.tokenizer
                .decode(&inst.system_equip("S", &supported).unwrap(), false)
                .contains("declaration:get_weather"),
        );
        assert!(inst.tool_call_grammar(&supported).is_some());

        for (label, schema) in unsupported_schemas() {
            let declared = inst
                .system_equip("S", std::slice::from_ref(&schema))
                .is_some();
            let constrained = inst
                .tool_call_grammar(std::slice::from_ref(&schema))
                .is_some();
            assert_eq!(declared, constrained, "{label}");
        }
    }

    #[test]
    fn supported_nested_shapes_render_and_constrain() {
        // The branches this module does implement must survive validation, so
        // fail-closed does not quietly become fail-everything.
        let inst = gemma4_tools();
        let schema = serde_json::json!({
            "name": "search",
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "slow"]},
                    "tags": {"type": "array", "items": {"type": ["string", "number"]}},
                    "opts": {
                        "type": "object",
                        "properties": {"deep": {"type": "string"}},
                        "required": ["deep"]
                    },
                    "limit": {"type": "number", "nullable": true}
                },
                "required": ["mode"]
            }
        })
        .to_string();
        let text = inst.tokenizer.decode(
            &inst
                .system_equip("S", std::slice::from_ref(&schema))
                .expect("supported toolset"),
            false,
        );
        assert!(text.contains("enum:[<|\"|>fast<|\"|>,<|\"|>slow<|\"|>]"));
        assert!(text.contains("items:{type:[<|\"|>STRING<|\"|>,<|\"|>NUMBER<|\"|>]}"));
        assert!(text.contains("properties:{deep:{type:<|\"|>STRING<|\"|>}}"));
        assert!(text.contains("nullable:true"));
        assert!(inst.tool_call_grammar(&[schema]).is_some());
    }

    #[test]
    fn cue_is_the_bare_model_turn() {
        // The pinned template's `add_generation_prompt` emits `<|turn>model\n`
        // alone in this state. It never emits an empty thought block anywhere,
        // and `enable_thinking` defaults to false, so nothing else belongs
        // here. An extra prefix would be a generation-prompt state the model
        // was never trained on.
        let inst = gemma4_tools();
        assert_eq!(inst.tokenizer.decode(&inst.cue(), false), CUE);
        assert!(
            !inst
                .tokenizer
                .decode(&inst.cue(), false)
                .contains("channel")
        );
    }

    #[test]
    fn cue_is_identical_with_and_without_tools() {
        let with_tools = gemma4_tools();
        let plain = gemma4();
        assert_eq!(
            with_tools.tokenizer.decode(&with_tools.cue(), false),
            plain.tokenizer.decode(&plain.cue(), false)
        );
    }

    #[test]
    fn seal_is_unchanged_by_the_tool_path() {
        // Generation on the tool path terminates via ToolEvent::Call, so the
        // stop set stays exactly the chat stop set.
        let inst = gemma4_tools();
        assert_eq!(inst.seal(), inst.stop_ids);
    }

    #[test]
    fn answer_emits_inline_tool_response_without_a_user_turn() {
        let inst = gemma4_tools();
        let text = inst
            .tokenizer
            .decode(&inst.answer("get_weather", "sunny"), false);
        assert_eq!(
            text,
            "<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>"
        );
        assert!(!text.contains("<|turn>user"));
        assert!(!text.contains("<turn|>"));
    }

    #[test]
    fn answer_defaults_a_missing_name_to_unknown() {
        let inst = gemma4_tools();
        let text = inst.tokenizer.decode(&inst.answer("", "sunny"), false);
        assert!(text.contains("response:unknown{"));
    }

    #[test]
    fn tool_call_grammar_none_without_tools() {
        assert!(gemma4_tools().tool_call_grammar(&[]).is_none());
    }

    #[test]
    fn tool_call_grammar_compiles_and_pins_tool_names() {
        let inst = gemma4_tools();
        let grammar = inst
            .tool_call_grammar(&[weather_schema()])
            .expect("gemma4 must support constrained tool calling");
        assert!(grammar.source.contains(r#"tool-name ::= "get_weather""#));
        assert!(grammar.source.contains(r#""<|tool_call>call:""#));
        assert!(grammar.source.contains(r#""}<tool_call|>""#));
        assert!(Grammar::from_ebnf(&grammar.source, "root").is_ok());
    }

    // ─── Matcher + decoder integration ───────────────────────

    /// Build the matcher exactly as the host's `tool_use::create_matcher`
    /// does: compile the instruct-provided grammar for this tokenizer, then
    /// pair it with `seal()` as the stop set.
    fn native_matcher(inst: &Gemma4Instruct, tools: &[String]) -> GrammarMatcher {
        let grammar = inst
            .tool_call_grammar(tools)
            .expect("gemma4 must support constrained tool calling");
        let compiled =
            CompiledGrammar::get_or_compile(&grammar.source, &grammar.grammar, &inst.tokenizer);
        GrammarMatcher::with_compiled(compiled, inst.tokenizer.clone(), inst.seal(), 10)
    }

    #[test]
    fn native_matcher_accepts_a_well_formed_call_and_the_decoder_reports_it() {
        // The full constrained-generation path: every token of a rendered
        // call is accepted by the compiled grammar, the matcher only permits
        // termination once the call has closed, and the decoder fed the same
        // stream reports exactly that call.
        let inst = gemma4_tools();
        let tools = vec![weather_schema()];
        let mut matcher = native_matcher(&inst, &tools);
        let mut decoder = inst.tool_decoder_for_tools(&tools);

        let rendered = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
        let tokens = inst.tokenizer.encode(rendered);
        let mut reported = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            assert!(
                matcher.accept_token(*token),
                "grammar rejected token {index} of a well-formed call"
            );
            assert!(
                !matcher.can_terminate() || index + 1 == tokens.len(),
                "grammar allowed termination mid-call at token {index}"
            );
            if let ToolEvent::Call(name, arguments) = decoder.feed(&[*token]) {
                reported.push((name, arguments));
            }
        }

        assert!(
            matcher.can_terminate(),
            "grammar must permit termination once the call closes"
        );
        assert_eq!(
            reported,
            vec![("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn native_matcher_rejects_an_undeclared_tool_name() {
        // Name pinning is what keeps the model from calling something the
        // system turn never declared.
        let inst = gemma4_tools();
        let mut matcher = native_matcher(&inst, &[weather_schema()]);
        let accepted = inst
            .tokenizer
            .encode("<|tool_call>call:read_file{}<tool_call|>")
            .into_iter()
            .all(|token| matcher.accept_token(token));
        assert!(!accepted, "grammar must reject an undeclared tool name");
    }

    #[test]
    fn native_matcher_rejects_an_unquoted_argument_value() {
        // The decoder refuses unquoted free text; the grammar must make it
        // ungeneratable in the first place, so the two agree.
        let inst = gemma4_tools();
        let mut matcher = native_matcher(&inst, &[weather_schema()]);
        let accepted = inst
            .tokenizer
            .encode("<|tool_call>call:get_weather{city:Paris}<tool_call|>")
            .into_iter()
            .all(|token| matcher.accept_token(token));
        assert!(!accepted, "grammar must reject an unquoted argument value");
    }

    #[test]
    fn native_matcher_is_unavailable_for_an_unsupported_toolset() {
        // `create_matcher` errors when `tool_call_grammar` is None, which is
        // the loud failure the fail-closed path relies on.
        let inst = gemma4_tools();
        for (label, schema) in unsupported_schemas() {
            assert!(
                inst.tool_call_grammar(&[schema]).is_none(),
                "{label}: matcher construction must fail closed"
            );
        }
    }

    /// Drive a decoder over a rendered call, token by token, and return the
    /// single call it reports, if any.
    fn decode_one(
        inst: &Gemma4Instruct,
        decoder: &mut Box<dyn ToolDecoder>,
        rendered: &str,
    ) -> Option<(String, String)> {
        let mut got = None;
        for token in inst.tokenizer.encode(rendered) {
            if let ToolEvent::Call(name, arguments) = decoder.feed(&[token]) {
                got = Some((name, arguments));
            }
        }
        got
    }

    #[test]
    fn no_toolset_decoder_is_legacy_lexical() {
        // The legacy, no-toolset path (`tool_decoder` / host create-decoder /
        // SDK Decoder::new) has no declared set, so it falls back to lexical
        // best-effort: any well-formed, lexically-valid call name is reported.
        let inst = gemma4_tools();
        let mut decoder = inst.tool_decoder();
        assert_eq!(
            decode_one(
                &inst,
                &mut decoder,
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            ),
            Some(("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string())),
            "a decoder built without a toolset reports any lexically-valid call"
        );
    }

    #[test]
    fn declared_tool_decoder_reports_declared_and_refuses_undeclared() {
        // The useful, declared path reports a call to an equipped tool and
        // still refuses one to a tool it was never shown — no grammar involved.
        let inst = gemma4_tools();
        let tools = vec![weather_schema()];
        let mut declared = inst.tool_decoder_for_tools(&tools);
        assert_eq!(
            decode_one(
                &inst,
                &mut declared,
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            ),
            Some(("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string())),
        );
        let mut undeclared = inst.tool_decoder_for_tools(&tools);
        assert_eq!(
            decode_one(
                &inst,
                &mut undeclared,
                "<|tool_call>call:read_file{}<tool_call|>"
            ),
            None,
            "an undeclared name must be refused even on the declared path"
        );
    }

    #[test]
    fn gemma4_wire_vocabulary_matches_the_pinned_e2b_revision() {
        // Every delimiter this protocol puts on the wire, paired with its id in
        // the pinned revision's vocabulary (see PINNED_E2B_REVISION).
        //
        // The fixture tokenizer used by the rest of these tests defines these
        // same strings as added tokens, which makes it self-consistent with the
        // implementation and therefore unable to catch a wrong spelling. This
        // table is the outside reference that breaks that circularity: it was
        // read off the pinned revision's tokenizer.json, so changing a
        // delimiter here fails against real ground truth rather than against
        // our own fixture.
        let pinned: [(&str, u32); 11] = [
            ("<bos>", 2),
            ("<eos>", 1),
            ("<|turn>", 105),
            ("<turn|>", 106),
            ("<|tool>", 46),
            ("<tool|>", 47),
            ("<|tool_call>", 48),
            ("<tool_call|>", 49),
            ("<|tool_response>", 50),
            ("<tool_response|>", 51),
            ("<|\"|>", 52),
        ];
        // The protocol's own constants, so a rename cannot drift from the table.
        assert!(pinned.iter().any(|(s, _)| *s == tool_protocol::TOOL_OPEN));
        assert!(pinned.iter().any(|(s, _)| *s == tool_protocol::TOOL_CLOSE));
        assert!(
            pinned
                .iter()
                .any(|(s, _)| *s == tool_protocol::TOOL_CALL_OPEN)
        );
        assert!(
            pinned
                .iter()
                .any(|(s, _)| *s == tool_protocol::TOOL_CALL_CLOSE)
        );
        assert!(
            pinned
                .iter()
                .any(|(s, _)| *s == tool_protocol::TOOL_RESPONSE_OPEN)
        );
        assert!(
            pinned
                .iter()
                .any(|(s, _)| *s == tool_protocol::TOOL_RESPONSE_CLOSE)
        );
        assert!(pinned.iter().any(|(s, _)| *s == tool_protocol::QUOTE));

        // Ids must be distinct: two delimiters sharing one id would make the
        // decoder unable to tell a call from a response.
        let mut ids: Vec<u32> = pinned.iter().map(|(_, id)| *id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "delimiter ids must be distinct");
    }

    /// Full-prompt parity against the real pinned tokenizer.
    ///
    /// `#[ignore]` on purpose. This needs the pinned revision on disk, and the
    /// version of this test it replaces returned early when the model was
    /// absent — so it reported success while asserting nothing, on every
    /// machine and CI runner that did not happen to have the weights cached.
    /// A test that cannot run must say so rather than pass. Run it with the
    /// tokenizer path in `GEMMA4_E2B_TOKENIZER`:
    ///
    /// ```text
    /// GEMMA4_E2B_TOKENIZER=/path/to/tokenizer.json \
    ///   cargo test -p pie --lib gemma4_matches_pinned_e2b_revision -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the pinned google/gemma-4-E2B-it tokenizer; set GEMMA4_E2B_TOKENIZER"]
    fn gemma4_matches_pinned_e2b_revision() {
        let path = std::env::var("GEMMA4_E2B_TOKENIZER")
            .expect("set GEMMA4_E2B_TOKENIZER to the pinned revision's tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(std::path::Path::new(&path)).expect("load pinned tokenizer");
        let inst = Gemma4Instruct::new(Arc::new(tokenizer));

        // Turn structure and message encoding against the real vocab.
        assert_eq!(
            inst.system_user(
                "You are a helpful benchmarking assistant.",
                "Write a short story about a robot. (Request #0)",
            ),
            PINNED_SYSTEM_USER_IDS
        );

        // The generation prompt is the bare model turn. The template's
        // `add_generation_prompt` emits `<|turn>model\n` and nothing else in
        // this state, and never emits an empty thought block anywhere.
        assert_eq!(inst.tokenizer.decode(&inst.cue(), false), CUE);
        assert_eq!(inst.cue(), PINNED_CUE_IDS);
    }
}
