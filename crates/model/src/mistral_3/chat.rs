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

use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, keep_tail};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

// The implementation below mirrors the published Mistral V3 jinja chat
// template; the verbatim copy that used to sit here as a static was never
// read — the checkpoint's own `chat_template` is the reference.

// =============================================================================
// MistralInstruct
// =============================================================================

pub struct MistralInstruct {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    // Delimiters
    inst_start: Vec<u32>,
    inst_end: Vec<u32>,
    sys_start: Vec<u32>,
    sys_end: Vec<u32>,
    tools_start: Vec<u32>,
    tools_end: Vec<u32>,
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

        let inst_start = encode("[INST]");
        let inst_end = encode("[/INST]");

        Self {
            stop_ids,
            inst_start,
            inst_end,
            sys_start: encode("[SYSTEM_PROMPT]"),
            sys_end: encode("[/SYSTEM_PROMPT]"),
            tools_start: encode("[AVAILABLE_TOOLS]"),
            tools_end: encode("[/AVAILABLE_TOOLS]"),
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
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(MistralToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            state: ToolState::Outside,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
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
        // The ONE early return, and it answers for an empty offer too: no
        // tools means no names means no alternation. A `tools.is_empty()`
        // guard above the walk used to say the same thing three lines
        // earlier, which is one more place for the two answers to differ
        // and none for them to differ usefully -- a control that deleted
        // it could not be told from the original.
        //
        // `None` and not `Some` of an empty alternation: `tool-name ::=`
        // with no body is not a rule that matches nothing, it is a grammar
        // the sampler cannot compile, and the fire carrying it fails at
        // the door rather than sampling freely.
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
        Some(ToolGrammar { source: grammar })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    InsideName,
    /// Carries the name already read off the `[TOOL_CALLS]…[ARGS]` header.
    /// Holding it in the state rather than beside it is what makes "there
    /// is a name whenever we are reading arguments" a fact rather than a
    /// convention with an unreachable branch guarding it.
    InsideArgs(String),
}

struct MistralToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    state: ToolState,
}

impl ToolDecoder for MistralToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        self.accumulated =
                            self.accumulated[pos + "[TOOL_CALLS]".len()..].to_string();
                        self.state = ToolState::InsideName;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        self.accumulated = keep_tail(&self.accumulated, 50).to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::InsideName => {
                    if let Some(pos) = self.accumulated.find("[ARGS]") {
                        let name = self.accumulated[..pos].trim().to_string();
                        self.accumulated = self.accumulated[pos + "[ARGS]".len()..].to_string();
                        self.state = ToolState::InsideArgs(name);
                        continue;
                    }
                    return ToolEvent::Start; // Wait for ARGS
                }
                ToolState::InsideArgs(ref name) => {
                    // Check for next marker
                    let mut end_pos = None;

                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        end_pos = Some(pos);
                    } else if let Some(pos) = self.accumulated.find("</s>") {
                        end_pos = Some(pos);
                    }

                    if let Some(pos) = end_pos {
                        let name = name.clone();
                        let args = self.accumulated[..pos].trim().to_string();
                        // The terminator stays put: a second `[TOOL_CALLS]`
                        // both ends this call and opens the next one.
                        self.accumulated = self.accumulated[pos..].to_string();
                        self.state = ToolState::Outside;
                        return ToolEvent::Call(name, args);
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.state = ToolState::Outside;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn mistral() -> MistralInstruct {
        let tok = make_tok(&[
            "<s>",
            "</s>",
            " ",
            "[INST]",
            "[/INST]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_CALLS]",
            "[ARGS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "Hello",
            "world",
            "func",
            "{",
            "}",
            ":",
            "\"",
            "arg",
            "name",
            "f",
            "Hi",
            "result",
            ",",
            "[",
            "]",
            "INST",
            "SYSTEM_PROMPT",
            "AVAILABLE_TOOLS",
            "TOOL_CALLS",
            "ARGS",
            "TOOL_RESULTS",
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
        assert_eq!(text, "[INST]Hi[/INST]");
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
        assert!(g.source.contains("tool-call ::= \"[TOOL_CALLS]\""));
        assert!(g.source.contains("foo"));
    }

    #[test]
    fn full_conversation() {
        let inst = mistral();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hi"));
        tokens.extend(inst.user("Hi"));
        tokens.extend(inst.assistant("Hi"));
        tokens.extend(inst.user("Hi"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "[SYSTEM_PROMPT]Hi[/SYSTEM_PROMPT]\
             [INST]Hi[/INST]\
             Hi</s>\
             [INST]Hi[/INST]"
        );
    }

    #[test]
    fn tool_decoder_parses_call() {
        let inst = mistral();
        let mut dec = inst.tool_decoder();
        dec.feed(&[9]); // [TOOL_CALLS]
        dec.feed(&[22]); // "f"
        dec.feed(&[10]); // [ARGS]
        let event = dec.feed(&[16, 17, 1]); // "{}" + "</s>"
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::shared::decoders::gbnf_sentence;

    /// The syllable has to be enumerated: `Pipeline::RawChar` looks up
    /// whole characters but falls back per byte, so a non-ASCII character
    /// absent from the vocabulary is dropped on encode.
    fn inst() -> MistralInstruct {
        let v: Vec<String> = [
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_CALLS]",
            "[ARGS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "가",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        MistralInstruct::new(Arc::new(Tokenizer::from_vocab(&v)))
    }

    fn stream(inst: &MistralInstruct, d: &mut dyn ToolDecoder, parts: &[&str]) -> Vec<ToolEvent> {
        parts
            .iter()
            .map(|p| d.feed(&inst.tokenizer.encode(p)))
            .collect()
    }

    fn last_call(events: &[ToolEvent]) -> Option<(String, String)> {
        events.iter().rev().find_map(|e| match e {
            ToolEvent::Call(n, a) => Some((n.clone(), a.clone())),
            ToolEvent::Start => None,
        })
    }

    /// The grammar constrains generation and the decoder reads the result
    /// back; nothing but this test binds them. llama-3 shipped a rule
    /// admitting text its own decoder rejected, so the binding is checked
    /// rather than assumed.
    #[test]
    fn the_grammar_admits_exactly_what_the_decoder_reads() {
        let inst = inst();
        let grammar = inst
            .tool_call_grammar(&[r#"{"function":{"name":"get_weather"}}"#.to_string()])
            .expect("a tool was offered");
        let sentence = gbnf_sentence(
            &grammar.source,
            "tool-call",
            &[("json-object", r#"{"city":"Oslo"}"#)],
        );
        assert_eq!(sentence, r#"[TOOL_CALLS]get_weather[ARGS]{"city":"Oslo"}"#);

        let mut decoder = inst.tool_decoder();
        // The arguments run to the next marker, so the turn has to end.
        let events = stream(&inst, decoder.as_mut(), &[&sentence, "</s>"]);
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), r#"{"city":"Oslo"}"#.into())),
            "the grammar admits {sentence:?}, which the decoder did not read as that call"
        );
    }

    /// Offering no tools tokenizes to nothing, and offering some does not.
    ///
    /// `equip` writes the `[AVAILABLE_TOOLS]` block into the prompt. With
    /// an empty list the early return is the whole behaviour: without it
    /// the model is handed an empty JSON array between the two markers,
    /// which reads as "these are the tools, and there are none" rather
    /// than as a turn with no tool block at all -- a distinction mistral's
    /// template makes and its training reflects.
    #[test]
    fn no_tools_offered_writes_no_tool_block() {
        let inst = inst();
        assert!(
            inst.equip(&[]).is_empty(),
            "an empty offer is no block, not an empty one"
        );
        let some = inst.equip(&[r#"{"name":"f"}"#.to_string()]);
        assert!(
            some.starts_with(&inst.tools_start[..]) && some.ends_with(&inst.tools_end[..]),
            "a non-empty offer is wrapped in the two markers"
        );
    }

    /// A grammar is offered only when a NAME was found, and the two ways
    /// to have none are different.
    ///
    /// `tool-name ::= {alternatives}` is the one rule the caller's tools
    /// contribute to. With no alternatives the rule's body is empty, which
    /// is not a GBNF rule that matches nothing -- it is a grammar the
    /// sampler cannot compile, and the fire that carries it fails at the
    /// door rather than generating unconstrained. So an offer that yields
    /// no names has to come back `None`, meaning "sample freely", and not
    /// `Some` of an empty alternation.
    ///
    /// Both ways to get there are asked: a string that is not JSON at all,
    /// and one that is JSON with no `name` anywhere the two spellings
    /// look.
    #[test]
    fn an_offer_that_names_nothing_constrains_nothing() {
        let inst = inst();
        assert!(
            inst.tool_call_grammar(&[]).is_none(),
            "no tools offered is no alternation, answered by the same line \
             that answers a nameless offer"
        );
        for offer in [
            "not json at all",
            r#"{"function":{"description":"no name here"}}"#,
            r#"{"arguments":{}}"#,
            "[]",
            "42",
        ] {
            assert!(
                inst.tool_call_grammar(&[offer.to_string()]).is_none(),
                "{offer} names no tool, so there is no alternation to constrain to"
            );
        }
    }

    /// A nameless offer beside a named one is DROPPED, not fatal.
    ///
    /// The walk skips what it cannot name and keeps going, so one
    /// malformed entry in a list does not cost the caller the constraint
    /// on the rest. The grammar that comes back must mention the name it
    /// found and nothing else -- an empty alternative left in the rule
    /// (`"f" | `) is the same uncompilable grammar as an empty rule.
    #[test]
    fn a_nameless_offer_beside_a_named_one_is_dropped_and_the_rest_still_binds() {
        let inst = inst();
        let grammar = inst
            .tool_call_grammar(&[
                "not json".to_string(),
                r#"{"function":{"name":"get_weather"}}"#.to_string(),
                r#"{"description":"nameless"}"#.to_string(),
            ])
            .expect("one of the three names a tool");
        let rule = grammar
            .source
            .lines()
            .find(|l| l.starts_with("tool-name ::="))
            .expect("the grammar states a tool-name rule");
        assert_eq!(
            rule, r#"tool-name ::= "get_weather""#,
            "the two nameless offers must leave no empty alternative behind"
        );
    }

    /// Both spellings of a name are read, and the nested one wins.
    ///
    /// OpenAI's schema nests the name under `function`; some callers send
    /// it flat. Reading only one spelling would silently drop half the
    /// callers' tools from the alternation, which is the same outcome as
    /// offering no grammar for them.
    #[test]
    fn a_name_is_read_from_either_spelling() {
        let inst = inst();
        for (offer, want) in [
            (r#"{"function":{"name":"nested"}}"#, "nested"),
            (r#"{"name":"flat"}"#, "flat"),
            // Both present: `function.name` is the schema's, and the flat
            // one is the fallback -- reading the fallback first would take
            // the wrong name off a well-formed OpenAI tool.
            (r#"{"name":"flat","function":{"name":"nested"}}"#, "nested"),
        ] {
            let grammar = inst
                .tool_call_grammar(&[offer.to_string()])
                .expect("the offer names a tool");
            assert!(
                grammar
                    .source
                    .contains(&format!(r#"tool-name ::= "{want}""#)),
                "{offer} should name {want}: {}",
                grammar.source
            );
        }
    }

    #[test]
    fn the_test_tokenizer_round_trips_the_markers() {
        let inst = inst();
        for part in ["[TOOL_CALLS]", "[ARGS]", "</s>", "가"] {
            let ids = inst.tokenizer.encode(part);
            assert_eq!(inst.tokenizer.decode(&ids, false), part, "part {part:?}");
        }
    }

    /// Regression: the buffer bound sliced at a byte offset computed by
    /// arithmetic, which lands inside a character whenever the model has
    /// been writing anything but ASCII.
    #[test]
    fn a_long_non_ascii_preamble_does_not_panic() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        let prose = format!("{}{}", "x".repeat(190), "가".repeat(20));
        assert!(!prose.is_char_boundary(prose.len() - 50));
        stream(&inst, d.as_mut(), &[&prose]);
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "[TOOL_CALLS]",
                "get_weather",
                "[ARGS]",
                r#"{"city":"Oslo"}"#,
                "</s>",
            ],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), r#"{"city":"Oslo"}"#.into()))
        );
    }

    #[test]
    fn the_window_keeps_a_marker_split_by_the_trim() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        let mut over = "y".repeat(240);
        over.push_str("[TOOL_");
        stream(&inst, d.as_mut(), &[&over]);
        let events = stream(&inst, d.as_mut(), &["CALLS]", "f", "[ARGS]", "{}", "</s>"]);
        assert_eq!(last_call(&events).map(|c| c.0), Some("f".into()));
    }

    /// A second `[TOOL_CALLS]` closes the previous call's arguments, so
    /// the marker has to survive into the next round of the search.
    #[test]
    fn a_second_call_closes_the_first_and_is_itself_decoded() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "[TOOL_CALLS]",
                "first",
                "[ARGS]",
                r#"{"a":1}"#,
                "[TOOL_CALLS]",
                "second",
                "[ARGS]",
                r#"{"b":2}"#,
                "</s>",
            ],
        );
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Call(n, a) => Some((n.as_str(), a.as_str())),
                ToolEvent::Start => None,
            })
            .collect();
        assert_eq!(calls, [("first", r#"{"a":1}"#), ("second", r#"{"b":2}"#)]);
    }

    #[test]
    fn the_name_may_arrive_across_several_steps() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "[TOOL_CALLS]",
                "get_",
                "weat",
                "her",
                "[ARGS]",
                "{}",
                "</s>",
            ],
        );
        assert_eq!(last_call(&events).map(|c| c.0), Some("get_weather".into()));
    }

    #[test]
    fn whitespace_around_the_name_and_arguments_is_dropped() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "[TOOL_CALLS]",
                "\n get_weather \n",
                "[ARGS]",
                "  {} ",
                "</s>",
            ],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), "{}".into()))
        );
    }

    #[test]
    fn reset_abandons_a_call_in_progress() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        stream(&inst, d.as_mut(), &["[TOOL_CALLS]", "f", "[ARGS]", "{}"]);
        d.reset();
        // Only a decoder still holding the call open would close it here.
        let after = stream(&inst, d.as_mut(), &["</s>"]);
        assert_eq!(last_call(&after), None);
    }

    /// The state alone is not enough: a half-received opening marker left
    /// in the buffer would complete itself against the next request.
    #[test]
    fn reset_forgets_buffered_text() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        stream(&inst, d.as_mut(), &["thinking about it [TOOL_"]);
        d.reset();
        let after = stream(&inst, d.as_mut(), &["CALLS]", "f", "[ARGS]", "{}", "</s>"]);
        assert_eq!(last_call(&after), None);
    }

    /// The name belongs to the state that reads arguments, so abandoning
    /// that state cannot leave a name behind for the next call to inherit.
    #[test]
    fn a_new_call_cannot_inherit_the_previous_name() {
        let inst = inst();
        let mut d = inst.tool_decoder();
        stream(
            &inst,
            d.as_mut(),
            &["[TOOL_CALLS]", "stale", "[ARGS]", "{}", "</s>"],
        );
        let events = stream(
            &inst,
            d.as_mut(),
            &["[TOOL_CALLS]", "fresh", "[ARGS]", "{}", "</s>"],
        );
        assert_eq!(last_call(&events).map(|c| c.0), Some("fresh".into()));
    }
}
