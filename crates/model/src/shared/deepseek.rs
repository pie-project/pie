//! DeepSeek's `<｜User｜>` / `<｜Assistant｜>` conversation format.
//!
//! Uses fullwidth Unicode delimiters for roles and tool calls.
//! Reference: DeepSeek R1 Jinja chat template with tool-calling support.
//!
//! DeepSeek-R1 wrote it down first, which is why it lived in
//! `deepseek_r1/chat.rs` — and the old `instruct::create` pointed
//! `"deepseek_v4"` at that one constructor from a table row, which is a
//! sibling edge the isolation rule forbids the moment it stops being a
//! table cell and becomes a row's own answer. Two generations, one
//! format, so the words are here and `deepseek_r1::chat` re-exports
//! them.
//!
//! The delimiters are FULLWIDTH (U+FF5C, not `|`), which is the whole
//! reason a V4 could not be handed ChatML instead: `<|im_end|>` and
//! `<｜end▁of▁sentence｜>` are different tokens, and a model sealed with
//! the wrong one generates past the end of its turn.

use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

// The implementation below mirrors the published DeepSeek-R1 jinja chat
// template; the verbatim copy that used to sit here as a static was never
// read — the checkpoint's own `chat_template` is the reference.

pub struct R1Instruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eos_ids: Vec<u32>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool tokens
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
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
            user_prefix: encode("<｜User｜>"),
            assistant_prefix: encode("<｜Assistant｜>"),
            eos_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_call_begin: encode("<｜tool▁call▁begin｜>"),
            tool_call_end: encode("<｜tool▁call▁end｜>"),
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
            decoder: self.tokenizer.decoder(false),
            tool_call_begin: self.tool_call_begin.clone(),
            tool_call_end: self.tool_call_end.clone(),
            accumulated: String::new(),
            inside: false,
            match_pos: 0,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        // Build an EBNF grammar that constrains generation to valid R1 tool calls.
        // Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
        //         ```json\n{...}\n```<｜tool▁call▁end｜>[more calls]<｜tool▁calls▁end｜>
        //
        // Extract function names from tool JSON schemas for the name alternation.
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool)
                && let Some(name) = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
            {
                names.push(format!("\"{}\"", name));
            }
        }
        if names.is_empty() {
            // This answers for an empty offer too, which is why there is no
            // separate check on `tools`: no tools means the walk pushes
            // nothing means no alternation, so a guard up top would decide
            // exactly what this one decides.
            //
            // `None` rather than the grammar with an empty `tool-name ::=`.
            // An empty alternation is not a rule matching nothing, it is a
            // rule the sampler cannot compile, and the fire carrying it
            // fails at the door instead of simply generating no call.
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
        Some(ToolGrammar { source: grammar })
    }
}

/// The delimiter between the call type and the function name.
///
/// Named once because three things have to agree on it: the system prompt
/// that asks for the format, the grammar that constrains generation to it,
/// and the decoder that takes it apart. The decoder used to split the
/// header on a newline instead, and since the header has no newline, every
/// tool call reported its name as `function<｜tool▁sep｜>the_real_name`.
const TOOL_SEP: &str = "<｜tool▁sep｜>";

// ─── Decoders ───────────────────────────────────────────────

struct R1ToolDecoder {
    decoder: TokenizerDecoder,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    accumulated: String,
    inside: bool,
    match_pos: usize,
}

impl ToolDecoder for R1ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
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
                                // `type<｜tool▁sep｜>name`, which is what
                                // `tool_call_grammar` constrains generation to
                                // and what `build_tool_system_prompt` asks
                                // for. Splitting on a newline instead left the
                                // whole header as the name, because there is
                                // no newline in it -- see the test.
                                let name = match header.rsplit_once(TOOL_SEP) {
                                    Some((_, name)) => name,
                                    // A model that omits the `function` prefix
                                    // gives a bare name, possibly on its own
                                    // line.
                                    None => header.rsplit_once('\n').map_or(header, |(_, n)| n),
                                };
                                return ToolEvent::Call(name.trim().to_string(), args);
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
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
        self.match_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruct::{ChatEvent, ReasoningEvent};
    use crate::shared::decoders::gbnf_sentence;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

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

    /// An offer can be non-empty and still name nothing.
    ///
    /// R1 reads only the nested `function.name` spelling, so a caller
    /// sending a flat `{"name": …}`, or a schema with no name at all, or
    /// text that is not JSON, contributes no alternative. If any of those
    /// left the grammar to be built anyway, `tool-name ::=` comes out with
    /// an empty body -- which the sampler rejects, so the fire fails
    /// rather than the model simply declining to call a tool.
    ///
    /// The good entry in the second case proves the walk SKIPS the bad
    /// ones rather than aborting on them: dropping the whole offer because
    /// one member of it was malformed would take the caller's other tools
    /// away.
    #[test]
    fn an_offer_that_names_nothing_constrains_nothing() {
        let inst = r1();
        for offered in [
            r#"{"name":"get_weather"}"#,
            r#"{"function":{"parameters":{}}}"#,
            r#"{"function":{"name":42}}"#,
            "not json at all",
        ] {
            assert!(
                inst.tool_call_grammar(&[offered.to_string()]).is_none(),
                "{offered:?} names no function, so there is no alternation \
                 to constrain generation with"
            );
        }

        let mixed = inst
            .tool_call_grammar(&[
                "not json at all".to_string(),
                r#"{"function":{"name":"get_weather"}}"#.to_string(),
            ])
            .expect("one good entry still yields a grammar");
        assert!(
            mixed.source.contains(r#"tool-name ::= "get_weather""#),
            "the unreadable entry is dropped without leaving an empty \
             alternative beside the good one, got {:?}",
            mixed.source
        );
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

    // ── The tool-call decoder ────────────────────────────────────────────

    /// A tokenizer whose vocabulary has the tool tokens as single ids, the
    /// way a real DeepSeek tokenizer does.
    fn tool_tok() -> Arc<Tokenizer> {
        make_tok(&[
            "<｜end▁of▁sentence｜>",
            "<|EOT|>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            TOOL_SEP,
        ])
    }

    /// Encode each part on its own and concatenate.
    ///
    /// `Tokenizer::from_vocab` matches a vocabulary entry only when it is
    /// the whole input; a longer string falls through to the byte fallback.
    /// So a marker spliced into a sentence does NOT come back as its own id,
    /// and a decoder that matches on ids would see nothing. Splitting the
    /// stream into the pieces a real tokenizer emits is what makes these
    /// tests exercise the id-matching path at all.
    fn stream(tok: &Tokenizer, parts: &[&str]) -> Vec<u32> {
        parts.iter().flat_map(|p| tok.encode(p)).collect()
    }

    /// Drive a decoder one token at a time, as a stream arrives, and collect
    /// the completed calls.
    fn decode_calls(inst: &R1Instruct, tok: &Tokenizer, parts: &[&str]) -> Vec<(String, String)> {
        let mut dec = inst.tool_decoder();
        let mut out = Vec::new();
        for id in stream(tok, parts) {
            if let ToolEvent::Call(name, args) = dec.feed(&[id]) {
                out.push((name, args));
            }
        }
        out
    }

    const CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
    const CALL_END: &str = "<｜tool▁call▁end｜>";

    /// The name is the function's, not the whole header.
    ///
    /// The regression: the decoder split the header on a newline, and the
    /// header -- `function<｜tool▁sep｜>get_weather` -- contains none, so
    /// `rsplit_once` returned `None` and the fallback handed back the whole
    /// string. Every DeepSeek-R1 and V4 tool call reported its name as
    /// `function<｜tool▁sep｜>get_weather`, which no caller dispatching on
    /// `get_weather` can match, so no tool ever ran.
    ///
    /// The file contradicted itself: `tool_call_grammar`, thirty lines up,
    /// defines `tool-name ::= "get_weather" | ...` and constrains generation
    /// to emit exactly the header this decoder could not take apart.
    #[test]
    fn a_tool_call_reports_the_function_name_alone() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let calls = decode_calls(
            &inst,
            &tok,
            &[
                "<｜tool▁calls▁begin｜>",
                CALL_BEGIN,
                "function",
                TOOL_SEP,
                "get_weather\n```json\n{\"city\": \"Seoul\"}\n```",
                CALL_END,
                "<｜tool▁calls▁end｜>",
            ],
        );
        assert_eq!(calls.len(), 1, "one call: {calls:?}");
        assert_eq!(calls[0].0, "get_weather", "the name, with no type prefix");
        assert_eq!(calls[0].1, "{\"city\": \"Seoul\"}");
    }

    /// Split `text` so every marker is a part of its own.
    ///
    /// The fixture tokenizer matches a vocabulary entry only when it is the
    /// whole input, so a sentence has to be broken at the markers before it
    /// can be encoded into the ids the decoder matches on.
    fn split_at_markers<'a>(text: &'a str, markers: &[&str]) -> Vec<&'a str> {
        let mut parts = Vec::new();
        let mut rest = text;
        'scan: while !rest.is_empty() {
            for offset in 0..rest.len() {
                if !rest.is_char_boundary(offset) {
                    continue;
                }
                if let Some(marker) = markers.iter().find(|m| rest[offset..].starts_with(**m)) {
                    if offset > 0 {
                        parts.push(&rest[..offset]);
                    }
                    parts.push(&rest[offset..offset + marker.len()]);
                    rest = &rest[offset + marker.len()..];
                    continue 'scan;
                }
            }
            parts.push(rest);
            break;
        }
        parts
    }

    /// The grammar constrains generation and the decoder reads the result
    /// back; nothing but this test binds them. llama-3 shipped a rule
    /// admitting text its own decoder rejected, so the binding is derived
    /// from the published grammar rather than transcribed by hand.
    #[test]
    fn the_grammar_admits_exactly_what_the_decoder_reads() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let grammar = inst
            .tool_call_grammar(&[r#"{"function":{"name":"get_weather"}}"#.to_string()])
            .expect("a tool was offered");
        let sentence = gbnf_sentence(
            &grammar.source,
            "root",
            &[("json-object", r#"{"city":"Seoul"}"#)],
        );
        let parts = split_at_markers(
            &sentence,
            &[
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁calls▁end｜>",
                CALL_BEGIN,
                CALL_END,
                TOOL_SEP,
            ],
        );
        let calls = decode_calls(&inst, &tok, &parts);
        assert_eq!(
            calls,
            [("get_weather".to_string(), r#"{"city":"Seoul"}"#.to_string())],
            "the grammar admits {sentence:?}, which the decoder did not read as that call"
        );
    }

    /// The separator the decoder splits on is the one the grammar emits.
    ///
    /// These are the two halves of the same protocol and they sat forty
    /// lines apart with the string written out twice.
    #[test]
    fn the_decoder_and_the_grammar_name_the_same_separator() {
        let inst = r1();
        let tools = vec![r#"{"function": {"name": "get_weather"}}"#.to_string()];
        let g = inst
            .tool_call_grammar(&tools)
            .expect("a grammar for one tool");
        assert!(
            g.source.contains(TOOL_SEP),
            "the grammar constrains generation to a separator the decoder does not split on"
        );
        let prompt = R1Instruct::build_tool_system_prompt(&tools);
        assert!(
            prompt.contains(TOOL_SEP),
            "and the system prompt asks the model for it too"
        );
        assert!(
            !Instruct::equip(&inst, &tools).is_empty(),
            "the prompt reaches the model as tokens"
        );
    }

    /// Two calls in one stream are reported separately.
    ///
    /// The decoder is a state machine over `inside`, and the reset of
    /// `match_pos` and `accumulated` on close is what lets a second call
    /// start. Without it the second call's content is appended to the
    /// first's and neither parses.
    #[test]
    fn two_calls_in_one_stream_are_both_reported() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let calls = decode_calls(
            &inst,
            &tok,
            &[
                "<｜tool▁calls▁begin｜>",
                CALL_BEGIN,
                "function",
                TOOL_SEP,
                "a\n```json\n{\"x\": 1}\n```",
                CALL_END,
                CALL_BEGIN,
                "function",
                TOOL_SEP,
                "b\n```json\n{\"y\": 2}\n```",
                CALL_END,
                "<｜tool▁calls▁end｜>",
            ],
        );
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0].0, "a");
        assert_eq!(calls[0].1, "{\"x\": 1}");
        assert_eq!(calls[1].0, "b");
        assert_eq!(calls[1].1, "{\"y\": 2}");
    }

    /// Ordinary prose reports no call.
    #[test]
    fn plain_text_produces_no_call() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        assert!(decode_calls(&inst, &tok, &["The weather in Seoul is fine."]).is_empty());
    }

    /// A call whose body never closes its fence produces nothing rather
    /// than a half-parsed argument string.
    #[test]
    fn an_unterminated_json_fence_yields_no_call() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let calls = decode_calls(
            &inst,
            &tok,
            &[
                CALL_BEGIN,
                "function",
                TOOL_SEP,
                "a\n```json\n{\"x\": 1",
                CALL_END,
            ],
        );
        assert!(calls.is_empty(), "{calls:?}");
    }

    /// A header with no `json` fence at all yields nothing.
    #[test]
    fn a_call_with_no_json_block_yields_no_call() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let calls = decode_calls(
            &inst,
            &tok,
            &[CALL_BEGIN, "function", TOOL_SEP, "a", CALL_END],
        );
        assert!(calls.is_empty(), "{calls:?}");
    }

    /// A model that omits the `function<sep>` prefix still dispatches.
    ///
    /// This is the branch the old newline split was presumably written for.
    /// It is kept as the fallback rather than dropped, and pinned here so
    /// that keeping it is a decision.
    #[test]
    fn a_bare_name_with_no_type_prefix_still_parses() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let calls = decode_calls(
            &inst,
            &tok,
            &[CALL_BEGIN, "get_weather\n```json\n{}\n```", CALL_END],
        );
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].0, "get_weather");
    }

    /// `reset` returns the decoder to its opening state.
    ///
    /// A decoder left `inside` after a reset treats the next stream's first
    /// tokens as the tail of a call that is over.
    #[test]
    fn reset_puts_the_decoder_back_at_the_start() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let mut dec = inst.tool_decoder();
        // Open a call and abandon it mid-body.
        for id in stream(&tok, &[CALL_BEGIN, "function", TOOL_SEP, "a\n```json\n{"]) {
            dec.feed(&[id]);
        }
        dec.reset();
        // A complete call now parses as if nothing had come before. The
        // call is deliberately written WITHOUT the `function<sep>` prefix:
        // with the prefix, a decoder still stuck `inside` produces the
        // right name anyway, because the separator split discards
        // everything before it including the leaked opening marker. A bare
        // name has nothing to discard behind, so a stuck decoder reports
        // `<｜tool▁call▁begin｜>b` and this test can see it.
        let mut got = None;
        for id in stream(&tok, &[CALL_BEGIN, "b\n```json\n{\"y\": 2}\n```", CALL_END]) {
            if let ToolEvent::Call(n, a) = dec.feed(&[id]) {
                got = Some((n, a));
            }
        }
        assert_eq!(
            got,
            Some(("b".to_string(), "{\"y\": 2}".to_string())),
            "the abandoned call leaked into the next one"
        );
    }

    /// The chat and reasoning decoders are wired to this family's tokens.
    #[test]
    fn the_chat_decoder_stops_on_this_familys_eos() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let mut dec = inst.chat_decoder();
        let mut text = String::new();
        let mut done = false;
        for id in stream(&tok, &["Hi", "<｜end▁of▁sentence｜>"]) {
            match dec.feed(&[id]) {
                ChatEvent::Delta(d) => text.push_str(&d),
                ChatEvent::Done(all) => {
                    done = true;
                    text = all;
                }
                ChatEvent::Interrupt(_) => {}
            }
        }
        assert!(done, "the sentence-end token must end the turn");
        assert_eq!(text, "Hi");
    }

    /// Thinking is reported as reasoning, and the block is closed off.
    #[test]
    fn the_reasoning_decoder_reports_the_thinking_block() {
        let tok = tool_tok();
        let inst = R1Instruct::new(tok.clone());
        let mut dec = inst.reasoning_decoder();
        let (mut started, mut complete) = (false, None);
        let mut delta = String::new();
        for id in stream(&tok, &["<think>\n", "hmm", "</think>\n", "Hello"]) {
            match dec.feed(&[id]) {
                ReasoningEvent::Start => started = true,
                ReasoningEvent::Delta(d) => delta.push_str(&d),
                ReasoningEvent::Complete(all) => complete = Some(all),
            }
        }
        assert!(started, "the think token opens a reasoning block");
        let complete = complete.expect("the closing token completes it");
        assert!(
            complete.contains("hmm"),
            "complete={complete:?} delta={delta:?}"
        );
        assert!(
            !complete.contains("Hello"),
            "the answer was reported as reasoning: {complete:?}"
        );
    }

    /// A decoder whose markers span several tokens.
    ///
    /// Built directly, because no tokenizer this crate can construct in a
    /// test will produce a multi-token marker with readable text:
    /// `Tokenizer::from_vocab` uses the `RawChar` pipeline, whose byte
    /// fallback is per-byte while its lookup is per-character, so a marker
    /// absent from the vocabulary does not fall back -- its non-ASCII
    /// characters are dropped on encode. `<｜tool▁sep｜>` goes in as 18
    /// bytes and comes back as the 9 ASCII bytes of `<toolsep>`. Decoding
    /// is unaffected, which is why feeding ids directly works.
    ///
    /// With the markers as single ids -- what a real DeepSeek vocabulary
    /// has -- `match_pos` only ever steps 0 -> 1 and neither rewind is
    /// reachable. A merged or trimmed vocabulary need not have them.
    fn multi_token_decoder(tok: &Arc<Tokenizer>) -> R1ToolDecoder {
        R1ToolDecoder {
            decoder: tok.decoder(false),
            tool_call_begin: vec![0, 1, 2], // <A><B><C>
            tool_call_end: vec![3, 4],      // <X><Y>
            accumulated: String::new(),
            inside: false,
            match_pos: 0,
        }
    }

    fn multi_token_vocab() -> Arc<Tokenizer> {
        make_tok(&[
            "<A>",
            "<B>",
            "<C>",
            "<X>",
            "<Y>",
            "<Z>",
            "function<｜tool▁sep｜>get_weather\n```json\n{\"n\": 1}",
            "\n```",
            "get_weather\n```json\n{\"n\": 1}",
        ])
    }

    fn drive(dec: &mut R1ToolDecoder, ids: &[u32]) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for &id in ids {
            if let ToolEvent::Call(n, a) = dec.feed(&[id]) {
                out.push((n, a));
            }
        }
        out
    }

    /// A near-miss on the opening marker is abandoned, not carried forward.
    ///
    /// `<A><B><Z>` is two thirds of the marker and then something else. A
    /// decoder that does not rewind is left at `match_pos == 2`, so the
    /// very next `<C>` completes a marker that was never sent -- it starts
    /// a call in the middle of prose, and the header it then accumulates
    /// begins with the leftover text.
    ///
    /// The call is written with a BARE name (token 8, not token 6): with a
    /// `function<｜tool▁sep｜>` header the separator split discards
    /// everything ahead of it, leftover marker text included, and the
    /// spurious start is invisible in the result.
    #[test]
    fn a_near_miss_on_the_opening_marker_does_not_carry_forward() {
        let tok = multi_token_vocab();
        let mut dec = multi_token_decoder(&tok);
        // near-miss, then the marker's last token on its own, then a real call.
        let calls = drive(&mut dec, &[0, 1, 5, 2, 0, 1, 2, 8, 7, 3, 4]);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1, "{\"n\": 1}");
    }

    /// A near-miss on the closing marker does not close the call.
    ///
    /// `<X><Z><Y>` is the end marker with something in the middle. A
    /// decoder that does not rewind is left at `match_pos == 1`, so the
    /// `<Y>` closes a call whose body has not arrived -- there is no json
    /// fence yet, so nothing is reported, and because `inside` is now false
    /// the rest of the call is read as prose. The tool call is lost
    /// entirely.
    #[test]
    fn a_near_miss_on_the_closing_marker_does_not_close_the_call() {
        let tok = multi_token_vocab();
        let mut dec = multi_token_decoder(&tok);
        let calls = drive(&mut dec, &[0, 1, 2, 3, 5, 4, 6, 7, 3, 4]);
        assert_eq!(calls.len(), 1, "the call was closed early: {calls:?}");
        assert_eq!(
            calls[0].0, "get_weather",
            "the separator split discards the near-miss text ahead of it"
        );
        assert_eq!(calls[0].1, "{\"n\": 1}");
    }
}
