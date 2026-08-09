//! OLMo 3 instruct implementation.
//!
//! Implements OLMo 3 chat template:
//! - ChatML-style: <|im_start|>role\ncontent<|im_end|>\n
//! - Tools defined in <functions>...</functions> within system/user messages.
//! - Tool calls in <function_calls>...</function_calls> within assistant messages.
//! - Tool outputs in <|im_start|>environment\ncontent<|im_end|>\n.
//! - Generation prompt adds <|im_start|>assistant\n<think>

use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent};
use crate::shared::decoders::{GenericChatDecoder, ThinkingDecoder, keep_tail};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

// The implementation below mirrors the published OLMo-3 jinja chat
// template; the verbatim copy that used to sit here as a static was never
// read — the checkpoint's own `chat_template` is the reference.

pub struct OlmoInstruct {
    tokenizer: Arc<Tokenizer>,
    im_start: Vec<u32>,
    im_end: Vec<u32>,
    newline: Vec<u32>,
    // Roles
    system_role: Vec<u32>,
    user_role: Vec<u32>,
    assistant_role: Vec<u32>,
    environment_role: Vec<u32>,
    // Tools
    // Generation
    think_start: Vec<u32>,
    think_end: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl OlmoInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");
        let eos_token = encode("<|endoftext|>");

        let mut stop_ids = im_end.clone();
        stop_ids.extend(&eos_token);

        Self {
            im_start,
            im_end,
            newline,
            system_role: encode("system"),
            user_role: encode("user"),
            assistant_role: encode("assistant"),
            environment_role: encode("environment"),
            think_start: encode("<think>"),
            think_end: encode("</think>"),
            stop_ids,
            tokenizer,
        }
    }

    fn wrap(&self, role: &[u32], content: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(content));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }
}

impl Instruct for OlmoInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.system_role, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_role, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(&self.think_start);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let preamble = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024. ";
        let mut msg = preamble.to_string();
        msg.push_str("<functions>");
        msg.push_str(&tools.join("\n"));
        msg.push_str("</functions>");

        self.system(&msg)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.wrap(&self.environment_role, value)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        // The same list `seal()` publishes, not just the turn end. The guest
        // stops sampling on `seal()` and reads its text out of this decoder,
        // so a token that is terminal to one and ordinary to the other ends
        // generation without ever producing `Done` -- and an inferlet that
        // loops until `Done`, which is how the surface reads, loses the last
        // message of every conversation the model chose to end with
        // `<|endoftext|>` rather than `<|im_end|>`.
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        // Starts inside because cue() includes <think>; empty start_ids = starts inside
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![],
            self.think_end.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(OlmoToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            state: ToolState::Outside,
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct OlmoToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    state: ToolState,
}

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    Inside,
}

impl ToolDecoder for OlmoToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("<function_calls>") {
                        self.accumulated =
                            self.accumulated[pos + "<function_calls>".len()..].to_string();
                        self.state = ToolState::Inside;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        self.accumulated = keep_tail(&self.accumulated, 50).to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::Inside => {
                    if let Some(pos) = self.accumulated.find("</function_calls>") {
                        let content = self.accumulated[..pos].trim().to_string();
                        self.accumulated =
                            self.accumulated[pos + "</function_calls>".len()..].to_string();
                        self.state = ToolState::Outside;

                        // The block holds either a single call object or a
                        // list of them; OLMo emits the list form, but a
                        // bare object costs nothing to accept.
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                            let call = match val.as_array() {
                                Some(arr) => arr.first(),
                                None if val.is_object() => Some(&val),
                                None => None,
                            };
                            // Indexing a `Value` yields `Null` for a missing
                            // key, but indexing the `Map` behind it panics,
                            // so the lookup stays on the `Value`.
                            if let Some(call) = call
                                && let Some(name) = call["name"].as_str()
                            {
                                let args = call["arguments"].to_string();
                                return ToolEvent::Call(name.to_string(), args);
                            }
                        }
                        // Unparseable, or parsed but naming no function: a
                        // call no dispatcher could route is not a call.
                        return ToolEvent::Start;
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

    #[test]
    fn system_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "system", "Hello"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.system("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>system\nHello<|im_end|>\n"));
    }

    #[test]
    fn equip_format() {
        // Build the exact content string that equip() will encode so the
        // mock tokenizer's fast-path recognizes it as a single token.
        let tools = &["foo".to_string(), "bar".to_string()];
        let preamble = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024. ";
        let content = format!("{}<functions>{}</functions>", preamble, tools.join("\n"));
        let mut vocab: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "<functions>",
            "</functions>",
            "foo",
            "bar",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        vocab.push(content);
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.equip(tools);
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<functions>"));
        assert!(text.contains("foo"));
        assert!(text.contains("bar"));
        assert!(text.contains("</functions>"));
    }

    #[test]
    fn answer_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "environment", "result"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.answer("fn", "result");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>environment\nresult<|im_end|>\n"));
    }

    #[test]
    fn generation_cue_includes_think() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "assistant", "<think>"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.cue();
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>assistant\n<think>"));
    }

    fn olmo() -> OlmoInstruct {
        let tok = make_tok(&[
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "Hello",
            "user",
            "assistant",
            "environment",
            "<|endoftext|>",
            "<functions>",
            "</functions>",
            "<function_calls>",
            "</function_calls>",
            "<think>",
            "</think>",
        ]);
        OlmoInstruct::new(tok)
    }

    #[test]
    fn full_conversation() {
        let inst = olmo();
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
             <|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn answer_uses_environment_role() {
        let inst = olmo();
        let tokens = inst.answer("fn", "Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|im_start|>environment\nHello<|im_end|>\n");
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;
    use crate::instruct::{ChatEvent, ReasoningEvent};

    /// A vocabulary carrying the markers plus one non-ASCII syllable.
    ///
    /// The syllable has to be enumerated: `Pipeline::RawChar` looks up
    /// whole characters but falls back per byte, so a non-ASCII character
    /// absent from the vocabulary is dropped on encode rather than split.
    fn tok() -> Arc<Tokenizer> {
        let v: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "user",
            "assistant",
            "environment",
            "<|endoftext|>",
            "<function_calls>",
            "</function_calls>",
            "<think>",
            "</think>",
            "가",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// Feed each piece as its own step, so the split points are the ones
    /// the test names rather than whatever the tokenizer chooses.
    fn stream(inst: &OlmoInstruct, d: &mut dyn ToolDecoder, parts: &[&str]) -> Vec<ToolEvent> {
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

    /// Guard for every test below: the decoder must actually receive the
    /// text these tests spell out. A vocabulary that silently dropped a
    /// marker would leave the tool tests asserting nothing.
    #[test]
    fn the_test_tokenizer_round_trips_the_markers() {
        let inst = OlmoInstruct::new(tok());
        for part in [
            "<function_calls>",
            "</function_calls>",
            r#"[{"name":"get_weather","arguments":{"city":"Oslo"}}]"#,
            "가",
        ] {
            let ids = inst.tokenizer.encode(part);
            assert_eq!(inst.tokenizer.decode(&ids, false), part, "part {part:?}");
        }
    }

    // ─── The call itself ─────────────────────────────────────

    #[test]
    fn a_call_arrives_as_a_list_of_one() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"[{"name":"get_weather","arguments":{"city":"Oslo"}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), r#"{"city":"Oslo"}"#.into()))
        );
        // Only the closing marker resolves the call.
        assert!(matches!(events[0], ToolEvent::Start));
        assert!(matches!(events[1], ToolEvent::Start));
    }

    #[test]
    fn a_bare_object_is_accepted_too() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"{"name":"get_time","arguments":{}}"#,
                "</function_calls>",
            ],
        );
        assert_eq!(last_call(&events), Some(("get_time".into(), "{}".into())));
    }

    /// Regression: indexing the `Map` behind a `Value` panics on a missing
    /// key, so an argument-less call took the process down.
    #[test]
    fn a_call_with_no_arguments_key_does_not_panic() {
        let inst = OlmoInstruct::new(tok());
        for body in [r#"{"name":"get_time"}"#, r#"[{"name":"get_time"}]"#] {
            let mut d = inst.tool_decoder();
            let events = stream(
                &inst,
                d.as_mut(),
                &["<function_calls>", body, "</function_calls>"],
            );
            assert_eq!(
                last_call(&events),
                Some(("get_time".into(), "null".into())),
                "body {body:?}"
            );
        }
    }

    /// A `Call` whose name is empty is a call no dispatcher can route, so
    /// it is not reported as one.
    #[test]
    fn a_call_naming_no_function_is_not_a_call() {
        let inst = OlmoInstruct::new(tok());
        for body in [
            r#"{"arguments":{"city":"Oslo"}}"#,
            r#"[{"arguments":{}}]"#,
            r#"{"name":7}"#,
            "[]",
            "\"just a string\"",
            "17",
        ] {
            let mut d = inst.tool_decoder();
            let events = stream(
                &inst,
                d.as_mut(),
                &["<function_calls>", body, "</function_calls>"],
            );
            assert_eq!(last_call(&events), None, "body {body:?}");
        }
    }

    #[test]
    fn a_block_that_is_not_json_reports_nothing() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &["<function_calls>", "get_weather(Oslo)", "</function_calls>"],
        );
        assert_eq!(last_call(&events), None);
    }

    // ─── Framing ─────────────────────────────────────────────

    #[test]
    fn prose_before_the_block_is_ignored() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "Let me look that up for you. ",
                "<function_calls>",
                r#"[{"name":"get_weather","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(
            last_call(&events),
            Some(("get_weather".into(), "{}".into()))
        );
    }

    #[test]
    fn a_whole_block_delivered_in_one_step_still_resolves() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[r#"<function_calls>[{"name":"f","arguments":{"a":1}}]</function_calls>"#],
        );
        assert_eq!(last_call(&events), Some(("f".into(), r#"{"a":1}"#.into())));
    }

    #[test]
    fn two_blocks_in_a_row_both_resolve() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let first = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"[{"name":"first","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(last_call(&first).map(|c| c.0), Some("first".into()));
        let second = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"[{"name":"second","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(last_call(&second).map(|c| c.0), Some("second".into()));
    }

    #[test]
    fn text_trailing_the_close_is_carried_into_the_next_search() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"[{"name":"f","arguments":{}}]"#,
                r#"</function_calls>ok<function_calls>[{"name":"g","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        // The first block resolves on step 3, the second on step 4 —
        // the leftover after the close was not discarded.
        assert_eq!(last_call(&events[..3]).map(|c| c.0), Some("f".into()));
        assert_eq!(last_call(&events[3..]).map(|c| c.0), Some("g".into()));
    }

    #[test]
    fn reset_abandons_a_block_in_progress() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        stream(&inst, d.as_mut(), &["<function_calls>", r#"[{"name":"f","#]);
        d.reset();
        // A complete, well-formed body and its close: only a decoder that
        // still held the block open would resolve one out of it.
        let after = stream(
            &inst,
            d.as_mut(),
            &[r#"[{"name":"f","arguments":{}}]"#, "</function_calls>"],
        );
        assert_eq!(last_call(&after), None);
    }

    /// The state alone is not enough: a half-received opening marker left
    /// in the buffer would complete itself against the next request.
    #[test]
    fn reset_forgets_buffered_text() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        stream(&inst, d.as_mut(), &["thinking about it <function_c"]);
        d.reset();
        let after = stream(
            &inst,
            d.as_mut(),
            &[r#"alls>[{"name":"f","arguments":{}}]"#, "</function_calls>"],
        );
        assert_eq!(last_call(&after), None);
    }

    // ─── The sliding window ──────────────────────────────────

    /// Regression: the buffer bound sliced at a byte offset computed by
    /// arithmetic, which lands inside a character whenever the model has
    /// been writing anything but ASCII.
    #[test]
    fn a_long_non_ascii_preamble_does_not_panic() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        // 190 ASCII bytes then 20 three-byte syllables: the cut at
        // len() - 50 falls strictly inside the fourth of them.
        let prose = format!("{}{}", "x".repeat(190), "가".repeat(20));
        assert!(!prose.is_char_boundary(prose.len() - 50));
        stream(&inst, d.as_mut(), &[&prose]);
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "<function_calls>",
                r#"[{"name":"f","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(last_call(&events).map(|c| c.0), Some("f".into()));
    }

    #[test]
    fn the_window_keeps_a_marker_split_by_the_trim() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.tool_decoder();
        // The trim has to happen while the opening marker's own prefix is
        // sitting at the end of the buffer, so the overflow and the
        // partial marker must arrive in the same step.
        let mut over = "y".repeat(240);
        over.push_str("<function");
        stream(&inst, d.as_mut(), &[&over]);
        let events = stream(
            &inst,
            d.as_mut(),
            &[
                "_calls>",
                r#"[{"name":"f","arguments":{}}]"#,
                "</function_calls>",
            ],
        );
        assert_eq!(last_call(&events).map(|c| c.0), Some("f".into()));
    }

    #[test]
    fn prose_below_the_bound_is_kept_whole() {
        assert_eq!(crate::shared::decoders::keep_tail("가나다", 50), "가나다");
    }

    // ─── The rest of the instruct surface ────────────────────

    #[test]
    fn no_tools_means_no_preamble() {
        let inst = OlmoInstruct::new(tok());
        assert!(inst.equip(&[]).is_empty());
    }

    #[test]
    fn the_user_turn_uses_the_user_role() {
        let inst = OlmoInstruct::new(tok());
        let text = inst.tokenizer.decode(&inst.user("hi"), false);
        assert_eq!(text, "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[test]
    fn the_seal_stops_on_the_turn_end_and_on_eos() {
        let inst = OlmoInstruct::new(tok());
        let seal = inst.seal();
        assert_eq!(
            seal,
            [
                inst.tokenizer.encode("<|im_end|>"),
                inst.tokenizer.encode("<|endoftext|>")
            ]
            .concat()
        );
    }

    #[test]
    fn the_chat_decoder_stops_at_the_turn_end() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.chat_decoder();
        let hi = inst.tokenizer.encode("hi");
        assert!(matches!(d.feed(&hi), ChatEvent::Delta(t) if t == "hi"));
        let mut tail = inst.tokenizer.encode("!");
        tail.extend(inst.tokenizer.encode("<|im_end|>"));
        assert!(matches!(d.feed(&tail), ChatEvent::Done(t) if t == "hi!"));
    }

    /// The decoder stops on EVERY token `seal()` calls terminal.
    ///
    /// These were two tests that each checked their own half -- one that
    /// `seal()` returns the turn end AND eos, one that the decoder stops at
    /// the turn end -- and both passed while the two disagreed about
    /// `<|endoftext|>`. The guest stops sampling on `seal()` and reads its
    /// text out of this decoder, so a token terminal to one and ordinary to
    /// the other ends generation without ever producing `Done`: an inferlet
    /// that loops until `Done` loses the last message of every conversation
    /// the model chose to end with eos.
    ///
    /// Stated over the whole of `seal()` rather than by naming the token, so
    /// a third stop token added to one side has to be added to the other.
    #[test]
    fn every_token_the_seal_calls_terminal_also_ends_the_decoder() {
        let inst = OlmoInstruct::new(tok());
        for id in inst.seal() {
            let mut d = inst.chat_decoder();
            assert!(
                matches!(d.feed(&[id]), ChatEvent::Done(_)),
                "seal() calls {id} terminal but the chat decoder reads it as \
                 ordinary text"
            );
        }
    }

    /// `cue()` already emits `<think>`, so the reasoning decoder must
    /// begin inside the block rather than waiting for an opener that will
    /// never arrive.
    #[test]
    fn reasoning_begins_inside_because_the_cue_opened_it() {
        let inst = OlmoInstruct::new(tok());
        let mut d = inst.reasoning_decoder();
        let step = inst.tokenizer.encode("weighing it up");
        assert!(matches!(d.feed(&step), ReasoningEvent::Delta(t) if t == "weighing it up"));
        assert!(matches!(
            d.feed(&inst.tokenizer.encode("</think>")),
            ReasoningEvent::Complete(t) if t == "weighing it up"
        ));
    }
}
