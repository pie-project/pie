//! DeepSeek's turn format: bare role markers, no closing tag, and a tool call
//! written as a fenced JSON block between `<｜tool▁call▁begin｜>` and
//! `<｜tool▁call▁end｜>`.

use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::chatml::{JSON_GRAMMAR, tool_names};
use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special,
};

const END_OF_SENTENCE: &str = "<｜end▁of▁sentence｜>";
const END_OF_TURN: &str = "<|EOT|>";

/// The two markers a turn stops on, as one citable list — what a serving
/// row's tokenizer contract reads, and what `new` resolves its `stop_ids`
/// from, so the demand and the use cannot drift apart.
pub const STOP_TOKENS: &[&str] = &[END_OF_SENTENCE, END_OF_TURN];

const TOOL_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const TOOL_CALL_END: &str = "<｜tool▁call▁end｜>";
const TOOL_SEP: &str = "<｜tool▁sep｜>";
const JSON_OPEN: &str = "\n```json\n";
const JSON_CLOSE: &str = "\n```";

pub struct DeepSeek {
    tokenizer: Arc<Tokenizer>,
    user_prefix: u32,
    assistant_prefix: u32,
    end_of_sentence: u32,
    stop_ids: Vec<u32>,
    think_open: Vec<u32>,
    think_close: u32,

    tool_outputs_begin: u32,
    tool_outputs_end: u32,
    tool_output_begin: u32,
    tool_output_end: u32,
}

impl DeepSeek {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let end_of_sentence = special(&tokenizer, END_OF_SENTENCE);
        let mut think_open = vec![special(&tokenizer, "<think>")];
        think_open.extend(tokenizer.encode("\n"));

        Self {
            user_prefix: special(&tokenizer, "<｜User｜>"),
            assistant_prefix: special(&tokenizer, "<｜Assistant｜>"),
            end_of_sentence,
            stop_ids: STOP_TOKENS
                .iter()
                .map(|marker| special(&tokenizer, marker))
                .collect(),
            think_open,
            think_close: special(&tokenizer, "</think>"),
            tool_outputs_begin: special(&tokenizer, "<｜tool▁outputs▁begin｜>"),
            tool_outputs_end: special(&tokenizer, "<｜tool▁outputs▁end｜>"),
            tool_output_begin: special(&tokenizer, "<｜tool▁output▁begin｜>"),
            tool_output_end: special(&tokenizer, "<｜tool▁output▁end｜>"),
            tokenizer,
        }
    }

    fn without_thinking(msg: &str) -> &str {
        match msg.rfind("</think>") {
            Some(at) => &msg[at + "</think>".len()..],
            None => msg,
        }
    }

    fn tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "You are a helpful assistant with tool calling capabilities. \
             When a tool call is needed, you MUST use the following format to issue the call:\n\
             <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n\
             ```json\n\
             {\"param1\": \"value1\", \"param2\": \"value2\"}\n\
             ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n\
             Make sure the JSON is valid.\n\n\
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

impl Instruct for DeepSeek {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.tokenizer.encode(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.user_prefix];
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.assistant_prefix];
        tokens.extend(self.tokenizer.encode(Self::without_thinking(msg)));
        tokens.push(self.end_of_sentence);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        vec![self.assistant_prefix]
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.system(&Self::tool_system_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut tokens = vec![self.tool_outputs_begin, self.tool_output_begin];
        tokens.extend(self.tokenizer.encode(value));
        tokens.push(self.tool_output_end);
        tokens.push(self.tool_outputs_end);
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
            self.think_open.clone(),
            self.think_close,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(DeepSeekToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        let alternatives = tool_names(tools)?;
        let source = format!(
            r#"root ::= "<｜tool▁calls▁begin｜>" tool-call+ "<｜tool▁calls▁end｜>"
tool-call ::= "<｜tool▁call▁begin｜>" "function" "<｜tool▁sep｜>" tool-name "\n```json\n" json-object "\n```" "<｜tool▁call▁end｜>"
tool-name ::= {alternatives}
{JSON_GRAMMAR}"#
        );
        Some(ToolGrammar { source })
    }
}

/// Reads fenced tool calls back out of generated text.
///
/// One clock. What stood here matched the begin and end markers against token
/// IDS while accumulating the payload as TEXT, and the two ran at different
/// speeds: `accumulated.clear()` at the begin marker discarded whatever text
/// the same batch had already decoded past it, and `take()` at the end marker
/// swallowed everything after. Both markers are found in the text now, so a
/// batch carrying two calls yields two events and the tail survives.
struct DeepSeekToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

fn parse_call(span: &str) -> Option<(String, String)> {
    let open = span.find(JSON_OPEN)?;
    let header = &span[..open];
    let body = &span[open + JSON_OPEN.len()..];
    let close = body.find(JSON_CLOSE)?;
    let arguments = body[..close].to_string();
    let name = match header.rsplit_once(TOOL_SEP) {
        Some((_, name)) => name,
        None => header.rsplit_once('\n').map_or(header, |(_, name)| name),
    };
    Some((name.trim().to_string(), arguments))
}

impl ToolDecoder for DeepSeekToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(TOOL_CALL_END) else {
                    return events;
                };
                let span = self.accumulated[..at].to_string();
                self.accumulated = self.accumulated[at + TOOL_CALL_END.len()..].to_string();
                self.inside = false;
                if let Some((name, arguments)) = parse_call(&span) {
                    events.push(ToolEvent::Call(name, arguments));
                }
            } else {
                let Some(at) = self.accumulated.find(TOOL_CALL_BEGIN) else {
                    return events;
                };
                self.accumulated = self.accumulated[at + TOOL_CALL_BEGIN.len()..].to_string();
                self.inside = true;
                events.push(ToolEvent::Start);
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}
