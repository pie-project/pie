use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|eot|>", "<|end_of_text|>"];

pub struct Atem {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eot: u32,
    eom: u32,
    stop_ids: Vec<u32>,
    reasoning_open: Vec<u32>,
    generation_prefix: Vec<u32>,
    open: bool,
}

impl Atem {
    pub fn try_new(tokenizer: Arc<Tokenizer>) -> Result<Self, String> {
        Self::try_with(tokenizer, false)
    }

    /// `open`: the Muse Spark release's system turn, its preamble, date and
    /// recipients stated whether a system message is given or not.
    pub fn try_with(tokenizer: Arc<Tokenizer>, open: bool) -> Result<Self, String> {
        let stop_ids = specials(&tokenizer, STOP_TOKENS)?;

        let start = special(&tokenizer, "<|start|>")?;
        let message = special(&tokenizer, "<|message|>")?;
        let eot = special(&tokenizer, "<|eot|>")?;
        let eom = special(&tokenizer, "<|eom|>")?;

        let header = |role: &str| -> Vec<u32> {
            let mut tokens = vec![start];
            tokens.extend(tokenizer.encode(role));
            tokens.push(message);
            tokens
        };

        let mut reasoning_open = tokenizer.encode(" to=self");
        reasoning_open.push(message);

        let mut generation_prefix = vec![start];
        generation_prefix.extend(tokenizer.encode("assistant"));

        Ok(Self {
            bos: special(&tokenizer, "<|begin_of_text|>")?,
            system_prefix: header("system"),
            user_prefix: header("user"),
            assistant_prefix: header("assistant to=user"),
            eot,
            eom,
            stop_ids,
            reasoning_open,
            generation_prefix,
            tokenizer,
            open,
        })
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.push(self.eot);
        tokens
    }

    fn opening(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn(prefix, msg));
        tokens
    }
}

impl Instruct for Atem {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        if self.open {
            return self.opening(&self.system_prefix, &open_system(msg));
        }
        let mut body = msg.to_string();
        if !msg.to_ascii_lowercase().contains("reasoning strength") {
            body.push_str("\n\nReasoning strength: high.");
        }
        body.push_str("\n\n# Valid recipients: \"self\", \"user\".");
        self.opening(&self.system_prefix, &body)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        if self.open {
            let mut tokens = self.opening(&self.system_prefix, &open_system(""));
            tokens.extend(self.turn(&self.user_prefix, msg));
            return tokens;
        }
        self.opening(&self.user_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.assistant_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let mut tokens = vec![self.generation_prefix[0]];
        tokens.extend(self.tokenizer.encode(&format!("tool {name}")));
        tokens.push(self.reasoning_open[self.reasoning_open.len() - 1]);
        tokens.extend(self.tokenizer.encode(&format!(
            "<tool_output name=\"{name}\">\n{value}\n</tool_output>"
        )));
        tokens.push(self.eot);
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
            self.reasoning_open.clone(),
            self.eom,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

/// The Muse Spark release's system turn around `msg` (its chat template's
/// defaults: the 2026-01-04 cutoff, today's UTC date, strength 512).
fn open_system(msg: &str) -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400)
        .cast_signed();
    // Days since 1970-01-01 as a civil date (Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let mut parts = vec![
        "You are a helpful AI assistant.".to_string(),
        "Knowledge cutoff: 2026-01-04.".to_string(),
        format!("Current date: {year:04}-{month:02}-{day:02}."),
    ];
    if !msg.is_empty() {
        parts.push(msg.to_string());
    }
    if !msg.to_ascii_lowercase().contains("reasoning strength:") {
        parts.push("Reasoning strength: 512.".to_string());
    }
    parts.push(
        "Use the appropriate recipient for each message:\n\n\
         - \"self\": private reasoning and tool planning.\n\
         - \"commentary\": user-visible intermediate updates while the assistant will continue working, including messages sent to the user before or between tool calls.\n\
         - \"user\": messages that end the assistant turn, such as a completed response or clarification question that waits for the user's reply; do not use for pre-tool updates or partial responses.\n\n\
         # Valid recipients: \"self\", \"commentary\", \"user\"."
            .to_string(),
    );
    parts.join("\n\n")
}
