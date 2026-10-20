//! A program's resume log, as it travels with its output: what a turn moved
//! to another worker resumes from. A program states it (`session.set-resume`,
//! `session.send-with-resume`); each note goes out as one process event
//! [`EVENT`], in order with what the program sends, so whoever moves a turn
//! holds the log as of exactly what the client was sent. A client that moves
//! nothing sees only the messages the notes carry.

use serde::{Deserialize, Serialize};

/// The process event a note rides; its value is the note as JSON.
pub const EVENT: &str = "resume";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Note {
    /// Replaces the log, before `more` is appended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Appended to the log.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub more: String,
    /// Sent to the client with it, as a `message` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl Note {
    /// The log after this note, from `log` before it.
    #[must_use]
    pub fn apply(&self, log: Option<String>) -> Option<String> {
        let mut log = match &self.state {
            Some(state) => state.clone(),
            None if self.more.is_empty() => return log,
            None => log.unwrap_or_default(),
        };
        log.push_str(&self.more);
        Some(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_is_set_then_grows_by_what_each_note_appends() {
        let set = Note {
            state: Some("head\n".into()),
            ..Note::default()
        };
        let sent = Note {
            more: "1 2 ".into(),
            message: Some("hi".into()),
            ..Note::default()
        };
        let log = sent.apply(sent.apply(set.apply(None)));
        assert_eq!(log.as_deref(), Some("head\n1 2 1 2 "));
        assert_eq!(
            Note::default().apply(None),
            None,
            "a note of nothing keeps no log"
        );
        let wire = serde_json::to_string(&sent).unwrap();
        assert_eq!(serde_json::from_str::<Note>(&wire).unwrap(), sent);
    }
}
