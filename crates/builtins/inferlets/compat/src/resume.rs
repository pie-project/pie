//! A turn that moved here from another worker, continuing where it left off.
//!
//! The gateway may move a running turn: it ends it on one worker and launches
//! it on another, which hands the program the resume log it kept
//! (`session.resumed`). The log is the response id the client knows, on a
//! line of its own, then each token the client has been sent, followed by a
//! space. The program appends those tokens to its prompt and generates the
//! rest under the same id; its first frames (an assistant role, an echoed
//! prompt) are not sent again.
//!
//! So that a turn can move, the first frame states the id, and every frame
//! appends the tokens it covers since the one before, in one step with the
//! frame (`session.send-with-resume`).

use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resume {
    pub id: String,
    pub tokens: Vec<u32>,
}

impl Resume {
    /// A log as [`Resume::head`] and the frames after it wrote it.
    pub fn parse(log: &str) -> Option<Self> {
        let (id, tokens) = log.split_once('\n')?;
        Some(Self {
            id: id.to_string(),
            tokens: tokens
                .split_ascii_whitespace()
                .map(|token| token.parse().ok())
                .collect::<Option<_>>()?,
        })
    }

    /// A log's head: the response id.
    fn head(id: &str) -> String {
        format!("{id}\n")
    }
}

/// What the tokens since the last frame add to the log.
fn tokens(tokens: &[u32]) -> String {
    tokens.iter().map(|token| format!("{token} ")).collect()
}

thread_local! {
    /// `None` until read: the log this run resumes from, if it moved here.
    static RESUME: RefCell<Option<Option<Resume>>> = const { RefCell::new(None) };
    /// The log has its head: it was resumed, or a frame has gone out.
    static HEADED: RefCell<bool> = const { RefCell::new(false) };
    static PENDING: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// The turn this run continues, if it moved here.
pub fn get() -> Option<Resume> {
    RESUME.with_borrow_mut(|slot| {
        slot.get_or_insert_with(|| {
            let resume = inferlet::session::resumed()
                .as_deref()
                .and_then(Resume::parse);
            HEADED.set(resume.is_some());
            resume
        })
        .clone()
    })
}

pub fn resuming() -> bool {
    get().is_some()
}

/// A token generated here, to be named by the next frame.
pub(crate) fn note(token: u32) {
    PENDING.with_borrow_mut(|pending| pending.push(token));
}

/// Sends `frame`, appending to the log the tokens generated since the frame
/// before; the first frame of a run that did not resume heads the log.
pub(crate) fn send(frame: &str) {
    get();
    if !HEADED.replace(true) {
        inferlet::session::set_resume(&Resume::head(&crate::short_id()));
    }
    let pending = PENDING.with_borrow_mut(std::mem::take);
    if pending.is_empty() {
        inferlet::session::send(frame);
    } else {
        inferlet::session::send_with_resume(frame, &tokens(&pending));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_reads_back_as_the_id_and_the_tokens_sent() {
        let log = format!(
            "{}{}{}",
            Resume::head("cmpl-1"),
            tokens(&[5, 6]),
            tokens(&[7])
        );
        assert_eq!(
            Resume::parse(&log),
            Some(Resume {
                id: "cmpl-1".into(),
                tokens: vec![5, 6, 7]
            })
        );
        assert_eq!(Resume::parse("no head"), None);
    }
}
