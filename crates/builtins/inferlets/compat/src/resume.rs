//! A turn that moved here from another worker, continuing where it left off.
//!
//! The gateway may move a running turn: it ends it on one worker and launches
//! it on another, which hands the program the resume log it kept
//! (`session.resumed`). The log is the response id the client knows, on a
//! line of its own, then each frame the client has been sent: the tokens it
//! covers since the frame before, each followed by a space, then `;`. The
//! program appends those tokens to its prompt and generates the rest under
//! the same id. The first frames it would send, as many as the log counts,
//! it builds and drops: an adapter is the same over the same tokens, so its
//! stream (an assistant role, an echoed prompt, a message or response opened,
//! the blocks and items under way) comes to stand where it stood, and the
//! client sees none of it again.
//!
//! So that a turn can move, the first frame states the id, and every frame
//! goes into the log in one step with it (`session.send-with-resume`).

use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resume {
    pub id: String,
    pub tokens: Vec<u32>,
    /// The frames the client has been sent.
    pub frames: usize,
}

impl Resume {
    /// A log as [`Resume::head`] and the frames after it wrote it.
    pub fn parse(log: &str) -> Option<Self> {
        let (id, frames) = log.split_once('\n')?;
        let sent = frames.strip_suffix(';').map_or(frames, |all| all);
        if !frames.is_empty() && !frames.ends_with(';') {
            return None;
        }
        Some(Self {
            id: id.to_string(),
            tokens: sent
                .split(|c: char| c == ';' || c.is_ascii_whitespace())
                .filter(|token| !token.is_empty())
                .map(|token| token.parse().ok())
                .collect::<Option<_>>()?,
            frames: frames.matches(';').count(),
        })
    }

    /// A log's head: the response id.
    fn head(id: &str) -> String {
        format!("{id}\n")
    }
}

/// What a frame covering `tokens` adds to the log.
fn frame(tokens: &[u32]) -> String {
    let mut more: String = tokens.iter().map(|token| format!("{token} ")).collect();
    more.push(';');
    more
}

thread_local! {
    /// `None` until read: the log this run resumes from, if it moved here.
    static RESUME: RefCell<Option<Option<Resume>>> = const { RefCell::new(None) };
    /// The log has its head: it was resumed, or a frame has gone out.
    static HEADED: RefCell<bool> = const { RefCell::new(false) };
    /// The frames a resumed run still drops: those the client has.
    static MUTED: RefCell<usize> = const { RefCell::new(0) };
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
            MUTED.set(resume.as_ref().map_or(0, |resume| resume.frames));
            resume
        })
        .clone()
    })
}

/// A token generated here, to be named by the next frame.
pub(crate) fn note(token: u32) {
    PENDING.with_borrow_mut(|pending| pending.push(token));
}

/// Sends `message`, the log taking it with the tokens generated since the
/// frame before; the first frame of a run that did not resume heads the log.
/// A resumed run's first frames, which the client has, are dropped.
pub(crate) fn send(message: &str) {
    get();
    let pending = PENDING.with_borrow_mut(std::mem::take);
    if MUTED.with_borrow_mut(|muted| std::mem::replace(muted, muted.saturating_sub(1)) > 0) {
        return;
    }
    if !HEADED.replace(true) {
        inferlet::session::set_resume(&Resume::head(&crate::short_id()));
    }
    inferlet::session::send_with_resume(message, &frame(&pending));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_reads_back_as_the_id_the_tokens_and_the_frames_sent() {
        let log = format!(
            "{}{}{}{}",
            Resume::head("cmpl-1"),
            frame(&[]),
            frame(&[5, 6]),
            frame(&[7])
        );
        assert_eq!(
            Resume::parse(&log),
            Some(Resume {
                id: "cmpl-1".into(),
                tokens: vec![5, 6, 7],
                frames: 3,
            })
        );
        assert_eq!(Resume::parse("no head"), None);
        assert_eq!(Resume::parse("cmpl-1\n5 6 "), None, "a frame cut short");
    }
}
