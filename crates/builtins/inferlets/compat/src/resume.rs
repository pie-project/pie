//! A turn that moved here from another worker, continuing where it left off.
//!
//! The gateway may move a running turn: it ends it on one worker and launches
//! it on another with a `resume` beside the body, naming the response id the
//! client knows and the tokens it has been sent. The program appends those
//! tokens to its prompt and generates the rest under the same id; its first
//! frames (an assistant role, an echoed prompt) are not sent again.
//!
//! So that a turn can move, every frame names the tokens it covers since the
//! one before (`plex.tokens`) and the response's id (`plex.id`); the gateway
//! strips them before the client sees the frame.

use std::cell::RefCell;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resume {
    pub id: String,
    pub tokens: Vec<u32>,
}

impl Resume {
    pub fn parse(value: &Value) -> Option<Self> {
        Some(Self {
            id: value.get("id")?.as_str()?.to_string(),
            tokens: value
                .get("tokens")?
                .as_array()?
                .iter()
                .map(|t| t.as_u64().map(|t| t as u32))
                .collect::<Option<_>>()?,
        })
    }
}

thread_local! {
    static RESUME: RefCell<Option<Resume>> = const { RefCell::new(None) };
    static PENDING: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn set(resume: Option<Resume>) {
    RESUME.with_borrow_mut(|slot| *slot = resume);
}

/// The turn this run continues, if it moved here.
pub fn get() -> Option<Resume> {
    RESUME.with_borrow(Clone::clone)
}

pub fn resuming() -> bool {
    RESUME.with_borrow(Option::is_some)
}

/// A token generated here, to be named by the next frame.
pub(crate) fn note(token: u32) {
    PENDING.with_borrow_mut(|pending| pending.push(token));
}

/// The tokens generated since the last frame.
pub(crate) fn take() -> Vec<u32> {
    PENDING.with_borrow_mut(std::mem::take)
}
