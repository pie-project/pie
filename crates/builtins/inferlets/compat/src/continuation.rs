//! A turn the gateway continues on another worker after its worker was lost
//! mid-stream.
//!
//! These are the built-in text APIs, whose meaning the gateway knows, so it
//! does this for them alone; no other program is continued. Launched through
//! the gateway's HTTP routes, whose input says `"continuable": true`, each
//! frame this program sends carries, under `continuation`, the tokens it
//! covers since the frame before (and the first frame the response id),
//! which the gateway keeps and takes off before the client sees the frame.
//! Launched any other way (a websocket client, `pie run`), its frames are
//! plain and a `continuation` in the input is refused. If the worker is
//! lost after the client had output, the gateway launches the request again
//! elsewhere with what it kept, as the input's `continuation`:
//!
//! ```json
//! {"id": "1a2b3c", "tokens": [5, 6, 7], "frames": 3}
//! ```
//!
//! The run appends the tokens to its prompt and generates the rest under the
//! same id. The first `frames` frames it would send it builds and drops: an
//! adapter is the same over the same tokens, so its stream (an assistant
//! role, a message or response opened, the blocks and items under way) comes
//! to stand where it stood, and the client sees none of it again.

use std::cell::RefCell;

use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Continuation {
    pub(crate) id: String,
    pub(crate) tokens: Vec<u32>,
    /// The frames the client has been sent.
    pub(crate) frames: usize,
}

impl Continuation {
    /// The input's `continuation`, if it has one that reads.
    #[must_use]
    pub(crate) fn of(value: &Value) -> Option<Self> {
        Some(Self {
            id: value.get("id")?.as_str()?.to_string(),
            tokens: value
                .get("tokens")?
                .as_array()?
                .iter()
                .map(|token| token.as_u64().and_then(|t| u32::try_from(t).ok()))
                .collect::<Option<_>>()?,
            frames: usize::try_from(value.get("frames")?.as_u64()?).ok()?,
        })
    }
}

thread_local! {
    /// The gateway keeps what the frames cover: they name it.
    static CONTINUABLE: RefCell<bool> = const { RefCell::new(false) };
    /// The turn this run continues, as the input stated it.
    static CONTINUES: RefCell<Option<Continuation>> = const { RefCell::new(None) };
    /// The first frame has gone out (or this run continues one that did).
    static OPENED: RefCell<bool> = const { RefCell::new(false) };
    /// The frames a continued run still drops: those the client has.
    static MUTED: RefCell<usize> = const { RefCell::new(0) };
    static PENDING: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// This run's frames name what they cover (`continuable`), and it
/// continues `continuation` (from the launch input).
pub(crate) fn begin(continuable: bool, continuation: Option<Continuation>) {
    CONTINUABLE.set(continuable);
    OPENED.set(continuation.is_some());
    MUTED.set(continuation.as_ref().map_or(0, |c| c.frames));
    CONTINUES.set(continuation);
}

/// This run is not continued, whatever its input said: its frames go out
/// plain, so the gateway keeps nothing to continue it from. A request of
/// several choices generates them side by side, and their tokens would
/// interleave in what the frames name.
pub fn opt_out() {
    begin(false, None);
}

/// The turn this run continues, if it does.
#[must_use]
pub(crate) fn get() -> Option<Continuation> {
    CONTINUES.with_borrow(Clone::clone)
}

/// A token generated here, to be named by the next frame.
pub(crate) fn note(token: u32) {
    PENDING.with_borrow_mut(|pending| pending.push(token));
}

/// Sends `frame` with the tokens generated since the frame before; the first
/// frame of a run names the response id. A continued run's first frames,
/// which the client has, are dropped.
pub(crate) fn send(mut frame: Value) {
    if MUTED.with_borrow_mut(|muted| std::mem::replace(muted, muted.saturating_sub(1)) > 0) {
        return;
    }
    let tokens = PENDING.with_borrow_mut(std::mem::take);
    if !CONTINUABLE.with_borrow(|on| *on) {
        inferlet::session::send(&frame.to_string());
        return;
    }
    let mut covers = json!({ "tokens": tokens });
    if !OPENED.replace(true) {
        covers["id"] = json!(crate::short_id());
    }
    frame["continuation"] = covers;
    inferlet::session::send(&frame.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_continuation_reads_back_as_the_id_the_tokens_and_the_frames_sent() {
        assert_eq!(
            Continuation::of(&json!({"id": "1a2b", "tokens": [5, 6, 7], "frames": 3})),
            Some(Continuation {
                id: "1a2b".into(),
                tokens: vec![5, 6, 7],
                frames: 3,
            })
        );
        assert_eq!(Continuation::of(&json!({"id": "1a2b"})), None);
        assert_eq!(
            Continuation::of(&json!({"id": "1a2b", "tokens": [-1], "frames": 1})),
            None
        );
    }
}
