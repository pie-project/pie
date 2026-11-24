pub use crate::pie::inferlet::chat::{
    Decoder, Event, ReasoningEffort, assistant, cue, cue_without_thinking, first_user, open,
    prefix, seal, stop_tokens, system, user,
};

use std::sync::OnceLock;

/// A system message, then the first user turn: [`open`] with the message
/// and no tools, then [`user`] (or [`first_user`] where the template opened
/// with nothing).
#[must_use]
pub fn system_user(system: &str, message: &str) -> Vec<u32> {
    let mut tokens = open(Some(system), &[], None);
    tokens.extend(if tokens.is_empty() {
        first_user(message)
    } else {
        user(message)
    });
    tokens
}

/// [`cue`], or [`cue_without_thinking`] when `think` is false.
#[must_use]
pub fn cue_thinking(think: bool) -> Vec<u32> {
    if think { cue() } else { cue_without_thinking() }
}

/// Whether `token` is one of the model's [`stop_tokens`], the list read once
/// for the process.
#[must_use]
pub fn is_stop(token: u32) -> bool {
    static STOP: OnceLock<Vec<u32>> = OnceLock::new();
    listed(&STOP, stop_tokens, token)
}

/// Whether `list`, read by `read` the first time it is asked, holds `token`.
fn listed(list: &OnceLock<Vec<u32>>, read: impl FnOnce() -> Vec<u32>, token: u32) -> bool {
    list.get_or_init(read).contains(&token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// The list is read on the first ask and answers every later one.
    #[test]
    fn the_stop_list_is_read_once() {
        let list = OnceLock::new();
        let reads = Cell::new(0);
        let read = || {
            reads.set(reads.get() + 1);
            vec![7, 9]
        };
        assert!(listed(&list, read, 9));
        assert!(!listed(&list, read, 8));
        assert!(listed(&list, || unreachable!("read twice"), 7));
        assert_eq!(reads.get(), 1);
    }
}
