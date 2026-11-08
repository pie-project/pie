pub use crate::pie::inferlet::chat::{
    Decoder, Event, ReasoningEffort, assistant, cue, cue_without_thinking, first_user, open,
    prefix, seal, stop_tokens, system, user,
};

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
