pub use crate::pie::inferlet::chat::{
    Decoder, Event, assistant, cue, cue_without_thinking, first_user, prefix, seal, stop_tokens,
    system, system_user, user,
};

/// [`cue`], or [`cue_without_thinking`] when `think` is false.
#[must_use]
pub fn cue_thinking(think: bool) -> Vec<u32> {
    if think { cue() } else { cue_without_thinking() }
}
