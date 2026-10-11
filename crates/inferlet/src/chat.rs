pub use crate::pie::inferlet::chat::{
    Decoder, Event, assistant, cue, first_user, prefix, seal, stop_tokens, system, system_user,
    user,
};

/// The tokens that tell a thinking model not to think this turn, to follow
/// [`cue`]: an empty thinking block, as the model's own template emits for
/// `enable_thinking: false`. Empty when the vocabulary has no single-token
/// block markers, where there is no thinking to turn off.
#[must_use]
pub fn no_thinking_cue() -> Vec<u32> {
    let (open, close) = (crate::model::encode("<think>"), crate::model::encode("</think>"));
    if open.len() != 1 || close.len() != 1 {
        return Vec::new();
    }
    let gap = crate::model::encode("\n\n");
    [open, gap.clone(), close, gap].concat()
}
