//! Chat-template templating + parsing: the generated `pie:inferlet/chat` bindings, re-exported.
//! Template knowledge lives in the Pie runtime, not here. [`Decoder::feed`] yields one [`Event`]
//! per call, and an empty [`Event::Delta`] means the batch produced no visible character.

pub use crate::pie::inferlet::chat::{
    Decoder, Event, assistant, cue, first_user, seal, stop_tokens, system, system_user, user,
};
