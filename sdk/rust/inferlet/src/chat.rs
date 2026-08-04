//! Chat-template templating + parsing — the generated `pie:inferlet/chat`
//! bindings, re-exported.
//!
//! Chat-template knowledge lives in the Pie runtime, not here. The fillers
//! ([`system`], [`user`], [`first_user`], [`system_user`], [`assistant`],
//! [`cue`], [`seal`], [`stop_tokens`]) produce token sequences for the bound
//! model's template; [`Decoder`] parses generated tokens back into visible
//! text plus structural [`Event`]s.
//!
//! [`Decoder::feed`] yields one [`Event`] per call. An empty [`Event::Delta`]
//! means the batch consumed tokens that produced no visible character — it is
//! surfaced as-is rather than remapped, so the caller decides.

pub use crate::pie::inferlet::chat::{
    Decoder, Event, assistant, cue, first_user, seal, stop_tokens, system, system_user, user,
};
