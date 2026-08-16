//! What a trace that states one of `xqa`'s symbols binds to.
//!
//! `max_pages_per_seq` is a PER-REQUEST maximum -- `keys::KvMaxPagesPerRequest`
//! -- not the batch-wide total `Cx::num_pages_in_batch` answers.

use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] =
    &[Bound::derived("attn::attention_xqa_decode_bf16_prepared")];
