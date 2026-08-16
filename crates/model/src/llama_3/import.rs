//! Llama 3's tensor names, in every vocabulary that spells them.
//!
//! This generation's rows name the same thirteen tensors as `llama_3`,
//! `mistral_3`, `olmo_2`, `olmo_3` and `phi_3` -- character for character,
//! which is a measured fact and not a family resemblance. So the table is
//! [`crate::shared::llama_like::import`]'s, named here rather than copied.
//!
//! Naming it is the whole content of this module, and that is not ceremony:
//! before, the fact that this generation is spelled HuggingFace's way was
//! recorded nowhere, and a respelling of pie's artifact would have had to
//! FIND the generations it applied to. Now each one points at the table it
//! publishes in, and diverging is writing rows here instead of this line.

use crate::shared::vocabulary::Vocab;

/// Every tensor this generation publishes, and what each vocabulary calls it.
///
/// The five generations of this lineage share one table. The day this one
/// stops sharing it -- a norm moved, a projection split -- it writes its own
/// rows here, and nothing else has to notice.
pub const VOCAB: Vocab = crate::shared::llama_like::import::VOCAB;
