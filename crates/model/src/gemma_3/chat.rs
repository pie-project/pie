//! Gemma 3's chat template, which is also gemma-3n's and therefore lives in [`crate::shared`]; this re-export is the path every caller spells.

pub use crate::shared::gemma_chat::{Gemma3Instruct, Gemma3Variant};
