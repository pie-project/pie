//! Predicate evaluation for the union path.

use model_compiler::lower::Row;

use crate::device::{
    SLOT_HAS_CUSTOM_MASK, SLOT_HAS_LORA, SLOT_HAS_STAGE_HOOKS, SLOT_HAS_WRITE_DESC, SLOT_TOKENS_GT,
    SLOT_TOKENS_LE, SLOT_TOKENS_MULTIPLE, SLOT_WANTS_ATTN_SCORE, SLOT_WINDOW_ONE,
};

/// Evaluate one predicate slot against a fire's rows. Must mirror
/// `lower::select`, or replay and eager run different programs. `None` if the
/// slot has no row-level meaning.
#[must_use]
pub fn predicate_of(slot: u32, param: u32, rows: &[Row]) -> Option<bool> {
    Some(match slot {
        SLOT_HAS_WRITE_DESC => rows.iter().any(|r| r.write_desc),
        SLOT_TOKENS_LE => rows.len() as u32 <= param,
        SLOT_TOKENS_GT => rows.len() as u32 > param,
        // `param == 0` is false, not a division; mirrors `TokensMultipleOf`.
        SLOT_TOKENS_MULTIPLE => param != 0 && (rows.len() as u32).is_multiple_of(param),
        SLOT_WANTS_ATTN_SCORE => rows.iter().any(|r| r.wants_scores),
        SLOT_HAS_CUSTOM_MASK => rows.iter().any(|r| r.custom_mask),
        SLOT_HAS_STAGE_HOOKS => rows.iter().any(|r| r.hooked),
        SLOT_HAS_LORA => rows.iter().any(|r| r.lora),
        SLOT_WINDOW_ONE => !rows.iter().any(|r| r.multi_token),
        _ => return None,
    })
}
