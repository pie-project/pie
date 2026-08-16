//! One function, because one function is what the gate can hold.
//!
//! The union cache in [`super::recordings`] is `feature = "abi"`: everything
//! in it holds a `GraphExec`, a `StreamRef` or a `Scratch`, and a build that
//! cannot fire cannot reach any of it. [`predicate_of`] is the exception, and
//! not by accident — it is the ONE step of the union path that is arithmetic
//! over a fire's rows and nothing else. That is what makes it checkable
//! without a device, and `tests/union_lower.rs` is the target that checks it.
//!
//! So the split is the gate's shape, not a taxonomy. Left in `recordings` the
//! function compiled only into builds that could not run the test that proves
//! it, which is the exact inversion of what its own doc comment claims.

use model_compiler::lower::Row;

use crate::device::{
    SLOT_HAS_CUSTOM_MASK, SLOT_HAS_LORA, SLOT_HAS_STAGE_HOOKS, SLOT_HAS_WRITE_DESC, SLOT_TOKENS_GT,
    SLOT_TOKENS_LE, SLOT_TOKENS_MULTIPLE, SLOT_WANTS_ATTN_SCORE, SLOT_WINDOW_ONE,
};

/// Evaluate one predicate slot against a fire's rows.
///
/// This is `lower::select`'s body, and it MUST stay that — the resolved
/// lowering answers a guard by calling `select`, and the captured one
/// answers the same guard by reading this byte out of device memory. If
/// the two ever disagree, the eager leg and the replay leg run different
/// programs and nothing type-checks the difference.
///
/// `None` for a slot with no row-level meaning (the Peel endpoint bits,
/// which are a property of the row SPLIT rather than of the rows).
///
/// Public, and deliberately free of any device object: the equivalence
/// between the eager leg and the captured leg is a HOST fact, so it must
/// be provable without a GPU.
#[must_use]
pub fn predicate_of(slot: u32, param: u32, rows: &[Row]) -> Option<bool> {
    Some(match slot {
        SLOT_HAS_WRITE_DESC => rows.iter().any(|r| r.write_desc),
        SLOT_TOKENS_LE => rows.len() as u32 <= param,
        SLOT_TOKENS_GT => rows.len() as u32 > param,
        // `param == 0` is false rather than a division; see
        // `GuardPred::TokensMultipleOf`, whose evaluation in
        // `model_compiler::lower` this mirrors.
        SLOT_TOKENS_MULTIPLE => param != 0 && (rows.len() as u32).is_multiple_of(param),
        SLOT_WANTS_ATTN_SCORE => rows.iter().any(|r| r.wants_scores),
        SLOT_HAS_CUSTOM_MASK => rows.iter().any(|r| r.custom_mask),
        SLOT_HAS_STAGE_HOOKS => rows.iter().any(|r| r.hooked),
        SLOT_HAS_LORA => rows.iter().any(|r| r.lora),
        SLOT_WINDOW_ONE => !rows.iter().any(|r| r.multi_token),
        _ => return None,
    })
}
