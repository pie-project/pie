//! `pie_cuda_encode`: the multimodal towers, run outside a fire.
//!
//! # THE TOWERS HAVE NO WEIGHTS (R3)
//!
//! This file used to bind gemma-4's vision and audio encoders: it resolved
//! `model_legacy::shared::tower_names`' slot order to device pointers out of
//! `LoadedModel::weights` and called `kernels_cuda::tower::*`. Both halves of
//! that are gone.
//!
//! `LoadedModel::weights` was the LEGACY LOAD CONTRACT's arena — the one
//! `model_legacy::boot::compile_load_plan` authored and
//! `weights::stage::stage_plan_weights` staged. R2 kept it alive after the
//! legacy fire path died *because this file read it*; R3 deletes the
//! contract, so there is no arena and no legacy name to resolve. A fire's
//! weights come from `model::produce` through the SKU's own import table now,
//! and NO IMPORT TABLE IN THE CATALOG READS A TOWER TENSOR — `gemma_4`'s
//! `import_hf` writes the text tower and nothing else. Every one of the
//! tower's weights is a checkpoint tensor no production row asks for, which
//! `baker_load`'s untaken count says out loud.
//!
//! So the honest answer is the one below: a refusal by name. Binding a
//! pointer that does not exist would have been a fault inside a kernel, and
//! answering `PIE_STATUS_OK` without encoding would have handed the
//! scheduler embedding rows of whatever the buffer held.
//!
//! # What it takes to bring them back
//!
//! An import table that produces the tower's banks, and a `#[points]`
//! declaration for the tower's launches so they resolve through the claim
//! table like every other fire. That is the same work the text tower already
//! had done for it, and it is P-series work rather than R-series: nothing
//! here is legacy that needs retiring, it is a capability that needs
//! declaring.

use driver_api::local::{PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_UNSUPPORTED};

use super::guard;
use super::state::Shell;

/// The multimodal encode: media in, embedding rows out.
impl Shell {
    /// Encode media into the model's embedding space.
    ///
    /// # Two refusals, and the order is the point
    ///
    /// A malformed plan is the CALLER's fault and is still named as one — a
    /// payload with no anchor to attach it to, a misaligned pixel buffer, a
    /// CSR that does not partition its bytes. `MediaEncodePlan::validate` is
    /// the only thing that checks any of it, so it runs first and answers
    /// `Invalid`, exactly as it did when there were towers behind it.
    ///
    /// What comes after is this BUILD's gap, and it answers `Unsupported`.
    /// Collapsing the two would tell an operator with a bad plan that pie
    /// cannot do multimodal, and an operator with a good one that their
    /// arguments were wrong.
    ///
    /// # Errors
    ///
    /// `Invalid` for a plan that does not describe itself, `Unsupported`
    /// otherwise — until a catalog import table produces a tower's weights.
    pub fn encode(
        &mut self,
        encode: &mut driver_api::MediaEncodePlan,
        completion: driver_api::completion::CompletionTarget,
    ) -> Result<(), i32> {
        let _ = completion;
        guard("encode", Err(PIE_STATUS_UNSUPPORTED), move || {
            if let Err(why) = encode.validate() {
                eprintln!("[driver-cuda] encode: {why}");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            eprintln!(
                "[driver-cuda] encode: this build produces no multimodal \
                 tower weights — no catalog import table reads a tower \
                 tensor, so there is nothing resident to encode with. \
                 `load_model` advertises `supports_media_encode: false`, \
                 which is why the engine should not have asked."
            );
            Err(PIE_STATUS_UNSUPPORTED)
        })
    }
}
