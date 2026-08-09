//! What this driver tells the loader, and the plan it gets back.
//!
//! The C++ (`loader/load_plan.hpp`) reached the Rust loader through the C
//! ABI: open a checkpoint handle, marshal a request, get a marshalled plan.
//! This port calls the same crates — `model` for the author registry,
//! `model-loader` for the compiler — in-process, so the wire structs and
//! their lifetime rules disappear; what remains is what the driver alone
//! knows and must state.
//!
//! That is one thing now. Which backend this is ([`metal_storage_target`], a
//! call into `StorageTarget::for_backend`).
//!
//! It was three. The mask of transforms this driver's kernels implement was
//! stated here AND in `model_loader::plan::passes::tile`, with a test
//! comparing them; the loader keeps it, because the loader is where the
//! consequence lands — it decides which plans compile, and it owns the host
//! fallback every claimed transform must have.
//!
//! The loading policy went the same way. [`compile_load_plan`] here used to
//! state all seven [`model::shared::policy::Policy`] fields, and
//! `driver-cuda`'s copy stated the same seven, differing in exactly two —
//! its own comment said the block was carried "bit for bit". Two copies of a
//! policy is not a spelling problem: a field added to `Policy` gets a
//! considered value on the copy its author was looking at and a `Default` on
//! the other, and both still compile and both still boot. The five shared
//! answers are [`model::boot`]'s now, and what stays here is the two this
//! driver alone knows — named [`Binding::MLX_IN_PLACE`] — plus the checkpoint
//! parse `model::boot` deliberately does not do.

use std::path::Path;

use model::boot::Binding;
use model::catalog::Variant;
use model::encoding::Encoding;
use model::shared::policy::Mxfp4MoePolicy;
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::plan::{LoadPlan, StorageTarget};
use model_loader::types::BackendKind;

/// This device's storage capability.
///
/// One definition, two readers: the device facts published at create time,
/// and the target supplied with every compile request.
///
/// The alignment, the tile budget and the transform mask were stated here as
/// three constants and stated again in `model_loader::plan::passes::tile`,
/// with a test comparing the masks. They are `StorageTarget::for_backend`'s
/// now — one statement, on the side that owns the consequence: the loader
/// decides which plans compile and owns the host fallback every claimed
/// transform has to have.
#[must_use]
pub fn metal_storage_target() -> StorageTarget {
    StorageTarget::for_backend(BackendKind::Metal, 0, 1)
}

/// Why a load plan was not produced.
///
/// Two variants, and the split says which side refused. [`Self::Checkpoint`]
/// is this module's own step — reading the snapshot directory, which
/// [`model::boot`] deliberately leaves to its caller. [`Self::Plan`] is
/// everything the shared load path can refuse: the descriptor, the family
/// registry, the compiler, the file check.
#[derive(Debug)]
pub enum LoadPlanError {
    /// The snapshot directory did not read as a checkpoint.
    Checkpoint(String),
    /// The shared load path refused; the value says why.
    Plan(model::boot::LoadPlanError),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Both arms keep the `load plan: ` prefix the four variants carried,
        // because it is what a `Error::Create { what }` message reads as at
        // the boot site.
        match self {
            LoadPlanError::Checkpoint(err) => write!(f, "load plan: {err}"),
            LoadPlanError::Plan(err) => write!(f, "load plan: {err}"),
        }
    }
}

impl std::error::Error for LoadPlanError {}

impl From<model::boot::LoadPlanError> for LoadPlanError {
    fn from(err: model::boot::LoadPlanError) -> Self {
        LoadPlanError::Plan(err)
    }
}

/// Compile the plan: the row and its encoding in, plan out.
///
/// This driver reads the snapshot directory itself and then hands the shared
/// load path everything else. The two answers it contributes are
/// [`Binding::MLX_IN_PLACE`] — MLX tensor names, projections left as stored
/// — and they are claims about the lowering rather than preferences: the
/// bind path looks up MLX names, and the attention and MLP kernels here read
/// the separate `q`/`k`/`v` tensors, so a fused request would produce
/// operands this driver cannot find.
///
/// Everything else — the other five policy fields, the author call, the
/// plan compile, and the check that every declared file is still on disk at
/// the size the plan states — is [`model::boot::compile_load_plan_for`]'s,
/// which `driver-cuda` calls with its own [`Binding`]. That is the point:
/// equal requests author equal contracts because there is one policy, not
/// two that happen to agree today.
///
/// # What replaced the descriptor argument
///
/// This took a `descriptor_json: &str` — a `pie.model/1` document whose
/// `model_type` string selected an author out of a registry. It takes the
/// ROW instead, because the row IS the author: `catalog::identify` matched
/// it against this checkpoint's own tensor names and extents, so the thing
/// that authors the contract and the thing the checkpoint actually contains
/// cannot be two different models any more. The `encoding` is the one fact
/// a row genuinely cannot state — Qwen3-8B is one model and four downloads,
/// and a group size is not an extent of any tensor.
///
/// The returned [`Mxfp4MoePolicy`] is the author's resolved answer — a family
/// may override the device rule — handed back rather than recomputed, so the
/// bind path cannot disagree with the contract it binds.
///
/// # Errors
///
/// The snapshot directory does not read as a checkpoint, or the shared load
/// path refuses; see [`LoadPlanError`].
pub fn compile_load_plan_for(
    snapshot_dir: &Path,
    target: &StorageTarget,
    row: &dyn Variant,
    encoding: &Encoding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let metadata = parse_checkpoint_metadata(snapshot_dir)
        .map_err(|err| LoadPlanError::Checkpoint(err.to_string()))?;
    Ok(model::boot::compile_load_plan_for(
        snapshot_dir,
        &metadata,
        target,
        row,
        encoding,
        Binding::MLX_IN_PLACE,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_target_states_the_device_and_nothing_optimistic() {
        let target = metal_storage_target();
        assert_eq!(target.backend, BackendKind::Metal);
        assert_eq!(target.preferred_alignment, 256);
        assert_eq!(target.max_tile_bytes, 64 * 1024 * 1024);
        assert!(!target.native_mxfp4_moe);
        assert_eq!(target.fusion_mask, 0, "no fused transcode kernels here");
    }

    /// The binding this driver contributes, asserted rather than assumed.
    ///
    /// It is TWO claims about the lowering — MLX names, projections left as
    /// stored — and both are load-bearing: the bind path looks up MLX names,
    /// and the attention and MLP kernels read separate `q`/`k`/`v`, so a
    /// fused request would author operands this driver cannot find. Stated
    /// here because [`compile_load_plan_for`] passes the constant and nothing
    /// else in this crate would notice if the constant changed under it.
    #[test]
    fn this_driver_asks_for_mlx_names_and_unfused_projections() {
        assert_eq!(
            Binding::MLX_IN_PLACE.naming,
            model::shared::policy::Naming::Mlx
        );
        assert_eq!(
            Binding::MLX_IN_PLACE.projections,
            model::shared::policy::Projections::InPlace
        );
    }

    /// A row's LOAD shape is what this path now carries, in place of the
    /// `TestFacts`/`descriptor_for_testing` pair that used to be here.
    ///
    /// Those two built a `pie.model/1` document by hand — `model_type`,
    /// `num_hidden_layers`, `tie_word_embeddings`, `quant_bits`,
    /// `quant_group_size` — so a probe could state the two or three facts
    /// its family needed and have the driver's own reader parse them back.
    /// The document does not exist any more and neither does the reader:
    /// `catalog::find` takes the id an operator types and hands back the row
    /// itself, and the row STATES its shape. There is nothing left to
    /// round-trip, which is the point — a helper that writes a schema by
    /// hand is a second statement of that schema, and the two drift.
    #[test]
    fn a_row_states_its_own_load_shape_and_needs_no_document_to_carry_it() {
        let id = model::catalog::ids()
            .into_iter()
            .next()
            .expect("this build serves at least one model");
        let row = model::catalog::find(id).expect("an id from `ids` finds its row");
        let shape = row.load_shape();
        assert_eq!(row.id(), id);
        assert!(shape.layers > 0, "`{id}` states no layers");
        assert!(shape.head_dim > 0, "`{id}` states no head dim");
    }

    /// An id no row claims is refused HERE, where the operator typed it.
    ///
    /// The predecessor of this refusal was a `pie.model/1` document with a
    /// `model_type` no author claimed, which surfaced as
    /// `LoadPlanError::UnknownFamily` from inside the plan compiler — after
    /// the checkpoint had been opened and its metadata parsed. A typo in a
    /// boot file is worth answering before any of that.
    #[test]
    fn an_id_no_row_claims_finds_nothing() {
        assert!(model::catalog::find("qwen3-0.6b-but-spelled-wrong").is_none());
    }
}
