//! What this driver tells the loader, and the plan it gets back.
//!
//! The same crates the C++ reached through the C ABI — `model` for the author
//! registry, `model-loader` for the compiler — called in-process, so what
//! remains is what the driver alone knows and must state.
//!
//! That is which backend this is ([`metal_storage_target`]). The transform
//! mask lives in `model_loader::plan::passes::tile`, where the consequence
//! lands; five of the seven [`model::shared::policy::Policy`] fields are
//! [`model::boot`]'s, since two copies of a policy drift silently.

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
/// One definition, two readers: the device facts published at create time, and
/// the target supplied with every compile request. Alignment, tile budget and
/// transform mask are `StorageTarget::for_backend`'s — one statement, on the
/// side that owns the consequence.
#[must_use]
pub fn metal_storage_target() -> StorageTarget {
    StorageTarget::for_backend(BackendKind::Metal, 0, 1)
}

/// Why a load plan was not produced.
///
/// The split says which side refused. [`Self::Checkpoint`] is this module's own
/// step, reading the snapshot directory, which [`model::boot`] deliberately
/// leaves to its caller; [`Self::Plan`] is everything the shared load path can
/// refuse.
#[derive(Debug)]
pub enum LoadPlanError {
    /// The snapshot directory did not read as a checkpoint.
    Checkpoint(String),
    /// The shared load path refused; the value says why.
    Plan(model::boot::LoadPlanError),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Both arms keep the `load plan: ` prefix, because it is what an
        // `Error::Create { what }` message reads as at the boot site.
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
/// This driver reads the snapshot directory itself and hands the shared load
/// path everything else. The two answers it contributes are
/// [`Binding::MLX_IN_PLACE`] — MLX tensor names, projections left as stored —
/// and they are claims about the lowering rather than preferences: the bind
/// path looks up MLX names, and the attention and MLP kernels here read the
/// separate `q`/`k`/`v` tensors, so a fused request would produce operands this
/// driver cannot find.
///
/// Everything else is [`model::boot::compile_load_plan_for`]'s, which
/// `driver-cuda` calls with its own [`Binding`], so equal requests author equal
/// contracts because there is one policy rather than two that agree today.
///
/// The ROW is the author: `catalog::identify` matched it against this
/// checkpoint's own tensor names and extents, so the thing that authors the
/// contract and the thing the checkpoint contains cannot be two models. The
/// `encoding` is the one fact a row genuinely cannot state — Qwen3-8B is one
/// model and four downloads, and a group size is not an extent of any tensor.
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
    /// Both claims are load-bearing: the bind path looks up MLX names, and the
    /// attention and MLP kernels read separate `q`/`k`/`v`, so a fused request
    /// would author operands this driver cannot find. Stated here because
    /// nothing else in this crate would notice the constant changing.
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

    /// A row's LOAD shape is what this path carries.
    ///
    /// `catalog::find` takes the id an operator types and hands back the row
    /// itself, and the row STATES its shape, so there is nothing to round-trip
    /// through a hand-built document — a helper that writes a schema by hand is
    /// a second statement of that schema, and the two drift.
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

    /// An id no row claims is refused HERE, where the operator typed it,
    /// rather than as an `UnknownFamily` from inside the plan compiler after
    /// the checkpoint has been opened and its metadata parsed.
    #[test]
    fn an_id_no_row_claims_finds_nothing() {
        assert!(model::catalog::find("qwen3-0.6b-but-spelled-wrong").is_none());
    }
}
