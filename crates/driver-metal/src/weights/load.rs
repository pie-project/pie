//! Loading a checkpoint into something a fire can bind.
//!
//! Two halves existed and nothing called them in sequence.
//! [`compile_load_plan_for`] authors the plan and checks its files;
//! [`stage_plan_weights`] runs it and stages every tensor into one device
//! region. This is the call between them, plus the one conversion that makes
//! the result answer a trace's questions: a [`Handle`](crate::device::Handle) map becomes a
//! [`Slice`] map, which is what [`resolve::Store`] reads.
//!
//! # Why the conversion is not a wrapper for its own sake
//!
//! A `Handle` is a checked view that owns a reference to its buffer; a `Slice`
//! is an address and an extent. The binder takes the second on purpose — see
//! `lowering::executor`'s docs — so that it stays portable and provable with no
//! device in the build. The region is kept beside the map here, because a map
//! of addresses whose buffer has been dropped is a map of dangling pointers.
//!
//! [`compile_load_plan_for`]: crate::loader::compile_load_plan_for
//! [`stage_plan_weights`]: crate::weights::stage_plan_weights
//! [`resolve::Store`]: crate::lowering::resolve::Store

use std::collections::HashMap;
use std::path::Path;

use crate::device::{Allocation, Context};
use crate::error::{Error, Result};
use crate::layout::region::Region as _;
use crate::loader::{compile_load_plan_for, metal_storage_target};
use crate::lowering::executor::Slice;
use crate::weights::stage_plan_weights;

/// A checkpoint on the device: the region that holds it, and where each
/// tensor sits in it.
pub struct Loaded {
    /// The staged region. **Held, not dropped** — every address in `tensors`
    /// points into it, and it is resident for exactly as long as this
    /// `Loaded` lives. Dropping it while a fire is bound against those
    /// addresses takes the weights out of the residency set under a running
    /// GPU, which is why it is a field rather than a local.
    ///
    /// CHUNKED, and never one: a recorded command cannot bind past 4 GiB
    /// (`device/recording.rs`), so `stage_weights` cuts the staged bytes on
    /// tensor boundaries into buffers no larger than that. Every address in
    /// `tensors` still points into exactly one of them.
    pub regions: Vec<Allocation>,
    /// Checkpoint tensor name → its address and extent.
    pub tensors: HashMap<String, Slice>,
    /// Weights the plan leaves in MXFP4, by name.
    ///
    /// Read off the plan by [`LoadPlan::mxfp4_tensor_names`], which is where
    /// the reasoning for it lives — including why it is a set of names and
    /// not a flag. This driver used to compute it here by matching on
    /// `Encoding::Quant` and on a `QuantScheme` variant, which is two of the
    /// loader's enums read structurally by a crate that should be reading
    /// answers.
    ///
    /// [`LoadPlan::mxfp4_tensor_names`]: model_loader::plan::LoadPlan::mxfp4_tensor_names
    pub mxfp4: std::collections::HashSet<String>,
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl Loaded {
    /// Every tensor name the checkpoint published, sorted.
    ///
    /// What a resolver's misses are diagnosed against: a name the text asks
    /// for and this list does not contain is either a spelling the map has
    /// wrong or a tensor the plan did not publish, and the two are told apart
    /// by looking.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        out.sort_unstable();
        out
    }
}

/// Author the plan for `snapshot_dir` against `row`, run it, and stage every
/// tensor.
///
/// # Why the row and not a descriptor
///
/// This took the `pie.model/1` descriptor JSON, whose `model_type` string
/// selected an author out of a registry. It takes the identified
/// [`Variant`](model::catalog::Variant) instead: the row was matched against
/// this checkpoint's OWN TENSOR NAMES AND EXTENTS by
/// `model::catalog::identify`, so the author that writes the contract and
/// the bytes the contract is authored over cannot describe two different
/// models. `encoding` is the one fact a row genuinely cannot state — the
/// same checkpoint is published at 4 bits and at 8, and a group size is not
/// an extent of any tensor.
///
/// # Errors
///
/// A plan that will not compile (a shape the row asserts and the checkpoint
/// contradicts, a declared file that is not on disk at the size declared), or
/// a staging that will not allocate.
pub fn load(
    context: &Context,
    snapshot_dir: &Path,
    row: &dyn model::catalog::Variant,
    encoding: &model::encoding::Encoding,
) -> Result<Loaded> {
    let target = metal_storage_target();
    let (plan, _moe) =
        compile_load_plan_for(snapshot_dir, &target, row, encoding).map_err(|err| {
            Error::Create {
                what: "load plan",
                // `{err}`, not `{err:?}`. `LoadPlanError`'s Display quotes the
                // compiler's own words and names the tensor it refused over; its
                // Debug prints `Plan(Compile("..."))`. An operator reads this
                // message.
                message: err.to_string(),
            }
        })?;
    let (regions, staged) = stage_plan_weights(context, &plan, snapshot_dir)?;
    let tensors = staged
        .into_iter()
        .map(|(name, handle)| {
            (
                name,
                Slice {
                    address: handle.gpu_address(),
                    bytes: handle.len(),
                },
            )
        })
        .collect();
    let mxfp4 = plan.mxfp4_tensor_names();
    Ok(Loaded {
        regions,
        tensors,
        mxfp4,
    })
}
