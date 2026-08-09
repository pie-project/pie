//! Loading a checkpoint into something a fire can bind.
//!
//! Two halves existed and nothing called them in sequence.
//! [`compile_load_plan`] authors the plan and checks its files;
//! [`stage_plan_weights`] runs it and stages every tensor into one device
//! region. This is the call between them, plus the one conversion that makes
//! the result answer a trace's questions: a [`Handle`](crate::gpu::Handle) map becomes a
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
//! [`compile_load_plan`]: crate::loader::compile_load_plan
//! [`stage_plan_weights`]: crate::gpu::stage_plan_weights
//! [`resolve::Store`]: crate::lowering::resolve::Store

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::loader::{compile_load_plan, metal_storage_target};
use crate::gpu::{Allocation, Context, stage_plan_weights};
use crate::lowering::executor::Slice;
use crate::layout::region::Region as _;

/// A checkpoint on the device: the region that holds it, and where each
/// tensor sits in it.
pub struct Loaded {
    /// The staged region. **Held, not dropped** — every address in `tensors`
    /// points into it, and it is resident for exactly as long as this
    /// `Loaded` lives. Dropping it while a fire is bound against those
    /// addresses takes the weights out of the residency set under a running
    /// GPU, which is why it is a field rather than a local.
    pub region: Allocation,
    /// Checkpoint tensor name → its address and extent.
    pub tensors: HashMap<String, Slice>,
    /// Weights the plan leaves in MXFP4, by name.
    ///
    /// The load's job is to get the bytes onto the device unchanged; what
    /// they MEAN is the binder's business (`.wiki/new-driver/next.md`,
    /// priority 2). This is how the load tells the binder, and it is a set of
    /// names rather than a flag because a checkpoint need not be uniform:
    /// `mlx-community/gpt-oss-20b-MXFP4-Q4` names 98 tensors as affine/64/4
    /// in its `quantization` block and leaves the expert banks out, so they
    /// take the top-level default -- mxfp4, group 32.
    ///
    /// Reading a bank with the dense format is not a near miss. Every scale
    /// comes from the wrong offset and bf16 garbage is NaN more often than
    /// not: measured, the fire bound every name, ran all 484 statements, and
    /// produced NaNs from the first routed projection of layer 0 onward while
    /// every structural gate passed.
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

/// Author the plan for `snapshot_dir`, run it, and stage every tensor.
///
/// # Errors
///
/// A plan that will not compile (an unknown family, a descriptor that does not
/// parse, a declared file that is not on disk at the size declared), or a
/// staging that will not allocate.
pub fn load(context: &Context, snapshot_dir: &Path, descriptor_json: &str) -> Result<Loaded> {
    let target = metal_storage_target();
    let (plan, _moe) =
        compile_load_plan(snapshot_dir, &target, descriptor_json).map_err(|err| Error::Create {
            what: "load plan",
            message: format!("{err:?}"),
        })?;
    let (region, staged) = stage_plan_weights(context, &plan, snapshot_dir)?;
    let tensors = staged
        .into_iter()
        .map(|(name, handle)| {
            (name, Slice {
                address: handle.gpu_address(),
                bytes: handle.len(),
            })
        })
        .collect();
    let mxfp4 = plan
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                &t.encoding,
                model_loader::types::Encoding::Quant(spec)
                    if spec.scheme == model_loader::types::QuantScheme::Mxfp4E2M1E8M0
            )
        })
        .map(|t| t.name.clone())
        .collect();
    Ok(Loaded {
        region,
        tensors,
        mxfp4,
    })
}
