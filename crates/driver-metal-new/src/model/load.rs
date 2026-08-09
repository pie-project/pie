//! Loading a checkpoint into something a fire can bind.
//!
//! Two halves existed and nothing called them in sequence.
//! [`compile_load_plan`] authors the plan and checks its files;
//! [`stage_plan_weights`] runs it and stages every tensor into one device
//! region. This is the call between them, plus the one conversion that makes
//! the result answer a trace's questions: a [`Handle`] map becomes a
//! [`Slice`] map, which is what [`resolve::Store`] reads.
//!
//! # Why the conversion is not a wrapper for its own sake
//!
//! A `Handle` is a checked view that owns a reference to its buffer; a `Slice`
//! is an address and an extent. The binder takes the second on purpose — see
//! `model::executor`'s docs — so that it stays portable and provable with no
//! device in the build. The region is kept beside the map here, because a map
//! of addresses whose buffer has been dropped is a map of dangling pointers.
//!
//! [`compile_load_plan`]: crate::loader::compile_load_plan
//! [`stage_plan_weights`]: crate::metal::stage_plan_weights
//! [`resolve::Store`]: crate::model::resolve::Store

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::loader::{compile_load_plan, metal_storage_target};
use crate::metal::{Context, Handle, stage_plan_weights};
use crate::model::executor::Slice;
use crate::region::Region as _;

/// A checkpoint on the device: the region that holds it, and where each
/// tensor sits in it.
pub struct Loaded {
    /// The staged region. **Held, not dropped** — every address in `tensors`
    /// points into it.
    pub region: Handle,
    /// Checkpoint tensor name → its address and extent.
    pub tensors: HashMap<String, Slice>,
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
    Ok(Loaded { region, tensors })
}
