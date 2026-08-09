//! The decode step's resident storage: weights, KV, GDN state, IO and the
//! scratch pool, allocated and staged.
//!
//! The C++ (`loader/heap_bind.cpp::stage_decode_storage`) allocates each
//! region and stages the load plan's weights with its own transform loops.
//! This port allocates the same regions, but the weights come from
//! `model_loader::executor::host::execute_plan` — the engine `transcode.hpp`
//! was a mirror of (see `.wiki/driver/progress-metal.md`) — so every TileMap the plan
//! carries (MXFP4 decode, affine encode, casts) has already run by the time
//! a byte reaches the device buffer.
//!
//! Weights land in ONE shared region and each tensor is a [`Handle`] slice
//! of it, keyed by runtime name: hundreds of per-tensor buffers would be
//! hundreds of residency entries, and the argument tables want stable
//! GPU addresses, not a map that reallocates.
//!
//! What is deliberately NOT here yet, implementation-first (the ledger
//! carries each):
//!
//! * zero-copy mapping and weight streaming (`resolve_mappable`, the
//!   stream pack) — memory optimizations; every checkpoint loads correctly
//!   through the copy path, some just resident-larger;
//! * the expert-slab staging arm (`ExpertSlabRequest`) — the slab type
//!   exists (`crate::loader::ExpertSlab`); wiring it here needs the paging
//!   fire path;
//! * elastic sizing of KV/scratch (`alloc_zeroed`'s initial-commit
//!   parameter) — regions allocate at full size for now.

use std::collections::HashMap;
use std::path::Path;

use model_loader::error::Error as LoaderError;
use model_loader::executor::host::execute_plan_into_arena;
use model_loader::plan::LoadPlan;

use crate::region::Region as _;
use crate::{Error, Result};

use super::context::Context;
use super::handle::Handle;
use super::ring::allocate;





/// Whether a model of `bytes` may be staged on this device, and why not.
///
/// Asked BEFORE a byte is read, which is the only moment it can be asked
/// usefully: `persistent_bytes` is the plan's own layout total, known from
/// the plan, and everything after the call either allocates it or writes
/// into it.
///
/// # Why this had to come back
///
/// The C++ had `fits_on_this_gpu` and the port dropped it. The machinery to
/// answer SURVIVED — `Memory::probe`/`headroom` reads both the device's
/// `recommendedMaxWorkingSetSize` and the kernel's reclaimable pages, and
/// `the_tighter_ceiling_wins_and_silence_is_not_a_refusal` has been testing
/// it the whole time — with no caller anywhere. Implemented, tested, and
/// unreachable, which is the pattern that makes a capability a caps lie.
///
/// What its absence costs is recorded in the tree already:
/// `device_checkpoint_names.rs` notes that *"the 26B gemma4 on this machine
/// is SIGKILLed by the staging test"*, and the guard existed to keep a
/// process from being left hung and unkillable instead. A refusal naming
/// both numbers is strictly better than the kernel's answer.
///
/// # Errors
///
/// [`Error::Create`], naming the model's bytes and the device's ceiling.
/// Never on a machine that would not answer: `headroom` is TRUE when neither
/// ceiling could be read, because refusing on silence turns a diagnostic
/// into an outage.
fn fits_on_this_gpu(memory: &super::memory::Memory, bytes: u64) -> Result<()> {
    if memory.headroom(bytes) {
        return Ok(());
    }
    Err(Error::Create {
        what: "weight arena",
        message: format!(
            "this checkpoint's resident weights are {bytes} bytes and this \
             device's ceiling is {} -- refused before staging rather than \
             after, because the alternative is the kernel deciding and the \
             process is not reliably killable while it does",
            memory.ceiling().unwrap_or(0)
        ),
    })
}

/// Stage the plan's weights into one region, each tensor a named slice.
///
/// # The region IS the arena
///
/// A resident plan lays its weights out at `persistent_offset` in one
/// contiguous arena. This allocates that arena as a Metal buffer and hands
/// its bytes to the executor, which writes the laid-out weights into their
/// final home. **Nothing is copied afterwards** — the tensor map is offsets
/// into a buffer that was written once.
///
/// This used to call `execute_plan`, which allocates the arena as a
/// `Vec<u8>`, fills it, and returns it — so the whole model was resident
/// TWICE, once in that vector and once in the region it was copied into.
/// On a machine where the model is a meaningful fraction of RAM that is the
/// difference between loading and being killed, and it is why nothing here
/// has ever been held to a 26B checkpoint.
///
/// Metal makes this cleaner than it would be elsewhere: a
/// `StorageModeShared` buffer is host-addressable, so the executor's
/// ordinary host writes land in device memory with no upload step between.
///
/// Tensors the plan publishes OUTSIDE the arena have no offset to be written
/// at, so they still arrive through a sink and are appended after it.
pub fn stage_plan_weights(
    context: &Context,
    plan: &LoadPlan,
    snapshot_dir: &Path,
) -> Result<(Handle, HashMap<String, Handle>)> {
    let names_by_tensor: HashMap<_, _> = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.id, tensor.name.as_str()))
        .collect();
    // (name, arena offset, bytes) for every tensor the arena holds. Known
    // from the PLAN, before a byte is read -- which is what lets the region
    // be allocated first and written into.
    let mut in_arena: Vec<(&str, u64, u64)> = Vec::new();
    for buffer in &plan.buffers {
        let (Some(offset), Some(tensor)) = (buffer.persistent_offset, buffer.tensor) else {
            continue;
        };
        if let Some(name) = names_by_tensor.get(&tensor) {
            in_arena.push((name, offset, buffer.bytes));
        }
    }
    let arena_len = plan.memory.persistent_bytes;
    fits_on_this_gpu(&super::memory::Memory::probe(context), arena_len)?;

    // The tensors that are NOT in the arena, collected as they finalize. A
    // sink rather than a second pass: the executor publishes each exactly
    // once and the arena-resident ones are dropped on the floor here, because
    // their bytes are already where they belong.
    let arena_names: std::collections::HashSet<&str> =
        in_arena.iter().map(|(name, _, _)| *name).collect();
    struct Outside<'a> {
        wanted: &'a std::collections::HashSet<&'a str>,
        tensors: Vec<(String, Vec<u8>)>,
    }
    impl model_loader::executor::sink::TensorSink for Outside<'_> {
        fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoaderError> {
            if !self.wanted.contains(name) {
                self.tensors.push((name.to_string(), bytes.to_vec()));
            }
            Ok(())
        }
    }

    // Allocated BEFORE execution and sized from the plan, then written
    // through. The extra span for the outside tensors is not known until they
    // arrive, so this over-allocates by their total -- which is small, and
    // the alternative is a second region.
    let outside_budget: u64 = plan
        .buffers
        .iter()
        .filter(|b| b.persistent_offset.is_none())
        .filter(|b| {
            b.tensor
                .and_then(|t| names_by_tensor.get(&t))
                .is_some_and(|name| !arena_names.contains(name))
        })
        .map(|b| b.bytes.div_ceil(256) * 256)
        .sum();
    let region = allocate(context, (arena_len + outside_budget).max(1), "weights region")?;

    let mut sink = Outside {
        wanted: &arena_names,
        tensors: Vec::new(),
    };
    {
        // SAFETY: the region was just allocated and no GPU work references it,
        // so the executor is the only writer. `arena_len` is the region's own
        // sizing term, so the slice is inside it.
        let arena = unsafe {
            std::slice::from_raw_parts_mut(
                region.contents().cast::<u8>().as_ptr(),
                usize::try_from(arena_len).unwrap_or(usize::MAX),
            )
        };
        execute_plan_into_arena(plan, snapshot_dir, arena, &mut sink, &mut |_| {}).map_err(
            |err| Error::Create {
                what: "staged weights",
                message: err.to_string(),
            },
        )?;
    }

    let mut weights = HashMap::new();
    for (name, offset, bytes) in in_arena {
        weights.insert(name.to_string(), region.slice(offset, bytes)?);
    }
    sink.tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut at = arena_len;
    for (name, bytes) in &sink.tensors {
        // SAFETY: as above; `outside_budget` reserved this span.
        unsafe { region.write(at, bytes)? };
        weights.insert(name.clone(), region.slice(at, bytes.len() as u64)?);
        at += (bytes.len() as u64).div_ceil(256) * 256;
    }
    Ok((region, weights))
}




#[cfg(test)]
mod tests {
    use super::fits_on_this_gpu;
    use crate::metal::memory::Memory;

    /// The guard the port dropped, restored and held to a number.
    ///
    /// `Memory::headroom` was implemented AND tested the whole time and had
    /// no caller — so what this pins is not the arithmetic (which
    /// `the_tighter_ceiling_wins_and_silence_is_not_a_refusal` already owns)
    /// but that the staging path consults it, refuses by name, and does not
    /// refuse on silence.
    #[test]
    fn a_model_past_the_ceiling_is_refused_before_a_byte_is_staged() {
        let device = Memory {
            device_working_set: 8 << 30,
            reclaimable: 32 << 30,
            ..Memory::default()
        };
        fits_on_this_gpu(&device, 4 << 30).expect("half the ceiling stages");
        fits_on_this_gpu(&device, 8 << 30).expect("exactly the ceiling stages");

        let err = fits_on_this_gpu(&device, 26 << 30)
            .expect_err("a 26 GB model on an 8 GB ceiling must be refused");
        let message = format!("{err:?}");
        assert!(
            message.contains("27917287424") && message.contains("8589934592"),
            "the refusal has to carry BOTH numbers, or the operator learns \
             nothing the SIGKILL would not have told them: {message}"
        );

        // A machine that will not answer is not a machine with no memory.
        fits_on_this_gpu(&Memory::default(), u64::MAX)
            .expect("an unreadable ceiling must not refuse a load");
    }
}
