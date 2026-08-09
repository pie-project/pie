//! The decode step's resident storage: weights, KV, GDN state, IO and the
//! scratch pool, allocated and staged.
//!
//! The C++ (`loader/heap_bind.cpp::stage_decode_storage`) allocates each
//! region and stages the load plan's weights with its own transform loops.
//! This port allocates the same regions, but the weights come from
//! `model_loader::executor::Execution` — the engine `transcode.hpp`
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

use std::ptr::NonNull;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use model_loader::error::Error as LoaderError;
use model_loader::executor::Execution;
use model_loader::executor::chunked::{Chunk, Chunked, chunk_of};
use model_loader::plan::LoadPlan;
use model_loader::plan::spans::{Span, publish_spans};

use crate::layout::region::Region;
use crate::{Error, Result};

use crate::device::allocation::Allocation;
use crate::device::context::Context;
use crate::device::handle::Handle;

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
fn fits_on_this_gpu(memory: &crate::device::memory::Memory, bytes: u64) -> Result<()> {
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
) -> Result<(Vec<Allocation>, HashMap<String, Handle>)> {
    // Where every tensor lands, known from the PLAN before a byte is read --
    // which is what lets the region be allocated first and written into.
    //
    // This used to be a walk over `persistent_offset` written here, and it was
    // WRONG in a way nothing reported: a `CreateView` gives an alias no offset
    // of its own, so every one of them missed the arena and arrived through
    // the sink to be appended as a second copy of bytes already in the region.
    // In a qwen3-0.6B plan that is 140 of 141 tensors. `driver-cuda` resolved
    // the chains and this did not, which is the argument for the walk living
    // in the loader: one answer, or two that differ.
    let published = publish_spans(plan);
    let arena_len = plan.memory.persistent_bytes;
    fits_on_this_gpu(&crate::device::memory::Memory::probe(context), arena_len)?;

    // The tensors that are NOT in the arena, collected as they finalize. A
    // sink rather than a second pass: the executor publishes each exactly
    // once and the arena-resident ones are dropped on the floor here, because
    // their bytes are already where they belong.
    struct Outside<'a> {
        in_arena: &'a BTreeMap<String, Span>,
        tensors: Vec<(String, Vec<u8>)>,
    }
    impl model_loader::executor::sink::TensorSink for Outside<'_> {
        fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoaderError> {
            if !self.in_arena.contains_key(name) {
                self.tensors.push((name.to_string(), bytes.to_vec()));
            }
            Ok(())
        }
    }

    // Allocated BEFORE execution and sized from the plan, then written
    // through. The plan states what it leaves outside and how big each one is,
    // so the region can hold everything and the alternative -- a second region,
    // or one allocation per straggler -- is not needed.
    let outside_budget: u64 = published
        .outside
        .values()
        .map(|bytes| bytes.div_ceil(256) * 256)
        .sum();
    let total = (arena_len + outside_budget).max(1);

    // CHUNKED at 4 GiB, because a recorded command cannot bind past it.
    // `setKernelBuffer:offset:atIndex:` truncates its offset to 32 bits on
    // this hardware (`device_icb.rs` asks the device), so a weight staged
    // more than 4 GiB into one buffer is bound to the wrong bytes when a
    // fire is replayed -- every launch succeeds and the logits are NaN.
    //
    // One region was the right shape for the reason the header gives: a
    // residency entry per tensor, and addresses that move. Five regions for
    // a 17 GB checkpoint keep both properties and cost four more residency
    // entries.
    //
    // Cuts land on tensor boundaries, so no tensor straddles two buffers --
    // a tensor is bound as one operand and must be wholly inside the buffer
    // it is bound from. The writes the executor makes may still straddle,
    // and `Chunked` splits them.
    let cuts = cut_at_tensor_boundaries(&published, arena_len, total);
    let mut chunks = Vec::with_capacity(cuts.len());
    for (i, span) in cuts.windows(2).enumerate() {
        let _ = i;
        chunks.push(Allocation::new(
            context,
            span[1] - span[0],
            "weights region",
        )?);
    }

    let mut sink = Outside {
        in_arena: &published.in_arena,
        tensors: Vec::new(),
    };
    {
        let mut backing = Chunked::new(&chunks, &cuts).map_err(|err| Error::Create {
            what: "staged weights",
            message: err.to_string(),
        })?;
        Execution::new(plan, snapshot_dir)
            .arena(&mut backing)
            .sink(&mut sink)
            .run()
            .map_err(|err| Error::Create {
                what: "staged weights",
                message: err.to_string(),
            })?;
    }

    let slice_at = |offset: u64, bytes: u64| -> Result<Handle> {
        let i = chunk_of(&cuts, offset);
        chunks[i].slice(offset - cuts[i], bytes)
    };
    let mut weights = HashMap::new();
    for (name, span) in &published.in_arena {
        weights.insert(name.clone(), slice_at(span.offset, span.bytes)?);
    }
    sink.tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut at = arena_len;
    for (name, bytes) in &sink.tensors {
        let i = chunk_of(&cuts, at);
        // SAFETY: no GPU work references the chunks yet, and `outside_budget`
        // reserved this span -- which `cut_at_tensor_boundaries` kept whole.
        unsafe { chunks[i].write(at - cuts[i], bytes)? };
        weights.insert(name.clone(), slice_at(at, bytes.len() as u64)?);
        at += (bytes.len() as u64).div_ceil(256) * 256;
    }
    Ok((chunks, weights))
}

/// The most a recorded command can carry as a buffer offset, plus one.
///
/// `device/recording.rs` refuses a bind past this and says why; staging in
/// chunks no larger than it is how a checkpoint stays recordable.
const FOUR_GIB: u64 = 1 << 32;

/// Where to cut `total` bytes into chunks of at most [`FOUR_GIB`], such that
/// no tensor crosses a cut.
///
/// Returns the boundaries, `[0, .., total]`, so chunk `i` is
/// `cuts[i]..cuts[i + 1]` and holds every tensor starting in that range.
///
/// A tensor larger than [`FOUR_GIB`] would force a cut inside itself, and
/// none exists: the largest in any checkpoint here is a 262 144 x 5 376
/// 4-bit embedding table, 0.7 GB. If one ever does, its chunk is simply
/// larger than the ceiling and `record` refuses that fire — which is the
/// same correct-and-slower answer as before, for one model instead of all.
fn cut_at_tensor_boundaries(
    published: &model_loader::plan::spans::Published,
    arena_len: u64,
    total: u64,
) -> Vec<u64> {
    // Every tensor as (start, end), in address order. The arena's spans come
    // out of a `BTreeMap` keyed by name, so they are sorted by name and have
    // to be re-sorted; the stragglers are appended in the order the writer
    // above lays them down.
    let mut spans: Vec<(u64, u64)> = published
        .in_arena
        .values()
        .map(|s| (s.offset, s.offset + s.bytes))
        .collect();
    let mut at = arena_len;
    let mut outside: Vec<(&String, &u64)> = published.outside.iter().collect();
    outside.sort_by(|a, b| a.0.cmp(b.0));
    for (_, bytes) in outside {
        spans.push((at, at + bytes));
        at += bytes.div_ceil(256) * 256;
    }
    spans.sort_unstable();

    let mut cuts = vec![0u64];
    for &(start, end) in &spans {
        let open = *cuts.last().unwrap_or(&0);
        // Cut BEFORE this tensor, not after the last one: a gap between two
        // tensors belongs to whichever side keeps the chunk under the
        // ceiling, and putting it in front of the tensor that would overflow
        // is what keeps the cut on a boundary.
        if end > open + FOUR_GIB && start > open {
            cuts.push(start);
        }
    }
    cuts.push(total);
    cuts.dedup();
    cuts
}

/// `Allocation` as a span the loader may address.
///
/// The whole of what `model_loader::executor::chunked` needs from this
/// driver: a host-visible pointer and a length. The arithmetic over them is
/// the loader's, because the arena is the loader's — see that module for why
/// it is not `executor::metal`.
///
/// # Safety
///
/// `Allocation` owns its `MTLBuffer`, so distinct allocations cannot overlap,
/// and `contents` stays valid for `len` bytes for as long as the allocation
/// is held.
unsafe impl Chunk for Allocation {
    fn base(&self) -> NonNull<u8> {
        self.contents().cast::<u8>()
    }

    fn len(&self) -> u64 {
        Region::len(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{FOUR_GIB, chunk_of, cut_at_tensor_boundaries, fits_on_this_gpu};
    use crate::device::memory::Memory;

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

    /// A tensor is bound as ONE operand, so it must be wholly inside the
    /// buffer it is bound from.
    ///
    /// This is the property the 4 GiB cut exists to keep, and it is the one
    /// that a size-based cut would break: chopping every 4 GiB lands inside
    /// whichever tensor spans the boundary, and that tensor is then bound
    /// from a buffer that holds half of it. Nothing would report it — the
    /// launch succeeds and reads the pages after the buffer.
    #[test]
    fn a_cut_never_lands_inside_a_tensor() {
        // Six 0.7 GB tensors and one 3 GB straggler, which is where a naive
        // cut at exactly 4 GiB would fall.
        let big = 700 * (1 << 20);
        let mut spans = std::collections::BTreeMap::new();
        let mut at = 0u64;
        for i in 0..6 {
            spans.insert(
                format!("t{i}"),
                model_loader::plan::spans::Span {
                    offset: at,
                    bytes: big,
                },
            );
            at += big;
        }
        let huge = 3 * (1u64 << 30);
        spans.insert(
            "straddler".to_string(),
            model_loader::plan::spans::Span {
                offset: at,
                bytes: huge,
            },
        );
        at += huge;

        let published = model_loader::plan::spans::Published {
            in_arena: spans.clone(),
            outside: std::collections::BTreeMap::new(),
        };
        let cuts = cut_at_tensor_boundaries(&published, at, at);

        assert!(
            cuts.len() > 2,
            "7.1 GiB of weights does not fit one chunk: {cuts:?}"
        );
        for span in spans.values() {
            let start = chunk_of(&cuts, span.offset);
            let last = chunk_of(&cuts, span.offset + span.bytes - 1);
            assert_eq!(
                start, last,
                "a tensor at {} for {} bytes crosses a cut in {cuts:?}",
                span.offset, span.bytes
            );
        }
        for pair in cuts.windows(2) {
            assert!(
                pair[1] - pair[0] <= FOUR_GIB,
                "chunk {}..{} is past what a recorded command can bind",
                pair[0],
                pair[1]
            );
        }
    }
}
