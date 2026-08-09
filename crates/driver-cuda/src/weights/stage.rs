//! Executing a compiled plan into device memory.
//!
//! One arena, allocated from the plan before a byte is read, and every tensor
//! a named span inside it. What replaced ~250 lines of hand-written
//! `fuse_llama_like` / `alias_gemma4` / `alias_qwen3_5` in the shell: the
//! joins, the views and the renames are all instructions in the plan, so the
//! bytes land fused and aliased the first and only time they are copied.
//!
//! # What is left here
//!
//! Allocation, and nothing else the plan already answers.
//!
//! This module used to walk the plan itself — resolving `CreateView` chains to
//! find where each tensor lands — and to carry its own `ArenaBacking`
//! implementation, with the pinned staging and the three transform kernels in
//! it. Both were the loader's work done on the driver's side of the boundary,
//! and both had a near-copy in `driver-metal` that had already drifted: only
//! this one resolved views, only that one budgeted for what a plan leaves
//! outside the arena. They are `model_loader::plan::spans::publish_spans` and
//! `model_loader::executor::cuda::CudaArena` now — asked once, answered the
//! same way for both backends.
//!
//! What a driver alone can state is where the memory comes from. `CudaArena`
//! takes a pointer and a length rather than an allocator, so the pool, the VMM
//! reservation or the plain `cudaMalloc` behind it stays this crate's
//! business. That is the whole of the division.

use std::collections::BTreeMap;
use std::path::Path;

use model_loader::error::Error;
use model_loader::executor::Execution;
use model_loader::executor::cuda::CudaArena;
use model_loader::executor::sink::TensorSink;
use model_loader::plan::LoadPlan;
use model_loader::plan::spans::{Span, publish_spans};

use crate::device::{Allocator, DeviceBuffer, OwnedStream};

/// Where one tensor's bytes are: a span of the arena, or its own allocation.
#[derive(Clone, Copy, Debug)]
pub struct WeightSpan {
    /// The device address of the tensor's first byte.
    pub ptr: *mut std::ffi::c_void,
    /// Its length in bytes.
    pub bytes: usize,
}

// SAFETY: a device address is a number; nothing here dereferences it on the
// host, and the CUDA context it belongs to outlives the shell that holds it.
unsafe impl Send for WeightSpan {}
unsafe impl Sync for WeightSpan {}

/// The device residency of an executed plan.
pub struct StagedWeights {
    /// Every tensor the plan names, by the contract's name.
    pub spans: BTreeMap<String, WeightSpan>,
    /// The arena, and any allocation for a tensor published outside it. Held
    /// only to keep the spans alive.
    pub owned: Vec<DeviceBuffer>,
}

/// Tensors with no arena offset arrive through the sink instead. Collected on
/// the host and uploaded after, because their size is not known until they
/// finalize; they are the small minority a plan leaves outside.
///
/// `in_arena` is the LOADER's answer about which names the arena holds, so a
/// tensor that reaches the sink and also has an arena offset — every alias
/// does, which in a qwen3-0.6B plan is 140 of 141 tensors — is dropped here
/// instead of becoming a second resident copy of bytes already on the device.
struct Outside<'a> {
    in_arena: &'a BTreeMap<String, Span>,
    tensors: Vec<(String, Vec<u8>)>,
}

impl TensorSink for Outside<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        if !self.in_arena.contains_key(name) {
            self.tensors.push((name.to_string(), bytes.to_vec()));
        }
        Ok(())
    }
}

/// Allocate the plan's arena on the device, run the plan into it, and name
/// every tensor it holds.
///
/// # Errors
///
/// The device could not hold the arena, or the plan did not execute.
pub fn stage_plan_weights(
    plan: &LoadPlan,
    snapshot_dir: &Path,
    alloc: &Allocator,
) -> Result<StagedWeights, Error> {
    // Known from the PLAN, before a byte is read, which is what lets the arena
    // be allocated first and written straight through.
    let published = publish_spans(plan);

    // The resident tensors AND the scratch behind them: a plan whose device
    // runs its own load-time transforms stages their operands in the arena,
    // so `persistent_bytes` alone is short by exactly that region.
    let arena_len = usize::try_from(plan.memory.arena_bytes())
        .map_err(|_| Error::Contract("arena does not fit the address space".into()))?
        .max(1);
    // The plan's own budget bounds the largest single write, which is what the
    // staging slots have to fit; sizing them from a constant would pin for the
    // largest checkpoint on every load of the smallest.
    let max_write = usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX);

    let buf = alloc.alloc(arena_len).map_err(cuda)?;
    let stream = OwnedStream::new(0).map_err(cuda)?;
    let base = buf.as_ptr();
    {
        // SAFETY: `buf` is a live allocation of `arena_len` bytes that outlives
        // this scope, and `stream` outlives it too.
        let mut arena =
            unsafe { CudaArena::new(base, arena_len, max_write, stream.as_ref().as_raw().cast())? };
        let mut sink = Outside {
            in_arena: &published.in_arena,
            tensors: Vec::new(),
        };
        Execution::new(plan, snapshot_dir)
            .arena(&mut arena)
            .sink(&mut sink)
            .run()?;

        let mut spans = BTreeMap::new();
        for (name, span) in &published.in_arena {
            let offset = usize::try_from(span.offset).unwrap_or(usize::MAX);
            spans.insert(name.clone(), {
                let bytes = usize::try_from(span.bytes).unwrap_or(0);
                // CHECKED, not argued. The SAFETY comment here read
                // "the offset is the plan's own and the arena was
                // sized from `persistent_bytes`, which the executor
                // already refused to exceed" — true, and unavailable
                // to the compiler. The buffer knows its own length.
                let ptr = buf.ptr_at(offset, bytes).ok_or_else(|| {
                    Error::Internal(format!(
                        "{name}: span {offset}..{} leaves an arena of \
                             {arena_len}",
                        offset + bytes
                    ))
                })?;
                WeightSpan { ptr, bytes }
            });
        }

        let mut owned = vec![buf];
        for (name, bytes) in sink.tensors {
            let mut b = alloc.alloc(bytes.len().max(1)).map_err(cuda)?;
            b.copy_from_host(&bytes, stream.as_ref()).map_err(cuda)?;
            spans.insert(
                name,
                WeightSpan {
                    ptr: b.as_ptr(),
                    bytes: bytes.len(),
                },
            );
            owned.push(b);
        }
        stream.as_ref().synchronize().map_err(cuda)?;
        Ok(StagedWeights { spans, owned })
    }
}

fn cuda(e: crate::Error) -> Error {
    Error::Contract(format!("staging: {e:?}"))
}

/// Read a staged tensor's bytes back to the host.
///
/// For the handful of load-time scalars a family needs on the HOST — gemma-4's
/// per-layer `layer_scalar`, the C++ `read_bf16_scalar_once`. Synchronous, and
/// meant to be: it runs a few dozen times at load and never again.
///
/// # Errors
///
/// The copy faulted.
pub fn read_span(span: WeightSpan) -> Result<Vec<u8>, Error> {
    let mut out = vec![0u8; span.bytes];
    let stream = OwnedStream::new(0).map_err(cuda)?;
    // SAFETY: `span` names a live device allocation of `bytes` bytes —
    // `stage_plan_weights` bounds-checked it against the arena — and
    // `stream` outlives the copy.
    unsafe { crate::device::read_raw_span(span.ptr, &mut out, stream.as_ref()) }.map_err(cuda)?;
    stream.as_ref().synchronize().map_err(cuda)?;
    Ok(out)
}
