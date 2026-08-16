//! Executing a compiled plan into device memory: one arena, allocated before
//! a byte is read, every tensor a named span inside it.

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

// SAFETY: a device address is a number, never dereferenced on the host; its CUDA
// context outlives the shell that holds it.
unsafe impl Send for WeightSpan {}
unsafe impl Sync for WeightSpan {}

/// The device residency of an executed plan.
pub struct StagedWeights {
    /// Every tensor the plan names, by the contract's name.
    pub spans: BTreeMap<String, WeightSpan>,
    /// The arena and any out-of-arena allocations, held to keep spans alive.
    pub owned: Vec<DeviceBuffer>,
}

/// Tensors with no arena offset arrive through the sink and are uploaded
/// after; one that also has an arena offset is skipped here.
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

/// Allocate the plan's arena, run the plan into it, and name every tensor.
/// # Errors
/// The device could not hold the arena, or the plan did not execute.
pub fn stage_plan_weights(
    plan: &LoadPlan,
    snapshot_dir: &Path,
    alloc: &Allocator,
) -> Result<StagedWeights, Error> {
    let published = publish_spans(plan);

    // arena_bytes includes load-time transform scratch, unlike persistent_bytes.
    let arena_len = usize::try_from(plan.memory.arena_bytes())
        .map_err(|_| Error::Contract("arena does not fit the address space".into()))?
        .max(1);
    let max_write = usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX);

    let buf = alloc.alloc(arena_len).map_err(|e| {
        Error::Contract(format!(
            "staging: arena of {arena_len} bytes ({:.1} GiB) did not fit the \
             device: {e:?}",
            arena_len as f64 / (1024.0 * 1024.0 * 1024.0),
        ))
    })?;
    let stream = OwnedStream::new(0).map_err(cuda)?;
    let base = buf.as_ptr();
    {
        // SAFETY: `buf` is a live `arena_len`-byte allocation outliving this scope;
        // `stream` outlives it too.
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
                // `ptr_at` bounds the span against the buffer's own length.
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

/// Read a staged tensor's bytes back to the host; load-time scalars only,
/// synchronous, run a few dozen times at load. # Errors: the copy faulted.
pub fn read_span(span: WeightSpan) -> Result<Vec<u8>, Error> {
    let mut out = vec![0u8; span.bytes];
    let stream = OwnedStream::new(0).map_err(cuda)?;
    // SAFETY: `span` names a live allocation bounds-checked by `stage_plan_weights`;
    // `stream` outlives it.
    unsafe { crate::device::read_raw_span(span.ptr, &mut out, stream.as_ref()) }.map_err(cuda)?;
    stream.as_ref().synchronize().map_err(cuda)?;
    Ok(out)
}
