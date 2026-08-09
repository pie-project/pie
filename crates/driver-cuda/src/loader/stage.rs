//! Executing a compiled plan into device memory.
//!
//! One arena, allocated from the plan before a byte is read, and every tensor
//! a named span inside it. What replaced ~250 lines of hand-written
//! `fuse_llama_like` / `alias_gemma4` / `alias_qwen3_5` in the shell: the
//! joins, the views and the renames are all instructions in the plan, so the
//! bytes land fused and aliased the first and only time they are copied.
//!
//! The old load read each checkpoint tensor into a host `Vec`, uploaded it,
//! then read three of them BACK off the device to concatenate a `qkv` and
//! uploaded that. Three round trips for a tensor the plan can lay out in one.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use model_loader::error::Error;
use model_loader::executor::host::execute_plan_into_backing;
use model_loader::executor::sink::TensorSink;
use model_loader::plan::LoadPlan;

use super::arena::DeviceArena;
use crate::cuda::{Allocator, DeviceBuffer, OwnedStream};

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
    let names: HashMap<_, _> = plan
        .tensors
        .iter()
        .map(|t| (t.id, t.name.as_str()))
        .collect();
    // (name, offset, bytes) for everything the arena holds — known from the
    // PLAN, before a byte is read, which is what lets the arena be allocated
    // first and written straight through.
    let mut in_arena: Vec<(&str, u64, u64)> = Vec::new();
    for b in &plan.buffers {
        let (Some(offset), Some(tensor)) = (b.persistent_offset, b.tensor) else {
            continue;
        };
        if let Some(name) = names.get(&tensor) {
            in_arena.push((name, offset, b.bytes));
        }
    }
    // VIEWS ARE NOT COPIES. A `CreateView` gives a buffer no
    // `persistent_offset` of its own even when its input has one — the alias
    // `layer.3.attn_norm` onto `model.layers.3.input_layernorm.weight` is one,
    // and so is every unfused fallback view into a fused projection. Resolving
    // them here is what keeps them from arriving through the sink as a second
    // resident copy of bytes already on the device: 140 of 141 tensors in a
    // qwen3-0.6B plan.
    type ViewOf = HashMap<model_loader::types::BufferId, (model_loader::types::BufferId, u64, u64)>;
    let mut view_of: ViewOf = HashMap::new();
    for instr in &plan.instrs {
        if let model_loader::plan::StorageInstr::CreateView {
            input,
            output,
            view,
            ..
        } = instr
        {
            let bytes = view
                .stride
                .dims
                .iter()
                .try_fold(1u64, |n, d| u64::try_from(d.count).ok().map(|c| n * c))
                .unwrap_or(0)
                * u64::from(view.stride.element_bytes);
            view_of.insert(
                *output,
                (*input, view.offset + view.stride.base_offset, bytes),
            );
        }
    }
    let offset_of: HashMap<_, _> = plan
        .buffers
        .iter()
        .filter_map(|b| Some((b.id, b.persistent_offset?)))
        .collect();
    let arena_offset = |mut id: model_loader::types::BufferId| -> Option<u64> {
        let mut extra = 0;
        // A view of a view is legal; the chain is short and acyclic by
        // construction, so the bound is a guard and not a search limit.
        for _ in 0..16 {
            if let Some(base) = offset_of.get(&id) {
                return Some(base + extra);
            }
            let (parent, delta, _) = *view_of.get(&id)?;
            extra += delta;
            id = parent;
        }
        None
    };
    for b in &plan.buffers {
        if b.persistent_offset.is_some() {
            continue;
        }
        let (Some(tensor), Some(offset)) = (b.tensor, arena_offset(b.id)) else {
            continue;
        };
        // A view's own `bytes` is zero — its length lives in the extent the
        // `CreateView` carries.
        let bytes = if b.bytes == 0 {
            view_of.get(&b.id).map_or(0, |(_, _, len)| *len)
        } else {
            b.bytes
        };
        if let Some(name) = names.get(&tensor) {
            in_arena.push((name, offset, bytes));
        }
    }
    let arena_names: HashSet<&str> = in_arena.iter().map(|(n, _, _)| *n).collect();

    // Tensors with no arena offset arrive through the sink instead. Collected
    // on the host and uploaded after, because their size is not known until
    // they finalize; they are the small minority a plan leaves outside.
    struct Outside<'a> {
        wanted: &'a HashSet<&'a str>,
        tensors: Vec<(String, Vec<u8>)>,
    }
    impl TensorSink for Outside<'_> {
        fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
            if !self.wanted.contains(name) {
                self.tensors.push((name.to_string(), bytes.to_vec()));
            }
            Ok(())
        }
    }

    let arena_len = usize::try_from(plan.memory.persistent_bytes)
        .map_err(|_| Error::Contract("arena does not fit the address space".into()))?;
    // The plan's own budget bounds the largest single write, which is what
    // the staging slots have to fit; sizing them from a constant would pin
    // for the largest checkpoint on every load of the smallest.
    let max_write = usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX);
    let mut arena = DeviceArena::new(arena_len.max(1), max_write, alloc)?;
    let mut sink = Outside {
        wanted: &arena_names,
        tensors: Vec::new(),
    };
    execute_plan_into_backing(plan, snapshot_dir, &mut arena, &mut sink, &mut |_| {})?;
    // A decline is silent by design — the load is correct either way — so
    // say it once here rather than leave "the loader transforms on the GPU"
    // as a claim with no observation behind it.
    let declined = arena.declined_transforms();
    if declined > 0 {
        eprintln!(
            "[driver-cuda] load: {declined} transform(s) ran on the host; \
             the device arena declined them"
        );
    }
    let buf = arena.into_buffer()?;

    let base = buf.as_ptr();
    let mut spans = BTreeMap::new();
    for (name, offset, bytes) in in_arena {
        let offset = usize::try_from(offset).unwrap_or(usize::MAX);
        spans.insert(
            name.to_string(),
            WeightSpan {
                // SAFETY: the offset is the plan's own and the arena was sized
                // from `persistent_bytes`, which the executor already refused
                // to exceed.
                ptr: unsafe { base.byte_add(offset) },
                bytes: usize::try_from(bytes).unwrap_or(0),
            },
        );
    }

    let mut owned = vec![buf];
    if !sink.tensors.is_empty() {
        let stream = OwnedStream::new(0).map_err(cuda)?;
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
    }
    Ok(StagedWeights { spans, owned })
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
    // SAFETY: `span` names a live device allocation of `bytes` bytes, and the
    // destination is a host `Vec` of the same length.
    let status = unsafe {
        cudarc::runtime::sys::cudaMemcpyAsync(
            out.as_mut_ptr().cast(),
            span.ptr,
            span.bytes,
            cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.as_ref().as_raw(),
        )
    };
    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        return Err(Error::Contract(format!("read_span: {status:?}")));
    }
    stream.as_ref().synchronize().map_err(cuda)?;
    Ok(out)
}
