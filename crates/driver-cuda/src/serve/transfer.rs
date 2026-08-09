//! Moving and resizing what the fires read: KV pages, recurrent state, and
//! the pools that hold them.
//!
//! Three exports that touch device memory and nothing else. They share a
//! shape — validate every index, then move — which `.wiki/driver/graph.md`
//! §3 rule 3 states as a rule and `store/control.rs` on the Metal side
//! records the bug it prevents.

use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR,
    PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_OK,
    PIE_STATUS_UNSUPPORTED,
    PieCompletion,
    PieDriver,
    PieKvCopyDesc,
    PiePoolResizeDesc,
    PieStateCopyDesc,
};
use super::{checked, guard};
use super::state::{KvState, SwapPool, shell, slice_of};

/// KV copies across all four domains: whole-page moves (`src_page_ids` →
/// `dst_page_ids`, every layer, every buffer) and beam-repair CELL moves
/// through the bridged `copy_kv_cells_bf16`.
///
/// The host legs go through `pools::swap_pool` and `layout::swap_plan`,
/// which is a change of substance and not of spelling. This used to
/// refuse them, then quietly grew an inline pool that assumed TWO buffers
/// per layer of ONE width — so a quantized cache (four buffers: `k`, `v`,
/// `k_scale`, `v_scale`) and gemma-4 (layers disagreeing on head dim)
/// were both turned away by a stride check rather than served. The ported
/// pool derives its regions from `KvCacheLayout::page_buffers`, which
/// answers per layer and per buffer, and `SwapPlan` builds the copy list
/// against that same geometry.
pub fn pie_cuda_copy_kv(
    driver: *mut PieDriver,
    copy: *const PieKvCopyDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_copy_kv", PIE_STATUS_DRIVER_ERROR, move || {
        use driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE;

        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(copy, driver_api::local::validate_kv_copy_desc, "copy_kv") {
            Ok(d) => d,
            Err(status) => return status,
        };
        let host_src = desc.src_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
        let host_dst = desc.dst_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
        if host_src && host_dst {
            eprintln!("[driver-cuda] copy_kv: host-to-host moves have no device leg");
            return PIE_STATUS_UNSUPPORTED;
        }
        let (Some(model), Some(_kv)) = (state.model.as_ref(), state.kv.as_ref()) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let src_pages = slice_of(desc.src_page_ids.ptr, desc.src_page_ids.len);
        let dst_pages = slice_of(desc.dst_page_ids.ptr, desc.dst_page_ids.len);
        if src_pages.len() != dst_pages.len() {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let cells = slice_of(desc.cells.ptr, desc.cells.len);
        if (host_src || host_dst) && !cells.is_empty() {
            return PIE_STATUS_INVALID_ARGUMENT; // cell moves are device-only
        }
        let (kv_heads, head_dim) = (
            i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
            i32::try_from(model.deployment.shape.head_dim_alloc()).unwrap_or(0),
        );
        let page_size: i32 = 16;
        let layers_n = model.deployment.layers as usize;

        // THE POOL, planned rather than assumed. `page_buffers` answers
        // per layer AND per buffer, so a quantized cache's scale planes and
        // gemma-4's differing head dims are geometry rather than a refusal.
        if host_src || host_dst {
            use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};
            let kv_ref = state.kv.as_ref().expect("checked");
            let layout = kv_ref.cache.layout();
            let device_buffers: Vec<Vec<u64>> = (0..kv_ref.layers())
                .map(|l| {
                    layout
                        .page_buffers(i32::try_from(l).unwrap_or(0))
                        .into_iter()
                        .map(|(_, bytes)| bytes)
                        .collect()
                })
                .collect();
            let host_ids = if host_src { src_pages } else { dst_pages };
            let need = host_ids.iter().copied().max().map_or(1, |m| m + 1);
            let plan = crate::pools::swap_pool::SwapPoolLayout::for_cache(
                &device_buffers,
                i32::try_from(need).unwrap_or(i32::MAX),
                page_size,
                kv_heads,
                head_dim,
            );
            // Reuse when the geometry AND the capacity still cover this
            // move. `check_against` is the pool's own test that the device
            // side has not changed shape underneath it — a resize that
            // moved a stride would otherwise write host pages of the wrong
            // width, which is corruption rather than degradation.
            let reusable = matches!(&state.swap, Some(sp)
                if sp.plan.num_pages() >= plan.num_pages()
                    && sp.plan.geometry() == plan.geometry());
            if !reusable {
                let mut ops = LiveStagingOps;
                let mut regions = Vec::with_capacity(plan.buffers().len());
                for b in plan.buffers() {
                    let Some(p) = ops.malloc_host(usize::try_from(b.nbytes).unwrap_or(0)) else {
                        for &r in &regions {
                            ops.free_host(r);
                        }
                        return PIE_STATUS_EXHAUSTED;
                    };
                    regions.push(p);
                }
                // The two stream ROLES the plan asks for, created once and
                // kept: a restore is on the critical path — an evicted
                // process cannot run until its pages are back — and queueing
                // it behind pending evictions is the stall the second stream
                // exists to avoid. The old code made a fresh stream per
                // call, which is neither.
                let st = plan.streams();
                let mk = |want: bool| -> Option<crate::device::OwnedStream> {
                    want.then(|| crate::device::OwnedStream::new(0).ok()).flatten()
                };
                let (evict, restore) = (mk(st.evict), mk(st.restore));
                if let Some(old) = state.swap.take() {
                    old.free();
                }
                state.swap = Some(SwapPool { regions, plan, evict, restore });
            }
        }

        // The stream the DEVICE legs and the cell moves ride. The host
        // legs take the pool's own evict/restore streams instead, which is
        // why those are made with the pool rather than here.
        let stream = match crate::device::OwnedStream::new(0) {
            Ok(s) => s,
            Err(_) => return PIE_STATUS_DRIVER_ERROR,
        };
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let kv_ref = state.kv.as_ref().expect("checked");
        for (s_id, d_id) in src_pages.iter().zip(dst_pages) {
            if (!host_src && *s_id >= kv_ref.num_pages)
                || (!host_dst && *d_id >= kv_ref.num_pages)
            {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }

        // THE COPY LIST IS BUILT, not nested by hand.
        //
        // `SwapPlan::build` walks layer × page × buffer and emits one
        // `CopyOp` per contiguous move, with the offsets in each side's own
        // index space. That last part is the reason it exists: the two
        // pools generally have different capacities, so a transposed
        // src/dst is not reliably caught by a bounds check, and `Direction`
        // names which side is which so a call site cannot read
        // ambiguously.
        use crate::layout::swap_plan::{Direction, Pool, SwapPlan};
        let direction = match (host_src, host_dst) {
            (false, true) => Direction::DeviceToHost,
            (true, false) => Direction::HostToDevice,
            _ => Direction::DeviceToDevice,
        };
        // The DEVICE geometry, which is what a device-to-device move is in.
        // A host leg uses the pool's, which was planned from this same
        // `page_buffers` and therefore agrees by construction.
        let layout = kv_ref.cache.layout();
        let dev_geometry = crate::layout::swap_plan::PoolGeometry::new(
            (0..kv_ref.layers())
                .map(|l| {
                    layout
                        .page_buffers(i32::try_from(l).unwrap_or(0))
                        .into_iter()
                        .map(|(_, bytes)| bytes)
                        .collect()
                })
                .collect(),
        );
        let geometry = match (host_src || host_dst, state.swap.as_ref()) {
            (true, Some(sp)) => sp.plan.geometry().clone(),
            _ => dev_geometry,
        };
        let Ok(plan) = SwapPlan::build(&geometry, direction, src_pages, dst_pages) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        // A host leg rides the pool's own stream for its direction; a
        // device leg rides this call's.
        let leg = match (direction, state.swap.as_ref()) {
            (Direction::DeviceToHost, Some(sp)) => sp.evict.as_ref(),
            (Direction::HostToDevice, Some(sp)) => sp.restore.as_ref(),
            _ => None,
        }
        .map_or_else(|| stream.as_ref(), crate::device::OwnedStream::as_ref);
        for op in plan.ops() {
            // A layer that owns no pages has nothing to move: gemma-4's
            // KV-shared trailing layers read through their source's pool,
            // so visiting them would move the same bytes twice.
            let resolve = |e: Pool| -> Option<*mut u8> {
                match e {
                    Pool::Device { layer, buffer } => {
                        let (k, v) = kv_ref.owned(layer as usize)?;
                        // Buffers beyond the k/v pair are the quantized
                        // format's scale planes, which this shell does not
                        // hand out yet -- and cannot move what it cannot
                        // address.
                        match buffer {
                            0 => Some(k.cast::<u8>()),
                            1 => Some(v.cast::<u8>()),
                            _ => None,
                        }
                    }
                    Pool::Host { layer, buffer } => {
                        state.swap.as_ref()?.region(layer, buffer)
                    }
                }
            };
            let (Some(dst), Some(src)) = (resolve(op.dst), resolve(op.src)) else {
                // Both sides absent is a layer with no pages; one side
                // absent is a buffer this shell cannot address, and a
                // partial move is worse than none.
                if resolve(op.dst).is_some() != resolve(op.src).is_some() {
                    eprintln!(
                        "[driver-cuda] copy_kv: this cache carries a buffer the \
                         shell cannot address, so the move would be partial"
                    );
                    // DRAIN WHAT IS ALREADY QUEUED. Earlier buffers of
                    // this same op enqueued onto `leg`, and a later
                    // `copy_kv` that finds the pool unusable does
                    // `swap.take(); old.free()` — a `cudaFreeHost` of
                    // regions those copies are still reading.
                    let _ = leg.synchronize();
                    return PIE_STATUS_UNSUPPORTED;
                }
                continue;
            };
            let kind = match direction {
                Direction::DeviceToHost => cudaMemcpyKind::cudaMemcpyDeviceToHost,
                Direction::HostToDevice => cudaMemcpyKind::cudaMemcpyHostToDevice,
                Direction::DeviceToDevice => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                Direction::HostToHost => cudaMemcpyKind::cudaMemcpyHostToHost,
            };
            let code = unsafe {
                cudaMemcpyAsync(
                    dst.add(usize::try_from(op.dst_offset).unwrap_or(0)).cast(),
                    src.add(usize::try_from(op.src_offset).unwrap_or(0)).cast_const().cast(),
                    usize::try_from(op.bytes).unwrap_or(0),
                    kind,
                    leg.as_raw(),
                )
            };
            if code != cudaError::cudaSuccess {
                // Same reason as the partial-move return above.
                let _ = leg.synchronize();
                return PIE_STATUS_DRIVER_ERROR;
            }
        }

        // Cell moves: the bridged beam-repair launcher, per layer. Disjoint
        // spans are the CALLER's contract, as the kernel's header states.
        if !cells.is_empty() {
            let alloc = crate::device::Allocator::new();
            let up = |vals: &[u32]| -> Result<crate::device::DeviceBuffer, i32> {
                let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut b = alloc.alloc(bytes.len())?;
                b.copy_from_host(&bytes, stream.as_ref())?;
                Ok(b)
            };
            let dp: Vec<u32> = cells.iter().map(|c| c.dst_page_id).collect();
            let doff: Vec<u32> = cells.iter().map(|c| c.dst_token_offset).collect();
            let sp: Vec<u32> = cells.iter().map(|c| c.src_page_id).collect();
            let soff: Vec<u32> = cells.iter().map(|c| c.src_token_offset).collect();
            let (d_dp, d_doff, d_sp, d_soff) = match (up(&dp), up(&doff), up(&sp), up(&soff)) {
                (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
                _ => return PIE_STATUS_EXHAUSTED,
            };
            // Only the layers that OWN pages: a shared layer's cells live
            // in its source's pool, so visiting it too would copy them
            // twice.
            for i in 0..kv_ref.layers() {
                let (Some((k, v)), Some(d)) = (kv_ref.owned(i), kv_ref.head_dim(i)) else {
                    continue;
                };
                let layer = crate::bind::abi::KvCacheLayerView {
                    layer: i as i32,
                    source_layer: i as i32,
                    num_pages: kv_ref.num_pages as i32,
                    page_size,
                    num_kv_heads: kv_heads,
                    head_dim: d,
                    scheme: crate::bind::abi::KvCacheScheme::Native,
                    storage_dtype: crate::dtype::DType::Bf16,
                    block_size: 0,
                    k_pages: k,
                    v_pages: v,
                    k_scales: core::ptr::null_mut(),
                    v_scales: core::ptr::null_mut(),
                    k_bf16_pages: k,
                    v_bf16_pages: v,
                    k_env_min: core::ptr::null_mut(),
                    k_env_max: core::ptr::null_mut(),
                    hnd_layout: false,
                    native_bf16: true,
                };
                unsafe {
                    crate::bind::abi::ffi::pie_k_attn_copy_kv_cells_bf16(
                        layer,
                        d_dp.as_ptr().cast(),
                        d_doff.as_ptr().cast(),
                        d_sp.as_ptr().cast(),
                        d_soff.as_ptr().cast(),
                        i32::try_from(cells.len()).unwrap_or(i32::MAX),
                        stream.as_ref().as_raw().cast(),
                    );
                }
            }
        }
        // BOTH STREAMS, and it synchronized only one.
        //
        // A HOST leg rides the pool's own `evict`/`restore` stream — that
        // is what `leg` selects above — while the cell moves ride this
        // call's. `OwnedStream::new` asks for `cudaStreamNonBlocking`
        // precisely so unrelated streams do NOT order against each other,
        // so synchronizing `stream` said nothing about the page copies.
        //
        // The engine was therefore told an evicted process's pages were
        // back while the H2D was still queued: it schedules a fire, the
        // fire reads KV pages that are partly unwritten, and the logits
        // are wrong rather than late. The other direction frees device
        // pages the D2H is still reading.
        //
        // `leg` is `stream` when the pool has no stream for this
        // direction, so the second synchronize is a no-op in the device
        // case rather than a special case here.
        // BOTH, not `||`. Short-circuiting on the first failure left the
        // pool's stream loaded, which is the state the whole fix is about.
        let (a, b) = (stream.as_ref().synchronize(), leg.synchronize());
        if a.is_err() || b.is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if let Some(notify) = state.notify {
            unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
        }
        PIE_STATUS_OK
    })
}

/// Direct recurrent-state copies: WHOLE-SLOT d2d over the hybrid's GDN
/// slabs (conv + recurrent, every linear layer), the C++ shape
/// (`context.cpp` ignores the token fields — those ride for the rs
/// BUFFER pool, spec-decode machinery). Slot ids are the engine's; the
/// slabs grow with migration to cover them.
pub fn pie_cuda_copy_state(
    driver: *mut PieDriver,
    copy: *const PieStateCopyDesc,
    _completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_copy_state", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(copy, driver_api::local::validate_state_copy_desc, "copy_state") {
                Ok(d) => d,
                Err(status) => return status,
            };
        let Some(gdn) = state.gdn.as_mut() else {
            // No recurrent family is loaded — the C++ shape: state copies
            // only mean something once the rs cache exists.
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let ranges = slice_of(desc.slot_ranges.ptr, desc.slot_ranges.len);
        let Ok(stream) = crate::device::OwnedStream::new(0) else {
            return PIE_STATUS_DRIVER_ERROR;
        };
        let alloc = crate::device::Allocator::new();
        let need = ranges
            .iter()
            .map(|r| r.src_slot_id.max(r.dst_slot_id) + 1)
            .max()
            .unwrap_or(0);
        // The epoch rides IN, because growing is what bumps it. This
        // call site used to do it by hand and the two in the fire path
        // did too — three copies of one rule, and a fourth caller that
        // forgot would be a capture replaying into a freed slab.
        if let Err(code) = gdn.ensure_slots(need, &mut state.fire_arrays.epoch, &alloc, &stream) {
            return code;
        }
        for range in ranges {
            // WHOLE SLOTS, and this comment used to name the function it
            // was reimplementing: "The C++ (`context.cpp::copy_state`)
            // copies WHOLE SLOTS (`rs_cache->copy_slot_d2d(src, dst)`)".
            // It calls it now. The token fields still ride for the rs
            // BUFFER pool, which is spec-decode machinery.
            //
            // `copy_slot_d2d` carries the MTP pending-hidden row too where
            // `copy_linear_state_slot_d2d` (the fold's) deliberately does
            // not -- a state copy is a clone and wants everything, a fold
            // is a rollback and would overwrite a newer MTP value with an
            // older one.
            let Ok(ops) = gdn.cache.copy_slot_d2d(
                i32::try_from(range.src_slot_id).unwrap_or(-1),
                i32::try_from(range.dst_slot_id).unwrap_or(-1),
            ) else {
                eprintln!("[driver-cuda] copy_state: a range names a slot the cache lacks");
                return PIE_STATUS_INVALID_ARGUMENT;
            };
            if let Err(code) = gdn.apply(&ops, stream.as_ref()) {
                return code;
            }
        }
        if stream.as_ref().synchronize().is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        PIE_STATUS_OK
    })
}

/// Resize the KV pool to `target_pages`, MIGRATING the surviving pages —
/// the migration the launch-time growth deliberately skipped. Shrinks
/// drop the tail; `map_ranges`/`unmap_ranges` (the elastic-VMM form) are
/// accepted but the shell's pools are plain allocations, so the target
/// page count is the whole contract here — stated, not hidden.
pub fn pie_cuda_resize_pool(
    driver: *mut PieDriver,
    resize: *const PiePoolResizeDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_resize_pool", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(resize, driver_api::local::validate_pool_resize_desc, "resize_pool") {
                Ok(d) => d,
                Err(status) => return status,
            };
        let Ok(target) = u32::try_from(desc.target_pages) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        if target == 0 {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let Some(model) = state.model.as_ref() else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        // The head dim is per LAYER on the row and no longer needed as a
        // scalar here: `PerLayer::head_dim` below takes the table.
        let kv_heads = i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0);
        let page_size: usize = 16;

        let stream = match crate::device::OwnedStream::new(0) {
            Ok(s) => s,
            Err(_) => return PIE_STATUS_DRIVER_ERROR,
        };
        let alloc = crate::device::Allocator::new();
        // BORROWED, not taken. This read `state.kv.take()`, and seven
        // early returns sit between here and the `install_kv` that
        // repopulates it — each one dropping the cache, freeing every KV
        // page, and leaving the shell with `kv: None`.
        //
        // The realistic one is `materialize` answering
        // `PIE_STATUS_EXHAUSTED`, which is exactly what an engine asking
        // to GROW past available VRAM gets and exactly the status it is
        // meant to treat as retryable. It saw a recoverable "not enough
        // memory" and an unrecoverably dead driver: every later
        // `pie_cuda_launch` fails on `state.kv.as_ref()` and every
        // request's context is gone.
        //
        // `old` is only ever READ here — the geometry and the page
        // contents to copy forward — so a borrow is all it needed.
        // `install_kv` drops the old one at the end, where the new one is
        // already in hand. `pie_cuda_copy_kv` in this same file shows the
        // shape: build the replacement, then release what it replaces.
        let old = state.kv.as_ref();
        // Per-layer page bytes: an existing pool states its own stride
        // (the two-head-dim families' rows disagree); before any pool
        // exists the DEPLOYMENT decides.
        //
        // It used to be re-derived here from `hf.layer_types` — a
        // gemma-4 sniff on a weight name, a `num_kv_shared_layers`
        // subtraction, and a reverse scan for the last earlier layer of
        // the same kind — which is `fire/launch.rs`'s per-layer table
        // written a second time, from a different source, and required
        // to come out identical. It is not a hypothetical: a resize that
        // laid out one stride while the fire path bound another would
        // read every shared layer's pages at the wrong pitch and emit
        // plausible tokens.
        //
        // The row states the table once. Both readers take it.
        let dep = &model.deployment;
        let n_layers = dep.layers as usize;

        // A resize changes the page COUNT and nothing else, so an
        // existing cache states its own geometry and nothing is
        // re-derived. Before any cache exists there is nothing to ask,
        // and the deployment answers.
        let layout = match old.as_ref().map(|o| o.cache.layout().with_num_pages(target as i32)) {
            Some(Ok(l)) => l,
            Some(Err(_)) => return PIE_STATUS_INVALID_ARGUMENT,
            None => {
                let per = crate::pools::kv_cache::PerLayer {
                    head_dim: dep.attention.iter().map(|a| a.head_dim as i32).collect(),
                    // gemma-4's tail attends through the last earlier
                    // layer of its own kind; every other family owns its
                    // pages. A fact about a LAYER, and the row says it.
                    kv_source_layer: dep.attention.iter().map(|a| a.kv_source as i32).collect(),
                    num_kv_heads: vec![kv_heads; n_layers],
                };
                if per.check_sharing().is_err() {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                let f = state.kv_format;
                match crate::pools::kv_cache::KvCacheLayout::plan_per_layer(
                    n_layers as i32,
                    target as i32,
                    page_size as i32,
                    kv_heads,
                    per,
                    f,
                    false,
                ) {
                    Ok(l) => l,
                    Err(_) => return PIE_STATUS_INVALID_ARGUMENT,
                }
            }
        };

        let mut ops = crate::pools::kv_cache_live::LiveKvCacheOps::new(
            stream.as_ref(),
            &alloc,
        );
        let Ok(cache) = crate::pools::kv_cache_live::KvCache::materialize(layout, &mut ops) else {
            return PIE_STATUS_EXHAUSTED;
        };
        let mut held = ops.into_held();
        for b in &mut held {
            if b.memset(0, stream.as_ref()).is_err() {
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        let fresh = KvState { cache, _held: held, num_pages: target };

        // Carry over what still fits. A layer that owns no pages has none
        // to carry, and a shrink keeps the low pages.
        if let Some(old_kv) = &old {
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            for i in 0..fresh.layers() {
                let (Some((nk, nv)), Some((ok_, ov)), Some(pb)) =
                    (fresh.owned(i), old_kv.owned(i), fresh.page_bytes(i))
                else {
                    continue;
                };
                let keep = old_kv.num_pages.min(target) as usize * pb;
                for (dst, src) in [(nk, ok_), (nv, ov)] {
                    let code = unsafe {
                        cudaMemcpyAsync(
                            dst,
                            src.cast_const(),
                            keep,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_ref().as_raw(),
                        )
                    };
                    if code != cudaError::cudaSuccess {
                        return PIE_STATUS_DRIVER_ERROR;
                    }
                }
            }
        }
        if stream.as_ref().synchronize().is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        // A RESIZE MOVES THE KV PAGES, and a captured graph baked their old
        // addresses into every attention launch. `install_kv` is what tells
        // `Recordings` to recapture instead of replaying into memory the
        // pool no longer owns — which showed up as a segfault the moment decode
        // fires became capturable at all.
        crate::serve::state::install_kv(&mut state.kv, &mut state.fire_arrays.epoch, fresh);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if let Some(notify) = state.notify {
            unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
        }
        PIE_STATUS_OK
    })
}

