//! Moving and resizing what the fires read: KV pages, recurrent state, and
//! the pools that hold them.
//!
//! Three exports that touch device memory and nothing else, sharing one shape:
//! validate every index, then move.

use super::guard;
use super::state::{KvState, Shell, SwapPool};
use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED, PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_UNSUPPORTED,
};

/// KV copies across all four domains: whole-page moves (every layer, every
/// buffer) and beam-repair cell moves through `copy_kv_cells_bf16`. Host legs
/// derive regions per layer and per buffer from `KvCacheLayout::page_buffers`,
/// so scale planes and gemma-4's differing head dims are geometry, not refusal.
impl Shell {
    /// Move KV pages, within this device or across the host boundary.
    ///
    /// # Errors
    ///
    /// A plan whose ends are not both in a domain this driver owns, or a
    /// page list the pool does not hold.
    pub fn copy_kv(
        &mut self,
        copy: &driver_api::KvCopyPlan,
        completion: driver_api::completion::CompletionTarget,
    ) -> Result<(), i32> {
        guard("copy_kv", Err(PIE_STATUS_DRIVER_ERROR), move || {
            use driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE;

            let state = self;
            // The two rules a `Vec` does not state: both domains name a real
            // one, and the page lists are parallel.
            if let Err(why) = copy.validate() {
                eprintln!("[driver-cuda] copy_kv: {why}");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            let desc = copy;
            let host_src = desc.src_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
            let host_dst = desc.dst_domain != PIE_MEMORY_DOMAIN_CUDA_DEVICE;
            if host_src && host_dst {
                eprintln!("[driver-cuda] copy_kv: host-to-host moves have no device leg");
                return Err(PIE_STATUS_UNSUPPORTED);
            }
            // The pool is ELASTIC: it holds what the frames so far have needed,
            // not what the scheduler is entitled to hand out, and a copy plan's
            // destination can name a page one past that high-water mark. Grow
            // for it rather than refusing.
            //
            // `driver-vulkan` and `driver-wgpu` each carry a device test for
            // exactly this (`a copy plan whose destination is above the pool
            // GROWS it`), found by the curated-inferlet sweep on
            // `prefix-tree-kv-cache` and only when it ran after the others --
            // the signature of a driver whose answer depends on what preceded
            // it. On CUDA it read
            //
            //     pre-launch KV copy rejected: driver-cuda copy_kv failed with
            //     status -1
            //
            // and the bounds check below is right about the pool as it IS and
            // has no way to know what it could be, so a prefix share aimed one
            // page past the last prefill's high-water mark died, and the
            // conversation died with it.
            //
            // DESTINATIONS only, and only when the destination is the device.
            // A SOURCE above the pool stays refused: this pool only ever grows
            // on demand, so a page it has never held is a page nothing has ever
            // written, and growing for it would turn a refusal into a copy of
            // freshly zeroed memory -- history-shaped silence rather than an
            // error.
            if !host_dst {
                let need = desc
                    .dst_page_ids
                    .iter()
                    .copied()
                    .chain(desc.cells.iter().map(|cell| cell.dst_page_id))
                    .max()
                    .map_or(0, |page| page.saturating_add(1));
                let have = state.kv.as_ref().map_or(0, |kv| kv.num_pages);
                if need > have {
                    state.hold_kv_pages(need)?;
                }
            }

            let (Some(model), Some(_kv)) = (state.model.as_ref(), state.kv.as_ref()) else {
                eprintln!(
                    "[driver-cuda] copy_kv: no model or no KV cache is loaded; \
                     a page move needs both"
                );
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            let src_pages = desc.src_page_ids.as_slice();
            let dst_pages = desc.dst_page_ids.as_slice();
            if src_pages.len() != dst_pages.len() {
                eprintln!(
                    "[driver-cuda] copy_kv: {} source pages against {} destination \
                     pages; the lists are positional and must be parallel",
                    src_pages.len(),
                    dst_pages.len()
                );
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            let cells = desc.cells.as_slice();

            if (host_src || host_dst) && !cells.is_empty() {
                // Cell moves run a device kernel over the paged cache; a host
                // domain has no cache to run it over.
                eprintln!(
                    "[driver-cuda] copy_kv: {} cell moves with a host domain on \
                     one side (src {}, dst {}); cell moves are device-only",
                    cells.len(),
                    desc.src_domain,
                    desc.dst_domain
                );
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            let (kv_heads, head_dim) = (
                i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
                i32::try_from(model.deployment.shape.head_dim_kernel).unwrap_or(0),
            );
            let page_size: i32 = crate::boot::KV_PAGE_SIZE;
            let _layers_n = model.deployment.layers as usize;

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
                // Reuse only when geometry and capacity still cover this move;
                // else a moved stride writes host pages of the wrong width.
                let reusable = matches!(&state.swap, Some(sp)
                if sp.plan.num_pages() >= plan.num_pages()
                    && sp.plan.geometry() == plan.geometry());
                if !reusable {
                    let mut ops = LiveStagingOps;
                    let mut regions = Vec::with_capacity(plan.buffers().len());
                    for b in plan.buffers() {
                        let Some(p) = ops.malloc_host(usize::try_from(b.nbytes).unwrap_or(0))
                        else {
                            for &r in &regions {
                                ops.free_host(r);
                            }
                            return Err(PIE_STATUS_EXHAUSTED);
                        };
                        regions.push(p);
                    }
                    // The two stream roles the plan asks for, created once: a
                    // restore is on the critical path, so it must not queue
                    // behind pending evictions — the second stream avoids that.
                    let st = plan.streams();
                    let mk = |want: bool| -> Option<crate::device::OwnedStream> {
                        want.then(|| crate::device::OwnedStream::new(0).ok())
                            .flatten()
                    };
                    let (evict, restore) = (mk(st.evict), mk(st.restore));
                    if let Some(old) = state.swap.take() {
                        old.free();
                    }
                    state.swap = Some(SwapPool {
                        regions,
                        plan,
                        evict,
                        restore,
                    });
                }
            }

            // The stream the device legs and cell moves ride; host legs take
            // the pool's own evict/restore streams instead.
            let stream = match crate::device::OwnedStream::new(0) {
                Ok(s) => s,
                Err(_) => return Err(PIE_STATUS_DRIVER_ERROR),
            };
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            let kv_ref = state.kv.as_ref().expect("checked");
            for (s_id, d_id) in src_pages.iter().zip(dst_pages) {
                if (!host_src && *s_id >= kv_ref.num_pages)
                    || (!host_dst && *d_id >= kv_ref.num_pages)
                {
                    eprintln!(
                        "[driver-cuda] copy_kv: page {s_id} -> {d_id} is outside \
                         this pool's {} pages (device side: src {host_src:?} is \
                         host, dst {host_dst:?} is host)",
                        kv_ref.num_pages
                    );
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
            }

            // `SwapPlan::build` walks layer × page × buffer and emits one
            // `CopyOp` per contiguous move, with offsets in each side's own
            // index space: the two pools differ in capacity, so a transposed
            // src/dst is not caught by a bounds check, and `Direction` names
            // which side is which.
            use crate::layout::swap_plan::{Direction, Pool, SwapPlan};
            let direction = match (host_src, host_dst) {
                (false, true) => Direction::DeviceToHost,
                (true, false) => Direction::HostToDevice,
                _ => Direction::DeviceToDevice,
            };
            // The device geometry a device-to-device move is in; a host leg
            // uses the pool's.
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
                eprintln!(
                    "[driver-cuda] copy_kv: SwapPlan::build declined {} pages in \
                     direction {direction:?} against this geometry",
                    src_pages.len()
                );
                return Err(PIE_STATUS_INVALID_ARGUMENT);
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
                // KV-shared trailing layers read through their source's pool.
                let resolve = |e: Pool| -> Option<*mut u8> {
                    match e {
                        Pool::Device { layer, buffer } => {
                            let (k, v) = kv_ref.owned(layer as usize)?;
                            // Buffers beyond the k/v pair are the quantized
                            // format's scale planes, not addressable yet.
                            match buffer {
                                0 => Some(k.cast::<u8>()),
                                1 => Some(v.cast::<u8>()),
                                _ => None,
                            }
                        }
                        Pool::Host { layer, buffer } => state.swap.as_ref()?.region(layer, buffer),
                    }
                };
                let (Some(dst), Some(src)) = (resolve(op.dst), resolve(op.src)) else {
                    // Both absent is a layer with no pages; one absent is an
                    // unaddressable buffer, and a partial move is worse than none.
                    if resolve(op.dst).is_some() != resolve(op.src).is_some() {
                        eprintln!(
                            "[driver-cuda] copy_kv: this cache carries a buffer the \
                         shell cannot address, so the move would be partial"
                        );
                        // Drain what is already queued: earlier buffers of this
                        // op enqueued onto `leg`, and a later `copy_kv` that
                        // frees the pool would `cudaFreeHost` regions those
                        // copies are still reading.
                        let _ = leg.synchronize();
                        return Err(PIE_STATUS_UNSUPPORTED);
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
                        src.add(usize::try_from(op.src_offset).unwrap_or(0))
                            .cast_const()
                            .cast(),
                        usize::try_from(op.bytes).unwrap_or(0),
                        kind,
                        leg.as_raw(),
                    )
                };
                if code != cudaError::cudaSuccess {
                    // Same reason as the partial-move return above.
                    let _ = leg.synchronize();
                    return Err(PIE_STATUS_DRIVER_ERROR);
                }
            }

            // Cell moves: the bridged beam-repair launcher, per layer. Disjoint
            // spans are the caller's contract.
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
                    _ => return Err(PIE_STATUS_EXHAUSTED),
                };
                // Only the layers that own pages: a shared layer's cells live in
                // its source's pool, so visiting it too would copy them twice.
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
                        // The return is `#[must_use]`: `Declined(NoCells)` is
                        // the `N <= 0` exit, which cannot happen under the
                        // `!cells.is_empty()` this loop runs under.
                        let moved = crate::fire::kv_paged::copy_kv_cells_bf16(
                            layer,
                            d_dp.as_ptr().cast(),
                            d_doff.as_ptr().cast(),
                            d_sp.as_ptr().cast(),
                            d_soff.as_ptr().cast(),
                            i32::try_from(cells.len()).unwrap_or(i32::MAX),
                            stream.as_ref().as_raw().cast(),
                        )
                        .map_err(|why| {
                            eprintln!("[driver-cuda] copy_kv: the cell move refused: {why}");
                            PIE_STATUS_DRIVER_ERROR
                        })?;
                        debug_assert!(
                            matches!(moved, crate::fire::kv_paged::CopyKvCells::Launched),
                            "copy_kv_cells_bf16 declined a non-empty cell list"
                        );
                    }
                }
            }
            // Synchronize both streams, not just `stream`: `leg` may be the
            // pool's own non-blocking evict/restore stream, which does not
            // order against `stream`, so leaving it unsynchronized would let a
            // fire read KV pages the H2D has not finished writing. Both, not
            // `||`: short-circuiting on the first failure leaves the other
            // stream loaded.
            let (a, b) = (stream.as_ref().synchronize(), leg.synchronize());
            if a.is_err() || b.is_err() {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::serve::settle_control(&state.broker, completion);
            Ok(())
        })
    }

    /// Move recurrent state: whole-slot d2d over the hybrid's GDN slabs. The
    /// token fields ride for the rs buffer pool (spec-decode machinery).
    ///
    /// # Errors
    ///
    /// No recurrent state allocated, or a range outside a slot.
    pub fn copy_state(
        &mut self,
        copy: &driver_api::StateCopyPlan,
        completion: driver_api::completion::CompletionTarget,
    ) -> Result<(), i32> {
        guard("copy_state", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            let desc = copy;
            let Some(gdn) = state.gdn.as_mut() else {
                // No recurrent family is loaded: state copies only mean
                // something once the rs cache exists.
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            let ranges = desc.slot_ranges.as_slice();
            let Ok(stream) = crate::device::OwnedStream::new(0) else {
                return Err(PIE_STATUS_DRIVER_ERROR);
            };
            let alloc = crate::device::Allocator::new();
            let need = ranges
                .iter()
                .map(|r| r.src_slot_id.max(r.dst_slot_id) + 1)
                .max()
                .unwrap_or(0);
            // The epoch rides in because growing is what bumps it; a caller
            // that forgot would let a capture replay into a freed slab.
            if let Err(code) = gdn.ensure_slots(need, &mut state.fire_arrays.epoch, &alloc, &stream)
            {
                return Err(code);
            }
            for range in ranges {
                // Whole slots. `copy_slot_d2d` carries the MTP pending-hidden
                // row too, where the fold's `copy_linear_state_slot_d2d`
                // deliberately does not: a state copy is a clone and wants
                // everything, a fold is a rollback and must not overwrite a
                // newer MTP value with an older one.
                let Ok(ops) = gdn.cache.copy_slot_d2d(
                    i32::try_from(range.src_slot_id).unwrap_or(-1),
                    i32::try_from(range.dst_slot_id).unwrap_or(-1),
                ) else {
                    eprintln!("[driver-cuda] copy_state: a range names a slot the cache lacks");
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                };
                if let Err(code) = gdn.apply(&ops, stream.as_ref()) {
                    return Err(code);
                }
            }
            if stream.as_ref().synchronize().is_err() {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::serve::settle_control(&state.broker, completion);
            Ok(())
        })
    }

    /// Resize the KV pool to `target_pages`, migrating the surviving pages;
    /// shrinks drop the tail.
    ///
    /// # Errors
    ///
    /// A target past the reserved address space, or memory the arena could
    /// not get back.
    pub fn resize_pool(
        &mut self,
        resize: &driver_api::PoolResizePlan,
        completion: driver_api::completion::CompletionTarget,
    ) -> Result<(), i32> {
        guard("resize_pool", Err(PIE_STATUS_DRIVER_ERROR), move || {
            let state = self;
            let desc = resize;
            let Ok(target) = u32::try_from(desc.target_pages) else {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            if target == 0 {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            state.hold_kv_pages(target)?;
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::serve::settle_control(&state.broker, completion);
            Ok(())
        })
    }

    /// Reallocate the KV pool to hold exactly `target` pages, carrying the
    /// surviving pages forward.
    ///
    /// Factored out of [`Self::resize_pool`] because it has a SECOND caller:
    /// [`Self::copy_kv`] grows through it when a plan names a destination
    /// above the pool. It is the whole of what a resize does apart from
    /// settling the completion, so `resize_pool` is now that call plus the
    /// settle.
    ///
    /// # Errors
    ///
    /// `INVALID_ARGUMENT` if no model is loaded or the geometry will not
    /// plan, `EXHAUSTED` if the new pool will not fit in VRAM (the old one is
    /// still installed and intact in that case), `DRIVER_ERROR` if the
    /// carry-forward copy or its synchronize fails.
    fn hold_kv_pages(&mut self, target: u32) -> Result<(), i32> {
        {
            let state = self;
            let Some(model) = state.model.as_ref() else {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            let kv_heads = i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0);
            let page_size = crate::boot::KV_PAGE_SIZE.unsigned_abs() as usize;

            let stream = match crate::device::OwnedStream::new(0) {
                Ok(s) => s,
                Err(_) => return Err(PIE_STATUS_DRIVER_ERROR),
            };
            let alloc = crate::device::Allocator::new();
            // Borrowed, not taken: `old` is only read (its geometry and the
            // pages to copy forward), and several early returns sit between
            // here and `install_kv`. Taking it would drop the cache and leave
            // the shell `kv: None` on an otherwise-retryable `EXHAUSTED` from
            // growing past VRAM. `install_kv` drops the old cache at the end,
            // once the new one is in hand.
            let old = state.kv.as_ref();
            // Per-layer page bytes: an existing pool states its own stride (the
            // two-head-dim families disagree); before any pool exists the
            // deployment decides. The resize and fire paths must read the same
            // table, or a shared layer's pages get read at the wrong pitch.
            let dep = &model.deployment;
            let n_layers = dep.layers as usize;

            // A resize changes the page count and nothing else, so an existing
            // cache states its own geometry; before any exists, the deployment
            // answers.
            let layout = match old
                .as_ref()
                .map(|o| o.cache.layout().with_num_pages(target as i32))
            {
                Some(Ok(l)) => l,
                Some(Err(_)) => return Err(PIE_STATUS_INVALID_ARGUMENT),
                None => {
                    let per = crate::pools::kv_cache::PerLayer {
                        head_dim: dep.attention.iter().map(|a| a.head_dim as i32).collect(),
                        // gemma-4's tail attends through the last earlier layer
                        // of its own kind; every other family owns its pages.
                        kv_source_layer: dep.attention.iter().map(|a| a.kv_source as i32).collect(),
                        // The layer's own count, as `fire::launch` builds it:
                        // a two-kind tower may disagree here too, and a table
                        // the two paths filled differently is a shared layer's
                        // pages read at two pitches.
                        num_kv_heads: dep.attention.iter().map(|a| a.kv_heads as i32).collect(),
                    };
                    if per.check_sharing().is_err() {
                        return Err(PIE_STATUS_INVALID_ARGUMENT);
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
                        Err(_) => return Err(PIE_STATUS_INVALID_ARGUMENT),
                    }
                }
            };

            let mut ops = crate::pools::kv_cache_live::LiveKvCacheOps::new(stream.as_ref(), &alloc);
            let Ok(cache) = crate::pools::kv_cache_live::KvCache::materialize(layout, &mut ops)
            else {
                return Err(PIE_STATUS_EXHAUSTED);
            };
            let mut held = ops.into_held();
            for b in &mut held {
                if b.memset(0, stream.as_ref()).is_err() {
                    return Err(PIE_STATUS_DRIVER_ERROR);
                }
            }
            let fresh = KvState {
                cache,
                _held: held,
                num_pages: target,
            };

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
                            return Err(PIE_STATUS_DRIVER_ERROR);
                        }
                    }
                }
            }
            if stream.as_ref().synchronize().is_err() {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
            // A resize moves the KV pages, and a captured graph baked their old
            // addresses into every attention launch; `install_kv` tells
            // `Recordings` to recapture instead of replaying into memory the
            // pool no longer owns.
            crate::serve::state::install_kv(&mut state.kv, &mut state.fire_arrays.epoch, fresh);
            Ok(())
        }
    }
}
