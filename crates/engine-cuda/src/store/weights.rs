//! The weights the pool holds: routed banks streamed an expert at a time,
//! tables streamed by their rows, and the weight residency the runtime states.

#[allow(clippy::wildcard_imports)]
use super::*;

impl Pools {
    /// The params of the routed banks streamed an expert at a time.
    #[must_use]
    pub fn expert_params(&self) -> Vec<u32> {
        self.chunks
            .layout()
            .experts
            .iter()
            .flat_map(|shape| shape.banks.iter().flatten().map(|plane| plane.param))
            .collect()
    }

    /// Opens the cache of each expert shape: `hosts` gives each bank's
    /// mapped host copy by its param. No expert is cached yet; each is read
    /// from the host copy.
    ///
    /// `groups` names each packed bank by its codes' param with the params
    /// of its planes in the order its select's table lists them (the codes
    /// first; `None` a plane the bank lacks): one table for the bank, an
    /// expert's entry a plane. A plane in no group is a dense bank, a table
    /// of its own.
    pub fn seat_experts(
        &mut self,
        hosts: &[(u32, u64)],
        groups: &[(u32, Vec<Option<u32>>)],
    ) -> Result<()> {
        self.expert_caches.clear();
        for shape in &self.chunks.layout().experts {
            let experts = shape.experts as usize;
            let mut planes: Vec<ExpertBank> = Vec::new();
            for (layer, bank) in shape.banks.iter().enumerate() {
                let first = planes.len();
                // A companion plane is read through its bank's table alone.
                let companion = |param: u32| {
                    groups.iter().any(|(codes, members)| {
                        *codes != param && members.iter().any(|member| *member == Some(param))
                    })
                };
                let mut counted = false;
                for plane in bank {
                    let Some(&(_, host)) = hosts.iter().find(|(param, _)| *param == plane.param)
                    else {
                        return Err(Fault::program(
                            "store::experts",
                            format!(
                                "param {} streams by its experts and has no host copy",
                                plane.param
                            ),
                        ));
                    };
                    let head = !companion(plane.param);
                    // The planes its table lists, as places among the
                    // layer's (resolved below, once all are known).
                    let members = match groups.iter().find(|(codes, _)| *codes == plane.param) {
                        Some((_, members)) => members.clone(),
                        None => vec![Some(plane.param)],
                    };
                    planes.push(ExpertBank {
                        plane: *plane,
                        layer: layer as u32,
                        host,
                        head,
                        members: members
                            .iter()
                            .map(|member| member.map(|param| param as usize))
                            .collect(),
                        table: crate::device::Buffer::zeroed(experts * members.len().max(1) * 8)?,
                        // A layer's reads are counted at its first bank.
                        hits: if head && !std::mem::replace(&mut counted, true) {
                            Some(crate::device::Buffer::zeroed(experts * 4)?)
                        } else {
                            None
                        },
                        whole_rows: false,
                    });
                }
                // Each member's param to its place among the cache's planes.
                let places: Vec<(usize, usize)> = planes[first..]
                    .iter()
                    .enumerate()
                    .map(|(at, held)| (held.plane.param as usize, first + at))
                    .collect();
                for held in &mut planes[first..] {
                    for member in &mut held.members {
                        if let Some(param) = *member {
                            let place = places
                                .iter()
                                .find(|(held, _)| *held == param)
                                .map(|&(_, place)| place)
                                .ok_or_else(|| {
                                    Fault::program(
                                        "store::experts",
                                        format!(
                                            "param {param} is a plane of a bank whose experts \
                                             the recipe does not lay it with"
                                        ),
                                    )
                                })?;
                            *member = Some(place);
                        }
                    }
                }
            }
            self.expert_caches.push(ExpertCache {
                shape: shape.clone(),
                planes,
                slots: vec![None; shape.count() as usize],
            });
        }
        self.stage_expert_tables(core::ptr::null_mut())
    }

    /// Each bank's table as its experts now lie, written behind what
    /// `stream` holds (a null stream: at once, before any fire).
    pub(super) fn stage_expert_tables(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for at in 0..self.expert_caches.len() {
            for bank in 0..self.expert_caches[at].planes.len() {
                let cache = &self.expert_caches[at];
                let held = &cache.planes[bank];
                if !held.head {
                    continue;
                }
                let first = (held.layer * cache.shape.experts) as usize;
                // Arming's stand-in: the shape's first slot the pool holds.
                let stand_in = self
                    .stand_in
                    .then(|| cache.slots.iter().flatten().next().copied())
                    .flatten();
                let mut entries =
                    Vec::with_capacity(cache.shape.experts as usize * held.members.len() * 8);
                let base = self.chunks.base();
                let mut whole_rows = true;
                for expert in 0..cache.shape.experts as usize {
                    let slot = match cache.slots[first + expert].or(stand_in) {
                        Some(slot) => Some(self.chunks.weight_slot(
                            cache.shape.per_chunk,
                            cache.shape.slot_bytes,
                            slot,
                        )?),
                        None => None,
                    };
                    // An expert's entry a plane of the bank, in the order
                    // its select lists them; 0 a plane the bank lacks.
                    for member in &held.members {
                        let address = match member.map(|place| &cache.planes[place]) {
                            None => 0,
                            Some(plane) => match slot {
                                Some(slot) => {
                                    let row = plane.plane.row;
                                    whole_rows &= row > 0
                                        && (slot + plane.plane.offset - base).is_multiple_of(row);
                                    slot + plane.plane.offset
                                }
                                None => plane.host + expert as u64 * plane.plane.bytes,
                            },
                        };
                        entries.extend_from_slice(&address.to_le_bytes());
                    }
                }
                let held = &mut self.expert_caches[at].planes[bank];
                held.whole_rows = whole_rows;
                if stream.is_null() {
                    held.table.write(0, &entries)?;
                } else {
                    held.table.stage(stream, 0, &entries)?;
                }
            }
        }
        Ok(())
    }

    /// The expert table of each streamed bank, by its param: what its
    /// select reads its experts through. A bank whose experts all lie in
    /// pool slots the runtime holds for the load's life is located, any
    /// select's, where each entry is a whole number of its plane's rows past
    /// the pool's base or no kernel of this device maps the window (a TMA
    /// map indexes an entry by that row, `windows_mapped`); any other may
    /// name the host copy by the next fire, or lies where no map row indexes
    /// it, and is the plain-load selects' (a recorded body keeps the select
    /// it chose).
    #[must_use]
    pub fn expert_tables(&self) -> Vec<(u32, kernels_cuda::linear::moe::ExpertTable)> {
        let window = (self.chunks.base(), self.chunks.bytes());
        // A size-only walk before the weights are seated reads them as
        // arming will: located, where the fit claimed them whole (the load
        // preseats them before it arms). Nothing reads through the table.
        let walked_held = crate::sizing::active()
            && self.weight_pool_bytes() / self.chunks.layout().chunk_bytes.max(1)
                <= u64::from(self.fitted);
        self.expert_caches
            .iter()
            .flat_map(|cache| {
                cache
                    .planes
                    .iter()
                    .filter(|bank| bank.head)
                    .map(move |bank| {
                        let first = (bank.layer * cache.shape.experts) as usize;
                        let located = walked_held
                            || ((self.weights_held || self.preseated)
                                && (bank.whole_rows || !windows_mapped())
                                && cache.slots[first..first + cache.shape.experts as usize]
                                    .iter()
                                    .all(Option::is_some));
                        (
                            bank.plane.param,
                            kernels_cuda::linear::moe::ExpertTable {
                                table: bank.table.ptr(),
                                hits: bank.hits.as_ref().map_or(0, crate::device::Buffer::ptr),
                                located,
                                window: if located { window } else { (0, 0) },
                            },
                        )
                    })
            })
            .collect()
    }

    /// Opens the cache of each row-streamed table: `hosts` gives each
    /// table's mapped host copy by its param. Every block starts where that
    /// copy has it, read in place until the runtime names a slot for it.
    pub fn seat_rows(&mut self, hosts: &[(u32, u64)]) -> Result<()> {
        self.row_caches.clear();
        for shape in &self.chunks.layout().rows {
            let Some(&(_, host)) = hosts.iter().find(|(param, _)| *param == shape.param) else {
                return Err(Fault::program(
                    "store::rows",
                    format!(
                        "param {} streams by its rows and has no host copy",
                        shape.param
                    ),
                ));
            };
            let blocks = shape.blocks as usize;
            let cache = RowCache {
                shape: shape.clone(),
                host,
                table: crate::device::Buffer::zeroed(blocks * 8)?,
                hits: crate::device::Buffer::zeroed(blocks * 4)?,
                slots: vec![None; blocks],
            };
            self.row_caches.push(cache);
        }
        self.stage_row_tables(core::ptr::null_mut())
    }

    /// Each cache's table as its blocks now lie, written behind what
    /// `stream` holds (a null stream: at once, before any fire).
    pub(super) fn stage_row_tables(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for at in 0..self.row_caches.len() {
            let cache = &self.row_caches[at];
            let block = u64::from(cache.shape.block_rows) * cache.shape.row_bytes;
            let mut entries = Vec::with_capacity(cache.slots.len() * 8);
            let stand_in = self
                .stand_in
                .then(|| cache.slots.iter().flatten().next().copied())
                .flatten();
            for (atom, slot) in cache.slots.iter().enumerate() {
                let address = match slot.as_ref().or(stand_in.as_ref()) {
                    Some(slot) => self.chunks.row_slot(&cache.shape, *slot)?,
                    None => cache.host + atom as u64 * block,
                };
                entries.extend_from_slice(&address.to_le_bytes());
            }
            let cache = &mut self.row_caches[at];
            if stream.is_null() {
                cache.table.write(0, &entries)?;
            } else {
                cache.table.stage(stream, 0, &entries)?;
            }
        }
        Ok(())
    }

    /// The params of the tables streamed by their rows.
    #[must_use]
    pub fn row_params(&self) -> Vec<u32> {
        self.chunks
            .layout()
            .rows
            .iter()
            .map(|rows| rows.param)
            .collect()
    }

    /// The row table of each streamed table, by its param: what its gather
    /// reads its rows through, counting each block's reads.
    #[must_use]
    pub fn row_tables(&self) -> Vec<(u32, kernels_cuda::linear::moe::RowTable)> {
        self.row_caches
            .iter()
            .map(|cache| {
                (
                    cache.shape.param,
                    kernels_cuda::linear::moe::RowTable::located(
                        cache.table.ptr(),
                        cache.shape.block_rows,
                    )
                    .counting(cache.hits.ptr()),
                )
            })
            .collect()
    }

    /// See `WeightCache::set_residency`: the blocks newly named a slot are
    /// filled from the host copy and every table rewritten, on `stream`
    /// behind the fires it holds and before the next. A block no longer
    /// named is read from the host copy again. Returns whether a bank's
    /// table became located or stopped being so: the selects a recorded
    /// body chose for it are then not the ones a fire would choose.
    pub fn weight_residency(
        &mut self,
        stream: *mut core::ffi::c_void,
        residency: &engine::chunk::WeightResidency,
    ) -> Result<bool> {
        if self.weights_held && !residency.held {
            return Err(Fault::Unbound {
                what: "a weight residency that is not held after one that was: a held set \
                       is the load's for life"
                    .to_string(),
            });
        }
        let located = |pools: &Pools| -> Vec<bool> {
            pools
                .expert_tables()
                .iter()
                .map(|(_, table)| table.located)
                .collect()
        };
        if self.weights_filed {
            let named = |shape: engine::chunk::ShapeId| {
                residency
                    .resident
                    .iter()
                    .filter(|resident| resident.atom.shape == shape)
                    .count()
            };
            let whole = self
                .row_caches
                .iter()
                .map(|cache| (cache.shape.id, cache.slots.len()))
                .chain(
                    self.expert_caches
                        .iter()
                        .map(|cache| (cache.shape.id, cache.slots.len())),
                )
                .all(|(shape, atoms)| named(shape) >= atoms);
            if !residency.held || !whole {
                // The load was told its weights would be held, and the
                // runtime streams them after all (its pool does not seat
                // every atom): the copy a miss is read from is made now.
                self.pin_weight_hosts()?;
            }
        }
        let before = located(self);
        // An atom the load seated that the runtime names another slot for
        // is filled there again: said, since a runtime that adopted the
        // load's seating moves none.
        if self.preseated {
            let moved = residency
                .resident
                .iter()
                .filter(|resident| {
                    self.row_caches
                        .iter()
                        .map(|cache| (cache.shape.id, &cache.slots))
                        .chain(
                            self.expert_caches
                                .iter()
                                .map(|cache| (cache.shape.id, &cache.slots)),
                        )
                        .find(|(id, _)| *id == resident.atom.shape)
                        .and_then(|(_, slots)| slots.get(resident.atom.atom as usize))
                        .is_some_and(|held| *held != Some(resident.slot))
                })
                .count();
            if moved > 0 {
                tracing::info!(
                    "the first weight residency names {moved} of {} atoms a slot \
                     other than the one the load seated them in (`WeightShape::seated`); each \
                     is read from the serving file again",
                    residency.resident.len()
                );
            }
        }
        // The runtime's set stands from here on, the load's own no longer.
        self.preseated = false;
        self.weights_held = residency.held;
        for at in 0..self.row_caches.len() {
            let shape = self.row_caches[at].shape.clone();
            let mut named: Vec<Option<u32>> = vec![None; shape.blocks as usize];
            for resident in residency
                .resident
                .iter()
                .filter(|resident| resident.atom.shape == shape.id)
            {
                let Some(seat) = named.get_mut(resident.atom.atom as usize) else {
                    return Err(Fault::Ceiling {
                        what: "row blocks of a streamed table",
                        need: u64::from(resident.atom.atom) + 1,
                        have: u64::from(shape.blocks),
                    });
                };
                *seat = Some(resident.slot);
            }
            let block = u64::from(shape.block_rows) * shape.row_bytes;
            let table_bytes = shape.rows * shape.row_bytes;
            let host = self.row_caches[at].host;
            let mut fills = Vec::new();
            for (atom, slot) in named.iter().enumerate() {
                let Some(slot) = *slot else { continue };
                if self.row_caches[at].slots[atom] == Some(slot) {
                    continue;
                }
                let from = atom as u64 * block;
                let bytes = block.min(table_bytes.saturating_sub(from));
                fills.push((
                    self.chunks.row_slot(&shape, slot)?,
                    host + from,
                    usize::try_from(bytes).unwrap_or(0),
                ));
            }
            self.fill_weights(stream, &fills)?;
            self.row_caches[at].slots = named;
        }
        for at in 0..self.expert_caches.len() {
            let shape = self.expert_caches[at].shape.clone();
            let mut named: Vec<Option<u32>> = vec![None; shape.count() as usize];
            for resident in residency
                .resident
                .iter()
                .filter(|resident| resident.atom.shape == shape.id)
            {
                let Some(seat) = named.get_mut(resident.atom.atom as usize) else {
                    return Err(Fault::Ceiling {
                        what: "experts of a streamed shape",
                        need: u64::from(resident.atom.atom) + 1,
                        have: u64::from(shape.count()),
                    });
                };
                *seat = Some(resident.slot);
            }
            let mut fills = Vec::new();
            // The lane-order banks' slots filled from the serving file, as
            // it lies: laid once filled (and the fp8 banks' codes put in
            // their order).
            let mut laid = Vec::new();
            let mut fp8 = Vec::new();
            for (atom, slot) in named.iter().enumerate() {
                let Some(slot) = *slot else { continue };
                if self.expert_caches[at].slots[atom] == Some(slot) {
                    continue;
                }
                let (layer, expert) = (
                    atom as u32 / shape.experts,
                    u64::from(atom as u32 % shape.experts),
                );
                let to = self
                    .chunks
                    .weight_slot(shape.per_chunk, shape.slot_bytes, slot)?;
                let banks = || {
                    self.expert_caches[at]
                        .planes
                        .iter()
                        .filter(move |bank| bank.layer == layer)
                };
                for bank in banks() {
                    fills.push((
                        to + bank.plane.offset,
                        bank.host + expert * bank.plane.bytes,
                        usize::try_from(bank.plane.bytes).unwrap_or(0),
                    ));
                }
                if self.weights_filed {
                    for bank in &self.fp8 {
                        if let Some(codes) = banks().find(|b| b.plane.param == bank.codes) {
                            fp8.push((to + codes.plane.offset, bank.rows, bank.k));
                        }
                    }
                    for lane in &self.lanes {
                        let at_of = |param: u32| {
                            banks()
                                .find(|bank| bank.plane.param == param)
                                .map(|b| to + b.plane.offset)
                        };
                        if let (Some(codes), Some(exps)) = (at_of(lane.codes), at_of(lane.scales)) {
                            laid.push((codes, exps, lane.rows, lane.k));
                        }
                    }
                }
            }
            self.fill_weights(stream, &fills)?;
            self.lay_lanes(stream, &laid)?;
            self.lay_fp8(stream, &fp8)?;
            self.expert_caches[at].slots = named;
        }
        self.stage_row_tables(stream)?;
        self.stage_expert_tables(stream)?;
        Ok(before != located(self))
    }

    /// The chunks `residency` names weight slots in.
    #[must_use]
    pub fn weight_chunks(
        &self,
        residency: &engine::chunk::WeightResidency,
    ) -> std::collections::BTreeSet<u32> {
        let per_chunk = |shape: engine::chunk::ShapeId| {
            self.row_caches
                .iter()
                .map(|cache| (cache.shape.id, cache.shape.per_chunk))
                .chain(
                    self.expert_caches
                        .iter()
                        .map(|cache| (cache.shape.id, cache.shape.per_chunk)),
                )
                .find(|(id, _)| *id == shape)
                .map(|(_, per_chunk)| per_chunk.max(1))
        };
        residency
            .resident
            .iter()
            .filter_map(|resident| Some(resident.slot / per_chunk(resident.atom.shape)?))
            .collect()
    }

    /// Whether a page this pool seated for a lane lies in one of `chunks`.
    #[must_use]
    pub fn seated_in(&self, chunks: &std::collections::BTreeSet<u32>) -> bool {
        self.chunks.seated_in(chunks)
    }

    /// See [`chunks::Chunks::keep_clear`].
    pub fn keep_clear(&mut self, chunks: std::collections::BTreeSet<u32>) {
        self.chunks.keep_clear(chunks);
    }

    /// Fills weight slots from the host copies, `(slot, host, bytes)` each.
    pub(super) fn fill_weights(
        &mut self,
        stream: *mut core::ffi::c_void,
        fills: &[(u64, u64, usize)],
    ) -> Result<()> {
        if self.hosts_dropped && !fills.is_empty() {
            return Err(Fault::Unbound {
                what: "a weight atom named a new slot after the held set's host copy was \
                       dropped: a held atom stays where it was named"
                    .to_string(),
            });
        }
        if self.weights_filed {
            // Pageable memory: a copy a span, staged by the runtime. A span
            // of the serving file's mapping is let go once copied; one of a
            // held load's pageable copy (its laid planes, beside planes left
            // in the file) is no file's pages, and letting it go would zero it.
            for &(slot, file, bytes) in fills {
                crate::device::copy_any(stream, slot, file, bytes)?;
                if !crate::device::alloc::in_pageable_copy(file) {
                    self.file_spans.push((file, bytes));
                }
            }
            return Ok(());
        }
        crate::device::copy_batch(stream, false, fills)
    }

    /// The caches' host copies are the serving file's mapping: a slot is
    /// filled from it, and no atom is read where it lies. A residency that
    /// does not hold every atom makes the copy a miss is read from first
    /// ([`Pools::pin_weight_hosts`]).
    pub fn file_weights(&mut self) {
        self.weights_filed = true;
    }

    /// The MXFP4 banks the load lays in lane order
    /// (`store::layout::lane_ordered`): their codes and exponents' params,
    /// rows and K.
    pub fn lane_banks(&mut self, lanes: Vec<LaneBank>) {
        self.lanes = lanes;
    }

    /// The fp8 banks the load puts in the FP8 matrix core's order
    /// (`store::layout::fp8_ordered`): their codes' params, rows and K.
    pub fn fp8_banks(&mut self, banks: Vec<Fp8Bank>) {
        self.fp8 = banks;
    }

    /// Each `(codes, rows, k)` span of e4m3 codes put in the FP8 matrix
    /// core's order and made canonical where it lies, host or device
    /// (`weights::hip::order_fp8_span`); a span holding a NaN code refuses
    /// the load.
    pub(super) fn lay_fp8(
        &self,
        stream: *mut core::ffi::c_void,
        spans: &[(u64, u64, u64)],
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            crate::weights::order_fp8_spans(stream, spans)
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (stream, spans);
            Ok(())
        }
    }

    /// Each `(codes, exponents)` span of `rows` rows of K `k` laid in lane
    /// order where it lies, host or device, through the device.
    pub(super) fn lay_lanes(
        &self,
        stream: *mut core::ffi::c_void,
        spans: &[(u64, u64, u64, u64)],
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            let Some(widest) = spans.iter().map(|&(_, _, rows, k)| rows * k).max() else {
                return Ok(());
            };
            let codes_span = (widest / 2).next_multiple_of(16);
            let span = codes_span + widest / 32;
            let staging =
                crate::device::Buffer::zeroed(usize::try_from(2 * span).unwrap_or(usize::MAX))?;
            // SAFETY: `stream` is the caller's live stream.
            let ctx = unsafe { kernels_cuda::Ctx::on(stream.cast()) };
            let (in_c, in_e) = (staging.ptr() + span, staging.ptr() + span + codes_span);
            let (out_c, out_e) = (staging.ptr(), staging.ptr() + codes_span);
            for &(codes, exps, rows, k) in spans {
                let (cb, eb) = (
                    usize::try_from(rows * k / 2).unwrap_or(usize::MAX),
                    usize::try_from(rows * k / 32).unwrap_or(usize::MAX),
                );
                crate::device::copy_any(stream, in_c, codes, cb)?;
                crate::device::copy_any(stream, in_e, exps, eb)?;
                kernels_cuda::linear::moe::mxfp4_lanes(
                    &ctx,
                    "linear.moe_select",
                    in_c,
                    out_c,
                    in_e,
                    out_e,
                    rows,
                    k,
                )?;
                crate::device::copy_any(stream, codes, out_c, cb)?;
                crate::device::copy_any(stream, exps, out_e, eb)?;
                crate::device::ctx::sync(stream)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (stream, spans);
            Ok(())
        }
    }

    /// Whether a span of the serving file's mapping was copied out since
    /// its pages were last let go.
    #[must_use]
    pub fn copied_from_file(&self) -> bool {
        !self.file_spans.is_empty()
    }

    /// Lets go of the pages of the serving file's mapping that were copied
    /// out of it (into pool slots, or the pinned copy): they are read again
    /// only to fill a slot, from the page cache. The copies have landed
    /// (the caller synchronized the stream that carried them).
    pub fn release_file_pages(&mut self) {
        const PAGE: u64 = 4096;
        for (at, bytes) in std::mem::take(&mut self.file_spans) {
            let first = at.next_multiple_of(PAGE);
            let end = (at + bytes as u64) / PAGE * PAGE;
            if end > first {
                // Whole pages inside a read-only file mapping the weight
                // store holds for the load's life.
                crate::device::alloc::drop_file_pages(first as usize, (end - first) as usize);
            }
        }
    }

    /// The bytes every weight the caches state takes in the pool: each
    /// shape's atoms in whole chunks of its own, as
    /// [`Pools::preseat_plan`] seats them.
    #[must_use]
    pub fn weight_pool_bytes(&self) -> u64 {
        let chunks: u64 = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.per_chunk, cache.slots.len()))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.per_chunk, cache.slots.len())),
            )
            .map(|(per_chunk, atoms)| (atoms as u64).div_ceil(u64::from(per_chunk.max(1))))
            .sum();
        chunks.saturating_mul(self.chunks.layout().chunk_bytes)
    }

    /// The residency a load seats its weights by before the runtime names
    /// any slot: every atom of every shape, each shape's in chunks of its
    /// own from the pool's first on. `None` where the pool cannot hold them
    /// all with a chunk to spare: the load then keeps a copy its fires read
    /// in place ([`Pools::pin_weight_hosts`]), and the runtime decides what
    /// a pool that small may serve.
    #[must_use]
    pub fn preseat_plan(&self) -> Option<engine::chunk::WeightResidency> {
        let shapes = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.id, cache.shape.per_chunk, cache.slots.len()))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.id, cache.shape.per_chunk, cache.slots.len())),
            );
        let mut first = 1u32;
        let mut resident = Vec::new();
        for (shape, per_chunk, atoms) in shapes {
            let per_chunk = per_chunk.max(1);
            resident.extend((0..atoms as u32).map(|atom| engine::chunk::ResidentWeight {
                atom: engine::chunk::WeightAtom { shape, atom },
                slot: first * per_chunk + atom,
            }));
            first = first.checked_add((atoms as u32).div_ceil(per_chunk))?;
        }
        // No more chunks than the fit left the pool (stated bytes are all
        // the cache rows'): the range is limited to them,
        // under whatever was seated past it.
        let most = self.fitted_chunks();
        (first < most).then_some(engine::chunk::WeightResidency {
            staging: Vec::new(),
            resident,
            held: true,
        })
    }

    /// Seats the load's weights by `plan` ([`Pools::preseat_plan`]): the
    /// slots' segments backed, each filled from the serving file, the
    /// tables located. The load arms over them; the runtime's first
    /// residency then names the slots it chose, and an atom it puts
    /// elsewhere is filled there again.
    /// A measuring load's second fit on a flat range (one allocation, made
    /// at the first fit: the least range the bodies armed over): where the
    /// card could back more chunks than that allocation holds, everything
    /// seated in it is let go (the weights preseated in it, the segments
    /// arming's fires backed), so the refit may reserve the range anew at
    /// what the card backs ([`chunks::Chunks::reshape`]); the caller seats
    /// the weights again. Whether it did: a range mapped on demand, one a
    /// body names, or one already as large as asked, keeps all it holds.
    /// Whether the load seated its weights in the pool ahead of the
    /// runtime's first residency ([`Pools::preseat_weights`]).
    #[must_use]
    pub const fn preseated(&self) -> bool {
        self.preseated
    }

    pub fn free_for_refit(&mut self) -> Result<bool> {
        if elastic::mapped_on_demand() || self.pinned || self.asked_chunks <= self.chunks.total() {
            return Ok(false);
        }
        for cache in &mut self.row_caches {
            cache.slots.fill(None);
        }
        for cache in &mut self.expert_caches {
            cache.slots.fill(None);
        }
        self.preseated = false;
        self.release_segments(0)?;
        Ok(true)
    }

    pub fn preseat_weights(
        &mut self,
        stream: *mut core::ffi::c_void,
        plan: &engine::chunk::WeightResidency,
    ) -> Result<()> {
        let segment = self.chunks.layout().segment_chunks.max(1);
        let last = self.weight_chunks(plan).into_iter().max().unwrap_or(0);
        self.commit_segments((last + 1).div_ceil(segment))?;
        self.weight_residency(stream, plan)?;
        self.weights_held = false;
        self.preseated = true;
        Ok(())
    }

    /// Gives the caches a copy a fire can read a miss from, out of the
    /// serving file's mapping they were left in (or the pageable copy a load
    /// told its weights are held made): the mapping itself where
    /// the device faults mapped pages in, else a pinned copy of every
    /// streamed table and bank, refused where the machine has not the
    /// memory for it.
    pub fn pin_weight_hosts(&mut self) -> Result<()> {
        self.weights_filed = false;
        if crate::experts::pageable_access() {
            return Ok(());
        }
        let align = |bytes: u64| bytes.next_multiple_of(crate::weights::ALIGN);
        // Each copy's place in the pinned span, its source and its bytes.
        let mut spans: Vec<(u64, u64, u64)> = Vec::new();
        let mut total = 0u64;
        let mut lay = |file: u64, bytes: u64, total: &mut u64| {
            spans.push((*total, file, bytes));
            *total += align(bytes);
        };
        for cache in &self.row_caches {
            lay(
                cache.host,
                cache.shape.rows * cache.shape.row_bytes,
                &mut total,
            );
        }
        for cache in &self.expert_caches {
            for bank in &cache.planes {
                lay(
                    bank.host,
                    u64::from(cache.shape.experts) * bank.plane.bytes,
                    &mut total,
                );
            }
        }
        let have = crate::weights::available_memory();
        if total > have {
            return Err(Fault::Residency(format!(
                "the runtime streams this load's weights, which needs a pinned host copy of \
                 them, {:.1} GiB, and this machine has {:.1} GiB of memory left for it. Give \
                 the pool room to hold them all (a larger `[engine.memory] budget_device`, or \
                 a device to itself), or use a device with pageable memory access (HMM).",
                total as f64 / f64::from(1u32 << 30),
                have as f64 / f64::from(1u32 << 30),
            )));
        }
        let pinned =
            crate::device::Pinned::mapped_uninit(usize::try_from(total).unwrap_or(usize::MAX))?;
        for &(at, file, bytes) in &spans {
            let len = usize::try_from(bytes).unwrap_or(usize::MAX);
            // SAFETY: `file` and `bytes` are a plane of the serving file's
            // mapping the weight store bound and holds for the load's life.
            let plane = unsafe { core::slice::from_raw_parts(file as *const u8, len) };
            if !pinned.write(usize::try_from(at).unwrap_or(usize::MAX), plane) {
                return Err(Fault::program(
                    "store::weights",
                    "a streamed weight does not fit the pinned copy sized for it",
                ));
            }
            self.file_spans.push((file, len));
        }
        let base = pinned.device();
        let mut copies = spans.iter().map(|&(at, _, _)| base + at);
        for cache in &mut self.row_caches {
            cache.host = copies.next().unwrap_or(cache.host);
        }
        for cache in &mut self.expert_caches {
            for bank in &mut cache.planes {
                bank.host = copies.next().unwrap_or(bank.host);
            }
        }
        // The file holds an fp8 bank's codes as it lies: its pinned copy's
        // are put in the FP8 matrix core's order, an expert at a time.
        let mut fp8 = Vec::new();
        for cache in &self.expert_caches {
            for bank in &self.fp8 {
                for codes in cache.planes.iter().filter(|b| b.plane.param == bank.codes) {
                    for expert in 0..u64::from(cache.shape.experts) {
                        fp8.push((codes.host + expert * codes.plane.bytes, bank.rows, bank.k));
                    }
                }
            }
        }
        self.lay_fp8(core::ptr::null_mut(), &fp8)?;
        // The file holds a lane-order bank as it lies: its pinned copy, what
        // misses read and slots are filled from, is laid in lane order.
        let mut laid = Vec::new();
        for cache in &self.expert_caches {
            for lane in &self.lanes {
                for codes in cache
                    .planes
                    .iter()
                    .filter(|bank| bank.plane.param == lane.codes)
                {
                    let Some(exps) = cache
                        .planes
                        .iter()
                        .find(|bank| bank.plane.param == lane.scales && bank.layer == codes.layer)
                    else {
                        continue;
                    };
                    for expert in 0..u64::from(cache.shape.experts) {
                        laid.push((
                            codes.host + expert * codes.plane.bytes,
                            exps.host + expert * exps.plane.bytes,
                            lane.rows,
                            lane.k,
                        ));
                    }
                }
            }
        }
        self.lay_lanes(core::ptr::null_mut(), &laid)?;
        tracing::info!(
            "the runtime streams weights this load was told it would hold; {} \
             MiB of them pinned on the host from the serving file, for misses",
            total >> 20
        );
        self.miss_copy = Some(pinned);
        // The tables name the copy from here on: at once, where no fire has
        // read them yet (a load that seats none of its weights arms over
        // the copy); a residency restages them behind its stream after.
        self.stage_row_tables(core::ptr::null_mut())?;
        self.stage_expert_tables(core::ptr::null_mut())?;
        Ok(())
    }

    /// The params whose host copies no fire reads and no fill needs again:
    /// every streamed table and bank, where the runtime holds the set for
    /// the load's life and it names every atom. Empty otherwise.
    #[must_use]
    pub fn held_weights(&self) -> Vec<usize> {
        let whole = self.weights_held
            && self
                .row_caches
                .iter()
                .map(|cache| &cache.slots)
                .chain(self.expert_caches.iter().map(|cache| &cache.slots))
                .all(|slots| slots.iter().all(Option::is_some));
        if !whole {
            return Vec::new();
        }
        self.row_params()
            .into_iter()
            .chain(self.expert_params())
            .map(|param| param as usize)
            .collect()
    }

    /// The host copies are gone: nothing is filled from them again.
    pub fn drop_weight_hosts(&mut self) {
        self.hosts_dropped = true;
    }

    /// Arming over a streamed set reads every atom on the host copy through
    /// a slot of its shape the pool holds (`on`), and the tables are staged
    /// so at once; off, as they lie. A synthetic fire's rows are never read
    /// back, so what the slot holds does not matter, and nothing it records
    /// names an entry: a body reads the table as it stands when it replays.
    /// The tables stay unlocated either way, so a body takes the select
    /// that reads the host copy.
    pub fn stand_in(&mut self, on: bool) -> Result<()> {
        self.stand_in = on;
        self.stage_row_tables(core::ptr::null_mut())?;
        self.stage_expert_tables(core::ptr::null_mut())
    }

    /// Whether every atom of every weight shape lies in a pool slot (true
    /// where the load states none).
    #[must_use]
    pub fn weights_seated_whole(&self) -> bool {
        self.row_caches
            .iter()
            .all(|cache| cache.slots.iter().all(Option::is_some))
            && self
                .expert_caches
                .iter()
                .all(|cache| cache.slots.iter().all(Option::is_some))
    }

    /// Whether every streamed bank's table is located: each expert in a
    /// slot of a set that is held.
    #[must_use]
    pub fn weights_located(&self) -> bool {
        self.expert_tables().iter().all(|(_, table)| table.located)
    }

    /// The row-streamed tables' blocks named a slot, and all of them.
    #[must_use]
    pub fn row_blocks(&self) -> (u64, u64) {
        self.row_caches.iter().fold((0, 0), |(named, all), cache| {
            (
                named + cache.slots.iter().flatten().count() as u64,
                all + cache.slots.len() as u64,
            )
        })
    }

    /// Zeroes the use record on `stream`, behind the fires that counted:
    /// what the shell's own synthetic fires read (arming the lattice) is
    /// no use the runtime orders its weight cache by. Waits for the
    /// stream, so no snapshot taken before lands after the record restarts.
    pub fn forget_weight_uses(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for (at, bytes) in self.counters() {
            crate::device::zero_span_on(stream, at, bytes)?;
        }
        crate::device::ctx::sync(stream)?;
        if let Some(record) = &mut self.use_record {
            record.host.zero();
            record.base.fill(0);
        }
        Ok(())
    }

    /// Every read counter of the caches as `(device address, bytes)`: a
    /// row cache's a block, an expert bank's an expert, in the order
    /// [`Pools::weight_uses`] reads them.
    pub(super) fn counters(&self) -> Vec<(u64, usize)> {
        self.row_caches
            .iter()
            .map(|cache| (cache.hits.ptr(), cache.slots.len() * 4))
            .chain(self.expert_caches.iter().flat_map(|cache| {
                let experts = cache.shape.experts as usize;
                cache.planes.iter().filter_map(move |bank| {
                    bank.hits.as_ref().map(|hits| (hits.ptr(), experts * 4))
                })
            }))
            .collect()
    }

    /// See `WeightCache::uses`: each block's reads since the last call.
    /// The counters run on (never zeroed but by
    /// [`Pools::forget_weight_uses`]); each call enqueues a snapshot of them
    /// into mapped host memory on `stream` behind the fires that counted,
    /// and answers with what the previous call's snapshot holds past the
    /// one before it. No call waits for the stream, so the record lags a
    /// call behind the fires, which the runtime's windows of several calls
    /// absorb.
    pub fn weight_uses(
        &mut self,
        stream: *mut core::ffi::c_void,
    ) -> Result<engine::chunk::WeightUses> {
        let counters = self.counters();
        if self.use_record.is_none() {
            let bytes: usize = counters.iter().map(|&(_, bytes)| bytes).sum();
            let host = crate::device::Pinned::mapped(bytes.max(4))?;
            host.zero();
            self.use_record = Some(UseRecord {
                host,
                base: vec![0; bytes / 4],
            });
        }
        let record = self
            .use_record
            .as_mut()
            .expect("the use record was just made");
        let mut reads = Vec::with_capacity(record.base.len());
        for (at, base) in record.base.iter_mut().enumerate() {
            // SAFETY: inside the record's mapped host span; a snapshot copy
            // may land meanwhile, and an aligned u32 is read whole.
            let now = unsafe { record.host.host().cast::<u32>().add(at).read_volatile() };
            reads.push(now.wrapping_sub(*base));
            *base = now;
        }
        let mut spans = Vec::with_capacity(counters.len());
        let mut into = record.host.host() as u64;
        for &(at, bytes) in &counters {
            spans.push((into, at, bytes));
            into += bytes as u64;
        }
        crate::device::copy_batch(stream, true, &spans)?;
        let mut uses = Vec::new();
        let mut reads = reads.into_iter();
        for cache in &self.row_caches {
            for (atom, reads) in reads.by_ref().take(cache.slots.len()).enumerate() {
                if reads > 0 {
                    uses.push((
                        engine::chunk::WeightAtom {
                            shape: cache.shape.id,
                            atom: atom as u32,
                        },
                        reads,
                    ));
                }
            }
        }
        for cache in &self.expert_caches {
            let experts = cache.shape.experts as usize;
            for bank in cache.planes.iter().filter(|bank| bank.hits.is_some()) {
                for (expert, reads) in reads.by_ref().take(experts).enumerate() {
                    if reads > 0 {
                        uses.push((
                            engine::chunk::WeightAtom {
                                shape: cache.shape.id,
                                atom: bank.layer * cache.shape.experts + expert as u32,
                            },
                            reads,
                        ));
                    }
                }
            }
        }
        let held = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.id, &cache.slots))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.id, &cache.slots)),
            )
            .map(|(shape, slots)| {
                let resident = slots.iter().flatten().count() as u32;
                engine::chunk::WeightHeld {
                    shape,
                    resident,
                    host: slots.len() as u32 - resident,
                }
            })
            .collect();
        Ok(engine::chunk::WeightUses { uses, held })
    }
}

/// A row-streamed table's cache: where each block of its rows lies (a pool
/// slot the runtime named, or the mapped host copy), the device table its
/// gather reads them through, and each block's reads since last asked.
#[derive(Debug)]
pub(super) struct RowCache {
    pub(super) shape: layout::RowsShape,
    /// The table's mapped host copy.
    pub(super) host: u64,
    /// A u64 address a block.
    pub(super) table: crate::device::Buffer,
    /// A u32 counter a block.
    pub(super) hits: crate::device::Buffer,
    /// The slot ref each block is cached in.
    pub(super) slots: Vec<Option<u32>>,
}

/// The caches' read counters as last snapshotted: mapped host memory a
/// snapshot copy lands in, every counter in [`Pools::counters`] order, and
/// each counter's value as the last answer read it.
#[derive(Debug)]
pub(super) struct UseRecord {
    pub(super) host: crate::device::Pinned,
    pub(super) base: Vec<u32>,
}

/// One bank of an expert shape's cache: its plane in an expert's slot, the
/// layer it is of, its mapped host copy, the device table its select reads
/// its experts through, and (a layer's first bank) each expert's reads.
#[derive(Debug)]
pub(super) struct ExpertBank {
    pub(super) plane: layout::ExpertPlane,
    pub(super) layer: u32,
    pub(super) host: u64,
    /// Whether a select reads this plane as its bank (a dense bank, or a
    /// packed bank's codes): it has the table. A packed bank's other planes
    /// are read through that table.
    pub(super) head: bool,
    /// The planes an expert's table entry lists, as places among the
    /// cache's planes; `None` one the bank lacks.
    pub(super) members: Vec<Option<usize>>,
    /// A u64 address an expert.
    pub(super) table: crate::device::Buffer,
    /// A u32 counter an expert.
    pub(super) hits: Option<crate::device::Buffer>,
    /// Whether, as the table was last written, every entry lay in the pool a
    /// whole number of its plane's rows past the pool's base: what a TMA map
    /// of the window indexes it by.
    pub(super) whole_rows: bool,
}

/// An MXFP4 bank ROCm lays in the matrix core's lane order: its codes' and
/// exponents' params, an expert's rows and K.
#[derive(Debug, Clone, Copy)]
pub struct LaneBank {
    pub codes: u32,
    pub scales: u32,
    pub rows: u64,
    pub k: u64,
}

/// An fp8 bank ROCm puts in the FP8 matrix core's order
/// (`store::layout::fp8_ordered`): its codes' param, an expert's rows and
/// K.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Bank {
    pub codes: u32,
    pub rows: u64,
    pub k: u64,
}

/// An expert shape's cache: where each expert of each layer lies (a pool
/// slot the runtime named, or the mapped host copy).
#[derive(Debug)]
pub(super) struct ExpertCache {
    pub(super) shape: layout::ExpertShape,
    pub(super) planes: Vec<ExpertBank>,
    /// The slot ref each atom is cached in.
    pub(super) slots: Vec<Option<u32>>,
}

/// One plane a page holds bytes in: `offset` into the page's slot of the
/// state pool.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlaneAt {
    pub(super) kind: Kind,
    pub(super) offset: u64,
}

pub(super) const POOL_STATE: Dtype = Dtype::Bf16;

/// A compressor's state for the kv row `row` its gather keys it by: two
/// planes (the projection, then its score) of `width` a token, `offset`
/// into the row's pages' slots.
#[derive(Debug)]
pub(super) struct PoolRow {
    pub(super) row: u32,
    pub(super) width: u64,
    /// Keyed by a windowed row's pages.
    pub(super) windowed: bool,
    /// The list those pages are slots of ([`Shape::Kv`]'s).
    pub(super) list: u8,
    pub(super) offset: u64,
}

impl PoolRow {
    /// The bytes one of its planes holds a page.
    pub(super) fn plane_bytes(&self, page_size: u32) -> u64 {
        u64::from(page_size) * self.width * u64::from(elem_size(POOL_STATE))
    }
}
