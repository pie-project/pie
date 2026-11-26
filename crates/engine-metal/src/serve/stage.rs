//! The shell's staging of a fire: its lanes' windows, seats and epilogues,
//! its inputs and feeds, and the walk that encodes it.

use super::*;

/// How [`Shell::walk`] encodes a forward.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Form {
    /// To run, sharing with the Neural Engine what this load splits.
    Fire,
    /// For its pipelines alone, split as [`Form::Fire`] splits it.
    DryShared,
    /// For its pipelines alone, the GPU taking all of it, as after the
    /// split is retired.
    DryAlone,
}

impl Shell {
    /// A windowed lane's window table for this fire (/// `store::windowed`): its ring's pages below the buffer's base, its
    /// window-buffer slots from it, and the fire's own rows past them: in
    /// their ring rows where none of them overwrites a row a query of the
    /// fire reads ([`crate::store::windowed::direct_allowed`]), else in pages
    /// of the range's own-row chunks (`Shell::own_top`, the cursor
    /// `Shell::own_used`), the one holding rows folded before the fire
    /// filled from its ring page first. The fold the fire makes is planned
    /// here and encoded after its forward.
    pub(super) fn windowed_pages(
        &mut self,
        k: usize,
        seated: &Seated<'_>,
        fold: engine::fire::WindowFold,
        port_fold: Option<u32>,
        have: u32,
        rows: u32,
    ) -> Result<Vec<u32>> {
        let refuse = |why: String| Fault::program("serve::windowed", why);
        let page = self.pools.window_paging(k).page_size.max(1);
        let window = self.pools.layout().window(k).map_or(0, |list| list.rows);
        let per_chunk = self.pools.window_slots_a_chunk(k)?;
        let ring_rows = crate::chunks::MetalRing::rows(window, page);
        let (ring_refs, shared): (Vec<u32>, bool) = match seated.ring {
            // A shared ring (`cow != run`) was copied run onto cow before the
            // fire (`Shell::copy_records`): the fire reads and writes cow.
            // Ring page j: piece j / per_chunk, the window slot j % per_chunk
            // of that chunk (a ring piece is one slot of one chunk, its ref
            // the chunk).
            Some(pieces) => (
                (0..ring_rows / page)
                    .map(|j| {
                        pieces
                            .get((j / per_chunk) as usize)
                            .map(|&[_, cow]| cow * per_chunk + j % per_chunk)
                            .ok_or_else(|| {
                                refuse(format!(
                                    "a ring of {} pages in {} piece(s) of {per_chunk} pages",
                                    ring_rows / page,
                                    pieces.len()
                                ))
                            })
                    })
                    .collect::<Result<_>>()?,
                pieces
                    .first()
                    .is_some_and(|&[_, cow]| self.ring_shared.contains(&cow)),
            ),
            // A seat-addressed lane: its seat's ring, its own.
            None if !seated.refs => (
                self.pools
                    .window_seat_ring(k, seated.lane.slot, ring_rows / page)?,
                false,
            ),
            None => {
                return Err(refuse(
                    "a windowed lane states no ring pieces in its record".to_string(),
                ));
            }
        };
        let ring = self.pools.window_pages(k, &ring_refs)?;
        let ring_at: Vec<u64> = ring_refs
            .iter()
            .map(|&slot| self.pools.window_slot_offset(k, slot))
            .collect::<Result<_>>()?;
        let buffer_refs = seated.windows.get(k).map_or(&[][..], Vec::as_slice);
        let buffer = self.pools.window_pages(k, buffer_refs)?;
        let buffer_at: Vec<u64> = buffer_refs
            .iter()
            .map(|&slot| self.pools.window_slot_offset(k, slot))
            .collect::<Result<_>>()?;
        let end = have + rows;
        let held = u64::from(fold.base) + buffer.len() as u64 * u64::from(page);
        // The fire's own rows past the buffer: their ring rows, or own pages.
        let direct = crate::store::windowed::direct_allowed(shared, rows, ring_rows, window);
        // Rows of the fire no buffer slot holds: below the buffer's base (the
        // rows it folds itself) or past its last slot.
        let outside = have < fold.base || u64::from(end) > held;
        let (own, own_at, own_page) = if direct || !outside {
            (Vec::new(), Vec::new(), 0)
        } else {
            // From the fire's first row's page: a buffer slot's page reads
            // the slot first ([`WindowedLane::page_of`]).
            let own_page = have / page;
            let count = end.div_ceil(page).saturating_sub(own_page);
            let (first, chunks) = self.own_top;
            // This fire's part of the region: one arm's share.
            let per_arm = chunks.saturating_sub(1).saturating_mul(per_chunk) / self.own_arms;
            let mut ids = Vec::with_capacity(count as usize);
            let mut at = Vec::with_capacity(count as usize);
            for _ in 0..count {
                if self.own_used >= per_arm {
                    return Err(refuse(format!(
                        "a fire's windowed rows take more own-row pages than an arm's share of \
                         the {chunks} chunk(s) the load reserved holds ({per_arm} pages)"
                    )));
                }
                let slot = self.own_arm * per_arm + self.own_used;
                self.own_used += 1;
                let (byte, id) =
                    self.pools
                        .window_slot_at(k, first + slot / per_chunk, slot % per_chunk)?;
                ids.push(id);
                at.push(byte);
            }
            (ids, at, own_page)
        };
        let lane = crate::store::windowed::WindowedLane {
            page,
            ring: &ring,
            base: fold.base,
            buffer: &buffer,
            own: &own,
            own_page,
        };
        let pages = lane.pages(end)?;
        // The same lane over indices into `bytes`, for the fills and the fold.
        let (r, b, o) = (
            ring_at.len() as u32,
            buffer_at.len() as u32,
            own_at.len() as u32,
        );
        let ring_ix: Vec<u32> = (0..r).collect();
        let buffer_ix: Vec<u32> = (r..r + b).collect();
        let own_ix: Vec<u32> = (r + b..r + b + o).collect();
        let bytes: Vec<u64> = ring_at
            .iter()
            .chain(&buffer_at)
            .chain(&own_at)
            .copied()
            .collect();
        let by_index = crate::store::windowed::WindowedLane {
            page,
            ring: &ring_ix,
            base: fold.base,
            buffer: &buffer_ix,
            own: &own_ix,
            own_page,
        };
        // An own-row page takes its ring page's rows whole: the fire writes
        // its own rows over them after.
        let mut fills: Vec<(u64, u64, u32, u32)> = by_index
            .own_fills(seated.folded)
            .into_iter()
            .map(|(ring, own)| (bytes[ring as usize], bytes[own as usize], 0, page))
            .collect();
        // A buffer slot holding rows folded before the fire (its first, from
        // `base` up to `folded`: rows a fire folded itself into the ring and
        // wrote to no slot) takes those rows from its ring page, and those
        // alone: its rows from `folded` on are unfolded, held in the slot only.
        for (i, &slot_at) in buffer_at.iter().enumerate() {
            let n = fold.base / page + u32::try_from(i).unwrap_or(u32::MAX);
            let first = u64::from(n) * u64::from(page);
            if first < u64::from(seated.folded) {
                let rows = (u64::from(seated.folded) - first).min(u64::from(page));
                fills.push((
                    ring_at[n as usize % ring_at.len().max(1)],
                    slot_at,
                    0,
                    u32::try_from(rows).unwrap_or(page),
                ));
            }
        }
        if !fills.is_empty() {
            self.window_fills.push((k, fills));
        }
        // A device-counted fold reads its count from the lane's `fold` port,
        // as the recurrent state's does, bounded by the rows the lane holds.
        let advance = match fold.fold {
            engine::fire::FoldLen::Host(rows) => rows,
            engine::fire::FoldLen::Device => port_fold.ok_or_else(|| {
                refuse(
                    "a lane states a device-counted window fold, and the program attached to \
                     it resolved no `fold` port"
                        .to_string(),
                )
            })?,
        }
        .min(end.saturating_sub(seated.folded));
        if u64::from(end) > held && u64::from(seated.folded) + u64::from(advance) < u64::from(end) {
            tracing::warn!(
                "windowed: lane in slot {} keeps rows past its buffer unfolded: have {have} rows \
                 {rows} folded {} advance {advance} base {} buffer {} pages of {page}",
                seated.lane.slot,
                seated.folded,
                fold.base,
                buffer.len()
            );
        }
        if advance > 0 {
            let copies: Vec<(u64, u64, u32, u32)> = by_index
                .fold_copies(seated.folded, seated.folded + advance)?
                .into_iter()
                .map(|(source, ring, first, rows)| {
                    (bytes[source as usize], bytes[ring as usize], first, rows)
                })
                .collect();
            if !copies.is_empty() {
                self.window_folds.push((k, copies));
            }
        }
        Ok(pages)
    }

    /// Where lane `lane`'s voxels of grid output `out` start: the plane's
    /// offset there, its width and dtype.
    pub(super) fn grid_seat(
        &self,
        prepared: &Prepared<'_>,
        lane: usize,
        out: usize,
    ) -> Result<(u64, u32, Dtype)> {
        let value = self.outputs[out].values[0];
        let (offset, width, dtype) = self.plane_row(prepared, value)?;
        let rows = prepared.slots.0[value.0 as usize].map_or(0, |rect| rect.rows);
        let first_clip: u32 = prepared
            .composition
            .lanes()
            .iter()
            .filter(|row| (row.source as usize) != lane)
            .filter(|row| row.voxel_offset < Self::first_voxel(prepared, lane))
            .map(|row| row.clips)
            .sum();
        let Some(chain) = self.grid_chains[out].as_ref() else {
            if first_clip == 0 {
                return Ok((offset, width, dtype));
            }
            return Err(Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "lane {lane}'s pixels start past the plane's first row, and this load \
                     could not walk the `conv.grid` chain from the port grid to the \
                     plane's — so where they start is not a thing this fire can state"
                ),
            });
        };
        let mut table = prepared.voxel_grid.clone();
        for rule in chain {
            table = rule.apply(&table).ok_or_else(|| Fault::Program {
                at: "serve::enqueue",
                why: format!("a clip's box does not map through {rule:?}"),
            })?;
        }
        let out_voxels: u64 = table
            .as_chunks::<4>()
            .0
            .iter()
            .map(|clip| {
                u64::try_from(clip[0]).unwrap_or(0)
                    * u64::try_from(clip[1]).unwrap_or(0)
                    * u64::try_from(clip[2]).unwrap_or(0)
            })
            .sum();
        // The carve sizes a plane past a downsampling rule at the port's
        // rows, its voxels at their head.
        if out_voxels > u64::from(rows) {
            return Err(Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "replaying this plane's `conv.grid` chain over the fire's own port \
                     grid lands {out_voxels} voxels and the carve gave the plane {} rows; \
                     the chain walked at load is not the one this fire runs",
                    rows
                ),
            });
        }
        if first_clip == 0 {
            return Ok((offset, width, dtype));
        }
        let at = table
            .as_chunks::<4>()
            .0
            .get(first_clip as usize)
            .map(|clip| clip[3])
            .ok_or_else(|| Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "lane {lane}'s first clip is {first_clip} and the output grid holds {}",
                    table.len() / 4
                ),
            })?;
        let element = poem_compiler::arena::elem_bytes(dtype).unwrap_or(0);
        Ok((
            offset + u64::try_from(at).unwrap_or(0) * u64::from(width) * element,
            width,
            dtype,
        ))
    }

    /// The voxels lane `lane` holds.
    pub(super) fn lane_voxels(prepared: &Prepared<'_>, lane: usize) -> u32 {
        prepared
            .composition
            .lanes()
            .iter()
            .find(|row| row.source as usize == lane)
            .map_or(0, |row| row.voxels)
    }

    pub(super) fn first_voxel(prepared: &Prepared<'_>, lane: usize) -> u32 {
        prepared
            .composition
            .lanes()
            .iter()
            .find(|row| row.source as usize == lane)
            .map_or(0, |row| row.voxel_offset)
    }

    pub(super) fn encode_epilogues(
        &mut self,
        frame: &mut Frame,
        prepared: &Prepared<'_>,
    ) -> Result<Vec<engine::InstanceId>> {
        if !prepared.attachments.iter().any(|a| !a.adapter_only) {
            return Ok(Vec::new());
        }

        let count = prepared.lanes.len();
        let mut first_row = vec![0u32; count];
        let mut lane_rows = vec![0u32; count];
        for row in prepared.composition.lanes() {
            let at = row.source as usize;
            if at < count {
                first_row[at] = row.row_offset;
                lane_rows[at] = row.rows;
            }
        }

        // The epilogues read the forward's planes through their addresses.
        #[cfg(target_vendor = "apple")]
        frame.barrier(&self.device)?;

        let mut owed = Vec::with_capacity(prepared.attachments.len());
        match self.stage_epilogues(frame, prepared, &first_row, &lane_rows, &mut owed) {
            Ok(()) => Ok(owed),
            Err(fault) => {
                for instance in owed {
                    self.programs.abandon_launched(instance);
                }
                Err(fault)
            }
        }
    }

    pub(super) fn stage_epilogues(
        &mut self,
        frame: &mut Frame,
        prepared: &Prepared<'_>,
        first_row: &[u32],
        lane_rows: &[u32],
        owed: &mut Vec<engine::InstanceId>,
    ) -> Result<()> {
        // A lane's row of an output, and how many rows from it are the
        // lane's own: for one gathered at the readout rows, its first among
        // them and its readout rows; for one written over every token row,
        // its first named row and the lane's rows from there, else its last
        // row alone.
        let row_of = |lane: usize, gathered: bool| -> Result<(u32, u32)> {
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            if owned == 0 {
                return Err(Fault::Ceiling {
                    what: "rows in the lane an epilogue is attached to",
                    need: 1,
                    have: 0,
                });
            }
            if gathered {
                if prepared.readout_count.get(lane).copied().unwrap_or(0) == 0 {
                    return Err(Fault::Ceiling {
                        what: "readout rows in the lane an epilogue is attached to",
                        need: 1,
                        have: 0,
                    });
                }
                return Ok((
                    prepared.readout_first.get(lane).copied().unwrap_or(0),
                    prepared.readout_count.get(lane).copied().unwrap_or(0),
                ));
            }
            let last = first_row[lane] + owned - 1;
            match prepared.lanes.get(lane).and_then(|seated| seated.readout) {
                None => Ok((last, 1)),
                Some(rows) => {
                    for &row in rows {
                        if row >= owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: u64::from(row) + 1,
                                have: u64::from(owned),
                            });
                        }
                    }
                    Ok(rows
                        .first()
                        .map_or((last, 1), |&row| (first_row[lane] + row, owned - row)))
                }
            }
        };
        let mut staged = Vec::with_capacity(prepared.attachments.len());
        for attached in prepared.attachments.iter().filter(|a| !a.adapter_only) {
            let reads = self.programs.reads(attached.instance)?;
            for k in 0..attached.lanes.max(1) {
                let at = attached.lane as usize + k as usize;
                let Some((fire_lane, row)) = prepared
                    .composition
                    .lanes()
                    .iter()
                    .enumerate()
                    .find(|(_, row)| row.source as usize == at)
                else {
                    continue;
                };
                for out in 0..self.outputs.len() {
                    let decl = &self.outputs[out].decl;
                    // Read at this lane, or its row read by the lane-0 read:
                    // either way from this lane's own seat, never on lane
                    // 0's pitch into rows another request may hold.
                    if !self.outputs[out].classes.contains(row.class as usize)
                        || !(reads
                            .iter()
                            .any(|(name, lane)| *name == decl.name && *lane == k)
                            || self.programs.covers(attached.instance, &decl.name, k)?)
                    {
                        continue;
                    }
                    let (base, offset, width, dtype, stride, rows_held) = match decl.name.as_str() {
                        // A capturing lane's rows are its reader
                        // instance's state, not the arena's.
                        name @ ("attn.scores" | "attn.scores.pages") => {
                            let (_, holder) = prepared
                                .capture_rows
                                .0
                                .iter()
                                .find(|(source, ..)| *source as usize == at)
                                .copied()
                                .ok_or_else(|| Fault::Program {
                                    at: "serve::stage",
                                    why: format!(
                                        "lane {at} reads `{name}`, and its fire wrote it no score row"
                                    ),
                                })?;
                            let (buffer, keys_at) =
                                self.programs.capture_row(holder)?.ok_or_else(|| {
                                    Fault::Program {
                                        at: "serve::stage",
                                        why: format!("lane {at}'s reader holds no score rows"),
                                    }
                                })?;
                            let offset = if name == "attn.scores" {
                                keys_at
                            } else {
                                keys_at
                                    + self
                                        .capture
                                        .as_ref()
                                        .map_or(0, |layout| capture_row_bytes(layout.keys, 0).0)
                            };
                            // Its own score row.
                            (
                                buffer,
                                offset,
                                decl.width,
                                eta_ir::Dtype::F32,
                                decl.width,
                                1,
                            )
                        }
                        _ => {
                            let (offset, width, dtype, stride, held) =
                                self.output_seat(prepared, out, at, fire_lane, &row_of)?;
                            (
                                self.arena.store().clone(),
                                offset,
                                width,
                                dtype,
                                stride,
                                held,
                            )
                        }
                    };
                    let name = self.outputs[out].decl.name.clone();
                    self.programs.bind_output(
                        attached.instance,
                        &name,
                        k,
                        &base,
                        offset,
                        dtype,
                        width,
                        stride,
                        rows_held,
                    )?;
                }
            }
            // Every lane bound: a read past the rows its lanes hold would
            // read other requests' rows. Refused, its lane alone.
            if let Some(why) = self.programs.rows_unbounded(attached.instance)? {
                return Err(Fault::OutputRows {
                    lane: attached.lane,
                    instance: attached.instance.0,
                    why,
                });
            }
            staged.push(attached.instance);
        }
        let mut refusal = None;
        for (instance, launched) in self.programs.stage_batched(frame, &staged, prepared.arm)? {
            match launched {
                crate::program::Launched::Airborne => owed.push(instance),
                crate::program::Launched::Refused(fired) => {
                    refusal.get_or_insert((instance, fired));
                }
            }
        }
        if let Some((instance, fired)) = refusal {
            return Err(refused(&fired, instance));
        }
        Ok(())
    }

    /// Where lane `at` (the fire's lane `fire_lane`) reads output `out`: a
    /// lane output at its row (a layer a plane after another), a row output
    /// at its readout row, a voxel output from its first voxel.
    pub(super) fn output_seat(
        &self,
        prepared: &Prepared<'_>,
        out: usize,
        at: usize,
        fire_lane: usize,
        row_of: &dyn Fn(usize, bool) -> Result<(u32, u32)>,
    ) -> Result<(u64, u32, Dtype, u32, u32)> {
        let output = &self.outputs[out];
        let name = &output.decl.name;
        let (offset, width, dtype) = self.plane_row(prepared, output.values[0])?;
        let element = poem_compiler::arena::elem_bytes(dtype).unwrap_or(0);
        let at_row = |row: u64| offset + row * u64::from(width) * element;
        Ok(match output.decl.axis {
            poem_ir::InputAxis::Lane => {
                let mut stride = width;
                if let Some(&next) = output.values.get(1) {
                    let gap = self.plane_row(prepared, next)?.0.wrapping_sub(offset);
                    for pair in output.values.windows(2) {
                        let (a, b) = (
                            self.plane_row(prepared, pair[0])?.0,
                            self.plane_row(prepared, pair[1])?.0,
                        );
                        if b.wrapping_sub(a) != gap || element == 0 || gap % element != 0 {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "output `{name}`'s layers, which do not sit one plane \
                                     after another"
                                ),
                            });
                        }
                    }
                    stride = u32::try_from(gap / element).unwrap_or(u32::MAX);
                }
                // Its layers, one plane after another.
                let layers = u32::try_from(output.values.len())
                    .unwrap_or(u32::MAX)
                    .max(1);
                (at_row(fire_lane as u64), width, dtype, stride, layers)
            }
            poem_ir::InputAxis::Row => {
                let (row, held) = row_of(at, output.decl.readout)?;
                (at_row(u64::from(row)), width, dtype, width, held)
            }
            poem_ir::InputAxis::Voxel if output.grid.is_some() => {
                let (offset, width, dtype) = self.grid_seat(prepared, at, out)?;
                (offset, width, dtype, width, Self::lane_voxels(prepared, at))
            }
            poem_ir::InputAxis::Voxel => (
                at_row(u64::from(Self::first_voxel(prepared, at))),
                width,
                dtype,
                width,
                Self::lane_voxels(prepared, at),
            ),
        })
    }

    /// Waits out the frames carrying `instances` (and those sharing a ring
    /// with them). With `ahead`, an instance that runs ahead
    /// ([`crate::program::Session::runs_ahead`]) keeps its newest frame in
    /// flight, its next pass staged in its other set of tables while that
    /// one runs; every other is held to none, its next fire staged only
    /// once its last landed. A close waits out every one (`ahead` false).
    pub(super) fn fence_instances(
        &mut self,
        instances: &[engine::InstanceId],
        ahead: bool,
    ) -> Result<()> {
        if instances.is_empty() {
            return Ok(());
        }
        let cohort = self.programs.cohort(instances);
        self.reap()?;
        // An instance runs ahead only while no sharer of its rings fires in
        // this step or has a frame in flight: its next pass is staged on its
        // own commits, and a sharer's would move the cursors under it.
        let in_flight: Vec<engine::InstanceId> = self
            .inflight
            .iter()
            .flat_map(|flight| flight.attached.iter().map(|(held, _)| *held))
            .collect();
        let alone: BTreeMap<engine::InstanceId, bool> = instances
            .iter()
            .map(|&held| {
                let sharers = self.programs.cohort(&[held]);
                let busy = sharers
                    .iter()
                    .any(|sharer| instances.contains(sharer) || in_flight.contains(sharer));
                (held, !busy)
            })
            .collect();
        let allowed = |programs: &crate::program::Plane, held: &engine::InstanceId| -> usize {
            if ahead && !cohort.contains(held) && alone.get(held).copied().unwrap_or(false) {
                programs.ahead_depth(*held)
            } else {
                0
            }
        };
        loop {
            let mut carried: BTreeMap<engine::InstanceId, usize> = BTreeMap::new();
            for flight in &self.inflight {
                for (held, _) in &flight.attached {
                    if instances.contains(held) || cohort.contains(held) {
                        *carried.entry(*held).or_insert(0) += 1;
                    }
                }
            }
            // The oldest frame carrying an instance past what it may keep.
            let over = self
                .inflight
                .iter()
                .find(|flight| {
                    flight.attached.iter().any(|(held, _)| {
                        carried
                            .get(held)
                            .is_some_and(|&n| n > allowed(&self.programs, held))
                    })
                })
                .map(|flight| flight.seq);
            let Some(seq) = over else {
                return Ok(());
            };
            self.harvest_through(seq)?;
        }
    }

    /// Fired before staging, so descriptor reads and input feeds see the cells a prologue put.
    /// The step's prologues in one command buffer and one wait
    /// (`Programs::fire_batched`): their status and the cells they put are
    /// read on the host before the forward is encoded (each prologue's
    /// verdict, the descriptor ports and input feeds), so the wait stays,
    /// once a step rather than once a region an instance.
    pub(super) fn fire_prologues(&mut self, attachments: &[Attached]) -> Result<()> {
        let ids: Vec<engine::InstanceId> = attachments
            .iter()
            .filter(|a| !a.adapter_only && self.programs.prologue_runs(a.instance))
            .map(|a| a.instance)
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        for (instance, fired) in self
            .programs
            .fire_batched(&self.device, &ids, Stage::Prologue)?
        {
            match fired {
                crate::Fired::Committed => {}
                other => {
                    return Err(Fault::program(
                        "serve::prologue",
                        format!(
                            "instance {instance}'s prologue did not commit ({other:?}), so the \
                             forward has no cells to read"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn admit_attachments(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<()> {
        if attachments.is_empty() {
            return Ok(());
        }

        let instances: Vec<engine::InstanceId> = attachments.iter().map(|a| a.instance).collect();
        self.fence_instances(&instances, true)?;

        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
            if let Some(blocked) = self.programs.ready(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is not ready to fire: {blocked}, and an epilogue \
                         that discovered this after the forward would leave the lane's \
                         tokens in the cache with the guest's pass unrun",
                        attached.instance
                    ),
                ));
            }
            let stated = lanes
                .get(attached.lane as usize)
                .and_then(|seated| seated.readout);
            if let Some(rows) = stated
                && !rows.windows(2).all(|pair| pair[1] == pair[0] + 1)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {} states a readout list that is not one ascending run \
                         ({rows:?}), and instance {}'s readout rows are one binding at \
                         one offset: this plane can point a guest at `start .. start + k` \
                         and at nothing else",
                        attached.lane, attached.instance
                    ),
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::arc_with_non_send_sync)]
    pub(super) fn stage<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
        let StepView {
            lanes,
            attachments,
            media,
            clips,
            done,
        } = step;

        while self.inflight.len() >= self.arms.depth() {
            self.harvest_one()?;
        }
        let arm = self.arms.free().ok_or(Fault::Ceiling {
            what: "in-flight steps",
            need: self.arms.depth() as u64 + 1,
            have: self.arms.depth() as u64,
        })?;

        let Resolved {
            resolved,
            resolved_of,
            envelope_of,
            beam_groups,
            group_of,
        } = self.resolve_attachments(lanes, attachments)?;

        let DevicePorts {
            device_pages,
            device_writes,
            device_masks,
            lane_rows,
        } = self.device_ports(lanes, &resolved, &envelope_of)?;

        let media_of = self.media_rows(media, &lane_rows)?;
        let clips_of = self.clip_rows(clips, lanes.len())?;
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match (media_of[at], clips_of[at]) {
                (None, None) => FireLane::new(seated.lane.word, rows),
                (Some(shot), _) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().copied().fold(0u32, u32::saturating_add),
                ),
                (None, Some(tile)) => FireLane::with_clips(
                    seated.lane.word,
                    rows,
                    tile.boxes.len() as u32,
                    tile.boxes
                        .iter()
                        .map(|b| b[0].saturating_mul(b[1]).saturating_mul(b[2]))
                        .fold(0u32, u32::saturating_add),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // A load for a runtime lays no seat runs: every chunk is the
        // runtime's to name, so a lane it did not address by ref has none.
        if self.addressing == super::Addressing::Refs
            && let Some(seated) = lanes.iter().find(|seated| !seated.refs)
        {
            return Err(Fault::Unbound {
                what: format!(
                    "a seat-addressed lane (seat {}) on a load placed for the runtime: its \
                     pool lays no seat runs, and every lane names its slots by ref",
                    seated.lane.slot
                ),
            });
        }
        // A seat-addressed lane writes the seat runs' chunks; a weight slot
        // named there would be overwritten under the cache's feet.
        if lanes.iter().any(|seated| !seated.refs) && self.pools.weights_in_seats() {
            return Err(Fault::Unbound {
                what: "a seat-addressed lane while the weight cache holds a slot in a chunk \
                       the seat runs take"
                    .to_string(),
            });
        }
        let seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let kv_less: Vec<bool> = Vec::with_capacity(lanes.len());
        let tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // Per window list, per lane: its pages.
        let window_lists = self.pools.layout().window_lists().count();
        // A fire's windowed own-row pages, fills and folds start afresh.
        self.own_used = 0;
        // Rings two or more of this fire's lanes write (a beam group on one
        // record, unforked): none of them writes its own rows into it
        // directly, since a sibling's queries still read the rows it would
        // overwrite.
        self.ring_shared.clear();
        {
            let mut seen = std::collections::HashSet::new();
            for seated in lanes {
                if let Some(&[_, cow]) = seated.ring.and_then(|pieces| pieces.first())
                    && !seen.insert(cow)
                {
                    self.ring_shared.insert(cow);
                }
            }
        }
        self.window_fills.clear();
        self.window_folds.clear();
        let window_tables: Vec<Vec<Vec<u32>>> = vec![Vec::with_capacity(lanes.len()); window_lists];
        let tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        // Token rows copied from a cell on the device before the forward:
        // the row, and the cell's buffer and byte offset.
        let token_feeds: Vec<(u32, crate::device::Buffer, u64)> = Vec::new();
        // Rotation rows copied from a positions cell on the device before
        // the forward: the row, and the cell's buffer and byte offset.
        let position_feeds: Vec<(u32, crate::device::Buffer, u64)> = Vec::new();
        let positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each row's stated position, the mrope table's fill where a row
        // carries no stream of its own.
        let stated_rows: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each band's lane and row tables of record piece ids, filled lane
        // by lane and joined band after band below.
        let widths = self.pools.bands();
        let lane_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let row_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let state_top = 0u32;
        // Every physical fixed slot the fire's records name, each admitted
        // (checked backed) once the fire's segments are committed.
        let state_ids: Vec<u32> = Vec::new();
        let rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(lanes.len());
        // Each lane's buffered reads as one run ending at its first row
        // (`engine::fire::lane_one_run`), the input the shared compactness
        // helper takes, from the same verb values the plan is built from.
        let rs_one_run: Vec<bool> = Vec::with_capacity(lanes.len());
        // Per seat, its buffer's base and its set's rs pages (slab slots),
        // where it buffers: what a hybrid's beams resolve over (hybrid beams).
        let rs_sets: Vec<Option<(u32, Vec<u32>)>> = Vec::with_capacity(lanes.len());
        // Per seat, its own rows: what its carried rows count toward the
        // rung by (`engine::fire::lane_carried`).
        let rs_own: Vec<u32> = Vec::with_capacity(lanes.len());
        // Per reporting lane whose rows a port places, one past the last
        // slot they land at, in the set's own numbering (`KvWrites::end`).
        let ends: Vec<engine::KvWrites> = Vec::new();
        let rs_active = false;
        // Per fire lane, the rows inside it whose state lands in a slot of its
        // own and then those slots (`kernels_metal::ssm::Snapshots`).
        let snap_bands: Vec<Vec<i32>> = Vec::new();
        let snap_slots = 0u32;
        let request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        let masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        let beginning: Vec<Vec<u32>> = Vec::new();
        let writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        // What each lane writes and reads of its set (`engine::extent`).
        let extents = engine::extent::FireExtents::default();
        // Each beam group's fold: the slot its
        // set's state is folded to before any of its lanes reads it, as its
        // lanes state it (the runtime states it on the first); the fold the
        // fire runs, `[seat, rows]` a group; and per seat the rows its page
        // table row starts past its buffer's first (the group's fold).
        let mut beam_to: Vec<Option<u32>> = vec![None; beam_groups.len()];
        for (source, seated) in lanes.iter().enumerate() {
            if let (
                Some(group),
                engine::fire::RsVerb::Buffer {
                    folded,
                    fold: engine::fire::FoldLen::Host(n),
                    ..
                },
            ) = (group_of.get(source).copied().flatten(), seated.rs)
            {
                let to = folded.saturating_add(*n);
                beam_to[group] = Some(beam_to[group].map_or(to, |at: u32| at.max(to)));
            }
        }
        let beam_folded: Vec<bool> = vec![false; beam_groups.len()];
        let group_folds: Vec<i32> = Vec::new();
        let rs_skips: Vec<u32> = Vec::with_capacity(lanes.len());
        let mut acc = LaneAccum {
            seats,
            kv_less,
            tables,
            window_tables,
            tokens,
            token_feeds,
            position_feeds,
            positions,
            rotations,
            stated_rows,
            lane_bands,
            row_bands,
            state_top,
            state_ids,
            rs_plans,
            rs_one_run,
            rs_sets,
            rs_own,
            ends,
            rs_active,
            snap_bands,
            snap_slots,
            request_of_token,
            masks,
            adapter_routes,
            beginning,
            writes,
            extents,
            beam_groups,
            beam_folded,
            group_folds,
            rs_skips,
        };
        {
            let inputs = LaneInputs {
                lanes,
                envelope_of: &envelope_of,
                resolved: &resolved,
                resolved_of: &resolved_of,
                device_pages: &device_pages,
                device_writes: &device_writes,
                device_masks: &device_masks,
                group_of: &group_of,
                beam_to: &beam_to,
                widths: &widths,
                any_adapter,
            };
            for row in composition.lanes() {
                self.stage_lane(row, &inputs, &mut acc)?;
            }
        }
        let LaneAccum {
            seats,
            kv_less,
            tables,
            window_tables,
            tokens,
            token_feeds,
            position_feeds,
            positions,
            rotations,
            stated_rows,
            lane_bands,
            row_bands,
            state_top,
            state_ids,
            rs_plans,
            rs_one_run,
            rs_sets,
            rs_own,
            ends,
            rs_active,
            snap_bands,
            snap_slots,
            request_of_token,
            masks,
            adapter_routes,
            beginning,
            writes,
            mut extents,
            mut beam_groups,
            beam_folded: _,
            group_folds,
            rs_skips,
        } = acc;
        // A page holds what its last owner left until its set writes it: no
        // lane reads past what its set holds written.
        extents
            .check()
            .map_err(|why| Fault::program("serve::prepare", why))?;
        self.written.commit(extents);

        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let paging = self.pools.paging();
        let written = writes
            .iter()
            .flatten()
            .map(|&(page, _)| u64::from(self.pools.list_slot(page.max(0) as u32)).saturating_add(1))
            .max()
            .unwrap_or(0);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .zip(&kv_less)
                // A kv-less lane takes no pages, so none is committed for it.
                .filter(|(_, free)| !**free)
                .map(|((seat, table), _)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        paging.base(seat.slot).saturating_add(pages)
                    } else {
                        table
                            .iter()
                            .take(pages as usize)
                            .map(|&page| self.pools.list_slot(page))
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .chain(std::iter::once(written))
                .max()
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: state_top.max(snap_slots),
            workspace: 0,
        };
        // A caller firing directly (every lane seat-addressed) has no
        // runtime to commit ahead of it: the segments its seats and its
        // demand touch back now.
        if lanes.iter().all(|seated| !seated.refs) && !self.warming {
            self.pools
                .commit_seats(demand.kv_pages, demand.state_slots)?;
        }
        // Every page and record piece the fire reads or writes is checked
        // backed, ref by ref: a page in an uncommitted chunk under the
        // highest one (two-ended growth backs a bottom run and the span's
        // top, not what lies between) is refused by name, not read.
        // The load's pipeline walk (`warming`) stages fires it drops
        // uncommitted: no kernel reads or writes what they name, so nothing
        // is checked backed and no record is zeroed for them. Its lanes are
        // seat 0's, and a recurrent model's seat-0 record lies past every
        // seat's kv run (seat runs follow one another shape by shape): the
        // walk refused such a load ("this fire wants 13113 committed chunks
        // of the state pool and the load reserved 58").
        for ((seat, table), &free) in seats.iter().zip(&tables).zip(&kv_less) {
            if free || table.is_empty() || self.warming {
                continue;
            }
            let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
            let pages = usize::try_from(after.div_ceil(page_size).max(1)).unwrap_or(usize::MAX);
            for &page in table.iter().take(pages) {
                self.pools.admit_list_page(page)?;
            }
        }
        for &(page, _) in writes.iter().flatten().filter(|_| !self.warming) {
            self.pools.admit_list_page(page.max(0) as u32)?;
        }
        for &id in state_ids.iter().filter(|_| !self.warming) {
            self.pools.admit_state_slot(id)?;
        }

        // A fresh record is zeroed by the fire that first reads it, at the
        // head of its own command buffer (`walk`): in queue order behind the
        // frames before it. Zeroed on the host, it drained every frame in
        // flight first, and each hybrid / SSM admission ran at depth 1.
        let zeroed: Vec<Vec<u32>> = if self.warming { Vec::new() } else { beginning };
        let slot_ids: Vec<i32> = lane_bands.concat();
        let rs_snaps: Vec<i32> = snap_bands.concat();
        let slot_of_row: Vec<i32> = row_bands.concat();

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let window_refs: Vec<Vec<&[u32]>> = window_tables
            .iter()
            .map(|lists| lists.iter().map(Vec::as_slice).collect())
            .collect();
        let mut geometries = Vec::with_capacity(self.page_tables);
        for &paging_of in self.pools.tables() {
            geometries.push(match paging_of {
                crate::store::TablePaging::List => kv::geometry_with(&paging, &seats, &table_refs)?,
                crate::store::TablePaging::Window(k) => {
                    let k = usize::from(k);
                    let refs = window_refs.get(k).ok_or_else(|| {
                        Fault::program("serve::prepare", format!("window list {k} has no table"))
                    })?;
                    kv::geometry_with(&self.pools.window_paging(k), &seats, refs)?
                }
            });
        }
        if writes.iter().any(Option::is_some) {
            for (geometry, &paging_of) in geometries.iter_mut().zip(self.pools.tables()) {
                if paging_of != crate::store::TablePaging::List {
                    continue;
                }
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "row {row} states an explicit write descriptor and the \
                                 page arithmetic placed {} row(s)",
                                geometry.write_page.len()
                            ),
                        ));
                    };
                    *write_page = page;
                    *write_offset = offset;
                }
            }
        }
        let geometries = geometries;
        // The beam groups' head step, planned over the tables as staged:
        // its records point into this arm's inputs.
        let (mut beam_records, beam_faults) = if beam_groups.is_empty() || self.warming {
            (crate::inputs::BeamRecords::default(), Vec::new())
        } else {
            for group in &mut beam_groups {
                group.lanes.sort_by_key(|lane| lane.source);
            }
            let inputs = &self.inputs[self.arena.current()][arm];
            let addresses = inputs.beam_addresses().ok_or_else(|| {
                Fault::program("serve::prepare", "the beam step's inputs have no address")
            })?;
            let tables = (0..geometries.len())
                .map(|table| {
                    inputs.table_addresses(table).ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!("page table {table} has no staged address"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let staged = crate::beam::Staged {
                indptr: geometries
                    .iter()
                    .map(|geometry| geometry.indptr.as_slice())
                    .collect(),
                tables,
                windowed: self
                    .pools
                    .tables()
                    .iter()
                    .any(|paging_of| *paging_of != crate::store::TablePaging::List),
                page: self.pools.paging().page_size,
                addresses,
                planes: self.pools.kv_page_planes(),
                max_lanes: self.programs.beam_max_lanes(),
            };
            let pools = &self.pools;
            crate::beam::plan(
                &beam_groups,
                &staged,
                &|source| lanes[source].translation.to_vec(),
                &|source| lanes[source].folded,
                &mut |refs| {
                    // The group's translation names slot refs (logical):
                    // translated through the chunk table first, then each
                    // physical page admitted. Admitting the refs themselves
                    // checked the wrong chunk once an evacuation moved one.
                    let ids = pools.kv_pages(refs, None)?;
                    for &page in &ids {
                        pools.admit_list_page(page)?;
                    }
                    Ok((ids, pools.kv_page_addresses(refs)?))
                },
            )?
        };
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        let bucket = self
            .budgets
            .tokens
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            composition.voxel_classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,
                enabled: self.copies && masks.iter().all(|lane| lane.mask.is_none()),
                page_tables: &geometries,
                positions: &positions,
                rotations: &rotations,
                request_of_token: &request_of_token,
            },
        )?;
        let boundaries = windows.packed();

        let staged =
            crate::mask::stage(&masks).map_err(|fault| fault.of_rows(composition.lanes()))?;

        let patch_rows = composition.patch_rows() as usize;
        let mut patch_payload: Vec<u8> = Vec::new();
        let mut patch_segments: Vec<i32> = Vec::new();
        let mut patch_routes: Vec<i32> = Vec::new();
        let mut patch_positions: Vec<i32> = Vec::new();
        let mut patch_grids: Vec<i32> = Vec::new();
        if patch_rows > 0 {
            let seat = self.patch_seat.expect(
                "a composition with patch rows came out of budgets with a patch ladder, and \
                 the seat is derived from the same trace the ladder admitted",
            );
            let stride = seat.row_bytes as usize;
            patch_payload = vec![0u8; patch_rows * stride];
            patch_positions = vec![0i32; patch_rows * 3];
            patch_grids = vec![0i32; composition.images() as usize * 3];
            patch_routes = vec![
                if self.drops_patch_rows {
                    PATCH_ROUTE_DROP
                } else {
                    0
                };
                patch_rows
            ];
            let mut per_image: Vec<u32> = vec![0; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                patch_payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);

                place_routes(
                    &mut patch_routes,
                    row.patch_offset,
                    row.patches,
                    row.row_offset,
                    self.patch_fold,
                    shot.routes,
                );
                let triples = row.patch_offset as usize * 3;
                patch_positions[triples..triples + shot.positions.len()]
                    .copy_from_slice(shot.positions);
                let at_grids = row.image_offset as usize * 3;
                patch_grids[at_grids..at_grids + shot.grids.len()].copy_from_slice(shot.grids);
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }
            patch_segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            patch_segments.push(at);
            for rows in per_image {
                at = at.saturating_add(rows as i32);
                patch_segments.push(at);
            }
        }

        let mut mrope_positions: Vec<i32> = Vec::new();
        if self.states_mrope {
            mrope_positions = vec![0i32; rows as usize * 3];
            for row in composition.lanes() {
                let stated = media_of[row.source as usize]
                    .map(|shot| shot.token_positions)
                    .filter(|stream| !stream.is_empty());
                let at = row.row_offset as usize * 3;
                match stated {
                    Some(stream) => {
                        mrope_positions[at..at + stream.len()].copy_from_slice(stream);
                    }
                    None => {
                        for i in 0..row.rows as usize {
                            let p = stated_rows[row.row_offset as usize + i];
                            mrope_positions[at + 3 * i] = p;
                            mrope_positions[at + 3 * i + 1] = p;
                            mrope_positions[at + 3 * i + 2] = p;
                        }
                    }
                }
            }
        }

        let rs_replay: Vec<i32> = rs_plans.iter().map(|plan| plan.replay as i32).collect();
        let rs_commit: Vec<i32> = rs_plans.iter().map(|plan| plan.commit as i32).collect();
        let rs_page_ints = self.inputs[self.arena.current()][0].rs_page_ints();
        let rs_pages = match (rs_active, self.rs_buffers.as_ref()) {
            (true, Some(buffers)) => {
                crate::rs::page_table(&rs_plans, buffers.page_tokens(), rs_page_ints)?
            }
            _ => None,
        };
        // A compact fire (every recurrent op on the paged forms, a page
        // table, no snapshots): a decode lane reads its carried rows from
        // its buffer wherever it lies, and takes no extended rows; the rows
        // each lane takes are what it counts toward the rung
        // (`engine::fire::lane_carried`). Else every carried row is
        // re-stepped through the extended rows.
        let compact = rs_pages.is_some()
            && rs_snaps.is_empty()
            && self.rs_layout.as_ref().is_some_and(|layout| layout.compact);
        let rs_ext_replay: Vec<u32> = rs_plans
            .iter()
            .zip(&rs_own)
            .map(|(plan, &own)| {
                if compact {
                    engine::fire::lane_carried(own, plan.replay)
                } else {
                    plan.replay
                }
            })
            .collect();
        // The fire's rows against what its rung's rs scratch holds, by the
        // one function the runtime's rung choice also calls
        // (`engine::fire::rung_holds` over the bounds this load states,
        // `Shell::rung_replay`: own T, carried T, and the rung's rs scratch
        // bytes).
        // A fire past them was cut or chosen wrong: refused by name before
        // anything is encoded.
        if rs_active {
            let tokens = self.arena.current_tokens();
            let rung = self.rung_replay(tokens).ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    "an rs fire on a load that lays no rs layout".to_string(),
                )
            })?;
            // The fire's compactness by the one function the runtime also
            // evaluates on this fire (`engine::fire::scratch_compact`: the
            // load's stated layout fact, every lane one run, no snapshot),
            // so the rung the runtime chose and this check agree. What the
            // fire really lays in its scratch stays the seat's bytes check
            // below, the named guard against the two ever parting.
            let load_compact = self.rs_layout.as_ref().is_some_and(|layout| layout.compact)
                && self.rs_buffers.is_some();
            let fire_compact = engine::fire::scratch_compact(
                load_compact,
                true,
                rs_one_run.iter().all(|&one| one),
                !rs_snaps.is_empty(),
            );
            if fire_compact != compact {
                tracing::warn!(
                    helper = fire_compact,
                    laid = compact,
                    "engine::fire::scratch_compact and the rs scratch Metal lays disagree for \
                     this fire (a contract bug): the bytes check below guards it"
                );
            }
            let held = engine::ReplayRows::of_lanes(
                rs_plans
                    .iter()
                    .zip(&rs_own)
                    .map(|(plan, &own)| (own, plan.replay)),
                fire_compact,
            );
            if engine::fire::replay_check_on() {
                engine::fire::replay_agrees(
                    lanes
                        .iter()
                        .zip(&rs_own)
                        .map(|(seated, &own)| (own, seated.replay_stated)),
                    &rung,
                    held,
                    fire_compact,
                    load_compact,
                    true,
                    !rs_snaps.is_empty(),
                )
                .map_err(|why| Fault::program("serve::rs", why))?;
            }
            if let Err(past) = engine::fire::rung_holds(&rung, held) {
                return Err(Fault::program(
                    "serve::rs",
                    format!(
                        "this fire's lanes carry {} recurrent rows into it ({} own, {} in its \
                         rs scratch), past the {tokens} tokens of the rung it stages in: {past} \
                         (engine::fire::rung_holds); the fire is refused and writes nothing",
                        held.carried, held.own, held.scratch
                    ),
                ));
            }
        }
        let rows_ext = crate::rs::scratch_rows(rows, rs_ext_replay.iter().sum::<u32>());
        let rs_ext_replay_i32: Vec<i32> = rs_ext_replay.iter().map(|&n| n as i32).collect();
        // A beam lane whose group folds first replays from the fold: its
        // row starts that many rows on, the rows below it the group fold's
        // (the group fold), which reads them through the same row.
        let rs_pages = match rs_pages {
            Some(mut table) => {
                let ints = rs_page_ints as usize;
                for (at, &skip) in rs_skips.iter().enumerate() {
                    if skip > 0 && table[at * ints] >= 0 {
                        table[at * ints] += skip as i32;
                    }
                }
                Some(table)
            }
            None if !group_folds.is_empty() => {
                return Err(Fault::program(
                    "serve::rs",
                    "a beam group folds its shared rows before its step, and its lanes' rows lie \
                     in no one stretch of pages a lane: a group fold reads them through a page \
                     table; the fire is refused and writes nothing",
                ));
            }
            None => None,
        };
        // A hybrid's beams: each beam's rs page table,
        // resolved over its list by the head step, and its rs copies over
        // every rs plane, through its set's rs entries.
        if !beam_records.groups.is_empty()
            && let (Some(_), Some(buffers), Some(layout)) = (
                rs_pages.as_ref(),
                self.rs_buffers.as_ref(),
                self.rs_layout.as_ref(),
            )
        {
            let inputs = &self.inputs[self.arena.current()][arm];
            let addresses = inputs.beam_addresses().ok_or_else(|| {
                Fault::program("serve::prepare", "the beam step's inputs have no address")
            })?;
            let table = inputs.rs_pages_address().ok_or_else(|| {
                Fault::program("serve::prepare", "the rs page table has no staged address")
            })?;
            let slab = buffers.slab_address().ok_or_else(|| {
                Fault::program("serve::prepare", "the rs buffer's pool has no GPU address")
            })?;
            let planes = buffers
                .plane_offsets()
                .iter()
                .zip(&layout.planes)
                .map(|(&offset, plane)| kernels_metal::runtime::beam::PagePlane {
                    offset,
                    cell: plane.row_bytes,
                })
                .collect();
            crate::beam::plan_rs(
                &mut beam_records,
                &beam_groups,
                &crate::beam::RsStaged {
                    sets: &rs_sets,
                    plans: &rs_plans,
                    table,
                    ints: rs_page_ints,
                    page_tokens: buffers.page_tokens(),
                    page_bytes: buffers.page_bytes(),
                    slab,
                    planes,
                },
                &addresses,
            )?;
        }
        if rs_active {
            let layout = self
                .rs_layout
                .as_ref()
                .expect("rs_active implies a layout, checked at the lane");
            let need = crate::rs::Seat::scratch_bytes(layout, rows_ext, lane_count);
            let have = self.arena.rs_region().1;
            if need > have {
                return Err(Fault::Ceiling {
                    what: "bytes of the rung's rs scratch in the span (its rows and the rows its lanes carry in)",
                    need,
                    have,
                });
            }
        }

        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated.inputs.iter().any(|feed| feed.id == merged.seat.id);
                merge_lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        at_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    fed,
                });
            }
        }

        let mut input_feeds: Vec<InputFeedPlan> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for (seat, readers) in &self.feeds.inputs {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let Some(feed) = seated.inputs.iter().find(|feed| feed.id == seat.id) else {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} runs in a class that reads guest input {}, and the \
                             lane feeds no channel for it — the input's rectangle would \
                             carry the last fire's rows",
                            row.source, seat.id
                        ),
                    });
                };
                let voxel = seat.voxel();
                let cells = if seat.per_lane() {
                    1
                } else if voxel {
                    u64::from(row.voxels)
                } else {
                    u64::from(row.rows)
                };
                let offset = if seat.per_lane() {
                    at_lane as u64
                } else if voxel {
                    u64::from(row.voxel_offset)
                } else {
                    u64::from(row.row_offset)
                };
                let Some(attached) = attachments
                    .iter()
                    .find(|a| (a.lane..a.lane + a.lanes.max(1)).contains(&row.source))
                else {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} feeds guest input {} and has no attachment; an input \
                             is read through the instance the lane is attached to",
                            row.source, seat.id
                        ),
                    });
                };
                input_feeds.push(InputFeedPlan {
                    id: seat.id,
                    voxel,
                    at: offset * seat.row_bytes(),
                    bytes: cells * seat.row_bytes(),
                    channel: feed.channel,
                    dtype: seat.dtype,
                    instance: attached.instance,
                    lane: row.source,
                });
            }
            for feed in seated.inputs {
                if self.feeds.seat(feed.id).is_none() {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} feeds guest input {}, and this plan reads no such input",
                            row.source, feed.id
                        ),
                    });
                }
            }
        }

        let mut voxel_grid: Vec<i32> = Vec::new();
        let mut voxel_slots: Vec<i32> = Vec::new();
        let mut voxel_token_grid: Vec<i32> = Vec::new();
        let mut voxel_payload: Vec<u8> = Vec::new();
        if composition.clips() > 0 {
            let element = self.voxel_seat.map_or(2u64, |seat| {
                poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(2)
            });
            let channels = self.voxel_seat.map_or(1u64, |seat| seat.channels);
            let token_patch = self.voxel_seat.and_then(|seat| seat.token_patch);
            let row_bytes = element * channels;
            voxel_payload = vec![0u8; composition.voxel_rows() as usize * row_bytes as usize];
            for row in composition.lanes() {
                let Some(tile) = clips_of[row.source as usize] else {
                    continue;
                };
                let mut offset = row.voxel_offset;
                let mut token_row = row.row_offset;
                for boxed in tile.boxes {
                    voxel_grid.extend_from_slice(&[
                        narrow(u64::from(boxed[0])),
                        narrow(u64::from(boxed[1])),
                        narrow(u64::from(boxed[2])),
                        narrow(u64::from(offset)),
                    ]);
                    let seated = &lanes[row.source as usize];
                    voxel_slots.push(self.state_id(seated)? as i32);
                    let voxels = boxed[0].saturating_mul(boxed[1]).saturating_mul(boxed[2]);
                    offset = offset.saturating_add(voxels);
                    if let Some(p) = token_patch {
                        token_row = token_cell(&mut voxel_token_grid, *boxed, p, token_row)
                            .ok_or_else(|| token_refusal(row.source))?;
                    }
                }
                if token_patch.is_some() && token_row > row.row_offset + row.rows {
                    return Err(token_refusal(row.source));
                }
                if !tile.payload.is_empty() {
                    let at = u64::from(row.voxel_offset) * row_bytes;
                    let end = at + tile.payload.len() as u64;
                    if end > voxel_payload.len() as u64 {
                        return Err(Fault::Ceiling {
                            what: "voxel payload bytes",
                            need: end,
                            have: voxel_payload.len() as u64,
                        });
                    }
                    voxel_payload[at as usize..end as usize].copy_from_slice(tile.payload);
                }
            }
        }

        let (group_of_lane, packings) = if self.feeds.selections.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let facts: Vec<poem_exec::fire::LaneFacts> = lanes
                .iter()
                .map(|seated| poem_exec::fire::LaneFacts {
                    group: seated.group,
                })
                .collect();
            let group_of_lane = poem_exec::fire::group_of_lane(composition.lanes(), &facts);
            let mut packings = Vec::with_capacity(self.feeds.selections.len());
            for &select in &self.feeds.selections {
                packings.push(
                    poem_exec::fire::pack(select, composition.lanes(), &group_of_lane, rows)
                        .map_err(|fault| Fault::from(poem_exec::Error::Fire(fault)))?,
                );
            }
            (group_of_lane, packings)
        };
        let group_masks =
            self.group_masks_of(lanes, &composition, &group_of_lane, &packings, &clips_of)?;
        let packing_fires: Vec<crate::inputs::PackingFire<'_>> = packings
            .iter()
            .map(|packed| crate::inputs::PackingFire {
                group_indptr: &packed.group_indptr,
                lane_indptr: &packed.lane_indptr,
                permutation: &packed.permutation,
            })
            .collect();

        let mut readout_first: Vec<u32>;
        let mut readout_count: Vec<u32>;
        let mut readout_indptr = vec![0u32];
        let readout_rows: Vec<i32> = {
            let mut placed: Vec<(u32, u32, usize)> = composition
                .lanes()
                .iter()
                .map(|row| (row.row_offset, row.rows, row.source as usize))
                .collect();
            placed.sort_unstable();
            let mut table = Vec::with_capacity(placed.len());
            readout_first = vec![0u32; lanes.len()];
            readout_count = vec![0u32; lanes.len()];
            for (row_offset, owned, source) in placed {
                let wanted: Vec<u32> = match lanes.get(source).and_then(|seated| seated.readout) {
                    Some(rows) if !rows.is_empty() => rows.to_vec(),
                    _ => vec![owned.saturating_sub(1)],
                };
                readout_first[source] = table.len() as u32;
                readout_count[source] = wanted.len() as u32;
                for row in wanted {
                    if row >= owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(row) + 1,
                            have: u64::from(owned),
                        });
                    }
                    table.push(i32::try_from(row_offset + row).unwrap_or(0));
                }
                readout_indptr.push(table.len() as u32);
            }
            table
        };

        let readouts_ceiling = poem_compiler::arena::readouts_ceiling(&self.budgets.tokens);
        if readout_rows.len() as u64 > readouts_ceiling {
            return Err(Fault::Ceiling {
                what: "rows one fire reads logits for",
                need: readout_rows.len() as u64,
                have: readouts_ceiling,
            });
        }
        let bound = self.inputs[self.arena.current()][arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                readout_rows: &readout_rows,
                positions: &positions,
                rotations: &rotations,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_ext_replay: rs_active.then_some(rs_ext_replay_i32.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                rs_snaps: (!rs_snaps.is_empty()).then_some(rs_snaps.as_slice()),
                rs_pages: rs_pages.as_deref(),
                rs_folds: &group_folds,
                request_of_token: &request_of_token,
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                page_tables: &geometries,
                mask: staged.as_ref(),
                patches: (patch_rows > 0).then_some(crate::inputs::PatchFire {
                    payload: &patch_payload,
                    segments: &patch_segments,
                    routes: &patch_routes,
                    positions: &patch_positions,
                    grids: &patch_grids,
                }),
                voxels: (composition.clips() > 0).then_some(crate::inputs::VoxelFire {
                    grid: &voxel_grid,
                    token_grid: &voxel_token_grid,
                    slots: &voxel_slots,
                    payload: &voxel_payload,
                }),
                mrope_positions: self.states_mrope.then_some(mrope_positions.as_slice()),
                group_of_lane: &group_of_lane,
                packings: &packing_fires,
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;
        let (beam, beam_check) = if beam_records.groups.is_empty() {
            (None, None)
        } else {
            let inputs = &mut self.inputs[self.arena.current()][arm];
            let staged = inputs.stage_beam(&self.handles, &beam_records)?;
            let reader = inputs.beam_fault_reader(beam_records.lanes)?;
            (staged, Some((reader, beam_faults)))
        };
        let group_masks = self.inputs[self.arena.current()][arm]
            .stage_group_masks(&self.handles, &group_masks)?;

        let fire = FireRows {
            tokens: u64::from(rows),
            lanes: u64::from(lane_count),
            patches: u64::from(composition.patch_rows()),
            images: u64::from(composition.images()),
            voxels: u64::from(composition.voxel_widest()),
            clips: u64::from(composition.clips()),
            readouts: readout_rows.len() as u64,
        };
        // The rung's carve was backed when the runtime raised to it
        // (`set_rung`); a fire never backs anything.
        if !self.pools.covers(self.arena.reach(fire)) {
            return Err(Fault::Program {
                at: "serve::stage",
                why: format!(
                    "a fire of {} tokens over {} lanes is wider than the rung the runtime \
                     raised to, whose carve alone is backed",
                    fire.tokens, fire.lanes
                ),
            });
        }
        let slots = self.arena.fire_slots(&self.handles, fire)?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[self.arena.current()][arm].seats(
                &self.handles,
                &bound,
                pages,
                rows,
                lane_count,
            ),
            self.tuning.kv_q8(),
        )?;

        let mut geometry = Vec::with_capacity(self.page_tables);
        for table in 0..self.page_tables {
            let seat = bound.page_tables[table];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(bound.row_valid),
                request_of_token: Some(bound.request_of_token),
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
            });
        }
        let patch_seats = bound.patches;
        // The capturing lanes' score rows: each lane's reader instance's.
        let (capture_seat, capture_rows) = self.lay_capture(&composition, attachments, arm)?;
        self.last_capture = capture_rows.clone();
        let bindings = FireBindings {
            tokens: bound.tokens,
            positions: bound.positions,
            rotations: bound.rotations,
            adapter_routes: bound.adapter_routes,
            readout_rows: bound.readout_rows,
            readout_indptr,
            nan_flags: match self.nan_flags.as_ref() {
                Some(plane) => {
                    let words = self.trace.values.len() as u32 + 1;
                    Some(kernels_metal::Tensor::new(
                        self.handles.bind(plane, 0, u64::from(words) * 4)?,
                        words,
                        1,
                        Dtype::U32,
                    ))
                }
                None => None,
            },
            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),
            patch_grids: patch_seats.map(|seats| seats.grids),
            mrope_positions: bound.mrope_positions,
            group_of_lane: bound.group_of_lane,
            packings: self
                .feeds
                .selections
                .iter()
                .copied()
                .zip(bound.packings.iter().copied())
                .collect(),
            group_masks,
            voxels: bound.voxels,
            inputs: bound.inputs,
            geometry,
            window_tables: self
                .pools
                .layout()
                .window_lists()
                .enumerate()
                .map(|(k, window)| {
                    (
                        window.rows,
                        self.pools.table_of(crate::store::TablePaging::Window(
                            u8::try_from(k).unwrap_or(u8::MAX),
                        )),
                    )
                })
                .collect(),
            tables: FireTables {
                request_of_token: bound.request_of_token,
                readout_positions: bound.readout_positions,
                readout_request_of_token: bound.readout_request_of_token,
                readout_mask_enabled: bound.readout_mask_enabled,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },
            rs_snaps: bound.rs_snaps,
            rs_snap_bands: bound.rs_snap_bands.clone(),
            capture: capture_seat,
            rs: if rs_active {
                let (Some(layout), Some(buffers)) = (&self.rs_layout, &self.rs_buffers) else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat of a fire whose plan buffers nothing".to_string(),
                    });
                };
                let (Some(replay), Some(commit), Some(ext_table)) =
                    (bound.rs_replay, bound.rs_commit, bound.rs_ext_replay)
                else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat's per-lane tables".to_string(),
                    });
                };
                Some(std::sync::Arc::new(crate::rs::Seat::mint(
                    &self.handles,
                    layout,
                    buffers,
                    (self.arena.store(), self.arena.rs_region()),
                    rs_plans,
                    (rs_ext_replay, ext_table),
                    replay,
                    commit,
                    bound.slot_ids,
                    rows_ext,
                    bound.rs_pages,
                    rs_page_ints,
                    bound.rs_folds,
                )?))
            } else {
                None
            },
        };

        Ok(Prepared {
            beam,
            beam_check,
            zeroed,
            capture_rows,
            ends,
            lanes,
            input_feeds,
            token_feeds,
            position_feeds,
            merge_lands,
            attachments,
            done,
            arm,
            composition,
            readout_first,
            readout_count,
            voxel_grid,
            descriptor,
            seats,
            tables,
            windows,
            slots,
            caches,
            bindings,
            demand,
        })
    }

    /// Each capturing lane of `composition`, in lane order, given the key
    /// row and page row its reader instance holds (the instance attached at
    /// its lane, reading `attn.scores` or `attn.scores.pages`), and the
    /// arm's tables of every lane's row starts written for the fire's score
    /// kernels: staged each fire, so a lane's rows are read wherever its
    /// reader's slots lie now. The load's pipeline walk names no row: it
    /// launches nothing. Every capturing lane a fire carries has its reader
    /// attached: only the fire that runs a pass's program captures, and
    /// `engine::fire::Step::validate_captures` refuses at submit a lane that
    /// captures with none.
    pub(super) fn lay_capture(
        &mut self,
        composition: &Composition,
        attachments: &[Attached],
        arm: usize,
    ) -> Result<(Option<crate::run::CaptureSeat>, CaptureRows)> {
        let Some(layout) = self.capture.as_ref() else {
            return Ok((None, CaptureRows::default()));
        };
        let lanes = composition.lanes();
        let (key_bytes, _) = capture_row_bytes(layout.keys, layout.pages);
        let mut keys_at = vec![0u64; lanes.len().max(1)];
        let mut pages_at = vec![0u64; lanes.len().max(1)];
        let mut rows = Vec::new();
        let mut captures = false;
        for (at, row) in lanes.iter().enumerate() {
            if !layout.classes.contains(row.class as usize) {
                continue;
            }
            captures = true;
            if self.warming {
                continue;
            }
            let source = row.source;
            let attached = attachments
                .iter()
                .find(|attached| {
                    !attached.adapter_only
                        && (attached.lane..attached.lane + attached.lanes.max(1)).contains(&source)
                })
                .ok_or_else(|| Fault::Program {
                    at: "serve::lay_capture",
                    why: format!(
                        "lane {source} captures its attention scores and no reader is attached \
                         at it: only the fire that runs a pass's program captures \
                         (engine::fire::Step::validate_captures refuses this at submit)"
                    ),
                })?;
            if attached.lanes > 1 {
                return Err(Fault::Program {
                    at: "serve::lay_capture",
                    why: format!(
                        "lane {source} captures its attention scores, and instance {} is \
                         attached at {} lanes: an instance holds one lane's score rows",
                        attached.instance, attached.lanes
                    ),
                });
            }
            let instance = attached.instance;
            let (buffer, offset) =
                self.programs
                    .capture_row(instance)?
                    .ok_or_else(|| Fault::Program {
                        at: "serve::lay_capture",
                        why: format!(
                            "lane {source} captures its attention scores, and instance {instance} \
                         attached at it reads none of them, so it holds no score rows"
                        ),
                    })?;
            let address = buffer.address_at(offset).ok_or_else(|| Fault::Device {
                call: "score rows",
                why: format!("lane {source}'s reader's score rows name no GPU address"),
            })?;
            keys_at[at] = address;
            pages_at[at] = address + key_bytes;
            rows.push((source, instance));
        }
        if !captures {
            return Ok((None, CaptureRows::default()));
        }
        let Some(layout) = self.capture.as_ref() else {
            return Ok((None, CaptureRows::default()));
        };
        let (keys_table, pages_table) = self.inputs[self.arena.current()][arm].stage_captures(
            &self.handles,
            &keys_at,
            &pages_at,
        )?;
        let seat = crate::run::CaptureSeat {
            keys_at: keys_table,
            pages_at: pages_table,
            keys: layout.keys,
            pages: layout.pages,
            page_size: layout.page_size,
        };
        Ok((Some(seat), CaptureRows(rows)))
    }

    /// The last fire's capturing lanes' rows, read back from their
    /// readers: each one's source lane, key row and page row (none where no
    /// lane of it captured). A later fire may write its own over them.
    #[allow(clippy::type_complexity)]
    pub fn score_rows(&mut self) -> Result<Vec<(u32, Vec<f32>, Vec<f32>)>> {
        self.drain()?;
        let Some(layout) = self.capture.as_ref() else {
            return Ok(Vec::new());
        };
        let (key_bytes, _) = capture_row_bytes(layout.keys, layout.pages);
        let read = |instance: engine::InstanceId, at: u64, floats: u32| -> Result<Vec<f32>> {
            let mut out = vec![0u8; floats as usize * 4];
            self.programs.read_capture(instance, at, &mut out)?;
            Ok(out
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        self.last_capture
            .0
            .iter()
            .map(|&(source, instance)| {
                Ok((
                    source,
                    read(instance, 0, layout.keys)?,
                    read(instance, key_bytes, layout.pages)?,
                ))
            })
            .collect()
    }

    /// Copies each token row a pass in flight put into the fire's token
    /// seat, on the device, ahead of the forward: the queue runs it after
    /// that pass, so the cell holds its token by then.
    pub(super) fn feed_positions(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.position_feeds.is_empty() {
            return Ok(());
        }
        let (seat, base) = self.inputs[self.arena.current()][p.arm].rotations_seat();
        for (row, buffer, cell) in &p.position_feeds {
            frame.copy_span(buffer, *cell, seat, base + 4 * u64::from(*row), 4)?;
        }
        frame.next_pass()?;
        Ok(())
    }

    pub(super) fn feed_tokens(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.token_feeds.is_empty() {
            return Ok(());
        }
        let (seat, base) = self.inputs[self.arena.current()][p.arm].tokens_seat();
        for (row, buffer, cell) in &p.token_feeds {
            frame.copy_span(buffer, *cell, seat, base + 4 * u64::from(*row), 4)?;
        }
        frame.next_pass()?;
        Ok(())
    }

    pub(super) fn feed_inputs(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.input_feeds.is_empty() {
            return Ok(());
        }
        let mut casts: Vec<(u32, u32, u32)> = Vec::new();
        for feed in &p.input_feeds {
            let (store, at) = if feed.voxel {
                self.inputs[self.arena.current()][p.arm]
                    .voxel_payload()
                    .ok_or_else(|| Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds voxel guest input {} and this load carved no voxel \
                             payload",
                            feed.lane, feed.id
                        ),
                    })?
            } else {
                self.inputs[self.arena.current()][p.arm]
                    .input_seat(feed.id)
                    .ok_or_else(|| Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds guest input {}, and this load carved no rectangle \
                             for it",
                            feed.lane, feed.id
                        ),
                    })?
            };
            let (ring, committed, cell_bytes, cell_dtype) =
                self.programs.feed_cell(feed.instance, feed.channel)?;
            let element = cell_dtype;
            if element == feed.dtype {
                if cell_bytes < feed.bytes {
                    return Err(Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds guest input {} needing {} bytes off channel {}, \
                             whose cell holds {cell_bytes}",
                            feed.lane, feed.id, feed.bytes, feed.channel
                        ),
                    });
                }
                frame.copy(
                    ring.slab(),
                    committed,
                    store.slab(),
                    at + feed.at,
                    feed.bytes,
                )?;
                continue;
            }
            if element != Dtype::F32 || feed.dtype != Dtype::Bf16 {
                return Err(Fault::Program {
                    at: "serve::feed_inputs",
                    why: format!(
                        "lane {} feeds {:?} guest input {} off channel {}, whose cell is \
                         {element:?}; a cell rides the input's element or arrives f32 and is \
                         cast",
                        feed.lane, feed.dtype, feed.id, feed.channel
                    ),
                });
            }
            let values = feed.bytes / 2;
            if cell_bytes < values * 4 {
                return Err(Fault::Program {
                    at: "serve::feed_inputs",
                    why: format!(
                        "lane {} feeds guest input {} of {values} value(s) off channel {}, \
                         whose f32 cell holds {cell_bytes} bytes",
                        feed.lane, feed.id, feed.channel
                    ),
                });
            }
            casts.push((
                self.handles.bind(&ring, committed, values * 4)?,
                self.handles.bind(store, at + feed.at, feed.bytes)?,
                u32::try_from(values).unwrap_or(u32::MAX),
            ));
        }
        #[cfg(target_vendor = "apple")]
        frame.next_pass()?;
        if !casts.is_empty() {
            let sink = Sink::new(&self.device, frame, &self.pipelines, &self.handles);
            for (source, into, values) in casts {
                kernels_metal::elemwise::pointwise::cast_f32_to_bf16(
                    &sink,
                    kernels_metal::Tensor::new(source, values, 1, Dtype::F32),
                    kernels_metal::Tensor::new(into, values, 1, Dtype::Bf16),
                )
                .map_err(crate::error::kernel)?;
            }
        }
        Ok(())
    }

    pub(super) fn land_merges(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.merge_lands.is_empty() {
            return Ok(());
        }
        for land in &p.merge_lands {
            let column = p.slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a merge over a guest input, which the carve gave no rectangle",
                    land.merge.0
                ),
            })?;
            let row_bytes = land.seat.row_bytes();
            let bytes = u64::from(land.rows) * row_bytes;
            let (into, into_at) = {
                let row = self.handles.get(column.buf).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, a merged column's, which this fire minted no row for",
                        column.buf
                    ),
                })?;
                (
                    row.slab().clone(),
                    row.offset() + u64::from(land.first) * row_bytes,
                )
            };
            if !land.fed {
                frame.fill_zero(&into, into_at, bytes)?;
                continue;
            }
            let (store, at) = self.inputs[self.arena.current()][p.arm]
                .input_seat(land.seat.id)
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "guest input {}'s rectangle, which this load carved none of",
                        land.seat.id
                    ),
                })?;
            frame.copy(
                store.slab(),
                at + u64::from(land.first) * row_bytes,
                &into,
                into_at,
                bytes,
            )?;
        }
        #[cfg(target_vendor = "apple")]
        frame.next_pass()?;
        Ok(())
    }

    pub(super) fn walk_once(&self, p: &Prepared<'_>) -> Result<Walked> {
        self.walk(p, Form::Fire)
    }

    /// The forward encoded; a dry one for a frame that is dropped uncommitted
    /// (the load's pipeline walk): no early head is flushed to run, and a
    /// split's fences and Neural Engine requests are left out, so nothing of
    /// it reaches a queue.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    pub(super) fn walk(&self, p: &Prepared<'_>, form: Form) -> Result<Walked> {
        let dry = form != Form::Fire;
        let place = At::new();
        let mut frame = self.device.frame()?;
        if let Some(plane) = self.nan_flags.as_ref() {
            frame.fill_zero(plane.slab(), 0, plane.bytes())?;
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        if !dry && !p.zeroed.is_empty() {
            for record in &p.zeroed {
                self.pools.clear_into(&mut frame, record)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        // The windowed boundary pages' fills, before anything reads them (a
        // dry walk encodes for no fire; a fill again is the same bytes).
        if !dry && !self.window_fills.is_empty() {
            for (k, fills) in &self.window_fills {
                self.pools.fold_window_rows(&mut frame, *k, fills)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        self.feed_inputs(&mut frame, p)?;
        self.feed_tokens(&mut frame, p)?;
        self.feed_positions(&mut frame, p)?;
        self.land_merges(&mut frame, p)?;
        if !dry && let Some(beam) = p.beam {
            // The beam groups' head step: reorder and
            // resolve, then the forks' tail copies, ahead of every append
            // and read of the fire, in its own command buffer.
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
            {
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::beam_step(
                    &sink,
                    beam.groups,
                    beam.group_count,
                    beam.widest,
                )
                .map_err(crate::error::kernel)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
            {
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::beam_copy_pages(
                    &sink,
                    beam.copies,
                    beam.planes,
                    beam.copy_count,
                    beam.plane_count,
                )
                .map_err(crate::error::kernel)?;
            }
            // A hybrid's rs copies, after the kv
            // copies and ahead of the fire's first append or read.
            if beam.rs_copy_count > 0 {
                #[cfg(target_vendor = "apple")]
                frame.next_pass()?;
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::copy_rs_rows(
                    &sink,
                    beam.rs_copies,
                    beam.planes,
                    beam.rs_copy_count,
                    beam.plane_count,
                )
                .map_err(crate::error::kernel)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            #[cfg(target_vendor = "apple")]
            let sink = sink
                .with_handoff(
                    self.ane
                        .as_ref()
                        .filter(|_| !dry)
                        .map(crate::ane::Ane::handoff),
                )
                .dry(dry)
                .flushing_after(if dry { 0 } else { HEAD_LAUNCHES });
            let mut run = Run::new(
                &sink,
                &self.handles,
                &self.trace.values,
                &self.trace.nodes,
                self.weights.table(),
                &p.slots,
                &p.caches,
                p.bindings.clone(),
                &p.windows,
                &place,
                &self.scratch,
                &self.tuning,
            )
            .with_ane(self.ane.as_ref().filter(|_| form != Form::DryAlone), dry)
            .with_regions(self.compiled.template());
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
        }

        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked { frame, launches })
    }

    pub(super) fn advance(&mut self, p: &Prepared<'_>) {
        for (seat, table) in p.seats.iter().zip(&p.tables) {
            if table.is_empty()
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
    }

    /// The fire's attached instances: each one's envelope and the lanes it
    /// describes, refused by name where two claim a lane, its ports run past
    /// the fire, or a beam group's parents and state disagree with its lanes.
    fn resolve_attachments(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<Resolved> {
        let mut resolved: Vec<engine::ports::Envelope> = Vec::new();
        // Each resolved envelope's instance, by the same index.
        let mut resolved_of: Vec<engine::InstanceId> = Vec::new();
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        // A decode envelope's beam groups (`crate::beam`): each one's
        // parents cell, state and lanes; and each lane's group.
        let mut beam_groups: Vec<crate::beam::Group> = Vec::new();
        let mut group_of: Vec<Option<usize>> = vec![None; lanes.len()];
        if !attachments.is_empty() {
            let instances: Vec<engine::InstanceId> =
                attachments.iter().map(|a| a.instance).collect();
            self.fence_instances(&instances, true)?;
            for attached in attachments {
                let Some(mut envelope) = self.programs.envelope(attached.instance)? else {
                    // A host-class instance has no device envelope, and a beam
                    // step remaps the device-placed geometry its envelope
                    // lays: one that states parents would run its beams as
                    // plain lanes on one working set, each beam's rows
                    // written over another's. Refused by name, never run.
                    if self.programs.parents_cell(attached.instance)?.is_some() {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents (a beam group) and its geometry \
                                 is host-class: a beam step remaps device-placed geometry \
                                 only, so its beams would run as plain lanes on one working \
                                 set; put the pass's tokens and positions from the device \
                                 (an epilogue's put), as a beam search does",
                                attached.instance
                            ),
                        ));
                    }
                    continue;
                };
                let first = attached.lane as usize;
                envelope.host_rows = lanes
                    .iter()
                    .skip(first)
                    .take(attached.lanes.max(1) as usize)
                    .map(|seated| seated.lane.tokens.len() as u32)
                    .collect();
                let carried = envelope.lanes();
                // Its ports describe its own lanes only: past them they
                // would set another request's positions and kv slots.
                let owned = attached.lanes.max(1) as usize;
                if carried > owned {
                    return Err(Fault::GeometryLanes {
                        lane: attached.lane,
                        instance: attached.instance.0,
                        carried: u32::try_from(carried).unwrap_or(u32::MAX),
                        owned: owned as u32,
                    });
                }
                if first + carried > lanes.len() {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "instance {} is attached at lane {first} and its \
                             `embed_indptr` port describes {carried} lane(s), which runs \
                             past the {} this fire carries; its descriptor ports have no \
                             rows to describe",
                            attached.instance,
                            lanes.len()
                        ),
                    ));
                }
                let held = resolved.len();
                for lane in 0..carried {
                    if envelope_of[first + lane].is_some() {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {} is claimed by two attached instances, the second \
                                 being {}; a lane's descriptor ports have one author, and \
                                 two would decide the same rows twice",
                                first + lane,
                                attached.instance
                            ),
                        ));
                    }
                    envelope_of[first + lane] = Some((held, lane));
                }
                if let Some((cell, at)) = self.programs.parents_cell(attached.instance)? {
                    let Some((state, layout)) = self.programs.beam_state(attached.instance)? else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents and holds no beam state",
                                attached.instance
                            ),
                        ));
                    };
                    let parents = cell.address_at(at).ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {}'s parents cell has no address",
                                attached.instance
                            ),
                        )
                    })?;
                    if carried as u32 != layout.lanes {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents for {} beam(s) and runs {carried}                                  lane(s); every beam runs every fire",
                                attached.instance, layout.lanes
                            ),
                        ));
                    }
                    for lane in 0..carried {
                        group_of[first + lane] = Some(beam_groups.len());
                    }
                    beam_groups.push(crate::beam::Group {
                        instance: attached.instance,
                        parents,
                        state,
                        layout,
                        lanes: Vec::new(),
                    });
                }
                resolved.push(envelope);
                resolved_of.push(attached.instance);
            }
        }
        Ok(Resolved {
            resolved,
            resolved_of,
            envelope_of,
            beam_groups,
            group_of,
        })
    }

    /// The pages, writes and masks each attached lane's device ports state,
    /// translated through the lane's table, and each lane's rows.
    fn device_ports(
        &self,
        lanes: &[Seated<'_>],
        resolved: &[engine::ports::Envelope],
        envelope_of: &[Option<(usize, usize)>],
    ) -> Result<DevicePorts> {
        let page_tokens = self.pools.paging().page_size.max(1);
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; lanes.len()];
        let mut device_masks: Vec<Option<Masking>> = vec![None; lanes.len()];
        let mut lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();
        for source in 0..lanes.len() {
            let Some((held, at)) = envelope_of[source] else {
                continue;
            };
            let ports = resolved[held].lane(at, source)?;
            let table = lanes[source].translation;
            let translate = |page: u32, port: &str| -> Result<u32> {
                table.get(page as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source}'s `{port}` port names working-set page {page} \
                             and the table this fire was handed maps {} page(s); a guest \
                             holds relative indexes and the pool's ids are the runtime's, \
                             so an index past the table addresses somebody else's cache",
                            table.len()
                        ),
                    )
                })
            };
            if ports.owns_slots() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            if let (Some(slots), Some(reach)) = (ports.slots(rows)?, ports.reach(rows)?) {
                device_pages[source] = Some(
                    (0..reach.div_ceil(page_tokens))
                        .map(|page| translate(page, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                );
                device_writes[source] = Some((
                    slots
                        .iter()
                        .map(|&slot| translate(slot / page_tokens, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                    slots.iter().map(|&slot| slot % page_tokens).collect(),
                ));
            }
            if let Some((cells, stride)) = ports.mask(rows)? {
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane that \
                             states its own slots has no such order this shell can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(engine::ports::from_dense(cells, stride));
            }
        }
        Ok(DevicePorts {
            device_pages,
            device_writes,
            device_masks,
            lane_rows,
        })
    }

    /// Each lane's media row, checked against the load's patch seat and the
    /// lane's rows.
    fn media_rows<'m>(
        &self,
        media: &'m [Media<'m>],
        lane_rows: &[u32],
    ) -> Result<Vec<Option<&'m Media<'m>>>> {
        let lanes_len = lane_rows.len();
        let mut media_of: Vec<Option<&'m Media<'m>>> = vec![None; lanes_len];
        for shot in media {
            let Some(seat) = self.patch_seat else {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless { lane: shot.lane },
                )));
            };
            let at = shot.lane as usize;
            if at >= lanes_len {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "a media row names lane {} of the {} this fire has",
                        shot.lane, lanes_len
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {} carries two media rows; one lane's spans are one \
                         concatenation with one payload order",
                        shot.lane
                    ),
                ));
            }
            let patch_rows = shot.rows.iter().copied().fold(0u32, u32::saturating_add) as u64;
            let rows_here = u64::from(lane_rows[at]);
            let owed = patch_rows.saturating_mul(seat.row_bytes);
            for (what, have, want) in [
                ("payload bytes", shot.patches.len() as u64, owed),
                ("routes", shot.routes.len() as u64, patch_rows),
                (
                    "grid positions",
                    shot.positions.len() as u64,
                    patch_rows * 3,
                ),
                (
                    "image grids",
                    shot.grids.len() as u64,
                    shot.rows.len() as u64 * 3,
                ),
            ] {
                if have != want {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        what,
                        have,
                        want,
                    });
                }
            }
            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64 != rows_here * 3
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    what: "trunk rotation triples",
                    have: shot.token_positions.len() as u64,
                    want: rows_here * 3,
                });
            }
            let drop = self.drops_patch_rows;
            let rows_here32 = lane_rows[at];
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows_here32)
            }) {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows: rows_here32,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }
        Ok(media_of)
    }

    /// Each lane's clip record, checked against the load's voxel seat.
    fn clip_rows<'c>(
        &self,
        clips: &'c [Clips<'c>],
        lanes_len: usize,
    ) -> Result<Vec<Option<&'c Clips<'c>>>> {
        let mut clips_of: Vec<Option<&'c Clips<'c>>> = vec![None; lanes_len];
        for tile in clips {
            if self.voxel_seat.is_none() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} submits VAE clips and this load seats no voxel row",
                        tile.lane
                    ),
                });
            }
            let at = tile.lane as usize;
            if at >= lanes_len {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "a clip submission names lane {} of the {} this fire has",
                        tile.lane, lanes_len
                    ),
                });
            }
            if tile.boxes.is_empty() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} carries a voxel row naming no clips; a lane with no clip \
                         constructs no voxel row at all",
                        tile.lane
                    ),
                });
            }
            if clips_of[at].is_some() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} submits two clip records; a lane's clips are one \
                         concatenation with one payload order",
                        tile.lane
                    ),
                });
            }
            clips_of[at] = Some(tile);
        }
        Ok(clips_of)
    }

    /// One lane row of a fire, staged into the accumulators the fire builds
    /// ([`LaneAccum`]): its seat and page tables, tokens and positions, record
    /// bands, rs plan, mask, adapter routes and writes.
    fn stage_lane<'i, 'a: 'i>(
        &mut self,
        row: &poem_exec::fire::LaneRow,
        inputs: &LaneInputs<'i, 'a>,
        acc: &mut LaneAccum<'i, 'a>,
    ) -> Result<()> {
        let LaneInputs {
            lanes,
            envelope_of,
            resolved,
            resolved_of,
            device_pages,
            device_writes,
            device_masks,
            group_of,
            beam_to,
            widths,
            any_adapter,
            ..
        } = *inputs;
        let LaneAccum {
            beam_groups,
            beam_folded,
            seats,
            kv_less,
            tables,
            window_tables,
            tokens,
            token_feeds,
            position_feeds,
            positions,
            rotations,
            stated_rows,
            lane_bands,
            row_bands,
            state_top,
            state_ids,
            rs_plans,
            rs_one_run,
            rs_sets,
            rs_own,
            ends,
            rs_active,
            snap_bands,
            snap_slots,
            request_of_token,
            masks,
            adapter_routes,
            beginning,
            writes,
            extents,
            group_folds,
            rs_skips,
            ..
        } = acc;
        let source = row.source as usize;
        let seated = &lanes[source];
        let lane = &seated.lane;
        let ports = match envelope_of[source] {
            Some((held, at)) => Some(resolved[held].lane(at, source)?),
            None => None,
        };
        let have = match ports.as_ref().filter(|ports| ports.owns_slots()) {
            Some(ports) => {
                let after = ports.reach(row.rows as usize)?.ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} states where its rows land and reaches no \
                             slot; the page count, the last page's fill and the \
                             attention schedules are all carved from what it reads"
                        ),
                    )
                })?;
                if after < row.rows {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} reads a KV extent of {after} and this fire \
                             writes {} row(s) into it; the extent is AFTER the append, \
                             so it can never be shorter than what the append adds",
                            row.rows
                        ),
                    ));
                }
                after - row.rows
            }
            None => match seated.held {
                Some(held) => held,
                None => self
                    .held
                    .get(lane.slot as usize)
                    .copied()
                    .ok_or(Fault::Ceiling {
                        what: "slots",
                        need: u64::from(lane.slot) + 1,
                        have: self.held.len() as u64,
                    })?,
            },
        };
        debug_assert_eq!(
            row.row_offset as usize,
            tokens.len(),
            "a lane's rows stand where the composition placed them"
        );
        let fresh = match seated.rs_reset {
            engine::fire::RsReset::Inferred => have == 0,
            engine::fire::RsReset::Fresh => true,
            engine::fire::RsReset::Held => false,
        };
        let record = self.record_of(seated)?;
        for (k, lists) in window_tables.iter_mut().enumerate() {
            lists.push(if seated.kv_less {
                Vec::new()
            } else if self.pools.ring().is_none() {
                // Windowed layers are a lane's ring and window buffer,
                // on every load that has them.
                return Err(Fault::program(
                    "serve::windowed",
                    "a load with window lists and no rings: Metal serves windowed layers \
                     as ring + buffer only"
                        .to_string(),
                ));
            } else if let Some(fold) = seated.window {
                let port_fold = ports.as_ref().and_then(engine::ports::LanePorts::fold);
                self.windowed_pages(k, seated, fold, port_fold, have, row.rows)?
            } else if seated.ring.is_some() {
                // A lane whose record holds rings states how it folds
                // them; its buffer slots are no whole window.
                return Err(Fault::program(
                    "serve::windowed",
                    format!(
                        "lane in slot {} of a load whose windowed layers are ring + buffer \
                         states no window fold (`Lane::window`)",
                        lane.slot
                    ),
                ));
            } else if !seated.refs {
                // A seat-addressed lane (a caller firing directly, the
                // tuner): its seat's ring, every row it writes folded in
                // the fire, no buffer.
                let page_size = self.pools.window_paging(k).page_size.max(1);
                let after = have.saturating_add(row.rows);
                let fold = engine::fire::WindowFold {
                    base: after / page_size * page_size,
                    fold: engine::fire::FoldLen::Host(row.rows),
                    lineage: engine::fire::Lineage::Set,
                };
                let seat = Seated {
                    folded: have,
                    windows: &[],
                    ..*seated
                };
                self.windowed_pages(k, &seat, fold, None, have, row.rows)?
            } else if !self.warming {
                return Err(Fault::program(
                    "serve::windowed",
                    format!(
                        "lane in slot {} of a load whose windowed layers are ring + buffer \
                         states no ring pieces and no window fold (`Lane::window`)",
                        lane.slot
                    ),
                ));
            } else {
                // The pipeline walk drops its fires: nothing reads what
                // one computes, so every page it names is one scratch
                // page, in the own-row region's last chunk (past every
                // arm's share).
                let page_size = self.pools.window_paging(k).page_size.max(1);
                let (first, chunks) = self.own_top;
                if chunks == 0 {
                    return Err(Fault::program(
                        "serve::windowed",
                        "a ring + buffer load with no own-row region for a fold-less lane"
                            .to_string(),
                    ));
                }
                let (_, id) = self.pools.window_slot_at(k, first + chunks - 1, 0)?;
                let after = have.saturating_add(row.rows);
                vec![id; after.div_ceil(page_size).max(1) as usize]
            });
        }
        *state_top = record
            .iter()
            .map(|&id| id.saturating_add(1))
            .fold(*state_top, u32::max);
        state_ids.extend_from_slice(&record);
        if fresh {
            beginning.push(record.clone());
        }
        if let Some(group) = group_of[source] {
            beam_groups[group].lanes.push(crate::beam::GroupLane {
                source,
                seat: seats.len(),
                row_offset: row.row_offset,
                have,
                rows: row.rows,
            });
        }
        seats.push(Seat {
            slot: lane.slot,
            have,
            rows: row.rows,
        });
        kv_less.push(seated.kv_less);
        tables.push(match &device_pages[source] {
            Some(pages) => std::borrow::Cow::Owned(self.pools.kv_pages(pages, None)?),
            None if seated.kv_less => std::borrow::Cow::Borrowed(seated.pages),
            None => {
                let after = u64::from(have) + u64::from(row.rows);
                let page_size = u64::from(self.pools.paging().page_size.max(1));
                let pages = u32::try_from(after.div_ceil(page_size).max(1)).unwrap_or(u32::MAX);
                std::borrow::Cow::Owned(
                    self.pools
                        .kv_pages(seated.pages, Some((lane.slot, pages)))?,
                )
            }
        });
        // A kv-less lane's mask goes to its group-packed attention, not the paged prefill.
        let masking = if seated.kv_less {
            None
        } else {
            device_masks[source].as_ref().or(seated.mask)
        };
        let runs_masked_arm = self.masked.contains(row.class as usize);
        if masking.is_some() && self.masked.is_empty() {
            return Err(Fault::Maskless { lane: row.source });
        }
        // The mask is per-lane data: a lane stating none reads every key
        // in a masked arm too. One stated where no arm applies it is refused.
        if masking.is_some() && !runs_masked_arm {
            return Err(Fault::MaskUnread {
                lane: row.source,
                word: lane.word,
            });
        }
        masks.push(crate::mask::LaneMask {
            mask: masking,
            have,
            folded: seated.folded.min(have),
            rows: row.rows,
            wide: self.wide.contains(row.class as usize),
        });
        let runs_correction = self.corrected.contains(row.class as usize);
        if seated.adapter.is_some() && self.corrected.is_empty() {
            return Err(Fault::Adapterless { lane: row.source });
        }
        let unreachable = seated.adapter.is_some()
            && !runs_correction
            && !self
                .compiled
                .classes
                .correction_reaches(&self.corrected, lane.word);
        if seated.adapter.is_some() != runs_correction && !unreachable {
            return Err(Fault::AdapterWord {
                lane: row.source,
                word: lane.word,
                runs_correction,
            });
        }
        if any_adapter {
            let id = seated
                .adapter
                .map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
            adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
        }
        // This lane's index: its seat was pushed above.
        let at_lane = seats.len() as i32 - 1;
        {
            const MAX: usize = kernels_metal::ssm::RS_SNAPSHOTS as usize;
            let snaps = seated.rs_snapshots;
            if snaps.len() > MAX
                || (!snaps.is_empty() && row.rows as usize != lane.tokens.len())
                || snaps
                    .iter()
                    .any(|snap| snap.row == 0 || snap.row >= row.rows)
            {
                return Err(Fault::program(
                    "serve::rs",
                    format!(
                        "lane {source} snapshots its recurrent state at rows {:?} and runs \
                         {} of its {} rows in this fire; a fire lands at most {MAX} \
                         interior rows of a lane it runs whole",
                        snaps.iter().map(|snap| snap.row).collect::<Vec<_>>(),
                        row.rows,
                        lane.tokens.len()
                    ),
                ));
            }
            if !snaps.is_empty() && snap_bands.is_empty() {
                *snap_bands = widths
                    .iter()
                    .map(|&w| vec![0i32; at_lane as usize * MAX * (1 + w as usize)])
                    .collect();
            }
            // Lane 0 leaves the tables empty above, so emptiness cannot
            // stand for "no lane has snapshots".
            if !snaps.is_empty() || !snap_bands.is_empty() {
                // Each snapshot's record, a physical id a piece
                // (`RsSnapshot::pieces`, its slot the first; a seat's
                // record where the lane is seat-addressed).
                let mut records = Vec::with_capacity(snaps.len());
                for snap in snaps {
                    let record = if !seated.refs {
                        self.pools.record(None, snap.slot)?
                    } else if snap.pieces.is_empty() {
                        vec![self.pools.state_slot(snap.slot, false)?]
                    } else {
                        self.pools.record(Some(&snap.pieces), lane.slot)?
                    };
                    *snap_slots = record
                        .iter()
                        .map(|&id| id.saturating_add(1))
                        .fold(*snap_slots, u32::max);
                    state_ids.extend_from_slice(&record);
                    records.push(self.pools.band_ids(&record));
                }
                let mut at = 0usize;
                for (band, &w) in snap_bands.iter_mut().zip(widths) {
                    let w = w as usize;
                    let mut entry = vec![0i32; MAX * (1 + w)];
                    for (n, (snap, ids)) in snaps.iter().zip(&records).enumerate() {
                        entry[n] = narrow(u64::from(snap.row));
                        entry[MAX + n * w..MAX + (n + 1) * w].copy_from_slice(&ids[at..at + w]);
                    }
                    band.extend(entry);
                    at += w;
                }
            }
        }
        if !matches!(seated.rs, engine::fire::RsVerb::Fold) {
            if self.rs_layout.is_none() {
                return Err(Fault::program(
                    "serve::rs",
                    format!(
                        "lane {source} asks a recurrent verb of a plan that declares no \
                         recurrent state to buffer"
                    ),
                ));
            }
            *rs_active = true;
        }
        let stated = match &ports {
            Some(ports) => ports.slots(row.rows as usize)?,
            None => None,
        };
        let placed = stated.is_some();
        let first = stated
            .and_then(|slots| slots.first().copied())
            .unwrap_or(have);
        // The buffer's rows live in the rs list: the verb's pages, in
        // token order from its base.
        // Where the record stands: the stated frozen length, or past it
        // where this engine's last fold of the record froze more than
        // the runtime has yet been told.
        let held_key = record.first().copied();
        if fresh && let Some(key) = held_key {
            self.frozen_ahead.remove(&key);
        }
        let rs = match seated.rs {
            engine::fire::RsVerb::Buffer {
                pages,
                base,
                folded,
                fold,
                lineage,
            } => {
                // A beam group's set is folded by the runtime alone
                // (the group fold): its lanes read the length it states, and an
                // entry left on the record's slot by an earlier set
                // (a fold writes a fresh record) is dropped.
                let beam_lane = group_of[source].is_some();
                if beam_lane && let Some(key) = held_key {
                    self.frozen_ahead.remove(&key);
                }
                let ahead = held_key
                    .filter(|_| !beam_lane)
                    .and_then(|key| self.frozen_ahead.get(&key).copied());
                let folded = match ahead {
                    Some(ahead) if ahead > *folded => ahead,
                    _ => {
                        if let Some(key) = held_key {
                            self.frozen_ahead.remove(&key);
                        }
                        *folded
                    }
                };
                engine::fire::RsVerb::Buffer {
                    pages: self
                        .pools
                        .rs_pages(pages, (!seated.refs).then_some(lane.slot))?,
                    base: *base,
                    folded,
                    fold: *fold,
                    lineage: *lineage,
                }
            }
            verb => verb.clone(),
        };
        let plan = engine::rs::LanePlan::of(
            &rs,
            row.rows,
            first,
            masking,
            ports.as_ref().and_then(engine::ports::LanePorts::fold),
        )
        .map_err(|error| Fault::program("serve::rs", format!("lane {source}: {error}")))?;
        rs_one_run.push(match &rs {
            engine::fire::RsVerb::Buffer {
                folded, lineage, ..
            } => engine::fire::lane_one_run(masking, *folded, first, *lineage),
            _ => true,
        });
        // A beam group's lanes share their set's state: what its lanes
        // fold is folded once for all of them, from the first's buffered
        // rows, before any reads it (the group fold). Each lane then replays only
        // the rows past it, from its page table row started past them,
        // and folds none itself.
        let beam_group =
            group_of[source].filter(|_| matches!(rs, engine::fire::RsVerb::Buffer { .. }));
        let mut plan = plan;
        let (skip, beam_fold_to) = match (beam_group, &rs) {
            (Some(group), engine::fire::RsVerb::Buffer { folded, .. }) => {
                let to = beam_to[group].unwrap_or(*folded).min(first).max(*folded);
                let skip = to - folded;
                if beam_to[group].is_some_and(|stated| stated > first) {
                    tracing::warn!(
                        instance = %beam_groups[group].instance,
                        lane = source,
                        folded = *folded,
                        stated = beam_to[group].unwrap_or(0),
                        first,
                        "a beam group's fold is stated past its lanes' first row (the group fold)"
                    );
                }
                if skip > 0 && plan.gathers.len() != 1 {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {source} of a beam group folds {skip} shared row(s) before \
                             its step and replays {} run(s) of its buffer; a group fold \
                             reads one run from the set's fold",
                            plan.gathers.len()
                        ),
                    ));
                }
                plan.replay = plan.replay.saturating_sub(skip);
                plan.commit = 0;
                if skip > 0 && !beam_folded[group] {
                    beam_folded[group] = true;
                    group_folds.push(rs_plans.len() as i32);
                    group_folds.push(skip as i32);
                }
                (skip, Some(to))
            }
            _ => (0, None),
        };
        rs_skips.push(skip);
        // A reporting lane that buffers tells, where the fire settles,
        // how many slots past its state's frozen length this fire froze
        // (as `LanePlan::of` reckons it) and, where a port places its
        // rows, one past the last slot they landed at.
        if seated.reports != 0 {
            let end = first.saturating_add(row.rows);
            let fold = match &rs {
                engine::fire::RsVerb::Buffer { folded, fold, .. } => {
                    let fold = match fold {
                        engine::fire::FoldLen::Host(n) => Some(*n),
                        engine::fire::FoldLen::Device => {
                            ports.as_ref().and_then(engine::ports::LanePorts::fold)
                        }
                    };
                    // A beam lane folds nothing itself: its group's fold
                    // is the runtime's, applied to the set (the group fold).
                    if beam_fold_to.is_some() {
                        Some(0)
                    } else {
                        fold.map(|n| folded.saturating_add(n).min(end) - folded)
                    }
                }
                // A lane with no recurrent state to buffer (a kv-only
                // model's): its fold is what its program's `fold` port
                // states for it, the count the runtime adds to the set's
                // frozen length when the fire settles, as for a buffering
                // lane. Unreported, the set's cursor never comes to rest
                // and a fork or a save of it is refused.
                _ => ports.as_ref().and_then(engine::ports::LanePorts::fold),
            };
            ends.push(engine::KvWrites {
                tag: seated.reports,
                fold,
                end: placed.then_some(end),
                ..engine::KvWrites::default()
            });
        }
        if let (engine::fire::RsVerb::Buffer { folded, fold, .. }, Some(key)) = (&rs, held_key) {
            let n = match fold {
                engine::fire::FoldLen::Host(n) => Some(*n),
                engine::fire::FoldLen::Device => {
                    ports.as_ref().and_then(engine::ports::LanePorts::fold)
                }
            };
            // A beam group's set is folded by the runtime as it prepares
            // the fire (the group fold): nothing is frozen ahead of what it states.
            let to = n.map_or(*folded, |n| {
                folded.saturating_add(n).min(first.saturating_add(row.rows))
            });
            if beam_fold_to.is_none() && to > *folded {
                self.frozen_ahead.insert(key, to);
            }
        }
        if !matches!(seated.rs, engine::fire::RsVerb::Fold) && crate::diag::on().rs_trace {
            eprintln!(
                "recurrent seat: lane {} rows {} replay {} commit {} gather {:?} scatter {:?}",
                row.source, row.rows, plan.replay, plan.commit, plan.gather, plan.scatter
            );
        }
        // The state the lane's kernels read and fold: the record it
        // writes, but a lane of a copy-on-write group that folds nothing
        // reads the run side, which no other lane writes this fire;
        // its buffered rows still land in `record`.
        // A beam group's lanes read the record its fold, if any, has
        // written before them (the group fold).
        let state = match seated.record_run {
            Some(run) if plan.commit == 0 && beam_fold_to.is_none() => {
                self.pools.record(Some(run), lane.slot)?
            }
            _ => record.clone(),
        };
        *state_top = state
            .iter()
            .map(|&id| id.saturating_add(1))
            .fold(*state_top, u32::max);
        state_ids.extend_from_slice(&state);
        let band_ids = self.pools.band_ids(&state);
        {
            let mut at = 0usize;
            for ((lanes_band, rows_band), &width) in
                lane_bands.iter_mut().zip(row_bands.iter_mut()).zip(widths)
            {
                let ids = &band_ids[at..at + width as usize];
                lanes_band.extend_from_slice(ids);
                for _ in 0..row.rows {
                    rows_band.extend_from_slice(ids);
                }
                at += width as usize;
            }
        }
        rs_sets.push(match &rs {
            engine::fire::RsVerb::Buffer { pages, base, .. } => Some((*base, pages.clone())),
            _ => None,
        });
        rs_own.push(row.rows);
        rs_plans.push(plan);
        if !seated.positions.is_empty() && seated.positions.len() != lane.tokens.len() {
            return Err(Fault::Positions {
                lane: row.source,
                stated: seated.positions.len() as u64,
                rows: lane.tokens.len() as u64,
            });
        }
        let rows_here = row.rows as usize;
        match ports.as_ref() {
            Some(ports) => {
                ports.check_extent(have.saturating_add(row.rows))?;
                // A pass of its instance still in flight puts the token
                // this one reads: copied from that cell on the device
                // before the forward, not read on the host.
                let in_flight = match envelope_of[source] {
                    Some((held, at)) => resolved_of
                        .get(held)
                        .and_then(|&instance| self.programs.token_cell(instance).transpose())
                        .transpose()?
                        .map(|(buffer, cell)| (buffer, cell + 4 * at as u64)),
                    None => None,
                };
                if let Some((buffer, cell)) = in_flight {
                    token_feeds.push((tokens.len() as u32, buffer, cell));
                }
                for &token in ports.tokens_for(rows_here)? {
                    tokens.push(token as i32);
                }
                let rows: Vec<i32> = (0..rows_here)
                    .map(|at| narrow(u64::from(have) + at as u64))
                    .collect();
                // Positions its program computes in a channel: copied
                // from the cell on the device, never read here.
                let position_cell = match envelope_of[source] {
                    Some((held, at)) => resolved_of
                        .get(held)
                        .and_then(|&instance| self.programs.positions_cell(instance).transpose())
                        .transpose()?
                        .map(|(buffer, cell)| (buffer, cell + 4 * at as u64)),
                    None => None,
                };
                if position_cell.is_some() && rows_here != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} carries its positions on the device and {rows_here} \
                             rows; a decode envelope's lane states one position a row, one row"
                        ),
                    ));
                }
                let stated: Vec<i32> = match position_cell
                    .is_none()
                    .then(|| ports.positions_for(have, rows_here))
                    .transpose()?
                    .flatten()
                {
                    Some(stated) => stated.iter().map(|&at| narrow(u64::from(at))).collect(),
                    None if !seated.positions.is_empty() => seated
                        .positions
                        .iter()
                        .map(|&at| narrow(u64::from(at)))
                        .collect(),
                    None => rows.clone(),
                };
                // The causal bound is the row order, the cache row; a
                // stated position rotates the row and bounds nothing.
                positions.extend(&rows);
                if let Some((buffer, cell)) = position_cell {
                    position_feeds.push((rotations.len() as u32, buffer, cell));
                }
                rotations.extend(if self.states_mrope { &rows } else { &stated });
                stated_rows.extend(&stated);
                match &device_writes[source] {
                    Some((slots, offsets)) => {
                        for (&page, &off) in slots.iter().zip(offsets) {
                            let page = self.pools.kv_page(page)?;
                            writes.push(Some((narrow(u64::from(page)), narrow(u64::from(off)))));
                        }
                    }
                    None => writes.extend(std::iter::repeat_n(None, rows_here)),
                }
                for _ in 0..rows_here {
                    request_of_token.push(at_lane);
                }
            }
            None => {
                for (at, token) in lane.tokens.iter().enumerate() {
                    tokens.push(*token as i32);
                    let row = narrow(u64::from(have) + at as u64);
                    let stated = seated
                        .positions
                        .get(at)
                        .map_or(row, |&p| narrow(u64::from(p)));
                    // The causal bound is the row order, the cache row; a
                    // stated position rotates the row and bounds nothing.
                    // With mrope the rotation rides its own table.
                    positions.push(row);
                    rotations.push(if self.states_mrope { row } else { stated });
                    stated_rows.push(stated);
                    request_of_token.push(at_lane);
                }
                writes.extend(std::iter::repeat_n(None, rows_here));
            }
        }
        if !seated.kv_less {
            let reach = have.saturating_add(row.rows);
            let set = (seated.set, seated.readable);
            match ports
                .as_ref()
                .map(|ports| ports.slots(rows_here))
                .transpose()?
            {
                Some(Some(slots)) => extents.lane(
                    &self.written,
                    source,
                    lane.slot,
                    set,
                    slots.iter().map(|&slot| slot..slot.saturating_add(1)),
                    reach,
                ),
                _ => extents.lane(
                    &self.written,
                    source,
                    lane.slot,
                    set,
                    std::iter::once(have..reach),
                    reach,
                ),
            }
        }
        Ok(())
    }
}

/// What [`Shell::resolve_attachments`] finds of a fire's attached instances.
struct Resolved {
    resolved: Vec<engine::ports::Envelope>,
    /// Each resolved envelope's instance, by the same index.
    resolved_of: Vec<engine::InstanceId>,
    envelope_of: Vec<Option<(usize, usize)>>,
    /// A decode envelope's beam groups (`crate::beam`): each one's parents
    /// cell, state and lanes; and each lane's group.
    beam_groups: Vec<crate::beam::Group>,
    group_of: Vec<Option<usize>>,
}

/// What [`Shell::device_ports`] reads of each lane's device ports.
struct DevicePorts {
    device_pages: Vec<Option<Vec<u32>>>,
    device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>>,
    device_masks: Vec<Option<Masking>>,
    lane_rows: Vec<u32>,
}

/// What [`Shell::stage_lane`] reads of the fire it stages.
#[derive(Clone, Copy)]
struct LaneInputs<'i, 'a> {
    lanes: &'a [Seated<'a>],
    envelope_of: &'i [Option<(usize, usize)>],
    resolved: &'i [engine::ports::Envelope],
    resolved_of: &'i [engine::InstanceId],
    device_pages: &'i [Option<Vec<u32>>],
    device_writes: &'i [Option<(Vec<u32>, Vec<u32>)>],
    device_masks: &'i [Option<Masking>],
    group_of: &'i [Option<usize>],
    beam_to: &'i [Option<u32>],
    widths: &'i [u32],
    any_adapter: bool,
}

/// What a fire's lane rows build, row by row ([`Shell::stage_lane`]).
struct LaneAccum<'i, 'a> {
    seats: Vec<Seat>,
    kv_less: Vec<bool>,
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    window_tables: Vec<Vec<Vec<u32>>>,
    tokens: Vec<i32>,
    token_feeds: Vec<(u32, crate::device::Buffer, u64)>,
    position_feeds: Vec<(u32, crate::device::Buffer, u64)>,
    positions: Vec<i32>,
    rotations: Vec<i32>,
    stated_rows: Vec<i32>,
    lane_bands: Vec<Vec<i32>>,
    row_bands: Vec<Vec<i32>>,
    state_top: u32,
    state_ids: Vec<u32>,
    rs_plans: Vec<engine::rs::LanePlan>,
    rs_one_run: Vec<bool>,
    rs_sets: Vec<Option<(u32, Vec<u32>)>>,
    rs_own: Vec<u32>,
    ends: Vec<engine::KvWrites>,
    rs_active: bool,
    snap_bands: Vec<Vec<i32>>,
    snap_slots: u32,
    request_of_token: Vec<i32>,
    masks: Vec<crate::mask::LaneMask<'i>>,
    adapter_routes: Vec<i32>,
    beginning: Vec<Vec<u32>>,
    writes: Vec<Option<(i32, i32)>>,
    extents: engine::extent::FireExtents,
    beam_groups: Vec<crate::beam::Group>,
    beam_folded: Vec<bool>,
    group_folds: Vec<i32>,
    rs_skips: Vec<u32>,
}
