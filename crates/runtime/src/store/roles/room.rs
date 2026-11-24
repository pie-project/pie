//! Taking room: parts of a role's shape, guest scratch runs, an unpark's
//! pieces and the claims held for it, and giving room back.

use super::*;

impl Roles {
    pub(super) fn shape(&self, role: SlotRole) -> Result<FixedShape, RolesError> {
        self.shapes.of(role).ok_or(RolesError::NoShape(role))
    }

    /// `slots` of `role` on `on`: any free ones, or (`contiguous`) one run
    /// of adjacent ones in a chunk, never split.
    pub(super) fn take(
        &self,
        on: &Shared,
        role: SlotRole,
        slots: u32,
        contiguous: bool,
    ) -> Result<Part, RolesError> {
        let part = |refs| Part {
            role,
            refs,
            contiguous,
            pieces: 0,
            chunks: 0,
            run: 0,
            held: false,
        };
        if slots == 0 {
            return Ok(part(Vec::new()));
        }
        let shape = self.shape(role)?;
        let mut pool = on.lock().unwrap();
        let claim = if Arc::ptr_eq(on, &self.device) { self.claim } else { None };
        if !contiguous {
            return pool
                .claiming(claim, |pool| pool.take_refs(shape.pool, u64::from(slots)))
                .map(part)
                .ok_or(RolesError::Short { role, slots });
        }
        match pool.claiming(claim, |pool| pool.take_run(shape.pool, u64::from(slots))) {
            Ok(refs) => Ok(part(refs)),
            Err(SlotError::RunPastChunk { per_chunk, .. }) => Err(RolesError::PastChunk {
                role,
                slots,
                slot_bytes: shape.slot_bytes,
                per_chunk,
            }),
            Err(_) => Err(RolesError::NoRun { role, slots }),
        }
    }

    /// The dead part `bytes` of guest scratch take, pieces of `run`
    /// chunks each: one run of `Instance` slots in a chunk where a second
    /// state of its size fits the same chunk beside it, else whole runs of
    /// `run` adjacent chunks. A state two of which do not fit one chunk
    /// gains nothing from a run of slots, its chunk holding no other state
    /// of its program, and only fragments: what it leaves of its chunk
    /// takes some smaller item, which keeps the chunk from coming free, and
    /// a run of that many slots needs a chunk all but empty, which a pool
    /// under pressure seldom has though whole chunks free up as processes
    /// leave. In whole chunks it comes back to any free chunk.
    pub(super) fn scratch_part(&self, bytes: u64, run: u32) -> Result<Part, RolesError> {
        let shape = self.shape(SlotRole::Instance)?;
        let slot = shape.slot_bytes.max(1);
        let run = run.max(1);
        let slots = bytes.div_ceil(slot);
        let two_share_a_chunk = slots.saturating_mul(2) <= u64::from(self.chunk_slots);
        let (pieces, slots) = if two_share_a_chunk {
            (0, u32::try_from(slots).unwrap_or(u32::MAX))
        } else {
            let chunk = self.device.lock().unwrap().pool().chunk_bytes().max(1);
            let piece = chunk.saturating_mul(u64::from(run));
            (u32::try_from(bytes.div_ceil(piece)).unwrap_or(u32::MAX), 0)
        };
        Ok(Part {
            role: SlotRole::Instance,
            refs: Vec::new(),
            contiguous: true,
            pieces,
            chunks: if pieces > 0 { run } else { 0 },
            run: slots,
            held: false,
        })
    }

    /// Room on the device for `bytes` of guest scratch (`what`) in pieces
    /// of `run` chunks ([`Self::scratch_part`]); all or none. Where no
    /// free run holds it and the segments still to commit within the
    /// budget would, the pool counts it short ([`GUEST_WAITER`]) and the
    /// bind waits on a commit ([`RolesError::GuestGrow`]); else it is
    /// refused by name, with what it asked and what was free.
    /// Counted short in the pool where `heads`, driving the walk.
    pub(super) fn take_scratch(
        &self,
        bytes: u64,
        run: u32,
        what: impl Fn() -> String,
        heads: bool,
    ) -> Result<Part, RolesError> {
        let part = self.scratch_part(bytes, run)?;
        let shape = self.shape(SlotRole::Instance)?;
        let room = match self.retake(&part) {
            Ok(part) => {
                if heads {
                    let mut device = self.device.lock().unwrap();
                    device.pool_mut().unwait(GUEST_WAITER);
                    device.set_guest_head(None);
                }
                return Ok(part);
            }
            Err(room) => room,
        };
        let (chunks, pieces) = part.chunk_runs();
        let mut device = self.device.lock().unwrap();
        let free = match room {
            GuestRoom::Runs { free_runs, .. } => free_runs,
            GuestRoom::Slots { .. } => 0,
        };
        // Runs a commit could make, only where one can come: at the
        // governor's target the walk commits nothing, and a bind short of
        // room goes to the planner (BindShort) instead of waiting on it.
        let committable = if device.commit_can_come() {
            device.pool().committable_runs(chunks)
        } else {
            0
        };
        let pool = device.pool_mut();
        let room = room.with_committable(committable);
        if free.saturating_add(committable) < pieces {
            if heads {
                pool.unwait(GUEST_WAITER);
                device.set_guest_head(None);
            }
            let short = RolesError::GuestShort {
                what: what(),
                bytes,
                room,
            };
            let state_chunks = u64::from(device.state_chunks_now());
            let asks = u64::from(chunks) * u64::from(pieces);
            return Err(if asks <= state_chunks {
                RolesError::BindShort(Box::new(short))
            } else {
                short
            });
        }
        let slots = u64::from(chunks) * u64::from(pieces) * u64::from(self.chunk_slots);
        let prefix = pool.prefix_segments();
        if heads {
            pool.wait(GUEST_WAITER, shape.pool, slots);
            // The walk commits until the head's runs stand free, not only
            // its slot count ([`SharedPool::walk_goal`]).
            device.set_guest_head(Some((chunks, pieces)));
        }
        Err(RolesError::GuestGrow {
            what: what(),
            bytes,
            room,
            run: chunks,
            prefix,
        })
    }

    /// Fresh device room for dead part `like`: its run of slots, or its
    /// runs of whole chunks, each inside one segment; all or none, else
    /// what it asked and what was free.
    pub(super) fn retake(&self, like: &Part) -> Result<Part, GuestRoom> {
        let shape = self.shape(SlotRole::Instance).map_err(|_| like.room())?;
        let mut pool = self.device.lock().unwrap();
        let taken = pool.claiming(self.claim, |pool| {
            if like.pieces > 0 {
                pool.take_chunk_runs(shape.pool, like.chunks, like.pieces)
            } else {
                pool.take_run(shape.pool, u64::from(like.run))
            }
        });
        match taken {
            Ok(refs) => Ok(Part {
                refs,
                ..like.clone()
            }),
            Err(SlotError::NoChunkRun {
                run,
                pieces,
                free_runs,
                free_chunks,
            }) => Err(GuestRoom::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                committable: 0,
                singles: 0,
            }),
            // Any other refusal (the budget, a run across a segment's end)
            // is said by name: read as the part's ask alone, its room's
            // zeros look like a full device ("free_runs: 0, free_chunks: 0,
            // committable: 0" on a device mostly free).
            Err(error) => {
                tracing::warn!(%error, run = like.chunks, pieces = like.pieces, "roles: a guest part's fresh room refused by the pool");
                Err(like.room())
            }
        }
    }

    /// The bytes dead part `part` holds on the device.
    pub(super) fn dead_bytes(&self, part: &Part) -> u64 {
        if part.pieces > 0 {
            let chunk = self.device.lock().unwrap().pool().chunk_bytes();
            u64::from(part.pieces) * u64::from(part.chunks) * chunk
        } else {
            let slot = self
                .shapes
                .of(part.role)
                .map_or(0, |shape| shape.slot_bytes);
            u64::from(part.run) * slot
        }
    }

    /// Whether `pieces` free runs of `run` adjacent `Instance` chunks stand
    /// beside where `holder`'s unpark puts its parked dead parts back
    /// (guest scratch and pair state, given back at its park), each placed
    /// as the unpark's take would place it, in the unpark's order: a read
    /// of the device pool under its lock, nothing taken
    /// ([`super::super::slots::SlotPool::chunk_runs_beside`]), so the room its own
    /// park gave back never counts toward its bind.
    #[must_use]
    pub fn fits_beside_unpark(&self, holder: Holder, run: u32, pieces: u32) -> bool {
        use super::super::slots::OwnPiece;
        let Some(shape) = self.shapes.of(SlotRole::Instance) else {
            return true;
        };
        let mut own: Vec<OwnPiece> = Vec::new();
        for (owner, item) in &self.items {
            let parked = matches!(item.tier, Tier::Host | Tier::Disk) || item.takes_fresh();
            if !parked || !self.unparks_with(*owner, item, holder) {
                continue;
            }
            let below = matches!(item.tier, Tier::Host | Tier::Disk);
            // Its guest parts laid fresh: the dead ones given back, and a
            // held one coming up from a tier below.
            for part in item
                .parts
                .iter()
                .filter(|part| (part.dead() && part.refs.is_empty()) || (part.held && below))
            {
                own.push(if part.pieces > 0 {
                    OwnPiece::Chunks {
                        run: part.chunks,
                        pieces: part.pieces,
                    }
                } else {
                    OwnPiece::Slots {
                        shape: shape.pool,
                        slots: part.run,
                    }
                });
            }
        }
        // The unpark lays its whole-chunk runs first, the longest first,
        // then its runs of slots ([`Self::prepare_unpark`]).
        own.sort_by_key(|piece| match *piece {
            OwnPiece::Chunks { run, .. } => (0, std::cmp::Reverse(run)),
            OwnPiece::Slots { .. } | OwnPiece::Take { .. } => (1, std::cmp::Reverse(0)),
        });
        self.device
            .lock()
            .unwrap()
            .pool()
            .chunk_runs_beside(&own, run, pieces)
    }

    /// Every piece `holder`'s unpark takes on the device, in the order it
    /// takes them, as the pool's placement reads them
    /// ([`super::super::slots::OwnPiece`]): each dead part with no slots (given
    /// back at its park, or short of slots its rebind names) as its runs of
    /// whole chunks or its run of slots, and each live part of an item
    /// parked below the device as the take that brings it back (a run of
    /// slots where it is contiguous, single slots else); the whole-chunk
    /// runs first, the longest first, then the rest
    /// ([`Self::prepare_unpark`]).
    pub(super) fn unpark_pieces(&self, holder: Holder) -> Vec<super::super::slots::OwnPiece> {
        use super::super::slots::OwnPiece;
        let (groups, order) = self.unpark_order(holder);
        order
            .into_iter()
            .filter_map(|(at, index)| {
                let part = &groups[at].2[index];
                let shape = self.shapes.of(part.role)?;
                // As [`Self::take_parts`] takes it on the device: a dead
                // part retaken fresh ([`Self::retake`]), a live one by
                // [`Self::take`].
                Some(if part.shaped() {
                    if part.pieces > 0 {
                        OwnPiece::Chunks {
                            run: part.chunks.max(1),
                            pieces: part.pieces,
                        }
                    } else {
                        OwnPiece::Slots {
                            shape: shape.pool,
                            slots: part.run,
                        }
                    }
                } else {
                    let slots = u32::try_from(part.refs.len()).unwrap_or(u32::MAX);
                    if part.contiguous {
                        OwnPiece::Slots {
                            shape: shape.pool,
                            slots,
                        }
                    } else {
                        OwnPiece::Take {
                            shape: shape.pool,
                            slots: u64::from(slots),
                        }
                    }
                })
            })
            .collect()
    }

    /// The items `holder`'s unpark brings back, each with whether it shed
    /// its dead parts at its park (it takes those fresh, its live parts
    /// where they lie), and the order its parts are taken in: its runs of
    /// whole chunks first, the longest first, then its runs of slots and
    /// its live parts, in the items' order. Taken in the items' order, a
    /// single chunk opened for a table at the lowest free chunk could split
    /// the only stretch a run of two stands in, and the unpark be refused
    /// the run its need counted on. What the unpark takes
    /// ([`Self::prepare_unpark`]) and what its placement reads
    /// ([`Self::unpark_pieces`]): one order.
    #[allow(clippy::type_complexity)]
    pub(super) fn unpark_order(&self, holder: Holder) -> (Vec<(Owner, bool, Vec<Part>)>, Vec<(usize, usize)>) {
        // Items that shed their dead parts at the park take fresh ones in
        // their class, their live parts where they lie: nothing is copied.
        // So does one on the device with a dead part its rebind must name
        // and no slots for it ([`Item::takes_fresh`]): the restore commits
        // only with every part its rebinds name in place.
        let groups: Vec<(Owner, bool, Vec<Part>)> = self
            .items
            .iter()
            .filter(|(owner, item)| item.takes_fresh() && self.unparks_with(**owner, item, holder))
            .map(|(owner, item)| (*owner, true, item.parts.clone()))
            .chain(
                self.items
                    .iter()
                    .filter(|(owner, item)| {
                        matches!(item.tier, Tier::Host | Tier::Disk)
                            && self.unparks_with(**owner, item, holder)
                    })
                    .map(|(owner, item)| (*owner, false, item.parts.clone())),
            )
            .collect();
        let mut order: Vec<(usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(at, (_, shed, parts))| {
                parts
                    .iter()
                    .enumerate()
                    .filter(move |(_, part)| !*shed || (part.dead() && part.refs.is_empty()))
                    .map(move |(index, _)| (at, index))
            })
            .collect();
        order.sort_by_key(|&(at, index)| {
            let part = &groups[at].2[index];
            if part.shaped() && part.pieces > 0 {
                (0, std::cmp::Reverse(part.chunks))
            } else {
                (1, std::cmp::Reverse(0))
            }
        });
        (groups, order)
    }

    /// The runs of adjacent chunks `holder`'s unpark lays fresh, one entry
    /// per run length, the longest first, read from the pool's own
    /// placement of every piece it takes ([`Self::unpark_pieces`]): its
    /// runs of whole chunks, and the chunks its runs and takes of slots
    /// open beyond the partly filled ones, in single chunks. A run of 465
    /// slots asks a chunk wherever no partly filled chunk of its class holds
    /// it, whatever free slots other chunks hold: counted by free slots, it
    /// would ask none, the need would stand, and the unpark be refused for
    /// good.
    /// Where the pool cannot place them all now, each run of slots and
    /// take asks a chunk of its own.
    #[must_use]
    pub fn unpark_need(&self, holder: Holder) -> Vec<(u32, u32)> {
        use super::super::slots::OwnPiece;
        let own = self.unpark_pieces(holder);
        let mut runs: BTreeMap<u32, u64> = BTreeMap::new();
        let mut whole = 0u32;
        for piece in &own {
            match *piece {
                OwnPiece::Chunks { run, pieces } => {
                    *runs.entry(run).or_default() += u64::from(pieces);
                    whole = whole.saturating_add(run.saturating_mul(pieces));
                }
                OwnPiece::Slots { .. } | OwnPiece::Take { .. } => {}
            }
        }
        // Where its placement does not stand now, its placement alone: a
        // part each a whole chunk was no placement (`SlotPool::opens_alone`).
        let device = self.device.lock().unwrap();
        let pool = device.pool();
        let singles = pool
            .opens_beside(&own)
            .unwrap_or_else(|| pool.opens_alone(&own))
            .saturating_sub(whole);
        drop(device);
        if singles > 0 {
            *runs.entry(1).or_default() += u64::from(singles);
        }
        runs.into_iter()
            .rev()
            .map(|(run, pieces)| (run, u32::try_from(pieces).unwrap_or(u32::MAX)))
            .collect()
    }

    /// `holder`'s whole restore placed by the takes themselves
    /// ([`super::super::slots::SlotPool::place`]): `takes` (its kv pages still to
    /// gather and the single slots its serve takes, by shape) first, then
    /// every piece its unpark takes, in the unpark's order. The pieces it
    /// asked, beside where they went: the one placement a restore is sent
    /// on, its moves planned by, and its pending bind read against.
    #[must_use]
    pub fn restore_placement(
        &self,
        holder: Holder,
        takes: &[(super::super::slots::ShapeId, u64)],
    ) -> (Vec<super::super::slots::OwnPiece>, super::super::slots::Placement) {
        use super::super::slots::OwnPiece;
        let own: Vec<OwnPiece> = takes
            .iter()
            .map(|&(shape, slots)| OwnPiece::Take { shape, slots })
            .chain(self.unpark_pieces(holder))
            .collect();
        let placement = self.device.lock().unwrap().pool().place(&own);
        (own, placement)
    }

    /// Holds for `holder`'s served restore the free chunks its unpark's
    /// pieces and then `bind` (a pending bind's runs) open as the takes
    /// place them ([`super::super::slots::SlotPool::place`]): no other take opens
    /// them until its unpark and bind have taken them or it is released.
    /// Whether its whole placement stands; held only where it does.
    pub fn reserve_restore(&mut self, holder: Holder, bind: &[(u32, u32)]) -> bool {
        use super::super::slots::OwnPiece;
        self.release_claim(holder);
        let own: Vec<OwnPiece> = self
            .unpark_pieces(holder)
            .into_iter()
            .chain(bind.iter().map(|&(run, pieces)| OwnPiece::Chunks { run, pieces }))
            .collect();
        let mut device = self.device.lock().unwrap();
        let placement = device.pool().place(&own);
        if !placement.stands() {
            return false;
        }
        let ticket = device.pool_mut().reserve_chunks(placement.opened.iter().copied());
        drop(device);
        self.claims.insert(holder, ticket);
        true
    }

    /// `own` placed for `holder` by the takes themselves
    /// ([`super::super::slots::SlotPool::place`]), what is held for its served
    /// restore open to them: the one placement its pending bind is checked
    /// by once its unpark has taken its own. Read as a count of free runs
    /// without its own reservation, a restore whose bind runs are held for
    /// it would find none of them, every serve (`PendingBindShort`).
    #[must_use]
    pub fn place_claimed(&self, holder: Holder, own: &[super::super::slots::OwnPiece]) -> super::super::slots::Placement {
        let ticket = self.claims.get(&holder).copied();
        self.device
            .lock()
            .unwrap()
            .pool_mut()
            .with_claim(ticket, |pool| pool.place(own))
    }

    /// `holder`'s reservation, its chunks not yet taken, goes back.
    pub fn release_claim(&mut self, holder: Holder) {
        if let Some(ticket) = self.claims.remove(&holder) {
            self.device.lock().unwrap().pool_mut().release_reservation(ticket);
        }
    }

    /// What `holder`'s restore would take back, in one line for a stall
    /// census: each parked item (instance, channel, adapter, pair) with its
    /// tier, and each part by role and size (scratch runs of whole chunks,
    /// or a run of slots; dead between fires or live, with its refs), then
    /// whether `need` runs stand beside the dead parts as the pool stands
    /// ([`Self::fits_beside_unpark`]: the live parts are not placed there),
    /// the free runs of `need`'s length, and each role shape's free slots.
    #[must_use]
    pub fn unpark_census(&self, holder: Holder, need: Option<(u32, u32)>) -> String {
        let mut items = Vec::new();
        for (owner, item) in &self.items {
            let parked = matches!(item.tier, Tier::Host | Tier::Disk) || item.takes_fresh();
            if !parked || !self.unparks_with(*owner, item, holder) {
                continue;
            }
            let parts: Vec<String> = item
                .parts
                .iter()
                .map(|part| {
                    format!(
                        "{:?}:{}{}x{}c/{}s/refs{}",
                        part.role,
                        if part.dead() { "dead:" } else { "live:" },
                        part.pieces,
                        part.chunks,
                        part.run,
                        part.refs.len()
                    )
                })
                .collect();
            items.push(format!("{owner:?}@{:?}{}[{}]", item.tier, if item.shed { "(shed)" } else { "" }, parts.join(",")));
        }
        let beside = need.map(|(run, pieces)| self.fits_beside_unpark(holder, run, pieces));
        let free_runs = need.map(|(run, _)| self.free_runs(run));
        let shapes: Vec<String> = [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter]
            .into_iter()
            .filter_map(|role| {
                let shape = self.shapes.of(role)?;
                let pool = self.device.lock().unwrap();
                Some(format!("{role:?}:free_slots={}", pool.pool().available(shape.pool)))
            })
            .collect();
        format!(
            "parked=[{}] need={need:?} beside_dead_parts={beside:?} free_runs={free_runs:?} shapes=[{}]",
            items.join("; "),
            shapes.join(",")
        )
    }

    /// The free runs of `run` adjacent `Instance` chunks inside one
    /// segment the device pool holds now: how many more pieces of a guest
    /// state that lies in runs of that length it could place.
    #[must_use]
    pub fn free_runs(&self, run: u32) -> u32 {
        self.device.lock().unwrap().pool().free_chunk_runs(run)
    }

    /// Whether `pieces` free runs of `run` chunks stand, and `singles` free
    /// chunks beside them.
    #[must_use]
    pub fn room_free(&self, run: u32, pieces: u32, singles: u32) -> bool {
        let pool = self.device.lock().unwrap();
        let pool = pool.pool();
        pool.free_chunk_runs(run) >= pieces
            && pool.free_chunk_runs(1) >= run.max(1).saturating_mul(pieces).saturating_add(singles)
    }

    /// Gives `parts` back on `on`: a failed take's rollback, which frees no
    /// room a waiter lacked (a take is all or none).
    pub(super) fn give(&self, on: &Shared, parts: &[Part]) {
        let mut pool = on.lock().unwrap();
        for part in parts.iter().filter(|part| !part.refs.is_empty()) {
            if let Some(shape) = self.shapes.of(part.role) {
                pool.give_refs(shape.pool, &part.refs);
            }
        }
    }

    /// Tells the binds waiting on room ([`Self::room_watch`]) to look again.
    pub(super) fn tell_room(&self) {
        self.freed
            .gives
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.freed.notify.notify_waiters();
    }

    /// A frame naming `instances` (its attachments) was handed to the
    /// engine: each instance's item, and the pair, channels and adapter it
    /// shares, are read until it lands. Answers what it named, for
    /// [`Self::landed`]. A pair's batch tables are read by every grouped
    /// fire of its instances: given back at the pair's last close while
    /// one was in flight, their chunks went to another owner under the
    /// fire (Metal's 0xa0, FUSED_GEOMETRY_MISMATCH).
    pub fn reading(&mut self, instances: impl IntoIterator<Item = InstanceId>) -> Vec<Owner> {
        let mut named: Vec<Owner> = Vec::new();
        for id in instances {
            let Some(item) = self.items.get(&Owner::Instance(id)) else {
                continue;
            };
            named.push(Owner::Instance(id));
            named.extend(item.channels.iter().map(|&channel| Owner::Channel(channel)));
            named.extend(item.adapter.map(Owner::Adapter));
            named.extend(item.pair.map(Owner::Pair));
        }
        named.retain(|owner| self.items.contains_key(owner));
        for owner in &named {
            if let Some(item) = self.items.get_mut(owner) {
                item.reads += 1;
            }
        }
        named
    }

    /// The frame that named `named` ([`Self::reading`]) landed: what was
    /// let go under it goes back now.
    pub fn landed(&mut self, named: &[Owner]) {
        for &owner in named {
            if let Some(item) = self.items.get_mut(&owner) {
                item.reads = item.reads.saturating_sub(1);
            }
            self.free_if_done(owner);
        }
        self.release_unheld_pairs();
    }

    /// `holder`'s guest bind no longer waits: it bound, was refused, or
    /// its process is gone. Where the pool counts its need, it counts it no
    /// more, so the walk commits for no one gone.
    pub fn leave_guest_wait(&mut self, holder: Holder) {
        if self.guest_waiter != Some(holder) {
            return;
        }
        self.guest_waiter = None;
        let mut device = self.device.lock().unwrap();
        device.pool_mut().unwait(GUEST_WAITER);
        device.set_guest_head(None);
    }

    /// Gives back `parts` a closed or parked item held on `on`: guest state
    /// given back on the device is the room a bind short of it waits on
    /// ([`Self::room_watch`]), and its waiters are told.
    pub(super) fn give_freeing(&self, on: &Shared, parts: &[Part]) {
        self.give(on, parts);
        if Arc::ptr_eq(on, &self.device)
            && parts
                .iter()
                .any(|part| part.dead() && !part.refs.is_empty())
        {
            self.tell_room();
        }
    }

    /// Takes the slots of every part, all or none: a whole part's fresh
    /// chunks on the device, nothing off it (`device`: `on` is the device).
    pub(super) fn take_parts(
        &self,
        on: &Shared,
        like: &[Part],
        device: bool,
    ) -> Result<Vec<Part>, RolesError> {
        let mut taken = Vec::with_capacity(like.len());
        for part in like {
            let took = if device && part.shaped() {
                // Laid as runs on the device, dead or held: its fresh runs,
                // a held part's bytes copied into them.
                self.retake(part).map_err(|room| {
                    let (run, _) = part.chunk_runs();
                    let committable = self.device.lock().unwrap().pool().committable_runs(run);
                    RolesError::GuestShort {
                        what: "a parked guest state coming back".into(),
                        bytes: self.dead_bytes(part),
                        room: room.with_committable(committable),
                    }
                })
            } else if part.dead() {
                Ok(Part {
                    refs: Vec::new(),
                    ..part.clone()
                })
            } else {
                // Below the device, its slots anywhere; what it is (held,
                // its runs' shape for the way back) goes with it.
                self.take(on, part.role, part.refs.len() as u32, part.contiguous)
                    .map(|taken| Part {
                        refs: taken.refs,
                        ..part.clone()
                    })
            };
            match took {
                Ok(part) => taken.push(part),
                Err(error) => {
                    self.give(on, &taken);
                    return Err(error);
                }
            }
        }
        Ok(taken)
    }

    // ---- programs and instances ----
}
