//! Binds and registrations: an instance's parts, its guest state, the bind
//! lifecycle, and a channel's ring.

use super::*;

impl Roles {
    /// The engine registered `program`, whose instances hold `facts`.
    pub fn program_registered(&mut self, program: ProgramId, facts: ProgramFacts) {
        self.programs.insert(program, facts);
    }

    /// `holder`'s claim of whole chunks for a bind: the largest room any
    /// of its binds asked so far, its pair's state counted as laid fresh
    /// (a bind may lay it so), in runs of whole chunks; none where it has
    /// bound nothing that takes whole chunks. What the capacity gate
    /// reserves for the finisher before it asks
    /// (`planner::ResidencyPlanner::chunk_spare`).
    #[must_use]
    pub fn bind_claim(&self, holder: Holder) -> Vec<(u32, u32)> {
        self.bind_claims.get(&holder).cloned().unwrap_or_default()
    }

    /// `holder` asked a bind of `runs`: its claim is the larger of the two.
    pub(super) fn claim(&mut self, holder: Holder, runs: Vec<(u32, u32)>) {
        let chunks = |runs: &[(u32, u32)]| {
            runs.iter()
                .map(|&(run, pieces)| u64::from(run) * u64::from(pieces))
                .sum::<u64>()
        };
        let claim = self.bind_claims.entry(holder).or_default();
        if chunks(&runs) > chunks(claim) {
            *claim = runs;
        }
    }

    /// `holder` left: its claim goes.
    pub fn forget_claim(&mut self, holder: Holder) {
        self.bind_claims.remove(&holder);
    }

    /// Whether `program`'s facts are known here.
    #[must_use]
    pub fn knows(&self, program: ProgramId) -> bool {
        self.programs.contains_key(&program)
    }

    /// The slots an instance of `program` bound to `channels` takes:
    /// `footprint.slots` of the `Instance` shape (cells are packed whole, so
    /// the engine counts, not the bytes), for each ring of the program no
    /// registered channel covers (`registered`), its slots of the `Channel`
    /// shape, and for the adapter it lands (`adapter`, its key) the slots
    /// of the `Adapter` shape another instance already holds for that key,
    /// else fresh ones; with them the guest state `guest`, taken before
    /// anything was registered for the bind ([`Self::take_guest`]), which
    /// goes back with the rest where the take fails.
    /// The fresh chunks the parts a bind of `program` takes beside its
    /// guest state may open: its tables, each ring of its own (a channel
    /// `registered` names holds its ring) and its adapter where none is on
    /// the device, as [`Self::take_instance`] takes them, each role's parts
    /// in as few chunks as their slots fill.
    pub(super) fn instance_parts_fresh(
        &self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
    ) -> u32 {
        let Some(facts) = self.programs.get(&program) else {
            return 0;
        };
        let rings: u32 = facts
            .rings
            .iter()
            .filter(|ring| {
                !channels
                    .get(ring.channel as usize)
                    .is_some_and(|&id| registered(id))
            })
            .map(|ring| ring.footprint.slots)
            .sum();
        let adapter = adapter
            .filter(|&key| {
                !matches!(self.items.get(&Owner::Adapter(key)), Some(item) if item.tier == Tier::Device)
            })
            .map_or(0, |_| facts.adapter.slots);
        let Ok(instance) = self.shape(SlotRole::Instance) else {
            return 0;
        };
        let chunk_bytes = u64::from(self.chunk_slots) * instance.slot_bytes;
        [
            (SlotRole::Instance, facts.instance.slots),
            (SlotRole::Channel, rings),
            (SlotRole::Adapter, adapter),
        ]
        .into_iter()
        .filter(|&(_, slots)| slots > 0)
        .map(|(role, slots)| {
            let bytes = self
                .shape(role)
                .map_or(chunk_bytes, |shape| u64::from(slots) * shape.slot_bytes);
            u32::try_from(bytes.div_ceil(chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
        })
        .sum()
    }

    pub fn take_instance(
        &mut self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
        guest: GuestTaken,
    ) -> Result<Taken, (RolesError, u32)> {
        let Some(facts) = self.programs.get(&program).cloned() else {
            self.untake_guest(guest);
            return Err((RolesError::Program(program), 0));
        };
        let mut shared = Vec::new();
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id)) {
                if item.tier != Tier::Device {
                    let holder = item.holder;
                    self.untake_guest(guest);
                    return Err((
                        RolesError::Parked {
                            channel: id,
                            holder,
                        },
                        0,
                    ));
                }
                shared.push(id);
            }
        }
        let device = Arc::clone(&self.device);
        let mut taken = Taken {
            channels: shared,
            chunk_slots: self.chunk_slots,
            ..Taken::default()
        };
        let footprint = facts.instance;
        match self.take(
            &device,
            SlotRole::Instance,
            footprint.slots,
            footprint.contiguous,
        ) {
            Ok(part) => taken.parts.push(part),
            Err(error) => {
                self.untake_guest(guest);
                return Err((error, 0));
            }
        }
        for ring in &facts.rings {
            let covered = channels
                .get(ring.channel as usize)
                .is_some_and(|&id| registered(id));
            if covered {
                continue;
            }
            let footprint = ring.footprint;
            match self.take(
                &device,
                SlotRole::Channel,
                footprint.slots,
                footprint.contiguous,
            ) {
                Ok(part) => {
                    taken.parts.push(part);
                    taken.rings.push(ring.channel);
                }
                Err(error) => {
                    let opened = u32::try_from(taken.parts.len()).unwrap_or(u32::MAX);
                    self.give(&device, &taken.parts);
                    self.untake_guest(guest);
                    return Err((error, opened));
                }
            }
        }
        let footprint = facts.adapter;
        if let Some(key) = adapter.filter(|_| footprint.slots > 0) {
            taken.adapter = match self.items.get(&Owner::Adapter(key)) {
                Some(item) if item.tier == Tier::Device => {
                    Some((key, item.parts[0].clone(), false))
                }
                Some(_) => {
                    self.give(&device, &taken.parts);
                    self.untake_guest(guest);
                    return Err((RolesError::AdapterParked, 0));
                }
                None => match self.take(
                    &device,
                    SlotRole::Adapter,
                    footprint.slots,
                    footprint.contiguous,
                ) {
                    Ok(part) => Some((key, part, true)),
                    Err(error) => {
                        let opened = u32::try_from(taken.parts.len()).unwrap_or(u32::MAX);
                        self.give(&device, &taken.parts);
                        self.untake_guest(guest);
                        return Err((error, opened));
                    }
                },
            };
        }
        taken.parts.extend(guest.scratch);
        taken.pair = guest.pair;
        self.pin_shared(&taken, true);
        Ok(taken)
    }

    /// The pair and the adapter `taken` names that are items already: held
    /// by its bind (`pin`), or let go of at its bind or its give back.
    pub(super) fn pin_shared(&mut self, taken: &Taken, pin: bool) {
        let owners = [
            taken.adapter.as_ref().map(|(key, ..)| Owner::Adapter(*key)),
            taken.pair.as_ref().map(|(key, ..)| Owner::Pair(*key)),
        ];
        for owner in owners.into_iter().flatten() {
            if let Some(item) = self.items.get_mut(&owner) {
                item.binds = if pin {
                    item.binds + 1
                } else {
                    item.binds.saturating_sub(1)
                };
            }
        }
    }

    /// A guest program's state for a bind of an instance on `channels`
    /// ([`GuestState`], of the pair `pair`): the instance's scratch, whole
    /// `Instance` chunks, and the pair's chunks its other instances hold
    /// already, else fresh ones. Taken first, before the bind registers
    /// anything: a bind short of it ([`RolesError::BindShort`],
    /// [`RolesError::GuestGrow`]) waits for room with nothing to undo. A
    /// channel it binds parked with its process refuses it first, as
    /// [`Self::take_instance`] would.
    /// As [`Self::take_guest_now`], for `holder`'s bind. Every bind takes
    /// room as it comes, none waiting behind another: a smaller one never
    /// takes a longer run's chunks ([`crate::store::slots`] size classes).
    /// The first bind short of room that waits on a commit has its need
    /// counted toward the walk's goal until it binds or leaves; one short
    /// meanwhile leaves that need as it is, and waits on the commit and the
    /// planner as it does (the planner asks the walk for its need too).
    pub fn take_guest(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        holder: Option<Holder>,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        self.claim = holder.and_then(|holder| self.claims.get(&holder).copied());
        if let Some(ticket) = self.claim {
            tracing::debug!(
                ?holder,
                chunks = ?self.device.lock().unwrap().pool().reserved(ticket),
                "roles: a bind takes from what its restore held for it"
            );
        }
        let taken = self.take_guest_claimed(channels, (pair, state), holder, beside);
        self.claim = None;
        // Its bind took what was held for it: the rest goes back.
        if taken.is_ok()
            && let Some(holder) = holder
        {
            self.release_claim(holder);
        }
        taken
    }

    pub(super) fn take_guest_claimed(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        holder: Option<Holder>,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        let heads =
            holder.is_none_or(|holder| self.guest_waiter.is_none_or(|counted| counted == holder));
        let taken = self.take_guest_now(channels, (pair, state), heads, beside);
        if let Some(holder) = holder.filter(|_| heads) {
            match &taken {
                Err(RolesError::GuestGrow { .. }) => self.guest_waiter = Some(holder),
                _ if self.guest_waiter == Some(holder) => self.guest_waiter = None,
                _ => {}
            }
        }
        taken
    }

    pub(super) fn take_guest_now(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        heads: bool,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        // What segregating runs by length holds back: each class's partly
        // filled Instance chunks and their free bytes, at debug.
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(shape) = self.shape(SlotRole::Instance)
        {
            let device = self.device.lock().unwrap();
            let classes: Vec<(u32, usize, u64)> = device
                .pool()
                .partial_classes(shape.pool)
                .into_iter()
                .map(|(class, chunks, free)| (class, chunks, free * shape.slot_bytes))
                .collect();
            tracing::debug!(
                ?classes,
                "Instance chunks partly filled, by class: (run length, chunks, free bytes)"
            );
        }
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id))
                && item.tier != Tier::Device
            {
                return Err(RolesError::Parked {
                    channel: id,
                    holder: item.holder,
                });
            }
        }
        let mut taken = GuestTaken::default();
        let run = state.chunks_a_piece();
        // The instance's own bytes: its resident part and every scratch arm
        // (none taken per fire yet), placed at the bind.
        let per_instance = state.instance_bytes();
        if (per_instance > 0 || state.per_pair > 0) && self.stated.insert(pair) {
            self.state_room(pair, state);
        }
        // Every part the bind takes fresh (its scratch, and its pair's state
        // where the pair holds none on the device now), in runs of the
        // longest, and a single chunk for each role of its tables, rings
        // and adapter (`beside`): what a refusal for want of room asks
        // ([`RolesError::asking`]), so the room it waits on holds the bind.
        // Those parts found slots in chunks open when it was refused; once
        // its process was suspended and restored for the room asked, those
        // slots had gone, the bind was short again of the chunks they then
        // opened and its process went down a second time for one bind (the
        // simulation's field mix, seed 86: refused opening 2 chunks,
        // restored with 2, refused again opening 4).
        let pair_fresh = state.per_pair > 0
            && !matches!(
                self.items.get(&Owner::Pair(pair)),
                Some(item) if item.tier == Tier::Device && !item.shed
            );
        // Its pair's state is asked too where another holds it on the
        // device now: suspended for this room, the pair's holders may park
        // meanwhile, and the bind, restored with the room asked, lays the
        // pair fresh (the simulation's seed 380: asked a run of three,
        // restored with it, short of the pair's second run). Over, never
        // under: the room a pair joined in place leaves goes back with the
        // take.
        let _ = pair_fresh;
        let fresh: Vec<(u32, u32)> = [
            (per_instance > 0).then_some(per_instance),
            (state.per_pair > 0).then_some(state.per_pair),
        ]
        .into_iter()
        .flatten()
        .filter_map(|bytes| self.scratch_part(bytes, run).ok())
        .map(|part| part.chunk_runs())
        .collect();
        let longest = fresh.iter().map(|&(run, _)| run).max().unwrap_or(1).max(1);
        let whole: u32 = fresh
            .iter()
            .map(|&(run, pieces)| run.div_ceil(longest).saturating_mul(pieces))
            .sum::<u32>();
        if per_instance > 0 {
            match self.take_scratch(
                per_instance,
                run,
                || format!("an instance of {pair}'s scratch"),
                heads,
            ) {
                // Resident bytes beside the scratch: the part is held,
                // copied at a park and back at an unpark, never dead.
                Ok(part) => {
                    taken.scratch = Some(Part {
                        held: state.resident > 0,
                        ..part
                    });
                }
                Err(error) => {
                    self.untake_guest(taken);
                    return Err(error.asking(whole, beside));
                }
            }
        }
        if state.per_pair > 0 {
            taken.pair = match self.items.get(&Owner::Pair(pair)) {
                Some(item) if item.tier == Tier::Device && !item.shed => {
                    Some((pair, item.parts[0].clone(), false, item.generation))
                }
                // Parked with its instances: its contents are dead between
                // fires, so it comes back on its own, fresh room on the
                // device for this bind; its parked instances are rebound to
                // it as they unpark.
                // Parked where it lies, its dead part given back
                // ([`Item::shed`]), it comes back the same way: bound to its
                // emptied part, the bind named no place.
                Some(item)
                    if item.tier == Tier::Host || (item.tier == Tier::Device && item.shed) =>
                {
                    let like = item.parts[0].clone();
                    match self.retake(&like) {
                        Ok(part) => {
                            let generation = self.next_generation();
                            let item = self.items.get_mut(&Owner::Pair(pair)).expect("present");
                            item.parts = vec![part.clone()];
                            item.tier = Tier::Device;
                            item.shed = false;
                            item.generation = generation;
                            taken.pair_retaken = true;
                            Some((pair, part, false, generation))
                        }
                        Err(room) => {
                            let short = RolesError::GuestShort {
                                what: format!("{pair}'s state coming back"),
                                bytes: self.dead_bytes(&like),
                                room,
                            };
                            let fits = self.fits_budget(&[Owner::Pair(pair)]);
                            self.untake_guest(taken);
                            return Err(if fits {
                                RolesError::BindShort(Box::new(short))
                            } else {
                                short
                            }
                            .asking(whole, beside));
                        }
                    }
                }
                Some(_) => {
                    self.untake_guest(taken);
                    return Err(RolesError::PairParked(pair));
                }
                None => {
                    match self.take_scratch(
                        state.per_pair,
                        run,
                        || format!("{pair}'s state"),
                        heads,
                    ) {
                        // Its item from the take on, held by the binds on
                        // their way ([`Self::pin_shared`]) and then by its
                        // instances: a second bind of the pair taken before
                        // this one bound joins it. Taken fresh again, two
                        // binds of one pair each named a taking of its own,
                        // the engine laid the group at one under the other's
                        // member, and the first bound left the other's chunk
                        // held by no item for good (the simulation's field
                        // mix, seed 20: pair 3 at chunk 6 for instance 16, at
                        // chunk 19 for 15 and 17, at one generation; seed
                        // 115: a beam pair's chunk out of service from step 0).
                        Ok(part) => {
                            let generation = self.next_generation();
                            let mut item = Item::new(None, vec![part.clone()]);
                            // Held by its instances alone: freed at the last
                            // close, or with its last bind given back.
                            item.open = false;
                            item.generation = generation;
                            self.items.insert(Owner::Pair(pair), item);
                            Some((pair, part, true, generation))
                        }
                        Err(error) => {
                            self.untake_guest(taken);
                            return Err(error.asking(whole, beside));
                        }
                    }
                }
            };
        }
        Ok(taken)
    }

    /// States, once a pair, what its guest state asks of the pool and how
    /// many of it the pool holds free as it stands: the run of adjacent
    /// chunks a piece takes, the pieces an instance and the pair take, and
    /// the free runs of that length, so a refusal for want of a run is
    /// visible before it happens.
    pub(super) fn state_room(&self, pair: FireKey, state: GuestState) {
        let run = state.chunks_a_piece();
        let pieces = |bytes: u64| self.scratch_part(bytes, run).map_or(0, |part| part.pieces);
        tracing::info!(
            %pair,
            resident = state.resident,
            scratch = state.scratch,
            scratch_arms = state.scratch_arms,
            per_pair = state.per_pair,
            run,
            instance_pieces = pieces(state.instance_bytes()),
            pair_pieces = pieces(state.per_pair),
            free_runs = self.free_runs(run),
            "a guest program's room: pieces of `run` adjacent chunks, and the free runs of that length"
        );
    }

    /// A bind of `holder`'s was taken: it is on its way until bound or
    /// given back.
    pub(super) fn bind_taken_for(&mut self, holder: Option<Holder>) {
        if let Some(holder) = holder {
            *self.binding.entry(holder).or_default() += 1;
        }
    }

    /// A bind of `holder`'s was bound or given back.
    pub(super) fn bind_ended_for(&mut self, holder: Option<Holder>) {
        let Some(holder) = holder else {
            return;
        };
        if let Some(count) = self.binding.get_mut(&holder) {
            *count -= 1;
            if *count == 0 {
                self.binding.remove(&holder);
            }
        }
        self.bind_landed.notify_waiters();
    }

    /// Whether a bind of `holder`'s is on its way: taken, its slots and
    /// its pair held, and not yet bound by the engine or given back. A
    /// park of `holder` waits for it ([`Self::bind_landed`]): parked
    /// meanwhile, the instance it binds stood on the device outside its
    /// process's park, its pair given fresh slots at a restore the
    /// instance's bind never named, and those slots went to another
    /// program's bind (the simulation, a yield between a bind's take and
    /// its bind: `guest::claims` refusing an instance over a live pair's
    /// run).
    #[must_use]
    pub fn binds_on_way(&self, holder: Holder) -> bool {
        self.binding.contains_key(&holder)
    }

    /// Told each time a bind on its way is bound or given back.
    #[must_use]
    pub fn bind_landed(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.bind_landed)
    }

    /// The bind of `taken` failed: its slots go back.
    pub fn untake(&mut self, taken: Taken) {
        self.off_way(&taken);
        self.bind_ended_for(taken.holder);
        self.pin_shared(&taken, false);
        let shared = [
            taken.adapter.as_ref().map(|(key, ..)| Owner::Adapter(*key)),
            taken.pair.as_ref().map(|(key, ..)| Owner::Pair(*key)),
        ];
        self.give(&self.device, &taken.parts);
        if let Some((_, part, true)) = taken.adapter {
            self.give(&self.device, &[part]);
        }
        // A pair taken fresh is an item from its take ([`Self::take_guest`]):
        // it goes below with its last bind, unless another joined it.
        // What the bind held and no one else holds goes now: a pair whose
        // members all parked while the bind was on its way, an adapter or
        // pair with none attached.
        for owner in shared.into_iter().flatten() {
            self.free_if_done(owner);
        }
        self.release_unheld_pairs();
    }

    /// A bind's or a registration's take short of slots (`Short`, `NoRun`):
    /// a free chunk of the budget holds any one item (a run past a chunk is
    /// `PastChunk`), so it waits for room as a guest state does
    /// ([`RolesError::BindShort`]). Anything else stands as it is.
    /// `guest`: the runs its guest state's fresh parts took; `opened`: the
    /// parts of its own (tables, rings, adapter) taken before the one
    /// short, each of which may have opened a chunk.
    pub(super) fn bind_short(
        &self,
        error: RolesError,
        guest: Option<(u32, u32)>,
        opened: u32,
    ) -> RolesError {
        match error {
            RolesError::Short { .. } | RolesError::NoRun { .. }
                if self.device.lock().unwrap().state_chunks_now() > 0 =>
            {
                // Short of its tables' chunk beside the fresh chunks its guest
                // state took (and gave back): the room asked is all of them,
                // never the tables' one chunk alone (else a scratch chunk and
                // a pair chunk taken, the tables short of a third, one chunk
                // served, short again: a spin). Runs of more than one chunk ask one more
                // run of their length for the tables: over, never under.
                let short = match guest.or((opened > 0).then_some((1, 0))) {
                    Some((run, pieces)) => RolesError::GuestShort {
                        what: format!(
                            "its tables, rings and adapter beside its guest state's {pieces} fresh \
                             runs ({error})"
                        ),
                        bytes: 0,
                        room: if run <= 1 {
                            GuestRoom::Slots {
                                slots: 0,
                                committable: 0,
                                chunks: pieces + opened + 1,
                            }
                        } else {
                            GuestRoom::Runs {
                                run,
                                pieces,
                                free_runs: 0,
                                free_chunks: 0,
                                committable: 0,
                                singles: opened + 1,
                            }
                        },
                    },
                    None => error,
                };
                RolesError::BindShort(Box::new(short))
            }
            error => error,
        }
    }

    /// A guest state taken for a bind that registered nothing, or failed
    /// before its instance's take: its chunks go back.
    pub fn untake_guest(&mut self, guest: GuestTaken) {
        if let Some(part) = &guest.scratch {
            self.give(&self.device, std::slice::from_ref(part));
        }
        // A pair it took fresh is an item from its take: gone with it
        // unless another bind joined it meanwhile.
        if let Some((key, _, true, _)) = guest.pair {
            self.free_if_done(Owner::Pair(key));
        }
    }

    /// The engine bound `taken` as instance `id`, for `holder`.
    pub fn bound(&mut self, id: InstanceId, holder: Option<Holder>, taken: Taken) {
        self.off_way(&taken);
        self.bind_ended_for(taken.holder);
        self.pin_shared(&taken, false);
        for &channel in &taken.channels {
            if let Some(item) = self.items.get_mut(&Owner::Channel(channel)) {
                item.attached.insert(id);
            }
        }
        let mut item = Item::new(holder, taken.parts);
        item.rings = taken.rings;
        item.channels = taken.channels;
        if let Some((key, part, _)) = taken.adapter {
            let shared = self.items.entry(Owner::Adapter(key)).or_insert_with(|| {
                let mut adapter = Item::new(holder, vec![part]);
                // Held by its instances alone: freed at the last close.
                adapter.open = false;
                adapter
            });
            shared.attached.insert(id);
            // A live member again: no longer parked where it lies, so no
            // move down takes it from under this instance.
            shared.shed = false;
            item.adapter = Some(key);
        }
        if let Some((key, part, _, generation)) = taken.pair {
            let shared = self.items.entry(Owner::Pair(key)).or_insert_with(|| {
                let mut pair = Item::new(holder, vec![part]);
                // Held by its instances alone: freed at the last close.
                pair.open = false;
                pair.generation = generation;
                pair
            });
            shared.attached.insert(id);
            item.pair = Some(key);
        }
        if item.parts.iter().all(|part| part.refs.is_empty())
            && item.channels.is_empty()
            && item.adapter.is_none()
            && item.pair.is_none()
        {
            return;
        }
        self.items.insert(Owner::Instance(id), item);
    }

    /// The engine closed instance `id`: its slots go back, and a closed
    /// channel it was the last on with them.
    pub fn close_instance(&mut self, id: InstanceId) {
        let owner = Owner::Instance(id);
        self.held.retain(|&(instance, _)| instance != id);
        let Some(item) = self.items.get_mut(&owner) else {
            return;
        };
        tracing::debug!(
            target: "runtime::planner::roles",
            ?owner,
            holder = ?item.holder,
            tier = ?item.tier,
            parts = %traced(&item.parts),
            "roles: close instance"
        );
        item.open = false;
        let shared: Vec<Owner> = item
            .channels
            .iter()
            .map(|&channel| Owner::Channel(channel))
            .chain(item.adapter.map(Owner::Adapter))
            .chain(item.pair.map(Owner::Pair))
            .collect();
        self.free_if_done(owner);
        for owner in shared {
            if let Some(item) = self.items.get_mut(&owner) {
                item.attached.remove(&id);
            }
            self.free_if_done(owner);
        }
    }

    // ---- channels ----

    /// The slots a channel whose ring `facts` states takes: one run of
    /// adjacent slots where it says `contiguous`.
    pub fn take_channel(&mut self, facts: &RingFacts) -> Result<TakenRing, RolesError> {
        let footprint = facts.footprint;
        self.take(
            &self.device,
            SlotRole::Channel,
            footprint.slots,
            footprint.contiguous,
        )
        .map(TakenRing)
    }

    /// The channel's registration failed: its slots go back.
    pub fn untake_channel(&mut self, taken: TakenRing) {
        self.give(&self.device, &[taken.0]);
    }

    /// The engine registered channel `id` over `taken`, for `holder`.
    pub fn channel_registered(&mut self, id: ChannelId, holder: Option<Holder>, taken: TakenRing) {
        if taken.0.refs.is_empty() {
            return;
        }
        self.items
            .insert(Owner::Channel(id), Item::new(holder, vec![taken.0]));
    }

    /// The engine closed channel `id`: its ring goes back once no bound
    /// instance holds it. Until then the channel is the engine's still
    /// (closing, `Engine::close_channel`): its ring parks and unparks with
    /// its instances' holder and is rebound like an open channel's
    /// ([`Roles::rebinds`]), and goes back only after the engine closed
    /// the last instance on it ([`Roles::close_instance`], which the lane
    /// runs once `Engine::close_instance` returned), so no slot the engine
    /// may still address is handed elsewhere.
    pub fn close_channel(&mut self, id: ChannelId) {
        let owner = Owner::Channel(id);
        if let Some(item) = self.items.get_mut(&owner) {
            item.open = false;
            tracing::debug!(
                target: "runtime::planner::roles",
                ?owner,
                holder = ?item.holder,
                tier = ?item.tier,
                attached = item.attached.len(),
                "roles: close channel"
            );
        }
        self.free_if_done(owner);
    }
}
