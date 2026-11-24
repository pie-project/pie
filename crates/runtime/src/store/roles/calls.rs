//! The roles' side of the engine's calls: a program learned, a ring taken
//! and registered, an instance bound, pumped and rebound.

use super::*;

/// Asks `engine` what an instance of `program` holds, where `roles` has
/// not yet: once a program, at its registration.
pub fn learn_program(
    roles: &RolesHandle,
    engine: &dyn Engine,
    program: ProgramId,
) -> ::engine::Result<()> {
    if roles.lock().unwrap().knows(program) {
        return Ok(());
    }
    let facts = engine.program_facts(program)?;
    roles.lock().unwrap().program_registered(program, facts);
    Ok(())
}

/// The slots a registration of `plan` takes for its ring, before anything
/// is registered: short of room a budget holds, [`RolesError::BindShort`].
/// `None` where no roles are kept or the engine keeps the ring itself.
pub fn take_ring(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
) -> anyhow::Result<Option<TakenRing>> {
    let Some(roles) = roles else {
        return Ok(None);
    };
    let Some(facts) = engine.ring_facts(plan)? else {
        return Ok(None);
    };
    let mut roles = roles.lock().unwrap();
    Ok(Some(
        roles
            .take_channel(&facts)
            .map_err(|error| roles.bind_short(error, None, 0))?,
    ))
}

/// The rings of every registration of `plans` ([`take_ring`]), all or
/// none.
pub fn take_rings(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plans: &[ChannelRegistration],
) -> anyhow::Result<Vec<Option<TakenRing>>> {
    let mut rings = Vec::with_capacity(plans.len());
    for plan in plans {
        match take_ring(roles, engine, plan) {
            Ok(ring) => rings.push(ring),
            Err(error) => {
                untake_rings(roles, rings);
                return Err(error);
            }
        }
    }
    Ok(rings)
}

/// The channels of `plans`, registered by the bind they come with, whose
/// rings the engine holds ([`Engine::holds_channel`]): with the channels
/// it holds registered already, what a bind's `registered` names
/// ([`take_bind`]). A channel the engine leaves to the runtime is each
/// instance's own ring, registered by this bind or not.
///
/// [`Engine::holds_channel`]: ::engine::Engine::holds_channel
#[must_use]
pub fn held_by(engine: &dyn Engine, plans: &[ChannelRegistration]) -> Vec<ChannelId> {
    plans
        .iter()
        .filter(|plan| engine.holds_channel(plan))
        .map(|plan| plan.id)
        .collect()
}

/// Gives back rings taken for registrations that will not happen.
pub fn untake_rings(roles: Option<&RolesHandle>, rings: Vec<Option<TakenRing>>) {
    if let Some(roles) = roles {
        let mut roles = roles.lock().unwrap();
        for ring in rings.into_iter().flatten() {
            roles.untake_channel(ring);
        }
    }
}

/// Registers `plan` over the ring [`take_ring`] took for it (`taken`).
pub fn register_taken<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    taken: Option<TakenRing>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let (Some(roles), Some(taken)) = (roles, taken) else {
        return register(engine, plan);
    };
    let plan = ChannelRegistration {
        slots: taken.slots(),
        ..plan.clone()
    };
    match register(engine, &plan) {
        Ok(registered) => {
            roles
                .lock()
                .unwrap()
                .channel_registered(plan.id, holder, taken);
            Ok(registered)
        }
        Err(error) => {
            roles.lock().unwrap().untake_channel(taken);
            Err(error)
        }
    }
}

/// Everything a bind of `binding` takes of the pool, before the bind
/// registers anything: its guest state ([`Roles::take_guest`]), then its
/// tables, its own rings and its adapter ([`Roles::take_instance`]); a
/// ring of a channel `registered` names (registered with the engine, or
/// to be by this bind where the engine holds it: [`held_by`]) is the
/// channel's. Short of room a budget holds, the bind is
/// answered [`RolesError::BindShort`] and waits for it with nothing to
/// roll back. `None` where no roles are kept.
pub fn take_bind(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    registered: impl Fn(ChannelId) -> bool,
    holder: Option<Holder>,
    resident: impl Fn(Holder) -> bool,
) -> anyhow::Result<Option<Taken>> {
    let Some(roles) = roles else {
        return Ok(None);
    };
    let roles_handle = roles;
    if let Some(holder) = holder.filter(|&holder| !resident(holder)) {
        return Err(RolesError::HolderParked(holder).into());
    }
    learn_program(roles, engine, binding.program)?;
    let state = engine.guest_state(binding.program, &binding.extents, binding.geometry)?;
    let adapter = engine.adapter_key(binding)?;
    let pair = FireKey {
        program: binding.program,
        extents: binding.extents,
    };
    let mut roles = roles.lock().unwrap();
    let beside =
        roles.instance_parts_fresh(binding.program, &binding.channels, &registered, adapter);
    let guest = roles.take_guest(&binding.channels, (pair, state), holder, beside)?;
    let fresh = guest.fresh_runs();
    let mut taken = roles
        .take_instance(
            binding.program,
            &binding.channels,
            &registered,
            adapter,
            guest,
        )
        .map_err(|(error, opened)| roles.bind_short(error, fresh, opened))?;
    taken.holder = holder;
    roles.bind_taken_for(holder);
    roles.on_way_take(&mut taken);
    drop(roles);
    // Read again past the take: a park that began since the first read
    // either saw this bind on its way and waits for it, or parked before
    // it was taken, and then the holder reads parked here and the take
    // goes back.
    if let Some(holder) = holder.filter(|&holder| !resident(holder)) {
        roles_handle.lock().unwrap().untake(taken);
        return Err(RolesError::HolderParked(holder).into());
    }
    Ok(Some(taken))
}

/// Binds `binding` over what [`take_bind`] took for it (`taken`): the
/// engine's bind, and what it took kept for `holder`, or given back where
/// the engine refuses it.
pub fn bind_taken(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    taken: Option<Taken>,
) -> anyhow::Result<BoundInstance> {
    let (Some(roles), Some(taken)) = (roles, taken) else {
        return Ok(engine.bind_instance(binding)?);
    };
    let binding = InstanceBinding {
        instance_slots: taken.instance_slots(),
        ring_slots: taken.ring_slots(),
        adapter_slots: taken.adapter_slots(),
        scratch: taken.scratch(),
        pair: taken.pair(),
        pair_generation: taken.pair_generation(),
        ..binding.clone()
    };
    match engine.bind_instance(&binding) {
        Ok(bound) => {
            roles.lock().unwrap().bound(bound.id, holder, taken);
            Ok(bound)
        }
        Err(error) => {
            roles.lock().unwrap().untake(taken);
            Err(error.into())
        }
    }
}

/// Runs `bind`, a register+bind of process `pid`'s, until it lands or is
/// refused by name. Refused because a channel it binds is parked with
/// `pid` (suspended, its rings with it), it waits by `resident`, the wait a
/// fire takes for `pid` to be resident again (the planner's restore), and
/// binds again: a suspension holds a bind back as it holds a fire, and
/// never refuses it. A channel parks with `pid` only while `pid` is not
/// resident, so each bind again follows a restore. A wait that ends
/// otherwise (the process failed or ended) is the bind's refusal.
///
/// Short of a free run for its guest state where a segment still to
/// commit within the budget holds one ([`RolesError::GuestGrow`]), it
/// waits by `grown(prefix, run)` ([`grow_for_guest`]: the walk commits a
/// segment past the committed prefix `prefix`, as it does for a kv
/// admission short of room) and binds again; a wait that ends with no
/// segment left to commit binds again too, short of room the planner then
/// makes ([`RolesError::BindShort`]), never refused for it.
///
/// This is not a retry: nothing is attempted again on the chance it now
/// fits, and nothing is degraded or made smaller. The bind is sent again
/// only after the planner has restored `pid`, or the walk has committed a
/// segment, the event the refusal named, exactly as a fire held back by a
/// suspension is sent after its restore.
///
/// However the wait ends (bound, refused, or dropped with its process),
/// its need is no longer counted (`leave`, [`Roles::leave_guest_wait`]).
///
/// Before any wait, its process's fires already submitted are settled
/// (`settle`): their leases on its working sets go back as they land, as
/// they do at any other wait of the process's ([`crate::inferlet::process::gate`]).
/// Waiting with them unsettled, the process was held between an eviction
/// that waits for those leases and a bind that waits on that eviction.
/// A wait for room its own eviction withdrew
/// ([`crate::planner::BindRoomWait::Withdrawn`]) waits for its restore.
#[allow(clippy::too_many_arguments)]
pub async fn bind_when_resident<T, E, B, W, G, P, S>(
    pid: Holder,
    mut bind: impl FnMut() -> B,
    resident: impl Fn() -> W,
    grown: impl Fn(u32, u32) -> G,
    room: impl Fn() -> Option<RoomWatch>,
    fits: impl Fn(u32, u32, u32) -> bool,
    suspend: impl Fn(u32, u32, u32) -> bool,
    planned: impl Fn(u32, u32, u32) -> P,
    leave: impl Fn(),
    waiting: impl Fn(bool),
    settle: impl Fn() -> S,
) -> anyhow::Result<T>
where
    E: std::fmt::Display,
    B: std::future::Future<Output = anyhow::Result<T>>,
    W: std::future::Future<Output = Result<(), E>>,
    G: std::future::Future<Output = Result<(), String>>,
    P: std::future::Future<Output = Result<crate::planner::BindRoomWait, String>>,
    S: std::future::Future<Output = ()>,
{
    /// Runs its `leave` on drop: the wait's every end, its process gone
    /// (the future dropped) too.
    struct Leaves<'a, F: Fn()>(&'a F);
    impl<F: Fn()> Drop for Leaves<'_, F> {
        fn drop(&mut self) {
            (self.0)();
        }
    }
    let _leaves = Leaves(&leave);
    // Waiting for room, the process yields to the planner as an idle one
    // does (`waiting`); past the wait, however it ends, it no longer does.
    let waits = || waiting(true);
    struct Ends<'a, F: Fn(bool)>(&'a F);
    impl<F: Fn(bool)> Drop for Ends<'_, F> {
        fn drop(&mut self) {
            (self.0)(false);
        }
    }
    let _ends = Ends(&waiting);
    // Suspended for this bind's room once and restored: short again, it
    // waits parked for room the planner makes, never suspended and evicted
    // again (mx-2: suspend, evict, restore, short, every ~25 ms, 222,924
    // evictions and no serve).
    let mut restored_for_room = false;
    loop {
        // Taken before the bind: room given back meanwhile is not lost.
        let watch = room();
        // Binding again, it is no victim while the bind is under way.
        waiting(false);
        let error = match bind().await {
            Ok(bound) => return Ok(bound),
            Err(error) => error,
        };
        // Every way on from here waits, but a refusal: what the process
        // has in flight lands and gives its leases back first.
        let waits_on = error.downcast_ref::<RolesError>().is_some_and(|error| {
            matches!(
                error,
                RolesError::PairParked(_) | RolesError::GuestGrow { .. }
            ) || error.room_runs().is_some()
                || matches!(error, RolesError::Parked { holder: Some(holder), .. } if *holder == pid)
                || matches!(error, RolesError::HolderParked(holder) if *holder == pid)
        });
        if waits_on {
            settle().await;
        }
        if let Some(RolesError::PairParked(key)) = error.downcast_ref::<RolesError>()
            && let Some(watch) = &watch
        {
            tracing::debug!(%pid, %key, "a bind waits for its pair's move to land");
            let freed = watch.clone().freed();
            crate::inferlet::process::said("bind: its pair's move", pid, async move {
                // Waiting on its pair's move: its owner is parked here.
                let _parked = crate::inferlet::process::parks_here();
                freed.await
            })
            .await;
            continue;
        }
        if let Some((run, pieces, singles)) = error
            .downcast_ref::<RolesError>()
            .and_then(RolesError::room_runs)
        {
            // Holding guest state of its own, it would wait on itself: it is
            // suspended, its state given back, and binds once restored with
            // room for this bind (`run`, `pieces`) beside its own.
            if !restored_for_room && suspend(run, pieces, singles) {
                tracing::debug!(%pid, %error, "a bind short of room suspends its process");
                // Waiting for its restore, not for room: its need is not
                // counted meanwhile.
                leave();
                if let Err(why) = resident().await {
                    return Err(error.context(format!(
                        "its process was suspended for room and is not resident again: {why}"
                    )));
                }
                restored_for_room = true;
                continue;
            }
            // Else it waits, queued in the planner, for room: given back
            // (a close or a park, `watch`), or made for it as for a
            // restore short of room for its roles (`planned`: idle cache
            // dropped first, then a resident parked by the planner's
            // victim choice).
            // Room given back wakes it only where its runs (`fits`) now
            // stand free: it is sent again on the room it waits for, never
            // on any give.
            tracing::debug!(%pid, %error, "a bind waits in the planner for room");
            waits();
            let freed = async {
                let Some(mut watch) = watch else {
                    return std::future::pending().await;
                };
                loop {
                    let next = room();
                    watch.freed().await;
                    if fits(run, pieces, singles) {
                        return;
                    }
                    match next {
                        Some(next) => watch = next,
                        None => return std::future::pending().await,
                    }
                }
            };
            tokio::select! {
                // In order, never at random: a seeded run takes the same branch.
                biased;
                () = freed => {}
                made = planned(run, pieces, singles) => match made {
                    Err(why) => {
                        return Err(error.context(format!("its wait for room ended: {why}")));
                    }
                    Ok(crate::planner::BindRoomWait::Served) => {}
                    Ok(crate::planner::BindRoomWait::Withdrawn) => {
                        tracing::debug!(%pid, "a bind's wait for room ends with its eviction");
                        leave();
                        waiting(false);
                        if let Err(why) = resident().await {
                            return Err(error.context(format!(
                                "its process was evicted and is not resident again: {why}"
                            )));
                        }
                    }
                },
            }
            continue;
        }
        if let Some(RolesError::GuestGrow {
            prefix,
            run,
            room: short,
            ..
        }) = error.downcast_ref::<RolesError>()
        {
            let (prefix, run) = (*prefix, *run);
            let (pieces, singles) = (short.chunk_runs().1, short.singles());
            tracing::debug!(%pid, %error, "a bind waits on a segment commit");
            waits();
            // A commit, room given back (its runs may stand free in it), or
            // the planner's serve: any ends the wait, and it binds again.
            // The planner is asked too, as for a bind short of room
            // (`planned`): where the governor's target stops the walk
            // committing after all, it is the only one that makes room
            // (idle cache dropped, a resident parked), and it parks no one
            // for a grow that is coming.
            let freed = async {
                match watch {
                    Some(watch) => watch.freed().await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                // In order, never at random: a seeded run takes the same branch.
                biased;
                // No segment left to commit within the budget by the time
                // it looked (the room the refusal counted was committed
                // for others, or the target came down): the room is the
                // planner's to make now, never the client's refusal. It
                // binds again, and a bind short of room then waits in the
                // planner (`BindShort`); only a need that can never fit
                // alone is refused (a read of "no segment is left to
                // commit" beside one of "the segments left ... hold more"
                // is two reads at two instants, not a refusal).
                grew = grown(prefix, run) => {
                    if let Err(why) = grew {
                        tracing::debug!(%pid, %why, "a bind's commit will not come: the planner makes its room");
                    }
                }
                () = freed => {}
                made = planned(run, pieces, singles) => match made {
                    Err(why) => {
                        return Err(error.context(format!("its wait for room ended: {why}")));
                    }
                    Ok(crate::planner::BindRoomWait::Served) => {}
                    Ok(crate::planner::BindRoomWait::Withdrawn) => {
                        leave();
                        waiting(false);
                        if let Err(why) = resident().await {
                            return Err(error.context(format!(
                                "its process was evicted and is not resident again: {why}"
                            )));
                        }
                    }
                },
            }
            continue;
        }
        let parked_with_it = matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::Parked { holder: Some(holder), .. } | RolesError::HolderParked(holder))
                if *holder == pid
        );
        if !parked_with_it {
            return Err(error);
        }
        tracing::debug!(%pid, %error, "a bind waits for its process to be resident again");
        if let Err(why) = resident().await {
            return Err(error.context(format!(
                "its process was suspended and is not resident again: {why}"
            )));
        }
    }
}

/// Waits until the walk of `engine`'s pool `shared` commits a segment past
/// the committed prefix `prefix`, for a guest bind short of a free run of
/// `run` chunks ([`RolesError::GuestGrow`]): the walk is asked to look
/// ([`crate::store::shared::ask_walk`]) and the wait is on the event a
/// commit tells ([`crate::store::shared::rung_watch`]), never a clock; the
/// walk's goal counts what the bind is short ([`GUEST_WAITER`]), within the
/// governor's target and yield, as it counts a kv admission's. Refused
/// (`Err`) once no segment is left to commit within the budget, the
/// pool's count of it cleared.
pub async fn grow_for_guest(
    shared: &Shared,
    engine: usize,
    prefix: u32,
    run: u32,
) -> Result<(), String> {
    loop {
        // Armed before the look: a commit between the two is not lost.
        let watch = crate::store::shared::rung_watch(engine);
        tokio::pin!(watch);
        watch.as_mut().enable();
        {
            let mut pool = shared.lock().unwrap();
            if pool.pool().prefix_segments() > prefix {
                return Ok(());
            }
            if pool.pool().committable_runs(run) == 0 {
                pool.pool_mut().unwait(GUEST_WAITER);
                pool.set_guest_head(None);
                return Err(format!(
                    "no free run of {run} adjacent chunks, and no segment is left to commit \
                     within the budget"
                ));
            }
        }
        crate::store::shared::ask_walk(engine);
        // Waiting on a segment's commit: its owner is parked here.
        let _parked = crate::inferlet::process::parks_here();
        watch.await;
    }
}

/// Runs `write`, a host write into `instance`'s rings, unless its slots
/// are away ([`Roles::writes_held`]): then it is held, to be replayed in
/// order once they are rebound ([`rebind`]).
pub fn pump(
    roles: Option<&RolesHandle>,
    instance: InstanceId,
    pump: Pump,
    write: impl FnOnce() -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    if let Some(roles) = roles {
        let mut roles = roles.lock().unwrap();
        if roles.writes_held(instance) {
            roles.hold(instance, pump);
            return Ok(());
        }
    }
    write()
}

/// Tells `engine` where moved slots lie now (`rebind`), then replays, in
/// order, the host writes held for the instances that may write again.
/// Before any later fire: the lane runs it between frames.
pub fn rebind(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    rebind: Rebind,
    mut replay: impl FnMut(&mut dyn Engine, InstanceId, Pump) -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    let owner = Owner::of(&rebind);
    // Run on the lane, after any close the lane ran since the restore read
    // its rebinds: a closed instance is the engine's no longer, and the
    // runtime never rebinds what it closed.
    if let Some(roles) = roles
        && !roles.lock().unwrap().rebind_stands(&rebind)
    {
        tracing::debug!(
            ?owner,
            "a rebind cancelled: its item closed or moved since it was read"
        );
        return Ok(());
    }
    let parked = matches!(rebind, Rebind::Parked { .. });
    engine.rebind_slots(rebind)?;
    // A park names no new slots: nothing is rebound, no held write goes.
    if parked {
        return Ok(());
    }
    let Some(roles) = roles else {
        return Ok(());
    };
    let released = {
        let mut roles = roles.lock().unwrap();
        roles.rebound(owner);
        roles.released()
    };
    for (instance, pump) in released {
        replay(engine, instance, pump)?;
    }
    Ok(())
}

/// An instance's own rings as the engine names them: channel `rings[k]` in
/// part `k + 1` (part 0 the tables).
pub(super) fn own_rings(rings: &[u32], parts: &[Part]) -> Vec<RingSlots> {
    rings
        .iter()
        .zip(parts.iter().skip(1))
        .map(|(&channel, part)| RingSlots {
            channel,
            slots: part.refs.clone(),
        })
        .collect()
}
