//! The shell's guest programs and adapters: registration, prewarm, bind and
//! rebind of instances, shared channels, and adapter banks and landing.

use super::*;

impl Shell {

    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<engine::ProgramId> {
        self.programs.register(&self.device, registration)
    }

    /// The runtime's prewarm: each of `registrations` (its standard
    /// samplers) compiled into the program cache before ready, so a request
    /// that registers one builds nothing.
    pub fn prewarm_programs(
        &mut self,
        registrations: &[engine::program::ProgramRegistration],
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let steps_before = crate::program::compile::step_pipelines_built();
        let compiled = self.programs.prewarm(&self.device, registrations)?;
        tracing::info!(
            "engine-metal: guest regions prewarmed for {} standard program(s): {} compiled \
             ({} step pipeline(s)), in {} ms",
            registrations.len(),
            compiled,
            crate::program::compile::step_pipelines_built().saturating_sub(steps_before),
            started.elapsed().as_millis()
        );
        Ok(())
    }

    /// The guest region modules the program cache has compiled.
    #[must_use]
    pub fn program_compilations(&self) -> u64 {
        self.programs.stats().compilations
    }

    /// The wall time every guest region compile since load took, in
    /// nanoseconds: an OS shader-cache hit reads apart from a cold compile.
    #[must_use]
    pub fn program_compile_nanos(&self) -> u64 {
        self.programs.compile_nanos()
    }

    /// See `Engine::prepare_program`: the registration's plan, on this
    /// thread; [`crate::program::Registering::compile`] builds its libraries
    /// and pipelines on any thread, while this shell fires on.
    pub fn prepare_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<crate::program::Registering> {
        self.programs.prepare_register(&self.device, registration)
    }

    /// See `Engine::finish_program`: the compiled registration taken in.
    pub fn finish_program(
        &mut self,
        registered: crate::program::Registered,
    ) -> Result<engine::ProgramId> {
        self.programs.finish_register(registered)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_program(
        &mut self,
        program_id: engine::ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[engine::ChannelId],
        instance_slots: &[u32],
        ring_slots: &[engine::RingSlots],
        scratch: &engine::ScratchPlace,
        pair: &engine::ScratchPlace,
        pair_generation: u64,
    ) -> Result<engine::InstanceId> {
        let pooled = crate::program::session::Pooled {
            pools: &self.pools,
            instance: instance_slots,
            rings: ring_slots,
        };
        // Its state and its pair's lie in the `Instance` slots the runtime
        // took (`scratch`, `pair`): nothing of it in the span, nothing to
        // back there.
        let id = self.programs.bind(
            &self.device,
            program_id,
            seeds,
            extents,
            geometry,
            device,
            channels,
            pooled,
            scratch,
            pair,
            pair_generation,
        )?;
        if crate::diag::on().tier_trace {
            eprintln!(
                "residency: program {} bound at {extents:?}: its state and its pair's in \
                 `Instance` slots ({scratch:?}, {pair:?})",
                program_id.0
            );
        }
        Ok(id)
    }

    /// See `Engine::rebind_slots`: every frame lands first, so no fire in
    /// flight reads the old refs' addresses.
    pub fn rebind_slots(&mut self, rebind: &engine::Rebind) -> Result<()> {
        self.drain()?;
        if let engine::Rebind::Instance { id, adapter, .. } = rebind
            && !adapter.is_empty()
        {
            self.rebind_adapter(*id, adapter)?;
        }
        self.programs.rebind_slots(&self.pools, rebind)
    }

    /// What an instance of `program_id` holds in pool slots
    /// (`Engine::program_facts`).
    pub fn program_facts(&self, program_id: engine::ProgramId) -> Result<engine::ProgramFacts> {
        let mut facts = self.programs.facts(program_id, &self.pools)?;
        // The adapter its `lora` sink lands: a bank's plane an `Adapter`
        // slot, taken once a content key and shared by every holder.
        if let Some(banks) = self.adapter_banks(program_id)? {
            facts.adapter = engine::Footprint {
                bytes: banks.iter().map(|bank| bank.slot).sum(),
                slots: u32::try_from(banks.len()).unwrap_or(u32::MAX),
                contiguous: false,
            };
        }
        Ok(facts)
    }

    /// The `Channel` slots a registered channel's ring of `shape` takes.
    #[must_use]
    pub fn ring_facts(&self, shape: crate::program::ChannelShape) -> Option<engine::RingFacts> {
        let channel = self.pools.layout().channel?;
        crate::program::ring_facts(0, shape, channel.slot_bytes, channel.per_chunk)
    }

    pub fn register_shared_channel(
        &mut self,
        id: engine::ChannelId,
        shape: crate::program::ChannelShape,
        slots: &[u32],
    ) -> Result<()> {
        self.programs
            .register_channel(id, shape, &self.pools, slots)
    }

    pub fn close_shared_channel(&mut self, id: engine::ChannelId) -> bool {
        self.programs.close_channel(id)
    }

    pub fn program_instance(
        &mut self,
        instance_id: engine::InstanceId,
    ) -> Result<Option<&mut crate::program::Session>> {
        self.fence_instances(&[instance_id], false)?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// The session the host takes from `channel` of: the frames that landed
    /// harvested (their commits in), and, where the instance runs ahead and
    /// its program never takes from that channel, its passes still in
    /// flight left flying ([`crate::program::Plane::takes_ahead`]): what it
    /// takes is a committed cell, which a pass in flight does not touch
    /// (it puts past it, and its commit moves the tail). The runtime takes
    /// on each landing; waiting out the next pass there held a lane to one
    /// frame in flight. Any other access waits out every pass of it.
    pub fn program_instance_taking(
        &mut self,
        instance_id: engine::InstanceId,
        channel: u32,
    ) -> Result<Option<&mut crate::program::Session>> {
        if self.programs.takes_ahead(instance_id, channel) {
            self.reap()?;
            return Ok(self.programs.instance_mut(instance_id));
        }
        self.program_instance(instance_id)
    }

    /// Frames in flight carrying instance `id` (tests: what a take left
    /// flying).
    #[must_use]
    pub fn frames_carrying(&self, id: engine::InstanceId) -> usize {
        self.inflight
            .iter()
            .filter(|flight| flight.attached.iter().any(|(held, _)| *held == id))
            .count()
    }

    /// Releases a registered program no instance binds.
    pub fn close_program(&mut self, id: engine::ProgramId) -> Result<()> {
        self.programs.close_program(id)
    }

    pub fn close_program_instance(&mut self, instance_id: engine::InstanceId) -> Result<()> {
        // A close goes through whatever its fence met: a frame of the
        // instance whose buffer failed has let its pass go at that harvest,
        // and nothing of it is left to wait for.
        // Each refusal is a harvested frame, so this ends.
        while let Err(fault) = self.fence_instances(&[instance_id], false) {
            tracing::warn!(instance = %instance_id, %fault, "closing an instance past a failed frame");
        }
        self.programs.close_instance(instance_id)?;
        Ok(())
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.seats()
    }

    pub fn program_adapter_sink(
        &self,
        program_id: engine::ProgramId,
    ) -> Result<Option<crate::adapter::Sink>> {
        let program = self.programs.program(program_id).ok_or_else(|| {
            Fault::program(
                "serve::shell",
                format!("no program {program_id} to read an adapter sink off"),
            )
        })?;
        crate::adapter::sink_of(&program.plan.package)
    }

    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.blobs.mount(root);
    }

    /// The banks the adapter a program's `lora` sink lands fills, in
    /// bank-name order: one `Adapter` slot each. None: it lands none.
    pub fn adapter_banks(
        &self,
        program_id: engine::ProgramId,
    ) -> Result<Option<Vec<crate::weights::BankSeat>>> {
        let Some(sink) = self.program_adapter_sink(program_id)? else {
            return Ok(None);
        };
        let seats = self.weights.seats();
        let site = sink.site()?;
        let mut banks: Vec<crate::weights::BankSeat> = Vec::new();
        for (role, _) in &sink.planes {
            for bank in crate::adapter::banks_for(*role, site, &seats)? {
                if !banks.iter().any(|seen| seen.name == bank.name) {
                    banks.push(bank.clone());
                }
            }
        }
        banks.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Some(banks))
    }

    /// Lands the adapter `instance` carries (`planes`, a bank's plane each)
    /// in the `Adapter` slots the runtime took for its item (`slots`, in
    /// bank-name order): where another instance landed the same item there,
    /// it joins it and nothing is written. Its adapter id: what its lanes
    /// route by.
    pub fn land_adapter(
        &mut self,
        instance: engine::InstanceId,
        slots: &[u32],
        mut planes: Vec<(String, Vec<u8>)>,
    ) -> Result<u32> {
        planes.sort_by(|a, b| a.0.cmp(&b.0));
        if slots.len() != planes.len() || slots.is_empty() {
            return Err(Fault::Adapter {
                bank: planes
                    .first()
                    .map_or_else(String::new, |(name, _)| name.clone()),
                why: format!(
                    "lands {} planes and the runtime took {} `Adapter` slots for it; the \
                     facts state one slot a bank",
                    planes.len(),
                    slots.len()
                ),
            });
        }
        if let Some(landing) = self.landings.get_mut(&slots[0]) {
            landing.holders.insert(instance);
            self.held_landing.insert(instance, slots[0]);
            return Ok(landing.id);
        }
        let id = (0..crate::weights::ADAPTER_IDS)
            .find(|id| !self.landings.values().any(|landing| landing.id == *id))
            .ok_or(Fault::Ceiling {
                what: "adapter ids a bank's table holds",
                need: u64::from(crate::weights::ADAPTER_IDS) + 1,
                have: u64::from(crate::weights::ADAPTER_IDS),
            })?;
        let banks: Vec<String> = planes.iter().map(|(name, _)| name.clone()).collect();
        for ((bank, bytes), &slot) in planes.iter().zip(slots) {
            let (_, _, runs, slot_bytes) =
                self.pools.slot_runs(crate::store::layout::ADAPTER, slot)?;
            if bytes.len() as u64 > slot_bytes {
                return Err(Fault::Adapter {
                    bank: bank.clone(),
                    why: format!(
                        "a plane of {} bytes past an `Adapter` slot of {slot_bytes}",
                        bytes.len()
                    ),
                });
            }
            // The plane, zero past its bytes, through the slot's host runs
            // (a run a heap it crosses); no frame reads a slot nothing
            // routes to yet.
            for (from, len, host) in runs {
                let (from, len) = (
                    usize::try_from(from).unwrap_or(usize::MAX),
                    usize::try_from(len).unwrap_or(usize::MAX),
                );
                let have = bytes.len().saturating_sub(from).min(len);
                // SAFETY: `host` addresses `len` mapped bytes of the slot.
                #[allow(unsafe_code)]
                unsafe {
                    std::ptr::write_bytes(host.as_ptr(), 0, len);
                    if have > 0 {
                        std::ptr::copy_nonoverlapping(bytes[from..].as_ptr(), host.as_ptr(), have);
                    }
                }
            }
        }
        let landing = Landing {
            id,
            refs: slots.to_vec(),
            banks,
            holders: std::iter::once(instance).collect(),
        };
        self.point_landing(&landing)?;
        self.landings.insert(slots[0], landing);
        self.held_landing.insert(instance, slots[0]);
        Ok(id)
    }

    /// Points a landing's id at its slots in every bank's table.
    pub(super) fn point_landing(&mut self, landing: &Landing) -> Result<()> {
        for (bank, &slot) in landing.banks.iter().zip(&landing.refs) {
            let (buffer, offset, _, _) =
                self.pools.slot_place(crate::store::layout::ADAPTER, slot)?;
            let address = buffer.address_at(offset).ok_or_else(|| Fault::Unbound {
                what: format!("a GPU address for `Adapter` slot {slot}"),
            })?;
            self.weights.point(bank, landing.id, Some(address))?;
        }
        Ok(())
    }

    /// `instance` lets go of its adapter: the last holder's close forgets
    /// the landing and its id (the runtime frees the slots).
    pub fn detach_adapter(&mut self, instance: engine::InstanceId) {
        let Some(first) = self.held_landing.remove(&instance) else {
            return;
        };
        if let Some(landing) = self.landings.get_mut(&first) {
            landing.holders.remove(&instance);
            if landing.holders.is_empty() {
                // Forgotten: its id reads the zero plane until it lands anew.
                if let Some(landing) = self.landings.remove(&first) {
                    for bank in &landing.banks {
                        let _ = self.weights.point(bank, landing.id, None);
                    }
                }
            }
        }
    }

    /// `instance`'s adapter now lies in `slots` (parked and back, or moved):
    /// the item's bytes copied there by the runtime, its id pointed there.
    /// Every holder of one item names the same new refs; the first told
    /// moves the landing, the rest find it moved.
    pub fn rebind_adapter(&mut self, instance: engine::InstanceId, slots: &[u32]) -> Result<()> {
        let Some(&first) = self.held_landing.get(&instance) else {
            return Ok(());
        };
        let Some(&new_first) = slots.first() else {
            return Ok(());
        };
        if first == new_first {
            return Ok(());
        }
        if let Some(mut landing) = self.landings.remove(&first) {
            landing.refs = slots.to_vec();
            self.point_landing(&landing)?;
            for holder in &landing.holders {
                self.held_landing.insert(*holder, new_first);
            }
            self.landings.insert(new_first, landing);
        } else {
            self.held_landing.insert(instance, new_first);
        }
        Ok(())
    }

    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        self.compiled
            .classes
            .adapted_word(&self.corrected, bit, word)
    }

    #[must_use]
    pub fn copies(&self) -> bool {
        self.copies
    }

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
    }
}
