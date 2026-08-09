//! Registration: programs, channels, instances, and closing them.
//!
//! All four are the registry's, and what is here is the conversion between
//! the ABI's records and the registry's own — field for field the same facts
//! under two spellings.

use crate::error::Result;
use crate::gpu::serve::state::Shell;

impl Shell {
    /// Register a PTIR program: its launch package and whatever kernels the
    /// host generated for it.
    ///
    /// Memoised by hash inside the registry, so a re-registration costs a
    /// lookup — which is the caller's assumption and not an optimisation
    /// added here.
    ///
    /// # Errors
    ///
    /// A package the registry refuses (a channel whose shape it cannot serve,
    /// a stage it cannot read).
    pub fn register_program(&mut self, desc: &driver_api::ProgramRegistration) -> Result<u64> {
        Ok(self.registry.register_program(
            desc.program_hash,
            desc.launch.clone(),
            // Field for field the same record under two names: the ABI's
            // and the registry's. Converted rather than aliased, because
            // the registry's is the one it validates against and a type
            // alias would let a future field diverge silently.
            desc.emitted_kernels
                .iter()
                .map(|k| crate::channel::EmittedKernel {
                    kind: k.kind,
                    stage_index: k.stage_index,
                    region_index: k.region_index,
                    entry_name: k.entry_name.clone(),
                    source: k.source.clone(),
                    error: k.error.clone(),
                })
                .collect(),
        )?)
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory on this backend, exactly as it is on the dummy
    /// driver: `ChannelState` holds the cells and four control words, and the
    /// binding is their addresses. Nothing about the channel plane is on the
    /// GPU — it is a different layer from the model forward and always has
    /// been (`.wiki/driver/progress-metal.md`).
    ///
    /// # Errors
    ///
    /// A shape or dtype the registry will not serve, or a duplicate id.
    pub fn register_channel(
        &mut self,
        desc: &driver_api::ChannelRegistrationPlan,
    ) -> Result<driver_api::PieChannelEndpointBinding> {
        let spec = crate::channel::ChannelSpec {
            id: desc.channel_id,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            capacity: desc.capacity,
            role: crate::channel::HostRole::from_wire(desc.host_role),
            seeded: desc.seeded,
            direction: crate::channel::Direction::from_wire(desc.extern_dir),
            extern_name: desc.extern_name.clone(),
        };
        let endpoint = self.registry.register_channel(spec)?;
        Ok(driver_api::PieChannelEndpointBinding {
            channel_id: endpoint.channel_id,
            mirror_base: endpoint.mirror_base,
            word_base: endpoint.word_base,
            mirror_bytes: endpoint.mirror_bytes as u64,
            word_bytes: endpoint.word_bytes as u64,
            cell_bytes: endpoint.cell_bytes,
            capacity: endpoint.capacity,
            // The ABI's order, and `ChannelState`'s: head, tail, poison,
            // closed. Stated here as constants because the two sides index
            // the same four words and neither can move without the other.
            head_word_index: 0,
            tail_word_index: 1,
            poison_word_index: 2,
            closed_word_index: 3,
        })
    }

    /// Attach an instance of a registered program to its channels.
    ///
    /// `requested` is the id the caller wants, or `None` to be assigned one.
    /// `seeds` are the channel values an instance starts with.
    ///
    /// # Errors
    ///
    /// A program id the registry does not hold, a channel an instance may not
    /// attach to, or a geometry class it does not serve.
    pub fn bind_instance(
        &mut self,
        program_id: u64,
        requested: Option<u64>,
        geometry_class: u32,
        channel_ids: &[u64],
        seeds: &[(u64, Vec<u8>)],
    ) -> Result<driver_api::PieInstanceBinding> {
        let geometry = crate::channel::Geometry::from_wire(geometry_class)?;
        let instance_id =
            self.registry
                .bind_instance(program_id, requested, geometry, channel_ids, seeds)?;
        Ok(driver_api::PieInstanceBinding {
            instance_id,
            geometry_class,
            reserved0: 0,
        })
    }

    /// Release an instance.
    ///
    /// A close of an id the registry does not hold is not an error: the
    /// scheduler's close is idempotent from its side.
    pub fn close_instance(&mut self, id: u64) {
        // Discarded deliberately: the registry answers `Err` for an id it does
        // not hold, and a double close is how a teardown race reads from this
        // side rather than a fault.
        let _ = self.registry.close_instance(id);
    }

    /// Release a channel.
    ///
    /// Idempotent for the same reason [`Self::close_instance`] is.
    pub fn close_channel(&mut self, id: u64) {
        let _ = self.registry.close_channel(id);
    }
}
