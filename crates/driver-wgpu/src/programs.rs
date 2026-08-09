//! Programs, channels and instances: the engine's registry verbs.
//!
//! # What is here and what is not
//!
//! Almost nothing, and that is the finding. Five of the fourteen verbs the
//! engine's seam calls are registration, and none of them touches a device:
//! a program is a launch package adopted into a plan, a channel is a ring in
//! host memory with four control words, and an instance is a binding between
//! them. The `driver` crate owns all of it, takes the workspace lints in
//! full, and both the CUDA and Metal shells already read it.
//!
//! So this module is the CONVERSION between the ABI's records and the
//! registry's -- field for field the same facts under two spellings -- and
//! nothing else. `driver-metal`'s `serve/register.rs` says the same in the
//! same shape, and the two agreeing is not duplication: each is its own
//! crate's statement of which ABI record maps to which registry record, and
//! an alias would let a field diverge in one and not the other.
//!
//! # Why the registry is a real dependency here
//!
//! `crates/model` and `crates/model-loader` are NOT dependencies of this crate
//! at all, not even dev ones -- a driver that depended on a checkpoint FORMAT
//! would be a driver that could not be handed bytes -- and `tests/pure.rs`
//! holds the closure to it. The `driver` crate is different in kind: it is not
//! a format, it is the plane the engine registers work on, and serving those
//! verbs is the driver's job rather than a caller's. It costs the purity guard
//! nothing, since its own closure is `driver-api` and `tensor-ir`, both
//! already admitted -- and only because this crate's edge to it is
//! `default-features = false`, since `driver-api`'s `rpc` feature reaches
//! `js-sys` and `wasm-bindgen` by way of tokio.
//!
//! # What a program does NOT do yet
//!
//! Run. Registration is bookkeeping the engine needs before it can bind an
//! instance, and it is complete; EXECUTING a PTIR program -- the reference
//! pass over its stages -- is `driver`'s `step` and is not wired to a command
//! encoder here. A frame that carries programs launches its model rows and
//! reports the instance faults it was given, which is what the shell's `step`
//! will serve; a frame that expects its program to have computed something
//! would be wrong, and that is stated rather than silently half-served.

use driver::{ChannelSpec, Direction, EmittedKernel, Geometry, HostRole, Registry};

/// The program, channel and instance registry, plus the conversions.
///
/// A newtype over [`Registry`] rather than a re-export, so that the
/// conversions have somewhere to live and a caller cannot reach past them
/// into the registry's own spelling of a record.
#[derive(Debug, Default)]
pub struct Programs {
    registry: Registry,
}

/// Why a registration was refused.
///
/// One variant, because the registry answers one error kind and inventing
/// more here would be this module claiming to know things it was told.
#[derive(Debug)]
pub struct Unregistered(pub String);

impl std::fmt::Display for Unregistered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Unregistered {}

impl Programs {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a PTIR program: its launch package and whatever kernels the
    /// host generated for it.
    ///
    /// Memoised by hash inside the registry, so a re-registration is a lookup
    /// -- the caller's assumption, not an optimisation added here.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a package the registry refuses: no stages, a
    /// channel whose shape it cannot serve, a stage it cannot read.
    pub fn register_program(
        &mut self,
        desc: &driver_api::ProgramRegistration,
    ) -> Result<u64, Unregistered> {
        self.registry
            .register_program(
                desc.program_hash,
                desc.launch.clone(),
                // Field for field the same record under two names, converted
                // rather than aliased: the registry validates against its
                // own, and an alias would let a future field diverge with
                // nothing to notice.
                desc.emitted_kernels
                    .iter()
                    .map(|k| EmittedKernel {
                        kind: k.kind,
                        stage_index: k.stage_index,
                        region_index: k.region_index,
                        entry_name: k.entry_name.clone(),
                        source: k.source.clone(),
                        error: k.error.clone(),
                    })
                    .collect(),
            )
            .map_err(|e| Unregistered(format!("{e}")))
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory, as it is on Metal and on the dummy driver.
    /// Nothing about the channel plane is on the GPU -- it is a different
    /// layer from the model forward, and a device buffer for it would be a
    /// mapped round trip per cell for data no shader reads. That argument is
    /// if anything stronger here: a WebGPU buffer is read back by mapping it
    /// and awaiting the queue, so a host ring on the device would cost a
    /// submission per poll.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a shape or dtype the registry will not serve, or
    /// a duplicate id.
    pub fn register_channel(
        &mut self,
        desc: &driver_api::ChannelRegistrationPlan,
    ) -> Result<driver_api::PieChannelEndpointBinding, Unregistered> {
        let spec = ChannelSpec {
            id: desc.channel_id,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            capacity: desc.capacity,
            role: HostRole::from_wire(desc.host_role),
            seeded: desc.seeded,
            direction: Direction::from_wire(desc.extern_dir),
            extern_name: desc.extern_name.clone(),
        };
        let endpoint = self
            .registry
            .register_channel(spec)
            .map_err(|e| Unregistered(format!("{e}")))?;
        Ok(driver_api::PieChannelEndpointBinding {
            channel_id: endpoint.channel_id,
            mirror_base: endpoint.mirror_base,
            word_base: endpoint.word_base,
            mirror_bytes: endpoint.mirror_bytes as u64,
            word_bytes: endpoint.word_bytes as u64,
            cell_bytes: endpoint.cell_bytes,
            capacity: endpoint.capacity,
            // The ABI's order and `ChannelState`'s: head, tail, poison,
            // closed. Constants because the two sides index the same four
            // words and neither can move without the other.
            head_word_index: 0,
            tail_word_index: 1,
            poison_word_index: 2,
            closed_word_index: 3,
        })
    }

    /// Attach an instance of a registered program to its channels.
    ///
    /// `requested` is the id the caller wants, or `None` to be assigned one.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a program id the registry does not hold, a
    /// channel an instance may not attach to, or a geometry class it does not
    /// serve.
    pub fn bind_instance(
        &mut self,
        program_id: u64,
        requested: Option<u64>,
        geometry_class: u32,
        channel_ids: &[u64],
        seeds: &[(u64, Vec<u8>)],
    ) -> Result<driver_api::PieInstanceBinding, Unregistered> {
        let geometry =
            Geometry::from_wire(geometry_class).map_err(|e| Unregistered(format!("{e}")))?;
        let instance_id = self
            .registry
            .bind_instance(program_id, requested, geometry, channel_ids, seeds)
            .map_err(|e| Unregistered(format!("{e}")))?;
        Ok(driver_api::PieInstanceBinding {
            instance_id,
            geometry_class,
            reserved0: 0,
        })
    }

    /// Release an instance.
    ///
    /// A close of an id the registry does not hold is NOT an error, and the
    /// signature says so by not returning one: the scheduler's close is
    /// idempotent from its side, and a double close is how a teardown race
    /// reads here rather than a fault.
    pub fn close_instance(&mut self, id: u64) {
        let _ = self.registry.close_instance(id);
    }

    /// Release a channel. Idempotent for the same reason
    /// [`Self::close_instance`] is.
    pub fn close_channel(&mut self, id: u64) {
        let _ = self.registry.close_channel(id);
    }

    /// The registry itself, for a caller that runs a program's stages.
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A channel's four control words are at the indices the ABI names.
    ///
    /// # Why this is worth a test
    ///
    /// They are literals in this file, and the only other statement of them
    /// is in the engine's reader. A ring whose head and tail were swapped
    /// still runs -- it reports a full ring as empty and an empty one as
    /// full, which reads as a hang and not as a wrong index.
    ///
    /// The endpoint's own numbers are checked at the same time, since the
    /// binding is what the caller maps: a `mirror_bytes` that did not cover
    /// `capacity` cells would be a caller writing past the ring.
    #[test]
    fn a_registered_channel_reports_a_ring_the_caller_can_map() {
        let mut programs = Programs::new();
        let binding = programs
            .register_channel(&driver_api::ChannelRegistrationPlan {
                channel_id: 7,
                dtype: driver_api::PIE_CHANNEL_DTYPE_F32,
                shape: vec![2, 3],
                capacity: 4,
                host_role: driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
                seeded: false,
                extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
                extern_name: Vec::new(),
                driver_id: 0,
                reader_wait_id: 0,
                writer_wait_id: 0,
            })
            .expect("a well-formed channel");

        assert_eq!(binding.channel_id, 7, "the caller's id, not a fresh one");
        assert_eq!(
            (
                binding.head_word_index,
                binding.tail_word_index,
                binding.poison_word_index,
                binding.closed_word_index
            ),
            (0, 1, 2, 3),
            "the control words moved, and a swapped head and tail reads as a \
             hang rather than as a wrong index"
        );
        assert_eq!(binding.cell_bytes, 2 * 3 * 4, "six f32s to a cell");
        assert_eq!(binding.capacity, 4);
        assert!(
            binding.mirror_bytes >= u64::from(binding.cell_bytes) * u64::from(binding.capacity),
            "the ring is smaller than the cells it says it holds, so a writer \
             that filled it would write past it"
        );

        // ...and the same id twice is refused rather than silently rebound,
        // which would leave the first holder writing into a ring nobody
        // reads.
        programs
            .register_channel(&driver_api::ChannelRegistrationPlan {
                channel_id: 7,
                dtype: driver_api::PIE_CHANNEL_DTYPE_F32,
                shape: vec![1],
                capacity: 1,
                host_role: driver_api::PIE_CHANNEL_HOST_ROLE_NONE,
                seeded: false,
                extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
                extern_name: Vec::new(),
                driver_id: 0,
                reader_wait_id: 0,
                writer_wait_id: 0,
            })
            .expect_err("channel 7 is taken");
    }

    /// Closing something the registry never held is not an error.
    ///
    /// The scheduler closes from its own side without asking whether the
    /// driver still has it, so a teardown race reaches here as a close of an
    /// unknown id. Refusing it would make a normal shutdown log a fault per
    /// conversation.
    #[test]
    fn closing_what_was_never_registered_is_not_a_fault() {
        let mut programs = Programs::new();
        programs.close_instance(11);
        programs.close_channel(12);
        // ...and twice, which is the shape the race actually takes.
        programs.close_instance(11);
        programs.close_channel(12);
    }

    /// An instance of a program nobody registered is refused, by name.
    ///
    /// The alternative is a bound instance id that names nothing, which the
    /// engine would then launch: the fault would surface at the first stage,
    /// with a message about a plan rather than about a program id.
    #[test]
    fn an_instance_needs_a_program_that_exists() {
        let mut programs = Programs::new();
        let refused = programs
            .bind_instance(3, None, driver_api::PIE_GEOMETRY_CLASS_HOST, &[], &[])
            .expect_err("there is no program 3");
        assert!(
            format!("{refused}").contains('3'),
            "the refusal does not say which program: {refused}"
        );

        // A geometry class outside the three is refused too, and BEFORE the
        // program lookup would have accepted it -- so a caller gets the
        // reason that is actually wrong with the call.
        let refused = programs
            .bind_instance(3, None, 99, &[], &[])
            .expect_err("99 is not a geometry class");
        assert!(
            format!("{refused}").contains("99"),
            "the refusal does not say which class: {refused}"
        );
    }
}
