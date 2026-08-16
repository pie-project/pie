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
//! `crates/model` and `crates/model-loader` are dev-dependencies of this
//! crate on purpose -- a driver that depended on a checkpoint FORMAT could
//! not be handed bytes -- and `tests/pure.rs` holds the closure to it. The
//! `driver` crate is different in kind: it is not a format, it is the plane
//! the engine registers work on, and serving those verbs is the driver's job
//! rather than a caller's. It costs the purity guard nothing, since its own
//! closure is `driver-api` and `tensor-ir`, both already admitted -- though
//! only after this crate stopped pulling `driver-api`'s `rpc` feature in
//! through it, which reaches `js-sys` and `wasm-bindgen` by way of tokio.
//!
//! # What a program on this backend does and does not run
//!
//! It runs, on the HOST. [`Programs::fire`] is `driver`'s reference
//! interpreter over the instance's stages, and [`crate::frames::run_programs`]
//! is the loop a frame's launch drives it from -- so a frame that carries a
//! sampler now gets its token put on the channel the engine reads, which is
//! the half that was missing while this module was registration-only.
//!
//! It runs when the composer's decision still holds. Each fire is checked
//! against the ring words the scheduler pinned when it composed the batch --
//! see [`Programs::ready`] -- so a member whose ring moved between composition
//! and launch is skipped and re-posted rather than answering into somebody
//! else's cell, and one whose ring is closed or poisoned is faulted rather
//! than re-posted forever.
//!
//! What is NOT run is any program stage on the DEVICE. A registration may
//! carry emitted kernels for a backend that codegens them; this driver
//! advertises no codegen backend, holds them, and runs the stages the
//! interpreter can. That is the same shape `driver-metal` serves.
//!
//! # What a registration carries that this driver does not read
//!
//! Audited the way `frames::unserved_in`'s coverage was, and worth recording
//! because the answer is "nothing is silently dropped" and establishing that
//! took a probe.
//!
//! `ProgramRegistration` has six fields and this crate names three.
//! `emitted_kernels` and `region_analysis` are the other two that matter, and
//! both are EMPTY on every program of a full curated sweep -- the sentence
//! above about holding emitted kernels describes a path nothing currently
//! takes.
//!
//! `LaunchStagePlan` has fourteen and this crate names six. Of the eight it
//! does not: `signature_hash`, `source_ops`, `value_types`, `used_extents`,
//! `singleton` and `fused` describe how to EXECUTE a stage's ops, which is
//! the thing this driver does not do on the device; `mtp_rows` is zero on
//! every program measured. `channel_rules` is populated on every one of them
//! -- one to seventeen -- and its only consumer anywhere is
//! `driver-cuda/src/fire/lora.rs`.
//!
//! # Why ignoring the LoRA rules is safe, which is not obvious
//!
//! Not because this driver checks a capability. `PIE_REGION_SIG_LORA` rides
//! in `StepSubmission::region_sig` and this crate reads that field only in
//! test fixtures, so a region marked LoRA gets no refusal here.
//!
//! It is safe because a LoRA program cannot be LOWERED for this backend. The
//! kernel table states no LoRA symbol, so the model text cannot name one, and
//! a plan that named one anyway dies at `dispatch::row` with
//! `Undispatchable::Unknown` -- by symbol, with `a_symbol_no_row_states_is_
//! refused_by_name` as its control. The enforcement is the TABLE, one layer
//! down, and not a capability test in this file.
//!
//! That is worth knowing before adding one: a capability check here would be a
//! second gate on the same fact, and the two could disagree.

use driver::{ChannelSpec, Direction, EmittedKernel, Geometry, HostRole, Registry};

/// The program, channel and instance registry, plus the conversions.
///
/// A newtype over [`Registry`] rather than a re-export, so that the
/// conversions have somewhere to live and a caller cannot reach past them
/// into the registry's own spelling of a record.
#[derive(Debug, Default)]
pub struct Programs {
    registry: Registry,
    /// What each registered program does to each of its channels, by program
    /// id and dense channel slot.
    ///
    /// Derived once at registration rather than per fire, as `driver-metal`
    /// derives it: it is a property of the package -- which channel it takes
    /// from and which it puts to -- and re-deriving it on every fire would
    /// walk every op of every stage per request per step.
    effects: std::collections::BTreeMap<u64, Vec<driver::Effect>>,
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

/// What a registered program does to each of its channels.
///
/// The ports are folded in beside the stages, because a descriptor port
/// touches a channel too -- a consuming port takes and a peeking one reads --
/// and a fire gated only on its STAGES would be gated on half its effects.
/// `driver-metal` assembles the same two lists for the same call and this
/// mirrors it deliberately: two spellings of one derivation is how the two
/// backends stop agreeing.
///
/// The take/read distinction is spelled out and then does not matter:
/// [`driver::channel_effects`] folds both into one "reads this ring" flag,
/// because a ring that is only peeked at still has to hold the cell being
/// peeked at. It is written the long way anyway so that the day the two are
/// told apart, this side already says which port is which.
fn effects_of(program: &driver::Program) -> Result<Vec<driver::Effect>, Unregistered> {
    let package = &program.plan.package;
    let ports: Vec<driver_api::plan::LaunchOp> = package
        .ports
        .iter()
        .filter(|port| !port.is_const)
        .map(|port| driver_api::plan::LaunchOp {
            code: if driver::port_consumes(port.port) {
                u16::from(driver::tensor_ir::op::tags::CHAN_TAKE)
            } else {
                u16::from(driver::tensor_ir::op::tags::CHAN_READ)
            },
            channel: port.channel,
            ..driver_api::plan::LaunchOp::default()
        })
        .collect();
    let identity: Vec<u32> = (0..package.channels.len() as u32).collect();
    let mut stages: Vec<(&[driver_api::plan::LaunchOp], &[u32])> = package
        .plans
        .iter()
        .map(|plan| (plan.ops.as_slice(), plan.channel_bindings.as_slice()))
        .collect();
    stages.push((ports.as_slice(), identity.as_slice()));
    driver::channel_effects(&package.channels, &stages)
        .map_err(|bad| Unregistered(format!("channel {}: {}", bad.channel, bad.reason())))
}

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
        let id = self
            .registry
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
            .map_err(|e| Unregistered(format!("{e}")))?;
        // Derived HERE and not at fire time, and derived from the adopted
        // plan rather than from the caller's record, so that a program the
        // registry accepted and one this crate can gate are the same set.
        let effects =
            effects_of(self.registry.program(id).ok_or_else(|| {
                Unregistered(format!("program {id} was registered and is gone"))
            })?)?;
        self.effects.insert(id, effects);
        Ok(id)
    }

    /// Whether an instance may fire right now, against tickets the composer
    /// pinned.
    ///
    /// # What a ticket is, and why ignoring one is not safe
    ///
    /// A ticket is the head and tail a channel HAD when the scheduler
    /// composed the batch. It is not a readiness condition of its own -- it
    /// is the check that the ring has not moved underneath a decision already
    /// made. A fire composed against `head = 4` that arrives to find `head =
    /// 5` has had its cell taken by somebody else, and firing it consumes the
    /// wrong one.
    ///
    /// The interpreter's own gate cannot see this. It asks whether the ring
    /// is empty or full, which is a question about how MANY cells there are,
    /// and a ring that moved by one take and one put is neither empty nor
    /// full and holds a different cell. So a driver that skipped this check
    /// would answer from the wrong cell, fluently, in exactly the batches
    /// where the scheduler was right to be careful.
    ///
    /// # What a defaulted table would do, and why one is not supplied
    ///
    /// `tickets` must be one per channel of the instance; a shorter table is
    /// a layout mismatch and is answered as one. It is tempting to fill a
    /// missing table with unpinned entries and let the poison, closed, empty
    /// and full checks run anyway -- but a fire that PUTS and finds no pinned
    /// tail is [`driver::Reason::Unpinned`], which is a retry, forever. A
    /// defaulted table is therefore not a permissive default, it is a
    /// deadlock, and the caller decides instead: see
    /// [`crate::frames::run_programs`], which gates only the members whose
    /// frame actually pinned something.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for an instance id the registry does not hold, one
    /// whose program has gone, or one naming a channel that has been closed
    /// out from under it.
    pub fn ready(
        &self,
        instance: u64,
        tickets: &[driver::Ticket],
    ) -> Result<driver::Readiness, Unregistered> {
        let inst = self
            .registry
            .instance(instance)
            .ok_or_else(|| Unregistered(format!("no instance {instance}")))?;
        let effects = self.effects.get(&inst.program_id).ok_or_else(|| {
            Unregistered(format!(
                "instance {instance} names program {} which is gone",
                inst.program_id
            ))
        })?;
        let mut states = Vec::with_capacity(inst.channel_ids.len());
        for &id in &inst.channel_ids {
            let channel = self.registry.channel(id).ok_or_else(|| {
                Unregistered(format!(
                    "instance {instance} is attached to channel {id}, which is gone"
                ))
            })?;
            states.push(channel.state().as_ref());
        }
        Ok(driver::check(&states, effects, tickets))
    }

    /// Read one instance's device-resolved geometry off its channels.
    ///
    /// A PEEK, not a take: the ports that consume are consumed once, later,
    /// by the interpreter's own port loop when the program fires. Reading
    /// twice would drop a cell, and the symptom is a fire silently using the
    /// fire-after-next's tokens.
    ///
    /// `page` is the KV page size, which the `kv_len` port's contract needs.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for an instance id the registry does not hold, or one
    /// whose program has gone.
    pub fn geometry(&self, instance: u64, page: u32) -> Result<driver::Resolution, Unregistered> {
        let inst = self
            .registry
            .instance(instance)
            .ok_or_else(|| Unregistered(format!("no instance {instance}")))?;
        let program = self.registry.program(inst.program_id).ok_or_else(|| {
            Unregistered(format!(
                "instance {instance} names program {} which is gone",
                inst.program_id
            ))
        })?;
        Ok(driver::resolve(&program.plan, &inst.interp, page))
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory, as it is on Metal and on the dummy driver.
    /// Nothing about the channel plane is on the GPU -- it is a different
    /// layer from the model forward, and a device buffer for it would be a
    /// device round trip per cell for data neither shader reads.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for a shape or dtype the registry will not serve, or
    /// a duplicate id.
    pub fn register_channel(
        &mut self,
        desc: &driver_api::ChannelRegistrationPlan,
    ) -> Result<driver_api::ChannelBinding, Unregistered> {
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
        Ok(driver_api::ChannelBinding {
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
    ) -> Result<driver_api::InstanceBinding, Unregistered> {
        let geometry =
            Geometry::from_wire(geometry_class).map_err(|e| Unregistered(format!("{e}")))?;
        let instance_id = self
            .registry
            .bind_instance(program_id, requested, geometry, channel_ids, seeds)
            .map_err(|e| Unregistered(format!("{e}")))?;
        Ok(driver_api::InstanceBinding {
            instance_id,
            geometry_class,
            reserved0: 0,
        })
    }

    /// Run one instance's program for one fire.
    ///
    /// # What this is and is not
    ///
    /// It is the reference interpreter in the `driver` crate -- the same one
    /// `driver-metal` fires -- run over the instance's stages, reading this
    /// member's rows of the fire's distribution through `inputs` and
    /// advancing the instance's channels. It is NOT a device pass: no stage
    /// of a PTIR program is compiled to SPIR-V here, and the emitted kernels
    /// a registration carries are held rather than run.
    ///
    /// That distinction is the whole reason this is a separate verb from
    /// [`Self::register_program`]: registration was already complete and the
    /// module doc used to end by saying a frame carrying programs would not
    /// run them. It runs them now, on the host, which is where the channel
    /// plane lives.
    ///
    /// A [`driver::StepOutcome::Blocked`] is not a failure and is returned as
    /// itself: readiness is the program's own gate, a missed gate means the
    /// pass did not happen and changed nothing, and the caller re-posts.
    ///
    /// # Errors
    ///
    /// [`Unregistered`] for an instance id the registry does not hold, or one
    /// whose program has gone.
    pub fn fire(
        &mut self,
        id: u64,
        inputs: &driver::PassInputs,
    ) -> Result<driver::StepOutcome, Unregistered> {
        self.registry
            .fire(id, inputs)
            .map_err(|e| Unregistered(format!("{e}")))
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

    /// One echo program, whose epilogue puts the logits row it was handed
    /// onto its output channel.
    ///
    /// The smallest package that can be caught reading the WRONG row: it
    /// computes nothing, so whatever ends up on the ring is exactly what the
    /// caller bound into the intrinsic.
    fn echo(vocab: u32) -> driver_api::ProgramRegistration {
        use driver_api::local::{
            PIE_CHANNEL_HOST_VISIBLE, PIE_READINESS_NEEDS_EMPTY, PIE_VALUE_INTRINSIC,
        };
        use driver_api::plan::{
            LaunchChannel, LaunchPackage, LaunchPut, LaunchStage, LaunchStagePlan, LaunchValue,
        };

        driver_api::ProgramRegistration {
            program_hash: 0xECC0,
            launch: LaunchPackage {
                values: vec![LaunchValue {
                    id: 0,
                    source: PIE_VALUE_INTRINSIC,
                    dtype: 0,
                    intrinsic: driver::tensor_ir::op::intrinsic_tags::LOGITS as u8,
                    channel: 0,
                    literal_bits: 0,
                    shape: vec![vocab],
                }],
                channels: vec![LaunchChannel {
                    id: 0,
                    capacity: 1,
                    dtype: 0,
                    flags: PIE_CHANNEL_HOST_VISIBLE,
                    extern_dir: -1,
                    readiness: PIE_READINESS_NEEDS_EMPTY,
                    shape: vec![vocab],
                    extern_name: vec![],
                }],
                ports: vec![],
                names: vec![],
                stages: vec![LaunchStage {
                    kind: driver::tensor_ir::registry::Stage::Epilogue as u8,
                    ops: vec![],
                    puts: vec![LaunchPut {
                        channel: 0,
                        value: 0,
                    }],
                    takes: vec![],
                    reads: vec![],
                }],
                // The stage's normalized mirror. A package carries its ops
                // twice -- once in the stage body the interpreter walks, and
                // once in the stage PLAN, which is what a backend reads to
                // decide what the fire does to each channel. Both are filled
                // here because a real package has both, and a plan that
                // omitted the put would declare a channel that waits for room
                // and is never written.
                plans: vec![LaunchStagePlan {
                    ops: vec![driver_api::plan::LaunchOp {
                        code: u16::from(driver::tensor_ir::op::tags::CHAN_PUT),
                        channel: 0,
                        ..driver_api::plan::LaunchOp::default()
                    }],
                    channel_bindings: vec![0],
                    ..LaunchStagePlan::default()
                }],
            },
            ..Default::default()
        }
    }

    /// A channel this crate can bind an instance of `echo` to.
    ///
    /// `native` only, like the fires that use it: the tests below that bind a
    /// ring go through `frames::run_programs`, which is gated.
    #[cfg(feature = "native")]
    fn ring(id: u64, vocab: u32) -> driver_api::ChannelRegistrationPlan {
        driver_api::ChannelRegistrationPlan {
            channel_id: id,
            dtype: driver_api::PIE_CHANNEL_DTYPE_F32,
            shape: vec![vocab],
            capacity: 1,
            // The role the registry checks this against is the one the
            // PACKAGE declares for its host-visible channel, and a mismatch
            // is refused at bind time rather than at fire time.
            host_role: driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
            seeded: false,
            extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
            driver_id: 0,
            reader_wait_id: 0,
            writer_wait_id: 0,
        }
    }

    /// A step of three fired rows whose two requests answer from rows 0 and 2.
    ///
    /// `native` only: `turns` and `serve` are gated, and this crate's PORTABLE
    /// half has to compile on its own -- `default = []` says so, and the test
    /// targets are part of "compile".
    #[cfg(feature = "native")]
    fn three_rows() -> crate::turns::Step {
        crate::turns::Step {
            logits: crate::serve::Logits {
                rows: 3,
                vocab: 2,
                values: vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            },
            fired: crate::serve::Fired {
                dispatches: 0,
                submissions: 0,
                // The one counter this backend keeps that its siblings do not:
                // read-only operands copied out of a buffer their own dispatch
                // also writes. `driver-vulkan`'s `blocks`/`parsed` have no
                // counterpart here.
                shadowed: 0,
            },
            rows: 3,
            // Request 0 answers from fire row 0 and request 1 from fire row
            // 2 -- a prefill of two tokens followed by a decode, which is the
            // shape that makes "row `r`" and "request `r`" different numbers.
            readout_of: vec![0, 2],
            // One row each, which is the shape `readout_of` alone could
            // express. A request naming SEVERAL is what `readouts_of` exists
            // for, and it has its own test.
            readouts_of: vec![vec![0], vec![2]],
            positions: vec![0, 1, 0],
            pipelines: 0,
        }
    }

    /// Each member of a batched frame is fired over ITS request's
    /// distribution, and the ring proves which one it got.
    ///
    /// # The defect this pins
    ///
    /// `driver-metal` shipped a `run_programs` that handed every member the
    /// WHOLE read-out, and the interpreter binds its logits intrinsic from
    /// row zero and cannot be told otherwise. So in every batched frame all
    /// members sampled the FIRST request's distribution and returned its
    /// token: one fire, N requests, one answer repeated. Nothing faults,
    /// nothing is out of bounds, and no single-request test can see it --
    /// which is why this one has two members and two DIFFERENT rows.
    ///
    /// The second half is the `readout_of` indirection, which this crate has
    /// and Metal does not need: every row here samples, so a fire of three
    /// rows produces three distributions and request 1's answer is row 2.
    /// A `run_programs` that indexed the read-out by REQUEST would hand
    /// member 1 row 1 -- a real distribution, of the second token of the
    /// first request's prompt, and fluent nonsense.
    #[test]
    #[cfg(feature = "native")]
    fn each_member_is_fired_over_its_own_requests_distribution() {
        let mut programs = Programs::new();
        let program = programs
            .register_program(&echo(2))
            .expect("a well-formed package");
        programs.register_channel(&ring(10, 2)).expect("channel 10");
        programs.register_channel(&ring(11, 2)).expect("channel 11");
        let a = programs
            .bind_instance(
                program,
                None,
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[10],
                &[],
            )
            .expect("instance a")
            .instance_id;
        let b = programs
            .bind_instance(
                program,
                None,
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[11],
                &[],
            )
            .expect("instance b")
            .instance_id;

        let sub = driver_api::StepSubmission {
            plan: driver_api::LaunchPlan::default(),
            roster_rows: vec![0, 1],
            sub_batch_indptr: vec![0, 2],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1, 2],
            logical_fire_ids: vec![1, 2],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: Vec::new(),
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        };
        let mut faults = Vec::new();
        crate::frames::run_programs(&mut programs, &[a, b], &sub, &three_rows(), &mut faults)
            .expect("both members fire");
        assert!(faults.is_empty(), "a fault: {faults:?}");

        let got = |id: u64| match programs
            .registry
            .channel(id)
            .expect("the channel is registered")
            .state()
            .front()
        {
            driver::Value::F32(v) => v,
            other => panic!("channel {id} holds {other:?} rather than f32s"),
        };
        assert_eq!(got(10), vec![10.0, 11.0], "member 0 answered from row 0");
        assert_eq!(
            got(11),
            vec![30.0, 31.0],
            "member 1 answered from row {} rather than its own request's row 2",
            if got(11) == vec![10.0, 11.0] { 0 } else { 1 }
        );
    }

    /// A member whose ring moved since the batch was composed does not fire.
    ///
    /// # What a ticket catches that nothing else can
    ///
    /// The interpreter's own gate asks whether a ring is EMPTY or FULL, which
    /// is a question about how many cells there are. A ring that has had one
    /// cell taken and one put is neither, and holds a different cell than the
    /// one the scheduler composed against. Firing it puts an answer where
    /// somebody else's answer was expected, and nothing about it is out of
    /// bounds, poisoned or empty -- so no other check in this crate, or in
    /// the `driver` crate's `step`, can see it.
    ///
    /// The frame here pins the tail one PAST where the ring is, which is what
    /// a composer that had already counted another fire's put would state.
    /// The member must be skipped, unpoisoned and unfaulted, for the
    /// scheduler to re-compose and re-post.
    ///
    /// The second half of the test is the control that keeps the first half
    /// honest: the SAME frame with the tail pinned where the ring actually is
    /// fires and lands its row. Without it, a `run_programs` that refused
    /// every ticketed member would pass.
    #[test]
    #[cfg(feature = "native")]
    fn a_member_whose_ring_moved_since_the_batch_was_composed_is_not_fired() {
        let ticketed = |tail: u64| driver_api::StepSubmission {
            plan: driver_api::LaunchPlan::default(),
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![1],
            // One channel, so one ticket: the head is unpinned because the
            // program takes nothing, and the tail is the one that matters.
            channel_expected_head: vec![driver::NO_TICKET],
            channel_expected_tail: vec![tail],
            channel_ticket_indptr: vec![0, 1],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        };

        let mut programs = Programs::new();
        let program = programs.register_program(&echo(2)).expect("the package");
        programs.register_channel(&ring(10, 2)).expect("channel 10");
        let a = programs
            .bind_instance(
                program,
                None,
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[10],
                &[],
            )
            .expect("an instance")
            .instance_id;

        let mut faults = Vec::new();
        crate::frames::run_programs(
            &mut programs,
            &[a],
            &ticketed(1),
            &three_rows(),
            &mut faults,
        )
        .expect("a moved ring is not an error");
        assert!(
            faults.is_empty(),
            "an early fire was reported as a fault: {faults:?}"
        );
        assert_eq!(
            programs
                .registry
                .channel(10)
                .expect("the channel")
                .state()
                .tail(),
            0,
            "the member fired against a ring the composer had not pinned \
             it to, so its answer landed where another fire's was expected"
        );

        // ...and the same frame, pinned where the ring actually is, fires.
        crate::frames::run_programs(
            &mut programs,
            &[a],
            &ticketed(0),
            &three_rows(),
            &mut faults,
        )
        .expect("the fire");
        assert!(faults.is_empty(), "a fault: {faults:?}");
        assert_eq!(
            match programs
                .registry
                .channel(10)
                .expect("the channel")
                .state()
                .front()
            {
                driver::Value::F32(v) => v,
                other => panic!("the ring holds {other:?}"),
            },
            vec![10.0, 11.0],
            "a member whose ticket matches did not fire, so the gate refuses \
             everything rather than what moved"
        );
    }

    /// A member posted against a CLOSED ring is faulted, not quietly skipped.
    ///
    /// # Why the difference matters more than it looks
    ///
    /// Retry and Failed both end in "this member did not fire", and if the
    /// driver conflated them the frame would look identical from the outside.
    /// The scheduler reads them very differently: a retry is re-composed and
    /// re-posted, so a program whose ring will never accept another cell
    /// would be re-posted forever, at full speed, and the request behind it
    /// would never be told anything. A fault is delivered.
    ///
    /// A closed ring is the reachable version of this: the consumer has gone
    /// away and said so.
    #[test]
    #[cfg(feature = "native")]
    fn a_member_whose_ring_is_closed_is_faulted_rather_than_retried() {
        let mut programs = Programs::new();
        let program = programs.register_program(&echo(2)).expect("the package");
        programs.register_channel(&ring(10, 2)).expect("channel 10");
        let a = programs
            .bind_instance(
                program,
                None,
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[10],
                &[],
            )
            .expect("an instance")
            .instance_id;
        programs
            .registry
            .channel(10)
            .expect("the channel")
            .state()
            .close();

        let mut faults = Vec::new();
        crate::frames::run_programs(
            &mut programs,
            &[a],
            &driver_api::StepSubmission {
                plan: driver_api::LaunchPlan::default(),
                roster_rows: vec![0],
                sub_batch_indptr: vec![0, 1],
                sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
                terminal_cells: Vec::new(),
                program_row_indptr: vec![0, 1],
                logical_fire_ids: vec![1],
                channel_expected_head: vec![driver::NO_TICKET],
                channel_expected_tail: vec![0],
                channel_ticket_indptr: vec![0, 1],
                region_row_indptr: Vec::new(),
                region_sig: Vec::new(),
                region_k: Vec::new(),
            },
            &three_rows(),
            &mut faults,
        )
        .expect("one instance's dead ring is not the frame's failure");
        assert_eq!(
            faults.len(),
            1,
            "a closed ring was not reported, so the scheduler re-posts this \
             member forever and its request is never answered: {faults:?}"
        );
        assert_eq!(faults[0].0, a, "the fault names another instance");
        assert!(
            faults[0].1.contains("Closed"),
            "the fault does not say what is wrong: {}",
            faults[0].1
        );
    }

    /// A descriptor port is an effect on its channel; a const port is not.
    ///
    /// # Why the ports are worth deriving at all
    ///
    /// A package's effects are what the ticket gate is checked against, and
    /// most of them come from the stages. The descriptor PORTS touch channels
    /// too -- the token ids a pass embeds arrive on a ring, the page table
    /// beside them on another -- and a fire derived from the stages alone
    /// would say it does nothing to either, so the gate would wave through a
    /// fire whose input has not arrived.
    ///
    /// A const port was folded to its value at adoption and never reads its
    /// ring, so it contributes nothing; it is here to be sure it stays that
    /// way, because a const port that claimed an effect would gate a fire on
    /// a ring nobody ever fills.
    #[test]
    fn a_descriptor_port_is_an_effect_on_its_channel_and_a_const_port_is_not() {
        use driver_api::plan::{LaunchChannel, LaunchPort};

        let mut reg = echo(2);
        reg.program_hash = 0x9057;
        let package = &mut reg.launch;
        let mut more = |id: u32, readiness: u8| {
            package.channels.push(LaunchChannel {
                id,
                capacity: 1,
                dtype: 0,
                flags: 0,
                extern_dir: -1,
                readiness,
                shape: vec![1],
                extern_name: vec![],
            });
        };
        // A channel a fire reads has to be holding the cell it reads, so the
        // two ported ones wait for one; a channel touched by nothing must
        // declare no readiness, or registration refuses it.
        more(1, driver_api::local::PIE_READINESS_NEEDS_FULL);
        more(2, driver_api::local::PIE_READINESS_NEEDS_FULL);
        more(3, 0);
        package.ports = vec![
            LaunchPort {
                port: driver::tensor_ir::registry::Port::EmbedTokens as u8,
                is_const: false,
                const_dtype: 0,
                channel: 1,
                const_shape: vec![],
                const_data: vec![],
            },
            LaunchPort {
                port: driver::tensor_ir::registry::Port::Pages as u8,
                is_const: false,
                const_dtype: 0,
                channel: 2,
                const_shape: vec![],
                const_data: vec![],
            },
            LaunchPort {
                port: driver::tensor_ir::registry::Port::Positions as u8,
                is_const: true,
                const_dtype: 0,
                channel: 3,
                const_shape: vec![1],
                const_data: vec![0; 4],
            },
        ];

        let mut programs = Programs::new();
        let id = programs.register_program(&reg).expect("the package");
        let effects = &programs.effects[&id];
        assert!(
            effects[1].take && effects[2].take,
            "a ported channel derived no effect, so a fire whose input has \
             not arrived is not gated on it"
        );
        assert!(
            !effects[3].take && !effects[3].put,
            "a const port was folded at adoption and still claims its ring, \
             so the fire waits on a ring nobody fills"
        );
    }

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
