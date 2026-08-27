//! `impl Driver for Cuda` — the shell behind the contract.
//!
//! # Why a wrapper and not `impl Driver for Shell`
//!
//! Because a [`Shell`] **is a loaded model**. `Shell::load` binds the device,
//! compiles the plan, lands the checkpoint and reserves the pools in one
//! call, and every other method on it is about that load. The contract's
//! [`Driver`] is the other shape: a caller opens a driver first
//! (`engine::driver::backend::open::cuda`, from a boot config that has no
//! model in it), registers it, and only then calls
//! [`Driver::load`] with a traced `Plan`. There is no `Shell` to have a
//! `Driver` impl on until the verb that makes one has been called.
//!
//! So [`Cuda`] is a `Shell` that has not happened yet: the device knobs a
//! boot config states, an `Option<Shell>` that `load` fills, and the
//! [`Capabilities`] that load answered. Every verb before a load is a
//! refusal with a sentence.
//!
//! # The contract the wrapper cannot state, and how it is supplied
//!
//! [`LoadRequest`] carries `{ plan, checkpoint, budgets, ordinal }` and NOT a
//! `ModelContract` — deliberately, because `driver-api`'s dependency floor is
//! `model-ir`, `tensor-ir`, `serde`, `thiserror` (its own header), and a
//! contract type in it would put `model-loader` in the graph of everyone who
//! reads a `KvHandle`. But [`Weights::resident`](crate::weights::Weights)
//! needs one: how a checkpoint's tensors become this plan's params is the
//! MODEL's declaration, and the shell must not grow an arm per family to
//! rediscover it (`weights.rs`'s own header).
//!
//! The resolution is a function pointer, installed when the driver is opened:
//!
//! ```text
//!   engine (links `model`)                     driver-cuda (links no family)
//!   ----------------------                     -----------------------------
//!   fn contract_for(plan, path) -> Contract ──▶ Cuda::new(boot, contract_for)
//!     model::import_of(plan.name)                  … load(request) calls it
//! ```
//!
//! One pointer, resolved by the party that already links the catalog, and no
//! model name anywhere in this crate outside its own dev-dependencies.
//!
//! # What this driver does not serve, and says so
//!
//! `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
//! default bodies, which answer [`DriverError::Unsupported`]. That is the
//! honest report of what the v1 shell has: its pools are not virtual, it has
//! no peer-copy path, no recurrent-state mover and no multimodal encoder.
//! Stubbing any of them to `Ok(())` would make a prefix-cache hit, a swap or
//! an image prompt *appear* to work and silently read the wrong bytes, which
//! is the failure mode the contract's "refusal is a value" section exists to
//! prevent.

use std::path::{Path, PathBuf};

use driver::driver_api::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use driver::driver_api::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use driver::driver_api::error::{DriverError, Result as DriverResult};
use driver::driver_api::fire::{
    FireId, FireSubmission, FireTicket, LaneReadout, Readout,
};
use driver::driver_api::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use driver::driver_api::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use driver::driver_api::transfer::MemoryDomain;
use driver::driver_api::Driver;
use model_compiler::{Budgets, DeviceProfile};
use model_ir::Plan;
use model_loader::contract::ModelContract;
use driver::driver_api::tensor_ir::registry::{GeometryClass, ModelProfile, PortMask};
use driver::driver_api::tensor_ir::types::DType;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Graphs, Lane, Seated, Shell};

/// How a caller answers "what does this checkpoint's bytes mean for this
/// plan".
///
/// See the module header: the contract has no seat for a `ModelContract` and
/// this crate must not know a model family, so the party that links the
/// catalog supplies the lookup.
pub type ContractFor =
    fn(&Plan, &Path) -> std::result::Result<ModelContract, String>;

/// The device knobs a boot config states, before any model is loaded.
///
/// Everything here is a property of the MACHINE and the deployment, which is
/// what a driver is opened with; the model's own ceilings arrive later on
/// [`LoadRequest::budgets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBoot {
    /// Which device to bind.
    pub ordinal: i32,
    /// How much of a fire to record. Overridden by `PIE_CUDA_GRAPHS`.
    pub graphs: Graphs,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            ordinal: 0,
            graphs: Graphs::default(),
        }
    }
}

/// The CUDA shell, behind [`Driver`].
pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
}

impl Cuda {
    /// A driver bound to nothing yet.
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Cuda {
        Cuda {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
        }
    }

    /// The loaded shell, for a caller that wants the native surface — the
    /// A/B mode switch, the graph statistics, the footprint.
    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    /// The loaded shell, mutably.
    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    /// What this load can do, once there is one.
    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    /// Open a slot for a fresh sequence.
    ///
    /// Not a trait verb: the contract has no `open`, because
    /// [`KvDelta`](driver::driver_api::KvDelta) says a lane whose `pages` are
    /// empty is one whose page table the SHELL owns, and a shell that owns it
    /// opens the slot itself. An engine that keeps its own page table never
    /// calls this — it states the same fact as `KvDelta::held == 0`, and the
    /// fire path clears the slot's recurrent banks on reading it.
    ///
    /// # Errors
    ///
    /// [`DriverError::Load`] before a load, and whatever the pools said.
    pub fn open(&mut self, slot: u32) -> DriverResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    /// One bound instance's session, for the two channel doors.
    ///
    /// # Errors
    ///
    /// [`DriverError::Load`] before a load, [`DriverError::Closed`] for an
    /// instance this plane does not carry — `Closed` and not `Program`
    /// because "this handle is gone" is what the caller can act on, and a
    /// channel door is exactly where a torn-down instance is discovered.
    fn instance(&mut self, id: InstanceId) -> DriverResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .ok_or(DriverError::Closed {
                what: "instance",
                id,
            })
    }

    fn loaded(&self) -> DriverResult<&Shell> {
        self.shell.as_ref().ok_or_else(|| {
            DriverError::Load("the cuda driver has no model loaded".into())
        })
    }

    fn loaded_mut(&mut self) -> DriverResult<&mut Shell> {
        self.shell.as_mut().ok_or_else(|| {
            DriverError::Load("the cuda driver has no model loaded".into())
        })
    }
}

/// The shell's refusal, in the contract's vocabulary.
///
/// **THE TAXONOMY IS THE POINT.** `Exhausted` and `Impossible` are scheduling
/// answers the engine's lane loop acts on — retry behind something that frees
/// pages, or drop the request — and everything else is a failure it logs. A
/// [`Fault::Ceiling`] is `Impossible` and not `Exhausted` because every
/// ceiling this shell states was reserved at LOAD: no amount of freeing makes
/// a graph recorded for 8192 rows take 9000, which is exactly the distinction
/// `driver_api::error`'s header draws.
fn fault(fault: Fault) -> DriverError {
    match fault {
        Fault::Runtimeless | Fault::Device { .. } | Fault::Schedule { .. } => {
            DriverError::Device(fault.to_string())
        }
        Fault::Bake(_) | Fault::Load(_) | Fault::Param { .. } | Fault::Unbound { .. } => {
            DriverError::Load(fault.to_string())
        }
        Fault::Ceiling { what, need, have } => DriverError::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        // A region whose classes this fire's order does not make consecutive
        // is a BAKE-integrity break, not a submission the caller can fix; so
        // is a schedule built over more classes than its reader runs.
        Fault::Fragmented { .. } => DriverError::Device(fault.to_string()),
        Fault::Straddled { .. } => DriverError::Load(fault.to_string()),
        // A mask that does not describe its lane, or one against a plan with
        // no masked arm, is the SUBMISSION's — the caller stated it and the
        // caller can state it differently. `Invalid`, not `Impossible`: a
        // retry with a mask of the lane's own extent is a real answer.
        Fault::Mask { .. } | Fault::Maskless { .. } | Fault::MaskWord { .. } => {
            DriverError::Invalid(fault.to_string())
        }
        Fault::Compile(_) | Fault::Program { .. } => DriverError::Program(fault.to_string()),
        Fault::Fire(_) => DriverError::Invalid(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling.
///
/// Two names for one seven-role vector: [`ExtentRole`] is the tag space both
/// are written in, and the conversion is field for field so that adding a
/// role to one without the other is a compile error rather than a silently
/// unresolved axis.
fn extents(stated: &BindExtents) -> driver::Extents {
    driver::Extents {
        kv_len: stated.kv_len,
        page_count: stated.page_count,
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The ceilings the compiler bakes against, out of the ones the load states.
///
/// The contract carries seven numbers and `model_compiler::Budgets` takes
/// four; the other three (`page_size`, `max_context`, `slots`) are the POOLS'
/// and go to `Boot` directly. Converted in one place, which is the whole
/// reason `driver-api` states its own `Budgets` rather than depending on the
/// compiler (`load.rs`'s note).
fn bake_budgets(budgets: &LoadBudgets) -> Budgets {
    Budgets {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// The guest-visible profile of a loaded plan.
///
/// **CARRIED, NOT RECONSTRUCTED** (design §7 on `caps`): the engine used to
/// rebuild a `ModelProfile` at bind time out of eight `has_*` booleans on a
/// flat capability struct. Everything below is read off the plan and the
/// budgets — `num_layers` from the nodes' own `layer` stamps, `vocab` from
/// the width of the `out` seam — so there is one copy and nothing to keep in
/// step.
fn profile(shell: &Shell, budgets: &LoadBudgets) -> DriverResult<ModelProfile> {
    let plan = shell.plan();
    let layers = plan
        .nodes
        .iter()
        .filter_map(|node| node.layer)
        .max()
        .map_or(0, |top| top + 1);
    let vocab = u32::try_from(shell.out_width().map_err(fault)?).unwrap_or(u32::MAX);
    Ok(ModelProfile {
        vocab,
        page_size: budgets.page_size,
        num_layers: layers,
        // The INTERPRETER-VISIBLE materialization, which is what the field
        // means (`ModelProfile::activation`'s own doc): the device's own
        // activation type is bf16 and is not what a guest program reads.
        activation: DType::F32,
        // Every one of these is a MODEL-GATED intrinsic the guest-program
        // plane would have to point at a buffer of this fire's, and the
        // attachment binds exactly ONE — `IntrinsicId::Logits`, at the arena's
        // out-seam rectangle (`Shell::fire_attached`). There is no second
        // rectangle for an mtp draft, a value head or an attention score to
        // stand in, so advertising one would let a program bind against it and
        // then fail at its first fire, which is the opposite of a bind-time
        // contract.
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        has_attn_score: false,
        has_attn_page_mask: false,
        has_lora: false,
        kernels: Vec::new(),
    })
}

impl Driver for Cuda {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    /// **THE LANE THREAD SAYS IT IS THE ONE NOW.** `Shell::load` binds the
    /// device on the thread that CALLED it, which is the worker's boot
    /// thread; the engine then moves this driver onto its own lane thread and
    /// runs every verb after the load from there. `cudaSetDevice` is
    /// per-thread and does not travel with the value, so the lane thread
    /// announces itself once and this rebinds.
    ///
    /// Before a load there is no device to bind to and nothing to do — the
    /// announcement is legal at any point and answers for what it can.
    fn bind_thread(&mut self) -> DriverResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> DriverResult<Loaded> {
        if self.shell.is_some() {
            return Err(DriverError::Load(
                "this cuda driver already has a model loaded; one shell per driver".into(),
            ));
        }
        let LoadRequest {
            plan,
            checkpoint,
            budgets,
            ordinal,
        } = request;

        // `Checkpoint::None` — bind and bake, land nothing — is a shape the
        // contract states and this shell has no path for: `Weights::resident`
        // is what reserves the store, and a `WeightTable` of nulls would
        // panic at the first dispatch rather than at the load that asked for
        // it. Refused by name.
        let Checkpoint::Path(path) = checkpoint else {
            return Err(DriverError::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        let contract = (self.contract_for)(&plan, &path).map_err(DriverError::Load)?;

        let shell = Shell::load(Boot {
            plan,
            contract: &contract,
            checkpoint: &path,
            budgets: bake_budgets(&budgets),
            // `None` takes this device's measured SM count. Costs are input,
            // not knowledge (design §6): a deployment that has measured its
            // own would state them, and one that has not still bakes
            // something that runs.
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            // The REQUEST's ordinal wins over the boot config's when it names
            // one: `LoadRequest::ordinal` is the contract's field for "which
            // device, when the shell serves more than one", and a boot config
            // that also named one is the deployment's default.
            ordinal: if ordinal >= 0 { ordinal } else { self.boot.ordinal },
            graphs: self.boot.graphs,
        })
        .map_err(fault)?;

        let plan_name = shell.plan().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let paging = shell.paging();
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cuda".to_string(),
                domain: MemoryDomain::CudaDevice(
                    u32::try_from(shell.ordinal()).unwrap_or(0),
                ),
                sms: shell.sms(),
                unified_memory: false,
                // Neither is probed. Both are load-time answers about the
                // device's ARITHMETIC, and nothing in this shell reads them
                // yet — the mxfp4 MoE arm is the dispatch plane's decision
                // (design §6), not a capability the caller selects.
                fp8_native: false,
                native_mxfp4_moe: false,
                // What `cudaMalloc` itself guarantees, and what a matrix
                // operand wants under cuBLAS — the same 256 the weight store
                // aligns to.
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: Some("cuda".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                // The recurrent pool is reserved off the plan's own
                // `CacheRow::State` rows and the shell keeps no count of its
                // slots apart from the seats it opened, which is `slots`.
                state_slots: paging.slots,
                state_slot_bytes: 0,
                // palo B-adapters: design §8's banks are a budget, and this
                // shell reserves none — `Budgets::max_adapters` reaches the
                // compiler and nothing allocates for it yet.
                adapter_banks: 0,
                // The pools are not virtual, so `resize_pool` is not served
                // and zero is what says so.
                elastic_page_bytes: 0,
                elastic_budget_pages: 0,
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                // Every lane may name its whole slot's block, and a fire may
                // carry `max_lanes` of them.
                max_page_refs: paging
                    .pages_per_slot
                    .saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // THE DECODE ENVELOPE IS SERVED AND NOTHING WIDER IS (`palo
            // B3`). `crate::program::ports` reads `embed_tokens`, `positions`
            // and `kv_len` off the attached instance's own device rings at
            // fire time — the token because the device DECIDED it and no host
            // can know it, the other two because a port that is read is a
            // port that is served, and both are checked against the seat the
            // page arithmetic is carved from.
            //
            // The four ports above that — `pages`, `page_indptr`, `w_slot`,
            // `w_off` — are NOT claimed, and the mask says so. A
            // decode-envelope lane's page table is this shell's own
            // (`KvDelta::pages` empty is the submission saying so), and
            // `store::kv::geometry_with` derives all four from the seat;
            // claiming them would mean letting a guest's page ids reach the
            // pool, which is the pooled device-geometry class and a different
            // piece of work. A `DeviceGeometry` binding is still refused at
            // bind, by name.
            ports: PortMask::DECODE_ENVELOPE,
            geometry: GeometryClass::DecodeEnvelope,
            kv_copy: KvCopyDomains::default(),
            kv_handle: None,
            media_encode: false,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                plan_name,
                weight_bytes,
                arena_bytes,
                pool_bytes,
                input_bytes,
            },
            caps,
        })
    }

    fn fire(&mut self, submission: &FireSubmission) -> DriverResult<FireTicket> {
        submission.validate()?;
        // A GUEST PROGRAM THAT CANNOT COMMIT REFUSES THE FIRE, AND IT DOES SO
        // AS A SCHEDULING ANSWER (design §9, `palo B2`). The readiness gate is
        // asked over every attached instance before anything launches —
        // `Shell::fire_attached`'s own header argues why it has to be — but a
        // block is not a FAILURE: the guest's rings will be ready once the
        // host drains what it published, which is the run-ahead's own retry
        // shape (`DriverError::is_scheduling`, and the lane loop that reads
        // it). Asked here rather than inside the shell because `Exhausted` is
        // a contract word and the shell speaks `Fault`.
        for attachment in &submission.attachments {
            let shell = self.loaded()?;
            if let Some(channel) = shell
                .program_ready(attachment.instance)
                .map_err(fault)?
            {
                return Err(DriverError::Exhausted {
                    resource: "guest channel cells",
                    // The unit is CELLS on one ring: a blocked instance wants
                    // one more than the ring can give it right now, which is
                    // either an input that has not arrived or an output the
                    // host has not drained. Naming the channel in the numbers
                    // is what a log needs to tell those two apart.
                    wanted: u64::from(channel) + 1,
                    available: u64::from(channel),
                });
            }
        }
        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        let shell = self.loaded_mut()?;
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .map(|lane| {
                if let Some(adapter) = lane.adapter {
                    // palo B-adapters: `Lane::adapter` routes to a device
                    // bank (design §8) and this load reserved none. A lane
                    // that asked for one and got the base model would be
                    // silently wrong output, which is the whole reason the
                    // capacity is a budget rather than an admission cap.
                    return Err(DriverError::Unsupported {
                        verb: "adapter banks",
                        driver: "cuda",
                    })
                    .map_err(|error: DriverError| {
                        let _ = adapter;
                        error
                    });
                }
                if !lane.positions.is_empty() {
                    // The shell derives positions as `held .. held + rows`.
                    // An explicit list means a speculative fire re-feeding
                    // rejected positions or an mRoPE lane, and both need a
                    // staged `positions` vector this fire path does not take.
                    return Err(DriverError::Unsupported {
                        verb: "explicit lane positions",
                        driver: "cuda",
                    });
                }
                if !lane.kv.translation.is_empty() {
                    // A fork moved this lane's pages, and the engine states
                    // both the old ids and the new. The shell reads the page
                    // table it is given and never the one before it, so a
                    // translation is either already applied to `pages` — in
                    // which case stating it is redundant — or it is not, in
                    // which case honouring it needs a page mover this shell
                    // does not have (`copy_kv`, unsupported below).
                    return Err(DriverError::Unsupported {
                        verb: "kv page translation",
                        driver: "cuda",
                    });
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: lane.word,
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    // `palo B-mask`, closed on this side. The `masked` fact
                    // is a declared axis (design §0/§8) and the mask itself
                    // is a runtime input, so the shell carries the bits and
                    // the PLAN decides whether anything reads them: a mask
                    // against an artifact with no `attention.masked` arm is
                    // `Fault::Maskless`, named at the fire rather than
                    // refused for every model here. The word the engine
                    // stamped is what puts the lane in the masked class, and
                    // `compose` refuses a word this artifact has no class for
                    // — so a mask and a word that disagree cannot both pass.
                    mask: lane.mask.as_ref(),
                })
            })
            .collect::<DriverResult<Vec<_>>>()?;

        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                instance: attachment.instance,
                at: attachment.at,
            })
            .collect();
        let rows = shell.fire_attached(&seated, &attached).map_err(fault)?;

        // THE SHELL READS THE LAST ROW AND ONLY THE LAST ROW, which is
        // `Readout::Last` — the default, and the reason a prefill does not
        // hand back half a megabyte per teacher-forced position.
        let mut readouts = Vec::with_capacity(rows.len());
        for (lane, values) in submission.lanes.iter().zip(rows) {
            readouts.push(match &lane.readout {
                Readout::None => LaneReadout::default(),
                Readout::Last => LaneReadout {
                    rows: 1,
                    width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    values,
                },
                Readout::Rows(_) => {
                    // palo B-readout: reading an interior row means keeping
                    // the whole logits rectangle addressable after the walk,
                    // which the arena does — `slots.0[out]` is the rectangle
                    // — but the row list has to reach the read-back loop, and
                    // `Shell::fire` answers one row per lane by design.
                    return Err(DriverError::Unsupported {
                        verb: "row-selected readout",
                        driver: "cuda",
                    });
                }
            });
        }
        Ok(FireTicket { id, readouts })
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> DriverResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> DriverResult<RegisteredChannel> {
        // BINDING IS REGISTRATION HERE, and the verb's own doc says so. A
        // channel's DEVICE ring is allocated inside `Session::bind` —
        // `program/launch.rs` carves every instance's rings when the instance
        // is bound, from the package's own channel declarations — so a
        // standalone `register_channel` has nothing to allocate. What the
        // engine registers before a bind is its HOST ring, which it owns
        // (`engine::driver::channel`), and the two are joined by
        // `publish_channel`/`take_channel` around the fire rather than by a
        // second allocation here.
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> DriverResult<BoundInstance> {
        // REFUSED AT BIND, WHICH IS THE CONTRACT'S OWN READING: a class is a
        // claim about which descriptor ports the device resolves, and the
        // caps this load answered say which those are. Asked through
        // `Capabilities::admits` rather than by re-deriving the subset test
        // here, because the contract wrote that negotiation down once and a
        // second spelling of it is a second thing to keep in step. A program
        // bound in a class this load does not serve would otherwise fail at
        // its first fire, against a descriptor nobody wrote.
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| DriverError::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(DriverError::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and \
                 not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }
        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        // THE EXTENTS ARE THE CALLER'S ANSWER, NOT A GUESS (Build log 15).
        // Every stage's fire-path buffers are carved at this call, from these
        // numbers; `sampled_rows` is the one a model fire's epilogue reads,
        // and one carved for a single row when the fire hands it four
        // zero-fills three rows that no launch faults on. Converted here
        // because `BindExtents` is the contract's spelling and `Extents` is
        // the plane's — the same seven roles, and the tags are
        // `ExtentRole`'s in both.
        let id = self
            .loaded_mut()?
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
            )
            .map_err(fault)?;
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> DriverResult<()> {
        self.loaded_mut()?
            .close_program_instance(id)
            .map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> DriverResult<()> {
        // As `register_channel`: a channel's life is its instance's, and
        // closing one on its own is a door this plane does not have.
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> DriverResult<bool> {
        // THE DOOR THAT REPLACED THE POINTER. The engine used to write a cell
        // into this ring itself, through the addresses `ChannelBinding`
        // published; it owns a host ring of its own now and hands the bytes
        // over here (`engine::driver::channel`'s header, and the trait's).
        self.instance(instance)?.publish(channel, cell).map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> DriverResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    // `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
    // default bodies. See the module header: the v1 shell genuinely has
    // none of the four, and a stub that answered `Ok(())` would make a
    // prefix-cache hit, a swap or an image prompt appear to work.
}

// STILL `unsafe impl`, and for the reason `Shell` has always needed one: it
// holds the device's own raw handles inline (a `cublasContext`, CUDA events,
// the arena's `c_void` bases), none of which is `Send` to the compiler, and
// taking C out of the CALL did not change what a CUDA context is. What makes
// it sound is the driver's own rule: every verb takes `&mut self`, so exactly
// one thread touches a shell at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}
