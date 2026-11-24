#![allow(unsafe_code)]
use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use engine::transfer::{KvCopy, MemoryDomain};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use poem_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::experts;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Landed, Lane, Seated, Shell, StepView};
use crate::settle::Done;
use crate::weights::AdapterPlane;

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceBoot {
    pub adapter_dir: Option<std::path::PathBuf>,
    /// `$PIE_HOME/profiles`, where a user's device profiles live.
    pub profiles: Option<std::path::PathBuf>,
    /// `[metal] measure`: whether a load times what no stored or committed
    /// profile states under a matching key.
    pub measure: engine::rates::Measuring,
    /// The top rung a load serves: `None` the ruling's (`Top::Ruled`, every
    /// served load). A test that hand-builds frames wider than a miniature's
    /// ruled top (which the runtime's step builder never forms) loads every
    /// rung (`Top::EveryRung`), as the shell tests do.
    pub top: Option<crate::Top>,
}

struct PendingStep {
    readout: Vec<Readout>,
    landed: Landed,
    rows: Option<Vec<(String, Vec<f32>)>>,
}

pub struct Metal {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<PendingStep>)>,
    /// The adapter id each instance's lanes route by.
    adapters: BTreeMap<InstanceId, u32>,
    /// What the load armed and what its fires tallied (`engine::live`).
    live: engine::live::Live,
    /// Where a tier copy still landing is told landed
    /// (`Engine::copy_slots_landing`).
    landed: Option<engine::LandingSink>,
    /// Test builds: the slots the runtime gave back, poisoned before the
    /// next verb (`Engine::poison_queue`).
    #[cfg(feature = "test-slot-fill")]
    poison: Option<engine::chunk::PoisonQueue>,
}

impl Metal {
    /// Fires that reached a pipeline the load did not build, counted as
    /// `Tally::BuiltInFire`: once the load is sealed nothing builds on a
    /// fire's path here (the pipeline cache refuses the key, and the fire
    /// with it), so what a backend that builds there would count is these.
    /// It reads 0 where every pipeline a fire can reach was built before
    /// ready.
    fn tally_unbuilt(&self) {
        if let Some(shell) = self.shell.as_ref() {
            let missed = shell.take_unbuilt_pipelines();
            if missed > 0 {
                self.live.tally(engine::live::Tally::BuiltInFire, missed);
            }
        }
    }

    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Metal {
        Metal {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            pending: None,
            adapters: BTreeMap::new(),
            live: engine::live::Live::armed(),
            landed: None,
            #[cfg(feature = "test-slot-fill")]
            poison: None,
        }
    }

    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id: id.0,
            })
    }

    fn loaded(&self) -> EngineResult<&Shell> {
        self.shell
            .as_ref()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        let shell = self
            .shell
            .as_mut()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))?;
        // Test builds: what the runtime gave back since the last verb is
        // poisoned before this one touches the pool.
        #[cfg(feature = "test-slot-fill")]
        if let Some(queue) = &self.poison {
            let due = std::mem::take(&mut *queue.lock().expect("the poison queue's lock"));
            if !due.is_empty() {
                let poisoned = shell.poison_given_back(&due);
                tracing::trace!(target: "engine_metal::poison", due = due.len(), poisoned, "given-back slots poisoned");
            }
        }
        Ok(shell)
    }
}

fn kv_copy_direction(src: MemoryDomain, dst: MemoryDomain) -> &'static str {
    match (src, dst) {
        (MemoryDomain::HostPinned, MemoryDomain::HostPinned) => {
            "`copy_kv` host-pinned to host-pinned, which is the caller's own memmove"
        }
        (MemoryDomain::HostPinned, _) => {
            "`copy_kv` out of host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (_, MemoryDomain::HostPinned) => {
            "`copy_kv` into host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (MemoryDomain::MetalPrivate, _) | (_, MemoryDomain::MetalPrivate) => {
            "`copy_kv` with a Metal PRIVATE end, and every reservation this load made is \
             Shared — there are no private pages here to read or write"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

fn fault(fault: Fault) -> Error {
    // A fault one lane earns refuses that lane alone; the step's other
    // lanes are the runtime's to submit again (`engine::Error::Lanes`).
    if let Some(lane) = fault.lane() {
        return Error::lane(lane, fault);
    }
    match fault {
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),
        Fault::TierFull(_) => Error::TierFull(fault.to_string()),
        Fault::Refused { bytes } => Error::Exhausted {
            resource: "device memory",
            wanted: bytes,
            available: 0,
        },
        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Backing { .. }
        | Fault::Mapped { .. }
        | Fault::Param { .. }
        | Fault::Recipe(_)
        | Fault::Shader { .. }
        | Fault::Unprepared { .. }
        | Fault::Unbound { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Fragmented { .. } => Error::Device(fault.to_string()),
        Fault::Unaffine { .. } | Fault::Unstructured { .. } => Error::Load(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskUnread { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapterless { .. }
        | Fault::AdapterWord { .. }
        | Fault::GeometryLanes { .. }
        | Fault::OutputRows { .. } => {
            Error::Invalid(fault.to_string())
        }
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::Claimed(refusal) => Error::Claimed(refusal),
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) => Error::Invalid(fault.to_string()),
        Fault::UnderCarve { logical, .. } => Error::Placement {
            chunks: vec![logical],
            reason: fault.to_string(),
        },
        Fault::Residency(_) | Fault::FitShort(_) => Error::Impossible(fault.to_string()),
    }
}

fn extents(stated: &BindExtents) -> eta_exec::Extents {
    eta_exec::Extents {
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The narrowest forward width a load narrows its fires to
/// so its arena scratch fits beside resident weights.
const MIN_FITTED_FORWARD_TOKENS: u32 = 512;

const DEFAULT_STATE_SLOTS: u32 = 256;

/// Bounds what the load tells (`FireLimits`) to the fires it walks: its top
/// rung, as the ruling bounded it (`crate::top::bounded`). A rung of so
/// many tokens walks no wider fire, and the runtime lifts its largest rung
/// to the limit told: told wider, it packs fires the walk refuses.
fn told_within(budgets: &mut LoadBudgets, served: &Budget) {
    if served.max_tokens < budgets.max_tokens {
        budgets.max_tokens = served.max_tokens;
        budgets.max_lanes = budgets.max_lanes.min(served.max_lanes);
        budgets.buckets.clone_from(&served.buckets);
    }
}

/// The load's budgets: its logit readouts sized to what `trace` reads, not a
/// general figure, so a vocab-wide head carves only the rows a fire can read.
fn bake_budgets(budgets: &LoadBudgets, trace: &Trace) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::readouts_per_lane(trace),
    }
}

#[must_use]
pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, poem_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Patches))
    });
    if !declares_patches {
        return None;
    }

    let max_patches = budgets
        .max_patches
        .unwrap_or_else(|| budgets.max_tokens.min(DERIVED_PATCH_CEILING))
        .max(PATCH_LATTICE_FLOOR);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    Some(PatchLadder {
        max_images: budgets
            .max_images
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

fn patch_bytes(
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
    }
}

fn voxel_bytes(
    payload: &[f32],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::Bf16 => Ok(payload
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect()),
        poem_ir::Dtype::F32 => Ok(payload.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(
            "a voxel submission against a plan whose voxel element is neither `bf16` \
                  nor `f32`, which is the pair every VAE in this catalog computes in",
        ),
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// The adapter an instance bound with `seeds` lands, by its content: two
/// independent 64-bit digests (FNV-1a high, SipHash low) of the sink's site
/// word and, in the sink's plane order, each plane's bank role, its seed's
/// length and its seed, word for word as CUDA's `adapter_key_of`, so one
/// adapter keys the same on both. None: the program lands none.
pub fn adapter_key_of(
    shell: &Shell,
    program: engine::program::ProgramId,
    seeds: &[engine::channel::ChannelSeed],
) -> EngineResult<Option<u128>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    Ok(Some(adapter_digest(
        sink.sites,
        sink.planes.iter().map(|(role, channel)| {
            let wire = seeds
                .iter()
                .find(|seed| seed.channel == *channel)
                .map_or(&[][..], |seed| seed.bytes.as_slice());
            (role.bank(), wire)
        }),
    )))
}

/// The digest [`adapter_key_of`] states, over a sink's site word and its
/// planes `(bank role, seed)` in order.
#[must_use]
pub fn adapter_digest<'a>(sites: u32, planes: impl Iterator<Item = (&'a str, &'a [u8])>) -> u128 {
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    let mut sip = std::collections::hash_map::DefaultHasher::new();
    let mut eat = |bytes: &[u8]| {
        use std::hash::Hasher;
        for byte in bytes {
            fnv = (fnv ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        sip.write(bytes);
    };
    eat(&sites.to_le_bytes());
    for (bank, wire) in planes {
        eat(bank.as_bytes());
        eat(&(wire.len() as u64).to_le_bytes());
        eat(wire);
    }
    let sip = std::hash::Hasher::finish(&sip);
    (u128::from(fnv) << 64) | u128::from(sip)
}

/// Lands the adapter an instance of `program` carries, its planes the
/// bind's seeds, in the `Adapter` slots the runtime took (`slots`): where
/// another instance landed the same adapter there it joins it. Its adapter
/// id, what its lanes route by; None where it carries none.
fn adapter_of(
    shell: &mut Shell,
    program: engine::program::ProgramId,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
    slots: &[u32],
) -> EngineResult<Option<u32>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        let wire = seeds
            .iter()
            .find(|(seeded, _)| seeded == channel)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| {
                Error::Load(format!(
                    "this program's `lora` sink reads its `{}` plane out of channel \
                     {channel} and this bind seeded nothing into it; an adapter's \
                     weights are the seed, because the fire path never reads the cell, \
                     so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    let planes: Vec<(String, Vec<u8>)> = planes
        .iter()
        .map(|plane| (plane.bank.to_string(), plane.bytes.to_vec()))
        .collect();
    shell
        .land_adapter(instance, slots, planes)
        .map(Some)
        .map_err(fault)
}

impl Engine for Metal {
    /// The device shares the host's memory: the governor holds the pool
    /// within what the GPU can still be given (`Context::host_room`), read
    /// on its tick, so the host side growing beside the pool shrinks it.
    fn host_room(&self) -> Option<engine::HostRoom> {
        Some(std::sync::Arc::new(crate::device::Context::host_room))
    }

    fn cover_fingerprint(&self) -> Option<String> {
        self.shell.as_ref().map(crate::serve::Shell::cover_fingerprint)
    }

    fn committed_covers(&self) -> &'static [&'static str] {
        crate::chip::COVERS
    }

    fn skips_measuring(&self) -> bool {
        self.boot.measure == engine::rates::Measuring::Skip
    }

    /// What the load timed (`Shell::measured_at_load`): the device's rates
    /// and the model's kernel.dense curves, each where it measured them.
    /// Nothing before a load, which has timed nothing yet.
    fn measured_at_load(&self) -> Vec<String> {
        self.shell
            .as_ref()
            .map(|shell| shell.measured_at_load().to_vec())
            .unwrap_or_default()
    }

    fn kind(&self) -> &'static str {
        "metal"
    }

    /// Whether a `[rates]` key the load serves by a stale figure is left to
    /// time behind serving (`Shell::rates_behind`).
    fn measures_behind(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.rates_behind().is_some())
    }

    /// One slice behind serving: the rates step timing the next stale
    /// `[rates]` key alone (`rates::measure_keys`), each frame reading
    /// whether a request arrived and stopping where one did, so an arrival
    /// waits at most the frame in flight; the key is timed again whole at a
    /// later idle turn. Kept only where it had the device alone at a steady
    /// clock: no frame of this engine in flight as it starts, no other lane
    /// submitting while it runs, and the clock read after it no slower than
    /// the settled one before it (`rates::throttled`). Kept: stored in the
    /// user's profile for the next load; this load serves on by the stale
    /// figure. Else discarded and timed again later.
    fn measure_behind(&mut self, behind: &engine::BehindServing<'_>) -> EngineResult<bool> {
        if self.pending.is_some() {
            return Ok(false);
        }
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        let Some(key) = shell.rates_behind().map(str::to_string) else {
            return Ok(false);
        };
        let profile = shell.tuning().profile().clone();
        let device = shell.device();
        let submitted = (behind.others)();
        let began = std::time::Instant::now();
        // A `curve <n>x<k>` entry: that shape's dense curves, timed by the
        // step a load times them by (`tune::dense::curves_behind`); else
        // one `[rates]` key.
        let timed = crate::rates::clock_rate(device, profile.clone(), true).and_then(|before| {
            let rates = match key.strip_prefix(crate::tune::dense::CURVE_BEHIND) {
                Some(shape) => profile::fire::Rates {
                    kernel_tflops_by_shape: Some(crate::tune::dense::curves_behind(
                        device,
                        &profile,
                        shape,
                        behind.arrived,
                    )?),
                    ..profile::fire::Rates::default()
                },
                None => crate::rates::measure_keys(
                    device,
                    profile.clone(),
                    &[key.as_str()],
                    Some(behind.arrived),
                    &mut |_| {},
                )?,
            };
            Ok((before, rates))
        });
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let (before, rates) = match timed {
            Ok(timed) => timed,
            Err(why) if crate::rates::yielded(&why) || why == crate::tune::dense::YIELDED => {
                tracing::info!(
                    key = %key,
                    ms,
                    "rates behind serving: `{key}` yielded to an arrival; timed again later"
                );
                return Ok(true);
            }
            Err(why) => {
                return Err(Error::Load(format!(
                    "rates behind serving: `{key}` was not timed: {why}"
                )));
            }
        };
        let after = crate::rates::clock_rate(device, profile, false).map_err(Error::Load)?;
        let why = if (behind.others)() != submitted {
            Some("another lane submitted while it ran".to_string())
        } else {
            crate::rates::throttled(before, after).map(|ratio| {
                format!("the device's clock {ratio:.2}x slower after it than its settled start")
            })
        };
        if let Some(why) = why {
            tracing::info!(
                key = %key,
                ms,
                "rates behind serving: `{key}` discarded ({why}); timed again later"
            );
            return Ok(true);
        }
        if let Err(why) = crate::rates::save_rates(device, self.boot.profiles.as_deref(), &rates) {
            tracing::warn!(%why, key = %key, "rates behind serving: not kept in the profile");
        }
        let left = shell.rate_timed(&key);
        tracing::info!(
            key = %key,
            ms,
            left,
            "rates behind serving: `{key}` timed and stored for the next load (this load serves \
             on by its stale figure); {left} left"
        );
        Ok(true)
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this metal engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight,
            steps,
        } = request;

        if ordinal > 0 {
            return Err(Error::unsupported("metal", "device ordinal selection"));
        }
        no_host_tier(residency.host_kv_budget)?;

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the metal shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let (working_set, outside, caches) = {
            let device = crate::device::Context::bind().map_err(fault)?;
            (
                device.working_set(),
                device.allocated(),
                // The weight cache runs where the queue keeps a residency set.
                device
                    .hold_resident(std::iter::empty(), true)
                    .map_err(fault)?,
            )
        };
        // Dense weights always live in device memory, and so does every
        // weight the weight cache does not stream; of the streamable ones
        // (routed experts, token-id tables) only the floor the cache keeps
        // must: the rest leaves the device under streaming, and a held set
        // seats it in the pool.
        let planes = crate::weights::plane_bytes(&trace).map_err(fault)?;
        let (weights, streamable) = if caches {
            let recipe = crate::chunks::recipe(
                &trace,
                &poem_compiler::Budgets::of(bake_budgets(&budgets, &trace)),
            )
            .map_err(fault)?;
            crate::store::streamed::resident_demand(
                &trace,
                &recipe,
                &planes,
                crate::experts::rows_streamed(residency.weights_held),
            )
        } else {
            (
                planes
                    .iter()
                    .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
                    .sum::<u64>(),
                0,
            )
        };
        let slots = poem_compiler::state_slots(budgets.slots, &trace, DEFAULT_STATE_SLOTS);
        let paging = crate::store::kv::Paging::of(budgets.page_size, budgets.max_context, 1, 1)
            .map_err(|error| fault(Fault::from(error)))?;
        // The pool the weights must leave room for: the stated budget,
        // or one sequence when the pool fills what is left.
        let kv_pool = match residency.device_kv_budget {
            Some(budget) => budget,
            None => crate::store::PoolDemand::of(&trace, paging, None)
                .map_err(fault)?
                .bytes(u64::from(paging.pages_per_slot)),
        };
        let accounting = |weights: u64, scratch: u64| {
            crate::store::accounting::Accounting::of(
                working_set,
                outside,
                residency.device.budget,
                weights,
                scratch,
                kv_pool,
            )
            .with_streamable(streamable)
        };

        // Compiled before the residency check, so the check prices the arena
        // scratch the load will wire beside the weights and the kv pool. When
        // the weights fit resident and only the scratch the forward width
        // reserves pushes the load past the device (a 12B on a 16 GB Mac),
        // the width halves until it fits: a narrower prefill fire is slower,
        // a refused load serves nothing.
        let mut budgets = budgets;
        // The widest fire's activation (its carve, its rs scratch, its
        // staged rows) beside one working set past the allowance: the fit
        // refuses, and the widest fire halves until it fits, once at load
        // (nothing narrows while serving). What was cut and why is told once.
        let asked = budgets.max_tokens;
        let mut cut: Option<engine::chunk::FitShort> = None;
        let mut shell = loop {
            let mut prepared = loop {
                let room = accounting(weights, 0).pool.saturating_sub(kv_pool) / 2;
                let mut fitted = poem_compiler::fit_voxels(
                    &trace,
                    poem_compiler::Budgets {
                        tokens: bake_budgets(&budgets, &trace),
                        patches: patch_ladder(&trace, &budgets),
                        voxels: poem_compiler::VoxelLadder::of(
                            &trace,
                            budgets.max_voxels,
                            budgets.max_clips,
                            budgets.max_lanes,
                        ),
                    },
                    &DeviceProfile::default(),
                    room,
                );
                let slots = match slots {
                    0 => poem_compiler::fit_frame_slots(
                        &trace,
                        fitted.voxels.as_mut(),
                        room,
                        engine::fit::least_state_slots(1),
                        frames_in_flight,
                    ),
                    slots => slots,
                };
                let voxels = fitted.voxels;
                let prepared = Shell::prepare(Boot {
                    trace: trace.clone(),
                    contract: &contract,
                    checkpoint: &path,
                    budget: bake_budgets(&budgets, &trace),
                    patches: patch_ladder(&trace, &budgets),
                    voxels,
                    profile: None::<DeviceProfile>,
                    page_size: budgets.page_size,
                    context: budgets.max_context,
                    slots,
                    kv_budget: residency.device_kv_budget,
                    allowance: residency.device,
                    runahead: engine::runahead::Runahead::of(frames_in_flight),
                    residency: experts::Plan::default(),
                    profiles: self.boot.profiles.clone(),
                    top: self.boot.top.unwrap_or(crate::Top::Ruled),
                    measure: self.boot.measure,
                })
                .map_err(fault)?;
                let narrower = (budgets.max_tokens / 2).max(MIN_FITTED_FORWARD_TOKENS);
                if narrower >= budgets.max_tokens
                    || accounting(weights, prepared.arena_bytes()).admit().is_ok()
                    || accounting(weights, 0).admit().is_err()
                {
                    break prepared;
                }
                eprintln!(
                    "the arena scratch for {} forward tokens ({} bytes) does not fit beside the \
                 resident weights on this device: fires narrowed to {narrower} tokens",
                    budgets.max_tokens,
                    prepared.arena_bytes(),
                );
                budgets.max_tokens = narrower;
                budgets.buckets.retain(|&rows| rows <= narrower);
            };
            let scratch = prepared.arena_bytes();

            accounting(weights, scratch).admit().map_err(fault)?;

            prepared
                .set_residency(experts::Plan::default().with_weights_held(residency.weights_held));
            match prepared.load() {
                Ok(shell) => break shell,
                Err(Fault::FitShort(short)) => {
                    let narrower = (budgets.max_tokens / 2).max(MIN_FITTED_FORWARD_TOKENS);
                    if narrower >= budgets.max_tokens {
                        return Err(fault(Fault::FitShort(short)));
                    }
                    cut = Some(short);
                    budgets.max_tokens = narrower;
                    budgets.buckets.retain(|&rows| rows <= narrower);
                }
                Err(other) => return Err(fault(other)),
            }
        };
        if let Some(short) = cut {
            eprintln!(
                "the widest fire narrowed from {asked} to {} forward tokens at load: at the \
                 wider fire, {short}",
                budgets.max_tokens
            );
        }

        told_within(&mut budgets, shell.budget());
        shell.mount_adapters(self.boot.adapter_dir.clone());
        shell
            .seat_disk_kv(residency.disk_kv_budget, residency.disk_kv_dir.as_deref())
            .map_err(fault)?;

        if crate::diag::on().tier_trace {
            let audit = shell.audit();
            eprintln!(
                "ledger: {}; device {} MiB, {} KiB beside the entries",
                shell.ledger(),
                audit.device >> 20,
                audit.gap() / 1024
            );
        }
        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let (pool_committed_bytes, pool_high_water_bytes) = shell.pool_residency();
        let weights_warm = shell.weights_warm();

        let paging = shell.paging();
        let pools = shell.pool_facts();
        let chunks = shell.chunk_facts().map_err(fault)?;
        shell.log_chunk_recipe(&chunks);

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "metal".to_string(),
                domain: MemoryDomain::MetalShared,
                sms: 0,
                unified_memory: true,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: shell.max_buffer(),
                codegen_backend: Some("metal".to_string()),
                library_forms: eta_compiler::codegen::metal::LIBRARY_FORMS,
            },
            pools,
            chunks,
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
                prefill_rows: shell
                    .neural_engine()
                    .map_or(0, crate::ane::Ane::prefill_rows),
                prefill_lanes: 0,
                max_readouts: u32::try_from(poem_compiler::arena::readouts_ceiling(&bake_budgets(
                    &budgets, &trace,
                )))
                .unwrap_or(u32::MAX),
                // Each lane's rung bounds what it replays.
                replayed_rows: 0,
            },
            has_lora: true,
            has_attn_page_mask: false,
            draft_block: shell.trace().drafter.map_or(0, |d| d.rows),
            draft_mask_token: shell.trace().drafter.map_or(0, |d| d.mask_token),
            draft_proposals_from: shell.trace().drafter.map_or(1, |d| d.proposals_from),
            ports: PortMask::DEVICE_GEOMETRY
                .with(Port::AttnMask)
                .with(Port::Fold)
                .with(Port::Parents),
            geometry: GeometryClass::DeviceGeometry,
            kv_handle: None,
            media_encode: false,
            rs_verbs: shell.serves_rs_verbs(),
            // Windowed layers as ring + buffer (CONTRACT-BC 19, 21): a load
            // with any states the rings and the window buffer in its facts.
            windowed_state: shell.pool_ring().is_some(),
            rs_snapshots: crate::rs::snapshots(shell.trace()),
            bidirectional_attention: true,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.inputs(),
            classes: shell.classes(),
            // Metal states no bound on one submission yet
            // (`crate::rates::at_load`).
            guest_cost: engine::GuestCost::default(),
            // A beam group's page list (`beam_list`), read at a search's finish.
            host_ops: engine::HostOps { beam_list: true },
            mixed_lanes: engine::MixedLanes::Any,
            frames_in_flight: engine::stated_frames(shell.frames_in_flight())?,
            steps: engine::StatedSteps::default(),
        };

        // The step compositions the runtime's builder forms over this load's
        // facts (CONTRACT-BC 24), computed before the walk, which fires each.
        let mut caps = caps;
        caps.steps = engine::StatedSteps::of(
            &steps,
            &engine::StepFacts::stated(&caps, &engine::ReadoutFacts::default()),
        );
        // Metal records no bodies (its walk builds pipelines): every shape
        // is walked, none eager.
        caps.steps.eager.clear();
        tracing::info!(
            "engine-metal: step set: {}; the walk fires every one of the {}, none eager",
            caps.steps.shapes.census(),
            caps.steps.shapes.len(),
        );

        // Every pipeline the load's rungs dispatch is built now, before
        // ready; a key met after is refused by name and tallied.
        let began = std::time::Instant::now();
        // A later boot of the same load (the same key) builds the keys an
        // earlier boot listed, from its binary archive, without walking; a
        // missing, stale or unreadable cache only costs the walk.
        let cache = self.boot.profiles.as_ref().and_then(|profiles| profiles.parent()).map(|home| {
            let key = shell.pipeline_cache_key();
            let name = key
                .bytes()
                .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
            let dir = home.join("cache").join("metal");
            (key, dir.join(format!("{name:016x}.keys")), dir.join(format!("{name:016x}.metalarchive")))
        });
        #[cfg(target_vendor = "apple")]
        if let Some((_, _, archive)) = cache.as_ref() {
            shell.use_pipeline_archive(archive);
        }
        let listed = cache.as_ref().and_then(|(key, keys, _)| read_pipeline_keys(keys, key));
        let (fires, built) = match listed {
            Some(keys) => (0, shell.build_pipelines(&keys).map_err(fault)?),
            None => {
                let walked = shell.warm_pipelines_over(&caps.steps.shapes).map_err(fault)?;
                if let Some((key, keys, _archive)) = cache.as_ref() {
                    write_pipeline_keys(keys, key, &shell.pipeline_keys());
                    #[cfg(target_vendor = "apple")]
                    let _ = shell.save_pipeline_archive(_archive);
                }
                walked
            }
        };
        shell.seal_pipelines().map_err(fault)?;
        tracing::info!(
            fires,
            built,
            listed = fires == 0,
            ms = began.elapsed().as_millis() as u64,
            "metal: every pipeline the load dispatches is built before ready"
        );
        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                weights_resident: true,
                weights_from_cache: weights_warm,
                arena_bytes,
                pool_bytes,
                input_bytes,
                pool_committed_bytes,
                pool_high_water_bytes,
            },
            caps,
        })
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        Err(Error::Invalid(format!(
            "adapter {} is operator-declared, and no backend hosts operator-declared \
             adapters: an adapter arrives through a program's `lora` sink at bind, its \
             planes the bind's seeds, and lands in `Adapter` slots shared by every \
             instance that carries the same planes",
            registration.id
        )))
    }

    fn commit_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.commit_segments(segments).map_err(fault)
    }

    fn release_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.release_segments(segments).map_err(fault)
    }

    fn set_rung(&mut self, from: u32, to: u32) -> EngineResult<()> {
        self.loaded_mut()?.set_rung(from, to).map_err(fault)
    }

    fn commit_span(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.commit_span(segments).map_err(fault)
    }

    fn release_span(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.release_span(segments).map_err(fault)
    }

    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        self.loaded_mut()?.evacuate(evacuation).map_err(fault)
    }

    fn chunk_table(&self) -> EngineResult<engine::chunk::ChunkTable> {
        Ok(self.loaded()?.chunk_table().clone())
    }

    /// Waits on the last copy's command buffer: every copy submitted so far
    /// has read its sources and written its destinations.
    fn settle_copies(&mut self) -> EngineResult<()> {
        match self.shell.as_mut() {
            Some(shell) => shell.settle_copies().map_err(fault),
            None => Ok(()),
        }
    }

    /// A copy on the device rides a command buffer that may still be
    /// reading its sources when the call returns: told landed from that
    /// buffer's completion. A move to or from the host tier or the slot
    /// files is a host copy done inside the call: its sources are read.
    fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        token: u64,
    ) -> EngineResult<bool> {
        let landed = self.landed.clone();
        self.loaded_mut()?
            .copy_slots_landing(copy, landed.map(|sink| (sink, token)))
            .map_err(fault)
    }

    fn on_landed(&mut self, sink: engine::LandingSink) {
        self.landed = Some(sink);
    }

    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        self.loaded_mut()?.copy_slots(copy).map_err(fault)
    }

    fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> EngineResult<Vec<u8>> {
        self.loaded_mut()?.read_tier_slots(slots).map_err(fault)
    }

    fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> EngineResult<()> {
        self.loaded_mut()?
            .write_tier_slots(slots, bytes)
            .map_err(fault)
    }

    fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> EngineResult<u64> {
        // This engine holds back none of the chunks it states
        // (`ChunkFacts::held_bytes` is 0 at load): the answer repeats it.
        self.loaded_mut()?
            .weight_residency(residency)
            .map(|()| 0)
            .map_err(fault)
    }

    fn weight_uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        self.loaded_mut()?.weight_uses().map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::fire::Serves::NONE, engine::fire::Serves::of),
        )?;
        // The runtime's stamp of each step's stated shape, cross-checked
        // against this load's stated set in debug builds (CONTRACT-BC 24).
        if let Some(caps) = self.caps.as_ref() {
            frame.debug_check_shapes(&caps.steps);
        }
        if let Some(shell) = self.shell.as_ref() {
            let facts = &shell.trace().facts;
            for step in &frame.steps {
                step.validate_captures(|word| facts.holds(word, poem_ir::Builtin::Scores))?;
            }
        }
        // Test builds: what was given back is poisoned before any step.
        #[cfg(feature = "test-slot-fill")]
        if self.shell.is_some() {
            self.loaded_mut()?;
        }
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);
        // A frame's rows are held while it is the open one: the steps of one
        // its caller never settled are let go now.
        if let Some((_, unsettled)) = self.pending.take()
            && let Some(shell) = self.shell.as_mut()
        {
            for step in unsettled.iter().filter(|step| step.rows.is_none()) {
                shell.forget(&step.landed);
            }
        }

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut pending = Vec::with_capacity(frame.steps.len());
        let mut refused = None;
        for (index, step) in frame.steps.iter().enumerate() {
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            match self.fire_step(step, at) {
                Ok((ticket, step_pending)) => {
                    steps.push(ticket);
                    pending.push(step_pending);
                }
                // A step's lanes refused past the first step: the frame is
                // the steps before it (`engine::Error::Lanes`).
                Err(error) if error.stops_frame_at(index) => {
                    refused = Some(error);
                    break;
                }
                Err(error) => {
                    self.pending = None;
                    // The steps this frame fired before the one that failed:
                    // no one asks their rows.
                    if let Some(shell) = self.shell.as_mut() {
                        for step in &pending {
                            shell.forget(&step.landed);
                        }
                    }
                    self.tally_unbuilt();
                    return Err(error);
                }
            }
        }
        self.tally_unbuilt();
        let mut ticket = FrameTicket::of(id, steps, refused);
        self.pending = Some((id, pending));
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    fn settles_asynchronously(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.frames_in_flight() > 1)
    }

    fn live(&self) -> engine::live::Live {
        self.live.clone()
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already taken \
                 its readout seats",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 a step's rows live in the readout seat its arm owns. Ask for a \
                 frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut pending) = self.pending.take().expect("checked just above");
        let mut refused = None;
        match self.shell.as_mut() {
            None => {
                refused = Some(Error::Load("the metal engine has no model loaded".into()));
            }
            Some(shell) => {
                for step in &mut pending {
                    if step.rows.is_some() {
                        continue;
                    }
                    match shell.rows_of(&step.landed) {
                        Ok(rows) => step.rows = Some(rows),
                        Err(why) => {
                            refused = Some(fault(why));
                            break;
                        }
                    }
                }
            }
        }
        self.pending = Some((ticket.id, pending));
        if let Some(error) = refused {
            return Err(error);
        }
        let (_, pending) = self.pending.as_ref().expect("just put back");
        for (receipt, step) in ticket.steps.iter_mut().zip(pending) {
            receipt.readouts = readouts_of(step);
        }
        Ok(())
    }

    /// Commits the epilogues of every landed step (their sink was told from
    /// the completion handler) before the runtime takes their channels.
    fn publish_landed(&mut self) -> EngineResult<()> {
        if let Some(shell) = self.shell.as_mut() {
            shell.reap().map_err(fault)?;
        }
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
        }
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let shell = self.loaded_mut()?;
        let before = (shell.program_compilations(), shell.program_compile_nanos());
        let id = shell.register_program(registration).map_err(fault)?;
        // A registration after ready that compiled guest code is one the
        // load's prewarm did not prepare: built at its registration, before
        // its first fire (no fire waits on it). Its compile's wall time
        // reads an OS shader-cache hit apart from a cold compile.
        let built = shell.program_compilations() - before.0;
        if built > 0 {
            let ms = (shell.program_compile_nanos() - before.1) / 1_000_000;
            self.live.tally(engine::live::Tally::BuiltAtRegister, built);
            self.live.tally(engine::live::Tally::BuiltAtRegisterMs, ms);
            tracing::warn!(
                program = registration.program_hash,
                regions = built,
                ms,
                "metal: a registration after ready compiled guest code the prewarm did not"
            );
        }
        Ok(id)
    }

    fn prewarm_programs(&mut self, registrations: &[ProgramRegistration]) -> EngineResult<()> {
        self.loaded_mut()?
            .prewarm_programs(registrations)
            .map_err(fault)
    }

    /// The plan here; the libraries and pipelines in the compile, off this
    /// thread (cold, a sampler's take 0.1-0.4 s on an M1 Max).
    fn prepare_program(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let registering = self
            .loaded_mut()?
            .prepare_program(registration)
            .map_err(fault)?;
        Ok(Box::new(move || {
            engine::Prepared(Box::new(registering.compile()))
        }))
    }

    fn finish_program(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let registered = prepared
            .0
            .downcast::<crate::program::Registered>()
            .map_err(|_| {
                engine::Error::Program("a prepared program this engine did not prepare".into())
            })?;
        let shell = self.loaded_mut()?;
        let before = (shell.program_compilations(), shell.program_compile_nanos());
        let id = shell.finish_program(*registered).map_err(fault)?;
        // The compile's regions count at the finish: a registration after
        // ready that compiled guest code is one the prewarm did not prepare,
        // built at its registration (no fire waits on it). Its compile's
        // wall time, off this thread, reads a cache hit apart from a cold
        // compile.
        let built = shell.program_compilations() - before.0;
        if built > 0 {
            let ms = (shell.program_compile_nanos() - before.1) / 1_000_000;
            self.live.tally(engine::live::Tally::BuiltAtRegister, built);
            self.live.tally(engine::live::Tally::BuiltAtRegisterMs, ms);
            tracing::warn!(
                program = id.0,
                regions = built,
                ms,
                "metal: a registration after ready compiled guest code the prewarm did not"
            );
        }
        Ok(id)
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        self.loaded_mut()?.close_program(program).map_err(fault)
    }

    fn adapter_key(&self, binding: &engine::InstanceBinding) -> EngineResult<Option<u128>> {
        adapter_key_of(self.loaded()?, binding.program, &binding.seeds)
    }

    fn program_facts(&self, program: ProgramId) -> EngineResult<engine::ProgramFacts> {
        self.loaded()?.program_facts(program).map_err(fault)
    }

    /// A guest program's state: its instances' value scratch and its
    /// pair's batch sets, in `Instance` slots. Nothing of it lies in the
    /// span, so Metal states no fire term (`Engine::fire_facts`' default)
    /// and no standing bytes (`Engine::standing_after`'s).
    fn guest_state(
        &self,
        program: ProgramId,
        at: &engine::BindExtents,
        // The class bound: the engine half states per-class arms.
        _class: engine::ports::GeometryClass,
    ) -> EngineResult<engine::GuestState> {
        self.loaded()?
            .guest_state(program, &extents(at))
            .map_err(fault)
    }

    #[cfg(feature = "test-slot-fill")]
    fn poison_queue(&mut self, queue: engine::chunk::PoisonQueue) {
        self.poison = Some(queue);
    }

    fn rebind_slots(&mut self, rebind: engine::Rebind) -> EngineResult<()> {
        self.loaded_mut()?.rebind_slots(&rebind).map_err(fault)
    }

    /// A device channel's ring is a shared ring in its `Channel` slots; a
    /// channel the host reads or writes is the runtime's, each instance
    /// holding its own ring of it.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        registration.host_role == eta_ir::container::HostRole::None
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::RingFacts>> {
        if registration.host_role != eta_ir::container::HostRole::None {
            return Ok(None);
        }
        Ok(self.loaded()?.ring_facts(channel_shape(registration)))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        if registration.host_role != eta_ir::container::HostRole::None {
            return Err(self.unsupported("register_channel"));
        }
        let shape = channel_shape(registration);
        self.loaded_mut()?
            .register_shared_channel(registration.id, shape, &registration.slots)
            .map_err(fault)?;
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
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
        let shell = self.loaded_mut()?;
        let id = shell
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
                binding.device_resolved,
                &binding.channels,
                &binding.instance_slots,
                &binding.ring_slots,
                &binding.scratch,
                &binding.pair,
                binding.pair_generation,
            )
            .map_err(fault)?;
        let landed = adapter_of(shell, binding.program, id, &seeds, &binding.adapter_slots);
        let bound = match landed {
            Ok(bound) => bound,
            Err(why) => {
                let _ = shell.close_program_instance(id);
                return Err(why);
            }
        };
        if let Some(bound) = bound {
            self.adapters.insert(id, bound);
        }
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<u64> {
        self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        shell.detach_adapter(id);
        shell.close_program_instance(id).map_err(fault)?;
        Ok(0)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        if self.loaded_mut()?.close_shared_channel(id) {
            return Ok(());
        }
        Err(self.unsupported("close_channel"))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.instance(instance)?
            .publish(channel, cell)
            .map_err(fault)
    }

    fn beam_list(&mut self, instance: InstanceId, beam: u32) -> EngineResult<Vec<u32>> {
        self.loaded_mut()?.beam_list(instance, beam).map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.loaded_mut()?
            .program_instance_taking(instance, channel)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id: instance.0,
            })?
            .take(channel)
            .map_err(fault)
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if device.is_none_or(|device| copy.src != device || copy.dst != device) {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "metal",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }
}

impl Metal {
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                lanes: attachment.lanes,
                instance: attachment.instance,
                adapter_only: attachment.adapter_only,
            })
            .collect();

        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = self.loaded_mut()?.patch_element() else {
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            staged.reserve(submission.media.len());
            for row in &submission.media {
                staged.push(patch_bytes(&row.patches, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: "metal",
                    }
                })?);
            }
        }
        let mut voxel_payloads: Vec<Vec<u8>> = Vec::with_capacity(submission.voxels.len());
        if !submission.voxels.is_empty() && self.loaded_mut()?.voxel_element().is_none() {
            return Err(Error::Invalid(
                "this step submits VAE clips and the loaded plan states no voxel row; a \
                 model with no autoencoder has nothing to decode them with"
                    .into(),
            ));
        }
        let voxel_element = self
            .loaded_mut()?
            .voxel_element()
            .unwrap_or(poem_ir::Dtype::Bf16);
        for row in &submission.voxels {
            voxel_payloads.push(voxel_bytes(&row.payload, voxel_element).map_err(|why| {
                Error::Unsupported {
                    verb: why,
                    engine: "metal",
                }
            })?);
        }
        let clips: Vec<crate::serve::Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_payloads)
            .map(|(row, payload)| crate::serve::Clips {
                lane: row.lane,
                boxes: &row.clips,
                payload,
            })
            .collect();

        let media: Vec<crate::serve::Media<'_>> = submission
            .media
            .iter()
            .zip(&staged)
            .map(|(row, patches)| crate::serve::Media {
                lane: row.lane,
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                grids: &row.grids,
                token_positions: &row.token_positions,
            })
            .collect();

        let id = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);

        for (index, lane) in submission.lanes.iter().enumerate() {
            let listed = matches!(lane.readout, Readout::Rows(_));
            let served = attached
                .iter()
                .any(|a| a.lane as usize == index && !a.adapter_only);
            if listed && !served {
                return Err(Error::unsupported("metal", "row-selected readout"));
            }
        }

        let sink = self.sink.clone();
        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        if !self.adapters.is_empty() {
            for attachment in &submission.attachments {
                let Some(bound) = self.adapters.get(&attachment.instance) else {
                    continue;
                };
                if let Some(slot) = lane_adapters.get_mut(attachment.lane as usize) {
                    *slot = Some(*bound);
                }
            }
        }
        let shell = self.loaded_mut()?;
        let mut words: Vec<u64> = submission.lanes.iter().map(|lane| lane.word).collect();
        for (at, slot) in lane_adapters.iter().enumerate() {
            if slot.is_none() {
                continue;
            }
            words[at] = shell.adapted_word(words[at]).ok_or_else(|| {
                Error::Invalid(format!(
                    "lane {at} is attached to an instance that bound an adapter, and \
                     this load's model text has no corrected class for its fact word \
                     {:#x}: the text declares no `linear.lora_correct` arm, or the arm's \
                     window is not decided by one fact. A lane that asked for a \
                     correction and got the base model is the one wrong answer this \
                     axis refuses to give",
                    words[at]
                ))
            })?;
        }
        // A lane names its kv pages, its window pages and its state record
        // in its slot tables (CONTRACT-BC 4): this engine states one shared
        // pool, so the runtime states them. A device-geometry lane's list
        // table is its working set's whole mapped table, which its ports'
        // page indexes translate through.
        let mut pages_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut record_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut run_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut window_of: Vec<Vec<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut ring_of: Vec<Option<Vec<[u32; 2]>>> = Vec::with_capacity(submission.lanes.len());
        let mut forks: Vec<(u32, u32)> = Vec::new();
        for lane in &submission.lanes {
            pages_of.push(
                lane.tables
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::LIST)
                    .map(|(_, refs)| refs.clone()),
            );
            // One table a window list, in the load's order.
            let lists = shell.pool_layout().window_lists().count();
            let windows: Vec<Vec<u32>> = crate::store::layout::WINDOWS[..lists]
                .iter()
                .map(|id| {
                    lane.tables
                        .iter()
                        .find(|(shape, _)| shape == id)
                        .map(|(_, refs)| refs.clone())
                        .unwrap_or_default()
                })
                .collect();
            window_of.push(windows);
            ring_of.push(match &lane.record {
                engine::fire::LaneRecord::Pieces(pieces) => pieces
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::RING)
                    .map(|(_, pieces)| pieces.clone()),
                _ => None,
            });
            // A load that holds state is told every lane's record: its
            // pieces, or none (the null record).
            let fixed: Option<Vec<[u32; 2]>> = match &lane.record {
                engine::fire::LaneRecord::Pieces(pieces) => pieces
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::FIXED)
                    .map(|(_, pieces)| pieces.clone()),
                engine::fire::LaneRecord::Null => shell
                    .null_record()
                    .map(|null| null.iter().map(|&r| [r, r]).collect()),
                engine::fire::LaneRecord::Unstated if shell.null_record().is_some() => {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} states no record, and this load holds recurrent \
                         state: the runtime states every lane's record, or that it has none",
                        lane.slot
                    )));
                }
                engine::fire::LaneRecord::Unstated => None,
            };
            let fixed = fixed.as_ref();
            run_of.push(fixed.and_then(|pieces| {
                pieces
                    .iter()
                    .any(|&[run, cow]| run != cow)
                    .then(|| pieces.iter().map(|&[run, _]| run).collect())
            }));
            record_of.push(fixed.map(|pieces| {
                // `cow != run`: the record is copied before the fire, once
                // for every lane of a group naming the same pair.
                for &[run, cow] in pieces {
                    if run != cow && !forks.contains(&(run, cow)) {
                        forks.push((run, cow));
                    }
                }
                pieces.iter().map(|&[_, cow]| cow).collect()
            }));
        }
        let mut ring_forks: Vec<(u32, u32)> = Vec::new();
        for pieces in ring_of.iter().flatten() {
            for &[run, cow] in pieces {
                if run != cow && !ring_forks.contains(&(run, cow)) {
                    ring_forks.push((run, cow));
                }
            }
        }
        if !ring_forks.is_empty() {
            eprintln!("ringfork {} pair(s), first {:?}", ring_forks.len(), ring_forks.first());
        }
        if !forks.is_empty() || !ring_forks.is_empty() {
            shell.copy_records(&forks, &ring_forks).map_err(fault)?;
        }
        // Tail copies (an rs-list or kv slot a fork shares) land before any
        // lane of the fire reads or writes (CONTRACT-BC 20).
        let slot_copies: Vec<_> = submission
            .lanes
            .iter()
            .flat_map(|lane| lane.slot_copies.iter().cloned())
            .collect();
        if !slot_copies.is_empty() || !forks.is_empty() {
            eprintln!(
                "wanomaly submit slot_copies {} forks {} first copy {:?}",
                slot_copies.len(),
                forks.len(),
                slot_copies.first()
            );
        }
        shell.copy_slot_pairs(&slot_copies).map_err(fault)?;
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                Ok(Seated {
                    replay_stated: lane.replay_stated,
                    lane: Lane {
                        slot: lane.slot,
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    group: lane.group,
                    inputs: &lane.inputs,
                    pages: pages_of[at].as_deref().unwrap_or(&[]),
                    held: pages_of[at].is_some().then_some(lane.kv.held),
                    folded: lane.kv.folded,
                    readable: lane.kv.readable,
                    set: lane.kv.set,
                    mask: lane.mask.as_ref(),
                    kv_less: lane.kv_less,
                    adapter: lane_adapters[at].or(lane.adapter),
                    positions: &lane.positions,
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    // A beam group's lane resolves through its working set's
                    // whole reserve (the pool its list's entries name,
                    // BEAM-PAGE-LISTS §6.5); any other lane through its own
                    // table.
                    translation: if lane.reserve.is_empty() {
                        match lane.list {
                            engine::fire::ListKind::Translation => {
                                pages_of[at].as_deref().unwrap_or(&[])
                            }
                            engine::fire::ListKind::Pages => &[],
                        }
                    } else {
                        &lane.reserve
                    },
                    rs: &lane.rs,
                    rs_reset: lane.rs_reset,
                    rs_snapshots: &lane.rs_snapshots,
                    refs: true,
                    record: record_of[at].as_deref(),
                    record_run: run_of[at].as_deref(),
                    windows: &window_of[at],
                    ring: ring_of[at].as_deref(),
                    window: submission.lanes[at].window,
                    reports: lane.reports,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.map(|sink| Done { at, sink });
            let prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                    clips: &clips,
                    done,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
        };

        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            PendingStep {
                readout: submission
                    .lanes
                    .iter()
                    .map(|lane| lane.readout.clone())
                    .collect(),
                landed,
                rows: None,
            },
        ))
    }
}

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[(String, Vec<f32>)] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| match policy {
            Readout::None | Readout::Rows(_) => LaneReadout::default(),
            Readout::Last => {
                let (output, values) = rows.get(lane).cloned().unwrap_or_default();
                LaneReadout {
                    rows: 1,
                    width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    output,
                    values,
                    ..LaneReadout::default()
                }
            }
        })
        .collect()
}

// SAFETY: a loaded `Shell` holds retained Metal objects `objc2` doesn't mark
// `Send`, plus `RefCell`s and no `Rc` or thread-local state. Moving the engine
// moves sole ownership of all of them, and Metal's device, queue, pipeline and
// resource objects are documented usable from any thread.
unsafe impl Send for Metal {}

/// The ring a registered channel declares.
fn channel_shape(registration: &ChannelRegistration) -> crate::program::ChannelShape {
    crate::program::ChannelShape {
        capacity: registration.capacity.max(1),
        numel: registration
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1),
        dtype: registration.dtype.program_dtype(),
    }
}

type PipelineKeyText = (String, String, String, Option<[u64; 2]>);

/// The keys a boot of the load `key` listed at `path`: `None` where there
/// is no list, it was written for another key, or a line does not parse.
fn read_pipeline_keys(path: &std::path::Path, key: &str) -> Option<Vec<PipelineKeyText>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if lines.next()? != key {
        return None;
    }
    lines
        .map(|line| {
            let mut parts = line.split('\t');
            let file = parts.next()?.to_string();
            let entry = parts.next()?.to_string();
            let stamp = parts.next()?.to_string();
            let strides = match parts.next()? {
                "-" => None,
                pair => {
                    let (k, v) = pair.split_once(',')?;
                    Some([k.parse().ok()?, v.parse().ok()?])
                }
            };
            Some((file, entry, stamp, strides))
        })
        .collect()
}

/// Writes the load `key`'s pipeline keys to `path`; a failed write only
/// costs the next boot a walk.
fn write_pipeline_keys(path: &std::path::Path, key: &str, keys: &[PipelineKeyText]) {
    let mut text = format!("{key}\n");
    for (file, entry, stamp, strides) in keys {
        let strides = strides.map_or_else(|| "-".to_string(), |[k, v]| format!("{k},{v}"));
        text.push_str(&format!("{file}\t{entry}\t{stamp}\t{strides}\n"));
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, text);
}

#[cfg(test)]
mod told {
    use super::*;

    /// A load whose ruling picks a top rung under the asked fire tells that
    /// rung's tokens as its widest fire, and its lanes and buckets as the
    /// rung walks them; one ruled at what it asked tells what it asked.
    #[test]
    fn the_told_limit_is_the_ruled_top_rung() {
        let asked = LoadBudgets {
            max_lanes: 2048,
            max_tokens: 4096,
            buckets: vec![1, 64, 512, 2048, 4096],
            ..LoadBudgets::default()
        };
        let baked = Budget {
            max_lanes: asked.max_lanes,
            max_tokens: asked.max_tokens,
            buckets: asked.buckets.clone(),
            max_adapters: 0,
            readouts_per_lane: 1,
        };
        for top in [1024, 512, 64] {
            let served = crate::top::bounded(&baked, top);
            let mut told = asked.clone();
            told_within(&mut told, &served);
            assert_eq!(told.max_tokens, top, "the told widest fire is the ruled top rung");
            assert_eq!(told.max_lanes, top.min(asked.max_lanes));
            assert_eq!(told.buckets, served.buckets);
            assert!(told.buckets.iter().all(|&rows| rows <= top));
        }
        let mut told = asked.clone();
        told_within(&mut told, &crate::top::bounded(&baked, asked.max_tokens));
        assert_eq!(told.max_tokens, asked.max_tokens);
        assert_eq!(told.max_lanes, asked.max_lanes);
    }
}

/// The device's memory is the host's: a host tier would copy slots within
/// the same DRAM, freeing nothing physical and only moving them between
/// budgets. Suspended slots go to disk; a host budget stated for this
/// device is refused by name, not ignored.
fn no_host_tier(host_kv_budget: Option<u64>) -> EngineResult<()> {
    match host_kv_budget.filter(|&budget| budget > 0) {
        Some(budget) => Err(Error::Load(format!(
            "budget_host ({budget} bytes) on Metal: the device's memory is the host's, so \
             there is no host kv tier (a move there would copy within the same memory and \
             free nothing); suspended state goes to the disk tier where budget_disk states one. Remove \
             budget_host from [engine.memory]"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod no_host_tier_tests {
    /// A host budget on a device whose memory is the host's is refused by
    /// name; none stated (or zero) loads.
    #[test]
    fn a_host_budget_on_metal_is_refused_by_name() {
        let refused = super::no_host_tier(Some(8 << 30)).expect_err("refused");
        let text = refused.to_string();
        assert!(text.contains("budget_host") && text.contains("the device's memory is the host's"), "{text}");
        assert!(super::no_host_tier(None).is_ok());
        assert!(super::no_host_tier(Some(0)).is_ok());
    }
}

#[cfg(test)]
mod placement {
    use super::*;

    /// A fire naming a state slot inside the activation's carve is refused
    /// as a placement, naming the logical chunk: the runtime moves that
    /// state down and submits again, failing no lane.
    #[test]
    fn a_slot_under_the_carve_is_refused_naming_its_chunk() {
        let refused = fault(Fault::UnderCarve {
            logical: 2057,
            physical: 2057,
            carve: 482_998_272,
            range: 4_162_532_352,
        });
        let Error::Placement { chunks, reason } = &refused else {
            panic!("a placement refusal: {refused:?}");
        };
        assert_eq!(chunks, &vec![2057]);
        assert!(reason.contains("activation span's backed carve"), "{reason}");
        assert!(refused.to_string().starts_with("impossible submission: "), "{refused}");
    }
}
