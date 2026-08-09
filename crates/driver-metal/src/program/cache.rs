//! The M1 runtime shell and its program compile.
//!
//! `M1Runtime::compile_program` is 718 lines of C++ because the cache
//! identity, the three caches, the emitted-kernel index, the metadata walk
//! and the RNG splice are all written inline. Every one of those is a
//! portable module of this crate now — [`identity`], [`cache`],
//! [`stage_cache`], [`emitted`], [`meta`], [`shader`] — so what is left here
//! is the walk itself: validate the plan, collect every kernel the program
//! needs, compile them as one batch, and assemble the executables.
//!
//! ## The transaction dissolves
//!
//! The C++ wraps the whole compile in a `PsoCompileTransaction`, because it
//! *installs as it builds*: every PSO is registered with the context and
//! every ordinal is taken from the shared counter the moment it exists, so a
//! failure halfway needs an object whose destructor walks the registrations
//! back and resets the counter. Here nothing is installed until everything
//! has compiled — the PSOs live in a local vector that releases itself, the
//! ordinal counter is written back in the last statement of the success
//! path, and the stage cache's pending half is only staged at assembly, past
//! the last failure exit. Rollback is not implemented; it is what happens.
//!
//! ## What else the C++ did that this does not
//!
//! * **"Cache full" was a retryable failure.** Both the program and the
//!   stage cache refused their 65th entry forever; the caller retried
//!   against the one condition retrying cannot change. [`Bounded`] and
//!   [`Stages`] evict instead — argued in their own modules.
//! * **An in-flight signature collision blamed the program.** Two stages of
//!   one program sharing a signature hash with different identities was
//!   `reject_deterministic` — written into the negative cache against a
//!   program that is the collision's victim, not its cause. Here the second
//!   stage simply builds unshared.
//! * **Per-region archives become one archive per program.** The C++ writes
//!   `<identity>/region-N.mtl4archive` per kernel; here the whole batch is
//!   archived under one key — device, combined signature, versions — by a
//!   compiler created for this build, so the archive holds exactly this
//!   program's binaries. What is given up: a *new* program that shares a
//!   stage with an old one recompiles that stage once after a restart. The
//!   in-memory stage cache still dedups within a run.
//! * **`PIE_METAL_PTIR_TEST_FAIL_COMPILE_ONCE`** — an env var that makes one
//!   compile fail, needed because the C++ caches were unreachable from
//!   tests. A test here hands `compile` a source that does not compile.
//!
//! [`identity`]: crate::channel::cache_identity
//! [`cache`]: crate::channel::Bounded
//! [`stage_cache`]: crate::channel::Stages
//! [`emitted`]: crate::channel::Emitted
//! [`meta`]: crate::channel::op_metadata
//! [`shader`]: crate::layout::shader

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_FUSED, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS,
    PIE_KERNEL_SINGLETON, PIE_REGION_LIBRARY,
};
use driver_api::plan::{EmittedKernel, LaunchOp};
use tensor_ir::fnv1a64;
use tensor_ir::op::tags;
use tensor_ir::registry::Stage;

use crate::channel::{
    Backend, Bounded, CacheStats, Emitted, ExecPlan, Failure, Lookup, MAX_CHANNELS,
    MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions,
    cache_identity, channel_effects, combined_signature, op_metadata, port_consumes,
};
use crate::device::archive::Archives;
use crate::device::context::Context;
use crate::layout::shader;
use crate::program::compile::{Compiler, Math};
use crate::program::executable::{
    FusedExecutable, GroupedExecutable, ProgramExecutable, ProgramStage, Pso, RegionExecutable,
    StageExecutable,
};
use crate::{Error, Result};

/// `PTIR_OP_CHAN_TAKE`, as the plan encodes it.
const CHAN_TAKE: u16 = tags::CHAN_TAKE as u16;
/// `PTIR_OP_CHAN_READ`.
const CHAN_READ: u16 = tags::CHAN_READ as u16;
/// `PTIR_OP_CHAN_PUT`.
const CHAN_PUT: u16 = tags::CHAN_PUT as u16;
/// `PTIR_OP_TOP_K`.
const TOP_K: u16 = tags::TOP_K as u16;

/// `PTIR_LIBRARY_NUCLEUS_SAMPLE`. Mirror of
/// `tensor_compiler::plan::LibraryOp::NucleusSample`, checked by a
/// dev-dependency test.
const LIBRARY_NUCLEUS_SAMPLE: u8 = 0;
/// `PTIR_LIBRARY_TOP_K`. Mirror of `LibraryOp::TopK`.
const LIBRARY_TOP_K: u8 = 1;

/// The most channel slots one stage may bind directly on the fused path.
///
/// `kMetalM2MaxFusedChannels`: a fused kernel binds two buffers per channel
/// after seven fixed ones, and MSL stops at `[[buffer(30)]]`.
pub const MAX_FUSED_CHANNELS: usize = 12;

/// The most singleton regions one stage may compile.
pub const MAX_REGIONS_PER_STAGE: usize = 256;

/// The most singleton regions one program may compile across its stages.
pub const MAX_REGIONS_PER_PROGRAM: usize = 1024;

/// First argument-table ordinal the launch path may claim.
///
/// The decode/prefill DAGs and this runtime share one ordinal namespace, and
/// they were once separated only by both being small: a prefill sized for
/// 512 rows reaches 264192, which ran straight through the old 100000 base.
/// This is the C++ `kPrefillOrdinalLimit`: the prefill base (3 strides of
/// 4096) plus 1024 rows of one 4096-ordinal stride each.
pub const ORDINAL_BASE: u32 = (3 + 1024) * 4096;

/// A deterministic failure, in one breath.
fn det(reason: impl Into<String>) -> Failure {
    Failure::Deterministic {
        reason: reason.into(),
    }
}

/// Queue one compile job and remember how to blame it.
fn push(
    jobs: &mut Vec<(String, String)>,
    labels: &mut Vec<String>,
    source: String,
    entry: &str,
    label: String,
) -> usize {
    jobs.push((source, entry.to_owned()));
    labels.push(label);
    jobs.len() - 1
}

/// Take the next ordinal.
fn take(counter: &mut u32) -> u32 {
    let ordinal = *counter;
    *counter += 1;
    ordinal
}

/// One stage of the program being built, before its PSOs exist.
///
/// Job indices into the batch, resolved to pipelines at assembly.
struct Building {
    /// Index of the stage in the program.
    index: usize,
    /// The stage-cache key (hash of the cache identity string).
    key: u64,
    /// The graph identity checked against the key.
    identity: u64,
    /// One job per singleton region, in op order.
    singleton: Vec<usize>,
    /// `(job, fused region index)` per fused region, or the host's refusal.
    fused: std::result::Result<Vec<(usize, usize)>, String>,
    /// `(job, fused region index, nucleus, topk)` per grouped region, or the
    /// host's refusal.
    grouped: std::result::Result<Vec<(usize, usize, bool, bool)>, String>,
    /// `(job, singleton region index)` per grouped-singleton region.
    grouped_singleton: Vec<(usize, usize)>,
}

/// How one program stage will be satisfied.
enum Piece {
    /// The stage cache already holds it.
    Cached(Rc<StageExecutable>),
    /// An earlier stage of this same program is building it.
    Same(usize),
    /// It compiles in this batch.
    Built(Building),
}

/// The M1 runtime: the caches, the ordinal counter, and the compile.
///
/// `M1Runtime`, minus the context it owned — every method takes the
/// [`Context`] it runs against, because the runtime's state is caches and
/// counters, not device objects.
pub struct Runtime {
    /// Where the staged RNG preamble (and any other spliceable source) lives.
    kernels_dir: PathBuf,
    /// `ptir_rng.generated.metal`, read once at creation.
    ///
    /// The host emitter embeds the M1 runtime template into every emitted
    /// source, but that template still carries
    /// `#include "ptir/ptir_rng.generated.metal"`, and Metal's runtime
    /// compiler resolves no includes of its own.
    rng_preamble: String,
    /// Where program archives are looked up and written.
    archives: Archives,
    /// The next unclaimed argument-table ordinal.
    next_ordinal: u32,
    /// Compiled programs by registration hash.
    programs: Bounded<u64, Rc<ProgramExecutable>>,
    /// Compiled stages by cache identity, shared across programs.
    stages: Stages<Rc<StageExecutable>>,
    /// Deterministic failures by registration hash.
    negative: Bounded<u64, String>,
    /// The emitter-shared grouped effect pair, kept across programs.
    grouped_effects: Option<(Pso, Pso)>,
    /// What the caches did.
    stats: CacheStats,
}

impl Runtime {
    /// Create the runtime against a kernels directory and an archive cache.
    ///
    /// # Errors
    ///
    /// Fails when the RNG preamble under `kernels_dir` cannot be read or is
    /// empty — better one refusal at creation than a compile error inside
    /// every emitted kernel later.
    pub fn new(kernels_dir: impl Into<PathBuf>, archives: Archives) -> Result<Self> {
        let kernels_dir = kernels_dir.into();
        let rng_preamble = shader::read_source(kernels_dir.join("ptir/ptir_rng.generated.metal"))?;
        if rng_preamble.trim().is_empty() {
            return Err(Error::Create {
                what: "M1 runtime",
                message: format!(
                    "the RNG preamble under {} is empty; every emitted kernel splices it",
                    kernels_dir.display()
                ),
            });
        }
        Ok(Self {
            kernels_dir,
            rng_preamble,
            archives,
            next_ordinal: ORDINAL_BASE,
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            grouped_effects: None,
            stats: CacheStats::default(),
        })
    }

    /// What the caches have done so far.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Compiled programs held.
    #[must_use]
    pub fn program_entries(&self) -> usize {
        self.programs.len()
    }

    /// Compiled stages held.
    #[must_use]
    pub fn stage_entries(&self) -> usize {
        self.stages.len()
    }

    /// Remembered deterministic failures.
    #[must_use]
    pub fn negative_entries(&self) -> usize {
        self.negative.len()
    }

    /// Claim the next argument-table ordinal.
    ///
    /// For the placed paths, whose per-command ordinals are allocated at
    /// prepare time rather than at compile.
    pub(super) fn next_ordinal(&mut self) -> u32 {
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        ordinal
    }

    /// The grouped effect pair, once a program compile has produced it.
    pub(super) fn grouped_effects_pair(&self) -> Option<(Pso, Pso)> {
        self.grouped_effects.clone()
    }

    /// Compile `plan` into an executable program, through the caches.
    ///
    /// `versions` must come from the registration that shipped the kernels —
    /// the emitter version in particular is the host's, never a driver-side
    /// copy; the copy this replaces said 23 while the host said 36.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] when the program can never compile — the
    /// answer is remembered and replayed. [`Failure::Retryable`] when it
    /// could not compile *now*; nothing is remembered, nothing is installed,
    /// and the ordinal counter is untouched.
    pub fn compile(
        &mut self,
        context: &Context,
        program_hash: u64,
        plan: &ExecPlan,
        versions: Versions,
        kernels: &[EmittedKernel],
    ) -> std::result::Result<Rc<ProgramExecutable>, Failure> {
        if let Some(program) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(Rc::clone(program));
        }
        if let Some(reason) = self.negative.get(&program_hash) {
            let reason = reason.clone();
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic { reason });
        }
        let outcome = self.build(context, program_hash, plan, versions, kernels);
        if let Err(failure) = &outcome
            && failure.is_remembered()
            && self
                .negative
                .insert(program_hash, failure.reason().to_owned())
                .is_some()
        {
            self.stats.evictions += 1;
        }
        outcome
    }

    /// The compile itself: validate, collect, compile once, assemble.
    #[allow(clippy::too_many_lines)]
    fn build(
        &mut self,
        context: &Context,
        program_hash: u64,
        plan: &ExecPlan,
        versions: Versions,
        kernels: &[EmittedKernel],
    ) -> std::result::Result<Rc<ProgramExecutable>, Failure> {
        let emitted = Emitted::index(kernels).map_err(|duplicate| {
            det(format!(
                "host emitted two kernels for kind {} stage {} region {}",
                duplicate.kind, duplicate.stage, duplicate.region
            ))
        })?;

        if !plan.executable {
            return Err(det(plan
                .reject_reason
                .clone()
                .unwrap_or_else(|| "Metal M1 plan is not executable".to_owned())));
        }
        let package = &plan.package;
        let plans = &package.plans;

        // A forward-needing program may only have prologue and epilogue
        // stages, and a prologue makes it M2-only: its dispatches must be
        // placed around the forward rather than run on the single-lane path.
        let mut requires_m2_placement = false;
        if plan.needs_forward() {
            for stage in &package.stages {
                match Stage::from_u8(stage.kind) {
                    Some(Stage::Prologue) => requires_m2_placement = true,
                    Some(Stage::Epilogue) => {}
                    _ => return Err(det("Metal rejects forward-needing per-layer stages")),
                }
            }
            // A prologue put into a channel that a host-resolved descriptor
            // port reads would ask the descriptor phase to observe a value
            // that only exists once the placed command runs.
            let mut descriptor_channels = vec![false; package.channels.len()];
            for port in &package.ports {
                if !port.is_const
                    && let Some(flag) = descriptor_channels.get_mut(port.channel as usize)
                {
                    *flag = true;
                }
            }
            for (index, stage) in package.stages.iter().enumerate() {
                if Stage::from_u8(stage.kind) != Some(Stage::Prologue) {
                    continue;
                }
                let stage_plan = &plans[index];
                for op in &stage_plan.ops {
                    if op.code != CHAN_PUT || op.channel == u32::MAX {
                        continue;
                    }
                    let Some(&dense) = stage_plan.channel_bindings.get(op.channel as usize) else {
                        continue;
                    };
                    if descriptor_channels.get(dense as usize).copied() == Some(true) {
                        return Err(det(
                            "Metal M2 cannot consume a prologue pending value through a \
                             host-resolved descriptor",
                        ));
                    }
                }
            }
        }
        if package.channels.len() > MAX_CHANNELS {
            return Err(det(format!(
                "Metal M1 supports at most {MAX_CHANNELS} channel slots per lane"
            )));
        }

        // Per-stage structural checks and the result-base walk, before any
        // compile is attempted. The region bound takes the larger of the two
        // counts the C++ used inconsistently — it capped on the partition's
        // region count and compiled one region per op.
        let mut metas = Vec::with_capacity(plans.len());
        let mut total_regions = 0usize;
        for (index, stage_plan) in plans.iter().enumerate() {
            for &binding in &stage_plan.channel_bindings {
                if binding as usize >= package.channels.len() {
                    return Err(det("Metal M1 stage channel binding is out of range"));
                }
            }
            for op in &stage_plan.ops {
                if op.code != CHAN_TAKE && op.code != CHAN_READ && op.code != CHAN_PUT {
                    continue;
                }
                if op.channel == u32::MAX
                    || op.channel as usize >= stage_plan.channel_bindings.len()
                {
                    return Err(det("Metal M1 stage channel op has no binding slot"));
                }
            }
            let regions = stage_plan.singleton.len().max(stage_plan.ops.len());
            if regions > MAX_REGIONS_PER_STAGE || total_regions + regions > MAX_REGIONS_PER_PROGRAM
            {
                return Err(det(
                    "Metal M1 singleton executable exceeds the bounded region cache",
                ));
            }
            total_regions += regions;
            // A whole-stage singleton rejection arrives as a refusal at
            // region 0: the host validated the stage and declined it.
            if let Slot::Refused(reason) = emitted.get(PIE_KERNEL_SINGLETON, index as u32, 0) {
                return Err(det(reason));
            }
            metas.push(
                op_metadata(&stage_plan.ops, stage_plan.value_types.len())
                    .map_err(|malformed| det(malformed.reason()))?,
            );
        }

        // The program-wide channel effects. A consuming descriptor port is a
        // take the descriptor phase performs — the ops cannot say so, so it
        // is said here, as a synthetic op list bound by the identity table.
        let port_ops: Vec<LaunchOp> = package
            .ports
            .iter()
            .filter(|port| !port.is_const)
            .map(|port| LaunchOp {
                code: if port_consumes(port.port) {
                    CHAN_TAKE
                } else {
                    CHAN_READ
                },
                channel: port.channel,
                ..LaunchOp::default()
            })
            .collect();
        let identity_bindings: Vec<u32> = (0..package.channels.len() as u32).collect();
        let mut effect_stages: Vec<(&[LaunchOp], &[u32])> = plans
            .iter()
            .map(|stage_plan| {
                (
                    stage_plan.ops.as_slice(),
                    stage_plan.channel_bindings.as_slice(),
                )
            })
            .collect();
        effect_stages.push((port_ops.as_slice(), identity_bindings.as_slice()));
        let effects = channel_effects(&package.channels, &effect_stages)
            .map_err(|bad| det(format!("channel {}: {}", bad.channel, bad.reason())))?;

        // Collect every kernel the program needs, as one batch.
        let mut jobs: Vec<(String, String)> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        let mut pieces: Vec<Piece> = Vec::new();
        let mut in_flight: HashMap<u64, (usize, u64)> = HashMap::new();

        for (index, stage_plan) in plans.iter().enumerate() {
            let identity_string = cache_identity(
                Backend::Metal,
                context.cache_id(),
                stage_plan.signature_hash,
                versions,
            );
            let key = fnv1a64(identity_string.as_bytes());
            if let Some(&(piece, identity)) = in_flight.get(&key) {
                if identity == stage_plan.identity {
                    self.stats.memory_hits += 1;
                    pieces.push(Piece::Same(piece));
                    continue;
                }
                // Two stages of one program share a key with different
                // identities. The C++ called that a program error; it is a
                // collision, so the second stage builds unshared.
            } else if let (Lookup::Hit, Some(shared)) = self.stages.lookup(key, stage_plan.identity)
            {
                self.stats.memory_hits += 1;
                pieces.push(Piece::Cached(shared));
                continue;
            } else {
                in_flight.insert(key, (pieces.len(), stage_plan.identity));
            }

            let mut singleton = Vec::with_capacity(metas[index].len());
            for region in 0..metas[index].len() {
                let slot = emitted.get(PIE_KERNEL_SINGLETON, index as u32, region as u32);
                let Slot::Kernel { source, entry } = slot else {
                    let mut reason = format!(
                        "Metal M1 host emitter missing singleton kernel for stage {index} \
                         region {region}"
                    );
                    if let Slot::Refused(why) = slot {
                        reason = format!("{reason}: {why}");
                    }
                    return Err(det(reason));
                };
                let source = self.splice(source)?;
                singleton.push(push(
                    &mut jobs,
                    &mut labels,
                    source,
                    entry,
                    format!("Metal M1 compile failed for {entry}"),
                ));
            }

            // The fused path. A host refusal is data — the stage drops to
            // the singleton fallback — and it discards the stage's fused
            // kernels collected so far rather than compiling them for
            // nothing.
            let fused = if stage_plan.channel_bindings.len() > MAX_FUSED_CHANNELS {
                Err(format!(
                    "fused region exceeds the {MAX_FUSED_CHANNELS}-channel direct-binding limit"
                ))
            } else {
                let mut sources = Vec::new();
                let mut refusal = None;
                for region in 0..stage_plan.fused.len() {
                    match emitted.get(PIE_KERNEL_FUSED, index as u32, region as u32) {
                        Slot::Kernel { source, entry } => {
                            sources.push((self.splice(source)?, entry.to_owned(), region));
                        }
                        Slot::Absent => {
                            return Err(det(format!(
                                "Metal M2 host emitter missing fused kernel for stage {index} \
                                 region {region}"
                            )));
                        }
                        Slot::Refused(why) => {
                            refusal = Some(why.to_owned());
                            break;
                        }
                        Slot::Malformed => {
                            refusal = Some(format!(
                                "host declined the fused kernel for stage {index} region \
                                 {region} without a reason"
                            ));
                            break;
                        }
                    }
                }
                match refusal {
                    Some(reason) => Err(reason),
                    None => Ok(sources
                        .into_iter()
                        .map(|(source, entry, region)| {
                            (
                                push(
                                    &mut jobs,
                                    &mut labels,
                                    source,
                                    &entry,
                                    "Metal M2 fused compile failed".to_owned(),
                                ),
                                region,
                            )
                        })
                        .collect()),
                }
            };

            // Grouped-singleton occupies KERNEL_GROUPED regions
            // [0, singleton count); there is no fallback past it, so a miss
            // here is deterministic.
            let mut grouped_singleton = Vec::with_capacity(stage_plan.singleton.len());
            for region in 0..stage_plan.singleton.len() {
                match emitted.get(PIE_KERNEL_GROUPED, index as u32, region as u32) {
                    Slot::Kernel { source, entry } => {
                        let source = self.splice(source)?;
                        grouped_singleton.push((
                            push(
                                &mut jobs,
                                &mut labels,
                                source,
                                entry,
                                "Metal M3 grouped singleton compile failed".to_owned(),
                            ),
                            region,
                        ));
                    }
                    Slot::Refused(why) => {
                        return Err(det(format!(
                            "Metal M3 grouped singleton emission failed: {why}"
                        )));
                    }
                    Slot::Absent | Slot::Malformed => {
                        return Err(det(format!(
                            "Metal M3 grouped singleton emission failed: host emitter produced \
                             no source for stage {index} region {region}"
                        )));
                    }
                }
            }

            // Grouped-fused occupies the tail of KERNEL_GROUPED, offset past
            // the singleton block. A refusal drops the stage to the
            // grouped-singleton fallback.
            let grouped = {
                let mut sources = Vec::new();
                let mut refusal = None;
                for (region, region_plan) in stage_plan.fused.iter().enumerate() {
                    let slot = emitted.get(
                        PIE_KERNEL_GROUPED,
                        index as u32,
                        (stage_plan.singleton.len() + region) as u32,
                    );
                    match slot {
                        Slot::Kernel { source, entry } => {
                            let library = region_plan.kind == PIE_REGION_LIBRARY;
                            let parallel_nucleus =
                                library && region_plan.library == LIBRARY_NUCLEUS_SAMPLE;
                            // The C++ indexes `ops[nodes.front()]` unchecked;
                            // a node index past the ops is UB there and a
                            // plain "not the parallel path" here.
                            let parallel_topk = library
                                && region_plan.library == LIBRARY_TOP_K
                                && region_plan
                                    .nodes
                                    .first()
                                    .and_then(|&node| stage_plan.ops.get(node as usize))
                                    .is_some_and(|op| op.code == TOP_K);
                            sources.push((
                                self.splice(source)?,
                                entry.to_owned(),
                                region,
                                parallel_nucleus,
                                parallel_topk,
                            ));
                        }
                        Slot::Absent => {
                            return Err(det(format!(
                                "Metal M3 host emitter missing grouped kernel for stage {index} \
                                 fused region {region}"
                            )));
                        }
                        Slot::Refused(why) => {
                            refusal = Some(why.to_owned());
                            break;
                        }
                        Slot::Malformed => {
                            refusal = Some(format!(
                                "host declined the grouped kernel for stage {index} fused \
                                 region {region} without a reason"
                            ));
                            break;
                        }
                    }
                }
                match refusal {
                    Some(reason) => Err(reason),
                    None => Ok(sources
                        .into_iter()
                        .map(|(source, entry, region, nucleus, topk)| {
                            (
                                push(
                                    &mut jobs,
                                    &mut labels,
                                    source,
                                    &entry,
                                    "Metal M3 grouped compile failed".to_owned(),
                                ),
                                region,
                                nucleus,
                                topk,
                            )
                        })
                        .collect()),
                }
            };

            pieces.push(Piece::Built(Building {
                index,
                key,
                identity: stage_plan.identity,
                singleton,
                fused,
                grouped,
                grouped_singleton,
            }));
        }

        // An M2-required program with a stage that cannot fuse can never run:
        // its dispatches must be placed around the forward and there is no
        // fused executable to place.
        if requires_m2_placement {
            for piece in &pieces {
                let reason = match piece {
                    Piece::Cached(stage) => stage.fused.as_ref().err().cloned(),
                    Piece::Built(building) => building.fused.as_ref().err().cloned(),
                    // Points at an earlier piece, already checked.
                    Piece::Same(_) => None,
                };
                if let Some(reason) = reason {
                    return Err(det(format!(
                        "Metal M2-required stage has no fused executable: {reason}"
                    )));
                }
            }
        }

        // The effect kernels. Two families, not interchangeable: the
        // single-lane pair has this program's channel effects baked in, the
        // grouped pair reads the same decisions from per-channel flag words.
        // The host emits the grouped pair at (stage 0, region 0) and the
        // per-program pair at (stage 0, region 1).
        let slot = emitted.get(PIE_KERNEL_READINESS, 0, 1);
        let Slot::Kernel {
            source: readiness_source,
            entry: readiness_entry,
        } = slot
        else {
            let mut reason = "Metal readiness kernel missing from host emission".to_owned();
            if let Slot::Refused(why) = slot {
                reason = format!("{reason}: {why}");
            }
            return Err(det(reason));
        };
        let slot = emitted.get(PIE_KERNEL_COMMIT, 0, 1);
        let Slot::Kernel {
            source: commit_source,
            entry: commit_entry,
        } = slot
        else {
            let mut reason = "Metal commit kernel missing from host emission".to_owned();
            if let Slot::Refused(why) = slot {
                reason = format!("{reason}: {why}");
            }
            return Err(det(reason));
        };
        let readiness_source = self.splice(readiness_source)?;
        let readiness_job = push(
            &mut jobs,
            &mut labels,
            readiness_source,
            readiness_entry,
            "Metal M1 readiness compile failed".to_owned(),
        );
        let commit_source = self.splice(commit_source)?;
        let commit_job = push(
            &mut jobs,
            &mut labels,
            commit_source,
            commit_entry,
            "Metal M1 commit compile failed".to_owned(),
        );

        let (
            Slot::Kernel {
                source: grouped_readiness_source,
                entry: grouped_readiness_entry,
            },
            Slot::Kernel {
                source: grouped_commit_source,
                entry: grouped_commit_entry,
            },
        ) = (
            emitted.get(PIE_KERNEL_READINESS, 0, 0),
            emitted.get(PIE_KERNEL_COMMIT, 0, 0),
        )
        else {
            return Err(det(
                "Metal grouped effect kernels missing from host emission",
            ));
        };
        // The pair is shared across programs — it is named by emitter
        // version, not program — so once compiled it is reused, but its
        // emission is still required: a host that stopped shipping it has
        // drifted from this driver.
        let grouped_pair = match self.grouped_effects.clone() {
            Some(pair) => Ok(pair),
            None => {
                let source = self.splice(grouped_readiness_source)?;
                let readiness = push(
                    &mut jobs,
                    &mut labels,
                    source,
                    grouped_readiness_entry,
                    "Metal M3 grouped readiness compile failed".to_owned(),
                );
                let source = self.splice(grouped_commit_source)?;
                let commit = push(
                    &mut jobs,
                    &mut labels,
                    source,
                    grouped_commit_entry,
                    "Metal M3 grouped commit compile failed".to_owned(),
                );
                Err((readiness, commit))
            }
        };

        // One batch, one archive, keyed by everything that makes the text
        // what it is: device, the program's combined signature, versions.
        let program_identity = cache_identity(
            Backend::Metal,
            context.cache_id(),
            combined_signature(plans),
            versions,
        );
        let compiler =
            Compiler::with_archives(context, self.archives.clone()).map_err(|error| {
                Failure::Retryable {
                    reason: format!("Metal M1 compiler creation failed: {error}"),
                }
            })?;
        let compiled = compiler.compile_sources(
            context,
            &jobs,
            fnv1a64(program_identity.as_bytes()),
            Math::Fast,
        );
        let hit = compiled.archive.is_hit();
        let mut psos: Vec<Pso> = Vec::with_capacity(jobs.len());
        for (slot, result) in compiled.pipelines.into_iter().enumerate() {
            match result {
                Ok(pso) => psos.push(pso),
                Err(error) => {
                    return Err(Failure::Retryable {
                        reason: format!("{}: {error}", labels[slot]),
                    });
                }
            }
        }
        if hit {
            self.stats.persistent_hits += psos.len() as u64;
        } else {
            self.stats.compilations += psos.len() as u64;
        }

        // Assembly. Nothing past this point fails, which is what lets the
        // installs below run unguarded.
        let mut next_ordinal = self.next_ordinal;
        let mut shared: Vec<Rc<StageExecutable>> = Vec::with_capacity(pieces.len());
        let mut staged: Vec<(u64, u64, Rc<StageExecutable>)> = Vec::new();
        for piece in pieces {
            match piece {
                Piece::Cached(stage) => shared.push(stage),
                Piece::Same(piece) => shared.push(Rc::clone(&shared[piece])),
                Piece::Built(building) => {
                    let stage_plan = &plans[building.index];
                    let regions = building
                        .singleton
                        .iter()
                        .enumerate()
                        .map(|(region, &job)| RegionExecutable {
                            operation: metas[building.index][region],
                            pso: psos[job].clone(),
                            ordinal: take(&mut next_ordinal),
                        })
                        .collect();
                    let fused = building.fused.map(|list| {
                        list.into_iter()
                            .map(|(job, region)| FusedExecutable {
                                region: stage_plan.fused[region].clone(),
                                pso: psos[job].clone(),
                                ordinal: take(&mut next_ordinal),
                            })
                            .collect()
                    });
                    let grouped_singleton = building
                        .grouped_singleton
                        .into_iter()
                        .map(|(job, region)| GroupedExecutable {
                            region: stage_plan.singleton[region].clone(),
                            pso: psos[job].clone(),
                            parallel_nucleus: false,
                            parallel_topk: false,
                        })
                        .collect();
                    let grouped = building.grouped.map(|list| {
                        list.into_iter()
                            .map(|(job, region, parallel_nucleus, parallel_topk)| {
                                GroupedExecutable {
                                    region: stage_plan.fused[region].clone(),
                                    pso: psos[job].clone(),
                                    parallel_nucleus,
                                    parallel_topk,
                                }
                            })
                            .collect()
                    });
                    let stage = Rc::new(StageExecutable {
                        regions,
                        fused,
                        grouped,
                        grouped_singleton,
                    });
                    staged.push((building.key, building.identity, Rc::clone(&stage)));
                    shared.push(stage);
                }
            }
        }

        let (grouped_readiness, grouped_commit) = match grouped_pair {
            Ok(pair) => pair,
            Err((readiness, commit)) => {
                let pair = (psos[readiness].clone(), psos[commit].clone());
                self.grouped_effects = Some((pair.0.clone(), pair.1.clone()));
                pair
            }
        };

        let stages = shared
            .into_iter()
            .zip(plans.iter().zip(&package.stages))
            .map(|(executable, (stage_plan, trace))| ProgramStage {
                executable,
                plan: stage_plan.clone(),
                kind: trace.kind,
            })
            .collect();
        let readiness_ordinal = take(&mut next_ordinal);
        let commit_ordinal = take(&mut next_ordinal);

        let program = Rc::new(ProgramExecutable {
            program_hash,
            stages,
            effects,
            readiness: psos[readiness_job].clone(),
            commit: psos[commit_job].clone(),
            grouped_readiness,
            grouped_commit,
            readiness_ordinal,
            commit_ordinal,
            requires_m2_placement,
        });

        for (key, identity, stage) in staged {
            self.stages.stage(key, identity, stage);
        }
        self.stages.commit();
        self.next_ordinal = next_ordinal;
        if self
            .programs
            .insert(program_hash, Rc::clone(&program))
            .is_some()
        {
            self.stats.evictions += 1;
        }
        Ok(program)
    }

    /// Splice the RNG preamble (and any other local include) into an emitted
    /// source.
    ///
    /// The C++ find/replaces the one include it knows about; this resolves
    /// whatever the source names, serving the RNG preamble from the copy read
    /// at creation and anything else from the kernels directory.
    fn splice(&self, source: &str) -> std::result::Result<String, Failure> {
        let root = self.kernels_dir.join("__emitted__.metal");
        shader::splice_with(&root, |path| {
            if path == root {
                Ok(source.to_owned())
            } else if path
                .file_name()
                .is_some_and(|name| name == "ptir_rng.generated.metal")
            {
                Ok(self.rng_preamble.clone())
            } else {
                std::fs::read_to_string(path)
            }
        })
        .map_err(|error| det(format!("Metal M1 emitted source did not splice: {error}")))
    }
}

impl std::fmt::Debug for Runtime {
    /// The cache shape, not the pipelines.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("programs", &self.programs.len())
            .field("stages", &self.stages.len())
            .field("negative", &self.negative.len())
            .field("next_ordinal", &self.next_ordinal)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The library-op ids are mirrored because this crate does not build
    /// against the compiler; the dev-dependency holds them still.
    #[test]
    fn the_library_op_mirror_still_matches_the_compiler() {
        use tensor_compiler::plan::LibraryOp;
        assert_eq!(LIBRARY_NUCLEUS_SAMPLE, LibraryOp::NucleusSample as u8);
        assert_eq!(LIBRARY_TOP_K, LibraryOp::TopK as u8);
    }

    #[test]
    fn the_ordinal_base_clears_the_prefill_namespace() {
        // The C++ derivation: base 3 strides of 4096, plus 1024 rows of one
        // stride each. A 512-row prefill reaches 264192; the base must be
        // past anything the forward DAGs can claim.
        assert_eq!(ORDINAL_BASE, 4_206_592);
        assert!(u64::from(ORDINAL_BASE) > 264_192);
    }
}
