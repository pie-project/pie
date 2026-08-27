//! Guest-program registration — the launch package, purified.
//!
//! **The nouns here are kept, not redesigned** (design §7: "LaunchPackage
//! lineage — owned types adopted directly; keep, purify"). A driver reading a
//! [`LaunchPackage`] gets what it always got: the value table, the channels
//! and ports to allocate, the per-stage op DAGs to launch, and the per-stage
//! plan the emitted kernels were generated from. What changed is the
//! *encoding*.
//!
//! # What "purify" meant, concretely
//!
//! Every field below that used to be a `u8`, an `i8` or a bitmask was a tag in
//! a numbering that PTIR already owns. The tags were re-spelled in
//! `driver-api::local` as `PIE_CHANNEL_DTYPE_*`, `PIE_CHANNEL_HOST_ROLE_*`,
//! `PIE_CHANNEL_EXTERN_*`, `PIE_READINESS_*`, `PIE_VALUE_*`, `PIE_REGION_*`,
//! `PIE_LIBRARY_*`, `PIE_KERNEL_*`, `PIE_STAGE_REQUIRES_*`, `PIE_EXTENT_STATIC`
//! and `PIE_NO_CHANNEL` — 40-odd constants whose values had to agree, by hand,
//! with [`tensor_ir`]'s own. The agreement was not checked anywhere; the one
//! place it visibly failed is instructive:
//!
//! ```text
//! // the SAME concept, twice, in one file (driver/src/program/registry.rs)
//! Direction::of(channel)      // extern_dir: 0 => Import, 1 => Export
//! Direction::from_wire(byte)  // PIE_CHANNEL_EXTERN_IMPORT(1) => Import,
//!                             // PIE_CHANNEL_EXTERN_EXPORT(2) => Export
//! ```
//!
//! Both arms were live. Neither was wrong about its own input. Nothing in the
//! type system said they were about the same axis.
//!
//! So the tags are gone and the types are PTIR's:
//! [`ChanDType`], [`DType`], [`HostRole`], [`ExternDir`], [`Direction`],
//! [`Port`], [`Stage`], [`IntrinsicId`], [`RngKind`]. Where PTIR has no enum
//! because the concept is the driver's — a library op, an emitted kernel's
//! role, a symbolic extent — one is declared here, with the wire tag the
//! constant carried written on the variant so the numbering is preserved
//! exactly.
//!
//! Two sentinels also died. `PIE_NO_CHANNEL = u32::MAX` is `Option<ChannelIndex>`;
//! `PIE_EXTENT_STATIC = 0xff`, the byte that said "this entry of `extents` is
//! not an extent, read `dims` instead", is [`Axis`] — which also folds the two
//! parallel vectors it indexed into one, so the "these are different lengths"
//! refusal that guarded them is now unrepresentable.

use serde::{Deserialize, Serialize};

use tensor_ir::container::{ChanDType, ExternDir, HostRole};
use tensor_ir::op::{ChannelIndex, IntrinsicId};
use tensor_ir::registry::{GeometryClass, Port, Stage};
use tensor_ir::types::{DType, RngKind};
use tensor_ir::validate::Direction;

use crate::channel::ChannelSeed;

/// A registered program's id, minted by the driver.
pub type ProgramId = u64;

/// A bound instance's id, minted by the driver.
pub type InstanceId = u64;

// ──────────────────────────── the trace half ────────────────────────────

/// Where an SSA value in the trace comes from.
///
/// Was `LaunchValue::source: u8` over `PIE_VALUE_CONST` … `PIE_VALUE_OP_RESULT`.
/// The payload each variant needs stays in its own field rather than moving
/// into the variant, because the driver reads the table positionally and
/// folding it would be a redesign rather than a retype.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueSource {
    /// A trace-time constant; the bits are in `literal_bits`.
    Const = 0,
    /// A first-party value intrinsic; which one is in `intrinsic`.
    Intrinsic = 1,
    /// A `chan_take`; the channel is in `channel`.
    ChannelTake = 2,
    /// A `chan_read`; the channel is in `channel`.
    ChannelRead = 3,
    /// An op's result.
    #[default]
    OpResult = 4,
}

/// One row of the SSA value table, in the global numbering the stage bodies
/// use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchValue {
    /// The value's global id: stage `i`'s local `v` is `Σ|stage_types[..i]| + v`.
    pub id: u32,
    /// Where it comes from.
    pub source: ValueSource,
    /// Its element type.
    pub dtype: DType,
    /// Which intrinsic, when [`ValueSource::Intrinsic`]. `None` otherwise —
    /// which is the reason this is an `Option` and not a bare `IntrinsicId`:
    /// `IntrinsicId::Logits` is wire id **0**, so the old `intrinsic: u8` field
    /// read "logits" for every value that had no intrinsic at all, and every
    /// site that touched it had to test `source` first to find out whether the
    /// field meant anything.
    pub intrinsic: Option<IntrinsicId>,
    /// Which channel, when [`ValueSource::ChannelTake`] or
    /// [`ValueSource::ChannelRead`].
    pub channel: ChannelIndex,
    /// The literal's raw bits, when [`ValueSource::Const`].
    pub literal_bits: u32,
    /// Its shape, as dims.
    pub shape: Vec<u32>,
}

impl Default for LaunchValue {
    fn default() -> LaunchValue {
        LaunchValue {
            id: 0,
            source: ValueSource::OpResult,
            dtype: DType::F32,
            intrinsic: None,
            channel: 0,
            literal_bits: 0,
            shape: Vec::new(),
        }
    }
}

/// One op of a stage body, or of a stage plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOp {
    /// The PTIR wire tag — `tensor_ir::op::tags`, the one place an op tag is
    /// spelled as a number. Was `code: u16`; a tag is a `u8` and always was.
    pub tag: u8,
    /// How many SSA ids this op defines.
    pub result_count: u16,
    /// Its first result's id.
    pub result_id: u32,
    /// Which intrinsic, for `intrinsic_val`. See [`LaunchValue::intrinsic`] on
    /// why it is an `Option`.
    pub intrinsic: Option<IntrinsicId>,
    /// A `const`'s literal dtype.
    pub lit_dtype: DType,
    /// The result's element type.
    pub dtype: DType,
    /// A `pivot_threshold`'s predicate tag — `tensor_ir::wire::predicate_tags`.
    /// Stays a raw tag because PTIR's own `Predicate` carries a value id in
    /// every variant and this field is only the discriminant of it.
    pub pred_tag: u8,
    /// A `pivot_threshold`'s predicate payload — a value id for every tag, so
    /// it is remapped like an operand rather than left as an immediate.
    pub pred_payload: u32,
    /// An `rng`'s distribution.
    pub rng_kind: RngKind,
    /// A `const`'s raw bits, read per `lit_dtype`.
    pub lit_bits: u32,
    /// The channel this op touches, when it touches one. Was
    /// `channel: u32` with `PIE_NO_CHANNEL = u32::MAX` meaning none.
    pub channel: Option<ChannelIndex>,
    /// Index into the package's name table, for `kernel_call` / `sink_call`.
    pub name_index: u32,
    /// First trace-known immediate.
    pub imm: u32,
    /// Second immediate.
    pub imm2: u32,
    /// Third immediate.
    pub imm3: u32,
    /// Value-id operands, as the container encodes them.
    pub args: Vec<u32>,
    /// The result's shape, as dims.
    pub shape: Vec<u32>,
}

impl Default for LaunchOp {
    fn default() -> LaunchOp {
        LaunchOp {
            tag: 0,
            result_count: 0,
            result_id: 0,
            intrinsic: None,
            lit_dtype: DType::F32,
            dtype: DType::F32,
            pred_tag: 0,
            pred_payload: 0,
            rng_kind: RngKind::Uniform,
            lit_bits: 0,
            channel: None,
            name_index: 0,
            imm: 0,
            imm2: 0,
            imm3: 0,
            args: Vec::new(),
            shape: Vec::new(),
        }
    }
}

/// One channel the driver allocates and binds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    /// Its index in declaration order.
    pub id: ChannelIndex,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// The cell's element type. [`ChanDType::Act`] is the late-bound
    /// activation type; the driver allocates from
    /// [`ChanDType::program_dtype`].
    pub dtype: ChanDType,
    /// Whether the channel arrives seeded. Was bit 0 of `flags: u8`.
    pub seeded: bool,
    /// Which end the host holds, if any. Was bits 1 and 2 of `flags: u8` —
    /// `HOST_VISIBLE` plus `HOST_READER`, two bits encoding the three states
    /// this enum has, with the fourth bit pattern (reader-but-not-visible)
    /// meaning nothing and reachable anyway.
    pub host_role: HostRole,
    /// Whether this channel crosses to another instance, and which way. Was
    /// `extern_dir: i8` with `-1` for none — and, separately,
    /// `PIE_CHANNEL_EXTERN_NONE/IMPORT/EXPORT` as `0/1/2` for the same axis.
    pub extern_dir: Option<ExternDir>,
    /// Which bit the channel's first in-pass op needs. `None` is a channel no
    /// stage touches — was `PIE_READINESS_UNTOUCHED`.
    pub readiness: Option<Direction>,
    /// The cell's shape, as dims.
    pub shape: Vec<u32>,
    /// The extern binding's name, when `extern_dir` is `Some`.
    pub extern_name: Vec<u8>,
}

impl Default for LaunchChannel {
    fn default() -> LaunchChannel {
        LaunchChannel {
            id: 0,
            capacity: 0,
            dtype: ChanDType::Concrete(DType::F32),
            seeded: false,
            host_role: HostRole::None,
            extern_dir: None,
            readiness: None,
            shape: Vec::new(),
            extern_name: Vec::new(),
        }
    }
}

/// One descriptor-port binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPort {
    /// Which port — [`tensor_ir::registry::Port`], the registry that owns
    /// them (decision 19). Was `port: u8`.
    pub port: Port,
    /// Whether the binding is a trace-time constant rather than a channel.
    pub is_const: bool,
    /// The constant's element type, when `is_const`.
    pub const_dtype: DType,
    /// The channel bound, when not `is_const`.
    pub channel: ChannelIndex,
    /// The constant's shape, when `is_const`.
    pub const_shape: Vec<u32>,
    /// The constant's payload, when `is_const`.
    pub const_data: Vec<u8>,
}

impl Default for LaunchPort {
    fn default() -> LaunchPort {
        LaunchPort {
            port: Port::EmbedTokens,
            is_const: false,
            const_dtype: DType::F32,
            channel: 0,
            const_shape: Vec::new(),
            const_data: Vec::new(),
        }
    }
}

/// One `chan_put`: a value committed to a channel at pass end.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    /// The channel written.
    pub channel: ChannelIndex,
    /// The value written.
    pub value: u32,
}

/// One stage body: the ops it launches and the channel effects it commits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    /// Which attachment stage this is. Was `kind: u8`.
    pub stage: Stage,
    /// The ops, in program order.
    pub ops: Vec<LaunchOp>,
    /// The puts, committed at pass end.
    pub puts: Vec<LaunchPut>,
    /// The channels this stage takes from.
    pub takes: Vec<ChannelIndex>,
    /// The channels this stage reads.
    pub reads: Vec<ChannelIndex>,
}

impl Default for LaunchStage {
    fn default() -> LaunchStage {
        LaunchStage {
            stage: Stage::Prologue,
            ops: Vec::new(),
            puts: Vec::new(),
            takes: Vec::new(),
            reads: Vec::new(),
        }
    }
}

// ──────────────────────────── the plan half ─────────────────────────────

/// A library region's operation — a kernel the backend supplies rather than
/// emits.
///
/// Was `LaunchRegion::library: u8` over `PIE_LIBRARY_*`. The discriminants are
/// written out because they are a wire numbering a driver reads, and a
/// discriminant is a wire numbering only by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LibraryOp {
    /// Nucleus sampling.
    NucleusSample = 0,
    /// Top-k selection.
    TopK = 1,
    /// A sort.
    Sort = 2,
    /// A prefix scan.
    Scan = 3,
    /// A matmul.
    Matmul = 4,
    /// A second-party kernel, named through the package's name table.
    SecondParty = 5,
}

/// What a region is.
///
/// Was the `(kind: u8, library: u8)` pair, where `library` meant nothing
/// unless `kind == PIE_REGION_LIBRARY` and was `0` — a perfectly valid
/// `NucleusSample` — the rest of the time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    /// Source the backend emits and compiles.
    #[default]
    Generated,
    /// A library call.
    Library(LibraryOp),
}

/// One region of a compiled stage: a contiguous run of ops the backend
/// launches together.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    /// Emitted, or a library call.
    pub kind: RegionKind,
    /// The backend's schedule tag for the region.
    pub schedule: u8,
    /// The ops it covers, by index into the plan's op list.
    pub nodes: Vec<u32>,
    /// The values it reads.
    pub inputs: Vec<u32>,
    /// The values it defines.
    pub outputs: Vec<u32>,
    /// The puts it commits.
    pub sinks: Vec<LaunchPut>,
}

/// Which runtime quantity a symbolic axis resolves against.
///
/// Was the low byte space of `LaunchPlanValue::extents`, with
/// `PIE_EXTENT_STATIC = 0xff` carved out of it to mean "not one of these".
/// The tags are unchanged; the sentinel became [`Axis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExtentRole {
    /// The request's readable KV extent.
    KvLen = 0,
    /// How many pages its page list holds.
    PageCount = 1,
    /// How many rows the value has.
    RowCount = 2,
    /// How many tokens the fire carries.
    TokenCount = 3,
    /// How many rows the epilogue reads out.
    SampledRows = 4,
    /// The query length.
    QueryLen = 5,
    /// The key length.
    KeyLen = 6,
}

impl ExtentRole {
    /// Every role, in wire-tag order.
    pub const ALL: [ExtentRole; 7] = [
        ExtentRole::KvLen,
        ExtentRole::PageCount,
        ExtentRole::RowCount,
        ExtentRole::TokenCount,
        ExtentRole::SampledRows,
        ExtentRole::QueryLen,
        ExtentRole::KeyLen,
    ];

    /// The role this wire tag names, or `None`.
    #[must_use]
    pub fn from_wire(tag: u8) -> Option<ExtentRole> {
        ExtentRole::ALL.get(usize::from(tag)).copied()
    }
}

/// One axis of a plan value's type.
///
/// This is the fold that removed a refusal. It was two parallel vectors —
/// `extents: Vec<u8>` and `dims: Vec<u32>` — where entry `i` of one was read
/// only if entry `i` of the other held the sentinel, and the first thing every
/// reader did was check that they were the same length
/// (`Unresolvable::Mismatch`). One vector of this cannot be mismatched with
/// itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    /// A trace-known extent.
    Static(u32),
    /// An extent resolved per fire.
    Symbolic(ExtentRole),
}

/// One value's type in a stage plan: an element type and a list of axes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    /// The element type.
    pub dtype: DType,
    /// The axes, outermost first.
    pub axes: Vec<Axis>,
}

impl Default for LaunchPlanValue {
    fn default() -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: DType::F32,
            axes: Vec::new(),
        }
    }
}

/// One channel-binding rule of a grouped launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    /// The value bound.
    pub value: u32,
    /// Its local slot in the group.
    pub local: u32,
}

/// What a stage's grouped launch depends on.
///
/// Was `flags: u32` over eight `PIE_STAGE_*` bits. Eight named booleans, which
/// is what eight bits with eight names are; a reader asks for the one it is
/// about instead of remembering which shift it lives at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageNeeds {
    /// Reads the projected query intrinsic.
    pub query: bool,
    /// Reads the layer-index intrinsic.
    pub layer: bool,
    /// Reads the attention-score intrinsic.
    pub attn_score: bool,
    /// Calls a second-party kernel.
    pub kernel_call: bool,
    /// Reads the attention page mask.
    pub page_mask: bool,
    /// Reads the MTP draft rows.
    pub mtp_rows: bool,
    /// Reads a LoRA sink.
    pub lora: bool,
    /// The grouped launch path can cover this stage at all.
    pub grouped_valid: bool,
}

/// One compiled stage, as the emitted kernels were generated from it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStagePlan {
    /// The stage's structural signature — what makes two stages the same
    /// shape.
    pub signature_hash: u64,
    /// The stage's graph-cache identity: signature plus everything a compiled
    /// artifact depends on.
    pub identity: u64,
    /// What its grouped launch depends on.
    pub needs: StageNeeds,
    /// How many MTP draft rows it reads.
    pub mtp_rows: u32,
    /// Its normalized ops, in stage-local numbering.
    pub ops: Vec<LaunchOp>,
    /// For each normalized op, which source ops it came from.
    pub source_ops: Vec<Vec<u32>>,
    /// Its value table's types.
    pub value_types: Vec<LaunchPlanValue>,
    /// Which channel each bound value binds through.
    pub channel_bindings: Vec<u32>,
    /// The names its `kernel_call`s and `sink_call`s index.
    pub names: Vec<String>,
    /// Its singleton-partition regions.
    pub singleton: Vec<LaunchRegion>,
    /// Its fused-partition regions.
    pub fused: Vec<LaunchRegion>,
    /// Which runtime extents it depends on. Was `Vec<u8>` in the same tag
    /// space [`ExtentRole`] now names.
    pub used_extents: Vec<ExtentRole>,
    /// Its grouped channel rules.
    pub channel_rules: Vec<LaunchChannelRule>,
    /// Why the grouped plan could not be derived, if it could not. Empty means
    /// it could.
    pub error: String,
}

/// A driver-side fast path a region admits: an argmax the backend can answer
/// without running the region's generated source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectArgmax {
    /// Which op, by index into the plan's ops.
    pub node: u32,
    /// Which value it reduces.
    pub source_value: u32,
    /// Which intrinsic that value is.
    pub intrinsic: IntrinsicId,
    /// Whether the path is legal only for a single-row fire. Was
    /// `requires_single_row: u8`.
    pub requires_single_row: bool,
}

/// What the backend's analysis found about one region.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionAnalysis {
    /// Which stage.
    pub stage_index: u32,
    /// Which region within it.
    pub region_index: u32,
    /// The region may call a second-party kernel. Was bit 0 of `flags: u32`.
    pub second_party_supported: bool,
    /// The region's generated source is valid. Was bit 1.
    pub generated_valid: bool,
    /// The direct-argmax fast paths it admits.
    pub direct_argmax: Vec<DirectArgmax>,
    /// The ops the analysis found nothing to do for.
    pub skipped: Vec<u32>,
}

/// What an emitted kernel is for.
///
/// Was `EmittedKernel::kind: u32` over `PIE_KERNEL_*`. Discriminants written
/// out for the same reason [`LibraryOp`]'s are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum KernelKind {
    /// One region, launched alone.
    #[default]
    Singleton = 0,
    /// A fused run of regions.
    Fused = 1,
    /// The grouped launch covering a whole stage.
    Grouped = 2,
    /// The readiness control kernel.
    Readiness = 3,
    /// The commit control kernel.
    Commit = 4,
}

/// One kernel the compiler emitted for this program.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmittedKernel {
    /// What it is for.
    pub kind: KernelKind,
    /// Which stage.
    pub stage_index: u32,
    /// Which region within it (unused for the whole-stage kinds).
    pub region_index: u32,
    /// Its entry-point symbol.
    pub entry_name: String,
    /// Its source text.
    pub source: String,
    /// Why it could not be emitted, if it could not. Empty means it was.
    pub error: String,
}

/// A whole program, in the shape a driver executes it.
///
/// Deliberately not PTIR: a driver reading this never sees a container, a wire
/// format, or an identity to re-check — the compiler already decided all of
/// that.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPackage {
    /// The SSA value table, in global numbering.
    pub values: Vec<LaunchValue>,
    /// The channels to allocate.
    pub channels: Vec<LaunchChannel>,
    /// The descriptor ports to bind.
    pub ports: Vec<LaunchPort>,
    /// The name table `kernel_call` / `sink_call` / extern names index.
    pub names: Vec<String>,
    /// The stage bodies.
    pub stages: Vec<LaunchStage>,
    /// The compiled stage plans, parallel to `stages`.
    pub plans: Vec<LaunchStagePlan>,
}

/// Everything a program registration states.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    /// The program's identity — FNV-1a over its canonical container bytes.
    pub program_hash: u64,
    /// The kernels the compiler emitted.
    pub emitted_kernels: Vec<EmittedKernel>,
    /// Which emitter version produced them. A driver's compiled-artifact cache
    /// keys on it.
    pub emitter_version: u32,
    /// What the backend's region analysis found.
    pub region_analysis: Vec<RegionAnalysis>,
    /// The package to execute.
    pub launch: LaunchPackage,
    /// The canonical container bytes, for a driver that runs the reference
    /// interpreter beside the device and diffs the two.
    pub reference_ptir: Vec<u8>,
}

/// What a bound instance's symbolic value shapes resolve against.
///
/// **A GUESS ZERO-FILLS SILENTLY, SO IT IS STATED** (Build log 15). A stage
/// plan's value types are written in [`Axis::Symbolic`] over the seven
/// [`ExtentRole`]s; a driver carves each stage's fire-path buffers at BIND
/// time, from these numbers, and a buffer carved for one row when the fire
/// hands it four leaves three rows of zeroes that no launch faults on. So the
/// caller states them, and the one that matters at a model fire's boundary is
/// [`BindExtents::sampled_rows`] — how many readout rows the epilogue reads,
/// which is the fire's [`Readout`](crate::fire::Readout) and nothing the
/// driver can infer from the package.
///
/// [`BindExtents::default`] is every extent ONE, which is what a program that
/// resolves entirely from static dims reads (it never reads these at all) and
/// what a `Readout::Last` lane hands an epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindExtents {
    /// The request's readable KV extent.
    pub kv_len: u32,
    /// How many pages its page list holds.
    pub page_count: u32,
    /// How many rows the value has.
    pub row_count: u32,
    /// How many tokens the fire carries.
    pub token_count: u32,
    /// How many rows the epilogue reads out.
    pub sampled_rows: u32,
    /// The query length.
    pub query_len: u32,
    /// The key length.
    pub key_len: u32,
}

impl Default for BindExtents {
    fn default() -> BindExtents {
        BindExtents {
            kv_len: 1,
            page_count: 1,
            row_count: 1,
            token_count: 1,
            sampled_rows: 1,
            query_len: 1,
            key_len: 1,
        }
    }
}

impl BindExtents {
    /// What `role` resolves to.
    #[must_use]
    pub const fn get(&self, role: ExtentRole) -> u32 {
        match role {
            ExtentRole::KvLen => self.kv_len,
            ExtentRole::PageCount => self.page_count,
            ExtentRole::RowCount => self.row_count,
            ExtentRole::TokenCount => self.token_count,
            ExtentRole::SampledRows => self.sampled_rows,
            ExtentRole::QueryLen => self.query_len,
            ExtentRole::KeyLen => self.key_len,
        }
    }
}

/// Everything an instance binding states.
///
/// Was `InstanceBindingPlan`, which carried three more fields —
/// `driver_id: usize`, `pacing_wait_id: u64` and `requested_instance_id` — that
/// were the engine's bookkeeping travelling through the driver so it could
/// come back unchanged. The driver mints the id; the engine keeps its own
/// tables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceBinding {
    /// Which program to instantiate.
    pub program: ProgramId,
    /// The channels this instance binds, in the package's declaration order.
    pub channels: Vec<crate::channel::ChannelId>,
    /// The values its seeded channels start holding.
    pub seeds: Vec<ChannelSeed>,
    /// How much of the fire geometry this instance's descriptor resolves on
    /// the device.
    pub geometry: GeometryClass,
    /// What this instance's symbolic value shapes resolve against. See
    /// [`BindExtents`].
    pub extents: BindExtents,
}

impl Default for InstanceBinding {
    /// A binding of nothing, at every extent one — the shape a program with
    /// no symbolic axis is bound in. Written out rather than derived because
    /// a derived [`BindExtents`] would be every extent ZERO, and a zero
    /// extent carves a zero-row buffer that no launch faults on.
    fn default() -> InstanceBinding {
        InstanceBinding {
            program: 0,
            channels: Vec::new(),
            seeds: Vec::new(),
            geometry: GeometryClass::default(),
            extents: BindExtents::default(),
        }
    }
}

/// A bound instance, as the driver answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundInstance {
    /// The instance's id.
    pub id: InstanceId,
    /// The program it instantiates.
    pub program: ProgramId,
    /// The geometry class it was bound in — the driver's acknowledgement,
    /// which a caller compares against what it asked for.
    pub geometry: GeometryClass,
}
