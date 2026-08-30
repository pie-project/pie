//! The launch package — a program in the shape an engine executes it.
//!
//! This is the whole of what crosses the host→engine boundary for a program,
//! and it deliberately is not ETA. An engine reading it never sees a container,
//! a sidecar, a wire format, or an identity to re-check: the compiler already
//! decided all of that. What it gets is the value table, the channels and
//! ports to allocate and bind, the per-stage op DAGs to launch, and the
//! per-stage plan the emitted kernels were generated from.
//!
//! Two things an engine could plausibly re-derive are folded in here instead,
//! because they are decisions about the program rather than about the machine
//! — and an engine that re-derives them is a second implementation that has to
//! agree with this one forever:
//!
//! * each stage's **graph-cache identity** ([`crate::plan::stage_identity`]), and
//! * the **grouped static plan** — which runtime extents the stage depends on,
//!   which values bind through a channel, which intrinsics it reads, and
//!   whether the grouped launch path can cover it at all.
//!
//! The op projection is [`crate::codegen::op_view::OpView`], the same one the emitters
//! read, so the kernel and the description of the kernel cannot drift.
//!
//! # Why the declarations are here and not in the contract
//!
//! **The nouns here are kept, not redesigned** (design §7: "LaunchPackage
//! lineage — owned types adopted directly; keep, purify"). They were declared
//! in `engine`, the runtime↔engine contract, for as long as the contract
//! was the only crate both sides could name — which meant the compiler, the
//! thing that PRODUCES a launch package, depended on the contract to describe
//! its own output, and every type the compiler already had a word for was
//! spelled twice with a `match` in between. The producer owns its output type
//! now. What that deleted, by name:
//!
//! * `LibraryOp` and `RegionKind`, which [`crate::plan`] declares — the
//!   partitioner decides what a region IS, so its answer is the one that
//!   ships;
//! * `KernelKind` and `EmittedKernel`, which [`crate::codegen::program`]
//!   declares beside the walk that fills them;
//! * `RegionAnalysis` and `DirectArgmax`, which
//!   [`crate::codegen::cuda::region_analysis`] declares beside the analysis
//!   that answers them.
//!
//! # What "purify" meant, concretely
//!
//! Every field below that used to be a `u8`, an `i8` or a bitmask was a tag in
//! a numbering that ETA already owns. The tags were re-spelled in
//! `driver-api::local` as `PIE_CHANNEL_DTYPE_*`, `PIE_CHANNEL_HOST_ROLE_*`,
//! `PIE_CHANNEL_EXTERN_*`, `PIE_READINESS_*`, `PIE_VALUE_*`, `PIE_REGION_*`,
//! `PIE_LIBRARY_*`, `PIE_KERNEL_*`, `PIE_STAGE_REQUIRES_*`, `PIE_EXTENT_STATIC`
//! and `PIE_NO_CHANNEL` — 40-odd constants whose values had to agree, by hand,
//! with [`eta_ir`]'s own. The agreement was not checked anywhere; the one
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
//! So the tags are gone and the types are ETA's:
//! [`ChanDType`], [`Dtype`], [`HostRole`], [`ExternDir`], [`Direction`],
//! [`Port`], [`Stage`], [`IntrinsicId`], [`RngKind`]. Where ETA has no enum
//! because the concept is the toolchain's — a library op, an emitted kernel's
//! role, a symbolic extent — one is declared here or in the module that
//! decides it, with the wire tag the constant carried written on the variant
//! so the numbering is preserved exactly.
//!
//! Two sentinels also died. `PIE_NO_CHANNEL = u32::MAX` is `Option<ChannelIndex>`;
//! `PIE_EXTENT_STATIC = 0xff`, the byte that said "this entry of `extents` is
//! not an extent, read `dims` instead", is [`Dimension`] — which also folds the two
//! parallel vectors it indexed into one, so the "these are different lengths"
//! refusal that guarded them is now unrepresentable.

use serde::{Deserialize, Serialize};

use eta_ir::container::{ChanDType, ExternDir, HostRole, PortSource};
use eta_ir::op::{ChannelIndex, IntrinsicId, intrinsic_tags, tags};
use eta_ir::registry::{Port, Stage};
use eta_ir::types::{Dtype, RngKind, ValueType, from_wire};
use eta_ir::validate::{BoundTrace, Direction};

use crate::plan::{
    CompiledStage, Dimension, NodeIndex, Region, RegionKind, RegionPartition, SymbolicExtent,
    SymbolicType, stage_identity,
};

use crate::codegen::fault::FaultClass;
use crate::codegen::op_view::OpView;

// ──────────────────────────── the trace half ────────────────────────────

/// Where an SSA value in the trace comes from.
///
/// **Not [`eta_ir::op::ValueSource`]**, which shares neither a variant nor a
/// question with it. That one classifies an op's result by what DECIDES it —
/// `Device`, `Channel`, `Operands` — so a partial evaluator can ask whether
/// the host could have computed it. This one names the trace construct that
/// DEFINED the value, which is a wire numbering an engine reads positionally.
/// The two were `ValueSource` in two crates that never met; they meet here, so
/// this one is spelled for the question it answers.
///
/// Was `LaunchValue::source: u8` over `PIE_VALUE_CONST` … `PIE_VALUE_OP_RESULT`.
/// The payload each variant needs stays in its own field rather than moving
/// into the variant, because the engine reads the table positionally and
/// folding it would be a redesign rather than a retype.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueOrigin {
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
    pub source: ValueOrigin,
    /// Its element type.
    pub dtype: Dtype,
    /// Which intrinsic, when [`ValueOrigin::Intrinsic`]. `None` otherwise —
    /// which is the reason this is an `Option` and not a bare `IntrinsicId`:
    /// `IntrinsicId::Logits` is wire id **0**, so the old `intrinsic: u8` field
    /// read "logits" for every value that had no intrinsic at all, and every
    /// site that touched it had to test `source` first to find out whether the
    /// field meant anything.
    pub intrinsic: Option<IntrinsicId>,
    /// Which channel, when [`ValueOrigin::ChannelTake`] or
    /// [`ValueOrigin::ChannelRead`].
    pub channel: ChannelIndex,
    /// The literal's raw bits, when [`ValueOrigin::Const`].
    pub literal_bits: u32,
    /// Its shape, as dims.
    pub shape: Vec<u32>,
}

impl Default for LaunchValue {
    fn default() -> LaunchValue {
        LaunchValue {
            id: 0,
            source: ValueOrigin::OpResult,
            dtype: Dtype::F32,
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
    /// The ETA wire tag — `eta_ir::op::tags`, the one place an op tag is
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
    pub lit_dtype: Dtype,
    /// The result's element type.
    pub dtype: Dtype,
    /// A `pivot_threshold`'s predicate tag — `eta_ir::wire::predicate_tags`.
    /// Stays a raw tag because ETA's own `Predicate` carries a value id in
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
            lit_dtype: Dtype::F32,
            dtype: Dtype::F32,
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

/// One channel the engine allocates and binds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    /// Its index in declaration order.
    pub id: ChannelIndex,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// The cell's element type. [`ChanDType::Act`] is the late-bound
    /// activation type; the engine allocates from
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
            dtype: ChanDType::Concrete(Dtype::F32),
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
    /// Which port — [`eta_ir::registry::Port`], the registry that owns
    /// them (decision 19). Was `port: u8`.
    pub port: Port,
    /// Whether the binding is a trace-time constant rather than a channel.
    pub is_const: bool,
    /// The constant's element type, when `is_const`.
    pub const_dtype: Dtype,
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
            const_dtype: Dtype::F32,
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

/// One region of a compiled stage: a contiguous run of ops the backend
/// launches together.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    /// Emitted, or a library call. [`RegionKind`] is the partitioner's own
    /// enum: this field used to be a second declaration of it, filled by a
    /// six-arm `match`.
    pub kind: RegionKind,
    /// The backend's schedule tag for the region — [`crate::plan::ScheduleTemplate`]
    /// as a byte.
    ///
    /// The one tag in this package that is still a number, and the reason is
    /// that nothing on the engine side reads it as anything else: it is
    /// carried through to the device lane table. It is a candidate for the
    /// same treatment [`RegionKind`] just got, not an exception to it.
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

/// One value's type in a stage plan: an element type and a list of axes.
///
/// The axes are [`Dimension`]s and the symbolic ones are [`SymbolicExtent`]s —
/// the planner's own types, shipped as themselves. They were declared a second
/// time here, as `Axis` and `ExtentRole`, with `axis` and `extent_role`
/// mapping one pair onto the other arm for arm; both declarations and both
/// maps are gone. The argument for why the pair existed, and for what its
/// disappearance means, is on [`SymbolicExtent`] and [`Dimension`] — the
/// surviving spellings carry it now, because they are the ones that cross into
/// the artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    /// The element type.
    pub dtype: Dtype,
    /// The axes, outermost first.
    pub axes: Vec<Dimension>,
}

impl Default for LaunchPlanValue {
    fn default() -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: Dtype::F32,
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
    /// space [`SymbolicExtent`] names.
    pub used_extents: Vec<SymbolicExtent>,
    /// Its grouped channel rules.
    pub channel_rules: Vec<LaunchChannelRule>,
    /// Why the grouped plan could not be derived, if it could not. Empty means
    /// it could.
    pub error: String,
}

/// A whole program, in the shape an engine executes it.
///
/// Deliberately not ETA: an engine reading this never sees a container, a wire
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
    /// The fault-code table the emitted kernels were written against.
    ///
    /// **CARRIED, NOT SHARED.** An emitter writes fault codes INTO generated
    /// source; an engine reads one back off the device and has to say what it
    /// means. That is a decision this crate made, not vocabulary the two
    /// hold in common, so it rides with the program instead of being an
    /// import — a code is named by the table of the emitter that compiled the
    /// kernel that raised it, and not by whatever table the reading binary
    /// links. [`crate::codegen::fault`] carries the argument.
    ///
    /// Empty on a package nobody built (`Default`), which reads as "no class
    /// names available" and prints the raw code — the same answer an
    /// unrecognised code already got.
    pub fault_classes: Vec<FaultClass>,
}

/// Build the launch package for a bound trace and its compiled stages.
///
/// `stages` is `crate::plan::compile_bound(bound)`, in container order — the
/// caller already has it, because the emitters need the same value.
pub fn build(bound: &BoundTrace, stages: &[CompiledStage]) -> LaunchPackage {
    LaunchPackage {
        values: lower_values(bound),
        channels: lower_channels(bound),
        ports: lower_ports(bound),
        names: bound.container.names.clone(),
        stages: lower_stages(bound),
        plans: stages.iter().map(lower_plan).collect(),
        fault_classes: crate::codegen::fault::classes(),
    }
}

// ── the trace half: what to allocate, bind, and launch ──

/// The SSA value table, in the same global numbering the stage bodies use:
/// stage `i`'s local value `v` is global `Σ|stage_types[..i]| + v`.
fn lower_values(bound: &BoundTrace) -> Vec<LaunchValue> {
    let mut values = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let mut local = 0u32;
        let mut push = |local: u32, mut value: LaunchValue| {
            let value_type = types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(Dtype::F32));
            value.id = base + local;
            value.dtype = value_type.dtype;
            value.shape = value_type.shape.dims().to_vec();
            values.push(value);
        };
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    push(
                        local,
                        LaunchValue {
                            source: if view.tag == tags::CHAN_TAKE {
                                ValueOrigin::ChannelTake
                            } else {
                                ValueOrigin::ChannelRead
                            },
                            channel: view.chan as u32,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CONST => {
                    push(
                        local,
                        LaunchValue {
                            source: ValueOrigin::Const,
                            literal_bits: view.lit_bits,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::INTRINSIC_VAL => {
                    push(
                        local,
                        LaunchValue {
                            source: ValueOrigin::Intrinsic,
                            intrinsic: IntrinsicId::from_u16(view.intr),
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CHAN_PUT | tags::SINK_CALL => {}
                _ => {
                    for result in 0..view.results {
                        push(
                            local + result,
                            LaunchValue {
                                source: ValueOrigin::OpResult,
                                ..LaunchValue::default()
                            },
                        );
                    }
                    local += view.results;
                }
            }
        }
        base += types.len() as u32;
    }
    values
}

fn lower_channels(bound: &BoundTrace) -> Vec<LaunchChannel> {
    bound
        .container
        .channels
        .iter()
        .enumerate()
        .map(|(index, decl)| {
            let extern_decl = bound
                .container
                .externs
                .iter()
                .find(|entry| entry.chan as usize == index);
            // First-touch, in pass order. The sets a stage ships say *whether*
            // a channel is taken and put; only this says which one comes first,
            // and an `InPlace` channel is both.
            let readiness = bound
                .readiness
                .iter()
                .find(|entry| entry.chan as usize == index)
                .map(|entry| entry.dir);
            LaunchChannel {
                id: index as u32,
                capacity: decl.capacity,
                // The program-side element type, with a late-bound activation
                // dtype already materialized — the engine allocates cells from
                // this and never sees `ACT`.
                dtype: ChanDType::Concrete(bound.channel_types[index].dtype),
                seeded: decl.seeded,
                host_role: decl.host_role,
                extern_dir: extern_decl.map(|entry| entry.dir),
                readiness,
                shape: decl.shape.dims().to_vec(),
                extern_name: extern_decl
                    .and_then(|entry| bound.container.names.get(entry.name as usize))
                    .map(|name| name.as_bytes().to_vec())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn lower_ports(bound: &BoundTrace) -> Vec<LaunchPort> {
    bound
        .container
        .ports
        .iter()
        .map(|binding| match binding.source {
            PortSource::Channel(chan) => LaunchPort {
                port: binding.port,
                is_const: false,
                channel: chan,
                ..LaunchPort::default()
            },
            PortSource::Const {
                dtype,
                ref shape,
                ref data,
            } => LaunchPort {
                port: binding.port,
                is_const: true,
                const_dtype: dtype,
                const_shape: shape.dims().to_vec(),
                const_data: data.clone(),
                ..LaunchPort::default()
            },
        })
        .collect()
}

/// The stage bodies, with every operand remapped from stage-local to the
/// global numbering of [`lower_values`]. Ops that define no value — `chan_put`
/// — become stage effects instead of ops, exactly as the execution model wants
/// them: a put is committed at pass end, not launched.
fn lower_stages(bound: &BoundTrace) -> Vec<LaunchStage> {
    let mut stages = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let global = |local: u32| base + local;
        let result_type = |local: u32| {
            types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(Dtype::F32))
        };

        let mut stage = LaunchStage {
            stage: program.stage,
            ..LaunchStage::default()
        };
        let mut local = 0u32;
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    if view.tag == tags::CHAN_TAKE {
                        stage.takes.push(view.chan as u32);
                    } else {
                        stage.reads.push(view.chan as u32);
                    }
                    local += 1;
                }
                tags::CONST | tags::INTRINSIC_VAL => local += 1,
                tags::CHAN_PUT => stage.puts.push(LaunchPut {
                    channel: view.chan as u32,
                    value: global(view.args[0]),
                }),
                _ => {
                    let value_type = result_type(local);
                    stage.ops.push(LaunchOp {
                        tag: view.tag,
                        result_count: view.results as u16,
                        result_id: global(local),
                        // A stage body carries no `intrinsic_val` op — those
                        // became `LaunchValue`s above — so there is no
                        // intrinsic to name here. The field this replaced said
                        // `0`, which is `IntrinsicId::Logits`.
                        intrinsic: None,
                        lit_dtype: dtype(view.lit_dtype),
                        dtype: value_type.dtype,
                        pred_tag: view.pred_tag,
                        rng_kind: rng(view.kind),
                        lit_bits: view.lit_bits,
                        // Every predicate tag carries a value id, so it is
                        // remapped like any other operand rather than left as
                        // an immediate.
                        pred_payload: global(view.pred_payload),
                        channel: None,
                        name_index: u32::from(view.name_idx),
                        imm: view.imm,
                        imm2: view.imm2,
                        imm3: view.imm3,
                        args: view.args.iter().map(|arg| global(*arg)).collect(),
                        shape: value_type.shape.dims().to_vec(),
                    });
                    local += view.results;
                }
            }
        }
        stages.push(stage);
        base += types.len() as u32;
    }
    stages
}

// ── the plan half: what the emitted kernels were generated from ──

fn lower_plan(stage: &CompiledStage) -> LaunchStagePlan {
    let normalized = &stage.normalized;
    let ops: Vec<OpView> = OpView::of_all(&normalized.ops);
    let grouped = GroupedPlan::derive(
        &ops,
        &normalized.value_types,
        &normalized.names,
        normalized.channel_bindings.len(),
    );

    LaunchStagePlan {
        signature_hash: stage.signature.hash,
        identity: stage_identity(stage),
        needs: grouped.needs,
        mtp_rows: grouped.mtp_rows,
        ops: ops.iter().map(lower_plan_op).collect(),
        source_ops: normalized.source_ops.clone(),
        value_types: normalized
            .value_types
            .iter()
            .map(lower_plan_value)
            .collect(),
        channel_bindings: normalized.channel_bindings.clone(),
        names: normalized.names.clone(),
        singleton: lower_partition(&stage.singleton),
        fused: lower_partition(&stage.fused),
        used_extents: grouped.used_extents,
        channel_rules: grouped.channel_rules,
        error: grouped.error,
    }
}

/// A normalized op keeps its stage-local numbering: the plan's ops index the
/// plan's own value table, which is what the emitted kernels bind against.
fn lower_plan_op(view: &OpView) -> LaunchOp {
    LaunchOp {
        tag: view.tag,
        result_count: view.results as u16,
        result_id: 0,
        // `intr` is set by the `intrinsic_val` arm of the wire projection and
        // by nothing else, so asking the tag is what makes `None` mean "no
        // intrinsic" rather than "intrinsic zero".
        intrinsic: (view.tag == tags::INTRINSIC_VAL)
            .then(|| IntrinsicId::from_u16(view.intr))
            .flatten(),
        lit_dtype: dtype(view.lit_dtype),
        dtype: dtype(view.dtype),
        pred_tag: view.pred_tag,
        rng_kind: rng(view.kind),
        lit_bits: view.lit_bits,
        pred_payload: view.pred_payload,
        channel: u32::try_from(view.chan).ok(),
        name_index: u32::from(view.name_idx),
        imm: view.imm,
        imm2: view.imm2,
        imm3: view.imm3,
        args: view.args.clone(),
        shape: view.shape.clone(),
    }
}

/// A planned value's type, as the launch package spells it — which is now the
/// same spelling.
///
/// `.map(axis)` stood on this line, and `axis` called `extent_role`: two
/// functions, nine arms, whose whole job was to cross a crate line that no
/// longer exists. [`crate::plan::Dimension`] IS the package's axis type, so
/// the dims are cloned and nothing translates them.
fn lower_plan_value(value_type: &SymbolicType) -> LaunchPlanValue {
    LaunchPlanValue {
        dtype: value_type.dtype,
        axes: value_type.dims.clone(),
    }
}

/// The element type a wire byte names. `F32` for a byte no dtype claims — the
/// same fallback the `as u8` round trip this replaced had, made explicit.
fn dtype(byte: u8) -> Dtype {
    from_wire(byte).unwrap_or(Dtype::F32)
}

/// The distribution a wire byte names. Uniform is tag 0, which is what every
/// non-`rng` op carries.
fn rng(byte: u8) -> RngKind {
    if byte == RngKind::Gumbel as u8 {
        RngKind::Gumbel
    } else {
        RngKind::Uniform
    }
}

fn lower_partition(partition: &RegionPartition) -> Vec<LaunchRegion> {
    partition.regions.iter().map(lower_region).collect()
}

fn lower_region(region: &Region) -> LaunchRegion {
    LaunchRegion {
        // ONE ENUM NOW. This was a `match` onto a second `RegionKind`
        // declared in the contract crate, which called a second `match`
        // (`library_tag`) onto a second `LibraryOp`. Both second declarations
        // are gone, so the partitioner's answer travels to the engine as
        // itself.
        kind: region.kind,
        schedule: region.schedule as u8,
        // `LaunchRegion` is the engine ABI, which has one integer space;
        // the node tags stop here.
        nodes: region.nodes.iter().copied().map(NodeIndex::get).collect(),
        inputs: region.inputs.clone(),
        outputs: region.outputs.clone(),
        sinks: region
            .sinks
            .iter()
            .map(|sink| LaunchPut {
                channel: sink.channel_slot.get(),
                value: sink.value,
            })
            .collect(),
    }
}

/// The grouped launch path's view of a stage: which runtime extents it
/// depends on, which values bind through a channel, which intrinsics it reads,
/// and whether the path can cover it at all.
///
/// A decision about the program rather than about the device, so it is made
/// once here rather than in each engine's launch path.
#[derive(Default)]
struct GroupedPlan {
    needs: StageNeeds,
    mtp_rows: u32,
    used_extents: Vec<SymbolicExtent>,
    channel_rules: Vec<LaunchChannelRule>,
    error: String,
}

impl GroupedPlan {
    fn derive(
        ops: &[OpView],
        value_types: &[SymbolicType],
        names: &[String],
        channel_count: usize,
    ) -> Self {
        let mut plan = GroupedPlan {
            needs: StageNeeds {
                grouped_valid: true,
                ..StageNeeds::default()
            },
            ..GroupedPlan::default()
        };
        // No bounds check on the tag any more: `SymbolicExtent` is an enum, so
        // "unsupported runtime extent" is not a state a `Dimension` can be in.
        let mut seen = [false; SymbolicExtent::ALL.len()];
        for value_type in value_types {
            for dimension in &value_type.dims {
                let Dimension::Symbolic(extent) = *dimension else {
                    continue;
                };
                if !seen[extent as usize] {
                    seen[extent as usize] = true;
                    plan.used_extents.push(extent);
                }
            }
        }

        let mut value_bases = Vec::with_capacity(ops.len());
        let mut next_value = 0u32;
        for op in ops {
            value_bases.push(next_value);
            next_value += op.results;
        }

        for (node, op) in ops.iter().enumerate() {
            // A second-party kernel is not a grouped-coverable tag — only the
            // fused path can launch one — but this plan is the shared
            // lane-binding metadata the fused path resolves through, so it
            // describes the op rather than rejecting it. `requires_query` is
            // set because the kernel consumes the lane's post-rope query row.
            if op.tag == tags::KERNEL_CALL {
                plan.needs.query = true;
                plan.needs.kernel_call = true;
                continue;
            }
            // Sinks are grouped-walkable but only fused-executable.
            // Described here for the same reason. Which flag depends on which
            // sink: `lora` configures the whole forward, everything else is
            // the page mask.
            if op.tag == tags::SINK_CALL {
                if names.get(op.name_idx as usize).map(String::as_str) == Some("lora") {
                    plan.needs.lora = true;
                } else {
                    plan.needs.page_mask = true;
                }
            }
            if !grouped_supported_tag(op.tag) {
                return plan.invalid("stage contains an unsupported grouped op");
            }
            if op.tag == tags::INTRINSIC_VAL {
                match op.intr {
                    intrinsic_tags::QUERY => plan.needs.query = true,
                    intrinsic_tags::LAYER => plan.needs.layer = true,
                    intrinsic_tags::ATTN_SCORE => plan.needs.attn_score = true,
                    intrinsic_tags::MTP_LOGITS => {
                        let value = value_bases[node] as usize;
                        let rows = match value_types.get(value).map(|ty| ty.dims.as_slice()) {
                            Some([Dimension::Static(rows), _]) => *rows,
                            _ => return plan.invalid("MtpLogits has no static draft-row layout"),
                        };
                        if plan.needs.mtp_rows && plan.mtp_rows != rows {
                            return plan.invalid(
                                "MtpLogits stages declare incompatible draft-row layouts",
                            );
                        }
                        plan.needs.mtp_rows = true;
                        plan.mtp_rows = rows;
                    }
                    intrinsic_tags::LOGITS => {}
                    _ => return plan.invalid("stage uses an unsupported intrinsic"),
                }
            }

            let value = match op.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => value_bases[node],
                tags::CHAN_PUT if !op.args.is_empty() => op.args[0],
                _ => continue,
            };
            if value as usize >= value_types.len()
                || op.chan < 0
                || op.chan as usize >= channel_count
            {
                return plan.invalid("channel value is outside the grouped plan");
            }
            plan.channel_rules.push(LaunchChannelRule {
                value,
                local: op.chan as u32,
            });
        }
        plan
    }

    /// The stage cannot take the grouped path. What was derived so far still
    /// describes it — the fused path reads it — so only the validity flag and
    /// the reason change.
    fn invalid(mut self, reason: &str) -> Self {
        self.needs.grouped_valid = false;
        self.error = reason.to_string();
        self
    }
}

/// Whether the grouped runtime has a body for this tag.
///
/// Answered from the runtime source, and neither of the two shapes this
/// invites.
///
/// *Raw tag ranges* (`0x01..=0x07 | 0x10..=0x20 | ...`) classify by numeric
/// neighbourhood. The gaps between such ranges are exactly where new op tags
/// get allocated, so a new op is silently classified unsupported.
///
/// *An exception list* closes the gaps and creates a worse problem: if support
/// is **defined** as "not in the list", then a test walking the list to check
/// it asserts `!contains(x)` for `x in list` — a tautology that passes whether
/// the list is empty or holds `add`. A predicate must not be tested against
/// the thing that defines it.
///
/// So the authority is the runtime source itself:
/// `grouped_support_is_what_the_runtime_can_execute` holds this function to it.
fn grouped_supported_tag(tag: u8) -> bool {
    eta_ir::op::spec(tag).is_some()
}

#[cfg(test)]
mod grouped_coverage {
    use super::*;

    /// Grouped support is whatever `ptir_m1_execute` can dispatch on, because
    /// `emit_grouped_fused_region` pastes that runtime into every kernel it
    /// emits. Reading the arms out of the source makes this a claim about the
    /// emitted program rather than a restatement of a constant.
    #[test]
    fn grouped_support_is_what_the_runtime_can_execute() {
        let handled = crate::codegen::runtime_scan::tags_compared_in(
            crate::codegen::runtime_scan::function_body(
                crate::codegen::metal::RUNTIME_TEMPLATE,
                "void ptir_m1_execute",
            ),
        );
        assert!(
            handled.len() > 40,
            "only {} tag arms parsed out of ptir_m1_execute; the scan broke and \
             every comparison below would be vacuous",
            handled.len()
        );
        let mut checked = 0usize;
        for spec in eta_ir::op::OP_TABLE {
            assert_eq!(
                grouped_supported_tag(spec.tag),
                handled.contains(&spec.tag),
                "{} ({:#04x}): the classifier and ptir_m1_execute disagree",
                spec.name,
                spec.tag
            );
            checked += 1;
        }
        assert_eq!(checked, eta_ir::op::OP_TABLE.len());
        assert!(
            !grouped_supported_tag(0xFF),
            "a non-op tag is not supported"
        );
    }
}
