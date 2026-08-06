//! The arena that keeps a [`PieForwardPlan`]'s slices alive.
//!
//! Every pointer the driver reads points into buffers owned here, on the
//! model of `loader/src/ffi/arena.rs`: build once, freeze, leak into the
//! plan's `owner`, reclaim in `pie_forward_release`. Between those two
//! points nothing mutates, so handing the plan to another thread is sound.
//!
//! One structural difference from the loader: the plan *header* is not boxed
//! here. `pie_forward_trace_llama_like` writes it into caller-owned storage
//! (`out_plan: *mut PieForwardPlan`), so the only Rust-owned allocation is
//! the arena itself, and [`release`] resets the caller's header to
//! [`PieForwardPlan::default`] after freeing it — which is what makes a
//! double release a no-op instead of the double-free it would be in the
//! loader's boxed-header scheme.

use std::collections::HashMap;

use crate::trace::{ForwardPlan, Op, OpKind, StateRef, StateStore, ValueInfo};

use super::types::*;

/// Owns the backing storage for one plan.
///
/// The vectors are frozen once [`build`] returns: the published slices point
/// into their heap buffers, which moving or boxing the arena does not
/// disturb (the same property `loader/src/ffi/arena.rs` relies on).
pub struct PlanArena {
    values: Vec<PieForwardValue>,
    ops: Vec<PieForwardOp>,
    value_ids: Vec<u32>,
    names: Vec<PieForwardName>,
    name_bytes: Vec<u8>,
    /// The TRACED form the wire arrays were flattened from, kept so the
    /// lowering can be asked for later (`pie_forward_lower`). The wire
    /// form is lossy on purpose — it is what the driver's walk needs —
    /// and `lower` reads the plan, not the walk's view of it.
    ///
    /// Costs one plan per model, which is what the driver already holds
    /// one of.
    plan: ForwardPlan,
    /// The last lowering asked for, kept alive so the launch list handed
    /// back can point at its kernel names instead of copying them. One
    /// slot: the shadow compares a fire and moves on, and a second ask
    /// invalidates the first — which is stated on `pie_forward_lower`.
    shadow: Option<crate::lower::Lowered>,
    shadow_wire: Vec<super::types::PieForwardLaunch>,
    shadow_names: Vec<PieForwardName>,
    shadow_name_bytes: Vec<u8>,
}

/// Interns strings into the arena's name table during a build.
///
/// Separate from [`PlanArena`] because the map is scaffolding: it exists to
/// dedupe (`embed` is named by both `Embed` and a tied `LmHead`) and is
/// dropped when the build finishes, so it must not ride along in the frozen
/// arena the plan keeps alive.
#[derive(Default)]
struct Interner {
    index: HashMap<String, u32>,
}

impl Interner {
    fn intern(&mut self, arena: &mut PlanArena, name: &str) -> u32 {
        if let Some(&id) = self.index.get(name) {
            return id;
        }
        let offset = arena.name_bytes.len() as u32;
        arena.name_bytes.extend_from_slice(name.as_bytes());
        arena.names.push(PieForwardName {
            offset,
            len: name.len() as u32,
        });
        let id = (arena.names.len() - 1) as u32;
        self.index.insert(name.to_string(), id);
        id
    }
}

/// Flatten one [`ValueInfo`] into the inline-dims POD form.
///
/// Panics if a shape outgrows [`PIE_FORWARD_MAX_DIMS`]. That is a tracer
/// bug, not caller input — the tracer emits rank-2 shapes plus the MoE
/// trace's rank-3 route-expanded values — and the entry rules turn the
/// panic into an abort rather than a status
/// (`loader/src/ffi/entry.rs:1-19`).
fn flatten_value(info: &ValueInfo) -> PieForwardValue {
    assert!(
        info.shape.0.len() <= PIE_FORWARD_MAX_DIMS,
        "traced value of rank {} exceeds PIE_FORWARD_MAX_DIMS ({PIE_FORWARD_MAX_DIMS})",
        info.shape.0.len()
    );
    let mut dims = [PieForwardDim::default(); PIE_FORWARD_MAX_DIMS];
    for (slot, dim) in dims.iter_mut().zip(&info.shape.0) {
        *slot = (*dim).into();
    }
    PieForwardValue {
        rank: info.shape.0.len() as u32,
        dims,
        dtype: info.dtype.into(),
    }
}

/// One op's flattened slots, [`flatten_kind`]'s result.
///
/// `aux_names` is a range of NAME indices in the flat id array — empty for
/// every kind except `Launch`, whose consumed weight names it carries.
struct OpParts {
    kind: PieForwardOpKind,
    weight_name: u32,
    param0: u32,
    param1: u32,
    selector: u32,
    aux_names: PieForwardIdRange,
}

impl OpParts {
    fn plain(parts: (PieForwardOpKind, u32, u32, u32, u32)) -> Self {
        let (kind, weight_name, param0, param1, selector) = parts;
        Self {
            kind,
            weight_name,
            param0,
            param1,
            selector,
            aux_names: PieForwardIdRange { offset: 0, len: 0 },
        }
    }
}

/// Flatten one [`OpKind`] into its POD slots.
///
/// The mapping is the table documented on [`PieForwardOp`]; keeping the two
/// adjacent to one match arm each is what keeps the table honest. Only the
/// expert-indexed `Matmul` carries a selector; everything else rests at
/// [`PIE_FORWARD_NO_VALUE`].
fn flatten_kind(arena: &mut PlanArena, interner: &mut Interner, kind: &OpKind) -> OpParts {
    let mut name = |arena: &mut PlanArena, weight: &str| interner.intern(arena, weight);
    OpParts::plain(match kind {
        OpKind::Embed { weight } => (
            PieForwardOpKind::Embed,
            name(arena, weight),
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::Matmul {
            weight,
            beta_one,
            selector,
        } => (
            PieForwardOpKind::Matmul,
            name(arena, weight),
            u32::from(*beta_one),
            0,
            selector.unwrap_or(PIE_FORWARD_NO_VALUE),
        ),
        OpKind::Rmsnorm { weight, variant } => (
            PieForwardOpKind::Rmsnorm,
            name(arena, weight),
            PieForwardNormVariant::from(*variant) as u32,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::AddBias { weight } => (
            PieForwardOpKind::AddBias,
            name(arena, weight),
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::RmsnormPerHead {
            weight,
            head_dim,
            variant,
        } => (
            PieForwardOpKind::RmsnormPerHead,
            name(arena, weight),
            *head_dim,
            PieForwardNormVariant::from(*variant) as u32,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::SplitQkv { q_width, kv_width } => (
            PieForwardOpKind::SplitQkv,
            PIE_FORWARD_NO_NAME,
            *q_width,
            *kv_width,
            PIE_FORWARD_NO_VALUE,
        ),
        // Partial rope crosses as param1: the rotary channel count, 0 for
        // the full rotation (no real partial width is 0 — the driver
        // clamps to >= 2 — so the resting value cannot be mistaken).
        OpKind::Rope { kind, partial } => (
            PieForwardOpKind::Rope,
            PIE_FORWARD_NO_NAME,
            PieForwardRopeKind::from(*kind) as u32,
            partial.unwrap_or(0),
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::KvAppend { layer } => (
            PieForwardOpKind::KvAppend,
            PIE_FORWARD_NO_NAME,
            *layer,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::Attention { layer } => (
            PieForwardOpKind::Attention,
            PIE_FORWARD_NO_NAME,
            *layer,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::Swiglu { inter } => (
            PieForwardOpKind::Swiglu,
            PIE_FORWARD_NO_NAME,
            *inter,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::LmHead { weight } => (
            PieForwardOpKind::LmHead,
            name(arena, weight),
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::ResidualAdd => (
            PieForwardOpKind::ResidualAdd,
            PIE_FORWARD_NO_NAME,
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::TopK { k } => (
            PieForwardOpKind::TopK,
            PIE_FORWARD_NO_NAME,
            *k,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::WeightedSum { k } => (
            PieForwardOpKind::WeightedSum,
            PIE_FORWARD_NO_NAME,
            *k,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::SigmoidGateAdd => (
            PieForwardOpKind::SigmoidGateAdd,
            PIE_FORWARD_NO_NAME,
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::SplitGdn { width0, width1 } => (
            PieForwardOpKind::SplitGdn,
            PIE_FORWARD_NO_NAME,
            *width0,
            *width1,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::CausalConv1d {
            weight,
            layer,
            kernel,
        } => (
            PieForwardOpKind::CausalConv1d,
            name(arena, weight),
            *layer,
            *kernel,
            PIE_FORWARD_NO_VALUE,
        ),
        // The one kind that names two weights: the a_log name rides in the
        // weight slot, the dt_bias name as a param0 NAME INDEX (the table
        // on `PieForwardOp` documents this).
        OpKind::GdnPrep { a_log, dt_bias } => {
            let a_log = name(arena, a_log);
            let dt_bias = name(arena, dt_bias);
            (
                PieForwardOpKind::GdnPrep,
                a_log,
                dt_bias,
                0,
                PIE_FORWARD_NO_VALUE,
            )
        }
        OpKind::GatedDelta { layer } => (
            PieForwardOpKind::GatedDelta,
            PIE_FORWARD_NO_NAME,
            *layer,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::RmsnormGated { weight } => (
            PieForwardOpKind::RmsnormGated,
            name(arena, weight),
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::SplitQGate { heads, head_dim } => (
            PieForwardOpKind::SplitQGate,
            PIE_FORWARD_NO_NAME,
            *heads,
            *head_dim,
            PIE_FORWARD_NO_VALUE,
        ),
        OpKind::SigmoidGateMul => (
            PieForwardOpKind::SigmoidGateMul,
            PIE_FORWARD_NO_NAME,
            0,
            0,
            PIE_FORWARD_NO_VALUE,
        ),
        // The stated launch: the KERNEL name rides the weight slot (the
        // one name every Launch has), the weight names it consumes ride
        // `aux_names` (a range of NAME indices in the flat id array, in
        // signature order), and the state mark rides the params —
        // param0 = store (0 none, 1 kv-cache, 2 recurrent), param1 = the
        // state layer.
        OpKind::Launch {
            kernel,
            weights,
            state,
        } => {
            let kernel = name(arena, kernel);
            let ids: Vec<u32> = weights.iter().map(|w| name(arena, w)).collect();
            let aux = store_ids(arena, &ids);
            let (store, layer) = match state {
                None => (0, 0),
                Some(StateRef {
                    store: StateStore::KvCache,
                    layer,
                }) => (1, *layer),
                Some(StateRef {
                    store: StateStore::RecurrentState,
                    layer,
                }) => (2, *layer),
            };
            return OpParts {
                kind: PieForwardOpKind::Launch,
                weight_name: kernel,
                param0: store,
                param1: layer,
                selector: PIE_FORWARD_NO_VALUE,
                aux_names: aux,
            };
        }
        // The hook site: stage wire value in param0, layer in param1 (and
        // the op's own layer field). Observes its input; nothing else
        // crosses — sidebands are runtime data.
        OpKind::HookSite { stage, layer } => (
            PieForwardOpKind::HookSite,
            PIE_FORWARD_NO_NAME,
            match stage {
                crate::trace::HookStage::OnAttnProj => 0,
                crate::trace::HookStage::OnAttn => 1,
            },
            *layer,
            PIE_FORWARD_NO_VALUE,
        ),
        // The lowered branch chain over runtime inputs: arm count in
        // param0; the aux run is [kind0, payload0, len0, kind1, payload1,
        // len1, ..., else_len] — three u32s per arm plus the trailing
        // else-region length (the flat id array holds plain u32s; what a
        // range means is the kind's contract, as with Launch).
        OpKind::Guard { arms, else_ops } => {
            let mut run: Vec<u32> = Vec::with_capacity(arms.len() * 3 + 1);
            for arm in arms {
                let (kind, payload) = arm.pred.wire();
                run.extend([kind, payload, arm.ops]);
            }
            run.push(*else_ops);
            let aux = store_ids(arena, &run);
            return OpParts {
                kind: PieForwardOpKind::Guard,
                weight_name: PIE_FORWARD_NO_NAME,
                param0: arms.len() as u32,
                param1: 0,
                selector: PIE_FORWARD_NO_VALUE,
                aux_names: aux,
            };
        }
        // Loop peeling (A3): prefix-region length in param0, tail-region
        // length in param1; the row split is a runtime input of the
        // fire. The window AXIS crosses as a one-word aux run — absent
        // for the default hook-free axis (pre-window consumers read an
        // empty run), `[1]` for the unmasked-prefix axis (the spatial
        // mask split).
        OpKind::Peel {
            prefix_ops,
            tail_ops,
            window,
        } => {
            let aux = match window {
                crate::trace::PeelWindow::HookFreePrefix => store_ids(arena, &[]),
                crate::trace::PeelWindow::UnmaskedPrefix => store_ids(arena, &[1]),
            };
            return OpParts {
                kind: PieForwardOpKind::Peel,
                weight_name: PIE_FORWARD_NO_NAME,
                param0: *prefix_ops,
                param1: *tail_ops,
                selector: PIE_FORWARD_NO_VALUE,
                aux_names: aux,
            };
        }
    })
}

/// Append an op's operand ids to the flat array and describe the run.
fn store_ids(arena: &mut PlanArena, ids: &[u32]) -> PieForwardIdRange {
    let offset = arena.value_ids.len() as u32;
    arena.value_ids.extend_from_slice(ids);
    PieForwardIdRange {
        offset,
        len: ids.len() as u32,
    }
}

fn flatten_op(
    arena: &mut PlanArena,
    interner: &mut Interner,
    plan: &crate::trace::ForwardPlan,
    op: &Op,
) -> PieForwardOp {
    let parts = flatten_kind(arena, interner, &op.kind);
    let inputs = store_ids(arena, &op.inputs);
    let outputs = store_ids(arena, &op.outputs);
    PieForwardOp {
        kind: parts.kind,
        layer: op.layer.map_or(PIE_FORWARD_NO_LAYER, |l| l as i32),
        weight_name: parts.weight_name,
        param0: parts.param0,
        param1: parts.param1,
        selector: parts.selector,
        aux_names: parts.aux_names,
        // Derived at the boundary now (migration step 5): the wire word
        // is unchanged, but it is no longer IR vocabulary — 2 is the
        // kernel table's `depth_prefix_plan`, 1 is "layer-tagged under a
        // depth-declaring trace", 0 is outside the axis.
        depth_role: if plan.depth_prefix_plan(op) {
            2
        } else if plan.depth_windowed(op) {
            1
        } else {
            0
        },
        inputs,
        outputs,
    }
}

/// Convert a traced [`ForwardPlan`] into the POD form the driver walks.
///
/// The returned header's slices point into an arena leaked behind `owner`;
/// the caller (or the C caller holding the header) must hand it to
/// [`release`].
pub fn build(plan: &ForwardPlan) -> PieForwardPlan {
    let mut arena = PlanArena {
        values: Vec::new(),
        ops: Vec::new(),
        value_ids: Vec::new(),
        names: Vec::new(),
        name_bytes: Vec::new(),
        plan: plan.clone(),
        shadow: None,
        shadow_wire: Vec::new(),
        shadow_names: Vec::new(),
        shadow_name_bytes: Vec::new(),
    };
    let mut interner = Interner::default();

    let family = interner.intern(&mut arena, &plan.family);

    arena.values.reserve(plan.values.len());
    for value in &plan.values {
        let flat = flatten_value(value);
        arena.values.push(flat);
    }

    arena.ops.reserve(plan.ops.len());
    for op in &plan.ops {
        let flat = flatten_op(&mut arena, &mut interner, plan, op);
        arena.ops.push(flat);
    }

    let values = PieForwardValueSlice {
        ptr: arena.values.as_ptr(),
        len: arena.values.len(),
    };
    let ops = PieForwardOpSlice {
        ptr: arena.ops.as_ptr(),
        len: arena.ops.len(),
    };
    let value_ids = PieForwardU32Slice {
        ptr: arena.value_ids.as_ptr(),
        len: arena.value_ids.len(),
    };
    let names = PieForwardNameSlice {
        ptr: arena.names.as_ptr(),
        len: arena.names.len(),
    };
    let name_bytes = PieForwardBytes {
        ptr: arena.name_bytes.as_ptr(),
        len: arena.name_bytes.len(),
    };
    let owner = Box::into_raw(Box::new(arena)).cast::<std::ffi::c_void>();
    PieForwardPlan {
        family,
        values,
        ops,
        value_ids,
        names,
        name_bytes,
        compiler_version: super::compiler_version(),
        depth_window: u8::from(plan.depth_window),
        owner,
    }
}

/// Reclaim the arena behind a plan header produced by [`build`] and reset
/// the header to empty.
///
/// Idempotent: a released (or never-filled) header has a null `owner` and is
/// left untouched. The loader cannot offer this — its header is itself a Rust
/// allocation — but here the header belongs to the caller, so resetting it is
/// both possible and the honest thing to leave behind.
///
/// # Safety
///
/// `plan` is null, or points at a writable header whose `owner` is null or
/// was produced by [`build`] and not yet released.
pub unsafe fn release(plan: *mut PieForwardPlan) {
    if plan.is_null() {
        return;
    }
    let plan = unsafe { &mut *plan };
    if !plan.owner.is_null() {
        drop(unsafe { Box::from_raw(plan.owner.cast::<PlanArena>()) });
    }
    *plan = PieForwardPlan::default();
}

/// The arena is immutable once published, and every pointer it hands out is
/// into storage it exclusively owns, so a plan can be read from any thread
/// and freed from a thread other than the one that built it — the same
/// guarantee (and rationale) as `loader/src/ffi/arena.rs:466-474`.
unsafe impl Send for PlanArena {}
unsafe impl Sync for PlanArena {}
unsafe impl Send for PieForwardPlan {}
unsafe impl Sync for PieForwardPlan {}

/// Read a plan's contents back as safe Rust slices, for tests that assert
/// the POD form reproduces the trace it was built from.
#[cfg(test)]
pub(crate) mod view {
    use super::*;

    pub fn ops(plan: &PieForwardPlan) -> &[PieForwardOp] {
        if plan.ops.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.ops.ptr, plan.ops.len) }
    }

    pub fn values(plan: &PieForwardPlan) -> &[PieForwardValue] {
        if plan.values.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.values.ptr, plan.values.len) }
    }

    pub fn ids(plan: &PieForwardPlan, range: PieForwardIdRange) -> &[u32] {
        if plan.value_ids.ptr.is_null() {
            return &[];
        }
        let all = unsafe { std::slice::from_raw_parts(plan.value_ids.ptr, plan.value_ids.len) };
        &all[range.offset as usize..(range.offset + range.len) as usize]
    }

    pub fn name(plan: &PieForwardPlan, index: u32) -> &str {
        assert_ne!(index, PIE_FORWARD_NO_NAME, "op names no weight");
        let names = unsafe { std::slice::from_raw_parts(plan.names.ptr, plan.names.len) };
        let entry = names[index as usize];
        let bytes =
            unsafe { std::slice::from_raw_parts(plan.name_bytes.ptr, plan.name_bytes.len) };
        std::str::from_utf8(&bytes[entry.offset as usize..(entry.offset + entry.len) as usize])
            .expect("name table holds UTF-8")
    }
}

/// Lower `plan` over `rows` and publish the result in the plan's own
/// arena, so the returned view outlives the call without copying.
///
/// The previous lowering is dropped: one slot, because the shadow
/// compares a fire and moves on. A caller holding an older
/// [`PieForwardLowered`] across a second call is reading freed storage,
/// which is why the entry point says so.
pub fn lower(
    header: &mut PieForwardPlan,
    rows: &[crate::lower::Row],
) -> PieForwardLowered {
    if header.owner.is_null() {
        return PieForwardLowered::default();
    }
    // Borrowed, not taken: `release` still owns the box.
    let arena = unsafe { &mut *header.owner.cast::<PlanArena>() };

    let lowered = match crate::lower::lower(&arena.plan, rows) {
        Ok(lowered) => lowered,
        Err(why) => {
            arena.shadow = None;
            arena.shadow_wire.clear();
            arena.shadow_names.clear();
            arena.shadow_name_bytes.clear();
            return PieForwardLowered {
                uncovered: match why {
                    crate::lower::Uncovered::Rows { .. } => PieForwardUncovered::Rows,
                    crate::lower::Uncovered::WholeKernelSplit { .. } => {
                        PieForwardUncovered::WholeKernelSplit
                    }
                    crate::lower::Uncovered::Discontiguous { .. } => {
                        PieForwardUncovered::Discontiguous
                    }
                    crate::lower::Uncovered::UnknownBackend(_) => {
                        PieForwardUncovered::UnknownBackend
                    }
                },
                ..PieForwardLowered::default()
            };
        }
    };

    arena.shadow_wire.clear();
    arena.shadow_wire.reserve(lowered.launches.len());
    for launch in &lowered.launches {
        arena.shadow_wire.push(PieForwardLaunch {
            at_op: launch.args,
            kernel_name: launch.kernel as u32,
            row_lo: launch.rows.start,
            row_hi: launch.rows.end,
            layer_lo: launch.layers.start,
            layer_hi: launch.layers.end,
        });
    }
    arena.shadow_names.clear();
    arena.shadow_name_bytes.clear();
    for name in &lowered.kernels {
        let offset = arena.shadow_name_bytes.len() as u32;
        arena.shadow_name_bytes.extend_from_slice(name.as_bytes());
        arena.shadow_names.push(PieForwardName {
            offset,
            len: name.len() as u32,
        });
    }

    let view = PieForwardLowered {
        launches: arena.shadow_wire.as_ptr(),
        launches_len: arena.shadow_wire.len(),
        kernel_names: arena.shadow_names.as_ptr(),
        kernel_names_len: arena.shadow_names.len(),
        kernel_name_bytes: PieForwardBytes {
            ptr: arena.shadow_name_bytes.as_ptr(),
            len: arena.shadow_name_bytes.len(),
        },
        arena_bytes: lowered.arena_bytes,
        uncovered: PieForwardUncovered::None,
    };
    // Kept so a debugger (and any later accessor) can reach the residue
    // and rectangle count the wire form does not carry.
    arena.shadow = Some(lowered);
    view
}
