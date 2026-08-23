//! The binding half of the lowering: one executable [`Program`] per lane.
//!
//! `sweep` derives WHICH ops a fact word runs; this derives WHAT each op
//! touches at the fire — where every value lives, how wide it is, and what
//! call answers the point. Rows stay symbolic (the fire's row count); widths
//! and dtypes are settled here, per point, from the walk's rules.

use model_ir::kernels::Backend;
use model_ir::plan::{Cond, Op, Param, Plan, ValueDef, ValueId};

/// One lane, executable: the ops in issue order with their calls resolved,
/// and one slot per plan value saying where it lives when this lane fires.
#[derive(Debug, Clone)]
pub struct Program {
    /// The fact words this lane serves.
    pub words: Vec<u64>,
    pub steps: Vec<Step>,
    /// Indexed by [`ValueId`].
    pub slots: Vec<Slot>,
    /// Bytes of arena per fire row — the no-reuse MVP layout: every arena
    /// slot's row sits at `row * row_pitch + offset`.
    pub row_pitch: u64,
}

#[derive(Debug, Clone)]
pub struct Step {
    /// Index into `plan.ops`.
    pub op: u32,
    pub call: Call,
}

/// How the driver reaches the op's implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Call {
    /// A `#[routine(canon = ..)]` answer: fire by symbol through the plane's
    /// signature table.
    Symbol(&'static str),
    /// A `#[claims]` trait answer: fire through the plane's point shim.
    Point(String),
    /// A tier-2 statement: the symbol is the statement, verbatim.
    Tier2(String),
}

/// Where a value lives at the fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    /// Staged by the driver from the fire's own data, by name
    /// (`token_ids`, `positions`, `qo_indptr`).
    Runtime(String),
    /// A rectangle in the fire's arena: `row * program.row_pitch + offset`,
    /// `width` elements of `dtype` per row.
    Arena { offset: u64, width: u64, dtype: Dt },
    /// A merge: on this lane exactly one arm survives, and the value IS it.
    Alias(ValueId),
    /// An effect or an op this lane never runs: nothing to address.
    Absent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt {
    Bf16,
    F32,
    I32,
    U32,
    U8,
}

impl Dt {
    #[must_use]
    pub fn size(self) -> u64 {
        match self {
            Dt::Bf16 => 2,
            Dt::F32 | Dt::I32 | Dt::U32 => 4,
            Dt::U8 => 1,
        }
    }
}

/// One value's row: how many elements wide, riding which element.
type Size = (u64, Dt);

/// What one lane needed and the walk could not answer.
///
/// PER LANE AND NOT PER PLAN, because the measurement is per lane: qwen's
/// prefill leg states `ssm.gated_delta_chunked`, which no cuda routine
/// claims, and its decode leg states none of it. A plan-wide refusal would
/// report the hybrid as unrunnable when half of it runs today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// Index into `sweep::lanes(plan)`.
    pub lane: usize,
    /// The fact words that lane serves.
    pub words: Vec<u64>,
    /// Every point this lane asked for and the walk could not answer, one
    /// row per point in plan order — the whole measurement, not the first
    /// thing that went wrong.
    pub gaps: Vec<Gap>,
}

/// One point a lane states and the walk cannot bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    /// Index into `plan.ops`: the first statement that asked.
    pub op: u32,
    pub point: String,
    pub why: Why,
    /// How many of this lane's statements state the point.
    pub statements: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// [`call_of`] answered nothing: the plane neither claims the point nor
    /// spells a `canon` for it. The backlog `sweep::resolve` already counts.
    Unclaimed,
    /// The point resolves and the width table has no rule for it — the
    /// walk's OWN backlog, which is not the plane's.
    Unsized,
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let why = match self.why {
            Why::Unclaimed => "UNCLAIMED",
            Why::Unsized => "UNSIZED",
        };
        write!(
            f,
            "{} -> {why} (first at op {}, {} statements)",
            self.point, self.op, self.statements
        )
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lane {} words {:?}:", self.lane, self.words)?;
        for gap in &self.gaps {
            write!(f, " {gap};")?;
        }
        Ok(())
    }
}

/// Every lane of `plan`, bound. Refuses a plan whose lanes include a point
/// the walk cannot answer — the walk's own measured backlog, one row per
/// refusing lane.
///
/// # Errors
///
/// The refusals, one per lane that has one. A plan whose every lane sizes
/// returns `Ok`.
pub fn programs(plan: &Plan) -> Result<Vec<Program>, Vec<Refusal>> {
    let (mut built, mut refused) = (Vec::new(), Vec::new());
    for lane in bound(plan) {
        match lane {
            Ok(program) => built.push(program),
            Err(refusal) => refused.push(refusal),
        }
    }
    if refused.is_empty() {
        Ok(built)
    } else {
        Err(refused)
    }
}

/// Every lane of `plan`, bound or refused, in `sweep::lanes` order — one
/// entry per lane, so a report can print the lanes that run beside the ones
/// that do not.
#[must_use]
pub fn bound(plan: &Plan) -> Vec<Result<Program, Refusal>> {
    crate::sweep::lanes(plan)
        .iter()
        .enumerate()
        .map(|(at, lane)| bind(plan, at, lane))
        .collect()
}

fn bind(plan: &Plan, at: usize, lane: &crate::sweep::Lane) -> Result<Program, Refusal> {
    let mut gaps: Vec<Gap> = Vec::new();
    let mut note = |op: u32, why: Why| {
        let point = plan.ops[op as usize].kernel.as_str();
        match gaps.iter_mut().find(|g| g.point == point) {
            Some(seen) => seen.statements += 1,
            None => gaps.push(Gap {
                op,
                point: point.to_string(),
                why,
                statements: 1,
            }),
        }
    };

    let mut steps = Vec::with_capacity(lane.ops.len());
    let mut runs = vec![false; plan.ops.len()];
    // A point with no call is a point with no rectangle either: nothing
    // fires, so nothing is written, and sizing its results would measure a
    // launch that is not there. Poisoned, not sized — and its consumers are
    // poisoned with it, so the report stays the gap and not the wake.
    let mut fires = vec![false; plan.ops.len()];
    for &op in &lane.ops {
        runs[op as usize] = true;
        match call_of(plan.plane, plan.ops[op as usize].kernel.as_str()) {
            Some(call) => {
                fires[op as usize] = true;
                steps.push(Step { op, call });
            }
            None => note(op, Why::Unclaimed),
        }
    }

    // VALUE ORDER IS TOPOLOGICAL. The recorder pushes a statement's results
    // after its operands and a merge after its arms, so one forward pass
    // over the values sees every size it reads already settled.
    let mut sizes: Vec<Option<Size>> = vec![None; plan.values.len()];
    let mut poisoned = vec![false; plan.values.len()];
    let mut slots: Vec<Slot> = Vec::with_capacity(plan.values.len());
    for (id, def) in plan.values.iter().enumerate() {
        let slot = match def {
            ValueDef::Runtime(name) => Slot::Runtime(name.clone()),
            ValueDef::Merge(arms) => match surviving_arm(lane, arms) {
                None => Slot::Absent,
                Some(arm) => {
                    sizes[id] = sizes[arm as usize];
                    poisoned[id] = poisoned[arm as usize];
                    // Chase the chain: a merge of merges aliases the
                    // rectangle, not the alias.
                    match slots[arm as usize] {
                        Slot::Alias(through) => Slot::Alias(through),
                        _ => Slot::Alias(arm),
                    }
                }
            },
            ValueDef::Stmt(op) => {
                if !runs[*op as usize] {
                    Slot::Absent
                } else {
                    let stmt = &plan.ops[*op as usize];
                    if stmt.outputs[0] as usize == id {
                        let spoilt = !fires[*op as usize]
                            || stmt.inputs.iter().any(|v| poisoned[*v as usize]);
                        let ins: Vec<Option<Size>> =
                            stmt.inputs.iter().map(|v| sizes[*v as usize]).collect();
                        match out_sizes(&stmt.kernel, plan, stmt, &ins).filter(|_| !spoilt) {
                            Some(outs) => {
                                assert_eq!(
                                    outs.len(),
                                    stmt.outputs.len(),
                                    "`{}` states {} results and the width rule sizes {}",
                                    stmt.kernel,
                                    stmt.outputs.len(),
                                    outs.len()
                                );
                                for (v, size) in stmt.outputs.iter().zip(outs) {
                                    sizes[*v as usize] = Some(size);
                                }
                            }
                            None => {
                                if !spoilt {
                                    note(*op, Why::Unsized);
                                }
                                for v in &stmt.outputs {
                                    poisoned[*v as usize] = true;
                                }
                            }
                        }
                    }
                    match sizes[id] {
                        Some((width, dtype)) => Slot::Arena { offset: 0, width, dtype },
                        None => Slot::Absent,
                    }
                }
            }
        };
        slots.push(slot);
    }

    if !gaps.is_empty() {
        gaps.sort_by_key(|g| g.op);
        return Err(Refusal {
            lane: at,
            words: lane.words.clone(),
            gaps,
        });
    }

    // THE NO-REUSE MVP: every rectangle this lane mints keeps its own column
    // of the row, in value order. Merges alias and allocate nothing.
    let mut cursor = 0u64;
    for slot in &mut slots {
        if let Slot::Arena { offset, width, dtype } = slot {
            *offset = align16(cursor);
            cursor = *offset + *width * dtype.size();
        }
    }

    Ok(Program {
        words: lane.words.clone(),
        steps,
        slots,
        row_pitch: align16(cursor),
    })
}

fn align16(bytes: u64) -> u64 {
    (bytes + 15) & !15
}

/// Which arm of a merge survives on this lane, if any.
///
/// EVERY WORD OF THE LANE MUST AGREE, and that is not a hope: a lane is the
/// set of words that run the same ops, and an arm's condition is its
/// producing statement's condition. Two words that disagreed here would have
/// been two lanes.
fn surviving_arm(lane: &crate::sweep::Lane, arms: &[(ValueId, Cond)]) -> Option<ValueId> {
    let hit = |word: u64| {
        let mut holding = arms.iter().filter(|(_, cond)| cond.holds(word));
        let first = holding.next().map(|(id, _)| *id);
        assert!(
            holding.next().is_none(),
            "two arms of one merge hold at word {word:#b}"
        );
        first
    };
    let mut words = lane.words.iter();
    let first = hit(*words.next().expect("a lane serves at least one word"));
    for &word in words {
        assert_eq!(
            hit(word),
            first,
            "a lane's words disagree about which arm of a merge survives"
        );
    }
    first
}

/// The width and element of every result `op` states, or `None` when no rule
/// covers the point yet.
///
/// ROWS ARE NOT HERE. Every result is `[rows, width]` and `rows` is the
/// fire's row count — a number the plan does not hold and the walk does not
/// invent, so the table answers the half that is decided at trace time.
///
/// The rules are read off three places and nowhere else: the declaration in
/// `kernels::points` (which slot is `InOut`, which `Out` is spelled `f32`),
/// the builder in `model_dsl::kernels` (which value is at which index of
/// `op.inputs`, which scalar at which index of `op.params`), and
/// `plan.params` for a weight's own dimensions — which is what the Load
/// contract's parameter registration is FOR.
fn out_sizes(point: &str, plan: &Plan, op: &Op, ins: &[Option<Size>]) -> Option<Vec<Size>> {
    // AN EFFECT STATES NO RECTANGLE. `attention.kv_append` and its siblings
    // leave the fire's rows in a pool and return nothing, so there is
    // nothing for a width rule to answer and no slot to mint.
    if op.outputs.is_empty() {
        return Some(Vec::new());
    }
    let like = |at: usize| -> Option<Size> { *ins.get(at)? };
    let dtype = |at: usize| -> Option<Dt> { Some(like(at)?.1) };
    let param = |at: usize| -> Option<u64> { op.params.get(at).copied() };
    let bank = |at: usize| -> Option<&Param> {
        let name = op.weights.get(at)?;
        plan.params.iter().find(|p| &p.name == name)
    };
    let axis = |at: usize, dim: usize| -> Option<u64> { bank(at)?.shape.get(dim).copied() };

    match point {
        // ---- The operands ARE the results: an `InOut` slot rotated, added
        // to, scaled or gated in place. `norm.residual_add` is the one whose
        // `InOut` is not the receiver — the declaration reads
        // `(x: In, y: InOut)` and the builder records `x` first.
        "norm.residual_add" => Some(vec![like(1)?]),
        "norm.add_bias"
        | "norm.mul_scalar"
        | "norm.scale"
        | "dist.all_reduce"
        | "gate.sigmoid_mul"
        | "attention.sink"
        | "attention.lse_ln"
        | "attention.logit_softcap"
        | "rope.partial_q"
        | "rope.partial_last" => Some(vec![like(0)?]),
        "rope.full" | "rope.partial" | "rope.yarn" => Some(vec![like(0)?, like(1)?]),

        // ---- Like the first `In`: a normalisation, a convolution and an
        // attention reading all hand back the rectangle they were given.
        "norm.rmsnorm"
        | "norm.rmsnorm_per_head"
        | "norm.rmsnorm_plus_one"
        | "norm.rmsnorm_per_head_plus_one"
        | "norm.rmsnorm_no_scale"
        | "norm.res_blend"
        | "mlp.geglu_tanh"
        | "ssm.causal_conv1d"
        | "ssm.causal_conv1d_chunked"
        | "attention.decode"
        | "attention.prefill"
        | "attention.masked" => Some(vec![like(0)?]),

        // ---- The GATE decides, not `x`. Both gated norms take an f32 core
        // out of a recurrent mixer and a gate on the activation element, and
        // the declaration spells the result `Out<Tensor<T>>` — the gate's.
        "norm.rmsnorm_gated" | "norm.rmsnorm_gated_by" => Some(vec![like(1)?]),

        // ---- The packed activations: one `[gate | up]` row in, one
        // `intermediate` row out, and `intermediate` is the statement's
        // first param on every one of them.
        "mlp.swiglu"
        | "mlp.swiglu_clamp"
        | "mlp.swiglu_clamp_alpha"
        | "mlp.geglu_tanh_packed"
        | "mlp.situ" => Some(vec![(param(0)?, dtype(0)?)]),

        // ---- The bank's OUT axis. Every weight in the catalogue is
        // `[out, in]` (`o_proj: [hidden, q_heads * head_dim]`), so a matmul's
        // width is `shape[0]` and the element stays the activation's — a
        // quantized bank dequantizes into the row, it does not retype it.
        "gemm.matmul" | "gemm.lm_head" | "gemm.attention_landing" => {
            Some(vec![(axis(0, 0)?, dtype(0)?)])
        }

        // ---- Layout: a table's row, or a cut stated by its own params.
        // `embed` is the walk's ROOT — its operand is `token_ids`, which has
        // no width, so the table's second axis is the only place the first
        // activation width can come from.
        "layout.embed" => Some(vec![(axis(0, 1)?, activation(&bank(0)?.repr))]),
        "layout.split_qkv" => {
            let element = dtype(0)?;
            Some(vec![
                (param(0)?, element),
                (param(1)?, element),
                (param(1)?, element),
            ])
        }
        // HALVES, and the param is a pitch and not a width. The packed row
        // is `[rows, heads, 2 * head_dim]` and the kernel writes
        // `[rows, heads, head_dim]` twice (`layout/deinterleave.cuh`), so
        // each half is half the row; `head_dim` only says where the heads
        // are. qwen's bank agrees — `qg_proj: [2 * q_heads * head_dim, hidden]`.
        "layout.split_q_gate" => {
            let (packed, element) = like(0)?;
            let head_dim = param(0)?;
            if head_dim == 0 || packed % (2 * head_dim) != 0 {
                return None;
            }
            Some(vec![(packed / 2, element), (packed / 2, element)])
        }
        "layout.split_rows" => {
            let (row, element) = like(0)?;
            let cut = param(0)?;
            if cut > row {
                return None;
            }
            Some(vec![(cut, element), (row - cut, element)])
        }
        // NOT SIZABLE FROM THE PLAN. `select(relay, l)` slices one layer out
        // of a `[rows, layers * ple_dim]` relay, and `layers` is the one
        // number the statement does not carry: the param is WHICH layer, not
        // how many. Sizing it would take either a second param or the
        // consumer's width, and the walk invents neither. Unclaimed on every
        // plane besides, so gemma-e4b refuses here twice over.
        "layout.select" => None,

        // ---- The gated-delta seam. `gdn_prep`'s only operand is the
        // `[a | b]` projection and its result is the decay and beta columns
        // that projection becomes, so the row is `ba`'s row on the f32 the
        // declaration spells. (The cuda routine behind it writes five
        // rectangles from a wider operand list; `kernels-cuda/src/ssm.rs`
        // states that gap rather than faking a delegation, and the four
        // extra rows are the recurrence's own arithmetic, not this
        // statement's.)
        "ssm.gdn_prep" => Some(vec![(like(0)?.0, Dt::F32)]),
        // `v_heads * v_dim`, off the params the statement states — and f32,
        // which is exactly what `norm.rmsnorm_gated` downstream declares its
        // `x` to be.
        "ssm.gated_delta" | "ssm.gated_delta_chunked" => {
            Some(vec![(param(1)?.checked_mul(param(3)?)?, Dt::F32)])
        }
        "ssm.kda_step" | "ssm.kda_chunked" => {
            Some(vec![(param(0)?.checked_mul(param(1)?)?, Dt::F32)])
        }

        // ---- An attention that hands back its log-sum-exp: `o` is `q`, and
        // the lse is one f32 per head, so its width is `q`'s over the stated
        // `head_dim`.
        "attention.decode_lse" | "attention.prefill_lse" => {
            let (q, element) = like(0)?;
            let head_dim = param(1)?;
            if head_dim == 0 || q % head_dim != 0 {
                return None;
            }
            Some(vec![(q, element), (q / head_dim, Dt::F32)])
        }
        "attention.merge_lse" => Some(vec![like(0)?, like(1)?]),

        // ---- Tier-2. The operand is the PACKED qkv matmul's row and the
        // result is the roped `q` alone: the two kv planes are written
        // straight into the pages and never land in the arena, so the width
        // is the packed row less `2 * kv_heads * head_dim`. Both numbers are
        // params — `.norm(q).norm(k)` puts the two epsilons ahead of them,
        // which is why `kv_heads` is param 2 and not param 0.
        "cuda::qkv_fused_qknorm_rope_vnorm_write" => {
            let (packed, element) = like(0)?;
            let kv = param(2)?.checked_mul(param(3)?)?;
            Some(vec![(packed.checked_sub(2 * kv)?, element)])
        }

        // ---- The rows algebra the MVP does not do. `moe.*` is
        // `tokens * top_k` rows, `pool.*` is one row per boundary, `hc.*` is
        // one per stream, and `mla.*`/`index.*` mix latent pitches into
        // both. None of that is a WIDTH rule, and stating a width without
        // the rows it belongs to would be the fiction this table exists to
        // avoid.
        _ => None,
    }
}

/// The activation element a bank of `repr` puts in the arena.
///
/// A QUANTIZED BANK IS NOT AN ELEMENT. `mxfp4` and `wna16` say how the
/// weights are STORED; the row a fire writes out of them rides the plane's
/// activation element, which is bf16 on every claim in this tree.
fn activation(repr: &str) -> Dt {
    match repr {
        "f32" => Dt::F32,
        _ => Dt::Bf16,
    }
}

/// The call that answers `kernel` on `plane`, mirroring `sweep::resolve`.
#[must_use]
pub fn call_of(plane: Backend, kernel: &str) -> Option<Call> {
    if let Some(rest) = kernel.strip_prefix("cuda::") {
        return (plane == Backend::Cuda).then(|| Call::Tier2(rest.to_string()));
    }
    if model_ir::kernels::point_claims(plane).contains(&kernel) {
        return Some(Call::Point(kernel.to_string()));
    }
    model_ir::kernels::canon_symbol(plane, kernel).map(Call::Symbol)
}
