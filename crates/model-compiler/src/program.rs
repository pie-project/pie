//! The binding half of the lowering: one executable [`Program`] per lane.
//!
//! `sweep` derives WHICH ops a fact word runs; this derives WHAT each op
//! touches at the fire — where every value lives, how wide it is, and what
//! call answers the point. The fire's own row count stays symbolic; a
//! value's FACTOR over it ([`Rows`]), its width and its dtype are settled
//! here, per point, out of the declaration that states the point.
//!
//! ## Two halves: what a rectangle IS, and where it SITS
//!
//! `out_sizes` answers the first, and it answers it by READING RATHER THAN
//! KNOWING: every point's `#[shape]` is a row of `kernels::points`, and what
//! stands here is the interpreter of that column. A width table used to
//! stand here instead — sixty match arms restating in this crate what the
//! declarations already said — and the whole failure it made possible was a
//! declaration and a rule drifting apart with nothing to notice.
//! `carve` answers the second, and it does
//! it by LIVENESS: a value's rectangle is needed from the step that writes it
//! to the last step that reads it ([`spans`]), and two values whose steps
//! never coincide are given the same bytes. That is the difference between a
//! pitch that is the sum of everything a lane ever mints and one that is the
//! lane's busiest instant — gemma4-31b's row went from 21.8 MiB to 1 MiB,
//! qwen35-d0.8b's from 2.45 MiB to 487 KiB, and every lane of every
//! catalogue row lands exactly on the bound. [`clashes`] is the invariant's
//! guard, because a slab shared with the wrong value does not fault: it
//! computes.

use model_ir::kernels::Backend;
use model_ir::kernels::points::{Element, Fan, Prim, Width};
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
    /// Bytes of arena one fire row needs.
    ///
    /// THE OFFSETS REUSE. Two values this lane never holds live at the same
    /// step share bytes (`carve`), so the pitch is the arena's busiest
    /// instant and not the sum of everything the lane ever mints. What a
    /// slot's offset means to an executor is unchanged, and so is the
    /// arena's size: `rows * row_pitch`.
    ///
    /// The reading is VALUE-MAJOR and both executors read it that way: value
    /// `V` owns `[offset * rows, offset * rows + bytes * rows)` and its rows
    /// sit `width` elements apart inside it, because the marks a kernel is
    /// handed carry no stride. That is a uniform scaling of the per-row
    /// layout by `rows`, so it preserves exactly the disjointness (and the
    /// sharing) this walk assigned, and every rectangle stays inside
    /// `rows * row_pitch`.
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
    /// A rectangle in the fire's arena at `offset * rows`,
    /// `rows.factor() * width` elements of `dtype` per FIRE row.
    ///
    /// THE PITCH STAYS PER FIRE ROW. A [`Rows::FireTimes`] slot keeps its
    /// `k` routed rows CONTIGUOUS inside its own column, so it contributes
    /// `k * width * dtype.size()` bytes to `row_pitch` and one token's
    /// routed rows sit `width` elements apart. That is the legacy staging's
    /// `[tokens, top_k, width]` rectangle to the byte — the same buffer,
    /// said in the terms the plan can state.
    ///
    /// AN OFFSET IS NOT PRIVATE. Values whose lives do not overlap are given
    /// the same bytes on purpose; what a slot owns is its offset for the
    /// steps [`spans`] says it is live, and nothing outside those steps may
    /// read it. The one reader past the walk is the `out` seam, and the
    /// spans hold it open to fire end for exactly that reason.
    Arena {
        offset: u64,
        rows: Rows,
        width: u64,
        dtype: Dt,
    },
    /// A merge: on this lane exactly one arm survives, and the value IS it.
    Alias(ValueId),
    /// An effect or an op this lane never runs: nothing to address.
    Absent,
}

impl Slot {
    /// The bytes ONE FIRE ROW of this slot occupies, the routed factor
    /// included — zero for anything that is not a rectangle of the arena.
    ///
    /// What a kernel actually touches, before the 16 the carve rounds a
    /// reservation up to.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            Slot::Arena {
                rows, width, dtype, ..
            } => rows.factor() * width * dtype.size(),
            Slot::Runtime(_) | Slot::Alias(_) | Slot::Absent => 0,
        }
    }
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

/// How many rows a value has, in the only terms a plan can state.
///
/// A FIRE BRINGS ITS OWN ROW COUNT and the plan does not hold it: `fire_rows`
/// is one row per token, decided when the driver assembles the batch. What
/// the plan DOES hold is the FACTOR a value's row count carries over that
/// number, and this tree has exactly two of them.
///
/// The second is the whole reason the enum exists. A router picks `top_k`
/// experts per token and `moe.matmul_select` runs one matmul per PICK, so
/// its result is `fire_rows * top_k` rows and not `fire_rows` — and every
/// point between that fan-out and `moe.weighted_sum`'s fold back rides the
/// wider count. Calling those values `[fire_rows, width]` would have been
/// off by `top_k` in the row count and by `top_k` in the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows {
    /// One row per token this fire carries.
    Fire,
    /// `fire_rows * k` — one row per ROUTE, `k` the router's `top_k`.
    ///
    /// THE FACTOR IS READ, NEVER RESTATED. `moe.matmul_select` states no
    /// `top_k`; it states `routes`, whose rectangle IS `[fire_rows, top_k]`,
    /// so the width of that operand is where `k` comes from. A statement
    /// that restated the number could disagree with the router that made it.
    FireTimes(u32),
}

impl Rows {
    /// How many rows of this value ride ONE fire row.
    #[must_use]
    pub fn factor(self) -> u64 {
        match self {
            Rows::Fire => 1,
            Rows::FireTimes(k) => u64::from(k),
        }
    }
}

/// One value's rectangle, as far as the plan can state it: its row factor
/// over the fire's own count, how many elements wide, riding which element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Size {
    rows: Rows,
    width: u64,
    dtype: Dt,
}

/// `width` elements of `dtype`, on `rows` rows.
const fn sized(rows: Rows, width: u64, dtype: Dt) -> Size {
    Size { rows, width, dtype }
}

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

/// WHY A STATEMENT DID NOT BIND, and the three answers are three different
/// things to do about it.
///
/// The plane-gate answer is here because it used to be somewhere else.
/// `sweep::Resolution` separated `violations` — a tier-2 symbol stated on a
/// plane that does not declare it, which is a REFUSED PLAN — from
/// `unresolved`, the honest backlog; [`call_of`] collapsed both into
/// [`Why::Unclaimed`], and the driver read the poorer of the two answers
/// while a report binary read the richer one. One derivation answers all
/// three now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// [`call_for`] answered nothing: the plane neither claims the point nor
    /// spells a `canon` for it. The honest backlog — a point to write.
    Unclaimed,
    /// A PLANE-GATED SYMBOL ON THE WRONG PLANE. A `cuda::` statement is legal
    /// on cuda and nowhere else, so this is not a backlog row: nothing is
    /// missing, the plan is wrong. A text states one behind `inputs.cuda()`
    /// with an unfused chain as the else, and a lane that arrives here lost
    /// that gate.
    WrongPlane,
    /// The point resolves and its `#[shape]` cannot be answered on this
    /// statement — the walk's OWN backlog, which is not the plane's. Either
    /// no declaration states the point at all, or the rule read something the
    /// bound statement does not have: an operand with no rectangle, a cut
    /// that does not come out whole, a fan of nothing.
    Unsized,
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let why = match self.why {
            Why::Unclaimed => "UNCLAIMED",
            Why::WrongPlane => "WRONG PLANE",
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
        match call_for(plan.plane, plan.ops[op as usize].kernel.as_str()) {
            Ok(call) => {
                fires[op as usize] = true;
                steps.push(Step { op, call });
            }
            Err(why) => note(op, why),
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
                        Some(Size { rows, width, dtype }) => Slot::Arena {
                            offset: 0,
                            rows,
                            width,
                            dtype,
                        },
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

    let spans = spans_over(plan, &steps, &slots);
    let row_pitch = carve(&mut slots, &spans);

    Ok(Program {
        words: lane.words.clone(),
        steps,
        slots,
        row_pitch,
    })
}

fn align16(bytes: u64) -> u64 {
    (bytes + 15) & !15
}

/// The steps one value must survive, in the lane's own issue order.
///
/// `first` is the step that writes it and `last` the last step that reads
/// it, both inclusive — so a value with no reader at all still spans the one
/// step that minted it, because the launch writes through the pointer either
/// way.
///
/// INCLUSIVE IS THE WHOLE SAFETY ARGUMENT, and it is what makes the `InOut`
/// question answer itself. A statement at step `s` reads its operands and
/// writes its results in the same launch: the operand's span ends at `s`,
/// the result's begins at `s`, and the two touch — so the allocator can
/// never hand one the other's bytes. That is the right answer for an
/// ordinary point (the kernel is still reading `x` while it writes `y`) and
/// it is the only available answer for an `InOut` one, where whether the
/// launch may alias its own operand is a fact about the kernel's indexing
/// that no plan states. A plan that WANTS the aliasing says so the way it
/// already can — a merge, which allocates nothing and shares by
/// construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub first: u32,
    pub last: u32,
}

impl Span {
    /// Are both values live at some common step?
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// The step at which each value's rectangle is born and the last step that
/// reads it, indexed by [`ValueId`] — `None` for a value this lane gives no
/// rectangle.
///
/// The lane's answer, not the plan's: a value's span is measured over the
/// steps THIS lane runs, which is why it is derived from `program.steps`
/// rather than from `plan.ops`.
#[must_use]
pub fn spans(plan: &Plan, program: &Program) -> Vec<Option<Span>> {
    spans_over(plan, &program.steps, &program.slots)
}

fn spans_over(plan: &Plan, steps: &[Step], slots: &[Slot]) -> Vec<Option<Span>> {
    // ONE STEP PAST THE WALK. The driver's delivery reads the `out` seam
    // AFTER the last step has been issued (`driver-cuda/src/fire/launch.rs`
    // repitches `fire.rect(baked.out)` into the logits buffer;
    // `baker-smoke` reads the same rectangle the same way), so the value it
    // names is live at a time no statement occupies. Giving that time an
    // index rather than a special case is what keeps the rule one sentence:
    // the logits column is simply live longer than any step.
    let end = u32::try_from(steps.len()).expect("a lane of fewer steps than u32 counts");
    let mut spans: Vec<Option<Span>> = vec![None; slots.len()];
    for (at, step) in steps.iter().enumerate() {
        let at = u32::try_from(at).expect("a step index inside the lane");
        let op = &plan.ops[step.op as usize];
        // OPERANDS AND RESULTS AT THE SAME INSTANT, which is the inclusive
        // reading [`Span`] argues for. An effect (`attention.kv_append`)
        // states no result and still holds its operands open here.
        for v in op.inputs.iter().chain(op.outputs.iter()) {
            touch(slots, &mut spans, *v, at);
        }
    }
    // THE DELIVERY TAIL. Every value the `out` seam names lives to `end`.
    // Only that seam: the other seams a plan carries (`attn.q`, `attn.out`,
    // `recurrent`) are trace-time attach points, and an adapter that takes
    // one becomes STATEMENTS in the plan — which this walk already sees. A
    // rule that pinned every seam open would be paying for readers that do
    // not exist, and would hold one activation per layer live across the
    // whole stack.
    for seam in plan
        .seams
        .iter()
        .filter(|s| s.seam == model_ir::seam::OUT.name)
    {
        for &v in &seam.values {
            let v = root(slots, v) as usize;
            if matches!(slots[v], Slot::Arena { .. }) {
                // `first: 0` covers the value no step of this lane writes —
                // which cannot happen for a seam this lane binds, and is the
                // safe reading if it ever did.
                spans[v]
                    .get_or_insert(Span {
                        first: 0,
                        last: end,
                    })
                    .last = end;
            }
        }
    }
    // EVERY RECTANGLE ENDS UP WITH A SPAN, so that nothing downstream has to
    // decide what an absent one means. One cannot be absent — `bind` mints a
    // rectangle only for a statement this lane runs, and a statement this
    // lane runs is a step whose results are touched above — and if one ever
    // were, holding it open across the whole lane is the reading that cannot
    // be wrong.
    for (id, slot) in slots.iter().enumerate() {
        if matches!(slot, Slot::Arena { .. }) {
            spans[id].get_or_insert(Span {
                first: 0,
                last: end,
            });
        }
    }
    spans
}

fn touch(slots: &[Slot], spans: &mut [Option<Span>], v: ValueId, at: u32) {
    // A MERGE IS ITS ARM'S LIFE, NOT ITS OWN. Reading the merged value reads
    // the surviving arm's rectangle, so the read lands on the arm — which is
    // how an alias extends the life of what it points at.
    let v = root(slots, v) as usize;
    if !matches!(slots[v], Slot::Arena { .. }) {
        return;
    }
    match &mut spans[v] {
        Some(span) => {
            span.first = span.first.min(at);
            span.last = span.last.max(at);
        }
        None => {
            spans[v] = Some(Span {
                first: at,
                last: at,
            })
        }
    }
}

/// The rectangle an alias finally names. `bind` already collapses chains, so
/// this is one hop in practice and a loop for the reader's sake.
fn root(slots: &[Slot], mut v: ValueId) -> ValueId {
    for _ in 0..=slots.len() {
        match slots[v as usize] {
            Slot::Alias(to) => v = to,
            _ => return v,
        }
    }
    panic!("a cycle of aliases");
}

/// Give every rectangle an offset, sharing bytes between values that are
/// never live together, and answer the row pitch that covers them all.
///
/// # Why greedy-by-size, and how close it gets
///
/// The lower bound is the arena's busiest instant: the total bytes live at
/// whichever step holds the most ([`live_bound`]). Reaching it exactly is
/// dynamic storage allocation, which is NP-hard in general — but these
/// intervals are a transformer's, which is to say a few long-lived residuals
/// crossing a long chain of short-lived scratch, and placing the big blocks
/// first leaves gaps the small ones drop into. The measured tables sit ON
/// the bound for every catalogue row, so nothing more elaborate has earned
/// its way in.
///
/// # Deterministic, and that is a requirement
///
/// The same plan must lay out the same way on every host: a program is
/// cached, compared and fired by offsets. So the order is a TOTAL one —
/// bytes descending, then birth step, then value id — and never a hash's.
fn carve(slots: &mut [Slot], spans: &[Option<Span>]) -> u64 {
    // A ROUTED SLOT PAYS FOR ALL ITS ROWS HERE, once: `row_pitch` is bytes
    // per FIRE row, so a `FireTimes(k)` column is `k` sub-rows wide and the
    // pitch carries the whole fan-out. Nothing downstream multiplies again.
    //
    // Sizes are rounded to 16 so that a freed hole is 16-aligned too, which
    // is what keeps every offset aligned with no separate padding pass.
    let mut order: Vec<(u64, Span, usize)> = slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| matches!(slot, Slot::Arena { .. }))
        .map(|(id, slot)| {
            let span = spans[id].expect("`spans` answers every arena slot");
            (align16(slot.bytes()), span, id)
        })
        .collect();
    order.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.first.cmp(&b.1.first))
            .then(a.2.cmp(&b.2))
    });

    let mut placed: Vec<(u64, u64, Span)> = Vec::with_capacity(order.len());
    let mut blockers: Vec<(u64, u64)> = Vec::new();
    let mut pitch = 0u64;
    for (bytes, span, id) in order {
        // The lowest offset no value live beside this one already holds:
        // walk the blockers in address order, stepping past each one that
        // starts before the gap under consideration closes.
        blockers.clear();
        blockers.extend(
            placed
                .iter()
                .filter(|(_, _, live)| live.overlaps(span))
                .map(|(at, size, _)| (*at, at + size)),
        );
        blockers.sort_unstable();
        let mut at = 0u64;
        for (from, to) in &blockers {
            if *from >= at + bytes {
                break;
            }
            at = at.max(*to);
        }
        let Slot::Arena { offset, .. } = &mut slots[id] else {
            unreachable!("only arena slots are ordered")
        };
        *offset = at;
        placed.push((at, bytes, span));
        pitch = pitch.max(at + bytes);
    }
    pitch
}

/// The busiest instant: the most bytes this lane holds live at any one step,
/// each rounded to the 16 an offset has to sit on.
///
/// THE FLOOR NO LAYOUT CAN BEAT, and the number the reuse is measured
/// against — [`Program::row_pitch`] sitting ON it is the whole claim, and it
/// does on every lane of every catalogue row today.
#[must_use]
pub fn live_bound(plan: &Plan, program: &Program) -> u64 {
    let spans = spans(plan, program);
    let sized: Vec<(Span, u64)> = program
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| matches!(slot, Slot::Arena { .. }))
        .map(|(id, slot)| {
            let span = spans[id].expect("`spans` answers every arena slot");
            (span, align16(slot.bytes()))
        })
        .collect();
    let end = u32::try_from(program.steps.len()).unwrap_or(u32::MAX);
    (0..=end)
        .map(|at| {
            sized
                .iter()
                .filter(|(span, _)| span.first <= at && at <= span.last)
                .map(|(_, bytes)| *bytes)
                .sum()
        })
        .max()
        .unwrap_or(0)
}

/// Every pair of values this lane holds live at one step whose rectangles
/// nevertheless share a byte.
///
/// THE ARENA'S WHOLE INVARIANT, and cheap enough to keep as its guard. A
/// reused slab does not fault when it is wrong — the addresses stay inside
/// the block and the launches all succeed — so the only thing that catches a
/// layout mistake is arithmetic, either this or a checkpoint's logits.
/// Empty on every program `bind` builds.
#[must_use]
pub fn clashes(plan: &Plan, program: &Program) -> Vec<(ValueId, ValueId)> {
    let spans = spans(plan, program);
    // THE BYTES A KERNEL TOUCHES, not the 16-rounded reservation: a pair that
    // shared only padding would be a carve this walk never produces, and
    // reporting it would name a clash no launch can see.
    let live: Vec<(ValueId, Span, u64, u64)> = program
        .slots
        .iter()
        .enumerate()
        .filter_map(|(id, slot)| match slot {
            Slot::Arena { offset, .. } => Some((
                u32::try_from(id).ok()?,
                spans[id].expect("`spans` answers every arena slot"),
                *offset,
                slot.bytes(),
            )),
            _ => None,
        })
        .collect();
    let mut found = Vec::new();
    for (i, (a, a_span, a_at, a_bytes)) in live.iter().enumerate() {
        for (b, b_span, b_at, b_bytes) in &live[i + 1..] {
            if a_span.overlaps(*b_span) && *a_at < b_at + b_bytes && *b_at < a_at + a_bytes {
                found.push((*a, *b));
            }
        }
    }
    found
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

/// The rectangle of every result `op` states, or `None` when the walk cannot
/// answer one.
///
/// THIS IS AN INTERPRETER AND NOT A TABLE, and what it interprets is the
/// declaration. Sixty match arms used to stand here, one per point, each
/// restating in this crate which operand a result rode and how wide it was —
/// which made a point's arity a fact of `kernels::points` and a point's
/// GEOMETRY a fact of `model-compiler`, two crates that nothing held
/// together. A declaration that changed its slots and not this file compiled
/// and computed a different model. The rules moved to the `#[shape]` beside
/// each method, the walk reads [`Point::outs`], and the two cannot disagree
/// because there is only one of them.
///
/// THE FIRE'S OWN ROW COUNT IS NOT HERE. Every result is
/// `[fire_rows * rows.factor(), width]` and `fire_rows` is a number the plan
/// does not hold and the walk does not invent. What the declaration answers
/// is the FACTOR — one row per token, or one per route — beside the width and
/// the element.
///
/// The three things it reads are the three a bound statement has: the
/// operands' rectangles (`ins`, already settled by the topological walk), the
/// scalars the statement stated (`op.params`), and `plan.params` for a
/// weight's own dimensions — which is what the Load contract's parameter
/// registration is FOR.
fn out_sizes(point: &str, plan: &Plan, op: &Op, ins: &[Option<Size>]) -> Option<Vec<Size>> {
    let declared = model_ir::kernels::point_of(point)?;
    declared
        .outs
        .iter()
        .map(|shape| {
            Some(sized(
                rows_of(shape.rows, ins)?,
                width_of(&shape.width, plan, op, ins)?,
                elem_of(shape.elem, plan, op, ins)?,
            ))
        })
        .collect()
}

/// The result's row FACTOR, as the declaration states it.
fn rows_of(fan: Fan, ins: &[Option<Size>]) -> Option<Rows> {
    match fan {
        Fan::Fire => Some(Rows::Fire),
        Fan::Ride(at) => Some(operand(ins, at)?.rows),
        // THE FACTOR IS READ, NEVER RESTATED. `moe.matmul_select` states no
        // `top_k`; it states `routes`, whose rectangle IS `[fire_rows,
        // top_k]`, so the width of that operand is where `k` comes from. A
        // fan of nothing is a refusal and not an empty rectangle: a router
        // that picked no expert is a plan that cannot run.
        Fan::Per(at) => {
            let k = u32::try_from(operand(ins, at)?.width).ok()?;
            (k != 0).then_some(Rows::FireTimes(k))
        }
    }
}

/// The result's width, evaluated over the statement.
///
/// EVERY OPERATOR REFUSES RATHER THAN ROUNDS. A division in this algebra is
/// always a row being cut into equal pieces — heads out of a packed row,
/// streams out of a stack — so a remainder means the operand was not the
/// rectangle the statement thought it was, and the honest answer is the
/// walk's own backlog row rather than a width that is nearly right.
fn width_of(w: &Width, plan: &Plan, op: &Op, ins: &[Option<Size>]) -> Option<u64> {
    let two =
        |a: &Width, b: &Width| Some((width_of(a, plan, op, ins)?, width_of(b, plan, op, ins)?));
    match *w {
        Width::Of(at) => Some(operand(ins, at)?.width),
        Width::Stated(at) => op.params.get(at).copied(),
        Width::Axis(at, dim) => weight(plan, op, at)?.shape.get(dim).copied(),
        Width::Count(n) => Some(n),
        Width::Times(a, b) => {
            let (a, b) = two(a, b)?;
            a.checked_mul(b)
        }
        Width::Over(a, b) => {
            let (a, b) = two(a, b)?;
            (b != 0 && a % b == 0).then(|| a / b)
        }
        Width::Less(a, b) => {
            let (a, b) = two(a, b)?;
            a.checked_sub(b)
        }
    }
}

/// The element the result rides, which the declaration's own `Out` slot chose
/// and this only has to look up.
fn elem_of(elem: Element, plan: &Plan, op: &Op, ins: &[Option<Size>]) -> Option<Dt> {
    match elem {
        Element::Ride(at) => Some(operand(ins, at)?.dtype),
        Element::Weight(at) => Some(activation(&weight(plan, op, at)?.repr)),
        Element::Fixed(p) => match p {
            Prim::F32 => Some(Dt::F32),
            Prim::I32 => Some(Dt::I32),
            Prim::U32 => Some(Dt::U32),
            Prim::U8 => Some(Dt::U8),
            // A HOST SCALAR'S RUN AND NOT A RECTANGLE'S. `#[points]` refuses
            // a `bool` result at the declaration, so this arm is what that
            // refusal looks like from the other side.
            Prim::Bool => None,
        },
        Element::Activation => Some(Dt::Bf16),
    }
}

/// The rectangle bound to the statement's `at`-th operand, if it has one. A
/// runtime plane (`token_ids`, `positions`) has none, which is why the
/// declarations that read one root their rows in the fire instead.
fn operand(ins: &[Option<Size>], at: usize) -> Option<Size> {
    *ins.get(at)?
}

/// The Load-contract parameter at the statement's `at`-th weight column: a
/// `Const` carries an address and no rectangle at the fire, so its dimensions
/// and its repr are read off the registration a text made when it named the
/// weight.
fn weight<'p>(plan: &'p Plan, op: &Op, at: usize) -> Option<&'p Param> {
    let name = op.weights.get(at)?;
    plan.params.iter().find(|p| &p.name == name)
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

/// The call that answers `kernel` on `plane`, or [`Why`] it does not.
///
/// THE ONE DERIVATION, and it used to be two. `sweep::resolve` walked the
/// same three questions in the same order and answered them into three
/// vectors — `resolved`, `unresolved`, `violations` — while `call_of` (which
/// is what every driver and the width walk actually called) answered an
/// `Option` and lost the third. The header on that function said so out
/// loud: *"mirroring `sweep::resolve`"*. A mirror is a copy, and the copy
/// was the expressive one, read by a report binary and by nothing that
/// fires.
///
/// # The order is the gate, then the claim, then the canon
///
/// THE `cuda::` PREFIX IS THE PLANE GATE AND NOTHING ELSE. A tier-2
/// statement is legal on one plane, so a plan on any other is a
/// [`Why::WrongPlane`]; on that plane the gate has done its job and what is
/// left is the point's own name, which is what the plane's `TIER2_POINTS`
/// spells and what [`Call::Tier2`] carries. Answered as a tier-2 call and
/// not as a bare symbol, because a bare symbol is what a `canon` row answers
/// with and the two reach a driver by different doors — this one through the
/// generated dispatch, that one through a staging shim.
///
/// # Errors
///
/// [`Why::WrongPlane`] for a plane-gated statement off its plane,
/// [`Why::Unclaimed`] where the plane neither claims the point nor spells a
/// `canon` for it. [`Why::Unsized`] is never answered here: it is the width
/// walk's own backlog and needs a bound statement, not a name.
pub fn call_for(plane: Backend, kernel: &str) -> Result<Call, Why> {
    if let Some(rest) = kernel.strip_prefix("cuda::") {
        return match plane {
            Backend::Cuda => Ok(Call::Tier2(rest.to_string())),
            _ => Err(Why::WrongPlane),
        };
    }
    if model_ir::kernels::point_claims(plane).contains(&kernel) {
        return Ok(Call::Point(kernel.to_string()));
    }
    model_ir::kernels::canon_symbol(plane, kernel)
        .map(Call::Symbol)
        .ok_or(Why::Unclaimed)
}

/// [`call_for`] with the reason dropped — the answer a caller that only asks
/// "does this bind" wants.
#[must_use]
pub fn call_of(plane: Backend, kernel: &str) -> Option<Call> {
    call_for(plane, kernel).ok()
}
