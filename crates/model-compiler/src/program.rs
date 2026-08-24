//! The binding half of the lowering: one executable [`Program`] per lane.
//!
//! `sweep` derives WHICH ops a fact word runs; this derives WHAT each op
//! touches at the fire — where every value lives, how wide it is, and what
//! call answers the point. The fire's own row count stays symbolic; a
//! value's FACTOR over it ([`Rows`]), its width and its dtype are settled
//! here, per point, from the walk's rules.
//!
//! ## Two halves: what a rectangle IS, and where it SITS
//!
//! `out_sizes` answers the first — the width table, read off the
//! declarations and the builders. `carve` answers the second, and it does
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
    for seam in plan.seams.iter().filter(|s| s.seam == model_ir::seam::OUT.name) {
        for &v in &seam.values {
            let v = root(slots, v) as usize;
            if matches!(slots[v], Slot::Arena { .. }) {
                // `first: 0` covers the value no step of this lane writes —
                // which cannot happen for a seam this lane binds, and is the
                // safe reading if it ever did.
                spans[v].get_or_insert(Span { first: 0, last: end }).last = end;
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
            spans[id].get_or_insert(Span { first: 0, last: end });
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
        None => spans[v] = Some(Span { first: at, last: at }),
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

/// The rectangle of every result `op` states, or `None` when no rule covers
/// the point yet.
///
/// THE FIRE'S OWN ROW COUNT IS NOT HERE. Every result is
/// `[fire_rows * rows.factor(), width]` and `fire_rows` is a number the plan
/// does not hold and the walk does not invent. What the table answers is the
/// FACTOR — one row per token, or one per route — beside the width and the
/// element.
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
    let dtype = |at: usize| -> Option<Dt> { Some(like(at)?.dtype) };
    let width = |at: usize| -> Option<u64> { Some(like(at)?.width) };
    let rows = |at: usize| -> Option<Rows> { Some(like(at)?.rows) };
    // An operand's rows and element, at a width the statement decides.
    let riding = |at: usize, w: u64| -> Option<Size> { Some(sized(rows(at)?, w, dtype(at)?)) };
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
        //
        // THE ROWS RIDE THE OPERAND, and that is what makes the routed leg
        // work with no MoE special case here: a3b's `swiglu` is handed
        // `moe.matmul_select`'s result, so its rows are `FireTimes(top_k)`
        // and its width is still the ONE stated intermediate. Had the fan-out
        // been carried as a wider row instead, this rule would have had to
        // ask whether its operand was routed.
        "mlp.swiglu"
        | "mlp.swiglu_clamp"
        | "mlp.swiglu_clamp_alpha"
        | "mlp.geglu_tanh_packed"
        | "mlp.situ" => Some(vec![riding(0, param(0)?)?]),

        // ---- The bank's OUT axis. Every weight in the catalogue is
        // `[out, in]` (`o_proj: [hidden, q_heads * head_dim]`), so a matmul's
        // width is `shape[0]` and the element stays the activation's — a
        // quantized bank dequantizes into the row, it does not retype it.
        "gemm.matmul" | "gemm.lm_head" | "gemm.attention_landing" => {
            Some(vec![riding(0, axis(0, 0)?)?])
        }

        // ---- Layout: a table's row, or a cut stated by its own params.
        // `embed` is the walk's ROOT — its operand is `token_ids`, which has
        // no rectangle, so the table's second axis is the only place the
        // first activation width can come from AND this is the one rule that
        // states `Rows::Fire` outright rather than inheriting it. Every other
        // per-token rectangle in a plan is downstream of this one.
        "layout.embed" => Some(vec![sized(
            Rows::Fire,
            axis(0, 1)?,
            activation(&bank(0)?.repr),
        )]),
        "layout.split_qkv" => Some(vec![
            riding(0, param(0)?)?,
            riding(0, param(1)?)?,
            riding(0, param(1)?)?,
        ]),
        // HALVES, and the param is a pitch and not a width. The packed row
        // is `[rows, heads, 2 * head_dim]` and the kernel writes
        // `[rows, heads, head_dim]` twice (`layout/deinterleave.cuh`), so
        // each half is half the row; `head_dim` only says where the heads
        // are. qwen's bank agrees — `qg_proj: [2 * q_heads * head_dim, hidden]`.
        "layout.split_q_gate" => {
            let packed = width(0)?;
            let head_dim = param(0)?;
            if head_dim == 0 || packed % (2 * head_dim) != 0 {
                return None;
            }
            Some(vec![riding(0, packed / 2)?, riding(0, packed / 2)?])
        }
        "layout.split_rows" => {
            let row = width(0)?;
            let cut = param(0)?;
            if cut > row {
                return None;
            }
            Some(vec![riding(0, cut)?, riding(0, row - cut)?])
        }
        // THE STATED WIDTH, and it is the second param because the first
        // says WHICH layer. `select(relay, l, ple_dim)` slices one layer out
        // of a `[rows, layers * ple_dim]` relay; `layers` is the number the
        // statement never carried, so the width it slices to is stated
        // rather than divided out — the `rmsnorm_per_head` rule, on an
        // operand instead of a weight.
        "layout.select" => Some(vec![riding(0, param(1)?)?]),

        // ---- The gated-delta seam. `gdn_prep`'s only operand is the
        // `[a | b]` projection and its result is the decay and beta columns
        // that projection becomes, so the row is `ba`'s row on the f32 the
        // declaration spells. (The cuda routine behind it writes five
        // rectangles from a wider operand list; `kernels-cuda/src/ssm.rs`
        // states that gap rather than faking a delegation, and the four
        // extra rows are the recurrence's own arithmetic, not this
        // statement's.)
        "ssm.gdn_prep" => Some(vec![sized(rows(0)?, width(0)?, Dt::F32)]),
        // `v_heads * v_dim`, off the params the statement states — and f32,
        // which is exactly what `norm.rmsnorm_gated` downstream declares its
        // `x` to be.
        "ssm.gated_delta" | "ssm.gated_delta_chunked" => Some(vec![sized(
            rows(0)?,
            param(1)?.checked_mul(param(3)?)?,
            Dt::F32,
        )]),
        "ssm.kda_step" | "ssm.kda_chunked" => Some(vec![sized(
            rows(0)?,
            param(0)?.checked_mul(param(1)?)?,
            Dt::F32,
        )]),

        // ---- An attention that hands back its log-sum-exp: `o` is `q`, and
        // the lse is one f32 per head, so its width is `q`'s over the stated
        // `head_dim`.
        "attention.decode_lse" | "attention.prefill_lse" => {
            let q = width(0)?;
            let head_dim = param(1)?;
            if head_dim == 0 || q % head_dim != 0 {
                return None;
            }
            Some(vec![like(0)?, sized(rows(0)?, q / head_dim, Dt::F32)])
        }
        "attention.merge_lse" => Some(vec![like(0)?, like(1)?]),

        // ---- Tier-2. The operand is the PACKED qkv matmul's row and the
        // result is the roped `q` alone: the two kv planes are written
        // straight into the pages and never land in the arena, so the width
        // is the packed row less `2 * kv_heads * head_dim`. Both numbers are
        // params — `.norm(q).norm(k)` puts the two epsilons ahead of them,
        // which is why `kv_heads` is param 2 and not param 0.
        "cuda::qkv_fused_qknorm_rope_vnorm_write" => {
            let packed = width(0)?;
            let kv = param(2)?.checked_mul(param(3)?)?;
            Some(vec![riding(0, packed.checked_sub(2 * kv)?)?])
        }

        // ---- THE ROUTERS. Two results, never one, and `top_k` sizes both:
        // `routes` names the chosen experts (`i32`) and `weights` says how
        // much each counts (`f32`), which is what the declaration's two `Out`
        // slots spell. `top_k` is param 1 on all three — `experts` comes
        // first on every builder, and the two that renormalise put their
        // `renormalize`/`scaling` pair AFTER it.
        //
        // These two rectangles are the only place `top_k` is written down in
        // a bound program, and everything routed downstream reads it here.
        "moe.topk_softmax" | "moe.topk_sigmoid" | "moe.topk_sqrt_softplus" => {
            let top_k = param(1)?;
            Some(vec![
                sized(Rows::Fire, top_k, Dt::I32),
                sized(Rows::Fire, top_k, Dt::F32),
            ])
        }

        // ---- THE FAN-OUT. `y[r] = x[r] @ bank[routes[r]]`: one matmul per
        // ROUTE, so the result is one row per route — `fire_rows * top_k` —
        // and `top_k` is `routes`' own width, read off the router's rectangle
        // rather than restated on this statement (which carries no `top_k`
        // param at all).
        //
        // ALWAYS `FireTimes`, NEVER A MULTIPLY. The second `matmul_select` of
        // a text is handed an ALREADY routed operand (a3b's `down` leg reads
        // `swiglu(matmul_select(x, gate_up, routes))`), and its result is
        // still one row per route: the fan-out happens once, at the first
        // statement that consults a route. A rule that multiplied its
        // operand's factor would have said `top_k * top_k` there.
        //
        // The width is the bank's `N`. `bank` is the `[E, N, K]` expert
        // stack — three axes, not the two an ordinary `[out, in]` weight has
        // — so the out axis is `shape[1]` and `shape[0]` is the expert fan.
        //
        // AND THE ALIGNED STAGING IS NOT HERE, deliberately. cuda's grouped
        // leg is five launches (`moe_align_decode` → `gather_moe_aligned_
        // inputs` → `build_moe_ptrs_aligned` → `moe_grouped_gemm` →
        // `reorder_moe_aligned_output`) and four of them touch a PADDED
        // rectangle: the routes bucketed by expert, each bucket rounded up to
        // a block. Three facts put that rectangle on the executor's side of
        // the line and not the plan's.
        //
        // ITS ROW COUNT IS NOT A MULTIPLE OF THE FIRE'S. The legacy plan
        // carried it as `Dim::MoeAlignedRoutes`, whose rule is
        // `ceil((n*k + min(E, n*k)*(block-1)) / block) * block` — an upper
        // bound with an ADDITIVE `E*(block-1)` term, so no `fire_rows * j`
        // says it for any `j`. The count the fire actually uses is smaller
        // still and depends on the route histogram, which exists only once
        // the router has run.
        //
        // ITS BLOCK IS A DEVICE TILE, NOT A MODEL NUMBER. cuda's is 16; metal
        // sets `ROUTE_BLOCK_MATVEC = 1`, at which the aligned count is
        // EXACTLY the route count and the sort is a pure permutation. A plan
        // is plane-agnostic, and a number that changes with the tile a plane
        // picked cannot ride in one.
        //
        // NOTHING IN IT IS A STATEMENT'S VALUE. The per-block expert ids, the
        // gathered activations, the three pointer arrays — no declaration
        // carries any of them, which is the same sentence
        // `kernels-cuda/src/moe.rs` writes to explain why `moe.matmul_select`
        // is claim-only there.
        //
        // What the plan holds is the leg's LAST rectangle, the one
        // `reorder_moe_aligned_output` writes: `route_out`, with
        // `num_tokens = rows` and `num_routes = rows * width / hidden`, i.e.
        // the compact `[tokens, top_k, width]`. That is this slot, to the
        // byte. Everything between the align and the reorder is one fire's
        // scratch and belongs to whoever runs the fire.
        "moe.matmul_select" | "moe.matmul_select_bias" => {
            let top_k = u32::try_from(width(1)?).ok()?;
            if top_k == 0 {
                return None;
            }
            Some(vec![sized(
                Rows::FireTimes(top_k),
                axis(0, 1)?,
                dtype(0)?,
            )])
        }

        // ---- THE FOLD BACK. A token's `top_k` expert rows weighted into
        // one: the width survives, the fan-out does not, and the result is
        // per token again. This is the only point in the table that NARROWS
        // the row factor, which is what makes it the routed leg's closing
        // bracket.
        "moe.weighted_sum" => Some(vec![sized(Rows::Fire, width(0)?, dtype(0)?)]),
        // `y = routed + shared * sigmoid(gate)`: three per-token rows in, one
        // out. Nothing routed reaches it — `weighted_sum` already folded.
        "moe.sigmoid_gate_add" => Some(vec![like(0)?]),

        // ---- Hyper-connections (dsv4). The stack is `[tokens, streams,
        // hidden]` flattened to a row, so `streams` multiplies and divides a
        // WIDTH here and never a row count: `expand` broadcasts one row into
        // `streams` of them and `collapse` gates them back down, both inside
        // the row. The residual stays one row per token throughout.
        "hc.expand" => Some(vec![riding(0, width(0)?.checked_mul(param(0)?)?)?]),
        // THE WHOLE STACK, RETYPED. The statement carries no `stream_count`
        // — the only `Hc` point that does not — so there is no divisor to
        // narrow with, and the declaration says the result IS the stack
        // normalised ("the mixer's own input"). `like(0)` at the `f32` the
        // `Out` slot spells.
        //
        // The legacy text sized this `[tokens, hidden]` and fed it to
        // `hc_pre_postprocess`'s `[N, 2M + M*M]` mix-logit slot, which is the
        // missing mix PROJECTION already on record at `hc.collapse` — a
        // model-truth debt, not a width this statement could state.
        "hc.rmsnorm_f32" => Some(vec![sized(rows(0)?, width(0)?, Dt::F32)]),
        // Three results and the first is the one the block runs on: the
        // collapsed row (`streams` wide over `stream_count`), then the two
        // mixes the fold reads back. A mix matrix is `[M]` and `[M, M]` and
        // both ride f32, which the declaration spells and the kernels insist
        // on — rounding a doubly-stochastic mixer to bf16 leaves it
        // measurably un-stochastic.
        "hc.gates" => {
            let streams = param(0)?;
            if streams == 0 || width(1)? % streams != 0 {
                return None;
            }
            Some(vec![
                riding(1, width(1)? / streams)?,
                sized(rows(1)?, streams, Dt::F32),
                sized(rows(1)?, streams.checked_mul(streams)?, Dt::F32),
            ])
        }
        // The stack the fold writes back is the stack it was handed, and the
        // stack is operand ONE — `fold(x, streams, post_mix, comb_mix)` puts
        // the block's answer first because that is what a statement's
        // receiver is.
        "hc.fold" => Some(vec![like(1)?]),
        "hc.collapse" => {
            let streams = param(0)?;
            if streams == 0 || width(0)? % streams != 0 {
                return None;
            }
            Some(vec![riding(0, width(0)? / streams)?])
        }

        // ---- The compressed KV plane (dsv4). ONE ROW PER TOKEN, AND THE
        // KERNELS ARE WHY. A "pooled row" reading would need the boundary
        // COUNT, which is a histogram over the fire's positions and known
        // only at the fire — but no kernel here asks for one:
        // `dsv4_boundary_meta_decode` writes `out_pos[t]`/`out_req[t]` for
        // every `t < n` and marks a non-boundary with `-1`, and
        // `dsv4_compress_gather_paged` reads `boundary_pos[c]` per block and
        // falls a `bpos < 0` slot through as zeros. That is deliberate: a
        // compacted list needs a D2H copy and a stream sync, which makes the
        // layer ineligible for CUDA-graph capture. The list is fixed-length
        // by design, so the honest row count is the fire's and no runtime
        // quantity is needed.
        //
        // `positions` is a runtime plane with no rectangle, so the two metas
        // state `Rows::Fire` outright — the `layout.embed` reading, on the
        // other root a plan has.
        "pool.boundary_decode" | "pool.boundary_prefill" => Some(vec![
            sized(Rows::Fire, 1, Dt::I32),
            sized(Rows::Fire, 1, Dt::I32),
        ]),
        // One pooled entry per boundary SLOT, `head_dim` wide — the stated
        // width, which is what an `Out` the statement allocates is for.
        //
        // THE ELEMENT IS THE ACTIVATION'S and there is nowhere else to read
        // it: both operands are `i32` boundary planes and a `Cache` row
        // carries no dtype, so the entries ride the plane's activation the
        // way `layout.embed`'s row does. bf16 on every claim in this tree —
        // the `activation` helper below says so in one place.
        "pool.gather" => Some(vec![sized(Rows::Fire, param(0)?, Dt::Bf16)]),
        // The pooled attention hands back `q`'s rectangle and one f32 per
        // head beside it, which is `attention.prefill_lse`'s shape with the
        // head count STATED rather than divided out: `heads` is param 1
        // (`ratio` leads on every `Pool` builder).
        "pool.attention_lse" => Some(vec![like(0)?, sized(rows(0)?, param(1)?, Dt::F32)]),

        // ---- Multi-head latent attention. Every width here is a stated
        // number times a stated number, because a `Const` bank carries an
        // address and no rectangle: `kv_b` is
        // `[heads, nope_dim + v_head_dim, kv_lora_rank]` and each absorb
        // slices it itself, which is why the declarations state the half
        // they do NOT use.
        //
        // The cut, both ways. `kv_lora_rank` is param 1 — `.norm(n)` records
        // the weight and then its epsilon, so every `Mla` point that takes a
        // `Norm` has `eps` at param 0 — and the rope half is what is LEFT of
        // the projection, not a restatement of `rope_dim`. `mla.latents`
        // states no `rope_dim` at all, so the subtraction is the only rule
        // both forms can share.
        "mla.latents" | "mla.latents_rope" => {
            let rank = param(1)?;
            Some(vec![
                riding(0, rank)?,
                riding(0, width(0)?.checked_sub(rank)?)?,
            ])
        }
        // `[heads, nope_dim]` and `[heads, rope_dim]`, per head.
        "mla.split_q_b" => Some(vec![
            riding(0, param(0)?.checked_mul(param(1)?)?)?,
            riding(0, param(0)?.checked_mul(param(2)?)?)?,
        ]),
        // The query in the latent basis: `[heads, kv_lora_rank]`.
        "mla.absorb_q" => Some(vec![riding(0, param(0)?.checked_mul(param(1)?)?)?]),
        // Back out to the value basis: `[heads, v_head_dim]`, and
        // `v_head_dim` is param 2 here because `absorb_out` names the half it
        // SKIPS (`nope_dim`) last.
        "mla.absorb_out" => Some(vec![riding(0, param(0)?.checked_mul(param(2)?)?)?]),
        // All four attentions answer in the latent basis:
        // `[heads, kv_lora_rank]`, which is what the absorb back out reads.
        // The ragged forms put `indptr` between `q` and the rest, so the
        // params line up on all four.
        "mla.attention_decode"
        | "mla.attention_prefill"
        | "mla.attention_decode_selected"
        | "mla.attention_prefill_selected" => {
            Some(vec![riding(0, param(0)?.checked_mul(param(1)?)?)?])
        }

        // ---- The sparse indexer. Both rotations are IN PLACE, which is what
        // `InOut` says: the result is the row that was rotated.
        "index.layernorm_rope" | "index.rope" => Some(vec![like(0)?]),

        // ---- THE RESULT THIS TABLE COULD NOT SIZE, SIZED — by changing the
        // VALUE and not by inventing a number. `index.topk` used to answer a
        // byte MASK over the cached keys, and the kv extent it would be wide
        // by is a per-request runtime number that appears in no operand, no
        // param and no bank of the statement; the legacy
        // `dsa_index_topk_mask` only escaped that because it scored the
        // fire's own token plane and its `[tokens, tokens]` is a row count
        // this walk does not hold either. The point now answers the SELECTION
        // ITSELF — `[tokens, top_k]` of `i32`, ascending, `-1` past the end —
        // and `top_k` is param 2 of the statement. Same rule
        // `moe.topk_sigmoid`'s `routes` rides, for the same reason: a fixed
        // budget is sizable and the thing it was chosen out of is not.
        //
        // THE ROWS RIDE THE QUERY rather than being restated as `Rows::Fire`:
        // one selection row per query row is what the point means, and a
        // decode leg whose query is one row per request must not be told it
        // is one row per token by this table.
        "index.topk" => Some(vec![sized(rows(0)?, param(2)?, Dt::I32)]),

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
