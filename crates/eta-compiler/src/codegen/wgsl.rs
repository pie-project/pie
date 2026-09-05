//! WGSL region emitter — the source both portable shells run a guest pass
//! from. `engine-wgpu` hands what this produces to `wgpu` directly;
//! `engine-vulkan` compiles it to SPIR-V with naga. One emitter for the two,
//! so the backends cannot disagree about what a program computes.
//!
//! Emission is a pure function of the plan, as it is for the CUDA and Metal
//! arms: the same region emits the same bytes every time.
//!
//! ## What the shader is
//!
//! [`RUNTIME`] is the value store and one function per op tag, and the emitted
//! entry point is a straight line of calls through it in plan order with a
//! `storageBarrier` between them. One workgroup runs a whole region, so the
//! barrier is the only ordering needed and it always sits in uniform control
//! flow — which WGSL requires and which is why a reduce's ladder is spelled as
//! several calls here rather than a loop inside the op.
//!
//! That shape is pinned to ONE workgroup, because no barrier a shader can
//! write orders anything beyond its own. [`emit_launch_steps`] is the other
//! shape: one dispatch per step, the dispatch boundary doing the ordering, and
//! so as many workgroups as a shell cares to launch. The two emit the same
//! calls in the same order and differ only in what separates them.
//!
//! ## What it refuses
//!
//! A tag with no emitted form is refused by NAME. It is never quietly left to
//! a host interpreter: a pass either runs on the device whole or does not run.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::plan::{CompiledStage, Region};

/// The device runtime every emitted region carries.
pub const RUNTIME: &str = include_str!("../../runtime/wgsl/ptir_runtime.wgsl");

/// Invocations in the one workgroup a region runs on. Matches `PTIR_WG` in
/// [`RUNTIME`]; the two are checked against each other by
/// `the_runtime_and_the_emitter_agree_on_the_workgroup`.
pub const WORKGROUP: u32 = 256;

/// Levels a reduce ladder is emitted at. Matches `REDUCE_LEVELS`; `32^7`
/// exceeds any row a value can hold, so seven always finishes.
const REDUCE_LEVELS: u32 = 7;

/// Rounds a sort ladder is emitted at. Matches `SORT_ROUNDS`; `2^28` exceeds
/// any row the scratch bound can hold, and the count is even so the ladder's
/// ping-pong lands its answer in buffer 0.
const SORT_ROUNDS: u32 = 28;

/// The barrier the one-workgroup shape puts between ops.
///
/// **IT IS A STORAGE BARRIER, NOT A WORKGROUP ONE, AND THE DIFFERENCE IS NOT
/// COSMETIC.** Both are control barriers, so both make the lanes arrive
/// together; they differ in which memory they make visible. Every op in
/// [`RUNTIME`] reads and writes `heap`, a STORAGE buffer, and no body declares
/// `var<workgroup>` at all — so `workgroupBarrier` would order the one kind of
/// memory the runtime never touches and say nothing about the kind it lives
/// in. A lane could then read a slot the previous op wrote and get the old
/// bytes: a stale number, not a fault. `storageBarrier` orders the writes that
/// are actually there.
const BARRIER: &str = "  storageBarrier();\n";

/// `kWgslEmitterVersion` — bumped whenever emitted WGSL changes, so a shell's
/// pipeline cache keys on it.
///
/// **3** — a sort ladder emits one more rung, `ptir_pivot_pack`, and
/// `pivot_threshold`'s top-`p` arm now writes its keep flags to scratch for
/// that rung to fold rather than writing the output byte by byte. Version-2
/// bytes are a shader whose ladder stops one rung short, so the top-`p`
/// output would be whatever the scratch held: silent-wrong, which is the
/// worse kind, and why the key moves with the bytes.
///
/// **2** — the emitted entry point now sets `lanes`, and the barrier between
/// ops is `storageBarrier`. A cache still holding version-1 bytes would hand
/// the device a shader whose `lanes` is zero, and every strided loop in the
/// runtime advances by `lanes`: not a wrong answer but a hang, which is why
/// the key must move with the bytes.
pub const WGSL_EMITTER_VERSION: u16 = 3;

/// Why a region has no WGSL kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The entry name is not a WGSL identifier.
    EntryName(String),
    /// An op the runtime has no form for. Named, so the refusal says which.
    Op {
        /// The wire tag with no emitted form.
        tag: u8,
        /// Its name in `OP_TABLE`, so the refusal reads as the op the author
        /// wrote rather than a number.
        name: &'static str,
    },
    /// The region's node list does not index the stage's ops.
    NodeOutOfRange(u32),
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EntryName(name) => {
                write!(f, "`{name}` is not a WGSL identifier")
            }
            Self::Op { tag, name } => write!(
                f,
                "this backend emits no WGSL for `{name}` (tag {tag:#04x}), and a guest pass \
                 runs on the device whole or not at all"
            ),
            Self::NodeOutOfRange(node) => {
                write!(
                    f,
                    "the region names node {node}, which the stage does not have"
                )
            }
        }
    }
}

/// Every tag [`RUNTIME`]'s `ptir_step` has an arm for.
///
/// Answering `true` for a tag with no arm would emit a kernel that leaves its
/// output untouched — silent-wrong rather than refused — so this list is
/// spelled out and `every_emitted_tag_has_a_runtime_arm` reads it back against
/// the runtime source in both directions.
#[must_use]
pub fn emits(tag: u8) -> bool {
    matches!(
        tag,
        tags::EXP
            | tags::LOG
            | tags::NEG
            | tags::RECIP
            | tags::ABS
            | tags::SIGN
            | tags::CAST
            | tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::REM
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::GT
            | tags::GE
            | tags::EQ
            | tags::NE
            | tags::LT
            | tags::LE
            | tags::AND
            | tags::OR
            | tags::NOT
            | tags::SELECT
            | tags::REDUCE_SUM
            | tags::REDUCE_MAX
            | tags::REDUCE_MIN
            | tags::REDUCE_ARGMAX
            | tags::CUMSUM
            | tags::CUMPROD
            | tags::BROADCAST
            | tags::RESHAPE
            | tags::TRANSPOSE
            | tags::GATHER
            | tags::GATHER_ROW
            | tags::SCATTER_ADD
            | tags::SCATTER_SET
            | tags::IOTA
            | tags::MASK_APPLY_PACKED
            | tags::CAUSAL_MASK
            | tags::SLIDING_WINDOW_MASK
            | tags::SINK_WINDOW_MASK
            | tags::RNG
            | tags::RNG_KEYED
            | tags::CONST
            | tags::KERNEL_CALL
            | tags::SINK_CALL
            | tags::SORT_DESC
            | tags::TOP_K
            | tags::PIVOT_THRESHOLD
            | tags::MATMUL
    )
}

/// Whether a tag's work is a reduce ladder rather than one pass.
const fn is_reduce(tag: u8) -> bool {
    matches!(
        tag,
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX
    )
}

/// Whether a tag's answer is read off an ordered row, so its work is a merge
/// ladder rather than one pass.
const fn is_sort(tag: u8) -> bool {
    matches!(tag, tags::SORT_DESC | tags::TOP_K | tags::PIVOT_THRESHOLD)
}

/// Whether an op is the SHELL's to resolve rather than the device's to run.
///
/// **A BOUNDARY OP IS NOT A MISSING DEVICE FORM.** A `chan_take` reads a ring,
/// a `chan_put` writes one and an `intrinsic_val` binds the fire's logits:
/// each moves a value ACROSS the host/device line, which is the one thing a
/// shader cannot do for itself. The interpreter does not run them through
/// `eval_op` either — `exec_stage` resolves them as value ROOTS — which is why
/// [`emits`] does not claim them and why a stage full of them is not a stage
/// this emitter failed at.
///
/// `kernel_call` and `sink_call` share the `Intrinsic` family and are NOT
/// boundary ops: the runtime gives them real forms, matching `eval_op`.
///
/// It lives here rather than in a shell because both shells run the same
/// emitted source and so must draw the line in the same place.
#[must_use]
pub fn is_boundary(tag: u8) -> bool {
    if tag == tags::INTRINSIC_VAL {
        return true;
    }
    eta_ir::op::spec(tag).is_some_and(|row| row.family == eta_ir::op::Family::Channel)
}

/// Merge rounds a sort of `plan`'s node `at` needs, which is at most
/// [`SORT_ROUNDS`] and is usually fewer.
///
/// **WHY FEWER IS SAFE, AND WHY THE BOUND IS THE WHOLE VALUE.** The ladder
/// doubles its run length each round, so once `1 << r` reaches the row length
/// the rows are sorted and every later round is a full-row COPY that exists
/// only to keep the ping-pong's parity — `op_sort_round` says so itself. So
/// `ceil(log2(len))` rounds do all the work, and rounding that up to even
/// lands the answer in buffer 0 exactly as 28 did. On a 32k vocabulary row
/// that is 16 rounds rather than 28, and under [`emit_launch_steps`] each
/// round saved is a whole dispatch of the row saved.
///
/// The bound is the value's TOTAL element count, not its trailing axis,
/// because `sort_desc` sorts the value flat as one row while `top_k` and
/// `pivot_threshold` sort per row. The total is `>=` the row length under
/// either reading, so it can only ever over-estimate — and over-estimating
/// costs a copy, while under-estimating returns a half-sorted row and a wrong
/// token. A value with any symbolic axis has no total to count, so it keeps
/// the full ladder.
fn sort_rounds(plan: &crate::codegen::launch::LaunchStagePlan, at: usize) -> u32 {
    let Some(op) = plan.ops.get(at) else {
        return SORT_ROUNDS;
    };
    let Some(&arg) = op.args.first() else {
        return SORT_ROUNDS;
    };
    let Some(value) = plan.value_types.get(arg as usize) else {
        return SORT_ROUNDS;
    };
    let mut total: u64 = 1;
    for axis in &value.axes {
        let crate::plan::Dimension::Static(extent) = axis else {
            return SORT_ROUNDS;
        };
        total *= u64::from(*extent);
    }
    // `ceil(log2(total))`, then up to even. A row of 0 or 1 needs no round at
    // all: `run` is already `>= len` before the first one.
    let mut need = 0u32;
    while (1u64 << need) < total {
        need += 1;
        if need >= SORT_ROUNDS {
            return SORT_ROUNDS;
        }
    }
    let rounds = need + (need & 1);
    rounds.min(SORT_ROUNDS)
}

/// One op's calls, appended to `body`.
///
/// **THE ONE PLACE A LADDER IS SPELLED.** A reduce is several `ptir_reduce_*`
/// calls and a sort is a `rounds`-round merge ladder, because the barrier between
/// levels has to sit in uniform control flow and WGSL cannot prove that of one
/// inside an op body. A caller that spelled a ladder as one `ptir_step` would
/// leave the output untouched — silent-wrong, not refused — so no caller
/// spells it: they all come through here.
fn emit_op(body: &mut String, at: usize, tag: u8, rounds: u32) {
    if is_sort(tag) {
        let _ = writeln!(body, "  ptir_sort_seed({at}u);");
        body.push_str(BARRIER);
        for round in 0..rounds {
            let _ = writeln!(body, "  ptir_sort_round({at}u, {round}u);");
            body.push_str(BARRIER);
        }
        let _ = writeln!(body, "  ptir_sort_pre({at}u);");
        body.push_str(BARRIER);
        let _ = writeln!(body, "  ptir_step({at}u);");
        // `pivot_threshold`'s top-`p` walk owns a row and writes one word per
        // element; four bool lanes share a word, so a separate pass folds
        // them. Emitted for the whole family and a no-op for the rest, for
        // the same reason the ladder is: what a rung does is the runtime's
        // business, and how many rungs there are must not depend on a
        // predicate the emitter cannot see.
        body.push_str(BARRIER);
        let _ = writeln!(body, "  ptir_pivot_pack({at}u);");
    } else if is_reduce(tag) {
        for level in 0..REDUCE_LEVELS {
            let _ = writeln!(body, "  ptir_reduce_level({at}u, {level}u);");
            body.push_str(BARRIER);
        }
        let _ = writeln!(body, "  ptir_reduce_finish({at}u);");
    } else {
        let _ = writeln!(body, "  ptir_step({at}u);");
    }
    body.push_str(BARRIER);
}

/// **WHERE THE RUNTIME ENDS AND THE EMITTED ENTRY POINT BEGINS.**
///
/// A reader that wants to see what a stage sequenced — a test counting
/// ladder rungs, a person reading a dump — wants the entry point and not the
/// eight hundred lines of library above it. This is the line it splits on,
/// and it is a constant rather than a string spelled twice, because a marker
/// the emitter writes and a test greps for is a contract, and a contract
/// spelled in two places is a contract that drifts.
pub const ENTRY_MARKER: &str = "// ---- the entry point ----";

/// The runtime, then `body`, wrapped in an entry point named `entry_name`.
///
/// This is the ONE-WORKGROUP shape: the whole region is one dispatch, its ops
/// are ordered by the barriers `body` carries, and `lanes` is the workgroup's
/// own width because a barrier orders nothing wider. [`wrap_step`] is the
/// other shape.
fn wrap(entry_name: &str, body: &str) -> String {
    let mut source = String::with_capacity(RUNTIME.len() + body.len() + 256);
    source.push_str(RUNTIME);
    source.push('\n');
    source.push_str(ENTRY_MARKER);
    source.push_str("\n\n@compute @workgroup_size(");
    let _ = write!(source, "{WORKGROUP}");
    source.push_str(")\nfn ");
    source.push_str(entry_name);
    source.push_str("(@builtin(local_invocation_id) lid : vec3<u32>) {\n  tid = lid.x;\n");
    let _ = writeln!(source, "  lanes = {WORKGROUP}u;");
    source.push_str(body);
    source.push_str("}\n");
    source
}

/// A WGSL identifier: a letter or underscore, then alphanumerics and
/// underscores, and not the reserved double-underscore prefix.
fn valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if name.starts_with("__") {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The WGSL for one region of a stage.
///
/// # Errors
///
/// [`Refused`] when the entry name is not an identifier, when the region names
/// a node the stage does not have, or when an op has no emitted form.
pub fn emit_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, Refused> {
    if !valid_identifier(entry_name) {
        return Err(Refused::EntryName(entry_name.to_string()));
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let _ = result_bases(&ops);

    let mut body = String::new();
    for &node in &region.nodes {
        let at = node.index();
        let op = ops.get(at).ok_or(Refused::NodeOutOfRange(at as u32))?;
        if !emits(op.tag) {
            return Err(Refused::Op {
                tag: op.tag,
                name: eta_ir::op::spec(op.tag).map_or("?", |row| row.name),
            });
        }
        // The full ladder: this path holds a region rather than a launch plan,
        // so it has no value table to bound the sort by, and over-spelling a
        // ladder costs copies while under-spelling it returns a half-sorted
        // row.
        emit_op(&mut body, at, op.tag, SORT_ROUNDS);
    }

    Ok(wrap(entry_name, &body))
}

/// The WGSL for one stage of a [`LaunchPackage`] — the shape a SHELL holds.
///
/// [`emit_region`] takes a [`CompiledStage`], which is a compiler-side noun no
/// shell ever receives; a shell is handed a launch package. Both spell the
/// same straight line of calls through the same runtime, so they share
/// [`emit_op`] rather than restating it — a shell that restated the sequence
/// would run one `ptir_step` where a ladder was meant the first time this
/// emitter grew one, and answer wrong rather than refuse.
///
/// [`is_boundary`] ops are skipped: the shell stages their values into the
/// heap before the dispatch and reads them back after, and the shader reads
/// them where the shell put them.
///
/// # Errors
///
/// [`Refused::EntryName`] when the name is not a WGSL identifier, and
/// [`Refused::Op`] naming the first op with no emitted form.
///
/// [`LaunchPackage`]: crate::codegen::launch::LaunchPackage
pub fn emit_launch_stage(
    entry_name: &str,
    plan: &crate::codegen::launch::LaunchStagePlan,
) -> Result<String, Refused> {
    if !valid_identifier(entry_name) {
        return Err(Refused::EntryName(entry_name.to_string()));
    }
    let fusion = crate::codegen::wgsl_analysis::analyze_stage(plan);
    let mut body = String::new();
    for (at, op) in plan.ops.iter().enumerate() {
        if is_boundary(op.tag) || !fusion.emits_node(at) {
            continue;
        }
        if !emits(op.tag) {
            return Err(Refused::Op {
                tag: op.tag,
                name: eta_ir::op::spec(op.tag).map_or("?", |row| row.name),
            });
        }
        emit_op(&mut body, at, op.tag, sort_rounds(plan, at));
    }
    Ok(wrap(entry_name, &body))
}

/// One dispatch of a stepwise region: an entry point, and which node it runs.
///
/// A shell dispatches these in order into ONE command buffer, with a memory
/// barrier between, and never waits between them — the sequence costs one
/// submit however long it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// The entry point to dispatch. Every step of a stage lives in the one
    /// module [`Stepwise::source`] holds.
    pub entry: String,
    /// The op this step belongs to, by index into the stage's ops. Several
    /// steps share a node when the op is a ladder.
    pub node: u32,
}

/// A stage emitted as a SEQUENCE of dispatches rather than one.
///
/// **WHY THIS SHAPE EXISTS.** [`emit_launch_stage`] puts a whole region in one
/// entry point and orders its ops with `storageBarrier`. That barrier orders
/// nothing between workgroups, so such a region is pinned to ONE workgroup —
/// on a 142-multiprocessor card, one part in 142 of the machine, which
/// measured at 3.6 ns per element for a body doing almost nothing per element.
///
/// The bodies were never the problem: the runtime declares no `var<workgroup>`
/// and carries no barrier, so every op already communicates through `heap` and
/// already stripes by `lanes`. What pinned the region was the sequencing. Give
/// each step its own dispatch and the sequencing becomes the dispatch
/// boundary, which orders the whole grid — so `lanes` becomes the grid's width
/// and the region uses as much of the card as the shell cares to give it.
///
/// The cost is one dispatch command per step instead of one per region. That
/// is not one submit per step: a shell issues them all into one command buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stepwise {
    /// One module holding every step's entry point.
    pub source: String,
    /// The dispatches, in the order they must run.
    pub steps: Vec<Step>,
}

/// The runtime, then one entry point per step.
///
/// Each body is a single call — no barrier, because the dispatch boundary is
/// the ordering. `tid` is the global invocation id and `lanes` the whole grid,
/// so a wider dispatch does proportionally more work and a shell may size the
/// grid however it likes: every loop in the runtime strides by `lanes` and so
/// covers its work at any width, down to a single workgroup.
fn wrap_step(source: &mut String, entry_name: &str, call: &str) {
    source.push_str("\n@compute @workgroup_size(");
    let _ = write!(source, "{WORKGROUP}");
    source.push_str(")\nfn ");
    source.push_str(entry_name);
    source.push_str(
        "(@builtin(global_invocation_id) gid : vec3<u32>,\n   \
         @builtin(num_workgroups) grid : vec3<u32>) {\n  tid = gid.x;\n",
    );
    let _ = writeln!(source, "  lanes = grid.x * {WORKGROUP}u;");
    source.push_str(call);
    source.push_str("}\n");
}

/// One op's steps, appended to `steps`, with their entry points appended to
/// `source`.
///
/// **THE LADDER IS SPELLED HERE AND IN [`emit_op`], AND NOWHERE ELSE.** The
/// two shapes differ only in what separates the calls — a barrier there, a
/// dispatch here — so the rungs themselves are read from the same constants
/// and the same predicates, and `the_two_shapes_sequence_the_same_calls` holds
/// them against each other.
fn step_op(
    source: &mut String,
    steps: &mut Vec<Step>,
    prefix: &str,
    at: usize,
    tag: u8,
    rounds: u32,
) {
    let mut push = |call: String, suffix: &str| {
        let entry = alloc::format!("{prefix}_{at}_{suffix}");
        wrap_step(source, &entry, &call);
        steps.push(Step {
            entry,
            node: at as u32,
        });
    };
    if is_sort(tag) {
        push(alloc::format!("  ptir_sort_seed({at}u);\n"), "seed");
        for round in 0..rounds {
            push(
                alloc::format!("  ptir_sort_round({at}u, {round}u);\n"),
                &alloc::format!("r{round}"),
            );
        }
        push(alloc::format!("  ptir_sort_pre({at}u);\n"), "pre");
        push(alloc::format!("  ptir_step({at}u);\n"), "step");
        push(alloc::format!("  ptir_pivot_pack({at}u);\n"), "pack");
    } else if is_reduce(tag) {
        for level in 0..REDUCE_LEVELS {
            push(
                alloc::format!("  ptir_reduce_level({at}u, {level}u);\n"),
                &alloc::format!("l{level}"),
            );
        }
        push(alloc::format!("  ptir_reduce_finish({at}u);\n"), "fin");
    } else {
        push(alloc::format!("  ptir_step({at}u);\n"), "step");
    }
}

/// A stage emitted as one dispatch per step — the shape that can use more than
/// one workgroup. See [`Stepwise`] for why it exists.
///
/// `prefix` names the entry points: a step of node `n` is `<prefix>_<n>_<what>`,
/// so a shell can log which dispatch it is on without a table.
///
/// [`is_boundary`] ops are skipped and elided nodes are dropped, exactly as in
/// [`emit_launch_stage`] — the two emissions must agree about which nodes run
/// or a shell would get different answers from the two shapes.
///
/// # Errors
///
/// [`Refused::EntryName`] when the prefix is not a WGSL identifier, and
/// [`Refused::Op`] naming the first op with no emitted form.
pub fn emit_launch_steps(
    prefix: &str,
    plan: &crate::codegen::launch::LaunchStagePlan,
) -> Result<Stepwise, Refused> {
    if !valid_identifier(prefix) {
        return Err(Refused::EntryName(prefix.to_string()));
    }
    let fusion = crate::codegen::wgsl_analysis::analyze_stage(plan);
    let mut source = String::from(RUNTIME);
    source.push('\n');
    source.push_str(ENTRY_MARKER);
    source.push('\n');
    let mut steps = Vec::new();
    for (at, op) in plan.ops.iter().enumerate() {
        if is_boundary(op.tag) || !fusion.emits_node(at) {
            continue;
        }
        if !emits(op.tag) {
            return Err(Refused::Op {
                tag: op.tag,
                name: eta_ir::op::spec(op.tag).map_or("?", |row| row.name),
            });
        }
        step_op(
            &mut source,
            &mut steps,
            prefix,
            at,
            op.tag,
            sort_rounds(plan, at),
        );
    }
    Ok(Stepwise { source, steps })
}

#[cfg(test)]
mod tests {
    use super::{
        RUNTIME, Refused, SORT_ROUNDS, WORKGROUP, emits, is_reduce, is_sort, valid_identifier,
    };
    use alloc::format;
    use alloc::vec::Vec;
    use eta_ir::op::{OP_TABLE, tags};

    /// **THE LADDER'S HEIGHT IS ONE NUMBER, WRITTEN TWICE.**
    ///
    /// The emitter spells `REDUCE_LEVELS` calls to `ptir_reduce_level`, and
    /// `op_reduce_finish` walks `PTIR_REDUCE_LEVELS` rungs to work out which
    /// half of its scratch the last one left the answer in. If the two ever
    /// disagreed, `finish` would read the wrong buffer for exactly the rows
    /// whose fold stopped at the rung between them — a wrong answer, not a
    /// fault. So the runtime's number is read back here against the
    /// emitter's.
    #[test]
    fn the_runtime_and_the_emitter_agree_on_the_ladders_height() {
        let declared = RUNTIME
            .lines()
            .find_map(|line| {
                let line = line.trim();
                let rest = line.strip_prefix("const PTIR_REDUCE_LEVELS : u32 = ")?;
                rest.trim_end_matches(';')
                    .trim_end_matches('u')
                    .parse::<u32>()
                    .ok()
            })
            .expect("the runtime declares `PTIR_REDUCE_LEVELS`");
        assert_eq!(
            declared,
            super::REDUCE_LEVELS,
            "the runtime folds {declared} rungs and the emitter spells {}",
            super::REDUCE_LEVELS
        );
    }

    /// **THE ONE DRIFT THAT ANSWERS SILENTLY WRONG.** A reduce is a ladder of
    /// `ptir_reduce_*` calls and a sort a ladder of `ptir_sort_*` ones,
    /// because a barrier between levels must sit in uniform control flow.
    /// `emit_op` is the only place that sequence is spelled, so a ladder added
    /// to the runtime and not to it would emit one `ptir_step` where several
    /// calls were meant — the output left untouched, a wrong answer rather
    /// than a refusal. Every sequencing entry point the runtime declares must
    /// therefore be one `emit_op` can spell.
    #[test]
    fn the_runtime_grew_no_sequencing_the_emitter_cannot_spell() {
        // What `emit_op` writes, and the runtime's own predicates, which its
        // bodies read rather than an entry point calling them.
        let spelled = [
            "ptir_step",
            "ptir_reduce_level",
            "ptir_reduce_finish",
            "ptir_sort_seed",
            "ptir_sort_round",
            "ptir_sort_pre",
            "ptir_pivot_pack",
            "ptir_is_reduce",
            "ptir_is_sort",
        ];
        for name in RUNTIME
            .lines()
            .filter_map(|line| line.strip_prefix("fn ptir_"))
            .filter_map(|rest| rest.split('(').next())
            .map(str::trim)
        {
            let full = format!("ptir_{name}");
            assert!(
                spelled.contains(&full.as_str()),
                "the runtime declares `{full}`, which `emit_op` does not know how to \
                 sequence; a stage using it would run one `ptir_step` where a ladder \
                 was meant"
            );
        }
    }

    /// The runtime's `PTIR_WG` and the emitter's workgroup size are one
    /// number in two places, and a shader whose stripe stride is not its
    /// launch width silently skips lanes.
    #[test]
    fn the_runtime_and_the_emitter_agree_on_the_workgroup() {
        assert!(
            RUNTIME.contains(&format!("const PTIR_WG : u32 = {WORKGROUP}u;")),
            "the runtime does not declare PTIR_WG as {WORKGROUP}"
        );
    }

    /// Every tag [`emits`] claims must have a `case` in the runtime's switch,
    /// and every `case` must be claimed. A claim with no arm emits a kernel
    /// that leaves its output untouched, which is silent-wrong; an arm with
    /// no claim is dead source nobody can reach.
    #[test]
    fn every_emitted_tag_has_a_runtime_arm() {
        for row in OP_TABLE {
            let arm = format!("case {:#04X}u:", row.tag).replace("0X", "0x");
            let armed =
                RUNTIME.contains(&arm) || RUNTIME.contains(&arm.to_uppercase().replace("0X", "0x"));
            assert_eq!(
                emits(row.tag),
                armed,
                "`{}` (tag {:#04x}): emits() says {}, the runtime says {}",
                row.name,
                row.tag,
                emits(row.tag),
                armed
            );
        }
    }

    /// The reduce ladder is the one op shape the emitter spells as several
    /// calls, so the runtime has to offer all three entry points.
    #[test]
    fn the_reduce_ladder_has_its_three_entry_points() {
        for name in [
            "fn ptir_reduce_level(",
            "fn ptir_reduce_finish(",
            "fn ptir_step(",
        ] {
            assert!(RUNTIME.contains(name), "the runtime is missing `{name}`");
        }
    }

    /// `SORT_ROUNDS` is one number in two places, as the workgroup size is.
    /// It also has to be EVEN: the ladder ping-pongs between two temporary
    /// buffers and an odd count would leave the answer in the one the
    /// finishing pass does not read.
    #[test]
    fn the_runtime_and_the_emitter_agree_on_the_sort_rounds() {
        assert!(
            RUNTIME.contains(&format!("const SORT_ROUNDS : u32 = {SORT_ROUNDS}u;")),
            "the runtime does not declare SORT_ROUNDS as {SORT_ROUNDS}"
        );
        assert_eq!(
            SORT_ROUNDS % 2,
            0,
            "an odd ladder leaves the answer in the buffer the finish does not read"
        );
    }

    /// The sort family is spelled as a ladder for the reason the reduce is, so
    /// the runtime has to offer the same shapes.
    #[test]
    fn the_sort_ladder_has_its_entry_points() {
        for name in [
            "fn ptir_sort_seed(",
            "fn ptir_sort_round(",
            "fn ptir_sort_pre(",
            "fn ptir_pivot_pack(",
            "fn ptir_is_sort(",
        ] {
            assert!(RUNTIME.contains(name), "the runtime is missing `{name}`");
        }
    }

    /// The three ops read their answer off an ordered row and so carry a
    /// ladder; `matmul` is one pass and must not.
    #[test]
    fn only_the_ordered_ops_carry_a_ladder() {
        for tag in [tags::SORT_DESC, tags::TOP_K, tags::PIVOT_THRESHOLD] {
            assert!(emits(tag), "tag {tag:#04x} is not claimed");
            assert!(is_sort(tag), "tag {tag:#04x} needs the ordered row");
        }
        assert!(emits(tags::MATMUL));
        assert!(
            !is_sort(tags::MATMUL),
            "matmul sums a contraction; it orders nothing"
        );
    }

    /// The two scans are one pass, not a ladder.
    ///
    /// Their accumulator is sequential down a row, so unlike a reduce there is
    /// nothing to fold level by level and no barrier to put between levels —
    /// one invocation walks its own row. Claiming them as a ladder would emit
    /// `ptir_reduce_level` calls the runtime's reduce switch has no arm for,
    /// which is silent-wrong: the calls would do nothing and the output would
    /// keep whatever was in it.
    #[test]
    fn the_scans_are_one_pass_not_a_ladder() {
        for tag in [tags::CUMSUM, tags::CUMPROD] {
            assert!(emits(tag), "tag {tag:#04x} is not claimed");
            assert!(
                !is_reduce(tag),
                "tag {tag:#04x} walks its row; it folds no levels"
            );
            assert!(
                !is_sort(tag),
                "tag {tag:#04x} reads its row in order; it orders nothing"
            );
        }
    }

    /// The scans share one body, told apart by the identity and the combine,
    /// so the runtime offers one function for the two.
    #[test]
    fn the_runtime_has_the_scan_arm() {
        assert!(
            RUNTIME.contains("fn op_cumulative("),
            "the runtime is missing `fn op_cumulative(`"
        );
    }

    /// Every op the host interpreter evaluates has an emitted form.
    ///
    /// This is the arm's completeness claim, and it is spelled as "what is left
    /// out is exactly these four" so that a new op added to the IR and to
    /// `eval_op` fails here until the WGSL arm grows a form for it — which is
    /// the whole point of a guest pass running on the device or not at all.
    ///
    /// The four are not ops a device computes: `eta_exec::op::eval_op`'s switch
    /// has no arm for any of them. The channel three are effects, read by
    /// `eta_exec::meta`, and `intrinsic_val` is a value `eta_exec::params`
    /// resolves before a region runs.
    #[test]
    fn every_op_the_interpreter_evaluates_is_emitted() {
        let left_out: Vec<&str> = OP_TABLE
            .iter()
            .filter(|row| !emits(row.tag))
            .map(|row| row.name)
            .collect();
        assert_eq!(
            left_out,
            ["chan_take", "chan_read", "chan_put", "intrinsic_val"],
            "the emitted set is no longer every op `eval_op` evaluates"
        );
    }

    #[test]
    fn an_entry_name_that_is_not_an_identifier_is_refused() {
        assert!(valid_identifier("guest_pass_0"));
        assert!(!valid_identifier(""));
        assert!(!valid_identifier("0pass"));
        assert!(!valid_identifier("__reserved"));
        assert!(!valid_identifier("has space"));
    }

    /// A refusal says which op it could not emit — the point of refusing
    /// rather than falling back to a host interpreter.
    #[test]
    fn a_refusal_names_the_op() {
        let refused = Refused::Op {
            tag: tags::SORT_DESC,
            name: "sort_desc",
        };
        let said = format!("{refused}");
        assert!(said.contains("sort_desc"), "{said}");
        assert!(said.contains("0x50"), "{said}");
    }
}

#[cfg(test)]
mod stepwise_tests {
    use super::{RUNTIME, Refused, WORKGROUP, emit_launch_stage, emit_launch_steps};
    use crate::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
    use crate::plan::Dimension;
    use alloc::format;
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;
    use eta_ir::op::tags;
    use eta_ir::types::Dtype;

    fn value(dims: &[Dimension]) -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: Dtype::F32,
            axes: dims.to_vec(),
        }
    }

    fn op(tag: u8, result_id: u32, args: &[u32]) -> LaunchOp {
        LaunchOp {
            tag,
            result_count: 1,
            result_id,
            args: args.to_vec(),
            ..LaunchOp::default()
        }
    }

    fn plan(ops: Vec<LaunchOp>, values: Vec<LaunchPlanValue>) -> LaunchStagePlan {
        LaunchStagePlan {
            ops,
            value_types: values,
            ..LaunchStagePlan::default()
        }
    }

    /// Every `ptir_*` call in `source`, in order, with its arguments — what
    /// the shader actually asks the runtime to do.
    fn calls(source: &str) -> Vec<String> {
        source
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.starts_with("ptir_").then(|| line.to_string())
            })
            .collect()
    }

    /// **THE TWO SHAPES MUST COMPUTE THE SAME THING.** A shell picks between
    /// one dispatch for the whole region and one dispatch per step by how much
    /// of the card it wants to use — a choice about SPEED, which must not be a
    /// choice about the answer. The two emissions are separate code, so the
    /// only thing keeping them honest is reading the calls out of both and
    /// holding them side by side. A rung added to one ladder and not the other
    /// would leave a fold half-done: a wrong number, not a fault.
    #[test]
    fn the_two_shapes_sequence_the_same_calls() {
        let stage = plan(
            vec![
                op(tags::IOTA, 0, &[]),
                op(tags::EXP, 1, &[0]),
                op(tags::REDUCE_SUM, 2, &[1]),
                op(tags::SORT_DESC, 3, &[1]),
                op(tags::DIV, 4, &[1, 2]),
            ],
            (0..5)
                .map(|_| value(&[Dimension::Static(64)]))
                .collect::<Vec<_>>(),
        );

        let one = emit_launch_stage("whole", &stage).expect("the one-workgroup shape emits");
        let many = emit_launch_steps("step", &stage).expect("the stepwise shape emits");

        assert_eq!(
            calls(&one),
            calls(&many.source),
            "the two shapes ask the runtime for different work"
        );
        assert_eq!(
            many.steps.len(),
            calls(&many.source).len(),
            "every step is one dispatch of one call"
        );
    }

    /// A step's node says which op it belongs to, and a ladder's rungs all
    /// carry the node they are rungs of — so a shell can say which op a
    /// dispatch is on without a table of its own.
    #[test]
    fn a_ladder_is_many_steps_of_one_node() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::SORT_DESC, 1, &[0])],
            vec![
                value(&[Dimension::Static(64)]),
                value(&[Dimension::Static(64)]),
            ],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        let rungs = many.steps.iter().filter(|step| step.node == 1).count();
        assert_eq!(
            rungs, 10,
            "a 64-long row is seed, six rounds, pre, the step itself, and the pack"
        );
        assert_eq!(many.steps[0].node, 0, "the iota is its own single step");
        assert!(
            many.steps.iter().all(|step| step.entry.starts_with("s_")),
            "every entry point carries the prefix a shell asked for"
        );
    }

    /// **THE DISPATCH BOUNDARY IS THE ORDERING, SO NO BARRIER IS LEFT INSIDE.**
    /// A barrier in a stepwise entry point would be worse than
    /// useless: it orders one workgroup's lanes against each other while
    /// saying nothing about the other workgroups, which is exactly the false
    /// assurance this shape exists to stop making.
    #[test]
    fn a_stepwise_entry_point_carries_no_barrier() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::REDUCE_SUM, 1, &[0])],
            vec![
                value(&[Dimension::Static(64)]),
                value(&[Dimension::Static(1)]),
            ],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        let emitted = &many.source[RUNTIME.len()..];
        assert!(
            !emitted.contains("Barrier("),
            "a stepwise entry point must not pretend a barrier orders the grid"
        );
        assert!(
            emit_launch_stage("whole", &stage)
                .expect("emits")
                .contains("storageBarrier()"),
            "the one-workgroup shape still orders its steps in the shader"
        );
    }

    /// The whole point of the shape: `tid` spans the grid and `lanes` is the
    /// grid's width, so a body's stride covers its work at any dispatch size.
    #[test]
    fn a_stepwise_entry_point_strides_by_the_whole_grid() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[])],
            vec![value(&[Dimension::Static(64)])],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        let emitted = &many.source[RUNTIME.len()..];
        assert!(emitted.contains("global_invocation_id"));
        assert!(emitted.contains("num_workgroups"));
        assert!(emitted.contains("tid = gid.x;"));
        assert!(emitted.contains(&format!("lanes = grid.x * {WORKGROUP}u;")));

        let one = emit_launch_stage("whole", &stage).expect("emits");
        assert!(
            one.contains("tid = lid.x;") && one.contains(&format!("lanes = {WORKGROUP}u;")),
            "the one-workgroup shape is its own width, since a barrier reaches no further"
        );
    }

    /// An elided node is elided in BOTH shapes, or the shell would have to
    /// point `offs` two different ways depending on which it dispatched.
    #[test]
    fn neither_shape_emits_a_call_for_an_elided_node() {
        let stage = plan(
            vec![
                op(tags::IOTA, 0, &[]),
                op(tags::RESHAPE, 1, &[0]),
                op(tags::EXP, 2, &[1]),
            ],
            vec![
                value(&[Dimension::Static(4), Dimension::Static(8)]),
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
            ],
        );
        let one = emit_launch_stage("whole", &stage).expect("emits");
        let many = emit_launch_steps("s", &stage).expect("emits");
        assert!(!one.contains("ptir_step(1u)"), "the reshape reads no bytes");
        assert!(!many.source.contains("ptir_step(1u)"));
        assert!(
            many.steps.iter().all(|step| step.node != 1),
            "an elided node gets no dispatch of its own"
        );
        assert_eq!(calls(&one), calls(&many.source));
    }

    /// A prefix that is not an identifier is refused before anything is
    /// emitted, exactly as the one-workgroup shape refuses a bad entry name.
    #[test]
    fn a_stepwise_prefix_must_be_an_identifier() {
        let stage = plan(vec![], vec![]);
        assert_eq!(
            emit_launch_steps("not a name", &stage),
            Err(Refused::EntryName("not a name".to_string()))
        );
    }
}

#[cfg(test)]
mod sort_bound_tests {
    use super::{SORT_ROUNDS, sort_rounds};
    use crate::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
    use crate::plan::Dimension;
    use alloc::vec;
    use alloc::vec::Vec;
    use eta_ir::op::tags;
    use eta_ir::types::Dtype;

    fn sort_of(axes: Vec<Dimension>) -> LaunchStagePlan {
        LaunchStagePlan {
            ops: vec![
                LaunchOp {
                    tag: tags::IOTA,
                    result_count: 1,
                    result_id: 0,
                    ..LaunchOp::default()
                },
                LaunchOp {
                    tag: tags::SORT_DESC,
                    result_count: 1,
                    result_id: 1,
                    args: vec![0],
                    ..LaunchOp::default()
                },
            ],
            value_types: vec![
                LaunchPlanValue {
                    dtype: Dtype::F32,
                    axes,
                },
                LaunchPlanValue {
                    dtype: Dtype::F32,
                    axes: vec![Dimension::Static(1)],
                },
            ],
            ..LaunchStagePlan::default()
        }
    }

    /// **THE LADDER MUST NEVER BE SHORTER THAN THE ROW NEEDS.** A round short
    /// leaves the row half-merged, and a half-merged row's first element is
    /// not its largest — the sampler would then answer a token that merely won
    /// its half. So the bound is checked against the property the runtime
    /// actually relies on: after the last round the run length covers the row,
    /// and the count is even so the answer sits in buffer 0.
    #[test]
    fn the_bound_always_covers_the_row_and_keeps_the_parity() {
        for len in [0u32, 1, 2, 3, 4, 63, 64, 65, 1024, 32_000, 32_768, 262_144] {
            let rounds = sort_rounds(&sort_of(vec![Dimension::Static(len)]), 1);
            assert!(
                rounds.is_multiple_of(2),
                "a row of {len} got {rounds} rounds, so the ping-pong ends in buffer 1"
            );
            assert!(
                rounds <= SORT_ROUNDS,
                "a row of {len} asked for more than the runtime's {SORT_ROUNDS}"
            );
            assert!(
                1u64 << rounds >= u64::from(len),
                "a row of {len} got {rounds} rounds, which merges runs of only {}",
                1u64 << rounds
            );
        }
    }

    /// A 32k vocabulary row is what this is for.
    #[test]
    fn a_vocabulary_row_costs_sixteen_rounds_not_twenty_eight() {
        assert_eq!(
            sort_rounds(&sort_of(vec![Dimension::Static(32_768)]), 1),
            16
        );
    }

    /// **A VALUE WITH NO STATIC EXTENT KEEPS THE WHOLE LADDER.** A symbolic
    /// axis is a length the emitter does not know at emission and the shell
    /// fills in later, so any bound computed from it would be a guess — and
    /// the wrong guess is silent.
    #[test]
    fn a_symbolic_row_keeps_the_full_ladder() {
        let symbolic = sort_of(vec![
            Dimension::Static(4),
            Dimension::Symbolic(crate::plan::SymbolicExtent::RowCount),
        ]);
        assert_eq!(sort_rounds(&symbolic, 1), SORT_ROUNDS);
    }

    /// The bound reads the operand's total, not the result's, and a sort whose
    /// operand it cannot find keeps the full ladder rather than guessing.
    #[test]
    fn a_sort_with_no_operand_keeps_the_full_ladder() {
        let mut orphan = sort_of(vec![Dimension::Static(64)]);
        orphan.ops[1].args.clear();
        assert_eq!(sort_rounds(&orphan, 1), SORT_ROUNDS);
        assert_eq!(sort_rounds(&orphan, 99), SORT_ROUNDS);
    }
}
