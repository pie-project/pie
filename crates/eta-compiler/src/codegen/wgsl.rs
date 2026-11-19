use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::steps::{self, Call, Finish, Reduce, StepUniform};
pub use crate::codegen::steps::{REDUCE_LEVELS, SORT_ROUNDS, STEP_BYTES};
use crate::codegen::steps::{is_reduce, is_sort, reduce_levels, sort_rounds};
use crate::plan::{CompiledStage, Region};

pub const RUNTIME: &str = include_str!("../../runtime/wgsl/ptir_runtime.wgsl");

pub const WORKGROUP: u32 = 256;

/// Words the runtime's one-row select keeps just below the temporary area
/// (`SEL_WORDS` in the runtime), which a scratch layout must leave free.
pub const SELECT_WORDS: u32 = 10320;

/// What WGSL's kernels take of a lane's temporary area past the
/// whole-value rule (`eta_exec::stage_temporaries`): nothing, by statement.
/// Its sorts, reduction levels and pivots run over the whole value in
/// place, in at most four words an element of it (`sort_cell`, `tmp_v`,
/// `pivot_*` in the runtime: two buffers of two words), the rule's figure,
/// and its one-row select keeps its state in the [`SELECT_WORDS`] reserved
/// below the area, not in it.
pub const TEMPORARY_NEEDS: &[crate::codegen::temporary::TemporaryNeed] = &[];

const BARRIER: &str = "  storageBarrier();\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    EntryName(String),
    Op { tag: u8, name: &'static str },
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

#[must_use]
pub fn emits(tag: u8) -> bool {
    matches!(
        tag,
        tags::EXP
            | tags::LOG
            | tags::NEG
            | tags::RECIP
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
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

#[must_use]
pub fn is_boundary(tag: u8) -> bool {
    if tag == tags::OUTPUT {
        return true;
    }
    eta_ir::op::spec(tag).is_some_and(|row| row.family == eta_ir::op::Family::Channel)
}

/// The nodes of `plan` its WGSL launches, in the order it runs them: every
/// op but a boundary one (the host stages or reads its value) and a
/// reshape its source covers (its result is its source's bytes).
#[must_use]
pub fn launched_nodes(plan: &crate::codegen::launch::LaunchStagePlan) -> Vec<u32> {
    let fusion = crate::codegen::wgsl_analysis::analyze_stage(plan);
    plan.ops
        .iter()
        .enumerate()
        .filter(|&(at, op)| !is_boundary(op.tag) && fusion.emits_node(at))
        .map(|(at, _)| at as u32)
        .collect()
}

// Native drivers recompile everything an entry point reaches per pipeline, so
// naming the op cuts a stage's build from minutes to seconds; Dawn reuses the
// compile of the shared switch instead, and specialising defeats that.
fn specialize() -> bool {
    !cfg!(target_arch = "wasm32")
}

fn step_call(at: usize, tag: u8) -> String {
    if !specialize() {
        return alloc::format!("ptir_step({at}u)");
    }
    let p = alloc::format!("{at}u");
    match tag {
        0x01..=0x0B => alloc::format!("map_unary({p}, {}u)", tag - 0x01),
        0x10..=0x15 => alloc::format!("bin_arith({p}, {}u)", tag - 0x10),
        0x1F => alloc::format!("bin_arith({p}, 6u)"),
        0x16..=0x1B => alloc::format!("cmp_op({p}, {}u)", tag - 0x16),
        0x1C..=0x1E => alloc::format!("logic_op({p}, {}u)", tag - 0x1C),
        0x20 => alloc::format!("op_select({p})"),
        0x40 => alloc::format!("op_cumulative({p}, false)"),
        0x41 => alloc::format!("op_cumulative({p}, true)"),
        0x38 => alloc::format!("op_broadcast({p})"),
        0x39 | 0xA1 => alloc::format!("op_copy({p})"),
        0x3A => alloc::format!("op_transpose({p})"),
        0x60 => alloc::format!("op_gather({p})"),
        0x61 => alloc::format!("op_gather_row({p})"),
        0x62 => alloc::format!("op_scatter({p}, true)"),
        0x63 => alloc::format!("op_scatter({p}, false)"),
        0x64 => alloc::format!("op_iota({p})"),
        0x65 => alloc::format!("op_mask_apply({p})"),
        0x66..=0x68 => alloc::format!("op_struct_mask({p}, {}u)", tag - 0x66),
        0x70 => alloc::format!("op_rng({p}, false)"),
        0x71 => alloc::format!("op_rng({p}, true)"),
        0x50 => alloc::format!("op_sort_desc_finish({p})"),
        0x51 => alloc::format!("op_top_k_finish({p})"),
        0x58 => alloc::format!("op_pivot_finish({p})"),
        0x55 => alloc::format!("op_matmul({p})"),
        0x81 => alloc::format!("op_const({p})"),
        _ => alloc::format!("ptir_step({p})"),
    }
}

fn reduce_kind(tag: u8) -> Option<&'static str> {
    match tag {
        tags::REDUCE_SUM => Some("RED_SUM"),
        tags::REDUCE_MAX => Some("RED_MAX"),
        tags::REDUCE_MIN => Some("RED_MIN"),
        tags::REDUCE_ARGMAX => Some("RED_ARGMAX"),
        _ => None,
    }
}

fn reduce_level_call(at: usize, tag: u8, level: u32) -> String {
    if !specialize() {
        return alloc::format!("ptir_reduce_level({at}u, {level}u)");
    }
    match reduce_kind(tag) {
        Some(kind) => alloc::format!("op_reduce_level({at}u, {kind}, {level}u)"),
        None => alloc::format!("ptir_reduce_level({at}u, {level}u)"),
    }
}

fn reduce_finish_call(at: usize, tag: u8) -> String {
    if !specialize() {
        return alloc::format!("ptir_reduce_finish({at}u)");
    }
    match reduce_kind(tag) {
        Some(kind) => alloc::format!("op_reduce_finish({at}u, {kind})"),
        None => alloc::format!("ptir_reduce_finish({at}u)"),
    }
}

fn emit_op(body: &mut String, at: usize, tag: u8, rounds: u32, levels: u32) {
    if is_sort(tag) {
        let _ = writeln!(body, "  ptir_sort_seed({at}u);");
        body.push_str(BARRIER);
        for round in 0..rounds {
            let _ = writeln!(body, "  ptir_sort_round({at}u, {round}u);");
            body.push_str(BARRIER);
        }
        let _ = writeln!(body, "  ptir_sort_pre({at}u);");
        body.push_str(BARRIER);
        let _ = writeln!(body, "  {};", step_call(at, tag));
        body.push_str(BARRIER);
        let _ = writeln!(body, "  ptir_pivot_pack({at}u);");
    } else if is_reduce(tag) {
        for level in 0..levels {
            let _ = writeln!(body, "  {};", reduce_level_call(at, tag, level));
            body.push_str(BARRIER);
        }
        let _ = writeln!(body, "  {};", reduce_finish_call(at, tag));
    } else {
        let _ = writeln!(body, "  {};", step_call(at, tag));
    }
    body.push_str(BARRIER);
}

pub const ENTRY_MARKER: &str = "// ---- the entry point ----";

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
    source.push_str("  sel_lt = lid.x;\n  sel_group = 0u;\n  sel_groups = 1u;\n");
    source.push_str(body);
    source.push_str("}\n");
    source
}

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
        emit_op(&mut body, at, op.tag, SORT_ROUNDS, REDUCE_LEVELS);
    }

    Ok(wrap(entry_name, &body))
}

pub fn emit_launch_stage(
    entry_name: &str,
    plan: &crate::codegen::launch::LaunchStagePlan,
) -> Result<String, Refused> {
    if !valid_identifier(entry_name) {
        return Err(Refused::EntryName(entry_name.to_string()));
    }
    let mut body = String::new();
    for at in launched_nodes(plan) {
        let at = at as usize;
        let op = &plan.ops[at];
        if !emits(op.tag) {
            return Err(Refused::Op {
                tag: op.tag,
                name: eta_ir::op::spec(op.tag).map_or("?", |row| row.name),
            });
        }
        emit_op(
            &mut body,
            at,
            op.tag,
            sort_rounds(plan, at),
            reduce_levels(plan, at),
        );
    }
    Ok(wrap(entry_name, &body))
}

/// One dispatch of a stepwise stage: call `call` of [`step_calls`] (the
/// closed set, [`steps`]) on node `node` at `rung`, with `selector`. The
/// call reads all three from the step's uniform ([`STEP_BINDING`]), so its
/// pipeline names no program: every pipeline a stage can dispatch is one of
/// [`step_calls`], and a backend builds them all before it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// What the step is, for a reader: the prefix, the node and the rung.
    pub entry: String,
    pub node: u32,
    /// Its call's index in [`step_calls`].
    pub call: u32,
    pub rung: u32,
    pub selector: u32,
}

impl Step {
    /// What the step's uniform holds.
    #[must_use]
    pub fn uniform(&self) -> StepUniform {
        StepUniform {
            node: self.node,
            rung: self.rung,
            selector: self.selector,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stepwise {
    pub steps: Vec<Step>,
}

/// The version of what the stepwise emitter writes ([`step_calls`],
/// [`step_module`], [`emit_launch_steps`]): it moves with the emitted bytes,
/// pinned beside the other emitters' in `tests/eta_compiler/emitter_version.rs`.
/// 1: every step a call of the closed set, reading its node, rung and
/// selector from a uniform; before, each stage one module with its nodes
/// spelled in.
/// 3: every heap, descriptor, offset and op-param access is bounded by its
/// stage's cfg (the heap's words, its values, its ops), indices read from
/// the heap by their own bounds, and a refused access writes the runtimes'
/// fault status (state 3, `OUT_OF_BOUNDS`, the index, the limit).
/// 4: a step reads its tables, status and heap at the bases its record
/// states, in the two pool segments a backend binds whole (the instance's
/// slots, its heap), and its config from its config table: one bind group
/// serves every instance whose slots and heap share two segments.
pub const WGSL_STEP_EMITTER_VERSION: u16 = 4;

/// The binding of the record a step module reads its node, rung, selector
/// and bases from ([`StepUniform`]), beside the runtime's (0, 1 and 4): a
/// backend binds each step's at its own dynamic offset.
pub const STEP_BINDING: u32 = 6;

const NODE: &str = "pie_step.node";
const RUNG: &str = "pie_step.rung";
const SELECTOR: &str = "pie_step.selector";

/// The ops this backend steps: every one it emits but a boundary op.
fn stepped(tag: u8) -> bool {
    emits(tag) && !is_boundary(tag)
}

/// The closed set's calls, in [`step_calls`]' order.
fn calls() -> Vec<Call> {
    steps::calls_of(stepped)
}

fn reduce_name(reduce: Reduce) -> &'static str {
    match reduce {
        Reduce::Sum => "RED_SUM",
        Reduce::Max => "RED_MAX",
        Reduce::Min => "RED_MIN",
        Reduce::Argmax => "RED_ARGMAX",
    }
}

/// `call` in WGSL: the runtime function it runs, on the uniform's node,
/// rung and selector. Where the ops are not named (Dawn, which reuses the
/// compile of the runtime's switch), a call that runs one op's step runs
/// the switch instead.
fn call_text(call: Call) -> String {
    let (p, k, j) = (NODE, RUNG, SELECTOR);
    let flag = |name: &str| alloc::format!("{name}({p}, {j} != 0u)");
    let named = specialize();
    let body = match call {
        Call::ReduceLevel(reduce) if named => {
            alloc::format!("op_reduce_level({p}, {}, {k})", reduce_name(reduce))
        }
        Call::ReduceLevel(_) => alloc::format!("ptir_reduce_level({p}, {k})"),
        Call::ReduceFinish(reduce) if named => {
            alloc::format!("op_reduce_finish({p}, {})", reduce_name(reduce))
        }
        Call::ReduceFinish(_) => alloc::format!("ptir_reduce_finish({p})"),
        Call::SortSeed { .. } => alloc::format!("ptir_sort_seed({p})"),
        Call::SortRound { .. } => alloc::format!("ptir_sort_round({p}, {k})"),
        Call::SortPre { .. } => alloc::format!("ptir_sort_pre({p})"),
        Call::PivotPack { .. } => alloc::format!("ptir_pivot_pack({p})"),
        Call::SelInit => alloc::format!("sel_init({p})"),
        Call::SelHist => alloc::format!("sel_hist({p}, {k})"),
        Call::SelPick => alloc::format!("sel_pick({p}, {k})"),
        Call::SelRankMask => alloc::format!("sel_rank_mask({p})"),
        Call::SelTally => alloc::format!("sel_tally({p})"),
        Call::SelPlace => alloc::format!("sel_place({p})"),
        Call::SelWalk => alloc::format!("sel_walk({p})"),
        _ if !named => alloc::format!("ptir_step({p})"),
        Call::Unary => alloc::format!("map_unary({p}, {j})"),
        Call::Arith => alloc::format!("bin_arith({p}, {j})"),
        Call::Compare => alloc::format!("cmp_op({p}, {j})"),
        Call::Logic => alloc::format!("logic_op({p}, {j})"),
        Call::Select => alloc::format!("op_select({p})"),
        Call::Cumulative => flag("op_cumulative"),
        Call::Broadcast => alloc::format!("op_broadcast({p})"),
        Call::Copy => alloc::format!("op_copy({p})"),
        Call::Transpose => alloc::format!("op_transpose({p})"),
        Call::Gather => alloc::format!("op_gather({p})"),
        Call::GatherRow => alloc::format!("op_gather_row({p})"),
        Call::Scatter => flag("op_scatter"),
        Call::Iota => alloc::format!("op_iota({p})"),
        Call::MaskApply => alloc::format!("op_mask_apply({p})"),
        Call::StructMask => alloc::format!("op_struct_mask({p}, {j})"),
        Call::Rng => flag("op_rng"),
        Call::Matmul => alloc::format!("op_matmul({p})"),
        Call::Const => alloc::format!("op_const({p})"),
        Call::Finish { finish, .. } => match finish {
            Finish::SortDesc => alloc::format!("op_sort_desc_finish({p})"),
            Finish::TopK => alloc::format!("op_top_k_finish({p})"),
            Finish::Pivot => alloc::format!("op_pivot_finish({p})"),
        },
    };
    let guarded = matches!(
        call,
        Call::SortSeed { guarded: true }
            | Call::SortRound { guarded: true }
            | Call::SortPre { guarded: true }
            | Call::Finish { guarded: true, .. }
            | Call::PivotPack { guarded: true }
    );
    if guarded {
        // The ladder runs only for a row the select handed over.
        alloc::format!("  if (sel_ladder_idle()) {{ return; }}\n  {body};\n")
    } else {
        alloc::format!("  {body};\n")
    }
}

/// Every call a stepwise stage can dispatch, each once, in WGSL: the steps
/// of every ladder of every op this backend emits ([`steps::calls_of`]). A
/// stage's steps index it, so the pipelines a shell builds from it are
/// every one a guest program can run.
#[must_use]
pub fn step_calls() -> Vec<String> {
    calls().into_iter().map(call_text).collect()
}

/// The module of call `call` of [`step_calls`]: the runtime, the step's
/// uniform, and one entry point, `entry`, that runs the call across the
/// whole grid.
#[must_use]
pub fn step_module(entry: &str, call: &str) -> String {
    let mut source = String::with_capacity(RUNTIME.len() + 1024);
    source.push_str(RUNTIME);
    source.push('\n');
    // The words [`StepUniform::bytes`] lays.
    source.push_str(
        "struct PieStep {\n  node : u32,\n  rung : u32,\n  selector : u32,\n  pad : u32,\n  \
         status_base : u32,\n  descs_base : u32,\n  params_base : u32,\n  offs_base : u32,\n  \
         cfg_base : u32,\n  heap_base : u32,\n  pad1 : u32,\n  pad2 : u32,\n}\n",
    );
    let _ = writeln!(
        source,
        "@group(0) @binding({STEP_BINDING}) var<uniform> pie_step : PieStep;"
    );
    source.push_str(ENTRY_MARKER);
    source.push('\n');
    wrap_step(&mut source, entry, call);
    source
}

/// `call` as one step's own text: what it runs, spelled with its node,
/// rung and selector in place of the uniform's reads.
#[must_use]
pub fn spelled(call: &str, step: &Step) -> String {
    let flag = if step.selector != 0 { "true" } else { "false" };
    call.replace(NODE, &alloc::format!("{}u", step.node))
        .replace(&alloc::format!("{SELECTOR} != 0u"), flag)
        .replace(SELECTOR, &alloc::format!("{}u", step.selector))
        .replace(RUNG, &alloc::format!("{}u", step.rung))
}

fn wrap_step(source: &mut String, entry_name: &str, call: &str) {
    source.push_str("\n@compute @workgroup_size(");
    let _ = write!(source, "{WORKGROUP}");
    source.push_str(")\nfn ");
    source.push_str(entry_name);
    source.push_str(
        "(@builtin(global_invocation_id) gid : vec3<u32>,\n   \
         @builtin(local_invocation_id) lid : vec3<u32>,\n   \
         @builtin(workgroup_id) wid : vec3<u32>,\n   \
         @builtin(num_workgroups) grid : vec3<u32>) {\n  tid = gid.x;\n",
    );
    let _ = writeln!(source, "  lanes = grid.x * {WORKGROUP}u;");
    source.push_str("  sel_lt = lid.x;\n  sel_group = wid.x;\n  sel_groups = grid.x;\n");
    // The step's bases, its heap bound ([`RUNTIME`]'s `hld` / `hst`) and
    // its temporaries' first word, read once.
    source.push_str(
        "  b_status = pie_step.status_base;\n  b_descs = pie_step.descs_base;\n  \
         b_params = pie_step.params_base;\n  b_offs = pie_step.offs_base;\n  \
         b_heap = pie_step.heap_base;\n  \
         lim_heap = tables[pie_step.cfg_base + CFG_HEAP_WORDS];\n  \
         cfg_temporary = tables[pie_step.cfg_base + CFG_TEMPORARY];\n",
    );
    source.push_str(call);
    source.push_str("  heap_flush();\n}\n");
}

/// `plan`'s launches as steps of the closed set; a node no call serves is
/// refused by its op's name.
pub fn emit_launch_steps(
    prefix: &str,
    plan: &crate::codegen::launch::LaunchStagePlan,
) -> Result<Stepwise, Refused> {
    if !valid_identifier(prefix) {
        return Err(Refused::EntryName(prefix.to_string()));
    }
    let calls = calls();
    let mut out = Vec::new();
    for at in launched_nodes(plan) {
        let at = at as usize;
        let op = &plan.ops[at];
        let refused = || Refused::Op {
            tag: op.tag,
            name: eta_ir::op::spec(op.tag).map_or("?", |row| row.name),
        };
        if !emits(op.tag) {
            return Err(refused());
        }
        let ladder = steps::ladder_of(plan, at, op.tag);
        for step in steps::steps_of(op.tag, ladder).ok_or_else(refused)? {
            let index = calls
                .iter()
                .position(|listed| *listed == step.call)
                .ok_or_else(refused)?;
            out.push(Step {
                entry: alloc::format!("{prefix}_{at}_{}", step.suffix),
                node: at as u32,
                call: index as u32,
                rung: step.rung,
                selector: step.selector,
            });
        }
    }
    Ok(Stepwise { steps: out })
}

#[cfg(test)]
mod tests {
    use super::{
        RUNTIME, Refused, SORT_ROUNDS, WORKGROUP, emits, is_reduce, is_sort, valid_identifier,
    };
    use alloc::format;
    use alloc::vec::Vec;
    use eta_ir::op::{OP_TABLE, tags};

    #[test]
    fn wgsl_every_case() {
        the_runtime_and_the_emitter_agree_on_the_ladders_height();
        the_runtime_grew_no_sequencing_the_emitter_cannot_spell();
        every_arm_of_the_runtime_switch_is_spelled_directly();
        the_runtime_and_the_emitter_agree_on_the_workgroup();
        every_emitted_tag_has_a_runtime_arm();
        the_reduce_ladder_has_its_three_entry_points();
        the_runtime_and_the_emitter_agree_on_the_sort_rounds();
        the_sort_ladder_has_its_entry_points();
        only_the_ordered_ops_carry_a_ladder();
        the_scans_are_one_pass_not_a_ladder();
        the_runtime_has_the_scan_arm();
        every_op_the_interpreter_evaluates_is_emitted();
        an_entry_name_that_is_not_an_identifier_is_refused();
        a_refusal_names_the_op();
    }

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

    fn every_arm_of_the_runtime_switch_is_spelled_directly() {
        if !super::specialize() {
            return;
        }
        let switch = RUNTIME
            .split("fn ptir_step(p : u32) {")
            .nth(1)
            .and_then(|rest| rest.split("\n}\n").next())
            .expect("the runtime defines `ptir_step`");
        let mut arms = 0;
        for line in switch.lines() {
            let Some(rest) = line.trim().strip_prefix("case 0x") else {
                continue;
            };
            let (hex, call) = rest.split_once("u: {").expect("a switch arm");
            let tag = u8::from_str_radix(hex, 16).expect("a tag");
            let call = call
                .trim()
                .trim_end_matches('}')
                .trim()
                .trim_end_matches(';');
            let spelled = super::step_call(7, tag);
            let expected = call.replace("(p,", "(7u,").replace("(p)", "(7u)");
            if call.is_empty() {
                assert_eq!(spelled, "ptir_step(7u)", "tag {tag:#x} does nothing");
            } else {
                assert_eq!(spelled, expected, "tag {tag:#x}");
            }
            arms += 1;
        }
        assert!(arms > 40, "the switch has {arms} arms");
        for tag in [
            tags::REDUCE_SUM,
            tags::REDUCE_MAX,
            tags::REDUCE_MIN,
            tags::REDUCE_ARGMAX,
        ] {
            assert!(super::reduce_kind(tag).is_some());
        }
    }

    fn the_runtime_grew_no_sequencing_the_emitter_cannot_spell() {
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

    fn the_runtime_and_the_emitter_agree_on_the_workgroup() {
        assert!(
            RUNTIME.contains(&format!("const PTIR_WG : u32 = {WORKGROUP}u;")),
            "the runtime does not declare PTIR_WG as {WORKGROUP}"
        );
    }

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

    fn the_reduce_ladder_has_its_three_entry_points() {
        for name in [
            "fn ptir_reduce_level(",
            "fn ptir_reduce_finish(",
            "fn ptir_step(",
        ] {
            assert!(RUNTIME.contains(name), "the runtime is missing `{name}`");
        }
    }

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

    fn the_runtime_has_the_scan_arm() {
        assert!(
            RUNTIME.contains("fn op_cumulative("),
            "the runtime is missing `fn op_cumulative(`"
        );
    }

    fn every_op_the_interpreter_evaluates_is_emitted() {
        let left_out: Vec<&str> = OP_TABLE
            .iter()
            .filter(|row| !emits(row.tag))
            .map(|row| row.name)
            .collect();
        assert_eq!(
            left_out,
            ["chan_take", "chan_read", "chan_put", "output"],
            "the emitted set is no longer every op `eval_op` evaluates"
        );
    }

    fn an_entry_name_that_is_not_an_identifier_is_refused() {
        assert!(valid_identifier("guest_pass_0"));
        assert!(!valid_identifier(""));
        assert!(!valid_identifier("0pass"));
        assert!(!valid_identifier("__reserved"));
        assert!(!valid_identifier("has space"));
    }

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
    use super::{
        ENTRY_MARKER, RUNTIME, Refused, Stepwise, WORKGROUP, emit_launch_stage, emit_launch_steps,
        spelled, step_calls, step_module,
    };
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

    fn calls(source: &str) -> Vec<String> {
        source
            .split_once(ENTRY_MARKER)
            .map_or(source, |(_, after)| after)
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                [
                    "ptir_",
                    "op_",
                    "map_unary(",
                    "bin_arith(",
                    "cmp_op(",
                    "logic_op(",
                ]
                .iter()
                .any(|head| line.starts_with(head))
                .then(|| line.to_string())
            })
            .collect()
    }

    /// What `many`'s steps run, each with its node and rung spelled in.
    fn run(many: &Stepwise) -> String {
        let calls = step_calls();
        many.steps
            .iter()
            .map(|step| spelled(&calls[step.call as usize], step))
            .collect()
    }

    /// Every op the guest IR carries is refused at admission (this backend
    /// emits no WGSL for it), a boundary the host serves, or has a call of
    /// the closed set for every step of every ladder it can take; where the
    /// ops are named, none of those calls is the runtime's whole switch, a
    /// driver's seconds for every op at once. The negative control: a set
    /// short of any one call fails.
    #[test]
    fn the_closed_set_holds_every_step_of_every_op() {
        use eta_ir::op::OP_TABLE;
        let calls = super::calls();
        assert_eq!(super::steps::unlisted(&calls, super::stepped), None);
        let stepped = OP_TABLE
            .iter()
            .filter(|row| super::stepped(row.tag))
            .count();
        assert!(stepped > 40, "{stepped} ops take steps");
        if super::specialize() {
            for call in step_calls() {
                assert!(
                    !call.contains("ptir_step("),
                    "`{}` is the whole switch",
                    call.trim()
                );
            }
        }
        for gone in 0..calls.len() {
            let mut short = calls.clone();
            let call = short.remove(gone);
            assert!(
                super::steps::unlisted(&short, super::stepped).is_some(),
                "the set without {call:?} still passes"
            );
        }
    }

    /// A sink call is the engine's: its node takes no dispatch.
    #[test]
    fn a_sink_call_takes_no_step() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::SINK_CALL, 1, &[0])],
            vec![
                value(&[Dimension::Static(64)]),
                value(&[Dimension::Static(64)]),
            ],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        assert!(
            many.steps.iter().all(|step| step.node != 1),
            "{:?}",
            many.steps
        );
    }

    #[test]
    fn wgsl_1_every_case() {
        the_two_shapes_sequence_the_same_calls();
        a_ladder_is_many_steps_of_one_node();
        a_stepwise_entry_point_carries_no_barrier();
        a_stepwise_entry_point_strides_by_the_whole_grid();
        neither_shape_emits_a_call_for_an_elided_node();
        a_stepwise_prefix_must_be_an_identifier();
    }

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
            calls(&run(&many)),
            "the two shapes ask the runtime for different work"
        );
        assert_eq!(
            many.steps.len(),
            calls(&run(&many)).len(),
            "every step is one dispatch of one call"
        );
    }

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

    fn a_stepwise_entry_point_carries_no_barrier() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::REDUCE_SUM, 1, &[0])],
            vec![
                value(&[Dimension::Static(64)]),
                value(&[Dimension::Static(1)]),
            ],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        let calls = step_calls();
        for step in &many.steps {
            let module = step_module("s", &calls[step.call as usize]);
            assert!(
                !module[RUNTIME.len()..].contains("Barrier("),
                "a stepwise entry point must not pretend a barrier orders the grid"
            );
        }
        assert!(
            emit_launch_stage("whole", &stage)
                .expect("emits")
                .contains("storageBarrier()"),
            "the one-workgroup shape still orders its steps in the shader"
        );
    }

    /// The runtime's temporaries are what [`TEMPORARY_NEEDS`] states of
    /// them: two buffers of two words an element of the value a step runs
    /// over (four words, the whole-value rule's figure), and the select's
    /// state below the area.
    #[test]
    fn the_runtimes_temporaries_lie_within_the_whole_value_rule() {
        assert!(super::TEMPORARY_NEEDS.is_empty());
        for cell in [
            "return cfg_temporary + ((buf * n) + c) * 2u;",
            "return cfg_temporary + (buf * span + row * stride + c) * 2u;",
            "fn pivot_flag(n : u32, t : u32) -> u32 { return cfg_temporary + t; }",
            "fn pivot_meta(n : u32, row : u32) -> u32 { return cfg_temporary + n + row; }",
            "fn pivot_val(n : u32, t : u32) -> u32 { return cfg_temporary + 2u * n + t; }",
            "fn pivot_pos(n : u32, i : u32) -> u32 { return cfg_temporary + 3u * n + i; }",
            "fn sel_at(k : u32) -> u32 { return cfg_temporary - SEL_WORDS + k; }",
        ] {
            assert!(RUNTIME.contains(cell), "the runtime no longer lays `{cell}`");
        }
        let named = RUNTIME.matches("return cfg_temporary").count();
        assert_eq!(named, 7, "a temporary cell this statement has not read");
        assert!(RUNTIME.contains(&alloc::format!("const SEL_WORDS : u32 = {}u;", super::SELECT_WORDS)));
    }

    fn a_stepwise_entry_point_strides_by_the_whole_grid() {
        let stage = plan(
            vec![op(tags::IOTA, 0, &[])],
            vec![value(&[Dimension::Static(64)])],
        );
        let many = emit_launch_steps("s", &stage).expect("emits");
        let module = step_module("s", &step_calls()[many.steps[0].call as usize]);
        let emitted = &module[RUNTIME.len()..];
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
        let one_body = one.split(ENTRY_MARKER).nth(1).unwrap();
        let many_body = run(&many);
        assert!(
            !one_body.contains("(1u)") && !one_body.contains("(1u,"),
            "the reshape reads no bytes"
        );
        assert!(!many_body.contains("(1u)") && !many_body.contains("(1u,"));
        assert!(
            many.steps.iter().all(|step| step.node != 1),
            "an elided node gets no dispatch of its own"
        );
        assert_eq!(calls(&one), calls(&run(&many)));
    }

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

    #[test]
    fn wgsl_2_every_case() {
        the_bound_always_covers_the_row_and_keeps_the_parity();
        a_vocabulary_row_costs_sixteen_rounds_not_twenty_eight();
        a_symbolic_row_keeps_the_full_ladder();
        a_sort_with_no_operand_keeps_the_full_ladder();
    }

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

    fn a_vocabulary_row_costs_sixteen_rounds_not_twenty_eight() {
        assert_eq!(
            sort_rounds(&sort_of(vec![Dimension::Static(32_768)]), 1),
            16
        );
    }

    fn a_symbolic_row_keeps_the_full_ladder() {
        let symbolic = sort_of(vec![
            Dimension::Static(4),
            Dimension::Symbolic(crate::plan::SymbolicExtent::RowCount),
        ]);
        assert_eq!(sort_rounds(&symbolic, 1), SORT_ROUNDS);
    }

    fn a_sort_with_no_operand_keeps_the_full_ladder() {
        let mut orphan = sort_of(vec![Dimension::Static(64)]);
        orphan.ops[1].args.clear();
        assert_eq!(sort_rounds(&orphan, 1), SORT_ROUNDS);
        assert_eq!(sort_rounds(&orphan, 99), SORT_ROUNDS);
    }
}

#[cfg(test)]
mod select_tests {
    use super::{RUNTIME, SELECT_WORDS, emit_launch_steps, spelled, step_calls};
    use crate::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
    use crate::plan::Dimension;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use eta_ir::op::tags;
    use eta_ir::types::Dtype;

    fn pivot(pred_tag: u8, input: &[Dimension]) -> LaunchStagePlan {
        let value = |axes: &[Dimension]| LaunchPlanValue {
            dtype: Dtype::F32,
            axes: axes.to_vec(),
        };
        LaunchStagePlan {
            ops: vec![LaunchOp {
                tag: tags::PIVOT_THRESHOLD,
                result_count: 1,
                result_id: 2,
                pred_tag,
                args: vec![0, 1],
                ..LaunchOp::default()
            }],
            value_types: vec![value(input), value(&[Dimension::Static(1)]), value(input)],
            ..LaunchStagePlan::default()
        }
    }

    fn suffixes(plan: &LaunchStagePlan) -> Vec<String> {
        emit_launch_steps("main", plan)
            .expect("a pivot emits")
            .steps
            .into_iter()
            .map(|step| step.entry.trim_start_matches("main_0_").into())
            .collect()
    }

    const ROW: [Dimension; 2] = [Dimension::Static(1), Dimension::Static(248_320)];

    #[test]
    fn a_one_row_rank_cut_selects_and_masks_without_a_sort() {
        assert_eq!(
            suffixes(&pivot(0, &ROW)),
            [
                "sel", "h0", "p0", "h1", "p1", "h2", "p2", "h3", "p3", "mask"
            ]
        );
    }

    #[test]
    fn a_one_row_mass_cut_walks_its_candidates_and_keeps_the_ladder_behind() {
        let steps = suffixes(&pivot(1, &ROW));
        assert_eq!(
            steps[..12],
            [
                "sel", "h0", "p0", "h1", "p1", "h2", "p2", "h3", "p3", "tally", "place", "walk"
            ]
        );
        assert_eq!(steps[12], "seed");
        assert_eq!(steps.last().map(String::as_str), Some("pack"));
        let calls = step_calls();
        let source: String = emit_launch_steps("main", &pivot(1, &ROW))
            .unwrap()
            .steps
            .iter()
            .map(|step| spelled(&calls[step.call as usize], step))
            .collect();
        assert!(
            source.contains("if (sel_ladder_idle()) { return; }\n  ptir_sort_seed(0u);"),
            "the ladder runs only for a row the select handed over"
        );
    }

    #[test]
    fn a_threshold_cut_is_one_compare() {
        assert_eq!(suffixes(&pivot(2, &ROW)), ["step"]);
        let rows = [Dimension::Static(4), Dimension::Static(1024)];
        assert_eq!(suffixes(&pivot(2, &rows)), ["step"]);
    }

    #[test]
    fn several_rows_keep_the_sort_ladder() {
        let rows = [Dimension::Static(4), Dimension::Static(1024)];
        let steps = suffixes(&pivot(1, &rows));
        assert_eq!(steps.first().map(String::as_str), Some("seed"));
        assert!(!steps.iter().any(|step| step.starts_with('h')));
    }

    #[test]
    fn the_layout_reserves_what_the_runtime_addresses() {
        let declared = alloc::format!("const SEL_WORDS : u32 = {SELECT_WORDS}u;");
        assert!(
            RUNTIME.contains(&declared),
            "the runtime's SEL_WORDS is not SELECT_WORDS"
        );
        // State, a histogram per workgroup, a tally pair per workgroup, and
        // the candidates' keys and indices.
        assert_eq!(SELECT_WORDS, 16 + 32 * 256 + 32 * 2 + 1024 * 2);
    }
}
