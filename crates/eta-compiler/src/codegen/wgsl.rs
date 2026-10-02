use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::plan::{CompiledStage, Region};

pub const RUNTIME: &str = include_str!("../../runtime/wgsl/ptir_runtime.wgsl");

pub const WORKGROUP: u32 = 256;

const REDUCE_LEVELS: u32 = 7;

const SORT_ROUNDS: u32 = 28;

const BARRIER: &str = "  storageBarrier();\n";

pub const WGSL_EMITTER_VERSION: u16 = 4;

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

const fn is_reduce(tag: u8) -> bool {
    matches!(
        tag,
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX
    )
}

const fn is_sort(tag: u8) -> bool {
    matches!(tag, tags::SORT_DESC | tags::TOP_K | tags::PIVOT_THRESHOLD)
}

#[must_use]
pub fn is_boundary(tag: u8) -> bool {
    if tag == tags::INTRINSIC_VAL {
        return true;
    }
    eta_ir::op::spec(tag).is_some_and(|row| row.family == eta_ir::op::Family::Channel)
}

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
        let _ = writeln!(body, "  {};", step_call(at, tag));
        body.push_str(BARRIER);
        let _ = writeln!(body, "  ptir_pivot_pack({at}u);");
    } else if is_reduce(tag) {
        for level in 0..REDUCE_LEVELS {
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
        emit_op(&mut body, at, op.tag, SORT_ROUNDS);
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub entry: String,
    pub node: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stepwise {
    pub source: String,
    pub steps: Vec<Step>,
}

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
        push(alloc::format!("  {};\n", step_call(at, tag)), "step");
        push(alloc::format!("  ptir_pivot_pack({at}u);\n"), "pack");
    } else if is_reduce(tag) {
        for level in 0..REDUCE_LEVELS {
            push(
                alloc::format!("  {};\n", reduce_level_call(at, tag, level)),
                &alloc::format!("l{level}"),
            );
        }
        push(
            alloc::format!("  {};\n", reduce_finish_call(at, tag)),
            "fin",
        );
    } else {
        push(alloc::format!("  {};\n", step_call(at, tag)), "step");
    }
}

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
            ["chan_take", "chan_read", "chan_put", "intrinsic_val"],
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
    use super::{ENTRY_MARKER, RUNTIME, Refused, WORKGROUP, emit_launch_stage, emit_launch_steps};
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
            calls(&many.source),
            "the two shapes ask the runtime for different work"
        );
        assert_eq!(
            many.steps.len(),
            calls(&many.source).len(),
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
        let many_body = many.source.split(ENTRY_MARKER).nth(1).unwrap();
        assert!(
            !one_body.contains("(1u)") && !one_body.contains("(1u,"),
            "the reshape reads no bytes"
        );
        assert!(!many_body.contains("(1u)") && !many_body.contains("(1u,"));
        assert!(
            many.steps.iter().all(|step| step.node != 1),
            "an elided node gets no dispatch of its own"
        );
        assert_eq!(calls(&one), calls(&many.source));
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
