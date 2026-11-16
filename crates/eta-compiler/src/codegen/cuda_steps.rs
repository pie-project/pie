//! The closed set of calls ([`steps`]) spelled for the SIMT runtime
//! compilers (NVRTC, hiprtc), and the edges a SIMT engine runs on the
//! device beside them: a channel's read and put, and an output's rows. A
//! [`Kernel`] names no program, so an engine builds every one of
//! [`kernels`] at load and a stage's [`stage_steps`] only index them.

use alloc::string::String;
use alloc::vec::Vec;

use eta_ir::op::tags;

use crate::codegen::launch::LaunchStagePlan;
use crate::codegen::steps::{self, Call, Finish, Reduce, StepUniform};
use crate::codegen::wgsl;

/// The runtime every kernel's text starts with.
pub const RUNTIME: &str = include_str!("../../runtime/cuda/ptir_steps.cuh");

/// The threads of a kernel's block (the runtime's `PTIR_WG`).
pub const BLOCK: u32 = 256;

/// The arguments every kernel takes (`PS_ARGS`).
pub const ARITY: usize = 20;

/// What a SIMT engine runs at a boundary op, on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    /// A channel read or take: the cell into the value.
    Read,
    /// A channel put: the value into the pending cell.
    Put,
    /// An output: the lane's bound rows into the value.
    Output,
}

/// A kernel of the set: a call of the closed set, or an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Step(Call),
    Edge(Edge),
}

/// Whether a node of `tag` takes the closed set's calls here: every op
/// whose ladders all have calls, but a boundary op (an edge).
#[must_use]
pub fn stepped(tag: u8) -> bool {
    eta_ir::op::spec(tag).is_some()
        && !wgsl::is_boundary(tag)
        && steps::ladders_of(tag)
            .into_iter()
            .all(|ladder| steps::steps_of(tag, ladder).is_some())
}

/// Every kernel a stage can dispatch here, each once: the calls of every
/// stepped tag's ladders, then the edges.
#[must_use]
pub fn kernels() -> Vec<Kernel> {
    let mut all: Vec<Kernel> = steps::calls_of(stepped)
        .into_iter()
        .map(Kernel::Step)
        .collect();
    all.extend([Edge::Read, Edge::Put, Edge::Output].map(Kernel::Edge));
    all
}

const fn reduce_name(reduce: Reduce) -> &'static str {
    match reduce {
        Reduce::Sum => "sum",
        Reduce::Max => "max",
        Reduce::Min => "min",
        Reduce::Argmax => "argmax",
    }
}

const fn reduce_kind(reduce: Reduce) -> &'static str {
    match reduce {
        Reduce::Sum => "RED_SUM",
        Reduce::Max => "RED_MAX",
        Reduce::Min => "RED_MIN",
        Reduce::Argmax => "RED_ARGMAX",
    }
}

const fn finish_name(finish: Finish) -> (&'static str, u32) {
    match finish {
        Finish::SortDesc => ("sort_desc", 0),
        Finish::TopK => ("top_k", 1),
        Finish::Pivot => ("pivot", 2),
    }
}

fn guarded(name: &str, guarded: bool) -> String {
    if guarded {
        alloc::format!("{name}_guarded")
    } else {
        String::from(name)
    }
}

/// The kernel's entry point.
#[must_use]
pub fn entry(kernel: Kernel) -> String {
    let name: String = match kernel {
        Kernel::Step(call) => match call {
            Call::Unary => "unary".into(),
            Call::Arith => "arith".into(),
            Call::Compare => "compare".into(),
            Call::Logic => "logic".into(),
            Call::Select => "select".into(),
            Call::Cumulative => "cumulative".into(),
            Call::Broadcast => "broadcast".into(),
            Call::Copy => "copy".into(),
            Call::Transpose => "transpose".into(),
            Call::Gather => "gather".into(),
            Call::GatherRow => "gather_row".into(),
            Call::Scatter => "scatter".into(),
            Call::Iota => "iota".into(),
            Call::MaskApply => "mask_apply".into(),
            Call::StructMask => "struct_mask".into(),
            Call::Rng => "rng".into(),
            Call::Matmul => "matmul".into(),
            Call::Const => "const".into(),
            Call::ReduceLevel(reduce) => alloc::format!("reduce_level_{}", reduce_name(reduce)),
            Call::ReduceFinish(reduce) => {
                alloc::format!("reduce_finish_{}", reduce_name(reduce))
            }
            Call::SortSeed { guarded: g } => guarded("sort_seed", g),
            Call::SortRound { guarded: g } => guarded("sort_round", g),
            Call::SortPre { guarded: g } => guarded("sort_pre", g),
            Call::Finish { finish, guarded: g } => {
                guarded(&alloc::format!("finish_{}", finish_name(finish).0), g)
            }
            Call::PivotPack { guarded: g } => guarded("pivot_pack", g),
            Call::SelInit => "sel_init".into(),
            Call::SelHist => "sel_hist".into(),
            Call::SelPick => "sel_pick".into(),
            Call::SelRankMask => "sel_rank_mask".into(),
            Call::SelTally => "sel_tally".into(),
            Call::SelPlace => "sel_place".into(),
            Call::SelWalk => "sel_walk".into(),
        },
        Kernel::Edge(Edge::Read) => "edge_read".into(),
        Kernel::Edge(Edge::Put) => "edge_put".into(),
        Kernel::Edge(Edge::Output) => "edge_output".into(),
    };
    alloc::format!("ptir_{name}")
}

fn body(kernel: Kernel) -> String {
    let call = match kernel {
        Kernel::Edge(Edge::Read) => {
            return "  ps_edge_read(c, lane, channels, pending_flags, node);\n".into();
        }
        Kernel::Edge(Edge::Put) => {
            return "  ps_edge_put(c, lane, channels, pending_flags, commit, node);\n".into();
        }
        Kernel::Edge(Edge::Output) => {
            return "  ps_edge_output(c, ps_lane, intrinsic_bases, intrinsic_modes, \
                    intrinsic_widths, intrinsic_strides, intrinsic_offsets, pitch, commit, \
                    node);\n"
                .into();
        }
        Kernel::Step(call) => call,
    };
    let (guard, step): (bool, String) = match call {
        Call::Unary => (false, "map_unary(node, selector)".into()),
        Call::Arith => (false, "bin_arith(node, selector)".into()),
        Call::Compare => (false, "cmp_op(node, selector)".into()),
        Call::Logic => (false, "logic_op(node, selector)".into()),
        Call::Select => (false, "op_select(node)".into()),
        Call::Cumulative => (false, "op_cumulative(node, selector == 1u)".into()),
        Call::Broadcast => (false, "op_broadcast(node)".into()),
        Call::Copy => (false, "op_copy(node)".into()),
        Call::Transpose => (false, "op_transpose(node)".into()),
        Call::Gather => (false, "op_gather(node)".into()),
        Call::GatherRow => (false, "op_gather_row(node)".into()),
        Call::Scatter => (false, "op_scatter(node, selector == 1u)".into()),
        Call::Iota => (false, "op_iota(node)".into()),
        Call::MaskApply => (false, "op_mask_apply(node)".into()),
        Call::StructMask => (false, "op_struct_mask(node, selector)".into()),
        Call::Rng => (false, "op_rng(node, selector == 1u)".into()),
        Call::Matmul => (false, "op_matmul(node)".into()),
        Call::Const => (false, "op_const(node)".into()),
        Call::ReduceLevel(reduce) => (
            false,
            alloc::format!("op_reduce_level(node, {}, rung)", reduce_kind(reduce)),
        ),
        Call::ReduceFinish(reduce) => (
            false,
            alloc::format!("op_reduce_finish(node, {})", reduce_kind(reduce)),
        ),
        Call::SortSeed { guarded } => (guarded, "op_sort_seed(node)".into()),
        Call::SortRound { guarded } => (guarded, "op_sort_round(node, rung)".into()),
        Call::SortPre { guarded } => (guarded, "op_sort_pre(node)".into()),
        Call::Finish { finish, guarded } => (
            guarded,
            alloc::format!("op_finish(node, {}u)", finish_name(finish).1),
        ),
        Call::PivotPack { guarded } => (guarded, "op_pivot_pack(node)".into()),
        Call::SelInit => (false, "sel_init(node)".into()),
        Call::SelHist => (false, "sel_hist(node, rung)".into()),
        Call::SelPick => (false, "sel_pick(node, rung)".into()),
        Call::SelRankMask => (false, "sel_rank_mask(node)".into()),
        Call::SelTally => (false, "sel_tally(node)".into()),
        Call::SelPlace => (false, "sel_place(node)".into()),
        Call::SelWalk => (false, "sel_walk(node)".into()),
    };
    let guard = if guard { "  PS_GUARD;\n" } else { "" };
    alloc::format!("{guard}  c.{step};\n")
}

/// The kernel's whole text: the runtime and its entry point.
#[must_use]
pub fn source(kernel: Kernel) -> String {
    let mut source = String::with_capacity(RUNTIME.len() + 512);
    source.push_str(RUNTIME);
    source.push_str("\nextern \"C\" __global__ void __launch_bounds__(256) ");
    source.push_str(&entry(kernel));
    source.push_str("(PS_ARGS) {\n  PS_ENTER;\n");
    source.push_str(&body(kernel));
    source.push_str("}\n");
    source
}

/// One step of a stage: its kernel, and the uniform it runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub kernel: Kernel,
    pub uniform: StepUniform,
}

/// The steps a stage runs, in its order: each boundary op's edge, and each
/// launched node's ladder of calls; a reshape its source covers and a sink
/// call take none. `Err(tag)` where a node's op takes a step no kernel
/// serves.
pub fn stage_steps(plan: &LaunchStagePlan) -> Result<Vec<Step>, u8> {
    let launched = wgsl::launched_nodes(plan);
    let mut out = Vec::new();
    for (at, op) in plan.ops.iter().enumerate() {
        let node = u32::try_from(at).map_err(|_| op.tag)?;
        let edge = match op.tag {
            tags::OUTPUT => Some(Edge::Output),
            tags::CHAN_READ | tags::CHAN_TAKE => Some(Edge::Read),
            tags::CHAN_PUT => Some(Edge::Put),
            _ => None,
        };
        if let Some(edge) = edge {
            out.push(Step {
                kernel: Kernel::Edge(edge),
                uniform: StepUniform {
                    node,
                    ..StepUniform::default()
                },
            });
            continue;
        }
        if launched.binary_search(&node).is_err() {
            continue;
        }
        if !stepped(op.tag) {
            return Err(op.tag);
        }
        let ladder = steps::ladder_of(plan, at, op.tag);
        for step in steps::steps_of(op.tag, ladder).ok_or(op.tag)? {
            out.push(Step {
                kernel: Kernel::Step(step.call),
                uniform: StepUniform {
                    node,
                    rung: step.rung,
                    selector: step.selector,
                },
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_has_its_own_entry_and_its_text_names_it() {
        let all = kernels();
        assert!(all.len() > 30, "{} kernels", all.len());
        let mut names: Vec<String> = all.iter().map(|&kernel| entry(kernel)).collect();
        for (kernel, name) in all.iter().zip(&names) {
            let text = source(*kernel);
            assert!(text.contains(&alloc::format!(" {name}(PS_ARGS)")), "{name}");
        }
        names.sort();
        names.dedup();
        assert_eq!(names.len(), all.len(), "two kernels share an entry");
    }

    #[test]
    fn the_set_serves_every_stepped_tag() {
        let calls: Vec<Call> = kernels()
            .into_iter()
            .filter_map(|kernel| match kernel {
                Kernel::Step(call) => Some(call),
                Kernel::Edge(_) => None,
            })
            .collect();
        assert_eq!(steps::unlisted(&calls, stepped), None);
        for tag in [
            tags::EXP,
            tags::TOP_K,
            tags::PIVOT_THRESHOLD,
            tags::REDUCE_MAX,
        ] {
            assert!(stepped(tag), "{tag:#x}");
        }
        for tag in [tags::OUTPUT, tags::CHAN_PUT, tags::CHAN_READ] {
            assert!(!stepped(tag), "{tag:#x} is an edge");
        }
    }
}
