//! The closed set of calls ([`steps`]) spelled for the SIMT runtime
//! compilers (NVRTC, hiprtc), and the edges a SIMT engine runs on the
//! device beside them: a channel's read and put, and an output's rows. One
//! kernel ([`source`], entry [`ENTRY`]) interprets a stage: it runs the
//! stage's steps from a table ([`record`] each), the blocks of a lane
//! meeting between them, so a stage is one launch. It names no program,
//! so an engine builds it at load and a stage's [`stage_steps`] only index
//! the set ([`kernels`]).

use alloc::string::String;
use alloc::vec::Vec;

use eta_ir::op::tags;

use crate::codegen::launch::LaunchStagePlan;
use crate::codegen::steps::{self, Call, Finish, Reduce, StepUniform};
use crate::codegen::wgsl;

/// The runtime the stage kernel's text starts with.
pub const RUNTIME: &str = include_str!("../../runtime/cuda/ptir_steps.cuh");

/// The threads of a block (the runtime's `PTIR_WG`).
pub const BLOCK: u32 = 256;

/// The arguments the stage kernel takes (`PS_STAGE_ARGS`).
pub const ARITY: usize = 22;

/// The blocks of a lane the one-row select splits a row over (the
/// runtime's `SEL_GROUPS`): as many as a lane runs on.
pub const SELECT_GROUPS: u32 = 64;

/// The words the one-row select keeps just below a node's temporary area
/// (the runtime's `SEL_WORDS`): its state, each group's histogram and
/// counts, and the candidates.
pub const SELECT_WORDS: u32 = 32 + SELECT_GROUPS * 514 + 2 * 1024;

/// The stage kernel's entry point.
pub const ENTRY: &str = "ptir_stage";

/// The bytes of a step's record in a stage's table (`PsStep`).
pub const STEP_RECORD_BYTES: usize = 32;

/// A record's call word where the call runs only for a row the one-row
/// select handed the sort ladder (`PS_GUARDED`).
pub const GUARDED: u32 = 0x8000_0000;

/// A record's call word where the call is folded into the step after it
/// (`PS_SKIP`: a one-row select's pick), which runs nothing and meets none.
pub const SKIP: u32 = 0x2000_0000;

/// A record's call word where the call keeps a ladder's buffers (a sort's,
/// a reduce's), which lie in the fire's arena (`PS_ARENA`,
/// `eta_exec::tables::SteppedTables`).
pub const ARENA: u32 = 0x4000_0000;

/// What a SIMT engine runs at a boundary op, on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    /// A channel read or take: the cell into the value.
    Read,
    /// A channel put: the value into the pending cell.
    Put,
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
    all.extend([Edge::Read, Edge::Put].map(Kernel::Edge));
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

/// The kernel's name: the case its call takes in the stage kernel.
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
    };
    alloc::format!("ptir_{name}")
}

/// What a kernel runs at its step, and whether it runs only for a row the
/// one-row select handed the sort ladder.
fn call(kernel: Kernel) -> (bool, String) {
    let call = match kernel {
        Kernel::Edge(Edge::Read) => {
            return (
                false,
                "ps_edge_read(c, lane, channels, pending_flags, node)".into(),
            );
        }
        Kernel::Edge(Edge::Put) => {
            return (
                false,
                "ps_edge_put(c, lane, channels, pending_flags, commit, node)".into(),
            );
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
    (guard, alloc::format!("c.{step}"))
}

/// The interpreter's head: each block finds its lane slot and group, and
/// for each lane of the fire its slot holds, runs the stage's steps where
/// the lane's fire was admitted (its commit word) and still stands.
const STAGE_HEAD: &str = r#"
extern "C" __global__ void __launch_bounds__(256) ptir_stage(PS_STAGE_ARGS) {
  const ps_u32 slots = gridDim.x / groups;
  const ps_u32 slot = blockIdx.x / groups;
  const ps_u32 ps_group = blockIdx.x % groups;
  if (header == nullptr || slot >= slots) return;
  ps_u32* bar = barriers + 2u * slot;
  for (ps_u32 first = 0u; first < header->lane_count; first += slots) {
    const ps_u32 ps_lane = first + slot;
    if (ps_lane >= header->lane_count) break;
    const PtirLaneRecord& lane = lane_records[ps_lane];
    ps_u32* commit = reinterpret_cast<ps_u32*>(lane.commit_slot);
    if (threadIdx.x == 0u) ps_go = commit == nullptr ? 0u : *commit;
    __syncthreads();
    const bool go = ps_go != 0u;
    // Every block of the lane has read its commit before a step can clear it.
    ps_barrier(bar, groups);
    if (!go) continue;
    Ps c;
    c.descs = descs;
    c.params = params;
    c.offs = offs;
    c.heap = PsHeap{reinterpret_cast<ps_u32*>(all_scratch + (ps_u64)ps_lane * scratch_stride)};
    (void)cfg;
    c.tid = ps_group * PTIR_WG + threadIdx.x;
    c.lanes = groups * PTIR_WG;
    c.sel_lt = threadIdx.x;
    c.sel_group = ps_group;
    c.sel_groups = groups;
    c.lane = ps_lane;
    c.pitch = pitch;
    c.intrinsic_bases = intrinsic_bases;
    c.intrinsic_modes = intrinsic_modes;
    c.intrinsic_widths = intrinsic_widths;
    c.intrinsic_strides = intrinsic_strides;
    c.intrinsic_offsets = intrinsic_offsets;
    // Where this slot's arena lies, in words past the lane's scratch.
    const ps_u32 arena_words = (ps_u32)(((ps_u64)(arena + (ps_u64)slot * arena_bytes) -
                                         (ps_u64)(all_scratch + (ps_u64)ps_lane * scratch_stride)) / 4u);
    for (ps_u32 s = 0u; s < step_count; ++s) {
      const PsStep step = steps[s];
      c.temporary = step.temporary;
      // The select's ladder state was written before the last meeting, so
      // every block of the lane skips a guarded step alike.
      if ((step.call & PS_SKIP) != 0u) continue;
      if ((step.call & PS_GUARDED) != 0u && c.sel_ladder_idle()) continue;
      if ((step.call & PS_ARENA) != 0u) c.temporary = arena_words;
      const ps_u32 node = step.node;
      const ps_u32 rung = step.rung;
      const ps_u32 selector = step.selector;
      (void)rung;
      (void)selector;
      switch (step.call & ~(PS_GUARDED | PS_ARENA | PS_SKIP)) {
"#;

const STAGE_TAIL: &str = r"        default: break;
      }
      ps_barrier(bar, groups);
    }
  }
}
";

/// The stage kernel's whole text: the runtime, and the interpreter whose
/// switch spells each kernel of [`kernels`] at its index.
#[must_use]
pub fn source() -> String {
    let mut source = String::with_capacity(RUNTIME.len() + 8192);
    source.push_str(RUNTIME);
    source.push_str(STAGE_HEAD);
    for (at, kernel) in kernels().into_iter().enumerate() {
        let (_, step) = call(kernel);
        source.push_str(&alloc::format!(
            "        case {at}u: {{ {step}; break; }} // {}\n",
            entry(kernel)
        ));
    }
    source.push_str(STAGE_TAIL);
    source
}

/// One step of a stage: its kernel, and the uniform it runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub kernel: Kernel,
    pub uniform: StepUniform,
}

/// The steps a stage runs, in its order (`eta_exec::tables::stepped_tables`
/// lays its scratch over them): each channel op's edge, and each launched
/// node's ladder of calls; an output (read where it lies), a value
/// computed where it is read ([`computed_where_read`]), a reshape its
/// source covers (`wgsl_analysis::analyze_stepped`) and a sink call take
/// none. `Err(tag)` where a node's op takes a step no kernel serves.
pub fn stage_steps(plan: &LaunchStagePlan) -> Result<Vec<Step>, u8> {
    let fusion = crate::codegen::wgsl_analysis::analyze_stepped(plan);
    let computed = computed_where_read(plan);
    let mut out = Vec::new();
    for (at, op) in plan.ops.iter().enumerate() {
        let node = u32::try_from(at).map_err(|_| op.tag)?;
        if op.tag == tags::OUTPUT || computed[at] {
            continue;
        }
        let edge = match op.tag {
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
        if wgsl::is_boundary(op.tag) || !fusion.emits_node(at) {
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

/// How deep a chain of values computed where they are read may run.
const COMPUTED_DEPTH: u32 = 4;

/// Whether `tag`'s result is cheap to compute where it is read, element by
/// element: a map, an arithmetic, a select, a draw or an iota. (A broadcast
/// is, of one element only: its index otherwise walks the shapes an
/// element, which costs its reader more than the row it would lay.)
const fn computable(tag: u8) -> bool {
    matches!(
        tag,
        0x01..=0x0B
            | 0x10..=0x15
            | 0x1F
            | tags::SELECT
            | tags::RNG
            | tags::RNG_KEYED
            | tags::IOTA
            | tags::BROADCAST
    )
}

/// Whether `value` is one element, whatever a bind resolves.
fn one_element(plan: &LaunchStagePlan, value: u32) -> bool {
    plan.value_types.get(value as usize).is_some_and(|ty| {
        ty.axes
            .iter()
            .all(|axis| matches!(axis, crate::plan::Dimension::Static(1)))
    })
}

/// Whether a node of `tag` reads each element of its arguments where it
/// lies, in a way that can take a value computed there instead.
const fn reads_elementwise(tag: u8) -> bool {
    matches!(
        tag,
        0x01..=0x0B
            | 0x10..=0x1F
            | tags::SELECT
            | tags::BROADCAST
            | tags::REDUCE_SUM
            | tags::REDUCE_MAX
            | tags::REDUCE_MIN
            | tags::REDUCE_ARGMAX
            | tags::CUMSUM
            | tags::CUMPROD
            | tags::MASK_APPLY_PACKED
            | tags::CHAN_PUT
    )
}

/// Each node whose result is computed where it is read rather than laid
/// in a lane's scratch: one this stepper computes element by element
/// ([`computable`], not a bool), whose one reader reads it elementwise and
/// runs, a chain of them at most [`COMPUTED_DEPTH`] deep. Such a node takes
/// no step ([`stage_steps`]) and its value no bytes
/// (`eta_exec::tables::stepped_tables`).
#[must_use]
pub fn computed_where_read(plan: &LaunchStagePlan) -> Vec<bool> {
    let fusion = crate::codegen::wgsl_analysis::analyze_stepped(plan);
    let nodes = plan.ops.len();
    let values = plan.value_types.len();
    let mut producer = alloc::vec![u32::MAX; values];
    let mut base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        if op.result_count == 1
            && let Some(slot) = producer.get_mut(base as usize)
        {
            *slot = node as u32;
        }
        base += u32::from(op.result_count);
    }
    // A reshape the stepper elides is its source: its readers are the source's.
    let mut root: Vec<u32> = (0..values as u32).collect();
    for &(result, source) in &fusion.aliases {
        if let Some(slot) = root.get_mut(result as usize) {
            *slot = source;
        }
    }
    let mut readers = alloc::vec![0u32; values];
    let mut reader = alloc::vec![u32::MAX; values];
    for (node, op) in plan.ops.iter().enumerate() {
        if op.tag == tags::RESHAPE && !fusion.emits_node(node) {
            continue;
        }
        for value in op.reads() {
            let value = root.get(value as usize).copied().unwrap_or(value);
            if let Some(count) = readers.get_mut(value as usize) {
                *count += 1;
                reader[value as usize] = node as u32;
            }
        }
    }
    let mut computed = alloc::vec![false; nodes];
    let mut depth = alloc::vec![0u32; nodes];
    let mut base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        let result = base;
        base += u32::from(op.result_count);
        if op.result_count != 1 || !computable(op.tag) {
            continue;
        }
        if op.tag == tags::BROADCAST && !op.args.first().is_some_and(|&a| one_element(plan, a)) {
            continue;
        }
        let Some(ty) = plan.value_types.get(result as usize) else {
            continue;
        };
        if ty.dtype == eta_ir::Dtype::Bool || readers[result as usize] != 1 {
            continue;
        }
        let read_by = reader[result as usize] as usize;
        let Some(read) = plan.ops.get(read_by) else {
            continue;
        };
        if !reads_elementwise(read.tag) {
            continue;
        }
        let below = op
            .args
            .iter()
            .filter_map(|&arg| {
                let arg = root.get(arg as usize).copied().unwrap_or(arg);
                producer.get(arg as usize).copied()
            })
            .filter(|&at| at != u32::MAX && computed[at as usize])
            .map(|at| depth[at as usize])
            .max()
            .unwrap_or(0);
        if below + 1 > COMPUTED_DEPTH {
            continue;
        }
        computed[node] = true;
        depth[node] = below + 1;
    }
    computed
}

/// `step`'s record in its stage's table (`PsStep`): its kernel's index in
/// `set` ([`kernels`]), with [`GUARDED`] where it runs only for a handed
/// row, and its uniform. `None` for a kernel the set does not hold.
#[must_use]
pub fn record(step: &Step, set: &[Kernel]) -> Option<[u32; 4]> {
    let at = set.iter().position(|&kernel| kernel == step.kernel)?;
    let at = u32::try_from(at).ok()?;
    let guard = if call(step.kernel).0 { GUARDED } else { 0 };
    let arena = if matches!(
        step.kernel,
        Kernel::Step(
            Call::SortSeed { .. }
                | Call::ReduceLevel(_)
                | Call::ReduceFinish(_)
                | Call::SortRound { .. }
                | Call::SortPre { .. }
                | Call::Finish { .. }
                | Call::PivotPack { .. }
        )
    ) {
        ARENA
    } else {
        0
    };
    let skip = if step.kernel == Kernel::Step(Call::SelPick) {
        SKIP
    } else {
        0
    };
    Some([
        at | guard | arena | skip,
        step.uniform.node,
        step.uniform.rung,
        step.uniform.selector,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_runtime_and_the_layout_agree_on_the_selects_state() {
        let groups = alloc::format!("#define SEL_GROUPS {SELECT_GROUPS}u");
        let words = alloc::format!("#define SEL_WORDS {SELECT_WORDS}u");
        assert!(RUNTIME.contains(&groups) && RUNTIME.contains(&words));
    }

    #[test]
    fn the_stage_kernel_spells_every_kernel_at_its_index() {
        let all = kernels();
        assert!(all.len() > 30, "{} kernels", all.len());
        let text = source();
        for (at, kernel) in all.iter().enumerate() {
            let name = entry(*kernel);
            assert!(text.contains(&alloc::format!("case {at}u: ")), "{name}");
            assert!(text.contains(&alloc::format!("// {name}\n")), "{name}");
        }
        let mut names: Vec<String> = all.iter().map(|&kernel| entry(kernel)).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), all.len(), "two kernels share a name");
        let guarded = Step {
            kernel: Kernel::Step(Call::SortRound { guarded: true }),
            uniform: StepUniform {
                node: 3,
                rung: 2,
                selector: 0,
            },
        };
        let [word, node, rung, _] = record(&guarded, &all).expect("in the set");
        assert_eq!((word & GUARDED, node, rung), (GUARDED, 3, 2));
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
