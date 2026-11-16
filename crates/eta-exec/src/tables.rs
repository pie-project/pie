//! A guest stage's bind-time tables, as every backend lays them: each
//! value's descriptor ([`desc_words`], [`DESC_WORDS`] u32s), each op's
//! params ([`OpParams`], `PARAM_WORDS` u32s), each value's word offset in
//! the stage's scratch, and the stage's config ([`Cfg`]). A step of the
//! closed set (`eta_compiler::codegen::steps`) finds its node's operands
//! here: the node's params name its argument and result values, whose
//! descriptors and offsets these tables hold. [`stage_tables`] writes them
//! at a bind, against the bind's extents; [`table_sizes`] is what they
//! take, fixed by the stage's value and op counts.

use eta_compiler::codegen::launch::{LaunchRegion, LaunchStagePlan};
use eta_compiler::codegen::{steps, wgsl, wgsl_analysis};
use eta_ir::op::tags;

use crate::{
    Extents, Launch, Launches, Lifetime, OpParams, SELECT_RESERVE, ValueDesc, describe_launched,
    lay_out, layout_reusing, value_lifetimes,
};

pub const DESC_WORDS: usize = 9;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cfg {
    pub value_count: u32,
    pub temporary: u32,
    pub op_count: u32,
    pub lane: u32,
}

const _: () = assert!(size_of::<Cfg>() == 16);

pub fn desc_words(d: &ValueDesc) -> [u32; DESC_WORDS] {
    let mut out = [0u32; DESC_WORDS];
    out[0] = d.len;
    out[1] = d.rows;
    out[2] = d.last;
    out[3] = d.rank;
    out[4] = d.dtype;
    for (at, &dim) in d.dims.iter().enumerate() {
        out[5 + at] = dim;
    }
    out
}

pub fn words_le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

impl Cfg {
    #[must_use]
    pub fn to_le_bytes(self) -> Vec<u8> {
        words_le(&[self.value_count, self.temporary, self.op_count, self.lane])
    }
}

/// How a stage runs here, which its scratch is laid over: the emitted
/// WGSL launches one op after another in the stage's order, each alone in
/// its launch; the host writes a root's bytes before the first and reads
/// a put's after the last.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stepping {
    /// The nodes the WGSL launches ([`wgsl::launched_nodes`]).
    pub launched: Vec<u32>,
    /// The other nodes: boundary ops and reshapes their source covers.
    pub idle: Vec<u32>,
    /// What the host may write before the first launch: an output's, a
    /// channel read's, a take's and a constant's result.
    pub staged: Vec<u32>,
    /// What is read after the last launch: every value a boundary op reads.
    pub kept: Vec<u32>,
    /// The results of the reshapes not launched: their source's bytes
    /// ([`alias_reshapes`]).
    pub hollow: Vec<u32>,
    /// A launched node takes the one-row select's ladder, whose state lies
    /// just below the temporary area (`SELECT_RESERVE`).
    pub selects: bool,
    /// What each launched node that keeps state in the temporary area
    /// keeps there: the value it orders or folds, and how.
    pub temporaries: Vec<(u32, Temporary)>,
}

/// What a launch keeps in the temporary area for the value it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Temporary {
    /// A sort's ladder: two key-and-index buffers of the value, which a
    /// pivot's cut reuses (four words an element).
    Sort,
    /// A reduce's ladder: two buffers of each row's 32-fold partials, a
    /// value and an index each.
    Reduce,
}

impl Temporary {
    /// The bytes it keeps for a value `d` describes.
    #[must_use]
    pub fn bytes(self, d: &ValueDesc) -> u64 {
        match self {
            Temporary::Sort => 16 * u64::from(d.len),
            Temporary::Reduce => {
                let rows = rows_of(d);
                let len = u64::from(d.len) / rows.max(1);
                16 * rows * len.div_ceil(32)
            }
        }
    }
}

/// A value's rows as the runtime counts them: every axis but the last.
fn rows_of(d: &ValueDesc) -> u64 {
    if d.rank < 2 {
        return 1;
    }
    d.dims
        .iter()
        .take(d.rank as usize - 1)
        .map(|&dim| u64::from(dim))
        .product()
}

impl Stepping {
    #[must_use]
    pub fn of(plan: &LaunchStagePlan) -> Stepping {
        let launched = wgsl::launched_nodes(plan);
        let mut stepping = Stepping::default();
        let mut result_base = 0u32;
        for (node, op) in plan.ops.iter().enumerate() {
            let results = result_base..result_base.saturating_add(u32::from(op.result_count));
            result_base = results.end;
            if matches!(
                op.tag,
                tags::OUTPUT | tags::CHAN_READ | tags::CHAN_TAKE | tags::CONST
            ) {
                stepping.staged.extend(results);
            }
            let node = node as u32;
            if launched.binary_search(&node).is_ok() {
                continue;
            }
            stepping.idle.push(node);
            if wgsl::is_boundary(op.tag) {
                stepping.kept.extend(op.reads());
            }
        }
        stepping.hollow = wgsl_analysis::analyze_stage(plan)
            .aliases
            .into_iter()
            .map(|(result, _)| result)
            .collect();
        for &node in &launched {
            let at = node as usize;
            let op = &plan.ops[at];
            let ladder = steps::ladder_of(plan, at, op.tag);
            stepping.selects |= matches!(ladder, steps::Ladder::Select { .. });
            let kept = match ladder {
                steps::Ladder::Sort { .. } | steps::Ladder::Select { rank: false, .. } => {
                    Some(Temporary::Sort)
                }
                steps::Ladder::Reduce { .. } => Some(Temporary::Reduce),
                _ => None,
            };
            if let (Some(kept), Some(&arg)) = (kept, op.args.first()) {
                stepping.temporaries.push((arg, kept));
            }
        }
        stepping.launched = launched;
        stepping
    }

    /// The launches `descriptors` are laid over: below the temporary area
    /// the one-row select's state, in it the most a launch that sorts or
    /// reduces keeps there.
    #[must_use]
    pub fn launches(&self, descriptors: &[ValueDesc]) -> Launches<'_> {
        Launches {
            each: self.launched.iter().map(Launch::of_node).collect(),
            idle: self.idle.clone(),
            staged: self.staged.clone(),
            kept: self.kept.clone(),
            hollow: self.hollow.clone(),
            reserve: if self.selects { SELECT_RESERVE } else { 0 },
            temporary: self
                .temporaries
                .iter()
                .map(|&(value, kept)| {
                    descriptors
                        .get(value as usize)
                        .map_or(u64::MAX, |d| kept.bytes(d))
                })
                .max()
                .unwrap_or(0),
        }
    }
}

pub fn alias_reshapes(plan: &LaunchStagePlan, values: &mut [u64]) {
    for (result, source) in wgsl_analysis::analyze_stage(plan).aliases {
        let Some(&at) = values.get(source as usize) else {
            continue;
        };
        if let Some(slot) = values.get_mut(result as usize) {
            *slot = at;
        }
    }
}

/// The bytes of a stage's tables (descriptors, op params, value offsets, config): what its
/// value and op counts fix, whatever the extents a bind resolves.
#[must_use]
pub fn table_sizes(plan: &LaunchStagePlan) -> [u64; 4] {
    [
        (plan.value_types.len() * DESC_WORDS * 4).max(DESC_WORDS * 4) as u64,
        (plan.ops.len().max(1) * size_of::<OpParams>()) as u64,
        (plan.value_types.len().max(1) * 4) as u64,
        size_of::<Cfg>() as u64,
    ]
}

/// Every stage's tables, stage by stage, as an instance of `plans` lays
/// them in its slots.
#[must_use]
pub fn program_table_sizes(plans: &[LaunchStagePlan]) -> Vec<u64> {
    plans.iter().flat_map(table_sizes).collect()
}

/// What a stage resolves at a bind: its values' descriptors and scratch
/// offsets, the scratch's bytes, and its tables' bytes ([`table_sizes`]
/// each, in that order).
pub struct StageTables {
    pub descriptors: Vec<ValueDesc>,
    pub offsets: Vec<u64>,
    pub scratch: u64,
    pub tables: [Vec<u8>; 4],
}

/// Resolves `plan` against `extents`.
pub fn stage_tables(plan: &LaunchStagePlan, extents: &Extents) -> Result<StageTables, String> {
    stage_tables_launched(plan, extents, |_| true)
}

/// Resolves `plan` against `extents` where the regions `launches` says no
/// to are another party's (a sink the engine serves): a value only they
/// read is described empty, and takes no bytes.
pub fn stage_tables_launched(
    plan: &LaunchStagePlan,
    extents: &Extents,
    launches: impl Fn(&LaunchRegion) -> bool,
) -> Result<StageTables, String> {
    let descriptors = describe_launched(plan, extents, launches).map_err(|why| {
        format!("a guest value's shape does not resolve against this fire: {why:?}")
    })?;
    let mut scratch = lay_out(
        plan,
        &descriptors,
        &Stepping::of(plan).launches(&descriptors),
    )
    .map_err(|why| format!("a guest pass's scratch does not fit: {why:?}"))?;
    alias_reshapes(plan, &mut scratch.values);

    let tables = tables_of(
        plan,
        &descriptors,
        &scratch.values,
        u32::try_from(scratch.temporary / 4).unwrap_or(u32::MAX),
    );
    Ok(StageTables {
        descriptors,
        tables,
        offsets: scratch.values,
        scratch: scratch.total.max(256),
    })
}

/// A value's dtype word where its elements are an output's rows, read
/// where they lie (`codegen::cuda_steps`): the base dtype in the low byte,
/// this bit, and the output's node from [`EXTERNAL_NODE_SHIFT`] up.
pub const EXTERNAL: u32 = 0x100;

/// A value's dtype word where its elements are computed where they are
/// read (`codegen::cuda_steps::computed_where_read`), by its node (from
/// [`EXTERNAL_NODE_SHIFT`] up).
pub const COMPUTED: u32 = 0x200;

/// Where an external or computed value's dtype word holds its node.
pub const EXTERNAL_NODE_SHIFT: u32 = 10;

/// What a stepper that runs each node's steps in the stage's order
/// resolves at a bind ([`stepped_tables`]): the tables; each node's
/// temporary area in a lane's scratch (a word offset: the one-row select's
/// state lies just below it; 0 for a node that keeps none there); and the
/// most bytes a ladder keeps, which lie in an arena of the fire's rather
/// than in any lane's scratch.
pub struct SteppedTables {
    pub tables: StageTables,
    pub temporaries: Vec<u32>,
    pub arena: u64,
}

/// Resolves `plan` against `extents` for a stepper that runs every node in
/// the stage's order, boundary ops among them (`codegen::cuda_steps`),
/// where the regions `launches` says no to are another party's. Its
/// scratch is laid tighter than [`stage_tables`] lays it:
/// - an output's result (and a reshape of it) takes no bytes: it is read
///   where the output lies ([`EXTERNAL`] in its descriptor);
/// - a reshape that covers its source is the source's bytes
///   (`wgsl_analysis::analyze_stepped`);
/// - a read's, a take's and a constant's result is live from its node,
///   and a put's value until it;
/// - the one-row select keeps its state as a value of its own, live at its
///   node only, in bytes a value dead by then held;
/// - a ladder that orders or folds a value (a sort's, a top-k's, a
///   pivot's, the one the one-row select hands a row to, a reduce's) keeps
///   its buffers in the fire's arena ([`SteppedTables::arena`]), which a
///   lane uses only while it runs that node.
pub fn stepped_tables(
    plan: &LaunchStagePlan,
    extents: &Extents,
    launches: impl Fn(&LaunchRegion) -> bool,
) -> Result<SteppedTables, String> {
    let mut descriptors = describe_launched(plan, extents, launches).map_err(|why| {
        format!("a guest value's shape does not resolve against this fire: {why:?}")
    })?;
    let fusion = wgsl_analysis::analyze_stepped(plan);
    let computed = eta_compiler::codegen::cuda_steps::computed_where_read(plan);
    let values = descriptors.len();

    let mut hollow: Vec<u32> = fusion.aliases.iter().map(|&(result, _)| result).collect();
    let mut order: Vec<u32> = Vec::new();
    let mut idle: Vec<u32> = Vec::new();
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        let results = result_base..result_base.saturating_add(u32::from(op.result_count));
        result_base = results.end;
        let node = u32::try_from(node).map_err(|_| "a stage past a u32 of nodes".to_string())?;
        if op.tag == tags::OUTPUT {
            for result in results {
                if let Some(d) = descriptors.get_mut(result as usize) {
                    d.dtype = (d.dtype & 0xFF) | EXTERNAL | (node << EXTERNAL_NODE_SHIFT);
                }
                hollow.push(result);
            }
            idle.push(node);
            continue;
        }
        if computed[node as usize] {
            for result in results {
                if let Some(d) = descriptors.get_mut(result as usize) {
                    d.dtype = (d.dtype & 0xFF) | COMPUTED | (node << EXTERNAL_NODE_SHIFT);
                }
                hollow.push(result);
            }
            idle.push(node);
            continue;
        }
        let edge = matches!(op.tag, tags::CHAN_READ | tags::CHAN_TAKE | tags::CHAN_PUT);
        if edge || (!wgsl::is_boundary(op.tag) && fusion.emits_node(node as usize)) {
            order.push(node);
        } else {
            idle.push(node);
        }
    }
    // A reshape of an output, or of a computed value, is it too.
    for &(result, root) in &fusion.aliases {
        let root_dtype = descriptors.get(root as usize).map(|d| d.dtype);
        if let (Some(dtype), Some(d)) = (root_dtype, descriptors.get_mut(result as usize))
            && dtype & (EXTERNAL | COMPUTED) != 0
        {
            d.dtype = dtype;
        }
    }

    let run = Launches {
        each: order.iter().map(Launch::of_node).collect(),
        idle,
        hollow,
        ..Launches::default()
    };
    let mut lifetimes = value_lifetimes(plan, &run)
        .ok_or_else(|| "a guest stage's nodes are neither run nor idle".to_string())?;
    // A computed value's arguments live as long as it is read, latest
    // first, so a chain carries its reader's step down to its roots.
    let mut bases = Vec::with_capacity(plan.ops.len());
    let mut base = 0u32;
    for op in &plan.ops {
        bases.push(base);
        base += u32::from(op.result_count);
    }
    let alias_of = |value: u32| {
        fusion
            .aliases
            .iter()
            .find(|&&(result, _)| result == value)
            .map_or(value, |&(_, root)| root)
    };
    for (node, op) in plan.ops.iter().enumerate().rev() {
        if !computed[node] {
            continue;
        }
        let last = lifetimes
            .get(bases[node] as usize)
            .map_or(0, |life| life.last);
        for arg in op.reads() {
            for value in [arg, alias_of(arg)] {
                if let Some(life) = lifetimes.get_mut(value as usize) {
                    life.last = life.last.max(last);
                    life.launch_last = life.launch_last.max(last);
                }
            }
        }
    }
    let mut laid = descriptors.clone();
    let mut kept: Vec<(u32, u64, usize)> = Vec::new();
    let mut arena = 0u64;
    for (step, &node) in order.iter().enumerate() {
        let at = node as usize;
        let op = &plan.ops[at];
        let ladder = steps::ladder_of(plan, at, op.tag);
        let select = matches!(ladder, steps::Ladder::Select { .. });
        let temporary = match ladder {
            steps::Ladder::Sort { .. } | steps::Ladder::Select { rank: false, .. } => {
                Some(Temporary::Sort)
            }
            steps::Ladder::Reduce { .. } => Some(Temporary::Reduce),
            _ => None,
        };
        let bytes = op
            .args
            .first()
            .and_then(|&arg| descriptors.get(arg as usize))
            .zip(temporary)
            .map_or(0, |(d, kept)| kept.bytes(d));
        let reserve = if select { SELECT_RESERVE } else { 0 };
        // A ladder's buffers are the fire's arena's; a lane keeps only the
        // select's state.
        arena = arena.max(bytes);
        let bytes = 0;
        if bytes + reserve == 0 {
            continue;
        }
        let len = u32::try_from((bytes + reserve).div_ceil(4))
            .map_err(|_| "a guest temporary past a u32 of words".to_string())?;
        let step = u32::try_from(step).unwrap_or(u32::MAX);
        kept.push((node, reserve, laid.len()));
        laid.push(ValueDesc {
            len,
            rows: 1,
            last: len,
            rank: 1,
            dtype: 0,
            dims: [len, 0, 0, 0],
        });
        lifetimes.push(Lifetime {
            dead: false,
            def: step,
            last: step,
            reusable: true,
            launch_def: step,
            launch_last: step,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        });
    }
    let layout = layout_reusing(&laid, &lifetimes, 0, 0)
        .map_err(|why| format!("a guest pass's scratch does not fit: {why:?}"))?;
    let mut offsets = layout.values[..values].to_vec();
    for &(result, root) in &fusion.aliases {
        if let Some(&at) = offsets.get(root as usize)
            && let Some(slot) = offsets.get_mut(result as usize)
        {
            *slot = at;
        }
    }
    let mut temporaries = vec![0u32; plan.ops.len()];
    for (node, reserve, value) in kept {
        temporaries[node as usize] =
            u32::try_from((layout.values[value] + reserve) / 4).unwrap_or(u32::MAX);
    }
    let tables = tables_of(plan, &descriptors, &offsets, 0);
    Ok(SteppedTables {
        tables: StageTables {
            descriptors,
            offsets,
            scratch: layout.total.max(256),
            tables,
        },
        temporaries,
        arena: arena.next_multiple_of(256),
    })
}

/// The tables' bytes ([`table_sizes`] each): `descriptors`, the ops'
/// params, `offsets`, and the config with `temporary` words.
fn tables_of(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    offsets: &[u64],
    temporary: u32,
) -> [Vec<u8>; 4] {
    let sizes = table_sizes(plan);
    let sized = |mut bytes: Vec<u8>, at: usize| {
        bytes.resize(usize::try_from(sizes[at]).unwrap_or(usize::MAX), 0);
        bytes
    };
    let descs: Vec<u8> = descriptors
        .iter()
        .flat_map(|d| words_le(&desc_words(d)))
        .collect();
    let mut params = Vec::with_capacity(plan.ops.len() * size_of::<OpParams>());
    let mut result_base = 0u32;
    for op in &plan.ops {
        params.extend_from_slice(&OpParams::of(op, result_base, plan.output_slots).to_le_bytes());
        result_base += u32::from(op.result_count);
    }
    let offsets: Vec<u32> = offsets
        .iter()
        .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
        .collect();
    let config = Cfg {
        value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
        temporary,
        op_count: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        lane: 0,
    };
    [
        sized(descs, 0),
        sized(params, 1),
        sized(words_le(&offsets), 2),
        sized(config.to_le_bytes(), 3),
    ]
}
