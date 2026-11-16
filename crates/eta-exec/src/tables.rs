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
    Extents, Launch, Launches, OpParams, SELECT_RESERVE, ValueDesc, describe_launched, lay_out,
    whole_value_temporaries,
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
        stepping.selects = launched.iter().any(|&node| {
            let at = node as usize;
            matches!(
                steps::ladder_of(plan, at, plan.ops[at].tag),
                steps::Ladder::Select { .. }
            )
        });
        stepping.launched = launched;
        stepping
    }

    /// The launches `descriptors` are laid over: below the temporary area
    /// the one-row select's state, in it what a launch that sorts or
    /// reduces a whole value takes.
    #[must_use]
    pub fn launches(&self, descriptors: &[ValueDesc]) -> Launches<'_> {
        Launches {
            each: self.launched.iter().map(Launch::of_node).collect(),
            idle: self.idle.clone(),
            staged: self.staged.clone(),
            kept: self.kept.clone(),
            hollow: self.hollow.clone(),
            reserve: if self.selects { SELECT_RESERVE } else { 0 },
            temporary: whole_value_temporaries(descriptors).unwrap_or(u64::MAX),
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
    let offsets: Vec<u32> = scratch
        .values
        .iter()
        .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
        .collect();
    let config = Cfg {
        value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
        temporary: u32::try_from(scratch.temporary / 4).unwrap_or(u32::MAX),
        op_count: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        lane: 0,
    };
    Ok(StageTables {
        descriptors,
        tables: [
            sized(descs, 0),
            sized(params, 1),
            sized(words_le(&offsets), 2),
            sized(config.to_le_bytes(), 3),
        ],
        offsets: scratch.values,
        scratch: scratch.total.max(256),
    })
}
