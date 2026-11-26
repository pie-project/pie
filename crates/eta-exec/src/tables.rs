//! A guest stage's bind-time tables, as every backend lays them: each
//! value's descriptor ([`desc_words`], [`DESC_WORDS`] u32s), each op's
//! params ([`OpParams`], `PARAM_WORDS` u32s), each value's byte offset in
//! the stage's scratch, and the stage's config ([`Cfg`]); and each op's
//! temporary area, which its steps' records carry. A step of the
//! closed set (`eta_compiler::codegen::steps`) finds its node's operands
//! here: the node's params name its argument and result values, whose
//! descriptors and offsets these tables hold. [`stage_tables`] writes them
//! at a bind, against the bind's extents; [`table_sizes`] is what they
//! take, fixed by the stage's value and op counts.

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::codegen::{wgsl, wgsl_analysis};
use eta_ir::op::tags;

use crate::{
    Extents, Keeps, Launch, Launches, OpParams, SELECT_RESERVE, ValueDesc, describe, lay_out,
};

pub const DESC_WORDS: usize = 9;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cfg {
    pub value_count: u32,
    /// The words of the stage's heap, past which the runtime refuses any
    /// access (its `OUT_OF_BOUNDS` fault): word `CFG_HEAP_WORDS` of the
    /// runtime's config table.
    pub heap_words: u32,
    pub op_count: u32,
    pub lane: u32,
    pub pad: [u32; 4],
}

const _: () = assert!(size_of::<Cfg>() == 32);

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
        words_le(&[
            self.value_count,
            self.heap_words,
            self.op_count,
            self.lane,
            self.pad[0],
            self.pad[1],
            self.pad[2],
            self.pad[3],
        ])
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
    /// Each launched node's tag and first argument, what it keeps in the
    /// temporary area is stated over ([`wgsl::temporary_bytes`]).
    pub runs: Vec<(u8, Option<u32>)>,
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
        stepping.runs = launched
            .iter()
            .filter_map(|&node| plan.ops.get(node as usize))
            .map(|op| (op.tag, op.args.first().copied()))
            .collect();
        stepping.launched = launched;
        stepping
    }

    /// The launches `descriptors` are laid over, a launched node each, a
    /// node that keeps something in the temporary area keeping an area of
    /// its own, alive while it runs: what the runtime keeps there for it
    /// ([`wgsl::temporary_bytes`] of its first argument), with the one-row
    /// select's state below it where it runs the select ([`wgsl::selects`]).
    /// No area is the stage's.
    #[must_use]
    pub fn launches(&self, descriptors: &[ValueDesc]) -> Launches<'_> {
        let keeps = |&(tag, argument): &(u8, Option<u32>)| {
            let bytes = argument
                .and_then(|argument| descriptors.get(argument as usize))
                .map_or(0, |argument| {
                    wgsl::temporary_bytes(tag, u64::from(argument.len), u64::from(argument.rows))
                });
            let reserve = if wgsl::selects(tag) {
                SELECT_RESERVE
            } else {
                0
            };
            (bytes > 0 || reserve > 0).then_some(Keeps { reserve, bytes })
        };
        Launches {
            each: self
                .launched
                .iter()
                .zip(&self.runs)
                .map(|(node, run)| Launch::of_node(node, keeps(run)))
                .collect(),
            idle: self.idle.clone(),
            staged: self.staged.clone(),
            kept: self.kept.clone(),
            hollow: self.hollow.clone(),
            reserve: 0,
            temporary: 0,
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
    /// Each op's temporary area's first word in the heap, 0 where it keeps
    /// none: what its steps' uniforms carry (`StepUniform::area`).
    pub areas: Vec<u32>,
    pub scratch: u64,
    pub tables: [Vec<u8>; 4],
}

/// Resolves `plan` against `extents`.
pub fn stage_tables(plan: &LaunchStagePlan, extents: &Extents) -> Result<StageTables, String> {
    let descriptors = plan
        .value_types
        .iter()
        .map(|value| {
            describe(value, extents).map_err(|why| {
                format!("a guest value's shape does not resolve against this fire: {why:?}")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let stepping = Stepping::of(plan);
    let launches = stepping.launches(&descriptors);
    let mut scratch = lay_out(plan, &descriptors, &launches)
        .map_err(|why| format!("a guest pass's scratch does not fit: {why:?}"))?;
    alias_reshapes(plan, &mut scratch.values);

    let sizes = table_sizes(plan);
    let sized = |mut bytes: Vec<u8>, at: usize| -> Result<Vec<u8>, String> {
        let len = usize::try_from(sizes[at])
            .map_err(|_| format!("a guest stage's table {at} of {} bytes", sizes[at]))?;
        bytes.resize(len, 0);
        Ok(bytes)
    };
    let word = |what: &str, value: u64| {
        u32::try_from(value).map_err(|_| format!("a guest stage's {what} ({value}) past a u32"))
    };
    let descs: Vec<u8> = descriptors
        .iter()
        .flat_map(|d| words_le(&desc_words(d)))
        .collect();
    let mut params = Vec::with_capacity(plan.ops.len() * size_of::<OpParams>());
    let mut result_base = 0u32;
    for op in &plan.ops {
        params.extend_from_slice(&OpParams::of(op, result_base, plan.output_slots).to_le_bytes());
        result_base = result_base
            .checked_add(u32::from(op.result_count))
            .ok_or_else(|| "a guest stage's result ids past a u32".to_string())?;
    }
    // Each node's area where it keeps one ([`Stepping::launches`]), else
    // none: a node that keeps none reads no area. Its steps' records carry
    // it (`StepUniform::area`).
    let mut areas = vec![0u64; plan.ops.len()];
    let keeping = launches
        .each
        .iter()
        .filter(|launch| launch.keeps.is_some())
        .flat_map(|launch| launch.nodes.iter().copied());
    for (node, &at) in keeping.zip(&scratch.areas) {
        areas[node as usize] = at;
    }
    let offsets: Vec<u32> = scratch
        .values
        .iter()
        .map(|&at| word("value offset", at))
        .collect::<Result<_, _>>()?;
    let area_words: Vec<u32> = areas
        .iter()
        .map(|&at| word("area word", at / 4))
        .collect::<Result<_, _>>()?;
    // The heap the stage binds, which its runtime bounds every access by.
    let heap = scratch.total.max(256);
    let config = Cfg {
        value_count: word("value count", plan.value_types.len() as u64)?,
        op_count: word("op count", plan.ops.len() as u64)?,
        lane: 0,
        heap_words: word("heap words", heap / 4)?,
        pad: [0; 4],
    };
    Ok(StageTables {
        descriptors,
        tables: [
            sized(descs, 0)?,
            sized(params, 1)?,
            sized(words_le(&offsets), 2)?,
            sized(config.to_le_bytes(), 3)?,
        ],
        offsets: scratch.values,
        areas: area_words,
        scratch: heap,
    })
}

#[cfg(test)]
mod tests {
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{EngineCaps, Stage};
    use eta_ir::types::{Dtype, Shape};

    use super::stage_tables;
    use crate::Extents;

    /// A stage whose values lie past a u32 word of scratch is refused by the
    /// layout's bound, by name, before any offset could be written as
    /// `u32::MAX`; the offset words' own check backs that bound.
    #[test]
    fn a_stage_past_a_word_of_scratch_is_refused_by_name() {
        let wide = Shape::vector(1 << 30);
        let decl = |host_role| ChannelDecl {
            shape: wide,
            dtype: ChanDType::Concrete(Dtype::F32),
            capacity: 1,
            host_role,
            seeded: false,
        };
        let container = TraceContainer {
            channels: vec![decl(HostRole::Writer), decl(HostRole::Reader)],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Neg(0),
                    Op::Exp(1),
                    Op::Add(1, 2),
                    Op::ChanPut { chan: 1, value: 3 },
                ],
            }],
            ..TraceContainer::default()
        };
        let bound = eta_ir::validate::bind(container, vec![Vec::new()], EngineCaps::default())
            .expect("the stage binds");
        let stages = eta_compiler::plan::compile_bound(&bound);
        let package = eta_compiler::codegen::launch::build(&bound, &stages);
        let refused = package
            .plans
            .iter()
            .map(|plan| stage_tables(plan, &Extents::default()))
            .find_map(Result::err)
            .expect("a stage past a word of scratch is refused");
        assert!(
            refused.starts_with("a guest pass's scratch does not fit: Bound")
                && refused.contains(&format!("limit: {}", crate::scratch::MAX_BYTES)),
            "{refused}"
        );
    }
}
