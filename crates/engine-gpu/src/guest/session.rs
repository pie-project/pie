//! What a guest session shares across APIs: a stage's descriptor words and
//! config, and how wide it dispatches.

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, OpParams, ValueDesc, describe, layout};

pub const DESC_WORDS: usize = 9;

pub const BINDINGS: u32 = 6;

pub const CFG_BINDING: u32 = 5;

pub const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

pub const GROUPS_PER_CORE: u32 = 4;

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

pub fn alias_reshapes(plan: &LaunchStagePlan, values: &mut [u64]) {
    for (result, source) in eta_compiler::codegen::wgsl_analysis::analyze_stage(plan).aliases {
        let Some(&at) = values.get(source as usize) else {
            continue;
        };
        if let Some(slot) = values.get_mut(result as usize) {
            *slot = at;
        }
    }
}

pub fn groups_for(descriptors: &[ValueDesc], cores: u32) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    let cap = cores.max(1).saturating_mul(GROUPS_PER_CORE);
    wanted.clamp(1, cap)
}

/// A stage's bind-time launch tables, in the order they lie in the
/// instance's `Instance` slots: descriptors, op params, value offsets and
/// the config, bound at these bindings.
pub const TABLE_BINDINGS: [u32; 4] = [1, 2, 3, CFG_BINDING];

/// The bytes of a stage's tables ([`TABLE_BINDINGS`] order): what its
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
/// each, [`TABLE_BINDINGS`] order).
pub struct StageTables {
    pub descriptors: Vec<ValueDesc>,
    pub offsets: Vec<u64>,
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
    let mut scratch = layout(&descriptors)
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
