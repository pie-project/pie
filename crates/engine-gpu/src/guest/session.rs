//! What a guest session shares across APIs: a stage's descriptor words and
//! config, and how wide it dispatches.

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::ValueDesc;

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
