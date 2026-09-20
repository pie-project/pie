use engine::fire::{KvDelta, Lane, Step};
use eta_ir::registry::GeometryClass;

use crate::engine::completion::CellPtr;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FireRequest {
    pub lanes: Vec<Lane>,
    pub geometry: GeometryClass,
    pub max_layers: Option<u32>,
    pub single_token_mode: bool,
    pub has_user_mask: bool,
    pub boundary_program: bool,
    pub voxels: Vec<engine::fire::StepVoxels>,
    pub media: Vec<engine::fire::StepMedia>,
    pub cohort: Option<u32>,
}

impl FireRequest {
    #[must_use]
    pub fn one(lane: Lane) -> FireRequest {
        FireRequest {
            lanes: vec![lane],
            ..FireRequest::default()
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.lanes.len()
    }

    #[must_use]
    pub fn tokens(&self) -> usize {
        self.lanes.iter().map(|lane| lane.tokens.len()).sum()
    }

    pub fn pages(&self) -> impl Iterator<Item = u32> + '_ {
        self.lanes
            .iter()
            .flat_map(|lane| lane.kv.pages.iter().copied())
    }

    #[must_use]
    pub fn lane(&self) -> Option<&Lane> {
        self.lanes.first()
    }

    pub fn lane_mut(&mut self) -> Option<&mut Lane> {
        self.lanes.first_mut()
    }

    #[must_use]
    pub fn qo_indptr(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.lanes.len() + 1);
        let mut at = 0u32;
        out.push(0);
        for lane in &self.lanes {
            at = at.saturating_add(lane.rows());
            out.push(at);
        }
        out
    }

    #[must_use]
    pub fn last_page_len(&self, page_size: u32) -> u32 {
        if page_size == 0 {
            return 0;
        }
        let Some(lane) = self.lanes.last() else {
            return 0;
        };
        let after = lane.kv.held.saturating_add(lane.rows());
        match after % page_size {
            0 if after == 0 => 0,
            0 => page_size,
            rest => rest,
        }
    }
}

pub struct StepFire {
    pub submission: Step,
    pub terminal_cells: Vec<CellPtr>,
    pub instances: Vec<u64>,
    pub logical_fire_ids: Vec<u64>,
}

#[derive(Default)]
pub struct FrameFire {
    pub steps: Vec<StepFire>,
}

impl FrameFire {
    pub fn terminal_cells(&self) -> impl Iterator<Item = CellPtr> + '_ {
        self.steps
            .iter()
            .flat_map(|step| step.terminal_cells.iter().copied())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaskWords {
    pub request_indptr: Vec<u32>,
    pub word_indptr: Vec<u32>,
    pub words: Vec<u32>,
}

#[must_use]
pub fn bitmask_words(lanes: &[Lane]) -> MaskWords {
    let mut request_indptr = Vec::with_capacity(lanes.len() + 1);
    let mut word_indptr = vec![0u32];
    let mut words: Vec<u32> = Vec::new();
    request_indptr.push(0);
    for lane in lanes {
        for mask in lane.mask.iter().flat_map(|masking| masking.masks()) {
            let start = words.len();
            words.resize(start + mask.words(), 0);
            mask.expand_into(&mut words[start..]);
            word_indptr.push(u32::try_from(words.len()).unwrap_or(u32::MAX));
        }
        request_indptr.push(u32::try_from(word_indptr.len() - 1).unwrap_or(u32::MAX));
    }
    MaskWords {
        request_indptr,
        word_indptr,
        words,
    }
}

#[must_use]
pub fn lane_of(slot: u32, tokens: Vec<u32>, held: u32, pages: Vec<u32>) -> Lane {
    Lane {
        slot,
        tokens,
        kv: KvDelta {
            held,
            pages,
            translation: Vec::new(),
        },
        ..Lane::default()
    }
}
