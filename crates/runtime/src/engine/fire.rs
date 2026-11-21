use engine::fire::{KvDelta, Lane, Step};

pub use self::lane::FireLane;
use eta_ir::registry::GeometryClass;

use crate::engine::completion::CellPtr;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FireRequest {
    pub lanes: Vec<FireLane>,
    pub geometry: GeometryClass,
    pub single_token_mode: bool,
    pub has_user_mask: bool,
    pub boundary_program: bool,
    pub voxels: Vec<engine::fire::StepVoxels>,
    pub media: Vec<engine::fire::StepMedia>,
    /// The recurrent rows its lanes carry unfolded into the fire, which it
    /// replays ahead of its own, summed over them
    /// ([`engine::fire::rung_rows`]): their count where the host places
    /// its rows, a bound on it where the device does.
    pub carried: u32,
    /// The buffered recurrent rows its lanes replay ahead of their own,
    /// summed over every lane, a decode lane's included: what the engine's
    /// stated [`::engine::FireLimits::replayed_rows`] bounds a step by.
    pub replayed: u32,
    /// The ring ends this fire moves. Two fires moving one end never share
    /// a step; the engine picks each move's cell from its ring's cursor.
    pub ring_moves: Vec<RingMove>,
}

/// A fire takes from (`takes`) or puts to (`puts`) a ring's cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingMove {
    pub channel: engine::ChannelId,
    pub takes: bool,
    pub puts: bool,
}

impl FireRequest {
    #[must_use]
    pub fn one(lane: impl Into<FireLane>) -> FireRequest {
        FireRequest {
            lanes: vec![lane.into()],
            ..FireRequest::default()
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.lanes.len()
    }

    /// At least one a lane: a lane whose tokens the device resolves carries
    /// none here but still runs.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.lanes.iter().map(|lane| lane.tokens.len().max(1)).sum()
    }

    /// What the fire asks its activation rung for: its own rows, or the
    /// rows it carries where they are more ([`engine::fire::rung_rows`]).
    #[must_use]
    pub fn rung_rows(&self) -> usize {
        self.tokens().max(self.carried as usize)
    }

    /// The rows the fire reads out: a lane's last row, the rows it names,
    /// or none.
    #[must_use]
    pub fn readouts(&self) -> usize {
        self.lanes
            .iter()
            .map(|lane| match &lane.readout {
                engine::fire::Readout::Last => 1,
                engine::fire::Readout::Rows(rows) => rows.len(),
                engine::fire::Readout::None => 0,
            })
            .sum()
    }

    /// The rows the fire's sequence lanes read out: [`Self::readouts`]
    /// without its float lanes' (`kv_less`, an encoder's or an embedding's
    /// every row), which land through the engine's readout landing whatever
    /// its fixed store holds (`::engine::ReadoutFacts::store_rows`). What
    /// fires are cut to and refused against the store by.
    #[must_use]
    pub fn sequence_readouts(&self) -> usize {
        self.lanes
            .iter()
            .filter(|lane| !lane.kv_less)
            .map(|lane| match &lane.readout {
                engine::fire::Readout::Last => 1,
                engine::fire::Readout::Rows(rows) => rows.len(),
                engine::fire::Readout::None => 0,
            })
            .sum()
    }

    pub fn pages(&self) -> impl Iterator<Item = u32> + '_ {
        self.lanes
            .iter()
            .flat_map(|lane| lane.pages.iter().copied())
    }

    #[must_use]
    pub fn lane(&self) -> Option<&FireLane> {
        self.lanes.first()
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
    /// The completions those cells are of, resolved once the step lands.
    pub completions: Vec<crate::engine::WorkItemCompletion>,
    /// Which of the step's lanes are each fire's: fire `i` has lanes
    /// `member_lanes[i]..member_lanes[i + 1]`, so a lane that faults fails
    /// its own fire alone.
    pub member_lanes: Vec<u32>,
    pub instances: Vec<::engine::InstanceId>,
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
pub fn lane_of(slot: u32, tokens: Vec<u32>, held: u32, pages: Vec<u32>) -> FireLane {
    FireLane {
        lane: Lane {
            slot,
            tokens,
            kv: KvDelta {
                held,
                ..Default::default()
            },
            ..Lane::default()
        },
        pages,
        ..FireLane::default()
    }
}

mod lane {
    use engine::fire::Lane;

    /// A lane of a fire the runtime builds: the engine's lane, and the ids
    /// the runtime's stores place its state by, which the fire names to the
    /// engine in the lane's slot tables (`pipeline::fire::rs::apply_shared`,
    /// `apply_pinned`) and never otherwise.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct FireLane {
        pub lane: Lane,
        /// The kv store's page ids, in token order.
        pub pages: Vec<u32>,
        /// Where stated, the slot each of `pages` stands in (a shared
        /// prefix's), in its place.
        pub translation: Vec<u32>,
    }

    impl From<Lane> for FireLane {
        fn from(lane: Lane) -> Self {
            FireLane {
                lane,
                ..FireLane::default()
            }
        }
    }

    impl std::ops::Deref for FireLane {
        type Target = Lane;
        fn deref(&self) -> &Lane {
            &self.lane
        }
    }

    impl std::ops::DerefMut for FireLane {
        fn deref_mut(&mut self) -> &mut Lane {
            &mut self.lane
        }
    }
}
