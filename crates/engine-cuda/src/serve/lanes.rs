use engine::fire::{InputFeed, Masking, RsReset, RsSnapshot, RsVerb};

#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    pub slot: u32,
    pub word: u64,
    pub tokens: &'a [u32],
}

pub type Clips<'a> = crate::voxels::Clips<'a>;

#[derive(Debug, Clone)]
pub struct Seated<'a> {
    pub lane: Lane<'a>,
    pub positions: &'a [u32],
    pub pages: &'a [u32],
    pub held: Option<u32>,
    /// The set's committed boundary (`KvDelta::folded`): the lane's mask
    /// speaks of the keys from it on.
    pub folded: u32,
    /// The set's written end the runtime states, and the set
    /// (`KvDelta::readable`, `KvDelta::set`): what its rows may read.
    pub readable: u32,
    pub set: u64,
    pub kv_less: bool,
    pub translation: &'a [u32],
    /// One past the last slot a row the lane's program places may land at
    /// (`KvDelta::write_end`); its rows land in `[folded, write_end)`.
    pub write_end: u32,
    pub window: &'a [u32],
    /// The lane's slot refs of the window lists past the first, narrowest
    /// first: a list each, where lanes name slot refs.
    pub more_windows: &'a [&'a [u32]],
    pub mask: Option<&'a Masking>,
    pub adapter: Option<u32>,
    pub rs: RsVerb,
    pub rs_reset: RsReset,
    pub rs_slot: Option<u32>,
    /// The record the lane's state is read from where its fire folds
    /// nothing into it (a lane beside the one that folds, in one fire of one
    /// working set): the record as it stood before the fire, while the
    /// lane's rows land in `rs_slot`'s. `None`: `rs_slot`'s.
    pub rs_read: Option<u32>,
    pub rs_snapshots: &'a [RsSnapshot],
    /// The lane's replay as the runtime stated it (`Lane::replay_stated`):
    /// what the submit check's cross-check compares this engine's count
    /// with (`engine::fire::replay_agrees`).
    pub replay_stated: Option<engine::fire::LaneReplay>,
    pub readout: Option<&'a [u32]>,
    pub group: Option<u32>,
    pub inputs: &'a [InputFeed],
    pub reports: u64,
    pub tap: Option<&'a engine::fire::Tap>,
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            positions: &[],
            pages: &[],
            held: None,
            folded: 0,
            readable: 0,
            set: 0,
            kv_less: false,
            translation: &[],
            write_end: 0,
            window: &[],
            more_windows: &[],
            mask: None,
            adapter: None,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            rs_slot: None,
            rs_read: None,
            rs_snapshots: &[],
            replay_stated: None,
            readout: None,
            group: None,
            inputs: &[],
            reports: 0,
            tap: None,
        }
    }

    #[must_use]
    pub fn masked(lane: Lane<'a>, mask: &'a Masking) -> Seated<'a> {
        Seated {
            mask: Some(mask),
            ..Seated::of(lane)
        }
    }

    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
            adapter: Some(id),
            ..Seated::of(lane)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    /// The first of its pass's lanes, `lanes` of them in a row.
    pub lane: u32,
    pub lanes: u32,
    pub instance: engine::InstanceId,
    /// Carries the pass's adapter alone and runs none of its stages.
    pub adapter_only: bool,
}

impl Attached {
    /// The step's lanes its pass spans.
    #[must_use]
    pub fn span(&self) -> std::ops::Range<u32> {
        self.lane..self.lane + self.lanes.max(1)
    }
}

pub(crate) const MROPE_COORDS: usize = 3;

pub(crate) const PATCH_ROUTE_DROP: i32 = -1;

#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    pub lane: u32,
    pub rows: &'a [u32],
    pub patches: &'a [u8],
    pub routes: &'a [i32],
    pub positions: &'a [i32],
    pub grids: &'a [i32],
    pub token_positions: &'a [i32],
}
