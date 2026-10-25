use engine::fire::{AttnClasses, Boundary, InputFeed, Masking, RsReset, RsSnapshot, RsVerb};

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
    pub kv_less: bool,
    pub translation: &'a [u32],
    pub window: &'a [u32],
    pub window_copies: &'a [(u32, u32)],
    pub mask: Option<&'a Masking>,
    pub adapter: Option<u32>,
    pub captures_scores: bool,
    pub rs: RsVerb,
    pub rs_reset: RsReset,
    pub rs_slot: Option<u32>,
    pub rs_snapshots: &'a [RsSnapshot],
    pub readout: Option<&'a [u32]>,
    pub group: Option<u32>,
    pub peer: Option<u32>,
    pub inputs: &'a [InputFeed],
    pub attn_classes: Option<&'a AttnClasses>,
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
            kv_less: false,
            translation: &[],
            window: &[],
            window_copies: &[],
            mask: None,
            adapter: None,
            captures_scores: false,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            rs_slot: None,
            rs_snapshots: &[],
            readout: None,
            group: None,
            peer: None,
            inputs: &[],
            attn_classes: None,
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

    #[must_use]
    pub fn capturing(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            captures_scores: true,
            ..Seated::of(lane)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    pub lane: u32,
    pub instance: engine::InstanceId,
    pub at: Boundary,
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
