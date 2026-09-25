use engine::fire::{AttnClasses, Boundary, Masking, PortFeed, RsReset, RsVerb, SelfCondInput};

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
    pub pages: &'a [u32],
    pub held: Option<u32>,
    pub kv_less: bool,
    pub translation: &'a [u32],
    pub window: &'a [u32],
    pub window_copies: &'a [(u32, u32)],
    pub mask: Option<&'a Masking>,
    pub adapter: Option<u32>,
    pub drafts: bool,
    pub captures_scores: bool,
    pub bidirectional: bool,
    pub self_cond: Option<&'a SelfCondInput>,
    pub rs: RsVerb,
    pub rs_reset: RsReset,
    pub readout: Option<&'a [u32]>,
    pub stream: u8,
    pub group: Option<u32>,
    pub peer: Option<u32>,
    pub ports: &'a [PortFeed],
    pub attn_classes: Option<&'a AttnClasses>,
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            kv_less: false,
            translation: &[],
            window: &[],
            window_copies: &[],
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            bidirectional: false,
            self_cond: None,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            readout: None,
            stream: 0,
            group: None,
            peer: None,
            ports: &[],
            attn_classes: None,
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
    pub fn drafting(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            drafts: true,
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
    pub instance: u64,
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
    pub embed_rows: &'a [i32],
    pub embed_weights: &'a [f32],
    pub token_positions: &'a [i32],
}
