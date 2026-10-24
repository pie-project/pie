use crate::store::rs::RsUnits;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Reclaim {
    pub(super) pages: u32,
    pub(super) rs: RsUnits,
}

impl Reclaim {
    pub(super) fn covers(self, need: Reclaim) -> bool {
        self.pages >= need.pages && self.rs.covers(need.rs)
    }

    pub(super) fn fits(self, room: Reclaim) -> bool {
        self.pages <= room.pages && room.rs.covers(self.rs)
    }

    pub(super) fn add(self, other: Reclaim) -> Reclaim {
        Reclaim {
            pages: self.pages.saturating_add(other.pages),
            rs: self.rs.add(other.rs),
        }
    }

    pub(super) fn sub(self, other: Reclaim) -> Reclaim {
        Reclaim {
            pages: self.pages.saturating_sub(other.pages),
            rs: self.rs.sub(other.rs),
        }
    }
}
