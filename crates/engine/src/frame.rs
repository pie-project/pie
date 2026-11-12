pub trait Prepared {
    fn demand(&self) -> Demand;
}

pub trait Enqueued {
    fn launches(&self) -> u32;
}

pub trait Shell {
    type Step<'a>
    where
        Self: 'a;
    type Prepared<'a>: Prepared
    where
        Self: 'a;
    type Enqueued<'a>: Enqueued
    where
        Self: 'a;
    type Settled;
    type Error;

    fn prepare<'a>(
        &mut self,
        step: Self::Step<'a>,
        prev: Option<&Self::Prepared<'a>>,
    ) -> std::result::Result<Self::Prepared<'a>, Self::Error>
    where
        Self: 'a;

    fn enqueue<'a>(
        &mut self,
        prepared: Self::Prepared<'a>,
    ) -> std::result::Result<Self::Enqueued<'a>, Self::Error>
    where
        Self: 'a;

    fn settle<'a>(
        &mut self,
        enqueued: Self::Enqueued<'a>,
    ) -> std::result::Result<Self::Settled, Self::Error>
    where
        Self: 'a;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Demand {
    pub kv_pages: u32,
    pub state_slots: u32,
    pub workspace: u64,
}

impl Demand {
    pub const ZERO: Demand = Demand {
        kv_pages: 0,
        state_slots: 0,
        workspace: 0,
    };

    #[must_use]
    pub const fn union(self, other: Demand) -> Demand {
        Demand {
            kv_pages: if self.kv_pages > other.kv_pages {
                self.kv_pages
            } else {
                other.kv_pages
            },
            state_slots: if self.state_slots > other.state_slots {
                self.state_slots
            } else {
                other.state_slots
            },
            workspace: if self.workspace > other.workspace {
                self.workspace
            } else {
                other.workspace
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Demand;

    #[test]
    fn a_frames_demand_is_the_union_and_not_the_sum() {
        let a = Demand {
            kv_pages: 8,
            state_slots: 1,
            workspace: 4096,
        };
        let b = Demand {
            kv_pages: 3,
            state_slots: 5,
            workspace: 1024,
        };
        let union = a.union(b);
        assert_eq!(union.kv_pages, 8);
        assert_eq!(union.state_slots, 5);
        assert_eq!(union.workspace, 4096);
        assert_eq!(Demand::ZERO.union(a), a);
        assert_eq!(a.union(b), b.union(a));
    }
}
