use crate::store::rs::RsUnits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarveCause {
    NoSwapRoom,
    NoEligibleVictim,
    NoRsSlots,
}

impl StarveCause {
    fn describe(self) -> &'static str {
        match self {
            StarveCause::NoSwapRoom => {
                "no host or disk swap room to evict into (or no swap transport)"
            }
            StarveCause::NoEligibleVictim => {
                "no evictable victim — every page is held by a process OLDER \
                 than the head, which FCFS anti-thrash forbids evicting"
            }
            StarveCause::NoRsSlots => {
                "every RS folded slot is held and no admitted process is still \
                 running to return one — asking for more concurrent state slots \
                 than the pool has cannot be waited out"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerError {
    Impossible {
        need: u32,
        total: u32,
    },
    Hog {
        need: u32,
        held: u32,
        total: u32,
    },
    Starved {
        need: u32,
        free: u32,
        total: u32,
        cause: StarveCause,
    },
    RsHog {
        need: RsUnits,
        held: RsUnits,
        total: RsUnits,
    },
    /// Parked short of room only the policy can make (`tier-pressure`), and
    /// none came by the seat deadline: no discard, no new retention keys,
    /// no growth.
    Short {
        tier: crate::policy::pressure::Tier,
        bytes: u64,
        waited: std::time::Duration,
    },
    Cancelled,
}

impl std::fmt::Display for PlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlannerError::Impossible { need, total } => write!(
                f,
                "allocation of {need} units can never fit (pool total {total})"
            ),
            PlannerError::Hog { need, held, total } => write!(
                f,
                "KV pool exhausted by this process alone: it holds {held} pages and asks \
                 {need} more of a {total}-page pool — no eviction of others can cover it"
            ),
            PlannerError::Starved {
                need,
                free,
                total,
                cause,
            } => {
                let (pool, unit) = if matches!(cause, StarveCause::NoRsSlots) {
                    ("state", "slots")
                } else {
                    ("KV", "pages")
                };
                write!(
                    f,
                    "{pool} pool starved: {need} {unit} asked, {free} free of {total}, {}, and no \
                     fire in flight anywhere to complete and free {unit}",
                    cause.describe(),
                )
            }
            PlannerError::RsHog { need, held, total } => write!(
                f,
                "recurrent pools exhausted by this process alone: it holds {held} and asks \
                 {need} more of {total}, and no move of others can cover it"
            ),
            PlannerError::Short {
                tier,
                bytes,
                waited,
            } => write!(
                f,
                "{} memory short by {bytes} bytes for {waited:?}: the runtime drops no state \
                 on its own, and no policy discard, retention change or memory growth made \
                 room in time",
                tier.name()
            ),
            PlannerError::Cancelled => f.write_str("planner request cancelled"),
        }
    }
}

impl std::error::Error for PlannerError {}
