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
    /// Past what a shared pool leaves for kv and state once the weights,
    /// the activation and the reserved chunks are set aside: no eviction
    /// makes the room, so it fails up front and says how the pool divides.
    PastPool {
        need: u32,
        room: crate::store::shared::PoolRoom,
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
        /// Short of state (`NoRsSlots`): the victim's own ask of the state
        /// shapes, and what of them stands free and in all. `need`, `free`
        /// and `total` are kv pages either way.
        state: Option<(RsUnits, RsUnits, RsUnits)>,
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
            PlannerError::PastPool { need, room } => write!(
                f,
                "allocation of {need} kv pages can never fit: {}",
                room.describe(*need)
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
                state,
            } => match state {
                // The pages were never what it lacked: say the state.
                Some((need, free, total)) => write!(
                    f,
                    "state pool starved: {need} asked, {free} free of {total}, {}, and no \
                     fire in flight anywhere to complete and free state",
                    cause.describe(),
                ),
                None => write!(
                    f,
                    "KV pool starved: {need} pages asked, {free} free of {total}, {}, and no \
                     fire in flight anywhere to complete and free pages",
                    cause.describe(),
                ),
            },
            PlannerError::RsHog { need, held, total } => write!(
                f,
                "recurrent pools exhausted by this process alone: it holds {held} and asks \
                 {need} more of {total}, and no move of others can cover it"
            ),
            PlannerError::Short {
                tier: crate::policy::pressure::Tier::Host,
                bytes,
                waited,
            } => write!(
                f,
                "quota: the host tier is full (host_kv_budget): short by {bytes} bytes for \
                 {waited:?} of room to park state, and no policy let a set at rest go; close a \
                 set at rest or raise host_kv_budget"
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A waiter short of window slots says so in window slots, not in the
    /// kv pages that stood free beside them ("125 slots asked, 233 free of
    /// 1870" were pages).
    #[test]
    fn a_state_starvation_names_the_state_it_lacked() {
        let mut windows = [0; crate::store::rs::MAX_WINDOW_LISTS];
        windows[0] = 33;
        let need = RsUnits::of_windows(windows);
        windows[0] = 24;
        let free = RsUnits::of_windows(windows);
        windows[0] = 935;
        let total = RsUnits::of_windows(windows);
        let error = PlannerError::Starved {
            need: 125,
            free: 233,
            total: 1870,
            cause: StarveCause::NoRsSlots,
            state: Some((need, free, total)),
        };
        let said = error.to_string();
        assert!(said.starts_with("state pool starved: 0 state slot(s) and 33 buffer page(s) (window lists [33, 0, 0, 0]) asked, 0 state slot(s) and 24"), "{said}");
        assert!(!said.contains("233"), "{said}");
        let kv = PlannerError::Starved {
            need: 125,
            free: 0,
            total: 1870,
            cause: StarveCause::NoEligibleVictim,
            state: None,
        };
        assert!(
            kv.to_string()
                .starts_with("KV pool starved: 125 pages asked, 0 free of 1870")
        );
    }
}
