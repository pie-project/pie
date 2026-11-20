use crate::store::rs::RsUnits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarveCause {
    NoSwapRoom,
    NoEligibleVictim,
    NoRsSlots,
    /// The victim alone holds what the pool holds, and nothing of it runs.
    SoleHolder,
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
            StarveCause::SoleHolder => {
                "this process alone holds what the pool holds and nothing of it is running \
                 to free any: its parked ask cannot be placed beside what it holds, and a \
                 restart would replay the same asks into the same wall — run fewer lanes at \
                 once or raise [engine.memory] budget_device"
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
    /// A waiter whose need could never fit the device pool, even with
    /// nothing else resident: its kv pages and the pages that free the runs
    /// of whole chunks it asks (`runs`, as `(run, pieces)`), together past
    /// the pool's `total` pages. Refused by name at once; every need short
    /// of this waits, however long, while the planner makes room.
    PastDevice {
        need: u32,
        total: u32,
        runs: Option<(u32, u32)>,
    },
    /// An activation raise whose rung could never fit the pool, even with
    /// all state moved off it: the reserved chunks and the rung's span
    /// (`need` chunks) past the pool's `total` chunks. Refused by name at
    /// once; every raise short of this waits, funded, while the planner
    /// makes room.
    PastRung {
        rung: u32,
        need: u32,
        total: u32,
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
            PlannerError::PastDevice { need, total, runs } => {
                write!(
                    f,
                    "device memory can never hold this need: {need} pages"
                )?;
                if let Some((run, pieces)) = runs {
                    write!(f, " (its kv, and {pieces} runs of {run} adjacent chunks)")?;
                }
                write!(
                    f,
                    " past the whole {total}-page device pool, even with every other process \
                     moved off it — raise [engine.memory] budget_device or ask for less"
                )
            }
            PlannerError::PastRung { rung, need, total } => write!(
                f,
                "the activation rung of {rung} tokens can never fit: with the reserved \
                 chunks it takes {need} chunks of a {total}-chunk pool, even with all \
                 state moved off it"
            ),
            PlannerError::Cancelled => f.write_str("planner request cancelled"),
        }
    }
}

impl std::error::Error for PlannerError {}
