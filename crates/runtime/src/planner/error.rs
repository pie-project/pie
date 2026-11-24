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
    /// A process's end (the declared context, in kv pages: it states none
    /// of its own) past what the device holds alone: as the finisher it
    /// would grow there and never finish ([`super::inner::CapacityGate`]).
    EndPastDevice {
        end: u32,
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
    /// A waiter asking a run of `run` adjacent chunks longer than any
    /// arrangement of the pool lays for state: a run lies inside one
    /// segment, below the narrowest rung's activation and past the reserved
    /// chunks, and no segment holds `run` such chunks (`longest` at most).
    /// Refused by name at once; moving every holder never opens it.
    RunPastPool {
        run: u32,
        longest: u32,
    },
    /// An activation raise whose rung could never fit the pool, even with
    /// every other process moved off it: the reserved chunks, the rung's
    /// span and the `own` chunks its owner's state takes at that rung
    /// (`need` chunks) past the pool's `total` chunks. Refused by name at
    /// once; every raise short of this waits, funded, while the planner
    /// makes room.
    PastRung {
        rung: u32,
        need: u32,
        own: u32,
        total: u32,
    },
    /// Contention found every tier full with no move left to free the
    /// head's room: this process, the lowest in the policy's order, fails
    /// so the others proceed ([`super::ResidencyPlanner::fail_tiers_full`]). Each
    /// tier's free and total pages.
    TiersFull {
        device: (u32, u32),
        host: (u32, u32),
        disk: (u32, u32),
    },
    /// A newcomer's first ask that fits nowhere: past what the device has
    /// free and what the tiers below could take of the evictions it would
    /// force. Refused by name at admission; admitted processes wait and
    /// move instead. Its pages, and each tier's free and total pages.
    AdmissionRefused {
        need: u32,
        device: (u32, u32),
        host: (u32, u32),
        disk: (u32, u32),
    },
    Cancelled,
}

impl PlannerError {
    /// A stable code a client reads by machine, carried at the head of the
    /// error's text (`[code] ...`) through the process's result, the
    /// worker's reply and the gateway's error body (`code`): an admission
    /// refusal a client retries, a wedge failure it does not. `None` for
    /// errors with no code of their own.
    #[must_use]
    pub fn code(&self) -> Option<&'static str> {
        match self {
            PlannerError::AdmissionRefused { .. } => Some(ADMISSION_REFUSED),
            PlannerError::TiersFull { .. } => Some(TIERS_FULL),
            _ => None,
        }
    }
}

/// [`PlannerError::code`] of an admission refusal: retried by the client.
pub const ADMISSION_REFUSED: &str = "admission_refused";
/// [`PlannerError::code`] of a wedge failure: every tier full.
pub const TIERS_FULL: &str = "tiers_full";

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
            PlannerError::EndPastDevice { end, room } => write!(
                f,
                "a process may grow to the declared context, {end} kv pages, and it states no \
                 end of its own: that never fits the device alone, so it could never finish: {}",
                room.describe(*end)
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
            PlannerError::RunPastPool { run, longest } => write!(
                f,
                "device memory can never lay this need: a run of {run} adjacent chunks, past the \
                 {longest} any segment of the pool holds for state — a run lies inside one \
                 segment, so no move ever opens it; raise [engine.memory] budget_device or ask \
                 for less"
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
            PlannerError::PastRung {
                rung,
                need,
                own,
                total,
            } => write!(
                f,
                "the activation rung of {rung} tokens can never fit: with the reserved \
                 chunks and the {own} chunks the process's own state takes beside it, it \
                 takes {need} chunks of a {total}-chunk pool, even with every other \
                 process moved off it"
            ),
            PlannerError::TiersFull { device, host, disk } => write!(
                f,
                "[tiers_full] every tier is full (device {} of {} pages free, host {} of {}, disk {} of {}) \
                 and no move frees room for the process waiting: this one, the lowest in the \
                 policy's order, fails so the others proceed",
                device.0, device.1, host.0, host.1, disk.0, disk.1
            ),
            PlannerError::AdmissionRefused { need, device, host, disk } => write!(
                f,
                "[admission_refused] refused at admission: the first ask of {need} kv pages fits nowhere \
                 (device {} of {} pages free, host {} of {}, disk {} of {}): the device \
                 cannot hold it and the tiers below cannot take the evictions it would force",
                device.0, device.1, host.0, host.1, disk.0, disk.1
            ),
            PlannerError::Cancelled => f.write_str("planner request cancelled"),
        }
    }
}

impl std::error::Error for PlannerError {}
