use engine_api::program::{LaunchChannel, LaunchOp};
use tensor_ir::op::tags::{CHAN_PUT, CHAN_READ, CHAN_TAKE};
use tensor_ir::validate::Direction;

use super::readiness::Effect;

// `OpMeta` — one op's node index and result base — stood here with the
// `op_metadata` that computed a stage's whole table. Its consumer was the
// deleted device lane table's op side; `Malformed` below outlived it because
// `channel_effects` answers with it (alto E).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Malformed {
    ResultBaseOverflowed,

    ResultsExceedValues { results: u32, values: u32 },
}

impl Malformed {
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Malformed::ResultBaseOverflowed => "stage result bases overflow 32 bits",
            Malformed::ResultsExceedValues { .. } => {
                "stage produces more results than it declares values"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inconsistent {
    pub channel: u32,

    pub problem: Problem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    TouchedButUntouchedReadiness,

    NeedsFullButNeverReads,

    NeedsEmptyButNeverPuts,

    ZeroCapacity,
}

impl Inconsistent {
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self.problem {
            Problem::TouchedButUntouchedReadiness => {
                "channel is taken or put but declares no readiness"
            }
            Problem::NeedsFullButNeverReads => "channel waits to be filled but is never read",
            Problem::NeedsEmptyButNeverPuts => "channel waits for room but is never written",
            Problem::ZeroCapacity => "channel holds no cells",
        }
    }
}

pub fn channel_effects(
    channels: &[LaunchChannel],
    stages: &[(&[LaunchOp], &[u32])],
) -> Result<Vec<Effect>, Inconsistent> {
    let mut effects: Vec<Effect> = channels
        .iter()
        .map(|channel| Effect {
            requires_full: channel.readiness == Some(Direction::NeedsFull),
            requires_empty: channel.readiness == Some(Direction::NeedsEmpty),
            take: false,
            put: false,
            capacity: channel.capacity,
        })
        .collect();

    for (ops, bindings) in stages {
        for op in ops.iter() {
            let Some(channel) = op.channel else {
                continue;
            };
            let Some(&dense) = bindings.get(channel as usize) else {
                continue;
            };
            let Some(effect) = effects.get_mut(dense as usize) else {
                continue;
            };
            match op.tag {
                CHAN_TAKE | CHAN_READ => effect.take = true,
                CHAN_PUT => effect.put = true,
                _ => {}
            }
        }
    }

    for (index, effect) in effects.iter().enumerate() {
        let channel = u32::try_from(index).unwrap_or(u32::MAX);
        let problem = if effect.capacity == 0 {
            Some(Problem::ZeroCapacity)
        } else if (effect.take || effect.put) && !effect.requires_full && !effect.requires_empty {
            Some(Problem::TouchedButUntouchedReadiness)
        } else if effect.requires_full && !effect.take {
            Some(Problem::NeedsFullButNeverReads)
        } else if effect.requires_empty && !effect.put {
            Some(Problem::NeedsEmptyButNeverPuts)
        } else {
            None
        };
        if let Some(problem) = problem {
            return Err(Inconsistent { channel, problem });
        }
    }

    Ok(effects)
}
