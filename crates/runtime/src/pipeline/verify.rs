//! The verify budget (`[runtime] verify_budget`): how many proposals the
//! requests drafting right now may verify between them, and which of them
//! get the rows.
//!
//! Without it each speculating lane prices its own verify width from its own
//! clock, and a lane beside others either drafts as if alone or not at all.
//! But a verify row costs whatever it adds to the fire it batches into, which
//! every lane in that fire pays: while the batch's rows are few the device is
//! bound by reading the weights, a row is nearly free, and past the decode
//! curve's knee every row lengthens the step for everyone. So the budget is a
//! total, the extra rows a frame takes before it runs `SLACK` longer than one
//! of only the one-token arms (DeepInfer's choice of a total verify width
//! from the measured decode curve), or a fixed count; and it goes to the
//! likeliest tokens: each lane bids the chance of every proposal surviving,
//! and the budget's rows go to the largest of those chances across lanes.
//! Because a lane's chances only fall along its block, the rows any lane wins
//! are a prefix of its proposals.
//!
//! Lanes round at their own pace, so there is no step every bid lands in:
//! a lane's bid stands until it bids again, withdraws, closes its pipeline or
//! goes `BID_LIFETIME` without bidding, and each request is answered against
//! the bids standing then. The curve is read off each frame's rows and its
//! post-to-retire time, which a deep dispatch queue lengthens at every width
//! alike, so its ratios are what the budget reads.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use crate::rt::Instant;

/// How many proposals the drafting lanes may verify between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Budget {
    /// No allotment: every lane prices itself, and `allot` grants each bid
    /// whole.
    #[default]
    Off,
    /// The rows the measured decode curve carries before a frame runs
    /// `SLACK` longer.
    Measured,
    /// A fixed count of rows.
    Rows(u32),
}

impl FromStr for Budget {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.trim() {
            "off" | "" => Ok(Budget::Off),
            "auto" => Ok(Budget::Measured),
            rows => match rows.parse::<u32>() {
                Ok(0) => Ok(Budget::Off),
                Ok(rows) => Ok(Budget::Rows(rows)),
                Err(_) => Err(format!(
                    "verify_budget is \"off\", \"auto\" or a count of rows, not {text:?}"
                )),
            },
        }
    }
}

/// A frame of the one-token arms may run this much longer to carry verify
/// rows: about where a decode step on a memory-bound device leaves its
/// flat stretch.
const SLACK: f64 = 0.2;
/// A bid not renewed for this long is a lane that stopped drafting.
const BID_LIFETIME: Duration = Duration::from_millis(250);
/// Past the widest frame measured, the curve is assumed flat up to this
/// many times its width, so wider frames get tried and measured.
const EXPLORE: u32 = 2;
/// The widest budget the curve is searched to.
const MOST_ROWS: u32 = 4096;
/// Each width's latency is the recent frames', each weighed this much the
/// next's.
const DECAY: f64 = 0.9;

/// A frame's latency against its rows, one point per power of two of rows.
#[derive(Clone, Debug, Default)]
pub(crate) struct Curve {
    /// `(rows, latency_us)`, each averaged over the bucket's recent frames.
    points: [Option<(f64, f64)>; Curve::BUCKETS],
}

impl Curve {
    const BUCKETS: usize = 24;

    fn bucket(rows: u32) -> usize {
        (u32::BITS - rows.max(1).leading_zeros()) as usize - 1
    }

    pub(crate) fn record(&mut self, rows: u32, latency_us: f64) {
        if rows == 0 || !(latency_us.is_finite() && latency_us > 0.0) {
            return;
        }
        let slot = &mut self.points[Self::bucket(rows).min(Self::BUCKETS - 1)];
        let rows = f64::from(rows);
        *slot = Some(match *slot {
            None => (rows, latency_us),
            Some((r, l)) => (
                DECAY * r + (1.0 - DECAY) * rows,
                DECAY * l + (1.0 - DECAY) * latency_us,
            ),
        });
    }

    /// The latency of a frame of `rows`, on the line between the measured
    /// widths around it; below the narrowest, the narrowest's. `None` past
    /// the widest.
    fn latency(&self, rows: f64) -> Option<f64> {
        let points: Vec<(f64, f64)> = self.points.iter().flatten().copied().collect();
        let first = *points.first()?;
        if rows <= first.0 {
            return Some(first.1);
        }
        points.windows(2).find_map(|pair| {
            let ((r0, l0), (r1, l1)) = (pair[0], pair[1]);
            (rows <= r1).then(|| l0 + (l1 - l0) * (rows - r0) / (r1 - r0).max(f64::EPSILON))
        })
    }

    fn widest(&self) -> Option<f64> {
        self.points.iter().flatten().map(|p| p.0).reduce(f64::max)
    }

    /// The rows a frame of `base` one-token rows takes beyond them before
    /// it runs `SLACK` longer: the first width past the slack ends the
    /// search, and a width past what was measured counts as flat up to
    /// `EXPLORE` times the widest measured.
    pub(crate) fn budget(&self, base: u32) -> u32 {
        let base = base.max(1);
        let reach = self
            .widest()
            .map_or(f64::from(base), |w| w.max(f64::from(base)))
            * f64::from(EXPLORE);
        let Some(floor) = self.latency(f64::from(base)) else {
            // Nothing measured this wide: try the next width out.
            return (reach as u32).saturating_sub(base).min(MOST_ROWS);
        };
        let ceiling = floor * (1.0 + SLACK);
        let mut extra = 0;
        while extra < MOST_ROWS {
            let rows = f64::from(base + extra + 1);
            let fits = match self.latency(rows) {
                Some(latency) => latency <= ceiling,
                None => rows <= reach,
            };
            if !fits {
                break;
            }
            extra += 1;
        }
        extra
    }
}

/// How many of `mine`'s proposals win a row when `budget` rows go to the
/// largest survival chances among `bids` (which hold `mine`'s own bid
/// under `me`). A bid's chances are read as falling (each one at most the
/// one before, as a survival is), so the rows a lane wins are a prefix; a
/// tie goes to the earlier proposal, then the lower lane.
pub(crate) fn share<'a>(
    bids: impl IntoIterator<Item = (u64, &'a [f32])>,
    me: u64,
    budget: u32,
) -> u32 {
    let mut entries: Vec<(f32, usize, u64)> = Vec::new();
    for (lane, survival) in bids {
        let mut running = 1.0f32;
        for (position, &chance) in survival.iter().enumerate() {
            let chance = if chance.is_finite() {
                chance.clamp(0.0, 1.0)
            } else {
                0.0
            };
            running = running.min(chance);
            entries.push((running, position, lane));
        }
    }
    entries.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    entries
        .iter()
        .take(budget as usize)
        .filter(|entry| entry.2 == me)
        .count() as u32
}

struct Bid {
    survival: Vec<f32>,
    at: Instant,
}

#[derive(Default)]
struct Allotter {
    budget: Budget,
    bids: HashMap<u64, Bid>,
    curve: Curve,
}

static ALLOTTER: LazyLock<Mutex<Allotter>> = LazyLock::new(Default::default);
/// Whether frames feed the curve, read without the lock on every retire.
static MEASURING: AtomicBool = AtomicBool::new(false);

fn allotter() -> std::sync::MutexGuard<'static, Allotter> {
    ALLOTTER
        .lock()
        .expect("the verify budget's lock is poisoned")
}

/// Set at boot from `[runtime] verify_budget`.
pub fn configure(budget: Budget) {
    let mut allotter = allotter();
    allotter.budget = budget;
    allotter.bids.clear();
    MEASURING.store(budget == Budget::Measured, Ordering::Relaxed);
}

#[must_use]
pub fn budget() -> Budget {
    allotter().budget
}

/// The proposals `lane`'s next verify may carry of those `survival` bids
/// for (`survival[j]`: the chance the round keeps proposal `j` and every
/// one before it), given `base` one-token rows a frame. Every bid is granted
/// whole with no budget set; an empty bid withdraws.
pub fn allot(lane: u64, survival: &[f32], base: u32) -> u32 {
    let mut allotter = allotter();
    let total = match allotter.budget {
        Budget::Off => return survival.len() as u32,
        Budget::Rows(rows) => rows,
        Budget::Measured => allotter.curve.budget(base),
    };
    let now = Instant::now();
    allotter
        .bids
        .retain(|_, bid| now.duration_since(bid.at) < BID_LIFETIME);
    if survival.is_empty() {
        allotter.bids.remove(&lane);
        return 0;
    }
    allotter.bids.insert(
        lane,
        Bid {
            survival: survival.to_vec(),
            at: now,
        },
    );
    share(
        allotter
            .bids
            .iter()
            .map(|(&lane, bid)| (lane, bid.survival.as_slice())),
        lane,
        total,
    )
}

/// A closed pipeline drafts no more.
pub fn withdraw(lane: u64) {
    let mut allotter = allotter();
    if allotter.budget != Budget::Off {
        allotter.bids.remove(&lane);
    }
}

/// A retired frame of `rows` that took `latency` from post to retire.
pub(crate) fn record_frame(rows: usize, latency: Duration) {
    if !MEASURING.load(Ordering::Relaxed) {
        return;
    }
    let rows = u32::try_from(rows).unwrap_or(u32::MAX);
    allotter().curve.record(rows, latency.as_secs_f64() * 1e6);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A curve like a memory-bound decode step: flat at 10 ms to 32 rows,
    /// then 0.3 ms a row.
    fn measured() -> Curve {
        let mut curve = Curve::default();
        for rows in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
            let latency = 10_000.0 + 300.0 * f64::from(rows.saturating_sub(32));
            curve.record(rows, latency);
        }
        curve
    }

    #[test]
    fn the_rows_go_to_the_likeliest_proposals_across_lanes() {
        let sure = [0.95f32, 0.9, 0.85, 0.8];
        let unsure = [0.6f32, 0.3, 0.1, 0.05];
        let bids = [(1, &sure[..]), (2, &unsure[..])];
        assert_eq!(share(bids, 1, 5), 4);
        assert_eq!(share(bids, 2, 5), 1);
    }

    #[test]
    fn a_budget_wider_than_every_bid_grants_them_whole() {
        let a = [0.9f32, 0.5];
        let b = [0.2f32];
        let bids = [(1, &a[..]), (2, &b[..])];
        assert_eq!(share(bids, 1, 100), 2);
        assert_eq!(share(bids, 2, 100), 1);
    }

    #[test]
    fn no_budget_grants_nothing() {
        let a = [0.99f32; 4];
        assert_eq!(share([(1, &a[..])], 1, 0), 0);
    }

    #[test]
    fn a_lane_wins_a_prefix_even_where_its_chances_rise() {
        // A later proposal cannot be likelier to survive than an earlier
        // one: it rides on it.
        let rising = [0.1f32, 0.99, 0.99];
        let steady = [0.5f32, 0.5];
        let bids = [(1, &rising[..]), (2, &steady[..])];
        assert_eq!(share(bids, 1, 2), 0);
        assert_eq!(share(bids, 2, 2), 2);
    }

    #[test]
    fn a_tie_goes_to_the_earlier_proposal_then_the_lower_lane() {
        let a = [0.5f32, 0.5];
        let b = [0.5f32, 0.5];
        let bids = [(7, &a[..]), (3, &b[..])];
        assert_eq!(share(bids, 3, 3), 2);
        assert_eq!(share(bids, 7, 3), 1);
    }

    #[test]
    fn an_unreadable_chance_wins_nothing() {
        let a = [f32::NAN, 0.9];
        let b = [0.1f32];
        let bids = [(1, &a[..]), (2, &b[..])];
        assert_eq!(share(bids, 2, 1), 1);
        assert_eq!(share(bids, 1, 1), 0);
    }

    #[test]
    fn the_measured_budget_ends_at_the_decode_curves_knee() {
        // From 8 rows at 10 ms, 20% slack is 12 ms: 32 flat rows, then
        // 2 ms at 0.3 ms a row is 6 more, so 38 rows in all.
        assert_eq!(measured().budget(8), 30);
    }

    #[test]
    fn a_fuller_frame_leaves_fewer_rows_to_verify() {
        let curve = measured();
        assert!(
            curve.budget(64) < curve.budget(8),
            "{} vs {}",
            curve.budget(64),
            curve.budget(8)
        );
    }

    #[test]
    fn an_unmeasured_width_is_tried_up_to_twice_the_widest_measured() {
        let mut curve = Curve::default();
        curve.record(4, 10_000.0);
        assert_eq!(curve.budget(4), 4);
        assert_eq!(Curve::default().budget(6), 6);
    }

    #[test]
    fn the_budget_reads_from_its_setting() {
        assert_eq!("off".parse(), Ok(Budget::Off));
        assert_eq!("0".parse(), Ok(Budget::Off));
        assert_eq!("auto".parse(), Ok(Budget::Measured));
        assert_eq!("24".parse(), Ok(Budget::Rows(24)));
        assert!("wide".parse::<Budget>().is_err());
    }
}
