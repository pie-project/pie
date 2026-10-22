//! Where a fire's wall time goes between the host and the device: how long
//! the host spends staging and encoding it, how long the device sits between
//! its buffers and between fires, and on what. Every instant is in seconds on
//! the host's monotonic clock, the one the device stamps a command buffer's
//! start and end on, so host and device times subtract.
//!
//! On under the `fire-timeline` diagnostic; it prints a window of fires at a
//! time in the shape the per-kernel profile prints a fire in, each line one
//! mean in milliseconds per fire of a kind, behind the prefix the bench
//! surfaces from the server's log.

use std::collections::BTreeMap;

/// One committed command buffer of a fire.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Buffer {
    pub commit: f64,
    pub start: f64,
    pub end: f64,
}

/// A landed fire: when the host was handed its step, opened its frame and
/// heard it land, and its buffers in queue order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fire {
    pub staged: f64,
    pub opened: f64,
    /// When the last buffer's completion handler ran.
    pub handled: f64,
    pub buffers: Vec<Buffer>,
}

/// One fire's account, in seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Account {
    /// The previous fire's device end to this one's: the fire's share of
    /// the device's wall time.
    pub period: f64,
    /// The device's first start to its last end on the fire. A buffer's
    /// start is stamped once the queue schedules it, which may be while the
    /// one ahead still runs, so the buffers' own spans overlap and do not
    /// sum to this.
    pub span: f64,
    /// Step handed over to frame open: the host staging the fire.
    pub stage: f64,
    /// Frame open to the last commit: the host encoding the fire.
    pub encode: f64,
    /// The device idle between the fire's own buffers.
    pub inner: f64,
    /// The part of `inner` the next buffer was not yet committed for.
    pub inner_uncommitted: f64,
    /// The device idle between the previous fire's end and this one's start,
    /// split below by what the host was doing.
    pub before: f64,
    /// Waiting on the previous fire's completion handler.
    pub before_notify: f64,
    /// The handler run to this fire's step reaching the engine: the
    /// runtime and the program deciding the next step.
    pub before_runtime: f64,
    /// The engine staging this fire.
    pub before_stage: f64,
    /// The engine encoding this fire's head.
    pub before_head: f64,
    /// The head committed and the device not yet on it.
    pub before_start: f64,
}

/// The instants `points` make, held inside `[first, last]` and kept in
/// order, as the lengths between them.
fn segments<const N: usize>(points: [f64; N]) -> [f64; N] {
    let (low, high) = (points[0], points[N - 1]);
    let mut out = [0.0; N];
    let mut at = low;
    for (i, point) in points.iter().enumerate().skip(1) {
        let next = point.clamp(low, high.max(low)).max(at);
        out[i - 1] = next - at;
        at = next;
    }
    out
}

impl Account {
    /// Accounts `fire` after a previous fire (none for the first), given as
    /// its device end and the instant its completion handler ran.
    #[must_use]
    pub fn of(fire: &Fire, previous: Option<(f64, f64)>) -> Option<Account> {
        let first = fire.buffers.first()?;
        let last = fire.buffers.last()?;
        let mut account = Account {
            stage: (fire.opened - fire.staged).max(0.0),
            encode: (last.commit - fire.opened).max(0.0),
            period: last.end - previous.map_or(first.start, |(end, _)| end),
            ..Account::default()
        };
        account.span = (last.end - first.start).max(0.0);
        for pair in fire.buffers.windows(2) {
            let gap = (pair[1].start - pair[0].end).max(0.0);
            account.inner += gap;
            account.inner_uncommitted += (pair[1].commit - pair[0].end).clamp(0.0, gap);
        }
        if let Some((end, handled)) = previous {
            account.before = (first.start - end).max(0.0);
            let [notify, runtime, stage, head, start, _] = segments([
                end,
                handled,
                fire.staged,
                fire.opened,
                first.commit,
                first.start,
            ]);
            account.before_notify = notify;
            account.before_runtime = runtime;
            account.before_stage = stage;
            account.before_head = head;
            account.before_start = start;
        }
        Some(account)
    }

    fn add(&mut self, other: &Account) {
        for ((_, mine), (_, theirs)) in self.fields_mut().into_iter().zip(other.fields()) {
            *mine += theirs;
        }
        self.stage += other.stage;
        self.inner_uncommitted += other.inner_uncommitted;
    }

    fn fields(&self) -> [(&'static str, f64); 10] {
        let mut copy = *self;
        copy.fields_mut().map(|(name, value)| (name, *value))
    }

    /// The fields a window prints, ten so two kinds in two variants fit the
    /// forty lines the kernel summary keeps.
    fn fields_mut(&mut self) -> [(&'static str, &mut f64); 10] {
        [
            ("period", &mut self.period),
            ("device span", &mut self.span),
            ("host encode", &mut self.encode),
            ("device idle inside", &mut self.inner),
            ("device idle before", &mut self.before),
            ("  of it completion handler", &mut self.before_notify),
            ("  of it runtime to engine", &mut self.before_runtime),
            ("  of it engine staging", &mut self.before_stage),
            ("  of it head encoding", &mut self.before_head),
            ("  of it head committed, not started", &mut self.before_start),
        ]
    }
}

/// A fire's kind: an encode variant, its rows and the launches it encoded.
pub type Kind = (u8, u32, u32);

/// The prefix the bench surfaces a server line behind.
pub const PREFIX: &str = "[pie-fire-timing] ";

/// Fires accounted a window at a time.
#[derive(Debug, Default)]
pub struct Timeline {
    previous: Option<(f64, f64)>,
    began: Option<f64>,
    fires: u32,
    kinds: BTreeMap<Kind, (u32, Account)>,
}

impl Timeline {
    /// Fires a window holds before it is printed.
    pub const WINDOW: u32 = 200;

    /// Accounts `fire` of `kind`; returns the window's lines once it is full.
    pub fn land(&mut self, kind: Kind, fire: &Fire) -> Option<Vec<String>> {
        let account = Account::of(fire, self.previous)?;
        let first = fire.buffers.first().map_or(0.0, |b| b.start);
        let began = *self
            .began
            .get_or_insert(self.previous.map_or(first, |(end, _)| end));
        let end = fire.buffers.last().map_or(first, |b| b.end);
        self.previous = Some((end, fire.handled));
        let slot = self.kinds.entry(kind).or_default();
        slot.0 += 1;
        slot.1.add(&account);
        self.fires += 1;
        if self.fires < Self::WINDOW {
            return None;
        }
        let lines = self.lines(end - began);
        self.fires = 0;
        self.began = Some(end);
        self.kinds.clear();
        Some(lines)
    }

    /// The window's lines: a head the kernel summary reads as one fire,
    /// then each kind's means, kinds of a handful of fires left out.
    fn lines(&self, wall: f64) -> Vec<String> {
        let fires = f64::from(self.fires.max(1));
        let mut lines = vec![format!(
            "{PREFIX}kernels: fire of 1 row(s), {:.3} ms on the device:",
            wall / fires * 1e3
        )];
        for (&(variant, rows, launches), (count, sum)) in &self.kinds {
            if *count * 20 < self.fires {
                continue;
            }
            let count_f = f64::from(*count);
            for (name, value) in sum.fields() {
                lines.push(format!(
                    "{PREFIX}  {:>9.3} ms  {:>5} launch(es)  timeline v{variant} r{rows} d{launches}: {name}",
                    value / count_f * 1e3,
                    count
                ));
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::{Account, Buffer, Fire, PREFIX, Timeline, segments};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn segments_hold_inside_the_span_and_in_order() {
        let [a, b, c, _] = segments([1.0, 0.5, 3.0, 2.0]);
        // 0.5 is before the span: nothing; 3.0 is past it: up to 2.0; the
        // last point is already passed.
        assert!(close(a, 0.0));
        assert!(close(b, 1.0));
        assert!(close(c, 0.0));
    }

    #[test]
    fn the_gap_before_a_fire_splits_by_what_the_host_was_doing() {
        let fire = Fire {
            staged: 10.3,
            opened: 10.4,
            handled: 12.1,
            buffers: vec![
                Buffer {
                    commit: 10.5,
                    start: 10.6,
                    end: 10.8,
                },
                Buffer {
                    commit: 11.0,
                    start: 11.1,
                    end: 12.0,
                },
            ],
        };
        let account = Account::of(&fire, Some((10.0, 10.1))).expect("two buffers");
        assert!(close(account.period, 2.0));
        assert!(close(account.span, 1.4));
        assert!(close(account.stage, 0.1));
        assert!(close(account.encode, 0.6));
        assert!(close(account.inner, 0.3));
        assert!(close(account.inner_uncommitted, 0.2));
        assert!(close(account.before, 0.6));
        assert!(close(account.before_notify, 0.1));
        assert!(close(account.before_runtime, 0.2));
        assert!(close(account.before_stage, 0.1));
        assert!(close(account.before_head, 0.1));
        assert!(close(account.before_start, 0.1));
    }

    #[test]
    fn a_fire_committed_while_the_device_is_busy_waits_on_nothing_of_the_host() {
        let fire = Fire {
            staged: 0.5,
            opened: 1.0,
            handled: 4.1,
            buffers: vec![Buffer {
                commit: 1.5,
                start: 3.0,
                end: 4.0,
            }],
        };
        let account = Account::of(&fire, Some((3.0, 3.1))).expect("one buffer");
        assert!(close(account.before, 0.0));
        assert!(close(account.before_notify, 0.0));
        assert!(close(account.before_runtime, 0.0));
        assert!(close(account.before_head, 0.0));
        assert!(close(account.before_start, 0.0));
    }

    #[test]
    fn a_window_prints_once_full_and_starts_over() {
        let mut timeline = Timeline::default();
        let mut at = 0.0;
        for fire in 0..Timeline::WINDOW * 2 {
            let landed = timeline.land(
                (0, 16, 1000),
                &Fire {
                    staged: at,
                    opened: at,
                    handled: at + 0.010,
                    buffers: vec![Buffer {
                        commit: at + 0.001,
                        start: at + 0.001,
                        end: at + 0.010,
                    }],
                },
            );
            at += 0.010;
            assert_eq!(landed.is_some(), (fire + 1) % Timeline::WINDOW == 0);
            if let Some(lines) = landed {
                // The first window opens at its first start; the second at
                // the first's end, so it holds whole fire periods.
                let head = if fire < Timeline::WINDOW {
                    "kernels: fire of 1 row(s), 9.995 ms"
                } else {
                    "kernels: fire of 1 row(s), 10.000 ms"
                };
                assert!(lines.iter().all(|l| l.starts_with(PREFIX)));
                assert!(lines[0].contains(head), "{}", lines[0]);
                assert!(
                    lines
                        .iter()
                        .any(|l| l.contains("v0 r16 d1000: device span"))
                );
            }
        }
    }
}
