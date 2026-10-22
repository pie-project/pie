//! Where a fire's wall time goes between the host and the device: how long
//! the host spends encoding it, how long the device sits between its buffers
//! and between fires, and why. Every instant is in seconds on the host's
//! monotonic clock, the one the device stamps a command buffer's start and
//! end on, so host and device times subtract.
//!
//! On under the `fire-timeline` diagnostic; it prints a window of fires at a
//! time to stderr in the shape the per-kernel profile prints a fire in, each
//! line one mean in milliseconds per fire.

use std::collections::BTreeMap;

/// One committed command buffer of a fire.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Buffer {
    pub commit: f64,
    pub start: f64,
    pub end: f64,
}

/// A landed fire: when its frame opened and its buffers in queue order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fire {
    pub opened: f64,
    pub buffers: Vec<Buffer>,
}

/// One fire's account, in seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Account {
    /// Frame open to the last commit: the host encoding the fire.
    pub encode: f64,
    /// Frame open to the first commit: the host encoding the head.
    pub head: f64,
    /// The device running the fire's buffers.
    pub busy: f64,
    /// The device idle between the fire's own buffers.
    pub inner: f64,
    /// The part of `inner` the next buffer was not yet committed for.
    pub inner_uncommitted: f64,
    /// The device idle between the previous fire's end and this one's start.
    pub before: f64,
    /// The part of `before` the host had not opened this fire's frame yet.
    pub before_unopened: f64,
    /// The part of `before` the frame was open but its head uncommitted.
    pub before_encoding: f64,
    /// The first commit to the device starting it, when the device was free.
    pub start_lag: f64,
    /// The device's end to the host seeing the fire land.
    pub land: f64,
    /// The previous fire's device end to this one's: the fire's share of
    /// the device's wall time.
    pub period: f64,
}

impl Account {
    /// Accounts `fire`, landed at `landed`, after a previous fire the device
    /// ended at `previous_end` (none for the first).
    #[must_use]
    pub fn of(fire: &Fire, previous_end: Option<f64>, landed: f64) -> Option<Account> {
        let first = fire.buffers.first()?;
        let last = fire.buffers.last()?;
        let mut account = Account {
            encode: last.commit - fire.opened,
            head: first.commit - fire.opened,
            land: (landed - last.end).max(0.0),
            ..Account::default()
        };
        for buffer in &fire.buffers {
            account.busy += (buffer.end - buffer.start).max(0.0);
        }
        for pair in fire.buffers.windows(2) {
            let gap = (pair[1].start - pair[0].end).max(0.0);
            account.inner += gap;
            account.inner_uncommitted += (pair[1].commit - pair[0].end).clamp(0.0, gap);
        }
        account.period = last.end - previous_end.unwrap_or(first.start);
        let free = previous_end.map_or(first.commit, |end| end.max(first.commit));
        account.start_lag = (first.start - free).max(0.0);
        if let Some(end) = previous_end {
            let gap = (first.start - end).max(0.0);
            account.before = gap;
            account.before_unopened = (fire.opened - end).clamp(0.0, gap);
            account.before_encoding =
                (first.commit - end.max(fire.opened)).clamp(0.0, gap - account.before_unopened);
        }
        Some(account)
    }

    fn add(&mut self, other: &Account) {
        self.encode += other.encode;
        self.head += other.head;
        self.busy += other.busy;
        self.inner += other.inner;
        self.inner_uncommitted += other.inner_uncommitted;
        self.before += other.before;
        self.before_unopened += other.before_unopened;
        self.before_encoding += other.before_encoding;
        self.start_lag += other.start_lag;
        self.land += other.land;
        self.period += other.period;
    }

    fn fields(&self) -> [(&'static str, f64); 11] {
        [
            ("period", self.period),
            ("host encode", self.encode),
            ("host encode to head commit", self.head),
            ("device busy", self.busy),
            ("device idle inside", self.inner),
            ("  of it next buffer uncommitted", self.inner_uncommitted),
            ("device idle before", self.before),
            ("  of it frame unopened", self.before_unopened),
            ("  of it head encoding", self.before_encoding),
            ("device start lag after commit", self.start_lag),
            ("device end to host landed", self.land),
        ]
    }
}

/// A fire's kind: the encode variant it ran under, its rows and the
/// dispatches it encoded.
pub type Kind = (u8, u32, u32);

/// Fires accounted a window at a time.
#[derive(Debug, Default)]
pub struct Timeline {
    previous_end: Option<f64>,
    fires: u32,
    kinds: BTreeMap<Kind, (u32, Account)>,
}

impl Timeline {
    /// Fires a window accounts before it is printed.
    pub const WINDOW: u32 = 120;

    /// Rows past which a fire is a prompt's: it moves the device's clock on
    /// and is not accounted.
    pub const ROWS_MAX: u32 = 16;

    /// What every line starts with, a prefix the bench surfaces.
    pub const PREFIX: &'static str = "[fire timeline] ";

    /// Accounts `fire` of `kind`, landed at `landed`; returns the window's
    /// lines once it is full.
    pub fn land(&mut self, kind: Kind, fire: &Fire, landed: f64) -> Option<Vec<String>> {
        let account = Account::of(fire, self.previous_end, landed)?;
        self.previous_end = fire.buffers.last().map(|b| b.end);
        if kind.1 > Self::ROWS_MAX {
            return None;
        }
        let slot = self.kinds.entry(kind).or_default();
        slot.0 += 1;
        slot.1.add(&account);
        self.fires += 1;
        if self.fires < Self::WINDOW {
            return None;
        }
        let lines = self.lines();
        self.fires = 0;
        self.kinds.clear();
        Some(lines)
    }

    /// One block per variant, in the shape the per-kernel profile prints a
    /// fire in: variant `v` as a fire of `v + 1` rows, each line one mean in
    /// milliseconds per fire of a kind.
    fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let variants: std::collections::BTreeSet<u8> = self.kinds.keys().map(|k| k.0).collect();
        for variant in variants {
            let kinds = || self.kinds.iter().filter(move |(k, _)| k.0 == variant);
            let fires: u32 = kinds().map(|(_, (n, _))| n).sum();
            let period: f64 = kinds().map(|(_, (_, a))| a.period).sum();
            let busy: f64 = kinds().map(|(_, (_, a))| a.busy).sum();
            let per = |value: f64| value / f64::from(fires.max(1)) * 1e3;
            lines.push(format!(
                "{}kernels: fire of {} row(s), {:.3} ms on the device:",
                Self::PREFIX,
                u32::from(variant) + 1,
                per(period)
            ));
            lines.push(format!(
                "{}  {:>9.3} ms  {fires:>5} launch(es)  timeline v{variant} period per fire",
                Self::PREFIX,
                per(period)
            ));
            lines.push(format!(
                "{}  {:>9.3} ms  {fires:>5} launch(es)  timeline v{variant} device busy per fire",
                Self::PREFIX,
                per(busy)
            ));
            for (&(_, rows, dispatches), (count, sum)) in kinds() {
                for (name, value) in sum.fields() {
                    lines.push(format!(
                        "{}  {:>9.3} ms  {count:>5} launch(es)  timeline v{variant} r{rows} \
                         d{dispatches}: {name}",
                        Self::PREFIX,
                        value / f64::from(*count) * 1e3,
                    ));
                }
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::{Account, Buffer, Fire, Timeline};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn a_head_the_device_finishes_before_the_tail_is_committed_waits_on_the_host() {
        let fire = Fire {
            opened: 10.0,
            buffers: vec![
                Buffer {
                    commit: 10.1,
                    start: 10.2,
                    end: 10.5,
                },
                Buffer {
                    commit: 10.8,
                    start: 10.9,
                    end: 12.0,
                },
            ],
        };
        let account = Account::of(&fire, Some(9.5), 12.25).expect("two buffers");
        assert!(close(account.encode, 0.8));
        assert!(close(account.head, 0.1));
        assert!(close(account.busy, 0.3 + 1.1));
        assert!(close(account.inner, 0.4));
        assert!(close(account.inner_uncommitted, 0.3));
        assert!(close(account.before, 0.7));
        assert!(close(account.before_unopened, 0.5));
        assert!(close(account.before_encoding, 0.1));
        assert!(close(account.start_lag, 0.1));
        assert!(close(account.land, 0.25));
    }

    #[test]
    fn a_fire_committed_while_the_device_is_busy_waits_on_nothing_of_the_host() {
        let fire = Fire {
            opened: 1.0,
            buffers: vec![Buffer {
                commit: 1.5,
                start: 3.0,
                end: 4.0,
            }],
        };
        let account = Account::of(&fire, Some(3.0), 4.0).expect("one buffer");
        assert!(close(account.before, 0.0));
        assert!(close(account.start_lag, 0.0));
        assert!(close(account.before_unopened, 0.0));
        assert!(close(account.before_encoding, 0.0));
    }

    fn fire_at(at: f64) -> Fire {
        Fire {
            opened: at,
            buffers: vec![Buffer {
                commit: at + 0.001,
                start: at + 0.001,
                end: at + 0.010,
            }],
        }
    }

    #[test]
    fn a_window_prints_once_full_and_starts_over() {
        let mut timeline = Timeline::default();
        let mut at = 0.0;
        for fire in 0..Timeline::WINDOW * 2 {
            let landed = timeline.land((0, 16, 1000), &fire_at(at), at + 0.010);
            at += 0.010;
            assert_eq!(landed.is_some(), (fire + 1) % Timeline::WINDOW == 0);
            if let Some(lines) = landed {
                // The first fire has no previous end, so its period is its
                // own span; every later one is a whole 10 ms.
                let head = if fire < Timeline::WINDOW {
                    "[fire timeline] kernels: fire of 1 row(s), 9.992 ms"
                } else {
                    "[fire timeline] kernels: fire of 1 row(s), 10.000 ms"
                };
                assert!(lines[0].starts_with(head), "{}", lines[0]);
                assert!(
                    lines
                        .iter()
                        .any(|l| l.contains("timeline v0 r16 d1000: device busy"))
                );
            }
        }
    }

    #[test]
    fn a_prompt_moves_the_clock_on_and_each_variant_prints_its_own_block() {
        let mut timeline = Timeline::default();
        assert!(timeline.land((0, 512, 900), &fire_at(0.0), 0.010).is_none());
        let mut at = 0.010;
        let mut printed = None;
        for fire in 0..Timeline::WINDOW {
            printed = timeline.land(
                (u8::from(fire % 2 == 1), 16, 1000),
                &fire_at(at),
                at + 0.010,
            );
            at += 0.010;
        }
        let lines = printed.expect("a full window");
        let heads: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("kernels: fire"))
            .collect();
        assert_eq!(heads.len(), 2);
        assert!(
            heads[0].contains("fire of 1 row(s), 10.000 ms"),
            "{}",
            heads[0]
        );
        assert!(
            heads[1].contains("fire of 2 row(s), 10.000 ms"),
            "{}",
            heads[1]
        );
        assert!(!lines.iter().any(|l| l.contains("r512")));
    }
}
