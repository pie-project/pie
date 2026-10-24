//! Fire timing (`fire-timing` diagnostics): each serving fire's span on the
//! device, between two timing events its enqueue records on the compute
//! stream, before its first launch and after its last (bodies, eager
//! kernels, the side streams joined back). A host slower than the device
//! shows as gaps inside the span. The scheduler's bubble histogram is the
//! other side: the host-seen idle between two fires.
//!
//! Nothing waits: a fire's pair is read at a later fire, once the device
//! has passed it, and a fire that finds every pair still in flight goes
//! untimed. Arming's fires are not served, so not timed.

use std::collections::{BTreeMap, VecDeque};

use crate::device::graph::Event;
use crate::error::Result;

/// Event pairs in flight at once: deeper than the fires a host runs ahead.
const PAIRS: usize = 8;

/// Timed fires between two printed reports.
const REPORT: u64 = 1024;

/// The fires of one token bucket, since load.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FireTime {
    pub fires: u64,
    pub total_us: f64,
    pub max_us: f64,
}

impl FireTime {
    #[must_use]
    pub fn mean_us(&self) -> f64 {
        if self.fires == 0 {
            return 0.0;
        }
        self.total_us / self.fires as f64
    }
}

/// A fire's begin and end events.
struct Pair {
    begin: Event,
    end: Event,
}

/// The shell's fire timer.
pub struct FireTimer {
    free: Vec<Pair>,
    flying: VecDeque<(Pair, u32)>,
    times: BTreeMap<u32, FireTime>,
    timed: u64,
    untimed: u64,
}

/// A fire being timed: its pair, its begin recorded.
pub struct Timing(Pair);

impl FireTimer {
    pub fn new() -> Result<FireTimer> {
        let free = (0..PAIRS)
            .map(|_| {
                Ok(Pair {
                    begin: Event::timing()?,
                    end: Event::timing()?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(FireTimer {
            free,
            flying: VecDeque::new(),
            times: BTreeMap::new(),
            timed: 0,
            untimed: 0,
        })
    }

    /// Each token bucket's fires, by bucket, every pair the device has
    /// passed folded in.
    pub fn times(&mut self) -> Result<&BTreeMap<u32, FireTime>> {
        self.read()?;
        Ok(&self.times)
    }

    /// Fires that found every pair in flight.
    #[must_use]
    pub fn untimed(&self) -> u64 {
        self.untimed
    }

    /// Before a fire's launches on `stream`: reads the pairs the device has
    /// passed, then records a free pair's begin; none free, `None`.
    pub fn begin(&mut self, stream: *mut core::ffi::c_void) -> Result<Option<Timing>> {
        self.read()?;
        let Some(pair) = self.free.pop() else {
            self.untimed += 1;
            return Ok(None);
        };
        if let Err(fault) = pair.begin.record(stream) {
            self.free.push(pair);
            return Err(fault);
        }
        Ok(Some(Timing(pair)))
    }

    /// After the launches of a fire at `bucket`: records its end.
    pub fn end(
        &mut self,
        timing: Timing,
        stream: *mut core::ffi::c_void,
        bucket: u32,
    ) -> Result<()> {
        let Timing(pair) = timing;
        match pair.end.record(stream) {
            Ok(()) => {
                self.flying.push_back((pair, bucket));
                Ok(())
            }
            Err(fault) => {
                self.free.push(pair);
                Err(fault)
            }
        }
    }

    /// A fire that failed after its begin: its pair back, untimed.
    pub fn abandon(&mut self, timing: Timing) {
        self.free.push(timing.0);
    }

    /// Folds in every pair the device has passed, oldest first.
    fn read(&mut self) -> Result<()> {
        while let Some((pair, _)) = self.flying.front() {
            if !pair.end.done()? {
                break;
            }
            let us = pair.end.since(&pair.begin)?;
            let (pair, bucket) = self.flying.pop_front().expect("checked above");
            self.free.push(pair);
            let time = self.times.entry(bucket).or_default();
            time.fires += 1;
            time.total_us += us;
            time.max_us = time.max_us.max(us);
            self.timed += 1;
            if self.timed.is_multiple_of(REPORT) {
                self.report();
            }
        }
        Ok(())
    }

    fn report(&self) {
        for (bucket, time) in &self.times {
            eprintln!(
                "engine-cuda: [fire-time] bucket={bucket} fires={} mean={:.1}us max={:.1}us \
                 untimed={}",
                time.fires,
                time.mean_us(),
                time.max_us,
                self.untimed,
            );
        }
    }
}
