//! What Linux says about its memory: pressure stall information.
//!
//! `/proc/pressure/memory` states the share of time tasks stalled on memory.
//! Stalling is the congestion itself, where macOS has to infer it, so no
//! paging counters are read. Both thresholds are hypotheses: no Linux machine
//! has measured them.

use super::{Pressure, System};

/// `some avg10`, percent: a task stalled this share of the last ten seconds.
const CONGESTED: f64 = 5.0;
/// `full avg10`, percent: every task stalled, the machine doing nothing else.
const CRITICAL: f64 = 10.0;

/// The `avg10` of the `kind` line of a pressure file.
fn avg10(text: &str, kind: &str) -> Option<f64> {
    text.lines()
        .find(|line| line.starts_with(kind))?
        .split_whitespace()
        .find_map(|field| field.strip_prefix("avg10="))?
        .parse()
        .ok()
}

/// `None` for text that is not a pressure file.
pub fn read(text: &str) -> Option<Pressure> {
    let some = avg10(text, "some")?;
    let full = avg10(text, "full").unwrap_or(0.0);
    Some(if full >= CRITICAL {
        Pressure::Critical
    } else if some >= CONGESTED {
        Pressure::Congested
    } else {
        Pressure::Calm
    })
}

/// Reads the file a tick at a time. A kernel without PSI has no such file,
/// which is why [`Watch::open`] can refuse.
pub struct Watch;

impl Watch {
    const PATH: &str = "/proc/pressure/memory";

    #[must_use]
    pub fn open() -> Option<Watch> {
        read(&std::fs::read_to_string(Self::PATH).ok()?).map(|_| Watch)
    }
}

impl System for Watch {
    fn read(&mut self) -> Pressure {
        std::fs::read_to_string(Self::PATH)
            .ok()
            .and_then(|text| read(&text))
            .unwrap_or(Pressure::Calm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(some: f64, full: f64) -> String {
        format!(
            "some avg10={some:.2} avg60=0.00 avg300=0.00 total=1\n\
             full avg10={full:.2} avg60=0.00 avg300=0.00 total=1\n"
        )
    }

    #[test]
    fn a_stall_reads_as_how_much_of_the_machine_it_held() {
        assert_eq!(read(&file(0.0, 0.0)), Some(Pressure::Calm));
        assert_eq!(read(&file(4.9, 0.0)), Some(Pressure::Calm));
        assert_eq!(read(&file(6.0, 1.0)), Some(Pressure::Congested));
        assert_eq!(read(&file(40.0, 12.0)), Some(Pressure::Critical));
    }

    #[test]
    fn a_kernel_without_a_full_line_still_reads() {
        assert_eq!(
            read("some avg10=9.00 avg60=0.00 avg300=0.00 total=1\n"),
            Some(Pressure::Congested)
        );
    }

    #[test]
    fn anything_else_is_not_a_pressure_file() {
        assert_eq!(read(""), None);
        assert_eq!(read("some avg10=lots\n"), None);
    }
}
