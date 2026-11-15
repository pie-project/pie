//! What Linux says about its memory: pressure stall information.
//!
//! `/proc/pressure/memory` states the share of time tasks stalled on
//! memory, system-wide: the `some` line, the share of time at least one task
//! stalled, as averages over the kernel's own windows (10, 60 and 300 s),
//! which it recomputes every [`KERNEL_PERIOD`]. The pressure read is `some`
//! over the shortest window, as a share: `avg10 / 100`. `some`, not `full`
//! (every task stalled at once): the yield is to the rest of the machine,
//! and `some` rises while other work is hurt but the machine still runs.
//! No window or debounce of the runtime's is laid over it.
//!
//! Rising pressure is told by a trigger the kernel arms on the file, where
//! it grants one; the kernel tells nothing when a stall ends, so while the
//! reading is above 0, and wherever no trigger is granted (a container may
//! refuse the file's write), the file is read once each [`KERNEL_PERIOD`],
//! the period the kernel updates it in.

#[cfg(target_os = "linux")]
use super::Reading;

/// How often the kernel recomputes its averages (`PSI_FREQ`,
/// `kernel/sched/psi.c`): a read sooner says nothing new.
pub const KERNEL_PERIOD: std::time::Duration = std::time::Duration::from_secs(2);

/// The `avg10` of the `kind` line of a pressure file.
fn avg10(text: &str, kind: &str) -> Option<f64> {
    text.lines()
        .find(|line| line.starts_with(kind))?
        .split_whitespace()
        .find_map(|field| field.strip_prefix("avg10="))?
        .parse()
        .ok()
}

/// The pressure a pressure file states, in `[0, 1]`; `None` for text that
/// is not one.
#[must_use]
pub fn read(text: &str) -> Option<f64> {
    Some((avg10(text, "some")? / 100.0).clamp(0.0, 1.0))
}

#[cfg(target_os = "linux")]
const PATH: &str = "/proc/pressure/memory";

/// Starts reading the system's memory pressure; `None` on a kernel without
/// PSI, which has no such file.
#[cfg(target_os = "linux")]
pub fn watch() -> Option<Reading> {
    let first = read(&std::fs::read_to_string(PATH).ok()?)?;
    let (tell, reading) = tokio::sync::watch::channel(first);
    let trigger = Trigger::arm();
    if trigger.is_none() {
        tracing::info!(
            "memory pressure: the kernel grants no PSI trigger here; {PATH} is read each {}s",
            KERNEL_PERIOD.as_secs()
        );
    }
    std::thread::Builder::new()
        .name("pie-psi".into())
        .spawn(move || follow(&tell, trigger))
        .ok()?;
    Some(reading)
}

/// Tells each reading that changed, for as long as anything listens.
#[cfg(target_os = "linux")]
fn follow(tell: &tokio::sync::watch::Sender<f64>, mut trigger: Option<Trigger>) {
    // Woken by the trigger: its stall may not be in the averages yet, so
    // the next read waits out one kernel period whatever this one says.
    let mut woken = false;
    loop {
        let Some(pressure) = std::fs::read_to_string(PATH).ok().and_then(|t| read(&t)) else {
            return;
        };
        tell.send_if_modified(|last| {
            let changed = *last != pressure;
            *last = pressure;
            changed
        });
        if tell.is_closed() {
            return;
        }
        match &trigger {
            Some(armed) if pressure == 0.0 && !woken => {
                woken = armed.wait();
                if !woken {
                    tracing::info!(
                        "memory pressure: the PSI trigger failed; {PATH} is read each period"
                    );
                    trigger = None;
                }
            }
            _ => {
                woken = false;
                std::thread::sleep(KERNEL_PERIOD);
            }
        }
    }
}

/// A PSI trigger on the file: the least stall the kernel takes within a
/// window of one [`KERNEL_PERIOD`] (the window an unprivileged trigger may
/// have) wakes [`Trigger::wait`].
#[cfg(target_os = "linux")]
struct Trigger(std::fs::File);

#[cfg(target_os = "linux")]
impl Trigger {
    fn arm() -> Option<Trigger> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(PATH)
            .ok()?;
        // The kernel takes the stall and the window in microseconds, the
        // string with its nul.
        let spec = format!("some 1 {}\0", KERNEL_PERIOD.as_micros());
        file.write_all(spec.as_bytes()).ok()?;
        Some(Trigger(file))
    }

    /// Waits for the trigger: `false` where the kernel says it failed.
    #[allow(unsafe_code)]
    fn wait(&self) -> bool {
        use std::os::fd::AsRawFd;
        let mut poll = libc::pollfd {
            fd: self.0.as_raw_fd(),
            events: libc::POLLPRI,
            revents: 0,
        };
        loop {
            // SAFETY: one pollfd the call reads and writes, its fd open for
            // as long as `self` is.
            let n = unsafe { libc::poll(&raw mut poll, 1, -1) };
            if n > 0 {
                return poll.revents & libc::POLLERR == 0;
            }
            if n < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                return false;
            }
        }
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
    fn the_pressure_is_the_share_of_the_last_ten_seconds_some_task_stalled() {
        assert_eq!(read(&file(0.0, 0.0)), Some(0.0));
        assert_eq!(read(&file(25.0, 3.0)), Some(0.25));
        assert_eq!(read(&file(100.0, 100.0)), Some(1.0));
    }

    #[test]
    fn the_full_line_says_nothing_of_it() {
        assert_eq!(read(&file(10.0, 0.0)), read(&file(10.0, 90.0)));
        assert_eq!(
            read("some avg10=9.00 avg60=0.00 avg300=0.00 total=1\n"),
            Some(0.09)
        );
    }

    #[test]
    fn anything_else_is_not_a_pressure_file() {
        assert_eq!(read(""), None);
        assert_eq!(read("some avg10=lots\n"), None);
    }
}
