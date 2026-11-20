//! What macOS says about its memory.
//!
//! No system call says how much another process would like. The kernel's
//! pressure level and its paging counters say whether it is struggling, and
//! that is all [`Governor`](super::Governor) asks.

use runtime_sys::macos::Host;

use super::{Pressure, System, refaulting};

const NORMAL: u32 = 1;
const WARN: u32 = 2;
const CRITICAL: u32 = 4;

/// `kern.memorystatus_vm_pressure_level`: what the dispatch source reports,
/// without a dispatch queue to run on.
fn pressure_level() -> u32 {
    runtime_sys::macos::pressure_level().unwrap_or(NORMAL)
}

#[derive(Clone, Copy, Default)]
struct Paging {
    evicted: u64,
    returned: u64,
}

fn paging(host: Host) -> Option<Paging> {
    let stats = host.vm_counters()?;
    Some(Paging {
        evicted: stats.compressions + stats.swapouts,
        returned: stats.decompressions + stats.swapins,
    })
}

/// Reads the system a tick at a time: the paging counters only say anything
/// as a rate.
#[derive(Default)]
pub struct Watch {
    /// Each `mach_host_self` adds a reference to the port: one is taken.
    host: Host,
    last: Option<Paging>,
}

impl System for Watch {
    fn read(&mut self) -> Pressure {
        let now = paging(self.host);
        let refault = match (self.last, now) {
            (Some(last), Some(now)) => refaulting(
                now.evicted.saturating_sub(last.evicted),
                now.returned.saturating_sub(last.returned),
            ),
            _ => false,
        };
        self.last = now.or(self.last);
        match pressure_level() {
            CRITICAL => Pressure::Critical,
            WARN => {
                tracing::debug!("the kernel says warn");
                Pressure::Congested
            }
            _ if refault => {
                tracing::debug!("pages are coming back");
                Pressure::Congested
            }
            _ => Pressure::Calm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_counters_read_and_only_rise() {
        let host = Host::default();
        let first = paging(host).expect("the host answers");
        let second = paging(host).expect("the host answers");
        assert!(second.evicted >= first.evicted);
        assert!(second.returned >= first.returned);
    }

    #[test]
    fn the_level_is_one_the_kernel_states() {
        assert!(matches!(pressure_level(), NORMAL | WARN | CRITICAL));
    }
}
