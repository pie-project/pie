//! What macOS says about its memory.
//!
//! No system call says how much another process would like. The kernel's
//! pressure level and its paging counters say whether it is struggling, and
//! that is all [`Governor`](super::Governor) asks.
#![allow(unsafe_code)]

use std::mem::{MaybeUninit, size_of};

use super::{Pressure, System, refaulting};

const NORMAL: u32 = 1;
const WARN: u32 = 2;
const CRITICAL: u32 = 4;

/// `kern.memorystatus_vm_pressure_level`: what the dispatch source reports,
/// without a dispatch queue to run on.
fn pressure_level() -> u32 {
    let mut level: u32 = NORMAL;
    let mut size = size_of::<u32>();
    // SAFETY: the name is a nul-terminated literal and `level` holds `size`
    // bytes; the kernel writes at most that.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 { level } else { NORMAL }
}

#[derive(Clone, Copy, Default)]
struct Paging {
    evicted: u64,
    returned: u64,
}

fn paging(host: libc::mach_port_t) -> Option<Paging> {
    let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = (size_of::<libc::vm_statistics64>() / size_of::<libc::integer_t>()) as u32;
    // SAFETY: `count` states `stats`' size in the kernel's units; the call
    // writes no more, and answers how much it did.
    let status = unsafe {
        #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &raw mut count,
        )
    };
    if status != 0 {
        return None;
    }
    // SAFETY: the call succeeded, so the counters are written; those it did
    // not reach are the zeroes above.
    let stats = unsafe { stats.assume_init() };
    Some(Paging {
        evicted: stats.compressions + stats.swapouts,
        returned: stats.decompressions + stats.swapins,
    })
}

/// Reads the system a tick at a time: the paging counters only say anything
/// as a rate.
pub struct Watch {
    /// Each `mach_host_self` adds a reference to the port: one is taken.
    host: libc::mach_port_t,
    last: Option<Paging>,
}

impl Default for Watch {
    fn default() -> Self {
        Self {
            host: host_port(),
            last: None,
        }
    }
}

fn host_port() -> libc::mach_port_t {
    // SAFETY: no arguments; the call answers this task's host port.
    #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
    unsafe {
        libc::mach_host_self()
    }
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
        let host = host_port();
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
