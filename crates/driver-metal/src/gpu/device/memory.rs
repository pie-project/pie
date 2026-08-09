//! What the machine will actually give, as a snapshot rather than a question.
//!
//! Two ceilings bound a model, and they are not the same question.
//!
//! `recommendedMaxWorkingSetSize` is a DEVICE property: on Apple silicon it
//! is a flat fraction of installed RAM and never moves. It reports the same
//! number to the first model of the day and to one starting on a box where
//! twenty gigabytes are already spoken for. Under unified memory that second
//! case is the one that fails.
//!
//! [`Memory::reclaimable`] is the other ceiling -- what the kernel can hand
//! back right now without swapping. It counts free, purgeable and evictable
//! pages and deliberately excludes wired ones, because wired pages cannot be
//! paged out and neither can memory a GPU context has already committed. That
//! exclusion is the point rather than an omission: a wedged context's pages
//! are charged to no live process and appear in no process listing, so the
//! only way to see them is to ask what is LEFT rather than what is used.
//!
//! # Why wired bytes are worth reporting at all
//!
//! They are the tell for a leaked GPU context, and the failure they precede
//! is not a slow machine. The window server cannot get memory to composite a
//! frame, blocks in the kernel on its own submit, misses its 120-second
//! watchdog, and takes the desktop down with it. The C++ shell records that
//! happening twice, once ten hours after the run that caused it.
//!
//! # A snapshot, not four static functions
//!
//! The C++ spells these as statics that reach the machine on every call, so
//! testing a refusal that only fires on a device the models do not fit needs
//! a process-wide override for each one -- and then a fifth predicate,
//! `device_working_set_is_forced`, because a forced working set paired with
//! the real machine's free memory compares two unrelated worlds and makes the
//! answer depend on whatever else is running.
//!
//! A snapshot has none of that. [`Memory::probe`] reads the machine once;
//! any other `Memory` is a struct literal. There is no hook to install, no
//! hook to forget to remove, and no way to build half an imaginary machine,
//! so the fifth predicate has nothing to warn about and does not exist.

use std::ffi::c_int;

use libc::{integer_t, mach_msg_type_number_t, mach_port_t, vm_size_t, vm_statistics64};
use objc2_metal::MTLDevice;

use crate::device::context::Context;

// `libc` marks `mach_host_self` deprecated in favour of a crate this
// workspace does not use, and does not publish `mach_port_deallocate` or
// `host_page_size` at all. Three declarations are cheaper than a dependency.
unsafe extern "C" {
    fn mach_host_self() -> mach_port_t;
    fn mach_task_self() -> mach_port_t;
    fn mach_port_deallocate(task: mach_port_t, name: mach_port_t) -> c_int;
    fn host_page_size(host: mach_port_t, out: *mut vm_size_t) -> c_int;
}

/// The page counts a reclaimable figure is computed from.
///
/// Separated from the `host_statistics64` call so the arithmetic -- which is
/// the part that has been wrong -- can be tested against numbers a machine
/// will not produce on demand.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pages {
    /// Pages on no queue at all.
    pub free: u64,
    /// Pages an application has volunteered to lose.
    pub purgeable: u64,
    /// Pages not recently touched.
    pub inactive: u64,
    /// File-backed read-ahead.
    pub speculative: u64,
    /// File-backed pages, on whichever queue.
    pub external: u64,
    /// Pages that cannot be paged out.
    pub wired: u64,
}

/// How many pages the kernel could hand back without swapping.
///
/// The queues -- active, inactive, speculative, wired -- partition non-free
/// memory and are disjoint. `external` (file-backed) and `purgeable` are
/// ATTRIBUTES cutting across them, not queues of their own, so `inactive` and
/// `external` overlap and cannot be added.
///
/// The C++ shell used to resolve that by dropping `external` and keeping only
/// the file pages parked as speculative, which throws away every clean file
/// page parked as ACTIVE -- and a clean file page is the cheapest memory on
/// the machine to reclaim, since it is dropped rather than written. On a box
/// that has just read sixty gigabytes of checkpoints, which is any box that
/// has loaded one model and is about to load another, that is gigabytes of
/// real headroom made invisible. Measured on a 32 GiB M1 Max with 15.73 GiB
/// free: 6.89 GiB inactive against 8.30 GiB file-backed, so 1.4 GiB was
/// hidden and the fit check refused a 10.9 GiB model.
///
/// `max` rather than a sum is the union bound that cannot over-count: both
/// sets live inside active + inactive + speculative, so their union is at
/// least the larger and at most the total, and only the larger can be claimed
/// without knowing the split -- which Mach does not report. `speculative` is
/// file-backed read-ahead and therefore already inside `external`; it joins
/// the inactive arm alone, where it is disjoint and would otherwise be lost.
#[must_use]
pub const fn reclaimable_pages(pages: &Pages) -> u64 {
    let evictable = if pages.inactive + pages.speculative > pages.external {
        pages.inactive + pages.speculative
    } else {
        pages.external
    };
    pages.free + pages.purgeable + evictable
}

/// One reading of what the machine will give.
///
/// Every field is bytes. A zero means the machine would not say, which is
/// distinct from "nothing available" only in that a caller should not refuse
/// on it -- see [`Memory::headroom`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Memory {
    /// What the device says it will hold resident.
    ///
    /// A device property, so it does not move and does not know what else is
    /// running.
    pub device_working_set: u64,

    /// What the kernel could hand back right now without swapping.
    pub reclaimable: u64,

    /// Pages that cannot be paged out.
    pub wired: u64,

    /// Physical memory installed.
    pub installed: u64,
}

impl Memory {
    /// Read the device and the kernel once.
    #[must_use]
    pub fn probe(context: &Context) -> Self {
        let device_working_set = context.device().recommendedMaxWorkingSetSize();
        let host = Host::open();
        let (pages, page_bytes) = host.vm_statistics().unwrap_or_default();
        Self {
            device_working_set,
            reclaimable: reclaimable_pages(&pages) * page_bytes,
            wired: pages.wired * page_bytes,
            installed: installed_bytes(),
        }
    }

    /// The tighter of the two ceilings, or `None` if neither could be read.
    ///
    /// Both are consulted because each is blind to what the other sees: the
    /// working set does not know what else is running, and the reclaimable
    /// figure does not know what this device will agree to hold. A zero is
    /// skipped rather than treated as a refusal -- a machine that will not
    /// answer is not a machine with no memory, and refusing on silence would
    /// turn a diagnostic into an outage.
    #[must_use]
    pub const fn ceiling(&self) -> Option<u64> {
        match (self.device_working_set, self.reclaimable) {
            (0, 0) => None,
            (0, v) | (v, 0) => Some(v),
            (a, b) if a < b => Some(a),
            (_, b) => Some(b),
        }
    }

    /// Whether `bytes` fits under [`ceiling`](Self::ceiling).
    ///
    /// True when nothing could be read, for the reason given there.
    #[must_use]
    pub const fn headroom(&self, bytes: u64) -> bool {
        match self.ceiling() {
            Some(limit) => bytes <= limit,
            None => true,
        }
    }

    /// Wired memory as a fraction of installed, or `None` if either is
    /// unknown.
    ///
    /// The number to watch across runs. It does not fall on its own when a
    /// context leaks, which is what makes it a tell rather than noise.
    #[must_use]
    pub fn wired_fraction(&self) -> Option<f64> {
        if self.installed == 0 || self.wired == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(self.wired as f64 / self.installed as f64)
    }
}

/// A borrowed send right on the host port, given back when dropped.
///
/// `mach_host_self` hands out a reference each call. The C++ shell calls it
/// three times per probe and returns none of them; on a process that probes
/// per fire that is a port reference leaked per fire.
struct Host(mach_port_t);

impl Host {
    fn open() -> Self {
        // SAFETY: `mach_host_self` takes no arguments and cannot fail.
        Self(unsafe { mach_host_self() })
    }

    /// The VM page counts and the page size, or `None` if the kernel refused.
    fn vm_statistics(&self) -> Option<(Pages, u64)> {
        let mut page: vm_size_t = 0;
        // SAFETY: `self.0` is a live host port and `page` is a valid out
        // pointer for the duration of the call.
        if unsafe { host_page_size(self.0, &raw mut page) } != 0 {
            return None;
        }

        // SAFETY: `vm_statistics64` is a plain C struct of integers, for
        // which an all-zero pattern is valid.
        let mut vm: vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count =
            (size_of::<vm_statistics64>() / size_of::<integer_t>()) as mach_msg_type_number_t;
        // SAFETY: the buffer is exactly `count` `integer_t`s long, which is
        // what `HOST_VM_INFO64` is defined to write.
        let status = unsafe {
            libc::host_statistics64(
                self.0,
                libc::HOST_VM_INFO64,
                (&raw mut vm).cast(),
                &mut count,
            )
        };
        if status != 0 {
            return None;
        }

        Some((
            Pages {
                free: u64::from(vm.free_count),
                purgeable: u64::from(vm.purgeable_count),
                inactive: u64::from(vm.inactive_count),
                speculative: u64::from(vm.speculative_count),
                external: u64::from(vm.external_page_count),
                wired: u64::from(vm.wire_count),
            },
            page as u64,
        ))
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a send right this value owns and will not use
        // again.
        unsafe { mach_port_deallocate(mach_task_self(), self.0) };
    }
}

/// Physical memory installed, or 0 if `sysctl` would not say.
fn installed_bytes() -> u64 {
    let mut installed: u64 = 0;
    let mut len = size_of::<u64>();
    // SAFETY: the name is a NUL-terminated literal, and the out buffer is
    // exactly `len` bytes.
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut installed).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 { installed } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement the C++ comment records, in pages of 16 KiB.
    #[test]
    fn a_clean_file_page_parked_as_active_is_not_thrown_away() {
        let pages = Pages {
            free: 0,
            purgeable: 0,
            inactive: 451_379, // 6.89 GiB
            speculative: 0,
            external: 543_949, // 8.30 GiB
            wired: 0,
        };
        assert_eq!(
            reclaimable_pages(&pages),
            543_949,
            "the file-backed arm is the larger one here and is what must win"
        );

        // The old spelling -- inactive plus speculative only -- hides the
        // difference, which was 1.4 GiB and refused a 10.9 GiB model.
        let hidden = (543_949 - 451_379) * 16_384;
        assert!((1..2).contains(&(hidden / (1 << 30))), "{hidden}");
    }

    #[test]
    fn the_two_arms_are_a_union_bound_and_never_a_sum() {
        let pages = Pages {
            free: 10,
            purgeable: 5,
            inactive: 100,
            speculative: 20,
            external: 40,
            wired: 999,
        };
        // Inactive wins here, and `external` is not added on top of it.
        assert_eq!(reclaimable_pages(&pages), 10 + 5 + 120);
        assert_ne!(reclaimable_pages(&pages), 10 + 5 + 120 + 40);

        // Speculative belongs to the inactive arm alone: it is already inside
        // `external`, so adding it there would double-count.
        let externals_win = Pages {
            external: 500,
            ..pages
        };
        assert_eq!(reclaimable_pages(&externals_win), 10 + 5 + 500);
    }

    #[test]
    fn wired_pages_are_never_reclaimable() {
        let pages = Pages {
            wired: u64::from(u32::MAX),
            ..Pages::default()
        };
        assert_eq!(reclaimable_pages(&pages), 0);
    }

    #[test]
    fn the_tighter_ceiling_wins_and_silence_is_not_a_refusal() {
        let both = Memory {
            device_working_set: 1000,
            reclaimable: 400,
            ..Memory::default()
        };
        assert_eq!(both.ceiling(), Some(400));
        assert!(both.headroom(400));
        assert!(!both.headroom(401));

        let device_only = Memory {
            reclaimable: 0,
            ..both
        };
        assert_eq!(device_only.ceiling(), Some(1000));

        let host_only = Memory {
            device_working_set: 0,
            ..both
        };
        assert_eq!(host_only.ceiling(), Some(400));

        let silent = Memory::default();
        assert_eq!(silent.ceiling(), None);
        assert!(
            silent.headroom(u64::MAX),
            "a machine that will not answer is not a machine with no memory"
        );
    }

    #[test]
    fn a_wired_fraction_needs_both_numbers() {
        let m = Memory {
            wired: 1 << 30,
            installed: 4 << 30,
            ..Memory::default()
        };
        assert_eq!(m.wired_fraction(), Some(0.25));
        assert_eq!(Memory { installed: 0, ..m }.wired_fraction(), None);
    }

    /// The one thing that has to be asked of the real machine: that the
    /// syscalls answer at all, and answer something possible.
    #[test]
    fn the_kernel_answers_and_its_answer_is_possible() {
        let host = Host::open();
        let (pages, page_bytes) = host.vm_statistics().expect("host_statistics64");
        assert!(
            page_bytes >= 4096 && page_bytes.is_power_of_two(),
            "{page_bytes}"
        );

        let installed = installed_bytes();
        assert!(installed >= 1 << 30, "{installed} bytes installed?");
        assert!(
            pages.wired * page_bytes < installed,
            "more memory is wired than is installed"
        );
        assert!(
            reclaimable_pages(&pages) * page_bytes <= installed,
            "more memory is reclaimable than is installed"
        );
    }

}
