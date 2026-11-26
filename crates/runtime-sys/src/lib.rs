#![allow(unsafe_code)]
use wasmtime::component::Component;
use wasmtime::{Engine, Precompiled};

/// A component serialized by `Engine::precompile_component`. Wasmtime checks
/// the artifact's header against this engine's version and settings and
/// refuses a mismatch.
pub fn deserialize_component(engine: &Engine, precompiled: &[u8]) -> anyhow::Result<Component> {
    if Engine::detect_precompiled(precompiled) != Some(Precompiled::Component) {
        anyhow::bail!("not a component precompiled by wasmtime");
    }
    // SAFETY: deserializing trusts the bytes to be wasmtime's own output, as
    // wasmtime cannot verify the code inside. They come from the host that
    // embeds this runtime (the page, in a browser), which is trusted as much
    // as the runtime itself: it supplies the language component's source too.
    unsafe { Component::deserialize(engine, precompiled) }.map_err(|e| anyhow::anyhow!("{e}"))
}

/// What macOS says about its memory.
#[cfg(target_os = "macos")]
pub mod macos {
    use std::mem::{MaybeUninit, size_of};

    /// `kern.memorystatus_vm_pressure_level`, or `None` if the kernel does
    /// not answer.
    pub fn pressure_level() -> Option<u32> {
        let mut level: u32 = 0;
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
        (status == 0).then_some(level)
    }

    /// The paging counters of `HOST_VM_INFO64`.
    #[derive(Clone, Copy, Debug)]
    pub struct VmCounters {
        pub compressions: u64,
        pub decompressions: u64,
        pub swapins: u64,
        pub swapouts: u64,
    }

    /// This task's host port: each one adds a reference to it.
    #[derive(Clone, Copy, Debug)]
    pub struct Host(libc::mach_port_t);

    impl Default for Host {
        fn default() -> Self {
            // SAFETY: no arguments; the call answers this task's host port.
            #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
            Host(unsafe { libc::mach_host_self() })
        }
    }

    impl Host {
        /// The host's paging counters, or `None` if it does not answer.
        pub fn vm_counters(self) -> Option<VmCounters> {
            let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
            let mut count =
                (size_of::<libc::vm_statistics64>() / size_of::<libc::integer_t>()) as u32;
            // SAFETY: `count` states `stats`' size in the kernel's units; the call
            // writes no more, and answers how much it did.
            let status = unsafe {
                #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
                libc::host_statistics64(
                    self.0,
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
            Some(VmCounters {
                compressions: stats.compressions,
                decompressions: stats.decompressions,
                swapins: stats.swapins,
                swapouts: stats.swapouts,
            })
        }
    }
}

/// What the operating system says of this process's address space, and
/// whether a range of it can be reserved: what the sandbox pool is sized by.
pub mod address {
    /// `RLIMIT_AS`, where the process is held to one.
    #[cfg(unix)]
    #[must_use]
    pub fn limit() -> Option<u64> {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `limit` is an `rlimit` to fill.
        if unsafe { libc::getrlimit(libc::RLIMIT_AS, &raw mut limit) } != 0
            || limit.rlim_cur == libc::RLIM_INFINITY
        {
            return None;
        }
        Some(limit.rlim_cur)
    }

    /// The host's page size; Windows pages are 4 KiB on every architecture
    /// it runs.
    #[must_use]
    pub fn page_bytes() -> u64 {
        #[cfg(unix)]
        {
            // SAFETY: sysconf reads a constant of the system.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page > 0 {
                return page as u64;
            }
        }
        4096
    }

    /// This task's virtual size (`proc_pidinfo(PROC_PIDTASKINFO)`), or why
    /// the kernel did not answer.
    #[cfg(target_os = "macos")]
    pub fn task_virtual_size() -> std::io::Result<u64> {
        // SAFETY: a zeroed `proc_taskinfo` is valid plain data.
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        // SAFETY: `info` is a `proc_taskinfo` of `size` bytes.
        let wrote = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                (&raw mut info).cast(),
                size,
            )
        };
        if wrote != size {
            return Err(std::io::Error::last_os_error());
        }
        Ok(info.pti_virtual_size)
    }

    /// A mapping of this task (the shared region is one).
    #[cfg(target_os = "macos")]
    #[derive(Debug, Clone, Copy)]
    pub struct Mapping {
        pub address: u64,
        pub size: u64,
    }

    /// This task's first mapping that ends above `address`
    /// (`mach_vm_region_recurse`, its top level), `None` past the last, or
    /// the kernel's code where it did not answer.
    #[cfg(target_os = "macos")]
    pub fn mapping_from(address: u64) -> Result<Option<Mapping>, i32> {
        /// `VM_REGION_SUBMAP_SHORT_INFO_COUNT_64`: the kernel's words of
        /// `vm_region_submap_short_info_64`.
        const WORDS: u32 = 12;
        const KERN_INVALID_ADDRESS: i32 = 1;
        unsafe extern "C" {
            fn mach_vm_region_recurse(
                task: libc::mach_port_t,
                address: *mut u64,
                size: *mut u64,
                depth: *mut u32,
                info: *mut i32,
                words: *mut u32,
            ) -> i32;
        }
        let (mut at, mut size, mut depth, mut words) = (address, 0u64, 0u32, WORDS);
        let mut info = [0i32; WORDS as usize];
        // SAFETY: every pointer is to a live local; `words` states `info`'s
        // length and the kernel writes no more; it reads this task's map.
        let status = unsafe {
            #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
            let task = libc::mach_task_self();
            mach_vm_region_recurse(
                task,
                &raw mut at,
                &raw mut size,
                &raw mut depth,
                info.as_mut_ptr(),
                &raw mut words,
            )
        };
        match status {
            0 => Ok(Some(Mapping { address: at, size })),
            KERN_INVALID_ADDRESS => Ok(None),
            code => Err(code),
        }
    }

    /// Whether an address-only reservation of `bytes` is taken now: reserved
    /// without access or backing at an address the system picks, and let go
    /// at once.
    #[cfg(unix)]
    #[must_use]
    pub fn reserves(bytes: u64) -> bool {
        let Ok(len) = usize::try_from(bytes) else {
            return false;
        };
        // SAFETY: an anonymous mapping with no access at an address the
        // kernel picks; nothing reads it, and it is unmapped at once.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                -1,
                0,
            );
            if at == libc::MAP_FAILED {
                return false;
            }
            libc::munmap(at, len);
        }
        true
    }

    #[cfg(windows)]
    mod win {
        /// `MEMORY_BASIC_INFORMATION` as 64-bit Windows lays it.
        #[repr(C)]
        pub struct MemoryBasicInformation {
            pub base_address: usize,
            pub allocation_base: usize,
            pub allocation_protect: u32,
            pub partition_id: u16,
            pub region_size: usize,
            pub state: u32,
            pub protect: u32,
            pub kind: u32,
        }

        pub const MEM_FREE: u32 = 0x1_0000;
        pub const MEM_RESERVE: u32 = 0x2000;
        pub const MEM_RELEASE: u32 = 0x8000;
        pub const PAGE_NOACCESS: u32 = 0x01;

        unsafe extern "system" {
            pub fn VirtualQuery(
                address: *const std::ffi::c_void,
                buffer: *mut MemoryBasicInformation,
                length: usize,
            ) -> usize;
            pub fn VirtualAlloc(
                address: *mut std::ffi::c_void,
                size: usize,
                kind: u32,
                protect: u32,
            ) -> *mut std::ffi::c_void;
            pub fn VirtualFree(address: *mut std::ffi::c_void, size: usize, kind: u32) -> i32;
            pub fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
            pub fn GetCurrentProcess() -> *mut std::ffi::c_void;
            pub fn K32GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                counters: *mut ProcessMemoryCountersEx,
                size: u32,
            ) -> i32;
        }

        /// `MEMORYSTATUSEX`.
        #[repr(C)]
        pub struct MemoryStatusEx {
            pub length: u32,
            pub memory_load: u32,
            pub total_phys: u64,
            pub avail_phys: u64,
            pub total_page_file: u64,
            pub avail_page_file: u64,
            pub total_virtual: u64,
            pub avail_virtual: u64,
            pub avail_extended_virtual: u64,
        }

        /// `PROCESS_MEMORY_COUNTERS_EX`.
        #[repr(C)]
        pub struct ProcessMemoryCountersEx {
            pub cb: u32,
            pub page_fault_count: u32,
            pub peak_working_set_size: usize,
            pub working_set_size: usize,
            pub quota_peak_paged_pool_usage: usize,
            pub quota_paged_pool_usage: usize,
            pub quota_peak_non_paged_pool_usage: usize,
            pub quota_non_paged_pool_usage: usize,
            pub pagefile_usage: usize,
            pub peak_pagefile_usage: usize,
            pub private_usage: usize,
        }
    }

    /// The commit the system can still give without growing its paging
    /// file (`ullAvailPageFile`, within the physical memory free,
    /// `ullAvailPhys`) and what this process has committed
    /// (`PrivateUsage`); `None` where either is not read.
    #[cfg(windows)]
    #[must_use]
    pub fn commit() -> Option<(u64, u64)> {
        // SAFETY: zeroed out-structs of the layouts the calls write, their
        // lengths stated; the process handle is the pseudo-handle, never
        // closed.
        unsafe {
            let mut status: win::MemoryStatusEx = std::mem::zeroed();
            status.length = std::mem::size_of::<win::MemoryStatusEx>() as u32;
            if win::GlobalMemoryStatusEx(&raw mut status) == 0 {
                return None;
            }
            let mut counters: win::ProcessMemoryCountersEx = std::mem::zeroed();
            let size = std::mem::size_of::<win::ProcessMemoryCountersEx>() as u32;
            counters.cb = size;
            if win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &raw mut counters, size) == 0
            {
                return None;
            }
            Some((
                status.avail_page_file.min(status.avail_phys),
                counters.private_usage as u64,
            ))
        }
    }

    /// One region of the process's user address space (`VirtualQuery`).
    #[cfg(windows)]
    #[derive(Debug, Clone, Copy)]
    pub struct Region {
        pub base: usize,
        pub size: usize,
        /// Nothing has reserved it (`MEM_FREE`).
        pub free: bool,
    }

    /// The region holding `address`, or `None` past the top of user space.
    #[cfg(windows)]
    #[must_use]
    pub fn region(address: usize) -> Option<Region> {
        // SAFETY: a zeroed `MEMORY_BASIC_INFORMATION` is valid.
        let mut region: win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery reads the region holding `address` into
        // `region`, a buffer of the length stated; it maps nothing.
        let written = unsafe {
            win::VirtualQuery(
                address as *const std::ffi::c_void,
                &raw mut region,
                std::mem::size_of::<win::MemoryBasicInformation>(),
            )
        };
        (written != 0 && region.region_size != 0).then_some(Region {
            base: region.base_address,
            size: region.region_size,
            free: region.state == win::MEM_FREE,
        })
    }

    /// Whether one `VirtualAlloc` reservation of `bytes` at no fixed
    /// address is taken now; released whole at once, committing nothing.
    #[cfg(windows)]
    #[must_use]
    pub fn reserves(bytes: u64) -> bool {
        // SAFETY: a reservation at no fixed address of `bytes`, released
        // whole at once; it commits nothing.
        unsafe {
            let base = win::VirtualAlloc(
                std::ptr::null_mut(),
                bytes as usize,
                win::MEM_RESERVE,
                win::PAGE_NOACCESS,
            );
            !base.is_null() && win::VirtualFree(base, 0, win::MEM_RELEASE) != 0
        }
    }
}
