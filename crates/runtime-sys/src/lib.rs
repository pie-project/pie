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
