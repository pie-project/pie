//! What the kernel driver holds on the device for this process's queues:
//! each queue has a wave save area (CWSR, its control stack within it) and
//! a debugger area on every XCC, in device memory the runtime allocates
//! for the driver when the queue is made, sized as KFD sizes it
//! (`kfd_queue_ctx_save_restore_size`, `kfd_queue_acquire_buffers`). The
//! HIP runtime's free figure does not count them, so a load that sizes one
//! allocation to the card's free bytes holds them back by name.
//!
//! The area a queue takes is read from KFD's topology for the device. The
//! queues are counted by HIP's rule, in every PID namespace alike: one
//! utility queue, a hardware queue for each stream up to HIP's limit, and a
//! copy queue for each of the most non-null streams that ever copied at once
//! ([`warm_copies`]; HIP pools its copy queues by kind of copy and keeps
//! them past the streams that made them, so this counts at least as many
//! and a reload makes none). [`held`] reads KFD's list where it can. KFD's
//! own list of the process's queues ([`listed_queues`], readable from the
//! host's PID namespace only) agrees with it at every count the tests hold
//! it to.

use std::fs;
use std::path::{Path, PathBuf};

const TOPOLOGY: &str = "/sys/class/kfd/kfd/topology/nodes";
/// Where KFD lists each process it serves, a directory a pid.
pub const PROCESSES: &str = "/sys/class/kfd/kfd/proc";

/// HIP's limit on hardware queues a device, where `GPU_MAX_HW_QUEUES` does
/// not state one (CLR's default).
const HIP_HW_QUEUES: u32 = 4;

/// The device memory the driver holds for this process's queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveAreas {
    /// The queues, by HIP's rule.
    pub queues: u32,
    pub xcc: u32,
    /// Bytes a queue's wave save area takes on one XCC, its control stack
    /// within it (topology's `cwsr_size`).
    pub cwsr: u64,
    /// Bytes a queue's debugger area takes on one XCC: 32 a wave the XCC
    /// can hold, 64-aligned.
    pub debug: u64,
}

impl SaveAreas {
    /// What one queue holds, over every XCC.
    #[must_use]
    pub fn area(&self) -> u64 {
        ((self.cwsr + self.debug) * u64::from(self.xcc)).next_multiple_of(PAGE)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.queues) * self.area()
    }
}

impl core::fmt::Display for SaveAreas {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} MiB for {} queues' wave save areas ({} XCC x ({} save + {} debugger) bytes a \
             queue)",
            self.bytes() >> 20,
            self.queues,
            self.xcc,
            self.cwsr,
            self.debug,
        )
    }
}

/// The save areas the current device holds for this process now: its
/// queues as KFD lists them where that list is this process's to read, and
/// HIP's rule over every stream otherwise ([`save_areas`]), which counts
/// at least as many. Queues outlive the streams that made them and are
/// handed to the next, so a load after a dropped one holds no more.
pub fn held() -> Result<SaveAreas, String> {
    let ruled = save_areas(u32::MAX)?;
    Ok(listed_queues().map_or(ruled, |queues| SaveAreas { queues, ..ruled }))
}

/// The save areas the current device holds for this process by HIP's rule,
/// which has opened at least `streams` streams on it: one utility queue, a
/// hardware queue a stream up to `GPU_MAX_HW_QUEUES`, and the copy queues
/// [`warm_copies`] made.
pub fn save_areas(streams: u32) -> Result<SaveAreas, String> {
    let node = node_of_current_device()?;
    let read = |name: &str| -> Result<u64, String> {
        property(&node, name)?.ok_or_else(|| {
            format!(
                "KFD's topology for the device ({}) states no `{name}`",
                node.display()
            )
        })
    };
    let xcc = u32::try_from(read("num_xcc")?).map_err(|_| "num_xcc past u32".to_owned())?;
    let cwsr = read("cwsr_size")?;
    let debug = debug_bytes(
        read("gfx_target_version")?,
        read("simd_count")? / read("simd_per_cu")?.max(1) / u64::from(xcc.max(1)),
        read("array_count")? / read("simd_arrays_per_engine")?.max(1),
    )
    .ok_or_else(|| {
        format!(
            "KFD sizes no debugger area for this device ({}): its gfx version is past what \
             this crate knows",
            node.display()
        )
    })?;
    let limit = std::env::var("GPU_MAX_HW_QUEUES")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(HIP_HW_QUEUES);
    // Compute queues by HIP's rule, and the copy queues `warm_copies`'
    // streams made, which outlive them.
    let queues = 1 + streams.min(limit) + warmed();
    Ok(SaveAreas {
        queues,
        xcc,
        cwsr,
        debug,
    })
}

const PAGE: u64 = 4096;

/// The copy queues HIP's rule counts for [`warm_copies`]: the most non-null
/// streams it warmed at once (`most`), not the streams warmed now (`live`,
/// by address). HIP keeps its copy queues in a small pool by kind of copy,
/// kept past the streams that copied and handed to the next, so the rule
/// counts at least the queues it made, and a load after a dropped one adds
/// none.
struct Warmed {
    live: Vec<usize>,
    most: u32,
}

static WARMED: std::sync::Mutex<Warmed> = std::sync::Mutex::new(Warmed {
    live: Vec::new(),
    most: 0,
});

fn warmed_streams() -> std::sync::MutexGuard<'static, Warmed> {
    WARMED.lock().expect("warmed streams lock poisoned")
}

/// The copy queues the streams [`warm_copies`] warmed made, which the
/// process holds from then on.
#[must_use]
pub fn warmed() -> u32 {
    warmed_streams().most
}

/// `stream` is destroyed: the copy queue it used stays the process's, for
/// the next stream that copies to take.
pub fn forget(stream: super::sys::Stream) {
    warmed_streams().live.retain(|&live| live != stream.addr());
}

/// Makes each of `streams`' copy queue now: one host-to-device copy a stream
/// past HIP's blit threshold (`GPU_FORCE_BLIT_COPY_SIZE`, 16 KiB where unset;
/// a copy at or under it runs as a blit kernel and makes no queue), so the
/// queue a stream's first copy would make while serving, with its save
/// area, is made before the load sizes its pools. Device memory for the
/// copy's target is the caller's `scratch`, at least one byte past the
/// threshold.
///
/// # Safety
///
/// `scratch` is live device memory of at least [`blit_threshold`] + 1
/// bytes on the current device, unused by anything else until this
/// returns, and every stream in `streams` is a live stream of that device
/// (or the null stream).
pub unsafe fn warm_copies(
    streams: &[super::sys::Stream],
    scratch: *mut core::ffi::c_void,
) -> Result<(), String> {
    let bytes = blit_threshold() + 1;
    let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: a live out-parameter; freed below.
    let code = unsafe { super::rt::hipHostMalloc(&raw mut host, bytes, 0) };
    if code != super::sys::HIP_SUCCESS {
        return Err(format!("hipHostMalloc answered {code}"));
    }
    let mut outcome = Ok(());
    for &stream in streams {
        // SAFETY: `scratch` holds `bytes` (the caller's contract) and `host`
        // was just allocated with them; the stream is the caller's, live.
        let code = unsafe {
            super::rt::hipMemcpyAsync(
                scratch,
                host,
                bytes,
                super::sys::MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        // SAFETY: as above.
        let synced = unsafe { super::rt::hipStreamSynchronize(stream) };
        if code != super::sys::HIP_SUCCESS || synced != super::sys::HIP_SUCCESS {
            outcome = Err(format!("a stream's first copy answered {code} / {synced}"));
            break;
        }
        let mut warmed = warmed_streams();
        if !stream.is_null() && !warmed.live.contains(&stream.addr()) {
            warmed.live.push(stream.addr());
            warmed.most = warmed
                .most
                .max(u32::try_from(warmed.live.len()).unwrap_or(u32::MAX));
        }
    }
    // SAFETY: allocated above, freed once.
    unsafe { super::rt::hipHostFree(host) };
    outcome
}

/// HIP's blit threshold: a copy of more bytes runs on a copy queue.
#[must_use]
pub fn blit_threshold() -> usize {
    std::env::var("GPU_FORCE_BLIT_COPY_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(16 << 10, |kib| kib << 10)
}

/// A queue's debugger area on one XCC of `cus` compute units and
/// `arrays_per_engine` shader arrays an engine, as KFD sizes it: 32 bytes
/// for each wave it counts the XCC holding (`get_num_waves`), 64-aligned.
fn debug_bytes(gfx: u64, cus: u64, arrays: u64) -> Option<u64> {
    let waves = if gfx < 100_100 {
        (cus * 40).min(arrays * 512)
    } else if gfx < 120_500 {
        cus * 32
    } else if gfx <= 120_501 {
        cus * 64
    } else {
        return None;
    };
    Some((waves * 32).next_multiple_of(64))
}

/// The device memory KFD counts as this process's on the current device
/// (`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`): every allocation the
/// process holds there, the driver's own for its queues among them. `None`
/// where that list is not this process's (another PID namespace) or not
/// readable.
pub fn process_vram() -> Option<u64> {
    let node = node_of_current_device().ok()?;
    let gpu = fs::read_to_string(node.join("gpu_id")).ok()?;
    let path = Path::new(PROCESSES)
        .join(std::process::id().to_string())
        .join(format!("vram_{}", gpu.trim()));
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// KFD's list of this process's queues on the current device, compute and
/// copy alike, where that list is this process's to read (the host's PID
/// namespace): what the tests hold HIP's rule ([`save_areas`]) to.
pub fn listed_queues() -> Option<u32> {
    let node = node_of_current_device().ok()?;
    let gpu = fs::read_to_string(node.join("gpu_id")).ok()?;
    let gpu = gpu.trim();
    let dir = Path::new(PROCESSES)
        .join(std::process::id().to_string())
        .join("queues");
    let mut queues = 0u32;
    for entry in fs::read_dir(dir).ok()? {
        let queue = entry.ok()?.path();
        if fs::read_to_string(queue.join("gpuid"))
            .ok()
            .as_deref()
            .map(str::trim)
            == Some(gpu)
        {
            queues += 1;
        }
    }
    (queues > 0).then_some(queues)
}

/// The KFD topology node of the current device, matched by PCI location.
fn node_of_current_device() -> Result<PathBuf, String> {
    let (domain, location) = pci_location()?;
    let nodes = fs::read_dir(TOPOLOGY).map_err(|why| format!("{TOPOLOGY}: {why}"))?;
    for entry in nodes {
        let node = entry.map_err(|why| format!("{TOPOLOGY}: {why}"))?.path();
        if property(&node, "domain")? == Some(domain)
            && property(&node, "location_id")? == Some(location)
        {
            return Ok(node);
        }
    }
    Err(format!(
        "no KFD topology node under {TOPOLOGY} is the device at PCI {domain:04x}:{:02x}:{:02x}.{}",
        location >> 8,
        (location >> 3) & 0x1f,
        location & 7
    ))
}

/// The current device's PCI domain and KFD location id (bus << 8 | device
/// << 3 | function).
fn pci_location() -> Result<(u64, u64), String> {
    let mut device = 0;
    // SAFETY: writes one int.
    unsafe { super::rt::hipGetDevice(&raw mut device) };
    let mut bus = [0 as core::ffi::c_char; 32];
    // SAFETY: `bus` is a live buffer of the length passed.
    let code = unsafe {
        super::rt::hipDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as core::ffi::c_int, device)
    };
    if code != super::sys::HIP_SUCCESS {
        return Err(format!("hipDeviceGetPCIBusId answered {code}"));
    }
    // SAFETY: HIP wrote a NUL-terminated string into `bus`.
    let id = unsafe { core::ffi::CStr::from_ptr(bus.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    parse_bus_id(&id)
        .ok_or_else(|| format!("HIP's PCI bus id `{id}` is not domain:bus:device.function"))
}

fn parse_bus_id(id: &str) -> Option<(u64, u64)> {
    let mut parts = id.trim().split(':');
    let domain = u64::from_str_radix(parts.next()?, 16).ok()?;
    let bus = u64::from_str_radix(parts.next()?, 16).ok()?;
    let (device, function) = parts.next()?.split_once('.')?;
    let device = u64::from_str_radix(device, 16).ok()?;
    let function = u64::from_str_radix(function, 16).ok()?;
    Some((domain, bus << 8 | device << 3 | function))
}

/// A `name value` line of a topology node's properties.
fn property(node: &Path, name: &str) -> Result<Option<u64>, String> {
    let path = node.join("properties");
    let text = fs::read_to_string(&path).map_err(|why| format!("{}: {why}", path.display()))?;
    Ok(text.lines().find_map(|line| {
        let (key, value) = line.split_once(' ')?;
        (key == name).then(|| value.trim().parse().ok()).flatten()
    }))
}
