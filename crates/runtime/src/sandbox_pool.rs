//! How many inferlet instances the wasmtime pool may hold: what one instance
//! reserves of the address space, what this process has to give, and the
//! count that fits.
//!
//! The pooling allocator reserves every slot up front, at `Engine::new`: a
//! linear memory slot (its reservation and guard), an execution stack (and
//! its guard page) and a component's tables per instance, one guard before
//! the memory slab. A pool larger than the address space fails that mmap;
//! the count is held to what fits, and a configured count that does not fit
//! is refused by name.
//!
//! The pool leaves the rest of the process as much again as it maps when
//! the pool is laid (the engines, their drivers and the weights are loaded
//! by then): what it maps at boot is the measure of what it maps beside the
//! pool.
//!
//! Memory maps are not reckoned so: an engine's maps (a Vulkan engine's
//! recorded command buffers, tens of thousands of them) are laid at load
//! and do not grow after, so the rest of the process is left the maps it
//! held before its engines loaded ([`note_maps_before_engines`]), the part
//! of it that grows with serving, not a copy of all it holds.

use anyhow::{Result, bail};
use std::sync::OnceLock;

/// The memory maps this process held before its engines loaded.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
static MAPS_BEFORE_ENGINES: OnceLock<u64> = OnceLock::new();

/// Records the memory maps this process holds now, before its engines
/// load: what the pool leaves the rest of the process to grow by. The
/// first record stands; where maps are not read, nothing is recorded.
pub fn note_maps_before_engines() {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Some(used) = std::fs::read_to_string("/proc/self/maps")
        .ok()
        .and_then(|maps| parse_maps(&maps, None).ok())
        .and_then(|space| space.maps)
        .map(|maps| maps.used)
    {
        let _ = MAPS_BEFORE_ENGINES.set(used);
    }
}

/// What the pool reserves: per instance, and once for the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolCost {
    /// Address space one instance reserves: its memory slot, its stack and
    /// its tables.
    pub instance_bytes: u64,
    /// Address space the pool reserves once: the guard before the first
    /// memory slot.
    pub pool_bytes: u64,
    /// Memory maps one live instance splits off: its stack's guard page and
    /// body, and its memory's image, accessible rest and reserved tail.
    pub instance_maps: u64,
}

impl PoolCost {
    /// What a pool of `instances` reserves.
    pub fn bytes(&self, instances: u64) -> u64 {
        self.instance_bytes
            .saturating_mul(instances)
            .saturating_add(self.pool_bytes)
    }
}

/// The wasmtime settings an instance's reservation is laid by.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolLayout {
    pub page_bytes: u64,
    /// `Config::memory_reservation`: the slot codegen assumes is mapped.
    pub memory_reservation: u64,
    /// `PoolingAllocationConfig::max_memory_size`.
    pub max_memory: u64,
    /// `Config::memory_guard_size`.
    pub memory_guard: u64,
    /// `Config::guard_before_linear_memory`: one guard before the slab.
    pub guard_before: bool,
    /// `Config::async_stack_size`: the pool's stack per instance, 0 for none.
    pub stack_bytes: u64,
    /// Tables one component instance may hold.
    pub tables_per_instance: u64,
    /// `PoolingAllocationConfig::table_elements`.
    pub table_elements: u64,
    /// The widest table element wasmtime lays a pooled table for: a
    /// function reference, a pointer.
    pub table_element_bytes: u64,
}

impl PoolLayout {
    /// What the pool reserves, laid as wasmtime 49 lays it with one
    /// stripe (no memory protection keys): a slot is the larger of the
    /// reservation and the memory maximum plus its guard; a stack is its
    /// size plus a guard page; a table is its elements, page-rounded.
    pub fn cost(&self) -> PoolCost {
        let page = self.page_bytes.max(1);
        let round = |bytes: u64| bytes.div_ceil(page).saturating_mul(page);
        let guard = round(self.memory_guard);
        let slot = round(self.memory_reservation.max(self.max_memory)).saturating_add(guard);
        let stack = if self.stack_bytes == 0 {
            0
        } else {
            round(self.stack_bytes).saturating_add(page)
        };
        let tables = round(self.table_elements.saturating_mul(self.table_element_bytes))
            .saturating_mul(self.tables_per_instance);
        PoolCost {
            instance_bytes: slot.saturating_add(stack).saturating_add(tables),
            pool_bytes: if self.guard_before { guard } else { 0 },
            instance_maps: if self.stack_bytes == 0 { 0 } else { 2 } + 3,
        }
    }
}

/// What this process has of its address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AddressSpace {
    /// The largest unmapped range one mapping may take.
    pub largest_free: u64,
    /// Every unmapped range a mapping may take.
    pub free: u64,
    /// What the process maps now.
    pub mapped: u64,
    /// The process's cap on its address space (`RLIMIT_AS`), if any.
    pub limit: Option<u64>,
    /// Memory maps, where the process has a cap on them.
    pub maps: Option<MapCount>,
}

/// This process's memory maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MapCount {
    /// What the process holds now.
    pub used: u64,
    /// The most it may hold (`vm.max_map_count`).
    pub max: u64,
    /// What the rest of the process is left to grow by: the maps it held
    /// before its engines loaded (an engine's maps are laid at load and do
    /// not grow), or, unrecorded, all it holds now.
    pub reserve: u64,
}

impl MapCount {
    /// Maps the pool may take: the cap less what is held and the reserve.
    pub fn pool(&self) -> u64 {
        self.max
            .saturating_sub(self.used)
            .saturating_sub(self.reserve)
    }
}

impl std::fmt::Display for MapCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (the cap {}, less {} held, less {} kept for the rest of the process)",
            self.pool(),
            self.max,
            self.used,
            self.reserve
        )
    }
}

impl AddressSpace {
    /// Address space the pool may take: one range, leaving the rest of
    /// the process as much again as it maps now.
    pub fn pool_bytes(&self) -> u64 {
        let rest = self.mapped;
        let mut bytes = self.largest_free.min(self.free.saturating_sub(rest));
        if let Some(limit) = self.limit {
            bytes = bytes.min(limit.saturating_sub(self.mapped.saturating_add(rest)));
        }
        bytes
    }

    /// Memory maps the pool's instances may split off, leaving the rest of
    /// the process its reserve ([`MapCount::reserve`]).
    pub fn pool_maps(&self) -> Option<u64> {
        self.maps.map(|maps| maps.pool())
    }

    /// The most instances a pool laid at `cost` holds here.
    pub fn fits(&self, cost: &PoolCost) -> u64 {
        let by_bytes = self
            .pool_bytes()
            .checked_sub(cost.pool_bytes)
            .map_or(0, |bytes| bytes / cost.instance_bytes.max(1));
        match self.pool_maps() {
            Some(maps) if cost.instance_maps > 0 => by_bytes.min(maps / cost.instance_maps),
            _ => by_bytes,
        }
    }
}

/// The pool's instance count: `configured` where it fits, else refused by
/// name; unset, the most that fits. `space` is `Err` where the address
/// space could not be read: a configured count is then taken as stated
/// (wasmtime refuses by name what it cannot map), an unset one refused.
pub(crate) fn size_pool(
    configured: Option<u32>,
    space: Result<AddressSpace>,
    cost: &PoolCost,
) -> Result<u32> {
    let space = match (space, configured) {
        (Ok(space), _) => space,
        (Err(error), Some(n)) => {
            tracing::warn!(
                "[sandbox] max_instances = {n} taken as stated: this process's address space is not read ({error:#})"
            );
            return Ok(n);
        }
        (Err(error), None) => bail!(
            "[sandbox] max_instances is unset and this process's address space is not read \
             ({error:#}); state it"
        ),
    };
    let fits = space.fits(cost);
    match configured {
        Some(n) if u64::from(n) <= fits => Ok(n),
        Some(n) => {
            let needs = cost.bytes(u64::from(n));
            if needs > space.pool_bytes() {
                bail!(
                    "[sandbox] max_instances = {n} needs {} of address space; this process has {} \
                     (it holds {fits} instances of {} each)",
                    show(needs),
                    show(space.pool_bytes()),
                    show(cost.instance_bytes),
                );
            }
            bail!(
                "[sandbox] max_instances = {n} needs {} memory maps; this process has {} \
                 (it holds {fits} instances)",
                u64::from(n).saturating_mul(cost.instance_maps),
                space
                    .maps
                    .map_or_else(|| "0".to_string(), |maps| maps.to_string()),
            );
        }
        None if fits == 0 => bail!(
            "the sandbox pool holds no instance: one needs {} of address space and {} memory maps; \
             this process has {} and {}",
            show(cost.bytes(1)),
            cost.instance_maps,
            show(space.pool_bytes()),
            space
                .maps
                .map_or_else(|| "no cap on maps".to_string(), |maps| maps.to_string()),
        ),
        None => Ok(u32::try_from(fits).unwrap_or(u32::MAX)),
    }
}

/// `bytes` in the largest binary unit it reaches, with the exact count.
pub(crate) fn show(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {} ({bytes} bytes)", UNITS[unit])
    }
}

/// This process's address space, read off `/proc/self/maps`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    use anyhow::Context;
    let maps = std::fs::read_to_string("/proc/self/maps").context("read /proc/self/maps")?;
    // The lowest address a mapping may take; unread, no range below the
    // lowest mapping is counted.
    let floor = std::fs::read_to_string("/proc/sys/vm/mmap_min_addr")
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok());
    let max_maps = std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        // Unread (Android may not let an app read it): the kernel's own
        // default, `DEFAULT_MAX_MAP_COUNT`.
        .unwrap_or(u64::from(u16::MAX) - 5);
    let mut space = parse_maps(&maps, floor)?;
    space.limit = address_space_limit();
    if let Some(maps) = &mut space.maps {
        maps.max = max_maps;
        maps.reserve = MAPS_BEFORE_ENGINES.get().copied().unwrap_or(maps.used);
    }
    Ok(space)
}

/// The address space `/proc/self/maps` text `maps` leaves: the unmapped
/// ranges from `floor` up to the main stack, but for the one just below
/// the stack (its growth room, above where mmap places mappings). Ranges
/// in the kernel's half (x86_64's vsyscall page) are not the process's.
#[cfg(any(target_os = "linux", target_os = "android", test))]
pub(crate) fn parse_maps(maps: &str, floor: Option<u64>) -> Result<AddressSpace> {
    let mut ranges = Vec::new();
    for line in maps.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = line.split_whitespace();
        let range = fields.next().unwrap_or_default();
        let parsed = range.split_once('-').and_then(|(start, end)| {
            Some((
                u64::from_str_radix(start, 16).ok()?,
                u64::from_str_radix(end, 16).ok()?,
            ))
        });
        let Some((start, end)) = parsed else {
            bail!("/proc/self/maps: no address range in {line:?}");
        };
        if start >= 1 << 63 {
            continue;
        }
        let stack = fields.nth(4).is_some_and(|name| name == "[stack]");
        ranges.push((start, end, stack));
    }
    ranges.sort_unstable();
    let Some(&(lowest, _, _)) = ranges.first() else {
        bail!("/proc/self/maps lists no mapping");
    };
    let ceiling = ranges
        .iter()
        .find(|(_, _, stack)| *stack)
        .map_or_else(|| ranges.iter().map(|r| r.1).max().unwrap_or(0), |r| r.0);
    let mapped = ranges.iter().map(|(start, end, _)| end - start).sum();
    // The unmapped ranges below the stack, the one just below it dropped.
    let mut gaps = Vec::new();
    let mut cursor = floor.unwrap_or(lowest).min(lowest);
    for &(start, end, _) in ranges.iter().take_while(|r| r.0 < ceiling) {
        if start > cursor {
            gaps.push(start - cursor);
        }
        cursor = cursor.max(end);
    }
    let (largest_free, free) = gaps.iter().fold((0, 0), |(largest, free), &gap| {
        (largest.max(gap), free + gap)
    });
    Ok(AddressSpace {
        largest_free,
        free,
        mapped,
        limit: None,
        maps: Some(MapCount {
            used: ranges.len() as u64,
            max: 0,
            reserve: 0,
        }),
    })
}

/// 64-bit macOS gives a process the addresses from its 4 GiB page zero to
/// `MACH_VM_MAX_ADDRESS`; what it maps is its task's virtual size.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    const PAGE_ZERO: u64 = 1 << 32;
    #[cfg(target_arch = "aarch64")]
    const MAX_ADDRESS: u64 = 0x0000_7FFF_FE00_0000;
    #[cfg(not(target_arch = "aarch64"))]
    const MAX_ADDRESS: u64 = 0x0000_7FFF_FFE0_0000;
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
        bail!(
            "proc_pidinfo(PROC_PIDTASKINFO): {}",
            std::io::Error::last_os_error()
        );
    }
    let mapped = info.pti_virtual_size;
    let free = (MAX_ADDRESS - PAGE_ZERO).saturating_sub(mapped);
    // What the process maps does not lie in one place (the shared region,
    // the GPU's ranges, the weights), so the range free in one piece is
    // found, not reckoned; and, as `pool_bytes` leaves the rest of the
    // process as much again as it maps now, that range less what it maps:
    // the GPU's and the driver's later mappings may land in it.
    Ok(AddressSpace {
        largest_free: largest_reservation(free).saturating_sub(mapped),
        free,
        mapped,
        limit: address_space_limit(),
        maps: None,
    })
}

/// The largest range, to the GiB and at most `upper`, an address-only
/// reservation takes now: each try reserved without access or backing and
/// let go at once.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn largest_reservation(upper: u64) -> u64 {
    const GIB: u64 = 1 << 30;
    let reserves = |bytes: u64| -> bool {
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
    };
    let (mut fits, mut over) = (0u64, upper.saturating_add(GIB));
    while over - fits > GIB {
        let mid = fits + (over - fits) / 2 / GIB * GIB;
        let mid = mid.max(fits + GIB);
        if reserves(mid) {
            fits = mid;
        } else {
            over = mid;
        }
    }
    fits
}

/// Windows: the process's user address space, region by region
/// (`VirtualQuery` from the bottom up to the top of user space, where it
/// answers nothing). Its free ranges lie between what the process's
/// images, heaps and drivers have reserved, so the one range a mapping may
/// take is the largest free region, not the free total (`GlobalMemoryStatusEx`
/// states the total only: a pool sized to it is one reservation of nearly
/// all 128 TiB, which no free region holds).
#[cfg(windows)]
#[allow(unsafe_code)]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    /// `MEMORY_BASIC_INFORMATION` as 64-bit Windows lays it.
    #[repr(C)]
    struct MemoryBasicInformation {
        base_address: usize,
        _allocation_base: usize,
        _allocation_protect: u32,
        _partition_id: u16,
        region_size: usize,
        state: u32,
        _protect: u32,
        _kind: u32,
    }
    /// `MEM_FREE`: a region nothing has reserved.
    const MEM_FREE: u32 = 0x1_0000;
    unsafe extern "system" {
        fn VirtualQuery(
            address: *const std::ffi::c_void,
            buffer: *mut MemoryBasicInformation,
            length: usize,
        ) -> usize;
    }
    let mut space = AddressSpace {
        largest_free: 0,
        free: 0,
        mapped: 0,
        limit: None,
        maps: None,
    };
    let mut address = 0usize;
    loop {
        // SAFETY: a zeroed `MEMORY_BASIC_INFORMATION` is valid.
        let mut region: MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery reads the region holding `address` into
        // `region`, a buffer of the length stated; it maps nothing.
        let written = unsafe {
            VirtualQuery(
                address as *const std::ffi::c_void,
                &mut region,
                std::mem::size_of::<MemoryBasicInformation>(),
            )
        };
        if written == 0 || region.region_size == 0 {
            break;
        }
        let bytes = region.region_size as u64;
        if region.state == MEM_FREE {
            space.free += bytes;
            space.largest_free = space.largest_free.max(bytes);
        } else {
            space.mapped += bytes;
        }
        match region.base_address.checked_add(region.region_size) {
            Some(next) => address = next,
            None => break,
        }
    }
    if space.free == 0 {
        bail!(
            "VirtualQuery read no free region: {}",
            std::io::Error::last_os_error()
        );
    }
    space.largest_free = largest_reservation(space.largest_free);
    Ok(space)
}

/// The largest range one `VirtualAlloc` reservation takes here, at most
/// `region` (the largest free region): Windows charges a reservation more
/// than its own range (the Control Flow Guard bitmap that covers it), so
/// the largest free region is more than one reservation may take. Read by
/// reserving and releasing, halving the gap to a page; nothing is
/// committed and nothing is held.
#[cfg(windows)]
#[allow(unsafe_code)]
fn largest_reservation(region: u64) -> u64 {
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    unsafe extern "system" {
        fn VirtualAlloc(
            address: *mut std::ffi::c_void,
            size: usize,
            kind: u32,
            protect: u32,
        ) -> *mut std::ffi::c_void;
        fn VirtualFree(address: *mut std::ffi::c_void, size: usize, kind: u32) -> i32;
    }
    let reserves = |bytes: u64| {
        // SAFETY: a reservation at no fixed address of `bytes`, released
        // whole at once; it commits nothing.
        unsafe {
            let base = VirtualAlloc(
                std::ptr::null_mut(),
                bytes as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            );
            !base.is_null() && VirtualFree(base, 0, MEM_RELEASE) != 0
        }
    };
    if reserves(region) {
        return region;
    }
    let (mut fits, mut fails) = (0, region);
    while fails - fits > page_bytes() {
        let mid = fits + (fails - fits) / 2;
        if reserves(mid) {
            fits = mid;
        } else {
            fails = mid;
        }
    }
    fits
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    bail!("no reading of the address space on this platform")
}

/// `RLIMIT_AS`, where the process is held to one.
#[cfg(unix)]
#[allow(unsafe_code)]
fn address_space_limit() -> Option<u64> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is an `rlimit` to fill.
    if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut limit) } != 0
        || limit.rlim_cur == libc::RLIM_INFINITY
    {
        return None;
    }
    Some(limit.rlim_cur as u64)
}

/// The host's page size.
#[allow(unsafe_code)]
pub(crate) fn page_bytes() -> u64 {
    #[cfg(unix)]
    {
        // SAFETY: sysconf reads a constant of the system.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page > 0 {
            return page as u64;
        }
    }
    // Windows pages are 4 KiB on every architecture it runs.
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn maps(used: u64, max: u64, reserve: u64) -> MapCount {
        MapCount { used, max, reserve }
    }

    /// wasmtime's 64-bit defaults: a 4 GiB reservation, a 32 MiB guard
    /// before and after, a 2 MiB stack, 16 tables of 20 000 elements.
    fn default_layout() -> PoolLayout {
        PoolLayout {
            page_bytes: 4096,
            memory_reservation: 4 * GIB,
            max_memory: 4 * GIB,
            memory_guard: 32 * MIB,
            guard_before: true,
            stack_bytes: 2 * MIB,
            tables_per_instance: 16,
            table_elements: 20_000,
            table_element_bytes: 8,
        }
    }

    /// The memory slab the phone failed to map: 2048 slots of 4 GiB and
    /// a 32 MiB guard each, one guard before them, 0x81002000000 bytes.
    #[test]
    fn the_memory_slab_is_laid_as_wasmtime_lays_it() {
        let layout = PoolLayout {
            stack_bytes: 0,
            tables_per_instance: 0,
            ..default_layout()
        };
        assert_eq!(layout.cost().bytes(2048), 0x810_0200_0000);
        let cost = default_layout().cost();
        // A slot, a stack and its guard page, 16 tables of 160 KiB.
        assert_eq!(
            cost.instance_bytes,
            4 * GIB + 32 * MIB + 2 * MIB + 4096 + 16 * 163_840
        );
        assert_eq!(cost.pool_bytes, 32 * MIB);
        assert_eq!(cost.instance_maps, 5);
    }

    /// A 39-bit phone (512 GiB): the count fits in what is free after the
    /// process keeps as much again as it maps, and in one range.
    #[test]
    fn the_count_fits_the_address_space() {
        let cost = default_layout().cost();
        let space = AddressSpace {
            largest_free: 400 * GIB,
            free: 480 * GIB,
            mapped: 30 * GIB,
            limit: None,
            maps: Some(maps(2_000, 65_530, 2_000)),
        };
        // 480 - 30 = 450 GiB free beyond the rest; one range of 400 holds it.
        assert_eq!(space.pool_bytes(), 400 * GIB);
        let fits = space.fits(&cost);
        assert!(cost.bytes(fits) <= space.pool_bytes());
        assert!(cost.bytes(fits + 1) > space.pool_bytes());
        assert_eq!(fits, 99);
        assert_eq!(size_pool(None, Ok(space), &cost).unwrap(), 99);
        assert_eq!(size_pool(Some(64), Ok(space), &cost).unwrap(), 64);

        // RLIMIT_AS holds it to the cap less twice what is mapped.
        let capped = AddressSpace {
            limit: Some(100 * GIB),
            ..space
        };
        assert_eq!(capped.pool_bytes(), 40 * GIB);
        assert_eq!(capped.fits(&cost), 9);

        // Memory maps hold it where the address space does not.
        let maps = AddressSpace {
            largest_free: 100_000 * GIB,
            free: 100_000 * GIB,
            maps: Some(maps(2_000, 12_000, 2_000)),
            ..space
        };
        assert_eq!(maps.fits(&cost), (12_000 - 4_000) / 5);
    }

    /// The default 2048 on the phone is refused by name, with what it
    /// needs and what the process has; a pool of none is refused unset.
    #[test]
    fn an_oversized_count_is_refused_by_name() {
        let cost = default_layout().cost();
        let space = AddressSpace {
            largest_free: 400 * GIB,
            free: 480 * GIB,
            mapped: 30 * GIB,
            limit: None,
            maps: Some(maps(2_000, 65_530, 2_000)),
        };
        let error = size_pool(Some(2048), Ok(space), &cost)
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("[sandbox] max_instances = 2048 needs 8.07 TiB"),
            "{error}"
        );
        assert!(error.contains("this process has 400.00 GiB"), "{error}");

        let few_maps = AddressSpace {
            maps: Some(maps(2_000, 4_100, 2_000)),
            ..space
        };
        let error = size_pool(Some(64), Ok(few_maps), &cost)
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with(
                "[sandbox] max_instances = 64 needs 320 memory maps; this process has 100 \
                 (the cap 4100, less 2000 held, less 2000 kept for the rest of the process)"
            ),
            "{error}"
        );

        let full = AddressSpace {
            largest_free: GIB,
            ..space
        };
        let error = size_pool(None, Ok(full), &cost).unwrap_err().to_string();
        assert!(
            error.starts_with("the sandbox pool holds no instance"),
            "{error}"
        );

        // Unread, an unset count is refused and a stated one taken.
        let unread = || Err(anyhow::anyhow!("no maps"));
        assert!(size_pool(None, unread(), &cost).is_err());
        assert_eq!(size_pool(Some(7), unread(), &cost).unwrap(), 7);
    }

    /// The phone after the Vulkan engine's arm walk: ~51k maps held, most
    /// of them the engine's recorded command buffers, laid at load and not
    /// grown after. The rest of the process is left what it held before
    /// the engine loaded, not a copy of all it holds; doubled, the 14k
    /// free maps were read as none.
    #[test]
    fn the_maps_reserve_is_what_the_process_held_before_its_engines() {
        let cost = default_layout().cost();
        // Address space wide enough that the maps bind.
        let phone = AddressSpace {
            largest_free: 100_000 * GIB,
            free: 100_000 * GIB,
            mapped: 30 * GIB,
            limit: None,
            maps: Some(maps(51_000, 65_530, 1_200)),
        };
        assert_eq!(phone.pool_maps(), Some(65_530 - 51_000 - 1_200));
        assert_eq!(phone.fits(&cost), (65_530 - 51_000 - 1_200) / 5);
        assert_eq!(
            size_pool(None, Ok(phone), &cost).unwrap(),
            (65_530 - 51_000 - 1_200) / 5
        );

        // Where the reserve is all it holds, the maps do bind, and the
        // refusal names the cap, what is held and the reserve.
        let unrecorded = AddressSpace {
            maps: Some(maps(51_000, 65_530, 51_000)),
            ..phone
        };
        assert_eq!(unrecorded.pool_maps(), Some(0));
        let error = size_pool(None, Ok(unrecorded), &cost)
            .unwrap_err()
            .to_string();
        assert!(
            error.ends_with(
                "and 0 (the cap 65530, less 51000 held, less 51000 kept for the rest of the process)"
            ),
            "{error}"
        );
    }

    /// An Android arm64 process with 39-bit addresses: the ranges below
    /// the stack count, the stack's growth room does not.
    #[test]
    fn maps_are_read_up_to_the_stack() {
        let maps = "\
5800000000-5800100000 r-xp 00000000 fd:00 1 /system/bin/pie
5800100000-5800200000 rw-p 00000000 00:00 0 [anon:libc_malloc]
7000000000-7000400000 rw-p 00000000 00:00 0
7fc0000000-7fc0800000 r--p 00000000 fd:00 2 /apex/libc.so
7ffff00000-7ffff21000 rw-p 00000000 00:00 0 [stack]
";
        let space = parse_maps(maps, Some(0x8000)).unwrap();
        assert_eq!(space.mapped, 0x100000 * 2 + 0x400000 + 0x800000 + 0x21000);
        // Below the binary, between the heap and the mmap range, and up to
        // the libraries; not the range up to the stack.
        let gaps = [
            0x58_0000_0000 - 0x8000,
            0x70_0000_0000 - 0x58_0020_0000,
            0x7f_c000_0000 - 0x70_0040_0000,
        ];
        assert_eq!(space.free, gaps.iter().sum::<u64>());
        assert_eq!(space.largest_free, gaps[0]);
        assert_eq!(space.maps.map(|m| m.used), Some(5));
    }

    /// x86_64's vsyscall page sits in the kernel's half; it is not the
    /// process's, and the vdso above the stack bounds nothing.
    #[test]
    fn the_kernel_half_is_not_the_process_s() {
        let maps = "\
555555554000-555555556000 r-xp 00000000 08:01 1 /usr/bin/pie
7ffff7fc0000-7ffff7fc4000 r--p 00000000 00:00 0 [vvar]
7ffffffde000-7ffffffff000 rw-p 00000000 00:00 0 [stack]
ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0 [vsyscall]
";
        let space = parse_maps(maps, Some(0x10000)).unwrap();
        assert_eq!(space.maps.map(|m| m.used), Some(3));
        // Below the binary, and from it up to the vvar below the stack.
        assert_eq!(
            space.free,
            (0x5555_5555_4000 - 0x10000) + (0x7fff_f7fc_0000 - 0x5555_5555_6000)
        );
    }

    /// Windows: the range the pool may take is one this process can
    /// reserve, as wasmtime reserves its pool (one `VirtualAlloc`), and the
    /// free total is not one range.
    #[cfg(windows)]
    #[test]
    #[allow(unsafe_code)]
    fn the_pool_s_range_is_one_windows_reserves() {
        const MEM_RESERVE: u32 = 0x2000;
        const MEM_RELEASE: u32 = 0x8000;
        const PAGE_NOACCESS: u32 = 0x01;
        unsafe extern "system" {
            fn VirtualAlloc(
                address: *mut std::ffi::c_void,
                size: usize,
                kind: u32,
                protect: u32,
            ) -> *mut std::ffi::c_void;
            fn VirtualFree(address: *mut std::ffi::c_void, size: usize, kind: u32) -> i32;
        }
        let space = read_address_space().unwrap();
        assert!(space.mapped > 0);
        assert!(space.largest_free <= space.free);
        let bytes = space.pool_bytes();
        assert!(bytes > 0);
        // SAFETY: a reservation with no fixed address, released below; it
        // commits nothing.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                bytes as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            )
        };
        assert!(
            !base.is_null(),
            "reserving the pool's {} failed: {}",
            show(bytes),
            std::io::Error::last_os_error()
        );
        // SAFETY: `base` is the reservation made above, released whole.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }
}
