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
//! The pool is laid after the engines load, and what an engine maps (its
//! weights, its driver's ranges, a Vulkan engine's recorded command
//! buffers, tens of thousands of maps) is laid at load and does not grow
//! after. So the rest of the process is left what it mapped, in bytes and
//! in maps, before its engines loaded ([`note_maps_before_engines`]): the
//! part of it that grows with serving, not a copy of all it holds.
//! Unrecorded, it is left as much again as it holds.

use anyhow::{Result, bail};
use std::sync::OnceLock;

/// What this process held before its engines loaded.
#[derive(Debug, Clone, Copy)]
struct BeforeEngines {
    /// Address space it mapped.
    mapped: u64,
    /// Memory maps it held, where they are counted.
    maps: Option<u64>,
}

static BEFORE_ENGINES: OnceLock<BeforeEngines> = OnceLock::new();

/// Records the address space and memory maps this process holds now,
/// before its engines load: what the pool leaves the rest of the process
/// to grow by. The first record stands; where the address space is not
/// read, nothing is recorded.
pub fn note_maps_before_engines() {
    if BEFORE_ENGINES.get().is_some() {
        return;
    }
    if let Ok(held) = held_now() {
        let _ = BEFORE_ENGINES.set(held);
    }
}

/// What this process holds now.
#[cfg(target_os = "macos")]
fn held_now() -> Result<BeforeEngines> {
    Ok(BeforeEngines {
        mapped: mapped_now()?,
        maps: None,
    })
}

/// What this process holds now.
#[cfg(not(target_os = "macos"))]
fn held_now() -> Result<BeforeEngines> {
    let space = read_address_space()?;
    Ok(BeforeEngines {
        mapped: space.mapped,
        maps: space.maps.map(|maps| maps.used),
    })
}

/// Keeps for the rest of the process what it held before its engines
/// loaded, or, unrecorded, all it holds now.
#[cfg_attr(
    not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        windows
    )),
    allow(dead_code)
)]
fn keep_for_the_rest(space: &mut AddressSpace) {
    let before = BEFORE_ENGINES.get();
    space.reserve = before.map_or(space.mapped, |before| before.mapped);
    if let Some(maps) = &mut space.maps {
        maps.reserve = before.and_then(|before| before.maps).unwrap_or(maps.used);
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
    /// What the rest of the process is left to grow by: what it mapped
    /// before its engines loaded (an engine's mappings are laid at load
    /// and do not grow), or, unrecorded, all it maps now.
    pub reserve: u64,
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
    /// the process its reserve ([`AddressSpace::reserve`]).
    pub fn pool_bytes(&self) -> u64 {
        let mut bytes = self
            .largest_free
            .min(self.free.saturating_sub(self.reserve));
        if let Some(limit) = self.limit {
            bytes = bytes.min(
                limit
                    .saturating_sub(self.mapped)
                    .saturating_sub(self.reserve),
            );
        }
        bytes
    }

    /// The address space the pool may take, with what it is reckoned from.
    pub fn bytes_shown(&self) -> String {
        let mut shown = format!(
            "{} (the largest free range {}; {} free, less {} kept for the rest of the process",
            brief(self.pool_bytes()),
            brief(self.largest_free),
            brief(self.free),
            brief(self.reserve),
        );
        if let Some(limit) = self.limit {
            shown += &format!(
                "; RLIMIT_AS {}, less {} mapped and the {} kept",
                brief(limit),
                brief(self.mapped),
                brief(self.reserve),
            );
        }
        shown.push(')');
        shown
    }

    /// Memory maps the pool's instances may split off, leaving the rest of
    /// the process its reserve ([`MapCount::reserve`]).
    pub fn pool_maps(&self) -> Option<u64> {
        self.maps.map(|maps| maps.pool())
    }

    /// The memory maps the pool may split off, with what they are
    /// reckoned from.
    pub fn maps_shown(&self) -> String {
        match self.maps {
            Some(maps) => format!(
                "{} memory maps (the cap {}, less {} held, less {} kept for the rest of the process)",
                maps.pool(),
                maps.max,
                maps.used,
                maps.reserve
            ),
            None => "no cap on memory maps".to_string(),
        }
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
                    space.bytes_shown(),
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
            space.bytes_shown(),
            space.maps_shown(),
        ),
        None => Ok(u32::try_from(fits).unwrap_or(u32::MAX)),
    }
}

/// `bytes` in the largest binary unit it reaches.
pub(crate) fn brief(bytes: u64) -> String {
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
        format!("{value:.2} {}", UNITS[unit])
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
    }
    keep_for_the_rest(&mut space);
    Ok(space)
}

/// The address space `/proc/self/maps` text `maps` leaves: the unmapped
/// ranges from `floor` up to the main stack, but for the one just below
/// the stack (its growth room, above where mmap places mappings). Ranges
/// in the kernel's half (x86_64's vsyscall page) are not the process's.
#[cfg(any(target_os = "linux", target_os = "android"))]
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
        reserve: 0,
        limit: None,
        maps: Some(MapCount {
            used: ranges.len() as u64,
            max: 0,
            reserve: 0,
        }),
    })
}

/// The lowest address 64-bit macOS maps: past the 4 GiB page zero.
#[cfg(target_os = "macos")]
const PAGE_ZERO: u64 = 1 << 32;
/// `MACH_VM_MAX_ADDRESS`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MAX_ADDRESS: u64 = 0x0000_7FFF_FE00_0000;
/// `MACH_VM_MAX_ADDRESS`.
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
const MAX_ADDRESS: u64 = 0x0000_7FFF_FFE0_0000;

/// What this process maps: its task's virtual size.
#[cfg(target_os = "macos")]
fn mapped_now() -> Result<u64> {
    runtime_sys::address::task_virtual_size()
        .map_err(|error| anyhow::anyhow!("proc_pidinfo(PROC_PIDTASKINFO): {error}"))
}

/// 64-bit macOS gives a process the addresses from its 4 GiB page zero to
/// `MACH_VM_MAX_ADDRESS`; what it maps is its task's virtual size.
#[cfg(target_os = "macos")]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    let mapped = mapped_now()?;
    let free = (MAX_ADDRESS - PAGE_ZERO).saturating_sub(mapped);
    // What the process maps does not lie in one place (the shared region,
    // the GPU's ranges, the weights), so the range free in one piece is
    // found, not reckoned; and, as `pool_bytes` leaves the rest of the
    // process its reserve, that range less the reserve: the rest's later
    // mappings may land in it.
    let mut space = AddressSpace {
        largest_free: largest_gap()?.min(free),
        free,
        mapped,
        reserve: 0,
        limit: address_space_limit(),
        maps: None,
    };
    keep_for_the_rest(&mut space);
    space.largest_free = space.largest_free.saturating_sub(space.reserve);
    Ok(space)
}

/// The largest range no mapping of this process lies in, down to a whole
/// GiB, read off its mappings.
#[cfg(target_os = "macos")]
fn largest_gap() -> Result<u64> {
    const GIB: u64 = 1 << 30;
    let (mut cursor, mut largest) = (PAGE_ZERO, 0u64);
    while cursor < MAX_ADDRESS {
        let mapping = runtime_sys::address::mapping_from(cursor)
            .map_err(|code| anyhow::anyhow!("mach_vm_region_recurse: {code}"))?;
        let Some(mapping) = mapping.filter(|mapping| mapping.size > 0) else {
            break;
        };
        largest = largest.max(mapping.address.min(MAX_ADDRESS).saturating_sub(cursor));
        cursor = cursor.max(mapping.address.saturating_add(mapping.size));
    }
    largest = largest.max(MAX_ADDRESS.saturating_sub(cursor));
    Ok(largest / GIB * GIB)
}

/// Windows: the process's user address space, region by region
/// (`VirtualQuery` from the bottom up to the top of user space, where it
/// answers nothing). Its free ranges lie between what the process's
/// images, heaps and drivers have reserved, so the one range a mapping may
/// take is the largest free region, not the free total (`GlobalMemoryStatusEx`
/// states the total only: a pool sized to it is one reservation of nearly
/// all 128 TiB, which no free region holds).
#[cfg(windows)]
pub(crate) fn read_address_space() -> Result<AddressSpace> {
    let mut space = AddressSpace {
        largest_free: 0,
        free: 0,
        mapped: 0,
        reserve: 0,
        limit: None,
        maps: None,
    };
    let mut address = 0usize;
    while let Some(region) = runtime_sys::address::region(address) {
        let bytes = region.size as u64;
        if region.free {
            space.free += bytes;
            space.largest_free = space.largest_free.max(bytes);
        } else {
            space.mapped += bytes;
        }
        match region.base.checked_add(region.size) {
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
    keep_for_the_rest(&mut space);
    Ok(space)
}

/// The largest range one `VirtualAlloc` reservation takes here, at most
/// `region` (the largest free region): Windows charges a reservation more
/// than its own range (the Control Flow Guard bitmap that covers it), so
/// the largest free region is more than one reservation may take. Read by
/// reserving and releasing, halving the gap to a page; nothing is
/// committed and nothing is held.
#[cfg(windows)]
fn largest_reservation(region: u64) -> u64 {
    let reserves = runtime_sys::address::reserves;
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
fn address_space_limit() -> Option<u64> {
    runtime_sys::address::limit()
}

/// The host's page size.
pub(crate) fn page_bytes() -> u64 {
    runtime_sys::address::page_bytes()
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

    /// The default 2048 on the phone is refused by name, with what it
    /// needs and what the process has; a pool of none is refused unset.
    #[test]
    fn an_oversized_count_is_refused_by_name() {
        let cost = default_layout().cost();
        let space = AddressSpace {
            largest_free: 400 * GIB,
            free: 480 * GIB,
            mapped: 30 * GIB,
            reserve: 30 * GIB,
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

    /// Windows: the range the pool may take is one this process can
    /// reserve, as wasmtime reserves its pool (one `VirtualAlloc`), and the
    /// free total is not one range.
    #[cfg(windows)]
    #[test]
    fn the_pool_s_range_is_one_windows_reserves() {
        let space = read_address_space().unwrap();
        assert!(space.mapped > 0);
        assert!(space.largest_free <= space.free);
        let bytes = space.pool_bytes();
        assert!(bytes > 0);
        assert!(
            runtime_sys::address::reserves(bytes),
            "reserving the pool's {} failed: {}",
            show(bytes),
            std::io::Error::last_os_error()
        );
    }

    /// The range read off the process's mappings is the one an address-only
    /// reservation takes, found here by halving to the GiB.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_largest_gap_is_what_a_reservation_takes() {
        let reserves = runtime_sys::address::reserves;
        let (mut fits, mut over) = (0u64, MAX_ADDRESS);
        while over - fits > GIB {
            let mid = (fits + (over - fits) / 2 / GIB * GIB).max(fits + GIB);
            if reserves(mid) {
                fits = mid;
            } else {
                over = mid;
            }
        }
        let gap = largest_gap().unwrap();
        assert!(
            gap.abs_diff(fits) <= GIB,
            "{} read, {} reserved",
            show(gap),
            show(fits)
        );
    }
}
