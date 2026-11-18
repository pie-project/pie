//! The host kv tier as its governor steers it. Its ceiling is what the host
//! can give the load (`engine::host::host_kv_ceiling`, stated as its chunks
//! by the engine); its target follows the host's memory pressure by the
//! device's rule and `yield` key:
//!
//! ```text
//! target = floor + (ceiling - floor) * (1 - yield * pressure)
//! ```
//!
//! The tier commits as it fills, so a target is a cap on the chunks it
//! opens: no chunk at or past the target is opened from then on
//! ([`crate::store::slots::SlotPool::set_ceiling`]). A tier holding chunks
//! past its target with no room to give them back tells the policy
//! `tier-pressure` on the host, as the device's governor does on the
//! device; the policy decides what leaves (PLEX order: idle cache first).
//! Once none past the target is open, the engine gives the OS back their
//! pages ([`::engine::Engine::release_host_chunks`]): the tier's resident
//! memory comes down with its target, and nothing touches those chunks
//! again until the target rises past them.

use super::{Governor, Growth, Limits, PoolTarget, Reader, Retarget, Signal, Source, TICK, Yield};
use crate::policy::pressure::{self, Tier};
use crate::store::shared::Shared;

/// One engine's host tier: its shared pool of host chunks.
pub struct HostTier {
    pool: Shared,
    /// The engine gave back the pages of every chunk from this one on, and
    /// the cap has not risen past it since: none of them holds a page.
    returned: Option<u32>,
}

impl HostTier {
    #[must_use]
    pub const fn new(pool: Shared) -> Self {
        Self {
            pool,
            returned: None,
        }
    }

    /// After a turn: the chunk from which the engine should give the OS
    /// back its pages, where the cap fell under what was given back last
    /// (or nothing was) and every chunk from the cap on is quiet (none
    /// open, none owed to a fire). A cap risen past what was given back
    /// moves it up to the cap: the chunks under it may open, and commit,
    /// again; those from it on still hold no page.
    pub fn to_return(&mut self) -> Option<u32> {
        let pool = self.pool.lock().unwrap();
        let slots = pool.pool();
        let (cap, total) = (slots.ceiling(), slots.total_chunks());
        decide_return(&mut self.returned, cap, total, |chunk| slots.chunk_quiet(chunk))
    }
}

/// [`HostTier::to_return`]'s rule, over the cap, the tier's chunks and
/// whether a chunk is quiet.
fn decide_return(
    returned: &mut Option<u32>,
    cap: u32,
    total: u32,
    quiet: impl Fn(u32) -> bool,
) -> Option<u32> {
    // A cap risen past what went back: the chunks under it may open (and
    // commit) again, those from it on still hold no page.
    if returned.is_some_and(|from| cap > from) {
        *returned = Some(cap);
    }
    if cap >= total || returned.is_some_and(|from| from <= cap) {
        return None;
    }
    if !(cap..total).all(quiet) {
        return None;
    }
    *returned = Some(cap);
    Some(cap)
}

impl PoolTarget for HostTier {
    /// A chunk: the tier opens and commits a chunk at a time.
    fn segment_bytes(&self) -> u64 {
        self.pool.lock().unwrap().pool().chunk_bytes()
    }

    /// None: the tier may be empty, every suspended sequence going to the
    /// disk.
    fn floor(&self) -> u64 {
        0
    }

    /// Its chunks: what the host could give the load when it was seated.
    fn hard_limit(&self) -> u64 {
        let pool = self.pool.lock().unwrap();
        u64::from(pool.pool().total_chunks()) * pool.pool().chunk_bytes()
    }

    /// The chunks it has opened.
    fn held(&self) -> u64 {
        let pool = self.pool.lock().unwrap();
        u64::from(pool.pool().state_chunks()) * pool.pool().chunk_bytes()
    }

    fn growth(&self) -> Growth {
        Growth::Elastic
    }

    /// Opens no chunk at or past `segments`; short where a chunk past it is
    /// still open.
    fn retarget(&mut self, segments: u64) -> Retarget {
        let mut pool = self.pool.lock().unwrap();
        let slots = pool.pool_mut();
        let cap = u32::try_from(segments)
            .unwrap_or(u32::MAX)
            .min(slots.total_chunks());
        slots.set_ceiling(cap);
        let past = (cap..slots.total_chunks())
            .filter(|&chunk| slots.chunk_shape(chunk).is_some())
            .count();
        if past == 0 {
            Retarget::Reached
        } else {
            Retarget::NoRoom {
                held: u64::from(slots.state_chunks()),
            }
        }
    }
}

/// The gauge entry the host tier's governor of `engine` speaks under:
/// apart from the device's ([`super::steered::source`]) and the planner's.
#[must_use]
pub const fn source(engine: usize) -> usize {
    usize::MAX / 2 - engine
}

/// One turn: the rule's target for `pressure`, the tier moved toward it,
/// and what it could not give back, as a signal.
pub fn turn(governor: &mut Governor, tier: &mut impl PoolTarget, pressure: f64) -> Signal {
    let target = governor.tick(pressure);
    let segments = governor.limits().segments(target);
    match tier.retarget(segments) {
        // Moving toward it on its own: no pressure for the policy, and an
        // earlier one eases (the device governor's rule).
        Retarget::Reached | Retarget::Moving { .. } => Signal::Eased,
        Retarget::NoRoom { held } => Signal::TierPressure {
            held_bytes: held * governor.limits().segment_bytes,
            target_bytes: target,
        },
    }
}

/// Steers `engine`'s host tier by `yields` from now on. With yield 0 the
/// target is the ceiling and nothing is read, so no loop runs.
pub fn start(engine: usize, pool: Shared, yields: Yield) {
    let tier = HostTier::new(pool);
    let limits = Limits::of(&tier);
    if !yields.yields() {
        tracing::info!(
            engine,
            ceiling_mib = limits.hard_limit >> 20,
            "engine.memory.yield = 0: the host kv tier's target is its ceiling; the system's \
             memory pressure is not read"
        );
        return;
    }
    let reader = match Source::of_this_system() {
        Ok(source) => Reader::new(source),
        Err(why) => {
            tracing::info!(
                engine,
                ceiling_mib = limits.hard_limit >> 20,
                %why,
                "this system states no memory pressure: the host kv tier's target is its ceiling"
            );
            return;
        }
    };
    tracing::info!(
        engine,
        yields = yields.share(),
        source = reader.source().name(),
        ceiling_mib = limits.hard_limit >> 20,
        "the host kv tier's target follows the system's memory pressure: floor + (ceiling - \
         floor) * (1 - yield * pressure)"
    );
    crate::rt::spawn(watch(engine, tier, Governor::new(limits, yields), reader));
}

async fn watch(engine: usize, mut tier: HostTier, mut governor: Governor, mut reader: Reader) {
    let mut interval = crate::rt::time::interval(TICK);
    let mut short = false;
    loop {
        interval.tick().await;
        match turn(&mut governor, &mut tier, reader.read()) {
            Signal::TierPressure {
                held_bytes,
                target_bytes,
            } => {
                short = true;
                pressure::raise(
                    Tier::Host,
                    source(engine),
                    held_bytes.saturating_sub(target_bytes).max(1),
                );
            }
            Signal::Eased if short => {
                short = false;
                pressure::relieve(Tier::Host, source(engine));
            }
            Signal::Eased => {}
        }
        // Nothing past the target is open: their pages go back to the OS.
        if let Some(from) = tier.to_return() {
            let op = crate::scheduler::worker::PoolOp::ReleaseHostChunks(from);
            if let Err(why) = crate::scheduler::pool(engine, op).await {
                tracing::warn!(
                    engine,
                    from,
                    %why,
                    "the host kv tier's pages past its target could not be given back"
                );
                tier.returned = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tier of `chunks` chunks of `chunk` bytes, its target followed with
    /// `yields`.
    fn governed(chunks: u64, chunk: u64, yields: f64) -> Governor {
        Governor::new(
            Limits {
                floor: 0,
                hard_limit: chunks * chunk,
                segment_bytes: chunk,
            },
            Yield::new(yields).unwrap(),
        )
    }

    /// Yield 0.5 at pressure 1 targets the midpoint of the floor (none) and
    /// the ceiling; yield 0 or no pressure, the ceiling.
    #[test]
    fn half_the_yield_at_full_pressure_targets_the_midpoint() {
        let chunk = 4 << 20;
        let mut governor = governed(1000, chunk, 0.5);
        assert_eq!(governor.tick(1.0), 500 * chunk);
        assert_eq!(governor.limits().segments(governor.target()), 500);
        assert_eq!(governor.tick(0.0), 1000 * chunk);
        let mut held = governed(1000, chunk, 0.0);
        assert_eq!(held.tick(1.0), 1000 * chunk);
    }

    /// A tier that reaches its target eases; one holding chunks past it
    /// says by how much, for the policy's `tier-pressure` on the host.
    #[test]
    fn a_tier_past_its_target_says_what_it_holds() {
        struct Tier {
            chunks: u64,
            open: u64,
            cap: u64,
        }
        impl PoolTarget for Tier {
            fn segment_bytes(&self) -> u64 {
                1
            }
            fn floor(&self) -> u64 {
                0
            }
            fn hard_limit(&self) -> u64 {
                self.chunks
            }
            fn held(&self) -> u64 {
                self.open
            }
            fn growth(&self) -> Growth {
                Growth::Elastic
            }
            fn retarget(&mut self, segments: u64) -> Retarget {
                self.cap = segments;
                if self.open > segments {
                    Retarget::NoRoom { held: self.open }
                } else {
                    Retarget::Reached
                }
            }
        }
        let mut governor = Governor::new(
            Limits {
                floor: 0,
                hard_limit: 100,
                segment_bytes: 1,
            },
            Yield::new(0.5).unwrap(),
        );
        let mut tier = Tier {
            chunks: 100,
            open: 80,
            cap: 100,
        };
        assert_eq!(
            turn(&mut governor, &mut tier, 1.0),
            Signal::TierPressure {
                held_bytes: 80,
                target_bytes: 50
            }
        );
        assert_eq!(tier.cap, 50);
        tier.open = 40;
        assert_eq!(turn(&mut governor, &mut tier, 1.0), Signal::Eased);
    }
}

#[cfg(test)]
mod return_tests {
    use super::*;

    /// The rule: once the cap falls under what was given back (or nothing
    /// was), and every chunk from the cap on is quiet, the chunks from the
    /// cap on go back, once; a chunk still owed holds it; a cap risen past
    /// what went back counts only the chunks from the new cap as given
    /// back, so a later fall gives back what the rise let open.
    #[test]
    fn chunks_past_a_falling_cap_go_back_once_each_fall() {
        let mut returned = None;
        assert_eq!(decide_return(&mut returned, 8, 8, |_| true), None, "at the ceiling: nothing");
        assert_eq!(decide_return(&mut returned, 5, 8, |chunk| chunk != 6), None, "chunk 6 owed");
        assert_eq!(decide_return(&mut returned, 5, 8, |_| true), Some(5));
        assert_eq!(decide_return(&mut returned, 5, 8, |_| true), None, "once");
        assert_eq!(decide_return(&mut returned, 3, 8, |_| true), Some(3), "a further fall");
        assert_eq!(decide_return(&mut returned, 6, 8, |_| true), None, "a rise gives nothing back");
        assert_eq!(returned, Some(6), "chunks 3..6 may open again; 6.. still hold nothing");
        assert_eq!(decide_return(&mut returned, 4, 8, |_| true), Some(4), "a fall after the rise");
    }

    /// The tier's own memory as an engine holds it: one anonymous mapping
    /// of its chunks, committed as copies write it, and
    /// `Engine::release_host_chunks` as an engine answers it.
    #[allow(unsafe_code)]
    struct HostMemory {
        at: *mut u8,
        chunk: usize,
        chunks: usize,
    }

    #[allow(unsafe_code)]
    impl HostMemory {
        fn new(chunk: usize, chunks: usize) -> Self {
            // SAFETY: a fresh private anonymous mapping, unmapped on drop.
            let at = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    chunk * chunks,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(at, libc::MAP_FAILED);
            Self { at: at.cast(), chunk, chunks }
        }

        /// A move to the host writes chunk `chunk`.
        fn write(&mut self, chunk: usize) {
            // SAFETY: inside the mapping.
            unsafe { std::ptr::write_bytes(self.at.add(chunk * self.chunk), 0xa5, self.chunk) };
        }

        fn release_host_chunks(&mut self, from: u32) -> u64 {
            let from = from as usize;
            // SAFETY: the tier's own memory; the runtime opens none of these
            // chunks before writing them again.
            let given = unsafe {
                ::engine::host::give_back(self.at.add(from * self.chunk), (self.chunks - from) * self.chunk)
            };
            given as u64
        }

        /// Its resident bytes (Linux), or the process's footprint (macOS:
        /// pages given back as reusable leave the footprint, not the map).
        fn resident(&self) -> u64 {
            #[cfg(target_os = "linux")]
            {
                // SAFETY: a query of the page size.
                let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
                let len = self.chunk * self.chunks;
                let mut vec = vec![0u8; len.div_ceil(page)];
                // SAFETY: one byte a page of the live mapping.
                assert_eq!(unsafe { libc::mincore(self.at.cast(), len, vec.as_mut_ptr().cast()) }, 0);
                (vec.iter().filter(|&&v| v & 1 != 0).count() * page) as u64
            }
            #[cfg(target_vendor = "apple")]
            {
                let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
                // SAFETY: `info` is the flavor asked for.
                let done = unsafe {
                    libc::proc_pid_rusage(
                        std::process::id() as i32,
                        libc::RUSAGE_INFO_V4,
                        (&raw mut info).cast(),
                    )
                };
                assert_eq!(done, 0);
                info.ri_phys_footprint
            }
        }
    }

    #[allow(unsafe_code)]
    impl Drop for HostMemory {
        fn drop(&mut self) {
            // SAFETY: the mapping `new` made, once.
            unsafe { libc::munmap(self.at.cast(), self.chunk * self.chunks) };
        }
    }

    /// A pressure file at 1, yield 1: the target falls to nothing, under
    /// the chunks the tier holds. While a chunk past it is open the tier
    /// says so (the policy's `tier-pressure`) and gives nothing back; once
    /// the policy has let them go, the engine gives the OS back the pages
    /// of every chunk past the target, and the tier's resident memory
    /// comes down with it.
    #[test]
    fn a_pressure_file_drives_the_target_under_held_and_the_tier_s_pages_go_back() {
        let mut facts = crate::store::shared::tests::fixed_facts();
        facts.host_chunks = 8;
        let host = crate::store::shared::Tiers::of(&facts)
            .unwrap()
            .host
            .expect("a host tier");
        let (chunk, shape, spc) = {
            let pool = host.lock().unwrap();
            let shape = (0..facts.shapes.len() as u16)
                .find(|&s| pool.pool().shape(s).is_some_and(|f| f.slots_per_chunk >= 1))
                .expect("a shape");
            let spc = pool.pool().shape(shape).unwrap().slots_per_chunk;
            (pool.pool().chunk_bytes(), shape, spc)
        };
        let holder = crate::store::slots::Holder::WorkingSet(1);
        // Six chunks' worth of suspended state lands on the host.
        let slots = host
            .lock()
            .unwrap()
            .pool_mut()
            .acquire(holder, shape, u64::from(spc) * 6)
            .expect("room on the host");
        let mut memory = HostMemory::new(usize::try_from(chunk).unwrap(), 8);
        let mut opened: Vec<u32> = slots.iter().map(|slot| slot.chunk).collect();
        opened.dedup();
        for &c in &opened {
            memory.write(c as usize);
        }
        let full = memory.resident();

        let dir = std::env::temp_dir().join(format!("pie-host-return-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("pressure");
        std::fs::write(&file, "1.0").unwrap();
        let mut reader = Reader::new(Source::File(file));
        let mut tier = HostTier::new(host.clone());
        let mut governor = Governor::new(Limits::of(&tier), Yield::new(1.0).unwrap());

        // Held past the target: the policy is told, nothing goes back.
        assert!(matches!(
            turn(&mut governor, &mut tier, reader.read()),
            Signal::TierPressure { .. }
        ));
        assert_eq!(tier.to_return(), None, "its chunks are open");

        // The policy lets the state go (the disk takes it): the tier eases,
        // and every chunk past the target goes back to the OS.
        host.lock().unwrap().pool_mut().release(holder, &slots, 0);
        assert!(matches!(turn(&mut governor, &mut tier, reader.read()), Signal::Eased));
        let from = tier.to_return().expect("chunks past the target go back");
        assert_eq!(from, 0, "the target is nothing at full pressure and yield 1");
        let given = memory.release_host_chunks(from);
        assert_eq!(given, chunk * 8, "every chunk's pages");
        let after = memory.resident();
        assert!(
            full.saturating_sub(after) >= chunk * opened.len() as u64 * 3 / 4,
            "the tier's resident memory comes down: {full} -> {after}"
        );
        assert_eq!(tier.to_return(), None, "once");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
