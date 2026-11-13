//! Warm fires at load (`[engine] prewarm = true`): the fire shapes a server meets
//! first are traced, compiled (or loaded from the disk cache, where a warm
//! machine finds them) and run once before the first request, as engine-cuda
//! arms its bodies (`serve/arming.rs`), so no request waits on a compile.
//!
//! The ladder, over the canonical shapes `fire_inner` pads fires to:
//! - a decode at every lane rung, at every page bound from the page floor to
//!   the context's, as many lanes as the pool seats at half that bound;
//! - a prefill at every row rung from the prefill floor (16), over 1 lane
//!   (one that fills its rung, and one that leaves a padding lane) and, from
//!   64 rows, 4 and 8;
//! - a prefill beside a decode at every lane rung, at every row rung from
//!   256 (such a fire pads its prefill lanes to the mixed host rung).
//!
//! Nothing runs: each shape is traced, and the new programs compile side by
//! side.
//!
//! Lanes are synthetic (zero tokens, as `dry.rs` builds them) in the plain
//! text classes; they share pages, since what they write is never read. The
//! slots they touch are opened again afterwards.
#![allow(unsafe_code)]

use poem_ir::Request;

use super::{Lane, Seated, Shell};
use crate::error::Result;

/// One synthetic lane: its word, its rows, and the pages it holds before
/// them (a decode lane at a page bound holds all but one slot of them).
#[derive(Clone, Copy)]
struct Want {
    word: u64,
    rows: u32,
    held_pages: u32,
}

impl Shell {
    /// Fires the warm ladder; answers how many fires ran.
    pub fn prewarm(&mut self) -> Result<usize> {
        let facts = &self.trace().facts;
        let decode = facts.word(&Request::new(1, false));
        let prefill = facts.word(&Request::new(2, false));
        let classes = &self.compiled.classes;
        let plain = |shell: &Shell, word: u64| {
            classes.class_of(word & classes.mask).is_some_and(|class| {
                !shell.masked.contains(class)
                    && !shell.wide.contains(class)
                    && !shell.corrected.contains(class)
            })
        };
        if !plain(self, decode) || !plain(self, prefill) || self.patch_seat.is_some() {
            return Ok(0);
        }
        let buckets = self.budgets.tokens.buckets.clone();
        let pad = self.pad;
        // The widest fire the runtime sends (the top rung is padding room).
        let widest = match buckets.as_slice() {
            [.., below, top] if pad.room > 0 && top - below == pad.room => *below,
            [.., top] => *top,
            [] => return Ok(0),
        };
        let lanes_cap = self
            .budgets
            .tokens
            .max_lanes
            .saturating_sub(pad.mixed_host_rung)
            .min(self.held.len() as u32);
        let cap = pad.rung(self.paging().pages_per_slot.max(1));
        let pool = self.paging().pages();
        let fresh = |word, rows| Want {
            word,
            rows,
            held_pages: 0,
        };
        let hybrid = self.has_state();
        let mut shapes: Vec<Vec<Want>> = Vec::new();
        let mut pages = pad.rung(pad.page_floor).min(cap);
        while pages <= cap {
            // A lane at this bound holds more than half of it.
            let seats = u32::try_from(pool / u64::from(pages / 2).max(1)).unwrap_or(u32::MAX);
            // and the rung the most lanes it seats pad to (Qwen3.6-35B-A3B's
            // v5e pool seats 126: its c256 decodes ran 128 lanes, compiled
            // mid-run)
            let most = poem_exec::fire::rung_of(&buckets, lanes_cap.min(seats.max(1)));
            for &n in buckets.iter().filter(|&&n| n <= most) {
                let lane = Want {
                    word: decode,
                    rows: 1,
                    held_pages: pages,
                };
                shapes.push(vec![lane; n as usize]);
            }
            pages = pad.next(pages);
        }
        // A prefill pads its lanes to 16, 32 or 64 by its rows (`canonical_pads`),
        // or to 16 where the load holds fewer lanes than that: one lane at
        // every rung and every lane count its rows can take there.
        let max_lanes = self.budgets.tokens.max_lanes;
        let mut prev = 0;
        for &b in buckets.iter().filter(|&&b| b <= widest) {
            for (w, lo, hi) in [
                (pad.mixed_host_rung, 2, 480),
                (32, 481, 960),
                (64, 961, u32::MAX),
            ] {
                let w = if w <= max_lanes {
                    w
                } else {
                    pad.mixed_host_rung
                };
                let rows = (b + 1).saturating_sub(w).min(hi);
                if b >= 64 && w <= max_lanes && rows >= lo && rows + w > prev + 1 {
                    shapes.push(vec![fresh(prefill, rows)]);
                }
            }
            prev = b;
        }
        for &b in buckets
            .iter()
            .filter(|&&b| (pad.prefill_floor.max(2)..=widest).contains(&b))
        {
            shapes.push(vec![fresh(prefill, b)]);
            shapes.push(vec![fresh(prefill, b - 1)]);
            for n in [4u32, 8, 16, 32, 64] {
                // A hybrid's prompts, cut at their last whole page, fire many
                // and short; so do a load's whose lanes stop short of a wide
                // fire's 64, from 2048 rows (v6e gpt-oss c32: 16 prompts of
                // 2893 rows).
                let narrow = max_lanes < 64 && n == 32 && b >= 2048 && n <= max_lanes;
                let shown = if hybrid {
                    b / n >= 2
                } else {
                    n <= 8 && b >= 64
                };
                if (shown && n <= lanes_cap) || (!hybrid && narrow) {
                    // `n - 1` lanes and the padding lane the rest takes (two
                    // or three lanes pad to four).
                    shapes.push(vec![fresh(prefill, b / n); (n - 1) as usize]);
                }
            }
        }
        // A hybrid's prompts filling the widest rows at every lane rung: `n - 1`
        // lanes above fall a rung short of it (7 of 288 rows land 2048 rows,
        // not 2304), and at a `max_forward_tokens` between the powers of two
        // the prompts that fill it fire there (v6e Qwen3.6-35B-A3B pf256 at
        // 2304: four 555-row prompts and their pads, 8 lanes, traced 812 ms a
        // round, compiled 33 s on a fresh cache).
        if hybrid {
            for n in [4u32, 8, 16, 32, 64] {
                let k = n - 1;
                if n <= lanes_cap && widest / n >= 2 {
                    let rows = widest - 1;
                    let mut lanes = vec![fresh(prefill, rows / k); k as usize];
                    lanes[0].rows += rows % k;
                    shapes.push(lanes);
                }
            }
        }
        // A hybrid's full fire: its prompts fill the widest rows and the lanes
        // they pad to push it onto the top rung (v6e Qwen3.8-27B mini20 c32:
        // 12 prompts of 2045 rows compiled mid-run, 12.7 s).
        let top = *buckets.last().unwrap_or(&widest);
        if hybrid && top > widest {
            for n in [4u32, 8, 16, 32, 64] {
                let k = n / 2 + 1;
                if n <= lanes_cap && widest / k >= 2 {
                    let mut lanes = vec![fresh(prefill, widest / k); k as usize];
                    lanes[0].rows += widest % k;
                    shapes.push(lanes);
                }
            }
        }
        // Prefills beside decodes (`MIXED_HOST_RUNG` prefill lanes): every
        // decode lane rung at every row rung from 256 it fits in, and at the
        // padding rung, which a full fire's decode pads overflow into.
        let host = pad.mixed_host_rung;
        // A hybrid fires a prompt's tail behind its last whole page beside
        // the decodes: its mixed fires take every row rung.
        let least = if hybrid { pad.prefill_floor } else { 256 };
        // (A fire that prefills pads its decodes to `pad.rung`: a half rung of
        // the ladder is a decode-only fire's.)
        for &d in buckets
            .iter()
            .filter(|&&d| d <= pad.rung(lanes_cap) && pad.rung(d) == d)
        {
            for &b in buckets.iter().filter(|&&b| b >= least && b >= 2 * d) {
                let rows = if b == top {
                    // A full fire: the runtime's widest rows, decodes and all.
                    widest.saturating_sub(d)
                } else {
                    b.saturating_sub(d + host)
                };
                if rows < 2 || (b != top && d + rows + host > b) {
                    continue;
                }
                let mut lanes = vec![fresh(decode, 1); d as usize];
                lanes.push(fresh(prefill, rows));
                shapes.push(lanes);
                // A fire this wide holds many prompts beside its decodes: their
                // lanes pad to 32 or 64 (v6e gpt-oss c256 at 4096-row fires:
                // 27 prompts and a decode compiled mid-run, twice, 48 s), and
                // from 2048 rows beside a few decodes, as a burst of prompts
                // meets them (v6e gpt-oss c256: 18 prompts of 2857 rows and a
                // decode). (From 4096-row fires: what a narrower fire's short
                // prompts take compiles on demand.)
                if b >= widest && widest >= 4096 {
                    for n in [17u32, 33] {
                        let each = rows / n;
                        if each >= 2 && d + 2 * n <= self.budgets.tokens.max_lanes {
                            let mut lanes = vec![fresh(decode, 1); d as usize];
                            lanes.extend(std::iter::repeat_n(fresh(prefill, each), n as usize));
                            // A full fire's prompts fill the widest rows, and the
                            // lanes they pad to push it onto the top rung (v6e
                            // gpt-oss c256: 27 prompts and a decode, 4224 rows,
                            // compiled mid-run, 25 s).
                            if b == top && top > widest {
                                lanes[d as usize].rows += rows % n;
                            }
                            shapes.push(lanes);
                        }
                    }
                }
            }
        }
        // A full fire's many prompts beside more decodes than the lanes they
        // leave hold at the decodes' rung: the decodes pad to the lanes left
        // (`canonical_pads`), at the widest rows and the top (v6e gpt-oss
        // c256: 17 prompts beside 239 decodes compiled mid-run, 23 s).
        if widest >= 4096 && top > widest {
            for h in [32u32, 64, 128] {
                let left = max_lanes.saturating_sub(h);
                // The fewest decodes whose rung passes the lanes left.
                let mut below = 1;
                while pad.next(below) <= left {
                    below = pad.next(below);
                }
                let (d, n) = (below + 1, h / 2);
                if d > left || d + n > lanes_cap {
                    continue;
                }
                let ones = (left - d) + (h - 1 - n);
                for rows in [widest / 2, widest - d - ones] {
                    let mut lanes = vec![fresh(decode, 1); d as usize];
                    lanes.extend(std::iter::repeat_n(fresh(prefill, rows / n), n as usize));
                    lanes[d as usize].rows += rows % n;
                    shapes.push(lanes);
                }
            }
        }
        let started = std::time::Instant::now();
        // Trace every shape (nothing runs), then compile the new programs
        // side by side: a compile is host work, one core each.
        self.deferred = Some(Vec::new());
        let mut fired = 0;
        let mut failed = None;
        for shape in &shapes {
            match self.warm_fire(shape) {
                Ok(()) => fired += 1,
                Err(fault) => {
                    failed = Some(fault);
                    break;
                }
            }
        }
        let deferred = self.deferred.take().unwrap_or_default();
        if let Some(fault) = failed {
            return Err(fault);
        }
        let traced_s = started.elapsed().as_secs_f64();
        let skipped = self.compile_deferred(deferred)?;
        self.warm_gathers()?;
        let touched = shapes.iter().map(Vec::len).max().unwrap_or(0) as u32;
        for slot in 0..touched.min(self.held.len() as u32) {
            self.open(slot)?;
        }
        tracing::info!(
            fires = fired,
            programs = self.traced.len(),
            skipped,
            traced_s,
            s = started.elapsed().as_secs(),
            "xla warm ladder compiled"
        );
        Ok(fired)
    }

    /// Compiles what the ladder traced, a thread a core up to 8, and seats
    /// each program under its keys; answers how many programs it left out.
    ///
    /// A program whose temporaries do not fit the chip's HBM beside the pool
    /// (PJRT's `RESOURCE_EXHAUSTED`) is left out, not the load: the ladder
    /// covers shapes a fire may never take, and one that does compiles then
    /// and fails there, as every shape did before the ladder (Qwen3.8-Flash-
    /// Next's mini on a v5e refused its whole load over one, "Ran out of
    /// memory on HBM" at m=9772). Any other compile failure fails the load.
    fn compile_deferred(&mut self, deferred: Vec<super::Deferred>) -> Result<usize> {
        let mut texts: Vec<(&str, &crate::trace::Signature)> = Vec::new();
        let mut index: Vec<usize> = Vec::with_capacity(deferred.len());
        let mut seen: std::collections::HashMap<[u8; 32], usize> = std::collections::HashMap::new();
        for d in &deferred {
            let hash = *blake3::hash(d.text.as_bytes()).as_bytes();
            let at = *seen.entry(hash).or_insert_with(|| {
                texts.push((d.text.as_str(), &d.sig));
                texts.len() - 1
            });
            index.push(at);
        }
        // A compile's scratch is ~0.55 GiB a MiB of its text (v5e, Qwen3.6-
        // 35B-A3B's 9 MiB pf256 programs: 5 GB each; eight v6e ones 108 GB
        // at once): one is admitted while the host's free memory holds 0.75
        // GiB a MiB for it and for every compile still running (one that
        // started may not have taken its memory yet), each at its own text,
        // not the ladder's longest (at that a v5e host ran Qwen3.6-35B-A3B's
        // 168-program ladder one compile at a time, 73 compile-minutes past
        // the hour a round has). A v5e host's 47 GB stopped answering at four
        // started together.
        let need_of = |text: &str| ((hold_bytes(text) >> 6) * 3) << 8;
        let need = texts.iter().map(|(t, _)| need_of(t)).max().unwrap_or(0);
        let least = texts.iter().map(|(t, _)| need_of(t)).min().unwrap_or(0);
        // A compiled program held in host memory costs ~64 bytes a byte of its
        // text (0.5 GB a 9 MiB program; 65 of them 42 GB of a v5e host's 47),
        // and libtpu's allocator keeps what a dropped one freed for its own
        // next use, giving the host none back: what the ladder holds is
        // budgeted up front, the free memory less two compiles' room, in
        // ladder order. The rest are parked: compiled onto the disk cache (not
        // held) where they are not there yet, loaded (~1 s, never compiled) by
        // the first fire that takes their shape. With a disk cache the ladder
        // holds none it compiles while it compiles: a held program's memory is
        // a compile's room (Qwen3.6-35B-A3B's v5e ladder held 25 GB of a 47 GB
        // host by its end, its last compiles one at a time); those the free
        // memory holds load once the compiles are done.
        let parks = self.device.parks();
        let budget = available_memory().map_or(u64::MAX, |b| b.saturating_sub(2 * need));
        let held = std::sync::atomic::AtomicU64::new(0);
        let parked = std::sync::atomic::AtomicUsize::new(0);
        let running = std::sync::atomic::AtomicU64::new(0);
        let admit = std::sync::Mutex::new(());
        let room = available_memory().map_or(8, |b| b / least.max(1)) as usize;
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(8)
            .min(room)
            .clamp(1, texts.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let device = &self.device;
        let mut programs: Vec<Option<Result<std::sync::Arc<crate::device::Program>>>> =
            (0..texts.len()).map(|_| None).collect();
        let slots: Vec<std::sync::Mutex<Option<Result<std::sync::Arc<crate::device::Program>>>>> =
            (0..texts.len())
                .map(|_| std::sync::Mutex::new(None))
                .collect();
        use std::sync::atomic::Ordering::SeqCst;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((text, sig)) = texts.get(at) else {
                            break;
                        };
                        let hold = hold_bytes(text);
                        let kept = (!parks || device.on_disk(text))
                            && held
                                .fetch_update(SeqCst, SeqCst, |h| {
                                    (h + hold <= budget).then_some(h + hold)
                                })
                                .is_ok();
                        if !kept {
                            parked.fetch_add(1, SeqCst);
                            if device.on_disk(text) {
                                continue;
                            }
                        }
                        // Admitted one at a time: a compile in flight may
                        // not have taken its memory yet, so each counts as
                        // still to come.
                        let own = need_of(text);
                        loop {
                            let turn = admit.lock().expect("the warm admission lock poisoned");
                            let busy = running.load(SeqCst);
                            if busy == 0 || available_memory().is_none_or(|b| b >= busy + own) {
                                running.fetch_add(own, SeqCst);
                                break;
                            }
                            drop(turn);
                            std::thread::sleep(std::time::Duration::from_millis(200));
                        }
                        let program = device.program(text, (*sig).clone());
                        running.fetch_sub(own, SeqCst);
                        if kept || program.is_err() {
                            if let Ok(mut slot) = slots[at].lock() {
                                *slot = Some(program);
                            }
                        } else {
                            drop(program);
                            device.forget(text);
                        }
                        #[cfg(target_os = "linux")]
                        // SAFETY: `malloc_trim` only releases free heap pages.
                        unsafe {
                            libc::malloc_trim(0);
                        }
                    }
                });
            }
        });
        for (at, slot) in slots.into_iter().enumerate() {
            programs[at] = slot.into_inner().ok().flatten();
        }
        // A parked program the host's free memory holds once the compiles are
        // done (two compiles' room kept) loads from the disk cache here, at
        // load, not in the first fire that takes its shape (v5e Qwen3-0.6B
        // c256: 13 parked loads stalled a 43 s round 3-5 s), side by side in
        // ladder order, each admitted while the free memory holds it beside
        // those still loading (one at a time a v6e reloaded Qwen3.6-35B-A3B's
        // freshly compiled ladder in 150 s).
        let mut parked = parked.load(SeqCst);
        if parked > 0 {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let full = std::sync::atomic::AtomicBool::new(false);
            let loading = std::sync::atomic::AtomicU64::new(0);
            let loads: Vec<
                std::sync::Mutex<Option<Result<std::sync::Arc<crate::device::Program>>>>,
            > = (0..texts.len())
                .map(|_| std::sync::Mutex::new(None))
                .collect();
            let held = &programs;
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        loop {
                            let at = next.fetch_add(1, SeqCst);
                            let Some((text, sig)) = texts.get(at) else {
                                break;
                            };
                            if full.load(SeqCst) {
                                break;
                            }
                            if held[at].is_some() || !device.on_disk(text) {
                                continue;
                            }
                            let hold = hold_bytes(text);
                            {
                                let _turn = admit.lock().expect("the warm admission lock poisoned");
                                let busy = loading.load(SeqCst);
                                if available_memory().is_none_or(|b| b < hold + busy + 2 * need) {
                                    full.store(true, SeqCst);
                                    break;
                                }
                                loading.fetch_add(hold, SeqCst);
                            }
                            let program = device.program(text, (*sig).clone());
                            loading.fetch_sub(hold, SeqCst);
                            if let Ok(mut slot) = loads[at].lock() {
                                *slot = Some(program);
                            }
                        }
                    });
                }
            });
            for (at, slot) in loads.into_iter().enumerate() {
                if let Some(program) = slot.into_inner().ok().flatten() {
                    programs[at] = Some(program);
                    parked -= 1;
                }
            }
        }
        if parked > 0 {
            tracing::warn!(
                parked,
                held_gib = held.load(SeqCst) >> 30,
                "xla warm ladder parked programs on disk: the host's memory holds no more"
            );
        }
        // The compiles' scratch is freed by now: hand it back rather than
        // keep it in the allocator's arenas. glibc's call; the other libcs
        // have no equivalent, and macOS has to build this crate too.
        #[cfg(target_os = "linux")]
        // SAFETY: `malloc_trim` only releases free heap pages.
        unsafe {
            libc::malloc_trim(0);
        }
        let exhausted = format!(
            "(code {})",
            crate::pjrt::sys::PJRT_Error_Code_RESOURCE_EXHAUSTED
        );
        let mut left_out = std::collections::HashSet::new();
        for (d, at) in deferred.into_iter().zip(index) {
            let program = match programs[at].as_ref() {
                Some(Ok(program)) => std::sync::Arc::clone(program),
                Some(Err(fault)) => {
                    let why = fault.to_string();
                    if why.contains(&exhausted) {
                        if left_out.insert(at) {
                            tracing::warn!(%why, "xla warm shape left out: its program does not fit HBM");
                        }
                        continue;
                    }
                    return Err(crate::error::Fault::Program {
                        at: "serve::warm",
                        why,
                    });
                }
                None => continue,
            };
            self.probed.insert(d.key, d.probed);
            self.traced.insert(d.key, std::sync::Arc::clone(&program));
            if let Some(shape_key) = d.shape_key {
                self.shaped.insert(shape_key, program);
            }
        }
        Ok(left_out.len())
    }

    fn warm_fire(&mut self, shape: &[Want]) -> Result<()> {
        let slots = self.held.len().max(1) as u32;
        let page_size = self.paging().page_size.max(1);
        // Every lane reads the same scratch slots: what warm fires write is
        // never read back, and the pool need not hold a page per lane.
        let owned: Vec<(Vec<u32>, crate::store::Scratch)> = shape
            .iter()
            .map(|want| {
                let n = (held_of(want, page_size) + want.rows)
                    .div_ceil(page_size)
                    .max(1);
                (
                    vec![0; want.rows as usize],
                    self.pools.scratch(0, n as usize),
                )
            })
            .collect();
        for slot in 0..(shape.len() as u32).min(slots) {
            self.open(slot)?;
        }
        let seated: Vec<Seated<'_>> = shape
            .iter()
            .zip(&owned)
            .enumerate()
            .map(|(at, (want, (tokens, scratch)))| {
                let mut seated = Seated::of(Lane {
                    slot: at as u32 % slots,
                    word: want.word,
                    tokens,
                });
                seated.pages = &scratch.pages;
                seated.window = &scratch.window;
                seated.fixed = &scratch.fixed;
                seated.held = Some(held_of(want, page_size));
                seated
            })
            .collect();
        self.fire_full(&seated).map(|_| ())
    }
}

/// A decode lane at page bound `p` holds `p · page_size - 1` rows, so its
/// token lands in the last slot of its `p`-th page; a fresh lane holds none.
fn held_of(want: &Want, page_size: u32) -> u32 {
    (want.held_pages * page_size).saturating_sub(want.rows.min(1))
}

/// The host memory free for new allocations (`MemAvailable`), where the host
/// says.
fn available_memory() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib << 10)
}

/// The host memory a compiled program holds: ~64 bytes a byte of its text,
/// less its Mosaic kernels' bodies (`tpu_custom_call` configs, one HLO
/// instruction each however long its base64), counted at their own size.
fn hold_bytes(text: &str) -> u64 {
    const BODY: &str = "\\22body\\22: \\22";
    let (mut kernels, mut rest) = (0, text);
    while let Some(at) = rest.find(BODY) {
        let tail = &rest[at + BODY.len()..];
        let len = tail.find("\\22").unwrap_or(tail.len());
        kernels += len;
        rest = &tail[len..];
    }
    (((text.len() - kernels) as u64) << 6) + kernels as u64
}
