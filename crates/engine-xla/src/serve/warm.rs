//! Warm fires at load (`PIE_XLA_PREWARM=1`): the fire shapes a server meets
//! first are traced, compiled (or loaded from the disk cache, where a warm
//! machine finds them) and run once before the first request, as engine-cuda
//! arms its bodies (`serve/arming.rs`), so no request waits on a compile.
//!
//! The ladder, over the canonical shapes `fire_inner` pads fires to:
//! - a decode at every lane rung, at every page bound from `PAGE_FLOOR` to
//!   `PIE_XLA_PREWARM_PAGES` (default: the context's), as many lanes as the
//!   pool seats at half that bound;
//! - a prefill at every row rung from 16, over 1 lane (one that fills its
//!   rung, and one that leaves a padding lane) and, from 64 rows, 4 and 8;
//! - a prefill beside a decode at every lane rung, at every row rung from
//!   256 (such a fire pads its prefill lanes to `MIXED_HOST_RUNG`).
//!
//! Nothing runs: each shape is traced, and the new programs compile side by
//! side (`PIE_XLA_PREWARM_THREADS`, default 8).
//!
//! Lanes are synthetic (zero tokens, as `dry.rs` builds them) in the plain
//! text classes; they share pages, since what they write is never read. The
//! slots they touch are opened again afterwards.

use poem_ir::Request;

use super::{Lane, PAD_ROOM, PAGE_FLOOR, Seated, Shell};
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
                    && !shell.corrected.contains(class)
                    && !shell.capturing.contains(class)
            })
        };
        if !plain(self, decode) || !plain(self, prefill) || self.patch_seat.is_some() {
            return Ok(0);
        }
        let buckets = self.budgets.tokens.buckets.clone();
        // The widest fire the runtime sends (the top rung is padding room).
        let widest = match buckets.as_slice() {
            [.., below, top] if top - below == PAD_ROOM => *below,
            [.., top] => *top,
            [] => return Ok(0),
        };
        let lanes_cap = self
            .budgets
            .tokens
            .max_lanes
            .saturating_sub(super::MIXED_HOST_RUNG)
            .min(self.held.len() as u32);
        let cap = self.paging().pages_per_slot.max(1).next_power_of_two();
        let top_pages = std::env::var("PIE_XLA_PREWARM_PAGES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(cap)
            .min(cap);
        let pool = self.paging().pages();
        let fresh = |word, rows| Want {
            word,
            rows,
            held_pages: 0,
        };
        let mut shapes: Vec<Vec<Want>> = Vec::new();
        let mut pages = PAGE_FLOOR.min(cap);
        while pages <= top_pages {
            // A lane at this bound holds more than half of it.
            let seats = u32::try_from(pool / u64::from(pages / 2).max(1)).unwrap_or(u32::MAX);
            let most = lanes_cap.min(seats.max(1).next_power_of_two());
            for &n in buckets.iter().filter(|&&n| n <= most) {
                let lane = Want {
                    word: decode,
                    rows: 1,
                    held_pages: pages,
                };
                shapes.push(vec![lane; n as usize]);
            }
            pages *= 2;
        }
        for &b in buckets.iter().filter(|&&b| (16..=widest).contains(&b)) {
            shapes.push(vec![fresh(prefill, b)]);
            shapes.push(vec![fresh(prefill, b - 1)]);
            for n in [4u32, 8] {
                if b >= 64 && n <= lanes_cap {
                    // `n - 1` lanes and the padding lane the rest takes (two
                    // or three lanes pad to four).
                    shapes.push(vec![fresh(prefill, b / n); (n - 1) as usize]);
                }
            }
        }
        // Prefills beside decodes (`MIXED_HOST_RUNG` prefill lanes): every
        // decode lane rung at every row rung from 256 it fits in, and at the
        // padding rung, which a full fire's decode pads overflow into.
        let host = super::MIXED_HOST_RUNG;
        let top = *buckets.last().unwrap_or(&widest);
        for &d in buckets.iter().filter(|&&d| d <= lanes_cap) {
            for &b in buckets.iter().filter(|&&b| b >= 256 && b >= 2 * d) {
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
        self.compile_deferred(deferred)?;
        let touched = shapes.iter().map(Vec::len).max().unwrap_or(0) as u32;
        for slot in 0..touched.min(self.held.len() as u32) {
            self.open(slot)?;
        }
        tracing::info!(
            fires = fired,
            programs = self.traced.len(),
            traced_s,
            s = started.elapsed().as_secs(),
            "xla warm ladder compiled"
        );
        Ok(fired)
    }

    /// Compiles what the ladder traced, on up to `PIE_XLA_PREWARM_THREADS`
    /// (default 8) threads, and seats each program under its keys.
    fn compile_deferred(&mut self, deferred: Vec<super::Deferred>) -> Result<()> {
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
        let threads = std::env::var("PIE_XLA_PREWARM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, texts.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let device = &self.device;
        let mut programs: Vec<Option<Result<std::sync::Arc<crate::device::Program>>>> =
            (0..texts.len()).map(|_| None).collect();
        let slots: Vec<std::sync::Mutex<Option<Result<std::sync::Arc<crate::device::Program>>>>> =
            (0..texts.len())
                .map(|_| std::sync::Mutex::new(None))
                .collect();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((text, sig)) = texts.get(at) else {
                            break;
                        };
                        let program = device.program(text, (*sig).clone());
                        if let Ok(mut slot) = slots[at].lock() {
                            *slot = Some(program);
                        }
                    }
                });
            }
        });
        for (at, slot) in slots.into_iter().enumerate() {
            programs[at] = slot.into_inner().ok().flatten();
        }
        // The compiles' scratch is freed by now: hand it back rather than
        // keep it in the allocator's arenas. glibc's call; the other libcs
        // have no equivalent, and macOS has to build this crate too.
        #[cfg(target_os = "linux")]
        // SAFETY: `malloc_trim` only releases free heap pages.
        unsafe {
            libc::malloc_trim(0);
        }
        for (d, at) in deferred.into_iter().zip(index) {
            let program = match programs[at].as_ref() {
                Some(Ok(program)) => std::sync::Arc::clone(program),
                Some(Err(fault)) => {
                    return Err(crate::error::Fault::Program {
                        at: "serve::warm",
                        why: fault.to_string(),
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
        Ok(())
    }

    fn warm_fire(&mut self, shape: &[Want]) -> Result<()> {
        let slots = self.held.len().max(1) as u32;
        let page_size = self.paging().page_size.max(1);
        let pool = u32::try_from(self.paging().pages())
            .unwrap_or(u32::MAX)
            .max(1);
        let windowed = self.paging().window_pages().saturating_sub(1).max(1);
        let has_window = self.pools.has_windowed();
        // Every lane reads the same pages from 0: what warm fires write is
        // never read back, and the pool need not hold a page per lane.
        let owned: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = shape
            .iter()
            .map(|want| {
                let n = (held_of(want, page_size) + want.rows)
                    .div_ceil(page_size)
                    .max(1);
                let pages: Vec<u32> = (0..n).map(|p| p % pool).collect();
                let window = if has_window {
                    pages
                        .iter()
                        .map(|&page| 1 + u32::try_from(u64::from(page) % windowed).unwrap_or(0))
                        .collect()
                } else {
                    Vec::new()
                };
                (vec![0; want.rows as usize], pages, window)
            })
            .collect();
        for slot in 0..(shape.len() as u32).min(slots) {
            self.open(slot)?;
        }
        let seated: Vec<Seated<'_>> = shape
            .iter()
            .zip(&owned)
            .enumerate()
            .map(|(at, (want, (tokens, pages, window)))| {
                let mut seated = Seated::of(Lane {
                    slot: at as u32 % slots,
                    word: want.word,
                    tokens,
                });
                seated.pages = pages;
                seated.window = window;
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
