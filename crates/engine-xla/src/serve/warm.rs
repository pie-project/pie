//! Warm fires at load, every load, before it answers ready: the fire shapes a
//! server meets are traced, compiled (or loaded from the disk cache, where a warm
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
use crate::device::{Admission, HostRoom, available_memory};
use crate::error::Result;

/// One synthetic lane: its word, its rows, and the pages it holds before
/// them (a decode lane at a page bound holds all but one slot of them).
#[derive(Clone, Copy)]
struct Want {
    word: u64,
    rows: u32,
    held_pages: u32,
    /// The lane reads out every row (a drafting lane's window), not its
    /// last.
    all_rows: bool,
    /// The lane states a mask (every key allowed): its fire stages the
    /// load's mask plane, a program of its own.
    masked: bool,
    /// The lane buffers its recurrent rows (`RsVerb::Buffer`): its fire is
    /// the buffered treatment of its shape (`engine::Treatment::Buffers`).
    buffers: bool,
}

/// A fire's layout key (`Shell::layout_key`): bucket, buffered, and each
/// word's one-row and wider lane counts.
type LayoutKey = (u32, bool, Vec<(u64, bool, u32)>);

/// The stated compositions the ladder lays no program for
/// (`Shell::unlaid_compositions`).
#[derive(Default)]
struct Gaps {
    /// Canonical layouts no ladder shape takes, each with a shape that lays it.
    unlaid: Vec<(String, Vec<Want>)>,
    /// Compositions `canonical_layout` finds no layout for, and the first.
    uncanonical: u64,
    first_uncanonical: Vec<String>,
}

/// The step composition (`engine::StepComposition`) a fire of `shape`'s lanes
/// forms, under `treatment`.
fn composition(shape: &[Want], treatment: engine::Treatment) -> engine::StepComposition {
    let mut step = engine::StepComposition {
        treatment,
        ..engine::StepComposition::default()
    };
    for want in shape {
        if want.rows <= 1 {
            step.decode_lanes += 1;
            step.rows += 1;
        } else {
            step.lane_rows = if step.prompt_lanes == 0 {
                engine::shapes::Span {
                    lo: want.rows,
                    hi: want.rows,
                }
            } else {
                engine::shapes::Span {
                    lo: step.lane_rows.lo.min(want.rows),
                    hi: step.lane_rows.hi.max(want.rows),
                }
            };
            step.prompt_lanes += 1;
            step.prompt_rows += want.rows;
            step.rows += want.rows;
        }
        step.readout_rows += if want.all_rows { want.rows } else { 1 };
        step.attachments.mask |= want.masked;
    }
    step
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
                !shell.wide.contains(class)
                    && !shell.corrected.contains(class)
                    // a class reading a guest input fires only with its
                    // channel's cell, which a synthetic lane has none of
                    && !shell
                        .dit
                        .feeds()
                        .inputs
                        .iter()
                        .any(|(_, readers)| readers.contains(class))
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
            all_rows: false,
            masked: false,
            buffers: false,
        };
        // A decode lane beside prompts holds its own prompt, a page of it:
        // its fire is the one a served decode lane's is (on a load that
        // buffers recurrent rows, one that may carry them), at the page
        // bound the prompts beside it set.
        let beside = Want {
            word: decode,
            rows: 1,
            held_pages: 1,
            all_rows: false,
            masked: false,
            buffers: false,
        };
        let hybrid = self.has_state();
        let mut shapes: Vec<Vec<Want>> = Vec::new();
        // A class reads a lane's mask (the mask fold: data, not a class).
        let masks = !self.masked.is_empty();
        // Where each part of the ladder starts in `shapes`: the load line
        // states the programs each part adds.
        let mut parts: Vec<(&str, usize)> = vec![("decode", 0)];
        let mut pages = pad.rung(pad.page_floor).min(cap);
        while pages <= cap {
            // A fire's page bound is its longest lane's: that lane holds more
            // than half of it, every other lane a page at least (over-demand:
            // 64 decodes beside one long lane at 64 pages compiled mid-run,
            // where the ladder had seated every lane past half the bound).
            let seats = u32::try_from(1 + pool.saturating_sub(u64::from(pages / 2 + 1)))
                .unwrap_or(u32::MAX);
            // and the rung the most lanes it seats pad to (Qwen3.6-35B-A3B's
            // v5e pool seats 126: its c256 decodes ran 128 lanes, compiled
            // mid-run)
            let most = poem_exec::fire::rung_of(&buckets, lanes_cap.min(seats.max(1)));
            for &n in buckets.iter().filter(|&&n| n <= most) {
                let lane = Want {
                    word: decode,
                    rows: 1,
                    held_pages: pages,
                    all_rows: false,
                    masked: false,
                    buffers: false,
                };
                shapes.push(vec![lane; n as usize]);
                // the same decodes beside a lane stating a mask (a beam): the
                // mask is data, its plane the load's shape
                if masks {
                    let mut lanes = vec![lane; n as usize];
                    lanes[0].masked = true;
                    shapes.push(lanes);
                }
            }
            pages = pad.next(pages);
        }
        parts.push(("prefill", shapes.len()));
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
            // Every lane rung whose prompts (two rows at least, a decode's
            // one) fit the rows: a burst of short prompts fires many at once
            // (c64: 64 prompt tails of 3 rows in 256 rows compiled mid-run).
            for &p in buckets.iter().filter(|&&p| p >= 2 && p <= lanes_cap) {
                // the prompt counts that reach every form rung `p` pads to:
                // one short of it with the padding lane the rest takes, and
                // at it, filling the rows exactly (two prompts of 8 rows in
                // 16 fire as they are, `canonical_pads`)
                for n in lane_counts(p) {
                    if b / n >= 2 {
                        let mut lanes = vec![fresh(prefill, b / n); n as usize];
                        if n < p {
                            lanes.pop();
                        }
                        shapes.push(lanes);
                    }
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
        parts.push(("mixed", shapes.len()));
        // Prefills beside decodes: every decode lane rung at every row rung
        // it fits in, and at the padding rung, which a full fire's decode
        // pads overflow into. A fire's prompt lanes pad to a lane rung from
        // `MIXED_HOST_RUNG` up (`canonical_pads`), so every such rung `p` is
        // taken too, where `p / 2` prompts of two rows each (the fewest that
        // pad to it) fit beside the decodes in the rows and `p + d` in the
        // lanes: what the cutter can form, no more (c64 and over-demand:
        // 10 to 40 prompt tails of 3 rows beside decodes at 32 to 128 rows
        // compiled mid-run).
        let host = pad.mixed_host_rung;
        let least = pad.prefill_floor;
        // (A fire that prefills pads its decodes to `pad.rung`: a half rung of
        // the ladder is a decode-only fire's.)
        // (A fire of decodes beside prompts pads them to the decode lanes its
        // prompts' rung leaves, `canonical_pads`: one decode count a row
        // bucket and prompt rung.)
        for &d in buckets
            .iter()
            .filter(|&&d| d <= pad.rung(lanes_cap) && pad.rung(d) == d)
        {
            for &b in buckets.iter().filter(|&&b| b >= least && b >= 2 * d) {
                // The bucket filled: decodes, prompt pad lanes, the rest. At
                // the top also a full fire, the runtime's widest rows,
                // decodes and all (where the profile keeps padding room past
                // the widest rows, that one lands a bucket lower, and a fire
                // its pads carry into the top found no program: every
                // composition the load states laid, `Shell::mixed_lanes`,
                // has one).
                let filled = b.saturating_sub(d + host);
                let mut totals = vec![filled];
                if b == top && widest.saturating_sub(d) != filled {
                    totals.push(widest.saturating_sub(d));
                }
                for rows in totals {
                    if rows < 2 || d == 0 || d != self.mixed_decode_lanes(host) {
                        continue;
                    }
                    let mut lanes = vec![beside; d as usize];
                    lanes.push(fresh(prefill, rows));
                    shapes.push(lanes);
                }
                // (the wide fires below: the widest rows at the top, else
                // the bucket filled)
                let rows = if b == top {
                    widest.saturating_sub(d)
                } else {
                    filled
                };
                let mut p = pad.next(host);
                while p + d <= max_lanes {
                    if d != self.mixed_decode_lanes(p) {
                        p = pad.next(p);
                        continue;
                    }
                    // the fewest prompts that pad to `p` (`canonical_pads`:
                    // `n` prompts and the rest lane at their rung): `p / 2`
                    // only at a growth of 200
                    let Some(n) = (1..p).find(|&n| pad.rung(n + 1).max(host) == p) else {
                        p = pad.next(p);
                        continue;
                    };
                    // the prompts' rows: the bucket, less the decodes, the
                    // one-row lanes their count pads with and the rest lane
                    let r = b.saturating_sub(d + p - n);
                    if r >= 2 * n {
                        let mut lanes = vec![beside; d as usize];
                        lanes.extend(std::iter::repeat_n(fresh(prefill, r / n), n as usize));
                        lanes[d as usize].rows += r % n;
                        shapes.push(lanes);
                    }
                    p = pad.next(p);
                }
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
                        if each >= 2
                            && d + 2 * n <= self.budgets.tokens.max_lanes
                            && d == self.mixed_decode_lanes(pad.rung(n + 1).max(host))
                        {
                            let mut lanes = vec![beside; d as usize];
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
        // Prompts beside more decodes than their rung's `beside` holds, and
        // than the lanes they leave hold at the decodes' rung: the decodes
        // pad to the lanes left (`canonical_pads`), at every prompt rung and
        // bucket the cutter can form them in (v6e Qwen3-0.6B c64: 19 prompts
        // padded to 32 lanes beside 48 decodes in 256 rows compiled mid-run,
        // 10 s, every inferlet waiting).
        let mut p = host;
        while p < max_lanes {
            let left = max_lanes - p;
            let past = self.mixed_decode_lanes(p) + 1;
            let d = (past..=left).find(|&d| pad.rung(d) > left);
            let n = (1..=p).find(|&n| pad.rung(n + 1).max(host) == p);
            if let (Some(d), Some(n)) = (d, n)
                && d + n <= lanes_cap
            {
                // decodes' pad lanes, prompts' pad lanes; the rest lane a row
                let ones = (left - d) + (p - 1 - n);
                for &b in buckets.iter().filter(|&&b| b >= least && b <= widest) {
                    let r = b.saturating_sub(d + ones + 1);
                    if r >= 2 * n {
                        let mut lanes = vec![beside; d as usize];
                        lanes.extend(std::iter::repeat_n(fresh(prefill, r / n), n as usize));
                        lanes[d as usize].rows += r % n;
                        shapes.push(lanes);
                    }
                }
            }
            p = pad.next(p);
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
                    let mut lanes = vec![beside; d as usize];
                    lanes.extend(std::iter::repeat_n(fresh(prefill, rows / n), n as usize));
                    lanes[d as usize].rows += rows % n;
                    shapes.push(lanes);
                }
            }
        }
        // Drafting lanes (a load whose model drafts, `draft_depth`) read out
        // every row: their windows (a token and its drafts, as deep as the
        // model drafts) at every lane rung and page bound, and their prompts
        // at every lane rung and row rung the readout store holds (the MTP
        // inferlet, eight at once: windows of 2 rows over 2 to 8 lanes and
        // four 29-row prompts compiled mid-run). A fire of `n` lanes pads to
        // its rung by one-row lanes reading nothing (`canonical_pads`): `n`
        // at the rung, just past half of it and one short of it reach every
        // form that rung pads to.
        if let Some(depth) = self.trace().facts.draft_depth {
            let drafts = facts.word(&Request::new(2, false).drafting(true));
            let window = facts.word(&Request::new(1 + depth, false).drafting(true));
            // (a fire reads out no more rows than the load's store holds)
            let store = self.readout_store_rows().unwrap_or(u32::MAX);
            if plain(self, drafts) && plain(self, window) {
                self.draft_depth = depth;
                self.drafting = [drafts, window]
                    .iter()
                    .filter_map(|&w| classes.class_of(w & classes.mask))
                    .collect();
                let counts = lane_counts;
                parts.push(("draft windows", shapes.len()));
                for &p in buckets.iter().filter(|&&p| p <= lanes_cap) {
                    for n in counts(p) {
                        if n * (1 + depth) > store {
                            continue;
                        }
                        let mut pages = pad.rung(pad.page_floor).min(cap);
                        while pages <= cap {
                            let lane = Want {
                                word: window,
                                rows: 1 + depth,
                                held_pages: pages,
                                all_rows: true,
                                masked: false,
                                buffers: false,
                            };
                            shapes.push(vec![lane; n as usize]);
                            pages = pad.next(pages);
                        }
                    }
                }
                parts.push(("draft prompts", shapes.len()));
                for &b in buckets
                    .iter()
                    .filter(|&&b| (pad.prefill_floor.max(2)..=widest.min(store)).contains(&b))
                {
                    for &p in buckets.iter().filter(|&&p| p <= lanes_cap) {
                        for n in counts(p) {
                            if b / n < 2 {
                                continue;
                            }
                            let mut lanes = vec![
                                Want {
                                    all_rows: true,
                                    ..fresh(drafts, b / n)
                                };
                                n as usize
                            ];
                            lanes[0].rows += b % n;
                            lanes[0].rows -= 1;
                            shapes.push(lanes);
                        }
                    }
                }
            }
        }
        // The buffered treatment of every shape the step set states buffered
        // (a lane of it buffers its recurrent rows: a stop string's, a beam's
        // or a draft's, or a decode's run-ahead under the lagged fold): its
        // fire lays the replaying scratch, a program of its own, laid here
        // and nowhere the set does not state it.
        if self.pools.buffers_rs()
            && let Some(steps) = self.steps.clone()
        {
            let buffered: Vec<Vec<Want>> = shapes
                .iter()
                .filter(|shape| {
                    steps
                        .shape_of(&composition(shape, engine::Treatment::Buffers))
                        .is_some()
                })
                .map(|shape| {
                    shape
                        .iter()
                        .map(|w| Want {
                            buffers: true,
                            ..*w
                        })
                        .collect()
                })
                .collect();
            parts.push(("buffered", shapes.len()));
            shapes.extend(buffered);
        }
        // Every composition the stated set holds is laid (nothing compiles
        // after ready): one the ladder's shapes leave out is laid here as its
        // own shape, and one canonical_pads finds no layout for (a fire of it
        // keys on its own counts) refuses the load by name.
        let gaps = self.unlaid_compositions(&shapes, decode, prefill);
        if gaps.uncanonical > 0 {
            return Err(crate::error::Fault::Program {
                at: "serve::warm",
                why: format!(
                    "{} compositions the step set states take no padded layout (the first: {}): \
                     a fire of one would compile after ready",
                    gaps.uncanonical,
                    gaps.first_uncanonical.join("; ")
                ),
            });
        }
        if !gaps.unlaid.is_empty() {
            parts.push(("stated", shapes.len()));
            shapes.extend(gaps.unlaid.into_iter().map(|(_, shape)| shape));
        }
        let mut added: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        let started = std::time::Instant::now();
        // Trace every shape (nothing runs), then compile the new programs
        // side by side: a compile is host work, one core each.
        self.trace_scope = Some(self.trace_scope());
        self.deferred = Some(Vec::new());
        let mut fired = 0;
        let mut failed = None;
        // A load streaming experts serves each shape two ways: reading the
        // experts where they lie, and staging the ones in no slot first.
        let passes = if self.streams_experts() { 2 } else { 1 };
        'passes: for pass in 0..passes {
            self.stage_all = pass == 1;
            for (at, shape) in shapes.iter().enumerate() {
                let before = self.deferred.as_ref().map_or(0, Vec::len);
                let fired_part = warm_part(&parts, at);
                let r = self.warm_fire(shape);
                // each program it traced, in the order serving meets them:
                // decode first, by lanes, then drafts, mixed and prompts
                let use_order = (use_rank(fired_part), shape.len() as u32);
                if let Some(deferred) = self.deferred.as_mut() {
                    for d in deferred.iter_mut().skip(before) {
                        d.order = (use_order.0, use_order.1, d.rows);
                    }
                }
                if pass == 0 {
                    let new = self.deferred.as_ref().map_or(0, Vec::len) - before;
                    *added.entry(fired_part).or_insert(0) += new;
                }
                match r {
                    Ok(()) => fired += 1,
                    Err(fault) => {
                        failed = Some(fault);
                        break 'passes;
                    }
                }
            }
        }
        self.stage_all = false;
        let deferred = self.deferred.take().unwrap_or_default();
        if let Some(fault) = failed {
            return Err(fault);
        }
        let traced_s = started.elapsed().as_secs_f64();
        let skipped = self.compile_deferred(deferred)?;
        // A prompt reading every row of itself fires as the plain fire of its
        // rows and reads its rows again with its row count's readout program.
        let readouts = self.lay_readouts()?;
        self.warm_gathers()?;
        self.pools.warm_copies(&self.device, &self.handles)?;
        let touched = shapes.iter().map(Vec::len).max().unwrap_or(0) as u32;
        for slot in 0..touched.min(self.held.len() as u32) {
            self.open(slot)?;
        }
        tracing::info!(
            fires = fired,
            parts = ?added,
            programs = self.traced.len(),
            skipped,
            readouts,
            traced_s,
            s = started.elapsed().as_secs(),
            "xla warm ladder compiled"
        );
        self.laddered = fired > 0;
        self.load_behind();
        Ok(fired)
    }

    /// Runs a beam group's head (`crate::store::beam`) before its fire.
    pub fn run_beam_head(
        &mut self,
        layout: kernels_xla::beam::Layout,
        head: &crate::store::beam::Head,
    ) -> Result<crate::pjrt::Buffer> {
        self.pools
            .run_beam_head(&self.device, &self.handles, layout, head)
    }

    /// Lays a beam group's head program at every width rung up to the lane
    /// cap (powers of two; at most a pool row of parents, 128): a group runs
    /// on the smallest rung that holds it. None on a load that keeps no kv
    /// or reads windowed tables (a beam group's lists resolve over
    /// full-context tables only).
    pub fn lay_beam_heads(&mut self) -> Result<usize> {
        if self.pools.kv_shape().is_none() || self.pools.tables().contains(&true) {
            return Ok(0);
        }
        let started = std::time::Instant::now();
        let paging = self.paging();
        let frames = self.staging_facts().ring_slots.saturating_sub(1).max(1);
        let top = self.budgets.tokens.max_lanes.min(128);
        let mut rung = 1u32;
        let mut layouts = Vec::new();
        while rung <= top {
            layouts.push(crate::beam::rung_layout(
                rung,
                paging.page_size,
                paging.pages_per_slot,
                frames,
            ));
            rung *= 2;
        }
        self.pools
            .lay_beam_heads(&self.device, &self.handles, &layouts)?;
        let laid = layouts.len();
        tracing::info!(
            programs = laid,
            top_rung = rung / 2,
            ms = started.elapsed().as_secs_f64() * 1e3,
            "xla beam head programs laid: one a width rung"
        );
        Ok(laid)
    }

    /// Loads the parked programs from the disk cache behind serving, in the
    /// order serving meets them (`use_rank`: decode by lanes first), as many at once as the load rate's knee (the
    /// width past which it stops rising), each admitted by the host's room
    /// as the ladder's are ([`HostRoom`]). A fire of a shape not yet loaded
    /// loads it itself, ahead of the loader (which starts none while a fire
    /// waits on one, `Device::wait_demand`), or waits on the loader's load
    /// of it where one is under way (`Device::load_named`); never building
    /// it. What the host holds no room for stays parked, its first fire
    /// loading it.
    fn load_behind(&mut self) {
        let mut queue: Vec<_> = self
            .parked
            .iter()
            .map(|(key, p)| (*key, p.code.clone(), p.sig.clone(), p.order))
            .collect();
        queue.sort_by_key(|q| (q.3, q.0));
        let mut seen = std::collections::HashSet::new();
        queue.retain(|q| seen.insert(q.1.hash()));
        if queue.is_empty() {
            return;
        }
        let device = std::sync::Arc::clone(&self.device);
        let stated = self.host_programs;
        // after ready, nothing compiles but guest stages: their room (by
        // their measured needs, `guest::run::guest_reserve`) is what the
        // loader leaves free, read as it loads
        let floor = 0;
        let scope = self.trace_scope.clone();
        let stop = std::sync::Arc::clone(&self.behind.stop);
        let width = self.load_width.clone();
        let started = std::time::Instant::now();
        let spawned = std::thread::Builder::new()
            .name("xla-load-behind".into())
            .spawn(move || {
                use std::sync::atomic::Ordering::SeqCst;
                let room = HostRoom::new(stated, floor);
                let next = std::sync::atomic::AtomicUsize::new(0);
                let loaded = std::sync::atomic::AtomicUsize::new(0);
                let full = std::sync::atomic::AtomicBool::new(false);
                // One program's load: its text's bytes, `None` past the queue's
                // end or with the host full or the loader stopped.
                let one = || -> Option<u64> {
                    if stop.load(SeqCst) || full.load(SeqCst) {
                        return None;
                    }
                    let (key, code, sig, _) = queue.get(next.fetch_add(1, SeqCst))?;
                    // no load beside a guest stage compile measuring its need
                    let _beside = crate::guest::run::load_beside();
                    let Some(reserved) = room.take(code.len()) else {
                        full.store(true, SeqCst);
                        return None;
                    };
                    let program = device.load_behind(code, sig.clone());
                    room.done(reserved, matches!(program, Ok(Some(_))));
                    match program {
                        Ok(Some(_)) => {
                            loaded.fetch_add(1, SeqCst);
                        }
                        Ok(None) => {
                            if let (Some(scope), crate::device::Code::Cached { name, .. }) =
                                (scope.as_deref(), code)
                            {
                                device.mend_record(scope, key, name);
                            }
                        }
                        Err(fault) => {
                            tracing::warn!(%fault, "xla program loading behind serving failed: its first fire loads it");
                        }
                    }
                    #[cfg(target_os = "linux")]
                    // SAFETY: `malloc_trim` only releases free heap pages.
                    unsafe {
                        libc::malloc_trim(0);
                    }
                    Some(code.len() as u64)
                };
                // `width` loads at once, after any a fire waits on: their
                // text's bytes a second, `None` once nothing is left to load.
                let batch = |width: usize| -> Option<f64> {
                    device.wait_demand();
                    let started = std::time::Instant::now();
                    let bytes: Vec<Option<u64>> = std::thread::scope(|s| {
                        let loads: Vec<_> = (0..width).map(|_| s.spawn(one)).collect();
                        loads.into_iter().map(|l| l.join().ok().flatten()).collect()
                    });
                    let done: u64 = bytes.iter().flatten().sum();
                    (bytes.iter().all(Option::is_some) && done > 0)
                        .then(|| done as f64 / started.elapsed().as_secs_f64())
                };
                // The width: grown by one while the fastest of a width's
                // samples beats the last width's (the knee of the curve, as
                // the rates' curves find theirs); deserializes under way are
                // not preempted, so a wider loader only delays a fire's load
                // past it (v6e Qwen3.5: 44 at once put a first request at
                // 13.9-33.3 s, 2-4 at 9.95 s).
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                let knee = match (width.stated, width.measure) {
                    (Some(stated), _) => {
                        tracing::info!(width = stated, "xla loading behind serving: the width the device's profile states");
                        usize::try_from(stated).unwrap_or(1).max(1)
                    }
                    (None, false) => {
                        tracing::info!(
                            width = 1,
                            "xla loading behind serving: skipped measuring its width (`[engine] measure = false`) and no layer states `[rates] load_width` under a matching key; served by the estimate, one load at a time, which delays a fire's own load by one at most"
                        );
                        1
                    }
                    (None, true) => {
                        let (mut at, mut last) = (1usize, 0f64);
                        // the knee, or `None` where the queue ran out first
                        let found = 'grow: loop {
                            let mut fastest = 0f64;
                            for _ in 0..LOAD_SAMPLES {
                                match batch(at) {
                                    Some(rate) => fastest = fastest.max(rate),
                                    None => break 'grow None,
                                }
                            }
                            if fastest <= last {
                                break 'grow Some(at - 1);
                            }
                            if at == cores {
                                break 'grow Some(at);
                            }
                            last = fastest;
                            at += 1;
                        };
                        match found {
                            Some(knee) => {
                                tracing::info!(width = knee, cores, "xla loading behind serving: the width the load rate stops rising at");
                                keep_width(&width, knee);
                                knee
                            }
                            None => {
                                tracing::info!(
                                    width = at,
                                    "xla loading behind serving: the programs ran out before the load rate stopped rising; no width kept"
                                );
                                at
                            }
                        }
                    }
                };
                while batch(knee).is_some() {}
                let loaded = loaded.load(SeqCst);
                tracing::info!(
                    loaded,
                    of = queue.len(),
                    s = started.elapsed().as_secs_f64(),
                    stopped = stop.load(SeqCst),
                    host_full = full.load(SeqCst),
                    "xla warm programs loaded behind serving"
                );
            });
        match spawned {
            Ok(handle) => self.behind.handle = Some(handle),
            Err(why) => {
                tracing::warn!(%why, "xla programs not loaded behind serving: each loads in its first fire")
            }
        }
    }

    /// Keeps traced fire `d`'s record (`Device::trace_record`) for later
    /// loads, its program built and kept in the disk cache: the record names
    /// the file and its length as kept, all before ready.
    fn keep_record(&self, d: &super::Deferred) {
        let (Some(scope), crate::device::Code::Text(text)) = (self.trace_scope.as_deref(), &d.code)
        else {
            return;
        };
        let Some(name) = self.device.cached_name(text) else {
            return;
        };
        let Some(file_len) = self.device.cached_len(&name) else {
            return;
        };
        let record = super::TraceRecord {
            hash: d.code.hash(),
            name,
            len: text.len(),
            file_len,
            need: crate::device::compile_need(text),
            sig: d.sig.clone(),
            probed: d.probed.clone(),
        };
        if let Ok(bytes) = serde_json::to_vec(&record) {
            self.device.keep_trace_record(scope, &d.key, &bytes);
        }
    }

    /// Waits for the programs loading behind serving: every parked program
    /// the host holds room for is loaded once it returns.
    pub fn wait_loaded(&mut self) {
        if let Some(handle) = self.behind.handle.take() {
            let _ = handle.join();
        }
    }

    /// Compiles what the ladder traced, a thread a core up to 8, and seats
    /// each program under its keys; answers how many programs it left out.
    /// A program a record names is parked, loaded behind serving
    /// (`Shell::load_behind`).
    ///
    /// A program whose temporaries do not fit the chip's HBM beside the pool
    /// (PJRT's `RESOURCE_EXHAUSTED`) is left out, not the load: the ladder
    /// covers shapes a fire may never take, and one that does compiles then
    /// and fails there, as every shape did before the ladder (Qwen3.8-Flash-
    /// Next's mini on a v5e refused its whole load over one, "Ran out of
    /// memory on HBM" at m=9772). Any other compile failure fails the load.
    fn compile_deferred(&mut self, deferred: Vec<super::Deferred>) -> Result<usize> {
        let mut texts: Vec<(&crate::device::Code, &crate::trace::Signature)> = Vec::new();
        let mut index: Vec<usize> = Vec::with_capacity(deferred.len());
        let mut seen: std::collections::HashMap<[u8; 32], usize> = std::collections::HashMap::new();
        for d in &deferred {
            let at = *seen.entry(d.code.hash()).or_insert_with(|| {
                texts.push((&d.code, &d.sig));
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
        let need = texts
            .iter()
            .filter(|(c, _)| c.text().is_some())
            .map(|(c, _)| c.need())
            .max()
            .unwrap_or(0);
        let least = texts
            .iter()
            .filter(|(c, _)| c.text().is_some())
            .map(|(c, _)| c.need())
            .min()
            .unwrap_or(0);
        // A compiled program held in host memory takes what its load is
        // measured to take ([`HostRoom`]), and libtpu's allocator keeps what
        // a dropped one freed for its own next use, giving the host none
        // back: the ladder holds what the free memory does beside two
        // compiles' room, in ladder order, each load admitted while it fits
        // beside those under way. The rest are parked: compiled onto the disk
        // cache (not held) where they are not there yet, loaded (~1 s, never
        // compiled) by the first fire that takes their shape. With a disk
        // cache the ladder holds none it compiles while it compiles: a held
        // program's memory is a compile's room (Qwen3.6-35B-A3B's v5e ladder
        // held 25 GB of a 47 GB host by its end, its last compiles one at a
        // time); those the free memory holds load once the compiles are done.
        let parks = self.device.parks();
        // the programs' room: what the load states (`Residency::host_programs`),
        // else the host's free memory less two compiles'
        let stated = self.host_programs;
        let host_room = HostRoom::new(stated, 2 * need);
        let parked = std::sync::atomic::AtomicUsize::new(0);
        // The compiles' room, fixed here: two compiles', or what a stated
        // programs' room leaves of the free memory. A compile starts once
        // its own need and every running one's fit in it, woken by a compile
        // that ends (no poll of the free memory).
        let admission = Admission::new(match stated {
            Some(stated) => available_memory().map_or(u64::MAX, |b| b.saturating_sub(stated)),
            None => 2 * need,
        });
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
                        // A program a record names is loaded behind serving
                        // (`Shell::load_behind`), not before ready.
                        if text.text().is_none() {
                            parked.fetch_add(1, SeqCst);
                            continue;
                        }
                        let reserved = if !parks || device.code_on_disk(text) {
                            host_room.take(text.len())
                        } else {
                            None
                        };
                        let kept = reserved.is_some();
                        if !kept {
                            parked.fetch_add(1, SeqCst);
                            if device.code_on_disk(text) {
                                continue;
                            }
                        }
                        let program = {
                            // a cached program loads, never compiled
                            let _admitted = text.text().map(|t| admission.admit(t));
                            device.prepare_code(text, (*sig).clone())
                        };
                        if let Some(r) = reserved {
                            host_room.done(r, matches!(program, Ok(Some(_))));
                        }
                        // a text's program is built, never missing
                        let program = program.and_then(|p| {
                            p.ok_or_else(|| crate::error::Fault::Program {
                                at: "serve::warm",
                                why: "a traced program was neither built nor loaded".to_string(),
                            })
                        });
                        if kept || program.is_err() {
                            if let Ok(mut slot) = slots[at].lock() {
                                *slot = Some(program);
                            }
                        } else {
                            drop(program);
                            device.forget_code(text);
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
        // A parked program loads behind serving (`load_behind`), not here:
        // a load here filled the host before the guest programs registered
        // (v6e Qwen3.5 fresh cache: 115 GiB held, then a sampler stage's
        // compile OOM-killed the server at its cgroup's cap), and the loader
        // behind serving leaves the guest compiles' room
        // (`guest::run::guest_reserve`) and loads nothing beside one
        // measuring its need.
        let parked = parked.load(SeqCst);
        if parked > 0 {
            tracing::warn!(
                parked,
                held_gib = host_room.held() >> 30,
                "xla warm ladder parked programs on disk: the host's memory holds no more; the first fire of each one's shape loads it (`loaded_in_fire`)"
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
            // a traced fire's record, once its program is built and kept
            if !matches!(programs[at], Some(Err(_))) {
                self.keep_record(&d);
            }
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
                None => {
                    // parked in the disk cache: the first fire of its shape
                    // loads it (`Shell::load_parked`)
                    if let Some(shape_key) = d.shape_key {
                        self.parked_shapes.insert(shape_key, d.key);
                    }
                    self.parked.insert(
                        d.key,
                        super::Parked {
                            code: d.code,
                            sig: d.sig,
                            probed: d.probed,
                            order: d.order,
                        },
                    );
                    continue;
                }
            };
            self.probed.insert(d.key, d.probed);
            self.traced.insert(d.key, std::sync::Arc::clone(&program));
            if let Some(shape_key) = d.shape_key {
                self.shaped.insert(shape_key, program);
            }
        }
        Ok(left_out.len())
    }

    /// A fire's program as `canonical_layout` lays it out: its bucket, its
    /// treatment, and how many one-row and wider lanes each word takes once
    /// padded (a one-row lane runs the per-row walk, a wider one the chunked
    /// scan; how a prompt's rows split is data). `None`: the fire takes no
    /// canonical layout.
    fn layout_key(&self, lanes: &[(usize, u64, u32)], buffered: Option<u32>) -> Option<LayoutKey> {
        let real: u32 = lanes.iter().map(|l| l.2).sum();
        let pads = self.canonical_layout(lanes, real, buffered)?;
        let bucket = real + pads.iter().map(|p| p.0).sum::<u32>();
        // (word, any one-row lane, lanes)
        let mut counts: Vec<(u64, bool, u32)> = Vec::new();
        let all = lanes
            .iter()
            .map(|&(_, w, r)| (w, r))
            .chain(pads.iter().map(|&(r, w)| (w, r)));
        for (word, rows) in all {
            let one = rows <= 1;
            match counts.iter_mut().find(|c| c.0 == word) {
                Some(c) => {
                    c.1 |= one;
                    c.2 += 1;
                }
                None => counts.push((word, one, 1)),
            }
        }
        counts.sort_unstable();
        Some((bucket, buffered.is_some(), counts))
    }

    /// The compositions the stated set holds (`engine::StepShapes`: every
    /// prompt and decode count its mixed and prefill shapes span, their rows
    /// where padding changes its form) whose layout no shape of `laid`
    /// takes: a fire of one would compile after ready (v6e Qwen3.5 c64: 61
    /// prompts and a rest lane at 1792 rows, 67 s). Each with a shape that
    /// lays it.
    fn unlaid_compositions(&self, laid: &[Vec<Want>], decode: u64, prefill: u64) -> Gaps {
        let mut gaps = Gaps::default();
        let Some(steps) = self.steps.as_ref() else {
            return gaps;
        };
        let classes = &self.compiled.classes;
        let class = |word: u64| classes.class_of(word & classes.mask);
        let (Some(dc), Some(pc)) = (class(decode), class(prefill)) else {
            return gaps;
        };
        let rung_of_shape = |shape: &[Want]| {
            if shape.iter().any(|w| w.buffers) {
                steps
                    .fire_shape(&composition(shape, engine::Treatment::Buffers))
                    .ok()
                    .and_then(|f| steps.shapes.get(f.id as usize))
                    .map(|s| s.rung)
            } else {
                None
            }
        };
        let mut keys = std::collections::HashSet::new();
        for shape in laid {
            if shape.iter().all(|w| w.rows <= 1) || shape.iter().any(|w| w.all_rows || w.masked) {
                continue;
            }
            let lanes: Option<Vec<_>> = shape
                .iter()
                .map(|w| Some((class(w.word)?, w.word, w.rows)))
                .collect();
            if let Some(key) = lanes.and_then(|l| self.layout_key(&l, rung_of_shape(shape))) {
                keys.insert(key);
            }
        }
        let buckets = &self.budgets.tokens.buckets;
        let pad = self.pad;
        for stated in &steps.shapes {
            // (`attachments.mask`: a lane may state a mask, which is data)
            if stated.kind == engine::StepKind::Decode {
                continue;
            }
            let buffered = (stated.treatment == engine::Treatment::Buffers).then_some(stated.rung);
            let mixed = stated.kind == engine::StepKind::Mixed;
            // (a fire of prompts alone carries `prefill_lanes` at most: the
            // runtime forms no wider one)
            let widest = if mixed {
                stated.prompt_lanes.hi
            } else {
                stated.prompt_lanes.hi.min(self.prefill_lanes())
            };
            for n in stated.prompt_lanes.lo.max(1)..=widest {
                // a fire's lanes are the load's at most
                let most = self.budgets.tokens.max_lanes.saturating_sub(n);
                for d in stated.decode_lanes.lo..=stated.decode_lanes.hi.min(most) {
                    let mut prev = 0;
                    for &b in buckets {
                        let lo = prev + 1;
                        prev = b;
                        let exact = b.saturating_sub(pad.rung(n) - n);
                        let mut rows = if mixed {
                            vec![lo, b]
                        } else {
                            vec![lo, exact.saturating_sub(1), exact, b - 1, b]
                        };
                        rows.sort_unstable();
                        rows.dedup();
                        for r in rows {
                            let p = r.saturating_sub(d);
                            if r < lo
                                || !(stated.rows.lo..=stated.rows.hi).contains(&r)
                                || !(stated.prompt_rows.lo..=stated.prompt_rows.hi).contains(&p)
                                || p < n * stated.lane_rows.lo.max(2)
                                || p > n * stated.lane_rows.hi
                            {
                                continue;
                            }
                            let each = p / n;
                            let mut lanes = vec![(dc, decode, 1); d as usize];
                            lanes.extend((0..n).map(|i| {
                                (pc, prefill, if i == 0 { p - each * (n - 1) } else { each })
                            }));
                            let Some(key) = self.layout_key(&lanes, buffered) else {
                                // no canonical layout: a fire of it keys on
                                // its own counts (`fire_inner`'s one pad lane)
                                gaps.uncanonical += 1;
                                if gaps.first_uncanonical.len() < 4 {
                                    gaps.first_uncanonical.push(format!(
                                        "{} {n} prompts of {p} rows beside {d} decodes",
                                        stated.treatment
                                    ));
                                }
                                continue;
                            };
                            if keys.insert(key) {
                                let shape: Vec<Want> = lanes
                                    .iter()
                                    .map(|&(_, word, rows)| Want {
                                        word,
                                        rows,
                                        held_pages: u32::from(rows == 1),
                                        all_rows: false,
                                        masked: false,
                                        buffers: buffered.is_some(),
                                    })
                                    .collect();
                                gaps.unlaid.push((
                                    format!(
                                        "{} {} prompts of {p} rows beside {d} decodes ({r} rows)",
                                        stated.treatment, n
                                    ),
                                    shape,
                                ));
                            }
                        }
                    }
                }
            }
        }
        gaps
    }

    fn warm_fire(&mut self, shape: &[Want]) -> Result<()> {
        let slots = self.held.len().max(1) as u32;
        let page_size = self.paging().page_size.max(1);
        // Every lane reads the same scratch slots: what warm fires write is
        // never read back, and the pool need not hold a page per lane.
        type Owned = (
            Vec<u32>,
            crate::store::Scratch,
            Vec<u32>,
            Option<engine::fire::Masking>,
        );
        let owned: Vec<Owned> = shape
            .iter()
            .map(|want| {
                let keys = held_of(want, page_size) + want.rows;
                let n = keys.div_ceil(page_size).max(1);
                // every key allowed
                let mask = want.masked.then(|| {
                    engine::fire::Masking::Extent(engine::fire::Mask::new(
                        vec![0, keys],
                        u64::from(keys),
                    ))
                });
                (
                    vec![0; want.rows as usize],
                    self.pools.scratch(0, n as usize),
                    (0..want.rows).collect(),
                    mask,
                )
            })
            .collect();
        for slot in 0..(shape.len() as u32).min(slots) {
            self.open(slot)?;
        }
        let windowed_state = self.pools.layout().ring().is_some();
        // A buffered lane buffers what it lands, carrying nothing in: the
        // buffered program is one a shape whatever a fire carries.
        let verbs: Vec<Option<engine::fire::RsVerb>> = shape
            .iter()
            .zip(&owned)
            .map(|(want, (_, scratch, _, _))| {
                want.buffers.then(|| engine::fire::RsVerb::Buffer {
                    pages: scratch.rs.clone(),
                    base: 0,
                    folded: held_of(want, page_size),
                    fold: engine::fire::FoldLen::Host(0),
                    lineage: engine::fire::Lineage::Set,
                })
            })
            .collect();
        let seated: Vec<Seated<'_>> = shape
            .iter()
            .zip(&owned)
            .zip(&verbs)
            .enumerate()
            .map(|(at, ((want, (tokens, scratch, every, mask)), verb))| {
                let mut seated = Seated::of(Lane {
                    slot: at as u32 % slots,
                    word: want.word,
                    tokens,
                });
                seated.pages = &scratch.pages;
                seated.fixed = &scratch.fixed;
                seated.held = Some(held_of(want, page_size));
                if !windowed_state {
                    seated.window = &scratch.window;
                }
                if want.all_rows {
                    seated.readout = Some(every);
                }
                seated.mask = mask.as_ref();
                if let Some(verb) = verb {
                    seated.rs = verb;
                }
                seated
            })
            .collect();
        self.scratch_fire = true;
        // a buffered want fires as its stated shape does: at its rung
        if shape.iter().any(|w| w.buffers)
            && let Some(steps) = &self.steps
        {
            let stated = steps
                .fire_shape(&composition(shape, engine::Treatment::Buffers))
                .ok();
            self.shape_next(stated);
        }
        let fired = self.fire_full(&seated).map(|_| ());
        self.scratch_fire = false;
        fired
    }
}

/// A lane at page bound `p` holds `p · page_size` rows less its own, so
/// its last row lands in the last slot of its `p`-th page; a fresh lane
/// holds none.
fn held_of(want: &Want, page_size: u32) -> u32 {
    if want.held_pages == 0 {
        return 0;
    }
    (want.held_pages * page_size).saturating_sub(want.rows)
}

/// The ladder part shape `at` belongs to: the last that starts at or before it.
/// Where a ladder part's programs stand in the order they load behind
/// serving: what every request's steps take first (decode, and the drafts
/// a speculative decode fires beside it), then mixed fires, then prompts.
fn use_rank(part: &str) -> u8 {
    match part {
        "decode" => 0,
        "draft windows" => 1,
        "draft prompts" => 2,
        "mixed" | "stated" => 3,
        "prefill" => 4,
        _ => 5,
    }
}

fn warm_part<'a>(parts: &[(&'a str, usize)], at: usize) -> &'a str {
    parts
        .iter()
        .rev()
        .find(|(_, from)| *from <= at)
        .map_or("decode", |(name, _)| name)
}

/// The lane counts that reach every form a fire padded to lane rung `p`
/// takes (`canonical_pads`): at the rung, one short of it, and just past
/// half of it.
fn lane_counts(p: u32) -> Vec<u32> {
    let mut ns = vec![p / 2 + 1, p.saturating_sub(1), p];
    ns.retain(|&n| n >= 1 && n <= p);
    ns.sort_unstable();
    ns.dedup();
    ns
}

/// The samples a width of loading behind serving takes, the fastest its
/// rate: as many as the rates' curves time a point (engine-gpu
/// `rates::REPEATS`).
const LOAD_SAMPLES: usize = 3;

/// The measured loader width `knee`, kept in the user's file `width` names
/// under its key ([`crate::stored`]), said where it lies; a file that would
/// not take it is said, and the width serves this load either way.
fn keep_width(width: &super::LoadWidth, knee: usize) {
    let Some((path, on)) = &width.keep else {
        return;
    };
    let save = engine::rates::SaveTo {
        path: path.clone(),
        provenance: (
            "rates",
            format!(
                "\"a load on {} (pie {}): `load_width`, cached programs loaded from the disk \
                 cache behind serving, the width grown by one while the fastest of \
                 {LOAD_SAMPLES} samples of the load rate rose; at {}\"",
                on.replace('"', "'"),
                env!("CARGO_PKG_VERSION"),
                engine::rates::build_commit()
            ),
        ),
        registry: crate::chip::registry(),
    };
    let measured = profile::rates::Rates {
        load_width: u32::try_from(knee).ok(),
        measured_on: Some(on.clone()),
        measured_by: Some(vec![(
            profile::rates::LOAD_WIDTH.to_string(),
            crate::stored::load_width_by(),
        )]),
        ..profile::rates::Rates::default()
    };
    match engine::rates::save_rates(&save, &measured) {
        Ok(saved) => tracing::info!(profile = %saved, width = knee, "xla loader width kept"),
        Err(why) => tracing::warn!(%why, "xla loader width not kept: measured again next load"),
    }
}
