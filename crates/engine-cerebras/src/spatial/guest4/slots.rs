//! A multi-lane VM's per-lane state the host lays out (the guest4hz image's
//! step input and per-lane programs): each lane's step input (`gsdesc`, and `gsin` every launch),
//! and on an image with per-lane programs (`vm_lprog`) the op arena every
//! lane's table sits in (`gops`), each lane's window of the bank arena
//! (`glane`, `gbdesc`, `gbank`).
//!
//! The images here are a VM column's: rows 0..=H, the hub at the feeder
//! row (no vocabulary rows), each row's words back to back.

use std::sync::Arc;

use super::Table;
use super::wafer::{Seed, VmLayout, pack_ops};

type R<T> = std::result::Result<T, String>;

/// Entries a lane's step input has (`gsdesc`: 4 a lane, 3 words each).
pub const STEP_ENTRIES: usize = 4;
/// Pool ranges a lane's window has (`gbdesc`: 8 a lane).
pub const WINDOW_RANGES: usize = 8;

/// One home the host writes each step: a mask (`kind` 1: `n` rows into
/// registers `dst`.., one word a row and a PE, bit i its vocabulary row i)
/// or pool words (`kind` 2: `n` words at `dst`), from block word `src`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepEntry {
    pub chan: u32,
    pub kind: u32,
    pub dst: u32,
    pub src: u32,
    pub n: u32,
}

/// `table`'s step input on `lay`: masks first, then pool homes, in
/// `per_step` order; refused past the image's `vm_sin` words or 4 entries,
/// and for a vocabulary-wide input that is not a mask.
pub fn step_entries(lay: &VmLayout, table: &Table) -> R<Vec<StepEntry>> {
    let mut masks = Vec::new();
    let mut pools = Vec::new();
    for &chan in &table.per_step {
        let home = table
            .homes
            .iter()
            .find(|h| h.chan == chan)
            .ok_or_else(|| format!("per-step channel {chan} has no home"))?;
        if home.kind == "wide" {
            if home.dtype != eta_ir::Dtype::Bool {
                return Err(format!(
                    "per-step channel {chan} is vocabulary-wide {:?}, not a mask",
                    home.dtype
                ));
            }
            let rows = (home.n as usize).div_ceil(lay.vocab.max(1)).max(1) as u32;
            masks.push((chan, 1, home.at, rows));
        } else {
            pools.push((chan, 2, home.at, home.n));
        }
    }
    let mut src = 0;
    let entries: Vec<StepEntry> = masks
        .into_iter()
        .chain(pools)
        .map(|(chan, kind, dst, n)| {
            let e = StepEntry {
                chan,
                kind,
                dst,
                src,
                n,
            };
            src += n;
            e
        })
        .collect();
    if entries.len() > STEP_ENTRIES || src as usize > lay.sin {
        return Err(format!(
            "step input of {src} words in {} entries; the image takes {} words in {STEP_ENTRIES}",
            entries.len(),
            lay.sin
        ));
    }
    Ok(entries)
}

/// `gsdesc` (lanes x 12 words): each lane's entries `[kind, dst, src | n <<
/// 16]`, unused ones 0.
pub fn gsdesc(lanes: &[Vec<StepEntry>]) -> Vec<u32> {
    let mut words = vec![0u32; lanes.len() * STEP_ENTRIES * 3];
    for (l, entries) in lanes.iter().enumerate() {
        for (k, e) in entries.iter().enumerate() {
            let at = (l * STEP_ENTRIES + k) * 3;
            words[at..at + 3].copy_from_slice(&[e.kind, e.dst, e.src | e.n << 16]);
        }
    }
    words
}

/// A lane's step block for lane position `pos` (`sin` words): a mask
/// row's word bit i is vocabulary row `pos·Vt + i` (0 past the vocabulary
/// and on the hub, `pos` `None`), pool words as they are. `values` are the
/// entries' channel words, in entry order.
fn block(
    lay: &VmLayout,
    entries: &[StepEntry],
    values: &[Vec<u32>],
    pos: Option<usize>,
) -> Vec<u32> {
    let mut out = vec![0u32; lay.sin];
    for (e, words) in entries.iter().zip(values) {
        for k in 0..e.n as usize {
            out[e.src as usize + k] = if e.kind == 1 {
                pos.map_or(0, |pos| {
                    (0..lay.vr.min(32)).fold(0u32, |w, i| {
                        let v = pos * lay.vr + i;
                        let on =
                            v < lay.vocab && words.get(k * lay.vocab + v).is_some_and(|&x| x != 0);
                        w | u32::from(on) << i
                    })
                })
            } else {
                words.get(k).copied().unwrap_or(0)
            };
        }
    }
    out
}

/// A lane's step input for one launch: its entries and their channels'
/// words (`None`: a lane without).
pub type StepInput = Option<(Vec<StepEntry>, Vec<Vec<u32>>)>;

/// `gsin` for one lap of `lanes.len()` turns, a VM column at a time (rows
/// 0..=H, `depth` slots of `sin` words each): the VM puts slot l before
/// lane l's turn and slot L + l before its next step's pre section (run
/// again at the next launch, so the same block stands there). `lanes[l]`:
/// lane l's entries and their channel words, `None` for a lane without.
pub fn step_images(lay: &VmLayout, depth: usize, lanes: &[StepInput]) -> Vec<Vec<u32>> {
    let (nv, sin) = (lay.rows + 1, lay.sin);
    let n = lanes.len();
    lay.columns()
        .map(|(b, j, _)| {
            let mut col = vec![0u32; nv * depth * sin];
            for (l, lane) in lanes.iter().enumerate() {
                let Some((entries, values)) = lane else {
                    continue;
                };
                for pe in 0..nv {
                    let words = block(lay, entries, values, lay.position(b, j, pe));
                    for slot in [l, n + l]
                        .into_iter()
                        .filter(|&s| s < depth)
                        .collect::<Vec<_>>()
                    {
                        let at = (pe * depth + slot) * sin;
                        col[at..at + sin].copy_from_slice(&words);
                    }
                }
            }
            col
        })
        .collect()
}

/// What a lane carries between its turns on a per-lane-program image: pool
/// ranges `(offset, words)` (at most 8) and wide registers past W0. The
/// translator's live state at main's start, and every home besides: the
/// host reads its pool homes from the window after a launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub pool: Vec<(u32, u32)>,
    pub regs: Vec<u32>,
}

impl Window {
    pub fn of(table: &Table) -> Window {
        let mut ranges: Vec<(u32, u32)> = table
            .bank_pool
            .iter()
            .copied()
            .chain(
                table
                    .homes
                    .iter()
                    .filter(|h| h.kind == "pool")
                    .map(|h| (h.at, h.n)),
            )
            .filter(|&(_, n)| n > 0)
            .collect();
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::new();
        for (o, n) in ranges {
            match merged.last_mut() {
                Some((lo, ln)) if o <= *lo + *ln => *ln = (*ln).max(o + n - *lo),
                _ => merged.push((o, n)),
            }
        }
        // Down to 8 over the smallest gaps.
        while merged.len() > WINDOW_RANGES {
            let k = (0..merged.len() - 1)
                .min_by_key(|&i| merged[i + 1].0 - (merged[i].0 + merged[i].1))
                .unwrap_or(0);
            let (o, _) = merged[k];
            let (o2, n2) = merged.remove(k + 1);
            merged[k].1 = o2 + n2 - o;
        }
        let mut regs: Vec<u32> = table.bank_regs.clone();
        regs.retain(|&r| r > 0);
        regs.sort_unstable();
        regs.dedup();
        Window { pool: merged, regs }
    }

    /// Its words in the bank arena (Vt a register).
    pub fn words(&self, vt: usize) -> usize {
        self.pool.iter().map(|&(_, n)| n as usize).sum::<usize>() + self.regs.len() * vt
    }
}

/// A lane's program on a per-lane-program image: `key` names the table
/// (the program it was translated from).
#[derive(Clone)]
pub struct LaneProgram {
    pub key: u64,
    pub program: Arc<super::lane::Program>,
}

/// A lane's window off the VM (evicted): its program, its window and the
/// words every VM PE held for it, a VM column each.
#[derive(Clone, Debug)]
pub struct Parked {
    pub key: u64,
    pub window: Window,
    pub words: Vec<Vec<u32>>,
}

/// Where each lane's program and state sit: the op arena (each distinct
/// table once) and each lane's table base and window offset.
#[derive(Clone, Default)]
pub struct Slots {
    pub arena: Vec<[u32; 6]>,
    pub lanes: Vec<Option<LaneSlot>>,
}

#[derive(Clone)]
pub struct LaneSlot {
    pub key: u64,
    pub base: u32,
    pub tok: u32,
    pub off: usize,
    pub window: Window,
    pub steps: Vec<StepEntry>,
}

impl Slots {
    /// The arena and windows for `lanes` (packed from 0, lane order: a
    /// change compacts both), refused past the image's `max_ops` or
    /// `bank_words`.
    pub fn plan(lay: &VmLayout, lanes: &[Option<LaneProgram>]) -> R<Slots> {
        let mut arena: Vec<[u32; 6]> = Vec::new();
        let mut bases: Vec<(u64, u32)> = Vec::new();
        let mut off = 0;
        let mut out = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let Some(p) = lane else {
                out.push(None);
                continue;
            };
            let table = &p.program.table;
            if let Some(why) = lay.refuses(table) {
                return Err(why);
            }
            let base = match bases.iter().find(|(k, _)| *k == p.key) {
                Some(&(_, base)) => base,
                None => {
                    let base = arena.len() as u32;
                    arena.extend_from_slice(&table.ins);
                    bases.push((p.key, base));
                    base
                }
            };
            let window = Window::of(table);
            let slot = LaneSlot {
                key: p.key,
                base,
                tok: table.tok_home.unwrap_or_default(),
                off,
                steps: if table.per_step.is_empty() {
                    Vec::new()
                } else {
                    step_entries(lay, table)?
                },
                window,
            };
            off += slot.window.words(lay.vr);
            out.push(Some(slot));
        }
        if arena.len() > lay.max_ops {
            return Err(format!(
                "an op arena of {} instructions; the image holds {}",
                arena.len(),
                lay.max_ops
            ));
        }
        if lanes.len() > 1 && off > lay.bank_words {
            return Err(format!(
                "lane windows of {off} words; the image's bank holds {}",
                lay.bank_words
            ));
        }
        Ok(Slots { arena, lanes: out })
    }

    /// `gops`: the arena packed, `max_ops` instructions.
    pub fn ops(&self, lay: &VmLayout) -> R<Vec<u32>> {
        pack_ops(&self.arena, lay.max_ops)
    }

    /// `glane` (4 words a lane: table base, with its rows a turn above bit 16
    /// where the VM reads them, token home, window offset, register mask)
    /// and `gbdesc` (8 a lane: pool ranges `offset | words << 16`); a lane
    /// without a program is all zeros (the VM's empty lane).
    pub fn glane_gbdesc(&self, lay: &VmLayout) -> (Vec<u32>, Vec<u32>) {
        let n = self.lanes.len();
        let (mut glane, mut gbdesc) = (vec![0u32; n * 4], vec![0u32; n * WINDOW_RANGES]);
        for (l, slot) in self.lanes.iter().enumerate() {
            let Some(s) = slot else {
                continue;
            };
            let mask = s.window.regs.iter().fold(0u32, |m, &r| m | 1 << r);
            let rows = if lay.lane_rows { 1 << 16 } else { 0 };
            glane[l * 4..l * 4 + 4].copy_from_slice(&[s.base | rows, s.tok, s.off as u32, mask]);
            for (k, &(o, w)) in s.window.pool.iter().enumerate() {
                gbdesc[l * WINDOW_RANGES + k] = o | w << 16;
            }
        }
        (glane, gbdesc)
    }

    /// `gsdesc` for every lane.
    pub fn gsdesc(&self) -> Vec<u32> {
        gsdesc(
            &self
                .lanes
                .iter()
                .map(|s| s.as_ref().map_or_else(Vec::new, |s| s.steps.clone()))
                .collect::<Vec<_>>(),
        )
    }

    /// Lane `l`'s window as the bank images `banks` hold it: what it needs
    /// to come back exactly ([`Slots::rebank`]'s `parked`) after it leaves.
    pub fn park(&self, lay: &VmLayout, l: usize, banks: &[Vec<u32>]) -> Option<Parked> {
        let s = self.lanes.get(l)?.as_ref()?;
        let (n, bw) = (s.window.words(lay.vr), lay.bank_words);
        let words = banks
            .iter()
            .map(|bank| {
                (0..=lay.rows)
                    .flat_map(|pe| bank[pe * bw + s.off..pe * bw + s.off + n].iter().copied())
                    .collect()
            })
            .collect();
        Some(Parked {
            key: s.key,
            window: s.window.clone(),
            words,
        })
    }

    /// The bank arena after a change from `old` (its images `banks`, a VM
    /// column each, `bank_words` a row): a lane in `parked` on the program
    /// and window it left gets its words back, a lane in `seeds` starts
    /// from them, a lane on the same program keeps its window's words, any
    /// other is zero.
    pub fn rebank(
        &self,
        lay: &VmLayout,
        old: &Slots,
        banks: &[Vec<u32>],
        seeds: &[Option<Vec<Seed>>],
        parked: &[Option<Parked>],
    ) -> R<Vec<Vec<u32>>> {
        let (nv, bw) = (lay.rows + 1, lay.bank_words);
        let cols: Vec<(usize, usize, usize)> = lay.columns().collect();
        let mut out = vec![vec![0u32; nv * bw]; cols.len()];
        for (l, slot) in self.lanes.iter().enumerate() {
            let Some(s) = slot else {
                continue;
            };
            let n = s.window.words(lay.vr);
            if let Some(Some(p)) = parked.get(l)
                && p.key == s.key
                && p.window == s.window
                && p.words.len() == out.len()
            {
                for (column, words) in out.iter_mut().zip(&p.words) {
                    for pe in 0..nv {
                        column[pe * bw + s.off..pe * bw + s.off + n]
                            .copy_from_slice(&words[pe * n..(pe + 1) * n]);
                    }
                }
            } else if let Some(Some(seeds)) = seeds.get(l) {
                let (pool, wide) = super::wafer::seed_images(lay, seeds)?;
                for (column, &(b, j, _)) in out.iter_mut().zip(&cols) {
                    for pe in 0..nv {
                        let at = pe * bw + s.off;
                        let mut k = at;
                        for &(o, w) in &s.window.pool {
                            column[k..k + w as usize]
                                .copy_from_slice(&pool[o as usize..(o + w) as usize]);
                            k += w as usize;
                        }
                        for &r in &s.window.regs {
                            if let Some(lane) = lay.position(b, j, pe) {
                                for i in 0..lay.vr {
                                    column[k + i] =
                                        wide[(lane * lay.nreg + r as usize) * lay.vr + i].to_bits();
                                }
                            }
                            k += lay.vr;
                        }
                    }
                }
            } else if let Some(Some(was)) = old.lanes.get(l)
                && was.key == s.key
                && was.window == s.window
            {
                for (column, bank) in out.iter_mut().zip(banks) {
                    for pe in 0..nv {
                        let (from, to) = (pe * bw + was.off, pe * bw + s.off);
                        column[to..to + n].copy_from_slice(&bank[from..from + n]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Each lane's pool (`pool_words`, its window's ranges filled) from the
    /// hub's bank row: its pool homes as its last turn left them.
    pub fn pools(&self, lay: &VmLayout, hub_bank: &[u32]) -> Vec<Vec<u32>> {
        self.lanes
            .iter()
            .map(|slot| {
                let mut pool = vec![0u32; lay.pool_words];
                if let Some(s) = slot {
                    let mut k = s.off;
                    for &(o, w) in &s.window.pool {
                        let (o, w) = (o as usize, w as usize);
                        if let (Some(dst), Some(src)) =
                            (pool.get_mut(o..o + w), hub_bank.get(k..k + w))
                        {
                            dst.copy_from_slice(src);
                        }
                        k += w;
                    }
                }
                pool
            })
            .collect()
    }
}

/// The opcodes a table uses (bit o), for the image's op mask.
pub fn opcodes(table: &Table) -> u64 {
    table
        .ins
        .iter()
        .fold(0, |m, i| m | 1u64.checked_shl(i[0] & 0xff).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::super::isa;
    use super::*;
    use eta_ir::Dtype;

    fn lay() -> VmLayout {
        VmLayout {
            x0: 0,
            blocks: 1,
            rows: 2,
            cols: 1,
            vr: 32,
            block_cols: 5,
            root: 2,
            feeder: 1,
            max_ops: 16,
            nreg: 4,
            pool_words: 16,
            feat: 0xffff,
            vocab: 50,
            opmask: u64::MAX,
            sin: 4,
            lprog: true,
            bank_words: 200,
            set_b_cols: 0,
            lane_rows: false,
        }
    }

    fn table(ins: usize, homes: Vec<super::super::Home>, per_step: Vec<u32>) -> Table {
        Table {
            ins: vec![[isa::QCONST, 0, 0, 0, 0, 0]; ins],
            pre_len: 0,
            homes,
            outputs: Vec::new(),
            wide_outputs: Vec::new(),
            per_step,
            tok_home: Some(0),
            nreg: 2,
            pool_words: 4,
            feat: 0,
            remat: false,
            bank_regs: vec![1],
            bank_pool: vec![(2, 2)],
        }
    }

    fn prog(key: u64, t: &Arc<Table>) -> Option<LaneProgram> {
        let t = Table {
            ins: t.ins.clone(),
            homes: t
                .homes
                .iter()
                .map(|h| home(h.chan, h.kind, h.at, h.n, h.dtype))
                .collect(),
            ..table(0, Vec::new(), Vec::new())
        };
        Some(LaneProgram {
            key,
            program: Arc::new(super::super::lane::Program {
                table: t,
                epilogue: 0,
                homes: Default::default(),
                logits: String::new(),
            }),
        })
    }

    fn home(chan: u32, kind: &'static str, at: u32, n: u32, dtype: Dtype) -> super::super::Home {
        super::super::Home {
            chan,
            kind,
            at,
            n,
            dtype,
        }
    }

    #[test]
    fn a_mask_goes_first_and_packs_a_pe_s_rows_as_bits() {
        let t = table(
            2,
            vec![
                home(5, "pool", 3, 1, Dtype::F32),
                home(7, "wide", 1, 50, Dtype::Bool),
            ],
            vec![5, 7],
        );
        let lay = lay();
        let e = step_entries(&lay, &t).unwrap();
        assert_eq!(
            e,
            vec![
                StepEntry {
                    chan: 7,
                    kind: 1,
                    dst: 1,
                    src: 0,
                    n: 1
                },
                StepEntry {
                    chan: 5,
                    kind: 2,
                    dst: 3,
                    src: 1,
                    n: 1
                },
            ]
        );
        assert_eq!(
            gsdesc(std::slice::from_ref(&e))[..6],
            [1, 1, 1 << 16, 2, 3, 1 | 1 << 16]
        );
        // Rows 0, 33 and 49 allowed: position 0 bit 0, position 1 bits 1 and 17.
        let mut mask = vec![0u32; 50];
        mask[0] = 1;
        mask[33] = 1;
        mask[49] = 1;
        let values = vec![mask, vec![0x3f80_0000]];
        let imgs = step_images(&lay, 8, &[Some((e, values))]);
        let col = &imgs[0];
        let at = |pe: usize, slot: usize| (pe * 8 + slot) * 4;
        assert_eq!(col[at(0, 0)..at(0, 0) + 2], [1, 0x3f80_0000]);
        assert_eq!(col[at(1, 0)..at(1, 0) + 2], [0, 0x3f80_0000]); // the hub
        assert_eq!(col[at(2, 0)..at(2, 0) + 2], [1 << 1 | 1 << 17, 0x3f80_0000]);
        assert_eq!(col[at(2, 1)..at(2, 1) + 2], [1 << 1 | 1 << 17, 0x3f80_0000]);
    }

    #[test]
    fn a_vocabulary_wide_float_input_or_too_many_words_is_refused() {
        let lay = lay();
        let t = table(2, vec![home(7, "wide", 1, 50, Dtype::F32)], vec![7]);
        assert!(step_entries(&lay, &t).is_err());
        let t = table(2, vec![home(5, "pool", 0, 5, Dtype::U32)], vec![5]);
        assert!(step_entries(&lay, &t).unwrap_err().contains("5 words"));
        assert!(lay.refuses(&t).is_some());
        assert!(
            VmLayout { sin: 0, ..lay }
                .refuses(&table(2, vec![home(5, "pool", 0, 1, Dtype::U32)], vec![5]))
                .is_some()
        );
    }

    #[test]
    fn an_opcode_outside_the_mask_is_refused() {
        let lay = VmLayout {
            opmask: !(1u64 << isa::QCONST),
            ..lay()
        };
        assert!(
            lay.refuses(&table(1, Vec::new(), Vec::new()))
                .is_some_and(|w| w.contains("opcodes"))
        );
    }

    #[test]
    fn lanes_share_a_table_and_pack_windows_in_lane_order() {
        let lay = lay();
        let a = Arc::new(table(
            3,
            vec![home(1, "pool", 0, 1, Dtype::U32)],
            Vec::new(),
        ));
        let b = Arc::new(table(5, Vec::new(), Vec::new()));
        let lanes = vec![prog(1, &a), None, prog(2, &b), prog(1, &a)];
        let s = Slots::plan(&lay, &lanes).unwrap();
        assert_eq!(s.arena.len(), 8);
        // a: pool [0,1) and [2,4), W1: 3 + 32 words; b: [2,4), W1: 34.
        let (glane, gbdesc) = s.glane_gbdesc(&lay);
        assert_eq!(glane, [0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 35, 2, 0, 0, 69, 2]);
        // guest4hz on: a live lane's one row a turn above the base; the
        // idle lane stays all zeros (empty).
        let (hz, _) = s.glane_gbdesc(&VmLayout {
            lane_rows: true,
            ..lay
        });
        assert_eq!(hz[..8], [1 << 16, 0, 0, 2, 0, 0, 0, 0]);
        assert_eq!(hz[8], 3 | 1 << 16);
        assert_eq!(gbdesc[..2], [1 << 16, 2 | 2 << 16]);
        assert_eq!(gbdesc[16], 2 | 2 << 16);
        // Too small a bank, too small an arena.
        assert!(
            Slots::plan(
                &VmLayout {
                    bank_words: 100,
                    ..lay
                },
                &lanes
            )
            .is_err()
        );
        assert!(Slots::plan(&VmLayout { max_ops: 7, ..lay }, &lanes).is_err());
    }

    #[test]
    fn a_lane_keeps_its_window_when_others_move() {
        let lay = lay();
        let a = Arc::new(table(3, Vec::new(), Vec::new()));
        let b = Arc::new(table(5, Vec::new(), Vec::new()));
        let old = Slots::plan(&lay, &[prog(2, &b), prog(1, &a)]).unwrap();
        let (nv, bw) = (lay.rows + 1, lay.bank_words);
        let mut bank = vec![0u32; nv * bw];
        for pe in 0..nv {
            for k in 0..34 {
                bank[pe * bw + 34 + k] = 100 + k as u32;
            }
        }
        // Lane 0 leaves: lane 1's window moves to 0 with its words.
        let new = Slots::plan(&lay, &[None, prog(1, &a)]).unwrap();
        let out = new.rebank(&lay, &old, &[bank], &[None, None], &[]).unwrap();
        assert_eq!(out[0][bw..bw + 3], [100, 101, 102]);
        let pools = new.pools(&lay, &out[0][bw..2 * bw]);
        assert_eq!(pools[1][2..4], [100, 101]);
    }

    #[test]
    fn a_parked_window_comes_back_word_for_word_in_another_bank() {
        let lay = lay();
        let a = Arc::new(table(
            3,
            vec![home(0, "pool", 0, 1, Dtype::U32)],
            Vec::new(),
        ));
        let b = Arc::new(table(5, Vec::new(), Vec::new()));
        let old = Slots::plan(&lay, &[prog(1, &a), prog(2, &b)]).unwrap();
        let (nv, bw) = (lay.rows + 1, lay.bank_words);
        let bank: Vec<u32> = (0..nv * bw).map(|w| 1000 + w as u32).collect();
        let n = old.lanes[0].as_ref().unwrap().window.words(lay.vr);
        let parked = old.park(&lay, 0, std::slice::from_ref(&bank)).unwrap();
        // Lane 0 leaves; it comes back in bank 1, the other program in 0.
        let new = Slots::plan(&lay, &[prog(2, &b), prog(1, &a)]).unwrap();
        let off = new.lanes[1].as_ref().unwrap().off;
        let out = new
            .rebank(
                &lay,
                &old,
                std::slice::from_ref(&bank),
                &[None, Some(Vec::new())],
                &[None, Some(parked.clone())],
            )
            .unwrap();
        for pe in 0..nv {
            assert_eq!(
                out[0][pe * bw + off..pe * bw + off + n],
                bank[pe * bw..pe * bw + n]
            );
        }
        // Another program in that bank does not take the parked words.
        let other = Slots::plan(&lay, &[prog(2, &b), prog(3, &a)]).unwrap();
        let out = other
            .rebank(&lay, &old, &[bank], &[None, None], &[None, Some(parked)])
            .unwrap();
        assert!(out[0][off..off + n].iter().all(|&w| w == 0));
    }
}
