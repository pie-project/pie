//! §6.2 beam geometry host-replay (Design X — the M3-G2 driver gate's runtime half).
//!
//! The beam epilogue's geometry channels (pages/lens/np/klen/kvm/w_slot/w_off)
//! are DEVICE-produced — `map_geometry` can't resolve them and the single-page
//! projection can't express the [B,P] (beam × pages-per-beam) layout the beam
//! attention needs. Design X (host-replay, correctness-first) replays the
//! epilogue's freeze / designated-child / page-turn arithmetic HOST-SIDE from:
//!   * the harvested `out_par` channel (ch14, a Reader) = the device's per-lane
//!     `parent` (which beam each survivor forked from — the one decision we
//!     can't recompute without the logits), and
//!   * host-tracked previous geometry (pages/lens/np/tfill/tslot), and
//!   * the `fresh` slot grants the host issues each step,
//! producing the [B,P] `pages` (working-set slot ids) + `np`/`klen`/`kvm` +
//! `w_slot`/`w_off`. The caller then resolves slot→physical (in Rust, via the
//! working set + arena) and ships the STANDARD decode wire (physical
//! `kv_page_indices`, `kv_last_page_lens`, kvm→BRLE `masks`, 1 query/beam) —
//! the driver never sees slot ids and needs no new schema field (charlie's gate
//! derives the write target from `klen-1`).
//!
//! This module mirrors `s6_2_beam_epilogue_binds`
//! (sdk/rust/ptir/tests/p1_overview.rs) VERBATIM — it IS the golden contract.
//! Replay drift is bounded + caught by charlie's 3 driver goldens (continue-tail
//! / page-turn / fork-freeze), reproduced as unit tests below.

use pie_driver_abi::Brle;

/// Host-tracked beam geometry carried across fires (owned by the `Pipeline`
/// instance). Slot ids are working-set slots (resolved to physical pages at
/// fire time); `page_t` is the KV page size in tokens.
#[derive(Debug, Clone)]
pub struct BeamState {
    /// Beam width `B`.
    pub b: usize,
    /// Max pages per beam `P`.
    pub p: usize,
    /// KV page size in tokens (`PAGE_T`).
    pub page_t: u32,
    /// `[B*P]` per-beam page slot ids (row-major, padded to `P`).
    pub pages: Vec<u32>,
    /// `[B*P]` per-page valid fill count (row-major).
    pub lens: Vec<u32>,
    /// `[B]` live page count per beam.
    pub np: Vec<u32>,
    /// `[B]` tail-page fill (offset within the current write page).
    pub tfill: Vec<u32>,
    /// `[B]` tail-page slot id (the page currently being written).
    pub tslot: Vec<u32>,
}

/// One fire's replayed [B,P] geometry (slot-space — resolve to physical before
/// the wire). Byte-for-byte the epilogue's channel outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeamGeometry {
    /// `[B*P]` page slot ids (row-major).
    pub pages: Vec<u32>,
    /// `[B]` live page count per beam (drop the padding tail by this COUNT).
    pub np: Vec<u32>,
    /// `[B]` physical KV span per beam.
    pub klen: Vec<u32>,
    /// `[B*P*page_t]` per-cell validity mask (0/1).
    pub kvm: Vec<u8>,
    /// `[B]` slot the current token's K/V is written into.
    pub w_slot: Vec<u32>,
    /// `[B]` offset-in-page the current token is written at.
    pub w_off: Vec<u32>,
    /// `[B]` write mode: `true` = HEIR (continues a shared tail page in-place →
    /// `write_slot_shared_inplace`); `false` = FORK/fresh page (`cow_write_slot`).
    pub w_cont: Vec<bool>,
}

impl BeamState {
    /// Seed the state for a fresh single-token-per-beam start (`np=1`, one token
    /// in each beam's page): `slot0[b]` is beam `b`'s initial page slot.
    pub fn seeded(b: usize, p: usize, page_t: u32, slot0: &[u32]) -> Self {
        assert_eq!(slot0.len(), b, "one seed slot per beam");
        let mut pages = vec![0u32; b * p];
        let mut lens = vec![0u32; b * p];
        for l in 0..b {
            pages[l * p] = slot0[l];
            lens[l * p] = 1; // one prompt token in the first page
        }
        Self {
            b,
            p,
            page_t,
            pages,
            lens,
            np: vec![1u32; b],
            tfill: vec![1u32; b],
            tslot: slot0.to_vec(),
        }
    }

    /// Replay ONE epilogue step. `parent[l]` = the beam survivor `l` forked from
    /// (harvested from `out_par`); `fresh[l]` = the headroom slot granted to lane
    /// `l` this step. Mutates `self` to the post-step state and returns the fire
    /// geometry. Mirrors `s6_2_beam_epilogue_binds` op-for-op.
    pub fn step(&mut self, parent: &[u32], fresh: &[u32]) -> BeamGeometry {
        let (b, p, pt) = (self.b, self.p, self.page_t);
        assert_eq!(parent.len(), b, "parent is [B]");
        assert_eq!(fresh.len(), b, "fresh is [B]");

        // heir[par] = the LAST lane that forked from `par` — the designated
        // child that CONTINUES in `par`'s tail page (scatter_set(lanes,parent,
        // lanes) keeps the last write per parent). Others must fork a fresh page.
        let mut heir = vec![u32::MAX; b];
        for (l, &par) in parent.iter().enumerate() {
            heir[par as usize] = l as u32;
        }

        let mut new_pages = vec![0u32; b * p];
        let mut new_lens = vec![0u32; b * p];
        let mut np = vec![0u32; b];
        let mut klen = vec![0u32; b];
        let mut kvm = vec![0u8; b * p * pt as usize];
        let mut w_slot = vec![0u32; b];
        let mut w_off = vec![0u32; b];
        let mut w_cont = vec![false; b];

        for l in 0..b {
            let par = parent[l] as usize;
            // Inherit the parent's page geometry (gather by parent).
            let base = par * p;
            let pg = &self.pages[base..base + p];
            let pl = &self.lens[base..base + p];
            let n = self.np[par];
            let tf = self.tfill[par];

            // CONTINUE (heir + room) writes in place; else FORK a fresh page.
            let cont = heir[par] == l as u32 && tf < pt;
            let slot = if cont { self.tslot[par] } else { fresh[l] };
            let off = if cont { tf } else { 0 };
            let mut n2 = if cont { n } else { n + 1 };

            // Inherit the parent's page geometry (compaction may densify it).
            let dst = l * p;
            new_pages[dst..dst + p].copy_from_slice(pg);
            new_lens[dst..dst + p].copy_from_slice(pl);

            // §6.2 D4 compaction: repeated FORKs accumulate frozen-tail waste
            // (each fork opens a fresh 1-token page), growing np past P. When a
            // fork would exceed P, densify the inherited pages to
            // `ceil(prior_klen/page_t)` full-prefix pages — the host compact of
            // overview §6.2 (lines 837-860), reclaiming the waste so the fresh
            // page fits within P. NOTE: this densifies the LOGICAL geometry only;
            // the PHYSICAL KV pack (gather the live tokens into the dense slots)
            // is the driver's compact/gather_tokens kernel — coordinate before the
            // multi-step run. (Design B removes this replay re-impl entirely.)
            if !cont && n2 > p as u32 {
                let prior_klen: u32 = new_lens[dst..dst + n as usize].iter().sum();
                let dense_np = prior_klen.div_ceil(pt).clamp(1, (p - 1) as u32);
                for j in 0..p {
                    new_lens[dst + j] = if (j as u32) < dense_np {
                        (prior_klen - (j as u32) * pt).min(pt)
                    } else {
                        new_pages[dst + j] = 0; // reclaimed
                        0
                    };
                }
                n2 = dense_np + 1; // the fork still appends its one fresh page
            }

            // Write the tail column (n2-1) — the current token's page.
            let tcol = (n2 - 1) as usize;
            new_pages[dst + tcol] = slot;
            new_lens[dst + tcol] = off + 1;

            np[l] = n2;
            klen[l] = (n2 - 1) * pt + off + 1;

            // Per-cell mask: cell (l, pp, t) valid iff t < lens[l, pp].
            for pp in 0..p {
                let fill = new_lens[dst + pp];
                let mrow = (dst + pp) * pt as usize;
                for t in 0..pt {
                    kvm[mrow + t as usize] = u8::from(t < fill);
                }
            }

            w_slot[l] = slot;
            w_off[l] = off;
            w_cont[l] = cont;
        }

        // Advance the tracked state (tslot/tfill = this step's write target).
        self.pages = new_pages.clone();
        self.lens = new_lens.clone();
        self.np = np.clone();
        self.tslot = w_slot.clone();
        self.tfill = w_off.iter().map(|&o| o + 1).collect();

        BeamGeometry { pages: new_pages, np, klen, kvm, w_slot, w_off, w_cont }
    }

    /// The geometry of the CURRENT state WITHOUT stepping — used for the first
    /// fire (the seeded initial state, before any epilogue has run). `w_slot` /
    /// `w_off` point at the current tail write cursor (the next cell each beam
    /// writes). NOTE: the exact fire-0 / prompt-length seeding is refined during
    /// the 4090 bring-up (the pre-staged beam seeds a fresh single-token state).
    pub fn geometry(&self) -> BeamGeometry {
        let (b, p, pt) = (self.b, self.p, self.page_t);
        let mut kvm = vec![0u8; b * p * pt as usize];
        let mut klen = vec![0u32; b];
        for l in 0..b {
            for pp in 0..p {
                let fill = self.lens[l * p + pp];
                let mrow = (l * p + pp) * pt as usize;
                for t in 0..pt {
                    kvm[mrow + t as usize] = u8::from(t < fill);
                }
            }
            let tail = (self.np[l] - 1) as usize;
            klen[l] = (self.np[l] - 1) * pt + self.lens[l * p + tail];
        }
        BeamGeometry {
            pages: self.pages.clone(),
            np: self.np.clone(),
            klen,
            kvm,
            w_slot: self.tslot.clone(),
            w_off: self.tfill.iter().map(|&t| t.saturating_sub(1)).collect(),
            // Fire 0 writes each beam's first token into its (reserved) seed
            // slot → a fresh materialisation, never an in-place shared append.
            w_cont: vec![false; self.b],
        }
    }
}

impl BeamGeometry {
    /// Page-run CSR `kv_page_indptr` `[B+1]` = prefix-sum of `np` (dropping the
    /// per-beam padding tail by COUNT, never by a `pages==0` sentinel — slot 0 is
    /// a valid page).
    pub fn page_indptr(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.np.len() + 1);
        let mut acc = 0u32;
        out.push(0);
        for &n in &self.np {
            acc += n;
            out.push(acc);
        }
        out
    }

    /// `kv_last_page_lens` `[B]`: valid tokens in each beam's final live page.
    pub fn last_page_lens(&self, page_t: u32) -> Vec<u32> {
        self.klen
            .iter()
            .map(|&len| {
                if len == 0 || page_t == 0 {
                    0
                } else {
                    ((len - 1) % page_t) + 1
                }
            })
            .collect()
    }

    /// The flat slot ids for the live pages only (`[sum np]`), row-major per beam
    /// — the input to slot→physical resolution. Padding (`p >= np[b]`) dropped.
    pub fn live_page_slots(&self, p: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for (b, &n) in self.np.iter().enumerate() {
            for pp in 0..n as usize {
                out.push(self.pages[b * p + pp]);
            }
        }
        out
    }

    /// kvm → per-beam BRLE `masks` (1 query/beam) + `mask_indptr` `[B+1]`. Each
    /// beam's mask covers its live KV span (`np[b] * page_t` cells).
    pub fn masks(&self, p: usize, page_t: u32) -> (Vec<Brle>, Vec<u32>) {
        let mut masks = Vec::with_capacity(self.np.len());
        let mut indptr = Vec::with_capacity(self.np.len() + 1);
        indptr.push(0u32);
        for (b, &n) in self.np.iter().enumerate() {
            let span = (n * page_t) as usize;
            let base = b * p * page_t as usize;
            let bits: Vec<bool> = (0..span).map(|c| self.kvm[base + c] != 0).collect();
            masks.push(Brle::from_slice(&bits));
            indptr.push((b + 1) as u32);
        }
        (masks, indptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // B=2, P=3, PAGE_T=4 — the s6_2 golden shape.
    const B: usize = 2;
    const P: usize = 3;
    const PT: u32 = 4;

    /// Two beams, one prompt token each in slots 100/101.
    fn seed() -> BeamState {
        BeamState::seeded(B, P, PT, &[100, 101])
    }

    /// GOLDEN 1 — continue-tail: each beam continues from itself (heir, tail has
    /// room) → same slot, offset advances, no new page.
    #[test]
    fn golden_continue_tail() {
        let mut st = seed();
        let g = st.step(&[0, 1], &[200, 201]);
        assert_eq!(g.np, vec![1, 1], "no page turn");
        assert_eq!(g.w_slot, vec![100, 101], "wrote parent's tail slot");
        assert_eq!(g.w_off, vec![1, 1], "appended after the prompt token");
        assert_eq!(g.klen, vec![2, 2], "one more valid token");
        // pages unchanged (still the seeded tail slots); fresh grants unused.
        assert_eq!(&g.pages[0..1], &[100]);
        assert_eq!(&g.pages[P..P + 1], &[101]);
        // mask: 2 valid cells in beam 0's live page.
        let (masks, indptr) = g.masks(P, PT);
        assert_eq!(indptr, vec![0, 1, 2]);
        assert_eq!(masks[0].to_vec(), vec![true, true, false, false]);
    }

    /// GOLDEN 2 — page-turn: the tail page is FULL, so even the heir forks a
    /// fresh page (offset 0, np+1).
    #[test]
    fn golden_page_turn() {
        let mut st = seed();
        // Fill both tail pages to PAGE_T so the next step must turn the page.
        st.tfill = vec![PT, PT];
        st.lens[0] = PT; // beam 0 page 0 full
        st.lens[P] = PT; // beam 1 page 0 full
        let g = st.step(&[0, 1], &[200, 201]);
        assert_eq!(g.np, vec![2, 2], "page turned → 2 live pages");
        assert_eq!(g.w_slot, vec![200, 201], "wrote the FRESH grant");
        assert_eq!(g.w_off, vec![0, 0], "start of the fresh page");
        assert_eq!(g.klen, vec![PT + 1, PT + 1], "full page + 1 new token");
        assert_eq!(g.pages[1], 200, "beam 0 page 1 = fresh slot");
        assert_eq!(g.pages[P + 1], 201, "beam 1 page 1 = fresh slot");
        let indptr = g.page_indptr();
        assert_eq!(indptr, vec![0, 2, 4], "2 pages per beam");
    }

    /// GOLDEN 3 — fork-freeze: both survivors fork from beam 0. The heir (last
    /// lane) continues in-place; the non-heir forks a fresh page (the frozen
    /// sibling references beam 0's shared tail read-only, its own token on a
    /// fresh page — no physical copy).
    #[test]
    fn golden_fork_freeze() {
        let mut st = seed();
        let g = st.step(&[0, 0], &[200, 201]);
        // heir[0] = last lane with parent==0 = lane 1 → lane 1 continues.
        assert_eq!(g.w_slot[1], 100, "heir continues beam 0's tail slot");
        assert_eq!(g.w_off[1], 1, "heir appends in place");
        assert_eq!(g.np[1], 1, "heir: no new page");
        // lane 0 is a non-heir fork → fresh page.
        assert_eq!(g.w_slot[0], 200, "fork took the fresh slot");
        assert_eq!(g.w_off[0], 0, "fork writes a fresh page");
        assert_eq!(g.np[0], 2, "fork: new page (np+1)");
        // Both inherit beam 0's first (shared, frozen) page.
        assert_eq!(g.pages[0], 100, "fork keeps beam 0's shared page 0");
        assert_eq!(g.pages[P], 100, "heir keeps beam 0's page 0");
    }

    /// Two steps of continue-tail fill the page, then step 3 must turn it.
    #[test]
    fn page_fills_then_turns() {
        let mut st = seed(); // lens=1, tfill=1
        let _ = st.step(&[0, 1], &[200, 201]); // tfill→2
        let _ = st.step(&[0, 1], &[202, 203]); // tfill→3
        let g = st.step(&[0, 1], &[204, 205]); // tfill→4 (fills page)
        assert_eq!(g.klen, vec![4, 4]);
        assert_eq!(g.np, vec![1, 1], "page exactly full, not yet turned");
        // Now tfill==PAGE_T → next step turns the page.
        let g4 = st.step(&[0, 1], &[206, 207]);
        assert_eq!(g4.np, vec![2, 2], "page turned");
        assert_eq!(g4.w_slot, vec![206, 207]);
        assert_eq!(g4.w_off, vec![0, 0]);
    }

    /// §6.2 D4 compaction: every step both survivors fork from beam 0
    /// (`parent=[0,0]`) → lane 0 forks a fresh page each step → np would grow
    /// unboundedly (the `index out of bounds` panic charlie hit). Assert
    /// compaction keeps `np <= P` and `klen <= P*PAGE_T` across many steps.
    #[test]
    fn compaction_bounds_np_under_repeated_forks() {
        let mut st = seed();
        for step in 0..16u32 {
            let g = st.step(&[0, 0], &[200 + 2 * step, 201 + 2 * step]);
            assert!(
                g.np.iter().all(|&n| n <= P as u32),
                "step {step}: np {:?} exceeds P={P}",
                g.np
            );
            assert!(
                g.klen.iter().all(|&k| k <= P as u32 * PT),
                "step {step}: klen {:?} exceeds P*PAGE_T",
                g.klen
            );
        }
    }

    /// §6.2 golden cross-verify (charlie's `beam_csrs_test.cpp` fork-freeze):
    /// a single step where both survivors fork from beam 0 (`parent=[0,0]`) must
    /// produce charlie's exact driver-golden geometry —
    /// `np=[3,2], pages=[5,6,7, 5,6], klen=[9,7]` (P=3, page=4). Beam 0's lane
    /// FORKS a fresh page (slot 7); the heir lane (1) CONTINUES beam 0's tail
    /// (slot 6). This is the primary G2 correctness arm: the replay's [B,P]
    /// fork-freeze geometry == the geometry charlie's SEAM-1 gate consumes.
    #[test]
    fn golden_charlie_fork_freeze_csrs() {
        // Pre-step: beam 0 = {pages[5,6], lens[4,2], np=2, tail slot 6 filled 2}.
        let mut st = BeamState {
            b: 2,
            p: 3,
            page_t: 4,
            pages: vec![5, 6, 0, 5, 6, 0],
            lens: vec![4, 2, 0, 4, 2, 0],
            np: vec![2, 2],
            tfill: vec![2, 2],
            tslot: vec![6, 6],
        };
        let g = st.step(&[0, 0], &[7, 8]); // both fork from beam 0; fresh 7/8
        // charlie's golden: np=[3,2], pages=[5,6,7, 5,6,_], klen=[9,7].
        assert_eq!(g.np, vec![3, 2], "fork-freeze np");
        assert_eq!(&g.pages[0..3], &[5, 6, 7], "lane 0 forks a fresh page (slot 7)");
        assert_eq!(&g.pages[3..5], &[5, 6], "heir lane 1 continues beam 0's pages");
        assert_eq!(g.klen, vec![9, 7], "klen: (3-1)*4+1=9 ; (2-1)*4+3=7");
        // Write targets: lane 0 forks (fresh, off 0), lane 1 heir (shared tail 6, off 2).
        assert_eq!(g.w_slot, vec![7, 6]);
        assert_eq!(g.w_off, vec![0, 2]);
        assert_eq!(g.w_cont, vec![false, true], "lane 0 fork, lane 1 heir in-place");
    }
}
