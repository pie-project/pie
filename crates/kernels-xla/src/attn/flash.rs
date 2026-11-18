//! A prefill's causal (and sliding-window) attention over the keys the fire
//! itself appends, as one Mosaic flash kernel: the rows' own `k`, `v` (as
//! the fire hands them to `kv_append`) are the key rows, so a block of query
//! rows reads its keys as whole row blocks of that rectangle, never through
//! the page table, and the scores, the running max and sum and the
//! accumulator stay in VMEM (the paged walk writes each chunk's f32 scores
//! to HBM and reads them back).
//!
//! Row `r` reads key rows `[klo[r], khi[r])` of the rectangle: its segment
//! (its lane's consecutive positions in this fire) up to itself, cut by the
//! window. Keys a lane holds from before the fire (a cached prefix, of any
//! length) are read too, walked before the fresh blocks: each lane's pages
//! laid one after another as a virtual rectangle, its blocks copied page by
//! page out of the planes (pooled planes as the pool lays them, unpacked in
//! VMEM). A lane whose rows the fire did not append in its own slots (a
//! guest's stated write slots) reads all its keys that way, as the walk
//! reads them through its table: a choice made per lane, as data; so does a
//! lane with a row under a custom mask, whose mask columns gate its keys in
//! the kernel (copied by position, only in row blocks that hold such a row).
//! The kernel carries no run-time branch.
//!
//! Grid: one step per row block, its key blocks walked by a loop with the
//! next block's `k`/`v` copying in (two buffers) while the current one is
//! read. A turn of the block takes one kv head and `gf` of its query heads,
//! stacked along the rows (`[gf · bq, d]`), so one score tile serves them
//! all. The scores are kept transposed, `sᵀ = k · qᵀ` (`[bk, gf · bq]`, keys
//! on the sublanes), as splash attention keeps them: the softmax reduces
//! over the sublanes (plain vector maxima and sums, not cross-lane work), the
//! running max and sum are lane-dense rows, and the accumulator is `vᵀ · pᵀ`
//! (`[dv, gf · bq]`), turned once at the end. Probabilities meet values in
//! bf16, as the walk's do. The turns are unrolled: identical kernels compile
//! once a program (libtpu caches the module), well under a second.

#![allow(clippy::too_many_arguments)]

use crate::error::Error;
use crate::hlo::{Built, Cmp, Elem, Fold, Func, Malformed, Val};
use crate::mosaic::{Body, Compiled, Grid, ICmp, Kernel, MTy, Sem, Space, V, Window};
use crate::tensor::{Operand, PageMap};

use super::paged::{Flashed, Geom, Heads, cmp_i, read_rows_i32, round_up, with_i};

/// Keys per block.
const BK: i64 = 256;
/// Query rows per block.
const BQ: i64 = 256;
/// The most stacked rows one score tile holds.
const STACK: i64 = 1024;

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "attn.flash",
        detail: detail.into(),
    })
}

/// What the fire appended: `k`, `v` (`[r, kvh · d]` bf16) and each row's
/// write page and offset (`[r]` i32).
#[derive(Clone, Copy, Debug)]
pub struct Appended {
    pub k: Val,
    pub v: Val,
    pub page: Val,
    pub off: Val,
}

impl Appended {
    /// Whether `k`'s rows are as many as the kernel reads.
    #[must_use]
    pub fn worth(k: Operand) -> bool {
        i64::from(k.rows) >= BK
    }

    /// What `kv_append` read: its rows and their slots.
    pub fn of(
        f: &mut Func,
        k: Operand,
        v: Operand,
        page: Operand,
        off: Operand,
    ) -> Result<Self, Error> {
        let page = read_rows_i32(f, "attn.kv_append", page, k.rows)?;
        let off = read_rows_i32(f, "attn.kv_append", off, k.rows)?;
        Ok(Self {
            k: k.val,
            v: v.val,
            page,
            off,
        })
    }
}

/// One kernel's static shape.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Key rows (the rectangle's).
    r: i64,
    /// Query rows rounded up to whole row blocks (the grid's reach).
    rp: i64,
    bq: i64,
    bk: i64,
    h: Heads,
    /// Query heads stacked a turn, and turns per kv head.
    gf: i64,
    gc: i64,
    scale: f64,
    /// Where the older keys are read from.
    src: Src,
}

/// The planes a fire's older keys are read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    /// Planes of their own, `[slots, kvh · d]` bf16 each.
    Plain { slots: i64 },
    /// Both planes views of one pool (`u32 [rows, 128]`), their pages in
    /// the row-pair order (`crate::hlo::PoolOrder`), as `pages.rs` reads it.
    Pooled { rows: i64 },
}

impl Src {
    /// Where `keys` and `values` (pages of `ps`) are read from, if the
    /// kernel reads them: planes of their own, or both bf16 views of one
    /// pool in the row-pair order of `ps`-row pages (whole pairs of lane
    /// blocks a row).
    fn of(f: &Func, keys: Val, values: Val, ps: i64) -> Option<Self> {
        if f.elem(keys) != Elem::Bf16 || f.elem(values) != Elem::Bf16 {
            return None;
        }
        let pair = |v: Val| f.dims(v)[1] % 256 == 0;
        match (f.view_parts(keys), f.view_parts(values)) {
            (None, None) => Some(Self::Plain {
                slots: f.dims(keys)[0],
            }),
            (Some((pk, Elem::Bf16)), Some((pv, Elem::Bf16)))
                if pk == pv
                    && pair(keys)
                    && pair(values)
                    && f.view_order(keys).is_some_and(|o| o.tokens == ps)
                    && f.view_order(keys) == f.view_order(values) =>
            {
                Some(Self::Pooled {
                    rows: f.dims(pk)[0],
                })
            }
            _ => None,
        }
    }
}

/// A fire's older pages as the kernel reads them: `np` virtual pages of
/// `ps` rows, a list of the lanes' table pages (`-1` past the ones read)
/// every layer shares; each plane's page of a table page is its map's
/// (`page · stride + offset`), the sink for `-1`, found in the kernel.
#[derive(Clone, Copy, Debug)]
struct OldPages {
    np: i64,
    ps: i64,
    maps: [PageMap; 2],
    /// The plane page an unread virtual page copies (the planes' last).
    sink: i64,
    /// The custom mask's column plane, where the fire carries one: `i32
    /// [r, w]`, position `k` at column `k + BK`; prefetched each row's flag,
    /// its lane's shift (`Older::shift`) and each row block's flag (any
    /// flagged row).
    mask: Option<i64>,
}

impl Shape {
    fn of(r: i64, h: &Heads, scale: f64, bq: i64, src: Src) -> Self {
        let mut gf = h.g;
        while gf > 1 && (gf * bq > STACK || h.g % gf != 0) {
            gf -= 1;
        }
        Self {
            r,
            rp: round_up(r, bq),
            bq,
            bk: BK,
            h: *h,
            gf,
            gc: h.g / gf,
            scale,
            src,
        }
    }

    const fn turns(&self) -> i64 {
        self.h.kvh * self.gc
    }

    const fn n(&self) -> i64 {
        self.gf * self.bq
    }

    /// The VMEM the kernel holds: two buffers of each block, the scratch,
    /// and the score tile's temporaries.
    fn vmem(&self) -> i64 {
        let Shape { bq, bk, h, .. } = *self;
        let (qh, n, t) = (h.qh(), self.n(), self.turns());
        let blocks =
            bq * qh * h.d * 2 + bq * qh * h.dv * 2 + 2 * 8 * bq * 4 + 2 * round_up(qh, 8) * bq * 4;
        let bufs = 2 * bk * h.kvh * (h.d + h.dv) * 2;
        let scratch = bq * bk * 4 + t * (2 * 8 * n + h.dv * n) * 4;
        let temps = 6 * n * bk * 4 + 3 * n * h.dv * 4 + n * h.d * 2;
        // pooled: an older block's pages as they lie, beside the buffers
        let packed = match self.src {
            Src::Pooled { .. } => bufs,
            Src::Plain { .. } => 0,
        };
        2 * blocks + bufs + packed + scratch + temps
    }
}

/// The shape the kernel takes for `r` rows of these heads within
/// `vmem_cap`, if any: heads a whole number of lane tiles wide, the
/// rectangle at least a key block tall and tile-aligned at its end.
fn shape(
    r: i64,
    h: &Heads,
    scale: f64,
    vmem_cap: i64,
    narrow_rows: i64,
    src: Src,
    ps: i64,
) -> Option<Shape> {
    // an older block is whole pages, unpacked (pooled) 128 keys at a time
    if h.d % 128 != 0
        || h.dv % 128 != 0
        || r < BK
        || r % 16 != 0
        || h.g < 1
        || ps < 1
        || BK % ps != 0
        || 128 % ps != 0
    {
        return None;
    }
    // Narrow heads take the chip's rows first (`kernel.attn.flash_narrow_rows`).
    let first = if h.d <= 128 && h.dv <= 128 {
        narrow_rows
    } else {
        BQ
    };
    // Whole row blocks only: a partial last block's copies reach past the
    // fire's rows (gemma-4-31B's 640-row fires faulted the chip).
    [first, BQ / 2]
        .into_iter()
        .filter(|&bq| r % bq == 0)
        .map(|bq| Shape::of(r, h, scale, bq, src))
        .find(|s| s.vmem() <= vmem_cap)
}

/// Whether the kernel takes this fire: the static side of the decision.
pub(crate) fn fits(
    f: &Func,
    ap: &Appended,
    r: i64,
    h: &Heads,
    vmem_cap: i64,
    narrow_rows: i64,
    planes: (Val, Val),
    ps: i64,
) -> bool {
    let Some(src) = Src::of(f, planes.0, planes.1, ps) else {
        return false;
    };
    let kd = f.dims(ap.k);
    let vd = f.dims(ap.v);
    kd.len() == 2
        && kd[0] == r
        && kd[1] == h.kvh * h.d
        && vd == [r, h.kvh * h.dv]
        && f.elem(ap.k) == Elem::Bf16
        && f.elem(ap.v) == Elem::Bf16
        && f.dims(ap.page) == [r]
        && f.dims(ap.off) == [r]
        && shape(r, h, 1.0, vmem_cap, narrow_rows, src, ps).is_some()
}

/// The kernel over a fire's own key rows and its lanes' older keys, read
/// page by page out of the pool: the older keys lie as a virtual rectangle
/// of `old.np` pages (prefetched 4, and 5 for the values where their pages
/// differ: each virtual page's plane page), operands 5 and 6 each row's
/// range in it, prefetched 2 and 3 each row block's blocks of it. One walk
/// over the older blocks and then the fresh ones, one copy of the
/// arithmetic.
fn kernel(s: &Shape, vmem_cap: i64, old: &OldPages) -> Built<Compiled> {
    let Shape {
        r,
        rp,
        bq,
        bk,
        h,
        gf,
        gc,
        scale,
        src,
    } = *s;
    let (qh, kvh, g, d, dv) = (h.qh(), h.kvh, h.g, h.d, h.dv);
    let (wk, wv) = (kvh * d, kvh * dv);
    let ps = old.ps;
    let per_blk = bk / ps;
    // a pooled plane's page: `ps · w / 256` u32 rows (two bf16 rows of 128
    // lanes apiece, pair by pair)
    let (hk, hv) = (ps * wk / 256, ps * wv / 256);
    let nb = rp / bq;
    let n = s.n();
    let turns = s.turns();
    let rows = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![gr.ids[0], zero])
    };
    let cols = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![zero, gr.ids[0]])
    };
    let blocked = |array: Vec<i64>, elem: Elem, block: Vec<i64>, index: crate::mosaic::IndexMap| {
        crate::mosaic::Operand {
            array,
            elem,
            block,
            window: Window::Blocked,
            index,
        }
    };
    let any_of = |array: Vec<i64>, elem: Elem| crate::mosaic::Operand {
        array,
        elem,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let any = |array: Vec<i64>| any_of(array, Elem::Bf16);
    // Only each row block's ranges (and its mask flag) are prefetched: the
    // page list and the rows' mask flags and shifts are copied into SMEM in
    // the steps that read them, so a fire with no older keys and no flagged
    // row pays nothing for them.
    let mut prefetch = vec![vec![nb], vec![nb], vec![nb], vec![nb]];
    let mask_at = prefetch.len();
    if old.mask.is_some() {
        prefetch.push(vec![nb]);
    }
    let planes = match src {
        Src::Plain { slots } => vec![any(vec![slots, wk]), any(vec![slots, wv])],
        Src::Pooled { rows } => vec![any_of(vec![rows, 128], Elem::U32)],
    };
    let mut scratch = vec![
        MTy::Ref(vec![2, bk, wk], Elem::Bf16, Space::Vmem),
        MTy::Ref(vec![2, bk, wv], Elem::Bf16, Space::Vmem),
        MTy::Sem(vec![4]),
        MTy::Ref(vec![bk, bq], Elem::F32, Space::Vmem),
        MTy::Ref(vec![turns, 1, n], Elem::F32, Space::Vmem),
        MTy::Ref(vec![turns, 1, n], Elem::F32, Space::Vmem),
        MTy::Ref(vec![turns, dv, n], Elem::F32, Space::Vmem),
    ];
    if let Src::Pooled { .. } = src {
        // an older block's pages as they lie, unpacked into the key and
        // value buffers once copied
        scratch.push(MTy::Ref(vec![2, per_blk, hk, 128], Elem::U32, Space::Vmem));
        scratch.push(MTy::Ref(vec![2, per_blk, hv, 128], Elem::U32, Space::Vmem));
    }
    // the page list in SMEM, and its copy's semaphore
    let pscratch_at = scratch.len();
    scratch.push(MTy::Ref(vec![old.np], Elem::I32, Space::Smem));
    scratch.push(MTy::Sem(vec![1]));
    let mscratch_at = scratch.len();
    if old.mask.is_some() {
        // a flagged row block's rows' flags and shifts, in SMEM
        scratch.push(MTy::Ref(vec![bq], Elem::I32, Space::Smem));
        scratch.push(MTy::Ref(vec![bq], Elem::I32, Space::Smem));
        // a flagged row block's mask columns for an older block, a row's a
        // tile of its own (copied row by row), and their copies' semaphore
        scratch.push(MTy::Ref(vec![bq, 1, bk], Elem::I32, Space::Vmem));
        scratch.push(MTy::Sem(vec![1]));
    }
    let mut ins = vec![
        blocked(
            vec![r, qh * d],
            Elem::Bf16,
            vec![bq, qh * d],
            Box::new(rows),
        ),
        any(vec![r, wk]),
        any(vec![r, wv]),
        blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
        blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
        blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
        blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
    ];
    ins.extend(planes);
    let pages_at = ins.len();
    ins.push(any_of(vec![old.np], Elem::I32));
    let plane_at = ins.len();
    if let Some(w) = old.mask {
        ins.push(any_of(vec![r], Elem::I32));
        ins.push(any_of(vec![r], Elem::I32));
        ins.push(any_of(vec![r, w], Elem::I32));
    }
    let k = Kernel {
        name: "flash_prefill".into(),
        grid: vec![nb],
        semantics: vec![Sem::Arbitrary],
        prefetch,
        ins,
        outs: vec![
            blocked(
                vec![r, qh * dv],
                Elem::Bf16,
                vec![bq, qh * dv],
                Box::new(rows),
            ),
            blocked(vec![qh, r], Elem::F32, vec![qh, bq], Box::new(cols)),
            blocked(vec![qh, r], Elem::F32, vec![qh, bq], Box::new(cols)),
        ],
        vmem_limit: Some(u64::try_from((s.vmem() * 3 / 2 + (8 << 20)).min(vmem_cap)).unwrap_or(0)),
        scratch,
    };
    k.build(|b, blk| {
        let i = blk.grid.ids[0];
        let q_ref = blk.ins[0];
        let (fresh_k, fresh_v, klo_ref, khi_ref) = (blk.ins[1], blk.ins[2], blk.ins[3], blk.ins[4]);
        let (okl_ref, okh_ref) = (blk.ins[5], blk.ins[6]);
        // the planes the older pages are read from: their own, or the pool
        let (old_k, old_v) = match src {
            Src::Plain { .. } => (blk.ins[7], blk.ins[8]),
            Src::Pooled { .. } => (blk.ins[7], blk.ins[7]),
        };
        let (opk, psem, pages_in) = (
            blk.scratch[pscratch_at],
            blk.scratch[pscratch_at + 1],
            blk.ins[pages_at],
        );
        let packed = match src {
            Src::Pooled { .. } => Some((blk.scratch[7], blk.scratch[8])),
            Src::Plain { .. } => None,
        };
        let masked = old.mask.map(|_| {
            let pre = &blk.grid.prefetch;
            (
                (blk.ins[plane_at], blk.ins[plane_at + 1]),
                (blk.scratch[mscratch_at], blk.scratch[mscratch_at + 1]),
                pre[mask_at],
                blk.ins[plane_at + 2],
                blk.scratch[mscratch_at + 2],
                blk.scratch[mscratch_at + 3],
            )
        });
        let (o_ref, mo_ref, lo_ref) = (blk.outs[0], blk.outs[1], blk.outs[2]);
        let (kbuf, vbuf, sems, bias_ref, m_s, l_s, acc_s) = (
            blk.scratch[0],
            blk.scratch[1],
            blk.scratch[2],
            blk.scratch[3],
            blk.scratch[4],
            blk.scratch[5],
            blk.scratch[6],
        );
        let pre = &blk.grid.prefetch;
        let lo = b.load(pre[0], &[i])?;
        let hi = b.load(pre[1], &[i])?;
        let olo = b.load(pre[2], &[i])?;
        let ohi = b.load(pre[3], &[i])?;
        let i0 = b.index(0);
        let zero = b.i32(0);
        let one = b.i32(1);

        // Every turn's state starts empty.
        for t in 0..turns {
            let at = b.index(t);
            let ninf = b.splat_f(&[1, 1, n], Elem::F32, f64::NEG_INFINITY)?;
            b.vstore(ninf, m_s, &[at, i0, i0])?;
            let z = b.splat_f(&[1, 1, n], Elem::F32, 0.0)?;
            b.vstore(z, l_s, &[at, i0, i0])?;
            let z = b.splat_f(&[1, dv, n], Elem::F32, 0.0)?;
            b.vstore(z, acc_s, &[at, i0, i0])?;
        }

        // Step `t` reads older block `olo + t` while `t` is below their
        // count, then fresh block `lo + t - count`.
        let count = |b: &mut Body, lo: V, hi: V| -> Built<V> {
            let c = b.subi(hi, lo)?;
            let c = b.addi(c, one)?;
            b.maxsi(c, zero)
        };
        let nold = count(b, olo, ohi)?;
        // The page list, where the block reads older keys at all.
        let has_old = b.cmpi(ICmp::Sgt, nold, zero)?;
        b.when(has_old, |b| {
            let z = b.i32(0);
            let sem = b.ref_slice(psem, &[z], &[1])?;
            let sem = b.ref_squeeze(sem)?;
            b.dma_start(pages_in, opk, sem)?;
            b.dma_wait(sem, pages_in, opk)
        })?;
        let nfresh = count(b, lo, hi)?;
        let steps = b.addi(nold, nfresh)?;
        let older = |b: &mut Body, t: V| -> Built<V> { b.cmpi(ICmp::Slt, t, nold) };
        let block_of = |b: &mut Body, t: V| -> Built<V> {
            let o = b.addi(olo, t)?;
            let f0 = b.subi(t, nold)?;
            let f0 = b.addi(lo, f0)?;
            let is_old = b.cmpi(ICmp::Slt, t, nold)?;
            b.select(is_old, o, f0)
        };
        // A block's first row in its rectangle: a fresh block clamped so it
        // stays inside the fire's rows (the mask skips the rows a clamped
        // block repeats); the older rectangle is whole blocks.
        let block_row = |b: &mut Body, t: V| -> Built<V> {
            let j = block_of(b, t)?;
            let bkv = b.i32(bk);
            let at = b.muli(j, bkv)?;
            let top = b.i32(r - bk);
            let clamped = b.minsi(at, top)?;
            let is_old = b.cmpi(ICmp::Slt, t, nold)?;
            b.select(is_old, at, clamped)
        };
        let dma = |b: &mut Body, s: V, d: V, sidx: V, wait: bool| -> Built<()> {
            let sem = b.ref_slice(sems, &[sidx], &[1])?;
            let sem = b.ref_squeeze(sem)?;
            if wait {
                b.dma_wait(sem, s, d)
            } else {
                b.dma_start(s, d, sem)
            }
        };
        // Starts (`wait` false) or waits for the copies of step `t` into
        // buffer `slot`: an older block page by page (each virtual page's
        // plane page, pooled: as the pool lays it), a fresh block whole.
        let copies = |b: &mut Body, t: V, slot: V, wait: bool| -> Built<()> {
            let is_old = older(b, t)?;
            let is_fresh = b.cmpi(ICmp::Sge, t, nold)?;
            b.when(is_old, |b| {
                let j = block_of(b, t)?;
                let pbv = b.i32(per_blk);
                let v0 = b.muli(j, pbv)?;
                let z = b.i32(0);
                let two = b.i32(2);
                let vslot = b.addi(slot, two)?;
                for p in 0..per_blk {
                    let pv = b.i32(p);
                    let vp = b.addi(v0, pv)?;
                    let raw = b.load(opk, &[vp])?;
                    // each plane's page of the table page, by its map
                    let unread = b.cmpi(ICmp::Slt, raw, z)?;
                    let sink = b.i32(old.sink);
                    let mut planes_pages = [raw; 2];
                    for (at, map) in old.maps.iter().enumerate() {
                        let by = b.i32(map.stride);
                        let off = b.i32(map.offset);
                        let page = b.muli(raw, by)?;
                        let page = b.addi(page, off)?;
                        planes_pages[at] = b.select(unread, sink, page)?;
                    }
                    let [pk, pvv] = planes_pages;
                    for (page, plane, sidx, which) in
                        [(pk, old_k, slot, 0usize), (pvv, old_v, vslot, 1)]
                    {
                        let (s, d) = match packed {
                            None => {
                                let (buf, w) = [(kbuf, wk), (vbuf, wv)][which];
                                let psv = b.i32(ps);
                                let row = b.muli(page, psv)?;
                                let s = b.ref_slice(plane, &[row, z], &[ps, w])?;
                                let dr = b.i32(p * ps);
                                let d = b.ref_slice(buf, &[slot, dr, z], &[1, ps, w])?;
                                (s, b.ref_squeeze(d)?)
                            }
                            Some((pk_buf, pv_buf)) => {
                                let (buf, half) = [(pk_buf, hk), (pv_buf, hv)][which];
                                let hv_ = b.i32(half);
                                let row = b.muli(page, hv_)?;
                                let s = b.ref_slice(plane, &[row, z], &[half, 128])?;
                                let d = b.ref_slice(buf, &[slot, pv, z, z], &[1, 1, half, 128])?;
                                let d = b.ref_squeeze(d)?;
                                (s, b.ref_squeeze(d)?)
                            }
                        };
                        dma(b, s, d, sidx, wait)?;
                    }
                }
                Ok(())
            })?;
            b.when(is_fresh, |b| {
                let kb = block_row(b, t)?;
                let z = b.i32(0);
                let two = b.i32(2);
                let vslot = b.addi(slot, two)?;
                for (src, buf, w, sidx) in [(fresh_k, kbuf, wk, slot), (fresh_v, vbuf, wv, vslot)] {
                    let s = b.ref_slice(src, &[kb, z], &[bk, w])?;
                    let d = b.ref_slice(buf, &[slot, z, z], &[1, bk, w])?;
                    let d = b.ref_squeeze(d)?;
                    dma(b, s, d, sidx, wait)?;
                }
                Ok(())
            })
        };
        // A pooled older block's pages into the key and value buffers: a
        // u32 word's low half the even 128-lane block (as a float's high
        // bits), its high half the odd.
        const UNPACK: i64 = 128;
        let unpack = |b: &mut Body, slot: V| -> Built<()> {
            let Some((pk_buf, pv_buf)) = packed else {
                return Ok(());
            };
            let sidx = b.as_index(slot)?;
            let i0 = b.index(0);
            let sixteen = b.splat_i(&[UNPACK, 128], 16)?;
            let high = b.splat_i(&[UNPACK, 128], i64::from(0xFFFF_0000u32 as i32))?;
            for (from, to, w) in [(pk_buf, kbuf, wk), (pv_buf, vbuf, wv)] {
                for c in 0..bk / UNPACK {
                    let pg = b.index(c * UNPACK / ps);
                    for jp in 0..w / 256 {
                        let jr = b.index(jp * ps);
                        let x = b.vload(from, &[sidx, pg, jr, i0], &[1, UNPACK / ps, ps, 128])?;
                        let x = b.shape_cast(x, &[UNPACK, 128])?;
                        let x = b.bitcast_v(x, Elem::I32)?;
                        let lo = b.shli(x, sixteen)?;
                        let hi = b.andi(x, high)?;
                        for (hb, half) in [(0, lo), (1, hi)] {
                            let fl = b.bitcast_v(half, Elem::F32)?;
                            let bf = b.convert_f(fl, Elem::Bf16)?;
                            let bf = b.shape_cast(bf, &[1, UNPACK, 128])?;
                            let row = b.index(c * UNPACK);
                            let col = b.index((2 * jp + hb) * 128);
                            b.vstore(bf, to, &[sidx, row, col])?;
                        }
                    }
                }
            }
            Ok(())
        };
        let any_steps = b.cmpi(ICmp::Sgt, steps, zero)?;
        b.when(any_steps, |b| copies(b, zero, zero, false))?;
        b.for_range(zero, steps, one, |b, t| {
            let two = b.i32(2);
            let slot = b.remsi(t, two)?;
            let other = b.subi(one, slot)?;
            let next = b.addi(t, one)?;
            let more = b.cmpi(ICmp::Slt, next, steps)?;
            b.when(more, |b| copies(b, next, other, false))?;
            copies(b, t, slot, true)?;
            if packed.is_some() {
                let is_old = older(b, t)?;
                b.when(is_old, |b| unpack(b, slot))?;
            }

            // The block's admission, a bias of 0 or -inf per (key, row): a
            // row admits the key rows of its range in the block's
            // rectangle, from the block's own start.
            let kb = block_row(b, t)?;
            let j = block_of(b, t)?;
            let is_old = older(b, t)?;
            let sq = [bk, bq];
            let kr = b.broadcast_i(kb, &sq)?;
            let io = b.iota(&sq, 0);
            let kr = b.addi(kr, io)?;
            let bkv = b.i32(bk);
            let start = b.muli(j, bkv)?;
            let start = b.broadcast_i(start, &sq)?;
            let flo = b.vload_all(klo_ref)?;
            let olo_v = b.vload_all(okl_ref)?;
            let klo = b.select(is_old, olo_v, flo)?;
            let klo = b.broadcast(klo, &sq)?;
            let klo = b.maxsi(klo, start)?;
            let fhi = b.vload_all(khi_ref)?;
            let ohi_v = b.vload_all(okh_ref)?;
            let khi = b.select(is_old, ohi_v, fhi)?;
            let khi = b.broadcast(khi, &sq)?;
            let ge = b.cmpi(ICmp::Sge, kr, klo)?;
            let lt = b.cmpi(ICmp::Slt, kr, khi)?;
            let ok = b.andi(ge, lt)?;
            let zs = b.splat_f(&sq, Elem::F32, 0.0)?;
            let ninf = b.splat_f(&sq, Elem::F32, f64::NEG_INFINITY)?;
            let bias = b.select(ok, zs, ninf)?;
            b.vstore_all(bias, bias_ref)?;
            // A flagged row block's older block: each flagged row's mask
            // columns (its lane's positions are the block's rows less its
            // shift, a whole 128) gate its keys, as the walk's mask does.
            if let Some(((en_in, shift_in), (en_ref, shift_ref), flag_ref, plane, mt, msem)) =
                masked
            {
                let flag = b.load(flag_ref, &[i])?;
                let z = b.i32(0);
                let any = b.cmpi(ICmp::Ne, flag, z)?;
                let go = b.andi(any, is_old)?;
                b.when(go, |b| {
                    let ones = b.splat_i(&[bq, 1, bk], 1)?;
                    let i0 = b.index(0);
                    b.vstore(ones, mt, &[i0, i0, i0])?;
                    let z = b.i32(0);
                    let bqv = b.i32(bq);
                    let row0 = b.muli(i, bqv)?;
                    let bkv = b.i32(bk);
                    let v0 = b.muli(j, bkv)?;
                    let one = b.i32(1);
                    let sem = b.ref_slice(msem, &[z], &[1])?;
                    let sem = b.ref_squeeze(sem)?;
                    // the block's rows' flags and shifts, into SMEM
                    for (from, to) in [(en_in, en_ref), (shift_in, shift_ref)] {
                        let src = b.ref_slice(from, &[row0], &[bq])?;
                        b.dma_start(src, to, sem)?;
                        b.dma_wait(sem, src, to)?;
                    }
                    for wait in [false, true] {
                        b.for_range(z, bqv, one, |b, x| {
                            let row = b.addi(row0, x)?;
                            let en = b.load(en_ref, &[x])?;
                            let z = b.i32(0);
                            let on = b.cmpi(ICmp::Ne, en, z)?;
                            b.when(on, |b| {
                                let sh = b.load(shift_ref, &[x])?;
                                let at = b.subi(v0, sh)?;
                                let edge = b.i32(BK);
                                let at = b.addi(at, edge)?;
                                let at = b.assume_multiple(at, 128)?;
                                let z = b.i32(0);
                                let src = b.ref_slice(plane, &[row, at], &[1, bk])?;
                                let dst = b.ref_slice(mt, &[x, z, z], &[1, 1, bk])?;
                                let dst = b.ref_squeeze(dst)?;
                                if wait {
                                    b.dma_wait(sem, src, dst)
                                } else {
                                    b.dma_start(src, dst, sem)
                                }
                            })
                        })?;
                    }
                    let cols = b.vload(mt, &[i0, i0, i0], &[bq, 1, bk])?;
                    let cols = b.shape_cast(cols, &[bq, bk])?;
                    let cols = b.transpose(cols)?;
                    let zc = b.splat_i(&[bk, bq], 0)?;
                    let keep = b.cmpi(ICmp::Ne, cols, zc)?;
                    let bias = b.vload_all(bias_ref)?;
                    let ninf = b.splat_f(&[bk, bq], Elem::F32, f64::NEG_INFINITY)?;
                    let bias = b.select(keep, bias, ninf)?;
                    b.vstore_all(bias, bias_ref)
                })?;
            }

            let z = b.i32(0);
            let kslab = b.ref_slice(kbuf, &[slot, z, z], &[1, bk, wk])?;
            let kslab = b.ref_squeeze(kslab)?;
            let vslab = b.ref_slice(vbuf, &[slot, z, z], &[1, bk, wv])?;
            let vslab = b.ref_squeeze(vslab)?;
            for t2 in 0..turns {
                let (kh, gi) = (t2 / gc, t2 % gc);
                let col = b.index(kh * d);
                let kv = b.vload(kslab, &[i0, col], &[bk, d])?;
                let col = b.index(kh * dv);
                let vv = b.vload(vslab, &[i0, col], &[bk, dv])?;
                // The turn's `gf` query heads, stacked along the rows.
                let mut parts = Vec::with_capacity(gf as usize);
                for x in 0..gf {
                    let qi = kh * g + gi * gf + x;
                    let col = b.index(qi * d);
                    parts.push(b.vload(q_ref, &[i0, col], &[bq, d])?);
                }
                let qf = if gf == 1 {
                    parts[0]
                } else {
                    b.concat(&parts, 0)?
                };
                let zs = b.splat_f(&[bk, n], Elem::F32, 0.0)?;
                let st = b.matmul(kv, qf, zs, true)?;
                let st = if scale == 1.0 {
                    st
                } else {
                    let sc = b.splat_f(&[bk, n], Elem::F32, scale)?;
                    b.mulf(st, sc)?
                };
                let bias = b.vload_all(bias_ref)?;
                let bias = if gf == 1 {
                    bias
                } else {
                    let copies = vec![bias; gf as usize];
                    b.concat(&copies, 1)?
                };
                let st = b.addf(st, bias)?;
                let at = b.index(t2);
                let m0 = b.vload(m_s, &[at, i0, i0], &[1, 1, n])?;
                let m0 = b.shape_cast(m0, &[1, n])?;
                let l0 = b.vload(l_s, &[at, i0, i0], &[1, 1, n])?;
                let l0 = b.shape_cast(l0, &[1, n])?;
                let mc = b.reduce_cols(st, true)?;
                let mn = b.maxf(m0, mc)?;
                let ninf1 = b.splat_f(&[1, n], Elem::F32, f64::NEG_INFINITY)?;
                let alive = b.cmpf(true, mn, ninf1)?;
                let z1 = b.splat_f(&[1, n], Elem::F32, 0.0)?;
                let ms = b.select(alive, mn, z1)?;
                let msb = b.broadcast(ms, &[bk, n])?;
                let e = b.subf(st, msb)?;
                let p = b.exp(e)?;
                let ps = b.reduce_cols(p, false)?;
                let dm = b.subf(m0, ms)?;
                let al = b.exp(dm)?;
                let l = b.mulf(l0, al)?;
                let l = b.addf(l, ps)?;
                let mn3 = b.shape_cast(mn, &[1, 1, n])?;
                b.vstore(mn3, m_s, &[at, i0, i0])?;
                let l3 = b.shape_cast(l, &[1, 1, n])?;
                b.vstore(l3, l_s, &[at, i0, i0])?;
                let a0 = b.vload(acc_s, &[at, i0, i0], &[1, dv, n])?;
                let a0 = b.shape_cast(a0, &[dv, n])?;
                let ab = b.broadcast(al, &[dv, n])?;
                let a = b.mulf(a0, ab)?;
                let pb = b.convert_f(p, Elem::Bf16)?;
                let zo = b.splat_f(&[dv, n], Elem::F32, 0.0)?;
                let pv = b.matmul_tl(vv, pb, zo)?;
                let a = b.addf(a, pv)?;
                let a3 = b.shape_cast(a, &[1, dv, n])?;
                b.vstore(a3, acc_s, &[at, i0, i0])?;
            }
            Ok(())
        })?;

        // Each head's rows normalized and turned into their columns; the
        // max and sum a row per head.
        let mut mrows: Vec<(i64, V)> = Vec::with_capacity(qh as usize);
        let mut lrows: Vec<(i64, V)> = Vec::with_capacity(qh as usize);
        for t2 in 0..turns {
            let at = b.index(t2);
            let l = b.vload(l_s, &[at, i0, i0], &[1, 1, n])?;
            let l = b.shape_cast(l, &[1, n])?;
            let m = b.vload(m_s, &[at, i0, i0], &[1, 1, n])?;
            let m = b.shape_cast(m, &[1, n])?;
            let z1 = b.splat_f(&[1, n], Elem::F32, 0.0)?;
            let one1 = b.splat_f(&[1, n], Elem::F32, 1.0)?;
            let pos = b.cmpf(true, l, z1)?;
            let inv = b.divf(one1, l)?;
            let inv = b.select(pos, inv, z1)?;
            let a = b.vload(acc_s, &[at, i0, i0], &[1, dv, n])?;
            let a = b.shape_cast(a, &[dv, n])?;
            let ib = b.broadcast(inv, &[dv, n])?;
            let ot = b.mulf(a, ib)?;
            let o = b.transpose(ot)?;
            let o = b.convert_f(o, Elem::Bf16)?;
            let (kh, gi) = (t2 / gc, t2 % gc);
            for x in 0..gf {
                let qi = kh * g + gi * gf + x;
                let og = b.rows(o, x * bq, bq)?;
                let col = b.index(qi * dv);
                b.vstore(og, o_ref, &[i0, col])?;
                mrows.push((qi, b.slice(m, &[0, x * bq], &[1, bq])?));
                lrows.push((qi, b.slice(l, &[0, x * bq], &[1, bq])?));
            }
        }
        mrows.sort_by_key(|x| x.0);
        lrows.sort_by_key(|x| x.0);
        let ms: Vec<V> = mrows.iter().map(|x| x.1).collect();
        let ls: Vec<V> = lrows.iter().map(|x| x.1).collect();
        let mo = b.concat(&ms, 0)?;
        b.vstore_all(mo, mo_ref)?;
        let lsum = b.concat(&ls, 0)?;
        b.vstore_all(lsum, lo_ref)
    })
}

/// Each row's segment: the first row of the run of rows that are one lane's
/// consecutive positions, as `(start [r], p0 [r])`, the run's first row and
/// that row's position. A padding row (position below zero) stands alone.
fn segments(f: &mut Func, lane: Val, qpos: Val) -> Built<(Val, Val)> {
    let r = f.dims(lane)[0];
    let io = f.iota(Elem::I32, &[r], 0);
    let prev = |f: &mut Func, x: Val, fill: i64| -> Built<Val> {
        let head = f.slice_axis(x, 0, 0, r - 1)?;
        let c = f.const_i(Elem::I32, fill, &[]);
        f.pad(head, c, &[1], &[0], &[0])
    };
    let pl = prev(f, lane, -1)?;
    let pp = prev(f, qpos, -2)?;
    let pp = with_i(f, pp, 1, Func::add)?;
    let same = f.compare(Cmp::Eq, lane, pl)?;
    let next = f.compare(Cmp::Eq, qpos, pp)?;
    let same = f.and(same, next)?;
    let live = cmp_i(f, Cmp::Ge, qpos, 0)?;
    let same = f.and(same, live)?;
    let z = f.like_i(io, 0);
    let cand = f.select(same, z, io)?;
    let start = f.scan_steps(cand, 0, Fold::Max)?;
    let back = f.sub(io, start)?;
    let p0 = f.sub(qpos, back)?;
    Ok((start, p0))
}

/// The attention of `q4` (`[r, kvh, g, d]` bf16) rows over the fire's
/// appended keys and their lanes' older keys, read page by page out of the
/// planes: `lo`/`hi` bound each row's admitted positions, `lane` its lane
/// (clamped), `cap` the lane's capacity, `slots` the pool plane's rows.
/// `mask`: each row's custom-mask flag, the mask plane (`[r, pitch]` u8,
/// a row's byte at key position `k`) and its bound. A lane whose rows did
/// not all land in its own slots, or with a flagged row (`through_pages`),
/// reads every key through its pages, as the walk would; a flagged row's
/// keys are gated by its mask in the kernel. `o` is bf16.
pub(crate) fn attend(
    f: &mut Func,
    ap: &Appended,
    h: &Heads,
    geom: &Geom,
    slots: i64,
    planes: (Val, Val),
    q4: Val,
    lane: Val,
    qpos: Val,
    lo: Val,
    hi: Val,
    cap: Val,
    mask: Option<(Val, Val, i64)>,
    scale: f64,
    vmem_cap: i64,
    narrow_rows: i64,
    maps: [PageMap; 2],
) -> Built<Flashed> {
    let r = f.dims(q4)[0];
    let ps = geom.ps;
    let Some(src) = Src::of(f, planes.0, planes.1, ps) else {
        return bad("the planes are not read page by page");
    };
    let Some(s) = shape(r, h, scale, vmem_cap, narrow_rows, src, ps) else {
        return bad(format!("{r} rows of {h:?} do not fit"));
    };
    let (start, p0) = segments(f, lane, qpos)?;
    let live = cmp_i(f, Cmp::Ge, qpos, 0)?;
    // each live row's flag (none on a padded row)
    let flags = match mask {
        Some((en, _, _)) => {
            let z = f.like_i(en, 0);
            Some(f.select(live, en, z)?)
        }
        None => None,
    };
    let paged = through_pages(f, geom, slots, ap, lane, qpos, cap, live, flags, maps[0])?;
    // A paged lane's rows read `[lo, hi)` all as older keys.
    let reach = f.select(paged, hi, p0)?;
    let old = older(f, geom, slots, lane, lo, reach, live, maps)?;
    let fresh = f.not(paged);
    let fresh = f.and(fresh, live)?;
    let (klo, khi) = fresh_rows(f, lo, hi, p0, start, fresh)?;
    let cols = match (mask, flags) {
        (Some((_, plane, bound)), Some(en)) => {
            // position `k` at column `k + BK`: a block's columns from a
            // block before a lane's first key to a block past its span
            let w = BK + geom.span + BK;
            let x = super::pages::mask_columns(f, plane, bound, r, geom.span + BK)?;
            let zero = f.const_i(Elem::I32, 0, &[]);
            let x = f.pad(x, zero, &[0, BK], &[0, 0], &[0, 0])?;
            debug_assert_eq!(f.dims(x)[1], w);
            Some((en, x))
        }
        _ => None,
    };
    let (o, m, l) = run(f, &s, ap, q4, klo, khi, vmem_cap, &old, planes, cols)?;
    Ok(Flashed { o, m, l })
}

/// Each row's key rows `[klo, khi)` of the fire's own rectangle: its
/// segment's, from its window's start (the older keys are the virtual
/// rectangle's); none where `fresh` is false.
fn fresh_rows(
    f: &mut Func,
    lo: Val,
    hi: Val,
    p0: Val,
    start: Val,
    fresh: Val,
) -> Built<(Val, Val)> {
    let klo = f.max(lo, p0)?;
    let klo = f.sub(klo, p0)?;
    let klo = f.add(klo, start)?;
    let khi = f.sub(hi, p0)?;
    let khi = f.add(khi, start)?;
    let none = f.like_i(klo, 0);
    Ok((f.select(fresh, klo, none)?, f.select(fresh, khi, none)?))
}

/// Whether each row's lane reads all its keys through its pages: a lane
/// with a live row the fire did not append at its table slot (a guest's
/// stated write slots may differ from `have + i`), nor within its capacity,
/// or under a custom mask (`flags`: its columns are read by position).
/// The walk reads such a lane's keys through its table after the append;
/// so does the kernel, through the fire's page list. Every other lane reads
/// its fresh keys from the rectangle the fire appended.
fn through_pages(
    f: &mut Func,
    geom: &Geom,
    slots: i64,
    ap: &Appended,
    lane: Val,
    qpos: Val,
    cap: Val,
    live: Val,
    flags: Option<Val>,
    keys_map: PageMap,
) -> Built<Val> {
    let ps = geom.ps;
    let r = f.dims(lane)[0];
    let pix = with_i(f, qpos, ps, Func::div)?;
    let at = with_i(f, lane, geom.pages, Func::mul)?;
    let at = f.add(at, pix)?;
    let page = super::paged::take(f, geom.pt, at)?;
    let want = with_i(f, page, ps, Func::mul)?;
    let off = with_i(f, qpos, ps, Func::rem)?;
    let want = f.add(want, off)?;
    let got = with_i(f, ap.page, ps, Func::mul)?;
    let got = f.add(got, ap.off)?;
    let filed = f.compare(Cmp::Eq, got, want)?;
    let inside = cmp_i(f, Cmp::Ge, ap.page, 0)?;
    // the sink is the keys plane's last page, in its own pages
    let plane_page = keys_map.apply(f, ap.page)?;
    let below = cmp_i(f, Cmp::Lt, plane_page, slots / ps - 1)?;
    let filed = f.and(filed, inside)?;
    let filed = f.and(filed, below)?;
    let upto = with_i(f, qpos, 1, Func::add)?;
    let held = f.compare(Cmp::Le, upto, cap)?;
    let mut ok = f.and(filed, held)?;
    if let Some(en) = flags {
        let off = cmp_i(f, Cmp::Eq, en, 0)?;
        ok = f.and(ok, off)?;
    }
    let stray = f.not(ok);
    let stray = f.and(stray, live)?;
    // any stray row of a lane pages the whole lane
    let grid = [geom.lanes, r];
    let li = f.iota(Elem::I32, &grid, 0);
    let lb = f.broadcast(lane, &grid, &[1])?;
    let mine = f.compare(Cmp::Eq, lb, li)?;
    let sb = f.broadcast(stray, &grid, &[1])?;
    let hit = f.and(mine, sb)?;
    let hit = f.convert(hit, Elem::I32);
    let any = f.reduce(hit, &[1], Fold::Max)?;
    let paged = super::paged::take(f, any, lane)?;
    cmp_i(f, Cmp::Ne, paged, 0)
}

/// A fire's readout rows `q4` (`[R, kvh, g, d]` bf16, at fire rows `at`)
/// over their lanes' keys: the rows the fire appended (`ap`, one a fire row,
/// `lane` and `qpos` the fire's tables), a blocked walk over that rectangle,
/// and the older keys their lanes hold, each row copying its own lane's
/// pages (`pages::attend`; heads or planes the copies do not take walk the
/// rows' older pages, `paged::per_row`), merged by their log-sum-exps. `o`
/// is `[R, kvh, g, dv]` f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn readouts(
    f: &mut Func,
    ap: &Appended,
    h: &Heads,
    geom: &Geom,
    slots: i64,
    planes: (Val, Val),
    q4: Val,
    at: Val,
    lane: Val,
    qpos: Val,
    window: i64,
    scale: f64,
    ch: crate::tuning::Chunks,
    vmem_cap: i64,
    maps: [PageMap; 2],
    mask: Option<(Val, Val, i64)>,
) -> Built<Val> {
    let n = f.dims(lane)[0];
    let (start, p0) = segments(f, lane, qpos)?;
    let live = cmp_i(f, Cmp::Ge, qpos, 0)?;
    let lo = if window > 0 {
        let s = with_i(f, qpos, window - 1, Func::sub)?;
        let z = f.like_i(s, 0);
        f.max(s, z)?
    } else {
        f.like_i(qpos, 0)
    };
    let cap = geom.capacity(f, lane)?;
    let hi = with_i(f, qpos, 1, Func::add)?;
    let hi = f.min(hi, cap)?;
    // A lane whose rows did not all land in its slots reads every key
    // through its pages, as `attend`'s do.
    let paged = through_pages(f, geom, slots, ap, lane, qpos, cap, live, None, maps[0])?;
    let fresh = f.not(paged);
    let fresh = f.and(fresh, live)?;
    let (klo, khi) = fresh_rows(f, lo, hi, p0, start, fresh)?;
    // Its older keys: positions `[lo, p0)` of its lane (`[lo, hi)` paged;
    // none for a padded row).
    let dead = f.like_i(klo, 0);
    let reach = f.select(paged, hi, p0)?;
    let ohi = f.select(live, reach, dead)?;
    let olo = f.select(live, lo, dead)?;
    let pick = |f: &mut Func, v: Val| super::paged::take(f, v, at);
    let (klo, khi) = (pick(f, klo)?, pick(f, khi)?);
    let (olo, ohi, ol) = (pick(f, olo)?, pick(f, ohi)?, pick(f, lane)?);
    // A readout row under a custom mask (`mask`'s flag, a readout row's)
    // reads every key through its pages, its mask gating them: a causal
    // one to itself, a wide one (flag 2) to its lane's capacity.
    let (klo, khi, olo, ohi, mask) = match mask {
        Some((en, plane, bound)) => {
            let (live_r, cap_r) = (pick(f, live)?, pick(f, cap)?);
            let (lo_r, hi_r) = (pick(f, lo)?, pick(f, hi)?);
            let on = cmp_i(f, Cmp::Ne, en, 0)?;
            let on = f.and(on, live_r)?;
            let wide = cmp_i(f, Cmp::Eq, en, 2)?;
            let reach = f.select(wide, cap_r, hi_r)?;
            let none = f.like_i(klo, 0);
            (
                f.select(on, none, klo)?,
                f.select(on, none, khi)?,
                f.select(on, lo_r, olo)?,
                f.select(on, reach, ohi)?,
                Some((en, plane, bound)),
            )
        }
        None => (klo, khi, olo, ohi, None),
    };
    use super::paged::{KeyList, Planes, Rows, Rule, blocked, per_row};
    let a = if super::pages::fits(geom.span, h, vmem_cap)
        && Src::of(f, planes.0, planes.1, geom.ps).is_some()
    {
        super::pages::attend(
            f, q4, planes.0, planes.1, geom.pt, ol, olo, ohi, geom.pages, geom.ps, h, scale,
            vmem_cap, maps, mask,
        )?
    } else {
        let paged = Planes::Paged {
            keys: planes.0,
            values: planes.1,
            geom: *geom,
            latent: false,
            keys_map: maps[0],
            values_map: maps[1],
        };
        let mut rows = Rows::new(q4, olo, ohi);
        rows.lane = Some(ol);
        let mut rule = Rule::default();
        if let Some((en, plane, bound)) = mask {
            rows.enabled = Some(en);
            rows.mask = Some(plane);
            rule.bound = bound;
        }
        per_row(f, &paged, h, &rule, &rows, KeyList::Range, scale, ch)?
    };
    let b = blocked(
        f,
        &Planes::Plain { k: ap.k, v: ap.v },
        h,
        &Rule::default(),
        &Rows::new(q4, klo, khi),
        n,
        scale,
        ch,
    )?;
    // The two parts' softmaxes as one: each weighs its sum at the larger
    // max (a part that admitted nothing weighs none).
    let m = f.max(a.m, b.m)?;
    let weigh = |f: &mut Func, part: &Flashed| -> Built<Val> {
        let d = f.sub(part.m, m)?;
        let e = f.exp(d);
        let w = f.mul(e, part.l)?;
        let z = f.like_f(part.l, 0.0);
        let any = f.compare(Cmp::Gt, part.l, z)?;
        f.select(any, w, z)
    };
    let (wa, wb) = (weigh(f, &a)?, weigh(f, &b)?);
    let den = f.add(wa, wb)?;
    let z = f.like_f(den, 0.0);
    let pos = f.compare(Cmp::Gt, den, z)?;
    let one = f.like_f(den, 1.0);
    let inv = f.div(one, den)?;
    let inv = f.select(pos, inv, z)?;
    let od = f.dims(b.o).to_vec();
    let fa = f.mul(wa, inv)?;
    let fb = f.mul(wb, inv)?;
    let fa = f.broadcast(fa, &od, &[0, 1, 2])?;
    let fb = f.broadcast(fb, &od, &[0, 1, 2])?;
    let oa = f.convert(a.o, Elem::F32);
    let ob = f.convert(b.o, Elem::F32);
    let oa = f.mul(oa, fa)?;
    let ob = f.mul(ob, fb)?;
    f.add(oa, ob)
}

/// The older keys a fire's rows read (their lanes' keys from before the
/// fire): each lane's pages from the first any of its rows reads to the
/// last, laid lane after lane (in the order the lanes' first rows come) as
/// a virtual rectangle of `np` pages, every lane's whole table room, so a
/// prefix of any length is read; and each row's range `[klo, khi)` in it.
/// A lane's virtual rows lie a whole 128 rows from its positions (`shift`),
/// so a block's mask columns come in one aligned copy.
#[derive(Clone, Copy)]
struct Older {
    /// Each virtual page's table page, `[np]` (`-1` past the ones read):
    /// what every layer of the fire reads, each plane's page of it its
    /// map's, in the kernel.
    pages: Val,
    klo: Val,
    khi: Val,
    /// Each row's lane's virtual row less its position (`[r]`, a multiple
    /// of 128).
    shift: Val,
    np: i64,
    ps: i64,
    /// Where each plane's pages lie, and the planes' sink page.
    maps: [PageMap; 2],
    sink: i64,
}

/// Each row's older keys: positions `[lo, p0)` of its lane (`p0` its
/// segment's first position), read through the lane's page table.
fn older(
    f: &mut Func,
    geom: &Geom,
    slots: i64,
    lane: Val,
    lo: Val,
    p0: Val,
    live: Val,
    maps: [PageMap; 2],
) -> Built<Older> {
    let r = f.dims(lane)[0];
    let (lanes, pages, ps) = (geom.lanes, geom.pages, geom.ps);
    let reads = f.compare(Cmp::Lt, lo, p0)?;
    let reads = f.and(reads, live)?;
    let fp = with_i(f, lo, ps, Func::div)?;
    let ep = with_i(f, p0, ps - 1, Func::add)?;
    let ep = with_i(f, ep, ps, Func::div)?;
    // Each lane's first and end page over the rows that read.
    let grid = [lanes, r];
    let li = f.iota(Elem::I32, &grid, 0);
    let lb = f.broadcast(lane, &grid, &[1])?;
    let mine = f.compare(Cmp::Eq, lb, li)?;
    let rb = f.broadcast(reads, &grid, &[1])?;
    let mr = f.and(mine, rb)?;
    let per_lane = |f: &mut Func, x: Val, sel: Val, fill: i64, fold: Fold| -> Built<Val> {
        let xb = if f.dims(x).len() == 1 {
            f.broadcast(x, &grid, &[1])?
        } else {
            x
        };
        let fl = f.like_i(xb, fill);
        let y = f.select(sel, xb, fl)?;
        f.reduce(y, &[1], fold)
    };
    let first = per_lane(f, fp, mr, pages, Fold::Min)?;
    let end = per_lane(f, ep, mr, 0, Fold::Max)?;
    let np_l = f.sub(end, first)?;
    let np_l = super::paged::clamp_i(f, np_l, 0, pages)?;
    // The lanes in the order their first live rows come.
    let rowi = f.iota(Elem::I32, &grid, 1);
    let lv = f.broadcast(live, &grid, &[1])?;
    let ml = f.and(mine, lv)?;
    let fr = per_lane(f, rowi, ml, r, Fold::Min)?;
    let key = with_i(f, fr, lanes, Func::mul)?;
    let lio = f.iota(Elem::I32, &[lanes], 0);
    let key = f.add(key, lio)?;
    let ll = [lanes, lanes];
    let kj = f.broadcast(key, &ll, &[1])?;
    let ki = f.broadcast(key, &ll, &[0])?;
    let before = f.compare(Cmp::Lt, kj, ki)?;
    // a lane's room: its pages, whole 128-row blocks, and a block more, so
    // its first page lies at its position's offset in a block
    let a = 128 / ps;
    let room = with_i(f, np_l, 2 * a - 1, Func::add)?;
    let room = with_i(f, room, a, Func::div)?;
    let room = with_i(f, room, a, Func::mul)?;
    let nb = f.broadcast(room, &ll, &[1])?;
    let z = f.like_i(nb, 0);
    let ahead = f.select(before, nb, z)?;
    let base = f.reduce(ahead, &[1], Fold::Sum)?;
    let lead = with_i(f, first, a, Func::rem)?;
    let base = f.add(base, lead)?;
    // Each row's range in the rectangle.
    let base_r = super::paged::take(f, base, lane)?;
    let first_r = super::paged::take(f, first, lane)?;
    let shift = f.sub(base_r, first_r)?;
    let shift = with_i(f, shift, ps, Func::mul)?;
    let none = f.like_i(lo, 0);
    let klo = f.add(shift, lo)?;
    let klo = f.select(reads, klo, none)?;
    let khi = f.add(shift, p0)?;
    let khi = f.select(reads, khi, none)?;
    // Each virtual page's table page: its owner is the lane whose room
    // starts last at or before it (rooms lie in rank order, one after
    // another, each at least a block), its page that lane's `v - base`-th
    // from its first, `-1` past the lane's pages. A search over the lanes
    // and a gather, not a scatter (XLA's scatter runs its updates one by
    // one on a TPU).
    let np = round_up(
        lanes * (round_up(pages, 128 / ps) + 2 * (128 / ps)),
        BK / ps,
    );
    let rstart = f.sub(base, lead)?;
    let vl = [np, lanes];
    let v = f.iota(Elem::I32, &vl, 0);
    let rs = f.broadcast(rstart, &vl, &[1])?;
    let at_or_before = f.compare(Cmp::Le, rs, v)?;
    let li = f.iota(Elem::I32, &vl, 1);
    let rank_key = with_i(f, rs, lanes, Func::mul)?;
    let rank_key = f.add(rank_key, li)?;
    let none_key = f.like_i(rank_key, -1);
    let cand = f.select(at_or_before, rank_key, none_key)?;
    let best = f.reduce(cand, &[1], Fold::Max)?;
    let owner = with_i(f, best, lanes, Func::rem)?;
    let owner = super::paged::clamp_i(f, owner, 0, lanes - 1)?;
    let v1 = f.iota(Elem::I32, &[np], 0);
    let owner_base = super::paged::take(f, base, owner)?;
    let j = f.sub(v1, owner_base)?;
    let cnt = super::paged::take(f, np_l, owner)?;
    let ge = cmp_i(f, Cmp::Ge, j, 0)?;
    let lt = f.compare(Cmp::Lt, j, cnt)?;
    let valid = f.and(ge, lt)?;
    let owner_first = super::paged::take(f, first, owner)?;
    let pix = f.add(owner_first, j)?;
    let pix = super::paged::clamp_i(f, pix, 0, pages - 1)?;
    let row = with_i(f, owner, pages, Func::mul)?;
    let at = f.add(row, pix)?;
    let raw = super::paged::take(f, geom.pt, at)?;
    let unread = f.like_i(raw, -1);
    let pages = f.select(valid, raw, unread)?;
    Ok(Older {
        pages,
        klo,
        khi,
        shift,
        np,
        ps,
        maps,
        sink: slots / ps - 1,
    })
}

/// The kernel over `q4`, each row reading key rows `[klo, khi)`: `(o [r,
/// kvh, g, dv] bf16, m [r, kvh, g], l [r, kvh, g])`.
fn run(
    f: &mut Func,
    s: &Shape,
    ap: &Appended,
    q4: Val,
    klo: Val,
    khi: Val,
    vmem_cap: i64,
    old: &Older,
    planes: (Val, Val),
    mask: Option<(Val, Val)>,
) -> Built<(Val, Val, Val)> {
    let Shape {
        r, rp, bq, bk, h, ..
    } = *s;
    let nb = rp / bq;
    let q2 = f.reshape(q4, &[r, h.qh() * h.d])?;
    // A power-of-two scale (1/16 at d = 256) turns `q` exactly, before the
    // kernel: its score tiles then take no multiply.
    let unit = Shape { scale: 1.0, ..*s };
    let (q2, s) = if s.scale != 1.0 && s.scale.log2().fract() == 0.0 {
        (f.scale(q2, s.scale)?, &unit)
    } else {
        (q2, s)
    };
    let pad_rows = |f: &mut Func, x: Val, fill: i64| -> Built<Val> {
        if rp == r {
            return Ok(x);
        }
        let c = if f.elem(x).is_float() {
            f.const_f(f.elem(x), fill as f64, &[])
        } else {
            f.const_i(f.elem(x), fill, &[])
        };
        let rank = f.dims(x).len();
        let mut hi = vec![0; rank];
        hi[0] = rp - r;
        f.pad(x, c, &vec![0; rank], &hi, &vec![0; rank])
    };
    // Each row block's first and last key block (none: last below first).
    let blocks = |f: &mut Func, klo: Val, khi: Val| -> Built<(Val, Val)> {
        let empty = f.compare(Cmp::Ge, klo, khi)?;
        let far = f.like_i(klo, i64::from(i32::MAX));
        let lo_rows = f.select(empty, far, klo)?;
        let none = f.like_i(khi, 0);
        let hi_rows = f.select(empty, none, khi)?;
        let lo_rows = pad_rows(f, lo_rows, i64::from(i32::MAX))?;
        let hi_rows = pad_rows(f, hi_rows, 0)?;
        let lo_b = f.reshape(lo_rows, &[nb, bq])?;
        let lo_b = f.reduce(lo_b, &[1], Fold::Min)?;
        let hi_b = f.reshape(hi_rows, &[nb, bq])?;
        let hi_b = f.reduce(hi_b, &[1], Fold::Max)?;
        let any = cmp_i(f, Cmp::Gt, hi_b, 0)?;
        let lo_blk = with_i(f, lo_b, bk, Func::div)?;
        let zero = f.like_i(lo_blk, 0);
        let lo_blk = f.select(any, lo_blk, zero)?;
        let hi_blk = with_i(f, hi_b, 1, Func::sub)?;
        let hi_blk = with_i(f, hi_blk, bk, Func::div)?;
        let minus = f.like_i(hi_blk, -1);
        Ok((lo_blk, f.select(any, hi_blk, minus)?))
    };
    let (lo_blk, hi_blk) = blocks(f, klo, khi)?;
    // The rows are whole row blocks (`shape`), so none is padded.
    let klo = f.reshape(klo, &[1, r])?;
    let khi = f.reshape(khi, &[1, r])?;
    let (olo, ohi) = blocks(f, old.klo, old.khi)?;
    let okl = f.reshape(old.klo, &[1, r])?;
    let okh = f.reshape(old.khi, &[1, r])?;
    let pages = OldPages {
        np: old.np,
        ps: old.ps,
        maps: old.maps,
        sink: old.sink,
        mask: mask.map(|(_, cols)| f.dims(cols)[1]),
    };
    let kern = kernel(s, vmem_cap, &pages)?;
    let mut operands = vec![lo_blk, hi_blk, olo, ohi];
    if let Some((en, _)) = mask {
        // each row block's flag: any flagged row
        let on = cmp_i(f, Cmp::Ne, en, 0)?;
        let on = f.convert(on, Elem::I32);
        let on = pad_rows(f, on, 0)?;
        let on = f.reshape(on, &[nb, bq])?;
        let blk = f.reduce(on, &[1], Fold::Max)?;
        operands.push(blk);
    }
    operands.extend([q2, ap.k, ap.v, klo, khi, okl, okh]);
    match s.src {
        // both planes out of the pool itself
        Src::Pooled { .. } => match f.view_parts(planes.0) {
            Some((pool, _)) => operands.push(pool),
            None => return bad("pooled planes are not views"),
        },
        Src::Plain { .. } => operands.extend([planes.0, planes.1]),
    }
    operands.push(old.pages);
    if let Some((en, cols)) = mask {
        operands.extend([en, old.shift, cols]);
    }
    let out = kern.call(f, &operands)?;
    let o = f.reshape(out[0], &[r, h.kvh, h.g, h.dv])?;
    // The max and sum come a row per head.
    let heads = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.transpose(x, &[1, 0])?;
        f.reshape(x, &[r, h.kvh, h.g])
    };
    let m = heads(f, out[1])?;
    let l = heads(f, out[2])?;
    Ok((o, m, l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_stacks_a_kv_heads_query_heads() {
        let h = Heads {
            kvh: 8,
            g: 2,
            d: 128,
            dv: 128,
        };
        let plain = Src::Plain { slots: 4096 };
        let s = shape(1792, &h, 0.1, 110 << 20, BQ, plain, 16).unwrap();
        assert_eq!((s.rp, s.bq, s.bk, s.gf, s.gc), (1792, 256, 256, 2, 1));
        let old = OldPages {
            np: 256,
            ps: 16,
            maps: [PageMap::IDENTITY; 2],
            sink: 255,
            mask: None,
        };
        let c = kernel(&s, 110 << 20, &old).unwrap();
        assert!(c.module.contains("iteration_bounds = array<i64: 7>"));
        assert_eq!(
            c.module.matches("tpu.matmul").count(),
            2 * 8,
            "one copy of each turn"
        );
        assert!(c.module.contains("scratch_operands = 9 : i64"));
        assert!(
            c.module.contains("vector<256x512xf32>"),
            "scores keyed on the sublanes"
        );
        assert_eq!(c.operands.len(), 14);
        // an older block's 16 pages, each plane's copied by itself
        assert!(c.module.matches("tpu.enqueue_dma").count() >= 2 * 16 + 2);
        assert_eq!(c.results.len(), 3);
        if let Some(p) = std::env::var_os("FLASH_DUMP") {
            std::fs::write(p, &c.module).unwrap();
        }
    }

    #[test]
    fn wide_groups_split_into_turns() {
        let h = Heads {
            kvh: 1,
            g: 8,
            d: 256,
            dv: 256,
        };
        let plain = Src::Plain { slots: 4096 };
        let s = shape(2048, &h, 0.1, 110 << 20, BQ, plain, 16).unwrap();
        assert_eq!((s.bq, s.gf, s.gc), (256, 4, 2));
        assert!(shape(200, &h, 0.1, 110 << 20, BQ, plain, 16).is_none());
        assert!(shape(1000, &h, 0.1, 110 << 20, BQ, plain, 16).is_none());
    }

    /// Pooled planes: an older block's pages copied as the pool lays them
    /// (u32 rows, a page's `ps · w / 256` of them) and unpacked into the
    /// key and value buffers; the values' own page list where their map
    /// differs.
    #[test]
    fn pooled_older_pages_are_copied_as_they_lie() {
        let h = Heads {
            kvh: 8,
            g: 2,
            d: 128,
            dv: 128,
        };
        let pooled = Src::Pooled { rows: 1 << 20 };
        let s = shape(1792, &h, 0.1, 110 << 20, BQ, pooled, 16).unwrap();
        let old = OldPages {
            np: 64 * 256,
            ps: 16,
            maps: [
                PageMap::IDENTITY,
                PageMap {
                    stride: 2,
                    offset: 1,
                },
            ],
            sink: 255,
            mask: None,
        };
        let c = kernel(&s, 110 << 20, &old).unwrap();
        assert!(c.module.contains("scratch_operands = 11 : i64"));
        assert!(
            c.module.contains("memref<2x16x64x128xi32")
                || c.module.contains("memref<2x16x64x128xui32")
        );
        // four prefetched, seven blocks, the pool and the page list (copied
        // into SMEM where a block reads older keys)
        assert_eq!(c.operands.len(), 4 + 9);
    }
}
