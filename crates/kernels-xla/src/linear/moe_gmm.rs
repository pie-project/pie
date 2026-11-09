//! Grouped matmul (GMM) as a Mosaic kernel: activation rows sorted by expert
//! into row tiles, each tile multiplied by its expert's weights, which the
//! kernel's pipeline DMAs from HBM block by block. Only the experts a fire
//! routes to are read, with no per-expert dot launch: what XLA cannot do
//! (a dot over one expert's dynamic slice costs >= ~14 us on v6e; the whole
//! bank streams every expert).
//!
//! Operands (in order): `se` i32 `[tiles]`, the expert of each row tile
//! (entries past the tile count are ignored); `nt` i32 `[1]`, the tile
//! count; `xs` bf16 `[tiles·tm, K]`, the rows tile by tile; the bank. The
//! result is `[tiles·tm, out_width]` (f32, or bf16): row `r` of tile `t` against
//! expert `se[t]`, its `N` outputs starting at column `offsets()[se[t]]`.
//! Rows of tiles past the count are left unwritten.
//!
//! The bank is read in the layout the TPU keeps it in (`Orient`), so XLA
//! copies nothing in front of the call. Grid: `(n blocks, tiles)`, the tile
//! axis innermost, so a run of tiles on one expert reads its weights once;
//! steps past the tile count repeat the last block indices (no DMA) and
//! skip their dot. Weights convert to bf16 in VMEM (the VPU keeps up with
//! the DMA; measured: an e5m2 bank streams as fast with the dot as without).
//!
//! Tiling follows tpu-inference's megablox/`gmm_v2` (tokamax): a group-to-
//! tile schedule in scalar prefetch, the rhs block indexed by the tile's
//! group, the tile's rows masked by the caller.

use crate::hlo::{Built, Elem, Malformed};
use crate::mosaic::{Body, Compiled, Grid, ICmp, Kernel, MTy, Operand, Sem, Space, V, Window};

/// How the bank `[E·N, K]` sits in HBM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orient {
    /// `E·N` minor (the TPU's choice when K is not whole 128-lane tiles and
    /// `E·N` is): the kernel reads the free transpose `[K, E·N]`.
    KMajor,
    /// Row-major `[E·N, K]`.
    NMajor,
}

impl Orient {
    /// The layout the TPU gives a 2-d `[rows, k]` array by default (v6e,
    /// libtpu 0.0.48: it puts a 128-divisible axis minor when the declared
    /// minor axis is not).
    #[must_use]
    pub fn of(rows: i64, k: i64) -> Self {
        if k % 128 != 0 && rows % 128 == 0 {
            Self::KMajor
        } else {
            Self::NMajor
        }
    }
}

/// One grouped matmul's static shape.
#[derive(Clone, Copy, Debug)]
pub struct Gmm {
    /// Row tiles (the static bound; a fire's count is `nt`).
    pub tiles: i64,
    /// Rows per tile (a multiple of 16: bf16 rows pack in pairs).
    pub tm: i64,
    /// Weight bytes a block may hold (two are in flight, plus a bf16 copy).
    pub block_bytes: i64,
    pub vmem_cap: i64,
    pub k: i64,
    pub n: i64,
    pub experts: i64,
    /// The bank's element type (bf16, or f8 converted in VMEM).
    pub w: Elem,
    pub orient: Orient,
    /// The result's element type: f32, or bf16 (the f32 sums rounded).
    pub out: Elem,
}

/// How the weights are cut into blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// `N / tn` blocks of `tn` outputs, block-indexed.
    Blocked { tn: i64 },
    /// One element-offset window of `width` columns per expert, starting at
    /// the 128-lane tile holding the expert's first column.
    Element { width: i64 },
}

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "mosaic.gmm",
        detail: detail.into(),
    })
}

impl Gmm {
    fn wbytes(&self) -> i64 {
        i64::from(self.w.bits()) / 8
    }

    fn cut(&self) -> Built<Cut> {
        let (k, n, e) = (self.k, self.n, self.experts);
        let fits = |cols: i64| k * cols * self.wbytes() <= self.block_bytes;
        let blocked = (1..=n / 128)
            .rev()
            .map(|d| d * 128)
            .find(|&tn| n % tn == 0 && fits(tn));
        match self.orient {
            Orient::KMajor => {
                if let Some(tn) = blocked {
                    return Ok(Cut::Blocked { tn });
                }
                if n % 128 == 0 || (e * n) % 128 != 0 {
                    return bad(format!("no block of a [{k}, {e}x{n}] bank fits"));
                }
                let lag = (0..e).map(|i| i * n % 128).max().unwrap_or(0);
                let width = (n + lag + 127) / 128 * 128;
                if width > e * n || !fits(width) {
                    return bad(format!("a {width}-column window of a [{k}, {e}x{n}] bank"));
                }
                Ok(Cut::Element { width })
            }
            Orient::NMajor => {
                if let Some(tn) = blocked {
                    return Ok(Cut::Blocked { tn });
                }
                // Whole experts: the block's rows start at `e·N`, which the
                // f8/bf16 sublane tiling wants a multiple of 32.
                if n % 32 != 0 || !fits(n) {
                    return bad(format!("no block of a [{e}x{n}, {k}] bank fits"));
                }
                Ok(Cut::Blocked { tn: n })
            }
        }
    }

    /// Columns of a result row.
    pub fn out_width(&self) -> Built<i64> {
        Ok(match self.cut()? {
            Cut::Blocked { .. } => self.n,
            Cut::Element { width } => width,
        })
    }

    /// Where expert `e`'s outputs start in its rows of the result.
    pub fn offsets(&self) -> Built<Vec<i64>> {
        let cut = self.cut()?;
        Ok((0..self.experts)
            .map(|e| match cut {
                Cut::Blocked { .. } => 0,
                Cut::Element { width } => e * self.n - self.window_start(e, width),
            })
            .collect())
    }

    /// The first column of expert `e`'s window.
    fn window_start(&self, e: i64, width: i64) -> i64 {
        (e * self.n / 128 * 128).min(self.experts * self.n - width)
    }

    /// The bank operand's shape as the kernel takes it.
    #[must_use]
    pub fn bank_dims(&self) -> [i64; 2] {
        match self.orient {
            Orient::KMajor => [self.k, self.experts * self.n],
            Orient::NMajor => [self.experts * self.n, self.k],
        }
    }

    /// Whether the kernel fetches each expert's weights itself, the next
    /// expert's while the current one's tiles run (`kernel` then takes each
    /// tile's run and the next run's expert after `se` and `nt`).
    #[must_use]
    pub fn prefetching(&self) -> bool {
        std::env::var("PIE_XLA_GMM_PREFETCH").is_ok_and(|v| v == "1")
            && matches!(self.cut(), Ok(Cut::Blocked { tn }) if tn == self.n)
    }

    fn kernel_prefetching(&self) -> Built<Compiled> {
        let (k, n, tm) = (self.k, self.n, self.tm);
        let rows = self.tiles * tm;
        let orient = self.orient;
        let (rb, cb) = match orient {
            Orient::NMajor => (n, k),
            Orient::KMajor => (k, n),
        };
        fn last(b: &mut Body, g: &Grid) -> Built<V> {
            let zero = b.index(0);
            let nt = b.load(g.prefetch[1], &[zero])?;
            let one = b.i32(1);
            let top = b.subi(nt, one)?;
            let z = b.i32(0);
            let top = b.maxsi(top, z)?;
            b.minsi(g.ids[1], top)
        }
        let xs = Operand {
            array: vec![rows, k],
            elem: Elem::Bf16,
            block: vec![tm, k],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![last(b, g)?, b.i32(0)])),
        };
        let bank = Operand {
            array: self.bank_dims().to_vec(),
            elem: self.w,
            block: self.bank_dims().to_vec(),
            window: Window::Any,
            index: Box::new(|b: &mut Body, _: &Grid| Ok(vec![b.i32(0), b.i32(0)])),
        };
        let out_elem = self.out;
        let out = Operand {
            array: vec![rows, n],
            elem: out_elem,
            block: vec![tm, n],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![last(b, g)?, b.i32(0)])),
        };
        let wb = self.wbytes();
        let convert = if self.w == Elem::Bf16 { 0 } else { k * n * 2 };
        let vmem = 2 * tm * k * 2 + 2 * k * n * wb + convert + 2 * tm * n * 4;
        let floor = (32 << 20).min(self.vmem_cap);
        let limit = (vmem * 2 + (8 << 20)).clamp(floor, self.vmem_cap);
        let kernel = Kernel {
            name: "gmm".into(),
            grid: vec![1, self.tiles],
            semantics: vec![Sem::Arbitrary, Sem::Arbitrary],
            prefetch: vec![vec![self.tiles], vec![1], vec![self.tiles], vec![self.tiles]],
            ins: vec![xs, bank],
            outs: vec![out],
            vmem_limit: Some(limit as u64),
            scratch: vec![
                MTy::Ref(vec![2, rb, cb], self.w, Space::Vmem),
                MTy::Sem(vec![2]),
            ],
        };
        kernel.build(|b, blk| {
            let (wbuf, sems) = (blk.scratch[0], blk.scratch[1]);
            let bank = blk.ins[1];
            let pre = &blk.grid.prefetch;
            let t = blk.grid.ids[1];
            // The copy of expert `ex`'s weights into slot `s`.
            let copy = |b: &mut Body, ex: V, s: V| -> Built<(V, V, V)> {
                let z = b.i32(0);
                let nn = b.i32(n);
                let at = b.muli(ex, nn)?;
                let src = match orient {
                    Orient::NMajor => b.ref_slice(bank, &[at, z], &[n, k])?,
                    Orient::KMajor => b.ref_slice(bank, &[z, at], &[k, n])?,
                };
                let dst = b.ref_slice(wbuf, &[s, z, z], &[1, rb, cb])?;
                let dst = b.ref_squeeze(dst)?;
                let sem = b.ref_slice(sems, &[s], &[1])?;
                let sem = b.ref_squeeze(sem)?;
                Ok((src, dst, sem))
            };
            let zero = b.index(0);
            let nt = b.load(pre[1], &[zero])?;
            let live = b.cmpi(ICmp::Slt, t, nt)?;
            b.when(live, |b| {
                let ti = b.as_index(t)?;
                let r = b.load(pre[2], &[ti])?;
                let two = b.i32(2);
                let slot = b.remsi(r, two)?;
                let z = b.i32(0);
                let one = b.i32(1);
                let before = b.subi(t, one)?;
                let before = b.maxsi(before, z)?;
                let bi = b.as_index(before)?;
                let rp = b.load(pre[2], &[bi])?;
                let at0 = b.cmpi(ICmp::Eq, t, z)?;
                let moved = b.cmpi(ICmp::Ne, rp, r)?;
                let first = b.select(at0, at0, moved)?;
                b.when(at0, |b| {
                    let ex = b.load(pre[0], &[ti])?;
                    let (src, dst, sem) = copy(b, ex, slot)?;
                    b.dma_start(src, dst, sem)
                })?;
                b.when(first, |b| {
                    let z = b.i32(0);
                    let (src, dst, sem) = copy(b, z, slot)?;
                    b.dma_wait(sem, src, dst)?;
                    let nx = b.load(pre[3], &[ti])?;
                    let has = b.cmpi(ICmp::Sge, nx, z)?;
                    b.when(has, |b| {
                        let one = b.i32(1);
                        let other = b.subi(one, slot)?;
                        let (src, dst, sem) = copy(b, nx, other)?;
                        b.dma_start(src, dst, sem)
                    })
                })?;
                let z = b.i32(0);
                let held = b.ref_slice(wbuf, &[slot, z, z], &[1, rb, cb])?;
                let held = b.ref_squeeze(held)?;
                let x = b.vload_all(blk.ins[0])?;
                let w = b.vload_all(held)?;
                let w = b.convert_f(w, Elem::Bf16)?;
                let acc = b.splat_f(&[tm, n], Elem::F32, 0.0)?;
                let y = b.matmul(x, w, acc, orient == Orient::NMajor)?;
                let y = b.convert_f(y, out_elem)?;
                b.vstore_all(y, blk.outs[0])
            })
        })
    }

    pub fn kernel(&self) -> Built<Compiled> {
        if self.prefetching() {
            return self.kernel_prefetching();
        }
        if self.tm % 16 != 0 || self.tiles < 1 || self.k < 1 || self.n < 1 || self.experts < 1 {
            return bad(format!("{self:?}"));
        }
        if !matches!(self.w, Elem::Bf16 | Elem::F8E5m2 | Elem::F8E4m3fn) {
            return bad(format!("{:?} weights", self.w));
        }
        let cut = self.cut()?;
        let (k, n, tm, e) = (self.k, self.n, self.tm, self.experts);
        let rows = self.tiles * tm;
        let (cols, nblocks) = match cut {
            Cut::Blocked { tn } => (tn, n / tn),
            Cut::Element { width } => (width, 1),
        };
        let out_width = self.out_width()?;
        let orient = self.orient;

        // The last tile the fire has (tile 0 when it has none).
        fn last(b: &mut Body, g: &Grid) -> Built<V> {
            let zero = b.index(0);
            let nt = b.load(g.prefetch[1], &[zero])?;
            let one = b.i32(1);
            let top = b.subi(nt, one)?;
            let z = b.i32(0);
            let top = b.maxsi(top, z)?;
            b.minsi(g.ids[1], top)
        }
        fn expert(b: &mut Body, g: &Grid) -> Built<V> {
            let t = last(b, g)?;
            b.load(g.prefetch[0], &[t])
        }

        let xs = Operand {
            array: vec![rows, k],
            elem: Elem::Bf16,
            block: vec![tm, k],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![last(b, g)?, b.i32(0)])),
        };
        let bank = match (orient, cut) {
            (Orient::KMajor, Cut::Blocked { tn }) => Operand {
                array: self.bank_dims().to_vec(),
                elem: self.w,
                block: vec![k, tn],
                window: Window::Blocked,
                index: Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    let per = b.i32(n / tn);
                    let at = b.muli(ex, per)?;
                    let at = b.addi(at, g.ids[0])?;
                    Ok(vec![b.i32(0), at])
                }),
            },
            (Orient::KMajor, Cut::Element { width }) => Operand {
                array: self.bank_dims().to_vec(),
                elem: self.w,
                block: vec![k, width],
                window: Window::Element,
                index: Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    let nn = b.i32(n);
                    let col = b.muli(ex, nn)?;
                    let lane = b.i32(128);
                    let col = b.divsi(col, lane)?;
                    let col = b.muli(col, lane)?;
                    let top = b.i32(e * n - width);
                    let col = b.minsi(col, top)?;
                    // `E·N - width` is whole lane tiles too.
                    let col = b.assume_multiple(col, 128)?;
                    Ok(vec![b.i32(0), col])
                }),
            },
            (Orient::NMajor, Cut::Blocked { tn }) => Operand {
                array: self.bank_dims().to_vec(),
                elem: self.w,
                block: vec![tn, k],
                window: Window::Blocked,
                index: Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    let per = b.i32(n / tn);
                    let at = b.muli(ex, per)?;
                    let at = b.addi(at, g.ids[0])?;
                    Ok(vec![at, b.i32(0)])
                }),
            },
            (Orient::NMajor, Cut::Element { .. }) => return bad("an element window over rows"),
        };
        let out_elem = self.out;
        let out = Operand {
            array: vec![rows, out_width],
            elem: out_elem,
            block: vec![tm, cols],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![last(b, g)?, g.ids[0]])),
        };

        let wb = self.wbytes();
        let convert = if self.w == Elem::Bf16 {
            0
        } else {
            k * cols * 2
        };
        let vmem = 2 * tm * k * 2 + 2 * k * cols * wb + convert + 2 * tm * cols * 4;
        // At least the compiler's own scoped default, where the device has it.
        let floor = (32 << 20).min(self.vmem_cap);
        let limit = (vmem * 2 + (8 << 20)).clamp(floor, self.vmem_cap);
        let kernel = Kernel {
            name: "gmm".into(),
            grid: vec![nblocks, self.tiles],
            semantics: vec![Sem::Arbitrary, Sem::Arbitrary],
            prefetch: vec![vec![self.tiles], vec![1]],
            ins: vec![xs, bank],
            outs: vec![out],
            vmem_limit: Some(limit as u64),
            scratch: vec![],
        };
        kernel.build(|b, blk| {
            let zero = b.index(0);
            let nt = b.load(blk.grid.prefetch[1], &[zero])?;
            let live = b.cmpi(ICmp::Slt, blk.grid.ids[1], nt)?;
            b.when(live, |b| {
                let x = b.vload_all(blk.ins[0])?;
                let w = b.vload_all(blk.ins[1])?;
                let w = b.convert_f(w, Elem::Bf16)?;
                let acc = b.splat_f(&[tm, cols], Elem::F32, 0.0)?;
                let y = b.matmul(x, w, acc, orient == Orient::NMajor)?;
                let y = b.convert_f(y, out_elem)?;
                b.vstore_all(y, blk.outs[0])
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gmm(n: i64, orient: Orient) -> Gmm {
        Gmm {
            tiles: 48,
            tm: 16,
            block_bytes: crate::tuning::Tuning::DEFAULT.moe_gmm_block_bytes,
            vmem_cap: crate::tuning::Tuning::DEFAULT.vmem_cap(),
            k: 2880,
            n,
            experts: 32,
            w: Elem::F8E5m2,
            orient,
            out: Elem::F32,
        }
    }

    #[test]
    fn gpt_oss_banks_cut_into_aligned_blocks() {
        // gate/up: 45 lane tiles an expert, blocks of 15.
        let g = gmm(5760, Orient::KMajor);
        assert_eq!(g.cut().unwrap(), Cut::Blocked { tn: 1920 });
        assert_eq!(g.out_width().unwrap(), 5760);
        // down: 22.5 lane tiles an expert, one 23-tile window each.
        let d = gmm(2880, Orient::KMajor);
        assert_eq!(d.cut().unwrap(), Cut::Element { width: 2944 });
        let off = d.offsets().unwrap();
        assert_eq!(&off[..3], &[0, 64, 0]);
        assert_eq!(off[31], 64);
        assert!(d.kernel().unwrap().module.contains("element_window"));
    }
}
