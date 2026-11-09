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
//!
//! A plain bank under one whole-`N` block (`Gmm::prefetching`) is instead
//! copied by the kernel itself into two VMEM slots: a run's first tile
//! waits for its expert's copy and starts the next run's, so an expert's
//! weights stream in while the run before it computes (the pipeline
//! fetches a block one step ahead, one tile at most). Two more i32
//! `[tiles]` operands follow `nt`: each tile's run and the next run's
//! expert (-1 after the last).

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
    /// The bank's element type (bf16, or f8 converted in VMEM; e2m1 codes
    /// with `scaled`).
    pub w: Elem,
    pub orient: Orient,
    /// An e8m0 exponent byte per 32 codes along `K` follows the bank, as
    /// `[K / 32, E·N]` bytes (`E·N`-minor bank only): the codes convert to
    /// bf16 in VMEM and take their block's power of two, exactly.
    pub scaled: bool,
    /// An f32 factor per 128 codes of K follows the bank, as `[K / 128,
    /// E·N]` (`E·N`-minor e4m3 bank only, the factors of its 128x128 blocks
    /// laid a column each): each 128 codes of K contract on their own and
    /// their partial takes its block's factors (`tile_dot`).
    pub tile: bool,
    /// A bf16 bias row per expert follows (`[E, 1, N]`, a blocked cut
    /// only), added to the f32 product before it rounds.
    pub bias: bool,
    /// The result's element: f32, or the product (and its bias) rounded to
    /// bf16 as the select's result is.
    pub out: Elem,
    /// Whether the kernel copies each expert's weights itself, the next
    /// run's while a run's tiles go (a plain bank under a whole-`N` block
    /// only: see [`Gmm::prefetching`]).
    pub prefetch: bool,
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

/// Rows from which a block-FP8 tile widens its codes (`Gmm::widens`).
pub const WIDEN_ROWS: i64 = 512;

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

    /// A block's bytes in HBM.
    fn block_dma(&self, cols: i64) -> i64 {
        self.k * cols * i64::from(self.w.bits()) / 8
    }

    /// A block-FP8 prompt in one tile widens its block of codes under their
    /// factors to bf16 once and contracts it a whole K deep: the 128-deep
    /// dots leave half of a v6e MXU idle, and the widening's VPU work is
    /// shared by the tile's rows (widened again for each of several tiles,
    /// it costs more than it saves).
    fn widens(&self) -> bool {
        self.tile && self.tiles == 1 && self.tm >= WIDEN_ROWS
    }

    fn cut(&self) -> Built<Cut> {
        let (k, n, e) = (self.k, self.n, self.experts);
        // An e2m1 block counts a byte a code: its bf16 copy in VMEM is what
        // bounds it; a block-FP8 one its bf16 slices, or its f32 widening.
        let per = if self.widens() {
            4
        } else if self.tile {
            2
        } else {
            self.wbytes().max(1)
        };
        let fits = |cols: i64| k * cols * per <= self.block_bytes;
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

    /// Where expert `e`'s outputs start in its window: the kernel turns each
    /// row's lanes by it, so every row's outputs start at column 0.
    fn lags(&self, width: i64) -> Vec<i64> {
        (0..self.experts)
            .map(|e| e * self.n - self.window_start(e, width))
            .collect()
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
        self.prefetch
            && !self.scaled
            && !self.tile
            && !self.bias
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
            prefetch: vec![
                vec![self.tiles],
                vec![1],
                vec![self.tiles],
                vec![self.tiles],
            ],
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
        if self.tm % 16 != 0 || self.tiles < 1 || self.k < 1 || self.n < 1 || self.experts < 1 {
            return bad(format!("{self:?}"));
        }
        if self.prefetching() {
            return self.kernel_prefetching();
        }
        if !matches!(
            self.w,
            Elem::Bf16 | Elem::F8E5m2 | Elem::F8E4m3fn | Elem::F4E2m1fn
        ) || (self.w == Elem::F4E2m1fn) != self.scaled
            || (self.scaled && (self.orient != Orient::KMajor || self.k % 32 != 0))
            || (self.tile
                && (self.w != Elem::F8E4m3fn || self.orient != Orient::KMajor || self.k % 128 != 0))
        {
            return bad(format!("{:?} weights (scaled {})", self.w, self.scaled));
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
        // e2m1 codes enter as their bits (`u4`): Mosaic here has no f4 vector.
        let welem = if self.scaled { Elem::U4 } else { self.w };
        let bank = match (orient, cut) {
            (Orient::KMajor, Cut::Blocked { tn }) => Operand {
                array: self.bank_dims().to_vec(),
                elem: welem,
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
                elem: welem,
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
        // The exponent bytes of the block's codes: the bank's columns, a row
        // per 32 codes of K.
        let groups = if self.tile { k / 128 } else { k / 32 };
        let scales = (self.scaled || self.tile).then(|| {
            let window = bank.window;
            let index: crate::mosaic::IndexMap = match cut {
                Cut::Blocked { tn } => Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    let per = b.i32(n / tn);
                    let at = b.muli(ex, per)?;
                    let at = b.addi(at, g.ids[0])?;
                    Ok(vec![b.i32(0), at])
                }),
                Cut::Element { width } => Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    let nn = b.i32(n);
                    let col = b.muli(ex, nn)?;
                    let lane = b.i32(128);
                    let col = b.divsi(col, lane)?;
                    let col = b.muli(col, lane)?;
                    let top = b.i32(e * n - width);
                    let col = b.minsi(col, top)?;
                    let col = b.assume_multiple(col, 128)?;
                    Ok(vec![b.i32(0), col])
                }),
            };
            Operand {
                array: vec![groups, e * n],
                elem: if self.tile { Elem::F32 } else { Elem::U8 },
                block: vec![groups, cols],
                window,
                index,
            }
        });
        let bias = match cut {
            Cut::Blocked { tn } if self.bias => Some(Operand {
                array: vec![e, 1, n],
                elem: Elem::Bf16,
                block: vec![1, 1, tn],
                window: Window::Blocked,
                index: Box::new(move |b: &mut Body, g: &Grid| {
                    let ex = expert(b, g)?;
                    Ok(vec![ex, b.i32(0), g.ids[0]])
                }),
            }),
            Cut::Blocked { .. } | Cut::Element { .. } if self.bias => {
                return bad("a bias over an element window");
            }
            _ => None,
        };
        if !matches!(self.out, Elem::F32 | Elem::Bf16) || (self.bias && self.out != Elem::Bf16) {
            return bad(format!("a {:?} result (bias {})", self.out, self.bias));
        }
        let out = Operand {
            array: vec![rows, out_width],
            elem: self.out,
            block: vec![tm, cols],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![last(b, g)?, g.ids[0]])),
        };

        let convert = if self.w == Elem::Bf16 {
            0
        } else {
            k * cols * 2
        };
        // The scaled codes' exponents widen to f32 beside them.
        let scaling = if self.scaled {
            2 * groups * cols + 4 * k * cols
        } else if self.widens() {
            2 * 4 * groups * cols + 4 * k * cols + 4 * groups * 128 * cols
        } else if self.tile {
            2 * 4 * groups * cols + 4 * tm * cols
        } else {
            0
        };
        let vmem =
            2 * tm * k * 2 + 2 * self.block_dma(cols) + convert + scaling + 2 * tm * cols * 4;
        // At least the compiler's own scoped default, where the device has it.
        let floor = (32 << 20).min(self.vmem_cap);
        let limit = (vmem * 2 + (8 << 20)).clamp(floor, self.vmem_cap);
        let kernel = Kernel {
            name: "gmm".into(),
            grid: vec![nblocks, self.tiles],
            semantics: vec![Sem::Arbitrary, Sem::Arbitrary],
            prefetch: vec![vec![self.tiles], vec![1]],
            ins: [Some(xs), Some(bank), scales, bias]
                .into_iter()
                .flatten()
                .collect(),
            outs: vec![out],
            vmem_limit: Some(limit as u64),
            scratch: vec![],
        };
        kernel.build(|b, blk| {
            let zero = b.index(0);
            let nt = b.load(blk.grid.prefetch[1], &[zero])?;
            let live = b.cmpi(ICmp::Slt, blk.grid.ids[1], nt)?;
            b.when(live, |b| {
                let y = if self.widens() {
                    let x = b.vload_all(blk.ins[0])?;
                    let w = b.vload_all(blk.ins[1])?;
                    let s = b.vload_all(blk.ins[2])?;
                    let w = widen_codes(b, w, s, groups, cols)?;
                    let acc = b.splat_f(&[tm, cols], Elem::F32, 0.0)?;
                    b.matmul(x, w, acc, false)?
                } else if self.tile {
                    tile_dot(b, blk, tm, groups, cols)?
                } else {
                    let x = b.vload_all(blk.ins[0])?;
                    let w = b.vload_all(blk.ins[1])?;
                    let w = if self.scaled {
                        let s = b.vload_all(blk.ins[2])?;
                        scale_codes(b, w, s, groups, cols)?
                    } else {
                        b.convert_f(w, Elem::Bf16)?
                    };
                    let acc = b.splat_f(&[tm, cols], Elem::F32, 0.0)?;
                    b.matmul(x, w, acc, orient == Orient::NMajor)?
                };
                let y = if self.bias {
                    use crate::mosaic::MTy;
                    let r = blk.ins[blk.ins.len() - 1];
                    let row = b.vload_all(r)?;
                    let row = b.op(
                        "vector.shape_cast",
                        &[row],
                        "",
                        &[MTy::Vector(vec![1, cols], Elem::Bf16)],
                    )[0];
                    let row = b.convert_f(row, Elem::F32)?;
                    let row = b.op(
                        "vector.broadcast",
                        &[row],
                        "",
                        &[MTy::Vector(vec![tm, cols], Elem::F32)],
                    )[0];
                    b.addf(y, row)?
                } else {
                    y
                };
                let Cut::Element { width } = cut else {
                    let y = b.convert_f(y, self.out)?;
                    return b.vstore_all(y, blk.outs[0]);
                };
                // The expert's outputs turned to column 0 in VMEM (gathering
                // the rows at a per-row column offset afterwards relays every
                // row out: v6e gpt-oss, 0.2 ms a layer at 4096 tokens).
                let mut lags = self.lags(width);
                lags.sort_unstable();
                lags.dedup();
                let ex = expert(b, &blk.grid)?;
                let nn = b.i32(n);
                let col = b.muli(ex, nn)?;
                let lane = b.i32(128);
                let start = b.divsi(col, lane)?;
                let start = b.muli(start, lane)?;
                let top = b.i32(e * n - width);
                let start = b.minsi(start, top)?;
                let lag = b.subi(col, start)?;
                for o in lags {
                    let at = b.i32(o);
                    let here = b.cmpi(ICmp::Eq, lag, at)?;
                    b.when(here, |b| {
                        let y = if o == 0 {
                            y
                        } else {
                            b.op(
                                "tpu.rotate",
                                &[y],
                                &format!("amount = {} : si32, dimension = 1 : si32", width - o),
                                &[MTy::Vector(vec![tm, width], Elem::F32)],
                            )[0]
                        };
                        let y = b.convert_f(y, self.out)?;
                        b.vstore_all(y, blk.outs[0])
                    })?;
                }
                Ok(())
            })
        })
    }
}

/// e4m3 codes `w` (`[K, cols]`) as bf16, each times its 128-code block's
/// f32 factor (`s`: `[K / 128, cols]`), the product rounded once, as the
/// bf16 deployment holds the weight.
fn widen_codes(b: &mut Body, w: V, s: V, groups: i64, cols: i64) -> Built<V> {
    use crate::mosaic::MTy;
    let k = groups * 128;
    let f32s = |d: &[i64]| MTy::Vector(d.to_vec(), Elem::F32);
    let w = b.convert_f(w, Elem::F32)?;
    let s = b.op("vector.shape_cast", &[s], "", &[f32s(&[groups, 1, cols])])[0];
    let s = b.op("vector.broadcast", &[s], "", &[f32s(&[groups, 128, cols])])[0];
    let s = b.op("vector.shape_cast", &[s], "", &[f32s(&[k, cols])])[0];
    let w = b.mulf(w, s)?;
    b.convert_f(w, Elem::Bf16)
}

/// A tile's rows against block-FP8 codes (`ins[1]`, `[K, cols]` e4m3) under
/// f32 factors a 128 of K (`ins[2]`, `[K / 128, cols]`): a dot a 128-code
/// block of K, its f32 partial scaled by the block's factors and summed, f32
/// `[tm, cols]`. The codes enter the dot as e4m3 against the bf16 rows (each
/// code exact): widening them to bf16 on the VPU first holds a v6e to 0.87
/// TB/s of codes, against 1.43 as they are; widening each to f32 under its
/// factor, to 0.6.
fn tile_dot(
    b: &mut Body,
    blk: &crate::mosaic::Blocks,
    tm: i64,
    groups: i64,
    cols: i64,
) -> Built<V> {
    use crate::mosaic::MTy;
    let zero = b.index(0);
    let mut acc = b.splat_f(&[tm, cols], Elem::F32, 0.0)?;
    for g in 0..groups {
        let at = b.index(g * 128);
        let x = b.vload(blk.ins[0], &[zero, at], &[tm, 128])?;
        let w = b.vload(blk.ins[1], &[at, zero], &[128, cols])?;
        let z = b.splat_f(&[tm, cols], Elem::F32, 0.0)?;
        let p = b.matmul(x, w, z, false)?;
        let row = b.index(g);
        let s = b.vload(blk.ins[2], &[row, zero], &[1, cols])?;
        let s = b.op(
            "vector.broadcast",
            &[s],
            "",
            &[MTy::Vector(vec![tm, cols], Elem::F32)],
        )[0];
        let p = b.mulf(p, s)?;
        acc = b.addf(acc, p)?;
    }
    Ok(acc)
}

/// e2m1 codes `w` (`[K, cols]`, as `u4`) as bf16, each times
/// `2^(s - 127)` of its 32-code block's exponent byte (`s`: `[K / 32, cols]`),
/// built as f32 bits: an exact product, the codes holding two significant
/// bits. A magnitude code `m >= 2` is `2^((m >> 1) - 1) · (1 + (m & 1) / 2)`,
/// so its bits are `m << 22` with the exponent `s - 1` added; `m = 1` (0.5) is
/// that exponent alone and `m = 0` is zero.
fn scale_codes(b: &mut Body, w: V, s: V, groups: i64, cols: i64) -> Built<V> {
    use crate::mosaic::MTy;
    let k = groups * 32;
    let int = |d: &[i64]| MTy::Vector(d.to_vec(), Elem::I32);
    let full = int(&[k, cols]);
    let splat = |b: &mut Body, x: i64, t: &MTy| {
        b.op(
            "arith.constant",
            &[],
            &format!("value = dense<{x}> : {t}"),
            std::slice::from_ref(t),
        )[0]
    };
    let bin = |b: &mut Body, op: &str, x: V, y: V| {
        let t = b.ty(x).clone();
        let attrs = if matches!(op, "arith.shli" | "arith.addi" | "arith.subi") {
            "overflowFlags = #arith.overflow<none>"
        } else {
            ""
        };
        b.op(op, &[x, y], attrs, &[t])[0]
    };
    // The block's exponent, less one, in an f32 exponent field, every code's.
    let g = int(&[groups, cols]);
    let e = b.op("arith.extui", &[s], "", std::slice::from_ref(&g))[0];
    let one = splat(b, 1, &g);
    let e = bin(b, "arith.subi", e, one);
    let at = splat(b, 23, &g);
    let e = bin(b, "arith.shli", e, at);
    let e = b.op("vector.shape_cast", &[e], "", &[int(&[groups, 1, cols])])[0];
    let e = b.op("vector.broadcast", &[e], "", &[int(&[groups, 32, cols])])[0];
    let e = b.op("vector.shape_cast", &[e], "", std::slice::from_ref(&full))[0];
    let c = b.op("arith.extui", &[w], "", std::slice::from_ref(&full))[0];
    let seven = splat(b, 7, &full);
    let m = bin(b, "arith.andi", c, seven);
    let eight = splat(b, 8, &full);
    let sign = bin(b, "arith.andi", c, eight);
    let at = splat(b, 28, &full);
    let sign = bin(b, "arith.shli", sign, at);
    let at = splat(b, 22, &full);
    let big = bin(b, "arith.shli", m, at);
    let big = bin(b, "arith.addi", big, e);
    let zero = splat(b, 0, &full);
    let neg = bin(b, "arith.subi", zero, m);
    let small = bin(b, "arith.andi", e, neg);
    let two = splat(b, 2, &full);
    let low = b.op(
        "arith.cmpi",
        &[m, two],
        "predicate = 6 : i64",
        &[MTy::Vector(vec![k, cols], Elem::Pred)],
    )[0];
    let t = b.op(
        "arith.select",
        &[low, small, big],
        "",
        std::slice::from_ref(&full),
    )[0];
    let bits = bin(b, "arith.ori", t, sign);
    let f = b.op(
        "arith.bitcast",
        &[bits],
        "",
        &[MTy::Vector(vec![k, cols], Elem::F32)],
    )[0];
    b.convert_f(f, Elem::Bf16)
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
            scaled: false,
            tile: false,
            bias: false,
            out: Elem::F32,
            prefetch: false,
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
        let lags = d.lags(2944);
        assert_eq!(&lags[..3], &[0, 64, 0]);
        assert_eq!(lags[31], 64);
        assert!(d.kernel().unwrap().module.contains("element_window"));
    }
}
