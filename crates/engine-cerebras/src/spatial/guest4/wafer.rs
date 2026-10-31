//! The VM on the head's VM columns (serve2d's `VM` 1): `cols` VM columns after each head
//! block's root column, the lane the VM PEs (block b's row ry, column j:
//! lane position `(b·H + ry)·cols + j`, vocabulary rows `position·Vt ..`),
//! the feeder row F a hub that holds none.
//!
//! The host's side: a program's table
//! (packed four words an instruction), its pool and wide homes' seeds, the
//! token's home and `gvalid` on every VM PE; `gmode` 1 on every head PE so
//! sampled iterations take the VM's token; the head's `compute(steps)` runs
//! `run` and `run_pre` itself; after it, `gout` (D rows of W words, step s at
//! row s % D) on any VM PE holds each step's host-taken words.

use crate::sdk::{DataType, Order, Rect};
use crate::spatial::device::Device;

use super::lane::{OUT_WORDS, Ring};
use super::{Home, Table};
use crate::spatial::wafer::rect;

type R<T> = std::result::Result<T, String>;

/// The VM's shape, from the compiled program's params (`out.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmLayout {
    /// The head's first column (0 on a head-only image; serve2d's head sits
    /// east of the layer grids).
    pub x0: usize,
    /// Head blocks (the head's width over C + 1).
    pub blocks: usize,
    /// Region rows a block (H); a VM column is H + 1 high (the hub at F).
    pub rows: usize,
    /// VM columns a block (`VMC`): the head's Vr rows split `cols` ways.
    pub cols: usize,
    /// Vocabulary rows a VM PE (Vt = Vr / cols): a wide register's length.
    pub vr: usize,
    /// A block's columns (C + cols) and its root column (R2X).
    pub block_cols: usize,
    pub root: usize,
    /// The feeder row (H / 2).
    pub feeder: usize,
    /// The image's VM limits: instructions, wide registers, pool words.
    pub max_ops: usize,
    pub nreg: usize,
    pub pool_words: usize,
    /// The VM op groups the image was built with (`feat`, pe.csl's F_*
    /// bits): a table using a group the image left out would run wrong.
    pub feat: u32,
    pub vocab: usize,
    /// The opcodes the image runs (bit o: opcode o; `opmask_lo` /
    /// `opmask_hi`, all ones by default).
    pub opmask: u64,
    /// Step input words a turn (`vm_sin`; 0: the host cannot write homes
    /// each step).
    pub sin: usize,
    /// Per-lane programs (`vm_lprog`): an op arena of `max_ops`
    /// instructions and a bank arena of `bank_words` words.
    pub lprog: bool,
    pub bank_words: usize,
    /// A second VM set's columns after this one's (`VMB`): the
    /// head's `gmode` covers them too.
    pub set_b_cols: usize,
    /// glane word 0 carries a live lane's rows a turn (`base | rows << 16`,
    /// rows 0 an empty lane: guest4hz on); guest4hs reads the word as the
    /// base alone.
    pub lane_rows: bool,
}

/// The head a VM sits on: its first column, chunk columns (`C`), root
/// group width, region rows, vocabulary rows a PE and the vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadGeometry {
    pub x0: usize,
    pub c: usize,
    pub gx: usize,
    pub hh: usize,
    pub vr: usize,
    pub vocab: usize,
}

/// A VM image's limits: instructions, wide registers, pool words, op
/// groups and opcodes, step input words, and per-lane programs' bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmLimits {
    pub max_ops: usize,
    pub nreg: usize,
    pub pool_words: usize,
    pub feat: u32,
    pub opmask: u64,
    pub sin: usize,
    pub lprog: bool,
    pub bank_words: usize,
}

impl VmLimits {
    /// The limits among a build's params, each name after `prefix` (`b_`:
    /// set B's); the opmask all ones and the rest 0 where absent.
    pub fn read(get: &impl Fn(&str) -> Option<usize>, prefix: &str) -> Option<VmLimits> {
        let k = |n: &str| get(&format!("{prefix}{n}"));
        let half = |n: &str| k(n).map_or(0xffff_ffff, |v| v as u64 & 0xffff_ffff);
        Some(VmLimits {
            max_ops: k("max_ops")?,
            nreg: k("nreg")?,
            pool_words: k("pool_words")?,
            feat: k("feat").unwrap_or(0) as u32,
            opmask: half("opmask_lo") | half("opmask_hi") << 32,
            sin: k("vm_sin").unwrap_or(0),
            lprog: k("vm_lprog").unwrap_or(0) != 0,
            bank_words: k("bank_words").unwrap_or(0),
        })
    }
}

/// `opmask_lo` / `opmask_hi` as one mask (all ones where the build says
/// nothing).
fn opmask(get: &impl Fn(&str) -> Option<usize>) -> u64 {
    let half = |k: &str| get(k).map_or(0xffff_ffff, |v| v as u64 & 0xffff_ffff);
    half("opmask_lo") | half("opmask_hi") << 32
}

impl VmLayout {
    /// The VM of a guest4h* head image (`VMC` 1 and the VM limits among its
    /// params); `None` for a program without one.
    pub fn of(get: impl Fn(&str) -> Option<usize>) -> Option<VmLayout> {
        if get("VMC")? != 1 {
            return None;
        }
        let (w, c, gx) = (get("W")?, get("C")?, get("Gx")?);
        let ngx = c / gx;
        Some(VmLayout {
            x0: 0,
            blocks: w / (c + 1),
            rows: get("H")?,
            cols: 1,
            vr: get("Vr")?,
            block_cols: c + 1,
            root: (ngx / 2) * gx + gx / 2,
            feeder: get("H")? / 2,
            max_ops: get("max_ops")?,
            nreg: get("nreg")?,
            pool_words: get("pool_words")?,
            feat: get("feat")? as u32,
            vocab: get("vocab")?,
            opmask: opmask(&get),
            sin: get("vm_sin").unwrap_or(0),
            lprog: get("vm_lprog").unwrap_or(0) != 0,
            bank_words: get("bank_words").unwrap_or(0),
            set_b_cols: get("VMB").unwrap_or(0),
            lane_rows: false,
        })
    }

    /// serve2d's (or serve2d_pipe's) head (one block of `C` = PY columns at
    /// `x0`, `HH` region rows) when its build carries the VM (`VM` 1 and the
    /// VM limits among its params) on its `VMC` = Vr / 32 VM columns.
    pub fn serve2d(x0: usize, get: impl Fn(&str) -> Option<usize>) -> Option<VmLayout> {
        // `VM` 1: the VM's program on the VM columns (`VMC` alone leaves
        // them fillers).
        if get("VM")? != 1 {
            return None;
        }
        // Set A's image states its op groups.
        get("feat")?;
        let head = HeadGeometry {
            x0,
            c: get("PY")?,
            gx: get("Gx")?,
            hh: get("HH")?,
            vr: get("Vr")?,
            vocab: get("vocab")?,
        };
        VmLayout::on_head(
            head,
            get("VMC")?,
            VmLimits::read(&get, "")?,
            get("VMB").unwrap_or(0),
        )
    }

    /// The VM on a serve2d head's `cols` VM columns, the image's `limits`,
    /// and `set_b_cols` of a second set's after them; `None` where the
    /// columns do not split the head's rows.
    #[must_use]
    pub fn on_head(
        head: HeadGeometry,
        cols: usize,
        limits: VmLimits,
        set_b_cols: usize,
    ) -> Option<VmLayout> {
        if cols == 0 || !head.vr.is_multiple_of(cols) || head.gx == 0 {
            return None;
        }
        let ngx = head.c / head.gx;
        Some(VmLayout {
            x0: head.x0,
            blocks: 1,
            rows: head.hh,
            cols,
            vr: head.vr / cols,
            block_cols: head.c + cols,
            root: (ngx / 2) * head.gx + head.gx / 2,
            feeder: head.hh / 2,
            max_ops: limits.max_ops,
            nreg: limits.nreg,
            pool_words: limits.pool_words,
            feat: limits.feat,
            vocab: head.vocab,
            opmask: limits.opmask,
            sin: limits.sin,
            lprog: limits.lprog,
            bank_words: limits.bank_words,
            set_b_cols,
            lane_rows: false,
        })
    }

    /// Set B of a two-set head (serve2d_pipe's `VMB` columns right after
    /// this set's VM columns, its own image): its geometry and its own
    /// limits (the `b_` params), for a program to be weighed against the
    /// image it would run on. `None` without a set B.
    pub fn set_b(&self, get: impl Fn(&str) -> Option<usize>) -> Option<VmLayout> {
        let cols = get("VMB").filter(|&c| c > 0)?;
        let mut limits = VmLimits::read(&get, "b_")?;
        if get("b_feat").is_none() {
            limits.feat = self.feat;
        }
        self.with_set_b(cols, get("Vr")?, limits)
    }

    /// Set B on `cols` columns after this set's, over the head's `vr`
    /// vocabulary rows, with its image's `limits`.
    #[must_use]
    pub fn with_set_b(&self, cols: usize, vr: usize, limits: VmLimits) -> Option<VmLayout> {
        if cols == 0 || !vr.is_multiple_of(cols) {
            return None;
        }
        Some(VmLayout {
            root: self.root + self.cols,
            cols,
            vr: vr / cols,
            max_ops: limits.max_ops,
            nreg: limits.nreg,
            pool_words: limits.pool_words,
            feat: limits.feat,
            opmask: limits.opmask,
            sin: limits.sin,
            lprog: limits.lprog,
            bank_words: limits.bank_words,
            // The head (where `gmode` lives) is the same two-set head.
            ..*self
        })
    }

    /// Block b's VM column j.
    pub fn column_j(&self, b: usize, j: usize) -> usize {
        self.x0 + b * self.block_cols + self.root + 1 + j
    }

    /// Block b's first VM column.
    pub fn column(&self, b: usize) -> usize {
        self.column_j(b, 0)
    }

    /// Every VM column, with its block and its index in the block.
    pub fn columns(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.blocks).flat_map(move |b| (0..self.cols).map(move |j| (b, j, self.column_j(b, j))))
    }

    /// The lane position (a vocabulary slice of Vt rows) of VM PE row `pe`
    /// of block b's VM column j; `None` on the hub.
    pub fn position(&self, b: usize, j: usize, pe: usize) -> Option<usize> {
        if pe == self.feeder {
            return None;
        }
        let ry = if pe < self.feeder { pe } else { pe - 1 };
        Some((b * self.rows + ry) * self.cols + j)
    }

    /// Lane positions in all (Vt vocabulary rows each).
    pub fn positions(&self) -> usize {
        self.blocks * self.rows * self.cols
    }

    /// The head's PEs (every block's columns, its region rows and the
    /// feeder row): where `gmode` lives.
    pub fn head(&self) -> Rect {
        rect(
            self.x0,
            0,
            self.blocks * self.block_cols + self.set_b_cols,
            self.rows + 1,
        )
    }

    /// Block 0's hub (its VM column's feeder-row PE): the ring and the pool
    /// are read there.
    pub fn hub(&self) -> (usize, usize) {
        (self.column(0), self.feeder)
    }

    /// Why `table` cannot load on this image, if it cannot.
    pub fn refuses(&self, table: &Table) -> Option<String> {
        if table.ins.len() > self.max_ops {
            return Some(format!(
                "{} instructions; the image holds {}",
                table.ins.len(),
                self.max_ops
            ));
        }
        if table.nreg as usize > self.nreg {
            return Some(format!(
                "{} wide registers; the image has {}",
                table.nreg, self.nreg
            ));
        }
        if table.pool_words as usize > self.pool_words {
            return Some(format!(
                "{} pool words; the image has {}",
                table.pool_words, self.pool_words
            ));
        }
        let missing = table.feat & !self.feat;
        if missing != 0 {
            return Some(format!(
                "op groups {missing:#x} (feat {:#x}) the image was built without (feat {:#x})",
                table.feat, self.feat
            ));
        }
        let ops = super::slots::opcodes(table);
        if ops & !self.opmask != 0 {
            return Some(format!(
                "opcodes {:#x} the image was built without",
                ops & !self.opmask
            ));
        }
        if !table.per_step.is_empty() {
            if self.sin == 0 {
                return Some(
                    "the host writes some of its homes every step; the image has no step input"
                        .into(),
                );
            }
            if let Err(why) = super::slots::step_entries(self, table) {
                return Some(why);
            }
        }
        if !table.wide_outputs.is_empty() {
            return Some("vocabulary-wide host outputs (read from the VM's registers)".into());
        }
        // A per-lane-program image keeps each lane's window in its bank: one
        // past the bank never fits, however few lanes run beside it.
        if self.lprog {
            let words = super::slots::Window::of(table).words(self.vr);
            if words > self.bank_words {
                return Some(format!(
                    "a window of {words} words; the image's bank holds {}",
                    self.bank_words
                ));
            }
        }
        None
    }

    /// Why `table` cannot run where the head embeds the VM's token (one
    /// sequence, several steps a launch): [`VmLayout::refuses`], and a table
    /// without a token home. A multi-lane launch forces every lane's token
    /// and takes the sampled one from the out ring.
    pub fn refuses_head(&self, table: &Table) -> Option<String> {
        self.refuses(table).or_else(|| {
            table
                .tok_home
                .is_none()
                .then(|| "no token home (the head embeds the EmbedTokens channel's word)".into())
        })
    }
}

/// The translator's six-word instructions `[op, fl, d, a, b, c]` as the VM's four
/// words `[op | fl << 8, d | a << 16, b, c]`, padded to `max_ops`.
pub fn pack_ops(ins: &[[u32; 6]], max_ops: usize) -> R<Vec<u32>> {
    if ins.len() > max_ops {
        return Err(format!(
            "{} instructions for a table of {max_ops}",
            ins.len()
        ));
    }
    let mut out = vec![0u32; max_ops * 4];
    for (i, &[op, fl, d, a, b, c]) in ins.iter().enumerate() {
        if op >= 256 || fl >= 1 << 24 || d >= 1 << 16 || a >= 1 << 16 {
            return Err(format!(
                "instruction {i} {:?} does not pack",
                [op, fl, d, a, b, c]
            ));
        }
        out[i * 4..i * 4 + 4].copy_from_slice(&[op | fl << 8, d | a << 16, b, c]);
    }
    Ok(out)
}

/// A home's seed words (pool: u32 words; wide: the whole vocabulary row, f32
/// bits of the value, ints converted as eta reads them).
#[derive(Debug, Clone)]
pub struct Seed {
    pub kind: &'static str,
    pub at: u32,
    pub n: u32,
    pub dtype: eta_ir::Dtype,
    pub words: Vec<u32>,
}

impl Seed {
    #[must_use]
    pub fn of(home: &Home, words: Vec<u32>) -> Seed {
        Seed {
            kind: home.kind,
            at: home.at,
            n: home.n,
            dtype: home.dtype,
            words,
        }
    }
}

/// The pool block and the wide rows (lane position x register x Vr, f32) the
/// seeds make.
pub(crate) fn seed_images(lay: &VmLayout, seeds: &[Seed]) -> R<(Vec<u32>, Vec<f32>)> {
    use eta_ir::Dtype;
    let lanes = lay.positions();
    let mut pool = vec![0u32; lay.pool_words];
    let mut wide = vec![0f32; lanes * lay.nreg * lay.vr];
    for seed in seeds {
        let (at, n) = (seed.at as usize, seed.n as usize);
        if seed.kind == "pool" {
            let dst = pool
                .get_mut(at..at + n.min(seed.words.len()))
                .ok_or_else(|| format!("pool home {at}+{n} past {} words", lay.pool_words))?;
            dst.copy_from_slice(&seed.words[..dst.len()]);
        } else {
            if at >= lay.nreg {
                return Err(format!("wide home {at} past {} registers", lay.nreg));
            }
            for (v, &w) in seed.words.iter().enumerate().take(lanes * lay.vr) {
                let x = match seed.dtype {
                    Dtype::F32 => f32::from_bits(w),
                    Dtype::I32 => w as i32 as f32,
                    _ => w as f32,
                };
                let (lane, j) = (v / lay.vr, v % lay.vr);
                wide[(lane * lay.nreg + at) * lay.vr + j] = x;
            }
        }
    }
    Ok((pool, wide))
}

/// Loads `table` with its homes' `seeds` on every VM PE and sets `gmode`
/// (1: the VM picks sampled tokens) on every head PE; resets the step count.
pub fn load(rt: &mut dyn Device, lay: &VmLayout, table: &Table, seeds: &[Seed]) -> R<()> {
    if let Some(why) = lay.refuses_head(table) {
        return Err(format!("the VM image cannot run this program: {why}"));
    }
    let e = |e: crate::sdk::Error| e.to_string();
    let ops = pack_ops(&table.ins, lay.max_ops)?;
    let (pool, wide) = seed_images(lay, seeds)?;
    let tok_home = table.tok_home.unwrap_or_default();
    let nv = lay.rows + 1;
    let tile =
        |words: &[u32]| -> Vec<u32> { (0..nv).flat_map(|_| words.iter().copied()).collect() };
    let b32 = DataType::Bits32;
    for (b, j, x) in lay.columns() {
        let col = rect(x, 0, 1, nv);
        rt.memcpy_h2d("gops", &tile(&ops), col, ops.len(), Order::RowMajor, b32)
            .map_err(e)?;
        rt.memcpy_h2d("gpool", &tile(&pool), col, pool.len(), Order::RowMajor, b32)
            .map_err(e)?;
        rt.memcpy_h2d(
            "gvalid",
            &vec![lay.vocab as u32; nv],
            col,
            1,
            Order::RowMajor,
            b32,
        )
        .map_err(e)?;
        rt.memcpy_h2d(
            "gtok_home",
            &vec![tok_home; nv],
            col,
            1,
            Order::RowMajor,
            b32,
        )
        .map_err(e)?;
        rt.memcpy_h2d("gostep", &vec![0; nv], col, 1, Order::RowMajor, b32)
            .map_err(e)?;
        // Region rows hold lane positions; the hub (row F) holds none.
        let per = lay.nreg * lay.vr;
        let mut w = vec![0u32; nv * per];
        for pe in 0..nv {
            if let Some(lane) = lay.position(b, j, pe) {
                for k in 0..per {
                    w[pe * per + k] = wide[lane * per + k].to_bits();
                }
            }
        }
        rt.memcpy_h2d("gwide", &w, col, per, Order::RowMajor, b32)
            .map_err(e)?;
    }
    set_gmode(rt, lay, 1)
}

/// `gmode` on every head PE: 1, the VM picks the sampled iterations'
/// tokens (and runs every iteration); 0, the head's argmax (the VM idles).
pub fn set_gmode(rt: &mut dyn Device, lay: &VmLayout, mode: u32) -> R<()> {
    let head = lay.head();
    let n = (head.w * head.h) as usize;
    rt.memcpy_h2d(
        "gmode",
        &vec![mode; n],
        head,
        1,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())
}

/// `vm_conc`: the set's rows this launch (`gring[3]`; its head rows,
/// skipped turns included) on every one of its VM PEs, with the ring's
/// header as it reads. The VM reads it at every launch's start and keeps
/// it, so it is written before every launch.
pub fn write_conc_rows(
    rt: &mut dyn Device,
    lay: &VmLayout,
    ring: super::lane::Ring,
    rows: u32,
) -> R<()> {
    let at = rect(lay.column_j(0, 0), 0, lay.cols, lay.rows + 1);
    let n = lay.cols * (lay.rows + 1);
    let words: Vec<u32> = (0..n)
        .flat_map(|_| [ring.depth, ring.width, ring.lanes, rows])
        .collect();
    rt.memcpy_h2d("gring", &words, at, 4, Order::RowMajor, DataType::Bits32)
        .map_err(|e| e.to_string())
}

/// The VM's pool (every VM PE holds the same words): the pool homes' values
/// after its last step, the token's home among them.
pub fn read_pool(rt: &mut dyn Device, lay: &VmLayout) -> R<Vec<u32>> {
    let mut pool = vec![0u32; lay.pool_words];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gpool",
        &mut pool,
        rect(x, y, 1, 1),
        lay.pool_words,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    Ok(pool)
}

/// The out ring after a launch of `steps` steps (at most its depth) that
/// followed `first` steps since the program loaded: each step's words (W,
/// padded to the 16 a step's `out` has), read from block 0's hub.
pub fn read_out(
    rt: &mut dyn Device,
    lay: &VmLayout,
    ring: Ring,
    first: u64,
    steps: usize,
) -> R<Vec<Vec<u32>>> {
    if steps > ring.depth as usize {
        return Err(format!(
            "{steps} steps a launch; the VM's ring holds {}",
            ring.depth
        ));
    }
    let n = (ring.depth * ring.width) as usize;
    let mut words = vec![0u32; n];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gout",
        &mut words,
        rect(x, y, 1, 1),
        n,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    Ok((first..first + steps as u64)
        .map(|s| {
            let at = ring.index(s, 0);
            let mut row = words[at..at + ring.width as usize].to_vec();
            row.resize(OUT_WORDS as usize, 0);
            row
        })
        .collect())
}

/// The VM's per-lane banks (a time-multiplexed VM, `gring` lanes L > 1):
/// on every VM PE, L banks of `pool_words + (nreg - 1)·Vr` words, bank l
/// the lane's pool (its pool homes seeded) then its wide registers
/// W1..W(nreg-1) for that PE's lane position (zeros on the hub). `seeds[l]`
/// are lane l's homes' seeds. Laid out a VM column at a time: rows 0..=H,
/// each row's L banks.
pub fn bank_images(lay: &VmLayout, seeds: &[Vec<Seed>]) -> R<Vec<Vec<u32>>> {
    let (nv, wide_n) = (lay.rows + 1, (lay.nreg - 1) * lay.vr);
    let bank = lay.pool_words + wide_n;
    let cols: Vec<(usize, usize, usize)> = lay.columns().collect();
    let mut columns = vec![vec![0u32; nv * seeds.len() * bank]; cols.len()];
    for (l, lane_seeds) in seeds.iter().enumerate() {
        let (pool, wide) = seed_images(lay, lane_seeds)?;
        for (column, &(b, j, _)) in columns.iter_mut().zip(&cols) {
            for pe in 0..nv {
                let at = (pe * seeds.len() + l) * bank;
                column[at..at + lay.pool_words].copy_from_slice(&pool);
                let Some(lane) = lay.position(b, j, pe) else {
                    continue;
                };
                // W0 holds the logits; the bank keeps W1 on.
                for r in 1..lay.nreg {
                    for j in 0..lay.vr {
                        column[at + lay.pool_words + (r - 1) * lay.vr + j] =
                            wide[(lane * lay.nreg + r) * lay.vr + j].to_bits();
                    }
                }
            }
        }
    }
    Ok(columns)
}

/// What a VM PE received for the last sampled step: the logits row x 1/rms
/// (block by block, region rows in order, the hub skipped), vocabulary order.
pub fn read_plane(rt: &mut dyn Device, lay: &VmLayout) -> R<Vec<f32>> {
    let (nv, w) = (lay.rows + 1, lay.vr + 1);
    let mut plane = vec![0f32; lay.positions() * lay.vr];
    for (b, j, x) in lay.columns() {
        let mut vin = vec![0u32; nv * w];
        rt.memcpy_d2h(
            "vin",
            &mut vin,
            rect(x, 0, 1, nv),
            w,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
        for pe in 0..nv {
            let Some(lane) = lay.position(b, j, pe) else {
                continue;
            };
            let row = &vin[pe * w..(pe + 1) * w];
            let scale = f32::from_bits(row[lay.vr]);
            for (k, &v) in row[..lay.vr].iter().enumerate() {
                plane[lane * lay.vr + k] = f32::from_bits(v) * scale;
            }
        }
    }
    plane.truncate(lay.vocab);
    Ok(plane)
}

/// Writes every lane's bank ([`bank_images`]) to `gbank` on every VM PE:
/// the lanes' carried state before a launch (a request starting in lane l,
/// or all lanes at once).
pub fn write_banks(rt: &mut dyn Device, lay: &VmLayout, seeds: &[Vec<Seed>]) -> R<()> {
    let images = bank_images(lay, seeds)?;
    let nv = lay.rows + 1;
    let per = images.first().map_or(0, |c| c.len() / nv);
    for (image, (_, _, x)) in images.iter().zip(lay.columns()) {
        rt.memcpy_h2d(
            "gbank",
            image,
            rect(x, 0, 1, nv),
            per,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `gbank` as it stands on every VM column (rows 0..=H, each row's `lanes`
/// banks), for a read-modify-write of some lanes' banks.
pub fn read_banks(rt: &mut dyn Device, lay: &VmLayout, lanes: usize) -> R<Vec<Vec<u32>>> {
    let nv = lay.rows + 1;
    let per = lanes * (lay.pool_words + (lay.nreg - 1) * lay.vr);
    lay.columns()
        .map(|(_, _, x)| {
            let mut words = vec![0u32; nv * per];
            rt.memcpy_d2h(
                "gbank",
                &mut words,
                rect(x, 0, 1, nv),
                per,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(|e| e.to_string())?;
            Ok(words)
        })
        .collect()
}

/// [`read_banks`]' images back to `gbank`.
pub fn write_bank_images(rt: &mut dyn Device, lay: &VmLayout, images: &[Vec<u32>]) -> R<()> {
    let nv = lay.rows + 1;
    for (image, (_, _, x)) in images.iter().zip(lay.columns()) {
        rt.memcpy_h2d(
            "gbank",
            image,
            rect(x, 0, 1, nv),
            image.len() / nv,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Every lane's pool part of its bank (`lanes` banks), read at block 0's
/// hub: after a launch, each lane's pool homes (its next token's among
/// them) as its last turn left them.
pub fn read_bank_pools(rt: &mut dyn Device, lay: &VmLayout, lanes: usize) -> R<Vec<Vec<u32>>> {
    let bank = lay.pool_words + (lay.nreg - 1) * lay.vr;
    let mut words = vec![0u32; lanes * bank];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gbank",
        &mut words,
        rect(x, y, 1, 1),
        lanes * bank,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    Ok(words
        .chunks(bank)
        .map(|b| b[..lay.pool_words].to_vec())
        .collect())
}

/// `gbank` as `per` words a row on every VM column (a per-lane-program
/// image's whole bank arena: `bank_words`).
pub fn read_bank_words(rt: &mut dyn Device, lay: &VmLayout, per: usize) -> R<Vec<Vec<u32>>> {
    let nv = lay.rows + 1;
    lay.columns()
        .map(|(_, _, x)| {
            let mut words = vec![0u32; nv * per];
            rt.memcpy_d2h(
                "gbank",
                &mut words,
                rect(x, 0, 1, nv),
                per,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(|e| e.to_string())?;
            Ok(words)
        })
        .collect()
}

/// The bank arena's words at block 0's hub (the pool parts of every
/// lane's window).
pub fn read_hub_bank(rt: &mut dyn Device, lay: &VmLayout) -> R<Vec<u32>> {
    let mut words = vec![0u32; lay.bank_words];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gbank",
        &mut words,
        rect(x, y, 1, 1),
        lay.bank_words,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    Ok(words)
}

/// A per-lane-program image's lanes as `slots` places them: the op arena,
/// `glane`, `gbdesc`, `gsdesc` (with step input) and the bank arena
/// `banks` on every VM PE; `fresh` also the vocabulary, the step count
/// and a zero pool (the image's first load).
pub fn load_slots(
    rt: &mut dyn Device,
    lay: &VmLayout,
    slots: &super::slots::Slots,
    banks: &[Vec<u32>],
    fresh: bool,
) -> R<()> {
    let e = |e: crate::sdk::Error| e.to_string();
    let ops = slots.ops(lay)?;
    let (glane, gbdesc) = slots.glane_gbdesc(lay);
    let gsdesc = slots.gsdesc();
    let nv = lay.rows + 1;
    let tile =
        |words: &[u32]| -> Vec<u32> { (0..nv).flat_map(|_| words.iter().copied()).collect() };
    let b32 = DataType::Bits32;
    for ((_, _, x), bank) in lay.columns().zip(banks) {
        let col = rect(x, 0, 1, nv);
        rt.memcpy_h2d("gops", &tile(&ops), col, ops.len(), Order::RowMajor, b32)
            .map_err(e)?;
        rt.memcpy_h2d(
            "glane",
            &tile(&glane),
            col,
            glane.len(),
            Order::RowMajor,
            b32,
        )
        .map_err(e)?;
        rt.memcpy_h2d(
            "gbdesc",
            &tile(&gbdesc),
            col,
            gbdesc.len(),
            Order::RowMajor,
            b32,
        )
        .map_err(e)?;
        if lay.sin > 0 {
            rt.memcpy_h2d(
                "gsdesc",
                &tile(&gsdesc),
                col,
                gsdesc.len(),
                Order::RowMajor,
                b32,
            )
            .map_err(e)?;
        }
        rt.memcpy_h2d("gbank", bank, col, lay.bank_words, Order::RowMajor, b32)
            .map_err(e)?;
        if fresh {
            rt.memcpy_h2d(
                "gpool",
                &vec![0; nv * lay.pool_words],
                col,
                lay.pool_words,
                Order::RowMajor,
                b32,
            )
            .map_err(e)?;
            rt.memcpy_h2d(
                "gvalid",
                &vec![lay.vocab as u32; nv],
                col,
                1,
                Order::RowMajor,
                b32,
            )
            .map_err(e)?;
            rt.memcpy_h2d("gostep", &vec![0; nv], col, 1, Order::RowMajor, b32)
                .map_err(e)?;
        }
    }
    if fresh {
        set_gmode(rt, lay, 1)?;
    }
    Ok(())
}

/// The launch's step input ([`super::slots::step_images`]) to `gsin`.
pub fn write_step(rt: &mut dyn Device, lay: &VmLayout, images: &[Vec<u32>]) -> R<()> {
    let nv = lay.rows + 1;
    for (image, (_, _, x)) in images.iter().zip(lay.columns()) {
        rt.memcpy_h2d(
            "gsin",
            image,
            rect(x, 0, 1, nv),
            image.len() / nv,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The step descriptors of a single-program multi-lane VM (every lane the
/// same entries) to `gsdesc`.
pub fn write_gsdesc(rt: &mut dyn Device, lay: &VmLayout, words: &[u32]) -> R<()> {
    let nv = lay.rows + 1;
    let tile: Vec<u32> = (0..nv).flat_map(|_| words.iter().copied()).collect();
    for (_, _, x) in lay.columns() {
        rt.memcpy_h2d(
            "gsdesc",
            &tile,
            rect(x, 0, 1, nv),
            words.len(),
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The VM's sampled-turn count (`gostep`, at block 0's hub): where its next
/// out-ring row goes.
pub fn read_ostep(rt: &mut dyn Device, lay: &VmLayout) -> R<u64> {
    let mut word = [0u32; 1];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gostep",
        &mut word,
        rect(x, y, 1, 1),
        1,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    Ok(u64::from(word[0]))
}

/// `count` out-ring rows of a multi-lane VM from row `first` on: the VM
/// writes a row each sampled turn (its `ostep`, every lane's turns in turn
/// order), at `ostep % D`; padded to the 16 words a step's `out` has.
pub fn read_out_rows(
    rt: &mut dyn Device,
    lay: &VmLayout,
    ring: Ring,
    first: u64,
    count: usize,
) -> R<Vec<Vec<u32>>> {
    if count > ring.depth as usize {
        return Err(format!(
            "{count} sampled turns a launch; the VM's ring holds {}",
            ring.depth
        ));
    }
    let n = (ring.depth * ring.width) as usize;
    let mut words = vec![0u32; n];
    let (x, y) = lay.hub();
    rt.memcpy_d2h(
        "gout",
        &mut words,
        rect(x, y, 1, 1),
        n,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    let w = ring.width as usize;
    Ok((first..first + count as u64)
        .map(|r| {
            let at = (r % u64::from(ring.depth)) as usize * w;
            let mut row = words[at..at + w].to_vec();
            row.resize(OUT_WORDS as usize, 0);
            row
        })
        .collect())
}

fn dtype_of(s: &str) -> eta_ir::Dtype {
    match s {
        "F32" => eta_ir::Dtype::F32,
        "I32" => eta_ir::Dtype::I32,
        "Bool" => eta_ir::Dtype::Bool,
        _ => eta_ir::Dtype::U32,
    }
}

/// A VM case (JSON: a translated table, its homes, outputs and token home,
/// and the homes' seeds) as a table and its seeds: the simulator drivers'
/// way to load a program without an instance.
pub fn case_table(case: &serde_json::Value) -> R<(Table, Vec<Seed>)> {
    let u = |v: &serde_json::Value| v.as_u64().unwrap_or(0) as u32;
    let flat: Vec<u32> = case["table"]
        .as_array()
        .ok_or("the case has no table")?
        .iter()
        .flat_map(|i| match i.as_array() {
            Some(words) => words.iter().map(u).collect(),
            None => vec![u(i)],
        })
        .collect();
    let homes: Vec<Home> = case["homes"]
        .as_array()
        .ok_or("the case has no homes")?
        .iter()
        .map(|h| Home {
            chan: u(&h["chan"]),
            kind: if h["kind"] == "pool" { "pool" } else { "wide" },
            at: u(&h["at"]),
            n: u(&h["n"]),
            dtype: dtype_of(h["dtype"].as_str().unwrap_or("U32")),
        })
        .collect();
    let seeds = homes
        .iter()
        .filter_map(|h| {
            let words = case["seeds"][h.chan.to_string()].as_array()?;
            Some(Seed::of(h, words.iter().map(u).collect()))
        })
        .collect();
    let table = Table {
        ins: flat
            .as_chunks::<6>()
            .0
            .iter()
            .map(|w| [w[0], w[1], w[2], w[3], w[4], w[5]])
            .collect(),
        pre_len: case["pre_len"].as_u64().unwrap_or(0) as usize,
        homes,
        outputs: case["outputs"]
            .as_array()
            .ok_or("the case has no outputs")?
            .iter()
            .map(|o| {
                (
                    u(&o["chan"]),
                    u(&o["slot"]),
                    u(&o["n"]),
                    dtype_of(o["dtype"].as_str().unwrap_or("U32")),
                )
            })
            .collect(),
        wide_outputs: Vec::new(),
        per_step: case["per_step"]
            .as_array()
            .map(|p| p.iter().map(|e| u(&e["chan"])).collect())
            .unwrap_or_default(),
        tok_home: case["tok_home"].as_u64().map(|t| t as u32),
        nreg: u(&case["nreg"]),
        pool_words: u(&case["pool_words"]),
        feat: u(&case["feat"]),
        remat: false,
        bank_regs: case["bank_regs"]
            .as_array()
            .map(|r| r.iter().map(u).collect())
            .unwrap_or_default(),
        bank_pool: case["bank_pool"]
            .as_array()
            .map(|r| r.iter().map(|p| (u(&p[0]), u(&p[1]))).collect())
            .unwrap_or_default(),
    };
    Ok((table, seeds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_pack_as_run_py() {
        let packed = pack_ops(&[[7, 3, 2, 5, 9, 11]], 2).unwrap();
        assert_eq!(packed, [7 | 3 << 8, 2 | 5 << 16, 9, 11, 0, 0, 0, 0]);
        assert!(pack_ops(&[[300, 0, 0, 0, 0, 0]], 1).is_err());
    }

    #[test]
    fn layout_of_the_0112_image() {
        let params = [
            ("W", 528),
            ("VMC", 1),
            ("H", 300),
            ("C", 32),
            ("Vr", 32),
            ("Gx", 8),
            ("max_ops", 48),
            ("nreg", 6),
            ("pool_words", 64),
            ("feat", 99),
            ("vocab", 151_936),
        ];
        let lay = VmLayout::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap();
        // 16 blocks of 33 columns, the root at 20 (R2X), the VM after it.
        assert_eq!(
            (lay.blocks, lay.root, lay.column(1), lay.feeder),
            (16, 20, 54, 150)
        );
        assert!(lay.positions() * lay.vr >= lay.vocab);
    }

    #[test]
    fn banks_hold_each_lane_s_pool_then_its_wide_registers() {
        let params = [
            ("W", 9),
            ("VMC", 1),
            ("H", 2),
            ("C", 8),
            ("Vr", 2),
            ("Gx", 2),
            ("max_ops", 8),
            ("nreg", 3),
            ("pool_words", 2),
            ("feat", 99),
            ("vocab", 4),
        ];
        let lay = VmLayout::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap();
        let home = |kind: &'static str, at: u32| Home {
            chan: 0,
            kind,
            at,
            n: 1,
            dtype: eta_ir::Dtype::U32,
        };
        let (pool, wide) = (home("pool", 1), home("wide", 2));
        // Lane l: pool word 1 = 10 + l; register W2 = [l, l+1, l+2, l+3] over the vocabulary.
        let seeds: Vec<Vec<Seed>> = (0..2u32)
            .map(|l| {
                vec![
                    Seed::of(&pool, vec![10 + l]),
                    Seed::of(&wide, (l..l + 4).collect()),
                ]
            })
            .collect();
        let cols = bank_images(&lay, &seeds).unwrap();
        // One block, rows 0 (lane position 0), 1 (the hub), 2 (position 1); banks of 2 + 2·2 words.
        let bank = |pe: usize, l: usize| &cols[0][(pe * 2 + l) * 6..(pe * 2 + l + 1) * 6];
        let f = |x: u32| (x as f32).to_bits();
        assert_eq!(bank(0, 1), [0, 11, 0, 0, f(1), f(2)]);
        assert_eq!(bank(1, 0), [0, 10, 0, 0, 0, 0]);
        assert_eq!(bank(2, 0), [0, 10, 0, 0, f(2), f(3)]);
    }

    #[test]
    fn serve2d_vm_columns_split_the_head_s_rows() {
        // serve2d_pipe's VM at full size: Vr 192 over VMC 6 columns of 32.
        let params = [
            ("VM", 1),
            ("VMC", 6),
            ("PY", 128),
            ("Gx", 16),
            ("HH", 792),
            ("Vr", 192),
            ("max_ops", 48),
            ("nreg", 6),
            ("pool_words", 64),
            ("feat", 99),
            ("vocab", 151_936),
        ];
        let lay =
            VmLayout::serve2d(512, |k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap();
        // R2X 72: the VM columns 512 + 73 .. 512 + 78; the head 128 + 6 wide.
        assert_eq!((lay.vr, lay.column_j(0, 5), lay.head().w), (32, 590, 134));
        // Region row 2's column 3 holds vocabulary rows (2·6 + 3)·32 ..; the hub none.
        assert_eq!(
            (lay.position(0, 3, 2), lay.position(0, 0, 396)),
            (Some(15), None)
        );
        assert!(lay.positions() * lay.vr >= lay.vocab);
    }

    #[test]
    fn a_table_needs_every_op_group_it_uses() {
        let params = [
            ("W", 18),
            ("VMC", 1),
            ("H", 4),
            ("C", 8),
            ("Vr", 32),
            ("Gx", 2),
            ("max_ops", 64),
            ("nreg", 8),
            ("pool_words", 256),
            ("feat", 511),
            ("vocab", 250),
        ];
        let lay = VmLayout::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap();
        let case = |feat: u32| {
            serde_json::json!({"table": [[0, 0, 0, 0, 0, 0]], "homes": [], "outputs": [], "seeds": {},
                               "tok_home": 0, "nreg": 1, "pool_words": 1, "feat": feat})
        };
        let (fits, _) = case_table(&case(8 | 3)).unwrap();
        assert_eq!(lay.refuses(&fits), None);
        // Top-p's mass pivot (F_PMASS, 512) on an image built at 511.
        let (pmass, _) = case_table(&case(8 | 512)).unwrap();
        assert!(lay.refuses(&pmass).unwrap().contains("0x200"));
    }

    /// serve2d_pipe's two VM sets as built for hardware: set A, and set B
    /// with radix top-k (pool 776, bank 600, nreg 6, 136 ops, op 28 out of
    /// its mask).
    fn pipe_sets() -> (VmLayout, VmLayout) {
        let params = [
            ("VM", 1),
            ("VMC", 6),
            ("PY", 128),
            ("Gx", 16),
            ("HH", 792),
            ("Vr", 192),
            ("feat", 1007),
            ("vocab", 151_936),
            ("max_ops", 104),
            ("nreg", 12),
            ("pool_words", 211),
            ("vm_lprog", 1),
            ("bank_words", 520),
            ("opmask_lo", 3_757_088_255),
            ("opmask_hi", 256),
            ("VMB", 6),
            ("b_max_ops", 136),
            ("b_nreg", 6),
            ("b_pool_words", 776),
            ("b_vm_lprog", 1),
            ("b_bank_words", 600),
            ("b_opmask_lo", 2_146_450_875 & !(1 << 28)),
            ("b_opmask_hi", 256),
        ];
        let get = |k: &str| params.iter().find(|p| p.0 == k).map(|p| p.1);
        let a = VmLayout::serve2d(512, get).unwrap();
        let b = a.set_b(get).unwrap();
        (a, b)
    }

    /// A program of `ins` instructions with a real-vocabulary shape:
    /// `nreg` wide registers, `pool` pool words, a bank window of `window`
    /// pool words and one wide register (W1).
    fn program(ins: usize, nreg: u32, pool: u32, window: u32) -> Table {
        let case = serde_json::json!({"table": vec![[0, 0, 0, 0, 0, 0]; ins], "homes": [], "outputs": [], "seeds": {},
                                      "tok_home": 0, "nreg": nreg, "pool_words": pool, "feat": 0,
                                      "bank_pool": [[0, window]], "bank_regs": [1]});
        case_table(&case).unwrap().0
    }

    #[test]
    fn set_b_sits_after_set_a_s_columns_with_its_own_limits() {
        let (a, b) = pipe_sets();
        assert_eq!(b.column_j(0, 0), a.column_j(0, 5) + 1);
        assert_eq!(
            (b.vr, b.max_ops, b.nreg, b.pool_words, b.bank_words),
            (32, 136, 6, 776, 600)
        );
        assert_eq!(b.opmask & 1 << 28, 0);
        assert_eq!((a.max_ops, a.pool_words, a.bank_words), (104, 211, 520));
    }

    #[test]
    fn tail_free_fits_neither_set_and_stays_on_the_host() {
        // 189f13bf and d5e399e6 at the real vocabulary: windows 552 and 551.
        let (a, b) = pipe_sets();
        for t in [program(64, 5, 904, 520), program(63, 5, 903, 519)] {
            assert!(a.refuses(&t).is_some_and(|w| w.contains("pool words")));
            assert!(b.refuses(&t).is_some_and(|w| w.contains("pool words")));
        }
        // A window past the bank never fits either, whatever its pool.
        let wide = program(40, 4, 200, 590);
        assert!(
            a.refuses(&wide)
                .is_some_and(|w| w.contains("window of 622 words"))
        );
        assert!(
            b.refuses(&wide)
                .is_some_and(|w| w.contains("window of 622 words"))
        );
    }

    #[test]
    fn typical_twice_fits_set_b_s_bank() {
        // 10dff76b: a window of 263 + 32 = 295 words, two in 600.
        let (a, b) = pipe_sets();
        let t = program(50, 6, 775, 263);
        assert!(a.refuses(&t).is_some_and(|w| w.contains("pool words")));
        assert_eq!(b.refuses(&t), None);
        let lane = || {
            let program = super::super::lane::Program {
                table: program(50, 6, 775, 263),
                epilogue: 0,
                homes: Default::default(),
                logits: String::new(),
            };
            Some(super::super::slots::LaneProgram {
                key: 1,
                program: std::sync::Arc::new(program),
            })
        };
        let two = super::super::slots::Slots::plan(&b, &[lane(), lane()]).unwrap();
        assert_eq!(two.lanes[1].as_ref().unwrap().off, 295);
        assert!(super::super::slots::Slots::plan(&b, &[lane(), lane(), lane()]).is_err());
    }
}
