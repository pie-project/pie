//! The components a [`ProgramSpec`] is composed of, as the host drives
//! them. Each component owns its part of the host↔wafer contract: the
//! symbols it touches ([`abi`]), the route tables it writes before
//! `init`, the weights it places, the checkpoint it needs, and its own
//! host operations (the KV pool's page copies and host tier, the stage's
//! per-launch tables, the sampler's per-lane control).
//!
//! The driver goes through the composition ([`ProgramSpec::components`])
//! and the components' own methods; a component matches only on its own
//! variants. The design modules ([`super::serve2d`], [`super::serve8b`],
//! [`super::serve8b_region`]) are the host halves of their CSL sources:
//! tile layouts and route-table rules the components call into.
//!
//! Adding a component (a model family's layers, a sampler) means a
//! variant or a type here: its params in [`super::program`], its symbols in
//! [`abi`], its routes and placement in [`Component`], and a golden
//! scenario (`golden.rs`).

use super::ckpt::Ckpt;
use super::device::Device;
use super::paged::{self, HostPool, SYMBOLS as SYM, Shape};
use super::program::{HeadKind, Kv, Layers, PageCopy, ProgramSpec, Sampler, Stage};
use super::serve2d::{BatchLane, VmTurns};
use super::serve8b::Kernel;
use super::wafer::rect;
use crate::sdk::{DataType, Order};

type R<T> = std::result::Result<T, String>;

/// What a load places: the checkpoint and RoPE's theta.
pub struct Weights<'a> {
    pub ckpt: &'a Ckpt,
    pub rope_theta: f64,
}

/// One part of a wafer program, as the host drives it.
pub trait Component {
    /// Its name, for errors and the ABI table.
    fn name(&self) -> &'static str;

    /// The exported symbols the host touches on it.
    fn symbols(&self, spec: &ProgramSpec) -> Vec<abi::Symbol>;

    /// Refuses a checkpoint it cannot place.
    fn validate(&self, _spec: &ProgramSpec, _ckpt: &Ckpt) -> R<()> {
        Ok(())
    }

    /// Before `init`: the route tables it needs.
    fn routes(&self, _spec: &ProgramSpec, _dev: &mut dyn Device) -> R<()> {
        Ok(())
    }

    /// Its weights (and zeroed state).
    fn place(&self, _spec: &ProgramSpec, _w: &Weights, _dev: &mut dyn Device) -> R<()> {
        Ok(())
    }
}

impl ProgramSpec {
    /// The program's components, in the order the host loads them (routes,
    /// then placement): the layers, the head, the KV store, the sampler.
    pub fn components(&self) -> [&dyn Component; 4] {
        [&self.layers, &self.head, &self.kv, &self.sampler]
    }

    /// Every symbol the host touches on the program, by component.
    pub fn abi(&self) -> Vec<(&'static str, abi::Symbol)> {
        let mut out: Vec<(&'static str, abi::Symbol)> = Vec::new();
        for c in self.components() {
            for s in c.symbols(self) {
                out.push((c.name(), s));
            }
        }
        out.extend(abi::launch(self).into_iter().map(|s| ("launch", s)));
        out
    }

    /// `compute`'s arguments: the passes, and the second word (a ring's
    /// micro-batches to bootstrap, bit 16: the head defers the VM's tokens
    /// on an all-VM launch of two micro-batches or more; a batch build
    /// ignores it).
    #[must_use]
    pub fn compute_args(&self, steps: usize, micro_batches: usize, all_vm: bool) -> [u32; 2] {
        let mut second = match self.stage {
            Stage::Ring { .. } => micro_batches as u32,
            Stage::Single => 0,
        };
        if self.head.defer
            && matches!(self.stage, Stage::Ring { .. })
            && all_vm
            && micro_batches >= 2
        {
            second |= 1 << 16;
        }
        [steps as u32, second]
    }
}

// ------------------------------------------------------------------ layers

impl Layers {
    /// The pool's units in layer order: (corner, kernel, layers).
    pub fn units_of(&self, spec: &ProgramSpec) -> Vec<(usize, usize, Kernel, usize)> {
        match self {
            Layers::Units { units, .. } => units
                .units(spec)
                .into_iter()
                .map(|v| (v.x0, v.y0, v.k, v.layers.len()))
                .collect(),
            Layers::Grids { grid, .. } | Layers::Strips { grid, .. } => (0..spec.layers())
                .map(|l| {
                    let (x, y) = spec.origin(l);
                    (x, y, *grid, 1)
                })
                .collect(),
        }
    }

    /// Bytes of one KV cell: f32 on the qwen2d grids, bf16 on paged8b's.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        match self {
            Layers::Grids { .. } => 4,
            Layers::Strips { .. } | Layers::Units { .. } => 2,
        }
    }
}

impl Component for Layers {
    fn name(&self) -> &'static str {
        match self {
            Layers::Grids { .. } => "qwen2d grids",
            Layers::Strips { .. } => "paged8b strips",
            Layers::Units { .. } => "paged8b units",
        }
    }

    fn symbols(&self, spec: &ProgramSpec) -> Vec<abi::Symbol> {
        use abi::Symbol as S;
        let f32s = |names: &[&'static str]| names.iter().map(|&n| S::var32(n)).collect::<Vec<_>>();
        match self {
            // qwen2d's one-sequence grids, and qwen2d_paged's.
            Layers::Grids { .. } => match spec.kv {
                Kv::Sequence => f32s(&[
                    "wqkv", "qkn_w", "inv_freq", "wo", "wgu", "wd", "gamma1", "gamma2",
                ]),
                Kv::Paged { .. } => f32s(&[
                    "wqa", "wqv", "wv", "qkn_w", "rot_c", "rot_s", "wo", "wgu", "wd", "gamma1",
                    "gamma2",
                ]),
            },
            // paged8b's bf16 tiles and f32 vectors, and the handoff tables.
            Layers::Strips { .. } | Layers::Units { .. } => {
                let mut out: Vec<S> = ["wqa", "wqv", "wv", "wo", "wgu", "wd", "rtab"]
                    .into_iter()
                    .map(S::var16)
                    .collect();
                out.extend(f32s(&["qkn_w", "rot_c", "rot_s", "gamma1", "gamma2"]));
                if let Layers::Units { units, .. } = self
                    && units.hroute
                {
                    out.extend(f32s(&["rbits", "rcfg"]));
                }
                out
            }
        }
    }

    fn routes(&self, spec: &ProgramSpec, dev: &mut dyn Device) -> R<()> {
        match self {
            Layers::Grids { .. } => Ok(()),
            Layers::Strips { .. } => super::serve8b::put_routes(spec, dev),
            Layers::Units { units, .. } => {
                // HROUTE: the program's route encoding, read first.
                let rbits = if units.hroute {
                    let at = rect(units.rc * units.rpx, 0, 1, 1);
                    Some(
                        dev.get("rbits", at, 10)
                            .map_err(|e| format!("reading rbits: {e}"))?,
                    )
                } else {
                    None
                };
                super::serve8b_region::put_routes(spec, units, dev, rbits.as_deref())
            }
        }
    }

    fn place(&self, spec: &ProgramSpec, w: &Weights, dev: &mut dyn Device) -> R<()> {
        match self {
            Layers::Grids { .. } => super::serve2d::place_layers(spec, w.ckpt, w.rope_theta, dev),
            Layers::Strips { .. } => super::serve8b::place_layers(spec, w.ckpt, w.rope_theta, dev),
            Layers::Units { units, .. } => {
                super::serve8b_region::place_layers(spec, units, w.ckpt, w.rope_theta, dev)
            }
        }
    }
}

// -------------------------------------------------------------------- head

impl Component for super::program::Head {
    fn name(&self) -> &'static str {
        match self.kind {
            HeadKind::Tied => "head2d",
            HeadKind::Untied => "head2d_v10",
        }
    }

    fn symbols(&self, spec: &ProgramSpec) -> Vec<abi::Symbol> {
        use abi::Symbol as S;
        let mut out = vec![
            S::var32(SYM.in_tokens),
            S::var32(SYM.out_tokens),
            S::var32(SYM.yscale),
            S::var32("wt"),
            S::var32("gam"),
        ];
        // The logits rows: one sequence's `y`, a batch's lanes' `hlogits`.
        out.push(match spec.kv {
            Kv::Sequence => S::var32(SYM.logits),
            Kv::Paged { .. } => S::var32(SYM.logit_lanes),
        });
        if self.kind == HeadKind::Untied {
            // The embedding strip's table.
            out.push(S::var32("etab"));
        }
        out
    }

    /// The tied head computes the logits from the embedding; the untied
    /// one places `lm_head` (and the embedding apart).
    fn validate(&self, _spec: &ProgramSpec, ckpt: &Ckpt) -> R<()> {
        match self.kind {
            HeadKind::Untied if !ckpt.has("head") => Err(
                "the checkpoint has no `head`; serve8b serves untied models (a tied one would \
                 need the embedding as the head)"
                    .into(),
            ),
            HeadKind::Tied if !ckpt.tied_head()? => Err(
                "an untied LM head (`head` apart from `embed`); this wafer program computes \
                 the logits from the embedding (serve8b places a head)"
                    .into(),
            ),
            _ => Ok(()),
        }
    }

    fn place(&self, spec: &ProgramSpec, w: &Weights, dev: &mut dyn Device) -> R<()> {
        match self.kind {
            HeadKind::Tied => super::serve2d::place_head(spec, w.ckpt, dev),
            HeadKind::Untied => {
                super::serve8b::head(spec, w.ckpt, dev)?;
                super::serve8b::embed(spec, w.ckpt, dev)
            }
        }
    }
}

// ---------------------------------------------------------------------- kv

impl Component for Kv {
    fn name(&self) -> &'static str {
        match self {
            Kv::Sequence => "sequence kv",
            Kv::Paged { .. } => "paged kv",
        }
    }

    fn symbols(&self, spec: &ProgramSpec) -> Vec<abi::Symbol> {
        use abi::Symbol as S;
        let Kv::Paged { copy, staging, .. } = *self else {
            return vec![S::var32(SYM.kc), S::var32(SYM.vc)];
        };
        // The pool's cells at the KV's width; the staging slots as the
        // design packs them (f32 or bf16 pairs in 32-bit words, bf16 one a
        // word).
        let w = if spec.layers.kv_bytes() == 2 {
            S::var16
        } else {
            S::var32
        };
        let mut out = vec![w(SYM.kc), w(SYM.vc)];
        match (copy, &spec.layers) {
            (PageCopy::Kernels, Layers::Grids { .. }) => {
                out.extend([S::func(SYM.copy_page, 2), S::func(SYM.copy_move, 2)]);
            }
            (PageCopy::Kernels, _) => out.push(S::func("copy_kv", 3)),
            (PageCopy::Host, _) => {}
        }
        if staging > 0 {
            let slot = match spec.design.kv_staging().0 {
                "bf16" => S::var16,
                _ => S::var32,
            };
            out.extend([
                slot("kstage"),
                slot("vstage"),
                S::func("page_stage", 2),
                S::func("page_unstage", 2),
            ]);
        }
        out
    }
}

impl Kv {
    fn paged(&self) -> R<(Shape, PageCopy, usize)> {
        match *self {
            Kv::Paged {
                shape,
                copy,
                staging,
            } => Ok((shape, copy, staging)),
            Kv::Sequence => Err("page copies on a one-sequence build".into()),
        }
    }

    /// The f32 grids' pool as the host reads and writes it: every layer's
    /// grid.
    fn host_pool(spec: &ProgramSpec, shape: Shape) -> HostPool {
        HostPool {
            grids: (0..spec.layers()).map(|l| spec.origin(l)).collect(),
            cols: spec.px(),
            shape,
            cells: spec.dc() * shape.page_size,
            data_type: if spec.layers.kv_bytes() == 2 {
                DataType::Bits16
            } else {
                DataType::Bits32
            },
        }
    }

    /// `copy_kv`'s whole pages on every layer's pool: the copy kernels, or
    /// the host's read-patch-write of each unit's rows.
    pub fn copy_pages(
        &self,
        spec: &ProgramSpec,
        dev: &mut dyn Device,
        pairs: &[(u32, u32)],
    ) -> R<()> {
        let (shape, copy, _) = self.paged()?;
        match (&spec.layers, copy) {
            (Layers::Units { units, .. }, _) => {
                super::serve8b_region::copy_pages(spec, units, dev, pairs)
            }
            (Layers::Strips { .. }, PageCopy::Kernels) => {
                super::serve8b::copy_pages(spec, dev, pairs)
            }
            (Layers::Grids { .. }, PageCopy::Kernels) => paged::copy_pages(dev, &SYM, pairs),
            (_, PageCopy::Host) => paged::copy_pages_host(dev, &Kv::host_pool(spec, shape), pairs),
        }
    }

    /// Words of one page's host copy (the host tier), if the host can keep
    /// this build's pages.
    #[must_use]
    pub fn page_words(&self, spec: &ProgramSpec) -> Option<usize> {
        let (shape, _, _) = self.paged().ok()?;
        match spec.layers {
            Layers::Grids { .. } => Some(paged::page_words(&Kv::host_pool(spec, shape))),
            _ => super::serve8b::page_words(spec).ok(),
        }
    }

    /// Bytes of one page's host copy: its cells at the KV's own width.
    #[must_use]
    pub fn page_bytes(&self, spec: &ProgramSpec) -> Option<usize> {
        self.page_words(spec).map(|n| n * spec.layers.kv_bytes())
    }

    /// `pages`' cells off the device, through the staging slots on a build
    /// with them (a round of pages a copy), else whole rows a page at a
    /// time.
    pub fn read_pages(
        &self,
        spec: &ProgramSpec,
        dev: &mut dyn Device,
        pages: &[u32],
    ) -> R<Vec<Vec<u32>>> {
        let (shape, _, staging) = self.paged()?;
        match spec.layers {
            Layers::Grids { .. } => {
                let pool = Kv::host_pool(spec, shape);
                if staging > 0 {
                    paged::read_pages_staged(dev, &pool, staging, pages)
                } else {
                    pages
                        .iter()
                        .map(|&p| paged::read_page_host(dev, &pool, p))
                        .collect()
                }
            }
            _ => super::serve8b::page_out_many(spec, dev, pages),
        }
    }

    /// [`Kv::read_pages`]' reverse: host copies into their device pages.
    pub fn write_pages(
        &self,
        spec: &ProgramSpec,
        dev: &mut dyn Device,
        pages: &[(u32, Vec<u32>)],
    ) -> R<()> {
        let (shape, _, staging) = self.paged()?;
        match spec.layers {
            Layers::Grids { .. } => {
                let pool = Kv::host_pool(spec, shape);
                if staging > 0 {
                    paged::write_pages_staged(dev, &pool, staging, pages)
                } else {
                    pages
                        .iter()
                        .try_for_each(|(p, w)| paged::write_page_host(dev, &pool, *p, w))
                }
            }
            _ => super::serve8b::page_in_many(spec, dev, pages),
        }
    }
}

// ------------------------------------------------------------------- stage

/// A micro-batch's KV table for a unit of so many pool rows.
pub type TableRows<'a> = dyn Fn(&[BatchLane], usize) -> R<Vec<u32>> + 'a;

impl Stage {
    /// A launch's KV tables (`table_rows(batch, rows)`: a micro-batch's
    /// table for a unit of `rows` pool rows): one batch's on every grid
    /// row (`tab`, a unit's own rows on serve8b_region); a ring's per
    /// micro-batch, at the head's feeders (serve2d_pipe's `mtab`) or on every
    /// unit (serve8b_region_pipe's `mtab`).
    pub fn write_tables(
        &self,
        spec: &ProgramSpec,
        dev: &mut dyn Device,
        mbs: &[&[BatchLane]],
        table_rows: &TableRows,
    ) -> R<()> {
        let shape = spec
            .paged()
            .ok_or("a batched launch on a one-sequence build")?;
        let e = |e: crate::sdk::Error| e.to_string();
        match (self, &spec.layers) {
            (Stage::Ring { .. }, Layers::Units { .. }) => {
                let of = |rows: usize| {
                    mbs.iter()
                        .map(|mb| table_rows(mb, rows))
                        .collect::<R<Vec<_>>>()
                };
                super::serve8b_region::put_pipe_tables(spec, dev, &of)
            }
            (Stage::Ring { mmax }, _) => {
                let mtab = super::serve2d::pipe_tables(
                    spec,
                    &mbs.iter()
                        .map(|mb| table_rows(mb, shape.rows))
                        .collect::<R<Vec<_>>>()?,
                );
                let n = mmax * shape.row_words();
                super::serve2d::head_h2d(spec, dev, SYM.ring_table, &mtab, spec.feeder(), 1, n)
            }
            (Stage::Single, layers) => {
                // Every fabric row of the grids: its grid row's table row (a
                // serve8b_region unit's: its own rows').
                let lanes: Vec<BatchLane> = mbs.iter().flat_map(|mb| mb.iter().cloned()).collect();
                let units = matches!(layers, Layers::Units { .. });
                let mut tabs: Vec<(usize, Vec<u32>)> = Vec::new();
                for grids in spec.grid_areas() {
                    let rows = if units { grids.h as usize } else { spec.py() };
                    if !tabs.iter().any(|t| t.0 == rows) {
                        tabs.push((rows, table_rows(&lanes, rows)?));
                    }
                    let tab = &tabs.iter().find(|t| t.0 == rows).expect("built above").1;
                    let all: Vec<u32> = (0..grids.h as usize / rows)
                        .flat_map(|_| tab.iter().copied())
                        .collect();
                    dev.memcpy_h2d_rowbcast(
                        SYM.table,
                        &all,
                        grids,
                        shape.row_words(),
                        DataType::Bits32,
                    )
                    .map_err(e)?;
                }
                Ok(())
            }
        }
    }

    /// After a launch: serve8b_region_pipe's head checks that every
    /// micro-batch's x came with its own index word.
    pub fn check_launch(&self, spec: &ProgramSpec, dev: &mut dyn Device) -> R<()> {
        match (self, &spec.layers) {
            (Stage::Ring { .. }, Layers::Units { .. }) => {
                super::serve8b_region::check_pipe(spec, dev)
            }
            _ => Ok(()),
        }
    }
}

// ----------------------------------------------------------------- sampler

impl Component for Sampler {
    fn name(&self) -> &'static str {
        match self {
            Sampler::Argmax => "argmax",
            Sampler::SetA { .. } => "eta VM",
            Sampler::SetsAb { .. } => "eta VM sets A and B",
        }
    }

    fn symbols(&self, spec: &ProgramSpec) -> Vec<abi::Symbol> {
        use abi::Symbol as S;
        if matches!(self, Sampler::Argmax) {
            return Vec::new();
        }
        let mut out: Vec<S> = [
            "gmode",
            "gops",
            "gpool",
            "gvalid",
            "gtok_home",
            "gostep",
            "gwide",
            "gring",
            "gout",
        ]
        .into_iter()
        .map(S::var32)
        .collect();
        let sets = [spec.vm(), spec.vm_b()];
        // A lane's bank (a ring's lanes, or per-lane programs), and the
        // per-lane programs' arena descriptors.
        if spec.mmax().is_some() || sets.iter().flatten().any(|v| v.lprog) {
            out.push(S::var32("gbank"));
        }
        if sets.iter().flatten().any(|v| v.lprog) {
            out.extend(["glane", "gbdesc"].map(S::var32));
        }
        if sets.iter().flatten().any(|v| v.sin > 0) {
            out.extend(["gsdesc", "gsin"].map(S::var32));
        }
        if spec.mmax().is_some() {
            out.push(S::var32(SYM.vm_lanes));
        }
        out
    }
}

impl Sampler {
    /// A ring launch's per-virtual-lane VM control (`vmlane`: gmode_l, the
    /// lane's forced prefix, guest4hz's turn offset, set B's lane set), on
    /// the head's root column, written only when it changes
    /// (`written`: the control last written).
    pub fn write_turns(
        &self,
        spec: &ProgramSpec,
        dev: &mut dyn Device,
        mbs: &[&[BatchLane]],
        vm: Option<&VmTurns>,
        written: &mut Vec<u32>,
    ) -> R<()> {
        let (Some(abi), Stage::Ring { mmax }, Layers::Grids { .. }) =
            (spec.vm_abi(), spec.stage, &spec.layers)
        else {
            return Ok(());
        };
        let b_n = spec.paged().map_or(1, |s| s.lanes);
        let w = abi.lane_words;
        let mut control = vec![0u32; mmax * b_n * w];
        for (m, mb) in mbs.iter().enumerate() {
            for b in 0..b_n {
                let lane = mb.get(b).unwrap_or(&mb[0]);
                let v = m * b_n + b;
                control[v * w] = vm.and_then(|t| t.sampled.get(v).copied()).unwrap_or(0);
                control[v * w + 1] = vm
                    .and_then(|t| t.nforce)
                    .and_then(|f| f.get(v).copied())
                    .unwrap_or(lane.forced.len() as u32);
                if w >= 4 {
                    control[v * w + 3] = vm.and_then(|t| t.sets.get(v).copied()).unwrap_or(0);
                }
            }
        }
        // Only the root column reads it (its region PEs put each row's
        // ctl, its feeder decides): one column, not the head's width.
        if *written != control {
            let all: Vec<u32> = (0..=spec.hh())
                .flat_map(|_| control.iter().copied())
                .collect();
            super::serve2d::check_head_columns(spec, SYM.vm_lanes, spec.r2x()..spec.r2x() + 1)?;
            let col = rect(spec.head_x() + spec.r2x(), 0, 1, spec.hh() + 1);
            dev.memcpy_h2d(
                SYM.vm_lanes,
                &all,
                col,
                control.len(),
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(|e| e.to_string())?;
            *written = control;
        }
        Ok(())
    }
}

/// The host↔wafer symbols, declared once: each a name, a kind (a variable
/// copied at 16 or 32 bits, or a function and its arity). The components
/// list theirs ([`Component::symbols`]); a test holds them against the
/// generated sources' exports and against every call the goldens record.
pub mod abi {
    use super::{ProgramSpec, SYM, Stage};

    /// What a symbol is to the host.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Kind {
        /// A variable the host copies, a word an element at 32 bits or one
        /// a word's low half at 16.
        Var { bits: u32 },
        /// A function the host launches with `args` words.
        Func { args: usize },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Symbol {
        pub name: &'static str,
        pub kind: Kind,
    }

    impl Symbol {
        #[must_use]
        pub fn var32(name: &'static str) -> Symbol {
            Symbol {
                name,
                kind: Kind::Var { bits: 32 },
            }
        }

        #[must_use]
        pub fn var16(name: &'static str) -> Symbol {
            Symbol {
                name,
                kind: Kind::Var { bits: 16 },
            }
        }

        #[must_use]
        pub fn func(name: &'static str, args: usize) -> Symbol {
            Symbol {
                name,
                kind: Kind::Func { args },
            }
        }
    }

    /// The launch: `init` (every PE takes its place), `compute(steps,
    /// second)` ([`ProgramSpec::compute_args`]), the per-launch tables.
    #[must_use]
    pub fn launch(spec: &ProgramSpec) -> Vec<Symbol> {
        let mut out = vec![Symbol::func("init", 0), Symbol::func(SYM.compute, 2)];
        if spec.paged().is_some() {
            out.push(Symbol::var32(match spec.stage {
                Stage::Ring { .. } => SYM.ring_table,
                Stage::Single => SYM.table,
            }));
        }
        if let (Stage::Ring { .. }, super::Layers::Units { .. }) = (spec.stage, &spec.layers) {
            out.push(Symbol::var32("perr"));
        }
        out
    }
}
