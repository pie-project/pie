//! The program layer: [`ProgramSpec`], one typed account of a built wafer
//! program. The planner's parameters become one ([`ProgramSpec::from_params`]),
//! the codegen's `K:V` parameters are rendered from it
//! ([`ProgramSpec::to_params`], byte for byte what was given), the manifest
//! stores it, and the driver reads every shape, symbol layout and launch
//! convention from it: nothing downstream re-reads a parameter by name.
//!
//! A spec is a composition of explicit components, not one record of
//! optional fields:
//!
//! - [`Model`]: the dims the program was built for (hidden, query heads a
//!   kv head, vocabulary) and its launch's iterations (`CAP`);
//! - [`Layers`]: the decoder layers' pipeline: f32 qwen2d grids (serve2d),
//!   bf16 grids in strips (serve8b) or grids and several-layer regions
//!   (serve8b_region);
//! - [`Stage`]: one batch a launch, or a ring of `MMAX` micro-batches;
//! - [`Head`]: the head's rows, columns and logits lanes;
//! - [`Kv`]: one sequence's cache, or a paged pool with its page-copy and
//!   staging kernels;
//! - [`Sampler`]: none, the head's eta VM (set A), or two sets (A and B, by
//!   turns or concurrently), with the VM image's host ABI ([`VmAbi`]).
//!
//! Parameters the host never reads (kernel tuning: `PASSES`, `MODE`,
//! `PROF`, ...) ride along unread as the spec's `knobs`. A spec renders
//! its parameters in the order it was given them (the layout's declaration
//! order for a planned build). A typed value given in another spelling
//! than its component's (`GY:0` for the default, a flag of 2) renders as
//! given while the component holds what was read from it; the typed
//! values are the only source of every parameter they cover.

use std::collections::BTreeMap;

use super::codegen::Design;
use super::guest4::wafer::{HeadGeometry, VmLayout, VmLimits};
use super::paged::Shape;
use super::serve8b::Kernel;
use super::serve8b_region::{Units, VmBlock};
use super::wafer::rect;
use crate::sdk::Rect;

type R<T> = std::result::Result<T, String>;

/// Compile parameters as (name, value), in order.
pub type Params = Vec<(String, String)>;

/// `K:V,...` as params.
#[must_use]
pub fn parse_params(s: &str) -> Params {
    s.split(',')
        .filter_map(|kv| kv.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// The dims a program was built for, and its launch's iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    /// `HR`: the hidden size.
    pub hidden: usize,
    /// `Qh`: query heads a kv head.
    pub qh: usize,
    pub vocab: usize,
    /// `CAP`: head iterations a launch (a launch runs at most `CAP - 1`
    /// passes).
    pub cap: usize,
}

/// The decoder layers' pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layers {
    /// serve2d / serve2d_pipe: one f32 qwen2d grid a layer, `gw` across and
    /// `gh` down, serpentine; `tc` KV positions a PE row on a one-sequence
    /// build.
    Grids {
        grid: Kernel,
        gw: usize,
        gh: usize,
        tc: usize,
    },
    /// serve8b: bf16 grids in strips; strips past the first `full` skip
    /// their westmost slot.
    Strips {
        grid: Kernel,
        gw: usize,
        gh: usize,
        full: usize,
    },
    /// serve8b_region: grids and several-layer regions.
    Units { grid: Kernel, units: Units },
}

impl Layers {
    /// The (one-layer) grid's kernel shape.
    #[must_use]
    pub fn grid(&self) -> Kernel {
        match *self {
            Layers::Grids { grid, .. }
            | Layers::Strips { grid, .. }
            | Layers::Units { grid, .. } => grid,
        }
    }
}

/// How many micro-batches a launch carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// One batch of `B` lanes (or one sequence).
    Single,
    /// `mmax` micro-batches of `B` go round the ring of units and the head.
    Ring { mmax: usize },
}

/// Which head the program carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    /// head2d over f32 tiles of the embedding (a tied head), `PY` chunk
    /// columns.
    Tied,
    /// head2d_v10 over bf16 tiles of `lm_head` (an untied head), `PY / 2`
    /// chunk columns, and the embedding strip east of it.
    Untied,
}

/// The head: `hh` region rows of `vr` vocabulary rows (its feeder row
/// between them), its root's group width, the VM columns it steps over,
/// and the logits rows it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    pub kind: HeadKind,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    /// `VMC`: set A's VM columns (or their stand-ins) after the root column.
    pub vm_cols: usize,
    /// `VMB`: a second VM set's columns after them.
    pub vmb_cols: usize,
    /// `HLM`: the most virtual lanes whose logits rows the head keeps.
    pub hlm: Option<usize>,
    /// `HLBN`: the most lanes a launch samples on the host.
    pub hlbn: Option<usize>,
    /// `DEFER`: the head defers the VM's tokens on an all-VM ring launch.
    pub defer: bool,
}

/// How the program copies a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageCopy {
    /// The program's copy kernels (`KVCP` 1).
    Kernels,
    /// The host reads, patches and writes the pool rows (`KVCP` 0).
    Host,
}

/// The KV store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv {
    /// One sequence's cache (serve2d `GRID` 0).
    Sequence,
    /// PIE's pages: the pool's shape, how a page is copied, and the staging
    /// slots a PE (`KVST`; 0: the host reads whole rows).
    Paged {
        shape: Shape,
        copy: PageCopy,
        staging: usize,
    },
}

/// The VM image's host ABI: `gring`'s words and a virtual lane's control
/// words (`vmlane`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmAbi {
    /// `[depth, width]` (guest4hm) or `[depth, width, lanes]` (guest4hz).
    pub ring_words: usize,
    /// 2 (gmode, forced prefix); 4 on serve2d_pipe's guest4hz head (and
    /// the offset to the lane's turn's last row, and the lane's set).
    pub lane_words: usize,
}

/// Who samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// The head's argmax (the host samples from the logits rows).
    Argmax,
    /// The head's eta VM, set A.
    SetA { a: VmLayout, abi: VmAbi },
    /// Two VM sets, each its own image: by turns, or `concurrent`
    /// (`VMCONC`: both at once, each told its rows a launch).
    SetsAb {
        a: VmLayout,
        b: VmLayout,
        concurrent: bool,
        abi: VmAbi,
    },
}

/// A parameter a build states ([`ProgramSpec::compose`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stated {
    /// A parameter a component holds: its value is the component's.
    Typed(&'static str),
    /// A kernel knob no component reads, and its value.
    Knob(&'static str, usize),
    /// A parameter a component holds where the composition has it (the VM
    /// image's limits, a pool's shape), else a knob.
    Either(&'static str, usize),
}

/// One built wafer program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramSpec {
    pub design: Design,
    pub model: Model,
    pub layers: Layers,
    pub stage: Stage,
    pub head: Head,
    pub kv: Kv,
    pub sampler: Sampler,
    /// Parameters no component reads (kernel tuning), as given.
    knobs: BTreeMap<String, String>,
    /// A typed parameter given in another spelling than its component
    /// renders (a 0 meaning the default, a flag other than 0/1, spaces):
    /// the text as given and the value the component read from it. It
    /// renders as given while the component still holds that value.
    raw: BTreeMap<String, (String, usize)>,
    /// The parameters' order.
    order: Vec<String>,
}

/// Params by name as numbers.
struct Get<'a>(&'a [(String, String)]);

impl Get<'_> {
    fn opt(&self, k: &str) -> Option<usize> {
        self.0
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.trim().parse().ok())
    }

    fn req(&self, k: &str) -> R<usize> {
        self.opt(k).ok_or_else(|| format!("no param {k}"))
    }
}

impl ProgramSpec {
    /// The program `params` describe for `design`, read as the engine always
    /// read them (a 0 for `GY`/`RGY` meaning the default, a flag true only
    /// as the old reader took it). Refuses params that make no layout. The
    /// spec renders them back as given ([`ProgramSpec::to_params`]).
    pub fn from_params(design: Design, params: &[(String, String)]) -> R<ProgramSpec> {
        let get = Get(params);
        let mut spec = match design {
            Design::Serve2d | Design::Serve2dPipe => serve2d(design, &get)?,
            Design::Serve8b => serve8b(&get)?,
            Design::Serve8bRegion | Design::Serve8bRegionPipe => serve8b_region(design, &get)?,
        };
        spec.order = params.iter().map(|(k, _)| k.clone()).collect();
        spec.knobs = params
            .iter()
            .filter(|(k, _)| spec.param(k).is_none())
            .cloned()
            .collect();
        spec.raw = params
            .iter()
            .filter_map(|(k, v)| {
                let t = spec.param(k)?;
                (t.to_string() != *v).then(|| (k.clone(), (v.clone(), t)))
            })
            .collect();
        let back = spec.to_params();
        if let Some(((k, v), (_, w))) = params.iter().zip(&back).find(|(a, b)| a != b) {
            return Err(format!(
                "{}: param {k}:{v} renders as {w}; the program spec cannot hold it",
                design.name()
            ));
        }
        Ok(spec)
    }

    /// A program of these components, the head's argmax sampling, stating
    /// no parameters yet ([`ProgramSpec::compose`] states them).
    #[must_use]
    pub fn new(
        design: Design,
        model: Model,
        layers: Layers,
        stage: Stage,
        head: Head,
        kv: Kv,
    ) -> ProgramSpec {
        ProgramSpec {
            design,
            model,
            layers,
            stage,
            head,
            kv,
            sampler: Sampler::Argmax,
            knobs: BTreeMap::new(),
            raw: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    /// The eta VM on this spec's head: set A's image (`a`) on the head's
    /// `VMC` columns, set B's (`b`) on its `VMB` columns, by turns or
    /// `concurrent`, with the design image's ABI. The VM runs on a
    /// one-sequence build or serve2d_pipe's ring (whose lanes it
    /// time-multiplexes); elsewhere, and where the columns do not split the
    /// head's rows, the head's argmax samples.
    #[must_use]
    pub fn head_sampler(&self, a: VmLimits, b: Option<VmLimits>, concurrent: bool) -> Sampler {
        if !(self.paged().is_none() || self.mmax().is_some()) {
            return Sampler::Argmax;
        }
        let geometry = HeadGeometry {
            x0: self.head_x(),
            c: self.py(),
            gx: self.gx(),
            hh: self.hh(),
            vr: self.vr(),
            vocab: self.vocab(),
        };
        let Some(mut a) = VmLayout::on_head(geometry, self.head.vm_cols, a, self.head.vmb_cols)
        else {
            return Sampler::Argmax;
        };
        let mut b = b.and_then(|l| a.with_set_b(self.head.vmb_cols, self.vr(), l));
        // The image's ABI: guest4hm's ring is [depth, width], guest4hz's
        // [depth, width, lanes]; serve2d_pipe's head reads guest4hz's
        // vmlane as 4 words a virtual lane (`VLW`, with or without a set B:
        // gmode, forced prefix, the turn's last row, the lane's set).
        let hz = self.design.vm_image() == Some("guest4hz");
        let ring_words = if hz { 3 } else { 2 };
        let lane_words = match (hz, self.mmax()) {
            (true, Some(_)) => 4,
            _ => 2,
        };
        if lane_words >= 3 {
            // guest4hz on: glane carries a live lane's rows, on either set
            // (one image, two rectangles).
            a.lane_rows = true;
            if let Some(b) = b.as_mut() {
                b.lane_rows = true;
            }
        }
        let abi = VmAbi {
            ring_words,
            lane_words,
        };
        match b {
            Some(b) => Sampler::SetsAb {
                a,
                b,
                concurrent,
                abi,
            },
            None => Sampler::SetA { a, abi },
        }
    }

    /// The spec stating `stated` (a planned build's parameters), rendered
    /// in `declared`'s order (the layout's; a stated name it does not
    /// declare is not passed). A typed parameter's value is its
    /// component's; a knob's is its own; an `Either` is its component's
    /// where one holds it (and must agree), else a knob.
    pub fn compose(mut self, stated: &[Stated], declared: &[String]) -> R<ProgramSpec> {
        let mut names: Vec<&'static str> = Vec::new();
        for st in stated {
            let (k, knob) = match *st {
                Stated::Typed(k) => {
                    if self.param(k).is_none() {
                        return Err(format!("{}: no component holds `{k}`", self.design.name()));
                    }
                    (k, None)
                }
                Stated::Knob(k, v) => (k, Some(v)),
                Stated::Either(k, v) => match self.param(k) {
                    Some(t) if t != v => {
                        return Err(format!(
                            "{}: `{k}` is {v} as stated and {t} by its component",
                            self.design.name()
                        ));
                    }
                    Some(_) => (k, None),
                    None => (k, Some(v)),
                },
            };
            match knob {
                Some(v) => {
                    self.knobs.insert(k.to_string(), v.to_string());
                }
                None => {
                    self.knobs.remove(k);
                }
            }
            if !names.contains(&k) {
                names.push(k);
            }
        }
        self.order = declared
            .iter()
            .filter(|d| names.contains(&d.as_str()))
            .cloned()
            .collect();
        Ok(self)
    }

    /// The spec, if it reads back as itself from its rendered parameters
    /// ([`ProgramSpec::from_params`]): what a composed build must.
    pub fn read_back(self) -> R<ProgramSpec> {
        let rendered = self.to_params();
        let back = ProgramSpec::from_params(self.design, &rendered)?;
        if back != self {
            return Err(format!(
                "{}: the build's spec does not read back as itself ({rendered:?})",
                self.design.name()
            ));
        }
        Ok(self)
    }

    /// The spec of (name, value) pairs, its design inferred (tests).
    #[cfg(test)]
    pub(crate) fn of_pairs(pairs: &[(&str, usize)]) -> R<ProgramSpec> {
        let params: Params = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.to_string()))
            .collect();
        ProgramSpec::from_params(ProgramSpec::infer_design(&params)?, &params)
    }

    /// The spec of `K:V,...`, its design inferred (tests).
    #[cfg(test)]
    pub(crate) fn of_str(params: &str) -> R<ProgramSpec> {
        let params = parse_params(params);
        ProgramSpec::from_params(ProgramSpec::infer_design(&params)?, &params)
    }

    /// The design `params` (an `out.json` of a program compiled outside
    /// PIE) are a build of.
    pub fn infer_design(params: &[(String, String)]) -> R<Design> {
        let get = Get(params);
        Ok(if get.opt("RPX").is_some() {
            if get.opt("PIPE") == Some(1) {
                Design::Serve8bRegionPipe
            } else {
                Design::Serve8bRegion
            }
        } else if get.opt("GS").is_some() {
            Design::Serve8b
        } else if get.opt("HEADMODE").is_some() {
            if get.opt("MMAX").is_some() {
                Design::Serve2dPipe
            } else {
                Design::Serve2d
            }
        } else {
            return Err(
                "the params name no grid design (serve2d, serve2d_pipe, serve8b, \
                 serve8b_region, serve8b_region_pipe); strip programs (serve28*) are no \
                 longer served"
                    .into(),
            );
        })
    }

    /// The codegen parameters: every one the spec was made from, in order,
    /// each typed one rendered from its component.
    #[must_use]
    pub fn to_params(&self) -> Params {
        self.order
            .iter()
            .map(|k| {
                let v = match (self.param(k), self.raw.get(k)) {
                    (Some(t), Some((text, read))) if *read == t => text.clone(),
                    (Some(t), _) => t.to_string(),
                    (None, _) => self.knobs.get(k).cloned().unwrap_or_default(),
                };
                (k.clone(), v)
            })
            .collect()
    }

    /// The typed parameter `k`, from the component that holds it (`None`:
    /// a knob, or a parameter this design's components do not hold).
    #[must_use]
    pub fn param(&self, k: &str) -> Option<usize> {
        let m = &self.model;
        let g = self.layers.grid();
        let model = match k {
            "HR" => Some(m.hidden),
            "Qh" => Some(m.qh),
            "vocab" => Some(m.vocab),
            "CAP" => Some(m.cap),
            "PX" => Some(g.px),
            "PY" => Some(g.py),
            "Hc" => Some(g.hc),
            "Dc" => Some(g.dc),
            "Gs" => Some(g.gs),
            "Ic" => Some(g.ic),
            _ => None,
        };
        model
            .or_else(|| self.layers_param(k))
            .or_else(|| self.stage_param(k))
            .or_else(|| self.head_param(k))
            .or_else(|| self.kv_param(k))
            .or_else(|| self.sampler_param(k))
    }

    fn layers_param(&self, k: &str) -> Option<usize> {
        match (self.layers, k) {
            (Layers::Grids { gw, .. } | Layers::Strips { gw, .. }, "GW") => Some(gw),
            (Layers::Grids { gh, .. } | Layers::Strips { gh, .. }, "GH") => Some(gh),
            (Layers::Grids { tc, .. }, "Tc") => Some(tc),
            (Layers::Strips { full, .. }, "GS") => Some(full),
            (Layers::Units { units: u, .. }, _) => match k {
                "RPX" => Some(u.rpx),
                "RPY" => Some(u.rpy),
                "RHc" => Some(u.rhc),
                "RDc" => Some(u.rdc),
                "RGs" => Some(u.rgs),
                "RIc" => Some(u.ric),
                "RNL" => Some(u.rnl),
                "RC" => Some(u.rc),
                "RR" => Some(u.rr),
                "NCOL" => Some(u.ncol),
                "RTR" => Some(u.rtr),
                "GY" => Some(u.gy),
                "RGY" => Some(u.rgy),
                "HROUTE" => Some(usize::from(u.hroute)),
                "ROT8" => Some(usize::from(u.rot8)),
                "NOLAYER" => Some(usize::from(u.nolayer)),
                _ => {
                    let vm = u.vm?;
                    match k {
                        "VMX" | "VMREAL" => Some(1),
                        "max_ops" => Some(vm.max_ops),
                        "nreg" => Some(vm.nreg),
                        "pool_words" => Some(vm.pool_words),
                        "bank_words" => Some(vm.bank_words),
                        "vm_lanes" => Some(vm.lanes),
                        _ => None,
                    }
                }
            },
            _ => None,
        }
    }

    fn stage_param(&self, k: &str) -> Option<usize> {
        let region = matches!(self.layers, Layers::Units { .. });
        match (self.stage, k) {
            (Stage::Ring { mmax }, "MMAX") => Some(mmax),
            (Stage::Ring { .. }, "PIPE") if region => Some(1),
            _ => None,
        }
    }

    fn head_param(&self, k: &str) -> Option<usize> {
        let h = &self.head;
        match k {
            "HH" => Some(h.hh),
            "Vr" => Some(h.vr),
            "Gx" => Some(h.gx),
            "VMC" => Some(h.vm_cols),
            "VMB" if self.is_serve2d() => Some(h.vmb_cols),
            "HLM" => h.hlm,
            "HLBN" => h.hlbn,
            "DEFER" if self.is_serve2d() => Some(usize::from(h.defer)),
            _ => None,
        }
    }

    fn kv_param(&self, k: &str) -> Option<usize> {
        let serve2d = self.is_serve2d();
        match (self.kv, k) {
            (Kv::Sequence, "GRID") if serve2d => Some(0),
            (Kv::Paged { .. }, "GRID") if serve2d => Some(1),
            (Kv::Paged { shape, .. }, "B") => Some(shape.lanes),
            (Kv::Paged { shape, .. }, "PS") => Some(shape.page_size),
            (Kv::Paged { shape, .. }, "NB") => Some(shape.blocks),
            (Kv::Paged { shape, .. }, "E") => Some(shape.entries),
            (Kv::Paged { copy, .. }, "KVCP") => Some(usize::from(copy == PageCopy::Kernels)),
            (Kv::Paged { staging, .. }, "KVST") => Some(staging),
            _ => None,
        }
    }

    fn sampler_param(&self, k: &str) -> Option<usize> {
        let (a, b, concurrent) = match self.sampler {
            Sampler::Argmax => return None,
            Sampler::SetA { a, .. } => (a, None, false),
            Sampler::SetsAb {
                a, b, concurrent, ..
            } => (a, Some(b), concurrent),
        };
        let lo = |m: u64| (m & 0xffff_ffff) as usize;
        let hi = |m: u64| (m >> 32) as usize;
        let set = |vm: &VmLayout, k: &str| match k {
            "max_ops" => Some(vm.max_ops),
            "nreg" => Some(vm.nreg),
            "pool_words" => Some(vm.pool_words),
            "feat" => Some(vm.feat as usize),
            "opmask_lo" => Some(lo(vm.opmask)),
            "opmask_hi" => Some(hi(vm.opmask)),
            "vm_sin" => Some(vm.sin),
            "vm_lprog" => Some(usize::from(vm.lprog)),
            "bank_words" => Some(vm.bank_words),
            _ => None,
        };
        match k {
            "VM" => Some(1),
            "VMCONC" if b.is_some() => Some(usize::from(concurrent)),
            _ => match k.strip_prefix("b_") {
                Some(kb) => b.and_then(|b| set(&b, kb)),
                None => set(&a, k),
            },
        }
    }

    fn is_serve2d(&self) -> bool {
        matches!(self.design, Design::Serve2d | Design::Serve2dPipe)
    }

    // ------------------------------------------------------- the geometry

    pub fn px(&self) -> usize {
        self.layers.grid().px
    }
    pub fn py(&self) -> usize {
        self.layers.grid().py
    }
    pub fn hc(&self) -> usize {
        self.layers.grid().hc
    }
    pub fn dc(&self) -> usize {
        self.layers.grid().dc
    }
    pub fn gs(&self) -> usize {
        self.layers.grid().gs
    }
    pub fn ic(&self) -> usize {
        self.layers.grid().ic
    }
    pub fn hidden(&self) -> usize {
        self.model.hidden
    }
    pub fn qh(&self) -> usize {
        self.model.qh
    }
    pub fn vocab(&self) -> usize {
        self.model.vocab
    }
    pub fn cap(&self) -> usize {
        self.model.cap
    }
    /// KV positions a PE row of a one-sequence grid.
    pub fn tc(&self) -> usize {
        match self.layers {
            Layers::Grids { tc, .. } => tc,
            _ => 0,
        }
    }
    pub fn gw(&self) -> usize {
        match self.layers {
            Layers::Grids { gw, .. } | Layers::Strips { gw, .. } => gw,
            Layers::Units { .. } => 0,
        }
    }
    pub fn gh(&self) -> usize {
        match self.layers {
            Layers::Grids { gh, .. } | Layers::Strips { gh, .. } => gh,
            Layers::Units { .. } => 0,
        }
    }
    pub fn hh(&self) -> usize {
        self.head.hh
    }
    pub fn vr(&self) -> usize {
        self.head.vr
    }
    pub fn gx(&self) -> usize {
        self.head.gx
    }
    /// Every VM column the head steps over (set A's and set B's).
    pub fn vmc(&self) -> usize {
        self.head.vm_cols + self.head.vmb_cols
    }
    pub fn hlm(&self) -> Option<usize> {
        self.head.hlm
    }
    pub fn hlbn(&self) -> Option<usize> {
        self.head.hlbn
    }
    pub fn vm_defer(&self) -> bool {
        self.head.defer
    }
    /// The batched build's pool (`None`: one sequence).
    pub fn paged(&self) -> Option<Shape> {
        match self.kv {
            Kv::Paged { shape, .. } => Some(shape),
            Kv::Sequence => None,
        }
    }
    /// A ring's micro-batches a launch (`None`: one batch).
    pub fn mmax(&self) -> Option<usize> {
        match self.stage {
            Stage::Ring { mmax } => Some(mmax),
            Stage::Single => None,
        }
    }
    /// The host copies pages (no copy kernels).
    pub fn host_copy(&self) -> bool {
        matches!(
            self.kv,
            Kv::Paged {
                copy: PageCopy::Host,
                ..
            }
        )
    }
    /// KV staging slots a PE.
    pub fn kvst(&self) -> usize {
        match self.kv {
            Kv::Paged { staging, .. } => staging,
            Kv::Sequence => 0,
        }
    }
    /// The head's VM, set A.
    pub fn vm(&self) -> Option<VmLayout> {
        match self.sampler {
            Sampler::SetA { a, .. } | Sampler::SetsAb { a, .. } => Some(a),
            Sampler::Argmax => None,
        }
    }
    /// The second VM set.
    pub fn vm_b(&self) -> Option<VmLayout> {
        match self.sampler {
            Sampler::SetsAb { b, .. } => Some(b),
            _ => None,
        }
    }
    pub fn vm_conc(&self) -> bool {
        matches!(
            self.sampler,
            Sampler::SetsAb {
                concurrent: true,
                ..
            }
        )
    }
    /// The VM image's host ABI (`None`: no VM).
    pub fn vm_abi(&self) -> Option<VmAbi> {
        match self.sampler {
            Sampler::SetA { abi, .. } | Sampler::SetsAb { abi, .. } => Some(abi),
            Sampler::Argmax => None,
        }
    }
    /// Words of a virtual lane's VM control (`vmlane`).
    pub fn vmlane_words(&self) -> usize {
        self.vm_abi().map_or(2, |a| a.lane_words)
    }
    pub fn layers(&self) -> usize {
        match self.layers {
            Layers::Strips { full, gw, gh, .. } => full * gw + (gh - full) * (gw - 1),
            Layers::Units { units, .. } => units.layers(),
            Layers::Grids { gw, gh, .. } => gw * gh,
        }
    }

    /// The head's chunk columns (C; the VM columns not counted).
    pub fn chunk_cols(&self) -> usize {
        match self.head.kind {
            HeadKind::Tied => self.py(),
            HeadKind::Untied => self.py() / 2,
        }
    }

    /// The grids' area as rectangles that skip serve8b's free slots
    /// (serve8b_region: each unit's).
    pub fn grid_areas(&self) -> Vec<Rect> {
        let (px, py) = (self.px(), self.py());
        match self.layers {
            Layers::Units { units, .. } => units
                .units(self)
                .iter()
                .map(|u| rect(u.x0, u.y0, u.k.px, u.k.py))
                .collect(),
            Layers::Strips { full, gw, gh, .. } => {
                let mut a = vec![rect(0, 0, gw * px, full * py)];
                if gh > full {
                    a.push(rect(px, full * py, (gw - 1) * px, (gh - full) * py));
                }
                a
            }
            Layers::Grids { gw, gh, .. } => vec![rect(0, 0, gw * px, gh * py)],
        }
    }

    /// The program's rectangle of PEs (width, height): the layers, the
    /// head's chunk and VM columns, serve8b's embedding strip (and
    /// serve8b_region's VM block); down, the layers or the head's rows and
    /// its feeder row.
    pub fn rectangle(&self) -> (usize, usize) {
        let (px, py, hh) = (self.px(), self.py(), self.hh());
        let head = hh + 1;
        match self.layers {
            Layers::Grids { gw, gh, .. } => (gw * px + py + self.vmc(), (gh * py).max(head)),
            Layers::Strips { gw, gh, .. } => {
                (gw * px + py / 2 + self.vmc() + py / 4, (gh * py).max(head))
            }
            Layers::Units { units: u, .. } => {
                // The band and regions (RC RPX), the grid column (PX), the
                // head and the strip, and VMX's 5-column VM block.
                let vw = if u.vm.is_some() { 5 } else { 0 };
                let lh = (py + u.rr * u.rpy).max(py * (1 + u.ncol));
                (
                    u.rc * u.rpx + px + py / 2 + self.vmc() + py / 4 + vw,
                    lh.max(head),
                )
            }
        }
    }

    /// Head dim (a kv group's `Gs` slices of `Dc`).
    pub fn head_dim(&self) -> usize {
        self.gs() * self.dc()
    }

    pub fn positions(&self) -> usize {
        match self.kv {
            // A lane's slots: E pages on every row.
            Kv::Paged { shape, .. } => shape.max_page_refs() * shape.page_size,
            Kv::Sequence => self.py() * self.tc(),
        }
    }

    /// Lanes one launch carries: B, or MMAX micro-batches of B.
    pub fn launch_lanes(&self) -> usize {
        self.paged()
            .map_or(1, |s| s.lanes * self.mmax().unwrap_or(1))
    }

    /// Lanes whose last logits rows the head keeps (`HLB`: the first
    /// micro-batch's first four).
    pub fn logit_lanes(&self) -> usize {
        self.paged().map_or(1, |s| {
            let kept = match self.mmax() {
                // serve2d_pipe: HLB = min(MMAX·B, HLM) virtual lanes (HLM 4
                // before the build had it, then only micro-batch 0's B).
                Some(m) => self.hlm().map_or(s.lanes.min(4), |h| (m * s.lanes).min(h)),
                None => s.lanes.min(4),
            };
            // A launch takes host-sampled lanes up to this; more go in the
            // next launch.
            self.hlbn().map_or(kept, |n| kept.min(n.max(1)))
        })
    }

    /// The feeder row.
    pub fn feeder(&self) -> usize {
        self.head.hh / 2
    }

    /// The head's first column.
    pub fn head_x(&self) -> usize {
        match self.layers {
            // serve8b_region: the regions' columns, then the grid column.
            Layers::Units { units, .. } => units.rc * units.rpx + self.px(),
            _ => self.gw() * self.px(),
        }
    }

    /// The root column inside the head block.
    pub(crate) fn r2x(&self) -> usize {
        let ngx = self.chunk_cols() / self.gx();
        (ngx / 2) * self.gx() + self.gx() / 2
    }

    /// The final root (fabric): `in_tokens`, `out_tokens`, the root column.
    pub fn root(&self) -> (usize, usize) {
        (self.head_x() + self.r2x(), self.feeder())
    }

    /// Layer `l`'s grid's top-left PE (serve8b_region: its unit's).
    pub fn origin(&self, l: usize) -> (usize, usize) {
        let (gw, px, py) = (self.gw(), self.px(), self.py());
        let (sy, k, sxmin) = match self.layers {
            Layers::Units { units, .. } => {
                let all = units.units(self);
                let unit = all
                    .iter()
                    .find(|v| v.layers.contains(&l))
                    .unwrap_or(&all[0]);
                return (unit.x0, unit.y0);
            }
            // serve8b: strips past the full ones skip their westmost slot.
            Layers::Strips { full, .. } if l >= full * gw => {
                let m = l - full * gw;
                (full + m / (gw - 1), m % (gw - 1), 1)
            }
            _ => (l / gw, l % gw, 0),
        };
        let sx = if sy % 2 == 0 { gw - 1 - k } else { sxmin + k };
        (sx * px, sy * py)
    }
}

/// The paged pool's shape on `rows` pool rows.
fn shape(get: &Get, rows: usize) -> R<Shape> {
    Ok(Shape {
        lanes: get.req("B")?,
        rows,
        page_size: get.req("PS")?,
        blocks: get.req("NB")?,
        entries: get.req("E")?,
    })
}

fn kernel(get: &Get) -> R<Kernel> {
    Ok(Kernel {
        px: get.req("PX")?,
        py: get.req("PY")?,
        hc: get.req("Hc")?,
        dc: get.req("Dc")?,
        gs: get.req("Gs")?,
        ic: get.req("Ic")?,
    })
}

fn model(get: &Get) -> R<Model> {
    Ok(Model {
        hidden: get.req("HR")?,
        qh: get.req("Qh")?,
        vocab: get.req("vocab")?,
        cap: get.req("CAP")?,
    })
}

fn head(get: &Get, vmb: bool) -> R<Head> {
    Ok(Head {
        kind: if vmb {
            HeadKind::Tied
        } else {
            HeadKind::Untied
        },
        hh: get.req("HH")?,
        vr: get.req("Vr")?,
        gx: get.req("Gx")?,
        vm_cols: get.req("VMC")?,
        vmb_cols: if vmb { get.opt("VMB").unwrap_or(0) } else { 0 },
        hlm: if vmb { get.opt("HLM") } else { None },
        hlbn: get.opt("HLBN"),
        defer: vmb && get.opt("DEFER").unwrap_or(0) != 0,
    })
}

fn headmode(get: &Get, what: &str) -> R<()> {
    let mode = get.req("HEADMODE")?;
    if mode != 2 {
        return Err(format!(
            "HEADMODE {mode}: the engine serves {what} with its head (HEADMODE 2)"
        ));
    }
    Ok(())
}

/// serve2d / serve2d_pipe: f32 qwen2d grids, head2d east of them.
fn serve2d(design: Design, get: &Get) -> R<ProgramSpec> {
    headmode(get, "serve2d with head2d")?;
    if get.req("W16")? != 0 || get.req("KV16")? != 0 || get.req("WF32")? != 1 {
        return Err("the engine places f32 weights and KV (W16 0, KV16 0, WF32 1)".into());
    }
    let grid = kernel(get)?;
    let host_copy = get.opt("KVCP").unwrap_or(1) == 0;
    let mut spec = ProgramSpec {
        design,
        model: model(get)?,
        layers: Layers::Grids {
            grid,
            gw: get.req("GW")?,
            gh: get.req("GH")?,
            tc: get.req("Tc")?,
        },
        stage: Stage::Single,
        head: head(get, true)?,
        kv: Kv::Sequence,
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    if get.opt("GRID").unwrap_or(0) == 1 {
        if grid.px != grid.py || spec.model.qh != 2 || spec.model.hidden != grid.py * grid.hc {
            return Err(format!(
                "a batched serve2d build wants square grids, Qh 2 and no hidden padding: {spec:?}"
            ));
        }
        spec.kv = Kv::Paged {
            shape: shape(get, grid.py)?,
            copy: if host_copy {
                PageCopy::Host
            } else {
                PageCopy::Kernels
            },
            staging: get.opt("KVST").unwrap_or(0),
        };
        if let Some(m) = get.opt("MMAX").filter(|&m| m > 0) {
            spec.stage = Stage::Ring { mmax: m };
        }
    } else if get.opt("MMAX").is_some() || host_copy {
        return Err("MMAX and KVCP 0 are batched (GRID 1) builds' params".into());
    }
    // The eta VM (`VM` 1, its image's limits among the params).
    if get.opt("VM") == Some(1) && get.opt("feat").is_some() {
        let p = |k: &str| get.opt(k);
        if let Some(a) = VmLimits::read(&p, "") {
            let b = VmLimits::read(&p, "b_").map(|mut b| {
                if get.opt("b_feat").is_none() {
                    b.feat = a.feat;
                }
                b
            });
            spec.sampler = spec.head_sampler(a, b, get.opt("VMCONC").unwrap_or(0) != 0);
        }
    }
    if spec.py() * spec.hc() < spec.hidden()
        || spec.gs() == 0
        || spec.gx() == 0
        || (spec.vmc() > 1 && spec.vm().is_none())
    {
        return Err(format!("serve2d params do not make a layout: {spec:?}"));
    }
    Ok(spec)
}

/// serve8b: bf16 grids in strips, head2d_v10 and the embedding strip.
fn serve8b(get: &Get) -> R<ProgramSpec> {
    headmode(get, "serve8b")?;
    if get.req("W16")? != 1 || get.req("KV16")? != 1 {
        return Err("serve8b keeps weights and KV in bf16 (W16 1, KV16 1)".into());
    }
    let grid = kernel(get)?;
    let full = get.req("GS")?;
    let spec = ProgramSpec {
        design: Design::Serve8b,
        model: model(get)?,
        layers: Layers::Strips {
            grid,
            gw: get.req("GW")?,
            gh: get.req("GH")?,
            full,
        },
        stage: Stage::Single,
        head: head(get, false)?,
        kv: Kv::Paged {
            shape: shape(get, grid.py)?,
            copy: if get.opt("KVCP").unwrap_or(1) == 0 {
                PageCopy::Host
            } else {
                PageCopy::Kernels
            },
            staging: get.opt("KVST").unwrap_or(0),
        },
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    let (py, c) = (grid.py, spec.chunk_cols());
    let ok = py % 4 == 0
        && grid.gs > 0
        && grid.px.is_multiple_of(grid.gs)
        && grid.dc.is_multiple_of(2)
        && spec.gx() > 0
        && c.is_multiple_of(spec.gx())
        && spec.vmc() <= 1
        && full >= 1
        && full <= spec.gh()
        && spec.gw() >= 2
        && spec.hh().is_multiple_of(2)
        && spec.vr().is_multiple_of(2)
        && py * grid.hc >= spec.hidden()
        && c * 2 * grid.hc >= spec.hidden();
    if !ok {
        return Err(format!("serve8b params do not make a layout: {spec:?}"));
    }
    Ok(spec)
}

/// serve8b_region (and its ring): grids and regions, the head and the
/// embedding strip as serve8b's.
fn serve8b_region(design: Design, get: &Get) -> R<ProgramSpec> {
    headmode(get, "serve8b_region")?;
    if get.req("W16")? != 1 || get.req("KV16")? != 1 || get.opt("KVCP").unwrap_or(0) != 0 {
        return Err("serve8b_region keeps weights and KV in bf16 and copies pages by the host (W16 1, KV16 1, KVCP 0)".into());
    }
    let grid = kernel(get)?;
    let (px, gs) = (grid.px, grid.gs);
    let (rpx, rpy, rgs, rc) = (
        get.req("RPX")?,
        get.req("RPY")?,
        get.req("RGs")?,
        get.req("RC")?,
    );
    if px == 0 || rc * rpx % px != 0 {
        return Err(format!(
            "serve8b_region: RC RPX {} is not whole grids of {px}",
            rc * rpx
        ));
    }
    let u = Units {
        rpx,
        rpy,
        rhc: get.req("RHc")?,
        rdc: get.req("RDc")?,
        rgs,
        ric: get.req("RIc")?,
        rnl: get.req("RNL")?,
        rc,
        rr: get.req("RR")?,
        ncol: get.req("NCOL")?,
        rtr: get.req("RTR")?,
        gy: get.opt("GY").filter(|&g| g > 0).unwrap_or(gs),
        rgy: get.opt("RGY").filter(|&g| g > 0).unwrap_or(rgs),
        hroute: get.opt("HROUTE").unwrap_or(0) == 1,
        rot8: get.opt("ROT8").unwrap_or(0) == 1,
        nolayer: get.opt("NOLAYER").unwrap_or(0) == 1,
        band: rc * rpx / px + 1,
        vm: match (get.opt("VMX"), get.opt("VMREAL")) {
            (Some(1), Some(1)) => Some(VmBlock {
                max_ops: get.req("max_ops")?,
                nreg: get.req("nreg")?,
                pool_words: get.req("pool_words")?,
                bank_words: get.req("bank_words")?,
                lanes: get.req("vm_lanes")?,
            }),
            (Some(1), _) => {
                return Err(
                    "VMX 1 with the stand-in VM (VMREAL 0): a check build, not served".into(),
                );
            }
            _ => None,
        },
    };
    let spec = ProgramSpec {
        design,
        model: model(get)?,
        layers: Layers::Units { grid, units: u },
        // serve8b_region_pipe (`PIPE` 1): MMAX micro-batches of B lanes a
        // launch round the units.
        stage: if get.opt("PIPE").unwrap_or(0) == 1 {
            Stage::Ring {
                mmax: get.req("MMAX")?,
            }
        } else {
            Stage::Single
        },
        head: head(get, false)?,
        kv: Kv::Paged {
            shape: shape(get, grid.py.min(rpy))?,
            copy: PageCopy::Host,
            staging: get.opt("KVST").unwrap_or(0),
        },
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    let c = spec.chunk_cols();
    let (py, hc, dc, ic, hid) = (grid.py, grid.hc, grid.dc, grid.ic, spec.hidden());
    let ok = py % 4 == 0
        && gs > 0
        && rgs > 0
        && px % gs == 0
        && rpx % rgs == 0
        && px / gs == rpx / rgs
        && rgs * u.rdc == gs * dc
        && px * ic == rpx * u.ric
        && dc.is_multiple_of(2)
        && u.rdc.is_multiple_of(2)
        && spec.gx() > 0
        && c.is_multiple_of(spec.gx())
        && spec.vmc() <= 1
        && spec.hh().is_multiple_of(2)
        && spec.vr().is_multiple_of(2)
        && py * hc >= hid
        && rpy * u.rhc >= hid
        && c * 2 * hc >= hid;
    if !ok {
        return Err(format!(
            "serve8b_region params do not make a layout: {spec:?} {u:?}"
        ));
    }
    if let Some(m) = spec.mmax() {
        // A ring with every unit busy can deadlock (the handoffs are
        // synchronous); the VM block is not on the ring.
        let n = u.units(&spec).len();
        if m == 0 || m + 1 > n || u.vm.is_some() || spec.kvst() > 0 {
            return Err(format!(
                "serve8b_region_pipe: MMAX {m} on {n} units, VMX {}, KVST {}",
                u.vm.is_some(),
                spec.kvst()
            ));
        }
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every golden scenario's params (the build presets, the module tests'
    /// and the hardware builds') come back from their spec as given.
    #[test]
    fn params_round_trip_through_the_spec() {
        for (name, design, params) in crate::spatial::golden::scenario_params() {
            let spec = ProgramSpec::from_params(design, &params).expect(name);
            assert_eq!(spec.to_params(), params, "{name}");
            assert_eq!(
                ProgramSpec::infer_design(&params).ok(),
                Some(design),
                "{name}"
            );
        }
    }

    /// The typed components are the parameters' source: a changed
    /// component renders as a changed parameter, the knobs as given.
    #[test]
    fn components_render_their_params() {
        let (_, design, params) = crate::spatial::golden::scenario_params()
            .into_iter()
            .find(|s| s.0 == "serve2d_pipe_ab")
            .expect("the scenario");
        let mut spec = ProgramSpec::from_params(design, &params).unwrap();
        let get = |p: &Params, k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        let Kv::Paged { ref mut shape, .. } = spec.kv else {
            panic!("paged")
        };
        shape.lanes = 3;
        spec.stage = Stage::Ring { mmax: 5 };
        let Sampler::SetsAb {
            ref mut concurrent,
            ref mut b,
            ..
        } = spec.sampler
        else {
            panic!("two sets")
        };
        *concurrent = true;
        b.max_ops = 77;
        let out = spec.to_params();
        assert_eq!(get(&out, "B").as_deref(), Some("3"));
        assert_eq!(get(&out, "MMAX").as_deref(), Some("5"));
        assert_eq!(get(&out, "b_max_ops").as_deref(), Some("77"));
        // VMCONC was not among the params: a spec renders only those.
        assert_eq!(get(&out, "VMCONC"), None);
        assert_eq!(get(&out, "PASSES"), get(&params, "PASSES"));
        assert_eq!(out.len(), params.len());
    }

    /// The spec's VM ABI is the generated sources': `vmlane`'s words a
    /// virtual lane (the head's `VLW`) and `gring`'s words (the VM image's
    /// array), for every golden scenario with a VM.
    #[test]
    fn the_vm_abi_is_the_generated_sources() {
        let number_after = |text: &str, at: &str| -> Option<usize> {
            let rest = &text[text.find(at)? + at.len()..];
            rest.trim_start()
                .split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse()
                .ok()
        };
        let mut seen = 0;
        for (name, design, params) in crate::spatial::golden::scenario_params() {
            let spec = ProgramSpec::from_params(design, &params).unwrap();
            let Some(abi) = spec.vm_abi() else { continue };
            seen += 1;
            let files = design.files().unwrap();
            let all: String = files.iter().map(|f| f.1.as_str()).collect();
            // gring: guest4hm's `[2]u32`, guest4hz's `[if (CONC) 4 else 3]`
            // (VMCONC's fourth word the host writes as it needs).
            let ring = if all.contains("var gring = @zeros([2]u32)") {
                2
            } else {
                number_after(&all, "var gring = @zeros([if (CONC) 4 else").expect(name)
            };
            assert_eq!(abi.ring_words, ring, "{name}: gring");
            match spec.mmax() {
                Some(_) => {
                    let vlw = number_after(&all, "const VLW: u16 =").expect(name);
                    assert!(all.contains("const VLN: u16 = MMAX * SB * VLW;"), "{name}");
                    assert_eq!(abi.lane_words, vlw, "{name}: vmlane words a lane");
                }
                // One sequence: no vmlane; the host writes none.
                None => assert!(!all.contains("const VLW"), "{name}"),
            }
        }
        assert!(seen >= 4);
    }

    /// Other spellings the engine always read (a 0 meaning the default,
    /// flags other than 0/1) are read as before and render as given; a
    /// changed component renders its new value.
    #[test]
    fn other_spellings_read_as_before_and_render_as_given() {
        let scenario = |n: &str| {
            crate::spatial::golden::scenario_params()
                .into_iter()
                .find(|s| s.0 == n)
                .expect("the scenario")
        };
        let set = |params: &Params, k: &str, v: &str| -> Params {
            let mut p = params.clone();
            match p.iter_mut().find(|e| e.0 == k) {
                Some(e) => e.1 = v.to_string(),
                None => p.push((k.to_string(), v.to_string())),
            }
            p
        };
        let (_, design, params) = scenario("serve2d_pipe_conc");
        let odd = set(&set(&params, "VMCONC", "2"), "DEFER", "3");
        let spec = ProgramSpec::from_params(design, &odd).unwrap();
        assert!(spec.vm_conc() && spec.vm_defer());
        assert_eq!(spec.to_params(), odd);
        let (_, design, params) = scenario("serve8b_region");
        let odd = set(&set(&set(&params, "RGY", "0"), "GY", "0"), "ROT8", "2");
        let mut spec = ProgramSpec::from_params(design, &odd).unwrap();
        assert_eq!(spec.to_params(), odd);
        let Layers::Units { ref mut units, .. } = spec.layers else {
            panic!("units")
        };
        assert_eq!((units.rgy, units.gy, units.rot8), (units.rgs, 8, false));
        // A changed component renders what it now holds.
        units.rgy = 4;
        let out = spec.to_params();
        assert_eq!(
            out.iter().find(|e| e.0 == "RGY").map(|e| e.1.as_str()),
            Some("4")
        );
        assert_eq!(
            out.iter().find(|e| e.0 == "ROT8").map(|e| e.1.as_str()),
            Some("2")
        );
        assert!(ProgramSpec::infer_design(&parse_params("kernel_cols:64,lps:7")).is_err());
    }
}
