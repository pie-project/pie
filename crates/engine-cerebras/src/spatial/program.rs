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
//! - [`Layers`]: the decoder layers' pipeline: f32 qwen2d grids (serve2d;
//!   on serve_q35 each grid a Gated DeltaNet or gated-attention unit by its
//!   [`Hybrid`] mix), bf16 grids in strips (serve8b) or grids and
//!   several-layer regions (serve8b_region);
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
use super::serve_gptoss::{Gpt, Slots};
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
    /// `QKN`: the units norm q and k per head (0: Llama's identity; the
    /// units' sources are generated without the norm, codegen's `Variant`).
    pub qk_norm: bool,
    /// `EPS`: the RMSNorm epsilon's f32 bits (0: the kernels' 1e-6).
    pub eps_bits: u32,
    /// `SPD` 1: the units carry the speed op cuts (`Variant::speed`).
    pub speed: bool,
    /// `LSE` 1: the units' softmax against a carried reference (`Variant::lse`).
    pub lse: bool,
    /// `OP2` 1: the second op cuts (`Variant::ops2`).
    pub ops2: bool,
    /// `RGX`: the regions' X full all-reduces' phase-1 group (0: RGs;
    /// `Variant::rgx`).
    pub rgx: usize,
    /// `QKV1` 1: one QKV matvec over fused rows (`Variant::qkv1`).
    pub qkv1: bool,
    /// `ARF`: the regions' FIFO dual-input all-reduces (`Variant::arf`; 0 off).
    pub arf: usize,
    /// `F16`: 1 an IEEE f16 build (no `--fp16-format=bf16`; the host writes
    /// f16 weights), 2 also the regions' weight matvecs accumulating in f16
    /// (`Variant::f16`). 0: bf16.
    pub f16: usize,
    /// `MEM`: the region kernel's code cuts (`Variant::mem`; 0 off).
    pub mem: usize,
    /// `OP3`: the regions' third op cuts (`Variant::op3`; 0 off).
    pub op3: usize,
    /// `GRD`: bits of the regions' levers the grids take too (`Variant::grid`; 0 off).
    pub grid: usize,
    /// `F16CL` 1: the regions' f16 copy clamped, 0515's redo window (`Variant::f16cl`).
    pub f16cl: bool,
    /// `XAC` 1 (with ARF 3, KVCP 0): the regions' XA all-reduces on their own
    /// colours (`Variant::xac`).
    pub xac: bool,
    /// `TRM` bits: the units' op cuts (`Variant::trim`; 0 off).
    pub trim: usize,
    /// `ARY`: rows a group of the ARF Y all-reduces' streamed halves (the
    /// host's route words; 0: 8).
    pub ary: usize,
    /// `HVC` 1: vertical handoffs on centred columns (the host's tables).
    pub hvc: bool,
    /// `SLW` 1 (with `TRM` 1024): the f16 SiLU's wide tail (`Variant::slw`).
    pub slw: bool,
}

impl Model {
    /// The dims, Qwen3's q/k norms and the kernels' epsilon.
    #[must_use]
    pub fn dims(hidden: usize, qh: usize, vocab: usize, cap: usize) -> Model {
        Model {
            hidden,
            qh,
            vocab,
            cap,
            qk_norm: true,
            eps_bits: 0,
            speed: false,
            lse: false,
            ops2: false,
            rgx: 0,
            qkv1: false,
            arf: 0,
            f16: 0,
            mem: 0,
            op3: 0,
            grid: 0,
            f16cl: false,
            xac: false,
            trim: 0,
            ary: 0,
            hvc: false,
            slw: false,
        }
    }

    /// The RMSNorm epsilon the units compute with.
    #[must_use]
    pub fn eps(&self) -> f32 {
        if self.eps_bits == 0 {
            super::model::KERNEL_EPS
        } else {
            f32::from_bits(self.eps_bits)
        }
    }

    /// What the units' sources are generated for.
    #[must_use]
    pub fn variant(&self) -> super::codegen::Variant {
        super::codegen::Variant {
            identity_qk_norm: !self.qk_norm,
            eps_bits: (self.eps_bits != 0).then_some(self.eps_bits),
            speed: self.speed,
            lse: self.lse,
            ops2: self.ops2,
            rgx: self.rgx > 0,
            qkv1: self.qkv1,
            arf: u16::try_from(self.arf).unwrap_or(0),
            f16: u16::try_from(self.f16).unwrap_or(0),
            mem: u16::try_from(self.mem).unwrap_or(0),
            op3: u16::try_from(self.op3).unwrap_or(0),
            grid: u16::try_from(self.grid).unwrap_or(0),
            f16cl: self.f16cl,
            xac: self.xac,
            trim: u16::try_from(self.trim).unwrap_or(0),
            slw: self.slw,
        }
    }
}

/// The decoder layers' pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layers {
    /// serve2d / serve2d_pipe: one f32 qwen2d grid a layer, `gw` across and
    /// `gh` down, serpentine; `tc` KV positions a PE row on a one-sequence
    /// build. serve_q35: each grid a Gated DeltaNet or a gated-attention
    /// unit (`hybrid`).
    Grids {
        grid: Kernel,
        gw: usize,
        gh: usize,
        tc: usize,
        hybrid: Option<Hybrid>,
        /// serve_g4: Gemma 4's layers on the grids (`grid` is a sliding
        /// layer's of the shared prefix; [`Gemma`] the rest).
        gemma: Option<Gemma>,
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
    /// serve_gptoss: per layer an attention unit over a MoE grid (`grid`,
    /// whose rows the head's chunk triples follow) in a slot.
    Gptoss { grid: Kernel, gpt: Gpt },
}

/// serve_q35's layer mix and its Gated DeltaNet unit's splits: layer `l`
/// attends where bit `l` of `kinds` is set (`KINDS`), else is a Gated
/// DeltaNet of `gk` columns a k head, `dk` k dims a column, `vh` v heads a
/// k head of `dv` dims a column, a `cw`-wide causal conv; RoPE turns `rot`
/// dims of an attention head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hybrid {
    pub kinds: u64,
    pub gk: usize,
    pub dk: usize,
    pub vh: usize,
    pub dv: usize,
    pub cw: usize,
    pub rot: usize,
}

/// serve_g4's Gemma 4 layers: `nl` grids (the last strip's missing ones are
/// fillers), layer `l` reading globally where bit `l` of `kinds` is set
/// (`dcg` dims a column, else `dcs`), the first `nown` owning their kv (the
/// shared tail after them reads its reading's source's), the MLP `ics`
/// rows a column (the tail's `ict`), `pc` PLE dims a column, the PLE
/// region's `npc` columns of `rpp` vocabulary rows a PE; `esb`/`psb` the
/// f32 bits of sqrt(hidden) and sqrt(PLE)/sqrt(2); `win` the sliding
/// layers' window (slots a query sees, its own included; 0: every slot);
/// `g1` the grids' all-reduce groups (0: `Gs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma {
    pub kinds: u64,
    pub nl: usize,
    pub nown: usize,
    pub dcs: usize,
    pub dcg: usize,
    pub ics: usize,
    pub ict: usize,
    pub pc: usize,
    pub npc: usize,
    pub rpp: usize,
    pub esb: u32,
    pub psb: u32,
    pub win: usize,
    /// The all-reduces' phase-1 group (`G1`; 0: `Gs`).
    pub g1: usize,
}

impl Gemma {
    /// Whether layer `l` reads globally.
    #[must_use]
    pub fn global(&self, l: usize) -> bool {
        self.kinds >> l & 1 == 1
    }

    /// Layer `l`'s kv dims a column.
    #[must_use]
    pub fn dc(&self, l: usize) -> usize {
        if self.global(l) { self.dcg } else { self.dcs }
    }

    /// Layer `l`'s MLP rows a column.
    #[must_use]
    pub fn ic(&self, l: usize) -> usize {
        if l < self.nown { self.ics } else { self.ict }
    }

    /// The layer whose kv layer `l` reads: itself below `nown`, else the
    /// last owning layer of its reading.
    #[must_use]
    pub fn source(&self, l: usize) -> usize {
        if l < self.nown {
            return l;
        }
        (0..self.nown)
            .rev()
            .find(|&j| self.global(j) == self.global(l))
            .unwrap_or(l)
    }

    /// The PLE row's u32 words a lane (a strip's `gw` layers, two bf16 a
    /// word).
    #[must_use]
    pub fn row_words(&self, gw: usize, p: usize) -> usize {
        gw * p * self.pc / 2
    }
}

impl Hybrid {
    /// Whether layer `l` attends (its grid holds a KV pool).
    #[must_use]
    pub fn attends(&self, l: usize) -> bool {
        self.kinds >> l & 1 == 1
    }

    /// A Gated DeltaNet column's conv channels (q, k and v) and its words of
    /// one lane's conv window and recurrent state row.
    #[must_use]
    pub fn conv_channels(&self) -> usize {
        2 * self.dk + self.vh * self.dv
    }

    #[must_use]
    pub fn window_words(&self) -> usize {
        (self.cw - 1) * self.conv_channels()
    }

    #[must_use]
    pub fn state_words(&self) -> usize {
        self.vh * self.dv
    }
}

impl Layers {
    /// The (one-layer) grid's kernel shape.
    #[must_use]
    pub fn grid(&self) -> Kernel {
        match *self {
            Layers::Grids { grid, .. }
            | Layers::Strips { grid, .. }
            | Layers::Units { grid, .. }
            | Layers::Gptoss { grid, .. } => grid,
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
    /// head2d_v10 as serve_gptoss carries it: `PY / 3` chunk columns of
    /// the MoE grid's row triples, the embedding strip of the attention
    /// unit's rows east of it.
    Triples,
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
            Design::Serve2d | Design::Serve2dPipe | Design::ServeQ35 => serve2d(design, &get)?,
            Design::Serve8b => serve8b(&get)?,
            Design::Serve8bRegion | Design::Serve8bRegionPipe => serve8b_region(design, &get)?,
            Design::ServeG4 => serve_g4(&get)?,
            Design::ServeGptoss => serve_gptoss(&get)?,
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
        Ok(if get.opt("NOWN").is_some() && get.opt("NPC").is_some() {
            Design::ServeG4
        } else if get.opt("Q35") == Some(1) {
            Design::ServeQ35
        } else if get.opt("NE").is_some() && get.opt("AY").is_some() {
            Design::ServeGptoss
        } else if get.opt("RPX").is_some() {
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

    /// A knob's text (a parameter no component holds), if the program has it.
    #[must_use]
    pub fn knob_str(&self, k: &str) -> Option<&str> {
        self.knobs.get(k).map(String::as_str)
    }

    /// The units hold u4g64 weights (`W4 1`: codegen's W4 variant of
    /// serve8b_region, placed by `spatial::u4`).
    #[must_use]
    pub fn weights_u4(&self) -> bool {
        self.knobs.get("W4").is_some_and(|v| v.trim() == "1")
    }

    /// The head's lm_head tiles and the embedding strip hold u4g64 too
    /// (`W4H 1`, on a `W4 1` build: `spatial::u4::place_head`/`place_embed`).
    #[must_use]
    pub fn head_u4(&self) -> bool {
        self.weights_u4() && self.knobs.get("W4H").is_some_and(|v| v.trim() == "1")
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
            "QKN" => Some(usize::from(m.qk_norm)),
            "EPS" => Some(m.eps_bits as usize),
            "SPD" => Some(usize::from(m.speed)),
            "LSE" => Some(usize::from(m.lse)),
            "OP2" => Some(usize::from(m.ops2)),
            "RGX" => Some(m.rgx),
            "QKV1" => Some(usize::from(m.qkv1)),
            "ARF" => Some(m.arf),
            "F16" => Some(m.f16),
            "MEM" => Some(m.mem),
            "OP3" => Some(m.op3),
            "GRD" => Some(m.grid),
            "F16CL" => Some(usize::from(m.f16cl)),
            "XAC" => Some(usize::from(m.xac)),
            "TRM" => Some(m.trim),
            "ARY" => Some(m.ary),
            "HVC" => Some(usize::from(m.hvc)),
            "SLW" => Some(usize::from(m.slw)),
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
            (
                Layers::Grids {
                    hybrid: Some(h), ..
                },
                _,
            ) => match k {
                "Q35" => Some(1),
                "KINDS" => usize::try_from(h.kinds).ok(),
                "GK" => Some(h.gk),
                "DK" => Some(h.dk),
                "VH" => Some(h.vh),
                "DV" => Some(h.dv),
                "CW" => Some(h.cw),
                "ROT" => Some(h.rot),
                _ => None,
            },
            (
                Layers::Grids {
                    gemma: Some(g),
                    grid,
                    ..
                },
                _,
            ) => match k {
                "P" => Some(grid.px),
                "DcS" => Some(g.dcs),
                "DcG" => Some(g.dcg),
                "IcS" => Some(g.ics),
                "IcT" => Some(g.ict),
                "PC" => Some(g.pc),
                "KINDS" => usize::try_from(g.kinds).ok(),
                "NL" => Some(g.nl),
                "NOWN" => Some(g.nown),
                "NPC" => Some(g.npc),
                "RPP" => Some(g.rpp),
                "ESB" => Some(g.esb as usize),
                "PSB" => Some(g.psb as usize),
                "WIN" => Some(g.win),
                "G1" => Some(g.g1),
                _ => None,
            },
            (Layers::Strips { full, .. }, "GS") => Some(full),
            (Layers::Gptoss { gpt: g, .. }, _) => match k {
                "AY" => Some(g.slots.ay),
                "AHc" => Some(g.slots.ahc),
                "MY" => Some(g.slots.my),
                "MHc" => Some(g.slots.mhc),
                "GW" => Some(g.slots.gw),
                "GH" => Some(g.slots.gh),
                "NL" => Some(g.slots.nl),
                "NE" => Some(g.ne),
                "TK" => Some(g.tk),
                "AGY" => Some(g.agy),
                "MGY" => Some(g.mgy),
                "SLW" => usize::try_from(g.slw).ok(),
                "WIN" => Some(g.win),
                "RTR" => Some(g.rtr),
                "RTI" => Some(g.rti),
                "HROUTE" => Some(usize::from(g.hroute)),
                "KV16" => Some(usize::from(g.kv16)),
                _ => None,
            },
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
                "YD" => Some(u.yd),
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
        matches!(
            self.design,
            Design::Serve2d | Design::Serve2dPipe | Design::ServeQ35
        )
    }

    /// A knob's value as a number (`None`: unset or not a number).
    #[must_use]
    pub fn knob(&self, k: &str) -> Option<u64> {
        self.knobs.get(k).and_then(|v| v.trim().parse::<u64>().ok())
    }

    /// Whether the layers' weights and the head's tiles are bf16 (`W16` 1,
    /// `WF32` 0: serve_q35 past 0.8B); else f32.
    #[must_use]
    pub fn w16(&self) -> bool {
        self.knobs.get("W16").is_some_and(|v| v.trim() == "1")
    }

    /// The program's build directory: its design's files for the model's
    /// variant and, at `W4 1`, its u4g64 units ([`Design::files_for`]);
    /// serve_q35w's for a W16 serve_q35 (serve_q35 itself stays #1412's f32
    /// program); serve_gptoss's levered units with its `MOE` knob
    /// ([`super::codegen::serve_gptoss_with`]).
    pub fn files(&self) -> Result<Vec<super::codegen::File>, String> {
        let (variant, u4) = (self.model.variant(), self.weights_u4());
        if self.design == Design::ServeQ35 && self.w16() {
            if variant != super::codegen::Variant::default() || u4 {
                return Err(format!(
                    "serve_q35w: no model variant ({variant:?}) or u4g64 (W4) units"
                ));
            }
            return super::codegen::serve_q35w();
        }
        if self.design == Design::ServeGptoss {
            // (the attention units' levers from the model, the MoE units' MOE knob)
            let moe = self
                .knob_str("MOE")
                .map_or(Ok(0), |v| v.trim().parse::<u16>())
                .map_err(|e| format!("MOE: {e}"))?;
            return super::codegen::serve_gptoss_with(variant, moe);
        }
        self.design.files_for(variant, u4)
    }

    /// serve_q35's layer mix (`None`: every layer attends).
    #[must_use]
    pub fn hybrid(&self) -> Option<Hybrid> {
        match self.layers {
            Layers::Grids { hybrid, .. } => hybrid,
            _ => None,
        }
    }

    /// Whether layer `l`'s grid attends (holds a KV pool).
    #[must_use]
    pub fn attends(&self, l: usize) -> bool {
        self.hybrid().is_none_or(|h| h.attends(l))
    }

    /// serve_g4's Gemma 4 layers.
    #[must_use]
    pub fn gemma(&self) -> Option<Gemma> {
        match self.layers {
            Layers::Grids { gemma, .. } => gemma,
            _ => None,
        }
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
            Layers::Gptoss { gpt, .. } => gpt.slots.gw,
        }
    }
    pub fn gh(&self) -> usize {
        match self.layers {
            Layers::Grids { gh, .. } | Layers::Strips { gh, .. } => gh,
            Layers::Units { .. } => 0,
            Layers::Gptoss { gpt, .. } => gpt.slots.gh,
        }
    }
    /// serve_gptoss's params beyond its MoE grid.
    pub fn gptoss(&self) -> Option<Gpt> {
        match self.layers {
            Layers::Gptoss { gpt, .. } => Some(gpt),
            _ => None,
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
            Layers::Grids { gemma: Some(g), .. } => g.nl,
            Layers::Grids { gw, gh, .. } => gw * gh,
            Layers::Gptoss { gpt, .. } => gpt.slots.nl,
        }
    }

    /// The head's chunk columns (C; the VM columns not counted).
    pub fn chunk_cols(&self) -> usize {
        match self.head.kind {
            HeadKind::Tied => self.py(),
            HeadKind::Untied => self.py() / 2,
            HeadKind::Triples => self.py() / 3,
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
            Layers::Gptoss { gpt, .. } => gpt
                .slots
                .units()
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
            Layers::Grids {
                hybrid: Some(_), ..
            } => (0..self.layers())
                .filter(|&l| self.attends(l))
                .map(|l| {
                    let (x, y) = self.origin(l);
                    rect(x, y, px, py)
                })
                .collect(),
            // serve_g4: the full strips, then the last one's layers (its
            // missing grids are fillers).
            Layers::Grids {
                gemma: Some(g), gw, ..
            } => {
                let (full, rest) = (g.nl / gw, g.nl % gw);
                let mut a = Vec::new();
                if full > 0 {
                    a.push(rect(0, 0, gw * px, full * py));
                }
                if rest > 0 {
                    let x = if full % 2 == 0 { gw - rest } else { 0 };
                    a.push(rect(x * px, full * py, rest * px, py));
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
            Layers::Grids { gw, gh, gemma, .. } => (
                gw * px + py + self.vmc() + gemma.map_or(0, |g| g.npc),
                (gh * py).max(head),
            ),
            Layers::Strips { gw, gh, .. } => {
                (gw * px + py / 2 + self.vmc() + py / 4, (gh * py).max(head))
            }
            Layers::Units { units: u, .. } => {
                // The band and regions (RC RPX), the grid column (PX), the
                // head and the strip, and VMX's 5-column VM block.
                let vw = if u.vm.is_some() { 5 } else { 0 };
                let lh = if u.yd > 0 {
                    (u.rr * u.rpy).max(u.yd + u.ncol.max(1) * py)
                } else {
                    (py + u.rr * u.rpy).max(py * (1 + u.ncol))
                };
                (
                    u.rc * u.rpx + px + py / 2 + self.vmc() + py / 4 + vw,
                    lh.max(head),
                )
            }
            // The slots, the head's chunk triples and VM columns, the strip
            // (one column an attention row).
            Layers::Gptoss { gpt, .. } => {
                let s = gpt.slots;
                (
                    s.gw * px + py / 3 + self.vmc() + s.ay,
                    (s.gh * (s.ay + s.my)).max(head),
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
            // serve_gptoss: the layer's attention unit (its pool).
            Layers::Gptoss { gpt, .. } => {
                let all = gpt.slots.units();
                let unit = all.get(2 * l).unwrap_or(&all[0]);
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
        qk_norm: get.opt("QKN").unwrap_or(1) != 0,
        eps_bits: u32::try_from(get.opt("EPS").unwrap_or(0)).map_err(|e| format!("EPS: {e}"))?,
        speed: get.opt("SPD").unwrap_or(0) != 0,
        lse: get.opt("LSE").unwrap_or(0) != 0,
        ops2: get.opt("OP2").unwrap_or(0) != 0,
        rgx: get.opt("RGX").unwrap_or(0),
        qkv1: get.opt("QKV1").unwrap_or(0) != 0,
        arf: get.opt("ARF").unwrap_or(0),
        f16: get.opt("F16").unwrap_or(0),
        mem: get.opt("MEM").unwrap_or(0),
        op3: get.opt("OP3").unwrap_or(0),
        grid: get.opt("GRD").unwrap_or(0),
        f16cl: get.opt("F16CL").unwrap_or(0) != 0,
        xac: get.opt("XAC").unwrap_or(0) != 0,
        trim: get.opt("TRM").unwrap_or(0),
        ary: get.opt("ARY").unwrap_or(0),
        hvc: get.opt("HVC").unwrap_or(0) != 0,
        slw: get.opt("SLW").unwrap_or(0) != 0,
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
    // serve_q35 also places bf16 weights with bf16 head tiles (W16 1, WF32 0).
    let w16 = design == Design::ServeQ35 && get.req("W16")? == 1 && get.req("WF32")? == 0;
    if !w16 && (get.req("W16")? != 0 || get.req("WF32")? != 1) || get.req("KV16")? != 0 {
        return Err("the engine places f32 weights and KV (W16 0, KV16 0, WF32 1; serve_q35 also W16 1, WF32 0)".into());
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
            hybrid: None,
            gemma: None,
        },
        stage: Stage::Single,
        head: head(get, true)?,
        kv: Kv::Sequence,
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    if design == Design::ServeQ35 {
        let kinds = get.req("KINDS")? as u64;
        let hybrid = Hybrid {
            kinds,
            gk: get.req("GK")?,
            dk: get.req("DK")?,
            vh: get.req("VH")?,
            dv: get.req("DV")?,
            cw: get.req("CW")?,
            rot: get.req("ROT")?,
        };
        if get.opt("GRID") != Some(1) || get.opt("MMAX").is_some() || get.opt("B") != Some(1) {
            return Err("serve_q35 is a batched (GRID 1) build of one lane (B 1), no ring".into());
        }
        if let Layers::Grids { hybrid: h, .. } = &mut spec.layers {
            *h = Some(hybrid);
        }
    }
    let qh_ok = spec.model.qh == 2 || design == Design::ServeQ35;
    if get.opt("GRID").unwrap_or(0) == 1 {
        if grid.px != grid.py || !qh_ok || spec.model.hidden != grid.py * grid.hc {
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

/// serve_g4: Gemma 4's layers as bf16 gemma4_layer grids (serpentine as
/// serve2d's), the f32 head2d over the tied embedding, the PLE region east
/// of it; a paged pool on every grid (the shared tail's its own, fed by
/// the handoff), the head's argmax.
fn serve_g4(get: &Get) -> R<ProgramSpec> {
    let p = get.req("P")?;
    let hc = get.req("Hc")?;
    let g = Gemma {
        kinds: get.req("KINDS")? as u64,
        nl: get.req("NL")?,
        nown: get.req("NOWN")?,
        dcs: get.req("DcS")?,
        dcg: get.req("DcG")?,
        ics: get.req("IcS")?,
        ict: get.req("IcT")?,
        pc: get.req("PC")?,
        npc: get.req("NPC")?,
        rpp: get.req("RPP")?,
        esb: u32::try_from(get.req("ESB")?).map_err(|e| e.to_string())?,
        psb: u32::try_from(get.req("PSB")?).map_err(|e| e.to_string())?,
        win: get.opt("WIN").unwrap_or(0),
        g1: get.opt("G1").unwrap_or(0),
    };
    let (gw, gh) = (get.req("GW")?, get.req("GH")?);
    if g.nl > gw * gh || g.nl <= (gh - 1) * gw || g.nown > g.nl {
        return Err(format!(
            "serve_g4: {} layers do not fill {gw} x {gh} grids (the last strip part)",
            g.nl
        ));
    }
    if get.opt("VM").unwrap_or(0) != 0 || get.opt("KVST").unwrap_or(0) != 0 {
        return Err("serve_g4 carries no VM and no KV staging (VM 0, KVST 0)".into());
    }
    let grid = Kernel {
        px: p,
        py: p,
        hc,
        dc: g.dcs,
        gs: get.req("Gs")?,
        ic: g.ics,
    };
    let spec = ProgramSpec {
        design: Design::ServeG4,
        model: Model::dims(p * hc, get.req("Qh")?, get.req("vocab")?, get.req("CAP")?),
        layers: Layers::Grids {
            grid,
            gw,
            gh,
            tc: 0,
            hybrid: None,
            gemma: Some(g),
        },
        stage: Stage::Single,
        head: head(get, true)?,
        kv: Kv::Paged {
            shape: shape(get, p)?,
            copy: PageCopy::Kernels,
            staging: 0,
        },
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    if spec.gs() == 0 || spec.gx() == 0 || spec.vmc() != 1 || p % spec.gs() != 0 {
        return Err(format!("serve_g4 params do not make a layout: {spec:?}"));
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
    let yd = get.opt("YD").unwrap_or(0);
    if yd > 0 {
        // The last region: in row RR - 2 (RR odd, the snake) or row 1 (the
        // U); the column's first grid within its rows.
        let (rr, ncol) = (get.req("RR")?, get.req("NCOL")?);
        let k = yd / rpy.max(1);
        let row_ok = k == 1 || (rr % 2 == 1 && k + 2 == rr);
        if rc != 2 || rr < 3 || ncol == 0 || !row_ok || yd + grid.py > (k + 1) * rpy {
            return Err(format!(
                "serve8b_region YD {yd}: lever D takes RC 2, RR >= 3, NCOL >= 1 and the column's first grid within the rows of region row 1 or (RR odd) RR - 2 (RC {rc}, RR {rr}, NCOL {ncol})"
            ));
        }
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
        band: if get.opt("YD").unwrap_or(0) > 0 {
            1
        } else {
            rc * rpx / px + 1
        },
        yd: get.opt("YD").unwrap_or(0),
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

/// serve_gptoss: attention units over MoE grids in slots, head2d_v10 on
/// the MoE rows' triples, the strip east of it; the pool on the attention
/// units (`AY` rows), its pages copied by the host.
fn serve_gptoss(get: &Get) -> R<ProgramSpec> {
    let px = get.req("PX")?;
    let (my, mhc) = (get.req("MY")?, get.req("MHc")?);
    let slots = Slots {
        px,
        ay: get.req("AY")?,
        ahc: get.req("AHc")?,
        my,
        mhc,
        gw: get.req("GW")?,
        gh: get.req("GH")?,
        nl: get.req("NL")?,
    };
    let (dc, gs, ic) = (get.req("Dc")?, get.req("Gs")?, get.req("Ic")?);
    let gpt = Gpt {
        slots,
        dc,
        gs,
        ic,
        ne: get.req("NE")?,
        tk: get.req("TK")?,
        agy: get.req("AGY")?,
        mgy: get.req("MGY")?,
        slw: get
            .0
            .iter()
            .find(|(k, _)| k == "SLW")
            .and_then(|(_, v)| v.trim().parse().ok())
            .ok_or("no param SLW")?,
        win: get.req("WIN")?,
        rtr: get.req("RTR")?,
        rti: get.req("RTI")?,
        hroute: get.opt("HROUTE").unwrap_or(0) == 1,
        // Absent: the bf16 pool the first builds had.
        kv16: get.opt("KV16").unwrap_or(1) == 1,
        qkv1: get.opt("QKV1").unwrap_or(0) == 1,
        arf: get.opt("ARF").unwrap_or(0),
    };
    let grid = Kernel {
        px,
        py: my,
        hc: mhc,
        dc,
        gs,
        ic,
    };
    let mut head = head(get, false)?;
    head.kind = HeadKind::Triples;
    let spec = ProgramSpec {
        design: Design::ServeGptoss,
        model: model(get)?,
        layers: Layers::Gptoss { grid, gpt },
        stage: Stage::Single,
        head,
        kv: Kv::Paged {
            shape: shape(get, slots.ay)?,
            copy: PageCopy::Host,
            staging: 0,
        },
        sampler: Sampler::Argmax,
        knobs: BTreeMap::new(),
        raw: BTreeMap::new(),
        order: Vec::new(),
    };
    let hid = spec.hidden();
    let c = spec.chunk_cols();
    let ok = gs > 0
        && px % gs == 0
        && dc.is_multiple_of(2)
        && slots.nl > 0
        && slots.nl <= slots.gw + slots.gw * (slots.gh - 1)
        && slots.gw % 2 == 1
        && slots.ay * slots.ahc >= hid
        && my * mhc >= hid
        && my.is_multiple_of(3)
        && px * ic >= gpt.ne.min(1)
        && spec.gx() > 0
        && c.is_multiple_of(spec.gx())
        && spec.vmc() <= 1
        && spec.hh().is_multiple_of(2)
        && c * 3 * mhc >= hid
        && slots.ay.is_multiple_of(gpt.agy.max(1))
        && my.is_multiple_of(gpt.mgy.max(1));
    if !ok {
        return Err(format!(
            "serve_gptoss params do not make a layout: {spec:?}"
        ));
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
            let files = spec.files().unwrap();
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
