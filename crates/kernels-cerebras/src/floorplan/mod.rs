//! The floorplan: what a wafer program places, element by element, between the logical plan
//! ([`super::plan`]: counts and split factors) and the physical build ([`super::floor`],
//! [`super::wiring`], [`super::fabric`]). Each feature of the plan's layer program asks for its
//! own structural element, independently of the others ([`Requests`]):
//!
//! | feature | element |
//! |---|---|
//! | attention over a cache | the scores segments on the attention rows / columns ([`Segment::Scores`]) |
//! | the gated delta rule | its (k, q) multicast and o-norm segments ([`Segment::Gdn`], [`Segment::Onorm`]) |
//! | routed experts | a MoE column run ([`ColRole::Moe`]) and the router's segments ([`Segment::Router`]) |
//! | a window | the KV rows sized to it ([`UnitShape::kv_rows`], an attention row run) |
//! | a per-token side input | its storage above each unit column ([`Aux::Store`]) |
//! | a cache read across layers | the KV line from its source unit ([`Aux::KvLine`]) |
//!
//! The units are one [`UnitShape`] (column runs by role, row runs by dims and role, the control
//! lines, K layers) visited along a [`UnitLayout`] (a snake of slots with gap slots, or explicit
//! origins); the head is one [`Head`] (east of the units, a band below them, or fed by groups of
//! rows). A [`Floorplan`] is the general partition: the four layouts the compiler had are points
//! of it, each made from a preset's keys by its constructor ([`plan`]):
//!
//! | constructor | units | path | head | beside the units |
//! |---|---|---|---|---|
//! | [`snake`] | one plain column run, one row run | a snake of UNX x UNY slots, at most one gap slot | east | — |
//! | [`stacked`] | one plain column run, KV on PYW rows | explicit origins in two unit columns ([`crate::plan::Arrangement`]) | east, token bus on | storage, KV line |
//! | [`runs`] | one plain column run, attention / plain row runs | two unit columns, down then up | east, by row groups | — |
//! | [`split`] | attention / MoE column runs | two unit columns from the band, up then down | a band below | — |
//!
//! The kernel set the layer program's mixers take ([`KernelSet`]) picks the constructor its pinned
//! plans and the search's moves are keyed for; the composer ([`super::compose::blocks`]) and the
//! role kernels ([`super::kernels::of`]) read the elements, never the constructor.
//!
//! What a later regime adds plugs in here: a row stream through units laid one a layer (a
//! [`UnitLayout`] and a token stream link in [`super::wiring`]), replicas of a unit shape, a K / V
//! region an attention block ([`Aux`]), a grid region for convolutions, a reading with no head.

use std::ops::Range;

use serde::Serialize;

use super::block::gdn::Gdn;
use super::compose::Needs;
use super::floor::{UnitGrid, UnitLayout};
use super::kernels::KernelSet;
use super::plan::{Arrangement, LayerKind, Plan, Preset, RowRun, Shape};

pub mod runs;
pub mod snake;
pub mod split;
pub mod stacked;

pub use snake::snake_gap;

type R<T> = Result<T, String>;

/// The plan of `shape` under `preset`: the partition and contract the preset's keys make for the
/// model, by the constructor of the kernel set its layers take. Each constructor reads its own
/// keys (the snake's unit slots, the stacked columns' arrangement, the row runs' dims and head
/// groups, the split's attention / MoE columns) and refuses the shapes its cut and kernels cannot
/// take by name. Nothing physical is decided here.
pub fn plan(shape: &Shape, preset: &Preset) -> R<Plan> {
    match KernelSet::of_shape(shape) {
        KernelSet::Own => snake::plan(shape, preset),
        KernelSet::Shared => stacked::plan(shape, preset),
        KernelSet::Windows => runs::plan(shape, preset),
        KernelSet::Experts => split::plan(shape, preset),
    }
}

/// Layers `layers` in units of `k`, every unit the same kinds; the unit count.
pub fn units_of(layers: &[LayerKind], k: u32, what: &str) -> R<u32> {
    let n = layers.len() as u32;
    if k == 0 || !n.is_multiple_of(k) {
        return Err(format!("{what}: {n} layers in units of {k}"));
    }
    Ok(n / k)
}

/// `n / d` rounded up to a multiple of `m`.
pub fn ceil_to(n: u32, d: u32, m: u32) -> u32 {
    n.div_ceil(d).div_ceil(m) * m
}

/// The structural elements a layer program's features ask for, each on its own (see the module
/// doc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Requests {
    /// Attention over a layer's own cache: the scores segments.
    pub scores: bool,
    /// The gated delta rule: its (k, q) multicast and the o-norm segments.
    pub gdn: bool,
    /// Routed experts: a MoE column run and the router's segments.
    pub experts: bool,
    /// Attention over a window: KV rows sized to it.
    pub window: bool,
    /// A per-token side input: storage above each unit column.
    pub store: bool,
    /// A cache read across layers: the KV line from the unit that appends it.
    pub kv_line: bool,
}

impl Requests {
    /// What `needs` asks the floor for.
    #[must_use]
    pub fn of(needs: &Needs) -> Requests {
        Requests {
            scores: needs.attention,
            gdn: needs.gdn,
            experts: needs.experts,
            window: needs.sliding,
            store: needs.side_in,
            kv_line: needs.shared_cache,
        }
    }
}

/// What a run of a unit's weight columns holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ColRole {
    /// Every phase of the layer (the dense units).
    Plain,
    /// The attention projections and their cache (the attention half).
    Attn,
    /// The routed experts and their router (the MoE half).
    Moe,
}

/// A run of `n` weight columns of one role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ColRun {
    pub n: u32,
    pub role: ColRole,
}

/// A unit's control lines: the CX column's local x, the CY row's local y.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Control {
    pub cx: u32,
    pub cy: u32,
}

/// One unit's shape: its weight columns in runs by role, its weight rows in runs by dims and
/// role (attention rows hold the KV cache), its control lines, the layers it runs, the rows
/// that keep KV.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct UnitShape {
    pub col_runs: Vec<ColRun>,
    pub row_runs: Vec<RowRun>,
    pub control: Control,
    /// Layers a unit runs (`K`).
    pub layers: u32,
    /// Rows a unit keeps KV on (a window's ring where it divides it).
    pub kv_rows: u32,
}

impl UnitShape {
    /// Weight columns / rows (the control column / row not counted).
    #[must_use]
    pub fn cols(&self) -> u32 {
        self.col_runs.iter().map(|r| r.n).sum()
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.row_runs.iter().map(|r| r.rows.1 - r.rows.0).sum()
    }

    /// The local columns of the first run of `role`.
    #[must_use]
    pub fn cols_of(&self, role: ColRole) -> Option<Range<u32>> {
        let mut at = 0;
        for r in &self.col_runs {
            if r.role == role {
                return Some(at..at + r.n);
            }
            at += r.n;
        }
        None
    }

    /// The columns attention runs on: the attention run's, else every column.
    #[must_use]
    pub fn attn_cols(&self) -> Range<u32> {
        self.cols_of(ColRole::Attn).unwrap_or(0..self.cols())
    }

    /// Whether a MoE column run halves the unit.
    #[must_use]
    pub fn halves(&self) -> bool {
        self.cols_of(ColRole::Moe).is_some()
    }

    /// Whether the rows come in runs of different dims or roles.
    #[must_use]
    pub fn row_roles(&self) -> bool {
        self.row_runs.len() > 1
    }

    /// The weight rows of the attention row runs.
    pub fn attn_rows(&self) -> impl Iterator<Item = u32> + '_ {
        self.row_runs
            .iter()
            .filter(|r| r.attention)
            .flat_map(|r| r.rows.0..r.rows.1)
    }
}

/// The units: `count` of one shape, visited along `path` (the x handoff follows it; unit 0 takes
/// the embedding).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Units {
    pub shape: UnitShape,
    pub path: UnitLayout,
    pub count: u32,
}

impl Units {
    /// The unit grid the floor lays ([`super::floor::FloorBuilder::add_units`]).
    #[must_use]
    pub fn grid(&self) -> UnitGrid {
        UnitGrid {
            cols: self.shape.cols(),
            rows: self.shape.rows(),
            cx: self.shape.control.cx,
            cy: self.shape.control.cy,
            layout: self.path.clone(),
            layers_per_unit: self.shape.layers,
        }
    }
}

/// Where the head sits and how the last unit's x reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum HeadAt {
    /// East of the units, a head column fed by two CX rows, the control column in the middle.
    East,
    /// A band below the units, x down the MoE columns, the control row in the middle.
    Band,
    /// East of the units, a head column a group of CX rows (the plan's `head.groups`).
    Groups,
}

/// The head: its placement, weight columns, rows and vocabulary rows a head PE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Head {
    pub at: HeadAt,
    pub cols: u32,
    pub rows: u32,
    pub vocab_rows: u32,
}

/// A segment all-reduce kind laid across the units, `width` columns a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Segment {
    /// The scores on every attention row's attention columns (a KV head's columns).
    Scores { width: u32 },
    /// GDN's (k, q) multicast a key head's columns.
    Gdn,
    /// The o-norm on every CY row (a value head's columns).
    Onorm { width: u32 },
    /// The router on every CY row's MoE columns (every expert's output once).
    Router { width: u32 },
}

/// A side input's storage above one unit column: its units top to bottom, its rows, the tokens a
/// storage PE holds, the colour its bus prefers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StoreColumn {
    pub units: Vec<u32>,
    pub rows: u32,
    pub tokens: u32,
    pub bus: u8,
}

/// The blocks placed beside the units for a feature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Aux {
    /// A per-token side input's storage above each unit column.
    Store(Vec<StoreColumn>),
    /// The shared caches' KV line from the source unit through its sharers.
    KvLine,
}

/// The program's elements (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Floorplan {
    pub units: Units,
    /// In the order they are laid.
    pub segments: Vec<Segment>,
    pub aux: Vec<Aux>,
    pub head: Head,
    /// The on-wafer token loop closes the program over the head.
    pub token_loop: bool,
}

impl Floorplan {
    /// The elements of `plan` for layers that ask `requests`: each element from the plan's
    /// choices for it. The units' path: the stacked arrangement's origins where the plan has one;
    /// two unit columns from the band (visited upward first, so the first and the last unit sit
    /// on it) where the head is a band; else the plan's unit slots (a snake, at most one gap
    /// slot); else two unit columns visited down, then up (unit 0 and the last side by side on
    /// row 0).
    pub fn of(plan: &Plan, requests: &Requests) -> R<Floorplan> {
        let p = &plan.partition;
        let col_runs = match p.attn_cols {
            Some(ax) => {
                if ax >= p.cols {
                    return Err("split plan: no MoE columns".into());
                }
                vec![
                    ColRun {
                        n: ax,
                        role: ColRole::Attn,
                    },
                    ColRun {
                        n: p.cols - ax,
                        role: ColRole::Moe,
                    },
                ]
            }
            None if requests.experts => return Err("split plan: no attention columns".into()),
            None => vec![ColRun {
                n: p.cols,
                role: ColRole::Plain,
            }],
        };
        let shape = UnitShape {
            control: Control {
                cx: p.attn_cols.unwrap_or(p.cols / 2),
                cy: p.rows / 2,
            },
            col_runs,
            row_runs: p.row_dims.clone(),
            layers: p.layers_per_unit,
            kv_rows: p.kv_rows,
        };
        let head = Head {
            at: if p.head.groups.is_some() {
                HeadAt::Groups
            } else if shape.halves() {
                HeadAt::Band
            } else {
                HeadAt::East
            },
            cols: p.head.cols,
            rows: p.head.rows,
            vocab_rows: p.head.vocab_rows,
        };
        let arrangement = p
            .arrangement
            .as_ref()
            .map(|_| Arrangement::of(plan))
            .transpose()?;
        let path = match (arrangement, head.at, p.unit_grid) {
            (Some(arr), _, _) => UnitLayout::Explicit {
                origins: arr.origins(p.cols + 1, p.rows + 1)?,
            },
            (None, HeadAt::Band, _) => {
                if !p.units.is_multiple_of(2) {
                    return Err("split: an even unit count (the last unit next to the band)".into());
                }
                two_columns(p.units, true)
            }
            (None, _, Some(ug)) => UnitLayout::Snake {
                unx: ug.0,
                uny: ug.1,
                up_first: false,
                gaps: snake_gap(p.units, ug)?.into_iter().collect(),
            },
            (None, _, None) => {
                let n = p.units;
                if n == 0 || !n.is_multiple_of(2) {
                    return Err(format!(
                        "runs: {n} units (two unit columns, unit 0 and the last on row 0)"
                    ));
                }
                two_columns(n, false)
            }
        };
        let mut segments = Vec::new();
        if requests.scores {
            segments.push(Segment::Scores {
                width: p.segments.scores,
            });
        }
        if requests.experts {
            segments.push(Segment::Router {
                width: p.segments.router.ok_or("split plan: no router segments")?,
            });
        }
        if requests.gdn && Gdn::of(plan).is_some() {
            segments.push(Segment::Gdn);
            segments.push(Segment::Onorm {
                width: p.segments.onorm,
            });
        }
        // a segment within a half of a plain unit's columns (none spans the control column)
        if !shape.halves() {
            for s in &segments {
                let (n, sw) = match *s {
                    Segment::Scores { width } => ("SWA", width),
                    Segment::Gdn => ("SWG", p.segments.gdn),
                    Segment::Onorm { width } => ("SWV", width),
                    Segment::Router { .. } => continue,
                };
                if sw == 0 || !(p.cols / 2).is_multiple_of(sw) {
                    return Err(format!(
                        "floorplan: {n} {sw} does not divide XC {} (a segment would span the control column)",
                        p.cols / 2
                    ));
                }
            }
        }
        // an east head takes two CX rows a column
        if head.at == HeadAt::East && head.cols * 2 != shape.rows() {
            return Err("floorplan: head columns = CX rows / 2".into());
        }
        let mut aux = Vec::new();
        if requests.store {
            let arr = Arrangement::of(plan)?;
            aux.push(Aux::Store(
                arr.columns(p.cols + 1)
                    .into_iter()
                    .filter(|(_, units, _)| !units.is_empty())
                    .map(|(_, units, st)| StoreColumn {
                        units,
                        rows: st.rows,
                        tokens: st.tokens,
                        bus: st.bus,
                    })
                    .collect(),
            ));
        }
        if requests.kv_line {
            aux.push(Aux::KvLine);
        }
        Ok(Floorplan {
            units: Units {
                shape,
                path,
                count: p.units,
            },
            segments,
            aux,
            head,
            token_loop: plan.contract.token_loop,
        })
    }

    /// The side input's storage columns, if any.
    #[must_use]
    pub fn store(&self) -> Option<&[StoreColumn]> {
        self.aux.iter().find_map(|a| match a {
            Aux::Store(c) => Some(c.as_slice()),
            Aux::KvLine => None,
        })
    }
}

/// `n` units in two unit columns, the first visited down and the second up (`up`: the first up
/// and the second down).
fn two_columns(n: u32, up: bool) -> UnitLayout {
    UnitLayout::Snake {
        unx: 2,
        uny: n / 2,
        up_first: up,
        gaps: Vec::new(),
    }
}
