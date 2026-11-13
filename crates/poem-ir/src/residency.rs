//! Which weights a device may stream rather than hold: a pure reading of
//! the ops that read each param.
//!
//! A param is **expert-streamable** when every op that reads it is a
//! router-selected MoE op ([`Linear::MoeSelect`]'s bank, bias and input
//! scales, [`Linear::MoeBiasSum`]'s bias, and the selects inside
//! [`Fused::MoeSelectGlu`], [`Fused::MoeExperts`] and [`Fused::MoeMega`]):
//! a fire reads only the experts its routes pick, so the atom is one expert
//! (axis 0 of every plane). It is **row-streamable** when every op that reads
//! it is a token-id row gather ([`Gather::Embed`] without tap weights, and
//! the embeds [`Fused::EmbedScaleAdd`] and [`Fused::EmbedScaleAddSelect`]
//! fold): a fire reads the rows its ids name, so the atom is a row. Anything
//! else, read by any other op or by both kinds, is **resident**.
//!
//! A weight's companion planes (`{name}.scales`, `.biases`,
//! `.global_scales`, `.input_scales`) are read through their leading plane's
//! value and take its class; a plane whose first axis is not the atom axis
//! of the rest stays resident with the rest.

use crate::Operands;
use crate::ops::{Fused, Gather, Linear, Operation};
use crate::trace::{ParamPlacement, Trace};
use crate::value::{BIASES, Def, Dim, GLOBAL_SCALES, INPUT_SCALES, SCALES, Ty, ValueId};

/// How a param is read, so how a device may hold it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Residency {
    /// Held whole on the device.
    Resident,
    /// Read one expert at a time by router-selected MoE ops: `experts`
    /// along axis 0 of every plane, `top_k` of them a token.
    Expert(ExpertAtom),
    /// Read a few rows a token by token-id gathers: `rows` along axis 0.
    Rows(RowAtom),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertAtom {
    /// The experts the bank holds (its planes along axis 0).
    pub experts: u32,
    /// The most experts a token's route picks among those that read it.
    pub top_k: u32,
    /// The layer of the first op that reads it, where the trace states one.
    pub layer: Option<u32>,
    /// The routes the first select that reads it picks by: the banks one
    /// MoE layer's selects read share them, so an expert's whole FFN.
    pub routes: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowAtom {
    /// The table's rows (axis 0).
    pub rows: u64,
    /// The ids per token the gathers read: 1 for a token's own id, the
    /// width of its n-gram ids for a hashed table read side by side.
    pub ids_per_token: u32,
    /// Whether every gather reads it through hashed n-gram ids
    /// ([`Gather::NgramIds`]) rather than a token's id itself.
    pub hashed: bool,
    /// The layer of the first gather that reads it, where the trace states
    /// one; `None` for a table read outside every layer.
    pub layer: Option<u32>,
}

impl Residency {
    #[must_use]
    pub fn is_resident(&self) -> bool {
        matches!(self, Residency::Resident)
    }
}

/// The role one read gives a weight value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Other,
    Expert { top_k: u32, routes: ValueId },
    Rows { ids_per_token: u32, hashed: bool },
}

/// Each param's [`Residency`], indexed as `trace.params`.
#[must_use]
pub fn residency(trace: &Trace) -> Vec<Residency> {
    let lead = leads(trace);
    // Each param's roles: `None` until read.
    let mut seen: Vec<Option<Residency>> = vec![None; trace.params.len()];
    let mut mixed = vec![false; trace.params.len()];
    let mut reads = Vec::new();
    for node in &trace.nodes {
        reads.clear();
        roles(trace, &node.op, &mut reads);
        for &(v, role) in &reads {
            let Def::Weight(p) = trace.values[v.0 as usize].def else {
                continue;
            };
            let p = p as usize;
            let class = match role {
                Role::Other => Residency::Resident,
                Role::Expert { top_k, routes } => {
                    let experts = trace.params[p].experts.map_or_else(
                        || trace.params[p].shape.first().copied().unwrap_or(0),
                        |slots| u64::from(slots.experts),
                    );
                    Residency::Expert(ExpertAtom {
                        experts: u32::try_from(experts).unwrap_or(0),
                        top_k,
                        layer: node.layer,
                        routes,
                    })
                }
                Role::Rows {
                    ids_per_token,
                    hashed,
                } => Residency::Rows(RowAtom {
                    rows: trace.params[p].shape.first().copied().unwrap_or(0),
                    ids_per_token,
                    hashed,
                    layer: node.layer,
                }),
            };
            seen[p] = Some(match (seen[p], class) {
                (None, class) => class,
                (Some(Residency::Expert(a)), Residency::Expert(b)) if a.experts == b.experts => {
                    Residency::Expert(ExpertAtom {
                        top_k: a.top_k.max(b.top_k),
                        ..a
                    })
                }
                (Some(Residency::Rows(a)), Residency::Rows(b)) if a.rows == b.rows => {
                    Residency::Rows(RowAtom {
                        ids_per_token: a.ids_per_token.max(b.ids_per_token),
                        hashed: a.hashed && b.hashed,
                        ..a
                    })
                }
                (Some(_), _) => {
                    mixed[p] = true;
                    Residency::Resident
                }
            });
            if mixed[p] {
                seen[p] = Some(Residency::Resident);
            }
        }
    }
    // Companions take their lead's class: a companion along the atom axis
    // streams with it, a per-tensor one (no axis, or one of another
    // extent) stays on the device; one of two or more axes along another
    // extent, or one read in another role, keeps the whole weight resident.
    let mut out: Vec<Residency> = (0..trace.params.len())
        .map(|p| seen[lead[p]].unwrap_or(Residency::Resident))
        .collect();
    let first = |q: usize| {
        let param = &trace.params[q];
        param.experts.map_or_else(
            || param.shape.first().copied(),
            |s| Some(u64::from(s.experts)),
        )
    };
    for p in (0..trace.params.len()).filter(|p| lead[*p] == *p) {
        let extent = match out[p] {
            Residency::Resident => continue,
            Residency::Expert(atom) => u64::from(atom.experts),
            Residency::Rows(atom) => atom.rows,
        };
        let planes: Vec<usize> = (p..trace.params.len()).filter(|q| lead[*q] == p).collect();
        let kind = core::mem::discriminant(&out[p]);
        let whole = extent > 0
            && trace.params[p].shape.len() >= 2
            && first(p) == Some(extent)
            && planes.iter().all(|&q| {
                let read_alike = seen[q].is_none_or(|r| core::mem::discriminant(&r) == kind);
                read_alike && (first(q) == Some(extent) || trace.params[q].shape.len() < 2)
            });
        for &q in &planes {
            if !whole || first(q) != Some(extent) {
                out[q] = Residency::Resident;
            }
        }
    }
    out
}

/// The placement a param is served from: [`ParamPlacement::Host`] for the
/// leading plane of a hashed table the model reads at its input, outside
/// every layer, only through n-gram ids (a few rows a token out of a table
/// no device should hold whole); everything else, its companions included,
/// on the device.
///
/// This is the placement the packages once declared by hand (`.host()`),
/// now inferred: it is narrower than [`Residency::Rows`], which a weight
/// cache reads. A hashed table read inside a layer (DeepSeek-V4.1's Engram)
/// and a table gathered by token ids alone (an untied embedding, Gemma's
/// per-layer embeddings) are row-streamable but stay on the device here, as
/// they always were.
#[must_use]
pub fn placement(trace: &Trace, residency: &[Residency]) -> Vec<ParamPlacement> {
    let lead = leads(trace);
    (0..trace.params.len())
        .map(|p| match residency[p] {
            Residency::Rows(RowAtom {
                hashed: true,
                layer: None,
                ..
            }) if lead[p] == p => ParamPlacement::Host,
            _ => ParamPlacement::Device,
        })
        .collect()
}

/// Each param's leading plane: itself, or the plane whose name it extends
/// by a companion suffix, declared just before it.
#[must_use]
pub fn leads(trace: &Trace) -> Vec<usize> {
    let mut lead: Vec<usize> = (0..trace.params.len()).collect();
    for p in 1..trace.params.len() {
        let name = &trace.params[p].name;
        let Some(stem) = [SCALES, BIASES, GLOBAL_SCALES, INPUT_SCALES]
            .iter()
            .find_map(|suffix| name.strip_suffix(suffix))
        else {
            continue;
        };
        let mut q = p;
        while q > 0 {
            q -= 1;
            if trace.params[q].name == stem {
                lead[p] = lead[q];
                break;
            }
            if !trace.params[q].name.starts_with(stem) {
                break;
            }
        }
    }
    lead
}

/// Every input `op` reads, with the role it reads it in.
fn roles(trace: &Trace, op: &Operation, sink: &mut Vec<(ValueId, Role)>) {
    let mut ins = Vec::new();
    op.inputs(&mut ins);
    let start = sink.len();
    sink.extend(ins.iter().map(|v| (*v, Role::Other)));
    let give = |sink: &mut Vec<(ValueId, Role)>, v: ValueId, role: Role| {
        for (at, r) in sink[start..].iter_mut() {
            if *at == v {
                *r = role;
            }
        }
    };
    let linear = |sink: &mut Vec<(ValueId, Role)>, op: &Linear| match op {
        Linear::MoeSelect {
            bank,
            bias,
            routes,
            input_scales,
            ..
        } => {
            let role = Role::Expert {
                top_k: width(trace, *routes),
                routes: *routes,
            };
            for v in [Some(*bank), *bias, *input_scales].into_iter().flatten() {
                give(sink, v, role);
            }
        }
        Linear::MoeBiasSum { bias, routes, .. } => {
            give(
                sink,
                *bias,
                Role::Expert {
                    top_k: width(trace, *routes),
                    routes: *routes,
                },
            );
        }
        _ => {}
    };
    match op {
        Operation::Linear(op) => linear(sink, op),
        Operation::Fused(
            Fused::MoeSelectGlu { select, .. }
            | Fused::MoeExperts { select, .. }
            | Fused::MoeMega { select, .. },
        ) => {
            linear(sink, select);
            if let Operation::Fused(Fused::MoeExperts { down, .. } | Fused::MoeMega { down, .. }) =
                op
            {
                linear(sink, down);
            }
        }
        Operation::Gather(Gather::Embed {
            ids,
            weights: None,
            table,
            scales,
            concat,
            ..
        }) => {
            let role = Role::Rows {
                ids_per_token: if *concat {
                    width(trace, *ids).max(1)
                } else {
                    1
                },
                hashed: hashed(trace, *ids),
            };
            for v in [Some(*table), *scales].into_iter().flatten() {
                give(sink, v, role);
            }
        }
        Operation::Fused(
            Fused::EmbedScaleAdd { ids, table, .. } | Fused::EmbedScaleAddSelect { ids, table, .. },
        ) => give(
            sink,
            *table,
            Role::Rows {
                ids_per_token: 1,
                hashed: hashed(trace, *ids),
            },
        ),
        _ => {}
    }
}

/// The columns of a `[rows, width]` value: a route's picks, a token's ids.
fn width(trace: &Trace, v: ValueId) -> u32 {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.get(1) {
            Some(Dim::Const(w)) => u32::try_from(*w).unwrap_or(u32::MAX),
            None => 1,
            _ => 0,
        },
        Ty::Struct(_) => 0,
    }
}

/// Whether `ids` are hashed n-gram ids: made by [`Gather::NgramIds`] or
/// [`Gather::NgramIdsChunked`], or merged from arms that each are.
fn hashed(trace: &Trace, ids: ValueId) -> bool {
    match &trace.values[ids.0 as usize].def {
        Def::Op(n) => matches!(
            trace.nodes[*n as usize].op,
            Operation::Gather(Gather::NgramIds { .. } | Gather::NgramIdsChunked { .. })
        ),
        Def::Merge(arms) => !arms.is_empty() && arms.iter().all(|(arm, _)| hashed(trace, *arm)),
        _ => false,
    }
}
