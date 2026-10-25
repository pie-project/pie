//! The generated row kernels a load fires: one for each `fused.region`
//! region (kernels-ir's, as the compiler folded it) and width its trace
//! holds (`compiler.synth_regions`), each held to the row skeleton's contract
//! ([`kernels_cuda::synth::supports`]), printed by kernels-cuda and loaded
//! with the model, so that no fire compiles one or loads its module.
//!
//! And the contraction skeletons' epilogues a load fires: one for each
//! `fused.anchored` anchored region (`compiler.synth_anchors`), formed only
//! where kernels-cuda serves it ([`kernels_cuda::linear::skeleton::serves`])
//! and its every kernel loaded with the model.
//!
//! The fold forms only regions the skeleton takes; one it does not is the
//! compiler's bug, and the load fails naming it (`Fault::Region`) rather
//! than running something else in its place.

use std::collections::BTreeMap;

use kernels_ir::{Anchored, Region};
#[cfg(not(feature = "rocm"))]
use poem_ir::{Def, Dtype, Ty, ValueDecl};
use poem_ir::{Fused, Operation, RegionRead, Trace, ValueId};

use crate::error::{Fault, Result};

#[cfg(not(feature = "rocm"))]
use kernels_cuda::linear::skeleton::Epilogue;
#[cfg(not(feature = "rocm"))]
use kernels_cuda::synth::{In, Out, Synthesized};

/// Each generated kernel of a load, by body hash and width, and by how
/// its parameters are stored.
#[derive(Debug, Default)]
pub struct Regions {
    #[cfg(not(feature = "rocm"))]
    kernels: BTreeMap<(u64, u32), Vec<(Vec<In>, Synthesized)>>,
    /// Each anchored region's epilogue, by the region's key.
    #[cfg(not(feature = "rocm"))]
    anchors: BTreeMap<u64, &'static Epilogue>,
}

/// A load's regions where its trace holds none.
pub(crate) static NONE: Regions = Regions {
    #[cfg(not(feature = "rocm"))]
    kernels: BTreeMap::new(),
    #[cfg(not(feature = "rocm"))]
    anchors: BTreeMap::new(),
};

/// `trace` with each matmul and the pointwise ops over its product folded
/// into one `fused.anchored` op where the contraction skeletons serve it
/// (`compiler.synth_anchors`), and the anchored regions by key.
#[cfg(not(feature = "rocm"))]
#[must_use]
pub fn fold_anchors(trace: Trace) -> (Trace, BTreeMap<u64, Anchored>) {
    poem_compiler::fuse::anchor::fold(trace, &|anchored, ins, outs| {
        kernels_cuda::linear::skeleton::serves(anchored, ins, outs).is_ok()
    })
}

/// Every `fused.anchored` of `trace`: its node, key and the values it
/// reads and lands.
fn anchored(trace: &Trace) -> impl Iterator<Item = (u32, u64, &[ValueId], &[ValueId])> {
    (0u32..)
        .zip(&trace.nodes)
        .filter_map(|(node, n)| match &n.op {
            Operation::Fused(Fused::Anchored { ins, key, outs, .. }) => {
                Some((node, *key, ins.as_slice(), outs.as_slice()))
            }
            _ => None,
        })
}

/// One `fused.region` of a trace: its node, parameters (and how each is
/// read), width, region key and results. ROCm reads only where it lies
/// (to refuse it).
#[cfg_attr(feature = "rocm", allow(dead_code))]
struct Site<'t> {
    node: u32,
    ins: &'t [ValueId],
    reads: &'t [RegionRead],
    width: u32,
    body: u64,
    outs: &'t [ValueId],
}

/// Every `fused.region` of `trace`.
fn regions(trace: &Trace) -> impl Iterator<Item = Site<'_>> {
    (0u32..)
        .zip(&trace.nodes)
        .filter_map(|(node, n)| match &n.op {
            Operation::Fused(Fused::Region {
                ins,
                reads,
                width,
                key,
                outs,
            }) => Some(Site {
                node,
                ins,
                reads,
                width: *width,
                body: *key,
                outs,
            }),
            _ => None,
        })
}

impl Regions {
    /// The kernels `trace`'s regions fire, compiled (or read from the disk
    /// cache) and loaded on the current device, `regions` the compiler's.
    #[cfg(not(feature = "rocm"))]
    pub(crate) fn load(
        trace: &Trace,
        regions_of: &BTreeMap<u64, Region>,
        anchored_of: &BTreeMap<u64, Anchored>,
        tuning: &kernels_cuda::tuning::Tuning,
    ) -> Result<Regions> {
        let mut anchors = BTreeMap::new();
        for (node, key, ins, outs) in anchored(trace) {
            let broken = |why: String| Fault::Region {
                node,
                body: key,
                why,
            };
            if anchors.contains_key(&key) {
                continue;
            }
            let held = anchored_of.get(&key).ok_or_else(|| {
                broken("the compiled model holds no anchored region by its key".into())
            })?;
            if let Some(why) = aliased(ins, outs) {
                return Err(broken(why));
            }
            let stored = |vs: &[ValueId]| {
                vs.iter()
                    .map(|&v| {
                        dtype_of(trace, v)
                            .ok_or_else(|| broken(format!("value {} is a struct", v.0)))
                    })
                    .collect::<Result<Vec<Dtype>>>()
            };
            kernels_cuda::linear::skeleton::serves(held, &stored(ins)?, &stored(outs)?)
                .map_err(|why| broken(why.to_string()))?;
            let epi = Epilogue::interned(held)?;
            let c = &held.contraction;
            epi.load_every(c.n, c.k, tuning)?;
            anchors.insert(key, epi);
        }
        let mut kernels = BTreeMap::new();
        for Site {
            node,
            ins,
            reads,
            width,
            body,
            outs,
        } in regions(trace)
        {
            let broken = |why: String| Fault::Region { node, body, why };
            let region = regions_of
                .get(&body)
                .ok_or_else(|| broken("the compiled model holds no region by its key".into()))?;
            if let Some(why) = aliased(ins, outs) {
                return Err(broken(why));
            }
            let stored = |v: ValueId| {
                dtype_of(trace, v).ok_or_else(|| broken(format!("value {} is a struct", v.0)))
            };
            let forms = forms(&trace.values, ins, reads, |v| dtype_of(trace, v)).map_err(broken)?;
            let results = outs
                .iter()
                .map(|&v| stored(v).map(Out::Rows))
                .collect::<Result<Vec<Out>>>()?;
            kernels_cuda::synth::supports(region, &forms, &results)
                .map_err(|why| broken(why.to_string()))?;
            if region.width != width {
                return Err(broken(format!(
                    "the op states rows {width} wide and its region's are not"
                )));
            }
            let held: &mut Vec<(Vec<In>, Synthesized)> = kernels.entry((body, width)).or_default();
            if !held.iter().any(|(stored, _)| *stored == forms) {
                let kernel = kernels_cuda::synth::synth(region, &forms, outs.len())?;
                kernel.load()?;
                held.push((forms, kernel));
            }
        }
        Ok(Regions { kernels, anchors })
    }

    /// This backend generates no row kernel: a trace with a region is not
    /// served.
    #[cfg(feature = "rocm")]
    pub(crate) fn load(
        trace: &Trace,
        _regions: &BTreeMap<u64, Region>,
        _anchored: &BTreeMap<u64, Anchored>,
        _tuning: &kernels_cuda::tuning::Tuning,
    ) -> Result<Regions> {
        if let Some(Site { node, .. }) = regions(trace).next() {
            return Err(Fault::Unsupported(format!(
                "the `fused.region` at node {node}: this backend generates no row kernel; \
                 serve it with `compiler.synth_regions` off"
            )));
        }
        match anchored(trace).next() {
            Some((node, ..)) => Err(Fault::Unsupported(format!(
                "the `fused.anchored` at node {node}: this backend has no contraction skeletons; \
                 serve it with `compiler.synth_anchors` off"
            ))),
            None => Ok(Regions {}),
        }
    }

    /// How many kernels the load generated: one a region, width and
    /// storage, and one epilogue an anchored region.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(not(feature = "rocm"))]
        {
            self.kernels.values().map(Vec::len).sum::<usize>() + self.anchors.len()
        }
        #[cfg(feature = "rocm")]
        {
            0
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The epilogue of the anchored region of key `key`.
    #[cfg(not(feature = "rocm"))]
    pub(crate) fn anchor(&self, key: u64) -> Option<&'static Epilogue> {
        self.anchors.get(&key).copied()
    }

    /// The kernel of the region of key `body` over rows `width` wide, its
    /// parameters stored as `forms` says.
    #[cfg(not(feature = "rocm"))]
    pub(crate) fn get(&self, body: u64, width: u32, forms: &[In]) -> Option<Synthesized> {
        self.kernels
            .get(&(body, width))?
            .iter()
            .find(|(stored, _)| stored.as_slice() == forms)
            .map(|(_, kernel)| *kernel)
    }
}

/// How each of a region's parameters `ins` is stored: a weight of one
/// element, one row of a weight every row reads, rows picked by id (as
/// `reads` says), one element a row (a position: a value one element a
/// row), or each row its own; in the dtype its plane holds.
#[cfg(not(feature = "rocm"))]
pub(crate) fn forms(
    values: &[ValueDecl],
    ins: &[ValueId],
    reads: &[RegionRead],
    dtype: impl Fn(ValueId) -> Option<Dtype>,
) -> std::result::Result<Vec<In>, String> {
    (0..ins.len())
        .map(|k| {
            let v = ins[k];
            let dtype = dtype(v).ok_or_else(|| format!("value {} is a struct", v.0))?;
            let decl = &values[v.0 as usize];
            let read = reads.get(k).copied().unwrap_or_default();
            Ok(if read.by.is_some() {
                In::Gathered(dtype)
            } else if one(decl) {
                In::One(dtype)
            } else if shared(decl) {
                In::Shared(dtype)
            } else if each(decl) {
                In::Each(dtype)
            } else {
                In::Rows(dtype)
            })
        })
        .collect()
}

/// Whether `decl` is one element a row, its own: a value of rows one
/// element wide (a position).
#[cfg(not(feature = "rocm"))]
fn each(decl: &ValueDecl) -> bool {
    !matches!(decl.def, Def::Weight(_))
        && matches!(
            &decl.ty,
            Ty::Tensor { shape, .. }
                if matches!(shape.as_slice(), [d] | [d, poem_ir::Dim::Const(1)] if !matches!(d, poem_ir::Dim::Const(_)))
        )
}

/// Why a region's planes are not its own: a result twice, or over one of
/// its parameters. The skeleton's blocks read their rows whole before writing,
/// but a block's row is not another block's.
#[cfg(not(feature = "rocm"))]
fn aliased(ins: &[ValueId], outs: &[ValueId]) -> Option<String> {
    outs.iter().enumerate().find_map(|(k, out)| {
        if outs[..k].contains(out) {
            Some(format!("result {k} is an earlier result's value"))
        } else {
            ins.iter()
                .position(|v| v == out)
                .map(|p| format!("result {k} is parameter {p}'s value"))
        }
    })
}

/// Whether `decl` is one row every row reads: a weight of rank 1.
#[cfg(not(feature = "rocm"))]
pub(crate) fn shared(decl: &ValueDecl) -> bool {
    matches!(decl.def, Def::Weight(_))
        && matches!(&decl.ty, Ty::Tensor { shape, .. } if shape.len() == 1)
}

/// Whether `decl` is one element every row reads: a weight of one element
/// (a layer's scale).
#[cfg(not(feature = "rocm"))]
pub(crate) fn one(decl: &ValueDecl) -> bool {
    matches!(decl.def, Def::Weight(_))
        && matches!(&decl.ty, Ty::Tensor { shape, .. } if shape.as_slice() == [poem_ir::Dim::Const(1)])
}

/// How `v` is stored, a weight as its parameter states; `None` for a
/// struct.
#[cfg(not(feature = "rocm"))]
fn dtype_of(trace: &Trace, v: ValueId) -> Option<Dtype> {
    let decl = &trace.values[v.0 as usize];
    match (&decl.def, &decl.ty) {
        (Def::Weight(w), _) => Some(trace.params[*w as usize].dtype),
        (_, Ty::Tensor { dtype, .. }) => Some(*dtype),
        (_, Ty::Struct(_)) => None,
    }
}
