//! The generated row kernels a load fires: one for each `fused.region`
//! region (kernels-ir's, as the compiler folded it) and width its trace
//! holds (`compiler.synth_regions`), each held to the row frame's contract
//! ([`kernels_cuda::synth::supports`]), printed by kernels-cuda and loaded
//! with the model, so that no fire compiles one or loads its module.
//!
//! The fold forms only regions the frame takes; one it does not is the
//! compiler's bug, and the load fails naming it (`Fault::Region`) rather
//! than running something else in its place.

use std::collections::BTreeMap;

use kernels_ir::Region;
#[cfg(not(feature = "rocm"))]
use poem_ir::{Def, Dtype, Ty, ValueDecl};
use poem_ir::{Fused, Operation, Trace, ValueId};

use crate::error::{Fault, Result};

#[cfg(not(feature = "rocm"))]
use kernels_cuda::synth::{In, Out, Synthesized};

/// Each generated kernel of a load, by body hash and width.
#[derive(Debug, Default)]
pub struct Regions {
    #[cfg(not(feature = "rocm"))]
    kernels: BTreeMap<(u64, u32), Synthesized>,
}

/// A load's regions where its trace holds none.
pub(crate) static NONE: Regions = Regions {
    #[cfg(not(feature = "rocm"))]
    kernels: BTreeMap::new(),
};

/// Every `fused.region` of `trace`: its node, and its parameters, width,
/// region key and results.
fn regions(trace: &Trace) -> impl Iterator<Item = (u32, &[ValueId], u32, u64, &[ValueId])> {
    (0u32..)
        .zip(&trace.nodes)
        .filter_map(|(at, node)| match &node.op {
            Operation::Fused(Fused::Region {
                ins,
                width,
                key,
                outs,
            }) => Some((at, ins.as_slice(), *width, *key, outs.as_slice())),
            _ => None,
        })
}

impl Regions {
    /// The kernels `trace`'s regions fire, compiled (or read from the disk
    /// cache) and loaded on the current device, `regions` the compiler's.
    #[cfg(not(feature = "rocm"))]
    pub(crate) fn load(trace: &Trace, regions_of: &BTreeMap<u64, Region>) -> Result<Regions> {
        let mut kernels = BTreeMap::new();
        for (node, ins, width, body, outs) in regions(trace) {
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
            let forms = ins
                .iter()
                .map(|&v| {
                    let dtype = stored(v)?;
                    let decl = &trace.values[v.0 as usize];
                    Ok(if one(decl) {
                        In::One(dtype)
                    } else if shared(decl) {
                        In::Shared(dtype)
                    } else {
                        In::Rows(dtype)
                    })
                })
                .collect::<Result<Vec<In>>>()?;
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
            if let std::collections::btree_map::Entry::Vacant(slot) = kernels.entry((body, width)) {
                let kernel = kernels_cuda::synth::synth(region, &forms, outs.len())?;
                kernel.load()?;
                slot.insert(kernel);
            }
        }
        Ok(Regions { kernels })
    }

    /// This backend generates no row kernel: a trace with a region is not
    /// served.
    #[cfg(feature = "rocm")]
    pub(crate) fn load(trace: &Trace, _regions: &BTreeMap<u64, Region>) -> Result<Regions> {
        match regions(trace).next() {
            Some((node, ..)) => Err(Fault::Unsupported(format!(
                "the `fused.region` at node {node}: this backend generates no row kernel; \
                 serve it with `compiler.synth_regions` off"
            ))),
            None => Ok(Regions {}),
        }
    }

    /// How many kernels the load generated: one a region and width.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(not(feature = "rocm"))]
        {
            self.kernels.len()
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

    /// The kernel of the region of key `body` over rows `width` wide.
    #[cfg(not(feature = "rocm"))]
    pub(crate) fn get(&self, body: u64, width: u32) -> Option<Synthesized> {
        self.kernels.get(&(body, width)).copied()
    }
}

/// Why a region's planes are not its own: a result twice, or over one of
/// its parameters. The frame's blocks read their rows whole before writing,
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
