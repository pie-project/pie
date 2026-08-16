//! Loading a checkpoint into something a fire can bind.
//!
//! [`compile_load_plan_for`](crate::loader::compile_load_plan_for) authors the
//! plan and checks its files; `weights::stage_plan_weights` runs it and stages
//! every tensor into one device region. This is the call between them, plus
//! the one conversion that makes the result answer a trace's questions: a
//! [`Handle`](crate::device::Handle) map becomes a [`Slice`] map, which is
//! what `lowering::resolve::Store` reads.
//!
//! A `Handle` is a checked view that owns a reference to its buffer; a `Slice`
//! is an address and an extent. The binder takes the second so it stays
//! portable and provable with no device in the build, and the region is kept
//! beside the map here because a map of addresses whose buffer has been
//! dropped is a map of dangling pointers.

use std::collections::HashMap;
use std::path::Path;

use crate::device::{Allocation, Context};
use crate::error::{Error, Result};
use crate::layout::region::Region as _;
use crate::loader::{compile_load_plan_for, metal_storage_target};
use crate::lowering::executor::Slice;
use crate::weights::stage_plan_weights;

/// A checkpoint on the device: the region that holds it, and where each
/// tensor sits in it.
pub struct Loaded {
    /// The staged region. **Held, not dropped** — every address in `tensors`
    /// points into it, and it is resident for exactly as long as this
    /// `Loaded` lives. Dropping it while a fire is bound against those
    /// addresses takes the weights out of the residency set under a running
    /// GPU, which is why it is a field rather than a local.
    ///
    /// CHUNKED, and never one: a recorded command cannot bind past 4 GiB
    /// (`device/recording.rs`), so `stage_weights` cuts the staged bytes on
    /// tensor boundaries into buffers no larger than that. Every address in
    /// `tensors` still points into exactly one of them.
    pub regions: Vec<Allocation>,
    /// Checkpoint tensor name → its address and extent.
    pub tensors: HashMap<String, Slice>,
    /// Weights the plan leaves in MXFP4, by name.
    ///
    /// Read off the plan by [`LoadPlan::mxfp4_tensor_names`], which is where
    /// the reasoning lives -- including why it is a set of names and not a
    /// flag: a driver matching on `Encoding::Quant` structurally is reading
    /// the loader's enums instead of its answers.
    ///
    /// [`LoadPlan::mxfp4_tensor_names`]: model_loader::plan::LoadPlan::mxfp4_tensor_names
    pub mxfp4: std::collections::HashSet<String>,
    /// Every DISTINCT affine point the plan's tensors arrive at, sorted.
    ///
    /// Read off the plan by [`LoadPlan::affine_points`], answering
    /// [`Self::mxfp4`]'s question from the other side: `mxfp4` says which
    /// tensors are NOT affine, and this says how many affine points the ones
    /// that are arrived at. `binding::observed` builds ONE kernel set, and
    /// [`Self::affine_point`] reads this to decide WHICH -- and owns the
    /// refusal when there is more than one.
    ///
    /// [`LoadPlan::affine_points`]: model_loader::plan::LoadPlan::affine_points
    pub affine_points: Vec<(u32, u32)>,
    /// One TENSOR NAME per point in [`Self::affine_points`], same order.
    ///
    /// Carried only so the refusal can name what it refused over. A count of
    /// points tells an operator that a checkpoint is not servable; a witness
    /// tells them which tensors made it so, and for `gpt-oss` that is 24
    /// router gates and nothing else — a fact that turns a dead end into the
    /// next piece of work.
    pub affine_witnesses: Vec<((u32, u32), String)>,
    /// Every affine tensor's point, by name — [`LoadPlan::affine_by_name`].
    ///
    /// Read through [`Self::affine_point_of`], and put to exactly two names:
    /// `binding::EXPERT_BANK` and `binding::ROUTER_GATE`. It answers an
    /// ENCODING and not a model fact, which is the same licence
    /// [`Self::mxfp4`] holds and the same one `no_probe_decides_a_fact`
    /// checks.
    ///
    /// [`LoadPlan::affine_by_name`]: model_loader::plan::LoadPlan::affine_by_name
    pub affine_by_name: std::collections::HashMap<String, (u32, u32)>,
}

impl Loaded {
    /// The affine point ONE named tensor arrived at.
    ///
    /// `None` when the checkpoint has no such tensor, or has it raw or in
    /// MXFP4 — none of the three is read at an affine point.
    #[must_use]
    pub fn affine_point_of(&self, name: &str) -> Option<crate::batch::AffineFormat> {
        self.affine_by_name
            .get(name)
            .map(|(group, bits)| crate::batch::AffineFormat {
                bits: *bits,
                group: *group,
            })
    }

    /// THE AFFINE POINT THE BYTES ARRIVED IN -- measured, not declared.
    ///
    /// Not `Encoding::from_config_json`: the `bits`/`group_size` at the top of
    /// `config.json`'s quantization block is a DEFAULT that a checkpoint may
    /// override per tensor. `gpt-oss-20b-MXFP4-Q4` states `g32/b4 mxfp4` at
    /// the top -- which describes its EXPERT BANKS, the only tensors that are
    /// not affine at all -- then overrides every projection, the embedding and
    /// the head to `g64/b4 affine` and every `mlp.router` to `g64/b8`, so the
    /// declared point matches not one tensor in the file. Read at g32, a
    /// 64-wide group is walked as two 32-wide ones and every scale after the
    /// first comes off the wrong offset.
    ///
    /// The tensors CAN answer this: `scales` has shape `[rows, cols / group]`,
    /// so the group is `cols / scales_cols` exactly, which is the division
    /// [`LoadPlan::affine_points`] already performs per tensor. Declared stays
    /// the authority on METHOD -- mxfp4 or affine is not an extent -- and that
    /// question is `Self::mxfp4`.
    ///
    /// # Errors
    ///
    /// More than one point: `binding::observed` instantiates ONE kernel set,
    /// so the second point's tensors would be dequantised at the first's
    /// width. A checkpoint with NO affine tensors is not an error -- it is a
    /// dense one, and `{bits: 0, group: 0}` is how this driver spells that.
    ///
    /// [`LoadPlan::affine_points`]: model_loader::plan::LoadPlan::affine_points
    pub fn affine_point(&self, row: &str) -> Result<crate::batch::AffineFormat> {
        // THE POINTS NOTHING HAS ACCOUNTED FOR. The router gate is allowed
        // its own -- `MetalBinding::router_quant_group` carries it and the
        // text names a second `affine_point` for that one op -- so it is
        // subtracted before the count that refuses.
        //
        // Subtracting a point the gate SHARES with the stack removes
        // nothing that matters: the empty arm hands that same point back,
        // which is the uniform checkpoint's answer and the reason this needs
        // no rule for what a router name looks like.
        let router = crate::model::binding::router_point(|n| self.affine_point_of(n));
        let unaccounted: Vec<(u32, u32)> = self
            .affine_points
            .iter()
            .copied()
            .filter(|p| router.is_none_or(|r| (r.group, r.bits) != *p))
            .collect();
        match unaccounted.as_slice() {
            // Every affine tensor is the gate, or there are none at all.
            // `{0, 0}` is how this driver spells a dense checkpoint, and
            // `geometry_from_deployment` replaces it with a point some
            // shader is instantiated at rather than passing `gs_0_b_0` on.
            [] => Ok(router.unwrap_or(crate::batch::AffineFormat { bits: 0, group: 0 })),
            [(group, bits)] => Ok(crate::batch::AffineFormat {
                bits: *bits,
                group: *group,
            }),
            many => {
                let points = many
                    .iter()
                    .map(|(g, b)| {
                        let witness = self
                            .affine_witnesses
                            .iter()
                            .find(|(p, _)| p == &(*g, *b))
                            .map_or_else(String::new, |(_, n)| format!(" (`{n}`)"));
                        format!("g{g}/b{b}{witness}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(crate::Error::Create {
                    what: "checkpoint",
                    message: format!(
                        "`{row}` arrives at {} affine points beside its router \
                         gate's ({points}) and this driver instantiates ONE \
                         kernel set for the stack. Every tensor at another \
                         point would be dequantised at the first one's width — \
                         scales read from the wrong offset. Refused rather \
                         than served wrongly",
                        many.len()
                    ),
                })
            }
        }
    }
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl Loaded {
    /// Every tensor name the checkpoint published, sorted.
    ///
    /// What a resolver's misses are diagnosed against: a name the text asks
    /// for and this list does not contain is either a spelling the map has
    /// wrong or a tensor the plan did not publish, and the two are told apart
    /// by looking.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        out.sort_unstable();
        out
    }
}

/// Author the plan for `snapshot_dir` against `row`, run it, and stage every
/// tensor.
///
/// The identified [`Variant`](model::catalog::Variant) rather than a
/// `pie.model/1` descriptor: the row was matched against this checkpoint's OWN
/// TENSOR NAMES AND EXTENTS by `model::catalog::identify`, so the author that
/// writes the contract and the bytes the contract is authored over cannot
/// describe two different models. `encoding` is the one fact a row genuinely
/// cannot state -- the same checkpoint is published at 4 bits and at 8.
///
/// # Errors
///
/// A plan that will not compile (a shape the row asserts and the checkpoint
/// contradicts, a declared file that is not on disk at the size declared), or
/// a staging that will not allocate.
pub fn load(
    context: &Context,
    snapshot_dir: &Path,
    row: &dyn model::catalog::Variant,
    encoding: &model::encoding::Encoding,
) -> Result<Loaded> {
    let target = metal_storage_target();
    let (plan, _moe) =
        compile_load_plan_for(snapshot_dir, &target, row, encoding).map_err(|err| {
            Error::Create {
                what: "load plan",
                // `{err}`, not `{err:?}`. `LoadPlanError`'s Display quotes the
                // compiler's own words and names the tensor it refused over; its
                // Debug prints `Plan(Compile("..."))`. An operator reads this
                // message.
                message: err.to_string(),
            }
        })?;
    let (regions, staged) = stage_plan_weights(context, &plan, snapshot_dir)?;
    let tensors = staged
        .into_iter()
        .map(|(name, handle)| {
            (
                name,
                Slice {
                    address: handle.gpu_address(),
                    bytes: handle.len(),
                },
            )
        })
        .collect();
    let mxfp4 = plan.mxfp4_tensor_names();
    let affine_points = plan.affine_points();
    let affine_witnesses = plan.affine_point_witnesses();
    let affine_by_name = plan.affine_by_name();
    Ok(Loaded {
        regions,
        tensors,
        mxfp4,
        affine_points,
        affine_witnesses,
        affine_by_name,
    })
}
