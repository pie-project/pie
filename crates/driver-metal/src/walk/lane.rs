//! The lane: a `model_compiler::program::Program` per fire class, built at load
//! and walked per fire.
//!
//! # What this is
//!
//! `driver-cuda/src/baker/` is the reference implementation and this is the
//! half of it that a shader plane does not get to have an opinion about. The
//! chain is the same one end to end -- text → `Plan` → lane → `Program` → walk
//! → the plane's claim bodies -- and every module of this crate matches its
//! cuda sibling by name, so that a decision made once is findable twice.
//!
//! # ONE CATALOG (R3)
//!
//! The pools used to be the OTHER catalog's: `model_legacy::deployment` sized
//! and strided the KV pages and the recurrent slabs off a config-projected
//! registry, while a per-driver `Binding` carried what the LOAD had observed
//! back into the trace, so that the text a driver ran depended on what its own
//! loader had found in the checkpoint. Every one of those pieces is gone.
//!
//! [`Deployment::of`](model::deployment::Deployment::of) reads the pool
//! geometry off the SAME `Plan` the program is built from, so a pool and the
//! program that indexes it cannot describe different models -- which is why
//! cuda's `baker::Geometry::agrees_with` was deleted rather than ported: a
//! check that can only pass is not a check.
//!
//! # The plane is named, not observed
//!
//! `model::trace_of(sku)` takes a `model_ir::kernels::Backend`, and
//! [`Baked::of`] takes the plane whose arm to hand it as a TYPE PARAMETER --
//! `Baked::of::<Metal>(sku)`. That one argument is the whole of what used to be
//! the binding: which plane's claim tables `sweep::resolve` joins a lane's
//! points against, and therefore which lanes bind at all. What a bank is stored
//! as rides on the plan's own `repr` column and is read at the slot
//! (`BoundOp::form`), so a driver no longer measures a quantisation and hands
//! its answer back to the catalog.

use std::collections::BTreeMap;

use model_compiler::program::{Program, Refusal};
use model_ir::plan::{FireClass, Plan};

use crate::walk::resolve;
use crate::walk::{Plane, Unresolved};

/// The checkpoint flavor a baker driver's reader can produce from.
///
/// `model::snapshot::Snapshot` is a safetensors reader and nothing else, so a
/// SKU whose only import is `gguf-bf16` cannot be built here however legitimate
/// that import is. Matched as a PREFIX because the flavor names carry the
/// storage dtype (`safetensors-bf16`), which is the import's business and not
/// the reader's.
pub const READABLE_BASE: &str = "safetensors";

/// Row alignment inside the weight arena, in bytes.
///
/// 256 is `model/src/bin/baker_load.rs`'s figure and cuda's, and it is the
/// load's own arithmetic rather than a kernel requirement: every produced
/// tensor is dense and row-major, so a bank only needs a region, and 256 keeps
/// every one of them on a cache line and on any vectorised load's natural
/// boundary.
///
/// EACH SHADER PLANE WANTS IT FOR ONE MORE REASON, and the two are not the same
/// reason. Metal's `setBuffer:offset:` takes a byte offset whose required
/// alignment is the device's, which 256 clears everywhere. WebGPU's is harder:
/// a storage binding's offset must be a multiple of
/// `min_storage_buffer_offset_alignment`, which the spec FLOORS at 256 -- so an
/// arena packed at 256 clears the worst case rather than the local one, and a
/// checkpoint laid out here loads on an adapter that asks for the maximum. A
/// binding that failed it is a validation error, not a slow path.
pub const BANK_ALIGN: u64 = 256;

/// A weight on the device: a region, its shape, and the element the CHECKPOINT
/// stores it at.
///
/// The plan's `repr` column is not this, and the gap is measured rather than
/// papered over. A model text declares qwen's `a_log` and its gdn norm at the
/// activation dtype and the checkpoint ships both F32; `produce` reports the
/// storage, and the claim bodies agree with the CHECKPOINT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bank<S> {
    /// Where the bytes are, and how many there are.
    ///
    /// A REGION AND NOT AN ADDRESS, which is the one thing this struct says
    /// differently from cuda's. A shader plane binds with an extent and a
    /// binding past its end is an error rather than a fault, so the size the
    /// load computed travels with the region to the slot that binds it.
    pub slice: S,
    /// The extent the plan's parameter table registered, which is what a
    /// `Const` slot's width rules read (`Width::Axis(at, axis)`).
    pub shape: Vec<u64>,
    /// The element the CHECKPOINT stores it at.
    pub dtype: model::produce::Dtype,
    /// The plan's own `repr` column for this parameter, which the storage dtype
    /// above cannot stand in for and does not try to.
    ///
    /// A QUANTISED BANK'S FORM LIVES ONLY HERE. `mxfp4` codes and `e8m0` block
    /// exponents are both `U8` on disk, so `dtype` tells the two planes of one
    /// bank apart in neither direction; what says which is which is the name
    /// the model text declared them under and the repr it declared them at.
    /// `BoundOp::form` reads this and nothing else.
    pub repr: String,
}

/// Where a lane's `Program`s come from, and everything the load settled about
/// them.
///
/// THE PLAN SOURCE, and it is deliberately not a device thing. Tracing a
/// catalog row, binding its lanes and asking the plane whether it claims every
/// point are three questions that need no GPU and no adapter, so a build with
/// neither can answer all three -- which is what `trace <sku> <plane> | lanes`
/// reports and what each driver's own tests check.
///
/// IT CARRIES NO PLANE, which is why it is not generic. What a plane decided is
/// already inside `plan.plane`, put there by the trace [`Baked::of`] ran; the
/// structure that survives is the plan and the lanes bound against it, and both
/// read the same on every backend.
#[derive(Debug)]
pub struct Baked {
    /// Which catalog row this is -- the SKU, not the checkpoint's own id.
    pub sku: &'static str,
    /// The traced plan. Held whole because the fire reads it: `plan.ops` is
    /// what a `Step` indexes.
    pub plan: Plan,
    /// Every lane, bound or refused, in `sweep::lanes` order.
    ///
    /// KEPT WHOLE AND NOT NARROWED TO THE ONE THAT FIRES, which is what makes
    /// the refusals a report rather than a failure: a hybrid whose prefill leg
    /// states a point this plane does not claim still decodes, and a structure
    /// that kept only the built lanes would report it as unrunnable -- and on a
    /// plane where every lane refuses, would hold nothing at all and have
    /// nothing to say about why.
    pub lanes: Vec<Result<Program, Refusal>>,
    /// What the tower is, read off the same plan.
    pub deployment: model::deployment::Deployment,
    /// The value the plan's `out` seam names.
    pub out: model_ir::plan::ValueId,
}

impl Baked {
    /// Trace a catalog row for THE PLANE `P` NAMES, bind its lanes, and read the
    /// deployment off the same plan.
    ///
    /// No checkpoint, no device, no allocation: this is the half of a load that
    /// is a fact about the catalog, and it is separated from the half that is a
    /// fact about a file so that both can be asked on a machine with neither.
    ///
    /// # Errors
    ///
    /// A SKU that is not a catalog row, a plan with no `out` seam, or a
    /// deployment `model::deployment` refuses to read -- each naming what it
    /// could not answer.
    pub fn of<P: Plane>(sku: &str) -> Result<Self, String> {
        let row = model::serve::row(sku).ok_or_else(|| {
            format!(
                "`{sku}` is not a row of `model::catalog()`; did you mean {}?",
                model::serve::nearest_ids(sku, 3).join(", "),
            )
        })?;
        let trace = model::trace_of(row.id)
            .ok_or_else(|| format!("`{}` is not a row of `model::catalog()`", row.id))?;
        // THE ONE ARGUMENT THAT USED TO BE A BINDING. Naming the plane is what
        // decides which claim tables `sweep::resolve` joins against, and
        // therefore which of this row's lanes bind at all.
        let plan = trace(P::BACKEND);
        let lanes = model_compiler::program::bound(&plan);
        let out = plan
            .seams
            .iter()
            .find(|s| s.seam == model_ir::seam::OUT.name)
            .and_then(|s| s.values.first().copied())
            .ok_or_else(|| format!("`{}` states no `out` seam", row.id))?;
        let deployment = model::deployment::Deployment::of(
            &plan,
            model::deployment::Advertised {
                arch: row.arch,
                max_model_len: row.max_model_len,
            },
        )
        .map_err(|why| format!("`{}`: {why}", row.id))?;
        Ok(Self {
            sku: row.id,
            plan,
            lanes,
            deployment,
            out,
        })
    }

    /// The lane that serves a fire of `class` carrying (or not) a user mask, and
    /// the fact word that picked it.
    ///
    /// A REFUSAL AND NOT A PANIC when no lane serves: a driver's next line is
    /// somebody else's request.
    ///
    /// # Errors
    ///
    /// The sentence to print, naming the class and what the lanes said.
    pub fn lane(&self, class: FireClass, masked: bool) -> Result<(u64, &Program), String> {
        let word = word_of(&self.plan, class, masked)?;
        let mut refused = Vec::new();
        for lane in &self.lanes {
            match lane {
                Ok(p) if p.words.contains(&word) => return Ok((word, p)),
                Ok(_) => {}
                Err(r) => {
                    if r.words.contains(&word) {
                        return Err(format!(
                            "the {} lane of `{}` refuses: {r}",
                            class.suffix(),
                            self.sku
                        ));
                    }
                    refused.push(r.lane);
                }
            }
        }
        Err(format!(
            "no bound lane of `{}` serves the {} word {word:#b}; lanes {refused:?} refused",
            self.sku,
            class.suffix()
        ))
    }

    /// Every step of every bound lane, checked against `P`'s claim tables. See
    /// [`resolve::check`].
    #[must_use]
    pub fn unresolved<P: Plane>(&self) -> Vec<Unresolved> {
        self.lanes
            .iter()
            .filter_map(|l| l.as_ref().ok())
            .flat_map(|p| resolve::check::<P>(&self.plan, p))
            .collect()
    }
}

/// The fact word a fire of `class` sets, against the facts THIS text states.
///
/// A HAND MATCH ON THE FACT NAME STOOD HERE and was one of four copies — this
/// crate's, cuda's, metal's and the smoke's. They disagreed with each other
/// (cuda carried a 64-bit guard; the two shader planes did not) and with the
/// DECLARATION they implement. `model_ir::facts` is the closed vocabulary and
/// the one derivation now.
///
/// Re-exported rather than wrapped so `baker::word_of` keeps answering, which
/// is the path both drivers' call sites and their tests already ask for.
pub use model_ir::facts::word_of;

/// Where every produced tensor lands in one arena, and how big the arena is.
///
/// ONE ARENA AND NOT ONE ALLOCATION PER TENSOR, and every plane has its own
/// reason. Cuda does it for the address space. Metal does it for the residency
/// set: an argument table binds a BUFFER and a residency set tracks one, so 260
/// buffers would be 260 residency entries and 260 chances for the allocator's
/// live-byte accounting to disagree with the device's. WebGPU does it for
/// `max_storage_buffers_per_shader_stage` and the binding model -- 260 objects
/// to keep alive, 260 bind-group entries to build against, and 260 chances for
/// a `BufferBinding` to name the wrong one. One buffer with 260 offsets into it
/// is one object and one arithmetic.
///
/// THE PLACEMENT HAS NO DECISION IN IT, which is what makes this short: every
/// produced tensor is dense, row-major and canonical, so one contiguous upload
/// per bank and no restride, no repack, no cast.
#[must_use]
pub fn arena_of(produced: &[(String, model::produce::HostTensor)]) -> (Vec<u64>, u64) {
    let mut at = 0u64;
    let mut offsets = Vec::with_capacity(produced.len());
    for (_, t) in produced {
        offsets.push(at);
        at += (t.bytes.len() as u64).div_ceil(BANK_ALIGN) * BANK_ALIGN;
    }
    (offsets, at.max(1))
}

/// The join `baker_load` proves, restated as a load-time precondition.
///
/// A missing bank would be a zero-length binding at a `Const` slot and a shader
/// reading nothing, which is the worst place to find out. A bank whose SHAPE
/// disagrees is worse still: it binds, it runs, and it answers.
///
/// # Errors
///
/// The list of params the import does not satisfy, each with what the plan
/// wanted and what arrived.
pub fn join<S>(plan: &Plan, banks: &BTreeMap<String, Bank<S>>) -> Result<(), String> {
    let mut missing = Vec::new();
    for p in &plan.params {
        match banks.get(&p.name) {
            None => missing.push(p.name.clone()),
            Some(b) if b.shape != p.shape => missing.push(format!(
                "{} (plan wants {:?}, import produced {:?})",
                p.name, p.shape, b.shape
            )),
            Some(_) => {}
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} param(s) the import does not satisfy: {missing:?}",
        missing.len()
    ))
}

/// Which import flavor of `sku` a baker driver's reader can produce from.
///
/// # Errors
///
/// A SKU offering no `safetensors*` import, naming what it does offer.
pub fn readable_base(sku: &str) -> Result<&'static str, String> {
    model::bases_for(sku)
        .into_iter()
        .find(|b| b.starts_with(READABLE_BASE))
        .ok_or_else(|| {
            format!(
                "`{sku}` names no `{READABLE_BASE}*` import; this driver's reader is \
                 safetensors and the SKU offers {:?}",
                model::bases_for(sku)
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plan with the facts a test states and nothing else.
    ///
    /// `Backend::Metal` is a FIXTURE VALUE and not a claim: nothing in this
    /// module reads `plan.plane`, and the word is a bitmask over the facts a
    /// text declared in the order it declared them. Both drivers' copies of
    /// these tests differed in exactly this literal, which is what said they
    /// were one test.
    fn plan_of(facts: &[&str]) -> Plan {
        Plan {
            name: "a-test-row".into(),
            plane: model_ir::kernels::Backend::Metal,
            facts: facts.iter().map(|f| (*f).to_string()).collect(),
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            seams: Vec::new(),
        }
    }

    /// The decode word is the one a decode fires, and the class picks it.
    ///
    /// AGAINST `model_ir::facts::word_of`, which this crate re-exports and
    /// which states no tests of its own. These two rows were `driver-metal`'s
    /// and `driver-wgpu`'s, identical but for a `Backend` literal, and they
    /// are the only readers the floor's derivation has: they stay here because
    /// they are what a lane is picked by, and they belong beside it until
    /// `model-ir` grows a suite of its own.
    #[test]
    fn the_decode_word_sets_qo_one_and_prefill_clears_it() {
        let mut plan = plan_of(&["qo_one"]);
        assert_eq!(word_of(&plan, FireClass::Decode, false), Ok(0b1));
        assert_eq!(word_of(&plan, FireClass::Prefill, false), Ok(0b0));

        let why = word_of(&plan, FireClass::Decode, true).expect_err("refused");
        assert!(why.contains("`masked` fact"), "{why}");

        plan.facts.push("a_fact_nobody_declared".into());
        assert!(
            word_of(&plan, FireClass::Decode, false).is_err(),
            "an unknown fact refuses rather than reading as a clear bit",
        );
    }

    /// A text that DOES branch on the mask gets the lane it asked for.
    #[test]
    fn a_text_that_states_masked_selects_on_it() {
        let plan = plan_of(&["qo_one", "masked"]);
        assert_eq!(word_of(&plan, FireClass::Decode, false), Ok(0b01));
        assert_eq!(word_of(&plan, FireClass::Decode, true), Ok(0b11));
        assert_eq!(word_of(&plan, FireClass::Prefill, true), Ok(0b10));
    }

    /// Every SKU a baker driver can serve can be built from a checkpoint its
    /// reader can open.
    ///
    /// The failure this prevents is a row that traces, binds and fires and then
    /// cannot be loaded at all, because its only import is a flavor
    /// `model::snapshot` does not open. It is asked once here rather than once
    /// per driver because [`READABLE_BASE`] is one answer: every baker driver
    /// reads through `model::snapshot`, which is a safetensors reader.
    #[test]
    fn every_servable_sku_has_a_readable_import() {
        let mut unreadable = Vec::new();
        for row in model::serve::ROWS {
            // A tensor-parallel row is the same bytes cut at load and names no
            // import of its own; that is the table's rule, not a gap.
            if model::bases_for(row.id).is_empty() {
                continue;
            }
            if readable_base(row.id).is_err() {
                unreadable.push(row.id);
            }
        }
        assert!(
            unreadable.is_empty(),
            "these SKUs offer no `{READABLE_BASE}*` import: {unreadable:?}",
        );
    }

    /// The arena is packed in order with every bank on a 256-byte boundary, and
    /// it is never zero long.
    #[test]
    fn the_weight_arena_aligns_every_bank_and_is_never_empty() {
        let (offsets, bytes) = arena_of(&[]);
        assert!(offsets.is_empty());
        assert_eq!(bytes, 1, "a model with no params still needs an allocation");
    }
}
