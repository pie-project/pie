//! **THE EXPORT SEAM, AND THE TWO OP-VOCABULARY SCANS BESIDE IT** — pure IR
//! analysis, and none of it is call order.
//!
//! It lived in [`serve`](crate::serve) because [`Shell::load`] is what asks
//! the questions, and a file whose header says it has no logic was carrying
//! four passes over the trace and the template. Nothing here touches a
//! device, a stream or a fire: given a [`Trace`] and the [`CompiledModel`]
//! that baked it, each function answers one question about WHICH CLASSES RUN
//! WHICH NODE, which is a fact about the artifact and is true before a device
//! is bound.
//!
//! * [`Exports::of`] — the declared seams (`out`, `mtp`, `attn.scores`)
//!   resolved to values, layers and the class sets that fill them.
//! * [`masked_classes`] / [`corrected_classes`] — the same reading taken from
//!   the OP VOCABULARY rather than from a seam: which classes' windows run an
//!   `attention.masked` arm, and which run a `linear.lora_correct` one.
//! * [`regions_shifting`] — the op vocabulary read a third time and answered
//!   PER REGION instead of per class: which regions hold nothing but ops that
//!   address off the staged seat's start ([`crate::SHIFTED`]), and can
//!   therefore be replayed somewhere other than the fire's row zero.
//!
//! [`Shell::load`]: crate::serve::Shell::load

use model_compiler::CompiledModel;
use model_ir::{Trace, ValueId};

use crate::error::{Fault, Result};

/// The names `model_dsl::seam` states for the values a reader touches after
/// the graph has run — `out`, `mtp`, `attn.scores`, in that order.
///
/// **READ FROM THE COMPILER, NOT SPELLED AGAIN** (palo C3b). This crate does
/// not depend on the authoring surface, and until this wave it kept its own
/// copy of the literal `"out"` with a comment in each place saying the other
/// one existed. `model_compiler::arena` is what gives these values their
/// delivery tail, so it is the honest place for the list to live: a shell
/// reading a name the carve does not pin would be reading bytes the carve was
/// free to give away.
pub(crate) const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];
pub(crate) const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];
pub(crate) const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];

/// One declared export, resolved against this load's plan and bake.
///
/// **A VALUE AND THE CLASSES THAT FILL IT, AND BOTH HALVES ARE USED.** The
/// value is what the fire's carve turns into a rectangle; the class set is
/// what a lane's word is checked against, because an export is written by an
/// ARM and an arm runs over a window. `Shell::masked` and `Shell::corrected`
/// are the same reading taken from the op vocabulary; this one is taken from
/// the seam, because a draft head's attention and a trunk layer's attention
/// are the same `Attention::Prefill` variant and only the export tells them
/// apart.
#[derive(Debug, Clone)]
pub struct Export {
    /// The exported value, as the plan's `Seam` row names it.
    pub value: ValueId,
    /// Which transformer layer it came from, for a per-layer export.
    pub layer: u32,
    /// The classes whose window runs the node that writes it.
    pub classes: model_ir::ClassSet,
}

/// This load's declared exports (design §9), resolved once at boot.
#[derive(Debug, Clone)]
pub(crate) struct Exports {
    /// The trunk's logits. Required: a plan with no `out` seam computes
    /// nothing a reader can take.
    pub(crate) out: ValueId,
    /// The draft head's logits over the draft window, for a SKU whose model
    /// text declares one (palo C3).
    pub(crate) mtp: Option<Export>,
    /// The attention's per-query mass, one entry per attention layer that
    /// exports it, in the plan's own order (palo C4).
    pub(crate) scores: Vec<Export>,
    /// The union of every capture column's classes — the set a capturing
    /// lane's word must land in, and empty for an artifact with no capture
    /// arm at all.
    pub(crate) capturing: model_ir::ClassSet,
}

impl Exports {
    /// Resolve the export seams against a plan and the bake that placed them.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a plan with no `out` seam.
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"
                ),
            })?;
        let named = |name: &str| -> Vec<Export> {
            trace.seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| {
                    let layer = seam.layer.unwrap_or(0);
                    seam.values
                        .iter()
                        .map(move |value| (layer, *value))
                })
                .map(|(layer, value)| Export {
                    value,
                    layer,
                    classes: writer_classes(trace, compiled, value),
                })
                .collect()
        };
        let scores = named(SCORES_SEAM);
        let mut capturing = model_ir::ClassSet::default();
        for export in &scores {
            for class in export.classes.iter() {
                capturing.insert(class);
            }
        }
        Ok(Exports {
            out,
            mtp: named(MTP_SEAM).into_iter().next(),
            scores,
            capturing,
        })
    }
}

/// The classes whose window runs the node that writes `value`.
///
/// **THE NODE, NOT THE OP NAME.** An export is told apart from the trunk by
/// WHAT IT IS, not by which kernel wrote it: the draft head's readout and the
/// trunk's are both `linear.lm_head`, and the capture arm's output and a
/// pooled attention's are both `[rows, heads]` F32. Asking which regions hold
/// the writing node is the one reading that cannot be fooled by a model text
/// reusing an op.
fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    use model_ir::Operands;
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}


/// The classes whose window runs an `attention.masked` arm.
///
/// **WHETHER THIS ARTIFACT HAS ANYWHERE FOR A MASK TO GO.** `masked` is a
/// fact the model declares (design §8), so a plan with no `attention.masked`
/// arm cannot serve one, and accepting the bits anyway would answer with the
/// unmasked continuation.
///
/// A CLASS SET rather than a boolean, because the question a fire asks is per
/// lane: does the class this lane's word resolved to run the masked arm? The
/// word and the mask are stamped at two instants by two parties — the runtime
/// computes the word from the model's `Classify::of`, the caller states the
/// mask — and this set is what lets the shell check that they agree
/// (`Fault::{Maskless, MaskWord}`).
#[must_use]
pub(crate) fn masked_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Attention(model_ir::Attention::Masked { .. })
        )
    })
}

/// The classes whose window runs a `linear.lora_correct` arm.
///
/// [`masked_classes`]'s adapter-axis twin, read off the bake for the same
/// reason and checked against a submission the same way, with the same three
/// consequences: an artifact with no correction op has nowhere for an adapter
/// id to go (`Fault::Adapterless`), a lane whose word puts it outside the
/// correction's window may not carry one and a lane whose word puts it inside
/// must (`Fault::AdapterWord`), and a fire in whose composition NO class of
/// this set has rows never stages the routes vector, never binds the seat,
/// and never launches the arm.
#[must_use]
pub(crate) fn corrected_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Linear(model_ir::Linear::LoraCorrect { .. })
        )
    })
}

/// The classes whose window runs an `attention.decode` arm — the DECODE
/// classes, in the only vocabulary this shell has for the word.
///
/// [`masked_classes`]'s third twin, read off the same template for the same
/// reason, and it exists because the bodies path's load-time arming
/// (`Shell::arm_bodies`) has to synthesize a decode composition before any
/// fire has shown it one. A shell cannot compute a lane's fact word — the
/// word comes from the model's own `Classify::of`, runtime-side (decision
/// #18), and which bit is `qo_one` stays the model's business — so "the
/// decode class" cannot be asked as a question about bits. It can be asked as
/// a question about OPS, and this is that question: a class whose window runs
/// the one-query-row attention arm is a class a decode lane lands in, and
/// `Class::word` then names a word that resolves back to it.
///
/// A SET rather than one class, because a bake may split the decode world
/// further — masked decode beside plain, corrected beside uncorrected — and
/// each of those is a composition of its own with a body of its own. Empty
/// for a plan with no `attention.decode` arm at all (a purely recurrent text,
/// say), and then load-time arming has nothing to aim at and arms nothing.
#[must_use]
pub(crate) fn decoding_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Attention(model_ir::Attention::Decode { .. })
        )
    })
}

/// **WHICH TEMPLATE REGIONS CAN MOVE THEIR OWN BASE** — one `bool` per
/// region of [`CompiledModel::template`], in region order, `true` when EVERY
/// op in it is named by [`crate::SHIFTED`].
///
/// [`masked_classes`]'s and [`corrected_classes`]'s structural twin — the
/// same walk of the same template testing the same node ops — and it differs
/// in exactly two ways, both forced by what the answer is for. It asks ALL
/// rather than ANY, because one guard-only op in a region addresses the wrong
/// row for the whole region's launch; and it answers PER REGION rather than
/// per class, because the thing that gets a seat is a region's launch and the
/// thing that reads it is a region's kernel. A class set could not say it: two
/// classes share a region, and it is the region that either moves or does not.
///
/// **A REGION WITH NO NODES IS `true`, AND IT IS NOT MERELY THE VACUOUS
/// ANSWER.** `all` over an empty range is `true` for free, and here that is
/// also what the question means: a region carries a window because of its
/// MASK — `Windows::of` cuts one per template region off the class table and
/// never looks at `Region::nodes` — so an empty region can hold a windowed
/// rectangle and still launch nothing over it. Nothing in it can address a
/// row, so nothing in it can address the wrong one, and refusing it would
/// refuse a body over a region that computes no bytes. The compiler ships no
/// such region; this says what would be true of one.
///
/// A node index the trace does not hold reads as NOT shifting, which refuses
/// the region. The two tables are baked together and that cannot happen; if it
/// ever does, the narrow reading is the one that stays sound.
#[must_use]
pub(crate) fn regions_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace.nodes.get(node as usize).is_some_and(|node| {
                    let name = model_ir::Operands::name(&node.op);
                    crate::SHIFTED.contains(&name) || crate::PLANNED.contains(&name)
                })
            })
        })
        .collect()
}

/// The union of the region masks whose regions run a node `wanted` accepts.
///
/// The one shape both readings above are: a region is a window over classes,
/// a node is inside a region or it is not, and the answer is which classes'
/// windows carry at least one node of the family asked about.
fn classes_running(
    trace: &Trace,
    compiled: &CompiledModel,
    wanted: impl Fn(&model_ir::Operation) -> bool,
) -> model_ir::ClassSet {
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        let runs = region.nodes.clone().any(|node| {
            trace
                .nodes
                .get(node as usize)
                .is_some_and(|node| wanted(&node.op))
        });
        if runs {
            for class in region.mask.iter() {
                classes.insert(class);
            }
        }
    }
    classes
}

