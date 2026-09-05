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
//!   address off the staged seat's start ([`crate::shifted`]), and can
//!   therefore be replayed somewhere other than the fire's row zero.
//! * [`regions_lane_shifting`] — the same reading one AXIS over
//!   ([`crate::lane_shifted`]): which regions hold nothing but ops that find
//!   their own LANE inside the fire, and can therefore be replayed somewhere
//!   other than the fire's lane zero.
//!
//! [`Shell::load`]: crate::serve::Shell::load

use model_compiler::CompiledModel;
use model_ir::{Operands, Trace, ValueId};

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

/// The classes whose window SCATTERS TOWER OUTPUT INTO TOKEN ROWS — the
/// MEDIA classes, in the only vocabulary this shell has for the word (the
/// multi-unit bodies wave).
///
/// [`masked_classes`]'s fourth twin, and it exists for
/// [`decoding_classes`]'s reason exactly: the bodies path's load-time arming
/// (`Shell::arm_bodies`) has to synthesize a fire that carries an IMAGE
/// before any caller has shown it one, and a shell cannot compute a lane's
/// fact word — the word is the model's `Classify::of`, runtime-side, and
/// which bit is `media` stays the model's business (multimodal §15). So the
/// question is asked about OPS: a class whose window runs the embed merge is
/// a class an image lane lands in, and `Class::word` then names a word that
/// resolves back to it.
///
/// **THE MERGE AND NOT THE TOWER.** The tower's own regions are on the PATCH
/// axis and their rectangles are `Dim::Patches`; a class does not "run" them
/// in the sense this predicate means, because an axis-empty fire simply does
/// not launch that unit. What a media lane's class does run is the scatter
/// that puts the tower's soft tokens onto that lane's placeholder rows —
/// `layout.scatter_rows` or its dropping form — which is a TRUNK-unit node
/// with a class mask, guarded on the media fact. That guard is the thing this
/// reads.
///
/// Empty for every text-only artifact, and then the arming pass's tower arm
/// enumerates nothing at all — which is the same nothing a plan with no
/// decode arm gives the decode arm, and is why neither needs a second clause
/// anywhere.
#[must_use]
pub(crate) fn media_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Layout(
                model_ir::Layout::ScatterRows { .. } | model_ir::Layout::ScatterLiveRows { .. }
            )
        )
    })
}

/// Every request shape, classified by the model: per class, the requests
/// that land in it, fewest flags first. A class no request reaches is one
/// no caller can bring, and the arming pass does not synthesize it.
#[must_use]
pub(crate) fn landing_requests(
    classify: model_ir::ClassifyFn,
    classes: &model_ir::ClassTable,
) -> Vec<Vec<model_ir::Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    for bits in 0..128u32 {
        let request = model_ir::Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
            .adapted(bits & 4 != 0)
            .drafting(bits & 8 != 0)
            .capturing_scores(bits & 16 != 0)
            .with_media(bits & 32 != 0)
            .denoising(bits & 64 != 0);
        let word = classify(&request) & classes.mask;
        if let Some(class) = classes.class_of(word) {
            landing[class].push(request);
        }
    }
    for requests in &mut landing {
        requests.sort_by_key(request_flags);
    }
    landing
}

fn request_flags(request: &model_ir::Request) -> u32 {
    u32::from(request.query_len() != 1)
        + u32::from(request.has_custom_mask())
        + u32::from(request.has_adapter())
        + u32::from(request.drafts())
        + u32::from(request.captures_scores())
        + u32::from(request.has_media())
        + u32::from(request.denoise())
}

/// The DECODE classes: every request that lands in one carries a single
/// row, so a lane of it is one row and its rung is the lane ceiling.
#[must_use]
pub(crate) fn decoding_of(landing: &[Vec<model_ir::Request>]) -> model_ir::ClassSet {
    model_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| {
                !requests.is_empty() && requests.iter().all(|request| request.query_len() == 1)
            })
            .map(|(class, _)| class),
    )
}

/// **WHICH TEMPLATE REGIONS CAN MOVE THEIR OWN BASE** — one `bool` per
/// region of [`CompiledModel::template`], in region order, `true` when EVERY
/// op in it is named by [`crate::shifted`].
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
                    crate::shifted(name) || crate::PLANNED.contains(&name)
                })
            })
        })
        .collect()
}

/// **WHICH TEMPLATE REGIONS FIND THEIR OWN LANE** — one `bool` per region of
/// [`CompiledModel::template`], in region order, `true` when every op in it
/// either is named by [`crate::lane_shifted`] / [`crate::PLANNED`] or NAMES
/// NOTHING THAT IS LANE-INDEXED.
///
/// [`regions_shifting`]'s twin one axis over, and everything that function's
/// note argues about ALL-rather-than-ANY, about answering per REGION rather
/// than per class, and about an empty region reading `true` holds here word
/// for word.
///
/// **AND IT IS AN OPERAND WALK WHERE ITS TWIN IS A NAME LOOKUP, WHICH IS THE
/// ONE REAL DIFFERENCE AND IS FORCED BY WHERE THE HAZARD LIVES.** The row
/// axis's hazard is what a KERNEL does with the pointer it is handed, so only
/// a name can answer it. The lane axis's hazard is what THIS SHELL hands over:
/// `Run::pool` advances the page bounds and last-page fills by `lane_offset`,
/// `Run::recurrent` advances the slot map, the fold predicate and the commit
/// length, and `Run::cut`'s lane column advances every operand whose leading
/// `Dim` counts lanes or images. Those are the three doors, and an op reaches
/// them through its OPERANDS — a `Def::Cache` space for the first two, a lane
/// -shaped rectangle for the third. So an op that names neither cannot be
/// handed a `lane_offset`-baked pointer at all, whatever it is called, and
/// refusing it would cost a body for no hazard.
///
/// **WHICH LEAVES EXACTLY TWO WAYS TO PASS**, and they are the two
/// [`crate::lane_shifted`] enumerates and the one this walk adds:
///
/// * the op is on that list, so the tables it names are handed over WHOLE and
///   it finds its lane in a staged datum or off the seat's `win[3]`;
/// * or the op is a planner ([`crate::PLANNED`]), which puts no node in the
///   captured graph and rebuilds its schedule every fire against that fire's
///   own staged geometry, lane offset included;
/// * or the op names nothing lane-indexed, and the question does not arise.
///
/// A node index the trace does not hold, an operand the trace does not
/// declare, and an op family this walk cannot collect all read as NOT
/// lane-shifting — which refuses the region. That is the safe direction on the
/// axis where being wrong reads another lane's state, and it is
/// [`regions_shifting`]'s own tie-break.
#[must_use]
pub(crate) fn regions_lane_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .all(|node| lane_shifting_node(trace, node))
        })
        .collect()
}

/// One node's answer for [`regions_lane_shifting`] — the name lookup first,
/// because it is the cheap one and because a name on the list has already
/// argued its operands.
fn lane_shifting_node(trace: &Trace, node: u32) -> bool {
    let Some(node) = trace.nodes.get(node as usize) else {
        return false;
    };
    let name = Operands::name(&node.op);
    if crate::lane_shifted(name) || crate::PLANNED.contains(&name) {
        return true;
    }
    // **INPUTS AND OUTPUTS BOTH**, `window::copyable`'s reason exactly: a
    // rectangle this op WRITES is resolved through the same `Run::cut` the
    // ones it reads are, and a lane-shaped output would be advanced by the
    // same number.
    let mut operands: Vec<ValueId> = Vec::new();
    node.op.inputs(&mut operands);
    node.op.outputs(&mut operands);
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        // **A CACHE SPACE IS THE FIRST DOOR, PAGED AND RECURRENT ALIKE.** Both
        // `Run::pool` and `Run::recurrent` slice their per-lane tables at
        // `lane_offset`, and only the absolute doors beside them
        // (`pool_absolute`, `recurrent_absolute`) do not — which is what the
        // names on the list took and what nothing off it did.
        if matches!(&decl.def, model_ir::Def::Cache(_)) {
            return false;
        }
        // **AND A LANE-SHAPED RECTANGLE IS THE SECOND**, whatever declared it:
        // `Run::cut`'s lane column is `(span.lane_offset, span.lanes + k)` for
        // every one of these, which is a pointer advanced by a number the key
        // does not fix. `GeomKind::Indices` is spelled `Dim::Lanes` and cut is
        // excluded from slicing it — but its BOUNDS are not, and an op naming
        // one names the other, so nothing is bought by carving an exception
        // here.
        let model_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            // A plan payload is host state resolved through `Run::slot`, and
            // its own window is the region that BUILT it — which is this one,
            // because a schedule may only be read where it was built
            // (`model`'s `no_schedule_straddles_its_readers`).
            return true;
        };
        !matches!(
            shape.first(),
            Some(
                model_ir::Dim::Lanes
                    | model_ir::Dim::LanesPlus(_)
                    | model_ir::Dim::Images
                    | model_ir::Dim::ImagesPlus(_)
            )
        )
    })
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
