//! The lane: a `model_compiler::program::Program` per fire class, built at
//! load and fired by [`crate::fire::launch`].
//!
//! # What this is
//!
//! `baker-smoke` proved the whole chain outside the driver — text → `Plan`
//! → lane → `Program` → GPU → argmax, A/B-matched against the legacy
//! driver, one request and one fire. This module is that capability moved
//! to where real serving happens. The smoke is the executable spec and is
//! cited by file and line wherever a decision it documented is re-made
//! here; where the driver's answer differs from the smoke's, the difference
//! is stated at the difference and never silently taken.
//!
//! # INSTEAD OF, not beside (R2)
//!
//! This module arrived behind `[driver] baker` / `PIE_BAKER`, building a
//! second lane BESIDE a legacy one that still fired by default, so that an
//! A/B was one process and one checkpoint parse apart. The A/B is done: its
//! answers are banked in `tests/baker_serve.rs` and the legacy path is
//! deleted. The knob is gone, this is the only lane, and a SKU that does
//! not reach one is refused at `load_model` with the reason named.
//!
//! What that ended, and it was the cost of the arrangement: the second
//! weight residency. `produce`'s bytes were uploaded on top of the legacy
//! manifest's — 1.40 GiB twice for qwen35-d0.8b — because both loads ran.
//! Only [`load`] runs now.
//!
//! # ONE CATALOGUE (R3)
//!
//! The pools used to be the OTHER catalogue's: `model_legacy::deployment`
//! sized and strided the KV pages and the recurrent slabs, gated the KV
//! style and the GQA ratio at the door and published the caps, while this
//! lane fired a program traced out of the new one. `Geometry::agrees_with`
//! was the check that stood between two accounts of one checkpoint.
//!
//! There is one account now. `model::deployment::Deployment::of` reads the
//! pool geometry off the same [`Plan`] the program is built from, so the
//! check has nothing left to compare and is deleted with the crate that
//! made it necessary.

use std::collections::BTreeMap;

use model_compiler::program::{Program, Refusal};
use model_ir::plan::Plan;
use model_ir::trace::FireClass;

use crate::device::{Allocator, DeviceBuffer, OwnedStream};

pub(crate) mod bound;
pub(crate) mod fire;
pub(crate) mod marks;
pub(crate) mod resolve;
pub(crate) mod staging;


/// The checkpoint flavor this driver's reader can produce from.
///
/// `model::snapshot::Snapshot` is a safetensors reader and nothing else, so
/// a SKU whose only import is `gguf-bf16` cannot be built here however
/// legitimate that import is. Matched as a PREFIX because the flavor names
/// carry the storage dtype (`safetensors-bf16`), which is the import's
/// business and not the reader's.
const READABLE_BASE: &str = "safetensors";

/// Row alignment inside the weight arena, in bytes.
///
/// 256 is `model/src/bin/baker_load.rs`'s figure, and it is the load's own
/// arithmetic rather than a kernel requirement: every produced tensor is
/// dense and row-major, so a bank only needs an address, and 256 is what
/// keeps every one of them on a cache line and on any vectorised load's
/// natural boundary.
const BANK_ALIGN: usize = 256;

/// A weight on the device: an address, its shape, and the element the
/// CHECKPOINT stores it at.
///
/// The plan's `repr` column is not this, and the gap is measured rather
/// than papered over. A model text declares qwen's `a_log` and its gdn norm
/// at the activation dtype and the checkpoint ships both F32; `produce`
/// reports the storage. The routines agree with the CHECKPOINT --
/// `qwen_gdn_post_conv_prep_bf16` declares `a_log: Const<Tensor<f32>>` --
/// so this table carries the storage dtype and the shim asserts against it
/// (`baker-smoke/src/smoke.rs:479-497`).
#[derive(Clone, Debug)]
pub(crate) struct Bank {
    pub ptr: *mut std::ffi::c_void,
    pub shape: Vec<u64>,
    pub dtype: model::produce::Dtype,
    /// The plan's own `repr` column for this parameter, which the storage
    /// dtype above cannot stand in for and does not try to.
    ///
    /// A QUANTISED BANK'S FORM LIVES ONLY HERE. `mxfp4` codes and `e8m0`
    /// block exponents are both `U8` on disk, so `dtype` tells the two planes
    /// of one bank apart in neither direction; what says which is which is
    /// the name the model text declared them under and the repr it declared
    /// them at. `BoundOp::form` reads this and nothing else.
    pub repr: String,
}

// SAFETY: a device address is a number, never dereferenced on the host; its
// CUDA context outlives the shell that holds it.
unsafe impl Send for Bank {}
unsafe impl Sync for Bank {}

/// Everything the baker lane needs, built once at load.
pub(crate) struct Baked {
    /// Which catalog row this is — the SKU, not the checkpoint's own id.
    pub sku: String,
    /// The traced plan. Held whole because the fire reads it: `plan.ops` is
    /// what a `Step` indexes, and the staging shim walks `plan.values` to
    /// find the statement that wrote an operand it is handed.
    pub plan: Plan,
    /// Every lane, bound or refused, in `sweep::lanes` order.
    ///
    /// KEPT WHOLE AND NOT NARROWED TO THE ONE THAT FIRES, which is what
    /// makes the refusals a report: qwen's prefill leg states
    /// `ssm.gated_delta_chunked`, which no cuda routine claims yet (W2), and
    /// its decode leg states none of it. A structure that kept only the
    /// built lanes would report the hybrid as unrunnable when half of it
    /// runs today — and, since R2, would have made `load_model` refuse a
    /// checkpoint this driver serves.
    pub lanes: Vec<Result<Program, Refusal>>,
    /// Every param the plan named, on the device.
    ///
    /// THE SEAM IS CLOSED: these are `model::produce`'s bytes and not the
    /// legacy manifest's, and they used to be a SECOND residency of the same
    /// checkpoint because both loads ran. `LoadedModel::weights` and the
    /// `weights::stage` that filled it are both gone — R3 deleted the load
    /// contract and `serve::encode`'s towers with it — so this map is the
    /// one weight residency the driver has.
    pub banks: BTreeMap<String, Bank>,
    /// The arena the banks are spans of, held to keep them alive.
    ///
    /// Underscored like `KvState::_held`, and for the same reason: nothing
    /// reads it, and that is the point — every [`Bank`] above is a raw
    /// pointer INTO this allocation, so dropping it frees every weight the
    /// lane is about to fire against. It is the field's existence that is
    /// load-bearing, not its value.
    _owned: Vec<DeviceBuffer>,
    /// The value the plan's `out` seam names.
    pub out: model_ir::plan::ValueId,
}

impl Baked {
    /// The lane that serves a fire of `class` carrying (or not) a user mask,
    /// and the fact word that picked it.
    ///
    /// A REFUSAL AND NOT A PANIC when no lane serves: the smoke could
    /// `panic!` on an unknown fact because the next line was its own exit,
    /// and a driver's next line is somebody else's request.
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
            "no bound lane of `{}` serves the {} word {word:#b}; \
             lanes {refused:?} refused",
            self.sku,
            class.suffix()
        ))
    }
}

/// The fa2 schedules one lane's statements ask the driver to raise.
///
/// A schedule is not a per-statement thing — it is a workspace and a work
/// list planned on the host from the fire's CSRs — so the driver raises one
/// per fire and the lane's statements bind it. What the lane needs is
/// therefore a SET, and the set is read off the lane's own statements
/// rather than off a config: `attention.decode`'s params are
/// `[window, head_dim, sm_scale]` and `attention.prefill`'s are
/// `[window, head_dim, kv_heads, sm_scale]` (`model-dsl/src/kernels.rs`'s
/// `attention` module is where both are spelled).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AttnAsk {
    /// `(head_dim, full_attention)` for the decode schedule, if the lane
    /// states one. `full_attention` is `window == 0` — an unstated window
    /// is what flashinfer spells `window_left = -1`.
    pub decode: Option<(u32, bool)>,
    /// `head_dim` for the prefill schedule, if the lane states one.
    pub prefill: Option<u32>,
}

impl Baked {
    /// What `program`'s statements ask for, walked once per fire.
    ///
    /// # Two widths are a REFUSAL, not a max
    ///
    /// The planner bakes the head dim into the schedule, and this driver
    /// carries ONE decode plan and ONE prefill plan. A lane stating two
    /// widths would have half its statements bound to a work list built for
    /// the other one — every launch succeeding, every pointer in range, the
    /// answers wrong. The legacy path kept a second decode plan for exactly
    /// this case; nothing in `baker::staging` can bind a second, so the
    /// honest answer is to say so.
    ///
    /// # Errors
    ///
    /// A second width at either point, or a statement whose params do not
    /// carry the two this reads.
    pub(crate) fn attn_ask(&self, program: &Program) -> Result<AttnAsk, String> {
        let mut ask = AttnAsk::default();
        for step in &program.steps {
            let Some(op) = self.plan.ops.get(step.op as usize) else {
                continue;
            };
            let param = |at: usize| -> Result<u32, String> {
                op.params
                    .get(at)
                    .and_then(|v| u32::try_from(*v).ok())
                    .ok_or_else(|| {
                        format!("`{}` states no param {at} this driver can read", op.kernel)
                    })
            };
            match op.kernel.as_str() {
                "attention.decode" | "attention.decode_lse" => {
                    let want = (param(1)?, param(0)? == 0);
                    match ask.decode {
                        Some(held) if held != want => {
                            return Err(format!(
                                "`{}` states two decode attention schedules \
                                 ({held:?} and {want:?}) and this driver raises one",
                                self.sku
                            ));
                        }
                        _ => ask.decode = Some(want),
                    }
                }
                // `attention.masked` READS A PREFILL SCHEDULE, and it stood in
                // the decode arm above until the point became a claim body.
                // The routine it resolved through is
                // `dispatch_attention_flashinfer_prefill_custom`, which takes
                // `In<Struct<Fa2Prefill>>` and refuses an unplanned cache
                // (`prefill_plan_usable`); asking for a decode schedule left
                // its lane's prefill workspace unstamped and the point could
                // only ever have refused. Nothing caught it because no
                // executor answered the point at all.
                "attention.prefill" | "attention.prefill_lse" | "attention.masked" => {
                    let want = param(1)?;
                    match ask.prefill {
                        Some(held) if held != want => {
                            return Err(format!(
                                "`{}` states two prefill attention widths \
                                 ({held} and {want}) and this driver raises one",
                                self.sku
                            ));
                        }
                        _ => ask.prefill = Some(want),
                    }
                }
                _ => {}
            }
        }
        Ok(ask)
    }
}

/// The fact word a fire of `class` sets, computed off `plan.facts` rather
/// than assumed.
///
/// Bit `i` is `plan.facts[i]`. `qo_one` — every request's query is one
/// token — is exactly what `fire_class_of` already decided when it compared
/// `rows` to `requests` (`fire/launch.rs:57-68`), so the driver does not
/// re-derive it from the CSR: the two would be free to disagree, and the
/// one that picks the lane must be the one that named the class. `masked` is
/// the frame's `has_user_mask`, which the caller states and nothing derives.
///
/// A fact this does not know is a REFUSAL and not a zero. A zero would be a
/// guess that silently picks a lane, and the lane is the whole program.
///
/// # And a fact the FIRE holds that the TEXT does not name
///
/// That is the other half, and only `masked` needs it. A text that does not
/// branch on `qo_one` serves both cases with the same statements, which is
/// right — the fact is an optimisation the text declined. A text that does
/// not branch on `masked` has ONE attention arm and it is causal, so a masked
/// frame would have its mask staged and then attended over as if it were not
/// there: the right-looking wrong answer. So the mask is refused HERE, where
/// the lane is picked, and the refusal names the text rather than the flag.
fn word_of(plan: &Plan, class: FireClass, masked: bool) -> Result<u64, String> {
    let mut word = 0u64;
    let mut says_masked = false;
    for (bit, fact) in plan.facts.iter().enumerate() {
        if bit >= 64 {
            return Err(format!(
                "`{}` declares {} facts; a fact word is 64 bits",
                plan.name,
                plan.facts.len()
            ));
        }
        let holds = match fact.as_str() {
            "qo_one" => class == FireClass::Decode,
            "masked" => {
                says_masked = true;
                masked
            }
            other => {
                return Err(format!(
                    "`{other}` is a fact this driver cannot answer for a \
                     {} fire; name it in `baker::word_of` or the lane is a \
                     guess",
                    class.suffix()
                ));
            }
        };
        if holds {
            word |= 1 << bit;
        }
    }
    if masked && !says_masked {
        return Err(format!(
            "this frame carries a user attention mask and `{}` states no \
             `masked` fact, so every lane it has attends causally: the mask \
             would be staged and IGNORED, and the request answered as though \
             it had asked nothing",
            plan.name
        ));
    }
    Ok(word)
}

/// Which catalog SKU this snapshot IS.
///
/// # The bridge that STOOD HERE
///
/// `BRIDGE` was a one-row table mapping the legacy catalog's spelling
/// (`"qwen3.5-0.8b-base"`, what `model_legacy::catalog::identify` read off
/// the checkpoint's tensors) to this catalog's (`"qwen35-d0.8b-bf16-kv-bf16"`),
/// because the two catalogues did not share an id space and neither was
/// wrong. R3 deleted the legacy catalog, so there is one id space and
/// nothing to bridge: the checkpoint is matched against the IMPORT tables —
/// the same tensor list `produce` is about to read — and the answer is the
/// SKU itself.
///
/// The stated SKU still outranks the match, and for the reason it always
/// did: a deployment that names one through `[baker] sku` / `PIE_BAKER_SKU`
/// is telling the driver something the tensors cannot say, and a table that
/// silently won would make the knob untestable. It is also how a new row is
/// proven before its checkpoint is one this reader can tell apart.
///
/// # Errors
///
/// A stated SKU that is not a catalog row, or a snapshot no row matches (or
/// that two match) — every one carrying [`model::Unmatched`]'s own account
/// of what it found.
pub(crate) fn identify(
    snapshot: &model::snapshot::Snapshot,
    stated: Option<&str>,
) -> Result<&'static str, String> {
    if let Some(sku) = stated {
        return model::serve::row(sku).map(|row| row.id).ok_or_else(|| {
            format!(
                "`{sku}` is not a row of `model::catalog()`; did you mean {}?",
                model::serve::nearest_ids(sku, 3).join(", "),
            )
        });
    }
    model::identify(&|name| snapshot.shape_of(name)).map_err(|why| why.to_string())
}

/// Trace, bind, produce, upload and resolve — the whole baker load.
///
/// # Errors
///
/// Every way this can fail names what it could not answer. It is a `String`
/// and not a `crate::Error` on purpose: the caller decides whether a baker
/// refusal fails the load or merely leaves the lane unbuilt, and that
/// decision is `serve::load`'s to state, not this function's to force.
pub(crate) fn load(
    sku: &str,
    snapshot: &std::path::Path,
    alloc: &Allocator,
) -> Result<Baked, String> {
    // ── 1. The plan, and every lane it binds to. ────────────────────────
    let trace = model::trace_of(sku)
        .ok_or_else(|| format!("`{sku}` is not a row of `model::catalog()`"))?;
    // `model_dsl::Plane` IS `model_ir::kernels::Backend` (a re-export at
    // `model-dsl/src/lib.rs:20`), so this names the plane without the
    // authoring crate. See the `model-baker` dep line for why that matters.
    let plan = trace(model_ir::kernels::Backend::Cuda);
    let lanes = model_compiler::program::bound(&plan);

    let out = plan
        .seams
        .iter()
        .find(|s| s.seam == model_ir::seam::OUT.name)
        .and_then(|s| s.values.first().copied())
        .ok_or_else(|| format!("`{sku}` states no `out` seam"))?;

    // ── 2. The weights, through `produce` and not the manifest. ─────────
    let base = model::bases_for(sku)
        .into_iter()
        .find(|b| b.starts_with(READABLE_BASE))
        .ok_or_else(|| {
            format!(
                "`{sku}` names no `{READABLE_BASE}*` import; this driver's \
                 reader is safetensors and the SKU offers {:?}",
                model::bases_for(sku)
            )
        })?;
    let import = model::import_of(sku, base)
        .ok_or_else(|| format!("`{sku}` names no `{base}` import"))?;
    // `Snapshot::at` AND NOT `Snapshot::open`: `open` resolves a cache-dir
    // NAME under `$HOME/.cache/huggingface/hub`, and the driver is handed
    // the snapshot directory itself (`ModelLoadDesc::snapshot_dir`). Going
    // through `open` would mean reconstructing a cache name from a path
    // that need not be in a cache at all.
    let snap = model::snapshot::Snapshot::at(snapshot.to_path_buf())
        .ok_or_else(|| format!("no safetensors snapshot at {}", snapshot.display()))?;
    let produced = model::produce::produce(&import, &|n| snap.read(n))
        .map_err(|e| format!("production refused: {e}"))?;

    let (banks, owned) = upload(&produced, &plan, alloc)?;
    drop(produced);

    // The join `baker_load` proves, restated here as a precondition: a
    // missing bank would be a null pointer at a `Const` slot and a fault
    // inside a kernel, which is the worst place to find out.
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
    if !missing.is_empty() {
        return Err(format!(
            "{} param(s) the `{base}` import does not satisfy: {missing:?}",
            missing.len()
        ));
    }

    Ok(Baked {
        sku: sku.to_owned(),
        plan,
        lanes,
        banks,
        _owned: owned,
        out,
    })
}

/// Every produced tensor into one arena, one span each.
///
/// ONE ARENA AND NOT 260 ALLOCATIONS, which is the driver's idiom and not
/// the smoke's: the whole arena is allocated before a byte is read and every
/// tensor is a span inside it — the shape `weights::stage::stage_plan_weights`
/// held before R3 deleted it. The smoke took a `cudaMalloc` per tensor because
/// it had no allocator to reuse; the driver has one, and 260 separate
/// allocations would be 260 chances for the allocator's live-bytes accounting
/// to disagree with the device's.
///
/// THE UPLOAD HAS NO DECISION IN IT, which is what `baker_load` says and
/// what makes this loop short: every produced tensor is dense, row-major
/// and canonical, so one contiguous H2D per bank and no restride, no
/// repack, no cast.
fn upload(
    produced: &[(String, model::produce::HostTensor)],
    plan: &Plan,
    alloc: &Allocator,
) -> Result<(BTreeMap<String, Bank>, Vec<DeviceBuffer>), String> {
    let mut at = 0usize;
    let mut offsets = Vec::with_capacity(produced.len());
    for (_, t) in produced {
        offsets.push(at);
        at += t.bytes.len().div_ceil(BANK_ALIGN) * BANK_ALIGN;
    }
    let arena_len = at.max(1);

    let mut buf = alloc.alloc(arena_len).map_err(|e| {
        format!(
            "the baker weight arena of {arena_len} bytes ({:.2} GiB) did not \
             fit the device: {e:?}",
            arena_len as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })?;
    // A stream of this load's own, exactly as `stage_plan_weights` takes
    // one: `Shell::fire_stream` does not exist until the first launch, and
    // a load that borrowed it would be building an ordering the fire path
    // does not know about.
    let stream = OwnedStream::new(0).map_err(|e| format!("baker upload stream: {e:?}"))?;

    let mut banks = BTreeMap::new();
    for ((name, t), offset) in produced.iter().zip(offsets) {
        buf.write_at(offset, &t.bytes, stream.as_ref())
            .map_err(|e| format!("{name}: H2D of {} bytes: {e:?}", t.bytes.len()))?;
        // `ptr_at` bounds the span against the buffer's own length, which is
        // the check that makes the offset arithmetic above auditable.
        let ptr = buf
            .ptr_at(offset, t.bytes.len())
            .ok_or_else(|| format!("{name}: span at {offset} leaves a {arena_len}-byte arena"))?;
        banks.insert(
            name.clone(),
            Bank {
                ptr,
                shape: t.shape.clone(),
                dtype: t.dtype,
                // The demand side's own column, carried across by name. A
                // produced row the plan binds no param for keeps an empty
                // repr, which is a refusal at a bank slot and no statement
                // can name it anyway.
                repr: plan
                    .params
                    .iter()
                    .find(|p| p.name == *name)
                    .map_or_else(String::new, |p| p.repr.clone()),
            },
        );
    }
    stream
        .as_ref()
        .synchronize()
        .map_err(|e| format!("baker upload: {e:?}"))?;
    Ok((banks, vec![buf]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stated SKU outranks the tensors, and a stated non-row is refused
    /// by name rather than traced into a panic.
    #[test]
    fn a_stated_sku_must_be_a_catalog_row() {
        let empty = model::snapshot::Snapshot::at(std::path::PathBuf::from("/nonexistent"));
        assert!(empty.is_none(), "there is no snapshot at /nonexistent");

        // The stated arm never reads the snapshot, which is what lets this
        // be a host test: it is a catalog lookup and nothing else.
        assert_eq!(
            model::serve::row("qwen35-d0.8b-bf16-kv-bf16").map(|r| r.id),
            Some("qwen35-d0.8b-bf16-kv-bf16"),
        );
        assert!(model::serve::row("qwen3.5-0.8b-base").is_none(), "the legacy spelling is not an id any more");
        assert_eq!(
            model::serve::nearest_ids("qwen35-d0.8b-bf16-kv-bf1", 1),
            vec!["qwen35-d0.8b-bf16-kv-bf16"],
            "a typo is answered with the row it is a typo OF",
        );
    }

    /// Every SKU this driver can serve can be built from a checkpoint this
    /// driver can read.
    ///
    /// The failure this prevents is a row that traces, binds and fires and
    /// then cannot be loaded at all, because its only import is a flavor
    /// `model::snapshot` does not open.
    #[test]
    fn every_servable_sku_has_a_readable_import() {
        let mut unreadable = Vec::new();
        for row in model::serve::ROWS {
            let bases = model::bases_for(row.id);
            if bases.is_empty() {
                // A tensor-parallel row is the same bytes cut at load and
                // names no import of its own; that is the table's rule, not
                // a gap.
                continue;
            }
            if !bases.iter().any(|b| b.starts_with(READABLE_BASE)) {
                unreadable.push((row.id, bases));
            }
        }
        assert!(
            unreadable.is_empty(),
            "these SKUs offer no `{READABLE_BASE}*` import, so this driver's \
             reader cannot produce them: {unreadable:?}",
        );
    }

    /// The decode word is the one the smoke fired, and the class picks it.
    #[test]
    fn the_decode_word_sets_qo_one_and_prefill_clears_it() {
        let mut plan = Plan {
            name: "a-test-row".into(),
            plane: model_ir::kernels::Backend::Cuda,
            facts: vec!["qo_one".into()],
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            seams: Vec::new(),
        };
        assert_eq!(word_of(&plan, FireClass::Decode, false), Ok(0b1));
        assert_eq!(word_of(&plan, FireClass::Prefill, false), Ok(0b0));

        // A text with one fact has ONE attention arm and it is causal, so a
        // masked frame reaching it would be answered as if it had asked
        // nothing. The refusal names the text.
        let why = word_of(&plan, FireClass::Decode, true).expect_err("refused");
        assert!(why.contains("`masked` fact"), "{why}");

        plan.facts.push("a_fact_nobody_declared".into());
        assert!(
            word_of(&plan, FireClass::Decode, false).is_err(),
            "an unknown fact refuses rather than reading as a clear bit",
        );
    }

    /// A text that DOES branch on the mask gets the lane it asked for, and
    /// the same text unmasked gets the other one.
    #[test]
    fn a_text_that_states_masked_selects_on_it() {
        let plan = Plan {
            name: "a-masked-row".into(),
            plane: model_ir::kernels::Backend::Cuda,
            facts: vec!["qo_one".into(), "masked".into()],
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            seams: Vec::new(),
        };
        assert_eq!(word_of(&plan, FireClass::Decode, false), Ok(0b01));
        assert_eq!(word_of(&plan, FireClass::Decode, true), Ok(0b11));
        assert_eq!(word_of(&plan, FireClass::Prefill, true), Ok(0b10));
    }
}
