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
//! # What is still the OTHER catalogue's
//!
//! The pools. `model::deployment` sizes and strides the KV pages and the
//! recurrent slabs, gates the KV style and the GQA ratio at the door, and
//! publishes the caps — so this lane's program is fired against pools laid
//! out by the legacy catalog's account of the same checkpoint.
//! [`Geometry::agrees_with`] is the check that stands between those two
//! accounts, and R3 is where the second one dies.

use std::collections::BTreeMap;

use model_compiler::program::{Program, Refusal};
use model_ir::plan::Plan;
use model_ir::trace::FireClass;

use crate::device::{Allocator, DeviceBuffer, OwnedStream};

pub(crate) mod fire;
pub(crate) mod geometry;
pub(crate) mod marks;
pub(crate) mod points_shim;
pub(crate) mod resolve;
pub(crate) mod staging;

pub(crate) use geometry::Geometry;

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
    pub dtype: model_baker::produce::Dtype,
}

// SAFETY: a device address is a number, never dereferenced on the host; its
// CUDA context outlives the shell that holds it. `weights::stage`'s
// `WeightSpan` carries the same pair of impls for the same reason.
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
    /// THE SEAM, HALF CLOSED: these are `model::produce`'s bytes and not the
    /// legacy manifest's, and they used to be a SECOND residency of the same
    /// checkpoint because both loads ran. Only this one runs now.
    ///
    /// What is left of the seam is that `LoadedModel::weights` still exists
    /// beside this map — `weights::stage` still stages the manifest's arena,
    /// because `serve::encode`'s towers and the `Encode` verb read weights by
    /// the legacy names. A fire reads only this map.
    pub banks: BTreeMap<String, Bank>,
    /// The arena the banks are spans of, held to keep them alive.
    ///
    /// Underscored like `KvState::_held`, and for the same reason: nothing
    /// reads it, and that is the point — every [`Bank`] above is a raw
    /// pointer INTO this allocation, so dropping it frees every weight the
    /// lane is about to fire against. It is the field's existence that is
    /// load-bearing, not its value.
    _owned: Vec<DeviceBuffer>,
    /// The numbers the claim-only routines want and the statements do not
    /// carry, all read off the plan.
    pub geom: Geometry,
    /// The value the plan's `out` seam names.
    pub out: model_ir::plan::ValueId,
}

impl Baked {
    /// The lane that serves `class`, and the fact word that picked it.
    ///
    /// A REFUSAL AND NOT A PANIC when no lane serves: the smoke could
    /// `panic!` on an unknown fact because the next line was its own exit,
    /// and a driver's next line is somebody else's request.
    pub fn lane(&self, class: FireClass) -> Result<(u64, &Program), String> {
        let word = word_of(&self.plan, class)?;
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
                "attention.decode" | "attention.decode_lse" | "attention.masked" => {
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
                "attention.prefill" | "attention.prefill_lse" => {
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
/// one that picks the lane must be the one that named the class.
///
/// A fact this does not know is a REFUSAL and not a zero. A zero would be a
/// guess that silently picks a lane, and the lane is the whole program.
fn word_of(plan: &Plan, class: FireClass) -> Result<u64, String> {
    let mut word = 0u64;
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
    Ok(word)
}

/// The new-catalog SKU for a checkpoint the legacy catalog identified.
///
/// # Why this table exists
///
/// The two catalogues do not share an id space, and neither is wrong.
/// `model_legacy::catalog::identify` reads the checkpoint's own tensors and
/// answers what model it IS (`"qwen3.5-0.8b-base"`); the new catalog files
/// a row under a SKU that also states how the numbers are stored
/// (`"qwen35-d0.8b-bf16-kv-bf16"`), because a trace's dtypes are part of
/// what it traces. Nothing in the tree maps between them — I looked.
///
/// So the bridge is written down, here, with one row per checkpoint whose
/// baker lane has actually been fired. It is deliberately NOT a fuzzy match
/// on the family name: `"qwen3.5-4b"` and `"qwen35-d3b-bf16-kv-bf16"` are
/// close enough to pair by accident and are different models.
///
/// A deployment can name a SKU the table does not hold, through
/// `[baker] sku` / `PIE_BAKER_SKU` — which is also how a new row is proven
/// before it is added here.
///
/// **This table dies when the id spaces merge**, which is the real fix and
/// is not this work's.
const BRIDGE: &[(&str, &str)] = &[
    // Proven end to end: `baker-smoke --sku qwen35-d0.8b-bf16-kv-bf16`
    // fires 381 steps against this checkpoint's real weights.
    ("qwen3.5-0.8b-base", "qwen35-d0.8b-bf16-kv-bf16"),
];

/// Which new-catalog row to trace for `legacy_id`, if any.
///
/// The knob outranks the table: a deployment that states a SKU is telling
/// the driver something the table cannot know, and a table row that
/// silently won would make the knob untestable.
pub(crate) fn sku_for(legacy_id: &str, stated: Option<&str>) -> Option<String> {
    if let Some(sku) = stated {
        return Some(sku.to_owned());
    }
    BRIDGE
        .iter()
        .find(|(id, _)| *id == legacy_id)
        .map(|(_, sku)| (*sku).to_owned())
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
    let trace = model_baker::trace_of(sku)
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

    // The geometry every claim-only routine needs, read off the plan and
    // never off a config file. It needs a bound lane to read a slot width
    // out of, so it is derived from the decode lane specifically — the one
    // this work fires.
    let (_, decode) = {
        let word = word_of(&plan, FireClass::Decode)?;
        let mut found = None;
        for lane in &lanes {
            if let Ok(p) = lane {
                if p.words.contains(&word) {
                    found = Some((word, p));
                    break;
                }
            }
        }
        found.ok_or_else(|| {
            format!("`{sku}` binds no decode lane; the baker path has nothing to fire")
        })?
    };
    let geom = Geometry::of(&plan, decode)?;

    // ── 2. The weights, through `produce` and not the manifest. ─────────
    let base = model_baker::bases_for(sku)
        .into_iter()
        .find(|b| b.starts_with(READABLE_BASE))
        .ok_or_else(|| {
            format!(
                "`{sku}` names no `{READABLE_BASE}*` import; this driver's \
                 reader is safetensors and the SKU offers {:?}",
                model_baker::bases_for(sku)
            )
        })?;
    let import = model_baker::import_of(sku, base)
        .ok_or_else(|| format!("`{sku}` names no `{base}` import"))?;
    // `Snapshot::at` AND NOT `Snapshot::open`: `open` resolves a cache-dir
    // NAME under `$HOME/.cache/huggingface/hub`, and the driver is handed
    // the snapshot directory itself (`ModelLoadDesc::snapshot_dir`). Going
    // through `open` would mean reconstructing a cache name from a path
    // that need not be in a cache at all.
    let snap = model_baker::snapshot::Snapshot::at(snapshot.to_path_buf())
        .ok_or_else(|| format!("no safetensors snapshot at {}", snapshot.display()))?;
    let produced = model_baker::produce::produce(&import, &|n| snap.read(n))
        .map_err(|e| format!("production refused: {e}"))?;

    let (banks, owned) = upload(&produced, alloc)?;
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
        geom,
        out,
    })
}

/// Every produced tensor into one arena, one span each.
///
/// ONE ARENA AND NOT 260 ALLOCATIONS, which is the driver's idiom and not
/// the smoke's: `weights::stage::stage_plan_weights` allocates the whole
/// arena before a byte is read and names every tensor a span inside it,
/// and this mirrors it exactly (`weights/stage.rs:57-125`). The smoke took
/// a `cudaMalloc` per tensor because it had no allocator to reuse; the
/// driver has one, and 260 separate allocations would be 260 chances for
/// the allocator's live-bytes accounting to disagree with the device's.
///
/// THE UPLOAD HAS NO DECISION IN IT, which is what `baker_load` says and
/// what makes this loop short: every produced tensor is dense, row-major
/// and canonical, so one contiguous H2D per bank and no restride, no
/// repack, no cast.
fn upload(
    produced: &[(String, model_baker::produce::HostTensor)],
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

    /// The knob outranks the table, and the table answers the row it holds.
    #[test]
    fn the_stated_sku_outranks_the_bridge_table() {
        assert_eq!(
            sku_for("qwen3.5-0.8b-base", None).as_deref(),
            Some("qwen35-d0.8b-bf16-kv-bf16"),
        );
        assert_eq!(
            sku_for("qwen3.5-0.8b-base", Some("something-else")).as_deref(),
            Some("something-else"),
            "a deployment that states a SKU is not overruled by the table",
        );
        assert_eq!(
            sku_for("a-checkpoint-nobody-has-bridged", None),
            None,
            "an unbridged id answers nothing rather than guessing",
        );
    }

    /// Every SKU the table names is a row of the catalog it bridges TO.
    ///
    /// The failure this prevents is a table row that rots: the new catalog
    /// renames a SKU, the bridge keeps the old spelling, and the only
    /// symptom is that `PIE_BAKER=1` quietly stops building a lane.
    #[test]
    fn every_bridged_sku_is_a_catalog_row() {
        let catalog = model_baker::catalog();
        for (legacy, sku) in BRIDGE {
            assert!(
                catalog.iter().any(|(n, _)| n == sku),
                "the bridge maps `{legacy}` to `{sku}`, which `model::catalog()` \
                 does not ship",
            );
        }
    }

    /// A bridged SKU can be built from a checkpoint this driver can read.
    #[test]
    fn every_bridged_sku_has_a_readable_import() {
        for (_, sku) in BRIDGE {
            let bases = model_baker::bases_for(sku);
            assert!(
                bases.iter().any(|b| b.starts_with(READABLE_BASE)),
                "`{sku}` offers {bases:?}, none of which this driver's \
                 safetensors reader can produce from",
            );
        }
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
        assert_eq!(word_of(&plan, FireClass::Decode), Ok(0b1));
        assert_eq!(word_of(&plan, FireClass::Prefill), Ok(0b0));

        plan.facts.push("a_fact_nobody_declared".into());
        assert!(
            word_of(&plan, FireClass::Decode).is_err(),
            "an unknown fact refuses rather than reading as a clear bit",
        );
    }
}
