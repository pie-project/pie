//! **`Sampler` → IR sugar** (WS5 de-hardwiring). The legacy SDK `Sampler` enum
//! lowered to IR programs so the hardwired CUDA sampler kernels can be removed
//! and all sampling routed through the IR-JIT path.
//!
//! Selection is Gumbel-max at `stream:0` (Model B ambient seed); `stream:0` ≡
//! today's `sample_temp` hash, so parity holds by construction. Sampler params
//! (temperature/k/p) bake as `Const`; programs declare **no** host inputs.

use crate::builder::{Built, BuildError, Graph, LoweredProgram, OutputKind};
use crate::dynamic::{
    dyn_gumbel_argmax, dyn_mask_to_score, dyn_min_p_mask, dyn_softmax, dyn_temperature_scale,
    dselect,
};
use crate::ir::TensorKey;
use crate::kinds::{CanonicalKind, infer_kind};
use crate::standard::{StandardSampler, StdParamKeys, build_standard};
use alloc::vec::Vec;

/// A legacy sampler spec (mirrors the SDK `Sampler` enum). golf translates the
/// inferlet-facing `Sampler` into this.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SamplerSpec {
    Argmax,
    TopP { temperature: f32, p: f32 },
    TopK { temperature: f32, k: u32 },
    MinP { temperature: f32, p: f32 },
    TopKTopP { temperature: f32, k: u32, p: f32 },
    Multinomial { temperature: f32 },
}

/// Classify a [`SamplerSpec`] to its [`CanonicalKind`] via the frozen recognizer
/// ladder ([`infer_kind`]) — the SDK-side half of the de-hardwiring recognizer
/// (#8). The same kind is stamped on [`build_sampler`]'s [`Built`]; a drift-guard
/// test asserts it equals the runtime's `infer_sampler_kind` for the same params,
/// so the SDK sugar and the host recognizer can never silently diverge.
///
/// `Argmax` carries no temperature (it is intrinsically greedy ≡ `T=0`), so it
/// resolves through the `T<=0 -> Argmax` arm. The continuous params (temperature,
/// `p`, `min_p`) are passed straight to the ladder; they ride the per-row `pi`
/// carrier in production, so they classify the kind but never enter the tag.
pub fn canonical_kind(spec: SamplerSpec) -> CanonicalKind {
    let (t, k, top_p, min_p) = match spec {
        SamplerSpec::Argmax => (0.0, 0, 1.0, 0.0),
        SamplerSpec::Multinomial { temperature } => (temperature, 0, 1.0, 0.0),
        SamplerSpec::TopP { temperature, p } => (temperature, 0, p, 0.0),
        SamplerSpec::TopK { temperature, k } => (temperature, k, 1.0, 0.0),
        SamplerSpec::MinP { temperature, p } => (temperature, 0, 1.0, p),
        SamplerSpec::TopKTopP { temperature, k, p } => (temperature, k, p, 0.0),
    };
    infer_kind(t, k, top_p, min_p)
}

fn is_greedy(t: f32) -> bool {
    t <= 0.0
}

/// Build the IR program for `spec` at runtime `vocab`. `outputs = [Token]`.
///
/// # Legacy sugar — emits the param-VARIANT form
/// This bakes temperature / `p` as `constant_f32` **immediates**, so the bytecode
/// varies per param value and is **not recognizable or cacheable** by the driver's
/// hash recognizer. For a program you intend to attach + dispatch, use
/// [`lower_sampler_standard`] (the canonical, recognizable `standard_program` +
/// submit values). `build_sampler` remains for building/inspecting the legacy
/// sugar graph (e.g. the eval/structure tests); do not attach its lowering. See #17.
pub fn build_sampler(spec: SamplerSpec, vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(canonical_kind(spec));
    let logits = g.intrinsic_logits_dyn();

    let token = match spec {
        SamplerSpec::Argmax => logits.argmax(),

        SamplerSpec::Multinomial { temperature } => {
            if is_greedy(temperature) {
                logits.argmax()
            } else {
                let t = g.constant_f32_dyn(temperature);
                let scaled = dyn_temperature_scale(&logits, &t);
                dyn_gumbel_argmax(&g, &scaled, 0, vocab)
            }
        }

        SamplerSpec::TopP { temperature, p } => {
            if is_greedy(temperature) {
                logits.argmax()
            } else {
                let t = g.constant_f32_dyn(temperature);
                let pv = g.constant_f32_dyn(p);
                let scaled = dyn_temperature_scale(&logits, &t);
                let probs = dyn_softmax(&scaled);
                let keep = probs.pivot_cummass_le(&pv);
                let score = dyn_mask_to_score(&g, &keep, &scaled);
                dyn_gumbel_argmax(&g, &score, 0, vocab)
            }
        }

        SamplerSpec::TopK { temperature, k } => {
            if is_greedy(temperature) {
                logits.argmax()
            } else {
                let t = g.constant_f32_dyn(temperature);
                let kc = g.constant_i32_dyn(k as i32); // #25: k is a value-id (baked Const here)
                let scaled = dyn_temperature_scale(&logits, &t);
                let keep = logits.pivot_rank_le(&kc);
                let score = dyn_mask_to_score(&g, &keep, &scaled);
                dyn_gumbel_argmax(&g, &score, 0, vocab)
            }
        }

        SamplerSpec::MinP { temperature, p } => {
            if is_greedy(temperature) {
                logits.argmax()
            } else {
                let t = g.constant_f32_dyn(temperature);
                let pv = g.constant_f32_dyn(p);
                let scaled = dyn_temperature_scale(&logits, &t);
                let keep = dyn_min_p_mask(&logits, &pv);
                let score = dyn_mask_to_score(&g, &keep, &scaled);
                dyn_gumbel_argmax(&g, &score, 0, vocab)
            }
        }

        SamplerSpec::TopKTopP { temperature, k, p } => {
            if is_greedy(temperature) {
                logits.argmax()
            } else {
                let t = g.constant_f32_dyn(temperature);
                let pv = g.constant_f32_dyn(p);
                let kc = g.constant_i32_dyn(k as i32); // #25: k is a value-id (baked Const here)
                let scaled = dyn_temperature_scale(&logits, &t);
                let probs = dyn_softmax(&scaled);
                let keep_k = logits.pivot_rank_le(&kc);
                let keep_p = probs.pivot_cummass_le(&pv);
                let f = g.constant_bool_dyn(false).broadcast_vec(vocab);
                let keep = dselect(&keep_k, &keep_p, &f); // AND
                let score = dyn_mask_to_score(&g, &keep, &scaled);
                dyn_gumbel_argmax(&g, &score, 0, vocab)
            }
        }
    };

    g.output(&token, OutputKind::Token);
    g.build()
}

/// Convenience: build then lower in one step (the host de-hardwire entry — just
/// the bytecode; Model B needs no seed key).
#[deprecated(
    note = "emits the param-VARIANT sugar form: bakes T/p as constant_f32 immediates → \
            a bytecode-only LoweredProgram that the driver recognizer can't hash-match \
            (→ CustomJIT) and that drops the submit params. Use `lower_sampler_standard`, \
            which emits the canonical recognizable `standard_program` + its submit values. See #17."
)]
pub fn lower_sampler(spec: SamplerSpec, vocab: u32) -> Result<LoweredProgram, BuildError> {
    Ok(build_sampler(spec, vocab)?.lower())
}

/// Per-fire **submit values** for a standard program's continuous params:
/// `(TensorKey, f32-little-endian bytes)`, ready to pass to the inferlet's
/// `resolve_bindings` (T@key0, p/min_p@key1). Empty for greedy/argmax.
pub type SubmitValues = Vec<(TensorKey, Vec<u8>)>;

/// Lower a [`SamplerSpec`] to the **canonical** standard-sampler program
/// ([`build_standard`]) plus the per-fire **submit values** for its continuous
/// params — the #15/#12 production lowering (option B).
///
/// Unlike [`build_sampler`], which bakes temperature / `p` as `constant_f32`
/// IMMEDIATES (→ param-VARIANT bytecode, a different hash per `p`, un-recognizable
/// and un-cacheable), this emits the param-INVARIANT `standard_program` the driver
/// recognizes by hash and caches per `vocab`, returning T / `p` / `min_p` as
/// `(key, le bytes)` submit values keyed by [`StdParamKeys`]. Bind them through
/// the inferlet's `resolve_bindings` (the slots the sugar path left empty). Since
/// #25 `k` rides a `U32` submit value, appended last (after temperature / top-p).
///
/// The `SamplerSpec → StandardSampler` map runs through [`canonical_kind`], so the
/// greedy collapse (`T <= 0 → Argmax`, which submits no params) is preserved
/// exactly — the emitted program is byte-identical to what the driver baked, so
/// `recognize()` hash-hits and `extract_dedicated_params` reads these submit slots.
pub fn lower_sampler_standard(
    spec: SamplerSpec,
    vocab: u32,
) -> Result<(Built, SubmitValues), BuildError> {
    let std_kind = match canonical_kind(spec) {
        CanonicalKind::Argmax => StandardSampler::Argmax,
        CanonicalKind::Temperature => StandardSampler::Temperature,
        CanonicalKind::MinP => StandardSampler::MinP,
        CanonicalKind::TopP => StandardSampler::TopP,
        CanonicalKind::TopK => StandardSampler::TopK,
        CanonicalKind::TopKTopP => StandardSampler::TopKTopP,
        // SamplerSpec's variants all classify to a standard token-sampler kind;
        // Custom is unreachable from sugar, but route it to greedy defensively.
        CanonicalKind::Custom => StandardSampler::Argmax,
    };
    let (built, keys) = build_standard(std_kind, vocab)?;
    Ok((built, collect_submit_values(spec, &keys)))
}

/// The baked top-k cutoff for a k-bearing spec; `0` otherwise (never reached for a
/// non-k-bearing canonical kind).
fn spec_k(spec: SamplerSpec) -> u32 {
    match spec {
        SamplerSpec::TopK { k, .. } | SamplerSpec::TopKTopP { k, .. } => k,
        _ => 0,
    }
}

/// Collect the per-fire submit values for the params the chosen standard program
/// actually declares (i.e. has a `Some` key for). Greedy specs collapse to Argmax
/// (all keys `None`) → no submit values.
fn collect_submit_values(spec: SamplerSpec, keys: &StdParamKeys) -> SubmitValues {
    let (temperature, top_p, min_p) = match spec {
        SamplerSpec::Argmax => (0.0, 1.0, 0.0),
        SamplerSpec::Multinomial { temperature } => (temperature, 1.0, 0.0),
        SamplerSpec::TopP { temperature, p } => (temperature, p, 0.0),
        SamplerSpec::TopK { temperature, .. } => (temperature, 1.0, 0.0),
        SamplerSpec::MinP { temperature, p } => (temperature, 1.0, p),
        SamplerSpec::TopKTopP { temperature, p, .. } => (temperature, p, 0.0),
    };
    let mut submit = Vec::new();
    if let Some(key) = keys.temperature {
        submit.push((key, temperature.to_le_bytes().to_vec()));
    }
    if let Some(key) = keys.top_p {
        submit.push((key, top_p.to_le_bytes().to_vec()));
    }
    if let Some(key) = keys.min_p {
        submit.push((key, min_p.to_le_bytes().to_vec()));
    }
    // #25: k rides a U32 submit value (appended last), no longer a baked immediate.
    if let Some(key) = keys.k {
        submit.push((key, spec_k(spec).to_le_bytes().to_vec()));
    }
    submit
}
