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
use crate::kinds::{CanonicalKind, infer_kind};

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
                let scaled = dyn_temperature_scale(&logits, &t);
                let keep = logits.pivot_rank_le(k);
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
                let scaled = dyn_temperature_scale(&logits, &t);
                let probs = dyn_softmax(&scaled);
                let keep_k = logits.pivot_rank_le(k);
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
pub fn lower_sampler(spec: SamplerSpec, vocab: u32) -> Result<LoweredProgram, BuildError> {
    Ok(build_sampler(spec, vocab)?.lower())
}
