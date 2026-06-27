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

fn is_greedy(t: f32) -> bool {
    t <= 0.0
}

/// Build the IR program for `spec` at runtime `vocab`. `outputs = [Token]`.
pub fn build_sampler(spec: SamplerSpec, vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
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
