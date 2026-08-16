//! gemma3n's SHAPE: the numbers a gemma-3n checkpoint has.
//!
//! Ungated, for `shared/llama_like/spec.rs`'s reason: a catalog row is
//! written in these words and a row must exist under every aspect, so
//! the struct cannot exist only when the tracer is compiled in.
//!
//! Being a row's words also means being `const`-constructible, and this
//! is the generation where that bit. Two fields were `Vec`s — the
//! per-layer MLP width and the per-layer window — and both are genuinely
//! per layer, so neither could become a scalar the way gemma-2's window
//! did. They are `&'static [T]` instead: the same numbers, in the
//! binary, with the length fixed at compile time rather than at parse
//! time. What that costs is [`serde::Deserialize`], and the note on
//! [`Gemma3nFacts`] says why that cost is zero.

use serde::Serialize;

/// The AltUp residual.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Gemma3nAltUpFacts {
    /// How many streams. The ACTIVE one is the stream the layer body
    /// actually runs on; the rest are predicted and corrected.
    pub num_streams: u32,
    pub active: u32,
}

/// This family's attention IS a plain GQA block — see
/// [`model_ir::facts::GqaFacts`], which both families carried
/// field-identically.
pub type Gemma3nAttnFacts = model_ir::facts::GqaFacts;

/// The window schedule, as a rule evaluated at compile time.
///
/// gemma-3n alternates four sliding layers and one full one, and the
/// config states the result as a thirty-entry `layer_types` array. A
/// derivation used to walk that array; a row states the rule and this
/// `const fn` expands it, so the array in the binary and the rule that
/// generated it cannot come apart.
///
/// The predicate is [`model_ir::facts::full_attn_at`]'s —
/// `(l + 1) % interval == 0` — written out because that function is not
/// `const` and a `const` row cannot call it.
#[must_use]
pub const fn window_schedule<const N: usize>(
    full_attn_interval: u32,
    sliding_window: i32,
) -> [i32; N] {
    let mut out = [sliding_window; N];
    let mut l = 0;
    while l < N {
        if full_attn_interval > 0 && (l as u32 + 1).is_multiple_of(full_attn_interval) {
            out[l] = -1;
        }
        l += 1;
    }
    out
}

/// The whole family.
///
/// # No `Deserialize`
///
/// The two per-layer fields are `&'static [T]`, which serde cannot
/// deserialize into: there is nowhere for the bytes to live. That is a
/// cost of exactly nothing, because nothing ever read this struct back.
/// The derive was inherited from the fixture era and the goldens
/// (`tests/golden_plans.rs`) serialize the TRACED PLAN, not the shape
/// that produced it — a plan is what the discipline exists to pin, and a
/// shape reaching it is the input, which is now a `const` in the binary
/// and cannot drift between runs at all. `Serialize` stays because it is
/// free and a diagnostic dump of "what did we think this model was" is
/// worth keeping.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Gemma3nFacts {
    pub vocab: u32,
    pub hidden: u32,
    /// One intermediate width per layer — `cfg.gemma3n_per_layer_intermediate`,
    /// which the loader REFUSES unless its length is the layer count. So
    /// this is the layer count too, and there is no second place to
    /// disagree.
    pub per_layer_intermediate: &'static [u32],
    /// `cfg.laurel_rank`: the low-rank branch beside attention.
    pub laurel_rank: u32,
    /// `cfg.gemma_hidden_size_per_layer_input`: the width of the
    /// per-layer embedding that gates in.
    pub ple_width: u32,
    /// Activation sparsity: the gaussian top-k on the gate half. Zero
    /// means the layer takes the plain geglu.
    pub sparsity_layers: u32,    pub altup: Gemma3nAltUpFacts,
    pub attn: Gemma3nAttnFacts,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_ir::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Empty reads as "no
    /// window", which is what every fixture written before this field
    /// meant. It used to carry `#[serde(default)]` for that; the attr
    /// went with the `Deserialize` derive, and `&[]` says the same thing
    /// at the one place a row states it.
    pub window_left: &'static [i32],
}

impl Gemma3nFacts {
    pub fn layers(&self) -> u32 {
        self.per_layer_intermediate.len() as u32
    }
    pub fn intermediate(&self, l: u32) -> u32 {
        self.per_layer_intermediate[l as usize]
    }
    /// The leading layers that apply the gaussian top-k before the geglu.
    pub fn is_sparse(&self, l: u32) -> bool {
        l < self.sparsity_layers
    }

    /// The `std_multiplier` a sparse layer's `gaussian_topk` thresholds at:
    /// `gaussian_inverse_cdf(activation_sparsity)`.
    ///
    /// A CONSTANT and not a per-layer list, because
    /// `activation_sparsity_pattern` is `0.95` across the leading run and
    /// `0.0` after it in both published checkpoints, and WHERE the run ends
    /// is already carried by [`Gemma3nFacts::is_sparse`] — the statement is
    /// only emitted where that holds. Two encodings of one pattern is how
    /// they come to disagree.
    ///
    /// `Φ⁻¹(0.95)`, to the precision an f32 keeps.
    pub fn sparsity_std_mult(&self) -> f32 {
        1.644_853_6
    }

    /// The six-layer synthetic. Uniform widths and no window: it pins
    /// the golden FORM of the traced arms, not any deployment's truth.
    pub fn gemma3n_synthetic() -> Self {
        Gemma3nFacts {
            // The synthetic attends the whole context; a live gemma-3n
            // states its per-layer list.
            window_left: &[],
            vocab: 262_144,
            hidden: 2048,
            per_layer_intermediate: &[8192; 6],
            laurel_rank: 64,
            ple_width: 256,
            sparsity_layers: 3,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AltUp is a rank-K residual, and K > 1 is what makes it one — the
    /// same check deepseek_v4's hyper-connection fixture makes, for the
    /// same reason: at 1 it would lower like an ordinary residual and
    /// prove nothing.
    #[test]
    fn the_residual_is_actually_rank_k() {
        let f = Gemma3nFacts::gemma3n_synthetic();
        assert!(f.altup.num_streams > 1);
        assert!(f.altup.active < f.altup.num_streams);
    }

    /// The fixture must exercise BOTH activation paths, or the sparsity
    /// fact is untested.
    #[test]
    fn the_fixture_has_a_sparse_layer_and_a_dense_one() {
        let f = Gemma3nFacts::gemma3n_synthetic();
        assert!((0..f.layers()).any(|l| f.is_sparse(l)));
        assert!((0..f.layers()).any(|l| !f.is_sparse(l)));
    }

    /// The layer count still comes from the per-layer list and from
    /// nowhere else, which is what made that field the layer count in
    /// the first place.
    #[test]
    fn the_per_layer_list_is_the_layer_count() {
        let f = Gemma3nFacts::gemma3n_synthetic();
        assert_eq!(f.layers(), 6);
        assert_eq!(f.intermediate(5), 8192);
    }

    /// The `const fn` expands the same schedule the config states as an
    /// array: gemma-3n's E2B lists `full_attention` at 4, 9, 14, 19, 24
    /// and 29, which is every fifth layer counting from one.
    #[test]
    fn the_window_schedule_is_the_one_the_config_lists() {
        const W: [i32; 30] = window_schedule(5, 512);
        let full: Vec<usize> = (0..30).filter(|&l| W[l] == -1).collect();
        assert_eq!(full, vec![4, 9, 14, 19, 24, 29]);
        assert!((0..30).all(|l| W[l] == -1 || W[l] == 512));
    }

    /// An empty list reads as "no window" through the shared reader,
    /// which is the shape the synthetic relies on.
    #[test]
    fn an_empty_schedule_reads_as_no_window() {
        let f = Gemma3nFacts::gemma3n_synthetic();
        assert_eq!(model_ir::facts::window_left_at(f.window_left, 0), -1);
        assert_eq!(model_ir::facts::window_left_at(f.window_left, 5), -1);
    }
}
