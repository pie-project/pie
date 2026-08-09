//! gemma3n's shape.
//!
//! The second rank-K residual in the tree, and it shares no statement
//! with the first. deepseek_v4's hyper-connections MIX K streams every
//! layer; gemma3n's AltUp PREDICTS the other streams from the active one,
//! runs the layer on the prediction, then CORRECTS all of them from the
//! result. Same K, different arithmetic, different kernels.
//!
//! Two more things belong to this family alone: `laurel`, a low-rank
//! branch that lands beside attention, and per-layer embeddings (PLE) —
//! an embedding table read PER LAYER and gated in, which is why the
//! per-layer intermediate widths are a LIST here.

use serde::{Deserialize, Serialize};

/// The AltUp residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma3nAltUpFacts {
    /// How many streams. The ACTIVE one is the stream the layer body
    /// actually runs on; the rest are predicted and corrected.
    pub num_streams: u32,
    pub active: u32,
}

/// This family's attention IS a plain GQA block — see
/// [`model_compiler::facts::GqaFacts`], which both families carried
/// field-identically.
pub type Gemma3nAttnFacts = model_compiler::facts::GqaFacts;



/// The whole family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma3nFacts {
    pub vocab: u32,
    pub hidden: u32,
    /// One intermediate width per layer — `cfg.gemma3n_per_layer_intermediate`,
    /// which the loader REFUSES unless its length is the layer count. So
    /// this is the layer count too, and there is no second place to
    /// disagree.
    pub per_layer_intermediate: Vec<u32>,
    /// `cfg.laurel_rank`: the low-rank branch beside attention.
    pub laurel_rank: u32,
    /// `cfg.gemma_hidden_size_per_layer_input`: the width of the
    /// per-layer embedding that gates in.
    pub ple_width: u32,
    /// Activation sparsity: the gaussian top-k on the gate half. Zero
    /// means the layer takes the plain geglu.
    pub sparsity_layers: u32,
    pub altup: Gemma3nAltUpFacts,
    pub attn: Gemma3nAttnFacts,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_compiler::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Serde-defaulted, and
    /// empty reads as "no window", which is what every fixture written
    /// before this field meant.
    #[serde(default)]
    pub window_left: Vec<i32>,
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

    pub fn gemma3n_synthetic() -> Self {
        Gemma3nFacts {
            // The 5-layer synthetic attends the whole context; a live
            // gemma-3n states its per-layer list.
            window_left: Vec::new(),
            vocab: 262144,
            hidden: 2048,
            per_layer_intermediate: vec![8192; 6],
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
}
