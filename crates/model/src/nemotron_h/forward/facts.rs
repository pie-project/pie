//! nemotron_h's shape.
//!
//! The third hybrid in the tree and the only one whose schedule is a
//! LIST. qwen3_5 and kimi_k3 both derive their layer kinds from an
//! interval; nemotron_h reads `cfg.layer_types` — a per-layer string —
//! and the kinds are THREE, not two: `mamba`, `attention`, and a bare
//! `mlp` layer with no mixer at all.
//!
//! Mamba is the new vocabulary. Where GDN and KDA carry a decaying state
//! over key/value pairs, mamba's selective scan carries an explicit
//! `[heads, head_dim, state_size]` state and reads a per-token `dt` that
//! decides how much of it to keep — which is why `prepare_mamba_dt_da`
//! is its own statement and not folded into the scan.

use serde::{Deserialize, Serialize};

/// What a layer mixes with, per `cfg.layer_types`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NemotronLayerKind {
    Mamba,
    Attention,
    /// No mixer: the layer is its MLP and nothing else. Neither of the
    /// other hybrids has such a layer, and a schedule expressed as an
    /// interval could not say it.
    Mlp,
}

/// The mamba mixer's dims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NemotronMambaFacts {
    pub num_heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    pub n_groups: u32,
    pub conv_kernel: u32,
}

impl NemotronMambaFacts {
    /// The scan's working width.
    pub fn intermediate(&self) -> u32 {
        self.num_heads * self.head_dim
    }
    /// What the conv sees: the scan's width plus the two group-wide
    /// B and C projections that ride the same buffer.
    pub fn conv_dim(&self) -> u32 {
        self.intermediate() + 2 * self.n_groups * self.state_size
    }
    /// The in-projection's width: `[z | conv_dim | dt]`.
    pub fn in_proj_width(&self) -> u32 {
        self.intermediate() + self.conv_dim() + self.num_heads
    }
}

/// This family's attention IS a plain GQA block — see
/// [`model_compiler::facts::GqaFacts`], which both families carried
/// field-identically.
pub type NemotronAttnFacts = model_compiler::facts::GqaFacts;



/// This family's mixture IS the shared one — see
/// [`model_compiler::facts::MoeFacts`]. Three families carried
/// field-identical copies; the alias keeps every spelling working while
/// there is one definition.
pub type NemotronMoeFacts = model_compiler::facts::MoeFacts;


/// The whole family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NemotronHFacts {
    pub vocab: u32,
    pub hidden: u32,
    /// The schedule, verbatim — one entry per layer. `layers` is its
    /// length, so there is no second place to disagree about the count.
    pub layer_types: Vec<NemotronLayerKind>,
    pub mamba: NemotronMambaFacts,
    pub attn: NemotronAttnFacts,
    pub moe: NemotronMoeFacts,
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

impl NemotronHFacts {
    pub fn layers(&self) -> u32 {
        self.layer_types.len() as u32
    }
    pub fn kind(&self, l: u32) -> NemotronLayerKind {
        self.layer_types[l as usize]
    }

    pub fn nemotron_h_synthetic() -> Self {
        use NemotronLayerKind::*;
        NemotronHFacts {
            // Nemotron-H attends the whole context.
            window_left: Vec::new(),
            vocab: 131072,
            hidden: 2048,
            layer_types: vec![Mamba, Mlp, Mamba, Attention, Mamba, Mlp],
            mamba: NemotronMambaFacts {
                num_heads: 16,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 16,
                kv_heads: 4,
                head_dim: 128,
            },
            moe: NemotronMoeFacts {
                num_experts: 32,
                top_k: 4,
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use NemotronLayerKind::*;

    /// All THREE kinds, or the fixture is not exercising the schedule
    /// this family alone has.
    #[test]
    fn the_fixture_carries_every_layer_kind() {
        let f = NemotronHFacts::nemotron_h_synthetic();
        for k in [Mamba, Attention, Mlp] {
            assert!(
                (0..f.layers()).any(|l| f.kind(l) == k),
                "the fixture states no {k:?} layer"
            );
        }
        assert_eq!(f.layers(), f.layer_types.len() as u32);
    }

    /// The in-projection packs three things, and the scan reads the
    /// middle one through a conv that is wider than the scan itself.
    #[test]
    fn the_mamba_widths_nest_the_way_the_split_reads_them() {
        let m = NemotronHFacts::nemotron_h_synthetic().mamba;
        assert_eq!(m.intermediate(), 16 * 64);
        assert_eq!(m.conv_dim(), m.intermediate() + 2 * 8 * 128);
        assert_eq!(
            m.in_proj_width(),
            m.intermediate() + m.conv_dim() + m.num_heads
        );
    }
}
