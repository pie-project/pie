//! Nemotron-H's SHAPE: the numbers a checkpoint of the Mamba2 hybrid
//! has.
//!
//! Ungated, for `qwen_3_5::spec`'s reason and one this generation makes
//! sharper than any other. The layer SCHEDULE is what a manifest, a
//! `Deployment` and a trace are all projections of — and here the
//! schedule is a LIST, not an interval. qwen3.5 alternates on a period
//! and kimi_k3 does too; Nemotron-H reads `hybrid_override_pattern`, a
//! string of one character per layer, and the 47B's is
//!
//! ```text
//! M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-
//! ```
//!
//! which contains `M---MM---M` — three MLP layers with no mixer between
//! them, then two Mamba layers back to back. No period generates that.
//! A row that stated an interval would be stating a stack NVIDIA did not
//! ship, and the checkpoint would load and produce noise, because every
//! tensor it names would still be present under a different layer index.
//!
//! What stayed behind in [`forward::facts`] is the per-backend BINDING
//! facts. Those name kernels, so they belong to the aspect that has
//! them.
//!
//! # `Vec` became `&'static [T]`
//!
//! A row is a `const`, and a `const` cannot hold a heap allocation. The
//! two lists here — the schedule and the per-layer windows — are stated
//! as `&'static [T]` for that reason and lose nothing by it: both were
//! read-only from the moment they were built.
//!
//! [`forward::facts`]: super::forward::facts

use serde::Serialize;

/// What a layer mixes with, per `hybrid_override_pattern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NemotronLayerKind {
    /// `M` — a Mamba2 selective-scan mixer.
    Mamba,
    /// `*` — an ordinary GQA attention block.
    Attention,
    /// `-` — no mixer: the layer is its MLP and nothing else. Neither of
    /// the other hybrids has such a layer, and a schedule expressed as
    /// an interval could not say it.
    Mlp,
}

impl NemotronLayerKind {
    /// The character `hybrid_override_pattern` spells this kind with.
    ///
    /// Here rather than in a test so that the mapping is stated ONCE:
    /// the schedules below are transcribed from those strings by hand,
    /// and the tests re-read them through this function. A typo in a
    /// 98-character string is invisible; a typo checked against the
    /// string it came from is not.
    #[must_use]
    pub const fn glyph(self) -> char {
        match self {
            Self::Mamba => 'M',
            Self::Attention => '*',
            Self::Mlp => '-',
        }
    }
}

/// The Mamba2 mixer's dims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NemotronMambaFacts {
    pub num_heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    /// How many groups the B and C projections are shared across.
    ///
    /// The one number of this mixer that NO tensor extent carries: the
    /// checkpoint ships `2 * n_groups * state_size` rows of B and C
    /// fused into one bank, so a loader holding the tensors knows only
    /// the PRODUCT. `LoadShape::mamba_groups` exists for this, and
    /// `contract::layer_mamba_tp` is where a wrong factorization would
    /// cut a group in half.
    pub n_groups: u32,
    pub conv_kernel: u32,
}

impl NemotronMambaFacts {
    /// The scan's working width.
    #[must_use]
    pub const fn intermediate(&self) -> u32 {
        self.num_heads * self.head_dim
    }
    /// What the conv sees: the scan's width plus the two group-wide
    /// B and C projections that ride the same buffer.
    #[must_use]
    pub const fn conv_dim(&self) -> u32 {
        self.intermediate() + 2 * self.n_groups * self.state_size
    }
    /// The in-projection's width: `[z | conv_dim | dt]`.
    #[must_use]
    pub const fn in_proj_width(&self) -> u32 {
        self.intermediate() + self.conv_dim() + self.num_heads
    }
}

/// This family's attention IS a plain GQA block — see
/// [`model_ir::facts::GqaFacts`], which both families carried
/// field-identically.
pub type NemotronAttnFacts = model_ir::facts::GqaFacts;

/// This family's mixture IS the shared one — see
/// [`model_ir::facts::MoeFacts`]. Three families carried
/// field-identical copies; the alias keeps every spelling working while
/// there is one definition.
pub type NemotronMoeFacts = model_ir::facts::MoeFacts;

/// The whole family.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NemotronHFacts {
    pub vocab: u32,
    pub hidden: u32,
    /// The schedule, verbatim — one entry per layer. `layers` is its
    /// length, so there is no second place to disagree about the count.
    pub layer_types: &'static [NemotronLayerKind],
    pub mamba: NemotronMambaFacts,
    pub attn: NemotronAttnFacts,
    /// The MLP block, routed or dense.
    ///
    /// `num_experts == 0` is a DENSE stack, and then
    /// [`NemotronMoeFacts::moe_intermediate`] is the dense MLP's width
    /// — which is how the traced text reads it on an
    /// [`NemotronLayerKind::Mlp`] layer, one `up_proj` and one
    /// `down_proj` with ReLU² between, no gate half. Every published
    /// Nemotron-H is this shape; the routed arm is the synthetic
    /// fixture's.
    pub moe: NemotronMoeFacts,
    /// Whether the embedding table is also the output projection.
    ///
    /// `false` on every published Nemotron-H, which is worth stating
    /// because it is the minority answer in this crate and an absent
    /// `tie_word_embeddings` defaults to `true` in most of the lineage.
    pub tied_embeddings: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_ir::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Empty reads as "no
    /// window", which is what every Nemotron-H means: `sliding_window`
    /// is `null` in all three published configs.
    pub window_left: &'static [i32],
}

/// The 52-layer schedule the 4B and the 8B share.
///
/// `"M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"` — 24 Mamba
/// layers, 24 MLP layers and FOUR attention layers, at 7, 18, 29 and 40.
/// Two checkpoints of different widths run the identical schedule, so it
/// is named once.
pub const SCHEDULE_52: &[NemotronLayerKind] = &{
    use NemotronLayerKind::{Attention as A, Mamba as M, Mlp as F};
    [
        M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A,
        F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, F,
    ]
};

/// The 47B's 98-layer schedule.
///
/// `"M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-"`
/// — and the `M---MM---M` in the tail is why this generation carries a
/// list. Five attention layers, at 17, 38, 49, 60 and 86, spaced 21, 11,
/// 11 and 26 apart.
pub const SCHEDULE_98: &[NemotronLayerKind] = &{
    use NemotronLayerKind::{Attention as A, Mamba as M, Mlp as F};
    [
        M, F, M, F, M, F, M, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, F, M,
        F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M,
        A, F, M, F, M, F, M, F, M, F, M, F, M, F, M, F, F, F, M, M, F, F, F, M, F, M, A, F, M, F,
        M, F, M, F, M, F, M, F,
    ]
};

impl NemotronHFacts {
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layer_types.len() as u32
    }

    /// # Panics
    ///
    /// If `l` is past the end of the schedule, which is a caller that
    /// disagrees with [`Self::layers`] about how deep the stack is.
    #[must_use]
    pub fn kind(&self, l: u32) -> NemotronLayerKind {
        self.layer_types[l as usize]
    }

    /// The layers that carry a recurrent slab.
    ///
    /// The list `RecurrentShape::linear_layers` wants, and the reason
    /// the schedule cannot be summarized: an allocator that provisioned
    /// slabs for "every layer that is not attention" would allocate 24
    /// slabs the MLP layers never touch on the 4B, and 48 on the 47B.
    #[must_use]
    pub fn mamba_layers(&self) -> Vec<u32> {
        (0..self.layers())
            .filter(|&l| self.kind(l) == NemotronLayerKind::Mamba)
            .collect()
    }

    /// Whether the MLP block is routed.
    #[must_use]
    pub const fn is_mixture(&self) -> bool {
        self.moe.num_experts > 0
    }

    /// `nvidia/Nemotron-H-8B-Base-8K`.
    #[must_use]
    pub const fn nemotron_h_8b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 4096,
            layer_types: SCHEDULE_52,
            mamba: NemotronMambaFacts {
                num_heads: 128,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },
            // `attention_head_dim: 128` and NOT `hidden / heads`, which
            // is 128 here by coincidence and 128 on the 47B where
            // `hidden / heads` is also 128 — the two agree on every
            // published checkpoint and the config states the field
            // separately anyway, so it is transcribed rather than
            // derived.
            attn: NemotronAttnFacts {
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(21_504),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    /// `nvidia/Nemotron-H-4B-Base-8K` — the 8B's schedule at a narrower
    /// width, and 112 Mamba heads rather than 128.
    ///
    /// Written out rather than derived from its sibling with `..`:
    /// struct update syntax inherits every field nobody named, and the
    /// fields these two share are shared by coincidence of training, not
    /// by construction. A 4B that grew a fifth attention layer would
    /// have to be edited HERE, and a row that inherited would not show
    /// where.
    #[must_use]
    pub const fn nemotron_h_4b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 3072,
            layer_types: SCHEDULE_52,
            mamba: NemotronMambaFacts {
                num_heads: 112,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(12_288),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    /// `nvidia/Nemotron-H-47B-Base-8K` — 98 layers, and the irregular
    /// schedule this generation's list exists for.
    #[must_use]
    pub const fn nemotron_h_47b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 8192,
            layer_types: SCHEDULE_98,
            mamba: NemotronMambaFacts {
                num_heads: 256,
                head_dim: 64,
                // TWICE the smaller checkpoints' state. The scan's
                // working set is `heads * head_dim * state`, so this row
                // asks for four times the slab the 8B does per layer,
                // and it is the number an allocator gets wrong quietly:
                // too small a slab is a silent read past the end of a
                // state, not a shape error.
                state_size: 256,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 64,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(30_720),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    /// The synthetic four-layer stack the config-parser fixture states,
    /// widened into a routed MLP so the mixture arm has a shape to be
    /// projected from.
    ///
    /// `crates/driver-cuda/tests/hf_config_dump/corpus/synthetic--nemotron-h.json`
    /// spells `"M*E-"`: one Mamba layer, one attention layer, one
    /// expert layer and one MLP layer. `E` has no kind of its own here —
    /// a routed layer IS an MLP layer whose block has a router — so the
    /// schedule below reads it as [`NemotronLayerKind::Mlp`].
    #[must_use]
    pub const fn nemotron_h_synthetic() -> Self {
        use NemotronLayerKind::{Attention, Mamba, Mlp};
        Self {
            vocab: 131_072,
            hidden: 2048,
            layer_types: &[Mamba, Mlp, Mamba, Attention, Mamba, Mlp],
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
                // nemotron-h's router normalizes over the chosen experts.
                norm_topk_prob: true,
                // No nemotron-h config states the key.
                routed_scaling: 1.0,
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
            tied_embeddings: false,
            // Nemotron-H attends the whole context.
            window_left: &[],
        }
    }
}

/// A DENSE MLP block of `intermediate` width.
///
/// The dense width lives in `moe_intermediate` because that is the field
/// the traced MLP layer reads — see [`NemotronHFacts::moe`]. Written as
/// a constructor so the three zeros that make it dense are stated once
/// instead of three times, where the third copy is the one that would
/// get a stray `top_k`.
const fn dense(intermediate: u32) -> NemotronMoeFacts {
    NemotronMoeFacts {
        num_experts: 0,
        top_k: 0,
        // A dense row: no router reads this.
        norm_topk_prob: true,
        routed_scaling: 1.0,
        moe_intermediate: intermediate,
        shared_intermediate: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NemotronHFacts, NemotronLayerKind,
        NemotronLayerKind::{Attention, Mamba, Mlp},
        SCHEDULE_52, SCHEDULE_98,
    };

    /// The 4B's and 8B's `hybrid_override_pattern`, verbatim.
    const PATTERN_52: &str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-";

    /// The 47B's, verbatim.
    const PATTERN_98: &str = "M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-";

    /// The schedules are the config's strings, character for character.
    ///
    /// This is the test the whole generation rests on. A 98-entry table
    /// written by hand has 98 chances to slip a layer, and a slipped
    /// layer is not a crash: every tensor the row names still exists,
    /// under an index one off, so the load succeeds and the model
    /// produces fluent noise. Checking the table against the string it
    /// was transcribed from is the only reading that catches it.
    #[test]
    fn the_schedules_are_the_patterns_the_configs_publish() {
        for (pattern, schedule) in [(PATTERN_52, SCHEDULE_52), (PATTERN_98, SCHEDULE_98)] {
            assert_eq!(
                pattern.len(),
                schedule.len(),
                "the schedule states {} layers and the pattern states {}",
                schedule.len(),
                pattern.len()
            );
            let spelled: String = schedule.iter().map(|k| k.glyph()).collect();
            assert_eq!(
                spelled, pattern,
                "the transcribed schedule is not the published one"
            );
        }
    }

    /// The attention layers are where the config puts them.
    ///
    /// Stated separately from the glyph check because these four (and
    /// five) indices are what `KvStyle::Paged` provisions pages for, and
    /// naming them is cheaper to read than counting `*` in a string.
    #[test]
    fn attention_lands_on_the_published_indices() {
        let attn = |s: &[NemotronLayerKind]| -> Vec<usize> {
            s.iter()
                .enumerate()
                .filter(|(_, k)| **k == Attention)
                .map(|(i, _)| i)
                .collect()
        };
        assert_eq!(attn(SCHEDULE_52), vec![7, 18, 29, 40]);
        assert_eq!(attn(SCHEDULE_98), vec![17, 38, 49, 60, 86]);
    }

    /// The tail of the 47B is why this is a list.
    ///
    /// Layers 75-77 are three MLP layers in a row and 78-79 are two
    /// Mamba layers in a row. Every alternation this crate's other
    /// hybrids use — layer `l` is a mixer iff `(l + 1) % interval == 0`,
    /// or iff `l` is even — puts a non-mixer between every pair of
    /// mixers by construction, so no period reproduces either run. The
    /// list is not verbosity; it is the only form that can say this.
    #[test]
    fn no_interval_generates_the_47b_schedule() {
        let run_of = |kind: NemotronLayerKind, want: usize| {
            (0..=SCHEDULE_98.len() - want)
                .find(|&start| SCHEDULE_98[start..start + want].iter().all(|k| *k == kind))
        };
        assert_eq!(
            run_of(Mlp, 3),
            Some(75),
            "three consecutive mixerless layers, which no alternation produces"
        );
        assert_eq!(
            run_of(Mamba, 2),
            Some(78),
            "and two consecutive scans right after them"
        );

        // The check the run above stands in for: the ONE rule that fits
        // the first three quarters of this stack gets the last quarter
        // wrong.
        let alternating: Vec<NemotronLayerKind> = (0..SCHEDULE_98.len())
            .map(|l| match SCHEDULE_98[l] {
                Attention => Attention,
                _ if l % 2 == 0 => Mamba,
                _ => Mlp,
            })
            .collect();
        assert_ne!(
            alternating.as_slice(),
            SCHEDULE_98,
            "if a period reproduced this schedule the list would be dead weight"
        );
    }

    /// Every fixture states a stack that could exist.
    #[test]
    fn every_fixture_states_a_stack_that_could_exist() {
        for f in &[
            NemotronHFacts::nemotron_h_4b(),
            NemotronHFacts::nemotron_h_8b(),
            NemotronHFacts::nemotron_h_47b(),
            NemotronHFacts::nemotron_h_synthetic(),
        ] {
            assert!(
                f.hidden > 0 && f.vocab > 0,
                "a stack of zero width has nothing to mix"
            );
            assert_eq!(f.layers(), f.layer_types.len() as u32);
            assert!(f.layers() > 0);
            assert!(
                f.attn.heads % f.attn.kv_heads == 0,
                "{} query heads do not group evenly over {} kv heads",
                f.attn.heads,
                f.attn.kv_heads
            );
            let m = &f.mamba;
            assert!(m.num_heads > 0 && m.head_dim > 0 && m.state_size > 0);
            assert!(
                m.conv_kernel > 0,
                "a convolution of width zero reads nothing"
            );
            assert!(
                m.n_groups > 0 && m.intermediate() % m.n_groups == 0,
                "B and C are shared across {} groups, which must divide the scan's width",
                m.n_groups
            );
            // A mixture states all of its numbers or none of them. A
            // `top_k` of zero over 32 experts routes every token to
            // nothing; 32 experts with a `moe_intermediate` of zero
            // gives each of them no width to compute in.
            if f.is_mixture() {
                assert!(f.moe.top_k > 0 && f.moe.top_k <= f.moe.num_experts);
                assert!(f.moe.moe_intermediate > 0);
            } else {
                assert_eq!(
                    f.moe.top_k, 0,
                    "a dense stack has no router to take a k from"
                );
                assert_eq!(f.moe.shared_intermediate, 0);
                assert!(
                    f.moe.moe_intermediate > 0,
                    "the dense MLP's width lives here"
                );
            }
            assert!(
                f.window_left.is_empty(),
                "no Nemotron-H config states a sliding window"
            );
        }
    }

    /// All THREE kinds, or the fixture is not exercising the schedule
    /// this family alone has.
    #[test]
    fn the_fixtures_carry_every_layer_kind() {
        for f in &[
            NemotronHFacts::nemotron_h_8b(),
            NemotronHFacts::nemotron_h_synthetic(),
        ] {
            for k in [Mamba, Attention, Mlp] {
                assert!(
                    (0..f.layers()).any(|l| f.kind(l) == k),
                    "the fixture states no {k:?} layer"
                );
            }
        }
    }

    /// The in-projection packs three things, and the scan reads the
    /// middle one through a conv that is wider than the scan itself.
    #[test]
    fn the_mamba_widths_nest_the_way_the_split_reads_them() {
        let m = NemotronHFacts::nemotron_h_8b().mamba;
        assert_eq!(m.intermediate(), 128 * 64);
        assert_eq!(m.conv_dim(), m.intermediate() + 2 * 8 * 128);
        assert_eq!(
            m.in_proj_width(),
            m.intermediate() + m.conv_dim() + m.num_heads
        );
        assert!(
            m.conv_dim() > m.intermediate(),
            "B and C ride the same buffer as the scan's input"
        );
    }

    /// The slab list names the Mamba layers and only those.
    #[test]
    fn only_the_mamba_layers_carry_a_slab() {
        let f = NemotronHFacts::nemotron_h_8b();
        let mamba = f.mamba_layers();
        assert_eq!(mamba.len(), 24, "the 52-layer schedule has 24 Mamba layers");
        assert!(mamba.iter().all(|&l| f.kind(l) == Mamba));
        assert_eq!(
            mamba.len() + 4 + 24,
            f.layers() as usize,
            "24 Mamba, 4 attention and 24 MLP layers account for all 52"
        );
    }

    /// The three published rows are dense; the fixture is routed.
    #[test]
    fn the_published_checkpoints_are_dense_and_state_their_dense_width_once() {
        for (f, width) in &[
            (NemotronHFacts::nemotron_h_4b(), 12_288),
            (NemotronHFacts::nemotron_h_8b(), 21_504),
            (NemotronHFacts::nemotron_h_47b(), 30_720),
        ] {
            assert!(!f.is_mixture(), "no published Nemotron-H routes");
            assert_eq!(f.moe.moe_intermediate, *width, "the dense MLP's width");
            assert_eq!(f.moe.num_experts, 0);
        }
        assert!(NemotronHFacts::nemotron_h_synthetic().is_mixture());
    }

    /// The 4B and the 8B differ in exactly the ways the configs do.
    ///
    /// Three fixtures written out in full agree where they should and
    /// differ where they should, and this is what says so: the fields
    /// that are the same in both configs are asserted equal, and the
    /// ones the checkpoints disagree about are asserted different, so a
    /// copy-paste that carried a width across is a failure rather than a
    /// second 8B under a 4B's name.
    #[test]
    fn the_narrower_siblings_inherit_only_what_they_share() {
        let (small, big) = (
            NemotronHFacts::nemotron_h_4b(),
            NemotronHFacts::nemotron_h_8b(),
        );
        assert_eq!(
            small.layer_types, big.layer_types,
            "one schedule, two widths"
        );
        assert_eq!(small.attn, big.attn, "32 over 8 heads of 128 in both");
        assert_eq!(small.vocab, big.vocab);
        assert_ne!(small.hidden, big.hidden);
        assert_ne!(small.mamba.num_heads, big.mamba.num_heads);
        assert_ne!(small.moe.moe_intermediate, big.moe.moe_intermediate);

        let huge = NemotronHFacts::nemotron_h_47b();
        assert_eq!(
            huge.mamba.state_size, 256,
            "the 47B doubles the scan's state"
        );
        assert_eq!(huge.attn.heads, 64);
        assert_eq!(huge.layers(), 98);
    }

    /// A glyph per kind, and no two kinds sharing one.
    #[test]
    fn each_layer_kind_spells_itself_distinctly() {
        let glyphs: Vec<char> = [Mamba, Attention, Mlp].iter().map(|k| k.glyph()).collect();
        assert_eq!(glyphs, vec!['M', '*', '-']);
        let mut sorted = glyphs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            glyphs.len(),
            "two kinds spelled the same are one kind"
        );
    }

    /// Nothing here is tied.
    #[test]
    fn no_published_nemotron_ties_its_embedding_table() {
        for f in &[
            NemotronHFacts::nemotron_h_4b(),
            NemotronHFacts::nemotron_h_8b(),
            NemotronHFacts::nemotron_h_47b(),
        ] {
            assert!(
                !f.tied_embeddings,
                "`tie_word_embeddings` is false in all three configs, and defaulting it \
                 the other way would leave a 131 072-row `lm_head` unbound"
            );
        }
    }
}
