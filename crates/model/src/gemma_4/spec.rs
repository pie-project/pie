//! Gemma-4's semantic shape: the numbers a row IS.
//!
//! Ungated, and that is the whole reason this file exists apart from
//! `forward/facts.rs`. A row has to answer `manifest()`, `load_shape()`
//! and `deployment()` in a build with no `forward` feature at all — the
//! contract-only builds do exactly that — so the struct those three
//! answers are projected from cannot live behind `#[cfg(feature =
//! "forward")]`. What stays next door is the BACKEND's facts
//! ([`super::forward::facts::Gemma4CudaFacts`]): what the loader
//! materialised, built at trace time, per-layer `Vec`s and all.
//!
//! The dividing line is not "semantic vs backend" as a taste; it is
//! `const`. A row is a `const` item in `.rodata`, so a shape a row holds
//! may not own a heap allocation, and the two questions "how wide is
//! this model" and "did this load fuse its QKV" fall on opposite sides
//! of that line by themselves.

use serde::{Deserialize, Serialize};

/// Facts for the GEMMA-4 family — the third declared family, and the
/// first whose per-layer axis carries TWO HEAD DIMS.
///
/// # What makes this family its own declaration
///
/// gemma-4 alternates sliding and full attention on a regular interval,
/// like qwen3_5's hybrid — but where qwen3_5's two layer kinds are two
/// different ATTENTIONS, gemma-4's are the same attention at different
/// geometry: `head_dim` 256 on a sliding layer, `global_head_dim` 512 on
/// a full one, with partial rope on the full layers only. The window
/// itself is not trace vocabulary at all: it is a scalar `window_left`
/// the driver reads per layer from the deployment, which is why nothing
/// here names it.
///
/// Two things ARE structural and have no analogue in the families
/// declared so far:
///
/// * **KV sharing.** The last [`Self::kv_shared_layers`] layers project
///   no k/v, norm no k/v, rope no k, and write no cache — they attend
///   through the pages of the last earlier layer of the SAME kind. The
///   elision is per layer and total, so it is a fact the trace reads to
///   decide which statements exist, not a runtime branch.
/// * **PLE** (per-layer embeddings). A prologue that embeds a SECOND
///   table, projects it to `layers * ple_dim`, norms and scales it, and
///   transposes so each layer reads a contiguous slice; then a per-layer
///   epilogue gates that slice into the residual stream. It is the
///   reason gemma-4 cannot be a `llama_like` configuration.
///
/// (There is no altup here. That is gemma3n's mechanism —
/// `crates/driver-cuda/csrc/src/model/gemma4/` never mentions it.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Facts {
    pub hidden: u32,
    pub layers: u32,
    /// Full attention every `interval`-th layer, `l % interval ==
    /// interval - 1` — qwen3_5's formula, and the config agrees
    /// (E4B: full at 5, 11, …, 41 with interval 6).
    pub full_attn_interval: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// The SLIDING layers' head dim (`head_dim`).
    pub head_dim: u32,
    /// The FULL layers' head dim (`global_head_dim`). Different from
    /// [`Self::head_dim`] on E4B (512 vs 256), which is why the two
    /// kinds cannot share one width the way qwen3_5's do.
    pub global_head_dim: u32,
    /// `num_global_key_value_heads` — the FULL layers' kv-head count.
    ///
    /// The second half of the head shape, and the half that had no
    /// field. `global_head_dim` alone says a full layer's heads are
    /// wider; it does not say there are fewer of them, and there are:
    /// the 31b runs 16 sliding against 4 global, the 26b 8 against 2.
    /// A pool sizing a full layer's page at the sliding count reads
    /// three quarters past the end of its K — not a crash, a fluent
    /// model reading another layer's memory.
    ///
    /// HF defaults this key to `num_key_value_heads` when the config
    /// omits it, which is what the E-series rows state: one shape's
    /// count at two widths.
    pub global_kv_heads: u32,
    /// Partial-rotary width on the FULL layers, resolved the driver's
    /// way (`max(2, 2 * int(0.5 * factor * head_dim))`): 0.25 × 512
    /// gives 128. Sliding layers rotate fully.
    pub global_rotary_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,
    /// `num_kv_shared_layers` — the count of TRAILING layers that reuse
    /// an earlier layer's pages. `first_shared = layers - this`.
    pub kv_shared_layers: u32,
    /// `hidden_size_per_layer_input` — the PLE slice width per layer.
    pub ple_dim: u32,
    /// `use_double_wide_mlp`: the KV-SHARED layers carry an MLP of
    /// `2 * intermediate`. E2B sets it, E4B does not — the first
    /// gemma-4 axis where two deployments of one family disagree about
    /// a WIDTH rather than a count, so it is a fact and the widths it
    /// implies erase at trace time.
    pub double_wide_shared: bool,
    /// `final_logit_softcapping`; 0 means no cap.
    pub logit_softcap: f32,
}

impl Gemma4Facts {
    /// Whether layer `l` runs FULL attention — the same predicate the
    /// qwen3.5 hybrid states, because the two families schedule their
    /// layer kinds the same way.
    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// Whether layer `l` reuses another layer's KV pages, projecting and
    /// writing none of its own.
    #[must_use]
    pub fn is_kv_shared(&self, l: u32) -> bool {
        l >= self.layers.saturating_sub(self.kv_shared_layers)
    }

    /// This layer's MLP width. The double-wide variant widens exactly
    /// the KV-shared layers, which is why it keys on the same predicate
    /// rather than on a second count.
    #[must_use]
    pub fn intermediate_of(&self, l: u32) -> u32 {
        if self.double_wide_shared && self.is_kv_shared(l) {
            self.intermediate * 2
        } else {
            self.intermediate
        }
    }

    /// The layer whose pages `l` attends through: the last EARLIER layer
    /// of the same kind (`gemma4.cpp`'s load-time search). `None` for a
    /// layer that owns its pages.
    #[must_use]
    pub fn kv_source(&self, l: u32) -> Option<u32> {
        if !self.is_kv_shared(l) {
            return None;
        }
        let first_shared = self.layers.saturating_sub(self.kv_shared_layers);
        (0..first_shared)
            .rev()
            .find(|&j| self.is_full_attn(j) == self.is_full_attn(l))
    }

    /// This layer's head dim — the per-layer axis that makes gemma-4 its
    /// own family.
    #[must_use]
    pub fn head_dim_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// This layer's kv-head COUNT, which varies with the same predicate
    /// and by a different factor. See [`Self::global_kv_heads`].
    #[must_use]
    pub fn kv_heads_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_kv_heads
        } else {
            self.kv_heads
        }
    }

    /// `google/gemma-4-E4B-it`, read from the checkpoint's own
    /// `config.json` (`text_config`) — every value a field of that file
    /// or the driver's stated derivation from one.
    #[must_use]
    pub const fn gemma_4_e4b() -> Self {
        Self {
            hidden: 2560,
            layers: 42,
            full_attn_interval: 6,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            // `num_global_key_value_heads` is absent from this
            // row's config, and HF reads that as the sliding count.
            global_kv_heads: 2,
            // partial_rotary_factor 0.25 on `global_head_dim` 512.
            global_rotary_dim: 128,
            intermediate: 10_240,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 18,
            ple_dim: 256,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }

    /// gemma-4-E2B-it, read from the checkpoint's `config.json`
    /// (2026-08-07). The SECOND geometry, and it disagrees with E4B on
    /// nearly every axis that matters: 35 layers (odd, so the interval
    /// does not divide it), `kv_heads = 1` (MQA where E4B is GQA-4), 20
    /// of 35 KV-shared, tied embeddings, and `use_double_wide_mlp` — the
    /// only gemma-4 fixture where `intermediate_of` is not constant.
    ///
    /// Live-anchored: the driver's derivation on this checkpoint prints
    /// `layers=35 interval=5 shared=20 d=256/512`, and traces 488 decode
    /// / 524 prefill ops. Both classes are byte-identical to the
    /// hand-written pass on the parity gate.
    #[must_use]
    pub const fn gemma_4_e2b() -> Self {
        Self {
            hidden: 1536,
            layers: 35,
            full_attn_interval: 5,
            q_heads: 8,
            kv_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            // `num_global_key_value_heads` is absent from this
            // row's config, and HF reads that as the sliding count.
            global_kv_heads: 1,
            // 0.25 * 512 through `max(2, 2 * int(0.5 * f * d))`.
            global_rotary_dim: 128,
            intermediate: 6144,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 20,
            ple_dim: 256,
            double_wide_shared: true,
            logit_softcap: 30.0,
        }
    }

    /// `google/gemma-4-31B-it`, from its own `config.json`
    /// (`text_config`) — the DENSE 31B, and the only gemma-4 in the
    /// corpus this repo has real weights and an MLX reference for.
    ///
    /// It was missing from this table, which is a quieter failure than
    /// a wrong number: a checkpoint that matches no row is refused as
    /// "no model this build serves", and the model every gemma fix in
    /// the driver had been verified against stopped being servable
    /// without anything saying so.
    ///
    /// The two head shapes are its own and are the reason
    /// [`Self::global_kv_heads`] exists: 16x256 on a sliding layer,
    /// 4x512 on a full one, so BOTH numbers change and neither implies
    /// the other. `num_kv_shared_layers` is 0 and
    /// `hidden_size_per_layer_input` is 0, so there is no sharing and
    /// no PLE — the E-series' two defining structures are both absent
    /// here.
    #[must_use]
    pub const fn gemma_4_31b() -> Self {
        Self {
            hidden: 5376,
            layers: 60,
            full_attn_interval: 6,
            q_heads: 32,
            kv_heads: 16,
            head_dim: 256,
            global_head_dim: 512,
            global_kv_heads: 4,
            // partial_rotary_factor 0.25 on `global_head_dim` 512, from
            // this row's `rope_parameters.full_attention`.
            global_rotary_dim: 128,
            intermediate: 21_504,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 0,
            ple_dim: 0,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }

    /// `google/gemma-4-26B-A4B-it`, from its own `config.json`
    /// (`text_config`) — the MIXTURE geometry, and the one gemma-4 in
    /// the corpus that shares NO kv at all.
    ///
    /// Three of its numbers are the ones that read oddly, and all three
    /// are the file's: `num_kv_shared_layers` is 0 where every other
    /// gemma-4 shares half its stack, `hidden_size_per_layer_input` is 0
    /// so there is no PLE, and `intermediate_size` 2112 is the width of
    /// the DENSE MLP that sits beside the experts rather than of an
    /// expert (704, on [`Gemma4Mixture`]). The mixture and the
    /// `attention_k_eq_v` mode it ships with are the reason this row
    /// refuses to deploy — see `super::mod`'s `Gemma4::deployment`.
    #[must_use]
    pub const fn gemma_4_26b_a4b() -> Self {
        Self {
            hidden: 2816,
            layers: 30,
            full_attn_interval: 6,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 256,
            global_head_dim: 512,
            global_kv_heads: 2,
            global_rotary_dim: 128,
            intermediate: 2112,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 0,
            ple_dim: 0,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }
}

/// The routed part of a gemma-4 that has one.
///
/// A SEPARATE struct, and `Option` on the row, because a mixture is not
/// a set of fields a dense row leaves at zero — it is a different set of
/// tensors. `Gemma4Facts` describes the stack every gemma-4 has;
/// this describes the bank only the A4B ships, and a row holding
/// `None` cannot accidentally publish a router of width zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Mixture {
    /// Routed experts per layer.
    pub num_experts: u32,
    /// How many of them a token reaches — `top_k_experts`.
    pub experts_per_token: u32,
    /// ONE expert's inner width, which is not
    /// [`Gemma4Facts::intermediate`]: on the A4B the dense width is 2112
    /// and an expert is 704.
    pub moe_intermediate: u32,
}

impl Gemma4Mixture {
    /// `google/gemma-4-26B-A4B-it`: 128 experts of 704, eight of them
    /// live per token — which is where the name's `A4B` comes from.
    #[must_use]
    pub const fn gemma_4_26b_a4b() -> Self {
        Self { num_experts: 128, experts_per_token: 8, moe_intermediate: 704 }
    }
}

#[cfg(test)]
mod tests {
    use super::{Gemma4Facts, Gemma4Mixture};

    /// Every shape this module ships, so a property proved below is
    /// proved about all of them rather than about whichever one the test
    /// happened to name.
    fn fixtures() -> Vec<(&'static str, Gemma4Facts)> {
        vec![
            ("e4b", Gemma4Facts::gemma_4_e4b()),
            ("e2b", Gemma4Facts::gemma_4_e2b()),
            ("26b-a4b", Gemma4Facts::gemma_4_26b_a4b()),
        ]
    }

    /// The layer-kind schedule the checkpoint states, reproduced by the
    /// formula. `config.json` lists `full_attention` at 5, 11, 17, 23,
    /// 29, 35, 41 — seven of forty-two — and the interval must generate
    /// exactly that set, not merely contain it.
    #[test]
    fn the_schedule_is_the_one_the_config_lists() {
        let f = Gemma4Facts::gemma_4_e4b();
        let full: Vec<u32> = (0..f.layers).filter(|&l| f.is_full_attn(l)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41]);
    }

    /// The A4B's own schedule, from its own file: 30 layers on the same
    /// interval of six, so full attention lands on 5, 11, 17, 23, 29 and
    /// the last layer of the stack is a full one.
    #[test]
    fn the_mixture_row_schedules_five_full_layers_of_thirty() {
        let f = Gemma4Facts::gemma_4_26b_a4b();
        let full: Vec<u32> = (0..f.layers).filter(|&l| f.is_full_attn(l)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29]);
    }

    /// KV sharing: the trailing 18 layers own no pages, and each attends
    /// through the last EARLIER layer of its own kind. On E4B that means
    /// every shared SLIDING layer lands on 22 and every shared FULL one
    /// on 23 — the driver's load-time search, as a fact.
    #[test]
    fn every_shared_layer_finds_a_source_of_its_own_kind() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert_eq!(f.layers - f.kv_shared_layers, 24);
        for l in 0..f.layers {
            match f.kv_source(l) {
                None => assert!(!f.is_kv_shared(l), "layer {l} shares but found no source"),
                Some(src) => {
                    assert!(f.is_kv_shared(l));
                    assert!(src < 24, "layer {l} sources from a sharing layer {src}");
                    assert_eq!(
                        f.is_full_attn(src),
                        f.is_full_attn(l),
                        "layer {l} sources from the other attention kind"
                    );
                }
            }
        }
        assert_eq!(f.kv_source(24), Some(22));
        assert_eq!(f.kv_source(29), Some(23));
        assert_eq!(f.kv_source(41), Some(23));
    }

    /// A stack that shares nothing owns every page, and says so by
    /// answering `None` everywhere rather than by naming itself.
    #[test]
    fn a_stack_that_shares_nothing_sources_no_layer() {
        let f = Gemma4Facts::gemma_4_26b_a4b();
        assert_eq!(f.kv_shared_layers, 0);
        for l in 0..f.layers {
            assert!(!f.is_kv_shared(l), "layer {l} shares in a stack that shares nothing");
            assert_eq!(f.kv_source(l), None, "layer {l} named a source it does not read");
        }
    }

    /// The search can come up empty, and the shape that makes it do so
    /// is a real published one: `google/gemma-4-E4B-it-assistant` states
    /// four layers and `num_kv_shared_layers: 4`, so EVERY layer shares
    /// and there is no earlier layer of any kind to share from — its KV
    /// comes from the E4B backbone it rides, which is not in this stack.
    ///
    /// The answer is `None`, and a projection reading it must land the
    /// layer on itself rather than on a wrong neighbour. That checkpoint
    /// has no row here for exactly this reason.
    #[test]
    fn a_stack_shared_end_to_end_finds_no_source_at_all() {
        let f = Gemma4Facts {
            layers: 4,
            kv_shared_layers: 4,
            full_attn_interval: 4,
            hidden: 256,
            q_heads: 4,
            kv_heads: 2,
            intermediate: 2048,
            ple_dim: 0,
            logit_softcap: 0.0,
            ..Gemma4Facts::gemma_4_e4b()
        };
        for l in 0..f.layers {
            assert!(f.is_kv_shared(l));
            assert_eq!(f.kv_source(l), None, "layer {l} found a source that cannot exist");
        }
    }

    /// The two head dims are the per-layer axis. A family that had one
    /// would not need this fact at all, which is why it is worth a test
    /// that says the widths actually differ.
    #[test]
    fn the_two_layer_kinds_have_different_head_dims() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert_ne!(f.head_dim, f.global_head_dim);
        assert_eq!(f.head_dim_of(0), 256);
        assert_eq!(f.head_dim_of(5), 512);
        assert_eq!(f.global_rotary_dim, 2 * (0.5 * 0.25 * 512.0) as u32);
    }

    /// The double-wide MLP widens the shared tail and nothing else. E2B
    /// is the only fixture where the width is not constant across the
    /// stack, and a projection that read `intermediate` directly would
    /// size 20 of its 35 layers at half what they need.
    #[test]
    fn double_wide_widens_exactly_the_shared_tail() {
        let e2b = Gemma4Facts::gemma_4_e2b();
        assert!(e2b.double_wide_shared);
        assert_eq!(e2b.intermediate_of(0), 6144);
        assert_eq!(e2b.intermediate_of(14), 6144);
        assert_eq!(e2b.intermediate_of(15), 12_288, "the first shared layer is not widened");
        assert_eq!(e2b.intermediate_of(34), 12_288);

        let e4b = Gemma4Facts::gemma_4_e4b();
        assert!(!e4b.double_wide_shared);
        for l in 0..e4b.layers {
            assert_eq!(
                e4b.intermediate_of(l),
                e4b.intermediate,
                "E4B widened layer {l}, and its config does not ask for that"
            );
        }
    }

    /// Every fixture states a stack that could exist. Positive widths, a
    /// GQA ratio that divides, a schedule that fits inside the stack and
    /// a shared tail that is not longer than the stack it trails.
    #[test]
    fn every_fixture_states_a_stack_that_could_exist() {
        for (name, f) in fixtures() {
            assert!(f.hidden > 0 && f.layers > 0 && f.vocab > 0, "{name} states a zero extent");
            assert!(f.q_heads > 0 && f.kv_heads > 0, "{name} states a stack with no heads");
            assert_eq!(
                f.q_heads % f.kv_heads,
                0,
                "{name}: {} query heads do not group evenly over {} kv heads",
                f.q_heads,
                f.kv_heads
            );
            assert!(f.head_dim > 0 && f.global_head_dim >= f.head_dim, "{name} head dims");
            assert!(
                f.global_rotary_dim > 0 && f.global_rotary_dim <= f.global_head_dim,
                "{name} rotates {} of a {}-wide head",
                f.global_rotary_dim,
                f.global_head_dim
            );
            assert!(f.intermediate > 0, "{name} states an MLP of no width");
            assert!(
                f.full_attn_interval > 0 && f.full_attn_interval <= f.layers,
                "{name}: an interval of {} over {} layers schedules nothing",
                f.full_attn_interval,
                f.layers
            );
            assert!(
                f.kv_shared_layers < f.layers,
                "{name} shares every layer, so no layer owns a page"
            );
            assert!(
                (0..f.layers).any(|l| f.is_full_attn(l)),
                "{name} schedules no full-attention layer at all"
            );
        }
    }

    /// The mixture states all its numbers or the row holds none: 128
    /// experts of 704 with 8 live, which is a routed bank that could
    /// exist. The check that matters is `experts_per_token <=
    /// num_experts` — a top-k wider than the bank is the one arithmetic
    /// error a router cannot survive.
    #[test]
    fn the_mixture_states_all_its_numbers() {
        let m = Gemma4Mixture::gemma_4_26b_a4b();
        assert_eq!(m.num_experts, 128);
        assert_eq!(m.experts_per_token, 8);
        assert_eq!(m.moe_intermediate, 704);
        assert!(m.experts_per_token > 0 && m.experts_per_token <= m.num_experts);
        assert!(m.moe_intermediate > 0);
        assert_ne!(
            m.moe_intermediate,
            Gemma4Facts::gemma_4_26b_a4b().intermediate,
            "an expert's width and the dense width are different numbers on this checkpoint, \
             and a projection that used one for the other would size the workspace wrong"
        );
    }
}
