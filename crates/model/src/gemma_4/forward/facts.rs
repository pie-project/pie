//! `gemma-4`'s load-time facts.

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
/// the driver reads per layer from its own config, which is why nothing
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Whether layer `l` runs FULL attention — the same predicate
    /// [`Qwen35HybridFacts::is_full_attn`] states, because the two
    /// families schedule their layer kinds the same way.
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// Whether layer `l` reuses another layer's KV pages, projecting and
    /// writing none of its own.
    pub fn is_kv_shared(&self, l: u32) -> bool {
        l >= self.layers.saturating_sub(self.kv_shared_layers)
    }

    /// This layer's MLP width. The double-wide variant widens exactly
    /// the KV-shared layers, which is why it keys on the same predicate
    /// rather than on a second count.
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
    pub fn kv_source(&self, l: u32) -> Option<u32> {
        if !self.is_kv_shared(l) {
            return None;
        }
        let first_shared = self.layers - self.kv_shared_layers;
        (0..first_shared)
            .rev()
            .find(|&j| self.is_full_attn(j) == self.is_full_attn(l))
    }

    /// This layer's head dim — the per-layer axis that makes gemma-4 its
    /// own family.
    pub fn head_dim_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// `google/gemma-4-E4B-it`, read from the checkpoint's own
    /// `config.json` (`text_config`) — every value a field of that file
    /// or the driver's stated derivation from one.
    pub fn gemma_4_e4b() -> Self {
        Self {
            hidden: 2560,
            layers: 42,
            full_attn_interval: 6,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
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
    pub fn gemma_4_e2b() -> Self {
        Self {
            hidden: 1536,
            layers: 35,
            full_attn_interval: 5,
            q_heads: 8,
            kv_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            // 0.25 * 512 through `max(2, 2 * int(0.5 * f * d))`.
            global_rotary_dim: 128,
            intermediate: 6144,
            vocab: 262144,
            tied_embeddings: true,
            kv_shared_layers: 20,
            ple_dim: 256,
            double_wide_shared: true,
            logit_softcap: 30.0,
        }
    }
}

#[cfg(test)]
mod gemma4_tests {
    use super::Gemma4Facts;

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
}

/// The CUDA backend's load-time facts for gemma-4 — the BINDING
/// questions its class traces resolve at trace time.
///
/// Three, and all three are "what did the loader materialise", which is
/// the taxonomy's first row: a load-time fact is a trace-time `match`,
/// erased.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4CudaFacts {
    /// The loader bound one packed `[Hq + 2*Hk, hidden]` projection
    /// (`qkv_proj_fused`) — llama_like's `fused_qkv`, same question.
    pub fused_qkv: bool,
    /// The loader bound a packed gate‖up bank — llama_like's
    /// `gate_up_fused`, same question, different activation behind it.
    pub gate_up_fused: bool,
    /// The KV cache is native bf16, so the fused decode post may write
    /// pages directly. One of the four terms
    /// `can_fuse_packed_qkv_post` reads; the other three are the
    /// declaration's own (`partial` is a layer-kind fact, hooks and the
    /// fire class are class/guard vocabulary).
    pub kv_native_bf16: bool,
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

impl Gemma4CudaFacts {
    /// SYNTHETIC fixture — the same standing caveat every `*CudaFacts`
    /// constructor here carries: it pins the GOLDEN FORM of the traced
    /// arms, not a deployment's truth. The live derivation and its
    /// digest are the executor rung's, and the digest is what corrects a
    /// guess on first boot.
    pub fn gemma_4_e4b_synthetic() -> Self {
        Self {
            // The fixture attends the whole context; a live gemma-4
            // deployment states its per-layer list.
            window_left: Vec::new(),
            fused_qkv: true,
            gate_up_fused: true,
            kv_native_bf16: true,
        }
    }
}

// ── gpt-oss ────────────────────────────────────────────────────────────

