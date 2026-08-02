//! `gpt-oss`'s load-time facts.


/// gpt-oss's shape. The family rides `mixtral.cpp`, so these are the
/// facts that text reads — not a checkpoint dump.
///
/// Two of them are here because the driver ANSWERS them per layer and
/// the declaration would otherwise have to re-derive them per fire: the
/// alternating window kind, and whether a layer carries attention sinks.
/// Both are load-time, so both erase at trace time.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// One expert's MLP width (`intermediate_size`). gpt-oss's is equal
    /// to `hidden`, which is a coincidence of this checkpoint and not a
    /// rule the text may lean on.
    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,
    /// `swiglu_limit`; 0 means the plain SwiGLU. gpt-oss clamps at 7.0,
    /// and the clamp is a DIFFERENT KERNEL, so this decides which
    /// activation the text states rather than being a runtime scalar.
    pub swiglu_limit: f32,
    /// Whether the checkpoint biases q/k/v/o, the router, and the expert
    /// projections (`attention_bias`). gpt-oss biases all of them; the
    /// q/k/v biases FOLD INTO the projection's epilogue and the rest are
    /// their own launches.
    pub attention_bias: bool,
    /// Whether this deployment's rope is the YaRN-paper one. gpt-oss's
    /// config asks for it (factor 32 over an original 4096 context) and
    /// the driver resolves it at load, so it is a fact and not a fire's
    /// question — and a WRONG one here is not a crash but a silently
    /// unscaled rotation, which is how it went unnoticed.
    pub rope_yarn_original: bool,
    /// Every layer carries `attn_sinks` on gpt-oss. The driver asks
    /// `layer.attn_sinks != nullptr` per layer and only requests an LSE
    /// from attention where the answer is yes — so this is what decides
    /// whether the attention statement produces one value or two.
    pub attn_sinks: bool,
}

impl GptOssFacts {
    /// Whether layer `l` attends over the SLIDING window. gpt-oss
    /// alternates from layer 0 (`layer_types` reads
    /// sliding, full, sliding, full, …), which the driver reaches
    /// through `per_layer_window_left` — a scalar the text does not
    /// state, since the window is an argument and not a kernel.
    pub fn is_sliding(&self, l: u32) -> bool {
        l % 2 == 0
    }

    /// openai/gpt-oss-20b, read from the checkpoint's `config.json`
    /// (2026-08-06). `layer_types` alternates from sliding; the yarn
    /// `rope_scaling` is NOT in this list because the driver never
    /// applies it — `mixtral.cpp:320` passes the plain `rope_theta`, a
    /// latent bug this declaration must not launder into a fact.
    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
            attention_bias: true,
            rope_yarn_original: true,
            attn_sinks: true,
        }
    }
}

/// The CUDA backend's answers for a gpt-oss deployment — the bindings
/// and the admission thresholds, all resolved at load.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssCudaFacts {
    /// Whether the layer bank carries the per-expert POINTER ARRAYS the
    /// fused decode GEMV indexes (`expert_gate_up_packed_ptrs`). Built
    /// by the `RoutedDecode` MXFP4 policy, which is the engine default;
    /// the `NativeGemm` policy binds marlin views instead and reaches
    /// the experts through a per-expert loop no rectangle spells.
    pub mxfp4_decode_gemv: bool,
    /// `mxfp4_decode_max_routes` — the fused leg's admission threshold
    /// in ROUTES (`N * top_k`), default `32 * experts`. A fire past it
    /// takes the host-routed walk, which this declaration refuses by
    /// name rather than states.
    pub mxfp4_decode_max_routes: u32,
    /// Whether the experts are STREAMED through a slab cache. A streamed
    /// layer reaches the same fused kernels, but only after a host
    /// round-trip that decides what to page in — so a streamed
    /// deployment is outside the flat list until that is stated.
    pub streamed_experts: bool,
}

impl GptOssCudaFacts {
    /// The L40S deployment's set, as the driver derives it: no
    /// streaming, the default policy's pointer arrays, and the default
    /// cap. SYNTHETIC until a live digest judges it — the standing
    /// contract for every `*_synthetic` fixture in this file.
    pub fn gpt_oss_20b_synthetic() -> Self {
        Self {
            mxfp4_decode_gemv: true,
            mxfp4_decode_max_routes: 32 * 32,
            streamed_experts: false,
        }
    }
}

