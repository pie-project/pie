//! The decode-step ABI: regions, IO slots, kernel kinds, and the graph key.
//!
//! `decode_abi.hpp` is the backend-agnostic contract three lanes share — the
//! heap allocator, the Metal-4 wrappers and the encoder — and declares
//! itself pure: "NO Metal/ObjC types in this header". This module is its
//! vocabulary half: the heap [`Region`]s, the [`IoSlot`] table, the
//! [`Kernel`] kind enum every attribution/PSO/weight table is indexed by,
//! [`ArgmaxParams`], and the [`ForwardGraphKey`] the command-buffer cache
//! buckets on. The ~30 `bind::` argument-table layouts are *not* here: each
//! is the ABI of one kernel and lands beside the encoder that binds it,
//! where its slot documentation has something to be checked against.
//!
//! ## The count that was forty kinds short
//!
//! Every table indexed by `Kernel` is sized from the kind count, and the
//! C++ once spelled that count `G4PleResidual + 1` — forty-four kinds short
//! of the real end. `psos[LmHeadUntied]` then wrote and read past the
//! array, the untied head's dispatch got the multi-batch table's GDN
//! pipeline, and the logits buffer was left exactly as it found it: every
//! logit zero, every token 0, and not one error anywhere. The C++ fix made
//! `KindCount` an enum member so it tracks the end by construction. The
//! Rust fix is stronger: the `kernels!` macro emits the variant
//! list and [`Kernel::ALL`]/[`Kernel::COUNT`] from the *same* token list,
//! so a kind appended to the enum is counted because it is the enum — there
//! is no second spelling of the end to fall behind. And a `[T; Kernel::COUNT]`
//! table indexed through [`Kernel::index`] cannot be indexed past, because
//! the index is the discriminant of a value that exists.
//!
//! ## The values were ABI with a peer that is gone
//!
//! The C++ said "APPEND ONLY" five separate times: the numeric values of
//! `Kernel`, `IoSlot` and `Region` were part of the M=1 argument-table ABI
//! and the serialized surfaces it shared with `decode_abi.hpp`. **That
//! header no longer exists**, and nothing outside this module reads a
//! discriminant: `Kernel::index` is called only by these tests, `COUNT`
//! only by this module's own tables, and no `Kernel` is serialized, cached
//! to disk or sent on a wire. The append-only rule was a contract with one
//! reader, and that reader was deleted with the C++ tree.
//!
//! So the anchors below are kept for a narrower reason than the one they
//! were written for. They no longer protect a cross-language agreement;
//! they catch an insertion in the middle of the list, which would silently
//! shift every table this module indexes. When a renumbering IS intended --
//! merging two kinds that were one operation under two family prefixes --
//! they fail loudly and the anchor is moved deliberately, which is what
//! happened to the four values above 84.

/// One fixed region of the decode heap.
///
/// One `MTLHeap`, fixed offsets, residency requested once; per token only
/// the IO slot contents change (invariant I2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Region {
    /// Load-once read-only weights: matvec banks, norms, the tied head.
    Weights = 0,
    /// The M=1 contiguous K/V ring for the full-attention layers.
    Kv = 1,
    /// GDN resident conv + recurrent state, updated in place (I4).
    State = 2,
    /// The activation ping-pong pool ([`SCRATCH_POOL`] buffers).
    Scratch = 3,
    /// Per-token CPU/GPU-touched scalars and the logits.
    Io = 4,
    /// Multi-batch CSR IO buffers (the [`IoSlot`] tail). Zero-sized at M=1.
    MbIo = 5,
    /// The separate NHD paged K/V pool the paged kernels read. The M=1 ring
    /// above is untouched.
    KvPagePool = 6,
}

/// The activation ping-pong pool's size cap.
///
/// The cap, not the allocation: the executor commits `colors_used` slots —
/// six for a dense stack, eight routed, nine routed with a shared expert.
/// Deliberately the current peak and not a round number above it, so the
/// next value that does not fit says so instead of binding to nothing.
pub const SCRATCH_POOL: usize = 9;

/// One slot of the IO region: GPU-read buffers, never encode-time bytes.
///
/// Invariant I1: keeping the scalars in buffers is what makes the encoded
/// command buffer byte-identical every token, so encode(N+1) overlaps
/// GPU(N). M=1 writes index `[0]` of each scalar slot; the CSR tail is
/// bound only for M>1 fires. Values are append-only ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IoSlot {
    /// `u32[max_tokens]` — the fired token ids.
    TokenId = 0,
    /// `u32[max_tokens]` — absolute positions (rope and the KV append read).
    Position = 1,
    /// `u32[max_tokens]` — per-token KV extent (the decode SDPA reads).
    SeqLen = 2,
    /// `bf16[vocab]` out at M=1; `bf16[max_tokens, vocab]` for paged fires.
    Logits = 3,
    /// `u32[max_tokens]` — the optional device-argmax substrate (I3).
    NextToken = 4,
    /// `u32[R+1]` — per-request token spans.
    QoIndptr = 5,
    /// `u32[R+1]` — per-request page-list base.
    KvPageIndptr = 6,
    /// `u32[total_pages]` — flat physical page ids.
    KvPageIndices = 7,
    /// `u32[R]` — fill count of each request's last page.
    KvLastPageLens = 8,
    /// `u32[R]` — recurrent-state slot per request.
    RsSlotIds = 9,
    /// `u8[R]` — per-slot fresh/continue flags.
    RsSlotFlags = 10,
    /// `u32[N]` — per-token owning request.
    ReqOfToken = 11,
    /// `u32[N]` — `rs_slot_ids[req_of_token[t]]`; the slotted GDN kernel
    /// indexes state by token row, and keeping the expansion distinct from
    /// [`RsSlotIds`](Self::RsSlotIds) keeps mixed fires unambiguous.
    SlotOfToken = 12,
    /// `u32[N]` — explicit physical destination page per appended token.
    /// Separate from the read CSR: a fork may write a new page while
    /// retaining a shared prefix.
    WPage = 13,
    /// `u32[N]` — in-page destination offset per appended token.
    WOff = 14,
    /// `u8[N, stride]` — the dense attention allow-mask.
    AttnMask = 15,
    /// `u32[1]` — the dense mask's row stride.
    AttnMaskStride = 16,
    /// `u8[N]` — whether each row consumes the mask.
    AttnMaskEnabled = 17,
    /// `u32[S]` — which body rows the fire samples, in readout order. The
    /// tail runs over these and no others: the LM head is the step's most
    /// expensive dispatch and a prefill reads one row per request.
    SampleRows = 18,
}

/// How many [`IoSlot`]s there are.
pub const IO_SLOT_COUNT: usize = IoSlot::SampleRows as usize + 1;

/// The argmax kernel's constant block, replicated exactly.
///
/// The EOS ids ride inline (at most eight); `n_eos = 0` means the EOS flag
/// never fires. Shared storage, so the executor rewrites vocab and stop ids
/// per generation without a rebind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ArgmaxParams {
    /// The logits row width.
    pub vocab: u32,
    /// Valid entries in [`eos_ids`](Self::eos_ids).
    pub n_eos: u32,
    /// Stop-token ids the device compares the winner against.
    pub eos_ids: [u32; 8],
}

/// The size the Metal side agrees on.
const _: () = assert!(size_of::<ArgmaxParams>() == 40);

/// Declares [`Kernel`] and derives its list and count from one token list.
///
/// This is the structural fix for the count that was forty kinds short:
/// `ALL` and `COUNT` come from the same tokens the variants do, so there is
/// no second spelling of the enum's end to fall behind.
macro_rules! kernels {
    ($(#[$enum_meta:meta])* $vis:vis enum $name:ident {
        $($(#[$meta:meta])* $variant:ident = $kname:literal),+ $(,)?
    }) => {
        $(#[$enum_meta])*
        $vis enum $name {
            $($(#[$meta])* $variant),+
        }
        impl $name {
            /// Every kind, in ABI order.
            $vis const ALL: [$name; [$($name::$variant),+].len()] =
                [$($name::$variant),+];
            /// How many kinds there are — the size of every table indexed
            /// by [`Kernel::index`].
            $vis const COUNT: usize = Self::ALL.len();
            /// This kind's table index: its ABI discriminant.
            #[must_use]
            $vis const fn index(self) -> usize {
                self as usize
            }
            /// The kind's name: the ablation token, the dump tag, and the
            /// attribution row label.
            ///
            /// Total by construction — the macro will not accept a variant
            /// without one. The C++ named kinds in a hand-kept `switch`
            /// with a `default: return "unknown"`, and for a while 50 of
            /// its 99 kinds fell through it: the attribution report was
            /// blind to half the enum and the ablation knob could not name
            /// any of them.
            #[must_use]
            $vis const fn name(self) -> &'static str {
                match self { $($name::$variant => $kname),+ }
            }
            /// The kind a name denotes, if any.
            #[must_use]
            $vis fn from_name(name: &str) -> Option<$name> {
                Self::ALL.into_iter().find(|kind| kind.name() == name)
            }
        }
    };
}

kernels! {
    /// One dispatch kind of the decode DAG.
    ///
    /// A kind is a *weight name* as much as a kernel: `weights_for_kind`
    /// switches on it and nothing else, which is why families that reuse a
    /// kernel under a different tensor name get their own kinds, and why
    /// the numeric values are append-only ABI. Kinds sharing a `.metal`
    /// differ only by dispatch dims and golden tag.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(u8)]
    pub enum Kernel {
        /// Embedding gather off the tied 4-bit head bundle.
        EmbedGather = "embed_gather",
        /// Pre-attention RMS norm.
        Rms = "rms",
        /// GDN in-projection, 4-bit qkv.
        QmvIn = "qmv_in",
        /// GDN in-projection, 4-bit z gate.
        QmvInZ = "qmv_in_z",
        /// GDN `a` projection.
        ///
        /// QUANTIZED, like every other projection in the layer. It and its
        /// `b` sibling were once staged as dense bf16 -- what a Qwen3-Next
        /// preview repack shipped, and what no released checkpoint ships.
        /// Reading `in_proj_a.weight` as bf16 reads packed nibbles as
        /// floats: NaN in the first four output channels, small plausible
        /// wrong numbers in the other twelve, and the model produces token
        /// 0 forever. `model/src/qwen_3_5/forward/mod.rs` states both as
        /// an ordinary `MatW`, the same type every dense projection gets.
        GdnInA = "gdn_in_a",
        /// GDN `b` projection; quantized, see [`Kernel::GdnInA`].
        GdnInB = "gdn_in_b",
        /// The hoisted GDN q/k prologue (one dispatch per head).
        GdnPrep = "gdn_prep",
        /// The fused GDN core: conv+silu, norms, gating, recurrent step.
        GdnCore = "gdn_core",
        /// The gated RMS norm closing the GDN block.
        GatedRms = "gated_rms",
        /// GDN out-projection.
        QmvOut = "qmv_out",
        /// Residual add.
        Residual = "residual",
        /// Attention q projection.
        QmvQ = "qmv_q",
        /// Deinterleave of the 2x-wide gated-q projection.
        QSplit = "q_split",
        /// Attention k projection.
        QmvK = "qmv_k",
        /// Attention v projection.
        QmvV = "qmv_v",
        /// Per-head q norm.
        QNorm = "q_norm",
        /// Per-head k norm.
        KNorm = "k_norm",
        /// Rope on q.
        Rope = "rope",
        /// Rope on k.
        RopeK = "rope_k",
        /// The M=1 contiguous-ring KV append.
        KvAppend = "kv_append",
        /// The M=1 single-pass decode attention.
        Sdpa = "sdpa",
        /// `attn *= sigmoid(gate)` before the o projection.
        AttnGate = "gate",
        /// Attention o projection.
        QmvO = "qmv_o",
        /// Pre-FFN RMS norm.
        FfnRms = "ffn_rms",
        /// FFN gate projection.
        QmvGate = "qmv_gate",
        /// FFN up projection.
        QmvUp = "qmv_up",
        /// SwiGLU.
        SiluMul = "silu_mul",
        /// FFN down projection.
        QmvDown = "qmv_down",
        /// The layer's closing residual add.
        LayerOut = "layer_out",
        /// The final RMS norm.
        FinalRms = "final_rms",
        /// The tied LM head matvec.
        QmvLmHead = "qmv_lm_head",
        /// The optional device argmax (I3 substrate).
        Argmax = "argmax",
        /// The paged KV scatter (M>1).
        KvAppendPaged = "kv_append_paged",
        /// The paged-attention read (M>1).
        SdpaPaged = "sdpa_paged",
        /// The slot-indexed GDN core (S>1).
        GdnCoreSlotted = "gdn_core_slotted",
        /// The slot-indexed GDN prologue (S>1).
        GdnPrepSlotted = "gdn_prep_slotted",
        /// gemma4 `post_attention_layernorm`.
        G4AttnPostNorm = "g4_attn_post_norm",
        /// gemma4 `pre_feedforward_layernorm`.
        G4FfnPreNorm = "g4_ffn_pre_norm",
        /// gemma4 `post_feedforward_layernorm`.
        G4FfnPostNorm = "g4_ffn_post_norm",
        /// gemma4's weightless RMS on v before the KV write.
        G4VNorm = "g4_v_norm",
        /// gemma4 `gelu_tanh(gate) * up`.
        G4Geglu = "g4_geglu",
        /// gemma4's learned per-layer gain.
        G4LayerScalar = "g4_layer_scalar",
        /// gemma4 `cap * tanh(logits / cap)`.
        G4Softcap = "g4_softcap",
        /// gemma4's sampled-row compaction before the tail.
        G4RowGather = "g4_row_gather",
        /// gemma4 sliding-window decode attention.
        G4SdpaSliding = "g4_sdpa_sliding",
        /// gemma4 `embed_tokens_per_layer` gather.
        G4PleTokenGather = "g4_ple_token_gather",
        /// gemma4 `per_layer_model_projection` matvec.
        G4PleProjGemv = "g4_ple_proj_gemv",
        /// gemma4 `per_layer_projection_norm`.
        G4PleProjNorm = "g4_ple_proj_norm",
        /// gemma4 `(proj + token) * 1/sqrt(2)`.
        G4PleCombine = "g4_ple_combine",
        /// gemma4 `per_layer_input_gate` matvec.
        G4PleGateGemv = "g4_ple_gate_gemv",
        /// gemma4 `gelu_tanh(gate) * ple`.
        G4PleGeglu = "g4_ple_geglu",
        /// gemma4 `per_layer_projection` matvec.
        G4PleProjLayerGemv = "g4_ple_proj_layer_gemv",
        /// gemma4 `post_per_layer_input_norm`.
        G4PleNorm = "g4_ple_norm",
        /// gemma4 `hidden += ple`. The variant the wrong count once ended
        /// at, forty-four kinds early.
        G4PleResidual = "g4_ple_residual",
        /// gemma4's fused post-attention norm + residual.
        G4AttnPostResidual = "g4_attn_post_residual",
        /// gemma4's fused post-FFN norm + residual.
        G4FfnPostResidual = "g4_ffn_post_residual",
        /// gemma4's fused PLE norm + residual, scaled.
        G4PleResidualScaled = "g4_ple_residual_scaled",
        /// An untied quantized embedding (`model.embed_tokens`).
        EmbedUntied = "embed_untied",
        /// An untied quantized LM head. The kind whose dispatch once ran
        /// the wrong pipeline off the short table.
        LmHeadUntied = "lm_head_untied",
        /// gpt-oss biased q projection.
        GoQmvQ = "go_qmv_q",
        /// gpt-oss biased k projection.
        GoQmvK = "go_qmv_k",
        /// gpt-oss biased v projection.
        GoQmvV = "go_qmv_v",
        /// gpt-oss biased o projection.
        GoQmvO = "go_qmv_o",
        /// gpt-oss decode attention with the learned per-head sink.
        GoSdpaSink = "go_sdpa_sink",
        /// gpt-oss router (8-bit affine, biased).
        GoRouter = "go_router",
        /// gpt-oss routed expert gate projection.
        GoExpertGate = "go_expert_gate",
        /// gpt-oss routed expert up projection.
        GoExpertUp = "go_expert_up",
        /// gpt-oss routed expert down projection.
        GoExpertDown = "go_expert_down",
        /// Top-k + softmax over the router's logits.
        GoRouterTopK = "go_router_top_k",
        /// gpt-oss's clamped SwiGLU variant.
        GoSwiGlu = "go_swi_glu",
        /// The weighted sum of the k experts' outputs.
        ExpertCombine = "expert_combine",
        /// Qwen-MoE router (`mlp.gate`, no bias).
        Router = "router",
        /// Qwen-MoE stacked expert gate projections.
        ExpertGate = "expert_gate",
        /// Qwen-MoE stacked expert up projections.
        ExpertUp = "expert_up",
        /// Qwen-MoE stacked expert down projections.
        ExpertDown = "expert_down",
        /// The batched mixture's expert-major sort.
        MoeSort = "moe_sort",
        /// The sorted-row gather.
        MoeGather = "moe_gather",
        /// The sorted-results combine, through the sort's inverse.
        LlMoeCombine = "ll_moe_combine",
        /// Shared expert gate projection (`mlp.shared_expert.gate_proj`).
        SharedGate = "shared_gate",
        /// Shared expert up projection.
        SharedUp = "shared_up",
        /// Shared expert down projection.
        SharedDown = "shared_down",
        /// `mlp.shared_expert_gate` — hidden to one logit a token.
        SharedGateProj = "shared_gate_proj",
        /// `routed + sigmoid(gate) * shared`.
        LlSharedCombine = "ll_shared_combine",
        /// The mixture's SwiGLU over the sorted stack — split from
        /// [`SiluMul`](Kernel::SiluMul) because a routed layer runs both at
        /// different extents.
        LlExpertSiluMul = "ll_expert_silu_mul",
        /// gemma4 MoE router (`router.proj` + per-expert scale).
        G4Router = "g4_router",
        /// gemma4 router norm (`router.scale`, folded at load).
        G4RouterNorm = "g4_router_norm",
        /// gemma4 top-k + softmax + gain.
        G4RouterTopK = "g4_router_top_k",
        /// gemma4 `pre_feedforward_layernorm_2`.
        G4MoeNorm = "g4_moe_norm",
        /// gemma4 `post_feedforward_layernorm_1`.
        G4DenseBranchNorm = "g4_dense_branch_norm",
        /// gemma4 `post_feedforward_layernorm_2`.
        G4MoeBranchNorm = "g4_moe_branch_norm",
        /// GeGLU over the sorted stack — gemma's activation.
        G4ExpertGeglu = "g4_expert_geglu",
        /// The dense and mixture branches meeting.
        G4BranchAdd = "g4_branch_add",
        /// gpt-oss paged decode attention with the per-head sink (M>1).
        GoSdpaSinkPaged = "go_sdpa_sink_paged",
        /// gemma4's weightless V norm READING THE K PROJECTION — the
        /// k-eq-v layers project no V of their own. A kind rather than a
        /// flag on `G4VNorm`, because it must run BEFORE `KNorm`: the
        /// two diverge at the shared projection, and a V-norm scheduled
        /// where the dense one sits would norm an already-normed K.
        G4VNormFromK = "g4_v_norm_from_k",
    }
}

/// The bucketed command-buffer key.
///
/// Grid dims change with the batch shape, so "byte-identical command
/// buffer" relaxes to "byte-identical within a shape bucket": encoded
/// buffers are cached by this key and reused on a hit. M=1 single-stream is
/// one stable bucket — `{1, 1, bucket 0, pure}` every token — which is what
/// keeps encode(N+1) overlapping GPU(N).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForwardGraphKey {
    /// Requests in the fire.
    pub requests: u32,
    /// Total tokens across the batch.
    pub tokens: u32,
    /// `max_pages_in_batch`, coarsened by [`PAGE_BUCKET_GRAN`] so the cache
    /// does not thrash on every +1 page.
    pub page_bucket: u32,
    /// Every request contributes exactly one token.
    pub is_pure_decode: bool,
}

/// The page-count bucketing granularity for [`ForwardGraphKey`].
pub const PAGE_BUCKET_GRAN: u32 = 8;

impl ForwardGraphKey {
    /// The key for a fire of `requests`/`tokens` whose largest request
    /// holds `max_pages` pages.
    #[must_use]
    pub const fn of(requests: u32, tokens: u32, max_pages: u32, is_pure_decode: bool) -> Self {
        Self {
            requests,
            tokens,
            page_bucket: max_pages.div_ceil(PAGE_BUCKET_GRAN),
            is_pure_decode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count is derived from the variant list itself, so this holds by
    /// construction — the assertion documents the invariant the C++ lost.
    #[test]
    fn the_count_is_the_end_of_the_enum_by_construction() {
        assert_eq!(Kernel::COUNT, Kernel::ALL.len());
        assert_eq!(
            Kernel::ALL.last().map(|k| k.index()),
            Some(Kernel::COUNT - 1),
            "the last kind's discriminant is one below the count"
        );
        // Discriminants are dense and ordered: index i holds the kind whose
        // discriminant is i, which is what makes `[T; COUNT]` tables safe.
        for (position, kind) in Kernel::ALL.iter().enumerate() {
            assert_eq!(kind.index(), position);
        }
    }

    /// The numeric values are ABI, "APPEND ONLY" five times over in the
    /// C++. These anchors pin every block boundary: an insertion upstream
    /// of one moves it and fails here, instead of renumbering forty kinds
    /// silently.
    #[test]
    fn the_abi_anchor_values_hold() {
        assert_eq!(Kernel::EmbedGather.index(), 0);
        assert_eq!(Kernel::Residual.index(), 10);
        assert_eq!(Kernel::QmvO.index(), 22);
        assert_eq!(Kernel::Argmax.index(), 31);
        assert_eq!(Kernel::KvAppendPaged.index(), 32);
        assert_eq!(Kernel::G4AttnPostNorm.index(), 36);
        assert_eq!(
            Kernel::G4PleResidual.index(),
            53,
            "the variant the wrong count once ended at"
        );
        assert_eq!(Kernel::LmHeadUntied.index(), 58);
        assert_eq!(Kernel::Router.index(), 71);
        assert_eq!(Kernel::G4Router.index(), 84);
        assert_eq!(Kernel::G4BranchAdd.index(), 91);
        assert_eq!(
            Kernel::GoSdpaSinkPaged.index(),
            92,
            "appended, nothing renumbered"
        );
        assert_eq!(Kernel::G4VNormFromK.index(), 93);
        assert_eq!(Kernel::COUNT, 94);
        // The bug the count fix answers: the short spelling reached 54 of
        // 98, and psos[LmHeadUntied] at 58 indexed past it.
        assert!(Kernel::LmHeadUntied.index() > Kernel::G4PleResidual.index() + 1);

        assert_eq!(IoSlot::TokenId as usize, 0);
        assert_eq!(IoSlot::SampleRows as usize, 18);
        assert_eq!(IO_SLOT_COUNT, 19);
        assert_eq!(Region::KvPagePool as usize, 6);
    }

    /// Every kind has a name and every name is one kind: the C++'s
    /// hand-kept switch left 50 of 99 kinds answering "unknown", which
    /// blinded the attribution report and the ablation knob to half the
    /// enum at once.
    #[test]
    fn every_kind_has_a_unique_name_and_round_trips() {
        let mut seen = std::collections::HashSet::new();
        for kind in Kernel::ALL {
            let name = kind.name();
            assert!(!name.is_empty());
            assert_ne!(name, "unknown", "the fall-through answer is not a name");
            assert!(seen.insert(name), "duplicate kind name {name}");
            assert_eq!(Kernel::from_name(name), Some(kind));
        }
        // The one legacy exception, pinned: AttnGate's golden tag is `gate`.
        assert_eq!(Kernel::AttnGate.name(), "gate");
        assert_eq!(Kernel::from_name("no_such_kind"), None);
    }

    #[test]
    fn a_kind_indexed_table_covers_every_kind_by_construction() {
        let mut table = [0u32; Kernel::COUNT];
        for kind in Kernel::ALL {
            table[kind.index()] += 1;
        }
        assert!(table.iter().all(|&hits| hits == 1));
    }

    #[test]
    fn the_graph_key_buckets_pages_and_keeps_m1_stable() {
        // M=1 decode is one stable bucket however long the sequence grows
        // within a granule.
        let a = ForwardGraphKey::of(1, 1, 3, true);
        let b = ForwardGraphKey::of(1, 1, 8, true);
        assert_eq!(a, b, "3 and 8 pages share ceil(n/8) = 1");
        let c = ForwardGraphKey::of(1, 1, 9, true);
        assert_ne!(a, c, "9 pages crosses into bucket 2");
        assert_eq!(ForwardGraphKey::of(1, 1, 0, true).page_bucket, 0);
        // Shape changes re-key.
        assert_ne!(a, ForwardGraphKey::of(2, 2, 3, true));
        assert_ne!(a, ForwardGraphKey::of(1, 1, 3, false));
    }
}
