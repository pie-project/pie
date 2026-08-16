//! The Nemotron-H hybrid's projections: what its numbers imply about a
//! checkpoint, a deployment and a trace.
//!
//! Three functions, and two of them used to be paragraphs of
//! `deployment_cuda`. `nemotron_h_facts_from_hf` read a parsed
//! `config.json` for eighteen numbers; the vtable's `recurrent()`
//! returned a bare `true` and left the SLAB GEOMETRY — the conv stride,
//! the state stride, which layers even have one — to be re-derived at
//! fire time from a resident config. So "how big is a Mamba state" had
//! two answers in two crates, and the one that mattered was the second.
//!
//! Here it is one, on the row's own `Deployment`.
//!
//! # A defect this move surfaced
//!
//! The deleted derivation read the MLP width from `moe_intermediate_size`
//! and nothing else. No published Nemotron-H states that key — all three
//! are DENSE and state `intermediate_size` — so every real checkpoint
//! got a dense MLP width of ZERO, and the routed block after every mixer
//! layer traced a router over no experts. It never showed because the
//! only stack this family was ever fired on is the six-layer synthetic
//! fixture, which is a mixture. The rows below state the dense width
//! where the traced text reads it, and `forward` stops before the routed
//! block when there are no experts.

use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape, Towers,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::NemotronHFacts;

/// This row's tensors.
///
/// # The per-layer rows are a UNION, and they have to be
///
/// [`crate::manifest::Observed::logical`] collapses `layers.<n>.` onto
/// `layer.{}.`, so every layer of a stack lands on one key and a
/// manifest cannot say "layer 7 attends and layer 6 scans". What a
/// hybrid publishes under the collapsed name is the UNION over its layer
/// kinds, and for THIS generation that union is three-way: the Mamba
/// mixer's eight tensors, the attention block's four, and the MLP's two.
/// A checkpoint that ships all three is a Nemotron-H; one that ships
/// only the attention set is a llama-like model and matches a llama-like
/// row.
///
/// The SCHEDULE is not lost — it is [`NemotronHFacts::layer_types`], and
/// [`deployment`] and the trace are where it is read.
///
/// # Everything hangs off `backbone.`
///
/// Not `model.layers.`: NVIDIA's converter names the decoder
/// `backbone` and its final norm `norm_f`, and the prefix rule in
/// `logical` strips `language_model.` but not `backbone.` — correctly,
/// because `backbone` is a NAME this checkpoint gave a module and not a
/// nesting convention. A row that stated `layer.{}.mixer.in_proj` would
/// match nothing at all.
#[must_use]
pub fn manifest(f: &NemotronHFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let m = &f.mamba;
    let a = &f.attn;
    let (intermediate, conv_dim) = (u64::from(m.intermediate()), u64::from(m.conv_dim()));
    let heads = u64::from(m.num_heads);
    let mlp = u64::from(f.moe.moe_intermediate);

    Manifest::new(f.layers())
        .with(TensorSpec::required("backbone.embeddings", [vocab, hidden]))
        .with(TensorSpec::required("backbone.norm_f", [hidden]))
        // `tie_word_embeddings` is FALSE on every published Nemotron-H,
        // which makes this row a required tensor rather than a
        // forbidden one — the minority answer in this crate, and the one
        // that leaves a 131 072-row projection unbound if a row gets it
        // backwards.
        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])
        // One norm per layer, on all three kinds. The only per-layer
        // tensor that is not a union member.
        .with(TensorSpec::required("backbone.layer.{}.norm", [hidden]))
        // ── The Mamba mixer ──────────────────────────────────────────
        // `[z | x | B | C | dt]` in one bank, which is why the width is
        // an arithmetic statement and not a config field: getting it
        // wrong splits a band mid-head, and `contract::layer_mamba_tp`
        // is where that lands.
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.in_proj",
            [u64::from(m.in_proj_width()), hidden],
        ))
        // `[conv_dim, 1, kernel]` as HF stores it; `extents_agree`
        // squeezes the degenerate axis, so a converter that wrote
        // `[conv_dim, kernel]` still matches.
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.conv1d",
            [conv_dim, 1, u64::from(m.conv_kernel)],
        ))
        // A conv BIAS, which the GDN hybrids also ship and which
        // `use_conv_bias: true` is the config's way of saying. A row
        // that omitted it would still match — the spec would just never
        // ask — so it is stated to keep the comparison total.
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.conv1d.bias",
            [conv_dim],
        ))
        // One decay, one skip and one step bias PER HEAD. These three
        // extents are how a reader tells `num_heads` from
        // `intermediate`, which the packed banks above cannot say.
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.A_log",
            [heads],
        ))
        .with(TensorSpec::required("backbone.layer.{}.mixer.D", [heads]))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.dt_bias",
            [heads],
        ))
        // The GATED norm folds over the scan's whole width, not per
        // head — `zamba_rmsnorm_gated` reads `z` from the same split.
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.norm",
            [intermediate],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.out_proj",
            [hidden, intermediate],
        ))
        // ── The attention block, under `mixer.` like everything else ─
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.q_proj",
            [u64::from(a.q_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.k_proj",
            [u64::from(a.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.v_proj",
            [u64::from(a.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.o_proj",
            [hidden, u64::from(a.q_width())],
        ))
        // No q/k norms anywhere in this generation, and no rotary
        // tables. Both absences discriminate: a qwen-3 stack of the same
        // widths ships `q_norm`, and a row that did not forbid it would
        // claim one.
        .with(TensorSpec::absent("backbone.layer.{}.mixer.q_norm"))
        .with(TensorSpec::absent("backbone.layer.{}.mixer.k_norm"))
        // ── The MLP, dense or routed ─────────────────────────────────
        // ReLU², so ONE projection up and one down; there is no gate
        // half to pair with, and forbidding `gate_proj` is what says so.
        .with(TensorSpec::absent("backbone.layer.{}.mixer.gate_proj"))
        .either(
            !f.is_mixture(),
            "backbone.layer.{}.mixer.up_proj",
            [mlp, hidden],
        )
        .either(
            !f.is_mixture(),
            "backbone.layer.{}.mixer.down_proj",
            [hidden, mlp],
        )
        // The expert bank's extents are a PACKING decision — one tensor
        // per expert, or the packed slab `contract::packed_expert_views`
        // publishes — so the spec asks that it exist and says nothing
        // about its shape. A dense row FORBIDS it, which is how a dense
        // stack and a routed one of identical widths stay distinct.
        .with(if f.is_mixture() {
            TensorSpec::present("backbone.layer.{}.mixer.experts.0.up_proj")
        } else {
            TensorSpec::absent("backbone.layer.{}.mixer.experts.0.up_proj")
        })
        .with(if f.is_mixture() {
            TensorSpec::present("backbone.layer.{}.mixer.experts.0.down_proj")
        } else {
            TensorSpec::absent("backbone.layer.{}.mixer.experts.0.down_proj")
        })
}

/// This row's deployment.
///
/// `rope_theta` and `norm_eps` are the row's rather than the shape's:
/// neither is a tensor extent, so neither can be measured against a
/// checkpoint, and [`super::spec`] holds only what can.
///
/// # Every layer gets an entry, including the ones that do not attend
///
/// [`Deployment::attention`] is indexed by layer, so a stack of 98 must
/// hand back 98 rows or a driver reading layer 97's window reads past
/// the end. The Mamba and MLP layers' entries are the attention layers'
/// answers repeated — they are never read, and leaving them out would
/// make the index mean something different for this generation than for
/// every other.
#[must_use]
pub fn deployment(
    f: &NemotronHFacts,
    rope_theta: f32,
    norm_eps: f32,
    head_dim_kernel: u32,
) -> Deployment {
    let a = &f.attn;
    let head_dim = head_dim_kernel.max(a.head_dim);
    let attention = (0..f.layers())
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: a.kv_heads,
            head_dim,
            // `sliding_window` is null in all three published configs:
            // the four attention layers of a 52-layer stack see the
            // whole context, and the recurrent layers between them carry
            // the rest of it.
            window: -1,
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,
            // Full rotation at the head width. Stated as 0 the way every
            // non-partial row states it — see `LayerAttention`.
            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers(),
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            head_dim_kernel,
            // A dense stack's MLP width is the block's; a mixture's
            // dense width is ZERO, because in a routed Nemotron-H every
            // MLP layer is the router's. `widest_mlp()` is the max of
            // the two and is what sizes the forward workspace, so a
            // planner told only one of them under-sizes the other.
            intermediate: if f.is_mixture() {
                0
            } else {
                f.moe.moe_intermediate
            },
            moe_intermediate: if f.is_mixture() {
                f.moe.moe_intermediate
            } else {
                0
            },
            experts_per_token: if f.is_mixture() { f.moe.top_k } else { 0 },
            shared_intermediate: if f.is_mixture() {
                f.moe.shared_intermediate
            } else {
                0
            },
            vocab: f.vocab,
        },
        attention,
        // The four attention layers page ordinarily; the Mamba layers
        // hold no pages at all, which `recurrent` is what states.
        kv: KvStyle::Paged,
        recurrent: Some(mamba_shape(f)),
        prefill: PrefillStyle::Planned,
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: 0.0,
        // No ATTENTION cap: gemma-2's `attn_logit_softcapping` is
        // gemma-2's alone, and a zero here is "no cap" rather than a
        // cap at zero — which would flatten every score to `tanh(inf)`.
        attn_logit_softcap: 0.0,
        ple_dim: 0,
        norm: NormPlacement::Pre,
        // Not a gemma: the gain is the multiplier, stored directly.
        norm_unit_offset: false,
        v_norm: false,
        // The only family whose CUDA text names `topk_sigmoid_bias`,
        // which is the kernel that reads this off the launch context.
        norm_topk_prob: f.moe.norm_topk_prob,
        routed_scaling: f.moe.routed_scaling,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW: a family label and a published ceiling are
        // facts about a checkpoint, and a projection sees geometry.
        advertised: Advertised::default(),
        // Text only. Nemotron-H ships no encoder beside the decoder.
        rope_scaling: None,
        towers: Towers::default(),
    }
}

/// The recurrent slab geometry the Mamba layers need allocated.
///
/// `PlannedFamily::recurrent()` returned `true` here and nothing else,
/// which left the driver to re-derive these seven numbers from a
/// resident config at fire time. The GDN hybrid's `gdn_shape()` is the
/// same seven, so the field NAMES are the GDN ones; what they hold is
/// the selective scan's:
///
/// * `v_h` / `v_d` are the scan's heads and per-head width — the state
///   is `[heads, head_dim, state]` and those are its first two axes;
/// * `k_d` is the STATE SIZE, the third;
/// * `k_h` is `n_groups`, because B and C are shared across groups the
///   way keys are shared across a GQA group.
///
/// `state_stride` then falls out as `v_h * k_d * v_d`, which is the
/// identical arithmetic the GDN path uses — so one allocator sizes both
/// hybrids and neither has a formula of its own to get wrong.
#[must_use]
fn mamba_shape(f: &NemotronHFacts) -> RecurrentShape {
    let m = &f.mamba;
    RecurrentShape {
        linear_layers: f.mamba_layers(),
        conv_stride: (m.conv_kernel * m.conv_dim()) as usize,
        state_stride: (m.num_heads * m.head_dim * m.state_size) as usize,
        // The store is bf16, matching `RecurrentStateCache`'s only
        // allocator for it.
        state_elem: 2,
        // NOT the group count. `k_h` is a GATED-DELTA field and every
        // kernel reading it is a gdn kernel a mamba row never
        // dispatches — which is why `n_groups` could ride here unnoticed
        // while `GdnCtx::n_groups` stayed the literal zero the launch
        // wrote beside it.
        k_h: 0,
        v_h: m.num_heads as i32,
        k_d: m.state_size as i32,
        v_d: m.head_dim as i32,
        conv_dim: m.conv_dim() as i32,
        conv_k: m.conv_kernel as i32,
        n_groups: m.n_groups as i32,
    }
}

/// Why this build has no Metal text for a nemotron-h row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `nemotron_h_cuda`, a HYBRID stack: Mamba-2
/// state-space layers interleaved with attention. `llama_like_metal`
/// states attention layers only, so tracing it would drop every
/// recurrent layer in the model.
///
/// A `Refusal::Unsupported` and not a `Malformed`: the checkpoint is
/// fine, and a pie whose Metal half had this text would serve the same
/// row unchanged. What is missing is a TEXT in this build, which is a
/// fact about the build.
///
/// Stating it is the whole of what replaces `driver-metal`'s
/// `LLAMA_LIKE` — an eleven-entry table of architecture STRINGS,
/// reduced by a punctuation-stripping `canonical()`, consulted before
/// any text was traced and free to disagree with what the tracer would
/// actually do. It listed `gpt_oss`, which no publication of reaches a
/// Metal device here, and omitted `gemma3`, whose text it models. A row
/// that answers for itself cannot disagree with a list, because there
/// is no list.
pub const NO_METAL: &str = "nemotron-h has no Metal text in this build: its forward is `nemotron_h_cuda`, \
     a hybrid of Mamba-2 state-space layers and attention layers, and the one \
     Metal text here (`llama_like_metal`) states attention only — it has no \
     recurrent layer kind and takes a different shape; the CUDA backend serves \
     this row";

/// Trace this row's CUDA text for one fire class.
#[must_use]
pub fn trace(
    f: &NemotronHFacts,
    class: model_ir::trace::FireClass,
) -> model_ir::trace::ForwardPlan {
    super::forward::nemotron_h_cuda(f, class)
}

#[cfg(test)]
mod tests {
    use super::super::spec::{NemotronLayerKind, NemotronMoeFacts};
    use super::{
        Deployment, KvStyle, NemotronHFacts, NormPlacement, PrefillStyle, deployment, manifest,
    };
    use crate::deployment::{Advertised, AttnOutput, Towers};
    use crate::manifest::{Observed, Presence};

    const ROPE: f32 = 10_000.0;
    const EPS: f32 = 1e-5;

    fn eight_b() -> Deployment {
        deployment(&NemotronHFacts::nemotron_h_8b(), ROPE, EPS, 128)
    }

    /// Every spec a row states is satisfied by the checkpoint the row
    /// describes. A manifest that cannot match its own arithmetic is
    /// describing a stack that does not exist.
    #[test]
    fn a_row_satisfies_the_manifest_it_states() {
        for f in &[
            NemotronHFacts::nemotron_h_4b(),
            NemotronHFacts::nemotron_h_8b(),
            NemotronHFacts::nemotron_h_47b(),
            NemotronHFacts::nemotron_h_synthetic(),
        ] {
            let m = manifest(f);
            let implied = Observed::from_pairs(
                m.tensors
                    .iter()
                    .filter(|t| t.presence != Presence::Absent)
                    .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
            );
            assert!(
                m.check(&implied).is_ok(),
                "{}",
                m.check(&implied).unwrap_err()
            );
        }
    }

    /// The names are the ones NVIDIA's converter writes.
    ///
    /// Transcribed from `model.safetensors.index.json` of
    /// `nvidia/Nemotron-H-8B-Base-8K`, which publishes 311 tensors under
    /// `backbone.` — not `model.layers.`. A row that used the llama
    /// spelling would match nothing at all, and "matches nothing" is
    /// indistinguishable from "this checkpoint is some other model"
    /// unless a test states the names.
    #[test]
    fn the_manifest_names_the_tensors_nvidia_publishes() {
        let m = manifest(&NemotronHFacts::nemotron_h_8b());
        let named = |n: &str| m.tensors.iter().find(|t| t.name == n);
        for name in [
            "backbone.embeddings",
            "backbone.norm_f",
            "lm_head",
            "backbone.layer.{}.norm",
            "backbone.layer.{}.mixer.in_proj",
            "backbone.layer.{}.mixer.conv1d",
            "backbone.layer.{}.mixer.conv1d.bias",
            "backbone.layer.{}.mixer.A_log",
            "backbone.layer.{}.mixer.D",
            "backbone.layer.{}.mixer.dt_bias",
            "backbone.layer.{}.mixer.norm",
            "backbone.layer.{}.mixer.out_proj",
            "backbone.layer.{}.mixer.q_proj",
            "backbone.layer.{}.mixer.k_proj",
            "backbone.layer.{}.mixer.v_proj",
            "backbone.layer.{}.mixer.o_proj",
            "backbone.layer.{}.mixer.up_proj",
            "backbone.layer.{}.mixer.down_proj",
        ] {
            let spec = named(name).unwrap_or_else(|| panic!("no row for '{name}'"));
            assert_eq!(
                spec.presence,
                Presence::Required,
                "'{name}' is what this generation is"
            );
        }
        assert_eq!(m.layers, 52);
    }

    /// The extents ARE the arithmetic, which is what makes a manifest a
    /// check rather than a second statement.
    #[test]
    fn the_mamba_extents_are_the_widths_the_split_reads() {
        let f = NemotronHFacts::nemotron_h_8b();
        let m = manifest(&f);
        let want = |n: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated above")
                .extents
                .clone()
        };
        // 128 heads of 64 = 8192 scanning width; B and C add
        // 2 * 8 groups * 128 state = 2048, so the conv sees 10 240; and
        // `in_proj` stacks z (8192), the conv's 10 240 and one `dt` per
        // head (128) = 18 560 rows over a 4096-wide residual.
        assert_eq!(want("backbone.layer.{}.mixer.in_proj"), vec![18_560, 4096]);
        assert_eq!(want("backbone.layer.{}.mixer.conv1d"), vec![10_240, 1, 4]);
        assert_eq!(want("backbone.layer.{}.mixer.conv1d.bias"), vec![10_240]);
        assert_eq!(want("backbone.layer.{}.mixer.norm"), vec![8192]);
        assert_eq!(want("backbone.layer.{}.mixer.out_proj"), vec![4096, 8192]);
        for per_head in ["A_log", "D", "dt_bias"] {
            assert_eq!(
                want(&format!("backbone.layer.{{}}.mixer.{per_head}")),
                vec![128],
                "one {per_head} per scan head is how a reader tells 128 heads from 8192 channels"
            );
        }
        // 32 query heads of 128 over 8 kv heads of 128.
        assert_eq!(want("backbone.layer.{}.mixer.q_proj"), vec![4096, 4096]);
        assert_eq!(want("backbone.layer.{}.mixer.k_proj"), vec![1024, 4096]);
        assert_eq!(want("backbone.layer.{}.mixer.o_proj"), vec![4096, 4096]);
        assert_eq!(want("backbone.layer.{}.mixer.up_proj"), vec![21_504, 4096]);
        assert_eq!(
            want("backbone.layer.{}.mixer.down_proj"),
            vec![4096, 21_504]
        );
    }

    /// A dense stack forbids the expert bank and a routed one requires
    /// it, so no checkpoint matches both.
    #[test]
    fn dense_and_routed_rows_cannot_claim_one_checkpoint() {
        let presence = |f: &NemotronHFacts, name: &str| {
            manifest(f)
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect("stated")
                .presence
        };
        let dense = NemotronHFacts::nemotron_h_8b();
        // The 8B's own widths with a router bolted on, so the ONLY thing
        // the two manifests disagree about is the MLP. Compared against
        // the synthetic mixture instead, every extent would differ too
        // and the four faults worth reading would be buried.
        let routed = NemotronHFacts {
            moe: NemotronMoeFacts {
                num_experts: 8,
                top_k: 2,
                norm_topk_prob: true,
                routed_scaling: 1.0,
                moe_intermediate: 21_504,
                shared_intermediate: 0,
            },
            ..NemotronHFacts::nemotron_h_8b()
        };
        let bank = "backbone.layer.{}.mixer.experts.0.up_proj";
        assert_eq!(presence(&dense, bank), Presence::Absent);
        assert_eq!(presence(&routed, bank), Presence::Required);
        assert_eq!(
            presence(&dense, "backbone.layer.{}.mixer.up_proj"),
            Presence::Required
        );
        assert_eq!(
            presence(&routed, "backbone.layer.{}.mixer.up_proj"),
            Presence::Absent
        );

        // The checkpoint the dense row describes, offered to the routed
        // row: it must be refused, and the refusal must name the bank.
        let m = manifest(&dense);
        let seen = Observed::from_pairs(
            m.tensors
                .iter()
                .filter(|t| t.presence != Presence::Absent)
                .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
        );
        let err = manifest(&routed)
            .check(&seen)
            .expect_err("a dense stack is not a mixture");
        assert!(
            err.to_string().contains("experts.0"),
            "the refusal must say what is missing: {err}"
        );
        assert_eq!(
            err.faults.len(),
            4,
            "two banks missing and two dense projections present"
        );
    }

    /// Neither q/k norms nor a gate half, and the row says so.
    ///
    /// Three absences, and each one is a different family this stack
    /// would otherwise be confusable with: qwen-3 norms its heads, and
    /// every swiglu family in the crate ships a `gate_proj` beside its
    /// `up_proj`. ReLU² needs neither.
    #[test]
    fn the_absences_are_what_separate_this_from_a_llama_like_stack() {
        let m = manifest(&NemotronHFacts::nemotron_h_8b());
        for name in [
            "backbone.layer.{}.mixer.q_norm",
            "backbone.layer.{}.mixer.k_norm",
            "backbone.layer.{}.mixer.gate_proj",
        ] {
            let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
            assert_eq!(spec.presence, Presence::Absent, "'{name}'");
        }
    }

    /// Untied, which is the minority answer and the one a default gets
    /// wrong.
    #[test]
    fn an_untied_row_requires_the_output_projection() {
        let m = manifest(&NemotronHFacts::nemotron_h_8b());
        let head = m
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Required);
        assert_eq!(head.extents, vec![131_072, 4096]);

        let tied = NemotronHFacts {
            tied_embeddings: true,
            ..NemotronHFacts::nemotron_h_8b()
        };
        let head = manifest(&tied)
            .tensors
            .into_iter()
            .find(|t| t.name == "lm_head")
            .expect("row");
        assert_eq!(
            head.presence,
            Presence::Absent,
            "a tied stack ships one table, not two"
        );
    }

    /// The geometry is the row's own numbers.
    #[test]
    fn the_geometry_is_the_rows_own_numbers() {
        let d = eight_b();
        assert_eq!(d.layers, 52);
        assert_eq!(
            d.attention.len(),
            52,
            "one entry per layer, whether or not it attends"
        );
        assert_eq!(d.shape.hidden, 4096);
        assert_eq!(d.shape.q_heads, 32);
        assert_eq!(d.shape.kv_heads, 8);
        assert_eq!(d.shape.gqa_group(), 4);
        assert_eq!(d.shape.head_dim, 128);
        assert_eq!(
            d.shape.head_dim_kernel, 128,
            "128 is instantiated; nothing is padded"
        );
        assert_eq!(d.shape.intermediate, 21_504);
        assert_eq!(
            d.shape.moe_intermediate, 0,
            "a dense stack has no expert width"
        );
        assert_eq!(d.shape.widest_mlp(), 21_504);
        assert_eq!(d.shape.vocab, 131_072);
        assert_eq!(d.norm_eps, EPS);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.kv, KvStyle::Paged);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(
            d.advertised,
            Advertised::default(),
            "the row fills this, not the projection"
        );
        assert_eq!(d.towers, Towers::default(), "Nemotron-H is text only");
    }

    /// A mixture states its expert width where a planner reads it.
    #[test]
    fn a_routed_stack_states_the_expert_width_and_not_a_dense_one() {
        let f = NemotronHFacts::nemotron_h_synthetic();
        let d = deployment(&f, ROPE, EPS, 128);
        assert_eq!(
            d.shape.intermediate, 0,
            "no layer of a routed stack runs a dense block"
        );
        assert_eq!(d.shape.moe_intermediate, 1024);
        assert_eq!(d.shape.widest_mlp(), 1024);
    }

    /// Every attention entry states the same six answers, because every
    /// attention layer of this stack IS the same.
    #[test]
    fn every_layer_attends_the_whole_context_at_one_base() {
        let d = eight_b();
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(
                a.window, -1,
                "layer {l} states a window and no config here does"
            );
            assert_eq!(
                a.kv_source, l as u32,
                "no layer of this generation shares KV"
            );
            assert_eq!(a.head_dim, 128);
            assert_eq!(a.rope_theta, ROPE);
            assert_eq!(a.rotary_dim, 0, "full rotation at the head width");
        }
        let expected = 1.0 / 128f32.sqrt();
        assert!(
            d.attention
                .iter()
                .all(|a| (a.sm_scale - expected).abs() < 1e-9),
            "this generation's q and k carry no norm, so the softmax scale is the ordinary one"
        );
        assert!(
            !d.shares_kv(),
            "a stack whose layers each own their pages does not share"
        );
    }

    /// The slab list names the SSM layers and nothing else.
    ///
    /// The number that moves if this is wrong is not a shape: it is how
    /// many slabs get allocated. Provisioning one per layer would ask
    /// for 52 where 24 are used, and the excess comes out of the KV
    /// pool — quietly, as a shorter servable context.
    #[test]
    fn the_recurrent_slabs_are_the_mamba_layers_and_only_those() {
        let f = NemotronHFacts::nemotron_h_8b();
        let d = eight_b();
        let r = d
            .recurrent
            .as_ref()
            .expect("a hybrid carries recurrent state");
        assert_eq!(r.linear_layers, f.mamba_layers());
        assert_eq!(r.linear_layers.len(), 24);
        assert!(
            r.linear_layers
                .iter()
                .all(|&l| f.kind(l) == NemotronLayerKind::Mamba),
            "a slab was provisioned for a layer that has no scan"
        );
        for l in [7, 18, 29, 40] {
            assert!(
                !r.linear_layers.contains(&l),
                "layer {l} attends; it pages instead"
            );
        }
    }

    /// The two strides, from the two widths they are made of.
    #[test]
    fn the_strides_are_the_scans_working_set() {
        let r = eight_b()
            .recurrent
            .expect("a hybrid carries recurrent state");
        // The conv window spans the whole packed in-projection minus z
        // and dt: 10 240 channels, four taps deep.
        assert_eq!(r.conv_dim, 10_240);
        assert_eq!(r.conv_k, 4);
        assert_eq!(r.conv_stride, 4 * 10_240);
        // The state is `[128 heads, 64 wide, 128 deep]`.
        assert_eq!(r.state_stride, 128 * 64 * 128);
        assert_eq!(
            r.state_stride,
            (r.v_h * r.k_d * r.v_d) as usize,
            "the GDN arithmetic exactly"
        );
        assert_eq!(r.state_elem, 2, "bf16");
        assert_eq!((r.v_h, r.v_d, r.k_d), (128, 64, 128));
        // The group count under its own name, and `k_h` — a
        // gated-delta field — left empty rather than carrying it.
        assert_eq!(r.n_groups, 8);
        assert_eq!(r.k_h, 0);
    }

    /// The 47B's doubled state reaches the allocator.
    ///
    /// Four times the 8B's slab per layer, from ONE number — and a
    /// checkpoint whose row inherited its sibling's 128 would be handed
    /// half the state its scan writes into.
    #[test]
    fn the_largest_row_asks_for_four_times_the_slab() {
        let big = deployment(&NemotronHFacts::nemotron_h_47b(), ROPE, EPS, 128)
            .recurrent
            .expect("a hybrid carries recurrent state");
        let small = eight_b()
            .recurrent
            .expect("a hybrid carries recurrent state");
        assert_eq!(big.state_stride, 256 * 64 * 256);
        assert_eq!(big.state_stride, small.state_stride * 4);
        assert_eq!(big.linear_layers.len(), 45);
    }

    /// A head dim the build does not instantiate is ROUNDED, and the
    /// unrounded one is still what a TP split reads.
    #[test]
    fn a_padded_head_dim_moves_the_kernel_width_and_not_the_checkpoints() {
        let f = NemotronHFacts::nemotron_h_8b();
        let d = deployment(&f, ROPE, EPS, 256);
        assert_eq!(
            d.shape.head_dim, 128,
            "the checkpoint's own width is unchanged"
        );
        assert_eq!(d.shape.head_dim_kernel, 256);
        assert_eq!(d.shape.head_dim_alloc(), 256);
        assert!(
            d.attention.iter().all(|a| a.head_dim == 256),
            "a kernel is handed the width it was instantiated at"
        );
    }
}
