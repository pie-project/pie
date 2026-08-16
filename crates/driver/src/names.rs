//! What a checkpoint calls the tensor a text calls `layer.3.down.zeros`.
//!
//! # The measurement this module exists because of
//!
//! `tests/checkpoint.rs` compiled a real load plan for
//! `mlx-community/Qwen3-0.6B-4bit` and held its published tensor names against
//! the names `lower()` binds. **704 of 704 disagreed.** Not "most", not "the
//! sidecars" -- every single one, because the two sides share no convention at
//! all:
//!
//! | the text binds          | the loader publishes                    |
//! | ----------------------- | --------------------------------------- |
//! | `layer.0.down`          | `layers.0.mlp.down_proj.weight`         |
//! | `layer.0.down.zeros`    | `layers.0.mlp.down_proj.biases`         |
//! | `layer.0.attn_norm`     | `layers.0.input_layernorm.weight`       |
//! | `final_norm`            | `final_norm.weight`                     |
//! | `embed`                 | `shared_embedding.weight`               |
//!
//! So a driver that binds by the traced name finds nothing. This is the
//! translation, and without it no real weight ever reaches a shell.
//!
//! # Why it is here and not in a shell
//!
//! It stood in `driver-vulkan/src/names.rs`, and `driver-wgpu/src/names.rs`
//! held the same 412 lines byte for byte -- the second hand-written copy of
//! one golden table, which is the failure this crate's own `lib.rs` opens by
//! naming about the C++ interpreters. Of the nineteen files in each shell
//! those two were among the only two that matched exactly, so it was
//! duplication rather than two crates being one fork.
//!
//! Sharing is right here for a reason that does NOT generalise, and the
//! contrast is worth stating because this repository refuses the same move
//! elsewhere: `model`'s per-family import tables have nine rows that look
//! identical across four families and are deliberately not shared, because
//! they are identical by coincidence of two naming schemes agreeing and
//! nothing keeps them agreeing. This table is not that. It describes exactly
//! one thing -- the names `model::boot::compile_load_plan_for` publishes --
//! and every shell that reads it is reading the same producer. One producer
//! with N consumers is a substrate; N producers that currently agree is a
//! coincidence, and only the first one belongs in a crate named for what
//! every driver shares.
//!
//! # Why a table and not a decision
//!
//! A shell may not choose a kernel. A table does not: it answers
//! `layer.3.down` with a string. Removing it does not change which kernels
//! fire, only whether they find their operands -- and that is the difference
//! between translating a spelling and making a decision.
//!
//! `driver-metal` has the same table again, in `src/lowering/resolve.rs`, and
//! it stays where it is. That one is entangled with Metal's `Slice` and its
//! `Resolver` trait, so folding it in would put `objc` in this crate's
//! closure and `driver-vulkan/tests/pure.rs` forbids that -- which is a
//! statement about that copy's ENTANGLEMENT and not about its data. What is
//! reproduced here is only the data, and only the roles the texts these
//! shells can lower actually bind -- measured, twenty-two of them, rather
//! than every role that exists.

//!
//! # A role has SEVERAL spellings and the checkpoint picks
//!
//! One role, one name forces the driver to pick a table per family, and
//! picking is the thing this crate may not do. So a role lists every spelling
//! it has been seen under, in try order, and [`Naming::spellings`] returns them
//! all; the caller keeps the one its checkpoint publishes. Adding a
//! convention is adding a string.

/// How a checkpoint spells what a text names.
///
/// Data, not code. Constructed by [`Naming::mlx`]; there is no other
/// constructor, because there is no second convention this crate has measured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Naming {
    /// What goes in front of a layer index. `layers.` for what
    /// `model::boot::compile_load_plan_for` publishes -- **not** the
    /// `model.layers.` a raw HuggingFace export writes, because the contract
    /// has already renamed by the time a driver sees a plan.
    pub layer_prefix: &'static str,
    /// A text's layer-scoped role, and the paths within a layer it could be.
    pub roles: &'static [(&'static str, &'static [&'static str])],
    /// The roles with no layer at all.
    pub globals: &'static [(&'static str, &'static [&'static str])],
    /// What the packed tensor itself hangs under.
    ///
    /// `.weight` for a tensor under a module, and the EMPTY string for one
    /// that is the value: gpt-oss ships `self_attn.sinks`, not
    /// `self_attn.sinks.weight`, because a sink is a vector per head and not a
    /// linear layer.
    pub weight_suffix: &'static [&'static str],
    /// What the zero point the text spells `.zeros` is called.
    ///
    /// `.biases`, MLX's affine quantisation, and no checkpoint here spells it
    /// otherwise.
    ///
    /// It used to also list `.bias` "for the MXFP4 expert banks -- the same
    /// role one character apart". They are one character apart and they are
    /// not the same role. `mlx-community/gpt-oss-20b-MXFP4-Q4` publishes an
    /// expert `bias` of `[32, 2880]`, one per output row, and no `biases` at
    /// all; the zero point beside `scales` would be `[32, 2880, 90]`, one per
    /// group. `qmv_routed` reads them at two different bindings, and
    /// under `PIE_MXFP4` it does not declare the zero point's at all.
    pub zero_point_suffix: &'static [&'static str],
    /// What the additive bias the text spells `.bias` is called.
    ///
    /// One per output row of a routed expert bank. `routed_qmv` is the only
    /// site that names it, and only for the symbols that read one.
    pub bias_suffix: &'static [&'static str],
}

/// The twenty-two roles the six texts in `tests/arena.rs` bind, and no others.
///
/// Measured, not guessed: a scratch pass over every `Arg::Weight` those texts
/// state produced exactly this set. A role that no text this crate can lower
/// ever names would be a string nothing checks.
const ROLES: &[(&str, &[&str])] = &[
    ("q_proj", &["self_attn.q_proj"]),
    ("k_proj", &["self_attn.k_proj"]),
    ("v_proj", &["self_attn.v_proj"]),
    ("o_proj", &["self_attn.o_proj", "linear_attn.out_proj"]),
    // The Qwen-2 family's projection biases. A bias hangs under the
    // PROJECTION rather than under a module of its own, so the path carries
    // the `.bias` and the empty `weight_suffix` closes it -- the same shape
    // `attn_sinks` has and for the same reason.
    ("q_bias", &["self_attn.q_proj.bias"]),
    ("k_bias", &["self_attn.k_proj.bias"]),
    ("v_bias", &["self_attn.v_proj.bias"]),
    // The OUTPUT projection's bias and the router's, which this table could
    // not spell until gpt-oss's text started binding them.
    //
    // The three above were added for qwen-2 and stopped there, because
    // qwen-2's `o_proj` carries none. gpt-oss's does, and so does its router,
    // and a name this table cannot spell is not a refusal anywhere: `spellings`
    // answers with nothing, the loader allocates for the weights it could name,
    // and the rest stay bound to whatever the arena held. 48 of gpt-oss-20b's
    // 775 weights, in the layer loop, silently.
    //
    // Found by `tests/checkpoint.rs` after an upstream fixture change, which is
    // the only reason it was found: `driver-vulkan` carries the identical table
    // and its own checkpoint test does not sweep this text.
    ("o_bias", &["self_attn.o_proj.bias"]),
    ("router_bias", &["mlp.gate.bias", "mlp.router.bias"]),
    // THE GATED DELTANET's layer, read off a compiled `qwen3.5-0.8b-base`
    // plan rather than guessed. Its projections and constants hang under
    // `linear_attn.`, which is a module this table had never seen: 306 of the
    // 712 names a qwen3.5 lowering binds had NO SPELLING here, which by the
    // rule stated above is not a refusal anywhere -- the loader allocates what
    // it could name and the rest stay bound to whatever the arena held. That
    // is the gpt-oss defect at six times the scale.
    //
    // `out_proj` is listed under `o_proj` beside the attention spelling: a
    // hybrid's linear layers publish `linear_attn.out_proj` and its full
    // layers `self_attn.o_proj`, one text binds `layer.N.o_proj` for both, and
    // the two paths cannot collide because no layer is both kinds.
    ("conv_w", &["linear_attn.conv1d.weight"]),
    ("conv_b", &["linear_attn.conv1d.bias"]),
    ("a_log", &["linear_attn.A_log"]),
    ("dt", &["linear_attn.dt_bias"]),
    ("gate_norm", &["linear_attn.norm"]),
    ("in_proj_qkv", &["linear_attn.in_proj_qkv"]),
    ("in_proj_a", &["linear_attn.in_proj_a"]),
    ("in_proj_b", &["linear_attn.in_proj_b"]),
    ("in_proj_z", &["linear_attn.in_proj_z"]),
    ("q_norm", &["self_attn.q_norm"]),
    ("k_norm", &["self_attn.k_norm"]),
    // One learned logit per head, and the one role whose tensor hangs under
    // no module -- see `weight_suffix`.
    ("attn_sinks", &["self_attn.sinks"]),
    ("gate_proj", &["mlp.gate_proj"]),
    ("up_proj", &["mlp.up_proj"]),
    ("down", &["mlp.down_proj"]),
    // `mlp.gate` is MLX's name for the ROUTER. It is one character from
    // `mlp.gate_proj`, which is an expert's gate half and an entirely
    // different tensor, and the collision is worth spelling out because a
    // wrong guess here binds a 151936-wide table where a router expected
    // `experts` numbers and nothing would fail loudly.
    ("router", &["mlp.gate", "mlp.router"]),
    // The expert bank carries no expert index: all of them live in one
    // `[experts, out, in]` tensor and the routed kernel indexes it by the slot
    // it read. Three conventions for that one bank -- qwen3-moe's
    // `switch_mlp`, gemma4's `switch_glu`, gpt-oss's plain `experts`.
    (
        "expert_gate",
        &[
            "mlp.switch_mlp.gate_proj",
            "experts.switch_glu.gate_proj",
            "mlp.experts.gate_proj",
        ],
    ),
    (
        "expert_up",
        &[
            "mlp.switch_mlp.up_proj",
            "experts.switch_glu.up_proj",
            "mlp.experts.up_proj",
        ],
    ),
    (
        "expert_down",
        &[
            "mlp.switch_mlp.down_proj",
            "experts.switch_glu.down_proj",
            "mlp.experts.down_proj",
        ],
    ),
    ("attn_norm", &["input_layernorm"]),
    // TWO spellings, and the checkpoint decides by which one it ships. Under
    // a pre-norm placement the pre-FFN norm IS `post_attention_layernorm` --
    // it sits after the attention and before the MLP, and llama publishes
    // nothing else. gemma splits that position in two, so it must take the
    // second.
    //
    // Ordered gemma-first on purpose: the caller keeps the first spelling the
    // checkpoint HAS, a llama checkpoint ships no `pre_feedforward_layernorm`
    // and falls through, and the other order would bind gemma's
    // attention-output norm as its MLP input norm and silently drop two.
    (
        "mlp_norm",
        &["pre_feedforward_layernorm", "post_attention_layernorm"],
    ),
];

/// The three layer-less roles the six texts bind.
const GLOBALS: &[(&str, &[&str])] = &[
    // Tied: one table serves the embedding and the readout, which is why both
    // resolve to `shared_embedding`. An untied deployment (gpt-oss) publishes
    // `embed_tokens` and `lm_head` separately, so both spellings are listed
    // and the checkpoint says which it is.
    ("embed", &["shared_embedding", "embed_tokens"]),
    ("lm_head", &["shared_embedding", "lm_head"]),
    ("final_norm", &["final_norm"]),
];

impl Naming {
    /// The convention `model::boot::compile_load_plan_for` publishes.
    ///
    /// **Read off a compiled plan, not off a HuggingFace export.** The
    /// contract renames before a driver sees anything: `model.layers.N.…`
    /// becomes `layers.N.…`, `model.norm.weight` becomes `final_norm.weight`,
    /// and a tied `model.embed_tokens` becomes `shared_embedding`. A table
    /// written against the export would be self-consistent, would pass any
    /// test that held the text against it, and would find nothing at load.
    #[must_use]
    pub const fn mlx() -> Self {
        Self {
            layer_prefix: "layers.",
            roles: ROLES,
            globals: GLOBALS,
            // `.weight` first, then the bare name for a role that IS its
            // tensor.
            weight_suffix: &[".weight", ""],
            zero_point_suffix: &[".biases"],
            bias_suffix: &[".bias"],
        }
    }

    /// Every name `traced` could be published under, in try order.
    ///
    /// The cross product of the role's candidate paths and the sidecar's
    /// candidate suffixes. **Empty** for a name outside the text's shape,
    /// which is drift rather than a spelling this table has not learned -- and
    /// the caller must treat the two differently, because an empty answer that
    /// read as "no match" would turn a typo into a silent miss.
    #[must_use]
    pub fn spellings(&self, traced: &str) -> Vec<String> {
        let Some(t) = decompose(traced) else {
            return Vec::new();
        };
        let table = if t.layer.is_some() {
            self.roles
        } else {
            self.globals
        };
        let Some((_, paths)) = table.iter().find(|(role, _)| *role == t.role) else {
            return Vec::new();
        };
        let bases: Vec<String> = match t.layer {
            Some(l) => paths
                .iter()
                .map(|p| format!("{}{l}.{p}", self.layer_prefix))
                .collect(),
            None => paths.iter().map(|p| (*p).to_string()).collect(),
        };
        let suffixes: &[&str] = match t.sidecar {
            Sidecar::Packed => self.weight_suffix,
            // No checkpoint spells the scale two ways, so there is nothing to
            // choose between.
            Sidecar::Scales => &[".scales"],
            Sidecar::Zeros => self.zero_point_suffix,
            Sidecar::Bias => self.bias_suffix,
        };
        bases
            .iter()
            .flat_map(|b| suffixes.iter().map(move |s| format!("{b}{s}")))
            .collect()
    }
}

/// Which of a quantised weight's three tensors a traced name is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sidecar {
    /// The packed weight itself.
    Packed,
    Scales,
    Zeros,
    /// The additive term a routed expert bank carries beside its codec's
    /// planes — one value per output row, and not a zero point.
    Bias,
}

/// What a traced name decomposes into.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Traced<'a> {
    layer: Option<u32>,
    role: &'a str,
    sidecar: Sidecar,
}

/// Split `layer.3.down.zeros` into its three parts.
///
/// `None` for a name that is not in the text's shape at all -- `layer.x.down`,
/// or a bare name with no role. Distinguished from "a role this table does not
/// carry" by the caller, which is why this returns an option rather than a
/// default.
fn decompose(name: &str) -> Option<Traced<'_>> {
    let (rest, sidecar) = match name.rfind('.') {
        Some(at) if &name[at..] == ".scales" => (&name[..at], Sidecar::Scales),
        Some(at) if &name[at..] == ".zeros" => (&name[..at], Sidecar::Zeros),
        Some(at) if &name[at..] == ".bias" => (&name[..at], Sidecar::Bias),
        _ => (name, Sidecar::Packed),
    };
    if let Some(tail) = rest.strip_prefix("layer.") {
        let (index, role) = tail.split_once('.')?;
        Some(Traced {
            layer: Some(index.parse().ok()?),
            role,
            sidecar,
        })
    } else if rest.is_empty() {
        None
    } else {
        Some(Traced {
            layer: None,
            role: rest,
            sidecar,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_scoped_name_becomes_the_loaders_path() {
        assert_eq!(
            Naming::mlx().spellings("layer.11.attn_norm"),
            [
                "layers.11.input_layernorm.weight",
                "layers.11.input_layernorm"
            ]
        );
    }

    #[test]
    fn the_zero_point_the_text_calls_zeros_is_the_loaders_biases() {
        // The disagreement this table exists for. The text says `.zeros`, MLX
        // writes `.biases`, and both are right on their own side.
        //
        // One spelling and not two: `.bias` used to sit beside it for the
        // MXFP4 expert banks, until that was measured and found to be the
        // additive term rather than the codec's zero point. It has its own
        // row now, which the next test reads.
        assert_eq!(
            Naming::mlx().spellings("layer.0.down.zeros"),
            ["layers.0.mlp.down_proj.biases"]
        );
    }

    #[test]
    fn the_additive_bias_the_text_spells_bias_keeps_that_spelling() {
        // The row `.zeros` gave up. A routed expert bank publishes one of
        // these per output row beside its packed weight, which is a different
        // plane from the codec's zero point and asks for its own name.
        assert_eq!(
            Naming::mlx().spellings("layer.0.down.bias"),
            ["layers.0.mlp.down_proj.bias"]
        );
    }

    #[test]
    fn a_scale_has_exactly_one_spelling() {
        assert_eq!(
            Naming::mlx().spellings("layer.0.down.scales"),
            ["layers.0.mlp.down_proj.scales"]
        );
    }

    #[test]
    fn a_global_carries_no_layer_and_no_prefix() {
        assert_eq!(
            Naming::mlx().spellings("final_norm"),
            ["final_norm.weight", "final_norm"]
        );
    }

    #[test]
    fn a_tied_head_and_its_embedding_answer_to_the_same_first_spelling() {
        let n = Naming::mlx();
        assert_eq!(n.spellings("embed")[0], "shared_embedding.weight");
        assert_eq!(n.spellings("lm_head")[0], "shared_embedding.weight");
        // And an untied deployment's second candidate is where they part.
        assert_eq!(n.spellings("embed")[2], "embed_tokens.weight");
        assert_eq!(n.spellings("lm_head")[2], "lm_head.weight");
    }

    #[test]
    fn an_expert_bank_offers_every_convention_it_has_been_seen_under() {
        // Three spellings for one tensor, so a checkpoint chooses rather than
        // this crate choosing a table.
        let s = Naming::mlx().spellings("layer.2.expert_down");
        assert!(s.contains(&"layers.2.mlp.switch_mlp.down_proj.weight".to_string()));
        assert!(s.contains(&"layers.2.experts.switch_glu.down_proj.weight".to_string()));
        assert!(s.contains(&"layers.2.mlp.experts.down_proj.weight".to_string()));
    }

    #[test]
    fn a_role_whose_tensor_is_the_value_offers_the_bare_name() {
        // gpt-oss ships `self_attn.sinks` and no `.weight` under it.
        assert_eq!(
            Naming::mlx().spellings("layer.7.attn_sinks")[1],
            "layers.7.self_attn.sinks"
        );
    }

    #[test]
    fn the_mlp_norm_tries_gemmas_spelling_before_llamas() {
        // The order is load-bearing: the caller keeps the first spelling the
        // checkpoint HAS, and the other order would bind gemma's
        // attention-output norm as its MLP input norm.
        assert_eq!(
            Naming::mlx().spellings("layer.0.mlp_norm")[0],
            "layers.0.pre_feedforward_layernorm.weight"
        );
    }

    #[test]
    fn a_name_outside_the_texts_shape_is_drift_and_not_a_missing_spelling() {
        let n = Naming::mlx();
        // Not a number where a layer index goes.
        assert!(n.spellings("layer.x.down").is_empty());
        // A role no text binds.
        assert!(n.spellings("layer.0.invented").is_empty());
        assert!(n.spellings("invented").is_empty());
        assert!(n.spellings("").is_empty());
    }
}
