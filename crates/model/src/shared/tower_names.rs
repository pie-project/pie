//! The tensor names a multimodal TOWER publishes, in launcher order.
//!
//! The same rule [`weight_names`](super::weight_names) states, for the
//! encoders instead of the decoder: *"the map between them belongs here and
//! not in a driver."* A tower's tensors are named by the checkpoint and
//! consumed by a launcher, and neither end is a backend's to invent.
//!
//! It lived in `driver-cuda/src/serve/encode.rs`, which spelled roughly fifty
//! paths — `model.audio_tower.layers.{l}.lconv1d.depthwise_conv1d.weight` and
//! its neighbours — to build the flat pointer table the encode entry takes.
//! That made a backend the only place that knew how one family's audio
//! front-end is put together, and a second backend growing an encode path
//! would have needed the same fifty.
//!
//! # Why a flat ordered list
//!
//! The launcher takes a `const void**` and indexes it by a fixed stride: the
//! audio tower is 62 entries per layer, the vision tower 41. So the ORDER is
//! the ABI, and a name list that did not preserve it would be a different
//! contract wearing the same shape. [`Slot`] therefore carries only what the
//! caller must do differently — refuse, or bind a null — and the sequence
//! carries everything else.
//!
//! # Required and optional are not the same refusal
//!
//! A quantised linear ships `input_min`/`input_max`/`output_min`/`output_max`
//! beside its weight; an unquantised one ships the weight alone. The launcher
//! reads a null for the absent bounds and takes the unquantised path, so an
//! absent bound is a STATEMENT and a missing weight is a fault. Collapsing
//! them would either refuse every unquantised tower or bind a null where the
//! kernel dereferences one.

/// One entry of a tower's pointer table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    /// The tower does not run without this tensor; a caller that cannot find
    /// it refuses the encode.
    Required(String),
    /// Absent means "not quantised", which the launcher reads as a null
    /// pointer and handles. A caller binds null rather than refusing.
    Optional(String),
}

impl Slot {
    /// The checkpoint name, whichever kind of slot this is.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Required(n) | Self::Optional(n) => n,
        }
    }
}

/// The five entries a CLIPPED linear contributes: the weight, then its four
/// quantisation bounds.
fn clipped_linear(base: &str, out: &mut Vec<Slot>) {
    out.push(Slot::Required(format!("{base}.linear.weight")));
    for m in ["input_min", "input_max", "output_min", "output_max"] {
        out.push(Slot::Optional(format!("{base}.{m}")));
    }
}

/// The audio tower's HEAD tensors — the subsample projection, the output
/// projection and the embedding projection — in launcher argument order.
///
/// These are passed as named arguments rather than through the table, which
/// is why they are a separate list: the launcher's first eight parameters.
#[must_use]
pub fn audio_head(prefix: &str, embed: &str) -> Vec<Slot> {
    [
        "subsample_conv_projection.layer0.conv.weight",
        "subsample_conv_projection.layer0.norm.weight",
        "subsample_conv_projection.layer1.conv.weight",
        "subsample_conv_projection.layer1.norm.weight",
        "subsample_conv_projection.input_proj_linear.weight",
        "output_proj.weight",
        "output_proj.bias",
    ]
    .into_iter()
    .map(|n| Slot::Required(format!("{prefix}.{n}")))
    .chain(std::iter::once(Slot::Required(embed.to_owned())))
    .collect()
}

/// The audio tower's per-layer table: 62 slots a layer, `layers` layers.
///
/// The order is the launcher's and is not negotiable — two feed-forwards,
/// the attention norms, four clipped attention projections, the relative
/// position projection and per-dimension scale, the light convolution block,
/// and the output norm.
#[must_use]
pub fn audio_layers(prefix: &str, layers: u32) -> Vec<Slot> {
    /// The 22 slots one feed-forward contributes.
    fn feed_forward(base: &str, out: &mut Vec<Slot>) {
        out.push(Slot::Required(format!("{base}.pre_layer_norm.weight")));
        out.push(Slot::Required(format!("{base}.post_layer_norm.weight")));
        clipped_linear(&format!("{base}.ffw_layer_1"), out);
        clipped_linear(&format!("{base}.ffw_layer_2"), out);
    }

    let mut out = Vec::with_capacity(layers as usize * AUDIO_SLOTS_PER_LAYER);
    for l in 0..layers {
        let lp = format!("{prefix}.layers.{l}");
        feed_forward(&format!("{lp}.feed_forward1"), &mut out);
        feed_forward(&format!("{lp}.feed_forward2"), &mut out);
        out.push(Slot::Required(format!("{lp}.norm_pre_attn.weight")));
        out.push(Slot::Required(format!("{lp}.norm_post_attn.weight")));
        for p in ["q_proj", "k_proj", "v_proj", "post"] {
            clipped_linear(&format!("{lp}.self_attn.{p}"), &mut out);
        }
        out.push(Slot::Required(format!(
            "{lp}.self_attn.relative_k_proj.weight"
        )));
        out.push(Slot::Required(format!("{lp}.self_attn.per_dim_scale")));
        out.push(Slot::Required(format!(
            "{lp}.lconv1d.pre_layer_norm.weight"
        )));
        out.push(Slot::Required(format!("{lp}.lconv1d.conv_norm.weight")));
        clipped_linear(&format!("{lp}.lconv1d.linear_start"), &mut out);
        clipped_linear(&format!("{lp}.lconv1d.linear_end"), &mut out);
        out.push(Slot::Required(format!(
            "{lp}.lconv1d.depthwise_conv1d.weight"
        )));
        out.push(Slot::Required(format!("{lp}.norm_out.weight")));
    }
    out
}

/// Slots the audio launcher indexes per layer. The stride IS the ABI.
pub const AUDIO_SLOTS_PER_LAYER: usize = 62;

/// Slots the vision launcher indexes per layer.
pub const VISION_SLOTS_PER_LAYER: usize = 41;

/// The vision tower's HEAD tensors, in launcher argument order.
#[must_use]
pub fn vision_head(prefix: &str, embed: &str) -> Vec<Slot> {
    vec![
        Slot::Required(format!("{prefix}.patch_embedder.input_proj.weight")),
        Slot::Required(format!("{prefix}.patch_embedder.position_embedding_table")),
        Slot::Required(embed.to_owned()),
    ]
}

/// The vision tower's per-layer table: 41 slots a layer.
#[must_use]
pub fn vision_layers(prefix: &str, layers: u32) -> Vec<Slot> {
    let mut out = Vec::with_capacity(layers as usize * VISION_SLOTS_PER_LAYER);
    for l in 0..layers {
        let lp = format!("{prefix}.encoder.layers.{l}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            out.push(Slot::Required(format!("{lp}.{norm}.weight")));
        }
        for c in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            clipped_linear(&format!("{lp}.{c}"), &mut out);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stride IS the ABI: the launcher indexes `table[layer * 62 + k]`,
    /// so a slot added or dropped shifts every later layer's operands by one
    /// and reads a norm as a weight. Nothing about that fails loudly on
    /// device, which is why it is asserted here.
    #[test]
    fn the_audio_table_is_sixty_two_slots_a_layer() {
        let one = audio_layers("model.audio_tower", 1);
        assert_eq!(one.len(), AUDIO_SLOTS_PER_LAYER);
        let three = audio_layers("model.audio_tower", 3);
        assert_eq!(three.len(), 3 * AUDIO_SLOTS_PER_LAYER);
        assert_eq!(
            &three[..AUDIO_SLOTS_PER_LAYER],
            &one[..],
            "layer 0 is layer 0"
        );
    }

    /// The same claim for the vision tower's 41.
    #[test]
    fn the_vision_table_is_forty_one_slots_a_layer() {
        let one = vision_layers("model.vision_tower", 1);
        assert_eq!(one.len(), VISION_SLOTS_PER_LAYER);
        assert_eq!(
            vision_layers("model.vision_tower", 4).len(),
            4 * VISION_SLOTS_PER_LAYER
        );
    }

    /// A clipped linear's bounds are OPTIONAL and its weight is not. The
    /// launcher takes a null for an absent bound and the unquantised path;
    /// a missing weight is a fault. Collapsing the two would either refuse
    /// every unquantised tower or dereference a null.
    #[test]
    fn only_the_quantisation_bounds_are_optional() {
        let l = audio_layers("t", 1);
        let required = l.iter().filter(|s| matches!(s, Slot::Required(_))).count();
        let optional = l.iter().filter(|s| matches!(s, Slot::Optional(_))).count();
        assert_eq!(required + optional, AUDIO_SLOTS_PER_LAYER);
        // Ten clipped linears a layer, four bounds each.
        assert_eq!(optional, 40);
        assert!(
            l.iter().all(|s| match s {
                Slot::Optional(n) => !n.ends_with(".weight"),
                Slot::Required(_) => true,
            }),
            "no weight is optional"
        );
    }

    /// The first slots spell what the C++ table's layout says they do. Pinned
    /// because the ORDER is the contract and a reordering is invisible.
    #[test]
    fn the_first_layers_slots_are_the_launchers_order() {
        let l = audio_layers("model.audio_tower", 1);
        assert_eq!(
            l[0].name(),
            "model.audio_tower.layers.0.feed_forward1.pre_layer_norm.weight"
        );
        assert_eq!(
            l[1].name(),
            "model.audio_tower.layers.0.feed_forward1.post_layer_norm.weight"
        );
        assert_eq!(
            l[2].name(),
            "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"
        );
        assert_eq!(
            l[3].name(),
            "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.input_min"
        );
        assert_eq!(
            l[AUDIO_SLOTS_PER_LAYER - 1].name(),
            "model.audio_tower.layers.0.norm_out.weight"
        );
        let v = vision_layers("model.vision_tower", 1);
        assert_eq!(
            v[0].name(),
            "model.vision_tower.encoder.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            v[6].name(),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
        );
    }

    /// The audio head is EIGHT slots in one order, and the consumer
    /// destructures them positionally.
    ///
    /// `driver-cuda`'s `encode.rs` writes
    /// `let [sscp0_conv, sscp0_norm, sscp1_conv, sscp1_norm, sscp_proj,
    /// out_w, out_b, embed_p] = heads[..] else { return UNSUPPORTED }`.
    /// A slot added or reordered here does not fail that binding — it
    /// silently hands the launcher a norm where a conv belongs, or (for
    /// a length change) turns every audio model into an unexplained
    /// `PIE_STATUS_UNSUPPORTED`.
    #[test]
    fn the_audio_head_is_the_launchers_first_eight_in_order() {
        let head = audio_head("model.audio_tower", "model.embed_audio.proj.weight");
        let names: Vec<&str> = head
            .iter()
            .map(|s| match s {
                Slot::Required(n) | Slot::Optional(n) => n.as_str(),
            })
            .collect();
        assert_eq!(
            names,
            [
                "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
                "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
                "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
                "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
                "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
                "model.audio_tower.output_proj.weight",
                "model.audio_tower.output_proj.bias",
                "model.embed_audio.proj.weight",
            ]
        );
    }

    /// The vision head is three, and the last is the embedding
    /// projection — which is NOT under the tower prefix, because it
    /// belongs to the text side that receives the tower's output.
    #[test]
    fn the_vision_head_is_three_and_the_embedding_is_not_the_towers() {
        let head = vision_head("model.vision_tower", "model.embed_vision.proj.weight");
        let names: Vec<&str> = head
            .iter()
            .map(|s| match s {
                Slot::Required(n) | Slot::Optional(n) => n.as_str(),
            })
            .collect();
        assert_eq!(
            names,
            [
                "model.vision_tower.patch_embedder.input_proj.weight",
                "model.vision_tower.patch_embedder.position_embedding_table",
                "model.embed_vision.proj.weight",
            ]
        );
    }

    /// No head slot is optional, on either tower.
    ///
    /// The per-layer tables carry optionals — a quantisation bound the
    /// launcher takes a null for. A head tensor has no such convention:
    /// the consumer's `need()` returns an error code and the positional
    /// destructure has no room for an absent one.
    #[test]
    fn nothing_in_either_head_is_optional() {
        for (tower, head) in [
            ("audio", audio_head("t", "e")),
            ("vision", vision_head("t", "e")),
        ] {
            assert!(
                head.iter().all(|s| matches!(s, Slot::Required(_))),
                "{tower}: a head tensor the launcher takes a null for would \
                 be dereferenced"
            );
        }
    }

    /// The prefix reaches every tower slot, and the embed argument
    /// reaches exactly one.
    ///
    /// Both towers live under `model.` in one checkpoint and could live
    /// anywhere in another, so a hard-coded name here would bind the
    /// wrong tensor rather than fail to find one.
    #[test]
    fn the_prefix_is_a_parameter_and_the_embedding_is_a_separate_one() {
        for (tower, head, layers) in [
            ("audio", audio_head("PFX", "EMB"), audio_layers("PFX", 2)),
            ("vision", vision_head("PFX", "EMB"), vision_layers("PFX", 2)),
        ] {
            let name = |s: &Slot| match s {
                Slot::Required(n) | Slot::Optional(n) => n.clone(),
            };
            let embeds = head.iter().filter(|s| name(s) == "EMB").count();
            assert_eq!(embeds, 1, "{tower}: exactly one slot is the embedding");
            assert!(
                head.iter()
                    .chain(layers.iter())
                    .all(|s| name(s) == "EMB" || name(s).starts_with("PFX.")),
                "{tower}: every other slot is under the prefix"
            );
        }
    }
}
