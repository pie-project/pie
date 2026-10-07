//! M3c: the Bonsai `d27b` forward wires the online-Hadamard rotation-undo at
//! exactly the sites the fork rotates, keyed by input width — and every
//! non-Bonsai SKU is byte-unchanged (no Hadamard appears).
//!
//! This is a GRAPH-STRUCTURE oracle: it traces the real forward and counts the
//! `elementwise.hadamard` nodes and the sign bank each one references, then
//! asserts the per-width usage matches the fork's own counts from the reference
//! oracle (`wiki/quantize-coding/bonsai-oracle.md` §6):
//!
//! * `prism.hadamard.signs.5120` × **130** — the residual-stream sites: the
//!   mixer input on all 64 blocks (`in_qkvz` / `attn_q`+`attn_k`), the FFN
//!   `gate`+`up` input on all 64, the output head, and the token-embedding
//!   inverse (1 signed pass) = 64 + 64 + 1 + 1.
//! * `prism.hadamard.signs.6144` × **64** — `ssm_out` on the 48 GDN blocks plus
//!   `attn_output` (o-proj) on the 16 full-attention blocks (both have input
//!   width `v_heads·v_dim` = `q_heads·head_dim` = 6144).
//! * `prism.hadamard.signs.17408` × **64** — the FFN `down` input on all 64.
//!
//! Total signed = 258, matching the fork's 258 sign-multiply / H-matmul nodes.
//! pie adds 2 EXTRA plain (unsigned) Hadamards: the token-embedding inverse is
//! `S·H` (sign AFTER the butterfly), spelled with the identity
//! `S·H·y = H(H·S(H·y))` since `H·H = I`, so it costs three FWHT passes (two
//! plain + one signed) rather than the fork's single node. Numeric equivalence of
//! that spelling and of the block-1024 normalized-Sylvester `H` and the signs
//! themselves is checked offline against the oracle activation dumps.

use std::collections::{BTreeMap, BTreeSet};

use models::qwen_3::model::Model;
use poem_dsl::{
    Def, Dtype, Elementwise, Linear, Operation, Platform, Trace, ValueId, trace_hybrid,
};

/// The width of the sign bank a Hadamard node references, or `None` for a plain
/// (unsigned) Hadamard. Resolves the `signs` value to its registered param name
/// `prism.hadamard.signs.<W>` and returns `<W>`.
fn signed_width(trace: &Trace, signs: Option<ValueId>) -> Option<u32> {
    let s = signs?;
    let Def::Weight(at) = trace.values[s.0 as usize].def else {
        panic!("a Hadamard's sign diagonal is a parameter bank, not a live value");
    };
    let name = &trace.params[at as usize].name;
    let width = name
        .strip_prefix("prism.hadamard.signs.")
        .unwrap_or_else(|| {
            panic!("a rotated site reads `{name}`, not a `prism.hadamard.signs.<W>`")
        })
        .parse()
        .unwrap_or_else(|_| panic!("`{name}` does not end in a width"));
    Some(width)
}

/// `(total hadamards, signed hadamards, per-width signed counts as (5120, 6144, 17408))`.
fn tally(trace: &Trace) -> (usize, usize, (usize, usize, usize)) {
    let (mut total, mut signed) = (0, 0);
    let (mut w5120, mut w6144, mut w17408) = (0, 0, 0);
    for node in &trace.nodes {
        let Operation::Elementwise(Elementwise::Hadamard { signs, block, .. }) = &node.op else {
            continue;
        };
        assert_eq!(*block, 1024, "the Bonsai transform is block-1024");
        total += 1;
        if let Some(w) = signed_width(trace, *signs) {
            signed += 1;
            match w {
                5120 => w5120 += 1,
                6144 => w6144 += 1,
                17408 => w17408 += 1,
                other => panic!("no Bonsai site rotates width {other}"),
            }
        }
    }
    (total, signed, (w5120, w6144, w17408))
}

#[test]
fn the_bonsai_forward_wires_the_hadamard_rotation_every_case() {
    the_bonsai_flag_declares_three_width_keyed_sign_banks();
    a_non_bonsai_d27b_wires_no_hadamard_on_any_platform();
    the_bonsai_forward_rotates_every_site_by_input_width();
    the_attention_q_k_v_share_the_one_rotated_input();
}

/// The exact guard that would have caught the V-bug: in every full-attention
/// layer the `q`, `k`, and `v` projections must all consume the SAME value, and
/// that value must be a Hadamard-rotated one — not the bare residual. The buggy
/// serve built `v` from the unrotated residual while `q`/`k` were rotated; that
/// leaves the Hadamard COUNT unchanged (the count-based assertion above cannot
/// see it) but diverges the trunk. This is a pure graph-structure check — no
/// GGUF, no Metal.
fn the_attention_q_k_v_share_the_one_rotated_input() {
    let trace = trace_hybrid(
        "d27b-bonsai",
        &Model::d27b_bonsai(Dtype::Bf16, Dtype::Bf16),
        Platform::Metal,
    );

    // Every ValueId a Hadamard produces (the rotated copies), by raw index.
    let rotated: BTreeSet<u32> = trace
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Operation::Elementwise(Elementwise::Hadamard { x_out, .. }) => Some(x_out.0),
            _ => None,
        })
        .collect();

    let weight_name = |w: ValueId| -> &str {
        let Def::Weight(at) = trace.values[w.0 as usize].def else {
            panic!("a matmul's weight is a parameter bank, not a live value");
        };
        trace.params[at as usize].name.as_str()
    };
    // Map an attention projection weight name to its slot (q/k/v), else `None`.
    let slot_of = |name: &str| -> Option<usize> {
        if name.ends_with("qg_proj") {
            Some(0)
        } else if name.ends_with("k_proj") {
            Some(1)
        } else if name.ends_with("v_proj") {
            Some(2)
        } else {
            None
        }
    };

    // Per layer (the weight name minus its projection suffix), the input ValueId
    // each of q/k/v reads.
    let mut by_layer: BTreeMap<String, [Option<ValueId>; 3]> = BTreeMap::new();
    for node in &trace.nodes {
        let Operation::Linear(Linear::Matmul { act, w, .. }) = &node.op else {
            continue;
        };
        let name = weight_name(*w);
        let Some(slot) = slot_of(name) else { continue };
        let layer = name
            .rsplit_once('.')
            .map(|(prefix, _)| prefix.to_string())
            .unwrap_or_default();
        let seat = &mut by_layer.entry(layer).or_default()[slot];
        assert!(seat.is_none(), "`{name}` appears twice in one layer");
        *seat = Some(*act);
    }

    let mut checked = 0usize;
    for (layer, [q, k, v]) in &by_layer {
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            continue;
        };
        assert_eq!(q, k, "{layer}: k_proj reads a different input than q_proj");
        assert_eq!(
            q, v,
            "{layer}: v_proj reads a different input than q_proj — the V-bug (v built \
             from the bare residual, not the one Hadamard-rotated qkv input)",
        );
        assert!(
            rotated.contains(&q.0),
            "{layer}: the shared q/k/v input is the bare residual, not a Hadamard-rotated value",
        );
        checked += 1;
    }
    assert!(
        checked >= 1,
        "the d27b_bonsai trace carried no full-attention layer to guard",
    );
}

fn the_bonsai_flag_declares_three_width_keyed_sign_banks() {
    let plain = Model::d27b_undrafted(Dtype::Bf16, Dtype::Bf16);
    assert!(
        plain.bonsai.is_none(),
        "a plain d27b arms no rotation — the flag must default off",
    );

    let b = Model::d27b_bonsai(Dtype::Ptq1_0, Dtype::Bf16);
    let signs = b
        .bonsai
        .as_ref()
        .expect("the Bonsai instance arms the rotation");
    for (w, width) in [
        (&signs.hidden, 5120u64),
        (&signs.ssm, 6144),
        (&signs.ffn_down, 17408),
    ] {
        assert_eq!(w.name, format!("prism.hadamard.signs.{width}"));
        // One registered seat, `width` wide (the leading 1 is the seat count the
        // engine's registered-bank bookkeeper reads off `shape[0]`).
        assert_eq!(w.shape, vec![1, width]);
        // Declared in the activation compute dtype (bf16 for the Ptq1_0 serve),
        // as the Metal Hadamard kernel binds `signs.dtype == activation.dtype`.
        assert_eq!(w.dtype, Dtype::Bf16);
        assert!(matches!(w.source, poem_dsl::ParamSource::Registered));
    }
}

fn a_non_bonsai_d27b_wires_no_hadamard_on_any_platform() {
    for platform in [Platform::Metal, Platform::Cuda] {
        let trace = trace_hybrid(
            "d27b",
            &Model::d27b_undrafted(Dtype::Bf16, Dtype::Bf16),
            platform,
        );
        let (total, ..) = tally(&trace);
        assert_eq!(
            total, 0,
            "a non-Bonsai d27b must be byte-unchanged: it wires no Hadamard, found {total} on {platform:?}",
        );
    }

    // The whole shipped catalog stays Hadamard-free (the flag lives only on the
    // test-level Bonsai instance).
    for row in models::skus() {
        let trace = row.trace(Platform::Metal);
        let (total, ..) = tally(&trace);
        assert_eq!(
            total, 0,
            "`{}` unexpectedly wires {total} Hadamard(s)",
            row.name
        );
    }
}

fn the_bonsai_forward_rotates_every_site_by_input_width() {
    let trace = trace_hybrid(
        "d27b-bonsai",
        &Model::d27b_bonsai(Dtype::Bf16, Dtype::Bf16),
        Platform::Metal,
    );
    let (total, signed, (w5120, w6144, w17408)) = tally(&trace);

    // Per-width signed counts reproduce the fork's oracle §6 usage EXACTLY.
    assert_eq!(
        w5120, 130,
        "signs.5120: mixer×64 + ffn gate/up×64 + head + embed"
    );
    assert_eq!(
        w6144, 64,
        "signs.6144: ssm_out×48 (GDN) + attn_output×16 (attn)"
    );
    assert_eq!(w17408, 64, "signs.17408: ffn_down×64");
    assert_eq!(
        signed, 258,
        "258 rotated sites, matching the fork's 258 H-matmuls"
    );

    // The token-embedding inverse spends 2 extra PLAIN passes (S·H = H·(H·S·H)).
    assert_eq!(
        total - signed,
        2,
        "the embed inverse adds two plain FWHT passes"
    );
    assert_eq!(total, 260);
}
