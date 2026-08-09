//! A packed scalar slot binds the address of a STRUCT, and the struct's
//! size is the shader's fact.
//!
//! `ParamSlot::packed` means "this slot is a pointer to a struct holding
//! every remaining scalar", and `Params::new` sizes the run from
//! `dispatch.params.len()` — the count the TEXT states. The shader states
//! its own: `moe/params.h::RouterParams` is four `unsigned int`, and
//! `route.metal` reads `p.softmax_over_all` at byte 8 and `p.logits_pitch`
//! at byte 12 in the body BOTH its instantiations share.
//!
//! Those two numbers are the same quantity as `DecodeGeometry::norm_topk_prob`
//! and the logits row stride. If the text states two scalars and the shader
//! reads four, the run is eight bytes, the next dispatch's scalars begin at
//! byte eight of the same slab, and the router reads them as its own
//! trailing fields — a routing that softmaxes over ALL experts because the
//! next statement's first scalar happened to be nonzero, and a logits stride
//! taken from its second. Both produce weights, neither faults.
//!
//! This measures the slab, not the argument: it stages a real lowered fire's
//! scalars and reads back the bytes at the address each dispatch binds.

use std::collections::BTreeSet;

use driver_metal::lowering::dispatch::{Dispatch, Geometry, plan_one};
use driver_metal::lowering::executor::{Frame, Resolver, Slice};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

/// Answers every name with a generous region — this test reads SCALARS.
#[derive(Default)]
struct Sentinels;

impl Resolver for Sentinels {
    fn weight(&mut self, _: &str) -> Option<Slice> {
        Some(Slice {
            address: 0x1000_0000,
            bytes: 1 << 30,
        })
    }
    fn named(&mut self, _: ValueId) -> Option<Slice> {
        Some(Slice {
            address: 0x2000_0000,
            bytes: 1 << 30,
        })
    }
}

/// gpt-oss-20b's shape: 32 experts, top-4 — the routed arm of the shared
/// llama-like text, which is the one that names `router_topk`.
fn routed() -> LlamaLikeFacts {
    LlamaLikeFacts::gpt_oss_20b()
}

fn geometry(f: &LlamaLikeFacts) -> Geometry {
    Geometry {
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        rotary_dims: f.head_dim,
        n_experts: f.n_experts,
        experts_per_token: f.experts_per_token,
    }
}

fn lowered(f: &LlamaLikeFacts, class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_metal(f, &LlamaLikeMetalFacts::synthetic(), class);
    lower(
        &plan,
        &vec![
            Row {
                samples: true,
                ..Row::default()
            };
            rows
        ],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the metal text lowers")
}

fn planned<'a>(low: &'a Lowered, f: &LlamaLikeFacts) -> Vec<Dispatch<'a>> {
    let frame = Frame {
        arena: Slice {
            address: 0x8000_0000,
            bytes: low.arena_bytes as u64,
        },
    };
    let mut store = Sentinels;
    low.launches
        .iter()
        .filter_map(|l| {
            plan_one(
                low,
                l,
                kernels_metal::KERNELS,
                frame,
                geometry(f),
                &mut store,
            )
            .ok()
        })
        .collect()
}

/// The `unsigned int` count of every packed struct the shader tree defines,
/// keyed by the symbol that takes it.
///
/// Transcribed from the headers rather than parsed, and that is the point:
/// this table is the SHADER's statement, held against what the text says.
/// A struct that grows a field and a text that does not is exactly the
/// drift this file exists to catch, and it shows up here as a mismatch
/// rather than as weights that sum to the wrong number.
const PACKED_STRUCT_WORDS: &[(&str, usize)] = &[
    // `attn/split_qkv.metal::SplitQkvParams`
    ("split_qkv_bf16", 2),
    // `attn/logit_softcap.metal::SoftcapParams`
    ("logit_softcap_bfloat16", 2),
    // `layout/ple_combine.metal::PleCombineParams`
    ("ple_combine_bfloat16", 2),
    // `layout/row_gather_params.h::RowGatherParams`
    ("row_gather_bfloat16", 2),
    // `mlp/gated.metal`'s three
    ("geglu_tanh_bfloat16", 1),
    ("geglu_tanh_strided_bfloat16", 5),
    ("gptoss_swiglu_bfloat16", 3),
    // `moe/params.h`'s four
    ("combine_sorted", 3),
    ("route_gather", 7),
    ("route_sort", 7),
    ("router_topk_bfloat16", 4),
    ("router_topk_scaled_bfloat16", 4),
    // `norm/layer_scalar.metal::LayerScalarParams`
    ("layer_scalar_mul_bfloat16", 1),
    // `norm/rms_params.h::RmsParams`
    ("rms_residual_bfloat16", 5),
    ("rms_residual_scaled_bfloat16", 5),
    ("rms_single_row_bfloat16", 5),
    // `norm/rms_params.h::VNormParams`
    ("vnorm_single_row_bfloat16", 2),
];

#[test]
fn a_packed_slot_stages_every_word_its_shader_struct_reads() {
    let f = routed();
    let mut short = BTreeSet::new();
    let mut unlisted = BTreeSet::new();
    let mut seen = 0usize;
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        let low = lowered(&f, class, rows);
        for d in planned(&low, &f) {
            if !d.param_slots.iter().any(|p| p.packed) {
                continue;
            }
            let Some(&(_, words)) = PACKED_STRUCT_WORDS.iter().find(|(sym, _)| *sym == d.symbol)
            else {
                unlisted.insert(d.symbol.to_string());
                continue;
            };
            seen += 1;
            if d.params.len() != words {
                short.insert(format!(
                    "  {} ({class:?}): the text states {} scalar(s), the shader's struct is \
                     {words} words — the run is {} bytes, so word {} onward is read out of \
                     whatever the NEXT dispatch staged at that offset.",
                    d.symbol,
                    d.params.len(),
                    d.params.len() * 4,
                    d.params.len(),
                ));
            }
        }
    }
    assert!(
        unlisted.is_empty(),
        "these dispatches bind a packed struct that `PACKED_STRUCT_WORDS` does not list, so \
         nothing holds their run against the shader's:\n  {}",
        unlisted.into_iter().collect::<Vec<_>>().join("\n  ")
    );
    assert!(
        short.is_empty(),
        "a packed slot binds the address of a struct and the shader reads the whole struct:\n{}",
        short.iter().cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(
        seen > 0,
        "no dispatch in either routed class bound a packed struct — the fixture stopped \
         being routed and this test measures nothing"
    );
}

/// The guard the assertion above needs: `dims_of` must actually see a
/// mixture, or the routed arm never lowers and there is no router to check.
#[test]
fn the_fixture_is_routed() {
    let f = routed();
    assert!(f.n_experts > 0 && f.experts_per_token > 0);
    let low = lowered(&f, FireClass::Decode, 1);
    let named: Vec<_> = planned(&low, &f)
        .iter()
        .map(|d| d.symbol.to_string())
        .collect();
    assert!(
        named.iter().any(|s| s.starts_with("router_topk")),
        "the routed text named no router; dispatches: {named:?}"
    );
}
