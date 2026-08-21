//! A packed scalar slot binds the address of a STRUCT, and the struct's
//! size is the shader's fact — so this file held every packed run against
//! the struct it is read as. **There are none left in a model text.**
//!
//! `ParamSlot::packed` means "this slot is a pointer to a struct holding
//! every remaining scalar", and `Params::new` sizes the run from
//! `dispatch.params.len()` — the count the TEXT states. The shader stated
//! its own, and the two were never held together: `moe/params.h::RouterParams`
//! was four `unsigned int` and the text stated two, so `route.metal` read
//! `p.softmax_over_all` at byte 8 and `p.logits_pitch` at byte 12 out of
//! whatever the NEXT dispatch staged there — a routing that softmaxes over
//! ALL experts because a neighbouring statement's first word happened to be
//! nonzero, and a logits stride taken from its second. Both produce weights,
//! neither faults. That is what this file was written to catch.
//!
//! It is not caught here any more, because it can no longer happen. The
//! convention was unwound (`kernels-metal/kernels/moe/route.metal`, and the
//! same for `norm`, `layout`, `mlp` and `attn`): every field is a
//! `const constant uint&` of its own now, one mark per field, at ascending
//! buffer indices after the operands. A text that states fewer fields than
//! the shader declares leaves an argument slot UNBOUND, which faults rather
//! than reading a neighbour — the silent failure mode is gone by
//! construction rather than by measurement.
//!
//! So this file's table is empty by FACT and not by omission, and what
//! remains is a trip-wire: a dispatch that binds a packed slot again is a
//! return to the convention, and it fails here until someone transcribes the
//! shader's word count beside it. The one surviving `ctx.params()` caller is
//! `kernels-metal::ptir::copy_logits_bf16`, whose block is a `const device
//! PtirLogitsCopyParams*` ARRAY indexed by `tid.y` rather than one statement's
//! scalar run; it is fired by the sampler and not by a model text, so no
//! lowering reachable from here binds it.
//!
//! This measures the slab, not the argument: it stages a real lowered fire's
//! scalars and reads back the bytes at the address each dispatch binds.

use std::collections::BTreeSet;

use driver_metal::lowering::dispatch::{Dispatch, Geometry, plan_launch};
use driver_metal::lowering::executor::{Frame, Resolver, Slice};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_ir::trace::{FireClass, ValueId};

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
    /// THE POOL'S NUMBERS, WHICH THE DEFAULT REFUSES. `kv_append_paged` reaches
    /// for its page size and both cache strides, and a resolver that answers
    /// none of them refuses every routed launch -- so this file planned nothing
    /// and its own denominator said so. A pool of 16-token pages over 8 heads
    /// of 128 is plausible and nothing here reads the values, only that they
    /// are there.
    fn pool(&mut self, which: driver_metal::lowering::executor::FireTable) -> Option<u32> {
        use driver_metal::lowering::executor::FireTable;
        Some(match which {
            FireTable::KvPageSize => 16,
            FireTable::KvHeadStride => 128,
            FireTable::KvSeqStride => 8 * 128,
            _ => return None,
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
        ..Geometry::default()
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
        .flat_map(|l| {
            plan_launch(low, l, frame, geometry(f), &mut store)
                .ok()
                .unwrap_or_default()
        })
        .collect()
}

/// The `unsigned int` count of every packed struct the shader tree defines,
/// keyed by the symbol that takes it — and the tree defines none.
///
/// It held eighteen symbols: `split_qkv_bf16` and `logit_softcap_bfloat16`,
/// `ple_combine_bfloat16` and `row_gather_bfloat16`, the three of
/// `mlp/gated.metal`, `moe/params.h`'s five, `layer_scalar_mul_bfloat16`, the
/// four of `norm/rms_params.h`, and `vnorm_single_row_bfloat16`. Each of
/// those headers is gone and each of those rows now takes its fields one
/// `const constant uint&` at a time, so there is nothing left to transcribe.
///
/// Transcribed from the headers rather than parsed, and that is still the
/// point: this table is the SHADER's statement, held against what the text
/// says. An entry re-appearing here means the packed convention came back,
/// and it has to come back with its word count.
const PACKED_STRUCT_WORDS: &[(&str, usize)] = &[];

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
    // NO `seen > 0`. Zero is the answer this file now expects, and the
    // count is kept because it is the number the emptiness is read from:
    // `unlisted` is what fires when it stops being zero. A guard demanding
    // a packed dispatch would fail on a tree that correctly has none, and
    // that the fixture still lowers something routed is
    // `the_fixture_is_routed`'s question, not this one's.
    assert_eq!(
        seen, 0,
        "a dispatch bound a packed struct AND the table names its word count, so the \
         packed convention is back and `PACKED_STRUCT_WORDS` should say so in its text \
         as well as in its rows"
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
