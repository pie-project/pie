//! The generated wrappers — CUDA's and the metal plane's — are CURRENT,
//! and they round-trip.
//!
//! `wrappers_are_current` re-runs the generator (`tests/generator/mod.rs`) over
//! `crates/kernels-cuda/src` and diffs the result against the committed
//! `src/cuda/generated.rs` — the `model-loader/tests/golden_plans.rs`
//! idiom — and `metal_wrappers_are_current` does the same over
//! `crates/kernels-metal/src` against `src/metal/generated.rs`. A stale
//! file refuses;
//! `UPDATE_WRAPPERS=1 cargo test -p model-dsl --test wrappers_are_current`
//! rewrites it (and, like every golden rewrite, the SAME run still tests
//! the code compiled from the old file — run it once more to prove the
//! new one).
//!
//! The round-trip tests are B4-gen step 5 (design-no-ask §10): one
//! statement traced through the generated fn and through the hand-written
//! wrapper must record IDENTICAL ops — the only difference allowed is the
//! retired restatement at the CALL SITE (`intermediate` / `width`), which
//! the routine's `out(..)` rule now derives.

mod generator;

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Regenerate-and-diff for one plane: the shared half of the two currency
/// gates.
fn assert_current(want: &str, at: &PathBuf, plane: &str, kernels: &str) {
    let have = std::fs::read_to_string(at).unwrap_or_default();
    if want == have {
        return;
    }
    if std::env::var_os("UPDATE_WRAPPERS").is_some() {
        std::fs::write(at, want).unwrap_or_else(|e| panic!("rewriting {}: {e}", at.display()));
        return;
    }
    // Point at the first diverging line rather than dumping two files.
    let line = want
        .lines()
        .zip(have.lines())
        .position(|(w, h)| w != h)
        .map_or_else(
            || want.lines().count().min(have.lines().count()) + 1,
            |i| i + 1,
        );
    panic!(
        "{plane} is STALE against {kernels} \
         (first divergence at line {line}). The wrappers are generated, \
         never edited: regenerate with \
         UPDATE_WRAPPERS=1 cargo test -p model-dsl --test wrappers_are_current \
         and review the diff."
    );
}

#[test]
fn wrappers_are_current() {
    let kernels = manifest_dir().join("../kernels-cuda/src");
    let want = generator::generate(&kernels);
    let at = manifest_dir().join("src/cuda/generated.rs");
    assert_current(&want, &at, "src/cuda/generated.rs", "crates/kernels-cuda/src");
}

#[test]
fn metal_wrappers_are_current() {
    let kernels = manifest_dir().join("../kernels-metal/src");
    let want = generator::generate_metal(&kernels);
    let at = manifest_dir().join("src/metal/generated.rs");
    assert_current(
        &want,
        &at,
        "src/metal/generated.rs",
        "crates/kernels-metal/src",
    );
}

/// `mlp::chunked_swiglu`, stated-shape vs generated. The stated side is
/// what the deleted hand wrapper recorded — the same marker fired with the
/// result's `(Shape, DType)` written out; the generated fn derives it from
/// the routine's `out(y = rows(packed) x half(packed))` rule. Same
/// statement either way.
#[test]
fn chunked_swiglu_round_trips() {
    use model_dsl::fire::{Call, fire};
    use model_ir::trace::{DType, Dim, Shape};
    let stated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let packed = model_dsl::input(t, 512);
        let _y = fire::<kernels_cuda::mlp::chunked_swiglu>(
            t,
            Call {
                inputs: vec![packed.key()],
                outs: vec![(Shape(vec![Dim::Tokens, Dim::Const(256)]), DType::BF16)],
                ..Default::default()
            },
        );
    });
    let generated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let packed = model_dsl::input(t, 512);
        let _y = model_dsl::cuda::generated::chunked_swiglu(&packed, None, None);
    });
    assert_eq!(
        stated, generated,
        "the generated `chunked_swiglu` must record exactly what the stated \
         form records, with the `intermediate` restatement retired"
    );
}

/// `mlp::relu2`, stated-shape vs generated: the generated fn derives the
/// result from `out(y = like(x))`.
#[test]
fn relu2_round_trips() {
    use model_dsl::fire::{Call, fire};
    use model_ir::trace::{DType, Dim, Shape};
    let stated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let x = model_dsl::input(t, 384);
        let _y = fire::<kernels_cuda::mlp::relu2>(
            t,
            Call {
                inputs: vec![x.key()],
                outs: vec![(Shape(vec![Dim::Tokens, Dim::Const(384)]), DType::BF16)],
                ..Default::default()
            },
        );
    });
    let generated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let x = model_dsl::input(t, 384);
        let _y = model_dsl::cuda::generated::relu2(&x, None, None);
    });
    assert_eq!(
        stated, generated,
        "the generated `relu2` must record exactly what the stated form \
         records, with the `width` restatement retired"
    );
}

/// The metal plane's round-trip: `rms_single_row`, stated-shape vs
/// generated. The stated side is what the deleted hand `rms_norm` wrapper
/// recorded — the entrypoint symbol, the six-word params run with the
/// zero `rows` placeholder, the spliced token extent, and the result's
/// `(Shape, DType)` written out; the generated fn derives the result from
/// `out(out = like(x))` and splices `rows` by the plane's convention.
/// Same statement either way.
#[test]
fn metal_rms_single_row_round_trips() {
    use model_dsl::fire::{Call, fire_at};
    use model_ir::trace::{DType, Dim, Shape};
    let stated = model_dsl::trace_named("b4gen.metal.decode", |t| {
        let x = model_dsl::input(t, 512);
        let _y = fire_at::<kernels_metal::norm::rms_single_row>(
            t,
            "rms_single_row_bfloat16",
            Call {
                inputs: vec![x.key()],
                weights: vec!["norm.w".to_string()],
                params: vec![1e-6f32.to_bits(), 512, 1, 0, 1.0f32.to_bits(), 0],
                outs: vec![(Shape(vec![Dim::Tokens, Dim::Const(512)]), DType::BF16)],
                extents: vec![(5, Shape(vec![Dim::Tokens]))],
                ..Default::default()
            },
        );
    });
    let generated = model_dsl::trace_named("b4gen.metal.decode", |t| {
        let x = model_dsl::input(t, 512);
        let _y = model_dsl::metal::generated::rms_single_row(
            &x, "norm.w", 1e-6, 512, 1, 0, 1.0, None, None,
        );
    });
    assert_eq!(
        stated, generated,
        "the generated `rms_single_row` must record exactly what the hand \
         `rms_norm` wrapper recorded, result shape and rows extent derived"
    );
}

/// The metal plane's second round-trip: `silu_mul` — two operands, the
/// `out(out = like(gate))` rule, and the spliced `rows` extent read off
/// the first operand's own row axis.
#[test]
fn metal_silu_mul_round_trips() {
    use model_dsl::fire::{Call, fire_at};
    use model_ir::trace::{DType, Dim, Shape};
    let stated = model_dsl::trace_named("b4gen.metal.decode", |t| {
        let gate = model_dsl::input(t, 256);
        let up = model_dsl::input(t, 256);
        let _y = fire_at::<kernels_metal::mlp::silu_mul>(
            t,
            "silu_mul_bfloat16",
            Call {
                inputs: vec![gate.key(), up.key()],
                params: vec![0],
                outs: vec![(Shape(vec![Dim::Tokens, Dim::Const(256)]), DType::BF16)],
                extents: vec![(0, Shape(vec![Dim::Tokens]))],
                ..Default::default()
            },
        );
    });
    let generated = model_dsl::trace_named("b4gen.metal.decode", |t| {
        let gate = model_dsl::input(t, 256);
        let up = model_dsl::input(t, 256);
        let _y = model_dsl::metal::generated::silu_mul(&gate, &up, None, None);
    });
    assert_eq!(
        stated, generated,
        "the generated `silu_mul` must record exactly what the hand wrapper \
         recorded, with the `intermediate` restatement retired"
    );
}

/// B4-gen step 8, first half: every traced `#[routine]` in
/// `crates/kernels-cuda/src` has a generated wrapper in the COMMITTED
/// `src/cuda/generated.rs`. A routine added to the plane is a routine the
/// DSL can state with no wrapper written — this is that guarantee, pinned.
#[test]
fn every_traced_routine_has_a_generated_wrapper() {
    let kernels = manifest_dir().join("../kernels-cuda/src");
    let committed = std::fs::read_to_string(manifest_dir().join("src/cuda/generated.rs"))
        .expect("the generated wrappers are checked in");
    let missing: Vec<String> = generator::traced(&kernels)
        .into_iter()
        .filter(|(_, name)| !committed.contains(&format!("pub fn {name}(")))
        .map(|(symbol, _)| symbol)
        .collect();
    assert!(
        missing.is_empty(),
        "traced routines with no generated wrapper (regenerate with \
         UPDATE_WRAPPERS=1): {missing:?}"
    );
}

/// B4-gen step 8, second half: NO hand-written wrapper in
/// `src/cuda/**` states a symbol a generated fn also states — the
/// restatement this whole section deleted must not grow back. The
/// keepers are allowlisted BY SYMBOL, each with the one fact that keeps
/// it a decision rather than a restatement; an allowlist row nothing
/// uses any more refuses too, so the list cannot rot.
#[test]
fn no_hand_wrapper_shadows_a_generated_one() {
    // (symbol, why the hand form survives step 7)
    const KEEPERS: &[(&str, &str)] = &[
        (
            "rope::qk_rmsnorm_rope_bf16_devwin",
            "peel-window + spliced token extent: the walk fills slots no caller can state",
        ),
        (
            "attn::write_kv_explicit_bf16_devwin",
            "peel-window + spliced token extent",
        ),
        (
            "attn::attention_xqa_decode_bf16_prepared",
            "spliced request extent + guard-region result",
        ),
        (
            "attn::dispatch_attention_flashinfer_decode",
            "mints the decode plan prep + region-aware result + sm_scale policy",
        ),
        (
            "attn::dispatch_attention_flashinfer_decode_lse",
            "mints the decode plan; the window picks the schedule",
        ),
        (
            "attn::dispatch_attention_flashinfer_decode_capture",
            "guard-region (outless) + score-view operand",
        ),
        (
            "attn::dispatch_attention_flashinfer_prefill_bf16",
            "mints the prefill plan prep",
        ),
        (
            "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
            "guard-region + score-view operand",
        ),
        (
            "attn::dispatch_attention_flashinfer_prefill_custom",
            "mints the prefill plan + custom-mask view",
        ),
        (
            "attn::attention_flashinfer_prefill",
            "planless arm: region-aware result + host-mirror views + sm_scale policy",
        ),
        (
            "attn::attention_flashinfer_prefill_lse",
            "planless LSE arm: host-mirror views + sm_scale policy",
        ),
        (
            "attn::attention_naive_paged",
            "region-aware result (a guard may own the value)",
        ),
        (
            "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
            "the REGION form: records no result while the routine marks one",
        ),
        (
            "ssm::recurrent_gated_delta_step_batched",
            "spliced request extent + 4-way symbol choice (gqa x state dtype)",
        ),
        (
            "ssm::recurrent_gated_delta_step_batched_state_bf16",
            "spliced request extent + symbol choice",
        ),
        (
            "ssm::recurrent_gated_delta_step_batched_gqa",
            "spliced request extent + symbol choice",
        ),
        (
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
            "spliced request extent + symbol choice",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched",
            "guard-region (outless) + symbol choice by state dtype",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched_state_bf16",
            "guard-region + symbol choice",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched_cached",
            "guard-region + symbol choice",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
            "guard-region + symbol choice",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
            "guard-region + symbol choice",
        ),
        (
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
            "guard-region + symbol choice",
        ),
        (
            "attn::dsv4_boundary_meta_decode",
            "fire-class chooser (decode/paged pick different launchers)",
        ),
        ("attn::dsv4_boundary_meta_paged", "fire-class chooser"),
        (
            "attn::dsv4_compress_gather_paged_bf16",
            "derives `coff` from the ratio: the driver's rule, stated once",
        ),
        (
            "quant::dequant_fp8_e4m3_to",
            "dequant loader: symbol chosen by scale layout; weight-shaped statement",
        ),
        (
            "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            "dequant loader arm",
        ),
        (
            "quant::dequant_fp8_e4m3_to_bf16_per_group",
            "dequant loader arm",
        ),
        (
            "quant::dequant_mxfp4_to",
            "dequant loader: weight-shaped statement, no operand values",
        ),
        ("mlp::geglu_tanh", "packed/pair symbol chooser"),
        ("mlp::chunked_geglu_tanh", "packed/pair symbol chooser"),
        (
            "norm::scalar_mul",
            "optional-scalar policy + `scale.<name>` weight naming",
        ),
        ("moe::gather_moe_aligned_inputs", "spliced token extent"),
    ];

    let kernels = manifest_dir().join("../kernels-cuda/src");
    let generated: std::collections::BTreeSet<String> = generator::traced(&kernels)
        .into_iter()
        .map(|(symbol, _)| symbol)
        .collect();

    // Symbols the HAND sources state: quoted string literals shaped like
    // `ns::name`, with comment lines dropped so prose cannot trip it.
    let cuda_dir = manifest_dir().join("src/cuda");
    let mut hand: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&cuda_dir).expect("src/cuda listing") {
        let path = entry.expect("src/cuda entry").path();
        if path.file_name().is_some_and(|f| f == "generated.rs")
            || path.extension().and_then(|e| e.to_str()) != Some("rs")
        {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("hand wrapper source");
        for line in src.lines() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            let mut rest = code;
            while let Some(open) = rest.find('"') {
                let Some(len) = rest[open + 1..].find('"') else {
                    break;
                };
                let lit = &rest[open + 1..open + 1 + len];
                if lit.contains("::")
                    && lit.chars().all(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == ':'
                    })
                {
                    hand.insert(lit.to_string());
                }
                rest = &rest[open + 1 + len + 1..];
            }
        }
    }

    let shadowing: Vec<&String> = hand
        .iter()
        .filter(|s| generated.contains(*s) && !KEEPERS.iter().any(|(k, _)| *k == s.as_str()))
        .collect();
    assert!(
        shadowing.is_empty(),
        "hand-written wrappers state symbols the generated fns already \
         state; delete the hand form or allowlist it with its reason: \
         {shadowing:?}"
    );

    let stale: Vec<&str> = KEEPERS
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| !hand.contains(*k))
        .collect();
    assert!(
        stale.is_empty(),
        "allowlisted keeper symbols no hand wrapper states any more — \
         drop the rows: {stale:?}"
    );
}

/// The metal half of B4-gen step 8's first pin: every traced `#[routine]`
/// in `crates/kernels-metal/src` has a generated wrapper in the COMMITTED
/// `src/metal/generated.rs`.
#[test]
fn every_traced_metal_routine_has_a_generated_wrapper() {
    let kernels = manifest_dir().join("../kernels-metal/src");
    let committed = std::fs::read_to_string(manifest_dir().join("src/metal/generated.rs"))
        .expect("the generated wrappers are checked in");
    let missing: Vec<String> = generator::traced_metal(&kernels)
        .into_iter()
        .filter(|(_, name)| !committed.contains(&format!("pub fn {name}(")))
        .map(|(_, name)| name)
        .collect();
    assert!(
        missing.is_empty(),
        "traced metal routines with no generated wrapper (regenerate with \
         UPDATE_WRAPPERS=1): {missing:?}"
    );
}

/// The metal half of the no-shadowing pin. A metal statement's symbol is
/// the plane's instantiated ENTRYPOINT, so what a hand wrapper can shadow
/// is a FIXED symbol — one a routine's body names as its single literal
/// and the generated fn therefore states verbatim. Hand wrappers that
/// compose a symbol out of an instantiation point (`format!`, the point
/// tables) state no fixed literal and cannot shadow; they remain keepers
/// by the same categories as CUDA's, listed in the report of the sweep
/// rather than here.
///
/// The keepers below each state a fixed symbol a generated fn also
/// states, with the one fact that keeps the hand form a decision rather
/// than a restatement; a row nothing states any more refuses too, so the
/// list cannot rot.
#[test]
fn no_metal_hand_wrapper_shadows_a_generated_one() {
    // (fixed symbol, why the hand form survives step 7)
    const KEEPERS: &[(&str, &str)] = &[
        (
            "neox_mb_bfloat16",
            "the rope ladder chooser: one hand fn picks among three fixed \
             rotations, fans one call onto q and k, and mints the tier-2 \
             freqs stream where the ladder is a table",
        ),
        ("neox_prop_mb_bfloat16", "the rope ladder chooser's arm"),
        ("neox_freqs_mb_bfloat16", "the rope ladder chooser's table arm"),
        (
            "kv_append_bfloat16",
            "the paged/contiguous chooser; the paged row's fire extent is \
             spelled `tokens`, which the rows convention does not claim",
        ),
        ("kv_append_paged_bfloat16", "the paged/contiguous chooser's arm"),
        (
            "cast_qmm_input_strided_bfloat16_to_float16",
            "region form (`cast_qmm_input_when`'s arm records no value) + \
             the pitch and the F16 rectangle derived from the operand",
        ),
        (
            "router_topk_bfloat16",
            "scaled/unscaled chooser tied to the per-expert weight it \
             scales by",
        ),
        ("router_topk_scaled_bfloat16", "the scaled chooser's arm"),
        (
            "route_sort",
            "two spliced extents (the route pairs and the sorted stack) \
             the rows convention does not claim",
        ),
        (
            "route_gather",
            "spliced pair/stack extents under `n`/`padded`/`padded_rows`",
        ),
        (
            "combine_sorted",
            "spliced token extent spelled `tokens`: the fire's rows, not \
             the stack the first operand carries",
        ),
        (
            "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
            "repr symbol chooser (affine/mxfp4 x bias; the affine arms are \
             composed, this one is the fixed MXFP4 point) + the bias weight \
             list + a Tokens extent the first operand does not carry",
        ),
        (
            "gdn_prep_prefill_bfloat16",
            "the prompt's spliced extent is spelled `n_scan`, which the \
             rows convention does not claim",
        ),
        (
            "row_gather_bfloat16",
            "two spliced Requests extents under `count`/`row_count`",
        ),
    ];

    let kernels = manifest_dir().join("../kernels-metal/src");
    let generated: std::collections::BTreeSet<String> = generator::traced_metal(&kernels)
        .into_iter()
        .filter_map(|(symbol, _)| symbol)
        .collect();

    // Symbols the HAND sources state: quoted string literals shaped like
    // entrypoints, with comment lines dropped so prose cannot trip it.
    let metal_dir = manifest_dir().join("src/metal");
    let mut hand: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&metal_dir).expect("src/metal listing") {
        let path = entry.expect("src/metal entry").path();
        if path.file_name().is_some_and(|f| f == "generated.rs")
            || path.extension().and_then(|e| e.to_str()) != Some("rs")
        {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("hand wrapper source");
        for line in src.lines() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            let mut rest = code;
            while let Some(open) = rest.find('"') {
                let Some(len) = rest[open + 1..].find('"') else {
                    break;
                };
                let lit = &rest[open + 1..open + 1 + len];
                if !lit.is_empty()
                    && lit
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                {
                    hand.insert(lit.to_string());
                }
                rest = &rest[open + 1 + len + 1..];
            }
        }
    }

    let shadowing: Vec<&String> = hand
        .iter()
        .filter(|s| generated.contains(*s) && !KEEPERS.iter().any(|(k, _)| *k == s.as_str()))
        .collect();
    assert!(
        shadowing.is_empty(),
        "hand-written metal wrappers state fixed symbols the generated fns \
         already state; delete the hand form or allowlist it with its \
         reason: {shadowing:?}"
    );

    let stale: Vec<&str> = KEEPERS
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| !hand.contains(*k))
        .collect();
    assert!(
        stale.is_empty(),
        "allowlisted keeper symbols no hand wrapper states any more — \
         drop the rows: {stale:?}"
    );
}
