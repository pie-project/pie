//! The binder over a REAL lowering (retirement plan phase C, first brick).
//!
//! Not a synthetic launch list: `qwen3_0_6b`'s traced decode and prefill
//! forms — the parity-anchored deployment, the same texts the committed
//! `.inc`s are emitted from — lowered over plain rows, then EVERY launch
//! bound through `executor::bind`. What that proves, GPU-free:
//!
//! * every arena offset the lowering assigns is inside the arena it
//!   sized (`arena_bytes` and the offsets agree with each other);
//! * every weight and named value the trace states reaches the resolver
//!   (the map is the only per-family piece left, as designed);
//! * every kernel symbol the lowering emits has a STATED row in the
//!   bridge's tables — DSL or driver-internal — so a generated
//!   `pie_k_*` entry exists for the dispatch half to call. This is the
//!   claim that phase C's dispatch can be written at all.

#![cfg(feature = "_cuda")]

use std::collections::BTreeSet;
use std::ffi::c_void;

use driver_cuda_new::model::executor::{BindRefusal, Frame, Resolver, bind};
use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use model::families::llama_like::forward::llama_like_cuda;
use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

/// Answers every name with a distinct sentinel and records what was asked.
#[derive(Default)]
struct Sentinels {
    weights: BTreeSet<String>,
    named: BTreeSet<ValueId>,
}

impl Resolver for Sentinels {
    fn weight(&mut self, name: &str) -> Option<*const c_void> {
        self.weights.insert(name.to_string());
        Some(0x1000 as *const c_void)
    }
    fn named(&mut self, value: ValueId) -> Option<*mut c_void> {
        self.named.insert(value);
        Some(0x2000 as *mut c_void)
    }
}

fn plan_of(class: FireClass) -> model_compiler::trace::ForwardPlan {
    llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class,
    )
}

fn lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("the live form lowers")
}

/// The qwen3_5 hybrid (E-gate family #1): `Qwen3.5-0.8B-Base`'s facts
/// with the LIVE L40S cuda set — the `emissions.rs` values, not the
/// synthetic fixture (warp-tiled and cached prefill env-gated off,
/// prefill_decode on, dense MLP so the MoE fields are the no-fused-leg
/// zeros).
fn qwen35_live_cuda() -> Qwen35CudaFacts {
    Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        // Attends the whole context; a BF16 checkpoint.
        window_left: Vec::new(),
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
    }
}

fn qwen35_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan =
        qwen3_5_hybrid_cuda(&Qwen35HybridFacts::qwen3_5_0_8b(), &qwen35_live_cuda(), class);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("the hybrid lowers")
}

/// gemma-2 (E-gate family #2): the 9b facts, DECODE class — the only
/// class the family states today.
fn gemma2_lowered(rows: usize) -> Lowered {
    let plan = model::gemma_2::forward::gemma2_cuda(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gemma2 lowers")
}

#[test]
fn every_launch_of_the_gemma2_deployment_binds() {
    let l = gemma2_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gemma2: launch refused: {r:?}"));
        assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
    }
    // gemma2's ARG-level weights are all `scale.*` constants (which bind
    // without the resolver); the tensor weights ride the op join.
    let plan = model::gemma_2::forward::gemma2_cuda(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode,
    );
    let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan, &l);
    assert!(
        (0..l.launches.len()).any(|i| {
            dp.spec(i).weight.as_deref().is_some_and(|w| !w.starts_with("scale."))
        }),
        "a forward that names no tensor weights did not lower the model"
    );
}

#[test]
fn every_lowered_gemma2_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for symbol in &gemma2_lowered(4).kernels {
        if !bridged.contains(symbol.as_str()) {
            unreachable.insert(symbol.clone());
        }
    }
    assert!(
        unreachable.is_empty(),
        "gemma2 kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gemma2_vocabulary() {
    let l = gemma2_lowered(4);
    eprintln!("=== gemma2 decode: {} launches, arena {}", l.launches.len(), l.arena_bytes);
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    for launch in l.launches.iter().take(30) {
        let args = &l.args[launch.args.start as usize..launch.args.end as usize];
        eprintln!("  L {} rows={:?} args={args:?}", l.kernels[launch.kernel as usize], launch.rows);
    }
}

/// gemma-4 (the gemma anchor WITH a cached checkpoint — E2B; gemma-2's
/// 2b-it is gated upstream): both stated classes, the synthetic cuda set.
fn gemma4_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = model::gemma_4::forward::gemma4_cuda(
        &model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e2b(),
        &model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
        class,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gemma4 lowers")
}

#[test]
fn every_launch_of_the_gemma4_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = gemma4_lowered(class, rows);
        assert!(!l.launches.is_empty());
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("gemma4 {class:?}: launch refused: {r:?}"));
            assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
        }
    }
}

/// nemotron_h (E-gate family #3): the synthetic fixture — the family
/// has NO cached real deployment in this environment, so the fixture is
/// the coverage anchor and the real-weight A/B is a recorded blocker.
/// DECODE only: the family states no other class yet.
fn nemotron_lowered(rows: usize) -> Lowered {
    let plan = model::nemotron_h::forward::nemotron_h_cuda(
        &model::nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("nemotron_h lowers")
}

#[test]
fn every_launch_of_the_nemotron_deployment_binds() {
    let l = nemotron_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("nemotron_h: launch refused: {r:?}"));
        assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
    }
}

#[test]
fn every_lowered_nemotron_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for symbol in &nemotron_lowered(4).kernels {
        if !bridged.contains(symbol.as_str()) {
            unreachable.insert(symbol.clone());
        }
    }
    assert!(
        unreachable.is_empty(),
        "nemotron_h kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_nemotron_vocabulary() {
    let l = nemotron_lowered(4);
    eprintln!("=== nemotron_h Decode: {} launches, arena {}", l.launches.len(), l.arena_bytes);
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if k.contains("mamba") || k.contains("zamba") || k.contains("relu2")
            || k.contains("sigmoid_bias") || k.contains("gemv")
            || k.contains("weighted_sum") || k.contains("conv")
        {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
        }
    }
}

#[test]
fn every_lowered_gemma4_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &gemma4_lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "gemma4 kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gemma4_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = gemma4_lowered(class, rows);
        eprintln!("=== gemma4 {class:?}: {} launches, arena {}", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        for launch in &l.launches {
            let k = &l.kernels[launch.kernel as usize];
            if k.contains("packed") || k.contains("residual_add") || k.contains("rounded")
                || k.contains("naive") || k.contains("attention_flashinfer_prefill")
                || k.contains("transpose") || k.contains("no_scale") || k.contains("geglu")
                || k.contains("rope_partial") || k.contains("split_qkv")
                || k.contains("scalar_mul")
            {
                let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
            }
        }
    }
}

/// Every symbol the bridge can dispatch: the DSL tables plus the
/// driver-internal one.
/// Lowered symbol → the bridge row the executor actually binds it to.
///
/// One entry, and it is the weight-representation axis's leftover. A
/// DENSE weight is the one representation `MatW::gemm_symbol` declines
/// to name — there is nothing to choose — so it records the semantic
/// `OpKind::Matmul`, and the CUDA reading of that semantic is spelled
/// `gemm::act_x_w`, the header's ROUTING entry point. Every other
/// representation states its own symbol and lands on its own row.
///
/// The routing entry point takes a `WeightView` by value, and that
/// descriptor deliberately no longer crosses this ABI: a driver that
/// built one was choosing a kernel from a per-layer struct no statement
/// mentioned. So the dense arm binds `gemm::act_x_wt_bf16` instead,
/// which `gemm.hpp` defines as `act_x_w` with `WeightView::raw(W,
/// BF16)` — the one view that arm ever built, now assembled inside the
/// launcher.
///
/// It is a rename at the ABI, not a missing row, and it disappears the
/// day the CUDA reading of a dense `Matmul` is the bf16 entry point's
/// own name.
const RENAMED_AT_THE_ABI: &[(&str, &str)] = &[("gemm::act_x_w", "gemm::act_x_wt_bf16")];

fn bridged_symbols() -> BTreeSet<&'static str> {
    let rows: BTreeSet<&'static str> = kernels_cuda::KERNELS
        .iter()
        .chain(kernels_cuda::driver_internal::DRIVER_KERNELS)
        .filter(|k| !k.operands.is_empty())
        .map(|k| k.symbol)
        .collect();
    let mut reachable = rows.clone();
    for (lowered, row) in RENAMED_AT_THE_ABI {
        // The exception buys nothing if its TARGET is imaginary: a
        // rename is only reachable when the row it renames to is.
        assert!(
            rows.contains(row),
            "`{lowered}` is bound to `{row}`, which has no bridge row"
        );
        reachable.insert(lowered);
    }
    reachable
}

#[test]
fn every_launch_of_the_anchor_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = lowered(class, rows);
        assert!(!l.launches.is_empty(), "{class:?} lowered to nothing");

        // The arena the frame would allocate — the binder only addresses
        // it, so a dangling sentinel base is fine off-device.
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();

        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("{class:?}: launch refused: {r:?}"));
            assert!(!bound.kernel.is_empty());
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize,
                "every stated operand binds"
            );
        }
        assert!(
            !resolver.weights.is_empty(),
            "{class:?}: a forward that names no weights did not lower the model"
        );
    }
}

/// The hybrid's bind claim, GPU-free: every launch of the qwen3_5
/// deployment's decode and prefill texts binds — arena offsets inside
/// the sized arena, every weight and ctx value reaching the resolver.
/// The fire classes the shell fires today; the service classes
/// (StateOnly, CommitAdvance) join when spec-decode does.
#[test]
fn every_launch_of_the_hybrid_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = qwen35_lowered(class, rows);
        assert!(!l.launches.is_empty(), "{class:?} lowered to nothing");
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("hybrid {class:?}: launch refused: {r:?}"));
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize,
                "every stated operand binds"
            );
        }
        assert!(!resolver.weights.is_empty());
    }
}

/// The hybrid's dispatchability claim — same as the anchor's, separate
/// test so a missing row names the family that needs it.
#[test]
fn every_lowered_hybrid_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &qwen35_lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "hybrid kernels with no stated bridge row: {unreachable:?}"
    );
}

/// The dispatchability claim: nothing lowers to a kernel the bridge
/// cannot reach. A symbol failing here is not a test problem — it is a
/// row that needs writing (DSL family or driver-internal) BEFORE the
/// dispatch half meets it.
#[test]
fn every_lowered_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "lowered kernels with no stated bridge row: {unreachable:?}"
    );
}

/// The refusals refuse: an arena smaller than the lowering sized is
/// caught at the offending offset, and an unknown weight is named.
#[test]
fn the_binder_diagnoses_drift_rather_than_addressing_through_it() {
    let l = lowered(FireClass::Decode, 4);

    let starved = Frame { arena: 0x10000 as *mut c_void, arena_bytes: 1 };
    let mut resolver = Sentinels::default();
    let refusal = l
        .launches
        .iter()
        .find_map(|launch| bind(&l, launch, starved, &mut resolver).err());
    assert!(
        matches!(refusal, Some(BindRefusal::ArenaOutOfBounds { arena_bytes: 1, .. })),
        "a one-byte arena must refuse: {refusal:?}"
    );

    struct NoWeights;
    impl Resolver for NoWeights {
        fn weight(&mut self, _: &str) -> Option<*const c_void> {
            None
        }
        fn named(&mut self, _: ValueId) -> Option<*mut c_void> {
            Some(0x2000 as *mut c_void)
        }
    }
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let refusal = l
        .launches
        .iter()
        .find_map(|launch| bind(&l, launch, frame, &mut NoWeights).err());
    assert!(
        matches!(refusal, Some(BindRefusal::UnknownWeight(_))),
        "a weightless store must be diagnosed by NAME: {refusal:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_all_deployment_vocabularies() {
    // Each deployment's OWN cuda facts — the emissions fixtures' values.
    // Dense BF16, single GPU, whole context — the four emission targets'
    // shared tail. `head_dim_kernel` is the one that is NOT shared: phi3
    // pads 96 to 128 and says so, and the other three take the 0 that
    // reads as "the kernel head_dim is the model's".
    let tail = LlamaLikeCudaFacts {
        head_dim_kernel: 0,
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        tp_size: 1,
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
        xqa_decode: false, decode_fused_post: false, rope_table: true,
        force_prefill_path: false, head_dim_padded: false, gate_up_fused: true,
    };
    let deployments: Vec<(&str, LlamaLikeFacts, LlamaLikeCudaFacts)> = vec![
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), LlamaLikeCudaFacts {
            decode_fused_post: true, ..tail.clone()
        }),
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b(), LlamaLikeCudaFacts {
            force_prefill_path: true, ..tail.clone()
        }),
        ("mistral_7b_v03", LlamaLikeFacts::mistral_7b_v03(), LlamaLikeCudaFacts {
            decode_fused_post: true, ..tail.clone()
        }),
        ("phi3_mini", LlamaLikeFacts::phi3_mini(), LlamaLikeCudaFacts {
            head_dim_padded: true, head_dim_kernel: 128, ..tail.clone()
        }),
    ];
    let bridged = bridged_symbols();
    for (name, facts, cuda) in &deployments {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_cuda(facts, cuda, class);
            let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
            let l = lower(&plan, &rows, Fire { captures_across_splits: false })
                .expect("lowers");
            let missing: Vec<&String> = l
                .kernels
                .iter()
                .filter(|k| !bridged.contains(k.as_str()))
                .collect();
            let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
            let mut r = Sentinels::default();
            for launch in &l.launches {
                let _ = bind(&l, launch, frame, &mut r);
            }
            let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan, &l);
            for i in 0..l.launches.len() {
                if let Some(w) = &dp.spec(i).weight {
                    r.weights.insert(w.clone());
                }
            }
            let mut names: Vec<_> = r
                .weights
                .iter()
                .filter(|n| !n.contains("layer.") || n.contains("layer.0."))
                .collect();
            names.sort();
            eprintln!(
                "{name} {class:?}: kernels={:?}\n  MISSING_ROWS={missing:?}\n  weights0={names:?}",
                l.kernels
            );
            for launch in &l.launches {
                let k = &l.kernels[launch.kernel as usize];
                if k == "rope::rope_bf16" || k == "norm::residual_add_bf16" || k == "norm::add_bias_bf16" {
                    let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                    eprintln!("  L {k} rows={:?} args={args:?}", launch.rows);
                }
            }
        }
    }
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_hybrid_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = qwen35_lowered(class, rows);
        eprintln!("=== hybrid {class:?}: {} launches, arena {} bytes", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        for launch in l.launches.iter().take(40) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L kernel={} rows={:?} args={args:?}",
                l.kernels[launch.kernel as usize], launch.rows
            );
        }
    }
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_anchor_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = lowered(class, rows);
        eprintln!("=== {class:?}: {} launches, arena {} bytes", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        {
            let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
            let mut r = Sentinels::default();
            for (i, launch) in l.launches.iter().enumerate() {
                let _ = bind(&l, launch, frame, &mut r);
                let _ = i;
            }
            let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan_of(class), &l);
            for i in 0..l.launches.len() {
                if let Some(w) = &dp.spec(i).weight {
                    r.weights.insert(w.clone());
                }
            }
            let mut names: Vec<_> = r.weights.iter().collect();
            names.sort();
            let head: Vec<_> = names.iter().filter(|n| !n.contains("layer.") || n.contains("layer.0.") || n.contains("layer.27.")).collect();
            eprintln!("  weights: {head:?}");
        }
        for launch in l.launches.iter().take(14) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L kernel={} rows={:?} args={args:?}",
                l.kernels[launch.kernel as usize], launch.rows
            );
        }
    }
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_lora_vocabulary() {
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, lora: true, ..Row::default() }; 2];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    eprintln!("=== lora decode: {} launches", l.launches.len());
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if k.contains("lora") {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
        }
    }
}

/// gemma3n (E-gate family #7): the synthetic fixture — the family has no
/// cached deployment here, so the fixture is the coverage anchor and the
/// real-weight A/B is a recorded blocker. AltUp's rank-K residual is the
/// new vocabulary: predict from the active stream, run the body on the
/// prediction, correct every stream from the result.
fn gemma3n_lowered(rows: usize) -> Lowered {
    let plan = model::gemma3n::forward::gemma3n_cuda(
        &model::gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gemma3n lowers")
}

#[test]
fn every_launch_of_the_gemma3n_deployment_binds() {
    let l = gemma3n_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gemma3n: launch refused: {r:?}"));
        assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
    }
}

#[test]
fn every_lowered_gemma3n_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for symbol in &gemma3n_lowered(4).kernels {
        if !bridged.contains(symbol.as_str()) {
            unreachable.insert(symbol.clone());
        }
    }
    assert!(
        unreachable.is_empty(),
        "gemma3n kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gemma3n_vocabulary() {
    let l = gemma3n_lowered(4);
    eprintln!("=== gemma3n Decode: {} launches, arena {}", l.launches.len(), l.arena_bytes);
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    let mut seen = BTreeSet::new();
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if (k.contains("altup") || k.contains("hc_") || k.contains("tanh")
            || k.contains("gaussian") || k.contains("rms") || k.contains("mean_streams")
            || k.contains("magnitude"))
            && seen.insert(k.clone())
        {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
        }
    }
}

/// gpt-oss / mixtral (E-gate family #8): the Mixtral plan family, one
/// declaration for both model types. A 20b checkpoint IS cached here, so
/// this family's facts are the real ones.
fn gpt_oss_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = model::gpt_oss::forward::gpt_oss_cuda(
        &model::gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b(),
        &model::gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic(),
        class,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gpt_oss lowers")
}

#[test]
fn every_launch_of_the_gpt_oss_deployment_binds() {
    let l = gpt_oss_lowered(FireClass::Decode, 4);
    assert!(!l.launches.is_empty());
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gpt_oss: launch refused: {r:?}"));
        assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
    }
}

#[test]
fn every_lowered_gpt_oss_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for symbol in &gpt_oss_lowered(FireClass::Decode, 4).kernels {
        if !bridged.contains(symbol.as_str()) {
            unreachable.insert(symbol.clone());
        }
    }
    assert!(
        unreachable.is_empty(),
        "gpt_oss kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gpt_oss_vocabulary() {
    let l = gpt_oss_lowered(FireClass::Decode, 4);
    eprintln!("=== gpt_oss Decode: {} launches, arena {}", l.launches.len(), l.arena_bytes);
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    let mut seen = BTreeSet::new();
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if seen.insert(k.clone()) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
        }
    }
}

/// The four remaining plan families (E-gate #9–12), each lowered from
/// its own fixture: glm5 (permutation MoE + the DSA indexer), kimi_k2
/// and kimi_k3 (MLA, and KDA linear attention on k3), deepseek_v4 (MLA,
/// fp8 experts, hashed routing). One helper per family, and one pair of
/// claims over all four — every launch binds, every kernel has a stated
/// bridge row — which is what makes the arms writable at all.
fn glm5_lowered(rows: usize) -> Lowered {
    let plan = model::glm5::forward::glm5_cuda(
        &model::glm5::forward::facts::Glm5Facts::glm5_106b_a12b(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("glm5 lowers")
}

fn kimi_k2_lowered(rows: usize) -> Lowered {
    let plan = model::kimi_k2::forward::kimi_cuda(
        &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
        &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("kimi_k2 lowers")
}

fn kimi_k3_lowered(rows: usize) -> Lowered {
    let plan = model::kimi_k3::forward::kimi_k3_cuda(
        &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("kimi_k3 lowers")
}

fn dsv4_lowered(rows: usize) -> Lowered {
    let plan = model::deepseek_v4::forward::dsv4_cuda(
        &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("dsv4 lowers")
}

#[test]
fn every_launch_of_the_remaining_families_binds() {
    for (name, l) in [
        ("glm5", glm5_lowered(4)),
        ("kimi_k2", kimi_k2_lowered(4)),
        ("kimi_k3", kimi_k3_lowered(4)),
        ("deepseek_v4", dsv4_lowered(4)),
    ] {
        assert!(!l.launches.is_empty(), "{name} lowered to nothing");
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("{name}: launch refused: {r:?}"));
            assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
        }
    }
}

#[test]
fn every_lowered_kernel_of_the_remaining_families_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for l in [glm5_lowered(4), kimi_k2_lowered(4), kimi_k3_lowered(4), dsv4_lowered(4)] {
        for symbol in &l.kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "kernels with no stated bridge row: {unreachable:?}"
    );
}

/// Every symbol any family lowers to has an ARM, not merely a row.
///
/// The bridge-row claims above answer "can the launcher be CALLED"; this
/// one answers "does `dispatch` know what to call". They are different
/// questions and the gap between them cost this branch a GPU cycle per
/// instance during the `origin/rewrite` merges: a row exists, the crate
/// compiles, every non-GPU test passes, and the first thing that notices
/// is a fire refusing with `NoArm`.
///
/// It reads the arms out of the executor's SOURCE, which is worth being
/// upfront about. `dispatch` is one `match` on `&str` and there is no
/// value to enumerate; calling it per symbol would need a device and a
/// plausible operand for each. The source scan asks a narrower question
/// than the match answers — an arm may still refuse on arity, and does —
/// but the failure it catches is the one that keeps happening: a symbol
/// nothing matches at all.
#[test]
fn every_lowered_symbol_has_an_arm() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model/executor.rs"),
    )
    .expect("the executor's source");

    // Arms are `"sym" =>` or `"a" | "b" =>`, so collect every quoted
    // `family::symbol` on a line that ends in `=>`. Guard the shape: a
    // rewrite that changes how arms are spelled must not leave this
    // passing vacuously.
    let mut armed = BTreeSet::new();
    for line in src.lines() {
        let line = line.trim();
        if !line.ends_with("=>") && !line.ends_with("=> {") {
            continue;
        }
        for part in line.split('"').skip(1).step_by(2) {
            if part.contains("::") && !part.contains(char::is_whitespace) {
                armed.insert(part.to_string());
            }
        }
    }
    assert!(
        armed.len() > 60,
        "the arm scan found {} arms, so its shape assumption broke",
        armed.len()
    );

    // GENERATED branches count as armed, because they are. `dispatch`
    // runs them first and the hand-written match is the fallthrough, so a
    // symbol with a generated branch and no arm is served — and this test
    // said otherwise for exactly as long as the generator has existed.
    //
    // Read off `emit_rust_dispatch` rather than off the build's output:
    // the generator is the claim, and reading `OUT_DIR` would make this
    // pass or fail on whether a build script had run.
    let generated = kernels_cuda::abi::emit_rust_dispatch(&[
        kernels_cuda::attn::KERNELS,
        kernels_cuda::rope::KERNELS,
        kernels_cuda::norm::KERNELS,
        kernels_cuda::mlp::KERNELS,
        kernels_cuda::gemm::KERNELS,
        kernels_cuda::moe::KERNELS,
        kernels_cuda::ssm::KERNELS,
        kernels_cuda::quant::KERNELS,
        kernels_cuda::layout::KERNELS,
        kernels_cuda::sample::KERNELS,
        kernels_cuda::driver_internal::DRIVER_KERNELS,
    ]);
    for line in generated.lines() {
        // A branch opens `"symbol" if ... => unsafe {`.
        if !line.trim_start().starts_with('"') || !line.contains("=> unsafe") {
            continue;
        }
        if let Some(sym) = line.split('"').nth(1)
            && sym.contains("::")
        {
            armed.insert(sym.to_string());
        }
    }

    let mut every: BTreeSet<String> = BTreeSet::new();
    for l in [
        lowered(FireClass::Decode, 4),
        lowered(FireClass::Prefill, 7),
        qwen35_lowered(FireClass::Decode, 4),
        qwen35_lowered(FireClass::Prefill, 7),
        glm5_lowered(4),
        kimi_k2_lowered(4),
        kimi_k3_lowered(4),
        dsv4_lowered(4),
    ] {
        every.extend(l.kernels.iter().cloned());
    }

    // `gemm::act_x_w` is the one rename at the ABI; its arm is spelled
    // with the symbol the lowering states, so it is already in `armed`.
    //
    // The set below is the PORT'S REMAINING WORK, not an exemption. Three
    // subsystems, none of them started, each tracked: MLA and its latent
    // cache (glm5, kimi_k2), kimi_k3's KDA, and deepseek_v4's DSA indexer
    // with the hyper-connection residual. They lower — the declarations
    // are written and their rows are stated — and no arm executes them.
    //
    // It is a CLOSED set on purpose, and that is what this test is for:
    // the list shrinks as arms land, and a symbol that joins it because
    // some other change stopped matching an arm fails here instead of on
    // a GPU. Sorted, so the diff when one leaves is one line.
    #[rustfmt::skip]
    const UNARMED: &[&str] = &[
        // ── MLA: the latent-cache attention (glm5, kimi_k2) ──────────
        "attn::attention_compressed_paged_bf16",
        "attn::combine_attn_outputs_bf16",
        "attn::dispatch_attention_mla_bf16",
        "attn::kimi_split_kv_a_norm_bf16",
        "attn::kimi_split_q_b_bf16",
        "attn::lse_log2_to_ln",
        "attn::mla_prepare_bf16",
        "attn::write_mla_to_pages",
        "gemm::mla_absorb_latent_to_v_bf16",
        "gemm::mla_absorb_q_to_latent_bf16",
        "rope::rope_partial_last_bf16",
        // ── deepseek_v4: the DSA indexer and hyper-connections ────────
        "attn::dsa_index_knorm_rope_bf16",
        "attn::dsa_index_q_rope_bf16",
        "attn::dsa_index_topk_mask",
        "attn::dsv4_boundary_meta_decode",
        "attn::dsv4_compress_gather_paged_bf16",
        "attn::dsv4_store_comp_entries_bf16",
        "norm::hc_head_postprocess_bf16",
        "norm::hc_post_bf16",
        "norm::hc_pre_postprocess_bf16",
        "norm::hc_rmsnorm_to_f32",
        // ── kimi_k3: KDA, the per-key-channel delta rule ──────────────
        //
        // `ssm::bf16_to_fp32` left this list without an arm being
        // written: its row stated its sources, so a branch generates.
        // That is the shape every line here is meant to leave by.
        "ssm::kda_gate_beta_bf16",
        "ssm::kda_o_norm_gated_bf16",
        "ssm::kda_recurrent_step_batched",
        "ssm::l2norm_scale_bf16_to_fp32",
    ];

    // Compared as SETS: the list above is grouped by subsystem because
    // that is how a reader decides whether a line is theirs, and sorting
    // it would scatter the three groups into one alphabetical run.
    let unarmed: BTreeSet<&str> = every.difference(&armed).map(String::as_str).collect();
    let expected: BTreeSet<&str> = UNARMED.iter().copied().collect();
    assert_eq!(
        unarmed.len(),
        UNARMED.len(),
        "a line in UNARMED is duplicated"
    );
    assert_eq!(
        unarmed, expected,
        "the unarmed set moved. A symbol that LEFT means an arm landed — \
         delete its line. A symbol that JOINED lowers with nothing to \
         execute it, and a fire will refuse with NoArm."
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_remaining_families_vocabulary() {
    for (name, l) in [
        ("glm5", glm5_lowered(4)),
        ("kimi_k2", kimi_k2_lowered(4)),
        ("kimi_k3", kimi_k3_lowered(4)),
        ("deepseek_v4", dsv4_lowered(4)),
    ] {
        eprintln!("=== {name}: {} launches, arena {}", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        let mut seen = BTreeSet::new();
        for launch in &l.launches {
            let k = &l.kernels[launch.kernel as usize];
            if seen.insert(k.clone()) {
                let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                eprintln!("  L {k} rows={:?} layers={:?} args={args:?}",
                    launch.rows, launch.layers);
            }
        }
    }
}

/// The pair-form activations get their `up` back from the JOIN.
///
/// `swiglu` / `swiglu_clamp` / `situ` state one operand where their
/// launcher takes gate AND up: the DSL records one input, the C++ arm
/// reads `ws.up` from its workspace, and the declarations drop the
/// second projection outright. The join's pre-pass is what makes the arm
/// writable, so this pins it — every pair-form launch in every family
/// that states one must carry exactly one aux slot. A family that grows
/// a differently-named `up` projection fails here rather than at a
/// silent wrong number on device.
#[test]
fn every_pair_form_activation_recovers_its_up_projection() {
    use driver_cuda_new::model::executor::DispatchPlan;

    let cases: Vec<(&str, model_compiler::trace::ForwardPlan, Lowered)> = vec![
        (
            "kimi_k2",
            model::kimi_k2::forward::kimi_cuda(
                &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
                &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
                FireClass::Decode,
            ),
            kimi_k2_lowered(4),
        ),
        (
            "kimi_k3",
            model::kimi_k3::forward::kimi_k3_cuda(
                &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
                FireClass::Decode,
            ),
            kimi_k3_lowered(4),
        ),
        (
            "glm5",
            model::glm5::forward::glm5_cuda(
                &model::glm5::forward::facts::Glm5Facts::glm5_106b_a12b(),
                FireClass::Decode,
            ),
            glm5_lowered(4),
        ),
        (
            "deepseek_v4",
            model::deepseek_v4::forward::dsv4_cuda(
                &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
                FireClass::Decode,
            ),
            dsv4_lowered(4),
        ),
    ];
    let mut seen_any = false;
    for (name, plan, l) in &cases {
        let dp = DispatchPlan::new(plan, l);
        for (i, launch) in l.launches.iter().enumerate() {
            let k = &l.kernels[launch.kernel as usize];
            if matches!(
                k.as_str(),
                "mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16" | "mlp::situ_bf16"
            ) {
                seen_any = true;
                assert_eq!(
                    dp.spec(i).aux.len(),
                    1,
                    "{name} launch {i} ({k}, layer {:?}) has no `up` from the join",
                    launch.layers
                );
            }
        }
    }
    assert!(seen_any, "no family stated a pair-form activation — the claim is vacuous");
}
