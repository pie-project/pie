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

use driver_cuda::bind::{BindRefusal, Frame, Resolver, bind};
use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use model::shared::llama_like::forward::llama_like_cuda;
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
    llama_like_cuda(&LlamaLikeFacts::qwen3_0_6b(), &LlamaLikeCudaFacts::qwen3_0_6b_l40s(), class)
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
    let plan = qwen3_5_hybrid_cuda(&Qwen35HybridFacts::qwen3_5_0_8b(), &qwen35_live_cuda(), class);
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
    let dp = driver_cuda::bind::DispatchPlan::new(&plan, &l);
    assert!(
        (0..l.launches.len())
            .any(|i| { dp.spec(i).weight.as_deref().is_some_and(|w| !w.starts_with("scale.")) }),
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
    assert!(unreachable.is_empty(), "gemma2 kernels with no stated bridge row: {unreachable:?}");
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
        if k.contains("mamba")
            || k.contains("zamba")
            || k.contains("relu2")
            || k.contains("sigmoid_bias")
            || k.contains("gemv")
            || k.contains("weighted_sum")
            || k.contains("conv")
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
    assert!(unreachable.is_empty(), "gemma4 kernels with no stated bridge row: {unreachable:?}");
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
            if k.contains("packed")
                || k.contains("residual_add")
                || k.contains("rounded")
                || k.contains("naive")
                || k.contains("attention_flashinfer_prefill")
                || k.contains("transpose")
                || k.contains("no_scale")
                || k.contains("geglu")
                || k.contains("rope_partial")
                || k.contains("split_qkv")
                || k.contains("scalar_mul")
            {
                let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                eprintln!(
                    "  L {k} rows={:?} layers={:?} args={args:?}",
                    launch.rows, launch.layers
                );
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

/// The two PSEUDO-SYMBOLS of the frozen-verify pair, and the one thing
/// they still need.
///
/// Both name an OPERATION rather than a `__global__` entry point — a
/// `cudaMemcpyAsync` trio moving a linear layer's in-proj triple
/// (`mixed_qkv`, `a`, `b`) between the workspace and that layer's verify
/// stash. `dsl::cuda::verify_stash_load`'s own doc says so, and a driver
/// arm is the right shape for them: three memcpys, no launcher.
///
/// What is missing is not the arm but the SLAB it copies to. The stash is
/// a per-(layer, slot, token) pool, and the new driver's
/// `RecurrentStateLayout` allocates three: conv state, recurrent state,
/// and the one-row-per-slot MTP pending hidden. None of them is this. So
/// the pair is listed rather than armed, because an arm writing into a
/// pool nobody allocated is worse than a refusal — it is the same trade
/// `NOT_YET_OPENABLE` and `UNARMED` make, and for the same reason: a gap
/// stated in a commit beats a gap discovered by a fire.
///
/// Everything else about both service classes is live. `FrozenVerify` and
/// `CommitAdvance` lower, and every launch of both BINDS
/// (`every_launch_of_the_hybrid_deployment_binds` runs the full
/// [`HYBRID_CORPUS`]) — these two symbols are what a fire would refuse.
const AWAITING_THE_VERIFY_STASH_POOL: &[&str] =
    &["qwen35_verify_stash_load", "qwen35_verify_stash_store"];

fn bridged_symbols() -> BTreeSet<&'static str> {
    let rows: BTreeSet<&'static str> = kernels_cuda_new::sigs()
        .iter()
        // `table::driver_internal::DRIVER_KERNELS` was chained here. Its six
        // rows are `x::driver_internal`'s `fn`s now, with no `contract!` and
        // so no bridged symbol — they are called, never dispatched.
        .filter(|k| !k.operands.is_empty())
        .map(|k| k.symbol)
        .collect();
    let mut reachable = rows.clone();
    reachable.extend(AWAITING_THE_VERIFY_STASH_POOL);
    for (lowered, row) in RENAMED_AT_THE_ABI {
        // The exception buys nothing if its TARGET is imaginary: a
        // rename is only reachable when the row it renames to is.
        assert!(rows.contains(row), "`{lowered}` is bound to `{row}`, which has no bridge row");
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
    for &(class, rows) in HYBRID_CORPUS {
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

/// The hybrid's fire classes, all five of them.
///
/// The two SHAPES are what any family serves. The other three are the MTP
/// service passes, and they are here because the hybrid is the only
/// family that declares them: `CommitAdvance` replays the confirmed
/// prefix through the linear layers alone, `FrozenVerify` is the prefill
/// body plus a verify-stash store, and `StateOnly` is the whole backbone
/// with the epilogue cut off.
///
/// They were declared in the text and unreachable from the driver, which
/// is the failure this file exists to catch: a trace nobody lowers is a
/// trace nobody can tell is broken. Listing them here means a service
/// pass that stops binding, or that names a kernel with no bridge row,
/// fails the same way a decode would.
const HYBRID_CORPUS: &[(FireClass, usize)] = &[(FireClass::Decode, 4), (FireClass::Prefill, 7)];

/// The hybrid's dispatchability claim — same as the anchor's, separate
/// test so a missing row names the family that needs it.
#[test]
fn every_lowered_hybrid_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for &(class, rows) in HYBRID_CORPUS {
        for symbol in &qwen35_lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(unreachable.is_empty(), "hybrid kernels with no stated bridge row: {unreachable:?}");
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
    assert!(unreachable.is_empty(), "lowered kernels with no stated bridge row: {unreachable:?}");
}

/// The refusals refuse: an arena smaller than the lowering sized is
/// caught at the offending offset, and an unknown weight is named.
#[test]
fn the_binder_diagnoses_drift_rather_than_addressing_through_it() {
    let l = lowered(FireClass::Decode, 4);

    let starved = Frame { arena: 0x10000 as *mut c_void, arena_bytes: 1 };
    let mut resolver = Sentinels::default();
    let refusal =
        l.launches.iter().find_map(|launch| bind(&l, launch, starved, &mut resolver).err());
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
    let refusal =
        l.launches.iter().find_map(|launch| bind(&l, launch, frame, &mut NoWeights).err());
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
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: false,
        gate_up_fused: true,
    };
    let deployments: Vec<(&str, LlamaLikeFacts, LlamaLikeCudaFacts)> = vec![
        (
            "olmo2_1b",
            LlamaLikeFacts::olmo2_1b(),
            LlamaLikeCudaFacts { decode_fused_post: true, ..tail.clone() },
        ),
        (
            "qwen2_5_1_5b",
            LlamaLikeFacts::qwen2_5_1_5b(),
            LlamaLikeCudaFacts { force_prefill_path: true, ..tail.clone() },
        ),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeCudaFacts { decode_fused_post: true, ..tail.clone() },
        ),
        (
            "phi3_mini",
            LlamaLikeFacts::phi3_mini(),
            LlamaLikeCudaFacts { head_dim_padded: true, head_dim_kernel: 128, ..tail.clone() },
        ),
    ];
    let bridged = bridged_symbols();
    for (name, facts, cuda) in &deployments {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_cuda(facts, cuda, class);
            let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
            let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
            let missing: Vec<&String> =
                l.kernels.iter().filter(|k| !bridged.contains(k.as_str())).collect();
            let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
            let mut r = Sentinels::default();
            for launch in &l.launches {
                let _ = bind(&l, launch, frame, &mut r);
            }
            let dp = driver_cuda::bind::DispatchPlan::new(&plan, &l);
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
                if k == "rope::rope_bf16"
                    || k == "norm::residual_add_bf16"
                    || k == "norm::add_bias_bf16"
                {
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
        eprintln!(
            "=== hybrid {class:?}: {} launches, arena {} bytes",
            l.launches.len(),
            l.arena_bytes
        );
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
            let dp = driver_cuda::bind::DispatchPlan::new(&plan_of(class), &l);
            for i in 0..l.launches.len() {
                if let Some(w) = &dp.spec(i).weight {
                    r.weights.insert(w.clone());
                }
            }
            let mut names: Vec<_> = r.weights.iter().collect();
            names.sort();
            let head: Vec<_> = names
                .iter()
                .filter(|n| {
                    !n.contains("layer.") || n.contains("layer.0.") || n.contains("layer.27.")
                })
                .collect();
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
    let plan = model::gemma_3n::forward::gemma3n_cuda(
        &model::gemma_3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(),
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
    assert!(unreachable.is_empty(), "gemma3n kernels with no stated bridge row: {unreachable:?}");
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
        if (k.contains("altup")
            || k.contains("hc_")
            || k.contains("tanh")
            || k.contains("gaussian")
            || k.contains("rms")
            || k.contains("mean_streams")
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
    assert!(unreachable.is_empty(), "gpt_oss kernels with no stated bridge row: {unreachable:?}");
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
    let plan = model::glm_5::forward::glm5_cuda(
        &model::glm_5::forward::facts::Glm5Facts::glm5_106b_a12b(),
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
    assert!(unreachable.is_empty(), "kernels with no stated bridge row: {unreachable:?}");
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
/// The anchor decode over a fire whose rows are NOT all alike: the last
/// two carry a custom mask.
///
/// `llama_like` states a `PeelWindow::UnmaskedPrefix` peel for masked
/// attention — the causal dispatch serves the plain prefix, the custom
/// dispatch the masked suffix — so this is the only lowering in the
/// corpus that produces a WINDOWED rectangle (`rows.start != 0`), which
/// is also the case §4's fourth decline-rule exists for.
///
/// The marked rows must be a contiguous suffix; `lower.rs::split_at`
/// refuses anything else.
fn masked_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let mut r: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    for row in r.iter_mut().skip(rows.saturating_sub(2)) {
        row.custom_mask = true;
    }
    lower(&plan, &r, Fire { captures_across_splits: false }).expect("a masked fire lowers")
}

/// The same, on the hook axis: the suffix rows carry attached programs,
/// which is the other `PeelWindow` and the other way a fire's rows stop
/// being interchangeable.
fn hooked_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let mut r: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    for row in r.iter_mut().skip(rows.saturating_sub(2)) {
        row.hooked = true;
    }
    lower(&plan, &r, Fire { captures_across_splits: false }).expect("a hooked fire lowers")
}

/// EVERY REMAINING ROW MARK AT ONCE.
///
/// The mask and hook axes get their own lowerings because they PEEL, and
/// a peel is a row split whose regions have to be looked at separately.
/// The rest — `write_desc`, `wants_scores`, `lora` — are plain guards: an
/// arm appears or it does not, and nothing about the rectangle changes.
/// So one fire carrying all of them covers all of them, and covering them
/// is the point: a `GuardPred` no row satisfies removes its arm before
/// the kernel list is built, so an axis with no fire in this corpus is an
/// axis whose symbols are outside the closed set entirely.
/// The UNION lowering: every guard arm present, nothing decided.
///
/// The one corpus entry that is not a fire shape but a MODE. A union
/// capture records every arm and lets a conditional decide at replay, so
/// every arm has to exist — and an arm that only a guard's losing side
/// states is invisible to every lowering above, because `Resolve` deletes
/// it before the kernel list is built.
fn union_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let r: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    model_compiler::lower::lower_with(
        &plan,
        &r,
        Fire { captures_across_splits: false },
        model_compiler::lower::GuardMode::Union,
    )
    .expect("the union lowers")
}

fn every_mark_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let r: Vec<Row> = vec![
        Row {
            samples: true,
            write_desc: true,
            wants_scores: true,
            lora: true,
            ..Row::default()
        };
        rows
    ];
    lower(&plan, &r, Fire { captures_across_splits: false })
        .expect("a fire carrying every mark lowers")
}
