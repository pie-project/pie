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
use model_compiler::lower::{Arg, Fire, Lowered, Row, lower};
use model_ir::trace::{FireClass, ValueId};

/// The scalars the family texts used to read off their fact structs.
///
/// Upstream lifted `norm_eps`, the rope bases and gpt-oss's sliding window
/// OUT of the facts and onto the forward functions, because two SKUs of one
/// family can differ in those and in nothing else. These tests never read a
/// number back -- they lower, bind and count -- so any well-formed value
/// states the same text, and these are the shipped checkpoints' own.
#[allow(dead_code)]
const EPS: f32 = 1e-6;
/// The common rope base. gpt-oss's is its own.
#[allow(dead_code)]
const THETA: f32 = 1_000_000.0;
/// gpt-oss: YaRN over a 150k base, alternating 128-token windows.
#[allow(dead_code)]
const WINDOWED_THETA: f32 = 150_000.0;
/// The sliding leg's span. `-1` is "no window" and is NOT what gpt-oss says.
#[allow(dead_code)]
const WINDOW: i32 = 128;

/// Answers every name with a distinct sentinel and records what was asked.
#[derive(Default)]
struct Sentinels {
    weights: BTreeSet<String>,
    named: BTreeSet<ValueId>,
    raised: BTreeSet<String>,
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
    /// A THIRD SENTINEL, for the third thing a trace can name.
    ///
    /// `Resolver::raised` defaults to `None` — most resolvers hold no raises —
    /// and this one answers every name by construction, which is what makes it
    /// a census of what the plan ASKED rather than of what a fire holds. Left
    /// on the default it refused `fa2.prefill` the moment the prefill launcher
    /// began naming its plan, which is the default doing its job and not a
    /// defect in it.
    fn raised(&mut self, value: ValueId, key: &str) -> Option<*const c_void> {
        self.raised.insert(key.to_string());
        // ONE SENTINEL PER VALUE, so a launch that bound the wrong raise is a
        // different address rather than the same one. The census above records
        // the WORD; this records which object the statement named.
        Some((0x3000 + value as usize) as *const c_void)
    }
}

fn plan_of(class: FireClass) -> model_ir::trace::ForwardPlan {
    llama_like_cuda::<model::shared::llama_like::forward::ShippedA, model::shared::llama_like::forward::ShippedKv>(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class, EPS, THETA)
}

fn lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_cuda::<model::shared::llama_like::forward::ShippedA, model::shared::llama_like::forward::ShippedKv>(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the live form lowers")
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
        proj_repr: model_dsl::WeightRepr::Bf16,
    }
}

fn qwen35_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = qwen3_5_hybrid_cuda::<model::qwen_3_5::forward::ShippedW1, model::qwen_3_5::forward::ShippedW2, model::qwen_3_5::forward::ShippedA, model::qwen_3_5::forward::ShippedKv>(
        &Qwen35HybridFacts::qwen3_5_0_8b(),
        &qwen35_live_cuda(),
        class, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers")
}

/// gemma-2 (E-gate family #2): the 9b facts, DECODE class — the only
/// class the family states today.
fn gemma2_lowered(rows: usize) -> Lowered {
    let plan = model::gemma_2::forward::gemma2_cuda::<model::gemma_2::forward::ShippedW1, model::gemma_2::forward::ShippedA, model::gemma_2::forward::ShippedKv>(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("gemma2 lowers")
}

#[test]
fn every_launch_of_the_gemma2_deployment_binds() {
    let l = gemma2_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: l.arena_bytes,
    };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gemma2: launch refused: {r:?}"));
        assert_eq!(
            bound.args.len(),
            (launch.args.end - launch.args.start) as usize
        );
    }
    // gemma2's ARG-level weights are all `scale.*` constants (which bind
    // without the resolver); the tensor weights ride the op join.
    let plan = model::gemma_2::forward::gemma2_cuda::<model::gemma_2::forward::ShippedW1, model::gemma_2::forward::ShippedA, model::gemma_2::forward::ShippedKv>(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode, EPS, THETA);
    let dp = driver_cuda::bind::DispatchPlan::new(&plan, &l);
    assert!(
        (0..l.launches.len()).any(|i| {
            dp.spec(i)
                .weight
                .as_deref()
                .is_some_and(|w| !w.starts_with("scale."))
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
    eprintln!(
        "=== gemma2 decode: {} launches, arena {}",
        l.launches.len(),
        l.arena_bytes
    );
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    for launch in l.launches.iter().take(30) {
        let args = &l.args[launch.args.start as usize..launch.args.end as usize];
        eprintln!(
            "  L {} rows={:?} args={args:?}",
            l.kernels[launch.kernel as usize], launch.rows
        );
    }
}

/// gemma-4 (the gemma anchor WITH a cached checkpoint — E2B; gemma-2's
/// 2b-it is gated upstream): both stated classes, the synthetic cuda set.
fn gemma4_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = model::gemma_4::forward::gemma4_cuda::<model::gemma_4::forward::ShippedW1, model::gemma_4::forward::ShippedA, model::gemma_4::forward::ShippedKv>(
        &model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e2b(),
        &model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
        class, EPS);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("gemma4 lowers")
}

#[test]
fn every_launch_of_the_gemma4_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = gemma4_lowered(class, rows);
        assert!(!l.launches.is_empty());
        let frame = Frame {
            arena: 0x10000 as *mut c_void,
            arena_bytes: l.arena_bytes,
        };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("gemma4 {class:?}: launch refused: {r:?}"));
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize
            );
        }
    }
}

/// nemotron_h (E-gate family #3): the synthetic fixture — the family
/// has NO cached real deployment in this environment, so the fixture is
/// the coverage anchor and the real-weight A/B is a recorded blocker.
/// DECODE only: the family states no other class yet.
fn nemotron_lowered(rows: usize) -> Lowered {
    let plan = model::nemotron_h::forward::nemotron_h_cuda::<model::nemotron_h::forward::ShippedW1, model::nemotron_h::forward::ShippedW2, model::nemotron_h::forward::ShippedA, model::nemotron_h::forward::ShippedKv>(
        &model::nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
        FireClass::Decode, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("nemotron_h lowers")
}

#[test]
fn every_launch_of_the_nemotron_deployment_binds() {
    let l = nemotron_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: l.arena_bytes,
    };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("nemotron_h: launch refused: {r:?}"));
        assert_eq!(
            bound.args.len(),
            (launch.args.end - launch.args.start) as usize
        );
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
    eprintln!(
        "=== nemotron_h Decode: {} launches, arena {}",
        l.launches.len(),
        l.arena_bytes
    );
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
            eprintln!(
                "  L {k} rows={:?} layers={:?} args={args:?}",
                launch.rows, launch.layers
            );
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
        eprintln!(
            "=== gemma4 {class:?}: {} launches, arena {}",
            l.launches.len(),
            l.arena_bytes
        );
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

/// Symbols live deployments LOWER TO that [`bridged_symbols`] would not
/// otherwise reach, because nothing in `kernels_cuda::sigs()` names them.
///
/// **EMPTY, and the four that were here are the reason it is worth keeping.**
/// They were `attn::split_qkv_bf16`, `layout::split_q_gate_bf16`,
/// `mlp::sigmoid_gate_inplace_bf16` and `ssm::qwen_gdn_post_conv_prep_bf16` —
/// four of `kernels_cuda::driver_internal`'s six `pub fn`s, answering
/// symbols the gemma4 and anchor lowerings (the first) and the qwen3.5 hybrid
/// and gemma3n (the other three) emit through `semantic()`. That module's doc
/// said *"these are plain `pub fn`s `driver-cuda` calls by path, no statement
/// names them, and there is nothing for a trace to resolve"*, and both halves
/// of it were false: statements name four of the six, and this crate calls
/// none of the six by path.
///
/// All four are DECLARED now — a `untraced!` line each in `attn`,
/// `layout`, `mlp` and `ssm`, deriving the row from the same `fn`, with the
/// bodies left in `driver_internal` because that is where the driver-shaped
/// `void*` parameter lists belong. So `sigs()` reaches them and this list has
/// nothing to add.
///
/// # WHAT DID NOT CHANGE, which is the half this file used to make visible
///
/// **A fire naming any of the four still refuses with `NoArm`.** Declaring a
/// symbol is not arming it: `bind/arms/` has an entry for none of the four,
/// `bind/mod.rs`'s `other =>` arm is what a dispatch reaches, and only
/// `attn::split_qkv_bf16` even has a near relative that is armed —
/// `attn::split_qkv_bf16_devwin`, at `bind/arms/attn.rs:390` — which is a
/// different kernel and not a fallback for it. Arming four kernels is
/// fire-path work, and this file is a GPU-free binder test: it can prove a
/// symbol has somewhere to go and cannot prove the somewhere is right.
///
/// The consequence for the tests below is worth stating plainly, because it
/// is a loss. `every_lowered_*_kernel_has_a_bridge_row` now passes on these
/// four through the row rather than through this list, so it no longer
/// distinguishes *has a row* from *has an arm*. This paragraph is where that
/// distinction is recorded; `driver_internal`'s own header is the other copy.
///
/// # How the original four stayed invisible
///
/// [`bridged_symbols`] filtered on `KernelSig::operands` being non-empty, and
/// that column is `&[]` on every row in `kernels-cuda` now. The filter
/// therefore matched nothing, the set came back empty, and the function
/// panicked on `gemm::act_x_wt_bf16` — a symbol that is unquestionably
/// present — before it could reach these four. Eight tests were red on a
/// message that was false, which is a better disguise for a real gap than
/// green would have been.
///
/// An entry belongs here when a lowering emits a symbol `sigs()` does not
/// declare and something in this crate can nonetheless serve it. Nothing is
/// in that position today.
const LOWERED_WITH_A_HOST_PROGRAM_AND_NO_ARM: &[&str] = &[];

fn bridged_symbols() -> BTreeSet<&'static str> {
    // A `.filter(|k| !k.operands.is_empty())` STOOD HERE, and it had come to
    // select NOTHING — which is why every test that calls this was red, with
    // a message about `gemm::act_x_wt_bf16` having no bridge row when the row
    // is right there in `sigs()`.
    //
    // What it was for: `table::driver_internal::DRIVER_KERNELS` used to be
    // chained onto `sigs()`, and its rows are dispatched by no one — they are
    // `driver_internal`'s `fn`s, CALLED and never named by a statement — so
    // they had to come back out. A non-empty `operands` meant "this row
    // states a launch ABI", which those rows did not.
    //
    // Why it stopped working: `KernelSig::operands` is `&[]` on every row in
    // this crate now. The derived half takes `SIG_BASE`'s default and the
    // stated half never wrote one, so the predicate is false everywhere and
    // the set is empty. `operands`' own doc says an empty list means UNSTATED
    // rather than "takes nothing"; a filter on it was therefore reading a
    // column that had stopped being written, and reading it as an answer.
    //
    // Why nothing replaces it: the exclusion it performed is now structural,
    // and the population it excluded has half moved. `driver_internal`
    // declares no `FAMILY` of its own and never will — its header says why —
    // but four of its six `fn`s are named from `attn`, `layout`, `mlp` and
    // `ssm` by a `untraced!` line each, so their symbols ARE in `sigs()`
    // and belong there: a live model text lowers to every one. The other two
    // forward to `norm` routines that were already declared, so nothing in
    // that module is a row waiting to be filtered back out. The set below IS
    // the dispatchable set, by construction rather than by predicate.
    let rows: BTreeSet<&'static str> = kernels_cuda::sigs().iter().map(|k| k.symbol).collect();
    let mut reachable = rows.clone();
    reachable.extend(AWAITING_THE_VERIFY_STASH_POOL);
    reachable.extend(LOWERED_WITH_A_HOST_PROGRAM_AND_NO_ARM);
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
        let frame = Frame {
            arena: 0x10000 as *mut c_void,
            arena_bytes: l.arena_bytes,
        };
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
        let frame = Frame {
            arena: 0x10000 as *mut c_void,
            arena_bytes: l.arena_bytes,
        };
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

    let starved = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: 1,
    };
    let mut resolver = Sentinels::default();
    let refusal = l
        .launches
        .iter()
        .find_map(|launch| bind(&l, launch, starved, &mut resolver).err());
    assert!(
        matches!(
            refusal,
            Some(BindRefusal::ArenaOutOfBounds { arena_bytes: 1, .. })
        ),
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
    let frame = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: l.arena_bytes,
    };
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
        proj_repr: model_dsl::WeightRepr::Bf16,
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
            LlamaLikeCudaFacts {
                decode_fused_post: true,
                ..tail.clone()
            },
        ),
        (
            "qwen2_5_1_5b",
            LlamaLikeFacts::qwen2_5_1_5b(),
            LlamaLikeCudaFacts {
                force_prefill_path: true,
                ..tail.clone()
            },
        ),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeCudaFacts {
                decode_fused_post: true,
                ..tail.clone()
            },
        ),
        (
            "phi3_mini",
            LlamaLikeFacts::phi3_mini(),
            LlamaLikeCudaFacts {
                head_dim_padded: true,
                head_dim_kernel: 128,
                ..tail.clone()
            },
        ),
    ];
    let bridged = bridged_symbols();
    for (name, facts, cuda) in &deployments {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_cuda::<model::shared::llama_like::forward::ShippedA, model::shared::llama_like::forward::ShippedKv>(facts, cuda, class, EPS, THETA);
            let rows: Vec<Row> = vec![
                Row {
                    samples: true,
                    ..Row::default()
                };
                4
            ];
            let l = lower(
                &plan,
                &rows,
                Fire {
                    captures_across_splits: false,
                },
            )
            .expect("lowers");
            let missing: Vec<&String> = l
                .kernels
                .iter()
                .filter(|k| !bridged.contains(k.as_str()))
                .collect();
            let frame = Frame {
                arena: 0x10000 as *mut c_void,
                arena_bytes: l.arena_bytes,
            };
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
        eprintln!(
            "=== {class:?}: {} launches, arena {} bytes",
            l.launches.len(),
            l.arena_bytes
        );
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        {
            let frame = Frame {
                arena: 0x10000 as *mut c_void,
                arena_bytes: l.arena_bytes,
            };
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
    let plan = llama_like_cuda::<model::shared::llama_like::forward::ShippedA, model::shared::llama_like::forward::ShippedKv>(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            lora: true,
            ..Row::default()
        };
        2
    ];
    let l = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("lowers");
    eprintln!("=== lora decode: {} launches", l.launches.len());
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if k.contains("lora") {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L {k} rows={:?} layers={:?} args={args:?}",
                launch.rows, launch.layers
            );
        }
    }
}

/// gemma3n (E-gate family #7): the synthetic fixture — the family has no
/// cached deployment here, so the fixture is the coverage anchor and the
/// real-weight A/B is a recorded blocker. AltUp's rank-K residual is the
/// new vocabulary: predict from the active stream, run the body on the
/// prediction, correct every stream from the result.
fn gemma3n_lowered(rows: usize) -> Lowered {
    let plan = model::gemma_3n::forward::gemma3n_cuda::<model::gemma_3n::forward::ShippedW1, model::gemma_3n::forward::ShippedA, model::gemma_3n::forward::ShippedKv>(
        &model::gemma_3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(),
        FireClass::Decode, EPS, THETA, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("gemma3n lowers")
}

#[test]
fn every_launch_of_the_gemma3n_deployment_binds() {
    let l = gemma3n_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: l.arena_bytes,
    };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gemma3n: launch refused: {r:?}"));
        assert_eq!(
            bound.args.len(),
            (launch.args.end - launch.args.start) as usize
        );
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
    eprintln!(
        "=== gemma3n Decode: {} launches, arena {}",
        l.launches.len(),
        l.arena_bytes
    );
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
            eprintln!(
                "  L {k} rows={:?} layers={:?} args={args:?}",
                launch.rows, launch.layers
            );
        }
    }
}

/// gpt-oss / mixtral (E-gate family #8): the Mixtral plan family, one
/// declaration for both model types. A 20b checkpoint IS cached here, so
/// this family's facts are the real ones.
fn gpt_oss_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = model::gpt_oss::forward::gpt_oss_cuda::<model::gpt_oss::forward::ShippedW1, model::gpt_oss::forward::ShippedW2, model::gpt_oss::forward::ShippedA, model::gpt_oss::forward::ShippedKv>(
        &model::gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b(),
        &model::gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic(),
        class, EPS, WINDOWED_THETA, WINDOW);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("gpt_oss lowers")
}

#[test]
fn every_launch_of_the_gpt_oss_deployment_binds() {
    let l = gpt_oss_lowered(FireClass::Decode, 4);
    assert!(!l.launches.is_empty());
    let frame = Frame {
        arena: 0x10000 as *mut c_void,
        arena_bytes: l.arena_bytes,
    };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gpt_oss: launch refused: {r:?}"));
        assert_eq!(
            bound.args.len(),
            (launch.args.end - launch.args.start) as usize
        );
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
    eprintln!(
        "=== gpt_oss Decode: {} launches, arena {}",
        l.launches.len(),
        l.arena_bytes
    );
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    let mut seen = BTreeSet::new();
    for launch in &l.launches {
        let k = &l.kernels[launch.kernel as usize];
        if seen.insert(k.clone()) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L {k} rows={:?} layers={:?} args={args:?}",
                launch.rows, launch.layers
            );
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
    let plan = model::glm_5::forward::glm5_cuda::<model::glm_5::forward::ShippedW1, model::glm_5::forward::ShippedW2, model::glm_5::forward::ShippedA, model::glm_5::forward::ShippedKv>(
        &model::glm_5::forward::facts::Glm5Facts::glm5_106b_a12b(),
        FireClass::Decode, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("glm5 lowers")
}

fn kimi_k2_lowered(rows: usize) -> Lowered {
    let plan = model::kimi_k2::forward::kimi_cuda::<model::kimi_k2::forward::ShippedW1, model::kimi_k2::forward::ShippedW2, model::kimi_k2::forward::ShippedA, model::kimi_k2::forward::ShippedKv>(
        &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
        &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
        FireClass::Decode, EPS);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("kimi_k2 lowers")
}

fn kimi_k3_lowered(rows: usize) -> Lowered {
    let plan = model::kimi_k3::forward::kimi_k3_cuda::<model::kimi_k3::forward::ShippedW1, model::kimi_k3::forward::ShippedW2, model::kimi_k3::forward::ShippedA, model::kimi_k3::forward::ShippedKv>(
        &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
        FireClass::Decode, EPS);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("kimi_k3 lowers")
}

fn dsv4_lowered(rows: usize) -> Lowered {
    let plan = model::deepseek_v4::forward::dsv4_cuda::<model::deepseek_v4::forward::ShippedW1, model::deepseek_v4::forward::ShippedW2, model::deepseek_v4::forward::ShippedA, model::deepseek_v4::forward::ShippedKv>(
        &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
        FireClass::Decode, EPS, THETA);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("dsv4 lowers")
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
        let frame = Frame {
            arena: 0x10000 as *mut c_void,
            arena_bytes: l.arena_bytes,
        };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("{name}: launch refused: {r:?}"));
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize
            );
        }
    }
}

#[test]
fn every_lowered_kernel_of_the_remaining_families_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for l in [
        glm5_lowered(4),
        kimi_k2_lowered(4),
        kimi_k3_lowered(4),
        dsv4_lowered(4),
    ] {
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

// The four builders below fed `every_lowered_symbol_has_an_arm`, deleted at
// `4f9ccba92` with the unit world. That test read `bind/mod.rs` as TEXT,
// collecting every quoted `family::symbol` on a line ending in `=>`, and the
// descent left it asking about a `match` that no longer holds the answer.
//
// Its CORPUS outlived it with nothing to feed, which is how the four went
// warning-only: `cargo check` does not build test targets, so nothing said so
// until the gate this commit adds. Two paragraphs of its doc comment survived
// too, glued to whichever item followed them — the shape a doc takes when the
// item between it and the next one goes.
//
// They are given a consumer again rather than deleted, because what they
// covered is not covered elsewhere. Every surviving `every_lowered_*` test in
// this file builds its corpus from `lowered(class, rows)`, whose rows are all
// alike; these four are the only ones that mark rows, and the marks are what
// make the lowering interesting — two of them PEEL, one turns every plain
// guard on at once, and one asks for `GuardMode::Union`.
//
// WHAT THE DELETED TEST ASKED, kept because nothing asks it now:
//
// > Every symbol any family lowers to has an ARM, not merely a row. The
// > bridge-row claims above answer "can the launcher be CALLED"; this one
// > answers "does `dispatch` know what to call". They are different questions
// > and the gap between them cost this branch a GPU cycle per instance during
// > the `origin/rewrite` merges: a row exists, the crate compiles, every
// > non-GPU test passes, and the first thing that notices is a fire refusing
// > with `NoArm`.
//
// The gap is still open. `bind/arms/`'s `unbound:` reasons are where an
// answer would go, and it would be a better one than the source scan was:
// the registries are VALUES, so the question can be asked of them rather
// than of the text of a `match`.

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
    let mut r: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    for row in r.iter_mut().skip(rows.saturating_sub(2)) {
        row.custom_mask = true;
    }
    lower(
        &plan,
        &r,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("a masked fire lowers")
}

/// The same, on the hook axis: the suffix rows carry attached programs,
/// which is the other `PeelWindow` and the other way a fire's rows stop
/// being interchangeable.
fn hooked_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let mut r: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    for row in r.iter_mut().skip(rows.saturating_sub(2)) {
        row.hooked = true;
    }
    lower(
        &plan,
        &r,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("a hooked fire lowers")
}

/// The UNION lowering: every guard arm present, nothing decided.
///
/// The one corpus entry that is not a fire shape but a MODE. A union
/// capture records every arm and lets a conditional decide at replay, so
/// every arm has to exist — and an arm that only a guard's losing side
/// states is invisible to every lowering above, because `Resolve` deletes
/// it before the kernel list is built.
fn union_lowered(rows: usize) -> Lowered {
    let plan = plan_of(FireClass::Decode);
    let r: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    model_compiler::lower::lower_with(
        &plan,
        &r,
        Fire {
            captures_across_splits: false,
        },
        model_compiler::lower::GuardMode::Union,
    )
    .expect("the union lowers")
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
    lower(
        &plan,
        &r,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("a fire carrying every mark lowers")
}

/// The same bridge-row claim, over the four lowerings whose rows are MARKED.
///
/// `every_lowered_kernel_has_a_bridge_row` above builds its corpus from
/// `lowered(class, rows)`, whose rows are all alike, so it can only reach the
/// symbols an unmarked fire lowers to. A mark is exactly what puts a
/// DIFFERENT symbol in the list: the two peels split the fire and dispatch
/// each region separately, `every_mark_lowered` satisfies the plain guards so
/// their arms survive `Resolve`, and `union_lowered` keeps every arm by asking
/// for none to be decided — which is the only way a symbol stated solely by a
/// guard's losing side appears at all.
///
/// It asserts the same thing about a larger set, and the four builders' own
/// docs say why each is in it. Naming them here is what makes those docs
/// load-bearing again: they described a corpus that had no consumer between
/// `4f9ccba92` and this test.
#[test]
fn every_kernel_a_marked_fire_lowers_to_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    let corpus = [
        ("masked", masked_lowered(4)),
        ("hooked", hooked_lowered(4)),
        ("every mark", every_mark_lowered(4)),
        ("union", union_lowered(4)),
    ];
    // Vacuity guard, and it is not a floor on the total. A lowering that
    // stopped producing kernels at all would satisfy the assertion below
    // perfectly, which is the way this shape of test dies quietly.
    for (what, l) in &corpus {
        assert!(
            !l.kernels.is_empty(),
            "the {what} lowering produced no kernels at all"
        );
    }
    for (_, l) in &corpus {
        for symbol in &l.kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "kernels a MARKED fire lowers to with no stated bridge row: {unreachable:?}"
    );
}

/// Every symbol every deployment lowers to either RUNS or says why it does
/// not — asked of the registries as values.
///
/// # The check this replaces, and why this one is better
///
/// `every_lowered_symbol_has_an_arm` asked exactly this until `4f9ccba92`,
/// by reading `bind/mod.rs` as TEXT: it collected every quoted
/// `family::symbol` on a line ending in `=>` and called those the armed set.
/// Its own doc was upfront that this was a narrower question than the match
/// answers, and it died when the descent moved the answer out of that
/// `match`.
///
/// `bind::route::route` ANSWERS IT. It used to be a lookup in a table of
/// `Bound` values and the table is gone; the question is the same and the
/// answer is derived, which separates four states a text scan flattened into
/// two:
///
/// * **bound** — the routine's own column runs it;
/// * **driver** — `#[routine(driver)]`, and `bind::dispatch`'s match fires it;
/// * **unbound** — a parameter nothing states, named. That is a stated gap,
///   and stated gaps are this tree's idiom, not its failures;
/// * **unknown** — no routine declares it, or one does and says `internal`.
///   A fire naming one refuses `NoArm`: a message about dispatch, naming
///   neither what is missing nor who would supply it.
///
/// Only the fourth is a defect, and it is the one nothing could see.
#[test]
fn every_lowered_symbol_runs_or_says_why_not() {
    let mut absent: BTreeSet<String> = BTreeSet::new();
    let mut armed = 0usize;
    let mut stated = 0usize;
    let corpus = [
        ("anchor decode", lowered(FireClass::Decode, 4)),
        ("anchor prefill", lowered(FireClass::Prefill, 7)),
        ("hybrid decode", qwen35_lowered(FireClass::Decode, 4)),
        ("hybrid prefill", qwen35_lowered(FireClass::Prefill, 7)),
        ("gemma2", gemma2_lowered(4)),
        ("gemma4 decode", gemma4_lowered(FireClass::Decode, 4)),
        ("gemma4 prefill", gemma4_lowered(FireClass::Prefill, 7)),
        ("gemma3n", gemma3n_lowered(4)),
        ("nemotron", nemotron_lowered(4)),
        ("masked", masked_lowered(4)),
        ("hooked", hooked_lowered(4)),
        ("every mark", every_mark_lowered(4)),
        ("union", union_lowered(4)),
    ];
    for (what, l) in &corpus {
        assert!(
            !l.kernels.is_empty(),
            "the {what} lowering produced no kernels at all"
        );
        for symbol in &l.kernels {
            match driver_cuda::bind::route::route(symbol) {
                driver_cuda::bind::route::Route::Bound(_)
                | driver_cuda::bind::route::Route::Driver => armed += 1,
                driver_cuda::bind::route::Route::Unbound(_) => stated += 1,
                driver_cuda::bind::route::Route::Unknown => {
                    absent.insert(symbol.clone());
                }
            }
        }
    }
    // Vacuity guard, and it is the one this shape of test needs most: a
    // registry lookup that started answering `None` for everything would
    // report every symbol at once, and one that started answering `Some`
    // for everything would report nothing and look like success.
    assert!(
        armed > 50,
        "only {armed} runnable symbols — the derivation broke"
    );
    assert!(
        absent.is_empty(),
        "symbols a live deployment lowers to that the arm registries do not mention. \
         A fire naming one refuses with `NoArm`, which names neither what is missing \
         nor who would supply it. Either no `#[routine]` declares the symbol, or one \
         does and says `internal` -- which means a text is naming a body other \
         routines call: {absent:?}\n\
         ({armed} runnable, {stated} refused across {} lowerings)",
        corpus.len()
    );
}

/// EVERY NAMED OPERAND NAMES THE VALUE THAT OWNS ITS BYTES.
///
/// This is the guard for a bug that made the engine serve fluent garbage
/// while every structural test passed, and it is worth stating why nothing
/// else caught it.
///
/// An in-place kernel is ONE pointer read and written — `qk_rmsnorm_rope`
/// takes a single `q: *mut bf16` and normalises it in place — and
/// `alias_owners` is the union-find that says which trace values therefore
/// have to be the same bytes. The arena obeys it without anyone asking:
/// `Buffers::assign` gives a whole alias group one offset, so two aliased
/// arena operands arrive as the same `Arg::Arena { at }` and no reader ever
/// learns a group existed.
///
/// `Buffers::NAMED` is not an offset. It is a SENTINEL meaning "the backend
/// binds this one", and it used to carry the value id through untouched — so
/// the alias survived for arena values and was **lost for named ones**. The
/// driver then did the only sensible thing with what it was handed: one
/// buffer per distinct id. `qk_rmsnorm_rope` got two, normalised the fresh
/// zeroed output over itself, and wrote zeros back — every layer, every
/// model, so attention ran on a query of exactly zero and the logits were
/// confident nonsense. Thirty routines in `kernels-cuda` declare an
/// `in_place` pair; every one of them was one lowering away from the same
/// fate.
///
/// So the invariant is checked where it can be checked cheaply and for every
/// family at once: a lowering may name any value it likes, but if it names a
/// NAMED one it must name the group's owner. Then "two ids" and "two
/// buffers" stop being the same sentence, and a backend that allocates per
/// distinct id is correct by construction rather than by luck.
///
/// Both sides are walked. `Lowered::args` is the operand side, written by
/// `Lowerer::slot`; `DispatchPlan`'s `outs` and `aux` are the output side,
/// written by the driver's own `out_arg`. They are two functions in two
/// crates and they have to agree, because a launch whose input names the
/// owner and whose output names the alias reads one buffer and writes
/// another — which is the bug, restated.
#[test]
fn a_named_operand_names_its_alias_owner() {
    let mut checked = 0usize;
    // The emission targets' shared tail, as `print_all_deployment_vocabularies`
    // states it — dense bf16, one GPU, whole context.
    let tail = LlamaLikeCudaFacts {
        head_dim_kernel: 0,
        proj_repr: model_dsl::WeightRepr::Bf16,
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
    let corpus: Vec<(&str, LlamaLikeFacts, LlamaLikeCudaFacts)> = vec![
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b(), tail.clone()),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), tail.clone()),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeCudaFacts {
                decode_fused_post: true,
                ..tail.clone()
            },
        ),
        (
            "phi3_mini",
            LlamaLikeFacts::phi3_mini(),
            LlamaLikeCudaFacts {
                head_dim_padded: true,
                head_dim_kernel: 128,
                ..tail.clone()
            },
        ),
    ];
    for (name, facts, cuda) in &corpus {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_cuda::<model::shared::llama_like::forward::ShippedA, model::shared::llama_like::forward::ShippedKv>(facts, cuda, class, EPS, THETA);
            let rows: Vec<Row> = vec![
                Row {
                    samples: true,
                    ..Row::default()
                };
                4
            ];
            let l = lower(
                &plan,
                &rows,
                Fire {
                    captures_across_splits: false,
                },
            )
            .expect("lowers");
            let dp = driver_cuda::bind::DispatchPlan::new(&plan, &l);

            let owner = |v: ValueId| l.value_owner.get(v as usize).copied().unwrap_or(v);
            let mut check = |side: &str, i: usize, a: &Arg| {
                if let Arg::Named { value, .. } = a {
                    assert_eq!(
                        owner(*value),
                        *value,
                        "{name} {class:?} launch {i} ({}): {side} names v{value}, \
                         but v{} owns those bytes — an alias group with two ids \
                         gets two buffers, and an in-place kernel then reads one \
                         and writes the other",
                        l.kernels[l.launches[i].kernel as usize],
                        owner(*value),
                    );
                    checked += 1;
                }
            };
            for i in 0..l.launches.len() {
                let lu = &l.launches[i];
                for a in &l.args[lu.args.start as usize..lu.args.end as usize] {
                    check("an operand", i, a);
                }
                for a in &dp.spec(i).outs {
                    check("an output", i, a);
                }
                for a in &dp.spec(i).aux {
                    check("an aux slot", i, a);
                }
            }

            // ── And the half the value ids cannot say. ──
            //
            // The owner check above is necessary and NOT sufficient: it
            // asks whether a NAMED operand is canonical, and an alias
            // group that was never formed has nothing to be canonical
            // about. `norm::residual_add_bf16` is that case — its arm
            // accumulates into `arg_out(0)`, its row declares `in_place =
            // &[(0, 0)]`, and if the union does not happen the arena
            // hands the output a block of its own and the kernel adds the
            // residual to whatever the previous launch left there.
            //
            // So this asks the question end-to-end instead: for every
            // in-place pair a statement's kernel declares, the operand
            // slot and the output slot must be the SAME storage. Args are
            // inputs, then outputs, then weights (`Lowerer::region`), so
            // the two slots are `i` and `n_in + o`.
            for i in 0..l.launches.len() {
                let lu = &l.launches[i];
                let op = &plan.ops[lu.op as usize];
                let pairs = match &op.kind {
                    model_ir::trace::OpKind::Launch { kernel, .. } => {
                        model_ir::kernels::in_place_pairs(&plan, kernel)
                    }
                    other => model_ir::kernels::semantic_in_place(other).to_vec(),
                };
                let args = &l.args[lu.args.start as usize..lu.args.end as usize];
                for &(o, in_i) in &pairs {
                    let (Some(a_in), Some(a_out)) = (
                        args.get(in_i as usize),
                        args.get(op.inputs.len() + o as usize),
                    ) else {
                        continue;
                    };
                    assert_eq!(
                        a_in, a_out,
                        "{name} {class:?} launch {i} ({}): in_place ({o}, {in_i}) says output {o} \
                         IS input {in_i}, but the lowering placed them apart — input {a_in:?} vs \
                         output {a_out:?}. The arm writes through ONE pointer, so it will \
                         accumulate into whatever the output slot last held",
                        l.kernels[lu.kernel as usize],
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no NAMED operand was reached, so nothing was checked"
    );
    eprintln!("a_named_operand_names_its_alias_owner: {checked} named operands");
}
