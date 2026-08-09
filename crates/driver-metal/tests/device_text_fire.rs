//! **The whole text, on the GPU, through the generic executor.**
//!
//! This is the north star's fourth property with the checkpoint taken out: a
//! sealed frame's step becomes rows, the rows become rectangles, the
//! rectangles become grids, the grids become a command buffer, and the command
//! buffer runs. Every one of the 423 launches of `llama_like`'s Metal text
//! reaches the device, and nothing in the driver names a family, a kernel or a
//! model on the way.
//!
//! What it does NOT prove is that the numbers are right, and there are **two**
//! reasons, of which only the first is obvious:
//!
//! 1. the weights are sentinels, not a checkpoint;
//! 2. the text still does not state everything its kernels read. Every row
//!    it names states its operands now — `tests/text_conformance.rs` holds
//!    that at zero — so no launch is bound positionally any more. What the
//!    rows still carry as `Unbound` is the gap: the gathers' token ids and
//!    the paged attention's six fire tables are values no statement supplies.
//!
//!    A slot nobody bound is read anyway, and Metal does not validate a
//!    binding, so the answer is whatever the arena held.
//!
//! `tests/text_conformance.rs` measures the second and pins the number so it
//! can only shrink. That number is the honest distance between this file and
//! a model that answers, and shrinking it is the work between here and
//! token-exactness.
//!
//! So "it ran" is a milestone and not a result — and this is exactly the
//! failure this crate was built to make impossible to miss, arriving in its
//! own executor.

use std::collections::HashMap;
use std::path::PathBuf;

use driver_metal::bind::encode::Pipelines;
use driver_metal::device::{Allocation, Context};
use driver_metal::fire::run::run;
use driver_metal::lowering::dispatch::Geometry;
use driver_metal::lowering::executor::{Resolver, Slice};
use driver_metal::lowering::frame::{Step, lower_step};
use driver_metal::program::Compiler;
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::trace::{FireClass, ValueId};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

/// Every weight the text names, backed by one generous region.
///
/// Sentinels rather than a checkpoint: this test is about whether the fire
/// EXECUTES. A region large enough for any tensor means no kernel reads past
/// an allocation, which is what would turn an execution proof into a crash
/// that says nothing about the executor.
struct Sentinels {
    slice: Slice,
    /// A ZEROED region for the fire's tables, and the zeros are the point.
    ///
    /// The kernels used to be handed zero extents — no statement carried a
    /// scalar — so every one of them no-opped and "the whole text fires" was
    /// true for a reason that flattered it. Now they carry real extents and do
    /// real work, and a paged attention handed a GARBAGE page CSR walks pages
    /// until the GPU is abandoned. Measured: this test hung for sixty seconds
    /// on exactly that.
    ///
    /// Zeros make the CSR say "no pages", which is a fire that computes
    /// nothing and returns — the honest sentinel for a test about whether the
    /// path executes.
    tables: Slice,
    asked: HashMap<String, usize>,
}

impl Resolver for Sentinels {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        *self.asked.entry(name.to_string()).or_default() += 1;
        Some(self.slice)
    }
    fn named(&mut self, _: ValueId) -> Option<Slice> {
        Some(self.slice)
    }
    fn kv(&mut self, _: u16, _: bool) -> Option<Slice> {
        Some(self.slice)
    }
    fn fire(&mut self, _: driver_metal::lowering::executor::FireTable) -> Option<Slice> {
        Some(self.tables)
    }

    /// A pool shaped like something rather than like nothing.
    ///
    /// `page_size = 0` is what hung this test for sixty seconds: the paged
    /// attention divides by it, and the zeroed page CSR that makes the scan
    /// terminate does not save a kernel that never gets to the scan.
    fn pool(&mut self, which: driver_metal::lowering::executor::FireTable) -> Option<u32> {
        use driver_metal::lowering::executor::FireTable as F;
        Some(match which {
            F::KvHeadStride => 128,
            F::KvSeqStride => 128 * 8,
            F::KvPageSize => 16,
            _ => return None,
        })
    }
}

/// A region of zeros for the fire's tables.
///
/// Explicitly zeroed rather than trusted to be: a fresh Metal buffer is
/// usually zero and nothing promises it, and what the zeros buy here is a page
/// CSR that says "no pages". A garbage one walks pages until the GPU is
/// abandoned, which is how this test first found out its kernels had started
/// doing real work.
///
/// Returns the `Allocation`, not a `Handle`: the caller has to own the region
/// for as long as the fire binds it, and this used to hand back a view of one
/// nobody owned.
fn zeroed(context: &Context) -> Allocation {
    use driver_metal::layout::region::Region as _;
    let h = Allocation::new(context, 1 << 20, "zeroed fire tables").expect("a table region");
    // SAFETY: freshly allocated, nothing encoded against it yet.
    unsafe { h.zero(0, 1 << 20).expect("it zeroes") };
    h
}

fn geometry() -> Geometry {
    Geometry {
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        rotary_dims: 128,
        n_experts: 0,
        experts_per_token: 0,
    }
}

/// The MIXTURE reaches the device, through the same executor.
///
/// The point is what is NOT here. There is no mixture-aware code anywhere
/// between this text and the GPU: the router, the sort, the gather, the routed
/// matmuls and the combine walk `dispatch::plan_one` exactly as a projection
/// does, and `LaunchRule::RouteRows`/`RoutedQmv` read the expert counts off
/// the dims the same way `Qmv` reads `width`.
///
/// A routed FFN is the hardest thing to express portably -- its SHAPE depends
/// on a value the fire computes -- so a mixture firing without a per-family
/// branch is the strongest evidence the executor is general that this crate
/// has.
///
/// Four layers rather than forty-eight: the walk is what is under test and a
/// 48-layer mixture is the same six statements forty-eight times.
#[test]
fn a_mixture_fires_on_the_device_through_the_same_executor() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let facts = LlamaLikeFacts {
        layers: 4,
        ..LlamaLikeFacts::qwen3_30b_a3b()
    };
    let plan = llama_like_metal(&facts, &LlamaLikeMetalFacts::synthetic(), FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let routed = lowered
        .kernels
        .iter()
        .filter(|k| k.contains("routed") || k.starts_with("route_") || k.contains("router"))
        .count();
    assert!(
        routed >= 4,
        "a mixture states a router, a sort, a gather and three routed \
         matmuls; found {routed} in {:?}",
        lowered.kernels
    );

    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    let timing = run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            rotary_dims: 128,
            n_experts: 128,
            experts_per_token: 8,
        },
        &mut store,
    )
    .expect("the mixture fires");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "the stepper reported no encode time, so nothing was encoded"
    );
    assert!(
        store.asked.keys().any(|n| n.contains("expert")),
        "the fire bound no expert bank, so it cannot have been the mixture: {:?}",
        store.asked.keys().collect::<Vec<_>>()
    );
}

/// **gpt-oss reaches the device, through the same executor.**
///
/// Attention SINKS, gpt-oss's own SwiGLU and an alternating sliding window —
/// three facts, one extra weight per layer, one extra symbol. No family.
///
/// The `sinks` slot has been open on `sdpa_paged_decode`'s row since the rows
/// were written, with the comment "no statement fills it until a text that has
/// sinks does". This is that text.
#[test]
fn gpt_oss_fires_on_the_device_through_the_same_executor() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let facts = LlamaLikeFacts {
        layers: 4,
        ..LlamaLikeFacts::gpt_oss_20b()
    };
    let metal = LlamaLikeMetalFacts {
        window_left: vec![128, -1, 128, -1],
        ..LlamaLikeMetalFacts::gpt_oss_20b()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let sinked = lowered
        .kernels
        .iter()
        .filter(|k| k.contains("_sink"))
        .count();
    assert!(
        sinked > 0,
        "a sinked text names the sink symbol: {:?}",
        lowered.kernels
    );
    let swiglu = lowered
        .kernels
        .iter()
        .filter(|k| k.contains("swiglu"))
        .count();
    assert!(swiglu > 0, "and its own activation: {:?}", lowered.kernels);

    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    let timing = run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            rotary_dims: 64,
            n_experts: 32,
            experts_per_token: 4,
        },
        &mut store,
    )
    .expect("gpt-oss fires");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "the stepper reported no encode time, so nothing was encoded"
    );
    assert!(
        store.asked.keys().any(|n| n.contains("attn_sinks")),
        "the fire bound no sink, so it cannot have been gpt-oss: {:?}",
        store.asked.keys().collect::<Vec<_>>()
    );
}

/// **gemma's side network reaches the device.**
///
/// The PLE is the one thing in this arc that no set of facts could make
/// appear: a SECOND embedding table gathered once per step, projected, normed
/// and joined into `[n_layers, ple_dim]` that each layer reads its own slice
/// of. Four statements before the stack and four inside it.
///
/// With the KV-shared layers beside it, which are the other half of what makes
/// gemma4 a family: which dispatches EXIST moves per layer, because a shared
/// layer has no `k_proj` weight at all.
#[test]
fn gemmas_side_network_fires_on_the_device() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let facts = LlamaLikeFacts {
        layers: 6,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    let metal = LlamaLikeMetalFacts {
        window_left: vec![512, 512, 512, 512, 512, -1],
        kv_shared_layers: 2,
        ..LlamaLikeMetalFacts::gemma_like()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    for want in [
        "ple_combine",
        "geglu_tanh_strided",
        "geglu_tanh",
        "rms_residual_scaled",
    ] {
        assert!(
            lowered.kernels.iter().any(|k| k.starts_with(want)),
            "a gemma text names {want}: {:?}",
            lowered.kernels
        );
    }

    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    let timing = run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry(),
        &mut store,
    )
    .expect("the side network fires");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "the stepper reported no encode time, so nothing was encoded"
    );
    assert!(
        store.asked.keys().any(|n| n.starts_with("ple_")),
        "the fire bound no PLE weight: {:?}",
        store.asked.keys().collect::<Vec<_>>()
    );
    // The LAST two layers share their KV, so they name four projections
    // between them where an owning layer names six.
    assert!(
        !store.asked.contains_key("layer.5.k_proj"),
        "a KV-shared layer has no k_proj weight and must not name one"
    );
}

#[test]
fn the_whole_metal_text_fires_on_the_device() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    // One token a request, four lanes: the decode a scheduler posts.
    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let lowered = lower_step(&plan, &step).expect("the step lowers");
    assert!(
        lowered.launches.len() > 300,
        "a 24-layer decode should be hundreds of launches, not {}",
        lowered.launches.len()
    );

    // 256 MiB: wider than any tensor this text names, so a bound operand is
    // never the reason a dispatch fails.
    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    let timing = run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry(),
        &mut store,
    )
    .expect("the whole text fires");

    // The fire completed, and it compiled far fewer pipelines than it ran
    // dispatches — the cold start is bounded by the TEXT, not by the fire.
    assert!(
        timing.encode > std::time::Duration::ZERO,
        "the stepper reported no encode time, so nothing was encoded"
    );
    assert!(
        !store.asked.is_empty(),
        "the fire bound no weights, so it cannot have been the real text"
    );
    let restated = store.asked.values().filter(|&&n| n > 1).count();
    assert!(
        restated > 0,
        "no weight was asked for twice; a 24-layer text restates its shapes"
    );
}

/// **Ignored, and the reason is a finding rather than a flake.**
///
/// This passed until the statements started carrying their scalars. While
/// every kernel was handed zero extents they all no-opped, so "the batched
/// lane fires" was true for a reason that flattered it. Now they do real work
/// — and `sdpa_paged_decode` is one of the four statements whose scalars are
/// still unstated, so it runs with `page_size = 0` and walks pages until the
/// GPU is abandoned. Sixty seconds, measured, twice.
///
/// Zeroing the fire tables does not help, which is the useful half: the hang
/// is in the SCALARS and not the tables.
///
/// The scalars are stated now — the pool's geometry arrives through
/// `Resolver::pool` — so this runs. `Sentinels::pool` states a shape rather
/// than zeros for exactly the reason above.
#[test]
fn a_prefill_step_fires_too_so_both_lanes_reach_the_device() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    // SIXTEEN tokens in one request: a prefill, which takes the batched
    // symbols (`affine_qmm_t`, `embed_gather_mb_4bit`, the paged pair).
    //
    // Sixteen and not eight, and the difference is a real precondition rather
    // than a round number. `qmm_t.metal` has no `M` argument -- its header
    // says the driver only selects it when `M % BM == 0`, so the row count
    // lives in the grid and every tile is full. `QMM_BMS` starts at sixteen,
    // so eight rows tile no rung and `Rule::Qmm` refuses them
    // (`Ungeometric::PartialTile`). It used to substitute, and both
    // substitutions were measured wrong against a real checkpoint: the
    // matvec's grid under the GEMM's symbol gave NaN, and rounding the row
    // axis up gave a finite wrong answer plus fourteen rows of overrun into
    // the next value.
    let step = Step {
        token_ids: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        qo_indptr: &[0, 16],
        sampling_indices: &[15],
        ..Step::default()
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry(),
        &mut store,
    )
    .expect("the batched lane fires too");
}

/// The paged KV pool, allocated at the fire's geometry.
///
/// `metal::stage_decode_storage` has allocated `KvSlots` since the port, but
/// sized from `batch::DecodeGeometry` — a model definition inside the driver.
/// This is the same allocation with its arguments taken from the frame.
#[test]
fn the_kv_pool_allocates_at_the_geometry_the_fire_states() {
    use driver_metal::layout::kv::Shape;
    use driver_metal::pools::kv::{Pool, translate};

    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let g = geometry();
    let shape = Shape {
        layers: 24,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("the pool allocates");

    assert_eq!(pool.pages(), 64);
    assert_eq!(
        pool.bytes(),
        shape.layer_bytes_at(0) * 2 * 24,
        "a K and a V region for every layer"
    );
    let layer = pool.layer(0).expect("layer 0 has pages");
    assert_ne!(
        layer.k.gpu_address(),
        layer.v.gpu_address(),
        "K and V must be distinct regions; one address would make the append \
         to K overwrite V"
    );
    assert!(
        pool.layer(24).is_none(),
        "past the last layer there is none"
    );

    // And the frame's translation reads against it.
    let table = [0u32, 1, 63];
    assert_eq!(
        translate(&pool, &table, &[0, 3], 0).expect("a lane's pages"),
        &[0, 1, 63]
    );
    assert!(
        translate(&pool, &[64], &[0, 1], 0).is_err(),
        "a page past the pool addresses another layer's memory"
    );
}

/// A KV move, run on the pool, checked byte for byte.
///
/// The pages are `StorageModeShared`, so a move is a `memmove` and needs no
/// encoder — and the memmove semantics are not incidental: a compaction slides
/// rows toward the front, so source and destination overlap.
#[test]
fn a_move_plan_slides_rows_without_smearing_them() {
    use driver_metal::layout::kv::Shape;
    use driver_metal::layout::{CellCopy, CellMovePlan};
    use driver_metal::pools::kv::Pool;

    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    // One layer, one head, tiny pages: the arithmetic is the subject, not the
    // size.
    let shape = Shape {
        layers: 1,
        kv_heads: 1,
        head_dim: 4,
        page_size: 2,
        pages: 4,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("the pool allocates");
    let layer = pool.layer(0).expect("layer 0");
    let row = shape.row_bytes().expect("a uniform pool") as usize;

    // Each row is its own byte, so a misplaced one names itself.
    let total = shape.layer_bytes_at(0) as usize;
    let src: Vec<u8> = (0..total).map(|i| (i / row) as u8).collect();
    layer.k.write(0, &src).expect("the pattern fits");
    layer.v.write(0, &src).expect("and into v");

    // Slide page 1 onto page 0 — the overlapping case a compaction makes.
    let page = shape.page_bytes().expect("a uniform pool");
    pool.apply(&CellMovePlan {
        copies: vec![CellCopy {
            src_off: page,
            dst_off: 0,
            bytes: page,
        }],
        pages_touched: 2,
    })
    .expect("the move runs");

    let read = |p: &driver_metal::pools::kv::Pages| -> Vec<u8> {
        let at = p
            .host_span(0, total as u64)
            .expect("the pages are addressable");
        unsafe { std::slice::from_raw_parts(at.as_ptr().cast_const(), total) }.to_vec()
    };
    for (name, got) in [("k", read(&layer.k)), ("v", read(&layer.v))] {
        // Page 0 now holds what page 1 held: rows 2 and 3.
        assert_eq!(got[0], 2, "{name}: page 0 row 0 came from page 1 row 0");
        assert_eq!(got[row], 3, "{name}: page 0 row 1 came from page 1 row 1");
        // And page 1 is untouched — a smear would have overwritten it.
        assert_eq!(got[2 * row], 2, "{name}: page 1 row 0 still its own");
        assert_eq!(got[3 * row], 3, "{name}: page 1 row 1 still its own");
        // Pages 2 and 3 were never named.
        assert_eq!(got[4 * row], 4, "{name}: page 2 untouched");
    }
}

/// Two WHOLE-TEXT fires in flight: the second is queued while the first runs.
///
/// The executor-level statement of `.wiki/new-driver/next.md` priority 1, and
/// the reason it is here rather than only in `device_steps`: a synthetic
/// two-dispatch step proves the stepper can hold two timeline values, and
/// this proves a real 300-launch decode can — that nothing in planning,
/// staging or binding secretly serialises on the previous fire.
///
/// What it measures is the property, not a duration. `run::submit` returns
/// having committed and not waited, so holding two `InFlight` values at once
/// is something `run` cannot express at all: it ends in `await_value`, so the
/// call that would produce the second cannot return until the first is done.
///
/// It also pins the ownership rule that makes this safe. A committed fire's
/// command buffer addresses its arena, reads operands out of its argument
/// table and scalars out of its staged params; `InFlight` holds all three, so
/// dropping the handle a caller kept cannot free memory the GPU is reading.
/// The arenas being DIFFERENT is asserted, because two fires sharing one is
/// the first thing that would go wrong.
#[test]
fn two_whole_text_fires_are_in_flight_at_once() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let backing =
        Allocation::new(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let zeros = zeroed(&context);
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: 1 << 20,
        },
        asked: HashMap::new(),
    };

    // ONE stepper across both fires. A fresh one per fire — which is what
    // `run_keeping_arena` builds — has no timeline to compare against and no
    // allocator ring to alternate, so it could not pipeline even in
    // principle.
    let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
    // ONE pool too, for the same reason: reuse across fires is what makes an
    // address stable, and a fresh pool per fire is the allocation it replaces.
    let scratch = driver_metal::fire::Scratch::new();
    // No regions registered, so `record` refuses every operand and `submit`
    // falls back to encoding -- which is the behaviour this test wants, and
    // is also the fallback a deployment that has not registered its regions
    // gets. Both fires still commit.
    let mut regions = driver_metal::device::Regions::new();
    let mut recordings = driver_metal::fire::Recordings::new();

    let mut machine = driver_metal::fire::run::Machine {
        context: &context,
        compiler: &compiler,
        pipelines: &mut pipelines,
        stepper: &mut stepper,
        scratch: &scratch,
        regions: &mut regions,
        recordings: Some(&mut recordings),
    };
    let first = driver_metal::fire::run::submit(&mut machine, &lowered, geometry(), &mut store)
        .expect("the first fire commits");

    // Committed, not waited for. The second is planned, staged, bound and
    // committed with the first still outstanding.
    let second = driver_metal::fire::run::submit(&mut machine, &lowered, geometry(), &mut store)
        .expect("the second fire commits while the first is outstanding");

    // TWO regions, not one. The pool must not hand the second fire the arena
    // the first is still executing over -- `ALLOCATOR_COUNT = 2` bounds the
    // depth at two, so two arenas is exactly right and one would be a
    // use-after-free that a green run does not show.
    assert_ne!(
        first.arena.gpu_address(),
        second.arena.gpu_address(),
        "a fire in flight still owns its arena"
    );

    assert_eq!(
        second.value,
        first.value + 1,
        "two fires must be two timeline points"
    );
    assert_ne!(
        first.arena.gpu_address(),
        second.arena.gpu_address(),
        "two fires in flight shared one arena, so the second overwrote the \
         first's activations while it was still computing them"
    );

    stepper.wait_for(second.value).expect("both fires retire");
    assert!(
        stepper.has_passed(first.value) && stepper.has_passed(second.value),
        "the timeline reached {} but reports otherwise",
        second.value
    );
}
