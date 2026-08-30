//! **THE ATTENTION-SCORE DOOR, FROM BOTH SIDES** — the guest reading the
//! rectangle, and the graph writing it (`.wiki/alto/attn-score.md` §4, alto
//! campaign gates S-2 and S-3).
//!
//! Two claims, and they are about different halves of one seam:
//!
//! 1. **THE EPILOGUE READS WHAT THE HOST INTERPRETER READS** — a guest program
//!    that binds `IntrinsicId::AttnScore` gets, on the device, the same numbers
//!    `eta_exec`'s reference pass computes from the same rectangle. This is the
//!    F32 arm's claim, and nothing else can make it: the emitted `0xA0` handler
//!    reads its argument as `bfloat` for every other intrinsic, so a rectangle
//!    it read at the wrong element type would come back as garbage that still
//!    committed. `program_parity`'s method, over one hand-built subject rather
//!    than the golden corpus — the corpus has no score-reading program, and
//!    growing it is `eta-compiler`'s business rather than this shell's.
//!
//! 2. **THE GRAPH WRITES A DISTRIBUTION, AND WRITES NOTHING WHEN NOBODY ASKED**
//!    — S-2 and S-3. A captured plane IS the softmax the design defines,
//!    recomputed in this file from the same bf16 numbers the device was handed;
//!    the declared ceiling is zero-padded to on every path; a lane's block is
//!    its own; and a fire no lane captured leaves every byte of the slab as the
//!    reservation left it.
//!
//! **S-2 IS ASKED BOTH WAYS — OF THE KERNEL DIRECTLY, AND THROUGH A MODEL
//! FIRE.** The arithmetic claim is made where it can be made EXACTLY:
//! `device_floor`'s idiom, one dispatch over a synthetic pool whose answer
//! this file computes itself. The REACH claim is made where only a checkpoint
//! can make it — a capturing lane through `Shell::fire_seated`, whose every
//! plane comes back a distribution over the prompt's live keys.
//!
//! The second half was impossible until `kernels-metal` stamped the
//! log-sum-exp sdpa entries past head width 64. The capture is a side launch
//! at the end of the `attention.prefill_lse` arm, `qwen35-d0.8b` states
//! `head_dim: 256`, and a capturing lane on the workhorse SKU was refused one
//! axis over — before the capture was ever reached — by a sentence about the
//! sdpa ladder. `SDPA_LSE_WIDTHS` now runs the same rungs the plain sdpa
//! ladder does, so the pin this file used to carry came down and the gate it
//! was holding the door for
//! (`a_capturing_lane_writes_a_distribution_on_every_plane`) is on.
//!
//! # Gating
//!
//! Claims 1 and 2 need a Metal device and nothing else. The shell-level gates
//! need the checkpoint too, and SKIP with the reason rather than being
//! `#[ignore]`d — an ignored test on the one box that could run it is a test
//! nobody runs.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_observability_contract_holds \
//!   -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine::program::ProgramRegistration;
use engine_metal::device::{Buffer, Context, Handles, Pipelines, present};
use engine_metal::encode::Sink;
use engine_metal::program::{Fired, Plane};
use engine_metal::{Boot, Lane, Seated, Shell};
use eta_exec::{
    Extents, InterpInstance, PassInputs, StepOutcome, Value, adopt_launch_package, host_take,
    make_host_instance, step,
};
use eta_ir::Dtype;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::Shape;
use kernels_metal::Tensor;
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The published row width — the DSL, the engine and this gate read the same
/// constant, which is the whole reason it is one.
const KV_MAX: usize = eta_ir::registry::ATTN_SCORE_KV_MAX as usize;

/// How many planes the hand-built subject declares.
///
/// **TWO AND NOT ONE, BECAUSE ONE PROVES THE EASY HALF.** A single-row read
/// lands on the same bytes whatever the pitch is; a two-row read only lands
/// right if the emitted gather's consecutive walk IS the slab's stride, which
/// is the one shape `Prepared::bind_intrinsic` permits and the reason it
/// refuses every other.
const PLANES: usize = 2;

/// One shell at a time per process — the measurements below are only readable
/// one at a time, and claim 2 holds ~1.5 GiB resident.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<Context> {
    if !present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    match Context::bind() {
        Ok(context) => Some(context),
        Err(error) => {
            println!("SKIP {what}: the device does not bind ({error})");
            None
        }
    }
}

// ── (1) the guest half: the epilogue reads what the interpreter reads ────────

/// The subject: read the whole `[PLANES, KV_MAX]` rectangle at the epilogue
/// and publish its sum.
///
/// **A SUM OVER EVERY SLOT, WHICH IS WHY IT IS A SUM.** The claim is that the
/// device walked the rectangle at the right element type and the right pitch,
/// and a reduction that touched only the first row — or read `bfloat` where the
/// bytes are `float` — answers a different number. The staged values are exact
/// binary fractions so that the total is exact whatever order either half adds
/// them in; a disagreement here is about the walk, never about associativity.
fn subject() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![ChannelDecl {
            shape: Shape::vector(PLANES as u32),
            dtype: ChanDType::Concrete(Dtype::F32),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::AttnScore,
                    shape: Shape::matrix(PLANES as u32, eta_ir::registry::ATTN_SCORE_KV_MAX),
                    dtype: Dtype::F32,
                },
                Op::ReduceSum(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

/// The rectangle both halves are handed: exact binary fractions, sparse, and
/// different in every plane so a reader that folded them would be caught.
fn rectangle() -> Vec<f32> {
    let mut plane = vec![0.0f32; PLANES * KV_MAX];
    for row in 0..PLANES {
        let base = row * KV_MAX;
        // Powers of two, so every partial sum is exact.
        plane[base] = 0.5;
        plane[base + 1] = 0.25;
        plane[base + 7] = 0.125;
        plane[base + KV_MAX - 1] = if row == 0 { 0.125 } else { 0.0625 };
    }
    plane
}

/// The load this subject binds against: a text that observes.
fn observing_profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = 8;
    profile.has_attn_score = true;
    profile
}

fn registration(container: TraceContainer, profile: ModelProfile) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(container, profile)
        .unwrap_or_else(|why| panic!("the subject does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Metal;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);
    ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

/// **THE F32 ARM, ASKED AS A DIFF AND NOT AS A SPOT CHECK.**
///
/// The device runs the emitted M2 kernel against a staged F32 rectangle; the
/// host interpreter runs the reference pass against the same numbers. The
/// published cell has to be the same bytes, because the two halves were handed
/// the same rectangle and there is nothing lossy between them — an F32 slot
/// widens to nothing, which is the whole difference from the `logits` seam this
/// plane already diffs (`program_parity` stages bf16 and widens on purpose).
#[test]
fn a_guest_reads_the_score_rectangle_as_the_interpreter_reads_it() {
    let _guard = serialized();
    let Some(context) = device_or_skip("the score rectangle's guest half") else {
        return;
    };

    let registration = registration(subject(), observing_profile());
    let package = registration.launch.clone();
    let plan = adopt_launch_package(package.clone())
        .unwrap_or_else(|error| panic!("the subject does not adopt: {error}"));
    assert!(
        plan.executable,
        "this backend declines the subject — {}",
        plan.reject_reason.clone().unwrap_or_default()
    );
    // The one declaration the whole wave turns on: the plan says it reads the
    // rectangle, and the plane no longer refuses it for saying so.
    assert!(
        plan.needs_attn_scores,
        "the subject was supposed to read `attn_score`"
    );

    let mut interp: InterpInstance = make_host_instance(&plan, &BTreeMap::new(), &BTreeMap::new());
    let mut plane = Plane::new();
    let program = plane
        .register(&context, &registration)
        .unwrap_or_else(|error| panic!("the subject does not compile: {error}"));
    let instance = plane
        .bind(
            &context,
            program,
            &[],
            Extents::default(),
            GeometryClass::Host,
        )
        .unwrap_or_else(|error| panic!("the subject does not bind: {error}"));

    // **THE FIRE IS REFUSED BEFORE THE RECTANGLE IS BOUND, AND THAT IS THE
    // GUARD REPLACING THE OLD FLAT REFUSAL.** `program::session` used to turn
    // every score-reading program away; what it turns away now is one whose
    // rectangle nothing pointed at, which is the same sentence the `logits`
    // and `mtp_logits` seams already make.
    let unbound = plane
        .fire(&context, instance)
        .expect_err("a score-reading program with no rectangle bound may not fire");
    let said = format!("{unbound}");
    assert!(said.contains("attn_score"), "{said}");
    assert!(said.contains("no buffer has been bound"), "{said}");

    let numbers = rectangle();
    let bytes: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut buffer = Buffer::zeroed(&context, bytes.len() as u64)
        .unwrap_or_else(|error| panic!("the rectangle does not fit: {error}"));
    buffer
        .write(0, &bytes)
        .unwrap_or_else(|error| panic!("staging the rectangle: {error}"));

    // A rectangle that is not F32 is refused by name, because the emitted
    // handler picks its element type off the intrinsic id and cannot be told
    // another one.
    let wrong = plane
        .bind_intrinsic(
            instance,
            IntrinsicId::AttnScore,
            &buffer,
            0,
            eta_ir::registry::ATTN_SCORE_KV_MAX,
            Dtype::Bf16,
        )
        .expect_err("the score rectangle is F32 and only F32");
    assert!(format!("{wrong}").contains("F32"), "{wrong}");

    plane
        .bind_intrinsic(
            instance,
            IntrinsicId::AttnScore,
            &buffer,
            0,
            eta_ir::registry::ATTN_SCORE_KV_MAX,
            Dtype::F32,
        )
        .unwrap_or_else(|error| panic!("binding the rectangle: {error}"));

    let inputs = PassInputs {
        attn_score: Some(&numbers),
        ..PassInputs::none()
    };
    let host = step(&mut interp, &plan, &inputs);
    assert!(
        matches!(host, StepOutcome::Committed),
        "the host half did not commit: {host:?}"
    );
    let device = plane
        .fire(&context, instance)
        .unwrap_or_else(|error| panic!("the device half will not fire: {error}"));
    assert_eq!(
        device,
        Fired::Committed,
        "the device half did not commit where the host did"
    );

    let (_, published) = host_take(&interp, &plan, 0);
    let Some(Value::F32(expected)) = published else {
        panic!("the host published nothing on channel 0");
    };
    let landed = plane
        .instance_mut(instance)
        .expect("bound")
        .take(0)
        .unwrap_or_else(|error| panic!("taking channel 0: {error}"))
        .expect("the device published a cell");
    let seen: Vec<f32> = landed
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();

    // **THE NUMBERS THEMSELVES, STATED.** `ReduceSum` folds the LAST axis, so
    // the published cell is one total PER PLANE — which is the assertion that
    // makes `PLANES = 2` worth having: the two planes hold different mass, so
    // a reader that stopped after the first row, or walked its second row at
    // any pitch but the slab's, answers a different pair. A reader that took
    // the bytes for `bfloat` answers something with no relation to either.
    let per_plane: Vec<f32> = (0..PLANES)
        .map(|row| numbers[row * KV_MAX..(row + 1) * KV_MAX].iter().sum())
        .collect();
    assert_eq!(per_plane, vec![1.0, 0.9375], "the fixture is not what it was");
    assert_eq!(expected, per_plane, "the host summed something else");
    assert_eq!(
        seen, expected,
        "the device read the score rectangle differently from the interpreter"
    );
    plane.close_instance(instance).expect("the instance closes");
}

// ── (2) the graph half: the capture kernel, on the device ───────────────────

/// The synthetic geometry the capture is fired over. Small enough to compute
/// the answer by hand in this file, and NOT degenerate in any of the four ways
/// that would make the claim vacuous: the query heads group over the kv heads
/// (2 over 1), the request's keys span two pages, the pages are out of order in
/// the page table, and the observation window is shorter than the query run.
const HEAD_DIM: usize = 64;
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const PAGE_SIZE: usize = 4;
/// Which physical page holds each of the request's two logical pages —
/// deliberately not `[0, 1]`, so a kernel that ignored the table would read the
/// wrong keys and still produce a well-formed distribution.
const PAGES: [u32; 2] = [2, 0];
const POOL_PAGES: usize = 4;
/// The request's live KV after the append: two pages, the second half full.
const KV_LEN: usize = 6;
/// The query rows this fire feeds — the last three positions of the sequence.
const QO_LEN: usize = 3;
/// The observation window, shorter than the query run on purpose.
const OBSERVE: usize = 2;

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

/// The same value as the device will read it back: bf16 widened to f32. The
/// expectation below is computed on THESE numbers, so the tolerance is about
/// the softmax rather than about a rounding neither half agreed to.
fn widen(v: f32) -> f32 {
    f32::from_bits((v.to_bits() >> 16) << 16)
}

/// A small, distinct, bf16-exact value for `(row, column)`.
fn fixture(row: usize, column: usize) -> f32 {
    let n = (row * 7 + column * 3) % 11;
    widen(n as f32 * 0.125 - 0.5)
}

/// One query row's head `h`, as the shader will read it.
fn q_head(row: usize, head: usize) -> Vec<f32> {
    (0..HEAD_DIM).map(|d| fixture(row * Q_HEADS + head, d)).collect()
}

/// Key `j`'s kv head `h`, as the shader will read it.
fn k_head(j: usize, head: usize) -> Vec<f32> {
    (0..HEAD_DIM).map(|d| fixture(100 + j * KV_HEADS + head, d)).collect()
}

/// **THE ANSWER, COMPUTED HERE RATHER THAN COPIED FROM THE KERNEL.**
///
/// The design's own arithmetic, spelled in Rust: for each of the window's last
/// [`OBSERVE`] query rows, the softmax over that row's causal limit, averaged.
/// `kv_len` is the request's live extent, so row `w` of the window is query row
/// `qo_len - rows + w` and its limit is `kv_len - rows + w + 1`.
fn expected_plane(head: usize, kv_len: usize, qo_len: usize, scale: f32) -> Vec<f32> {
    let rows = OBSERVE.min(qo_len);
    let kv_head = head / (Q_HEADS / KV_HEADS);
    let mut out = vec![0.0f32; kv_len];
    for w in 0..rows {
        let q = q_head(qo_len - rows + w, head);
        let limit = (kv_len - rows + w + 1).min(kv_len);
        let scores: Vec<f32> = (0..limit)
            .map(|j| {
                let k = k_head(j, kv_head);
                scale * q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|s| (s - top).exp()).sum();
        for (j, s) in scores.iter().enumerate() {
            out[j] += (s - top).exp() / sum / rows as f32;
        }
    }
    out
}

/// Fire one capture over the synthetic geometry into `slab`, at `plane`.
///
/// `kv_len` and `qo_len` are the request's own; everything else is the
/// constants above. Returns nothing — the caller reads the slab back.
fn fire_capture(
    device: &Context,
    pipelines: &Pipelines,
    slab: &Buffer,
    kv_len: usize,
    qo_len: usize,
    plane: u32,
    plane_stride: u32,
    scale: f32,
) {
    let handles = Handles::new();

    // q, one row per query token.
    let mut q_bytes = Vec::new();
    for row in 0..qo_len {
        for head in 0..Q_HEADS {
            for value in q_head(row, head) {
                q_bytes.extend_from_slice(&bf16(value));
            }
        }
    }
    let mut q_store = Buffer::zeroed(device, q_bytes.len() as u64).expect("q reserves");
    q_store.write(0, &q_bytes).expect("q lands");

    // The pool: every slot of every page written, so a kernel reading the
    // wrong page reads real numbers rather than zeros and the diff is sharp.
    let row_stride = KV_HEADS * HEAD_DIM;
    let mut pool_bytes = vec![0u8; POOL_PAGES * PAGE_SIZE * row_stride * 2];
    for j in 0..kv_len {
        let page = PAGES[j / PAGE_SIZE] as usize;
        let slot = page * PAGE_SIZE + j % PAGE_SIZE;
        for head in 0..KV_HEADS {
            for (d, value) in k_head(j, head).into_iter().enumerate() {
                let at = ((slot * KV_HEADS + head) * HEAD_DIM + d) * 2;
                pool_bytes[at..at + 2].copy_from_slice(&bf16(value));
            }
        }
    }
    let mut pool = Buffer::zeroed(device, pool_bytes.len() as u64).expect("the pool reserves");
    pool.write(0, &pool_bytes).expect("the pool lands");

    let indices: Vec<u32> = PAGES.to_vec();
    let mut index_store = Buffer::zeroed(device, 4 * indices.len() as u64).expect("indices");
    index_store.write(0, words(&indices)).expect("indices land");

    let indptr: Vec<u32> = vec![0, kv_len.div_ceil(PAGE_SIZE) as u32];
    let mut indptr_store = Buffer::zeroed(device, 4 * indptr.len() as u64).expect("indptr");
    indptr_store.write(0, words(&indptr)).expect("indptr lands");

    // The causal bound: a query row's ABSOLUTE position, which is what this
    // plane's capture reads where the CUDA twin reconstructs `kv_len`.
    let positions: Vec<i32> = (0..qo_len)
        .map(|row| (kv_len - qo_len + row) as i32)
        .collect();
    let mut position_store = Buffer::zeroed(device, 4 * qo_len as u64).expect("positions");
    position_store
        .write(0, signed(&positions))
        .expect("positions land");

    let qo: Vec<i32> = vec![0, qo_len as i32];
    let mut qo_store = Buffer::zeroed(device, 4 * qo.len() as u64).expect("qo indptr");
    qo_store.write(0, signed(&qo)).expect("the qo indptr lands");

    let q_h = handles
        .bind(&q_store, 0, q_bytes.len() as u64)
        .expect("q binds");
    let qo_h = handles.bind(&qo_store, 0, 4 * qo.len() as u64).expect("qo binds");
    let pool_h = handles
        .bind(&pool, 0, pool_bytes.len() as u64)
        .expect("the pool binds");
    let index_h = handles
        .bind(&index_store, 0, 4 * indices.len() as u64)
        .expect("indices bind");
    let indptr_h = handles
        .bind(&indptr_store, 0, 4 * indptr.len() as u64)
        .expect("indptr binds");
    let position_h = handles
        .bind(&position_store, 0, 4 * qo_len as u64)
        .expect("positions bind");
    let slab_rows = plane_stride;
    let slab_h = handles
        .bind(slab, 0, u64::from(slab_rows) * KV_MAX as u64 * 4)
        .expect("the slab binds");

    let pool_view = kernels_metal::KvPool {
        keys: Tensor::new(pool_h, (POOL_PAGES * PAGE_SIZE) as u32, row_stride as u32, Dtype::Bf16),
        values: Tensor::new(pool_h, (POOL_PAGES * PAGE_SIZE) as u32, row_stride as u32, Dtype::Bf16),
        page_indices: Tensor::new(index_h, indices.len() as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(indptr_h, indptr.len() as u32, 1, Dtype::U32),
        page_size: PAGE_SIZE as i32,
        seq_stride: row_stride as u64,
        head_stride: HEAD_DIM as u64,
    };
    let plan = kernels_metal::PrefillPlan {
        positions: Tensor::new(position_h, qo_len as u32, 1, Dtype::I32),
        request_of_token: Tensor::new(position_h, qo_len as u32, 1, Dtype::I32),
        mask: Tensor::new(position_h, qo_len as u32, 1, Dtype::U8),
        mask_enabled: Tensor::new(position_h, qo_len as u32, 1, Dtype::U8),
        mask_stride: 1,
    };

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, &handles);
        kernels_metal::attn::score::capture(
            &sink,
            kernels_metal::RaggedTensor {
                data: Tensor::new(q_h, qo_len as u32, (Q_HEADS * HEAD_DIM) as u32, Dtype::Bf16),
                indptr: Tensor::new(qo_h, qo.len() as u32, 1, Dtype::I32),
            },
            &plan,
            &pool_view,
            None,
            HEAD_DIM as u32,
            KV_HEADS as u32,
            scale,
            OBSERVE as u32,
            0,
            plane_stride,
            plane,
            eta_ir::registry::ATTN_SCORE_KV_MAX,
            1,
            Tensor::new(slab_h, slab_rows, eta_ir::registry::ATTN_SCORE_KV_MAX, Dtype::F32),
        )
        .expect("the capture encodes");
    }
    frame.commit().expect("the capture completes");
}

fn words(values: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4) }
}

fn signed(values: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4) }
}

fn read_slab(slab: &Buffer, planes: u32) -> Vec<f32> {
    let mut raw = vec![0u8; planes as usize * KV_MAX * 4];
    slab.read(0, &mut raw).expect("the slab reads back");
    raw.chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// **S-2, THE WHOLE OF IT, ON THE DEVICE.** Every plane the capture writes is
/// a probability distribution over the request's live KV positions, exactly
/// zero on the declared ceiling past them, and EQUAL TO THE SOFTMAX THIS FILE
/// COMPUTES ITSELF.
///
/// The last clause is what separates this from a plumbing test. A kernel that
/// read the wrong page, folded the heads, took the softmax over the wrong
/// causal limit, or averaged the wrong query rows would still produce rows that
/// sum to one — every one of those mistakes is a well-formed distribution over
/// something else. So the assertion is against the arithmetic
/// `.wiki/alto/attn-score.md` §4 spells, recomputed here from the same bf16
/// numbers the device was handed.
///
/// The tolerance is `1e-3` and it is bf16's, not f32's: the keys are bf16 and
/// the dots accumulate in f32, so a row of a few dozen bf16 terms lands within
/// about a thousandth. Tightening it would be asserting about the storage
/// dtype rather than about the softmax.
#[test]
fn a_captured_plane_is_the_softmax_the_design_defines() {
    let _guard = serialized();
    let Some(device) = device_or_skip("the capture kernel") else {
        return;
    };
    let pipelines = Pipelines::new();
    let planes = Q_HEADS as u32;
    let slab = Buffer::zeroed(&device, u64::from(planes) * KV_MAX as u64 * 4)
        .expect("the slab reserves");
    let scale = 0.125f32;
    fire_capture(&device, &pipelines, &slab, KV_LEN, QO_LEN, 0, planes, scale);

    let got = read_slab(&slab, planes);
    for head in 0..Q_HEADS {
        let row = &got[head * KV_MAX..(head + 1) * KV_MAX];
        let want = expected_plane(head, KV_LEN, QO_LEN, scale);
        let mass: f32 = row[..KV_LEN].iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "plane {head} sums to {mass} over its {KV_LEN} live keys, not one"
        );
        for (j, (a, b)) in row.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "plane {head} key {j}: the device says {a}, the design says {b}"
            );
        }
        // **THE DECLARED CEILING IS ZERO-PADDED TO, NOT LEFT.** A position that
        // does not exist received no attention, so it sorts to the bottom of
        // every eviction ranking without a sentinel.
        assert!(
            row[KV_LEN..].iter().all(|p| *p == 0.0),
            "plane {head} has a non-zero tail past its {KV_LEN} live keys"
        );
        // And it is not uniform: a kernel that answered `1 / kv_len` everywhere
        // would pass every clause above and carry no information at all.
        let flat = 1.0 / KV_LEN as f32;
        assert!(
            row[..KV_LEN].iter().any(|p| (p - flat).abs() > 1e-3),
            "plane {head} is the uniform distribution, which is not an observation"
        );
    }
    println!("  the capture is the design's softmax to 1e-3 on {planes} planes");
}

/// **A SLAB OUTLIVES A FIRE, AND THAT IS THE FAILURE THIS CATCHES.**
///
/// The rectangle is reserved at a ceiling and reused, so a short request
/// landing where a long one was is the one arrangement in which "the tail is
/// zero" can be true by accident on every fire but the second. The kernel
/// writes the whole row every time for exactly this reason; this is the
/// assertion that says it does.
#[test]
fn a_short_request_leaves_no_tail_of_the_long_one_before_it() {
    let _guard = serialized();
    let Some(device) = device_or_skip("the stale-tail gate") else {
        return;
    };
    let pipelines = Pipelines::new();
    let planes = Q_HEADS as u32;
    let slab = Buffer::zeroed(&device, u64::from(planes) * KV_MAX as u64 * 4)
        .expect("the slab reserves");
    let scale = 0.125f32;

    fire_capture(&device, &pipelines, &slab, KV_LEN, QO_LEN, 0, planes, scale);
    let long = read_slab(&slab, planes);
    assert!(
        long[KV_LEN - 1] > 0.0,
        "the long request must really have written its last key"
    );

    // The same slab, the same planes, a request half as long.
    let short_kv = 2;
    let short_qo = 1;
    fire_capture(&device, &pipelines, &slab, short_kv, short_qo, 0, planes, scale);
    let after = read_slab(&slab, planes);
    for head in 0..Q_HEADS {
        let row = &after[head * KV_MAX..(head + 1) * KV_MAX];
        let mass: f32 = row[..short_kv].iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "plane {head} of the short request sums to {mass}"
        );
        assert!(
            row[short_kv..].iter().all(|p| *p == 0.0),
            "plane {head} still carries the long request's mass past key {short_kv}"
        );
    }
}

/// **A LANE'S BLOCK IS ITS OWN, AND ONE LANE CANNOT PROVE THAT.**
///
/// Lane zero's base offset IS zero, so a kernel that observed lane zero for
/// everybody would be perfectly deterministic and perfectly wrong on a
/// single-lane fire. Two planes at two bases in one slab is the smallest
/// arrangement that can tell the difference — and the second write must leave
/// the first alone, which is the addressing claim the whole per-lane slab rests
/// on.
#[test]
fn a_capture_writes_its_own_planes_and_nobody_elses() {
    let _guard = serialized();
    let Some(device) = device_or_skip("the plane addressing gate") else {
        return;
    };
    let pipelines = Pipelines::new();
    // Two lanes' worth of blocks, `Q_HEADS` planes each.
    let planes = Q_HEADS as u32;
    let stride = planes * 2;
    let slab =
        Buffer::zeroed(&device, u64::from(stride) * KV_MAX as u64 * 4).expect("the slab reserves");
    let scale = 0.125f32;

    fire_capture(&device, &pipelines, &slab, KV_LEN, QO_LEN, planes, stride, scale);
    let got = read_slab(&slab, stride);
    // The block the capture was pointed at holds the mass.
    for head in 0..Q_HEADS {
        let at = planes as usize + head;
        let row = &got[at * KV_MAX..(at + 1) * KV_MAX];
        let mass: f32 = row[..KV_LEN].iter().sum();
        assert!((mass - 1.0).abs() < 1e-3, "plane {at} sums to {mass}");
    }
    // And the neighbouring block is untouched — not zeroed by the capture,
    // which never addressed it, but still the zeros the reservation had.
    for at in 0..planes as usize {
        let row = &got[at * KV_MAX..(at + 1) * KV_MAX];
        assert!(
            row.iter().all(|p| *p == 0.0),
            "plane {at} was written by a capture pointed at plane {planes}"
        );
    }
}

// ── (3) the shell: what serves, and the one thing that does not ─────────────

/// The catalog row whose model text declares the capture arm.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Long enough that the capture window is shorter than the query run, and
/// short enough to prefill in one fire under [`budgets`].
const PROMPT: &str = "The capital of France is Paris. Paris is a large European city \
                      with a long history. Paris is a large European city with a long \
                      history.";

/// Small on purpose: the arena reserves `max_tokens` rows of a wide logit
/// column, and these gates need a prompt rather than a batch.
fn budgets() -> Budget {
    Budget::new(4, 256)
}

/// The lane word the model's own `Classify` computes — the facts qwen declares,
/// and no third opinion about any of them.
fn word(query_len: u32, captures: bool) -> u64 {
    model::qwen_3::forward::Facts::of(
        &Request::new(query_len, false).capturing_scores(captures),
    )
    .word()
}

fn seat<'a>(slot: u32, tokens: &'a [u32], captures: bool) -> Seated<'a> {
    let lane = Lane {
        slot,
        word: word(tokens.len() as u32, captures),
        tokens,
    };
    if captures {
        Seated::capturing(lane)
    } else {
        Seated::of(lane)
    }
}


/// **S-3. A FIRE NO LANE CAPTURED PAYS THE AXIS NOTHING.**
///
/// The observation is a LAUNCH or it is nothing: `Run::capture_scores` returns
/// before it reaches an encoder for a load with no slab, for a fire no lane
/// captured, and for a node the score seam does not name. So a plain fire on a
/// load that CARVED a slab must leave every byte of it as the reservation left
/// it — and it must fire the launches it always fired.
///
/// Structural as well as measured: the slab is the SHELL's, so `model-compiler`
/// never hears about it and the artifact a pre-campaign SKU bakes is
/// byte-identical.
#[test]
fn a_fire_no_lane_captured_pays_the_observability_axis_nothing() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the zero-cost gate") else {
        return;
    };
    assert!(
        shell.observes_scores(),
        "this SKU's text declares a capture column, so the slab must exist"
    );
    let planes = shell.score_planes();
    println!(
        "  {planes} planes ({} layers x {} heads)",
        planes / shell.score_heads().max(1),
        shell.score_heads()
    );

    let prompt = encode(&tokenizer, PROMPT);
    shell.open(0).expect("the slot opens");
    let first = shell
        .fire_seated(&[seat(0, &prompt, false)])
        .expect("the plain fire lands");
    let before = shell.last_fire();

    shell.open(0).expect("the slot reopens");
    let second = shell
        .fire_seated(&[seat(0, &prompt, false)])
        .expect("the second plain fire lands");
    let after = shell.last_fire();

    assert_eq!(
        (before.launches, before.copied),
        (after.launches, after.copied),
        "two identical plain fires did not launch the same number of times"
    );
    assert_eq!(first, second, "a plain fire is not deterministic");
    // And the slab it never wrote is the slab it was given.
    let block = shell
        .observed(0)
        .expect("the slab reads back")
        .expect("this load observes");
    assert_eq!(block.len(), planes as usize * KV_MAX);
    assert!(
        block.iter().all(|p| *p == 0.0),
        "a fire no lane captured wrote into the observability slab"
    );
    println!("  a plain fire is {} launches and touches no plane", after.launches);
}

/// **THE TWO REFUSALS THE AXIS OWES, BY NAME.**
///
/// A lane that asks to be observed and whose word puts it outside the capture
/// window would get no mass at all, and a row of zeros is indistinguishable
/// from a sequence that attended to nothing. The reverse — a word inside the
/// window with no ask — writes a plane no epilogue is pointed at.
#[test]
fn a_capture_ask_and_a_word_that_disagree_are_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the capture word gate") else {
        return;
    };
    let prompt = encode(&tokenizer, PROMPT);

    shell.open(0).expect("the slot opens");
    // Asks, and its word says otherwise.
    let mut asking = seat(0, &prompt, false);
    asking.captures_scores = true;
    let why = shell
        .fire_seated(&[asking])
        .expect_err("an ask whose word skips the capture arm");
    assert!(format!("{why}").contains("outside the capture window"), "{why}");

    // Its word says so, and it does not ask.
    let quiet = Seated::of(Lane {
        slot: 0,
        word: word(prompt.len() as u32, true),
        tokens: &prompt,
    });
    let why = shell
        .fire_seated(&[quiet])
        .expect_err("a capturing word with no ask behind it");
    assert!(format!("{why}").contains("did not ask to be observed"), "{why}");
}

/// **S-2 END TO END, THROUGH A REAL FIRE ON A REAL CHECKPOINT.**
///
/// A capturing lane prefills the workhorse SKU, and EVERY plane of its block
/// comes back a probability distribution over the prompt's live keys.
///
/// This is the gate the file used to pin rather than run. Until the
/// log-sum-exp sdpa entries were stamped past head width 64, a capturing lane
/// on this SKU was refused at `attention.prefill_lse` for stating
/// `head_dim: 256` — one axis over from anything about scores — so the shell
/// half of S-2 could not be asked at all.
///
/// **AND IT IS THE REACH CLAIM, NOT THE ARITHMETIC ONE.** The softmax itself
/// is proved exactly above, against numbers this file computes
/// (`a_captured_plane_is_the_softmax_the_design_defines`); a checkpoint's
/// weights are not a fixture and recomputing this SKU's attention here would
/// be reimplementing the model. What only a model fire can say is that the arm
/// is REACHED — that the capture window admits this head width, that the
/// `attn.scores` seam names every exported layer's node, and that the plane a
/// layer owns is the plane it writes. So the assertion is the shape claim on
/// every plane at once: mass one over the prompt's live keys to bf16's
/// tolerance, EXACTLY zero past them, and not the uniform row a launch that
/// wrote nothing meaningful would leave.
#[test]
fn a_capturing_lane_writes_a_distribution_on_every_plane() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the shell-level S-2 gate") else {
        return;
    };
    assert!(
        shell.observes_scores(),
        "this SKU's text declares a capture column, so the slab must exist"
    );
    let prompt = encode(&tokenizer, PROMPT);
    let kv_len = prompt.len();
    assert!(
        kv_len > 0 && kv_len < KV_MAX,
        "the prompt has to fit inside the declared ceiling: {kv_len} keys"
    );

    shell.open(0).expect("the slot opens");
    shell
        .fire_seated(&[seat(0, &prompt, true)])
        .expect("the capturing fire lands");

    let planes = shell.score_planes();
    let heads = shell.score_heads().max(1);
    // Lane and not slot: a single-lane fire seriates to fire lane zero, which
    // is the row the capture arm addressed.
    let block = shell
        .observed(0)
        .expect("the slab reads back")
        .expect("this load observes");
    assert_eq!(block.len(), planes as usize * KV_MAX);

    for plane in 0..planes as usize {
        let row = &block[plane * KV_MAX..(plane + 1) * KV_MAX];
        let (layer, head) = (plane / heads as usize, plane % heads as usize);
        let mass: f32 = row[..kv_len].iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "layer {layer} head {head} sums to {mass} over the prompt's {kv_len} keys, not one"
        );
        // The declared ceiling is zero-padded to, on a real fire as on a
        // synthetic one — a position the prompt does not have received no
        // attention and must not outrank one it does.
        assert!(
            row[kv_len..].iter().all(|p| *p == 0.0),
            "layer {layer} head {head} has a non-zero tail past the prompt's {kv_len} keys"
        );
        // And a plane no capture reached would be all zeros, while one that
        // answered `1 / kv_len` everywhere would sum to one and carry nothing.
        let flat = 1.0 / kv_len as f32;
        assert!(
            row[..kv_len].iter().any(|p| (p - flat).abs() > 1e-3),
            "layer {layer} head {head} is the uniform distribution, which is not an observation"
        );
    }
    println!(
        "  {planes} planes ({} layers x {heads} heads), every one a distribution over the \
         prompt's {kv_len} keys",
        planes / heads
    );
}

// ── the checkpoint, or the reason there is none ─────────────────────────────

fn encode(tokenizer: &tokenizer::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text)
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    })
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// A loaded shell and the tokenizer that goes with it, or the reason there is
/// neither — printed rather than failed, for `serve_smoke`'s reason.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(snapshot) = snapshot() else {
        println!("SKIP {what}: no Qwen3.5-0.8B snapshot on this machine");
        return None;
    };
    let Some(container) = container(&snapshot) else {
        println!("SKIP {what}: the snapshot holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    // The runtime's half: trace the row, state the load contract. Neither is
    // the shell's — `Trace` crosses the boundary, `CompiledModel` never does.
    let trace = model::trace_of(SKU).expect("the catalog ships this SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &snapshot,
        budget: budgets(),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}
