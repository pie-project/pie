//! The paged attention bodies, against one reference and against each other.
//!
//! # What had no proof
//!
//! Attention is the only statement in this tree with THREE bodies. A fire
//! picks one by class and width — `sdpa_paged_decode` walks a query row per
//! threadgroup, `sdpa_paged_tiled` stages 32 rows through threadgroup memory,
//! `sdpa_paged_mma` puts eight of those rows on a simdgroup's matrix unit —
//! and all three are supposed to answer the same softmax.
//!
//! Nothing in this crate said so. The numeric proof attention had was
//! TRANSITIVE: `device_real_weights` runs whole checkpoints against MLX, which
//! holds every body the fire happens to pick at the width that checkpoint
//! happens to have. Both oracle checkpoints are `d = 64`, so EVERY OTHER
//! WIDTH the family is built at had never had its attention compared with
//! anything: 128 is `qwen3_0_6b`'s, the fixture `device_text_fire` fires end
//! to end, and 256 and 512 are shipped, compiled, pipeline-built and until now
//! never asked to be right. `device_text_fire` proves the text RUNS on
//! sentinels; it reads no value, and `device_kernels` proves a pipeline
//! BUILDS, which is not the same as answering.
//!
//! So this is the first per-kernel arithmetic on the family, and the first
//! arithmetic of any kind at 128, 256 and 512. It runs at all four.
//!
//! # Why a differential and not just a reference
//!
//! Both, because they fail differently. A CPU reference catches all three
//! bodies being wrong the same way — a page walk that reads the pool linearly,
//! a window applied to the wrong end. Agreement BETWEEN the bodies catches
//! what a reference's tolerance hides: the tolerance has to be loose enough
//! for a different summation order, and a real defect smaller than that slips
//! under it. Two bodies disagreeing by more than the output format's own
//! quantum is a defect no tolerance excuses, because they read identical bytes.
//!
//! # The fixture, and what each piece is for
//!
//! Every input value is a multiple of `1/8` in `[-1, 1]`, which bfloat16 holds
//! EXACTLY. The reference therefore sees the same numbers the device does and
//! the only rounding left in the comparison is the output store — see
//! `device_add_bias`, where a fixture the format could not hold tested the
//! format instead of the kernel.
//!
//! The page table is REVERSED (`[6, 4, 2, 0]` for request 0), so logical page
//! `p` is nowhere near physical page `p` and a body that read the pool
//! linearly answers something else — injected, and every arm fails. Two
//! requests share the pool, so a body that ignored `req_of_token` reads the
//! other request's history. Positions are `[6, 9, 2]` against a window of 4,
//! so one row's history is shorter than the window and two rows are clamped by
//! it. Three rows in a 32-row tile leaves 29 of them past `n_rows`, which is
//! the operand the tiled and mma rows state and the decode row does not.
//!
//! The MASK is on for one of the three rows. Every live fire in this tree
//! binds it disabled, so the arm that reads it is shipped, compiled, pipeline-
//! built and never taken; here row 1 takes it and rows 0 and 2 do not, which
//! is what makes the per-row flag observable rather than assumed. Its two drop
//! clauses are separated: a zero at key 7, and a `MASK_STRIDE` that puts key 9
//! past the end of the row. See [`MASK_HOLES`].
//!
//! # The one contract the three bodies do not share
//!
//! The requests are `[0, 0, 1]` and the order matters. The first draft used
//! `[0, 1, 0]` and `sdpa_paged_mma` ALONE disagreed, by 20% — far past
//! anything its `half` tiles could explain.
//!
//! It groups the tile into runs of equal request and decides membership with
//! `mine = my_req == r`, so a request owning rows in two runs of one tile
//! joins both and double-counts the keys their ranges share. What forbids that
//! is `qo_indptr`: `req_of_token` is derived from a CSR
//! (`driver-api/src/plan.rs:486-502`), so it is non-decreasing and a request
//! has exactly one run. `[0, 1, 0]` is a fire the planner cannot emit.
//!
//! So this is not a defect, and it was invisible until three bodies were asked
//! the same question — which is the argument for the differential. The
//! assumption now sits beside the line that makes it, in
//! `attn/sdpa_paged_mma.metal`.
//!
//! # What the first fixture could not have caught
//!
//! Its values had period 16 and every stride in the K/V pool is a multiple of
//! 16, so every page held the SAME vector. A softmax over identical planes is
//! that plane whatever weights it computed, so the page walk, the window and
//! the request split were all unobservable — the plain arms agreed with a
//! reference that was wrong in the same way. Only the SINK arms noticed,
//! because a denominator depends on the scores where a convex combination of
//! identical planes does not. See [`spread`].

use std::path::PathBuf;

use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;
use driver_metal::lowering::dispatch::{Dispatch, ParamSlot, Touches};
use driver_metal::lowering::executor::{BoundArg, Slice};

/// How much of its own tolerance the worst element used, and the band that
/// has to hold.
///
/// The check this replaced, in every test in this file, was: perturb a value
/// by four times the bound, assert the perturbation exceeds the bound. That
/// is `4b > b`. It is true of every bound anyone will ever write, it passes
/// just as brightly under a bound ten times too loose, and it stood under the
/// words "the tolerance discriminates".
///
/// `device_gdn` is where that cost something: a bound reasoned from bf16's
/// half-ulp rather than measured came out better than twice as loose as the
/// device, a fault injection that scaled the central gate of the kernel by
/// one percent PASSED underneath it, and this exact tautology approved.
///
/// What a tolerance check has to do is read the hardware. `worst` is the
/// largest `|got - want| / bound(want)` any comparison in the test actually
/// took, so it is the headroom in units of the bound. Above one the
/// assertions did not hold; below an eighth the bound is more than eight
/// times the arithmetic the device delivers, which is the room a wrong kernel
/// hides in. A hand that widens a bound to quiet a red run trips the floor
/// instead of getting away with it.
use driver_metal::skip::skipped;

fn tolerance_holds(worst: f32, what: &str) {
    // Exact agreement is not a loose bound; it is the absence of anything to
    // bound. Two paths that produce identical bits have no headroom to
    // measure, and on a device whose reduction happens to associate the same
    // way as the oracle's that is the RIGHT answer rather than a suspicious
    // one. The floor is about a bound that admits errors, and there is no
    // error here to admit.
    //
    // This matters because these bands were measured on one Mac. A different
    // GPU can land closer to the oracle than this one does -- the prefill
    // scan is already within a single fp32 ulp of the walked decode, and one
    // ulp from zero is not far -- and a floor that failed on perfect
    // agreement would turn a better device into a red build.
    if worst == 0.0 {
        return;
    }
    assert!(
        worst <= 1.0,
        "{what}: the worst element used {worst} of its bound, so an assertion above \
         passed by an accident of iteration order"
    );
    assert!(
        worst >= 0.125,
        "{what}: the worst element used only {worst} of its bound, so the tolerance is \
         more than eight times the arithmetic this device actually delivers -- tighten \
         the bound instead of trusting it"
    );
}

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Values a bfloat16 holds exactly: multiples of an eighth in `[-1, 1)`.
///
/// FIFTEEN of them, not sixteen, and this is the whole fixture. Every stride
/// in the K/V pool is a multiple of `head_dim` and every head_dim here is a
/// multiple of 16, so a period of 16 gives every page, every head and every
/// slot the SAME vector: the softmax then answers that vector whatever weights
/// it computed, and a body that walked the page table wrong is invisible. The
/// first draft of this file did exactly that, and only the sink arms noticed,
/// because a denominator depends on the scores where a convex combination of
/// identical planes does not. Fifteen is coprime with 64, 128 and 2, so no two
/// slots alias.
///
/// A `spread` of arbitrary floats would put the input's own rounding into the
/// comparison and buy nothing.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed) % 15) as f32 / 8.0 - 1.0)
        .collect()
}

const Q_HEADS: usize = 4;
const GQA: usize = 2;
const KV_HEADS: usize = Q_HEADS / GQA;
const ROWS: usize = 3;
const PAGE_SIZE: usize = 3;
const PAGES: usize = 7;
const SCALE: f32 = 0.125;
const WINDOW: i32 = 4;
const MASK_STRIDE: u32 = 9;
/// The mask is ON for row 1 alone, so a body that read `attention_mask` for
/// every row -- or for none -- answers a different thing than this states.
const MASK_ON: [u8; ROWS] = [0, 1, 0];
/// One zero inside EVERY row's kept window, not just the enabled row's.
///
/// Row 1's window keeps 6..=9 and the mask has to drop two of them by two
/// different clauses: the hole at 7, and `MASK_STRIDE` of 9, which puts 9
/// itself past the end of the row -- the shader treats that as absent rather
/// than as reading out of bounds. What survives is {6, 8}: NON-CONTIGUOUS, so
/// no window and no truncation can imitate it.
///
/// Rows 0 and 2 have the flag OFF, so their holes must have no effect at all.
/// They are here because without them the claim "the flag is read per row" is
/// unfalsifiable -- a reference that masked every row agreed anyway, since
/// nothing in their slices was ever zero. Their windows are 3..=6 and 0..=2.
const MASK_HOLES: [usize; ROWS] = [4, 7, 1];
const POSITIONS: [i32; ROWS] = [6, 9, 2];
const REQUESTS: [i32; ROWS] = [0, 0, 1];
const INDPTR: [u32; 3] = [0, 4, 7];
const INDICES: [u32; PAGES] = [6, 4, 2, 0, 5, 3, 1];

/// A body of the family, and whether it reads the `sinks` buffer it binds.
struct Arm {
    entrypoint: String,
    file: &'static str,
    /// Threads on x per query head, which is the threadgroup width: 1024 for
    /// the row and tile bodies, 128 for the matrix unit — see
    /// `lowering::launch`'s `SDPA_MMA_THREADS`, which is declared on the
    /// shader with `max_total_threads_per_threadgroup` and is not a knob.
    threads: u32,
    /// Whether the body owns 32 rows per group, which is also whether it
    /// states `n_rows` as an eighteenth operand.
    tiled: bool,
    sunk: bool,
    /// `(q_row_pitch, o_row_pitch)` in ELEMENTS, for the one entry point that
    /// states them. `None` is the packed layout every other point assumes and
    /// an eighteen-wide argument table; `Some` widens the table to twenty.
    pitch: Option<(u32, u32)>,
}

impl Arm {
    fn of(name: &str, head_dim: usize) -> Self {
        let mma = name.starts_with("sdpa_paged_mma");
        Self {
            entrypoint: format!("{name}_bfloat16_d_{head_dim}"),
            file: if mma {
                "attn/sdpa_paged_mma.metal"
            } else {
                "attn/sdpa_paged.metal"
            },
            threads: if mma { 128 } else { 1024 },
            tiled: mma || name.starts_with("sdpa_paged_tiled"),
            sunk: name.ends_with("_sink"),
            pitch: None,
        }
    }
}

/// Softmax attention over the kept keys, with an optional sink logit that
/// joins the denominator and has no value behind it.
fn reference(scores: &[f32], values: &[&[f32]], head_dim: usize, sink: Option<f32>) -> Vec<f32> {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let merged = sink.map_or(m, |s| m.max(s));
    let weights: Vec<f32> = scores.iter().map(|s| (s - merged).exp()).collect();
    let z: f32 = weights.iter().sum::<f32>() + sink.map_or(0.0, |s| (s - merged).exp());
    (0..head_dim)
        .map(|d| {
            let acc: f32 = weights.iter().zip(values).map(|(w, v)| w * v[d]).sum();
            acc / z
        })
        .collect()
}

/// One fire of one arm, into a fresh output.
///
/// The `args` vector is EIGHTEEN long whatever the row states, because
/// `bind::encode` binds `args[i]` at buffer `i` and the scalar operands of
/// this family sit at 4, 9, 10, 11, 13, 15 and 17 — interleaved among the
/// buffers rather than gathered past them. Those seven entries are overwritten
/// by the `param_slots` binds that follow in the same encode, so what they
/// hold does not matter; that they EXIST is what keeps `queries` at 0 and
/// `sinks` at 16.
#[allow(clippy::too_many_arguments)]
fn fire(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    arm: &Arm,
    head_dim: usize,
    page_size: usize,
    buffers: &[(usize, u64)],
    out: &Allocation,
) -> Vec<f32> {
    let args_wide = if arm.pitch.is_some() { 20 } else { 18 };
    let mut args = vec![
        BoundArg {
            slice: Slice {
                address: out.gpu_address(),
                bytes: 1 << 20,
            },
            width: 0,
        };
        args_wide
    ];
    for (slot, address) in buffers {
        args[*slot] = BoundArg {
            slice: Slice {
                address: *address,
                bytes: 1 << 20,
            },
            width: 0,
        };
    }

    // The scalars in the order the row states them, and the slots the shader
    // reads them at. `n_rows` is stated by the tiled and mma rows only.
    let mut params = vec![
        GQA as u32,
        page_size as u32,
        KV_HEADS as u32,
        SCALE.to_bits(),
        MASK_STRIDE,
        WINDOW as u32,
    ];
    let mut param_slots: Vec<ParamSlot> = [4u32, 9, 10, 11, 13, 15]
        .iter()
        .enumerate()
        .map(|(at, slot)| ParamSlot {
            slot: *slot as usize,
            at: (at * 4) as u32,
            bytes: 4,
            packed: false,
            // WHICH of the statement's scalars, not a placeholder.
            // `Params::stage` reads `value` as an index into `params` and
            // `at` as where it lands; `Some(0)` on every slot stages
            // `gqa_factor` seven times, which reads as `scale = 0` and
            // `window = 2` and answers a softmax over the wrong keys with
            // every score equal.
            value: Some(u8::try_from(at).expect("seven scalars")),
        })
        .collect();
    if arm.tiled {
        params.push(ROWS as u32);
        param_slots.push(ParamSlot {
            slot: 17,
            at: 24,
            bytes: 4,
            packed: false,
            value: Some(6),
        });
    }

    if let Some((q_pitch, o_pitch)) = arm.pitch {
        // 18 and 19, past every other point of the family. `q_row_pitch > 0`
        // is what switches the body off the packed `row * n_q_heads * D`.
        for (i, (slot, pitch)) in [(18usize, q_pitch), (19, o_pitch)].iter().enumerate() {
            params.push(*pitch);
            param_slots.push(ParamSlot {
                slot: *slot,
                at: (28 + i * 4) as u32,
                bytes: 4,
                packed: false,
                value: Some(u8::try_from(7 + i).expect("nine scalars")),
            });
        }
    }

    let rows_or_tiles = if arm.tiled {
        (ROWS as u32).div_ceil(32)
    } else {
        ROWS as u32
    };
    let dispatch = Dispatch {
        symbol: &arm.entrypoint,
        file: arm.file,
        stamp: "",
        grid: [Q_HEADS as u32 * arm.threads, rows_or_tiles, 1],
        threadgroup: [arm.threads, 1, 1],
        // One dispatch, so nothing to order against.
        touches: Touches::everything(&args),
        args,
        params,
        param_slots,
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, std::slice::from_ref(&dispatch))
        .unwrap_or_else(|why| panic!("`{}` builds a pipeline: {why}", arm.entrypoint));
    let staged =
        Params::stage(context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table = ArgumentTable::new(context, args_wide).expect("a table as wide as the row");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .unwrap_or_else(|why| panic!("`{}` fires: {why}", arm.entrypoint));

    // The whole output INCLUDING any pad between rows, so a caller that gave
    // a pitch can check the pad was not written as well as that the rows were.
    let pitch = arm.pitch.map_or(Q_HEADS * head_dim, |(_, o)| o as usize);
    let n = ROWS * pitch;
    let words = unsafe { core::slice::from_raw_parts(out.contents().as_ptr().cast::<u16>(), n) };
    words.iter().copied().map(from_bf16).collect()
}

/// The bodies of the paged attention family answer one softmax.
///
/// Run at both widths this tree instantiates for the tiled body. The matrix
/// unit joins at 64, which is the only width it is built at — see
/// `attn/sdpa_paged_mma.metal`'s instantiation block for the two reasons 128
/// is not built, neither of which is the threadgroup memory the header used to
/// blame.
#[test]
#[ignore = "needs a Metal 4 device"]
fn every_paged_attention_body_answers_the_same_softmax() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    for head_dim in [64usize, 128, 256, 512] {
        // The sink arms and the matrix unit are instantiated at 64 ONLY --
        // gpt-oss is the family with sinks and it is 64 wide, and
        // `attn/sdpa_paged_mma.metal` builds no other width. At the three
        // wider ones the two plain bodies are all there is to compare, which
        // is why the reference is not optional: above 64 the differential has
        // only one pair and could agree by agreeing on the same mistake.
        // Asking for `sdpa_paged_decode_sink_bfloat16_d_128` names a symbol
        // the library does not export, which `Pipelines::ensure` refuses by
        // name rather than by dispatching something arbitrary.
        let mut arms = vec![
            Arm::of("sdpa_paged_decode", head_dim),
            Arm::of("sdpa_paged_tiled", head_dim),
        ];
        if head_dim == 64 {
            arms.push(Arm::of("sdpa_paged_decode_sink", head_dim));
            arms.push(Arm::of("sdpa_paged_tiled_sink", head_dim));
            arms.push(Arm::of("sdpa_paged_mma", head_dim));
            arms.push(Arm::of("sdpa_paged_mma_sink", head_dim));
        }

        let q_seen = spread(ROWS * Q_HEADS * head_dim, 1);
        let pool = PAGES * PAGE_SIZE * KV_HEADS * head_dim;
        let k_seen = spread(pool, 5);
        let v_seen = spread(pool, 11);
        // One per HEAD and spread across the scores, so the shrink differs
        // head to head: a sink far below every score changes nothing and one
        // above them all halves the output.
        let sink_seen: Vec<f32> = (0..Q_HEADS).map(|h| h as f32 - 1.5).collect();

        let queries = alloc_bf16(&context, &q_seen, "queries");
        let k_pages = alloc_bf16(&context, &k_seen, "k_pages");
        let v_pages = alloc_bf16(&context, &v_seen, "v_pages");
        let sinks = alloc_bf16(&context, &sink_seen, "sinks");
        let position_ids = alloc_words(&context, &POSITIONS.map(|p| p as u32), "position_ids");
        let req_of_token = alloc_words(&context, &REQUESTS.map(|r| r as u32), "req_of_token");
        let kv_page_indices = alloc_words(&context, &INDICES, "kv_page_indices");
        let kv_page_indptr = alloc_words(&context, &INDPTR, "kv_page_indptr");
        // Every live fire in this tree binds the mask DISABLED, so the arm
        // that reads it is shipped, compiled and never taken. Row 1 takes it.
        // Rows 0 and 2 leave it off, which is the half that says the flag is
        // read per row: a body that masked everything once any row asked
        // fails on them, and a body that ignored the flag fails on row 1.
        let mut mask_bytes = vec![1u8; ROWS * MASK_STRIDE as usize];
        for (row, hole) in MASK_HOLES.iter().enumerate() {
            mask_bytes[row * MASK_STRIDE as usize + hole] = 0;
        }
        let attention_mask = alloc_bytes(&context, &mask_bytes, "mask");
        let attention_mask_enabled = alloc_bytes(&context, &MASK_ON, "mask_enabled");

        let buffers = [
            (0usize, queries.gpu_address()),
            (1, k_pages.gpu_address()),
            (2, v_pages.gpu_address()),
            (5, position_ids.gpu_address()),
            (6, req_of_token.gpu_address()),
            (7, kv_page_indices.gpu_address()),
            (8, kv_page_indptr.gpu_address()),
            (12, attention_mask.gpu_address()),
            (14, attention_mask_enabled.gpu_address()),
            (16, sinks.gpu_address()),
        ];

        let mut answers = Vec::new();
        for arm in &arms {
            let out = Allocation::new(
                &context,
                (ROWS * Q_HEADS * head_dim * 2) as u64,
                "attention out",
            )
            .expect("an output");
            let mut with_out = buffers.to_vec();
            with_out.push((3, out.gpu_address()));
            answers.push(fire(
                &context, &compiler, arm, head_dim, PAGE_SIZE, &with_out, &out,
            ));
        }

        let slot_of = |req: usize, kp: usize| -> usize {
            let phys = INDICES[INDPTR[req] as usize + kp / PAGE_SIZE] as usize;
            phys * PAGE_SIZE + kp % PAGE_SIZE
        };

        // The output is bfloat16: eight bits of significand, so the store
        // alone can move a value by 2^-9 of itself. The bound is that quantum
        // with room for a different summation order, and `tolerance_holds` at
        // the foot of the loop is what says it is not so loose as to accept
        // anything -- by measuring, which the check that used to stand there
        // did not.
        let bound = |want: f32| (want.abs() / 128.0).max(1.0 / 256.0);
        let mut worst = 0.0f32;

        for row in 0..ROWS {
            let req = REQUESTS[row] as usize;
            let q_pos = POSITIONS[row];
            let start = if WINDOW > 0 && q_pos >= WINDOW {
                q_pos - WINDOW + 1
            } else {
                0
            };
            let keeps: Vec<usize> = (start..=q_pos)
                .map(|kp| kp as usize)
                .filter(|kp| {
                    MASK_ON[row] == 0
                        || (*kp < MASK_STRIDE as usize
                            && mask_bytes[row * MASK_STRIDE as usize + *kp] != 0)
                })
                .collect();
            assert!(
                keeps.len() > 1,
                "row {row} keeps {} keys; a softmax over one key is that key \
                 whatever weight it computed, so the row would prove nothing",
                keeps.len()
            );
            for (q_head, &sink_of) in sink_seen.iter().enumerate() {
                let kv_head = q_head / GQA;
                let q_base = (row * Q_HEADS + q_head) * head_dim;
                let scores: Vec<f32> = keeps
                    .iter()
                    .map(|kp| {
                        let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                        (0..head_dim)
                            .map(|d| SCALE * q_seen[q_base + d] * k_seen[base + d])
                            .sum()
                    })
                    .collect();
                let planes: Vec<&[f32]> = keeps
                    .iter()
                    .map(|kp| {
                        let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                        &v_seen[base..base + head_dim]
                    })
                    .collect();
                let plain = reference(&scores, &planes, head_dim, None);
                let sunk = reference(&scores, &planes, head_dim, Some(sink_of));

                for (arm, got) in arms.iter().zip(&answers) {
                    let want = if arm.sunk { &sunk } else { &plain };
                    for d in 0..head_dim {
                        let seen = got[q_base + d];
                        worst = worst.max((seen - want[d]).abs() / bound(want[d]));
                        assert!(
                            (seen - want[d]).abs() <= bound(want[d]),
                            "{} at d={head_dim}: row {row} head {q_head} channel \
                             {d} is {seen} and the reference is {}",
                            arm.entrypoint,
                            want[d],
                        );
                    }
                }

                // The DIFFERENTIAL. Every arm read identical bytes, so two of
                // them may differ only by the order they summed in — which the
                // bfloat16 store rounds away almost always and never amplifies.
                // This is the claim a reference tolerance cannot make.
                for (arm, got) in arms.iter().zip(&answers) {
                    let (first, base) = arms
                        .iter()
                        .zip(&answers)
                        .find(|(other, _)| other.sunk == arm.sunk)
                        .expect("an arm agrees with its own half");
                    for d in 0..head_dim {
                        let a = base[q_base + d];
                        let b = got[q_base + d];
                        assert!(
                            (a - b).abs() <= (a.abs() / 128.0).max(1.0 / 256.0),
                            "{} and {} read the same bytes and answered {a} and \
                             {b} at d={head_dim} row {row} head {q_head} \
                             channel {d}",
                            first.entrypoint,
                            arm.entrypoint,
                        );
                    }
                }

                // The sink's DIRECTION, for every arm that reads it. A sink
                // joins the softmax with no value behind it: it adds to the
                // denominator and nothing to the numerator, so it can only
                // shrink an output toward zero, by one factor per head. A body
                // that divided by the sink instead would pass the value check
                // wherever the tolerance was loose and fails here always.
                for (arm, got) in arms.iter().zip(&answers) {
                    if !arm.sunk {
                        continue;
                    }
                    for d in 0..head_dim {
                        let with = got[q_base + d];
                        let without = plain[d];
                        assert!(
                            with.abs() <= without.abs() + bound(without),
                            "{}: row {row} head {q_head} channel {d} moved \
                             {without} to {with}, which is AWAY from zero",
                            arm.entrypoint,
                        );
                    }
                }
            }
        }

        tolerance_holds(worst, &format!("the paged differential at d={head_dim}"));

        // And the sink actually moved the head it was largest on, or the two
        // halves were always going to agree and the sink arms proved nothing.
        let Some(sunk_at) = arms.iter().position(|a| a.sunk) else {
            continue;
        };
        let hot = (Q_HEADS - 1) * head_dim;
        let moved = (0..head_dim)
            .filter(|d| answers[sunk_at][hot + d] != answers[0][hot + d])
            .count();
        assert!(
            moved > head_dim / 2,
            "the head with the largest sink ({}) moved only {moved} of \
             {head_dim} channels at d={head_dim}; pick a sink comparable with \
             the scores or the sink arms prove nothing",
            sink_seen[Q_HEADS - 1],
        );
    }
}

fn alloc_bf16(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let words: Vec<u16> = values.iter().copied().map(bf16).collect();
    let bytes = std::mem::size_of_val(words.as_slice()) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(0, cast(&words)).expect("the values fit");
    }
    a
}

fn alloc_words(context: &Context, values: &[u32], what: &'static str) -> Allocation {
    let bytes = std::mem::size_of_val(values) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(
            0,
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes as usize),
        )
        .expect("the words fit");
    }
    a
}

fn alloc_bytes(context: &Context, values: &[u8], what: &'static str) -> Allocation {
    let a = Allocation::new(context, (values.len() as u64).max(4), what).expect("an allocation");
    unsafe {
        a.write(0, values).expect("the bytes fit");
    }
    a
}

fn cast(v: &[u16]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// The page-shape tails, where the window and the mask are COMPILED OUT.
///
/// `sdpa_paged_decode`'s row states seven axis points under one heading,
/// "head dim and page shape", and states `window` and `attention_mask` among
/// its operands. Three of those points do not read either. `_p32`,
/// `_d_128_p32` and `_p32_sg8` instantiate the template with `FAST_FULL =
/// true`, and inside the kernel that constant does two things: `kv_start`
/// becomes `0` unconditionally, so the sliding window is gone, and the mask
/// test is behind `if constexpr (!FAST_FULL)`, so the mask is gone. They are
/// not a page shape of the same kernel. They are FULL ATTENTION, and the
/// axis says otherwise.
///
/// Nothing selects them. A text names `sdpa_paged_decode_bfloat16_d_<width>`
/// from its row's head dim and `METAL_SDPA_HEAD_DIMS` is the plain widths
/// alone, so no statement in this tree can spell a page tail — they are
/// compiled into every build, signed, pipeline-buildable, and unreachable.
/// That is why the divergence has never bitten: not because it is safe, but
/// because the door is shut. `sdpa_paged_tiled_strided` is dark for the same
/// class of reason and the ledger says so; these three are invisible to that
/// ledger because the ROW they belong to is live.
///
/// So this test is the door being opened once, on purpose. It binds a window
/// of 4 and an enabled mask with holes, and asserts the answer is the one
/// computed over the WHOLE history with no mask at all. Under the plain
/// points that assertion is false by a mile — which is the point. The same
/// test fires the PLAIN point of the same width over the same fixture and
/// requires it to DISAGREE, which is what says the fixture has a window and a
/// mask worth ignoring.
///
/// The page walk is the other half. `FAST_FULL` carries an incremental page
/// cursor instead of dividing `kp` -- `fast_page_off` starts at the
/// simdgroup, strides by `BN` and rolls into the next index -- which is a
/// different way of arriving at a slot than the plain path's `kp >> 5`. The
/// page table is reversed so the two cannot agree by accident, and `_sg8`
/// rolls over every fourth step where `_p32` rolls every step.
const PAGE_32: usize = 32;
const PAGES_32: usize = 4;
const INDPTR_32: [u32; 3] = [0, 3, 4];
const INDICES_32: [u32; PAGES_32] = [3, 1, 2, 0];
const POSITIONS_32: [i32; ROWS] = [40, 70, 5];
const REQUESTS_32: [i32; ROWS] = [0, 0, 1];

#[test]
#[ignore = "needs a Metal 4 device"]
fn a_page_shaped_decode_answers_the_whole_history() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    // The third entry's THREADGROUP is 256 and not 1024, and that is not a
    // tuning choice this test made. `BN` is the number of simdgroups the
    // online softmax strides keys by and the threadgroup is `BN * BD` with
    // `BD == 32`; `_p32_sg8` instantiates `BN = 8`. The header says what
    // happens if the launch disagrees -- "missing simdgroups mean missing
    // keys, not a smaller version of the same work" -- and at 1024 this point
    // answers 1.0546875 where the reference is -0.1139, because `simd_gid`
    // runs to 31 over threadgroup arrays sized for 8.
    //
    // `LaunchRule::SdpaVector` is 1024 threads for every point of this row.
    // So the launch geometry is a THIRD thing the page-shape axis silently
    // varies, after the window and the mask, and the one point that needs a
    // different one is the one nothing can select.
    for (head_dim, tail, threads) in [
        (64usize, "_p32", 1024u32),
        (128, "_p32", 1024),
        (64, "_p32_sg8", 256),
    ] {
        let entrypoint = format!("sdpa_paged_decode_bfloat16_d_{head_dim}{tail}");
        let arm = Arm {
            entrypoint: entrypoint.clone(),
            file: "attn/sdpa_paged.metal",
            threads,
            tiled: false,
            sunk: false,
            pitch: None,
        };
        // The plain point over the SAME fixture. It reads the window and the
        // mask this one ignores, so it is the control: if it agreed with the
        // full-history reference the fixture would be proving nothing.
        let plain = Arm::of("sdpa_paged_decode", head_dim);

        let q_seen = spread(ROWS * Q_HEADS * head_dim, 1);
        let pool = PAGES_32 * PAGE_32 * KV_HEADS * head_dim;
        let k_seen = spread(pool, 5);
        let v_seen = spread(pool, 11);

        let queries = alloc_bf16(&context, &q_seen, "queries");
        let k_pages = alloc_bf16(&context, &k_seen, "k_pages");
        let v_pages = alloc_bf16(&context, &v_seen, "v_pages");
        let sinks = alloc_bf16(&context, &[0.0f32; Q_HEADS], "sinks");
        let position_ids = alloc_words(&context, &POSITIONS_32.map(|p| p as u32), "position_ids");
        let req_of_token = alloc_words(&context, &REQUESTS_32.map(|r| r as u32), "req_of_token");
        let kv_page_indices = alloc_words(&context, &INDICES_32, "kv_page_indices");
        let kv_page_indptr = alloc_words(&context, &INDPTR_32, "kv_page_indptr");
        // ENABLED, on every row, with a hole inside every row's history. A
        // point that read this drops keys; the reference keeps them.
        let mut mask_bytes = vec![1u8; ROWS * MASK_STRIDE as usize];
        for (row, hole) in MASK_HOLES.iter().enumerate() {
            mask_bytes[row * MASK_STRIDE as usize + hole] = 0;
        }
        let attention_mask = alloc_bytes(&context, &mask_bytes, "mask");
        let attention_mask_enabled = alloc_bytes(&context, &[1u8; ROWS], "mask_enabled");

        let buffers = [
            (0usize, queries.gpu_address()),
            (1, k_pages.gpu_address()),
            (2, v_pages.gpu_address()),
            (5, position_ids.gpu_address()),
            (6, req_of_token.gpu_address()),
            (7, kv_page_indices.gpu_address()),
            (8, kv_page_indptr.gpu_address()),
            (12, attention_mask.gpu_address()),
            (14, attention_mask_enabled.gpu_address()),
            (16, sinks.gpu_address()),
        ];

        let mut answers = Vec::new();
        for a in [&arm, &plain] {
            let out = Allocation::new(
                &context,
                (ROWS * Q_HEADS * head_dim * 2) as u64,
                "attention out",
            )
            .expect("an output");
            let mut with_out = buffers.to_vec();
            with_out.push((3, out.gpu_address()));
            answers.push(fire(
                &context, &compiler, a, head_dim, PAGE_32, &with_out, &out,
            ));
        }

        let slot_of = |req: usize, kp: usize| -> usize {
            let phys = INDICES_32[INDPTR_32[req] as usize + kp / PAGE_32] as usize;
            phys * PAGE_32 + kp % PAGE_32
        };
        let bound = |want: f32| (want.abs() / 128.0).max(1.0 / 256.0);
        let mut worst = 0.0f32;
        let mut controlled = 0usize;

        for row in 0..ROWS {
            let req = REQUESTS_32[row] as usize;
            let q_pos = POSITIONS_32[row];
            // The WHOLE history: no window, no mask. That is the claim.
            let keeps: Vec<usize> = (0..=q_pos).map(|kp| kp as usize).collect();
            for q_head in 0..Q_HEADS {
                let kv_head = q_head / GQA;
                let q_base = (row * Q_HEADS + q_head) * head_dim;
                let scores: Vec<f32> = keeps
                    .iter()
                    .map(|kp| {
                        let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                        (0..head_dim)
                            .map(|d| SCALE * q_seen[q_base + d] * k_seen[base + d])
                            .sum()
                    })
                    .collect();
                let planes: Vec<&[f32]> = keeps
                    .iter()
                    .map(|kp| {
                        let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                        &v_seen[base..base + head_dim]
                    })
                    .collect();
                let want = reference(&scores, &planes, head_dim, None);

                for d in 0..head_dim {
                    let got = answers[0][q_base + d];
                    worst = worst.max((got - want[d]).abs() / bound(want[d]));
                    assert!(
                        (got - want[d]).abs() <= bound(want[d]),
                        "{entrypoint}: row {row} head {q_head} channel {d} is {got} and the \
                         whole-history reference is {}; `FAST_FULL` is supposed to compile the \
                         window and the mask out, so this point answers over every key",
                        want[d]
                    );
                    if (answers[1][q_base + d] - want[d]).abs() > bound(want[d]) {
                        controlled += 1;
                    }
                }
            }
        }

        tolerance_holds(worst, &format!("the page-shaped decode at d={head_dim}"));

        // The control. The plain point over this same fixture reads the
        // window and the mask, so it must DISAGREE with the whole-history
        // reference -- otherwise the fixture has no window and no mask worth
        // ignoring and the test above is vacuous.
        assert!(
            controlled > ROWS * Q_HEADS,
            "`sdpa_paged_decode_bfloat16_d_{head_dim}` agreed with the whole-history reference \
             on all but {controlled} channels, so this fixture cannot tell a windowed masked \
             read from a full one and `{entrypoint}` ignoring them proves nothing"
        );
    }
}

/// The last entry point of the family, and the only one with a pitch.
///
/// `sdpa_paged_tiled_strided` is the tiled body with `queries` and `out` rows
/// a uniform distance apart instead of packed. The DARK ledger carried it as
/// "no statement produces a row pitch", which was true and was also the whole
/// of what was known about it: it was compiled at `_d_256` alone, signed into
/// every build, and no test on any backend had ever asked it for a number.
/// Dark for want of a caller is not the same as untrusted, and the difference
/// is one dispatch.
///
/// It has since left that ledger -- `attn.rs::sdpa_paged_tiled_strided` is a
/// routine and `PAGED_TILED_STRIDED` its one point -- so the sentence above
/// is history and not a standing description. `routine::DARK` is one row now
/// and the row is `silu_mul_strided`. The dispatch below is why this one is
/// no longer on it.
///
/// The pitch is a PAD, not a reshape: rows are `Q_HEADS * D + PAD` apart and
/// the pad is filled with a value that would wreck any answer that read it —
/// so this fires two claims at once. The rows must be the same softmax the
/// packed body computes, and the pad must come back exactly as written, which
/// is what says the kernel did not stride into its neighbour. The prefill this
/// kernel exists for lays rows `scratch_widest_elems` apart, wider than
/// `n_q_heads * D`, and the shader's own comment names the failure it is
/// avoiding: "a kernel that assumed packed would walk into the next row's
/// tensor and read plausible garbage rather than fail". Plausible garbage is
/// what a pad of poison turns into an assertion.
const PAD: usize = 11;
const POISON: f32 = -7.5;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_strided_tile_reads_its_rows_a_pitch_apart() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let head_dim = 256usize;
    let packed = Q_HEADS * head_dim;
    let pitch = packed + PAD;
    let arm = Arm {
        entrypoint: format!("sdpa_paged_tiled_strided_bfloat16_d_{head_dim}"),
        file: "attn/sdpa_paged.metal",
        threads: 1024,
        tiled: true,
        sunk: false,
        pitch: Some((pitch as u32, pitch as u32)),
    };
    // The packed twin over the same keys, which is the reference's second
    // opinion: `sdpa_paged_tiled` at the same width reads the same pool and
    // must answer the same thing from a layout with no pad in it.
    let twin = Arm::of("sdpa_paged_tiled", head_dim);

    let dense = spread(ROWS * packed, 1);
    let mut strided = vec![POISON; ROWS * pitch];
    for row in 0..ROWS {
        strided[row * pitch..row * pitch + packed]
            .copy_from_slice(&dense[row * packed..(row + 1) * packed]);
    }

    let pool = PAGES * PAGE_SIZE * KV_HEADS * head_dim;
    let k_seen = spread(pool, 5);
    let v_seen = spread(pool, 11);

    let k_pages = alloc_bf16(&context, &k_seen, "k_pages");
    let v_pages = alloc_bf16(&context, &v_seen, "v_pages");
    let sinks = alloc_bf16(&context, &[0.0f32; Q_HEADS], "sinks");
    let position_ids = alloc_words(&context, &POSITIONS.map(|p| p as u32), "position_ids");
    let req_of_token = alloc_words(&context, &REQUESTS.map(|r| r as u32), "req_of_token");
    let kv_page_indices = alloc_words(&context, &INDICES, "kv_page_indices");
    let kv_page_indptr = alloc_words(&context, &INDPTR, "kv_page_indptr");
    let mut mask_bytes = vec![1u8; ROWS * MASK_STRIDE as usize];
    for (row, hole) in MASK_HOLES.iter().enumerate() {
        mask_bytes[row * MASK_STRIDE as usize + hole] = 0;
    }
    let attention_mask = alloc_bytes(&context, &mask_bytes, "mask");
    let attention_mask_enabled = alloc_bytes(&context, &MASK_ON, "mask_enabled");

    let shared = [
        (1usize, k_pages.gpu_address()),
        (2, v_pages.gpu_address()),
        (5, position_ids.gpu_address()),
        (6, req_of_token.gpu_address()),
        (7, kv_page_indices.gpu_address()),
        (8, kv_page_indptr.gpu_address()),
        (12, attention_mask.gpu_address()),
        (14, attention_mask_enabled.gpu_address()),
        (16, sinks.gpu_address()),
    ];

    let mut answers = Vec::new();
    for a in [&arm, &twin] {
        let wide = a.pitch.map_or(packed, |(q, _)| q as usize);
        let q_seen: &[f32] = if a.pitch.is_some() { &strided } else { &dense };
        let queries = alloc_bf16(&context, q_seen, "queries");
        // Pre-poisoned, so a pad the kernel leaves alone reads back as poison
        // and a pad it writes reads back as an answer.
        let out = alloc_bf16(&context, &vec![POISON; ROWS * wide], "attention out");
        let mut with_out = shared.to_vec();
        with_out.push((0, queries.gpu_address()));
        with_out.push((3, out.gpu_address()));
        answers.push(fire(
            &context, &compiler, a, head_dim, PAGE_SIZE, &with_out, &out,
        ));
    }

    let slot_of = |req: usize, kp: usize| -> usize {
        let phys = INDICES[INDPTR[req] as usize + kp / PAGE_SIZE] as usize;
        phys * PAGE_SIZE + kp % PAGE_SIZE
    };
    let bound = |want: f32| (want.abs() / 128.0).max(1.0 / 256.0);
    let mut worst = 0.0f32;

    for row in 0..ROWS {
        let req = REQUESTS[row] as usize;
        let q_pos = POSITIONS[row];
        let start = if WINDOW > 0 && q_pos >= WINDOW {
            q_pos - WINDOW + 1
        } else {
            0
        };
        let keeps: Vec<usize> = (start..=q_pos)
            .map(|kp| kp as usize)
            .filter(|kp| {
                MASK_ON[row] == 0
                    || (*kp < MASK_STRIDE as usize
                        && mask_bytes[row * MASK_STRIDE as usize + *kp] != 0)
            })
            .collect();
        for q_head in 0..Q_HEADS {
            let kv_head = q_head / GQA;
            let q_base = (row * Q_HEADS + q_head) * head_dim;
            let scores: Vec<f32> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| SCALE * dense[q_base + d] * k_seen[base + d])
                        .sum()
                })
                .collect();
            let planes: Vec<&[f32]> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * head_dim;
                    &v_seen[base..base + head_dim]
                })
                .collect();
            let want = reference(&scores, &planes, head_dim, None);

            for (d, &w) in want.iter().enumerate() {
                let at = row * pitch + q_head * head_dim + d;
                let got = answers[0][at];
                worst = worst.max((got - w).abs() / bound(w));
                assert!(
                    (got - w).abs() <= bound(w),
                    "{}: row {row} head {q_head} channel {d} is {got} and the reference is {}; \
                     the rows are {pitch} elements apart and the packed layout would put this \
                     channel at {}",
                    arm.entrypoint,
                    w,
                    row * packed + q_head * head_dim + d
                );
                let twin_at = row * packed + q_head * head_dim + d;
                worst = worst.max((answers[1][twin_at] - got).abs() / bound(w));
                assert!(
                    (answers[1][twin_at] - got).abs() <= bound(w),
                    "`sdpa_paged_tiled_strided` answers {got} where its packed twin answers {} \
                     at row {row} head {q_head} channel {d}; one body, two layouts, and the \
                     layout is not supposed to be visible in the number",
                    answers[1][twin_at]
                );
            }
        }
    }

    tolerance_holds(worst, "the strided tile");

    // The pad. Every element between one row's last channel and the next
    // row's first must still be poison: the kernel writes `o_row_pitch` apart
    // and owns `Q_HEADS * D` of every stride, and the rest belongs to whoever
    // laid the scratch out.
    for row in 0..ROWS {
        for d in 0..PAD {
            let at = row * pitch + packed + d;
            assert!(
                (answers[0][at] - POISON).abs() < 1.0 / 256.0,
                "element {d} of the pad after row {row} is {} and was written as {POISON}; the \
                 strided body wrote past the {packed} channels it owns",
                answers[0][at]
            );
        }
    }

    // And the pitch has to MATTER. Row 1 of the strided answer lives at
    // `pitch`, so reading it where the PACKED layout would put it -- at
    // `packed` -- lands in the pad and then in row 1's leading channels,
    // shifted by `PAD`. If that agreed with the packed answer the pitch would
    // not be separating the two layouts and every claim above would hold just
    // as well with `PAD == 0`.
    let moved = (0..packed)
        .filter(|d| (answers[0][packed + d] - answers[1][packed + d]).abs() > 1.0 / 256.0)
        .count();
    assert!(
        moved > packed / 4,
        "reading the strided answer at the PACKED offset agreed with the packed answer on all \
         but {moved} of {packed} channels, so this fixture's pitch is not separating the two \
         layouts and the pad proves nothing"
    );
}
