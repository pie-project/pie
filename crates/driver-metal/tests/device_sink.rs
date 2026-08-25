//! The lse plane and the sink rescale: `attention.{decode_lse, prefill_lse,
//! sink}`, the three points gpt-oss states that this plane could not answer.
//!
//! # What had no proof, and what this file is not
//!
//! `device_attention` compares the paged bodies' OUTPUTS. Every one of them
//! divides by its denominator on the way out, so the denominator itself was
//! never read by anything: a body could have computed the right output from
//! the wrong `sum_exp_score` -- by an offsetting error in `max_score`, which
//! the online softmax cancels between the two -- and passed every check in
//! that file. `attention.decode_lse` makes the denominator an OPERAND of the
//! next statement, so from here it is a value and not an accumulator.
//!
//! It is also the sharper probe of the two. An output is a convex combination
//! of the values it kept, so a body that dropped a key whose value happened to
//! sit near the mean of the others answers almost the same vector. The lse is
//! the SUM over the kept set: drop a key and it moves by that key's whole
//! weight, whatever its value was. That is why the fixture below keeps the
//! reversed page table, the window and the per-row mask -- the same three
//! things `device_attention` uses to make a key set observable, aimed at the
//! output that cannot average one away.
//!
//! # THE BASE, which is the whole of `attention.sink`
//!
//! An lse is a logarithm and a logarithm has a base. The floor states BASE
//! TWO -- flashinfer's, because its host folds `log2(e)` into `sm_scale` and
//! its kernels run on `exp2`, and that is the plane whose kernel this tree
//! does not own. THIS plane's softmax runs on `fast::exp`, natural log, so
//! `sdpa_lse_base2` rebases on the way out and the number that reaches the
//! next statement is the one the point promises.
//!
//! `attention.sink` is then the one point where two bases meet: a base-two
//! normaliser against `gpt-oss`'s `self_attn.sinks`, which HF wrote in natural
//! log. The `ln(2)` in `attn/attn_sink.metal` is not housekeeping in front of
//! the kernel; it IS the kernel's job, and dropping it is a factor of 0.693 in
//! a sigmoid argument -- which matches HF's top-1 on most prompts by accident
//! and then drifts. So the reference for that point is not a CPU model this
//! file wrote: it is the bf16 bytes `pie::attn::attn_sink_rescale` -- the CUDA
//! kernel the cuda plane serves `attention.sink` with -- produced on an L40S
//! from `kernels-cuda/kernels/attn/attn_sink.cuh`, carried over as
//! [`CUDA_O_OUT`]. A reference computed from the code that ships is worth more
//! than one computed from the same reading of the header twice.
//!
//! # And the question the third test settles
//!
//! `attn/sdpa_paged.metal` has `_sink` entry points that fold the same scalar
//! into the softmax denominator BEFORE the division, and they answer the same
//! numbers -- [`a_published_lse_rescaled_is_the_folded_sink_arm`] measures
//! exactly that, and it holds. It is still not the same statement: a folded
//! arm writes no lse, `attention.decode_lse` DECLARES one, and a point is a
//! contract about what is written. The two readings agreeing is what makes
//! this a decomposition rather than a rewrite; the declaration is what decides
//! which one a text may state.
//!
//! # Every check here can fail
//!
//! Two of the tests below are fault injections: they compile the shipped
//! shader with one constant changed and assert the comparison goes RED. A
//! check that cannot fail is worth nothing, and these are cheap because
//! `layout::shader::read_source` resolves includes from a directory -- so a
//! copy of `attn/` with one line different is a whole shader tree.

use std::path::{Path, PathBuf};

use driver_metal::baker::dispatch::{Dispatch, ParamSlot, Touches};
use driver_metal::baker::{BoundRegion as BoundArg, Slice};
use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;
use driver_metal::skip::skipped;

/// How much of its own tolerance the worst element used, and the band that has
/// to hold. The argument is `device_attention`'s and is not repeated here: a
/// bound nothing measured is a bound a wrong kernel hides in, and `4b > b` is
/// true of every bound anyone will ever write.
fn tolerance_holds(worst: f32, what: &str) {
    if worst == 0.0 {
        return;
    }
    assert!(
        worst <= 1.0,
        "{what}: the worst element used {worst} of its bound, so an assertion \
         above passed by an accident of iteration order"
    );
    assert!(
        worst >= 0.125,
        "{what}: the worst element used only {worst} of its bound, so the \
         tolerance is more than eight times the arithmetic this device \
         actually delivers -- tighten the bound instead of trusting it"
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

/// Values bfloat16 holds exactly: multiples of an eighth in `[-1, 1)`, period
/// fifteen. Fifteen is coprime with 64 and with every stride in the pool, so
/// no two slots alias and a body that walked the page table wrong is visible.
/// See `device_attention`'s note for what a period of sixteen cost.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed) % 15) as f32 / 8.0 - 1.0)
        .collect()
}

// ── the attention fixture ───────────────────────────────────────────────────
//
// `device_attention`'s, deliberately: the same reversed page table, the same
// two requests sharing a pool, the same three positions against a window of
// four, and the same per-row mask with two separate drop clauses. Sharing the
// numbers is what lets a disagreement between the two files be about the lse
// and not about the fixture.

/// The one width the `_lse` arms are stamped at. gpt-oss is the only family
/// that states these points on this plane and its heads are 64 wide; see the
/// instantiation block in `attn/sdpa_paged.metal`.
const HEAD_DIM: usize = 64;
const Q_HEADS: usize = 4;
const GQA: usize = 2;
const KV_HEADS: usize = Q_HEADS / GQA;
const ROWS: usize = 3;
const PAGE_SIZE: usize = 3;
const PAGES: usize = 7;
const SCALE: f32 = 0.125;
const WINDOW: i32 = 4;
const MASK_STRIDE: u32 = 9;
const MASK_ON: [u8; ROWS] = [0, 1, 0];
const MASK_HOLES: [usize; ROWS] = [4, 7, 1];
const POSITIONS: [i32; ROWS] = [6, 9, 2];
const REQUESTS: [i32; ROWS] = [0, 0, 1];
const INDPTR: [u32; 3] = [0, 4, 7];
const INDICES: [u32; PAGES] = [6, 4, 2, 0, 5, 3, 1];

/// What an unwritten lse slot holds, so "the kernel never got here" is a
/// different failure from "the kernel got here and computed the wrong thing".
const LSE_SENTINEL: f32 = -12345.0;

/// One of the four sinks per head, spread across the scores: one far below
/// every score (which changes nothing), one above them all (which roughly
/// halves the output), and two in between. A sink nothing shrinks proves
/// nothing, which `device_attention` records having found out.
fn sink_seen() -> Vec<f32> {
    (0..Q_HEADS).map(|h| h as f32 - 1.5).collect()
}

/// A shader tree, possibly with one line of it changed.
///
/// `read_source` splices `#include "..."` relative to the including file's
/// own directory, so a directory holding `attn/` IS a shader tree as far as
/// the compiler is concerned. A fault injection is then a copy with a
/// replacement applied -- and the replacement is asserted to have LANDED,
/// because a patch that matched nothing would leave the shipped kernel in
/// place and the test would pass while proving the opposite of what it says.
struct Tree {
    root: PathBuf,
    _held: Option<tempfile::TempDir>,
}

impl Tree {
    fn shipped() -> Self {
        Self {
            root: kernels_dir(),
            _held: None,
        }
    }

    fn patched(file: &str, from: &str, to: &str) -> Self {
        let held = tempfile::tempdir().expect("a temp dir");
        let attn = held.path().join("attn");
        std::fs::create_dir_all(&attn).expect("attn/");
        let mut landed = false;
        for entry in std::fs::read_dir(kernels_dir().join("attn")).expect("attn/ is readable") {
            let path = entry.expect("an entry").path();
            let name = path
                .file_name()
                .expect("a name")
                .to_string_lossy()
                .to_string();
            let text = std::fs::read_to_string(&path).expect("a shader reads");
            let text = if name == file {
                assert!(
                    text.contains(from),
                    "the injection `{from}` matches nothing in `{file}`, so it \
                     would have left the shipped kernel in place and the test \
                     would have proved the opposite of what it says"
                );
                landed = true;
                text.replace(from, to)
            } else {
                text
            };
            std::fs::write(attn.join(&name), text).expect("the copy writes");
        }
        assert!(landed, "`{file}` is not in `attn/`");
        Self {
            root: held.path().to_path_buf(),
            _held: Some(held),
        }
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

/// One dispatch, encoded and run. The caller states the whole argument table
/// and the scalars' slots, because the families in this file place their
/// scalars among their buffers rather than past them.
struct Launch<'a> {
    entrypoint: String,
    file: &'static str,
    grid: [u32; 3],
    threadgroup: [u32; 3],
    args_wide: usize,
    buffers: &'a [(usize, u64)],
    /// `(slot, value)` for each scalar, in the order they are staged.
    scalars: &'a [(usize, u32)],
}

fn run(context: &Context, tree: &Tree, launch: &Launch<'_>) {
    let compiler = driver_metal::program::Compiler::new(context).expect("a compiler");
    let placeholder = BoundArg {
        slice: Slice {
            address: launch.buffers.first().expect("at least one buffer").1,
            bytes: 1 << 20,
        },
        width: 0,
    };
    let mut args = vec![placeholder; launch.args_wide];
    for (slot, address) in launch.buffers {
        args[*slot] = BoundArg {
            slice: Slice {
                address: *address,
                bytes: 1 << 20,
            },
            width: 0,
        };
    }

    let params: Vec<u32> = launch.scalars.iter().map(|(_, v)| *v).collect();
    let param_slots: Vec<ParamSlot> = launch
        .scalars
        .iter()
        .enumerate()
        .map(|(at, (slot, _))| ParamSlot {
            slot: *slot,
            at: (at * 4) as u32,
            bytes: 4,
            // WHICH of this statement's scalars. `Some(0)` on every slot
            // stages the first one everywhere, which reads as `scale = 0` and
            // answers a softmax with every score equal.
            value: u8::try_from(at).expect("few scalars"),
        })
        .collect();

    let dispatch = Dispatch {
        // LEAKED: `Dispatch::symbol` is `&'static str` because a claim body
        // names an entry point as a literal. A test that builds its table at
        // run time pays one leak per launch.
        symbol: String::leak(launch.entrypoint.clone()),
        file: launch.file,
        stamp: "",
        grid: launch.grid,
        threadgroup: launch.threadgroup,
        touches: Touches::everything(&args),
        args,
        params,
        param_slots,
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(tree.root().to_path_buf());
    pipelines
        .ensure(context, &compiler, std::slice::from_ref(&dispatch))
        .unwrap_or_else(|why| panic!("`{}` builds a pipeline: {why}", launch.entrypoint));
    let staged =
        Params::stage(context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table =
        ArgumentTable::new(context, launch.args_wide).expect("a table as wide as the launch");
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
        .unwrap_or_else(|why| panic!("`{}` fires: {why}", launch.entrypoint));
}

/// The K/V pool, the page table and the launch operands every attention fire
/// in this file shares. Held together because the addresses have to outlive
/// the dispatch that names them.
struct Fixture {
    q_seen: Vec<f32>,
    k_seen: Vec<f32>,
    v_seen: Vec<f32>,
    mask_bytes: Vec<u8>,
    mask_on: [u8; ROWS],
    buffers: Vec<(usize, u64)>,
    #[allow(dead_code)]
    held: Vec<Allocation>,
}

fn fixture(context: &Context) -> Fixture {
    let mut mask_bytes = vec![1u8; ROWS * MASK_STRIDE as usize];
    for (row, hole) in MASK_HOLES.iter().enumerate() {
        mask_bytes[row * MASK_STRIDE as usize + hole] = 0;
    }
    fixture_masked(context, mask_bytes, MASK_ON)
}

/// The same pool and page table, with the mask stated by the caller.
///
/// The ONE row shape this fixture's own numbers cannot reach is a row that
/// keeps nothing: a query's own K is appended before attention, so `kp =
/// q_pos` survives every causal bound and every window, and only a mask can
/// take it away. That row is where `sum_exp_score` is zero and where an lse
/// written as `m + log(sum)` is a NaN rather than the `-inf` flashinfer
/// publishes -- so it gets a mask of its own below.
fn fixture_masked(context: &Context, mask_bytes: Vec<u8>, mask_on: [u8; ROWS]) -> Fixture {
    let q_seen = spread(ROWS * Q_HEADS * HEAD_DIM, 1);
    let pool = PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM;
    let k_seen = spread(pool, 5);
    let v_seen = spread(pool, 11);

    let queries = alloc_bf16(context, &q_seen, "queries");
    let k_pages = alloc_bf16(context, &k_seen, "k_pages");
    let v_pages = alloc_bf16(context, &v_seen, "v_pages");
    let sinks = alloc_bf16(context, &sink_seen(), "sinks");
    let position_ids = alloc_words(context, &POSITIONS.map(|p| p as u32), "position_ids");
    let req_of_token = alloc_words(context, &REQUESTS.map(|r| r as u32), "req_of_token");
    let kv_page_indices = alloc_words(context, &INDICES, "kv_page_indices");
    let kv_page_indptr = alloc_words(context, &INDPTR, "kv_page_indptr");

    let attention_mask = alloc_bytes(context, &mask_bytes, "mask");
    let attention_mask_enabled = alloc_bytes(context, &mask_on, "mask_enabled");

    let buffers = vec![
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

    Fixture {
        q_seen,
        k_seen,
        v_seen,
        mask_bytes,
        mask_on,
        buffers,
        held: vec![
            queries,
            k_pages,
            v_pages,
            sinks,
            position_ids,
            req_of_token,
            kv_page_indices,
            kv_page_indptr,
            attention_mask,
            attention_mask_enabled,
        ],
    }
}

/// The scalars the paged family states, at the slots the shaders read them
/// from. `n_rows` is stated by the tiled entry points only.
fn scalars(tiled: bool) -> Vec<(usize, u32)> {
    let mut out = vec![
        (4usize, GQA as u32),
        (9, PAGE_SIZE as u32),
        (10, KV_HEADS as u32),
        (11, SCALE.to_bits()),
        (13, MASK_STRIDE),
        (15, WINDOW as u32),
    ];
    if tiled {
        out.push((17, ROWS as u32));
    }
    out
}

/// What each row keeps, by this file's own reading of the fixture: the causal
/// bound, the window, and the mask's two drop clauses.
fn keeps(f: &Fixture, row: usize) -> Vec<usize> {
    let q_pos = POSITIONS[row];
    let start = if WINDOW > 0 && q_pos >= WINDOW {
        q_pos - WINDOW + 1
    } else {
        0
    };
    (start..=q_pos)
        .map(|kp| kp as usize)
        .filter(|kp| {
            f.mask_on[row] == 0
                || (*kp < MASK_STRIDE as usize
                    && f.mask_bytes[row * MASK_STRIDE as usize + *kp] != 0)
        })
        .collect()
}

fn slot_of(req: usize, kp: usize) -> usize {
    let phys = INDICES[INDPTR[req] as usize + kp / PAGE_SIZE] as usize;
    phys * PAGE_SIZE + kp % PAGE_SIZE
}

/// The scores of one `(row, head)` over the keys that row keeps.
fn scores(f: &Fixture, row: usize, q_head: usize) -> Vec<f32> {
    let req = REQUESTS[row] as usize;
    let kv_head = q_head / GQA;
    let q_base = (row * Q_HEADS + q_head) * HEAD_DIM;
    keeps(f, row)
        .iter()
        .map(|kp| {
            let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * HEAD_DIM;
            (0..HEAD_DIM)
                .map(|d| SCALE * f.q_seen[q_base + d] * f.k_seen[base + d])
                .sum()
        })
        .collect()
}

/// Softmax attention over the kept keys, and the log-sum-exp of the
/// denominator that normalised it -- IN BASE TWO, which is what the point
/// states. `log2(sum) + m*log2(e)` and not `ln(...)`: see the module header.
fn reference(f: &Fixture, row: usize, q_head: usize) -> (Vec<f32>, f32) {
    let req = REQUESTS[row] as usize;
    let kv_head = q_head / GQA;
    let s = scores(f, row, q_head);
    let planes: Vec<&[f32]> = keeps(f, row)
        .iter()
        .map(|kp| {
            let base = (slot_of(req, *kp) * KV_HEADS + kv_head) * HEAD_DIM;
            &f.v_seen[base..base + HEAD_DIM]
        })
        .collect();
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = w.iter().sum();
    let o = (0..HEAD_DIM)
        .map(|d| w.iter().zip(&planes).map(|(a, v)| a * v[d]).sum::<f32>() / z)
        .collect();
    (o, (m + z.ln()) * std::f32::consts::LOG2_E)
}

/// Fire one `_lse` arm and read back its two planes.
fn fire_lse(context: &Context, tree: &Tree, f: &Fixture, tiled: bool) -> (Vec<f32>, Vec<f32>) {
    let out = Allocation::new(
        context,
        (ROWS * Q_HEADS * HEAD_DIM * 2) as u64,
        "attention out",
    )
    .expect("an output");
    let lse = alloc_floats(
        context,
        &[LSE_SENTINEL; ROWS * Q_HEADS],
        "the log-sum-exp plane",
    );

    let mut buffers = f.buffers.clone();
    buffers.push((3, out.gpu_address()));
    buffers.push((if tiled { 18 } else { 17 }, lse.gpu_address()));

    let (entrypoint, args_wide, grid) = if tiled {
        (
            format!("sdpa_paged_tiled_lse_bfloat16_d_{HEAD_DIM}"),
            19usize,
            [Q_HEADS as u32 * 1024, (ROWS as u32).div_ceil(32), 1],
        )
    } else {
        (
            format!("sdpa_paged_decode_lse_bfloat16_d_{HEAD_DIM}"),
            18usize,
            [Q_HEADS as u32 * 1024, ROWS as u32, 1],
        )
    };

    run(
        context,
        tree,
        &Launch {
            entrypoint,
            file: "attn/sdpa_paged.metal",
            grid,
            threadgroup: [1024, 1, 1],
            args_wide,
            buffers: &buffers,
            scalars: &scalars(tiled),
        },
    );

    let n = ROWS * Q_HEADS * HEAD_DIM;
    let words = unsafe { core::slice::from_raw_parts(out.contents().as_ptr().cast::<u16>(), n) };
    let o = words.iter().copied().map(from_bf16).collect();
    let planes = unsafe {
        core::slice::from_raw_parts(lse.contents().as_ptr().cast::<f32>(), ROWS * Q_HEADS)
    };
    (o, planes.to_vec())
}

/// Fire a folded-sink arm, which writes no lse.
fn fire_folded(context: &Context, tree: &Tree, f: &Fixture, tiled: bool) -> Vec<f32> {
    let out = Allocation::new(
        context,
        (ROWS * Q_HEADS * HEAD_DIM * 2) as u64,
        "attention out",
    )
    .expect("an output");
    let mut buffers = f.buffers.clone();
    buffers.push((3, out.gpu_address()));

    let (entrypoint, args_wide, grid) = if tiled {
        (
            format!("sdpa_paged_tiled_sink_bfloat16_d_{HEAD_DIM}"),
            18usize,
            [Q_HEADS as u32 * 1024, (ROWS as u32).div_ceil(32), 1],
        )
    } else {
        (
            format!("sdpa_paged_decode_sink_bfloat16_d_{HEAD_DIM}"),
            17usize,
            [Q_HEADS as u32 * 1024, ROWS as u32, 1],
        )
    };

    run(
        context,
        tree,
        &Launch {
            entrypoint,
            file: "attn/sdpa_paged.metal",
            grid,
            threadgroup: [1024, 1, 1],
            args_wide,
            buffers: &buffers,
            scalars: &scalars(tiled),
        },
    );

    let n = ROWS * Q_HEADS * HEAD_DIM;
    let words = unsafe { core::slice::from_raw_parts(out.contents().as_ptr().cast::<u16>(), n) };
    words.iter().copied().map(from_bf16).collect()
}

/// `attn_sink_rescale` over a caller-supplied output, lse and sink, in place.
///
/// In place because `Attention::sink` states `o` as `InOut` and this plane
/// cuts an in-place mark into a read half and a write half at ONE address; the
/// two buffers below are that address twice, which is what the shipped claim
/// binds.
fn fire_sink(
    context: &Context,
    tree: &Tree,
    o: &[u16],
    lse: &[f32],
    sinks: &[u16],
    shape: [usize; 3],
) -> Vec<u16> {
    let [heads, head_dim, rows] = shape;
    let plane = alloc_raw(context, o, "o");
    let lse = alloc_floats(context, lse, "lse");
    let sinks = alloc_raw(context, sinks, "sinks");
    let buffers = [
        (0usize, plane.gpu_address()),
        (1, plane.gpu_address()),
        (2, lse.gpu_address()),
        (3, sinks.gpu_address()),
    ];
    run(
        context,
        tree,
        &Launch {
            entrypoint: "attn_sink_rescale_bfloat16".to_string(),
            file: "attn/attn_sink.metal",
            grid: [head_dim as u32, heads as u32, rows as u32],
            threadgroup: [head_dim as u32, 1, 1],
            args_wide: 4,
            buffers: &buffers,
            scalars: &[],
        },
    );
    let words =
        unsafe { core::slice::from_raw_parts(plane.contents().as_ptr().cast::<u16>(), o.len()) };
    words.to_vec()
}

// ── the reference `pie::attn::attn_sink_rescale` produced on an L40S ─────────
//
// Emitted by a host harness that includes `attn/attn_sink.cuh` from this tree
// and launches `attn_sink_rescale<bf16>` on the fixture below -- so the
// numbers are the cuda plane's own kernel's, not a second reading of its
// header. `nvcc -arch=sm_89`, CUDA 13.0.
//
// The three rows are the three cases the rescale has to separate:
//   row 0  finite lse spread around the sinks -- the factor is doing work,
//          and it is 0.1027, 0.6953, 0.2676, 0.8438 across the four heads;
//   row 1  lse far ABOVE every sink, so the factor is 1 and the output must
//          come back untouched -- a kernel that rescaled anyway fails here;
//   row 2  `lse = -inf`, the row that kept no key, where `isfinite` is what
//          keeps a NaN out of the output.
const SINK_ROWS: usize = 3;
const SINK_HEADS: usize = 4;
const SINK_D: usize = 8;

#[rustfmt::skip]
const O_IN: [u16; SINK_ROWS * SINK_HEADS * SINK_D] = [
    0xbf60, 0x0000, 0xbf80, 0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0,
    0x3f00, 0xbf00, 0x3ec0, 0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60,
    0x0000, 0xbf80, 0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00,
    0xbf00, 0x3ec0, 0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000,
    0xbf80, 0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00,
    0x3ec0, 0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80,
    0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0,
    0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00,
    0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0, 0xbf20,
    0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00, 0x3f40,
    0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0, 0xbf20, 0x3e80,
    0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00, 0x3f40, 0xbe80,
];

#[rustfmt::skip]
const SINK_LSE: [f32; SINK_ROWS * SINK_HEADS] = [
    0.5, 2.0, -3.25, 6.75,
    40.0, 12.5, 33.0, 27.25,
    f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
];

/// `gpt-oss`'s own `self_attn.sinks` values as the ledger records them
/// (2.515625, 0.55859375), plus a negative one so a head where the sink loses
/// to the normaliser is covered, and one larger than any of them.
#[rustfmt::skip]
const SINKS: [u16; SINK_HEADS] = [0x4021, 0x3f0f, 0xbfa0, 0x4040];

#[rustfmt::skip]
const CUDA_O_OUT: [u16; SINK_ROWS * SINK_HEADS * SINK_D] = [
    0xbdb8, 0x0000, 0xbdd2, 0xbc52, 0x3d9e, 0xbcd2, 0x3d83, 0xbd1e,
    0x3eb2, 0xbeb2, 0x3e86, 0xbedf, 0x3e32, 0xbf06, 0x3db2, 0xbf1c,
    0x0000, 0xbe89, 0xbd09, 0x3e4e, 0xbd89, 0x3e2c, 0xbdce, 0x3e09,
    0xbed8, 0x3ea2, 0xbf07, 0x3e58, 0xbf22, 0x3dd8, 0xbf3d, 0x0000,
    0xbf80, 0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00,
    0x3ec0, 0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80,
    0xbe00, 0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0,
    0xbf20, 0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00,
    0x3f40, 0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0, 0xbf20,
    0x3e80, 0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00, 0x3f40,
    0xbe80, 0x3f20, 0xbec0, 0x3f00, 0xbf00, 0x3ec0, 0xbf20, 0x3e80,
    0xbf40, 0x3e00, 0xbf60, 0x0000, 0xbf80, 0xbe00, 0x3f40, 0xbe80,
];

/// The bound a bfloat16 store alone can move a value by: eight bits of
/// significand, so `2^-9` of itself, with room for a different summation
/// order. `device_attention` uses the same one and for the same reason.
fn bound(want: f32) -> f32 {
    (want.abs() / 128.0).max(1.0 / 256.0)
}

/// The bound on the lse, which is an f32 and not a bfloat16 store away from
/// one. FOUR ULPS OF THE PUBLISHED VALUE, and the number is measured rather
/// than reasoned.
///
/// The device sums in f32 with `fast::exp` where the reference sums in f32
/// with `exp`, and takes the logarithm of a sum in a different association
/// order; that is a relative error in the denominator, so it stays relative
/// through the logarithm's own rounding. Measured over the twenty-four
/// `(row, head)` pairs of this fixture on an M-series part, both bodies: the
/// disagreement is exactly 0, 1, 2 or 3 ulps of the value -- 5.96e-8 and its
/// multiples, which is `2^-24` -- and the largest is 1.79e-7 on a value of
/// 0.809. Four ulps is `2^-21` relative, so the worst element uses 0.46 of
/// this bound and `tolerance_holds` at the foot of the test is what keeps
/// that honest. THE FIRST DRAFT SAID `2^-13` and the floor caught it at 512
/// times too loose, which is the whole reason that check is there.
fn lse_bound(want: f32) -> f32 {
    const FOUR_ULPS: f32 = 1.0 / 2_097_152.0;
    (want.abs() * FOUR_ULPS).max(FOUR_ULPS)
}

/// **The lse arms answer the same softmax and publish its denominator.**
///
/// Both bodies, against a reference this file computes from the fixture: the
/// output as `device_attention` checks it, and the log-sum-exp, which nothing
/// checked before because nothing was written.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_lse_arms_publish_a_base_two_log_sum_exp() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::shipped();
    let f = fixture(&context);

    let mut worst_o = 0.0f32;
    let mut worst_lse = 0.0f32;
    for tiled in [false, true] {
        let what = if tiled { "prefill_lse" } else { "decode_lse" };
        let (o, lse) = fire_lse(&context, &tree, &f, tiled);
        for row in 0..ROWS {
            assert!(
                keeps(&f, row).len() > 1,
                "row {row} keeps one key; a softmax over one key is that key \
                 whatever weight it computed, so the row would prove nothing"
            );
            for q_head in 0..Q_HEADS {
                let (want_o, want_lse) = reference(&f, row, q_head);
                let q_base = (row * Q_HEADS + q_head) * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    let seen = o[q_base + d];
                    worst_o = worst_o.max((seen - want_o[d]).abs() / bound(want_o[d]));
                    assert!(
                        (seen - want_o[d]).abs() <= bound(want_o[d]),
                        "{what}: row {row} head {q_head} channel {d} is {seen} \
                         and the reference is {}",
                        want_o[d],
                    );
                }
                let seen = lse[row * Q_HEADS + q_head];
                assert!(
                    seen != LSE_SENTINEL,
                    "{what}: row {row} head {q_head} was never written -- the \
                     plane still holds the sentinel"
                );
                worst_lse = worst_lse.max((seen - want_lse).abs() / lse_bound(want_lse));
                assert!(
                    (seen - want_lse).abs() <= lse_bound(want_lse),
                    "{what}: row {row} head {q_head} publishes {seen} where the \
                     base-two log-sum-exp of its kept keys is {want_lse}. \
                     Natural log would be {}",
                    want_lse / std::f32::consts::LOG2_E,
                );
            }
        }
    }
    tolerance_holds(worst_o, "the lse arms' output");
    tolerance_holds(worst_lse, "the lse arms' log-sum-exp");
}

/// **A row that keeps no key publishes `-inf`, and the sink then leaves it
/// alone.** The two halves of the empty row, in one test, because separately
/// neither is worth much.
///
/// This is the row shape the fixture above cannot reach. A query's own K is
/// appended before attention runs, so `kp = q_pos` survives every causal
/// bound and every window; only the mask can empty a row, and here it does,
/// for all three at once.
///
/// It matters because on that row the lse is not a number the softmax
/// computed. `sum_exp_score` is zero, so what gets published is a SENTINEL,
/// and the two sides of the point have to agree on which one: `-inf` is what
/// flashinfer publishes and what `attn_sink_rescale`'s `isfinite` branch is
/// written against, on both planes. A plane that published `0` instead --
/// which is what `log2` of an empty sum would be if the accumulator were
/// initialised to one rather than zero -- would take the FINITE branch on the
/// far side and rescale a row by `sigmoid(-sink)`, silently, on exactly the
/// rows where nothing is left to rescale.
///
/// So this asserts the contract end to end rather than one kernel: `-inf`
/// published, a zero output beside it, and the rescale downstream leaving
/// both alone. The last clause is the one worth having -- anything the sink
/// pass writes here rides the o_proj GEMM and the residual add into the row's
/// whole hidden state.
#[test]
#[ignore = "needs a Metal 4 device"]
fn a_row_that_keeps_no_key_publishes_negative_infinity() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::shipped();
    let f = fixture_masked(&context, vec![0u8; ROWS * MASK_STRIDE as usize], [1, 1, 1]);
    let sinks: Vec<u16> = sink_seen().iter().copied().map(bf16).collect();

    for tiled in [false, true] {
        let what = if tiled { "prefill_lse" } else { "decode_lse" };
        for row in 0..ROWS {
            assert!(
                keeps(&f, row).is_empty(),
                "{what}: row {row} still keeps a key, so this fixture is not \
                 asking the question it says it is"
            );
        }
        let (o, lse) = fire_lse(&context, &tree, &f, tiled);
        for (at, seen) in lse.iter().enumerate() {
            assert!(
                *seen == f32::NEG_INFINITY,
                "{what}: the empty row {} head {} publishes {seen}, not -inf",
                at / Q_HEADS,
                at % Q_HEADS,
            );
        }
        for (at, seen) in o.iter().enumerate() {
            assert!(
                *seen == 0.0,
                "{what}: the empty row {} channel {at} is {seen}, not zero",
                at / (Q_HEADS * HEAD_DIM),
            );
        }

        // And the rescale downstream, which is where a NaN would be made.
        let staged: Vec<u16> = o.iter().copied().map(bf16).collect();
        let rescaled = fire_sink(
            &context,
            &tree,
            &staged,
            &lse,
            &sinks,
            [Q_HEADS, HEAD_DIM, ROWS],
        );
        for (at, bits) in rescaled.iter().enumerate() {
            let seen = from_bf16(*bits);
            assert!(
                seen == 0.0 && !seen.is_nan(),
                "{what}: `attn_sink_rescale` turned an empty row's zero into \
                 {seen} at channel {at}"
            );
        }
    }
}

/// **And the lse check can fail**, which is the only reason to believe it.
///
/// One constant of `sdpa_lse_base2` changed -- `log2(e)` to `1.0`, which is
/// the plane publishing the natural log it accumulated in -- and the shipped
/// output plane is left exactly as it was. So this is the difference between
/// the two bases and nothing else, and if the check above could not see it,
/// the point's stated base would be decoration.
#[test]
#[ignore = "needs a Metal 4 device"]
fn an_lse_published_in_the_wrong_base_fails_the_check() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::patched(
        "sdpa_online.h",
        "constexpr float kLog2E = 1.44269504088896340736f;",
        "constexpr float kLog2E = 1.0f;",
    );
    let f = fixture(&context);
    let (_, lse) = fire_lse(&context, &tree, &f, false);

    let mut caught = Vec::new();
    for row in 0..ROWS {
        for q_head in 0..Q_HEADS {
            let (_, want) = reference(&f, row, q_head);
            let seen = lse[row * Q_HEADS + q_head];
            if (seen - want).abs() > lse_bound(want) {
                caught.push((row, q_head, seen, want));
            }
        }
    }
    assert!(
        !caught.is_empty(),
        "an lse published in natural log passed the base-two check, so the \
         check is not reading the base"
    );
    let (row, head, seen, want) = caught[0];
    println!(
        "injection caught: row {row} head {head} published {seen} where base \
         two is {want} (ratio {})",
        seen / want
    );
}

/// **The sink rescale is the kernel cuda serves this point with.**
///
/// Against [`CUDA_O_OUT`] -- `pie::attn::attn_sink_rescale`'s own bf16 bytes,
/// off an L40S. The three rows separate the three things the kernel does: the
/// rebase, the identity when the normaliser dominates, and the `isfinite`
/// guard on a row that kept no key.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_sink_rescale_is_the_kernel_cuda_serves() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::shipped();
    let got = fire_sink(
        &context,
        &tree,
        &O_IN,
        &SINK_LSE,
        &SINKS,
        [SINK_HEADS, SINK_D, SINK_ROWS],
    );

    let mut worst = 0.0f32;
    let mut moved = 0usize;
    for t in 0..SINK_ROWS {
        for h in 0..SINK_HEADS {
            for d in 0..SINK_D {
                let i = (t * SINK_HEADS + h) * SINK_D + d;
                let want = from_bf16(CUDA_O_OUT[i]);
                let seen = from_bf16(got[i]);
                worst = worst.max((seen - want).abs() / bound(want));
                assert!(
                    (seen - want).abs() <= bound(want),
                    "row {t} head {h} channel {d} is {seen} where \
                     `attn_sink_rescale` on the cuda plane answers {want} \
                     (lse {}, sink {})",
                    SINK_LSE[t * SINK_HEADS + h],
                    from_bf16(SINKS[h]),
                );
                // A sink adds to the denominator and nothing to the numerator,
                // so it can only shrink an output toward zero. A kernel that
                // divided by the factor instead passes the value check
                // wherever the tolerance is loose and fails this always.
                let before = from_bf16(O_IN[i]);
                assert!(
                    seen.abs() <= before.abs() + bound(before),
                    "row {t} head {h} channel {d} moved {before} to {seen}, \
                     which is AWAY from zero"
                );
                if got[i] != O_IN[i] {
                    moved += 1;
                }
            }
        }
    }
    assert!(
        moved >= SINK_HEADS * SINK_D / 2,
        "only {moved} of {} channels moved at all; the fixture's sinks are too \
         far below its normalisers for the rescale to prove anything",
        O_IN.len()
    );
    tolerance_holds(worst, "the sink rescale against the cuda plane's kernel");
}

/// **And the sink check can fail.**
///
/// `ln(2)` set to one -- the rebase removed, which is exactly the defect
/// `attn/attn_sink.cuh`'s header was written for and the one this plane would
/// have shipped if it had read `sigmoid(lse - sink)` off the point's spelling
/// without asking what base `lse` was in.
#[test]
#[ignore = "needs a Metal 4 device"]
fn a_sink_that_does_not_rebase_fails_the_check() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::patched(
        "attn_sink.metal",
        "constexpr float kLn2 = 0.69314718055994530942f;",
        "constexpr float kLn2 = 1.0f;",
    );
    let got = fire_sink(
        &context,
        &tree,
        &O_IN,
        &SINK_LSE,
        &SINKS,
        [SINK_HEADS, SINK_D, SINK_ROWS],
    );

    let mut caught = Vec::new();
    for i in 0..O_IN.len() {
        let want = from_bf16(CUDA_O_OUT[i]);
        let seen = from_bf16(got[i]);
        if (seen - want).abs() > bound(want) {
            caught.push((i, seen, want));
        }
    }
    assert!(
        !caught.is_empty(),
        "a rescale that never left base two matched the cuda plane's bytes, so \
         the comparison is not reading the rebase"
    );
    let (i, seen, want) = caught[0];
    println!(
        "injection caught: channel {i} answered {seen} where the cuda plane \
         answers {want}"
    );
}

/// **A published lse, rescaled, IS the folded sink arm** -- and that is why
/// the two are a decomposition and not a disagreement.
///
/// `decode_lse` then `sink` against `sdpa_paged_decode_sink`, which merges the
/// same scalar into the denominator before it divides; likewise for the tiled
/// pair. They read identical bytes, so anything past a bfloat16 store's own
/// quantum is a defect no tolerance excuses.
///
/// What it does NOT show is that the folded arm answers `attention.decode_lse`.
/// It writes no lse, and the point declares one; agreeing about `o` is not
/// answering a contract about `o` and `lse`. The floor's decomposition is what
/// gpt-oss's text states, and this is the measurement that says the plane pays
/// nothing for stating it that way.
#[test]
#[ignore = "needs a Metal 4 device"]
fn a_published_lse_rescaled_is_the_folded_sink_arm() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let tree = Tree::shipped();
    let f = fixture(&context);
    let sinks: Vec<u16> = sink_seen().iter().copied().map(bf16).collect();

    let mut worst = 0.0f32;
    for tiled in [false, true] {
        let what = if tiled { "prefill_lse" } else { "decode_lse" };
        let (o, lse) = fire_lse(&context, &tree, &f, tiled);
        let staged: Vec<u16> = o.iter().copied().map(bf16).collect();
        let rescaled = fire_sink(
            &context,
            &tree,
            &staged,
            &lse,
            &sinks,
            [Q_HEADS, HEAD_DIM, ROWS],
        );
        let folded = fire_folded(&context, &tree, &f, tiled);

        let mut moved = 0usize;
        for row in 0..ROWS {
            for q_head in 0..Q_HEADS {
                let q_base = (row * Q_HEADS + q_head) * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    let a = from_bf16(rescaled[q_base + d]);
                    let b = folded[q_base + d];
                    worst = worst.max((a - b).abs() / bound(b));
                    assert!(
                        (a - b).abs() <= bound(b),
                        "{what}: row {row} head {q_head} channel {d} is {a} \
                         through `sink` and {b} folded into the softmax -- two \
                         readings of one denominator, over identical bytes"
                    );
                    if (a - o[q_base + d]).abs() > bound(o[q_base + d]) {
                        moved += 1;
                    }
                }
            }
        }
        assert!(
            moved > ROWS * Q_HEADS * HEAD_DIM / 8,
            "{what}: the rescale moved only {moved} channels, so the two arms \
             agreed by both doing nothing"
        );
    }
    tolerance_holds(worst, "the published lse against the folded sink");
}

fn alloc_bf16(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let words: Vec<u16> = values.iter().copied().map(bf16).collect();
    alloc_raw(context, &words, what)
}

fn alloc_raw(context: &Context, words: &[u16], what: &'static str) -> Allocation {
    let bytes = std::mem::size_of_val(words) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(0, cast(words)).expect("the words fit");
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

fn alloc_floats(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let bytes = std::mem::size_of_val(values) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(
            0,
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes as usize),
        )
        .expect("the floats fit");
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
