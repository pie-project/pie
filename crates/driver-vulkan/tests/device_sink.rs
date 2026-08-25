//! THE LSE PLANE AND THE SINK RESCALE, FIRED ON THE CARD.
//!
//! `attention.{decode_lse, prefill_lse, sink}` — the three points gpt-oss states
//! that this plane could not answer, so the trace diff read 13/16 for both
//! gptoss rows while gemma4 was already complete.
//!
//! # What had no proof, and what this file is not
//!
//! Every paged body in `attn/sdpa_paged.slang` divides by its denominator on the
//! way out, so the denominator itself was read by nothing: a body could have
//! computed the right output from the wrong `sum_exp` — by an offsetting error
//! in `max_score`, which the online softmax cancels between the two — and
//! nothing in this tree would have noticed. `attention.decode_lse` makes the
//! denominator an OPERAND of the next statement, so from here it is a value and
//! not an accumulator, and this file is the first thing that reads it.
//!
//! It is also the sharper probe of the two. An output is a convex combination of
//! the values it kept, so a body that dropped a key whose value happened to sit
//! near the mean of the others answers almost the same vector. The lse is the
//! SUM over the kept set: drop a key and it moves by that key's whole weight,
//! whatever its value was. That is why the fixture below keeps the reversed page
//! table, the window and the per-row mask — three ways of making a key set
//! observable, aimed at the output that cannot average one away.
//!
//! # THE BASE, which is the whole of `attention.sink`
//!
//! An lse is a logarithm and a logarithm has a base. The floor states BASE TWO —
//! flashinfer's, because its host folds `log2(e)` into `sm_scale` and its kernels
//! run on `exp2`, and that is the plane whose kernel this tree does not own.
//!
//! THIS plane's softmax runs on Slang's `exp`, and what that is was MEASURED
//! rather than assumed: compiled at this tree's own flags it is
//! `OpExtInst %1 Exp`, GLSL.std.450's natural exponent, with no `log2(e)`
//! anywhere near it — and the constant scan below asserts exactly that, module by
//! module. So `sdpa_lse_base2` rebases on the way out and the number that reaches
//! the next statement is the one the point promises.
//!
//! `attention.sink` is then the one point where two bases meet: a base-two
//! normaliser against gpt-oss's `self_attn.sinks`, which HF wrote in natural log.
//! The `ln(2)` in `attn/attn_sink.slang` is not housekeeping in front of the
//! kernel; it IS the kernel's job, and dropping it is a factor of 0.693 in a
//! sigmoid argument — which matches HF's top-1 on most prompts by accident and
//! then drifts. So the reference for that point is not a CPU model this file
//! wrote: it is the bf16 bytes `pie::attn::attn_sink_rescale` — the CUDA kernel
//! the cuda plane serves `attention.sink` with — produced on the L40S this suite
//! runs on, from `kernels-cuda/kernels/attn/attn_sink.cuh`, carried over as
//! [`CUDA_O_OUT`]. A reference computed from the code that ships is worth more
//! than one computed from the same reading of the header twice.
//!
//! # And the question the last test settles
//!
//! `attn/sdpa_paged.slang` has `PIE_WITH_SINK` entry points that fold the same
//! scalar into the softmax denominator BEFORE the division, and they answer the
//! same numbers — [`a_published_lse_rescaled_is_the_folded_sink_arm`] measures
//! exactly that, and it holds. It is still not the same statement: a folded arm
//! writes no lse, `attention.decode_lse` DECLARES one, and a point is a contract
//! about what is written. The two readings agreeing is what makes this a
//! decomposition rather than a rewrite; the declaration is what decides which one
//! a text may state.
//!
//! # Every check here can fail, and the injections are SPIR-V and not source
//!
//! Two of the tests below are fault injections: they build a pipeline from the
//! SHIPPED module with one constant changed, and assert the comparison goes RED.
//!
//! WHERE THE SIBLING PLANES PATCH SOURCE, THIS ONE PATCHES THE MODULE, and that
//! is the plane's own shape rather than a shortcut. `driver-wgpu` holds WGSL
//! strings and `driver-metal` holds a `.metal` tree, so both can hand a compiler
//! a changed text; `kernels-vulkan` compiles Slang in its BUILD SCRIPT and this
//! crate receives `&'static [u8]` of SPIR-V. There is no source here to patch and
//! no `slangc` in a test's closure. What there is, is the shipped words —
//! and [`patched`] rewrites the literal of exactly one `OpConstant %float`,
//! asserting it matched exactly once in that entrypoint's own module. That is a
//! stricter statement than a string replace, not a weaker one: a text anchor can
//! land in a comment or a sibling body, and a constant-literal anchor cannot land
//! anywhere but a constant. [`the_two_bases_are_one_constant_each_and_no_more`]
//! is the check that keeps "exactly once" honest for the whole family at once.
//!
//! # How a point is fired here
//!
//! Through the REAL claim body. `kernels_vulkan::plane::Ctx` is `dyn Encode`, so
//! [`Rec`] below is a whole plane as far as `kernels_vulkan::attn` is concerned:
//! the body picks the entrypoint, computes the grid and states its own operand
//! run, and this file only lays that run out and dispatches it. What is under
//! test is therefore the shipped body and the shipped shader, not a
//! transcription of either.
//!
//! The two folded `_sink` arms are the exception and are hand-stated, because
//! nothing claims them: they are dark on this plane, which is the finding the
//! last test rests on.
//!
//! # Why `device` and not `native`
//!
//! `device_fire`'s reason exactly: these need a card and the compiled SPIR-V, and
//! need nothing at all from the plan walk. And not `#[ignore]`, for
//! `device_fire`'s other reason: an ignored test is skipped on the machine that
//! HAS the hardware too, which is the machine whose failures matter.

#![cfg(feature = "device")]
#![allow(clippy::print_stdout)]

use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard, OnceLock};

use driver_vulkan::device::{Bound, Buffer, Device, Pipelines};
use driver_vulkan::spirv;
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::Attention;
use kernels_vulkan::Capability;
use kernels_vulkan::plane::{ArgValue, Encode, Fire};
use kernels_vulkan::points::{Handle, bf16};
use kernels_vulkan::views::{AttnFireView, MaskView, PagedKvView, SplitView};

// ── the device, once per file ───────────────────────────────────────────────

static GPU: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

static NO_DEVICE: OnceLock<String> = OnceLock::new();

fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => Some(Mutex::new(d)),
        Err(e) => {
            eprintln!("skipped: {e}");
            let _ = NO_DEVICE.set(e.to_string());
            None
        }
    });
    held.as_ref()
        .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

macro_rules! gpu {
    () => {{
        if !kernels_vulkan::embedded() {
            eprintln!(
                "skipped: built without kernels-vulkan/native, so there are no \
                 modules to build a pipeline from"
            );
            return;
        }
        let Some(device) = gpu() else {
            return;
        };
        device
    }};
}

// ── the bounds, and what keeps them honest ──────────────────────────────────

/// How much of its own tolerance the worst element used, and the band that has
/// to hold.
///
/// A bound nothing measured is a bound a wrong kernel hides in, and `4b > b` is
/// true of every bound anyone will ever write — so the floor is asked in both
/// directions. The upper half says the assertions above really passed; the lower
/// half says they did not pass by a mile, which is the half that catches a
/// tolerance copied from somewhere it did not belong. Three waves across the
/// sibling planes wrote bounds from reasoning that were 300, 512 and 25 times too
/// loose, and this is the shape of check that found all three.
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
         tolerance is more than eight times the arithmetic this device actually \
         delivers — tighten the bound instead of trusting it"
    );
}

/// The bound a bfloat16 store alone can move a value by: eight bits of
/// significand, so `2^-9` of itself, with room for a different summation order.
fn bound(want: f32) -> f32 {
    (want.abs() / 128.0).max(1.0 / 256.0)
}

/// The bound on the lse, which is an f32 and not a bfloat16 store away from one.
/// TWO ULPS OF THE PUBLISHED VALUE, and the number is MEASURED rather than
/// reasoned.
///
/// The device sums the dot in a different association order from this file's
/// reference — the tiled arm folds 32 lanes through workgroup memory, the decode
/// arm splits the row across subgroups and merges them — exponentiates, and takes
/// a logarithm. That is a relative error in the denominator, so it stays relative
/// through the logarithm's own rounding.
///
/// Measured over the twelve `(row, query head)` pairs of this fixture on an L40S,
/// BOTH bodies; the run prints the raw figure and `tolerance_holds` at the foot
/// of the test is what keeps the fraction honest. The sibling planes' first
/// drafts of this number were reasoned rather than read and were 25 and 512 times
/// too loose, which is the whole reason that floor is there.
fn lse_bound(want: f32) -> f32 {
    const TWO_ULPS: f32 = 1.0 / 8_388_608.0;
    (want.abs() * TWO_ULPS).max(TWO_ULPS)
}

// ── bf16, both ways ─────────────────────────────────────────────────────────

/// `f32` to bf16, round-to-nearest-even — the rounding `f32_to_bf16` does in
/// `common/bf16.slang`, restated so the host reference reads the same values the
/// shader writes.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// bf16 back to `f32` — exact, since bf16 is the top half of an `f32`.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// A run of bf16 values as the `uint16_t` elements the shaders read.
///
/// ONE ELEMENT PER WORD-HALF AND NO PACKING, which is where this file parts
/// company with `driver-wgpu`'s twin: WGSL has no 16-bit storage type, so bf16
/// crosses there as `array<u32>` and every codec on that plane deals in pairs.
/// `PIE_ACT` is `uint16_t` here.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_bf16(*x).to_le_bytes()).collect()
}

fn bf16_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn bits_of(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Values bfloat16 holds exactly: multiples of an eighth in `[-1, 1)`, period
/// fifteen. Fifteen is coprime with 64 and with every stride in the pool, so no
/// two slots alias and a body that walked the page table wrong is visible.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 15) as f32 - 7.0) / 8.0)
        .collect()
}

// ── the attention fixture ───────────────────────────────────────────────────

/// The one width the `_lse` arms are stamped at. gpt-oss is the only family that
/// states these points on this plane and its heads are 64 wide; see the
/// instantiation block in `attn/sdpa_paged.slang`.
const HEAD_DIM: usize = 64;
const Q_HEADS: usize = 4;
const GQA: usize = 2;
const KV_HEADS: usize = Q_HEADS / GQA;
const ROWS: usize = 3;
const PAGE_SIZE: usize = 3;
const PAGES: usize = 7;
const SLOTS: usize = PAGES * PAGE_SIZE;
const SCALE: f32 = 0.125;
const WINDOW: i32 = 4;
const MASK_STRIDE: u32 = 12;

/// Which rows consult their mask, and the one key each masked row drops.
///
/// Rows 0 and 1 are masked and row 2 is not, so the fixture carries both arms of
/// `keeps`'s branch. The holes are inside each row's window — 4 is in row 0's
/// `[3, 6]` and 7 is in row 1's `[6, 9]` — because a hole outside the window is a
/// hole the causal bound already dug.
const MASK_ON: [u8; ROWS] = [1, 1, 0];
const MASK_HOLES: [usize; ROWS] = [4, 7, usize::MAX];

const POSITIONS: [i32; ROWS] = [6, 9, 2];
const REQUESTS: [i32; ROWS] = [0, 0, 1];
const INDPTR: [u32; 3] = [0, 4, 7];
/// Reversed, so a body that read the pool in slot order rather than through the
/// page table answers different numbers instead of the same ones.
const INDICES: [u32; PAGES] = [6, 4, 2, 0, 5, 3, 1];

/// What an unwritten lse slot holds, so "the kernel never got here" is a
/// different failure from "the kernel got here and computed the wrong thing".
/// Never zero and never `-inf`, both of which this point can legitimately mean.
const LSE_SENTINEL: f32 = -12345.0;

/// What an unwritten bf16 output word holds. `Device::empty` hands back zeros and
/// a correct output can be zero, so the sentinel is a value no softmax produces
/// here: `0x4747` is 51000.0.
const BF16_SENTINEL: u8 = 0x47;

/// One of the four sinks per head, spread across the scores: one far below every
/// score (which changes nothing), one above them all (which roughly halves the
/// output), and two in between. A sink nothing shrinks proves nothing.
fn sink_seen() -> Vec<f32> {
    vec![-8.0, 0.5, 2.515_625, 8.0]
}

// ── the handles this fixture binds ──────────────────────────────────────────

const H_Q: u32 = 0;
const H_K: u32 = 1;
const H_V: u32 = 2;
const H_O: u32 = 3;
const H_POS: u32 = 4;
const H_REQ: u32 = 5;
const H_INDICES: u32 = 6;
const H_INDPTR: u32 = 7;
const H_MASK: u32 = 8;
const H_MASK_ON: u32 = 9;
const H_SINKS: u32 = 10;
const H_LSE: u32 = 11;
const H_WPAGE: u32 = 12;
const H_WOFF: u32 = 13;
const H_SPLIT: u32 = 14;
const HANDLES: usize = 15;

// ── the recorder that stands where a driver's encoder would ─────────────────

/// One fire a claim body stated.
#[derive(Clone, Debug)]
struct Fired {
    entrypoint: &'static str,
    lanes: [u32; 3],
    args: Vec<ArgValue>,
}

/// A whole plane, as far as `kernels_vulkan::attn` is concerned.
///
/// `kernels_vulkan::plane::Ctx` is `dyn Encode`, so implementing three methods is
/// all it takes to drive the SHIPPED claim bodies — which is the point: the
/// entrypoint each test fires and the grid it fires over are the body's answers,
/// not this file's.
#[derive(Default)]
struct Rec {
    fires: RefCell<Vec<Fired>>,
}

impl Encode for Rec {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        self.fires.borrow_mut().push(Fired {
            entrypoint: fire.entrypoint,
            lanes: fire.lanes,
            args: args.to_vec(),
        });
        Ok(())
    }

    /// Nothing in this family stages a stream or opens a window; a point that
    /// tried would be a finding rather than something to answer.
    fn staged(&self, name: &'static str) -> Result<u32, Refusal> {
        Err(Refusal::Unstated { what: name })
    }

    fn windowed(&self, _of: u32, _at: u64) -> Result<u32, Refusal> {
        Err(Refusal::Unstated {
            what: "a window into a staged stream, which no attention point opens",
        })
    }
}

impl Rec {
    /// The one fire a point stated, or a panic naming how many it stated instead.
    /// Every point in this file plans exactly one dispatch.
    fn one(self) -> Fired {
        let mut fires = self.fires.into_inner();
        assert_eq!(fires.len(), 1, "this point plans exactly one dispatch");
        fires.pop().expect("just checked")
    }
}

// ── the SPIR-V fault injection ──────────────────────────────────────────────

/// The shipped module with the literal of exactly one `OpConstant %float`
/// changed.
///
/// A CONSTANT ANCHOR AND NOT A TEXT ANCHOR. `driver-metal` learned the hard way
/// that a fault injection whose anchor matched the wrong kernel body silently
/// mutates a kernel that is never fired, and both sibling planes now assert their
/// string matched exactly once. A string can land in a comment, in a sibling
/// entry point's arm, or in a define that this variant did not take; a 32-bit
/// literal inside an `OpConstant` whose result type is `%float` can land in none
/// of those, because by the time SPIR-V exists the preprocessor has run, the
/// unreached arms are gone and there are no comments left.
///
/// The walk is the module's own instruction stream — `OpConstant` is opcode 43,
/// and the type id is compared against the `OpTypeFloat 32` this module declares
/// — so a matching bit pattern that happens to be an id, a decoration operand or
/// a string fragment is not a candidate at all. Exactly one match is required, in
/// this entrypoint's own module and no other.
fn patched(code: &[u8], from: f32, to: f32) -> Vec<u8> {
    let mut words = spirv::words(code).expect("a built module is whole words");

    let mut float32: Option<u32> = None;
    let mut at = 5usize;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        assert!(count > 0, "a built module has no zero-length instruction");
        // `OpTypeFloat` is 22: result id, then the width in bits.
        if (words[at] & 0xffff) == 22 && count >= 3 && words[at + 2] == 32 {
            float32 = Some(words[at + 1]);
        }
        at += count;
    }
    let float32 = float32.expect("a module that carries a float constant declares `f32`");

    let want = from.to_bits();
    let mut hits = Vec::new();
    let mut at = 5usize;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        // `OpConstant` is 43: result type, result id, then the literal.
        if (words[at] & 0xffff) == 43
            && count == 4
            && words[at + 1] == float32
            && words[at + 3] == want
        {
            hits.push(at + 3);
        }
        at += count;
    }
    assert_eq!(
        hits.len(),
        1,
        "the fault injection's anchor {from} appears as {} `OpConstant %float` \
         in this entrypoint's own module, not once — a patch that matched \
         nothing would leave the shipped kernel in place and this test would \
         pass while proving the opposite of what it says",
        hits.len(),
    );
    words[hits[0]] = to.to_bits();
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

// ── laying a stated run out on the card ─────────────────────────────────────

/// Split a body's operand run into the buffer list and the push block.
///
/// The SAME rule `kernels_vulkan::bindings` applies, restated here for the reason
/// this whole file is in `tests/`: buffer-kinded operands take descriptor slots
/// in the order the body states them, and every other operand is appended to the
/// push block in the order the body states them. Every scalar in the attention
/// family is four bytes wide, so the block is a straight run — and the test
/// asserts that against what the module DECLARES rather than trusting it.
fn lay_out(fired: &Fired) -> (Vec<u32>, Vec<u8>) {
    let mut handles = Vec::new();
    let mut block = Vec::new();
    for arg in &fired.args {
        match *arg {
            ArgValue::Buffer { handle, .. } => handles.push(handle),
            ArgValue::I32(v) => block.extend_from_slice(&v.to_le_bytes()),
            ArgValue::U32(v) => block.extend_from_slice(&v.to_le_bytes()),
            ArgValue::F32(v) => block.extend_from_slice(&v.to_le_bytes()),
            other => panic!(
                "`{}` stated {}, which this rig does not lay out",
                fired.entrypoint,
                other.kind()
            ),
        }
    }
    (handles, block)
}

/// Groups from lanes, the way `driver_vulkan::serve` does it: a claim body states
/// INVOCATIONS and the driver divides by the module's own declared local size,
/// per axis.
fn groups(lanes: [u32; 3], local: [u32; 3]) -> [u32; 3] {
    [
        lanes[0].div_ceil(local[0].max(1)),
        lanes[1].div_ceil(local[1].max(1)),
        lanes[2].div_ceil(local[2].max(1)),
    ]
}

/// Build the pipeline a fire names, hold the run the body planned against what
/// the module declares, and dispatch it.
///
/// `patch` is a fault injection applied to the shipped module — see [`patched`].
/// `None` runs the shipped words.
///
/// The pipeline cache is FRESH per call, which is not laziness: `Pipelines` is
/// keyed by `(entrypoint, tier)` and a patched module carries the same entrypoint
/// name as the shipped one, so a shared cache would hand a later caller whichever
/// was compiled first.
fn run(device: &Device, bufs: &[Buffer], fired: &Fired, patch: Option<(f32, f32)>) {
    let mut cache = Pipelines::new();
    let (shipped, tier) = device
        .module_for(fired.entrypoint)
        .unwrap_or_else(|| panic!("the tree carries `{}`", fired.entrypoint));
    assert_eq!(
        tier,
        Capability::Baseline,
        "`{}` is stamped at one tier and it is the baseline",
        fired.entrypoint
    );
    let mutated;
    let code: &[u8] = match patch {
        None => shipped,
        Some((from, to)) => {
            mutated = patched(shipped, from, to);
            &mutated
        }
    };

    let (handles, block) = lay_out(fired);
    let pipeline = cache
        .get(
            device,
            fired.entrypoint,
            code,
            block.len() as u32,
            0,
            Capability::Baseline,
        )
        .unwrap_or_else(|why| panic!("`{}` builds no pipeline: {why:?}", fired.entrypoint));
    let declared = pipeline.declared();

    // A HOLE IS SKIPPED, NOT FILLED. `Device::run` binds the Nth buffer to the
    // Nth slot the module actually READS, so the run a body states is as long as
    // the module's declared bindings LESS its holes — which is how
    // `sdpa_paged_decode_lse`'s eleven operands reach bindings 0..9 and 11 with
    // the unread `sinks` at 10 passed over.
    let real = declared.bindings as usize - declared.holes();
    assert_eq!(
        handles.len(),
        real,
        "`{}`'s body bound {} buffers and the module reads {real} of the {} \
         slots it declares",
        fired.entrypoint,
        handles.len(),
        declared.bindings,
    );
    let packed: Vec<u32> = (0..block.len() as u32).step_by(4).collect();
    assert_eq!(
        packed, declared.push_offsets,
        "`{}`'s body stated its scalars at {packed:?} and the module declares \
         its push members at {:?}",
        fired.entrypoint, declared.push_offsets,
    );

    let grid = groups(fired.lanes, declared.local);
    let bound: Vec<Bound<'_>> = handles
        .iter()
        .map(|h| Bound::whole(&bufs[*h as usize]))
        .collect();
    device
        .run(pipeline, &bound, &block, grid)
        .unwrap_or_else(|why| panic!("`{}` dispatches: {why:?}", fired.entrypoint));
    cache.clear(device);
}

// ── the pool, the page table and the operands they all share ────────────────

/// The device-side fixture: one buffer per handle, plus the host copies of
/// everything a reference needs.
struct Rig<'a> {
    device: &'a Device,
    bufs: Vec<Buffer>,
    view: Box<AttnFireView>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn tensor<T: kernels::points::Scalar>(h: u32) -> kernels::shader::Tensor<T> {
    kernels::shader::Tensor::new(h)
}

/// The pool, the page table and the launch operands every attention fire in this
/// file shares. `mask` is the caller's, because the empty-row test needs a
/// different one and nothing else about the fixture changes.
fn rig<'a>(device: &'a Device, mask: &[u8], mask_on: [u8; ROWS]) -> Rig<'a> {
    let q = spread(ROWS * Q_HEADS * HEAD_DIM, 1);
    let k = spread(SLOTS * KV_HEADS * HEAD_DIM, 4);
    let v = spread(SLOTS * KV_HEADS * HEAD_DIM, 9);

    let mut bufs = vec![Buffer::placeholder(0); HANDLES];
    let mut put = |h: u32, bytes: &[u8]| {
        bufs[h as usize] = device.buffer(bytes).expect("the fixture uploads");
    };
    put(H_Q, &bf16_bytes(&q));
    put(H_K, &bf16_bytes(&k));
    put(H_V, &bf16_bytes(&v));
    put(H_O, &vec![BF16_SENTINEL; ROWS * Q_HEADS * HEAD_DIM * 2]);
    put(H_POS, &i32_bytes(&POSITIONS));
    put(H_REQ, &i32_bytes(&REQUESTS));
    put(H_INDICES, &u32_bytes(&INDICES));
    put(H_INDPTR, &u32_bytes(&INDPTR));
    put(H_MASK, mask);
    put(H_MASK_ON, &mask_on);
    put(H_SINKS, &bf16_bytes(&sink_seen()));
    put(H_LSE, &f32_bytes(&[LSE_SENTINEL; ROWS * Q_HEADS]));
    put(H_WPAGE, &u32_bytes(&[0; ROWS]));
    put(H_WOFF, &u32_bytes(&[0; ROWS]));
    // Never read by anything this file fires — every point here takes the
    // unsplit arm — but the view states it, so it is a real allocation rather
    // than a dangling handle.
    put(H_SPLIT, &f32_bytes(&[0.0; ROWS * Q_HEADS * (HEAD_DIM + 2)]));

    let view = Box::new(AttnFireView {
        kv: PagedKvView {
            keys: tensor::<kernels::shader::bf16>(H_K),
            values: tensor::<kernels::shader::bf16>(H_V),
            page_indices: tensor::<u32>(H_INDICES),
            page_indptr: tensor::<u32>(H_INDPTR),
            write_page: tensor::<u32>(H_WPAGE),
            write_offset: tensor::<u32>(H_WOFF),
            page_size: PAGE_SIZE as i32,
            seq_stride: kernels::shader::Usize(0),
            head_stride: kernels::shader::Usize(0),
        },
        positions: tensor::<i32>(H_POS),
        request_of_token: tensor::<i32>(H_REQ),
        mask: MaskView {
            mask: tensor::<u8>(H_MASK),
            enabled: tensor::<u8>(H_MASK_ON),
            stride: MASK_STRIDE,
        },
        split: SplitView {
            partials: tensor::<f32>(H_SPLIT),
            // ONE, which is what makes `attention.decode_lse` reachable at all:
            // a split holds only its own slice's denominator and the claim body
            // takes the unsplit arm. See the body's own note.
            splits: 1,
        },
        kv_heads: KV_HEADS as i32,
        head_dim: HEAD_DIM as i32,
    });

    Rig {
        device,
        bufs,
        view,
        q,
        k,
        v,
    }
}

/// The mask this file's own reading of [`MASK_ON`] and [`MASK_HOLES`] states:
/// every key allowed except one per masked row.
fn mask_bytes() -> Vec<u8> {
    let mut mask = vec![1u8; ROWS * MASK_STRIDE as usize];
    for (row, hole) in MASK_HOLES.iter().enumerate() {
        if *hole != usize::MAX {
            mask[row * MASK_STRIDE as usize + hole] = 0;
        }
    }
    mask
}

impl Rig<'_> {
    fn pages(&self) -> Cache<kernels::raises::Struct<kernels_vulkan::views::AttnFire>> {
        Cache {
            ptr: &*self.view as *const AttnFireView,
        }
    }

    fn read(&self, handle: u32) -> Vec<u8> {
        self.device
            .read(&self.bufs[handle as usize])
            .expect("the range reads back")
    }

    fn write(&self, handle: u32, bytes: &[u8]) {
        self.device
            .write(&self.bufs[handle as usize], bytes)
            .expect("the range writes");
    }

    /// Fill the lse plane with a value no softmax produces, so an unwritten slot
    /// is a different failure from a wrongly written one.
    fn poison_lse(&self) {
        self.write(H_LSE, &f32_bytes(&[LSE_SENTINEL; ROWS * Q_HEADS]));
    }

    /// What each row keeps, by this file's own reading of the fixture: the causal
    /// bound, the window and the mask's two clauses.
    fn keeps(&self, row: usize, mask: &[u8], mask_on: [u8; ROWS]) -> Vec<i32> {
        let q_pos = POSITIONS[row];
        let start = if WINDOW > 0 && q_pos >= WINDOW {
            q_pos - WINDOW + 1
        } else {
            0
        };
        (start..=q_pos)
            .filter(|kp| {
                if mask_on[row] == 0 {
                    return true;
                }
                if *kp as u32 >= MASK_STRIDE {
                    return false;
                }
                mask[row * MASK_STRIDE as usize + *kp as usize] != 0
            })
            .collect()
    }

    fn slot_of(&self, req: usize, kp: i32) -> usize {
        let page_ix = kp as usize / PAGE_SIZE;
        let page_off = kp as usize % PAGE_SIZE;
        let phys = INDICES[INDPTR[req] as usize + page_ix] as usize;
        phys * PAGE_SIZE + page_off
    }

    /// Softmax attention over the kept keys, and the log-sum-exp of the
    /// denominator that normalised it — IN BASE TWO, which is what the point
    /// states: `m*log2(e) + log2(sum)` and not `ln(...)`. See the module header.
    ///
    /// The scale is applied PER TERM because the shaders apply it per term, and
    /// hoisting it is a different f32 rounding. Everything accumulates in f64, so
    /// a disagreement is the kernel's and not this file's.
    fn reference(
        &self,
        row: usize,
        q_head: usize,
        mask: &[u8],
        mask_on: [u8; ROWS],
    ) -> (Vec<f32>, f32) {
        let kept = self.keeps(row, mask, mask_on);
        let kv_head = q_head / GQA;
        let q_base = (row * Q_HEADS + q_head) * HEAD_DIM;
        let scores: Vec<f64> = kept
            .iter()
            .map(|kp| {
                let slot = self.slot_of(REQUESTS[row] as usize, *kp);
                let k_base = (slot * KV_HEADS + kv_head) * HEAD_DIM;
                (0..HEAD_DIM)
                    .map(|d| {
                        f64::from(SCALE)
                            * f64::from(self.q[q_base + d])
                            * f64::from(self.k[k_base + d])
                    })
                    .sum()
            })
            .collect();
        if scores.is_empty() {
            return (vec![0.0; HEAD_DIM], f32::NEG_INFINITY);
        }
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f64 = weights.iter().sum();
        let mut out = vec![0.0f32; HEAD_DIM];
        for (at, kp) in kept.iter().enumerate() {
            let slot = self.slot_of(REQUESTS[row] as usize, *kp);
            let v_base = (slot * KV_HEADS + kv_head) * HEAD_DIM;
            for (d, cell) in out.iter_mut().enumerate() {
                *cell += (weights[at] * f64::from(self.v[v_base + d]) / denom) as f32;
            }
        }
        (out, (m * std::f64::consts::LOG2_E + denom.log2()) as f32)
    }

    fn free(self) {
        for b in self.bufs {
            self.device.free(b);
        }
    }
}

fn tin(h: u32, rows: usize, width: usize) -> In<Handle<bf16>> {
    In {
        ptr: Handle::new(h),
        rows: rows as i32,
        width: width as i32,
    }
}

fn tout(h: u32, rows: usize, width: usize) -> Out<Handle<bf16>> {
    Out {
        ptr: Handle::new(h),
        rows: rows as i32,
        width: width as i32,
    }
}

fn fout(h: u32, rows: usize, width: usize) -> Out<Handle<f32>> {
    Out {
        ptr: Handle::new(h),
        rows: rows as i32,
        width: width as i32,
    }
}

// ── firing the three points ─────────────────────────────────────────────────

/// `attention.decode_lse` or `attention.prefill_lse`, through the shipped claim
/// body, and the two planes it wrote.
fn fire_lse(rig: &Rig<'_>, tiled: bool, patch: Option<(f32, f32)>) -> (Vec<f32>, Vec<f32>) {
    rig.poison_lse();
    rig.write(H_O, &vec![BF16_SENTINEL; ROWS * Q_HEADS * HEAD_DIM * 2]);
    let rec = Rec::default();
    let width = Q_HEADS * HEAD_DIM;
    {
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        let q = tin(H_Q, ROWS, width);
        let o = tout(H_O, ROWS, width);
        let lse = fout(H_LSE, ROWS, Q_HEADS);
        if tiled {
            Attention::prefill_lse::<bf16>(
                ctx,
                q,
                In {
                    ptr: Handle::new(H_INDPTR),
                    rows: 1,
                    width: INDPTR.len() as i32,
                },
                rig.pages(),
                WINDOW as u32,
                HEAD_DIM as u32,
                KV_HEADS as u32,
                SCALE,
                o,
                lse,
            )
        } else {
            Attention::decode_lse::<bf16>(
                ctx,
                q,
                rig.pages(),
                WINDOW as u32,
                HEAD_DIM as u32,
                SCALE,
                o,
                lse,
            )
        }
        .expect("the claim body states this fire");
    }
    let fired = rec.one();
    assert_eq!(
        fired.entrypoint,
        if tiled {
            "sdpa_paged_tiled_lse_bfloat16_d_64"
        } else {
            "sdpa_paged_decode_lse_bfloat16_d_64"
        },
        "the claim body picks the entrypoint, and this is the one it picks",
    );
    run(rig.device, &rig.bufs, &fired, patch);
    (bf16_read(&rig.read(H_O)), f32_read(&rig.read(H_LSE)))
}

/// A folded `_sink` arm, which writes no lse.
///
/// HAND-STATED, and that is the finding rather than a shortcut: nothing in this
/// tree claims these entry points, so there is no body to drive. The operand run
/// below is `attention.decode`'s or `attention.prefill`'s exactly, with the real
/// sink buffer appended where the sinkless arms leave binding 10 a hole.
fn fire_folded(rig: &Rig<'_>, tiled: bool) -> Vec<f32> {
    rig.write(H_O, &vec![BF16_SENTINEL; ROWS * Q_HEADS * HEAD_DIM * 2]);
    let buf = |h: u32| ArgValue::Buffer {
        handle: h,
        writes: true,
        rows: 0,
        width: 0,
    };
    let mut args = vec![
        buf(H_Q),
        buf(H_K),
        buf(H_V),
        buf(H_O),
        ArgValue::I32(GQA as i32),
        buf(H_POS),
        buf(H_REQ),
        buf(H_INDICES),
        buf(H_INDPTR),
        ArgValue::I32(PAGE_SIZE as i32),
        ArgValue::I32(KV_HEADS as i32),
        ArgValue::F32(SCALE),
        buf(H_MASK),
        ArgValue::U32(MASK_STRIDE),
        buf(H_MASK_ON),
        ArgValue::I32(WINDOW),
    ];
    if tiled {
        args.push(ArgValue::I32(ROWS as i32));
    }
    args.push(buf(H_SINKS));
    let fired = Fired {
        entrypoint: if tiled {
            "sdpa_paged_tiled_sink_bfloat16_d_64"
        } else {
            "sdpa_paged_decode_sink_bfloat16_d_64"
        },
        // The grids `kernels_vulkan::attn`'s own helpers compute for this shape,
        // called rather than transcribed.
        lanes: if tiled {
            kernels_vulkan::attn::tiled_grid(Q_HEADS as i32, ROWS as i32).expect("a grid")
        } else {
            kernels_vulkan::attn::vector_grid(HEAD_DIM as i32, Q_HEADS as i32, ROWS as i32)
                .expect("a grid")
        },
        args,
    };
    run(rig.device, &rig.bufs, &fired, None);
    bf16_read(&rig.read(H_O))
}

/// `attention.sink`, through the shipped claim body, over a caller-supplied
/// output, lse and sink — in place.
///
/// In place because `Attention::sink` states `o` as `InOut` and this plane cuts
/// an in-place mark into a read half and a write half at ONE handle; the two
/// bindings below are that handle twice, which is what the shipped claim binds.
fn fire_sink(rig: &Rig<'_>, rows: usize, heads: usize, head_dim: usize, patch: Option<(f32, f32)>) {
    let rec = Rec::default();
    {
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        Attention::sink::<bf16>(
            ctx,
            InOut {
                ptr: Handle::new(H_O),
                rows: rows as i32,
                width: (heads * head_dim) as i32,
            },
            In {
                ptr: Handle::new(H_LSE),
                rows: rows as i32,
                width: heads as i32,
            },
            Const {
                v: Handle::new(H_SINKS),
            },
            head_dim as u32,
        )
        .expect("the claim body states this fire");
    }
    let fired = rec.one();
    assert_eq!(fired.entrypoint, "attn_sink_rescale_bfloat16");
    run(rig.device, &rig.bufs, &fired, patch);
}

// ── the reference `pie::attn::attn_sink_rescale` produced on this L40S ───────
//
// Emitted by a host harness that includes `attn/attn_sink.cuh` from this tree and
// launches `attn_sink_rescale<pie::bf16>` on the fixture below — so the numbers
// are the cuda plane's own kernel's, not a third reading of its header.
// `nvcc -arch=sm_89`, CUDA 13.0, on the same card this suite dispatches to.
//
// They are ALSO byte-for-byte the constants `driver-metal`'s and `driver-wgpu`'s
// `device_sink` carry, each produced independently for its own slice. THREE
// harnesses, one kernel, identical bytes: the reference is the kernel's and not
// any harness's.
//
// The three rows are the three cases the rescale has to separate:
//   row 0  finite lse spread around the sinks — the factor is doing work, and it
//          is 0.1027, 0.6953, 0.2676, 0.8438 across the four heads;
//   row 1  lse far ABOVE every sink, so the factor is 1 and the output must come
//          back untouched — a kernel that rescaled anyway fails here;
//   row 2  `lse = -inf`, the row that kept no key, where the finiteness test is
//          what keeps a NaN out of the output.
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

/// gpt-oss's own `self_attn.sinks` values as the ledger records them (2.515625,
/// 0.55859375), plus a negative one so a head where the sink loses to the
/// normaliser is covered, and one larger than any of them.
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

/// The rig the sink tests use: the `attn_sink_rescale` fixture written into the
/// handles the claim body binds. It shares [`rig`]'s allocation so that one
/// fixture serves both families; only the three buffers this point reads are
/// overwritten.
fn sink_rig<'a>(device: &'a Device) -> Rig<'a> {
    let rig = rig(device, &mask_bytes(), MASK_ON);
    let o: Vec<u8> = O_IN.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sinks: Vec<u8> = SINKS.iter().flat_map(|w| w.to_le_bytes()).collect();
    rig.write(H_O, &o);
    rig.write(H_LSE, &f32_bytes(&SINK_LSE));
    rig.write(H_SINKS, &sinks);
    rig
}

// ── the tests ───────────────────────────────────────────────────────────────

/// **The lse arms answer the same softmax and publish its denominator.**
///
/// Both bodies, against a reference this file computes from the fixture: the
/// output, and the log-sum-exp, which nothing checked before because nothing was
/// written.
#[test]
fn the_lse_arms_publish_a_base_two_log_sum_exp() {
    let device = gpu!();
    let mask = mask_bytes();
    let rig = rig(&device, &mask, MASK_ON);

    for tiled in [false, true] {
        let which = if tiled { "prefill_lse" } else { "decode_lse" };
        let (o, lse) = fire_lse(&rig, tiled, None);
        assert_eq!(o.len(), ROWS * Q_HEADS * HEAD_DIM);
        assert_eq!(lse.len(), ROWS * Q_HEADS);

        let mut worst_o = 0.0f32;
        let mut worst_lse = 0.0f32;
        let mut raw = 0.0f32;
        for row in 0..ROWS {
            for head in 0..Q_HEADS {
                let (want_o, want_lse) = rig.reference(row, head, &mask, MASK_ON);
                let got = lse[row * Q_HEADS + head];
                assert_ne!(
                    got, LSE_SENTINEL,
                    "{which}: ({row}, {head})'s lse slot was never written"
                );
                let used = (got - want_lse).abs() / lse_bound(want_lse);
                worst_lse = worst_lse.max(used);
                raw = raw.max((got - want_lse).abs() / want_lse.abs().max(1.0));
                assert!(
                    used <= 1.0,
                    "{which}: lse at ({row}, {head}): device {got}, host \
                     {want_lse}, {used} of its bound"
                );
                for (d, want) in want_o.iter().enumerate() {
                    let at = (row * Q_HEADS + head) * HEAD_DIM + d;
                    let used = (o[at] - want).abs() / bound(*want);
                    worst_o = worst_o.max(used);
                    assert!(
                        used <= 1.0,
                        "{which}: o at ({row}, {head}, {d}): device {}, host \
                         {want}, {used} of its bound",
                        o[at],
                    );
                }
            }
        }
        // THE KEY SETS ARE OBSERVABLE, which is what makes the check above worth
        // running: if the window and the mask took nothing away, a body that
        // ignored both would pass.
        let sizes: Vec<usize> = (0..ROWS)
            .map(|r| rig.keeps(r, &mask, MASK_ON).len())
            .collect();
        assert_eq!(
            sizes,
            vec![3, 3, 3],
            "the fixture's rows must keep a proper subset of their history"
        );
        // The RAW disagreement beside the fraction of the bound, so the number in
        // `lse_bound`'s note is one a reader can re-derive from a run rather than
        // one they have to take on trust.
        println!(
            "{which}: MATCHED {} outputs and {} lse values on {}; worst o \
             {worst_o:.3} of bound, worst lse {worst_lse:.3} of bound \
             ({raw:e} relative, {:.2} ulps)",
            o.len(),
            lse.len(),
            device.name(),
            raw / f32::EPSILON * 2.0,
        );
        tolerance_holds(worst_o, &format!("{which}'s output"));
        tolerance_holds(worst_lse, &format!("{which}'s lse"));
    }
    rig.free();
}

/// **A row that keeps no key publishes `-inf`, and the sink then leaves it
/// alone.** The two halves of the empty row, in one test, because separately
/// neither is worth much.
///
/// It matters because on that row the lse is not a number the softmax computed.
/// `sum_exp` is zero, so what gets published is a SENTINEL, and the two sides of
/// the point have to agree on which one: `-inf` is what flashinfer publishes and
/// what `attn_sink_rescale`'s finiteness branch is written against, on every
/// plane. A plane that published `0` instead — which is what `log2` of an empty
/// sum would be if the accumulator were seeded at one — would hand the sink a
/// finite normaliser and shrink a row that has nothing in it.
#[test]
fn a_row_that_keeps_no_key_publishes_negative_infinity() {
    let device = gpu!();
    // Every row masked, every key disallowed: the one shape this fixture's own
    // numbers cannot otherwise reach, because a query's own K is appended before
    // attention and `kp = q_pos` survives every causal bound and every window.
    let mask = vec![0u8; ROWS * MASK_STRIDE as usize];
    let rig = rig(&device, &mask, [1, 1, 1]);

    for tiled in [false, true] {
        let which = if tiled { "prefill_lse" } else { "decode_lse" };
        let (o, lse) = fire_lse(&rig, tiled, None);
        for (at, v) in lse.iter().enumerate() {
            assert!(
                v.is_infinite() && v.is_sign_negative(),
                "{which}: an empty row's lse at {at} is {v}, not -inf"
            );
        }
        for (at, v) in o.iter().enumerate() {
            assert_eq!(*v, 0.0, "{which}: an empty row's output at {at} is {v}");
        }
        println!("{which}: {} empty rows publish -inf", lse.len());
    }

    // And the rescale reads that sentinel rather than tripping over it. The
    // output is already zero, so what this really asserts is that no NaN came out
    // of `sigmoid(-inf * ln2 - sink)`.
    fire_sink(&rig, ROWS, Q_HEADS, HEAD_DIM, None);
    for (at, v) in bf16_read(&rig.read(H_O)).iter().enumerate() {
        assert!(
            v.is_finite() && *v == 0.0,
            "`attention.sink` turned an empty row's zero at {at} into {v}"
        );
    }
    println!("`attention.sink` leaves an -inf row's zeros alone");
    rig.free();
}

/// **The base is measurable, and a plane that published the wrong one fails
/// here.**
///
/// FAULT INJECTION. `sdpa_lse_base2`'s `kLog2E` is set to 1 in the shipped
/// module, which is exactly the plane that publishes its accumulator's own
/// natural log — the mistake `attn_sink.cuh`'s header records, and the one that
/// answers every OUTPUT check in this tree because the output never sees the lse.
///
/// BOTH ARMS, because both call the same rebase and a mutation that reddened only
/// one would say the other had its own copy.
#[test]
fn an_lse_published_in_the_wrong_base_fails_the_check() {
    let device = gpu!();
    let mask = mask_bytes();
    let rig = rig(&device, &mask, MASK_ON);

    for tiled in [false, true] {
        let which = if tiled { "prefill_lse" } else { "decode_lse" };
        let (_, shipped) = fire_lse(&rig, tiled, None);
        let (natural_o, natural) = fire_lse(&rig, tiled, Some((std::f32::consts::LOG2_E, 1.0)));

        let mut moved = 0;
        for (at, (a, b)) in shipped.iter().zip(&natural).enumerate() {
            let want = rig.reference(at / Q_HEADS, at % Q_HEADS, &mask, MASK_ON).1;
            assert!(
                (a - want).abs() <= lse_bound(want),
                "{which}: the shipped arm should agree at {at}"
            );
            if (b - want).abs() > lse_bound(want) {
                moved += 1;
            }
        }
        assert_eq!(
            moved,
            shipped.len(),
            "{which}: a natural-log lse passed the base-two check at {} of {} \
             slots — the check does not measure the base",
            shipped.len() - moved,
            shipped.len(),
        );
        // THE MUTATION IS CONFINED TO THE LSE. `kLog2E` is read nowhere but
        // `sdpa_lse_base2`, so the output must come back untouched — otherwise
        // the constant this test names is not the one it thinks it is.
        for (at, v) in natural_o.iter().enumerate() {
            let (want_o, _) = rig.reference(
                at / (Q_HEADS * HEAD_DIM),
                (at / HEAD_DIM) % Q_HEADS,
                &mask,
                MASK_ON,
            );
            let want = want_o[at % HEAD_DIM];
            assert!(
                (v - want).abs() <= bound(want),
                "{which}: removing the rebase moved the OUTPUT at {at}, so \
                 `kLog2E` is reached by more than the lse"
            );
        }
        println!(
            "{which}: all {moved} lse slots go red when the rebase is removed, and no output word moves"
        );
    }
    rig.free();
}

/// **`attention.sink` is the kernel cuda serves that point with.**
///
/// Against [`CUDA_O_OUT`] — `pie::attn::attn_sink_rescale<bf16>`'s own bf16 bytes
/// off this card — over identical input bytes. Not a tolerance: bf16 out, so the
/// two are compared as bit patterns and the claim is that they are the same
/// numbers.
#[test]
fn the_sink_rescale_is_the_kernel_cuda_serves() {
    let device = gpu!();
    let rig = sink_rig(&device);

    fire_sink(&rig, SINK_ROWS, SINK_HEADS, SINK_D, None);
    let got = bits_of(&rig.read(H_O));

    let mut differ = Vec::new();
    for (at, want) in CUDA_O_OUT.iter().enumerate() {
        if got[at] != *want {
            differ.push(format!(
                "[{}] vulkan 0x{:04x} ({}) cuda 0x{want:04x} ({})",
                at,
                got[at],
                from_bf16(got[at]),
                from_bf16(*want),
            ));
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {} bf16 words differ from `attn_sink_rescale`'s:\n  {}",
        differ.len(),
        CUDA_O_OUT.len(),
        differ.join("\n  "),
    );
    // The row the factor is 1 on has to have come back untouched, and the row the
    // factor is doing work on has to have MOVED — otherwise a kernel that wrote
    // its input back would pass the comparison above.
    let untouched = SINK_HEADS * SINK_D;
    assert_eq!(
        got[untouched..2 * untouched],
        O_IN[untouched..2 * untouched],
        "row 1's lse is far above every sink, so its factor is 1"
    );
    assert_ne!(
        got[..untouched],
        O_IN[..untouched],
        "row 0's sinks are inside its lse range, so its factor is not 1"
    );
    println!(
        "BIT-IDENTICAL to `pie::attn::attn_sink_rescale` over {} words on {}",
        CUDA_O_OUT.len(),
        device.name(),
    );
    rig.free();
}

/// **The rebase inside the sink is measurable too.**
///
/// FAULT INJECTION, and the mirror of the lse one: `attn/attn_sink.slang`'s
/// `kLn2` is set to 1 in the shipped module, which is the plane that reads a
/// base-two lse as though it were natural log. The sigmoid argument is then off
/// by a factor of 1/0.693 and the comparison against cuda has to go red.
#[test]
fn a_sink_that_does_not_rebase_fails_the_check() {
    let device = gpu!();
    let rig = sink_rig(&device);

    fire_sink(
        &rig,
        SINK_ROWS,
        SINK_HEADS,
        SINK_D,
        Some((std::f32::consts::LN_2, 1.0)),
    );
    let got = bits_of(&rig.read(H_O));

    let differ = got.iter().zip(&CUDA_O_OUT).filter(|(a, b)| a != b).count();
    assert!(
        differ > 0,
        "a sink that does not rebase matched `attn_sink_rescale` on all {} \
         words — the comparison does not measure the rebase",
        CUDA_O_OUT.len(),
    );
    // WHERE it goes red, and not only that it does. Only a row with a finite lse
    // in the sigmoid's own range can move: row 1's factor is 1 in either base
    // (its lse dwarfs every sink both times) and row 2 takes the `-inf` branch
    // before the constant is ever read. So the damage must be confined to row 0
    // and must cover most of it — a mutation that moved a handful of words
    // somewhere else would be measuring something other than the rebase.
    let row = SINK_HEADS * SINK_D;
    let row0 = got[..row]
        .iter()
        .zip(&CUDA_O_OUT[..row])
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differ,
        row0,
        "the missing `ln(2)` moved {} words outside row 0, where the factor is 1 \
         or the lse is -inf and the constant is not read at all",
        differ - row0,
    );
    assert!(
        row0 * 2 > row,
        "only {row0} of row 0's {row} words moved; row 0 is where the factor is \
         strictly between 0 and 1 in both bases, so a missing `ln(2)` has to \
         show across it and not in a corner of it"
    );
    println!(
        "{differ} of {} words go red when the rebase is removed, all {row0} of \
         them in row 0's {row}",
        CUDA_O_OUT.len(),
    );
    rig.free();
}

/// **A published lse, rescaled, IS the folded arm — and they are still not the
/// same statement.**
///
/// The decomposition check. `attn/sdpa_paged.slang`'s `PIE_WITH_SINK` arms fold
/// the sink into the denominator before the division and write no lse; firing
/// `decode_lse` then `sink` over the same bytes has to answer the same numbers,
/// or the two readings are not two readings of one thing.
///
/// They agree to within a bfloat16 store, which is what makes this a
/// DECOMPOSITION. The declaration is what makes it necessary: `decode_lse` states
/// `lse: Out<Tensor<f32>>` with a shape, and an arm that writes no such plane
/// cannot answer it however right its `o` is. That is why the folded arms stay
/// dark on this plane and this file has to hand-state them to fire one.
#[test]
fn a_published_lse_rescaled_is_the_folded_sink_arm() {
    let device = gpu!();
    let mask = mask_bytes();
    let rig = rig(&device, &mask, MASK_ON);
    rig.write(H_SINKS, &bf16_bytes(&sink_seen()));

    for tiled in [false, true] {
        let which = if tiled { "prefill" } else { "decode" };
        let folded = fire_folded(&rig, tiled);

        // The same bytes, the other way round: publish the denominator, then
        // rescale against it.
        let (unrescaled, plain_lse) = fire_lse(&rig, tiled, None);
        assert!(
            plain_lse.iter().all(|v| v.is_finite()),
            "{which}: this fixture's rows all keep keys"
        );
        fire_sink(&rig, ROWS, Q_HEADS, HEAD_DIM, None);
        let decomposed = bf16_read(&rig.read(H_O));

        let mut worst = 0.0f32;
        for (at, (a, b)) in decomposed.iter().zip(&folded).enumerate() {
            let used = (a - b).abs() / bound(*b);
            worst = worst.max(used);
            assert!(
                used <= 1.0,
                "{which}: element {at}: decomposed {a}, folded {b}, {used} of a \
                 bf16 store"
            );
        }
        // And the sink actually did something: an identical answer would also be
        // what two arms that both ignored the sink produce.
        let moved = unrescaled
            .iter()
            .zip(&folded)
            .filter(|(a, b)| (*a - *b).abs() > bound(**b))
            .count();
        assert!(
            moved > 0,
            "{which}: the folded arm answers the UNRESCALED output, so this \
             fixture's sinks change nothing and the agreement above is vacuous"
        );
        println!(
            "{which}: decomposed == folded over {} elements, worst {worst:.3} of \
             a bf16 store; the sink moves {moved} of them",
            folded.len(),
        );
        tolerance_holds(worst, &format!("{which}'s decomposition"));
    }
    rig.free();
}

/// **The two bases are one constant each, and the fault injections above have
/// nowhere else to land.**
///
/// GPU-FREE, and it is what makes "exactly once" a property of the family rather
/// than of two calls.
///
/// It is also the base MEASUREMENT, stated as an assertion. If Slang lowered
/// `exp` the way a fast-math backend might — `exp2(x * log2(e))` — then
/// `log2(e)` would appear once per `exp` call site in these modules, the anchor
/// would not be unique, and the accumulator would already be in base two so the
/// rebase would be wrong. It appears ONCE in each `_lse` module and NOWHERE in
/// the sinkless and folded arms, which says both halves at once: `exp` is
/// `OpExtInst Exp` and the only `log2(e)` on this plane is the one
/// `sdpa_lse_base2` writes.
///
/// The `-inf` beside it is the empty row's stated sentinel, and it is checked
/// here for the same reason: it must exist in the module that publishes it and
/// nowhere it could be confused for something else.
#[test]
fn the_two_bases_are_one_constant_each_and_no_more() {
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native");
        return;
    }

    fn floats(entrypoint: &str, value: f32) -> usize {
        let code = kernels_vulkan::code(entrypoint, Capability::Baseline)
            .unwrap_or_else(|| panic!("`{entrypoint}` is a module this build produced"));
        let words = spirv::words(code).expect("a built module is whole words");
        let mut float32 = None;
        let mut at = 5usize;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            assert!(count > 0);
            if (words[at] & 0xffff) == 22 && count >= 3 && words[at + 2] == 32 {
                float32 = Some(words[at + 1]);
            }
            at += count;
        }
        let Some(float32) = float32 else { return 0 };
        let want = value.to_bits();
        let mut n = 0;
        let mut at = 5usize;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            if (words[at] & 0xffff) == 43
                && count == 4
                && words[at + 1] == float32
                && words[at + 3] == want
            {
                n += 1;
            }
            at += count;
        }
        n
    }

    const LSE_ARMS: [&str; 2] = [
        "sdpa_paged_decode_lse_bfloat16_d_64",
        "sdpa_paged_tiled_lse_bfloat16_d_64",
    ];
    const NO_LSE: [&str; 3] = [
        "sdpa_paged_decode_bfloat16_d_64",
        "sdpa_paged_decode_sink_bfloat16_d_64",
        "sdpa_paged_tiled_sink_bfloat16_d_64",
    ];

    for arm in LSE_ARMS {
        assert_eq!(
            floats(arm, std::f32::consts::LOG2_E),
            1,
            "`{arm}` must carry exactly one `log2(e)`: the rebase, and no `exp` \
             lowered through it"
        );
        assert_eq!(
            floats(arm, f32::NEG_INFINITY),
            1,
            "`{arm}` must carry exactly one `-inf`: the empty row's sentinel"
        );
        assert_eq!(
            floats(arm, std::f32::consts::LN_2),
            0,
            "`{arm}` publishes a base; it does not consume one"
        );
    }
    for arm in NO_LSE {
        assert_eq!(
            floats(arm, std::f32::consts::LOG2_E),
            0,
            "`{arm}` writes no lse, so it has no base to rebase — a `log2(e)` \
             here would mean Slang's `exp` lowers through one and the whole \
             rebase argument is wrong"
        );
    }
    assert_eq!(
        floats("attn_sink_rescale_bfloat16", std::f32::consts::LN_2),
        1,
        "`attn_sink_rescale_bfloat16` must carry exactly one `ln(2)`: the point \
         where the two bases meet"
    );
    assert_eq!(
        floats("attn_sink_rescale_bfloat16", std::f32::consts::LOG2_E),
        0,
        "the sink consumes a base-two lse; it does not produce one"
    );
    println!(
        "the rebase is one constant in each of {} lse arms and the `ln(2)` is \
         one in the sink; {} sinkless arms carry neither",
        LSE_ARMS.len(),
        NO_LSE.len(),
    );
}

/// **The runner says whether it fired.**
///
/// Not gated on having a device, so a run that measured nothing says so rather
/// than reporting the same `ok` as a run that measured everything.
#[test]
fn the_sink_runner_states_whether_it_fired() {
    let required = std::env::var_os("PIE_VULKAN_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    match gpu() {
        Some(device) => {
            let name = device.name().to_string();
            drop(device);
            println!(
                "VULKAN DEVICE: PRESENT ({name}). The three lse/sink points were \
                 fired on it."
            );
        }
        None => {
            let why = NO_DEVICE.get().map_or("no reason recorded", String::as_str);
            println!(
                "VULKAN DEVICE: ABSENT ({why}). Every lse and sink claim here \
                 measured NOTHING."
            );
            assert!(
                !required,
                "PIE_VULKAN_REQUIRE_DEVICE is set and no device opened: {why}"
            );
        }
    }
}
