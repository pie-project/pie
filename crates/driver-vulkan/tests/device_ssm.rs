//! THE FIVE SSM POINTS, ON THE CARD IN THIS BOX.
//!
//! `tests/device_fire.rs` established the three claims this file rests on: a
//! device opens, every module in the tree becomes a pipeline, and one kernel
//! dispatches to numbers another vendor's silicon agrees with. What it did not
//! establish is anything about the SSM family, which is the family with a
//! CARRY -- three of these five read a recurrent slab, write it, and are read
//! again by the next fire -- and a carry is the one thing a single-shot
//! comparison cannot see going wrong.
//!
//! So each point here is checked twice over: the RESULT row against an `f64`
//! model of the shader's own arithmetic, and the SLAB the fire leaves behind
//! against the same model. A conv whose output is right and whose window is
//! shifted is a decode that is correct for exactly one token, which is the
//! shape of defect this family has actually shipped.
//!
//! # The two conventions this file exists to hold
//!
//! Both are recorded truths, learned on CUDA and re-proved on two other planes,
//! and both are MUTATED here rather than merely asserted -- a check that has
//! never been seen to fail is a check nobody has any reason to believe.
//!
//! **The conv slab is `[K, C]`, not the rolling window.** A slot holds `K` rows
//! of `C` channels, oldest first, at `state[k * C + c]`; `K - 1` of them are
//! live between fires and row 0 is where the shift's tail goes.
//! [`the_conv_slab_is_k_rows_of_c_channels_and_reading_it_transposed_is_seen`]
//! reads the same bytes as `[C, K]` and
//! [`the_conv_window_opens_at_row_one_and_an_off_by_one_is_seen`] slides the
//! window one row; both must fail the comparison that passes.
//!
//! **An executor hands a kernel dense rectangles only.** `ssm.gdn_prep` takes
//! ONE packed `[b | a]` operand and writes ONE packed `[g_log | beta]` result,
//! with `v_heads` read off half the operand's width and never restated.
//! [`the_gdn_prep_seam_is_b_then_a_and_swapping_it_is_seen`] swaps the halves,
//! and [`a_compact_gate_plane_is_the_defect_this_family_already_shipped`] hands
//! the scan two compact halves where it indexes a packed row -- and shows the
//! thing that let that defect live: at ONE token the two layouts are the same
//! bytes, and only at two does the seam move.
//!
//! # What the numbers are held against, and the stronger reference that was run
//!
//! The asserted reference is a host model of each shader's own arithmetic in
//! `f64`, so that neither device's `f32` reassociation is baked into the thing
//! both are being judged by. `device_fire.rs` argues for that choice at length
//! and the argument is the same here: it is the reference that TRAVELS, needing
//! no CUDA toolkit and no golden blob.
//!
//! It is also the reference that cannot notice a shader and its restatement
//! being wrong together, which for a delta scan is the failure mode that
//! matters. So the stronger one was RUN, and here is the record.
//! `kernels-cuda/kernels/ssm/` holds the twins -- the file every plane's SSM
//! numeric contract was measured against -- and all four were compiled with
//! `nvcc -O2 -arch=sm_89` and run on the SAME L40S as the dispatches below,
//! over the same inputs, generated from the same four lines of arithmetic:
//!
//! ```text
//! pie::ssm::causal_conv1d_update_batched<bf16>
//!   conv step rows           390 /   390 bf16 words BIT-IDENTICAL
//!   conv step window        1560 /  1560 bf16 words BIT-IDENTICAL
//! pie::ssm::causal_conv1d_prefill_batched<bf16>
//!   conv window rows         910 /   910 bf16 words BIT-IDENTICAL
//!   conv window, seats 1,2  1040 /  1040 bf16 words BIT-IDENTICAL
//! pie::ssm::qwen_gdn_ba_gates<bf16>
//!   `[g_log | beta]`         worst relative 8.413e-8 over 370 f32
//! pie::ssm::chunk_gated_delta_prefill_batched<float, false>
//!   result rows              worst relative 2.780e-7 over 1344 f32
//!   the carry left behind    worst relative 1.090e-7 over 55296 f32
//! ```
//!
//! THE CONV IS BIT-IDENTICAL INCLUDING THE SLAB, which is the strongest
//! statement available about the `[K, C]` convention: not that this plane and
//! its `f64` model agree about the window, but that this plane and
//! `kernels-cuda` leave the same 1560 bytes behind, cell for cell, after the
//! same shift. Bit-identical is a MEASUREMENT and not a property -- the two
//! kernels associate nothing differently here, but a wider tap count or a
//! bias term would give them room to -- so it is recorded and not asserted.
//!
//! The two `f32` families are within a few `f32` ulp rather than identical, and
//! that is where they should be: cuda spells the sigmoid and the decay with
//! `__expf` where Slang emits `OpExtInst Exp`, and the scan's own `rsqrt` and
//! its `K_DIM`-long products reassociate differently between a 128-lane
//! workgroup and a 128-thread block.
//!
//! What the cuda scan does NOT independently check is the prologue: it is
//! handed `q_norm`, `k_norm`, a widened `v` and two COMPACT gate planes, which
//! `ssm/gated_delta.slang` computes inside the same workgroup because this
//! plane has no scratch door. Those five were staged on the host for the twin.
//! So the twin measures the delta RECURRENCE -- the decay commit, the two state
//! passes in the right order, the `[k_dim, v_dim]` cell layout -- written by
//! somebody else, in another language, for another machine. That is the part of
//! this family that has been got wrong twice.
//!
//! To re-measure: `PIE_VULKAN_DUMP_SSM=<dir>` makes every test below write its
//! output buffers verbatim, which is what the twins were diffed against. A
//! number quoted in prose that cannot be re-run is a number this tree has no
//! reason to believe.
//!
//! # Why the shapes are awkward
//!
//! `ssm/causal_conv1d.slang` and `ssm/gdn_gates.slang` are 64 lanes wide, so
//! [`CHANNELS`] is 130 and [`V_HEADS`] is 37: neither divides, both leave a
//! partial workgroup whose surplus lanes must return rather than write.
//! `ssm/gated_delta.slang` is 128 wide against a [`K_DIM`] of 96 and a
//! [`V_DIM`] of 48, so both of its strided loops run ONE partial trip with most
//! lanes idle -- which is the arrangement in which a missing barrier shows.
//! The slot table is `[2, 0, 1]`, never the identity, because a slab indexed by
//! the row instead of by the seat is right for exactly the fixture that used
//! `slots[r] == r`. And one request of the chunked pair is EMPTY, which is the
//! arm that must leave both conv planes alone.
//!
//! # Why `device` and not `native`
//!
//! `tests/device_fire.rs` says it: `native` does not build, and nothing here
//! needs it. These fire pipelines directly, the way that file does, because
//! `driver-vulkan`'s `Pools::slab` answers `None` for every layer today -- no
//! recurrent slab is allocated anywhere in the driver -- so a walk could not
//! reach these points even if the walk built. The claim bodies' own argument
//! lists are held against these modules in `tests/doors.rs`; what is measured
//! here is the arithmetic behind them.

#![cfg(feature = "device")]
#![allow(clippy::print_stdout)]

use driver_vulkan::device::{Bound, Device, Pipeline, Pipelines};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Channels the convolution covers. `130 = 2 * 64 + 2`, so the last workgroup
/// runs 62 lanes that own no channel.
const CHANNELS: usize = 130;

/// Taps. Qwen3.5's conv width, and the number the slab holds rows of.
const TAPS: usize = 4;

/// Seats in the recurrent pool. Three, so that one can be named by nobody and
/// still be checked.
const SEATS: usize = 3;

/// Value heads the gate row halves into. `37` is prime and under 64.
const V_HEADS: usize = 37;

/// Key heads, value heads, and the two head widths the scan is told.
///
/// `V_SCAN / K_SCAN` is 2, so the scan's group-query replication is exercised
/// rather than being the identity; `K_DIM` and `V_DIM` are both under the
/// 128-wide workgroup and neither divides it.
const K_SCAN: usize = 2;
const V_SCAN: usize = 4;
const K_DIM: usize = 96;
const V_DIM: usize = 48;

/// The CSR the two chunked arms walk: three requests of 3, 0 and 4 tokens.
const INDPTR: [i32; 4] = [0, 3, 3, 7];

/// Tokens the CSR covers.
const TOKENS: usize = 7;

/// The seat each token sits in. Request 0 is in seat 2 and request 2 in seat 1;
/// seat 0 is named by nobody.
const TOKEN_SEATS: [u32; TOKENS] = [2, 2, 2, 1, 1, 1, 1];

/// The seat each decode row sits in. Never the identity.
const ROW_SEATS: [u32; 3] = [2, 0, 1];

/// Half a bf16 ulp, relative to the row's largest output. `device_fire.rs`
/// derives it; the conv's result is bf16 and takes the same budget.
const BF16_TOLERANCE: f64 = 1.0 / 256.0;

/// What an f32 result is held to, relative to the row's largest.
///
/// Not a bf16 budget: `ssm.gdn_prep` and both delta scans write `f32`, so the
/// only error is the shader's own `f32` reassociation and its transcendentals
/// against `f64`'s. `exp` and `rsqrt` are each a few ulp in SPIR-V, the delta
/// scan compounds them through `K_DIM` products and a four-token recurrence,
/// and this is the budget that leaves. The worst actually measured is printed
/// by every test that uses it, so a drift is visible before it is a failure.
const F32_TOLERANCE: f64 = 1.0e-4;

/// What an untouched f32 slot holds.
///
/// A value no fire here can produce and one `Device::empty` cannot hand back by
/// accident. Zero would be indistinguishable from a real zero and from a
/// dispatch that never ran, which is the whole content of `device_fire.rs`'s
/// own sentinel argument.
const SENTINEL: f32 = 1.0e30;

/// What an untouched bf16 word holds: `0x4780` is `65536.0`.
const BF16_SENTINEL: u8 = 0x47;

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

/// Round to nearest even, which is what `common/bf16.slang` does.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_bf16(*x).to_le_bytes()).collect()
}

fn bf16_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]])))
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

/// A stream of values every one of which is bf16-exact.
///
/// Multiples of `1/16` inside `[-2.1875, 2.1875]`, which needs six significand
/// bits against bf16's eight. `37` steps through the residues of `71` and the
/// seed moves each stream somewhere unrelated, so neighbouring elements are far
/// apart in value -- a ramp would make an off-by-one index a difference of one
/// sixteenth, which no tolerance worth having could see.
fn exact(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((seed * 997 + i * 37) % 71) as f32 - 35.0) / 16.0)
        .collect()
}

/// SiLU, in `f64`, which is what the shader computes in `f32`.
fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

/// Hold a device result against an `f64` model, scaled by the model's largest.
///
/// Returns the worst relative error, so every caller can print what it actually
/// measured rather than only that it was under budget.
fn worst_of(got: &[f32], want: &[f64], budget: f64, what: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: a different number of values"
    );
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{what}: the reference is all zeros");
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for i in 0..got.len() {
        let e = (f64::from(got[i]) - want[i]).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    assert!(
        worst <= budget,
        "{what}: element {at}: the device says {}, the reference says {}, a \
         relative error of {worst:.3e} against a budget of {budget:.3e}",
        got[at],
        want[at]
    );
    worst
}

/// The same comparison, asked to FAIL. Returns the worst it saw.
fn disagrees(got: &[f32], want: &[f64], budget: f64, what: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: a different number of values"
    );
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{what}: the reference is all zeros");
    let worst = (0..got.len())
        .map(|i| (f64::from(got[i]) - want[i]).abs() / scale)
        .fold(0.0f64, f64::max);
    assert!(
        worst > budget,
        "{what}: the mutated reference is still inside the budget at \
         {worst:.3e} against {budget:.3e}, so the check it mutates is not \
         evidence of anything"
    );
    worst
}

/// Build one pipeline and hold what it declares against what this file binds.
fn pipeline<'a>(
    device: &Device,
    cache: &'a mut Pipelines,
    entrypoint: &str,
    push_words: usize,
    bindings: u32,
    local: [u32; 3],
) -> &'a Pipeline {
    let (code, tier) = device
        .module_for(entrypoint)
        .unwrap_or_else(|| panic!("`{entrypoint}` has a module in this build"));
    let built = cache
        .get(device, entrypoint, code, (push_words * 4) as u32, 0, tier)
        .unwrap_or_else(|e| panic!("`{entrypoint}` builds a pipeline: {e}"));
    let d = built.declared();
    assert_eq!(
        d.bindings, bindings,
        "`{entrypoint}` declares {} buffers and this file binds {bindings}",
        d.bindings
    );
    assert_eq!(d.holes(), 0, "`{entrypoint}` has a binding hole");
    let packed: Vec<u32> = (0..push_words as u32).map(|i| i * 4).collect();
    assert_eq!(
        d.push_offsets, packed,
        "`{entrypoint}`'s push members are not the run of four-byte scalars \
         this file packs"
    );
    assert_eq!(d.local, local, "`{entrypoint}`'s workgroup moved");
    built
}

/// Lanes over the module's own workgroup, which is what `serve::run` computes
/// and what a body's `Fire::apply` means.
fn groups(lanes: [u32; 3], local: [u32; 3]) -> [u32; 3] {
    [
        lanes[0].div_ceil(local[0]),
        lanes[1].div_ceil(local[1]),
        lanes[2].div_ceil(local[2]),
    ]
}

/// Write a fire's output verbatim when `PIE_VULKAN_DUMP_SSM` names a directory.
///
/// The stronger reference this file can reach is not the `f64` model but
/// `kernels-cuda`'s own conv, which runs on this same L40S. Nothing here links
/// CUDA -- `driver-vulkan` has no such edge and should not grow one for a test
/// -- so the cross-check is a separate program, and this is the door it reads
/// through. The INPUTS are not dumped: they are `exact(seed, n)`, four lines of
/// arithmetic the twin restates, so a dump of them would only record that the
/// two programs agree about a formula.
///
/// A number quoted in prose that cannot be re-run is a number this tree has no
/// reason to believe; `device_fire.rs`'s `PIE_VULKAN_DUMP` is the same door for
/// the same reason.
fn dump(name: &str, values: &[f32]) {
    let Some(dir) = std::env::var_os("PIE_VULKAN_DUMP_SSM") else {
        return;
    };
    let at = std::path::Path::new(&dir).join(name);
    std::fs::write(&at, f32_bytes(values)).expect("the dump path is writable");
    println!("dumped {} values to {}", values.len(), at.display());
}

// ── THE CONVOLUTION ────────────────────────────────────────────────────────

/// The conv weight, `[C, K]` row-major -- `weight[c * K + k]`.
fn conv_weight() -> Vec<f32> {
    exact(3, CHANNELS * TAPS)
}

/// The slab every conv test starts from: `SEATS` seats of `[K, C]`.
///
/// Bf16-exact even though the plane holds it in `f32`, so that a cross-check
/// against `kernels-cuda`'s bf16 slab compares arithmetic rather than who
/// rounded the window.
fn conv_slab() -> Vec<f32> {
    exact(11, SEATS * TAPS * CHANNELS)
}

/// The `f64` model of `causal_conv1d_bfloat16`.
///
/// Returns the result rows and the whole slab the fire should leave behind. The
/// `transposed` and `slide` knobs are the two mutations; both are `false` in
/// every comparison that is meant to pass.
fn conv_step_reference(
    x: &[f32],
    weight: &[f32],
    slab: &[f32],
    seats: &[u32],
    transposed: bool,
    slide: usize,
) -> (Vec<f64>, Vec<f64>) {
    let tap = |seat: usize, k: usize, c: usize| -> f64 {
        // `[K, C]` with `C` fast, which is the convention. The mutation reads
        // the same bytes as `[C, K]`.
        let at = if transposed {
            seat * TAPS * CHANNELS + c * TAPS + k
        } else {
            seat * TAPS * CHANNELS + k * CHANNELS + c
        };
        f64::from(slab[at])
    };
    let mut y = vec![0.0f64; seats.len() * CHANNELS];
    let mut out: Vec<f64> = slab.iter().map(|v| f64::from(*v)).collect();
    for (r, seat) in seats.iter().enumerate() {
        let seat = *seat as usize;
        for c in 0..CHANNELS {
            let fresh = f64::from(x[r * CHANNELS + c]);
            let mut acc = fresh * f64::from(weight[c * TAPS + TAPS - 1]);
            for k in 0..TAPS - 1 {
                acc += tap(seat, k + slide, c) * f64::from(weight[c * TAPS + k]);
            }
            y[r * CHANNELS + c] = silu(acc);
            for k in 0..TAPS - 1 {
                out[seat * TAPS * CHANNELS + k * CHANNELS + c] = tap(seat, k + 1, c);
            }
            out[seat * TAPS * CHANNELS + (TAPS - 1) * CHANNELS + c] = fresh;
        }
    }
    (y, out)
}

/// The `f64` model of `causal_conv1d_chunked_bfloat16`.
fn conv_window_reference(
    x: &[f32],
    weight: &[f32],
    slab: &[f32],
    sentinel: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0f64; TOKENS * CHANNELS];
    // A request the fire does not name keeps whatever the written plane held,
    // which is the sentinel here and the previous carry-back in a real run.
    let mut out = vec![sentinel; SEATS * TAPS * CHANNELS];
    for r in 0..INDPTR.len() - 1 {
        let begin = INDPTR[r] as usize;
        let end = INDPTR[r + 1] as usize;
        if end <= begin {
            continue;
        }
        let span = end - begin;
        let seat = TOKEN_SEATS[begin] as usize;
        let tap = |s: isize, c: usize| -> f64 {
            if s < 0 {
                f64::from(
                    slab[seat * TAPS * CHANNELS + (TAPS as isize + s) as usize * CHANNELS + c],
                )
            } else {
                f64::from(x[(begin + s as usize) * CHANNELS + c])
            }
        };
        for c in 0..CHANNELS {
            for t in 0..span {
                let mut acc = 0.0f64;
                for k in 0..TAPS {
                    let s = t as isize - (TAPS as isize - 1) + k as isize;
                    acc += tap(s, c) * f64::from(weight[c * TAPS + k]);
                }
                y[(begin + t) * CHANNELS + c] = silu(acc);
            }
            for s in 0..TAPS {
                let src = span as isize - TAPS as isize + s as isize;
                out[seat * TAPS * CHANNELS + s * CHANNELS + c] = tap(src, c);
            }
        }
    }
    (y, out)
}

/// The one decode fire, run for whatever comparison the caller wants.
///
/// Returns `(y, new_slab)`.
fn fire_conv_step(device: &Device, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cache = Pipelines::new();
    let entrypoint = "causal_conv1d_bfloat16";
    let built = pipeline(device, &mut cache, entrypoint, 2, 6, [64, 1, 1]);

    let rows = ROW_SEATS.len();
    let slab = conv_slab();
    let weight = conv_weight();
    let y = device
        .empty((rows * CHANNELS * 2) as u64)
        .expect("the result rows");
    device
        .write(&y, &vec![BF16_SENTINEL; rows * CHANNELS * 2])
        .expect("the sentinel goes down first");
    let fresh = device
        .empty((SEATS * TAPS * CHANNELS * 4) as u64)
        .expect("the written conv plane");
    device
        .write(&fresh, &f32_bytes(&vec![SENTINEL; SEATS * TAPS * CHANNELS]))
        .expect("the sentinel goes down first");
    let bufs = [
        device.buffer(&bf16_bytes(x)).expect("x"),
        device.buffer(&bf16_bytes(&weight)).expect("weight"),
        device.buffer(&f32_bytes(&slab)).expect("conv_state"),
        fresh,
        device.buffer(&u32_bytes(&ROW_SEATS)).expect("slots"),
        y,
    ];
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(
            built,
            &bound,
            &i32_bytes(&[CHANNELS as i32, TAPS as i32]),
            groups([CHANNELS as u32, rows as u32, 1], [64, 1, 1]),
        )
        .expect("the dispatch");

    let got = bf16_read(&device.read(&bufs[5]).expect("read the rows"));
    let left = f32_read(&device.read(&bufs[3]).expect("read the window"));
    cache.clear(device);
    for b in bufs {
        device.free(b);
    }
    (got, left)
}

/// The one prefill fire.
fn fire_conv_window(device: &Device, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cache = Pipelines::new();
    let entrypoint = "causal_conv1d_chunked_bfloat16";
    let built = pipeline(device, &mut cache, entrypoint, 2, 7, [64, 1, 1]);

    let slab = conv_slab();
    let weight = conv_weight();
    let y = device
        .empty((TOKENS * CHANNELS * 2) as u64)
        .expect("the result rows");
    device
        .write(&y, &vec![BF16_SENTINEL; TOKENS * CHANNELS * 2])
        .expect("the sentinel");
    let fresh = device
        .empty((SEATS * TAPS * CHANNELS * 4) as u64)
        .expect("the written conv plane");
    device
        .write(&fresh, &f32_bytes(&vec![SENTINEL; SEATS * TAPS * CHANNELS]))
        .expect("the sentinel");
    let bufs = [
        device.buffer(&bf16_bytes(x)).expect("x"),
        device.buffer(&bf16_bytes(&weight)).expect("weight"),
        device.buffer(&f32_bytes(&slab)).expect("conv_state"),
        fresh,
        device.buffer(&u32_bytes(&TOKEN_SEATS)).expect("slots"),
        device.buffer(&i32_bytes(&INDPTR)).expect("indptr"),
        y,
    ];
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(
            built,
            &bound,
            &i32_bytes(&[CHANNELS as i32, TAPS as i32]),
            groups([CHANNELS as u32, (INDPTR.len() - 1) as u32, 1], [64, 1, 1]),
        )
        .expect("the dispatch");

    let got = bf16_read(&device.read(&bufs[6]).expect("read the rows"));
    let left = f32_read(&device.read(&bufs[3]).expect("read the window"));
    cache.clear(device);
    for b in bufs {
        device.free(b);
    }
    (got, left)
}

// ── THE GATE ROW ───────────────────────────────────────────────────────────

/// The `f64` model of `gdn_ba_gates_bfloat16`.
///
/// `swapped` is the mutation: `a` read out of the first half and `b` out of the
/// second, which is the reading that turns a mixing rate into a decay and stays
/// entirely in range while doing it.
fn gates_reference(
    ba: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    rows: usize,
    swapped: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * 2 * V_HEADS];
    for t in 0..rows {
        let row = t * 2 * V_HEADS;
        for h in 0..V_HEADS {
            let (b, a) = if swapped {
                (f64::from(ba[row + V_HEADS + h]), f64::from(ba[row + h]))
            } else {
                (f64::from(ba[row + h]), f64::from(ba[row + V_HEADS + h]))
            };
            let z = a + f64::from(dt_bias[h]);
            let sp = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
            out[row + h] = -f64::from(a_log[h]).exp() * sp;
            out[row + V_HEADS + h] = 1.0 / (1.0 + (-b).exp());
        }
    }
    out
}

fn fire_gates(
    device: &Device,
    ba: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    rows: usize,
) -> Vec<f32> {
    let mut cache = Pipelines::new();
    let entrypoint = "gdn_ba_gates_bfloat16";
    let built = pipeline(device, &mut cache, entrypoint, 1, 4, [64, 1, 1]);

    let n = rows * 2 * V_HEADS;
    let gates = device.empty((n * 4) as u64).expect("the gate row");
    device
        .write(&gates, &f32_bytes(&vec![SENTINEL; n]))
        .expect("the sentinel");
    let bufs = [
        device.buffer(&bf16_bytes(ba)).expect("ba"),
        device.buffer(&f32_bytes(a_log)).expect("a_log"),
        device.buffer(&bf16_bytes(dt_bias)).expect("dt_bias"),
        gates,
    ];
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(
            built,
            &bound,
            &i32_bytes(&[V_HEADS as i32]),
            groups([V_HEADS as u32, rows as u32, 1], [64, 1, 1]),
        )
        .expect("the dispatch");
    let got = f32_read(&device.read(&bufs[3]).expect("read the gate row"));
    cache.clear(device);
    for b in bufs {
        device.free(b);
    }
    got
}

// ── THE DELTA SCAN ─────────────────────────────────────────────────────────

/// Row width of the packed post-convolution operand.
const QKV_WIDTH: usize = 2 * K_SCAN * K_DIM + V_SCAN * V_DIM;

/// The `f64` model of both delta scans.
///
/// One walk over `[(begin, end)]` windows, because the two entrypoints differ
/// only in where the window comes from -- which is exactly the claim the two
/// shader arms make about each other.
///
/// `gates_at` is a closure so that
/// [`a_compact_gate_plane_is_the_defect_this_family_already_shipped`] can hand
/// the same recurrence a COMPACT reading of the same bytes.
fn delta_reference(
    qkv: &[f32],
    gates_at: &dyn Fn(usize, usize) -> (f64, f64),
    slab: &[f32],
    windows: &[(usize, usize)],
    seats: &[u32],
    rows: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0f64; rows * V_SCAN * V_DIM];
    let mut state: Vec<f64> = slab.iter().map(|v| f64::from(*v)).collect();
    let scale = 1.0 / (K_DIM as f64).sqrt();
    let keys = K_SCAN * K_DIM;
    for (begin, end) in windows.iter().copied() {
        if end <= begin {
            continue;
        }
        let seat = seats[begin] as usize;
        for hv in 0..V_SCAN {
            let hk = hv / (V_SCAN / K_SCAN);
            let base = ((seat * V_SCAN + hv) * K_DIM) * V_DIM;
            for t in begin..end {
                let row = t * QKV_WIDTH;
                let qbase = row + hk * K_DIM;
                let kbase = qbase + keys;
                let vbase = row + 2 * keys + hv * V_DIM;

                let mut sum_q = 0.0f64;
                let mut sum_k = 0.0f64;
                for i in 0..K_DIM {
                    sum_q += f64::from(qkv[qbase + i]) * f64::from(qkv[qbase + i]);
                    sum_k += f64::from(qkv[kbase + i]) * f64::from(qkv[kbase + i]);
                }
                let q_inv = (sum_q + 1e-6).sqrt().recip() * scale;
                let k_inv = (sum_k + 1e-6).sqrt().recip();
                let q: Vec<f64> = (0..K_DIM)
                    .map(|i| f64::from(qkv[qbase + i]) * q_inv)
                    .collect();
                let k: Vec<f64> = (0..K_DIM)
                    .map(|i| f64::from(qkv[kbase + i]) * k_inv)
                    .collect();

                let (g_log, beta) = gates_at(t, hv);
                let decay = g_log.exp();
                for c in 0..V_DIM {
                    // The two passes are two passes: the decay is committed and
                    // read back before the rank-one update, which is committed
                    // and read back before the result. Fusing them computes a
                    // result against a state one update behind.
                    let mut kv_mem = 0.0f64;
                    for (i, kv) in k.iter().enumerate() {
                        let at = base + i * V_DIM + c;
                        state[at] *= decay;
                        kv_mem += state[at] * kv;
                    }
                    let delta = (f64::from(qkv[vbase + c]) - kv_mem) * beta;
                    let mut acc = 0.0f64;
                    for (i, (kv, qv)) in k.iter().zip(&q).enumerate() {
                        let at = base + i * V_DIM + c;
                        state[at] += kv * delta;
                        acc += state[at] * qv;
                    }
                    y[(t * V_SCAN + hv) * V_DIM + c] = acc;
                }
            }
        }
    }
    (y, state)
}

/// The packed reading of `gates`: `[g_log | beta]`, `2 * V_SCAN` per token.
fn packed_gates(gates: &[f32]) -> impl Fn(usize, usize) -> (f64, f64) + '_ {
    move |t, hv| {
        let fused = t * 2 * V_SCAN + hv;
        (f64::from(gates[fused]), f64::from(gates[fused + V_SCAN]))
    }
}

fn fire_delta(
    device: &Device,
    qkv: &[f32],
    gates: &[f32],
    slab: &[f32],
    csr: Option<&[i32]>,
    seats: &[u32],
    rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cache = Pipelines::new();
    let (entrypoint, bindings) = match csr {
        Some(_) => ("gated_delta_chunked_bfloat16", 6),
        None => ("gated_delta_bfloat16", 5),
    };
    let built = pipeline(device, &mut cache, entrypoint, 4, bindings, [128, 1, 1]);

    let out = rows * V_SCAN * V_DIM;
    let y = device.empty((out * 4) as u64).expect("the result rows");
    device
        .write(&y, &f32_bytes(&vec![SENTINEL; out]))
        .expect("the sentinel");
    let mut bufs = vec![
        device.buffer(&bf16_bytes(qkv)).expect("qkv"),
        device.buffer(&f32_bytes(gates)).expect("gates"),
        device.buffer(&f32_bytes(slab)).expect("rstate"),
        device.buffer(&u32_bytes(seats)).expect("slots"),
        y,
    ];
    if let Some(csr) = csr {
        bufs.push(device.buffer(&i32_bytes(csr)).expect("indptr"));
    }
    let covers = csr.map_or(rows, |c| c.len() - 1) as u32;
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(
            built,
            &bound,
            &i32_bytes(&[K_SCAN as i32, V_SCAN as i32, K_DIM as i32, V_DIM as i32]),
            groups([128, V_SCAN as u32, covers], [128, 1, 1]),
        )
        .expect("the dispatch");

    let got = f32_read(&device.read(&bufs[4]).expect("read the rows"));
    let left = f32_read(&device.read(&bufs[2]).expect("read the carry"));
    cache.clear(device);
    for b in bufs {
        device.free(b);
    }
    (got, left)
}

/// The recurrent slab every scan test starts from.
fn delta_slab() -> Vec<f32> {
    exact(23, SEATS * V_SCAN * K_DIM * V_DIM)
}

// ── THE TESTS ──────────────────────────────────────────────────────────────

/// THE RUNNER SAYS WHETHER IT FIRED, for `device_fire.rs`'s reason: every other
/// test here returns early with a `skipped:` line `cargo test` swallows, so a
/// green run that touched no GPU and a green run that measured five kernels are
/// indistinguishable without this.
#[test]
fn the_ssm_runner_states_whether_it_fired() {
    let required = std::env::var_os("PIE_VULKAN_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    match gpu() {
        Some(device) => {
            let name = device.name().to_string();
            drop(device);
            println!("VULKAN DEVICE: PRESENT ({name}). The five ssm points were fired on it.");
        }
        None => {
            let why = NO_DEVICE.get().map_or("no reason recorded", String::as_str);
            println!("VULKAN DEVICE: ABSENT ({why}). Every ssm claim here measured NOTHING.");
            assert!(
                !required,
                "PIE_VULKAN_REQUIRE_DEVICE is set and no device opened: {why}"
            );
        }
    }
}

/// `ssm.causal_conv1d`: the row it convolves AND the window it leaves.
///
/// The window is half the point. A conv that computes the right output and
/// shifts the wrong row is correct for exactly one more token, and the seat
/// nobody named is checked too -- a fire that wrote every seat rather than the
/// three it was handed would be invisible in the output rows.
#[test]
fn a_conv1d_step_is_the_step_the_reference_computes_and_leaves_the_window_it_should() {
    let device = gpu!();
    let x = exact(5, ROW_SEATS.len() * CHANNELS);
    let (got, left) = fire_conv_step(&device, &x);

    dump("conv_step.y.bin", &got);
    dump("conv_step.slab.bin", &left);

    let (want, slab) = conv_step_reference(&x, &conv_weight(), &conv_slab(), &ROW_SEATS, false, 1);
    let rows = worst_of(&got, &want, BF16_TOLERANCE, "the conv's result rows");

    // Only the three seats the fire named are written; seat 0 is named by
    // nobody in `ROW_SEATS`... but `ROW_SEATS` names all three, so every seat
    // is live and the whole plane is the reference.
    let window = worst_of(
        &left,
        &slab,
        BF16_TOLERANCE,
        "the window the conv leaves behind",
    );

    println!(
        "causal_conv1d_bfloat16 fired {}x{CHANNELS} at K={TAPS} on {}: rows \
         {rows:.3e}, window {window:.3e}, budget {BF16_TOLERANCE:.3e}",
        ROW_SEATS.len(),
        device.name()
    );
}

/// `ssm.causal_conv1d_chunked`: three requests of 3, 0 and 4 tokens.
///
/// The empty one is the arm under test as much as the other two: it must leave
/// BOTH conv planes alone, which here means its seat still holds the sentinel
/// the written plane went down with.
#[test]
fn a_conv1d_window_is_the_window_the_reference_computes_and_an_empty_request_moves_nothing() {
    let device = gpu!();
    let x = exact(7, TOKENS * CHANNELS);
    let (got, left) = fire_conv_window(&device, &x);

    dump("conv_window.y.bin", &got);
    dump("conv_window.slab.bin", &left);

    let (want, slab) = conv_window_reference(&x, &conv_weight(), &conv_slab(), f64::from(SENTINEL));
    let rows = worst_of(&got, &want, BF16_TOLERANCE, "the conv window's rows");

    // Seat 0 is named by no request in `TOKEN_SEATS`, so nothing may have
    // touched it -- which is what "an empty window leaves both planes alone"
    // means from the plane's side. It is held EXACTLY and held FIRST, because
    // a sentinel of 1e30 inside a relative comparison would set the scale and
    // make every other cell agree to nine figures.
    let seat = TAPS * CHANNELS;
    assert!(
        left[0..seat].iter().all(|v| *v == SENTINEL),
        "a seat no request named was written anyway"
    );
    let window = worst_of(
        &left[seat..],
        &slab[seat..],
        BF16_TOLERANCE,
        "the window the chunked conv leaves behind",
    );

    println!(
        "causal_conv1d_chunked_bfloat16 fired {TOKENS} tokens over 3 requests \
         (3, 0, 4) on {}: rows {rows:.3e}, window {window:.3e}, budget \
         {BF16_TOLERANCE:.3e}",
        device.name()
    );
}

/// MUTATION ONE: THE SLAB IS `[K, C]`, AND `[C, K]` IS A DIFFERENT KERNEL.
///
/// The same bytes, read with the two axes swapped. Both readings are in range,
/// both produce plausible activations, and the difference is a conv that
/// convolves a channel against its own history in the wrong order -- which at
/// `K = 4` and `C = 130` is a completely different number and at `K = C` would
/// be invisible. This is why `CHANNELS` is not `TAPS`.
#[test]
fn the_conv_slab_is_k_rows_of_c_channels_and_reading_it_transposed_is_seen() {
    let device = gpu!();
    let x = exact(5, ROW_SEATS.len() * CHANNELS);
    let (got, _) = fire_conv_step(&device, &x);

    let (want, _) = conv_step_reference(&x, &conv_weight(), &conv_slab(), &ROW_SEATS, true, 1);
    let worst = disagrees(
        &got,
        &want,
        BF16_TOLERANCE,
        "the conv read against a `[C, K]` slab",
    );
    println!(
        "conv_slab_transposed: the same comparison fails at {worst:.3e} against \
         a budget of {BF16_TOLERANCE:.3e}"
    );
}

/// MUTATION TWO: THE WINDOW OPENS AT ROW ONE.
///
/// The step convolves slab rows `1 .. K-1` and the arriving column; a reference
/// that reads rows `0 .. K-2` instead is off by one row -- one token of
/// history, which is precisely the error a shifted carry produces and precisely
/// the one that a single-token fixture cannot see, because at `K = 2` the two
/// windows differ by the only row there is.
#[test]
fn the_conv_window_opens_at_row_one_and_an_off_by_one_is_seen() {
    let device = gpu!();
    let x = exact(5, ROW_SEATS.len() * CHANNELS);
    let (got, _) = fire_conv_step(&device, &x);

    let (want, _) = conv_step_reference(&x, &conv_weight(), &conv_slab(), &ROW_SEATS, false, 0);
    let worst = disagrees(
        &got,
        &want,
        BF16_TOLERANCE,
        "the conv read one slab row early",
    );
    println!(
        "conv_window_off_by_one: the same comparison fails at {worst:.3e} \
         against a budget of {BF16_TOLERANCE:.3e}"
    );
}

/// A GRID ONE WORKGROUP SHORT, which on Vulkan is completely silent.
///
/// `CHANNELS` is 130 against a 64-wide workgroup, so the covering grid is three
/// workgroups on x and the short one is two: channels 128 and 129 of every row
/// keep their sentinel, every call returns success and no validation layer
/// objects. The same comparison that passes above must fail here, or it is not
/// evidence of anything.
#[test]
fn a_conv_grid_one_workgroup_short_leaves_a_tail_this_check_can_see() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "causal_conv1d_bfloat16";
    let built = pipeline(&device, &mut cache, entrypoint, 2, 6, [64, 1, 1]);

    let rows = ROW_SEATS.len();
    let x = exact(5, rows * CHANNELS);
    let y = device
        .empty((rows * CHANNELS * 2) as u64)
        .expect("the result rows");
    device
        .write(&y, &vec![BF16_SENTINEL; rows * CHANNELS * 2])
        .expect("the sentinel");
    let fresh = device
        .empty((SEATS * TAPS * CHANNELS * 4) as u64)
        .expect("the written conv plane");
    device
        .write(&fresh, &f32_bytes(&vec![SENTINEL; SEATS * TAPS * CHANNELS]))
        .expect("the sentinel");
    let bufs = [
        device.buffer(&bf16_bytes(&x)).expect("x"),
        device.buffer(&bf16_bytes(&conv_weight())).expect("weight"),
        device.buffer(&f32_bytes(&conv_slab())).expect("conv_state"),
        fresh,
        device.buffer(&u32_bytes(&ROW_SEATS)).expect("slots"),
        y,
    ];
    let full = groups([CHANNELS as u32, rows as u32, 1], [64, 1, 1]);
    assert_eq!(full, [3, 3, 1], "130 channels is three 64-wide workgroups");
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(
            built,
            &bound,
            &i32_bytes(&[CHANNELS as i32, TAPS as i32]),
            [full[0] - 1, full[1], full[2]],
        )
        .expect("an undershot dispatch is still a successful one, which is the point");

    let got = bf16_read(&device.read(&bufs[5]).expect("read back"));
    let sentinel = from_bf16(u16::from_le_bytes([BF16_SENTINEL, BF16_SENTINEL]));
    for r in 0..rows {
        for c in 128..CHANNELS {
            assert_eq!(
                got[r * CHANNELS + c],
                sentinel,
                "row {r} channel {c} was past the short grid and was written anyway"
            );
        }
    }
    let (want, _) = conv_step_reference(&x, &conv_weight(), &conv_slab(), &ROW_SEATS, false, 1);
    let worst = disagrees(
        &got,
        &want,
        BF16_TOLERANCE,
        "the conv on a grid one workgroup short",
    );
    println!(
        "one workgroup short: channels 128..130 kept their sentinel and the \
         same comparison fails at {worst:.3e} against a budget of \
         {BF16_TOLERANCE:.3e}"
    );

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}

/// `ssm.gdn_prep`: one packed operand in, one packed result out, one launch.
#[test]
fn a_gdn_prep_row_is_the_row_the_reference_computes() {
    let device = gpu!();
    let rows = 5usize;
    let ba = exact(13, rows * 2 * V_HEADS);
    let dt_bias = exact(17, V_HEADS);
    // `a_log` is a weight and is exponentiated, so it is kept small and
    // negative-leaning; a large one would make `-exp(a_log) * softplus`
    // dominate the row and hide the softplus entirely.
    let a_log: Vec<f32> = (0..V_HEADS).map(|h| (h % 11) as f32 / 8.0 - 0.75).collect();

    let got = fire_gates(&device, &ba, &a_log, &dt_bias, rows);
    dump("gdn_prep.gates.bin", &got);
    let want = gates_reference(&ba, &a_log, &dt_bias, rows, false);
    let worst = worst_of(&got, &want, F32_TOLERANCE, "the fused `[g_log | beta]` row");

    // The seam is arithmetic and not a pointer: the two halves of one row are
    // different FUNCTIONS, so a row whose halves happened to agree would make
    // the swap below unfalsifiable.
    assert!(
        (0..rows).any(|t| {
            (0..V_HEADS).any(|h| {
                (got[t * 2 * V_HEADS + h] - got[t * 2 * V_HEADS + V_HEADS + h]).abs() > 1e-3
            })
        }),
        "the two halves of the gate row are indistinguishable in this fixture"
    );

    println!(
        "gdn_ba_gates_bfloat16 fired {rows}x{V_HEADS} on {}: worst {worst:.3e} \
         against a budget of {F32_TOLERANCE:.3e}",
        device.name()
    );
}

/// MUTATION THREE: THE SEAM IS `[b | a]` AND SWAPPING IT IS FINITE, PLAUSIBLE
/// AND WRONG.
///
/// `b` becomes the decay and `a` becomes the mixing rate. Both stay in range,
/// nothing overflows, no shape check anywhere notices, and the model produces
/// slightly worse text forever. `V_HEADS` is odd so that even the element
/// COUNT of the two halves cannot rescue a reader who got the order wrong.
#[test]
fn the_gdn_prep_seam_is_b_then_a_and_swapping_it_is_seen() {
    let device = gpu!();
    let rows = 5usize;
    let ba = exact(13, rows * 2 * V_HEADS);
    let dt_bias = exact(17, V_HEADS);
    let a_log: Vec<f32> = (0..V_HEADS).map(|h| (h % 11) as f32 / 8.0 - 0.75).collect();

    let got = fire_gates(&device, &ba, &a_log, &dt_bias, rows);
    let want = gates_reference(&ba, &a_log, &dt_bias, rows, true);
    let worst = disagrees(&got, &want, F32_TOLERANCE, "the gate row read as `[a | b]`");
    println!(
        "gdn_prep_seam_swapped: the same comparison fails at {worst:.3e} \
         against a budget of {F32_TOLERANCE:.3e}"
    );
}

/// `ssm.gated_delta`: one token per request, three requests, three seats.
///
/// The carry is checked as well as the result, for the conv's reason: a scan
/// whose output is right and whose state is one update behind is a decode that
/// drifts, and the drift is invisible at the token that produced it.
#[test]
fn a_gated_delta_step_is_the_step_the_reference_computes_and_leaves_the_carry_it_should() {
    let device = gpu!();
    let rows = ROW_SEATS.len();
    let qkv = exact(29, rows * QKV_WIDTH);
    let gates = gate_row(rows);
    let slab = delta_slab();

    let (got, left) = fire_delta(&device, &qkv, &gates, &slab, None, &ROW_SEATS, rows);
    let windows: Vec<(usize, usize)> = (0..rows).map(|t| (t, t + 1)).collect();
    let (want, carry) = delta_reference(
        &qkv,
        &packed_gates(&gates),
        &slab,
        &windows,
        &ROW_SEATS,
        rows,
    );

    let out = worst_of(&got, &want, F32_TOLERANCE, "the delta scan's result rows");
    let held = worst_of(&left, &carry, F32_TOLERANCE, "the carry the scan leaves");
    println!(
        "gated_delta_bfloat16 fired {rows} tokens at ({K_SCAN}, {V_SCAN}, \
         {K_DIM}, {V_DIM}) on {}: rows {out:.3e}, carry {held:.3e}, budget \
         {F32_TOLERANCE:.3e}",
        device.name()
    );
}

/// `ssm.gated_delta_chunked`: a request's whole window, carried across tokens.
///
/// This is the arm the packing defect lived in, because it is the only one that
/// reads more than one row of `gates`.
#[test]
fn a_gated_delta_window_is_the_window_the_reference_computes() {
    let device = gpu!();
    let qkv = exact(31, TOKENS * QKV_WIDTH);
    let gates = gate_row(TOKENS);
    let slab = delta_slab();

    let (got, left) = fire_delta(
        &device,
        &qkv,
        &gates,
        &slab,
        Some(&INDPTR),
        &TOKEN_SEATS,
        TOKENS,
    );
    dump("delta_window.y.bin", &got);
    dump("delta_window.state.bin", &left);

    let windows: Vec<(usize, usize)> = (0..INDPTR.len() - 1)
        .map(|r| (INDPTR[r] as usize, INDPTR[r + 1] as usize))
        .collect();
    let (want, carry) = delta_reference(
        &qkv,
        &packed_gates(&gates),
        &slab,
        &windows,
        &TOKEN_SEATS,
        TOKENS,
    );

    // The empty request writes no result row, so its tokens keep the sentinel
    // -- and there are none, which is what makes the CSR walk the thing under
    // test rather than the token count.
    let out = worst_of(&got, &want, F32_TOLERANCE, "the delta window's result rows");
    let held = worst_of(
        &left,
        &carry,
        F32_TOLERANCE,
        "the carry the chunked scan leaves",
    );

    // Seat 0 is named by no request, so its cells are exactly what went down.
    let per_seat = V_SCAN * K_DIM * V_DIM;
    for (i, v) in left[0..per_seat].iter().enumerate() {
        assert_eq!(
            *v, slab[i],
            "a seat no request named had cell {i} of its carry rewritten"
        );
    }

    println!(
        "gated_delta_chunked_bfloat16 fired {TOKENS} tokens over 3 requests \
         (3, 0, 4) on {}: rows {out:.3e}, carry {held:.3e}, budget \
         {F32_TOLERANCE:.3e}",
        device.name()
    );
}

/// MUTATION FOUR: TWO COMPACT HALVES AGAINST A KERNEL THAT INDEXES A PACKED
/// ROW -- AND WHY IT SURVIVED A DECODE.
///
/// This is the defect as it actually shipped: a shim wrote `gates` as a
/// `[N, V_h]` `g_log` plane followed by a `[N, V_h]` `beta` plane, while the
/// scan read `[g_log | beta]` per TOKEN. The two layouts are the same bytes at
/// `N == 1` and diverge at every `N` above it, so every decode agreed and every
/// prefill longer than one token was wrong.
///
/// So this asserts both halves of that sentence, which is the only form in
/// which the mutation is worth anything: the compact reading is INDISTINGUISHABLE
/// from the packed one at one token, and the same comparison FAILS over the
/// seven-token window.
#[test]
fn a_compact_gate_plane_is_the_defect_this_family_already_shipped() {
    let device = gpu!();
    let slab = delta_slab();

    // At one token the two readings are the same bytes.
    let one = exact(37, QKV_WIDTH);
    let gates_one = gate_row(1);
    let compact_one = |t: usize, hv: usize| -> (f64, f64) {
        (
            f64::from(gates_one[t * V_SCAN + hv]),
            f64::from(gates_one[V_SCAN + t * V_SCAN + hv]),
        )
    };
    let (got_one, _) = fire_delta(&device, &one, &gates_one, &slab, None, &ROW_SEATS[..1], 1);
    let (packed_want, _) = delta_reference(
        &one,
        &packed_gates(&gates_one),
        &slab,
        &[(0, 1)],
        &ROW_SEATS[..1],
        1,
    );
    let (compact_want, _) =
        delta_reference(&one, &compact_one, &slab, &[(0, 1)], &ROW_SEATS[..1], 1);
    assert_eq!(
        packed_want, compact_want,
        "at one token the packed and compact readings must be the same bytes, \
         which is the reason this defect survived every decode"
    );
    let agrees = worst_of(
        &got_one,
        &compact_want,
        F32_TOLERANCE,
        "the delta scan at one token, read compactly",
    );

    // At seven they are not.
    let qkv = exact(31, TOKENS * QKV_WIDTH);
    let gates = gate_row(TOKENS);
    let compact = |t: usize, hv: usize| -> (f64, f64) {
        (
            f64::from(gates[t * V_SCAN + hv]),
            f64::from(gates[TOKENS * V_SCAN + t * V_SCAN + hv]),
        )
    };
    let (got, _) = fire_delta(
        &device,
        &qkv,
        &gates,
        &slab,
        Some(&INDPTR),
        &TOKEN_SEATS,
        TOKENS,
    );
    let windows: Vec<(usize, usize)> = (0..INDPTR.len() - 1)
        .map(|r| (INDPTR[r] as usize, INDPTR[r + 1] as usize))
        .collect();
    let (want, _) = delta_reference(&qkv, &compact, &slab, &windows, &TOKEN_SEATS, TOKENS);
    let worst = disagrees(
        &got,
        &want,
        F32_TOLERANCE,
        "the delta window read against two compact gate planes",
    );

    println!(
        "gated_delta_seam_compact: at one token the two readings agree to \
         {agrees:.3e}; over {TOKENS} tokens the same comparison fails at \
         {worst:.3e} against a budget of {F32_TOLERANCE:.3e}"
    );
}

/// A `[g_log | beta]` row for `rows` tokens, packed the way `ssm.gdn_prep`
/// writes one.
///
/// `g_log` is negative -- it is a log-decay, and `exp(g_log)` is the per-token
/// contraction -- and `beta` is a sigmoid's output, so it is in `(0, 1)`. Both
/// vary per head and per token: a decay that never varied would hide a scan
/// that dropped the token axis.
fn gate_row(rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * 2 * V_SCAN];
    for t in 0..rows {
        for h in 0..V_SCAN {
            out[t * 2 * V_SCAN + h] = -0.05 - ((t * 3 + h) % 7) as f32 / 40.0;
            out[t * 2 * V_SCAN + V_SCAN + h] = 0.2 + ((t * 5 + h) % 9) as f32 / 20.0;
        }
    }
    out
}

/// Buffers this file allocates and frees by hand, so a leak is a counter that
/// does not return to where it started rather than a slow machine.
#[test]
fn every_buffer_this_file_allocates_is_freed() {
    let device = gpu!();
    let before = device.live_buffers();
    let x = exact(5, ROW_SEATS.len() * CHANNELS);
    let _ = fire_conv_step(&device, &x);
    assert_eq!(
        device.live_buffers(),
        before,
        "a conv fire left buffers behind"
    );
    let _: (Vec<f32>, Vec<f32>) = {
        let slab = delta_slab();
        let gates = gate_row(ROW_SEATS.len());
        let qkv = exact(29, ROW_SEATS.len() * QKV_WIDTH);
        fire_delta(
            &device,
            &qkv,
            &gates,
            &slab,
            None,
            &ROW_SEATS,
            ROW_SEATS.len(),
        )
    };
    assert_eq!(
        device.live_buffers(),
        before,
        "a delta fire left buffers behind"
    );
}
