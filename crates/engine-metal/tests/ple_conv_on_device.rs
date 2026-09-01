//! qwen4's two missing doors, on a real Apple GPU: the PLE n-gram hasher and
//! the causal convolution's DILATED form.
//!
//! **WHAT THIS FILE IS FOR.** Both doors were typed refusals on this plane
//! until this lane — `attention.ple_ngram_ids{,_chunked}` by NAME, and
//! `attention.ssm_causal_conv1d{,_chunked}` by PARAMETER at `dilation: 2..`
//! — so nothing here had ever run either arithmetic, and every `qwen38-flash-*`
//! row in the catalog was a bake that could not reach its second layer.
//!
//! The two are measured differently on purpose, because they fail
//! differently:
//!
//!   * **The hasher's output is a TABLE ROW.** A hash off by one indexes a
//!     different embedding, so there is no band at all here: every assert is
//!     an equality against `kernels_metal::attn::ple::reference`, which states
//!     the same arithmetic in host Rust and is itself pinned to integers in
//!     that crate's `--lib` suite.
//!
//!   * **The convolution's output is a NUMBER**, so it is measured in bf16
//!     quanta against a host fp32 reference — and beside that, in the one way
//!     a faithful-looking port can still be wrong: a kernel that IGNORED its
//!     new `dilation` argument would match the reference at 1 and fail
//!     nothing else this file could ask it. So the dilated and undilated
//!     answers are held APART as well as each against their own reference.
//!
//! # The gates
//!
//! ```text
//! (a) ple chunked     — two requests, one crossing an eos, vs the reference:
//!                       every hashed row and the window each lane is left
//!                       holding, exactly
//! (b) ple decode      — the same sequence stepped a token at a time: the
//!                       device's decode arm against its own chunked arm
//! (c) conv dilation 1 — vs the host fp32 reference, in bf16 quanta (the
//!                       regression: this is the arm every GDN mixer fires)
//! (d) conv dilation 3 — vs the same reference, and NOT equal to (c)
//! (e) conv split      — a chunked prefill then three decode steps, against
//!                       one chunked run over the whole sequence: the state
//!                       discipline, at dilation 3
//! ```
//!
//! # Gating
//!
//! As `device_floor` and `hc_on_device`: `cfg`'d to Apple at compile time, and
//! SKIPS at run time when `device::present()` says no, saying so.
//!
//! ```text
//! cargo test -p engine-metal --test ple_conv_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::ple::reference::Hash;
use kernels_metal::attn::{ple, ssm};
use kernels_metal::{RaggedTensor, RecurrentPool, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

// ---------------------------------------------------------------------------
// The hashing: `Qwen3.8-Flash-Next`'s own constants, cut to four heads.
// `kernels_metal::attn::ple`'s `--lib` pins hold these same numbers against
// hand-computed rows; here they are what the DEVICE is asked to reproduce.
// ---------------------------------------------------------------------------

const MULTS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

const PRIMES: [u64; 4] = [20_000_003, 20_000_023, 20_000_033, 20_000_047];

const OFFSETS: [u64; 4] = [0, 20_000_003, 40_000_026, 60_000_059];

const EOS: i32 = 248_044;

const HEADS_PER_NGRAM: u32 = 2;

const HEADS: u32 = 4;

/// The window a lane keeps: `ngram − 1`.
const SPAN: u32 = 2;

fn hashing() -> Hash<'static> {
    Hash {
        eos: EOS,
        mults: &MULTS,
        primes: &PRIMES,
        offsets: &OFFSETS,
        heads_per_ngram: HEADS_PER_NGRAM as usize,
    }
}

/// The two requests this file hashes: one of five tokens crossing an eos, one
/// of three that does not. Together they are twelve columns of output and two
/// lanes of state, which is enough that a kernel addressing lane 0 by accident
/// would be caught.
const REQ_A: [i32; 5] = [11, 22, EOS, 44, 55];

const REQ_B: [i32; 3] = [7, 8, 9];

/// How many state slots the fixtures use.
const SLOTS: u32 = 3;

// ---------------------------------------------------------------------------
// Gate (a): the chunked hasher.
// ---------------------------------------------------------------------------

/// **THE PREFILL ARM, EXACTLY.** Two requests in one fire, landing into two
/// different slots, held against the host reference row for row — and the
/// windows the two lanes are left holding, which is the state discipline the
/// next fire depends on.
#[test]
fn the_chunked_hasher_lands_the_rows_the_reference_computes() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the chunked n-gram hasher") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let ids: Vec<i32> = REQ_A.iter().chain(REQ_B.iter()).copied().collect();
    let indptr: Vec<i32> = vec![0, REQ_A.len() as i32, ids.len() as i32];
    // Request A owns slot 2, request B slot 0 — deliberately not the ordinals,
    // so a kernel reading the lane number as the slot would answer wrong.
    let slot_of_row: Vec<u32> = REQ_A
        .iter()
        .map(|_| 2u32)
        .chain(REQ_B.iter().map(|_| 0u32))
        .collect();

    let (rows, state) = fire_chunked_hash(
        &device,
        &pipelines,
        &handles,
        &ids,
        &indptr,
        &slot_of_row,
        &vec![0i32; (SLOTS * SPAN) as usize],
    );

    let h = hashing();
    let mut want_a = vec![0i32; SPAN as usize];
    let want_rows_a = ple::reference::walk(&h, &REQ_A, &mut want_a);
    let mut want_b = vec![0i32; SPAN as usize];
    let want_rows_b = ple::reference::walk(&h, &REQ_B, &mut want_b);
    let want_rows: Vec<i32> = want_rows_a.iter().chain(want_rows_b.iter()).copied().collect();

    assert_eq!(
        rows, want_rows,
        "the device's hashed rows are not the reference's — a hash is exact or it is a \
         different embedding, so there is no band this can be inside"
    );
    // Slot 2 is request A's window, slot 0 is request B's, and slot 1 is
    // untouched — the one lane nothing wrote, which is what proves the slab
    // stride is the slot's and not the lane's.
    assert_eq!(&state[4..6], &want_a[..], "request A's window");
    assert_eq!(&state[0..2], &want_b[..], "request B's window");
    assert_eq!(&state[2..4], &[0, 0], "the slot no request owned");
    println!(
        "(a) ple_ngram_ids_chunked: {} rows x {HEADS} heads, exact; windows {want_a:?} / {want_b:?}",
        ids.len()
    );
}

// ---------------------------------------------------------------------------
// Gate (b): the decode hasher.
// ---------------------------------------------------------------------------

/// **THE DECODE ARM IS THE SAME WALK, ONE TOKEN AT A TIME.** Request A stepped
/// token by token off its own state must land the rows the chunked arm landed
/// for the whole request — the property that lets a fire split its lanes by
/// `qo_one` and merge the answers, measured across the device boundary.
#[test]
fn the_decode_hasher_steps_the_walk_the_chunked_arm_takes() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the decode n-gram hasher") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let mut state = vec![0i32; (SLOTS * SPAN) as usize];
    let mut stepped: Vec<i32> = Vec::new();
    for id in REQ_A {
        // One row, in slot 2 — a decode lane is one token.
        let (rows, next) = fire_decode_hash(&device, &pipelines, &handles, &[id], &[2], &state);
        stepped.extend(rows);
        state = next;
    }

    let h = hashing();
    let mut want_state = vec![0i32; SPAN as usize];
    let want = ple::reference::walk(&h, &REQ_A, &mut want_state);

    assert_eq!(stepped, want, "the stepped rows are not the walked rows");
    assert_eq!(&state[4..6], &want_state[..], "the stepped window");
    println!(
        "(b) ple_ngram_ids: {} decode steps agree with the chunked walk, exactly",
        REQ_A.len()
    );
}

// ---------------------------------------------------------------------------
// The convolution: a shape whose taps are all live and whose channels loop.
// ---------------------------------------------------------------------------

const CHANNELS: usize = 6;

const TAPS: usize = 4;

/// qwen4's PLE mixes at `ngram_size`, which is three.
const DILATION: usize = 3;

/// Gate (e)'s split: a prefill of this many tokens, then decodes.
const PREFILL: usize = 5;

const CONV_ROWS: usize = 8;

const fn hist(taps: usize, dil: usize) -> usize {
    (taps - 1) * dil + 1
}

// ---------------------------------------------------------------------------
// Gates (c) and (d): the convolution, undilated and dilated.
// ---------------------------------------------------------------------------

/// **THE UNDILATED ARM IS THE ARM EVERY GDN MIXER FIRES**, so this is the
/// regression the dilation threading had to not break: `dilation = 1` collapses
/// `(k · dil + 1)` to `(k + 1)` and `hist` to `conv_width`, which is the launch
/// this file's kernel made before it took the argument at all.
///
/// **AND THE TWO ARE HELD APART.** A kernel that read its `dilation` argument
/// and then ignored it would pass every assert above this line. The last one is
/// the only thing in this file that can see that, and it is the reason the
/// fixture's taps are all non-zero.
#[test]
fn the_convolution_serves_dilation_one_and_three_and_they_differ() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the dilated causal convolution") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let fx = ConvFixture::new(0x91e0_0001);
    let mut worst = [0.0f32; 2];
    let mut answers: Vec<Vec<f32>> = Vec::new();
    for (i, dil) in [1usize, DILATION].into_iter().enumerate() {
        let state = fx.zero_state(dil);
        let (y, out_state) = fire_conv_chunked(
            &device,
            &pipelines,
            &handles,
            &fx,
            dil,
            &state,
            &[0, CONV_ROWS as i32],
        );
        let mut want_state = state.clone();
        let want = fx.reference(&fx.x, dil, &mut want_state);

        for (k, w) in want.iter().enumerate() {
            worst[i] = worst[i].max((y[k] - w).abs() / quantum(w.abs().max(0.05)));
        }
        // The state is a COPY of the fixture's own rows through f32, so it is
        // an equality and not a band — a drifted cell there would be a kernel
        // reading the wrong row, not a rounding.
        assert_eq!(
            out_state, want_state,
            "the window a dilation-{dil} chunk leaves is not the last {} rows of its input",
            hist(TAPS, dil)
        );
        println!(
            "({}) causal_conv1d_chunked at dilation {dil}: {:.2} bf16 quanta, state exact",
            if i == 0 { 'c' } else { 'd' },
            worst[i]
        );
        answers.push(y);
    }
    assert!(worst[0] <= 1.0, "the undilated arm drifted {:.2} quanta", worst[0]);
    assert!(worst[1] <= 1.0, "the dilated arm drifted {:.2} quanta", worst[1]);

    let differing = answers[0]
        .iter()
        .zip(&answers[1])
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > answers[0].len() / 4,
        "dilation 1 and dilation {DILATION} answered the same on {} of {} cells — a kernel that \
         took the argument and did not spend it would look exactly like this",
        answers[0].len() - differing,
        answers[0].len()
    );
    println!(
        "    and the two answers differ on {differing} of {} cells",
        answers[0].len()
    );
}

// ---------------------------------------------------------------------------
// Gate (e): the split.
// ---------------------------------------------------------------------------

/// **THE STATE DISCIPLINE, AT DILATION THREE.** A prefill of five tokens
/// followed by three decode steps has to land what one chunked run over all
/// eight lands — which is the whole reason the history is `(K−1)·dil + 1` rows
/// and shifts by ONE position rather than by one tap.
#[test]
fn a_dilated_prefill_then_decodes_is_one_dilated_run() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the dilated split") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let fx = ConvFixture::new(0x91e0_0002);
    let whole = fire_conv_chunked(
        &device,
        &pipelines,
        &handles,
        &fx,
        DILATION,
        &fx.zero_state(DILATION),
        &[0, CONV_ROWS as i32],
    );

    // The head: the first `PREFILL` rows, chunked.
    let head_fx = fx.slice(0, PREFILL);
    let (mut split_y, mut state) = fire_conv_chunked(
        &device,
        &pipelines,
        &handles,
        &head_fx,
        DILATION,
        &fx.zero_state(DILATION),
        &[0, PREFILL as i32],
    );
    // The tail: one row at a time, as a decode lane arrives.
    for t in PREFILL..CONV_ROWS {
        let step_fx = fx.slice(t, t + 1);
        let (y, next) = fire_conv_decode(&device, &pipelines, &handles, &step_fx, DILATION, &state);
        split_y.extend(y);
        state = next;
    }

    let mut worst = 0.0f32;
    for (k, w) in whole.0.iter().enumerate() {
        worst = worst.max((split_y[k] - w).abs() / quantum(w.abs().max(0.05)));
    }
    println!(
        "(e) a {PREFILL}-row dilated prefill plus {} decodes vs one {CONV_ROWS}-row run: \
         {worst:.2} bf16 quanta",
        CONV_ROWS - PREFILL
    );
    assert!(worst <= 1.0, "the split drifted {worst:.2} quanta");
    assert_eq!(state, whole.1, "the two paths left different windows");
}

// ---------------------------------------------------------------------------
// The convolution fixture and its host reference.
// ---------------------------------------------------------------------------

struct ConvFixture {
    /// `[rows, CHANNELS]` bf16, already rounded.
    x: Vec<f32>,
    /// `[CHANNELS, TAPS]` bf16, already rounded.
    w: Vec<f32>,
    rows: usize,
    /// One slot per row, all the same lane — the conv's chunked arm reads the
    /// request's FIRST row's slot, its decode arm reads the row's own.
    slot_of_row: Vec<u32>,
}

impl ConvFixture {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        Self {
            x: rng.bf16_plane(CONV_ROWS * CHANNELS),
            // Deliberately spread: a tap of zero would hide a mis-strided read.
            w: (0..CHANNELS * TAPS)
                .map(|k| f32_of(bf16_bits(0.25 + 0.125 * (k % 5) as f32)))
                .collect(),
            rows: CONV_ROWS,
            // Slot 1 of three — never zero, so a kernel that lost the slot
            // multiply would read the wrong slab.
            slot_of_row: vec![1; CONV_ROWS],
        }
    }

    fn slice(&self, from: usize, to: usize) -> Self {
        Self {
            x: self.x[from * CHANNELS..to * CHANNELS].to_vec(),
            w: self.w.clone(),
            rows: to - from,
            slot_of_row: vec![1; to - from],
        }
    }

    fn zero_state(&self, dil: usize) -> Vec<f32> {
        vec![0.0; SLOTS as usize * hist(TAPS, dil) * CHANNELS]
    }

    /// `ssm_causal_conv1d.metal`'s arithmetic in host fp32: tap `j` reads
    /// `dil · (TAPS − 1 − j)` positions back, left-padded out of the lane's
    /// history, silu on the way out. Advances `state` the way the shader does.
    fn reference(&self, x: &[f32], dil: usize, state: &mut [f32]) -> Vec<f32> {
        let h = hist(TAPS, dil);
        let rows = x.len() / CHANNELS;
        // Slot 1's slab, which is the only one these fixtures touch.
        let slab = h * CHANNELS;
        let mut y = vec![0.0f32; rows * CHANNELS];
        for t in 0..rows {
            for c in 0..CHANNELS {
                let mut acc = 0.0f32;
                for k in 0..TAPS {
                    let src = t as isize - ((TAPS - 1 - k) * dil) as isize;
                    let tap = if src < 0 {
                        state[slab + (h as isize + src) as usize * CHANNELS + c]
                    } else {
                        x[src as usize * CHANNELS + c]
                    };
                    acc += tap * self.w[c * TAPS + k];
                }
                // Left in fp32: the device rounds its store to bf16
                // round-to-nearest and this host's `bf16_bits` truncates, so
                // pre-rounding here would compare two different roundings and
                // report a whole quantum of "drift" that is neither shader's.
                y[t * CHANNELS + c] = acc / (1.0 + (-acc).exp());
            }
        }
        let mut next = vec![0.0f32; h * CHANNELS];
        for s in 0..h {
            for c in 0..CHANNELS {
                let src = rows as isize - h as isize + s as isize;
                next[s * CHANNELS + c] = if src < 0 {
                    state[slab + (h as isize + src) as usize * CHANNELS + c]
                } else {
                    x[src as usize * CHANNELS + c]
                };
            }
        }
        state[slab..slab + h * CHANNELS].copy_from_slice(&next);
        y
    }
}

// ---------------------------------------------------------------------------
// The fires.
// ---------------------------------------------------------------------------

fn fire_chunked_hash(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    ids: &[i32],
    indptr: &[i32],
    slot_of_row: &[u32],
    state_in: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let ids_b = staged(device, &encode_i32(ids));
    let indptr_b = staged(device, &encode_i32(indptr));
    let slots_b = staged(device, &encode_u32(slot_of_row));
    let hash_b = staged(device, &encode_u64(&ple::hash_constants(&MULTS, &PRIMES, &OFFSETS)));
    let state_b = staged(device, &encode_i32(state_in));
    let out = Buffer::zeroed(device, (ids.len() * HEADS as usize * 4) as u64)
        .expect("the hashed rows reserve");

    let pool = RecurrentPool {
        state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
        slots: Tensor::new(bind(handles, &slots_b), 1, slot_of_row.len() as u32, Dtype::U32),
        conv_state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
        new_conv_state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
    };
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        ple::ngram_ids_chunked(
            &sink,
            RaggedTensor {
                data: Tensor::new(bind(handles, &ids_b), ids.len() as u32, 1, Dtype::I32),
                indptr: Tensor::new(bind(handles, &indptr_b), indptr.len() as u32, 1, Dtype::I32),
            },
            &pool,
            Tensor::new(
                bind(handles, &hash_b),
                1,
                (MULTS.len() + 2 * PRIMES.len()) as u32,
                Dtype::U64,
            ),
            EOS as u32,
            &MULTS,
            &PRIMES,
            &OFFSETS,
            HEADS_PER_NGRAM,
            Tensor::new(bind(handles, &out), ids.len() as u32, HEADS, Dtype::I32),
        )
        .expect("the chunked hasher encodes");
    }
    frame.commit().expect("the chunked hasher completes");
    (
        decode_i32(&read_back(&out, ids.len() * HEADS as usize * 4)),
        decode_i32(&read_back(&state_b, state_in.len() * 4)),
    )
}

fn fire_decode_hash(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    ids: &[i32],
    slot_of_row: &[u32],
    state_in: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let ids_b = staged(device, &encode_i32(ids));
    let slots_b = staged(device, &encode_u32(slot_of_row));
    let hash_b = staged(device, &encode_u64(&ple::hash_constants(&MULTS, &PRIMES, &OFFSETS)));
    let state_b = staged(device, &encode_i32(state_in));
    let out = Buffer::zeroed(device, (ids.len() * HEADS as usize * 4) as u64)
        .expect("the hashed rows reserve");

    let pool = RecurrentPool {
        state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
        slots: Tensor::new(bind(handles, &slots_b), 1, slot_of_row.len() as u32, Dtype::U32),
        conv_state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
        new_conv_state: Tensor::new(bind(handles, &state_b), SLOTS, SPAN, Dtype::I32),
    };
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        ple::ngram_ids(
            &sink,
            Tensor::new(bind(handles, &ids_b), ids.len() as u32, 1, Dtype::I32),
            &pool,
            Tensor::new(
                bind(handles, &hash_b),
                1,
                (MULTS.len() + 2 * PRIMES.len()) as u32,
                Dtype::U64,
            ),
            EOS as u32,
            &MULTS,
            &PRIMES,
            &OFFSETS,
            HEADS_PER_NGRAM,
            Tensor::new(bind(handles, &out), ids.len() as u32, HEADS, Dtype::I32),
        )
        .expect("the decode hasher encodes");
    }
    frame.commit().expect("the decode hasher completes");
    (
        decode_i32(&read_back(&out, ids.len() * HEADS as usize * 4)),
        decode_i32(&read_back(&state_b, state_in.len() * 4)),
    )
}

fn conv_pool(
    handles: &Handles,
    state: &Buffer,
    slots: &Buffer,
    fx: &ConvFixture,
    dil: usize,
) -> RecurrentPool {
    let width = (hist(TAPS, dil) * CHANNELS) as u32;
    RecurrentPool {
        state: Tensor::new(bind(handles, state), SLOTS, width, Dtype::F32),
        slots: Tensor::new(bind(handles, slots), 1, fx.rows as u32, Dtype::U32),
        conv_state: Tensor::new(bind(handles, state), SLOTS, width, Dtype::F32),
        new_conv_state: Tensor::new(bind(handles, state), SLOTS, width, Dtype::F32),
    }
}

fn fire_conv_chunked(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    fx: &ConvFixture,
    dil: usize,
    state_in: &[f32],
    indptr: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let x_b = staged(device, &encode_bf16(&fx.x));
    let w_b = staged(device, &encode_bf16(&fx.w));
    let indptr_b = staged(device, &encode_i32(indptr));
    let slots_b = staged(device, &encode_u32(&fx.slot_of_row));
    let state_b = staged(device, &encode_f32(state_in));
    let y = Buffer::zeroed(device, (fx.rows * CHANNELS * 2) as u64).expect("the output reserves");

    let pool = conv_pool(handles, &state_b, &slots_b, fx, dil);
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        ssm::causal_conv1d_chunked(
            &sink,
            RaggedTensor {
                data: Tensor::new(bind(handles, &x_b), fx.rows as u32, CHANNELS as u32, Dtype::Bf16),
                indptr: Tensor::new(bind(handles, &indptr_b), indptr.len() as u32, 1, Dtype::I32),
            },
            Tensor::new(bind(handles, &w_b), CHANNELS as u32, TAPS as u32, Dtype::Bf16),
            &pool,
            TAPS as u32,
            dil as u32,
            Tensor::new(bind(handles, &y), fx.rows as u32, CHANNELS as u32, Dtype::Bf16),
        )
        .expect("the chunked conv encodes");
    }
    frame.commit().expect("the chunked conv completes");
    (
        decode_bf16(&read_back(&y, fx.rows * CHANNELS * 2)),
        decode_f32(&read_back(&state_b, state_in.len() * 4)),
    )
}

fn fire_conv_decode(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    fx: &ConvFixture,
    dil: usize,
    state_in: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let x_b = staged(device, &encode_bf16(&fx.x));
    let w_b = staged(device, &encode_bf16(&fx.w));
    let slots_b = staged(device, &encode_u32(&fx.slot_of_row));
    let state_b = staged(device, &encode_f32(state_in));
    let y = Buffer::zeroed(device, (fx.rows * CHANNELS * 2) as u64).expect("the output reserves");

    let pool = conv_pool(handles, &state_b, &slots_b, fx, dil);
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        ssm::causal_conv1d(
            &sink,
            Tensor::new(bind(handles, &x_b), fx.rows as u32, CHANNELS as u32, Dtype::Bf16),
            Tensor::new(bind(handles, &w_b), CHANNELS as u32, TAPS as u32, Dtype::Bf16),
            &pool,
            TAPS as u32,
            dil as u32,
            Tensor::new(bind(handles, &y), fx.rows as u32, CHANNELS as u32, Dtype::Bf16),
        )
        .expect("the decode conv encodes");
    }
    frame.commit().expect("the decode conv completes");
    (
        decode_bf16(&read_back(&y, fx.rows * CHANNELS * 2)),
        decode_f32(&read_back(&state_b, state_in.len() * 4)),
    )
}

// ---------------------------------------------------------------------------
// Host staging.
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (x - 0.5) * 2.0
    }

    fn bf16_plane(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32_of(bf16_bits(self.next_f32()))).collect()
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn encode_u64(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

fn bind(handles: &Handles, buffer: &Buffer) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .expect("the handle table has a row")
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    buffer.read(0, &mut out).expect("the bytes come back");
    out
}
