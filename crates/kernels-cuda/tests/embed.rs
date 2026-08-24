//! `layout.embed`'s claimed body: the gather, and the bound it clamps to.
//!
//! THE REFERENCE IS A COPY, so the bar is BIT EQUALITY and nothing softer.
//! This point moves rows; it does not compute with them. `layout/embed.cuh`
//! says "pure copy either way", and the two arms it means — one `float4` per
//! thread when `hidden % 8 == 0` and both addresses are 16-byte aligned, one
//! element per thread otherwise — are a HOST choice, made in the claim body
//! off a test the device cannot run on itself. Both arms are fired below, at
//! shapes that pick each.
//!
//! # The clamp is the reason this point states `vocab`
//!
//! `embed` was the last row of R4's backlog and the blocker was named: the
//! kernel bounds every token id against the table's ROW count so that a
//! runaway wire payload reads row zero instead of past the largest tensor in
//! the model, and a `Const` table at the fire is an address with no
//! rectangle. The declaration states the number now. A test that only ever
//! handed in-range ids would never touch it, so two of the checks below are
//! about the bound rather than the gather:
//!
//! * an id at or above the stated `vocab` must land on ROW ZERO, exactly —
//!   which is what makes a body that dropped the clamp (or passed
//!   `i32::MAX`) fail here and only here;
//! * `vocab = 0` must REFUSE by name rather than launch a kernel whose every
//!   id is out of range.
//!
//! The third mutation is the one the stated number invites: a vocab BELOW
//! the table's real row count is a lie the plane cannot detect — no operand
//! carries the truth — and its whole visible effect is that the ids it
//! excludes come back as row zero. That is measured, not asserted away: it
//! is what "the statement is believed" costs, and the cost is bounded (a
//! wrong row, never a wrong address).
//!
//! # And the rows must agree
//!
//! `ids` and `y` both carry a row count and the walk sizes the second from
//! the fire. A disagreement means the gather was sized from something that
//! is not the token count, so the body refuses instead of gathering the
//! shorter of the two.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Layout;
use kernels::routine::{Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time. `matmul_select_bias.rs`'s lock, verbatim and for its
/// reason.
static FIRE: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── the device, or a skip ────────────────────────────────────────────────

fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

fn device_or_skip(what: &str) -> bool {
    let Some(count) = quietly(|| {
        let mut n: i32 = 0;
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        (code == rt::cudaError::cudaSuccess).then_some(n)
    }) else {
        eprintln!("skipping {what}: no CUDA runtime library on this machine");
        return false;
    };
    match count {
        Some(n) if n > 0 => {}
        _ => {
            eprintln!("skipping {what}: no CUDA device this build can drive");
            return false;
        }
    }
    assert_eq!(
        unsafe { rt::cudaSetDevice(0) },
        rt::cudaError::cudaSuccess,
        "a device is present but cudaSetDevice(0) failed"
    );
    assert_eq!(
        unsafe { rt::cudaFree(core::ptr::null_mut()) },
        rt::cudaError::cudaSuccess,
        "a device is present but the primary context would not come up"
    );
    true
}

// ── device memory, freed when the run ends ───────────────────────────────

struct Slab {
    ptr: *mut c_void,
}

impl Slab {
    fn of(bytes: &[u8]) -> Slab {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes.len().max(1)) },
            rt::cudaError::cudaSuccess,
            "cudaMalloc({})",
            bytes.len()
        );
        let slab = Slab { ptr };
        if !bytes.is_empty() {
            assert_eq!(
                unsafe {
                    rt::cudaMemcpy(
                        slab.ptr,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )
                },
                rt::cudaError::cudaSuccess,
                "host to device"
            );
        }
        slab
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        let mut bytes = vec![0u8; elems * 2];
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.ptr,
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { rt::cudaFree(self.ptr) };
        }
    }
}

// ── elements ─────────────────────────────────────────────────────────────

/// `__float2bfloat16`: round to nearest, ties to even.
fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.bits() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

// ── one case ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Case {
    what: &'static str,
    /// Whether the claim body should pick the `float4` arm. Read off the
    /// same test the body makes, so a shape that stopped picking it would
    /// be caught by the assertion rather than silently retiring an arm.
    vectorised: bool,
    rows: usize,
    hidden: usize,
}

/// A table whose every row is distinguishable from every other, so a gather
/// that landed on the wrong row is visible at the first element rather than
/// statistically. The row index is folded into the value.
fn table_of(rng: &mut Rng, vocab: usize, hidden: usize) -> Vec<u16> {
    (0..vocab * hidden)
        .map(|i| narrow((i / hidden) as f32 + rng.unit() * 0.25))
        .collect()
}

/// Fire the point once and read the result back.
fn run(ctx: &Ctx<'_>, c: Case, table: &[u16], ids: &[i32], vocab: u32) -> Vec<u16> {
    let out_n = c.rows * c.hidden;
    let d_table = Slab::of(&bytes_of_u16(table));
    let d_ids = Slab::of(&bytes_of_i32(ids));
    // A poison no legitimate row can be: every table value below is finite,
    // so a NaN survivor is a slot the gather never wrote.
    let d_out = Slab::of(&bytes_of_u16(&vec![0x7FC0u16; out_n]));

    // The body's own vectorisation test, made from outside it. Both slabs
    // come from `cudaMalloc`, which is 256-byte aligned, so the alignment
    // half is always true here and `hidden % 8` is what decides.
    assert_eq!(
        kernels_cuda::layout::vectorisable(
            c.hidden as i32,
            d_table.ptr.cast_const().cast(),
            d_out.ptr.cast_const().cast(),
        ),
        c.vectorised,
        "[{}]: the arm this case exists to fire is not the one the body picks",
        c.what
    );

    Layout::embed::<bf16>(
        ctx,
        In {
            ptr: d_ids.ptr.cast(),
            rows: c.rows as i32,
            width: 1,
        },
        Const::new(d_table.ptr.cast_const().cast()),
        vocab,
        Out {
            ptr: d_out.ptr.cast(),
            rows: c.rows as i32,
            width: c.hidden as i32,
        },
    )
    .expect("the claimed `layout.embed` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the gather did not complete"
    );
    d_out.read_u16(out_n)
}

/// `y[n] = table[ids[n] in [0, vocab) ? ids[n] : 0]`, on the host.
fn want(c: Case, table: &[u16], ids: &[i32], vocab: u32) -> Vec<u16> {
    let mut out = Vec::with_capacity(c.rows * c.hidden);
    for n in 0..c.rows {
        let id = ids[n];
        let row = if id >= 0 && (id as u32) < vocab {
            id as usize
        } else {
            0
        };
        out.extend_from_slice(&table[row * c.hidden..(row + 1) * c.hidden]);
    }
    out
}

// ── the shapes ───────────────────────────────────────────────────────────

/// Both arms, at hidden sizes shaped like real ones. qwen35-d0.8b's hidden
/// is 1024 and gemma's PLE slice is 256; the odd width is what a relay whose
/// `layers * ple_dim` does not divide by eight would hand this point.
const SHAPES: &[Case] = &[
    Case {
        what: "the float4 arm, a decode row",
        vectorised: true,
        rows: 1,
        hidden: 1024,
    },
    Case {
        what: "the float4 arm, a prefill window",
        vectorised: true,
        rows: 37,
        hidden: 256,
    },
    Case {
        what: "the scalar arm, a hidden that is not eight-wide",
        vectorised: false,
        rows: 11,
        hidden: 133,
    },
];

const VOCAB: usize = 512;

fn sample(c: Case, seed: u64) -> (Vec<u16>, Vec<i32>) {
    let mut rng = Rng(seed);
    let table = table_of(&mut rng, VOCAB, c.hidden);
    let ids: Vec<i32> = (0..c.rows)
        .map(|n| ((rng.bits() >> 32) as usize % VOCAB + n) as i32 % VOCAB as i32)
        .collect();
    (table, ids)
}

#[test]
fn the_gather_is_the_table_row() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("layout.embed") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    for (i, c) in SHAPES.iter().enumerate() {
        let (table, ids) = sample(*c, 0x9e37_79b9_7f4a_7c15 ^ (i as u64 + 1));
        let got = run(&ctx, *c, &table, &ids, VOCAB as u32);
        let expected = want(*c, &table, &ids, VOCAB as u32);
        assert!(
            got.iter()
                .all(|b| f32::from_bits(u32::from(*b) << 16).is_finite()),
            "[{}]: the gather left a slot unwritten (the NaN poison survived)",
            c.what
        );
        let differ = (0..got.len()).filter(|i| got[*i] != expected[*i]).count();
        assert_eq!(
            differ,
            0,
            "[{}]: {differ}/{} elements are not the table row the id names",
            c.what,
            got.len()
        );
        eprintln!(
            "layout.embed [{}]: {}/{} bit-identical to the gathered rows",
            c.what,
            got.len(),
            got.len()
        );
    }
}

#[test]
fn the_stated_vocab_is_the_clamp() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("layout.embed") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    let c = Case {
        what: "the clamp",
        vectorised: true,
        rows: 8,
        hidden: 128,
    };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let table = table_of(&mut rng, VOCAB, c.hidden);
    // Eight ids spread over the table, ASCENDING, so a `vocab` cut anywhere
    // splits them into a kept prefix and a clamped tail.
    let ids: Vec<i32> = (0..c.rows).map(|n| (n * 61) as i32).collect();

    // ── the whole table stated: every id is its own row ──
    let full = run(&ctx, c, &table, &ids, VOCAB as u32);
    assert_eq!(
        full,
        want(c, &table, &ids, VOCAB as u32),
        "the unclamped gather is not the table"
    );

    // ── a vocab BELOW the ids: the tail must land on row zero ──
    //
    // THE MUTATION THE STATED NUMBER INVITES. Nothing at the fire can tell
    // the plane this is a lie, and the whole visible cost of believing it is
    // that the excluded ids come back as row zero rather than their own —
    // a wrong row, never a wrong address. If the body dropped `vocab`, or
    // passed `i32::MAX` the way a delegation would have had to, this run
    // would be bit-identical to the one above.
    const CUT: u32 = 200;
    let clamped = run(&ctx, c, &table, &ids, CUT);
    let zero_row = &table[..c.hidden];
    let mut kept = 0usize;
    let mut sent_to_zero = 0usize;
    for n in 0..c.rows {
        let block = &clamped[n * c.hidden..(n + 1) * c.hidden];
        if (ids[n] as u32) < CUT {
            assert_eq!(
                block,
                &full[n * c.hidden..(n + 1) * c.hidden],
                "id {} is inside the stated vocab and moved anyway",
                ids[n]
            );
            kept += 1;
        } else {
            assert_eq!(
                block, zero_row,
                "id {} is at or past the stated vocab {CUT} and did not clamp to row zero",
                ids[n]
            );
            sent_to_zero += 1;
        }
    }
    assert!(
        kept > 0 && sent_to_zero > 0,
        "the cut measured nothing: {kept} kept, {sent_to_zero} clamped"
    );
    eprintln!(
        "layout.embed [the clamp]: vocab {CUT} of {VOCAB} keeps {kept} ids and sends \
         {sent_to_zero} to row zero"
    );

    // ── a NEGATIVE id, which the same test in the kernel catches ──
    let mut negative = ids.clone();
    negative[3] = -7;
    let refused = run(&ctx, c, &table, &negative, VOCAB as u32);
    assert_eq!(
        &refused[3 * c.hidden..4 * c.hidden],
        zero_row,
        "a negative id did not clamp to row zero"
    );
}

#[test]
fn a_vocab_of_zero_refuses_and_so_do_rows_that_disagree() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("layout.embed") {
        return;
    }
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };

    let (rows, hidden) = (4usize, 64usize);
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let table = table_of(&mut rng, VOCAB, hidden);
    let ids: Vec<i32> = (0..rows).map(|n| n as i32).collect();
    let d_table = Slab::of(&bytes_of_u16(&table));
    let d_ids = Slab::of(&bytes_of_i32(&ids));
    let d_out = Slab::of(&bytes_of_u16(&vec![0u16; rows * hidden]));

    let fire = |vocab: u32, id_rows: i32| {
        Layout::embed::<bf16>(
            &ctx,
            In {
                ptr: d_ids.ptr.cast(),
                rows: id_rows,
                width: 1,
            },
            Const::new(d_table.ptr.cast_const().cast()),
            vocab,
            Out {
                ptr: d_out.ptr.cast(),
                rows: rows as i32,
                width: hidden as i32,
            },
        )
    };

    // A table with no rows is not a table; every id would be out of range
    // and every row of the result would be a clamp onto nothing.
    let empty = fire(0, rows as i32).expect_err("`vocab = 0` must refuse");
    eprintln!("layout.embed [vocab 0]: {empty:?}");

    // The ids and the result disagree about how many tokens this fire has.
    let short =
        fire(VOCAB as u32, rows as i32 - 1).expect_err("fewer ids than gathered rows must refuse");
    eprintln!("layout.embed [rows disagree]: {short:?}");

    // And the honest shape still fires, so the two refusals above are about
    // what they name and not about the rig.
    fire(VOCAB as u32, rows as i32).expect("the honest shape");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the gather did not complete"
    );
}
