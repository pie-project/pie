//! **WHAT THE GATED-DELTA PREFILL SCAN COSTS**, both ways, and whether the
//! two ways agree.
//!
//! A gated-delta stack's prefill is dominated by one serial kernel, and this
//! file is where that claim is a number rather than an intuition: it fires
//! the scan ALONE at qwen3.6-27B's shape (16 key heads, 48 value heads, 128
//! wide apiece) and multiplies by the stack's 48 gated-delta layers, so the
//! product is directly comparable to `throughput_probe`'s whole-prefill
//! milliseconds. At 512 tokens that product was 6449 ms of a 10663 ms
//! prefill — 60% of the fire — and it is what `ssm_gdn_scan.metal` was
//! written against.
//!
//! Two arms:
//!
//! 1. **The cost.** Both scans at three prompt lengths. The threadgroup scan
//!    is exactly linear in the tokens, which is what a serial walk at a fixed
//!    thread count looks like.
//! 2. **The agreement.** Both scans over the same pseudo-random operands for
//!    512 tokens, compared element for element. They are NOT bit-identical
//!    and are not meant to be: the two per-token folds sum the same `Dk`
//!    terms in different associations, and the parting compounds because the
//!    state a token leaves is the state the next one reads. What this arm
//!    asserts is only that the register scan produces finite numbers against
//!    a control that produced something; the SIZE of the parting is printed,
//!    and what the checkpoints say about it is `four_bit_first_light` and
//!    `session_c_first_light`, which pin tokens.
//!
//! The control is fired PAST `attn::ssm`'s selection rather than through a
//! moved knob, because `kernels_metal::tuning` freezes at the first
//! `current()` and a process can only answer once.
//!
//! ```text
//! cargo test -p engine-metal --release --test what_the_gated_delta_scan_costs -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::ssm;
use kernels_metal::{Arg, Ctx, Fire, Grid, RaggedTensor, RecurrentPool, Tensor};
use model_ir::Dtype;

// qwen3.6-27b's gated-delta shape (`model::qwen_3::model::d27b_dims`).
const K_HEADS: u32 = 16;
const V_HEADS: u32 = 48;
const K_DIM: u32 = 128;
const V_DIM: u32 = 128;
const LAYERS: u32 = 48;

const REPS: usize = 8;

struct Plane {
    _qkv: Buffer,
    _gates: Buffer,
    _indptr: Buffer,
    _slots: Buffer,
    _conv: Buffer,
    state: Buffer,
    y: Buffer,
    qkv: Tensor,
    indptr: Tensor,
    gates: Tensor,
    yt: Tensor,
    pool: RecurrentPool,
    t: u32,
}

fn plane(device: &Context, handles: &Handles, t: u32, filled: bool) -> Plane {
    let qkv_w = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM;
    let y_w = V_HEADS * V_DIM;
    let bf16 = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8_388_608.0 - 0.5
    };
    let hold = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(device, bytes.len() as u64).expect("reserve");
        b.write(0, bytes).expect("stage");
        b
    };
    let qkv_b = if filled {
        hold(
            &(0..t * qkv_w)
                .flat_map(|_| bf16(next()))
                .collect::<Vec<u8>>(),
        )
    } else {
        Buffer::zeroed(device, u64::from(t) * u64::from(qkv_w) * 2).expect("qkv")
    };
    // `[g_log | beta]`: g_log negative so `exp` decays, beta positive.
    let gates_b = if filled {
        let bytes: Vec<u8> = (0..t)
            .flat_map(|_| {
                let mut row = Vec::new();
                for _ in 0..V_HEADS {
                    row.extend((next().abs() * -2.0 - 0.05).to_le_bytes());
                }
                for _ in 0..V_HEADS {
                    row.extend((next().abs() + 0.1).to_le_bytes());
                }
                row
            })
            .collect();
        hold(&bytes)
    } else {
        Buffer::zeroed(device, u64::from(t) * u64::from(2 * V_HEADS) * 4).expect("gates")
    };
    let indptr_b = hold(&[0i32.to_le_bytes(), (t as i32).to_le_bytes()].concat());
    let slots_b = Buffer::zeroed(device, 4).expect("slots");
    let conv_b = Buffer::zeroed(device, 4).expect("conv");
    let state_b = Buffer::zeroed(
        device,
        u64::from(V_HEADS) * u64::from(V_DIM) * u64::from(K_DIM) * 4,
    )
    .expect("state");
    let y_b = Buffer::zeroed(device, u64::from(t) * u64::from(y_w) * 4).expect("y");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let pool = RecurrentPool {
        state: Tensor::new(bind(&state_b), 1, V_HEADS * V_DIM * K_DIM, Dtype::F32),
        slots: Tensor::new(bind(&slots_b), 1, 1, Dtype::U32),
        conv_state: Tensor::new(bind(&conv_b), 1, 1, Dtype::F32),
        new_conv_state: Tensor::new(bind(&conv_b), 1, 1, Dtype::F32),
    };
    Plane {
        qkv: Tensor::new(bind(&qkv_b), t, qkv_w, Dtype::Bf16),
        indptr: Tensor::new(bind(&indptr_b), 2, 1, Dtype::I32),
        gates: Tensor::new(bind(&gates_b), t, 2 * V_HEADS, Dtype::F32),
        yt: Tensor::new(bind(&y_b), t, y_w, Dtype::F32),
        pool,
        _qkv: qkv_b,
        _gates: gates_b,
        _indptr: indptr_b,
        _slots: slots_b,
        _conv: conv_b,
        state: state_b,
        y: y_b,
        t,
    }
}

impl Plane {
    /// The register scan, through the selection `attn::ssm` makes.
    fn scan(&self, sink: &Ctx<'_>) -> Result<(), kernels_metal::Error> {
        ssm::gated_delta_chunked(
            sink,
            RaggedTensor {
                data: self.qkv,
                indptr: self.indptr,
            },
            self.yt,
            self.gates,
            &self.pool,
            K_HEADS,
            V_HEADS,
            K_DIM,
            V_DIM,
            self.yt,
        )
    }

    /// The threadgroup scan, fired PAST the selection — the tuning table is
    /// frozen at the first `current()`, so a control that moved a knob could
    /// only move it once per process.
    fn chunked(&self, sink: &Ctx<'_>) -> Result<(), kernels_metal::Error> {
        let (kh, vh, kd, vd) = (K_HEADS as i32, V_HEADS as i32, K_DIM as i32, V_DIM as i32);
        sink.fire(
            Fire::at("attn/ssm_gated_delta.metal", "gated_delta_chunked_bfloat16")
                .apply(Grid::of([128, V_HEADS, 1], [128, 1, 1])),
            &[
                self.qkv.arg(),
                self.indptr.arg(),
                self.gates.arg(),
                self.pool.state.arg_mut(),
                self.pool.slots.arg(),
                self.yt.arg_mut(),
                kh.arg(),
                vh.arg(),
                kd.arg(),
                vd.arg(),
            ],
        )
    }

    fn wipe(&mut self) {
        for b in [&mut self.state, &mut self.y] {
            let blank = vec![0u8; b.bytes() as usize];
            b.write(0, &blank).expect("wipe");
        }
    }

    fn read(&self) -> Vec<f32> {
        let mut bytes = vec![0u8; self.y.bytes() as usize];
        self.y.read(0, &mut bytes).expect("read back");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[test]
fn gdn_scan_share_of_a_prefill() {
    if !device::present() {
        println!("SKIP: no Metal device");
        return;
    }
    let device = Context::bind().expect("device");
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    println!(
        "\n  gdn scan, qwen3.6-27b shape, {LAYERS} layers — lanes {} rows {}",
        kernels_metal::tuning::current().gdn_scan_lanes,
        kernels_metal::tuning::current().gdn_scan_rows,
    );
    for &t in &[128u32, 256, 512] {
        let p = plane(&device, &handles, t, false);
        for (label, register) in [("threadgroup", false), ("register   ", true)] {
            let burst = |n: usize| {
                let frame = device.frame().expect("frame");
                {
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for _ in 0..n {
                        if register {
                            p.scan(&sink).expect("encode");
                        } else {
                            p.chunked(&sink).expect("encode");
                        }
                    }
                }
                frame.commit().expect("commit");
            };
            burst(2);
            let start = Instant::now();
            burst(REPS);
            let ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
            println!(
                "  {label}  T={t:<4} {ms:>8.3} ms/layer   x{LAYERS} = {:>8.0} ms",
                ms * f64::from(LAYERS)
            );
        }
    }
}

#[test]
fn gdn_scan_agrees_with_the_threadgroup_scan() {
    if !device::present() {
        println!("SKIP: no Metal device");
        return;
    }
    let device = Context::bind().expect("device");
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let mut p = plane(&device, &handles, 512, true);

    let once = |p: &Plane, register: bool| {
        let frame = device.frame().expect("frame");
        {
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            if register {
                p.scan(&sink).expect("encode");
            } else {
                p.chunked(&sink).expect("encode");
            }
        }
        frame.commit().expect("commit");
    };

    p.wipe();
    once(&p, false);
    let want = p.read();
    p.wipe();
    once(&p, true);
    let got = p.read();

    let rms = |xs: &[f32]| {
        (xs.iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / xs.len() as f64)
            .sqrt()
    };
    let residual: Vec<f32> = got.iter().zip(&want).map(|(a, b)| a - b).collect();
    let worst = residual.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let parted = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the register scan wrote a NaN or an infinity"
    );
    println!(
        "\n  agreement over {} tokens: residual {:e} rms against {:e}, worst element {:e} — {:e}%",
        p.t,
        rms(&residual),
        rms(&want),
        worst,
        100.0 * rms(&residual) / rms(&want).max(f64::MIN_POSITIVE),
    );
    println!("  {parted} of {} elements differ in any bit", got.len());
    assert!(
        rms(&want) > 1e-6,
        "the control wrote nothing to compare against"
    );
}
