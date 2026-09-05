//! **THE THREE REGISTER SCANS LAND ONE SET OF BITS, AND THEY ARE THE OLD
//! KERNELS' NUMBERS** — the gate for `ssm_gdn_scan.metal`'s one-token step
//! and committed twin.
//!
//! The serving engine runs a plain decode through the step, a prefill through
//! the scan and a speculative window through the committed scan. A guest that
//! verifies a window against a plain decode compares their tokens, so what
//! this pins first is that the three kernels are ONE arithmetic: over the same
//! `T` tokens, `T` one-row steps, one committed run with `commit = T` and one
//! plain scan must leave the same bank and the same outputs **byte for byte**,
//! and a committed run with `commit = j < T` must leave the bank `j` steps
//! leave. Then, against the threadgroup kernels they replace
//! (`ssm_gated_delta.metal`'s `gated_delta` and `gated_delta_committed`), the
//! new kernels must agree to the reassociation floor — the shuffle tree sums
//! the same terms in another order, so the answers part in the last bits and
//! nowhere else.
//!
//! qwen3.6-27B's shape (16 key heads, 48 value heads, 128 wide), pseudo-random
//! operands, a dense pseudo-random starting bank so the recurrence has memory
//! to carry.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_gated_delta_scans_agree -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::ssm::{self, Committed};
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::Tensor;
use model_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 48;
const K_DIM: u32 = 128;
const V_DIM: u32 = 128;
/// The fused row: `[q | k | v]`.
const QKV_WIDTH: u32 = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM;
const Y_WIDTH: u32 = V_HEADS * V_DIM;
/// One bank: `[v_heads][v_dim][k_dim]` f32.
const BANK_FLOATS: u64 = V_HEADS as u64 * V_DIM as u64 * K_DIM as u64;
/// Tokens in the run — a full verify window.
const T: u32 = 16;
/// The truncated commit.
const J: u32 = 5;

const OLD_FILE: &str = "attn/ssm_gated_delta.metal";

/// Banks, one per arm.
const SLOTS: u32 = 8;
const SLOT_OLD_COMMITTED: u32 = 0;
const SLOT_NEW_COMMITTED: u32 = 1;
const SLOT_NEW_STEPS: u32 = 2;
const SLOT_NEW_SCAN: u32 = 3;
const SLOT_OLD_STEPS: u32 = 4;
const SLOT_NEW_COMMITTED_J: u32 = 5;
const SLOT_NEW_STEPS_J: u32 = 6;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

/// A pseudo-random float in `[-1, 1)`.
fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (((bits + rounding) >> 16) as u16).to_le_bytes()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// Relative rms of the difference, and the worst absolute difference.
fn compare(want: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(want.len(), got.len());
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    let mut worst = 0.0f64;
    for (a, b) in want.iter().zip(got) {
        let d = f64::from(a - b);
        diff += d * d;
        norm += f64::from(*a) * f64::from(*a);
        worst = worst.max(d.abs());
    }
    ((diff / norm.max(1e-30)).sqrt(), worst)
}

#[test]
fn the_scans_agree() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    // ── operands ─────────────────────────────────────────────────────────
    let mut qkv_b = Buffer::zeroed(&device, u64::from(T) * u64::from(QKV_WIDTH) * 2).expect("qkv");
    {
        let mut bytes = Vec::with_capacity((T * QKV_WIDTH * 2) as usize);
        for at in 0..u64::from(T) * u64::from(QKV_WIDTH) {
            bytes.extend_from_slice(&bf16(unit(at)));
        }
        qkv_b.write(0, &bytes).expect("write qkv");
    }
    // `[g_log | beta]` a row: a decay in (e^-0.4, e^-0.02), a beta in (0.3, 0.9).
    let mut gates_b = Buffer::zeroed(&device, u64::from(T) * u64::from(2 * V_HEADS) * 4).expect("gates");
    {
        let mut g = Vec::with_capacity((T * 2 * V_HEADS) as usize);
        for t in 0..T {
            for h in 0..V_HEADS {
                g.push(-0.02 - 0.38 * (unit(u64::from(t * V_HEADS + h) ^ 0xA5A5) * 0.5 + 0.5));
            }
            for h in 0..V_HEADS {
                g.push(0.3 + 0.6 * (unit(u64::from(t * V_HEADS + h) ^ 0x5A5A) * 0.5 + 0.5));
            }
        }
        gates_b.write(0, &as_bytes(&g)).expect("write gates");
    }
    // Every slot starts from the same dense bank.
    let mut state_b = Buffer::zeroed(&device, u64::from(SLOTS) * BANK_FLOATS * 4).expect("state");
    {
        let bank: Vec<f32> = (0..BANK_FLOATS).map(|at| 0.1 * unit(at ^ 0xC3C3)).collect();
        let bytes = as_bytes(&bank);
        for s in 0..u64::from(SLOTS) {
            state_b.write(s * BANK_FLOATS * 4, &bytes).expect("write bank");
        }
    }
    let work_b = Buffer::zeroed(&device, BANK_FLOATS * 4).expect("work");
    let y_b: Vec<Buffer> = (0..SLOTS)
        .map(|_| Buffer::zeroed(&device, u64::from(T) * u64::from(Y_WIDTH) * 4).expect("y"))
        .collect();
    // Tables: one lane, the run is the whole CSR, no replay.
    let mut indptr_b = Buffer::zeroed(&device, 8).expect("indptr");
    indptr_b.write(0, &ints(&[0, T as i32])).expect("write indptr");
    let mut replay_b = Buffer::zeroed(&device, 4).expect("replay");
    replay_b.write(0, &ints(&[0])).expect("write replay");
    let mut commit_t_b = Buffer::zeroed(&device, 4).expect("commit");
    commit_t_b.write(0, &ints(&[T as i32])).expect("write commit");
    let mut commit_j_b = Buffer::zeroed(&device, 4).expect("commit j");
    commit_j_b.write(0, &ints(&[J as i32])).expect("write commit j");
    // One slot table per arm, i32 for the committed tables and u32 for the pools.
    let slot_tables: Vec<Buffer> = (0..SLOTS)
        .map(|s| {
            let mut b = Buffer::zeroed(&device, 4).expect("slot");
            b.write(0, &ints(&[s as i32])).expect("write slot");
            b
        })
        .collect();

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let h_qkv = bind(&qkv_b);
    let h_gates = bind(&gates_b);
    let h_state = bind(&state_b);
    let h_work = bind(&work_b);
    let h_indptr = bind(&indptr_b);
    let h_replay = bind(&replay_b);
    let h_commit_t = bind(&commit_t_b);
    let h_commit_j = bind(&commit_j_b);
    let h_slot: Vec<u32> = slot_tables.iter().map(bind).collect();
    let h_y: Vec<u32> = y_b.iter().map(bind).collect();
    // Row `t` of qkv, gates and y as its own one-row tensor.
    let row = |b: &Buffer, t: u32, width: u32, elem: u64| {
        let stride = u64::from(width) * elem;
        handles.bind(b, u64::from(t) * stride, stride).expect("a row handle")
    };

    let qkv = Tensor::new(h_qkv, T, QKV_WIDTH, Dtype::Bf16);
    let gates = Tensor::new(h_gates, T, 2 * V_HEADS, Dtype::F32);
    let state = Tensor::new(h_state, SLOTS * V_HEADS * V_DIM, K_DIM, Dtype::F32);
    let work = Tensor::new(h_work, V_HEADS * V_DIM, K_DIM, Dtype::F32);
    let indptr = Tensor::new(h_indptr, 2, 1, Dtype::I32);
    let table = |h: u32| Tensor::new(h, 1, 1, Dtype::I32);
    let utable = |h: u32| Tensor::new(h, 1, 1, Dtype::U32);
    let y_of = |s: u32| Tensor::new(h_y[s as usize], T, Y_WIDTH, Dtype::F32);
    let pool = |s: u32| RecurrentPool {
        state,
        slots: utable(h_slot[s as usize]),
        conv_state: state,
        new_conv_state: state,
    };
    let committed = |s: u32, commit: u32| Committed {
        replay: table(h_replay),
        commit: table(commit),
        slots: table(h_slot[s as usize]),
        lane0: 0,
    };
    let splits = (V_DIM / 32) as i32;

    let run = |fire: &dyn Fn(&Sink)| {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        fire(&sink);
        frame.commit().expect("the commit");
    };

    // ── A: the old committed kernel, the whole run, commit = T ───────────
    run(&|sink| {
        sink.fire(
            Fire::at(OLD_FILE, "gated_delta_committed_bfloat16")
                .apply(Grid::of([128, V_HEADS, splits as u32], [128, 1, 1])),
            &[
                qkv.arg(),
                indptr.arg(),
                table(h_replay).arg(),
                table(h_commit_t).arg(),
                table(h_slot[SLOT_OLD_COMMITTED as usize]).arg(),
                0i32.arg(),
                gates.arg(),
                state.arg_mut(),
                work.arg_mut(),
                y_of(SLOT_OLD_COMMITTED).arg_mut(),
                (K_HEADS as i32).arg(),
                (V_HEADS as i32).arg(),
                (K_DIM as i32).arg(),
                (V_DIM as i32).arg(),
                splits.arg(),
            ],
        )
        .expect("the old committed kernel");
    });

    // ── B: the new committed scan, commit = T; B': commit = J ────────────
    for (slot, commit) in [(SLOT_NEW_COMMITTED, h_commit_t), (SLOT_NEW_COMMITTED_J, h_commit_j)] {
        run(&|sink| {
            ssm::gated_delta_committed(
                sink,
                qkv,
                indptr,
                &committed(slot, commit),
                gates,
                &pool(slot),
                work,
                K_HEADS,
                V_HEADS,
                K_DIM,
                V_DIM,
                y_of(slot),
            )
            .expect("the committed scan");
        });
    }

    // ── C: the new step, one token a fire; C': the first J only ──────────
    for (slot, steps) in [(SLOT_NEW_STEPS, T), (SLOT_NEW_STEPS_J, J)] {
        for t in 0..steps {
            let q = Tensor::new(row(&qkv_b, t, QKV_WIDTH, 2), 1, QKV_WIDTH, Dtype::Bf16);
            let g = Tensor::new(row(&gates_b, t, 2 * V_HEADS, 4), 1, 2 * V_HEADS, Dtype::F32);
            let y = Tensor::new(row(&y_b[slot as usize], t, Y_WIDTH, 4), 1, Y_WIDTH, Dtype::F32);
            run(&|sink| {
                ssm::gated_delta(sink, q, q, g, &pool(slot), K_HEADS, V_HEADS, K_DIM, V_DIM, y)
                    .expect("the step");
            });
        }
    }

    // ── D: the plain scan over the CSR ───────────────────────────────────
    run(&|sink| {
        ssm::gated_delta_chunked(
            sink,
            RaggedTensor { data: qkv, indptr },
            qkv,
            gates,
            &pool(SLOT_NEW_SCAN),
            K_HEADS,
            V_HEADS,
            K_DIM,
            V_DIM,
            y_of(SLOT_NEW_SCAN),
        )
        .expect("the scan");
    });

    // ── E: the old step kernel, one token a fire ─────────────────────────
    for t in 0..T {
        let q = Tensor::new(row(&qkv_b, t, QKV_WIDTH, 2), 1, QKV_WIDTH, Dtype::Bf16);
        let g = Tensor::new(row(&gates_b, t, 2 * V_HEADS, 4), 1, 2 * V_HEADS, Dtype::F32);
        let y = Tensor::new(row(&y_b[SLOT_OLD_STEPS as usize], t, Y_WIDTH, 4), 1, Y_WIDTH, Dtype::F32);
        run(&|sink| {
            sink.fire(
                Fire::at(OLD_FILE, "gated_delta_bfloat16")
                    .apply(Grid::of([128, V_HEADS, splits as u32], [128, 1, 1])),
                &[
                    q.arg(),
                    g.arg(),
                    state.arg_mut(),
                    utable(h_slot[SLOT_OLD_STEPS as usize]).arg(),
                    y.arg_mut(),
                    (K_HEADS as i32).arg(),
                    (V_HEADS as i32).arg(),
                    (K_DIM as i32).arg(),
                    (V_DIM as i32).arg(),
                    splits.arg(),
                ],
            )
            .expect("the old step kernel");
        });
    }

    // ── read back ────────────────────────────────────────────────────────
    let bank = |s: u32| {
        let raw = handles
            .read(handles.bind(&state_b, u64::from(s) * BANK_FLOATS * 4, BANK_FLOATS * 4).expect("a bank handle"), BANK_FLOATS * 4)
            .expect("read bank");
        floats(&raw)
    };
    let out = |s: u32| floats(&handles.read(h_y[s as usize], u64::from(T) * u64::from(Y_WIDTH) * 4).expect("read y"));
    let rows = |v: &[f32], n: u32| v[..(n * Y_WIDTH) as usize].to_vec();

    // 1. One arithmetic: steps, committed and scan land the same bytes.
    let steps_bank = bank(SLOT_NEW_STEPS);
    let steps_y = out(SLOT_NEW_STEPS);
    assert_eq!(bank(SLOT_NEW_COMMITTED), steps_bank, "committed(T) left other bank bits than T steps");
    assert_eq!(out(SLOT_NEW_COMMITTED), steps_y, "committed(T) answered other bits than T steps");
    assert_eq!(bank(SLOT_NEW_SCAN), steps_bank, "the scan left other bank bits than T steps");
    assert_eq!(out(SLOT_NEW_SCAN), steps_y, "the scan answered other bits than T steps");
    // 2. The commit is live: `commit = J` leaves the bank J steps leave, and
    //    still answers every row of the run.
    assert_eq!(bank(SLOT_NEW_COMMITTED_J), bank(SLOT_NEW_STEPS_J), "committed(J) left other bank bits than J steps");
    assert_ne!(bank(SLOT_NEW_COMMITTED_J), steps_bank, "J and T steps leave the same bank, so the commit claim can't tell them apart");
    assert_eq!(out(SLOT_NEW_COMMITTED_J), steps_y, "committed(J) answered the rows past J other bits than the steps");
    assert_eq!(rows(&out(SLOT_NEW_STEPS_J), J), rows(&steps_y, J));
    eprintln!("one arithmetic: {T} steps == committed({T}) == scan; committed({J}) bank == {J} steps  (byte for byte)");

    // 3. Against the threadgroup kernels: the reassociation floor.
    //    5.2e-9 rms over 512 tokens is what the scan header measured against
    //    the chunked kernel; 1e-5 is three orders above that and four below
    //    a wrong answer.
    let floor = 1e-5f64;
    for (name, old_slot, new_slot) in [
        ("committed", SLOT_OLD_COMMITTED, SLOT_NEW_COMMITTED),
        ("step", SLOT_OLD_STEPS, SLOT_NEW_STEPS),
    ] {
        let (bank_rms, bank_worst) = compare(&bank(old_slot), &bank(new_slot));
        let (y_rms, y_worst) = compare(&out(old_slot), &out(new_slot));
        eprintln!(
            "{name:9} old vs new: bank rel rms {bank_rms:.2e} (worst {bank_worst:.2e}), y rel rms {y_rms:.2e} (worst {y_worst:.2e})"
        );
        assert!(bank_rms <= floor, "{name}: the new kernel's bank parts from the old by {bank_rms:.2e} rms");
        assert!(y_rms <= floor, "{name}: the new kernel's output parts from the old by {y_rms:.2e} rms");
    }
    // The old two agree with each other the same way, which says the floor is
    // the reassociation's and not a bug shared by the two new kernels.
    let (rms, _) = compare(&bank(SLOT_OLD_COMMITTED), &bank(SLOT_OLD_STEPS));
    eprintln!("old committed vs old steps: bank rel rms {rms:.2e}");
}
