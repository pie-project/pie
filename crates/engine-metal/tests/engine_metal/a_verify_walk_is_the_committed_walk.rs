#![cfg(target_vendor = "apple")]
//! The verify arm of a buffered window, over Qwen3.6-35B-A3B's heads: the
//! staged scan, its gates derived as it stages them, leaves the bank bits
//! and lands the readout and gate bits that the gate prep and the committed
//! scan do, and the row-parallel conv the bits of the committed conv; each
//! lane's replayed rows folded first and its own rows landed, a lane longer
//! than one staged pass and a prompt lane's snapshots among them.
//! The microbenchmark prices both arms on one verify round's layer:
//!   cargo test --release -p engine-metal --test engine_metal \
//!     a_verify_walk -- --include-ignored --nocapture

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::attn::ssm::{self, Committed, RS_SNAPSHOTS, Snapshots, VerifyGates};
use kernels_metal::tensor::RecurrentPool;
use poem_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 32;
const DIM: u32 = 128;
const QKV_WIDTH: u32 = 2 * K_HEADS * DIM + V_HEADS * DIM;
const Y_WIDTH: u32 = V_HEADS * DIM;
const BA_WIDTH: u32 = 2 * V_HEADS;
const BANK_FLOATS: u64 = V_HEADS as u64 * DIM as u64 * DIM as u64;
const SLOTS: u32 = 32;
const TAPS: u32 = 4;
const CONV_FLOATS: u64 = TAPS as u64 * QKV_WIDTH as u64;
const LAYERS: u32 = 30;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (((bits + rounding) >> 16) as u16).to_le_bytes()
}

fn bf16s(n: u64, value: impl Fn(u64) -> f32) -> Vec<u8> {
    (0..n).flat_map(|at| bf16(value(at))).collect()
}

fn f32s(n: u64, value: impl Fn(u64) -> f32) -> Vec<u8> {
    (0..n).flat_map(|at| value(at).to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// One fire's lanes: each one's own rows, replayed rows, rows committed,
/// and the state slot the committed and the verify arm each walk.
struct Lanes {
    rows: Vec<i32>,
    replay: Vec<i32>,
    commit: Vec<i32>,
    old_slots: Vec<i32>,
    new_slots: Vec<i32>,
}

impl Lanes {
    fn indptr(&self) -> Vec<i32> {
        let mut at = vec![0];
        for rows in &self.rows {
            at.push(at.last().copied().unwrap_or(0) + rows);
        }
        at
    }

    fn own(&self) -> u32 {
        self.rows.iter().sum::<i32>() as u32
    }

    fn extended(&self) -> u32 {
        self.own() + self.replay.iter().sum::<i32>() as u32
    }

    /// Each lane's `(extended row, window row, own rows)`.
    fn spans(&self) -> Vec<(usize, usize, usize)> {
        let indptr = self.indptr();
        let mut begin = 0usize;
        let mut spans = Vec::new();
        for r in 0..self.rows.len() {
            let rep = self.replay[r] as usize;
            let rows = self.rows[r] as usize;
            spans.push((begin + rep, indptr[r] as usize, rows));
            begin += rep + rows;
        }
        spans
    }
}

struct Bench {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: std::cell::RefCell<Vec<Buffer>>,
}

impl Bench {
    fn new() -> Option<Bench> {
        let device = Context::bind().ok()?;
        eprintln!("device: {}", device.name());
        Some(Bench {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            keep: std::cell::RefCell::new(Vec::new()),
        })
    }

    fn upload(&self, bytes: &[u8], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let mut buffer =
            Buffer::zeroed(&self.device, (bytes.len() as u64).max(16)).expect("a buffer");
        buffer.write(0, bytes).expect("an upload");
        let handle = self
            .handles
            .bind(&buffer, 0, buffer.bytes())
            .expect("a handle");
        self.keep.borrow_mut().push(buffer);
        Tensor::new(handle, rows, width, dtype)
    }

    fn zeroed(&self, rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let bytes = u64::from(rows) * u64::from(width) * dtype.bits() / 8;
        self.upload(&vec![0u8; bytes as usize], rows, width, dtype)
    }

    fn i32s(&self, v: &[i32]) -> Tensor {
        self.upload(&ints(v), v.len() as u32, 1, Dtype::I32)
    }

    /// `bytes` bytes of `t` from byte `at`.
    fn read(&self, t: Tensor, at: u64, bytes: u64) -> Vec<u8> {
        let handle = self.handles.cut(t.buf, at, bytes).expect("a cut");
        self.handles.read(handle, bytes).expect("a read back")
    }

    /// Rows `from..from + n` of `t`.
    fn rows(&self, t: Tensor, from: usize, n: usize) -> Vec<u8> {
        let row = u64::from(t.width) * t.dtype.bits() / 8;
        self.read(t, from as u64 * row, n as u64 * row)
    }

    fn run(&self, fire: &dyn Fn(&Sink)) -> f64 {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit_timed().expect("the commit")
    }
}

/// The inputs of a layer's walk over `ext` extended rows.
struct Inputs {
    x: Tensor,
    weight: Tensor,
    qkv: Tensor,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: Tensor,
    conv: Tensor,
}

fn inputs(bench: &Bench, ext: u32) -> Inputs {
    let wide = |at: u64| unit(at) * 2.0f32.powi((noise(at ^ 0xabcd) % 9) as i32 - 4);
    let bank = f32s(BANK_FLOATS, |at| 0.1 * unit(at ^ 0xC3C3));
    let state: Vec<u8> = (0..SLOTS).flat_map(|_| bank.iter().copied()).collect();
    let window = f32s(CONV_FLOATS, |at| unit(at ^ 0x7777));
    let conv: Vec<u8> = (0..SLOTS).flat_map(|_| window.iter().copied()).collect();
    Inputs {
        x: bench.upload(
            &bf16s(u64::from(ext * QKV_WIDTH), |at| unit(at ^ 0x1111)),
            ext,
            QKV_WIDTH,
            Dtype::Bf16,
        ),
        weight: bench.upload(
            &bf16s(u64::from(QKV_WIDTH * TAPS), |at| 0.5 * unit(at ^ 0x2222)),
            QKV_WIDTH,
            TAPS,
            Dtype::Bf16,
        ),
        qkv: bench.upload(
            &bf16s(u64::from(ext * QKV_WIDTH), wide),
            ext,
            QKV_WIDTH,
            Dtype::Bf16,
        ),
        ba: bench.upload(
            &bf16s(u64::from(ext * BA_WIDTH), |at| 3.0 * unit(at ^ 0x3333)),
            ext,
            BA_WIDTH,
            Dtype::Bf16,
        ),
        dt_bias: bench.upload(
            &bf16s(u64::from(V_HEADS), |at| unit(at ^ 0x4444)),
            V_HEADS,
            1,
            Dtype::Bf16,
        ),
        a_log: bench.upload(
            &f32s(u64::from(V_HEADS), |at| unit(at ^ 0x5555)),
            V_HEADS,
            1,
            Dtype::F32,
        ),
        state: bench.upload(&state, SLOTS * V_HEADS * DIM, DIM, Dtype::F32),
        conv: bench.upload(&conv, SLOTS * TAPS, QKV_WIDTH, Dtype::F32),
    }
}

fn pool(x: &Inputs, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state: x.state,
        slots,
        conv_state: x.conv,
        new_conv_state: x.conv,
    }
}

fn snapshots(bench: &Bench, table: Option<&[i32]>) -> Option<Snapshots> {
    table.map(|table| Snapshots {
        table: bench.i32s(table),
        lane0: 0,
    })
}

/// The gate prep and the committed scan into the extended plane, then the
/// verify scan into the window's rows; each lane's banks, its own rows'
/// readouts and gates compared bit for bit, and each snapshot's bank too.
fn walks_agree(bench: &Bench, lanes: &Lanes, snaps: Option<(&[i32], &[i32])>) {
    let ext = lanes.extended();
    let own = lanes.own();
    let x = inputs(bench, ext);
    let indptr = bench.i32s(&lanes.indptr());
    let committed = |slots: &[i32]| Committed {
        replay: bench.i32s(&lanes.replay),
        commit: bench.i32s(&lanes.commit),
        slots: bench.i32s(slots),
        lane0: 0,
    };
    let ext_gates = bench.zeroed(ext, BA_WIDTH, Dtype::F32);
    let ext_y = bench.zeroed(ext, Y_WIDTH, Dtype::F32);
    let own_gates = bench.zeroed(own, BA_WIDTH, Dtype::F32);
    let own_y = bench.zeroed(own, Y_WIDTH, Dtype::F32);
    let old = committed(&lanes.old_slots);
    let new = committed(&lanes.new_slots);
    let tuning = engine_metal::chip::tuning("", false);
    let work = bench.zeroed(lanes.rows.len() as u32, BANK_FLOATS as u32, Dtype::F32);
    bench.run(&|sink| {
        ssm::gdn_prep(sink, x.ba, x.dt_bias, x.a_log, ext_gates).expect("the gate prep");
        ssm::gated_delta_committed_landing(
            sink,
            &tuning,
            x.qkv,
            indptr,
            &old,
            ext_gates,
            &pool(&x, old.slots),
            work,
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            ext_y,
            snapshots(bench, snaps.map(|s| s.0)),
        )
        .expect("the committed scan");
    });
    let gates = VerifyGates {
        ba: x.ba,
        dt_bias: x.dt_bias,
        a_log: x.a_log,
        gates: own_gates,
    };
    bench.run(&|sink| {
        ssm::gated_delta_verify(
            sink,
            x.qkv,
            indptr,
            &new,
            &gates,
            &pool(&x, new.slots),
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            own_y,
            snapshots(bench, snaps.map(|s| s.1)),
        )
        .expect("the verify scan");
    });
    let bank = |slot: i32| bench.read(x.state, slot as u64 * BANK_FLOATS * 4, BANK_FLOATS * 4);
    for (r, (from, at, rows)) in lanes.spans().into_iter().enumerate() {
        assert!(
            bank(lanes.old_slots[r]) == bank(lanes.new_slots[r]),
            "lane {r}: the verify scan committed other bank bits than the committed scan"
        );
        assert!(
            bench.rows(ext_y, from, rows) == bench.rows(own_y, at, rows),
            "lane {r}: the verify scan read its own rows out in other bits"
        );
        assert!(
            bench.rows(ext_gates, from, rows) == bench.rows(own_gates, at, rows),
            "lane {r}: the verify scan derived other gate bits than the gate prep"
        );
        assert!(
            bank(lanes.new_slots[r]) != bank(0),
            "lane {r}: its slot was never committed into"
        );
    }
    if let Some((old, new)) = snaps {
        let s = RS_SNAPSHOTS as usize;
        for r in 0..lanes.rows.len() {
            for i in (0..s).take_while(|&i| old[2 * s * r + i] > 0) {
                let at = 2 * s * r + s + i;
                assert!(
                    bank(old[at]) == bank(new[at]),
                    "lane {r}: snapshot {i} landed other bank bits"
                );
            }
        }
    }
}

/// The committed conv into the extended plane, then the verify conv into it
/// and the window's rows; every extended row, each lane's own rows, its
/// committed window and each snapshot's compared bit for bit.
fn convs_agree(bench: &Bench, lanes: &Lanes, snaps: Option<(&[i32], &[i32])>) {
    let ext = lanes.extended();
    let own = lanes.own();
    let x = inputs(bench, ext);
    let indptr = bench.i32s(&lanes.indptr());
    let committed = |slots: &[i32]| Committed {
        replay: bench.i32s(&lanes.replay),
        commit: bench.i32s(&lanes.commit),
        slots: bench.i32s(slots),
        lane0: 0,
    };
    let old_y = bench.zeroed(ext, QKV_WIDTH, Dtype::Bf16);
    let new_y = bench.zeroed(ext, QKV_WIDTH, Dtype::Bf16);
    let land = bench.zeroed(own, QKV_WIDTH, Dtype::Bf16);
    let old = committed(&lanes.old_slots);
    let new = committed(&lanes.new_slots);
    bench.run(&|sink| {
        ssm::causal_conv1d_committed_landing(
            sink,
            x.x,
            indptr,
            &old,
            x.weight,
            &pool(&x, old.slots),
            TAPS,
            1,
            old_y,
            snapshots(bench, snaps.map(|s| s.0)),
        )
        .expect("the committed conv");
    });
    bench.run(&|sink| {
        ssm::causal_conv1d_verify(
            sink,
            x.x,
            indptr,
            &new,
            x.weight,
            &pool(&x, new.slots),
            TAPS,
            1,
            new_y,
            land,
            snapshots(bench, snaps.map(|s| s.1)),
        )
        .expect("the verify conv");
    });
    let window = |slot: i32| bench.read(x.conv, slot as u64 * CONV_FLOATS * 4, CONV_FLOATS * 4);
    assert!(
        bench.rows(old_y, 0, ext as usize) == bench.rows(new_y, 0, ext as usize),
        "the verify conv convolved the extended rows into other bits"
    );
    for (r, (from, at, rows)) in lanes.spans().into_iter().enumerate() {
        assert!(
            bench.rows(old_y, from, rows) == bench.rows(land, at, rows),
            "lane {r}: the verify conv landed its own rows in other bits"
        );
        assert!(
            window(lanes.old_slots[r]) == window(lanes.new_slots[r]),
            "lane {r}: the verify conv committed another window"
        );
    }
    if let Some((old, new)) = snaps {
        let s = RS_SNAPSHOTS as usize;
        for r in 0..lanes.rows.len() {
            for i in (0..s).take_while(|&i| old[2 * s * r + i] > 0) {
                let at = 2 * s * r + s + i;
                assert!(
                    window(old[at]) == window(new[at]),
                    "lane {r}: conv snapshot {i} landed another window"
                );
            }
        }
    }
}

/// Lanes whose old and new walks start from the same slot contents: every
/// slot of the inputs holds the same bank.
fn cases() -> Vec<(Lanes, Option<(Vec<i32>, Vec<i32>)>)> {
    let s = RS_SNAPSHOTS as usize;
    let mut old = vec![0i32; 2 * s];
    let mut new = vec![0i32; 2 * s];
    old[..2].copy_from_slice(&[3, 19]);
    new[..2].copy_from_slice(&[3, 19]);
    old[s..s + 2].copy_from_slice(&[9, 10]);
    new[s..s + 2].copy_from_slice(&[11, 12]);
    vec![
        // A verify round's lane, a shorter replay, and a walk of 22 rows,
        // past one staged pass.
        (
            Lanes {
                rows: vec![8, 8, 14],
                replay: vec![5, 3, 8],
                commit: vec![4, 9, 20],
                old_slots: vec![1, 2, 3],
                new_slots: vec![4, 5, 6],
            },
            None,
        ),
        // One lane, nothing replayed, every row committed.
        (
            Lanes {
                rows: vec![8],
                replay: vec![0],
                commit: vec![8],
                old_slots: vec![7],
                new_slots: vec![8],
            },
            None,
        ),
        // A lane whose fold leaves rows of its old window in the new one.
        (
            Lanes {
                rows: vec![8],
                replay: vec![1],
                commit: vec![2],
                old_slots: vec![15],
                new_slots: vec![16],
            },
            None,
        ),
        // A prompt lane landing snapshots as it passes rows 3 and 19.
        (
            Lanes {
                rows: vec![24],
                replay: vec![0],
                commit: vec![24],
                old_slots: vec![13],
                new_slots: vec![14],
            },
            Some((old, new)),
        ),
    ]
}

#[test]
fn a_verify_walk_is_the_committed_walk() {
    let Some(bench) = Bench::new() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    for (lanes, snaps) in cases() {
        let snaps = snaps.as_ref().map(|(o, n)| (o.as_slice(), n.as_slice()));
        walks_agree(&bench, &lanes, snaps);
        convs_agree(&bench, &lanes, snaps);
    }
    eprintln!(
        "the verify arm's banks, windows, snapshots, readouts and gates are the committed arm's, bit for bit"
    );
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a microbenchmark"]
fn a_verify_walk_prices_against_the_committed_walk() {
    let Some(bench) = Bench::new() else {
        return;
    };
    // A verify round of the 35B: eight rows behind five replayed, each of
    // the 30 layers its own state, as cold as a round finds it.
    let lanes = Lanes {
        rows: vec![8],
        replay: vec![5],
        commit: vec![5],
        old_slots: vec![0],
        new_slots: vec![0],
    };
    let ext = lanes.extended();
    let x = inputs(&bench, ext);
    let indptr = bench.i32s(&lanes.indptr());
    let layers: Vec<Committed> = (0..LAYERS)
        .map(|l| Committed {
            replay: bench.i32s(&lanes.replay),
            commit: bench.i32s(&lanes.commit),
            slots: bench.i32s(&[(l % SLOTS) as i32]),
            lane0: 0,
        })
        .collect();
    let ext_gates = bench.zeroed(ext, BA_WIDTH, Dtype::F32);
    let ext_y = bench.zeroed(ext, Y_WIDTH, Dtype::F32);
    let ext_conv = bench.zeroed(ext, QKV_WIDTH, Dtype::Bf16);
    let land = bench.zeroed(8, QKV_WIDTH, Dtype::Bf16);
    let gates = VerifyGates {
        ba: x.ba,
        dt_bias: x.dt_bias,
        a_log: x.a_log,
        gates: bench.zeroed(8, BA_WIDTH, Dtype::F32),
    };
    let own_y = bench.zeroed(8, Y_WIDTH, Dtype::F32);
    let work = bench.zeroed(1, BANK_FLOATS as u32, Dtype::F32);
    let tuning = engine_metal::chip::tuning("", false);
    // A frame of the 30 layers, so the device runs at the clock a round
    // holds it at.
    let price = |fire: &dyn Fn(&Sink, &Committed)| {
        let mut v = Vec::new();
        for i in 0..30 {
            let s = bench.run(&|sink| {
                for c in &layers {
                    fire(sink, c);
                }
            });
            if i >= 10 {
                v.push(s * 1e6 / f64::from(LAYERS));
            }
        }
        median(v)
    };
    let old_conv = |sink: &Sink, c: &Committed| {
        ssm::causal_conv1d_committed(
            sink,
            x.x,
            indptr,
            c,
            x.weight,
            &pool(&x, c.slots),
            TAPS,
            1,
            ext_conv,
        )
        .expect("the committed conv");
    };
    let new_conv = |sink: &Sink, c: &Committed| {
        ssm::causal_conv1d_verify(
            sink,
            x.x,
            indptr,
            c,
            x.weight,
            &pool(&x, c.slots),
            TAPS,
            1,
            ext_conv,
            land,
            None,
        )
        .expect("the verify conv");
    };
    let old_walk = |sink: &Sink, c: &Committed| {
        ssm::gdn_prep(sink, x.ba, x.dt_bias, x.a_log, ext_gates).expect("the gate prep");
        ssm::gated_delta_committed(
            sink,
            &tuning,
            x.qkv,
            indptr,
            c,
            ext_gates,
            &pool(&x, c.slots),
            work,
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            ext_y,
        )
        .expect("the committed scan");
    };
    let new_walk = |geometry: (u32, u32, u32)| {
        let gates = &gates;
        let x = &x;
        move |sink: &Sink, c: &Committed| {
            ssm::gated_delta_verify_at(
                sink,
                geometry,
                x.qkv,
                indptr,
                c,
                gates,
                &pool(x, c.slots),
                K_HEADS,
                V_HEADS,
                DIM,
                DIM,
                own_y,
                None,
            )
            .expect("the verify scan");
        }
    };
    let geometries = [
        (32, 4, 4),
        (32, 8, 4),
        (32, 8, 2),
        (32, 4, 8),
        (16, 4, 1),
        (16, 4, 2),
        (8, 4, 1),
        (8, 4, 2),
        (8, 8, 1),
    ];
    let mut best = vec![f64::MAX; geometries.len() + 3];
    for _ in 0..3 {
        best[0] = best[0].min(price(&old_conv));
        best[1] = best[1].min(price(&new_conv));
        best[2] = best[2].min(price(&old_walk));
        for (at, geometry) in geometries.into_iter().enumerate() {
            best[at + 3] = best[at + 3].min(price(&new_walk(geometry)));
        }
    }
    eprintln!(
        "one layer of 5 + 8 rows: conv committed {:.1} us, verify {:.1} us",
        best[0], best[1]
    );
    let mut line = format!(
        "one layer of 5 + 8 rows: prep and committed scan {:.1} us; verify",
        best[2]
    );
    for ((l, g, v), us) in geometries.iter().zip(&best[3..]) {
        line += &format!(" l{l}s{g}v{v} {us:.1}");
    }
    eprintln!("{line}");
}
