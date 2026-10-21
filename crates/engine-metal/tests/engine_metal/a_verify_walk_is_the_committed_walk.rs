#![cfg(target_vendor = "apple")]
//! The verify arm of a buffered window, over Qwen3.6-35B-A3B's heads: the
//! staged scan leaves the bank bits and lands the readout bits the committed
//! scan does, each lane's replayed rows folded first and its own rows read
//! out, a lane's walk longer than one staged pass among them.
//! The microbenchmark prices both on one verify round's layer:
//!   cargo test --release -p engine-metal --test engine_metal \
//!     a_verify_walk_is_the_committed_walk -- --include-ignored --nocapture

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::attn::ssm::{self, Committed, Snapshots};
use kernels_metal::tensor::RecurrentPool;
use poem_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 32;
const DIM: u32 = 128;
const QKV_WIDTH: u32 = 2 * K_HEADS * DIM + V_HEADS * DIM;
const Y_WIDTH: u32 = V_HEADS * DIM;
const BANK_FLOATS: u64 = V_HEADS as u64 * DIM as u64 * DIM as u64;
const SLOTS: u32 = 32;

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

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// One fire's lanes: each one's own rows, replayed rows, rows committed,
/// and the state slot the committed and the verify scan each walk.
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

    fn zeroed(&self, bytes: u64, rows: u32, width: u32, dtype: Dtype) -> Tensor {
        self.upload(&vec![0u8; bytes as usize], rows, width, dtype)
    }

    fn read(&self, t: Tensor, bytes: u64) -> Vec<f32> {
        floats(&self.handles.read(t.buf, bytes).expect("a read back"))
    }

    fn run(&self, fire: &dyn Fn(&Sink)) -> f64 {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit_timed().expect("the commit")
    }
}

/// The inputs of a walk over `ext` extended rows.
struct Inputs {
    qkv: Tensor,
    gates: Tensor,
    state: Tensor,
}

fn inputs(bench: &Bench, ext: u32) -> Inputs {
    let mut qkv = Vec::with_capacity((ext * QKV_WIDTH * 2) as usize);
    for at in 0..u64::from(ext) * u64::from(QKV_WIDTH) {
        qkv.extend_from_slice(&bf16(
            unit(at) * 2.0f32.powi((noise(at ^ 0xabcd) % 9) as i32 - 4),
        ));
    }
    let mut gates = Vec::with_capacity((ext * 2 * V_HEADS * 4) as usize);
    for t in 0..ext {
        for h in 0..V_HEADS {
            let g = -0.02 - 0.38 * (unit(u64::from(t * V_HEADS + h) ^ 0xA5A5) * 0.5 + 0.5);
            gates.extend_from_slice(&g.to_le_bytes());
        }
        for h in 0..V_HEADS {
            let b = 0.3 + 0.6 * (unit(u64::from(t * V_HEADS + h) ^ 0x5A5A) * 0.5 + 0.5);
            gates.extend_from_slice(&b.to_le_bytes());
        }
    }
    let bank: Vec<u8> = (0..BANK_FLOATS)
        .flat_map(|at| (0.1 * unit(at ^ 0xC3C3)).to_le_bytes())
        .collect();
    let state: Vec<u8> = (0..SLOTS).flat_map(|_| bank.iter().copied()).collect();
    Inputs {
        qkv: bench.upload(&qkv, ext, QKV_WIDTH, Dtype::Bf16),
        gates: bench.upload(&gates, ext, 2 * V_HEADS, Dtype::F32),
        state: bench.upload(&state, SLOTS * V_HEADS * DIM, DIM, Dtype::F32),
    }
}

fn pool(state: Tensor, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    }
}

/// The committed scan into the extended plane, then the verify scan into the
/// window's rows; each lane's banks and its own rows' readouts compared bit
/// for bit, and each snapshot's bank too.
fn agree(bench: &Bench, lanes: &Lanes, snaps: Option<(&[i32], &[i32])>) {
    let ext = lanes.extended();
    let own = lanes.own();
    let n = lanes.rows.len() as u32;
    let x = inputs(bench, ext);
    let i32s = |v: &[i32]| bench.upload(&ints(v), v.len() as u32, 1, Dtype::I32);
    let indptr = i32s(&lanes.indptr());
    let replay = i32s(&lanes.replay);
    let commit = i32s(&lanes.commit);
    let committed = |slots: &[i32]| Committed {
        replay,
        commit,
        slots: i32s(slots),
        lane0: 0,
    };
    let snapshots = |table: Option<&[i32]>| {
        table.map(|table| Snapshots {
            table: i32s(table),
            lane0: 0,
        })
    };
    let ext_y = bench.zeroed(u64::from(ext * Y_WIDTH) * 4, ext, Y_WIDTH, Dtype::F32);
    let own_y = bench.zeroed(u64::from(own * Y_WIDTH) * 4, own, Y_WIDTH, Dtype::F32);
    let old = committed(&lanes.old_slots);
    let new = committed(&lanes.new_slots);
    let tuning = engine_metal::chip::tuning("", false);
    let work = bench.zeroed(BANK_FLOATS * 4 * u64::from(n), n, 1, Dtype::F32);
    bench.run(&|sink| {
        ssm::gated_delta_committed_landing(
            sink,
            &tuning,
            x.qkv,
            indptr,
            &old,
            x.gates,
            &pool(x.state, old.slots),
            work,
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            ext_y,
            snapshots(snaps.map(|s| s.0)),
        )
        .expect("the committed scan");
    });
    bench.run(&|sink| {
        ssm::gated_delta_verify(
            sink,
            x.qkv,
            indptr,
            &new,
            x.gates,
            &pool(x.state, new.slots),
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            own_y,
            snapshots(snaps.map(|s| s.1)),
        )
        .expect("the verify scan");
    });
    let bank = |slot: i32| {
        let handle = bench
            .handles
            .cut(x.state.buf, slot as u64 * BANK_FLOATS * 4, BANK_FLOATS * 4)
            .expect("a bank");
        floats(
            &bench
                .handles
                .read(handle, BANK_FLOATS * 4)
                .expect("a bank read"),
        )
    };
    let ext_rows = bench.read(ext_y, u64::from(ext * Y_WIDTH) * 4);
    let own_rows = bench.read(own_y, u64::from(own * Y_WIDTH) * 4);
    let width = Y_WIDTH as usize;
    let indptr_host = lanes.indptr();
    let mut begin = 0usize;
    for r in 0..lanes.rows.len() {
        let rows = lanes.rows[r] as usize;
        let rep = lanes.replay[r] as usize;
        let at = indptr_host[r] as usize;
        assert_eq!(
            bank(lanes.old_slots[r]),
            bank(lanes.new_slots[r]),
            "lane {r}: the verify scan committed other bank bits than the committed scan"
        );
        assert!(
            ext_rows[(begin + rep) * width..(begin + rep + rows) * width]
                == own_rows[at * width..(at + rows) * width],
            "lane {r}: the verify scan read its own rows out in other bits"
        );
        begin += rep + rows;
    }
    if let Some((old, new)) = snaps {
        for r in 0..lanes.rows.len() {
            let row = 2 * kernels_metal::attn::ssm::RS_SNAPSHOTS as usize * r;
            for i in 0..kernels_metal::attn::ssm::RS_SNAPSHOTS as usize {
                if old[row + i] <= 0 {
                    break;
                }
                let at = row + kernels_metal::attn::ssm::RS_SNAPSHOTS as usize + i;
                assert_eq!(
                    bank(old[at]),
                    bank(new[at]),
                    "lane {r}: snapshot {i} landed other bank bits"
                );
            }
        }
    }
    let committed_banks: Vec<i32> = lanes.new_slots.clone();
    let untouched = bank(0);
    for slot in committed_banks {
        assert_ne!(
            bank(slot),
            untouched,
            "slot {slot} was never committed into"
        );
    }
}

#[test]
fn a_verify_walk_is_the_committed_walk() {
    let Some(bench) = Bench::new() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    // Three lanes: a verify round's, a shorter replay, and a walk of 22
    // rows, past one staged pass.
    agree(
        &bench,
        &Lanes {
            rows: vec![8, 8, 14],
            replay: vec![5, 3, 8],
            commit: vec![4, 9, 20],
            old_slots: vec![1, 2, 3],
            new_slots: vec![4, 5, 6],
        },
        None,
    );
    // One lane, nothing replayed, every row committed.
    agree(
        &bench,
        &Lanes {
            rows: vec![8],
            replay: vec![0],
            commit: vec![8],
            old_slots: vec![7],
            new_slots: vec![8],
        },
        None,
    );
    // A prompt lane landing snapshots as it passes rows 3 and 19.
    let s = kernels_metal::attn::ssm::RS_SNAPSHOTS as usize;
    let mut old = vec![0i32; 2 * s];
    let mut new = vec![0i32; 2 * s];
    old[..2].copy_from_slice(&[3, 19]);
    new[..2].copy_from_slice(&[3, 19]);
    old[s..s + 2].copy_from_slice(&[9, 10]);
    new[s..s + 2].copy_from_slice(&[11, 12]);
    agree(
        &bench,
        &Lanes {
            rows: vec![24],
            replay: vec![0],
            commit: vec![24],
            old_slots: vec![13],
            new_slots: vec![14],
        },
        Some((&old, &new)),
    );
    eprintln!(
        "the verify scan's banks, snapshots and own readouts are the committed scan's, bit for bit"
    );
}

const LAYERS: u32 = 30;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a microbenchmark"]
fn a_verify_walk_prices_against_the_committed_walk() {
    use kernels_metal::encode::{Arg, Encode, Fire, Grid};
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
    let i32s = |v: &[i32]| bench.upload(&ints(v), v.len() as u32, 1, Dtype::I32);
    let indptr = i32s(&lanes.indptr());
    let layers: Vec<Committed> = (0..LAYERS)
        .map(|l| Committed {
            replay: i32s(&lanes.replay),
            commit: i32s(&lanes.commit),
            slots: i32s(&[(l % SLOTS) as i32]),
            lane0: 0,
        })
        .collect();
    let ext_y = bench.zeroed(u64::from(ext * Y_WIDTH) * 4, ext, Y_WIDTH, Dtype::F32);
    let own_y = bench.zeroed(u64::from(8 * Y_WIDTH) * 4, 8, Y_WIDTH, Dtype::F32);
    let work = bench.zeroed(BANK_FLOATS * 4, 1, 1, Dtype::F32);
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
    let old = |sink: &Sink, c: &Committed| {
        ssm::gated_delta_committed(
            sink,
            &tuning,
            x.qkv,
            indptr,
            c,
            x.gates,
            &pool(x.state, c.slots),
            work,
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            ext_y,
        )
        .expect("the committed scan");
    };
    let verify = |lanes: u32, groups: u32, vrows: u32| {
        move |sink: &Sink, c: &Committed| {
            sink.fire(
                Fire::at(
                    "attn/ssm_gdn_scan.metal",
                    ssm::gated_delta_verify_entry(lanes, groups, vrows).expect("a stamped entry"),
                )
                .apply(ssm::gated_delta_verify_grid(lanes, groups, vrows, V_HEADS)),
                &[
                    x.qkv.arg(),
                    indptr.arg(),
                    c.replay.arg(),
                    c.commit.arg(),
                    c.slots.arg(),
                    0i32.arg(),
                    x.gates.arg(),
                    x.state.arg_mut(),
                    own_y.arg_mut(),
                    (K_HEADS as i32).arg(),
                    (V_HEADS as i32).arg(),
                    sink.absent().expect("an absent table"),
                    (-1i32).arg(),
                ],
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
    let mut old_best = f64::MAX;
    let mut best = vec![f64::MAX; geometries.len()];
    for _ in 0..3 {
        old_best = old_best.min(price(&old));
        for (at, (l, g, v)) in geometries.into_iter().enumerate() {
            best[at] = best[at].min(price(&verify(l, g, v)));
        }
    }
    let mut line = format!("one layer's walk of 5 + 8 rows: committed {old_best:.1} us; verify");
    for ((l, g, v), us) in geometries.iter().zip(&best) {
        line += &format!(" l{l}s{g}v{v} {us:.1}");
    }
    eprintln!("{line}");
}
