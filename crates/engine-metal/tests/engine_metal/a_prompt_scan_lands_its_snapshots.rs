#![cfg(target_vendor = "apple")]

//! A prompt's scans land the state as of a row inside it in a slot of its
//! own, bit for bit the state the same scan over just the rows before it
//! leaves: the staged, simdgroup and plain prompt scans, the committed scans
//! of a fire whose seat buffers another lane, and the conv.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::ssm::{self, Committed, RS_SNAPSHOTS, Snapshots};
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use poem_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 32;
const T: u32 = 77;
const SNAPS: [u32; 2] = [16, 48];

const CHANNELS: u32 = 8192;
const WIDTH: u32 = 4;

fn unit(at: u64) -> f32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    ((x >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    (((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// One lane's snapshot row: `rows` into it, landing in `slots`.
fn table(rows: &[u32], slots: &[u32]) -> Vec<i32> {
    let max = RS_SNAPSHOTS as usize;
    let mut entry = vec![0i32; 2 * max];
    for (at, (&row, &slot)) in rows.iter().zip(slots).enumerate() {
        entry[at] = row as i32;
        entry[max + at] = slot as i32;
    }
    entry
}

struct Gpu {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Gpu {
    fn upload(&self, bytes: &[u8]) -> Buffer {
        let mut buffer = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    }

    fn bind(&self, b: &Buffer) -> u32 {
        self.handles.bind(b, 0, b.bytes()).expect("a handle")
    }

    fn ints(&self, v: &[i32], dtype: Dtype) -> Tensor {
        let b = self.upload(&i32s(v));
        Tensor::new(self.bind(&b), v.len() as u32, 1, dtype)
    }

    fn snapshots(&self, rows: &[u32], slots: &[u32]) -> Snapshots {
        Snapshots {
            table: self.ints(&table(rows, slots), Dtype::I32),
            lane0: 0,
        }
    }

    fn run(&self, fire: &dyn Fn(&Sink)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit().expect("the commit");
    }

    fn read(&self, b: &Buffer, at: u64, bytes: u64) -> Vec<u8> {
        self.handles
            .read(self.handles.bind(b, at, bytes).expect("a cut"), bytes)
            .expect("a read")
    }
}

/// The gated delta at head width `dim` under `tuning`: slot 0 runs the lane
/// whole and lands its snapshots in slots 1 and 2; slots 3 and 4 run just
/// the rows before each; slot 5 runs it whole with none. Slot 7 runs the
/// committed scan whole, landing the first snapshot in slot 6, against slot
/// 8's committed scan that keeps the state as of that row.
fn gated_delta(gpu: &Gpu, dim: u32, tuning: &kernels_metal::Tuning, what: &str) {
    let qkv_width = 2 * K_HEADS * dim + V_HEADS * dim;
    let bank_floats = u64::from(V_HEADS * dim * dim);
    let slots = 9u32;
    let qkv_b = gpu.upload(
        &(0..u64::from(T * qkv_width))
            .flat_map(|at| bf16(unit(at) * 0.5))
            .collect::<Vec<u8>>(),
    );
    let gates: Vec<f32> = (0..T * 2 * V_HEADS)
        .map(|at| {
            let u = unit(u64::from(at) ^ 0xA5A5) * 0.5 + 0.5;
            if at % (2 * V_HEADS) < V_HEADS {
                -0.02 - 0.38 * u
            } else {
                0.3 + 0.6 * u
            }
        })
        .collect();
    let gates_b = gpu.upload(&f32s(&gates));
    let bank: Vec<f32> = (0..bank_floats).map(|at| 0.1 * unit(at ^ 0xC3C3)).collect();
    let state_b = gpu.upload(&f32s(
        &(0..slots)
            .flat_map(|_| bank.iter().copied())
            .collect::<Vec<f32>>(),
    ));
    let work_b = gpu.upload(&vec![0u8; (bank_floats * 4) as usize]);
    let y_b = gpu.upload(&vec![0u8; (T * V_HEADS * dim * 4) as usize]);
    let state = Tensor::new(gpu.bind(&state_b), slots * V_HEADS * dim, dim, Dtype::F32);
    let qkv = Tensor::new(gpu.bind(&qkv_b), T, qkv_width, Dtype::Bf16);
    let gates_t = Tensor::new(gpu.bind(&gates_b), T, 2 * V_HEADS, Dtype::F32);
    let y = Tensor::new(gpu.bind(&y_b), T, V_HEADS * dim, Dtype::F32);
    let work = Tensor::new(gpu.bind(&work_b), V_HEADS * dim, dim, Dtype::F32);
    let pool = |slot: u32, rows: u32| RecurrentPool {
        state,
        slots: gpu.ints(&vec![slot as i32; rows as usize], Dtype::U32),
        conv_state: state,
        new_conv_state: state,
    };
    let indptr = |rows: u32| gpu.ints(&[0, rows as i32], Dtype::I32);
    let scan = |slot: u32, rows: u32, snapshots: Option<Snapshots>| {
        gpu.run(&|sink| {
            ssm::gated_delta_chunked_landing(
                sink,
                tuning,
                RaggedTensor {
                    data: qkv,
                    indptr: indptr(rows),
                },
                qkv,
                gates_t,
                &pool(slot, rows),
                K_HEADS,
                V_HEADS,
                dim,
                dim,
                y,
                snapshots,
            )
            .expect("the scan");
        });
    };
    scan(0, T, Some(gpu.snapshots(&SNAPS, &[1, 2])));
    scan(3, SNAPS[0], None);
    scan(4, SNAPS[1], None);
    scan(5, T, None);
    let committed = |slot: u32, commit: u32, snapshots: Option<Snapshots>| {
        gpu.run(&|sink| {
            ssm::gated_delta_committed_landing(
                sink,
                tuning,
                qkv,
                indptr(T),
                &Committed {
                    replay: gpu.ints(&[0], Dtype::I32),
                    commit: gpu.ints(&[commit as i32], Dtype::I32),
                    slots: gpu.ints(&[slot as i32], Dtype::I32),
                    lane0: 0,
                },
                gates_t,
                &pool(slot, T),
                work,
                K_HEADS,
                V_HEADS,
                dim,
                dim,
                y,
                snapshots,
            )
            .expect("the committed scan");
        });
    };
    committed(7, T, Some(gpu.snapshots(&SNAPS[..1], &[6])));
    committed(8, SNAPS[0], None);
    let bytes = bank_floats * 4;
    let read = |slot: u32| gpu.read(&state_b, u64::from(slot) * bytes, bytes);
    assert_eq!(
        read(1),
        read(3),
        "{what}: the first snapshot is the state as of its row"
    );
    assert_eq!(
        read(2),
        read(4),
        "{what}: the second snapshot is the state as of its row"
    );
    assert_eq!(
        read(0),
        read(5),
        "{what}: landing snapshots leaves the lane's own state"
    );
    assert_ne!(
        read(1),
        read(2),
        "{what}: the two rows leave different states"
    );
    assert_eq!(
        read(6),
        read(8),
        "{what}: the committed scan lands the state as of its row"
    );
}

#[test]
fn a_prompt_scan_lands_its_snapshots() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let gpu = Gpu {
        device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
    };
    let staged = engine_metal::chip::tuning("[kernel]\ngdn.scan = \"staged\"", false);
    let simd = engine_metal::chip::tuning("", false);
    gated_delta(&gpu, 128, &staged, "staged");
    gated_delta(&gpu, 128, &simd, "simdgroup");
    // 96 is no stamped scan's width: the plain kernels run it.
    gated_delta(&gpu, 96, &simd, "plain");

    // The conv: the window as of each snapshot row.
    let hist = WIDTH;
    let slots = 8u32;
    let x_b = gpu.upload(
        &(0..u64::from(T * CHANNELS))
            .flat_map(|at| bf16(unit(at ^ 0x77) * 2.0))
            .collect::<Vec<u8>>(),
    );
    let w_b = gpu.upload(
        &(0..u64::from(CHANNELS * WIDTH))
            .flat_map(|at| bf16(unit(at ^ 0x99) * 0.6))
            .collect::<Vec<u8>>(),
    );
    let carried: Vec<f32> = (0..hist * CHANNELS)
        .map(|at| unit(u64::from(at) ^ 0x5151))
        .collect();
    let conv_b = gpu.upload(&f32s(
        &(0..slots)
            .flat_map(|_| carried.iter().copied())
            .collect::<Vec<f32>>(),
    ));
    let conv = Tensor::new(gpu.bind(&conv_b), slots, hist * CHANNELS, Dtype::F32);
    let cy_b = gpu.upload(&vec![0u8; (T * CHANNELS * 2) as usize]);
    let x = Tensor::new(gpu.bind(&x_b), T, CHANNELS, Dtype::Bf16);
    let weight = Tensor::new(gpu.bind(&w_b), CHANNELS, WIDTH, Dtype::Bf16);
    let conv_pool = |slot: u32, rows: u32| RecurrentPool {
        state: conv,
        slots: gpu.ints(&vec![slot as i32; rows as usize], Dtype::U32),
        conv_state: conv,
        new_conv_state: conv,
    };
    for (slot, rows, snapshots) in [
        (0u32, T, Some(gpu.snapshots(&SNAPS, &[1, 2]))),
        (3, SNAPS[0], None),
        (4, SNAPS[1], None),
    ] {
        gpu.run(&|sink| {
            ssm::causal_conv1d_chunked_landing(
                sink,
                RaggedTensor {
                    data: x,
                    indptr: gpu.ints(&[0, rows as i32], Dtype::I32),
                },
                weight,
                None,
                &conv_pool(slot, rows),
                WIDTH,
                1,
                Tensor::new(gpu.bind(&cy_b), T, CHANNELS, Dtype::Bf16),
                snapshots,
            )
            .expect("the conv");
        });
    }
    gpu.run(&|sink| {
        ssm::causal_conv1d_committed_landing(
            sink,
            x,
            gpu.ints(&[0, T as i32], Dtype::I32),
            &Committed {
                replay: gpu.ints(&[0], Dtype::I32),
                commit: gpu.ints(&[T as i32], Dtype::I32),
                slots: gpu.ints(&[5], Dtype::I32),
                lane0: 0,
            },
            weight,
            &conv_pool(5, T),
            WIDTH,
            1,
            Tensor::new(gpu.bind(&cy_b), T, CHANNELS, Dtype::Bf16),
            Some(gpu.snapshots(&SNAPS[..1], &[6])),
        )
        .expect("the committed conv");
    });
    let bytes = u64::from(hist * CHANNELS) * 4;
    let window = |slot: u32| gpu.read(&conv_b, u64::from(slot) * bytes, bytes);
    assert_eq!(
        window(1),
        window(3),
        "the conv's first snapshot is its window as of the row"
    );
    assert_eq!(
        window(2),
        window(4),
        "the conv's second snapshot is its window as of the row"
    );
    assert_eq!(
        window(6),
        window(3),
        "the committed conv lands its window as of the row"
    );
    eprintln!("snapshots at rows {SNAPS:?} of {T} land every scan's state bit for bit");
}
