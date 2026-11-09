#![cfg(target_vendor = "apple")]

//! Every recurrent-state kernel addresses its bank through a segment table and a
//! chunk id per piece of each lane's record. Run once over the slab (each record
//! whole at `slot * stride`, the table `StateMap::whole`) and once over the same
//! records cut into pieces along the op's atom axis (GDN, KDA and Mamba-2 heads,
//! conv channels, frame-cache frames), the pieces in chunks of two slots spread
//! over segments laid backwards with poison between them, the kernels land the
//! same output bits, the same record bits, and leave the poison alone.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::conv::resample;
use kernels_metal::ssm::{self, Committed, Mamba2, RS_SNAPSHOTS, Snapshots};
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{StateMap, Tensor, Tuning};
use poem_ir::Dtype;

/// What every element outside a piece holds: a quiet NaN no kernel writes.
const POISON_F32: u32 = 0x7fc0_dead;
const POISON_BF16: u16 = 0x7fd5;
/// Elements of poison between slots, chunks and segments; a multiple of four
/// so a piece starts where a four-wide load may.
const GAP: u64 = 64;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x0BAD_5EED_1234_5678;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16s(n: u64, f: impl Fn(u64) -> f32) -> Vec<u8> {
    (0..n).flat_map(|at| bf16(f(at)).to_le_bytes()).collect()
}

fn f32s(n: u64, f: impl Fn(u64) -> f32) -> Vec<u8> {
    (0..n).flat_map(|at| f(at).to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

fn words32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn words16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: std::cell::RefCell<Vec<Buffer>>,
}

impl Rig {
    fn open() -> Option<Rig> {
        let device = Context::bind().ok()?;
        Some(Rig {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            keep: std::cell::RefCell::new(Vec::new()),
        })
    }

    fn tensor(&self, bytes: &[u8], rows: u32, width: u32, dtype: Dtype) -> Tensor {
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
        self.tensor(&vec![0u8; bytes as usize], rows, width, dtype)
    }

    fn ints(&self, v: &[i32], dtype: Dtype) -> Tensor {
        self.tensor(&ints(v), v.len() as u32, 1, dtype)
    }

    fn map(&self, map: &StateMap) -> Tensor {
        self.tensor(&map.bytes(), 1, map.words().len() as u32, Dtype::U64)
    }

    fn read(&self, t: Tensor, bytes: u64) -> Vec<u8> {
        self.handles.read(t.buf, bytes).expect("a read back")
    }

    fn run(&self, fire: &dyn Fn(&Sink)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit().expect("the commit");
    }
}

/// Where a record's atoms run: outermost (`[atoms][atom_elems]`: GDN, KDA and
/// Mamba-2 heads, frame-cache frames) or innermost (`[rows][atoms]`: a conv's
/// channels).
#[derive(Clone, Copy)]
enum Axis {
    Outer,
    Inner { atoms: u64 },
}

/// One bank's layout: its segment table and each record's piece ids.
struct Layout {
    map: StateMap,
    ids: Vec<Vec<u32>>,
    elems: u64,
}

impl Layout {
    fn slab(records: u32, stride: u64) -> Layout {
        Layout {
            map: StateMap::whole(stride),
            ids: (0..records).map(|r| vec![r]).collect(),
            elems: u64::from(records) * stride,
        }
    }

    /// `records` records of `stride` elements, each cut into `pieces` pieces,
    /// two slots a chunk, three chunks a segment, the segments laid
    /// backwards and the pieces' ids strided, poison between every one.
    fn scattered(records: u32, stride: u64, atoms: u32, pieces: u32) -> Layout {
        assert!(atoms.is_multiple_of(pieces) && stride.is_multiple_of(u64::from(pieces)));
        let piece = stride / u64::from(pieces);
        let total = records * pieces;
        let slot_stride = piece + GAP;
        let chunk_stride = 2 * slot_stride + GAP;
        let seg_chunks = 3u32;
        let chunks = total.div_ceil(2);
        let segs = u64::from(chunks.div_ceil(seg_chunks));
        let span = u64::from(seg_chunks) * chunk_stride + 3 * GAP;
        let bases = (0..segs).map(|s| GAP + (segs - 1 - s) * span).collect();
        let step = [7u32, 5, 3, 11, 13]
            .into_iter()
            .find(|k| total % k != 0)
            .expect("a stride coprime with the ids");
        let ids = (0..records)
            .map(|r| {
                (0..pieces)
                    .map(|p| ((r * pieces + p) * step + 1) % total)
                    .collect()
            })
            .collect();
        Layout {
            map: StateMap {
                slot_stride,
                chunk_slots: 2,
                chunk_stride,
                seg_chunks,
                piece_atoms: atoms / pieces,
                pieces,
                bases,
            },
            ids,
            elems: GAP + segs * span,
        }
    }

    fn pieces(&self) -> u32 {
        self.map.pieces
    }

    /// The slab's records (`stride` elements each) laid out here; poison
    /// everywhere else.
    fn place<T: Copy>(&self, axis: Axis, stride: u64, slab: &[T], poison: T) -> Vec<T> {
        let mut out = vec![poison; self.elems as usize];
        self.walk(axis, stride, |at, from| {
            out[at as usize] = slab[from as usize]
        });
        out
    }

    /// Every (element here, element of the slab) pair a record holds.
    fn walk(&self, axis: Axis, stride: u64, mut f: impl FnMut(u64, u64)) {
        let pieces = u64::from(self.pieces());
        for (r, ids) in self.ids.iter().enumerate() {
            let record = r as u64 * stride;
            for (p, &id) in ids.iter().enumerate() {
                let at = self.map.slot(id);
                let p = p as u64;
                match axis {
                    Axis::Outer => {
                        let piece = stride / pieces;
                        for e in 0..piece {
                            f(at + e, record + p * piece + e);
                        }
                    }
                    Axis::Inner { atoms } => {
                        let per = atoms / pieces;
                        for row in 0..stride / atoms {
                            for a in 0..per {
                                f(at + row * per + a, record + row * atoms + p * per + a);
                            }
                        }
                    }
                }
            }
        }
    }

    /// A slot table: each entry's record's piece ids.
    fn lanes(&self, slots: &[u32]) -> Vec<i32> {
        slots
            .iter()
            .flat_map(|&s| self.ids[s as usize].iter().map(|&id| id as i32))
            .collect()
    }

    /// A snapshot table: per lane, its snapshot rows, then each snapshot's
    /// record's piece ids.
    fn snaps(&self, lanes: &[(&[u32], &[u32])]) -> Vec<i32> {
        let max = RS_SNAPSHOTS as usize;
        let pieces = self.pieces() as usize;
        let mut out = Vec::new();
        for (rows, slots) in lanes {
            let mut entry = vec![0i32; max * (1 + pieces)];
            for (at, (&row, &slot)) in rows.iter().zip(slots.iter()).enumerate() {
                entry[at] = row as i32;
                for (p, &id) in self.ids[slot as usize].iter().enumerate() {
                    entry[max + at * pieces + p] = id as i32;
                }
            }
            out.extend(entry);
        }
        out
    }
}

/// Runs `fire` over the f32 bank `init` (`records` records of `stride`) in
/// the slab and in `scattered`, and holds the two to one another: `fire`'s
/// outputs bit for bit, and the scattered bank is the slab's records placed,
/// its poison untouched.
fn agree(
    rig: &Rig,
    what: &str,
    records: u32,
    stride: u64,
    axis: Axis,
    scattered: &Layout,
    fire: &dyn Fn(&Layout, Tensor, Tensor) -> Vec<u8>,
) {
    let slab = Layout::slab(records, stride);
    let init: Vec<u32> = (0..u64::from(records) * stride)
        .map(|at| (0.25 * unit(at ^ 0xC3C3)).to_bits())
        .collect();
    let mut outs = Vec::new();
    let mut banks = Vec::new();
    for layout in [&slab, scattered] {
        let placed = layout.place(axis, stride, &init, POISON_F32);
        let bytes: Vec<u8> = placed.iter().flat_map(|w| w.to_le_bytes()).collect();
        // Kernels read only the handle; the shape is the slab's, as an op states it.
        let bank = rig.tensor(&bytes, records, stride as u32, Dtype::F32);
        let map = rig.map(&layout.map);
        outs.push(fire(layout, bank, map));
        banks.push(words32(&rig.read(bank, layout.elems * 4)));
    }
    assert!(
        outs[0] == outs[1],
        "{what}: the scattered bank's outputs are not the slab's"
    );
    assert!(
        banks[0] != slab.place(axis, stride, &init, POISON_F32),
        "{what}: the slab run landed nothing"
    );
    assert!(
        banks[1] == scattered.place(axis, stride, &banks[0], POISON_F32),
        "{what}: the scattered bank is not the slab's records placed, poison between"
    );
    eprintln!(
        "{what}: {} output bytes and {} record elements agree over {} pieces a record",
        outs[0].len(),
        u64::from(records) * stride,
        scattered.pieces()
    );
}

fn pool(state: Tensor, slots: Tensor, segments: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
        segments,
    }
}

const RECORDS: u32 = 4;
/// Lane 0 walks `SPANS[0]` rows from record 0 and lands its state as of
/// rows `SNAP_ROWS` in records 2 and 3; lane 1 walks the rest from record 1.
const SPANS: [u32; 2] = [48, 23];
const SNAP_ROWS: [u32; 2] = [5, 17];
const SNAP_SLOTS: [u32; 2] = [2, 3];

fn rows() -> u32 {
    SPANS.iter().sum()
}

/// The ragged walk's tables in `layout`: indptr, a row's slot table and the
/// snapshot table.
fn walk_tables(rig: &Rig, layout: &Layout) -> (Tensor, Tensor, Snapshots) {
    let indptr = rig.ints(&[0, SPANS[0] as i32, rows() as i32], Dtype::I32);
    let per_row: Vec<u32> = (0..SPANS[0])
        .map(|_| 0)
        .chain((0..SPANS[1]).map(|_| 1))
        .collect();
    let slots = rig.ints(&layout.lanes(&per_row), Dtype::U32);
    let snaps = Snapshots {
        table: rig.ints(
            &layout.snaps(&[(&SNAP_ROWS, &SNAP_SLOTS), (&[], &[])]),
            Dtype::I32,
        ),
        lane0: 0,
    };
    (indptr, slots, snaps)
}

#[test]
fn a_scattered_state_bank_is_the_slab() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());
    let simd = engine_metal::chip::tuning("", false);
    let staged = engine_metal::chip::tuning("[kernel]\ngdn.scan = \"staged\"", false);
    gated_delta(&rig, "gdn scan", &simd, 128, 128);
    gated_delta(&rig, "gdn staged", &staged, 128, 128);
    gated_delta(&rig, "gdn fallback", &simd, 96, 64);
    kda(&rig);
    mamba2(&rig);
    conv(&rig, "conv x4", 64, 4);
    conv(&rig, "conv", 6, 2);
    frame_cache(&rig);
}

fn gated_delta(rig: &Rig, what: &str, tuning: &Tuning, k_dim: u32, v_dim: u32) {
    const K_HEADS: u32 = 2;
    const V_HEADS: u32 = 4;
    let width = 2 * K_HEADS * k_dim + V_HEADS * v_dim;
    let y_width = V_HEADS * v_dim;
    let t = rows();
    let head = u64::from(k_dim) * u64::from(v_dim);
    let stride = u64::from(V_HEADS) * head;
    let qkv = rig.tensor(
        &bf16s(u64::from(t * width), |at| unit(at) * 2.0),
        t,
        width,
        Dtype::Bf16,
    );
    let gates = rig.tensor(
        &f32s(u64::from(t * 2 * V_HEADS), |at| {
            let v = unit(at ^ 0xA5A5) * 0.5 + 0.5;
            if (at / u64::from(V_HEADS)).is_multiple_of(2) {
                -0.02 - 0.38 * v
            } else {
                0.3 + 0.6 * v
            }
        }),
        t,
        2 * V_HEADS,
        Dtype::F32,
    );
    let scattered = Layout::scattered(RECORDS, stride, V_HEADS, 2);

    // The prompt's ragged walk, landing two snapshots.
    agree(
        rig,
        &format!("{what}: prompt"),
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let (indptr, slots, snaps) = walk_tables(rig, layout);
            let y = rig.zeroed(t, y_width, Dtype::F32);
            rig.run(&|sink| {
                ssm::gated_delta_chunked_landing(
                    sink,
                    tuning,
                    RaggedTensor { data: qkv, indptr },
                    y,
                    gates,
                    &pool(bank, slots, map),
                    K_HEADS,
                    V_HEADS,
                    k_dim,
                    v_dim,
                    y,
                    Some(snaps),
                )
                .expect("the prompt scan");
            });
            rig.read(y, u64::from(t * y_width) * 4)
        },
    );

    // One step a lane: rows 0 and 1 over records 1 and 3.
    agree(
        rig,
        &format!("{what}: step"),
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let slots = rig.ints(&layout.lanes(&[1, 3]), Dtype::U32);
            let y = rig.zeroed(2, y_width, Dtype::F32);
            let qkv = Tensor { rows: 2, ..qkv };
            let gates = Tensor { rows: 2, ..gates };
            rig.run(&|sink| {
                ssm::gated_delta(
                    sink,
                    tuning,
                    qkv,
                    y,
                    gates,
                    &pool(bank, slots, map),
                    K_HEADS,
                    V_HEADS,
                    k_dim,
                    v_dim,
                    y,
                )
                .expect("the step");
            });
            rig.read(y, 2 * u64::from(y_width) * 4)
        },
    );

    // The committed walk: lane 0 keeps 30 of its rows, lane 1 all of its.
    agree(
        rig,
        &format!("{what}: committed"),
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let indptr = rig.ints(&[0, SPANS[0] as i32, t as i32], Dtype::I32);
            let committed = Committed {
                replay: rig.ints(&[0, 0], Dtype::I32),
                commit: rig.ints(&[30, SPANS[1] as i32], Dtype::I32),
                slots: rig.ints(&layout.lanes(&[0, 1]), Dtype::I32),
                lane0: 0,
            };
            let work = rig.zeroed(2 * V_HEADS * v_dim, k_dim, Dtype::F32);
            let y = rig.zeroed(t, y_width, Dtype::F32);
            let state = pool(bank, committed.slots, map);
            rig.run(&|sink| {
                ssm::gated_delta_committed_landing(
                    sink, tuning, qkv, indptr, &committed, gates, &state, work, K_HEADS, V_HEADS,
                    k_dim, v_dim, y, None,
                )
                .expect("the committed walk");
            });
            rig.read(y, u64::from(t * y_width) * 4)
        },
    );
}

fn kda(rig: &Rig) {
    const HEADS: u32 = 4;
    const DIM: u32 = 64;
    let plane = HEADS * DIM;
    let t = rows();
    let stride = u64::from(HEADS) * u64::from(DIM) * u64::from(DIM);
    let mixed = rig.tensor(
        &bf16s(u64::from(t * 3 * plane), |at| unit(at ^ 0x1111)),
        t,
        3 * plane,
        Dtype::Bf16,
    );
    let f = rig.tensor(
        &bf16s(u64::from(t * plane), |at| unit(at ^ 0x2222)),
        t,
        plane,
        Dtype::Bf16,
    );
    let b = rig.tensor(
        &bf16s(u64::from(t * HEADS), |at| unit(at ^ 0x3333)),
        t,
        HEADS,
        Dtype::Bf16,
    );
    let dt_bias = rig.tensor(
        &f32s(u64::from(plane), |at| unit(at ^ 0x4444)),
        1,
        plane,
        Dtype::F32,
    );
    let a_log = rig.tensor(
        &f32s(u64::from(HEADS), |at| unit(at ^ 0x5555)),
        1,
        HEADS,
        Dtype::F32,
    );
    let scattered = Layout::scattered(RECORDS, stride, HEADS, 4);
    agree(
        rig,
        "kda: prompt",
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let (indptr, slots, _) = walk_tables(rig, layout);
            let y = rig.zeroed(t, plane, Dtype::F32);
            rig.run(&|sink| {
                ssm::kda_chunked(
                    sink,
                    RaggedTensor {
                        data: mixed,
                        indptr,
                    },
                    f,
                    b,
                    dt_bias,
                    a_log,
                    &pool(bank, slots, map),
                    HEADS,
                    DIM,
                    1e-6,
                    0.0,
                    y,
                )
                .expect("the prompt walk");
            });
            rig.read(y, u64::from(t * plane) * 4)
        },
    );
    agree(
        rig,
        "kda: step",
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let slots = rig.ints(&layout.lanes(&[2, 0]), Dtype::U32);
            let y = rig.zeroed(2, plane, Dtype::F32);
            rig.run(&|sink| {
                ssm::kda_step(
                    sink,
                    Tensor { rows: 2, ..mixed },
                    Tensor { rows: 2, ..f },
                    Tensor { rows: 2, ..b },
                    dt_bias,
                    a_log,
                    &pool(bank, slots, map),
                    HEADS,
                    DIM,
                    1e-6,
                    0.0,
                    y,
                )
                .expect("the step");
            });
            rig.read(y, 2 * u64::from(plane) * 4)
        },
    );
}

fn mamba2(rig: &Rig) {
    const HEADS: u32 = 4;
    const GROUPS: u32 = 2;
    const N: u32 = 128;
    let inner = HEADS * 64;
    let width = inner + 2 * GROUPS * N;
    let t = rows();
    let stride = u64::from(HEADS) * 64 * u64::from(N);
    let xbc = rig.tensor(
        &bf16s(u64::from(t * width), |at| unit(at ^ 0x6666)),
        t,
        width,
        Dtype::Bf16,
    );
    let dt = rig.tensor(
        &bf16s(u64::from(t * HEADS), |at| unit(at ^ 0x7777) - 1.0),
        t,
        HEADS,
        Dtype::Bf16,
    );
    let z = rig.tensor(
        &bf16s(u64::from(t * inner), |at| unit(at ^ 0x8888)),
        t,
        inner,
        Dtype::Bf16,
    );
    let per_head = |salt: u64| {
        rig.tensor(
            &f32s(u64::from(HEADS), |at| 0.5 * unit(at ^ salt)),
            1,
            HEADS,
            Dtype::F32,
        )
    };
    let m = Mamba2 {
        a_log: per_head(0x9999),
        dt_bias: per_head(0xAAAA),
        d: per_head(0xBBBB),
        norm: rig.tensor(
            &bf16s(u64::from(inner), |at| 1.0 + 0.1 * unit(at ^ 0xCCCC)),
            1,
            inner,
            Dtype::Bf16,
        ),
        eps: 1e-5,
        heads: HEADS,
        groups: GROUPS,
    };
    let scattered = Layout::scattered(RECORDS, stride, HEADS, 2);
    agree(
        rig,
        "mamba2: prompt",
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let (indptr, slots, snaps) = walk_tables(rig, layout);
            let y = rig.zeroed(t, inner, Dtype::Bf16);
            rig.run(&|sink| {
                ssm::mamba2_chunked(
                    sink,
                    RaggedTensor { data: xbc, indptr },
                    dt,
                    z,
                    m,
                    &pool(bank, slots, map),
                    N,
                    y,
                    Some(snaps),
                )
                .expect("the prompt walk");
            });
            rig.read(y, u64::from(t * inner) * 2)
        },
    );
    agree(
        rig,
        "mamba2: step",
        RECORDS,
        stride,
        Axis::Outer,
        &scattered,
        &|layout, bank, map| {
            let slots = rig.ints(&layout.lanes(&[3, 1]), Dtype::U32);
            let y = rig.zeroed(2, inner, Dtype::Bf16);
            rig.run(&|sink| {
                ssm::mamba2(
                    sink,
                    Tensor { rows: 2, ..xbc },
                    Tensor { rows: 2, ..dt },
                    Tensor { rows: 2, ..z },
                    m,
                    &pool(bank, slots, map),
                    N,
                    y,
                )
                .expect("the step");
            });
            rig.read(y, 2 * u64::from(inner) * 2)
        },
    );
}

fn conv(rig: &Rig, what: &str, channels: u32, pieces: u32) {
    const WIDTH: u32 = 4;
    let hist = WIDTH;
    let t = rows();
    let stride = u64::from(hist) * u64::from(channels);
    let axis = Axis::Inner {
        atoms: u64::from(channels),
    };
    let x = rig.tensor(
        &bf16s(u64::from(t * channels), |at| unit(at ^ 0xDDDD)),
        t,
        channels,
        Dtype::Bf16,
    );
    let weight = rig.tensor(
        &bf16s(u64::from(channels * WIDTH), |at| 0.6 * unit(at ^ 0xEEEE)),
        channels,
        WIDTH,
        Dtype::Bf16,
    );
    let scattered = Layout::scattered(RECORDS, stride, channels, pieces);
    agree(
        rig,
        &format!("{what}: prompt"),
        RECORDS,
        stride,
        axis,
        &scattered,
        &|layout, bank, map| {
            let (indptr, slots, snaps) = walk_tables(rig, layout);
            let y = rig.zeroed(t, channels, Dtype::Bf16);
            rig.run(&|sink| {
                ssm::causal_conv1d_chunked_landing(
                    sink,
                    RaggedTensor { data: x, indptr },
                    weight,
                    None,
                    &pool(bank, slots, map),
                    WIDTH,
                    1,
                    y,
                    Some(snaps),
                )
                .expect("the prompt conv");
            });
            rig.read(y, u64::from(t * channels) * 2)
        },
    );
    agree(
        rig,
        &format!("{what}: step"),
        RECORDS,
        stride,
        axis,
        &scattered,
        &|layout, bank, map| {
            let slots = rig.ints(&layout.lanes(&[2, 1]), Dtype::U32);
            let y = rig.zeroed(2, channels, Dtype::Bf16);
            rig.run(&|sink| {
                ssm::causal_conv1d(
                    sink,
                    Tensor { rows: 2, ..x },
                    weight,
                    None,
                    &pool(bank, slots, map),
                    WIDTH,
                    1,
                    y,
                )
                .expect("the step conv");
            });
            rig.read(y, 2 * u64::from(channels) * 2)
        },
    );
    agree(
        rig,
        &format!("{what}: committed"),
        RECORDS,
        stride,
        axis,
        &scattered,
        &|layout, bank, map| {
            let indptr = rig.ints(&[0, SPANS[0] as i32, t as i32], Dtype::I32);
            let committed = Committed {
                replay: rig.ints(&[0, 0], Dtype::I32),
                commit: rig.ints(&[30, SPANS[1] as i32], Dtype::I32),
                slots: rig.ints(&layout.lanes(&[0, 1]), Dtype::I32),
                lane0: 0,
            };
            let y = rig.zeroed(t, channels, Dtype::Bf16);
            let state = pool(bank, committed.slots, map);
            rig.run(&|sink| {
                ssm::causal_conv1d_committed_landing(
                    sink, x, indptr, &committed, weight, &state, WIDTH, 1, y, None,
                )
                .expect("the committed conv");
            });
            rig.read(y, u64::from(t * channels) * 2)
        },
    );
}

/// A frame cache's slot is `[frames][plane][channels]` bf16, cut a frame a
/// piece; every clip here is one plane wide, so a slot is just its frames.
fn frame_cache(rig: &Rig) {
    const C: u32 = 3;
    const FRAMES: u32 = 2;
    const CLIPS: [[i32; 3]; 3] = [[3, 2, 2], [1, 2, 2], [2, 2, 2]];
    let plane = 4u32;
    let mut grid = Vec::new();
    let mut off = 0;
    for e in CLIPS {
        grid.extend_from_slice(&[e[0], e[1], e[2], off]);
        off += e[0] * e[1] * e[2];
    }
    let voxels = off as u32;
    let records = CLIPS.len() as u32;
    let cache_rows = records * FRAMES * plane;
    let stride = u64::from(FRAMES * plane * C);
    let x = rig.tensor(
        &bf16s(u64::from(voxels * C), |at| 4.0 * unit(at)),
        voxels,
        C,
        Dtype::Bf16,
    );
    let old = rig.tensor(
        &bf16s(u64::from(cache_rows * C), |at| 4.0 * unit(at + 7717)),
        cache_rows,
        C,
        Dtype::Bf16,
    );
    let grid = rig.tensor(&ints(&grid), records, 4, Dtype::I32);
    let clip_slots: [u32; 3] = [2, 0, 1];
    let slab = Layout::slab(records, stride);
    let scattered = Layout::scattered(records, stride, FRAMES, FRAMES);
    let mut runs = Vec::new();
    for layout in [&slab, &scattered] {
        let bank = rig.tensor(
            &vec![POISON_BF16; layout.elems as usize]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            1,
            layout.elems as u32,
            Dtype::Bf16,
        );
        let slots = rig.ints(&layout.lanes(&clip_slots), Dtype::I32);
        let map = rig.map(&layout.map);
        let slab_view = Tensor::new(bank.buf, FRAMES * plane, C, Dtype::Bf16);
        let back = rig.zeroed(cache_rows, C, Dtype::Bf16);
        rig.run(&|sink| {
            resample::cache_store(sink, x, old, slots, map, grid, FRAMES, slab_view, 0)
                .expect("the frame-cache store");
        });
        rig.run(&|sink| {
            resample::cache_gather(sink, slab_view, slots, map, grid, FRAMES, back)
                .expect("the frame-cache gather");
        });
        runs.push((
            words16(&rig.read(bank, layout.elems * 2)),
            rig.read(back, u64::from(cache_rows * C) * 2),
        ));
    }
    assert!(
        runs[0].1 == runs[1].1,
        "frame cache: the scattered gather is not the slab's"
    );
    assert!(
        runs[1].0 == scattered.place(Axis::Outer, stride, &runs[0].0, POISON_BF16),
        "frame cache: the scattered slab is not the slab's frames placed, poison between"
    );
    eprintln!(
        "frame cache: {} gathered bytes agree over {} pieces a slot",
        runs[0].1.len(),
        scattered.pieces()
    );
}
