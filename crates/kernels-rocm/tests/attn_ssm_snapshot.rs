//! A snapshot inside a lane lands the state a pass cut there folds, bit for
//! bit, and the lane's rows and its last state are the cut passes' too: the
//! gated delta rule and KDA, chunk-parallel and token by token.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::ssm;
use kernels_rocm::{RaggedTensor, RecurrentPool, Snapshots, Tensor};

const D: u32 = 128;
const HEADS: u32 = 4;

#[derive(Clone, Copy, Debug)]
enum Scan {
    Delta { serial: bool },
    Kda { serial: bool },
}

impl Scan {
    /// Each row's input planes, in order, as (width, dtype).
    fn planes(self) -> Vec<(u32, Dtype)> {
        match self {
            // qkv over two key heads, then [g_log | β].
            Scan::Delta { .. } => vec![(4 * D + HEADS * D, Dtype::Bf16), (2 * HEADS, Dtype::F32)],
            // [q | k | v], f, b.
            Scan::Kda { .. } => vec![
                (3 * HEADS * D, Dtype::Bf16),
                (HEADS * D, Dtype::Bf16),
                (HEADS, Dtype::Bf16),
            ],
        }
    }

    fn state(self) -> Dtype {
        match self {
            Scan::Delta { .. } => Dtype::Bf16,
            Scan::Kda { .. } => Dtype::F32,
        }
    }

    fn inputs(self, rows: usize) -> Vec<Vec<f32>> {
        self.planes()
            .into_iter()
            .enumerate()
            .map(|(at, (width, _))| {
                let w = width as usize;
                let v = values(rows * w, 7 + at as u64, 1.0);
                match (self, at) {
                    (Scan::Delta { .. }, 1) => v
                        .iter()
                        .enumerate()
                        .map(|(i, x)| {
                            if i % w < w / 2 {
                                -0.02 - 0.2 * x.abs()
                            } else {
                                0.5 + 0.45 * x
                            }
                        })
                        .collect(),
                    (Scan::Kda { .. }, 1) => v.into_iter().map(|x| round_bf16(2.0 * x)).collect(),
                    _ => v.into_iter().map(round_bf16).collect(),
                }
            })
            .collect()
    }
}

struct Fire<'a> {
    rows: std::ops::Range<u32>,
    lens: &'a [u32],
    slots: &'a [i32],
    snapshots: &'a [(u32, i32)],
}

/// One fire of `scan` over `slab` (its first lane snapshotting at
/// `snapshots`); its output rows.
fn fire(gpu: &Gpu, scan: Scan, inputs: &[Vec<f32>], slab: Tensor, f: &Fire) -> Vec<f32> {
    let rows = f.rows.len() as u32;
    let mut indptr = vec![0i32];
    for &len in f.lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let count = f.snapshots.len() as i32;
    let mut snap = vec![0, count];
    snap.resize(f.lens.len() + 1, count);
    let state = RecurrentPool {
        slab,
        slot_ids: gpu.column(f.slots),
        slot_stride_elems: i64::from(slab.width),
        snapshots: if f.snapshots.is_empty() {
            Snapshots::NONE
        } else {
            Snapshots {
                indptr: gpu.column(&snap),
                rows: gpu.column(&f.snapshots.iter().map(|s| s.0 as i32).collect::<Vec<_>>()),
                slots: gpu.column(&f.snapshots.iter().map(|s| s.1).collect::<Vec<_>>()),
                count: count as u32,
            }
        },
        ..STATE
    };
    let planes: Vec<Tensor> = scan
        .planes()
        .into_iter()
        .zip(inputs)
        .map(|((width, dtype), v)| {
            let w = width as usize;
            gpu.tensor(
                &v[f.rows.start as usize * w..f.rows.end as usize * w],
                rows,
                width,
                dtype,
            )
        })
        .collect();
    let indptr = gpu.column(&indptr);
    let mut y = gpu.zeros(rows, HEADS * D, Dtype::F32);
    let ctx = gpu.ctx();
    match scan {
        Scan::Delta { serial } => (if serial {
            ssm::gated_delta_chunked_serial
        } else {
            ssm::gated_delta_chunked
        })(
            &ctx,
            RaggedTensor {
                data: planes[0],
                indptr,
            },
            Tensor::ABSENT,
            planes[1],
            &state,
            2,
            HEADS,
            D,
            D,
            &mut y,
        ),
        Scan::Kda { serial } => (if serial {
            ssm::kda_chunked_serial
        } else {
            ssm::kda_chunked
        })(
            &ctx,
            RaggedTensor {
                data: planes[0],
                indptr,
            },
            planes[1],
            planes[2],
            gpu.tensor(
                &values((HEADS * D) as usize, 20, 1.0),
                1,
                HEADS * D,
                Dtype::F32,
            ),
            gpu.tensor(&[0.1, -0.2, 0.3, 0.0], 1, HEADS, Dtype::F32),
            &state,
            HEADS,
            D,
            1e-6,
            0.0,
            &mut y,
        ),
    }
    .unwrap();
    gpu.read(&y)
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn a_snapshot_lands_what_a_pass_cut_there_folds() {
    let Some(gpu) = Gpu::open() else { return };
    // Lane 0 snapshots mid-chunk and a whole chunk later; lane 1 rides along.
    let (a, b) = (200u32, 70u32);
    let cuts = [0u32, 100, 164, a];
    for scan in [
        Scan::Delta { serial: false },
        Scan::Delta { serial: true },
        Scan::Kda { serial: false },
        Scan::Kda { serial: true },
    ] {
        let inputs = scan.inputs((a + b) as usize);
        let stride = HEADS * D * D;
        let init: Vec<f32> = values(4 * stride as usize, 9, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let slot = |slab: &[f32], s: usize| slab[s * stride as usize..][..stride as usize].to_vec();

        let whole = gpu.tensor(&init, 4, stride, scan.state());
        let y = fire(
            &gpu,
            scan,
            &inputs,
            whole,
            &Fire {
                rows: 0..a + b,
                lens: &[a, b],
                slots: &[0, 1],
                snapshots: &[(cuts[1], 2), (cuts[2], 3)],
            },
        );
        let whole = gpu.read(&whole);

        let cut = gpu.tensor(&init, 4, stride, scan.state());
        let mut y_cut = Vec::new();
        for (piece, pair) in cuts.windows(2).enumerate() {
            y_cut.extend(fire(
                &gpu,
                scan,
                &inputs,
                cut,
                &Fire {
                    rows: pair[0]..pair[1],
                    lens: &[pair[1] - pair[0]],
                    slots: &[0],
                    snapshots: &[],
                },
            ));
            if piece < 2 {
                assert_eq!(
                    bits(&slot(&whole, 2 + piece)),
                    bits(&slot(&gpu.read(&cut), 0)),
                    "{scan:?}: the snapshot at row {} is not the state a pass cut there folds",
                    pair[1]
                );
            }
        }
        y_cut.extend(fire(
            &gpu,
            scan,
            &inputs,
            cut,
            &Fire {
                rows: a..a + b,
                lens: &[b],
                slots: &[1],
                snapshots: &[],
            },
        ));
        let cut = gpu.read(&cut);
        assert_eq!(
            bits(&y),
            bits(&y_cut),
            "{scan:?}: the rows differ from the cut passes'"
        );
        for s in 0..2 {
            assert_eq!(
                bits(&slot(&whole, s)),
                bits(&slot(&cut, s)),
                "{scan:?}: lane {s}'s last state differs"
            );
        }
    }
}
