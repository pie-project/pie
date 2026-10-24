//! Kimi Delta Attn against a host reference: the staging (norms, gates)
//! and the serial per-channel-decay recurrence in f64. The decode step (a
//! block a row and head, and the tiled step), the token-by-token prefill and
//! the chunk-parallel prefill, with commits, interior begins, fold masks,
//! steep decays and the staged window.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, round_bf16, sigmoid, values};
use dtype::Dtype;
use kernels_rocm::ssm;
use kernels_rocm::{Ctx, RaggedTensor, RecurrentPool, Tensor};

/// The fires share the context's named scratch planes, so the tests in this
/// file take turns on the device.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
struct Geo {
    heads: usize,
    d: usize,
    /// The decay's floor; zero leaves it unbounded.
    floor: f32,
    /// Decays of several nats a token, so a chunk's Γ spans hundreds.
    steep: bool,
}

impl Geo {
    fn width(self) -> usize {
        self.heads * self.d
    }
    fn stride(self) -> usize {
        self.heads * self.d * self.d
    }
}

/// The mixer's inputs over `rows` rows, and `slots` state slots.
struct Inputs {
    mixed: Vec<f32>,
    f: Vec<f32>,
    b: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    slab: Vec<f32>,
}

fn inputs(g: Geo, rows: usize, slots: usize, seed: u64) -> Inputs {
    let bf = |v: Vec<f32>| v.into_iter().map(round_bf16).collect::<Vec<_>>();
    let a = if g.steep { 8f32 } else { 0.5 };
    Inputs {
        mixed: bf(values(rows * 3 * g.width(), seed, 1.0)),
        f: bf(values(rows * g.width(), seed + 1, 2.0)),
        b: bf(values(rows * g.heads, seed + 2, 3.0)),
        dt_bias: values(g.width(), seed + 3, 1.0),
        a_log: values(g.heads, seed + 4, 0.3)
            .into_iter()
            .map(|x| (a * (1.0 + x)).ln())
            .collect(),
        slab: values(slots * g.stride(), seed + 5, 0.1),
    }
}

/// One row's staged q̂, k̂, v, gate ([H][D] each) and β ([H]), as the
/// staging kernels land them.
struct Token {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    gate: Vec<f64>,
    beta: Vec<f64>,
}

fn token(g: Geo, inp: &Inputs, row: usize) -> Token {
    let (w, d) = (g.width(), g.d);
    let mixed = &inp.mixed[row * 3 * w..(row + 1) * 3 * w];
    let normed = |plane: usize, scale: f32| -> Vec<f64> {
        let x = &mixed[plane * w..(plane + 1) * w];
        let mut out = vec![0f64; w];
        for h in 0..g.heads {
            let head = &x[h * d..(h + 1) * d];
            let ss: f32 = head.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss + EPS).sqrt();
            for (o, v) in out[h * d..(h + 1) * d].iter_mut().zip(head) {
                *o = f64::from(v * inv * scale);
            }
        }
        out
    };
    let mut gate = vec![0f64; w];
    for h in 0..g.heads {
        let a = inp.a_log[h].exp();
        for c in 0..d {
            let x = inp.f[row * w + h * d + c] + inp.dt_bias[h * d + c];
            gate[h * d + c] = f64::from(if g.floor < 0.0 {
                g.floor * sigmoid(a * x)
            } else {
                -a * if x > 20.0 { x } else { x.exp().ln_1p() }
            });
        }
    }
    Token {
        q: normed(0, 1.0 / (d as f32).sqrt()),
        k: normed(1, 1.0),
        v: mixed[2 * w..].iter().map(|&v| f64::from(v)).collect(),
        gate,
        beta: (0..g.heads)
            .map(|h| f64::from(sigmoid(inp.b[row * g.heads + h])))
            .collect(),
    }
}

/// The recurrence over one request's rows from `state` (one slot, [H][D][D]
/// as St[v][k]); lands the outputs in `y` ([rows][H·D]).
fn recur(
    g: Geo,
    inp: &Inputs,
    state: &mut [f64],
    rows: impl Iterator<Item = usize>,
    y: &mut [f32],
) {
    let d = g.d;
    for row in rows {
        let t = token(g, inp, row);
        for h in 0..g.heads {
            let at = |x: &[f64], c: usize| x[h * d + c];
            for vr in 0..d {
                let s = &mut state[(h * d + vr) * d..(h * d + vr + 1) * d];
                let mut kv = 0.0;
                for (c, x) in s.iter_mut().enumerate() {
                    *x *= at(&t.gate, c).exp();
                    kv += *x * at(&t.k, c);
                }
                let delta = (at(&t.v, vr) - kv) * t.beta[h];
                let mut o = 0.0;
                for (c, x) in s.iter_mut().enumerate() {
                    *x += at(&t.k, c) * delta;
                    o += *x * at(&t.q, c);
                }
                y[row * g.width() + h * d + vr] = o as f32;
            }
        }
    }
}

/// A request: its rows of the plane, its slot, and whether it folds.
struct Lane {
    rows: std::ops::Range<usize>,
    slot: usize,
    fold: bool,
}

fn reference(g: Geo, inp: &Inputs, rows: usize, lanes: &[Lane]) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0f32; rows * g.width()];
    let mut slab = inp.slab.clone();
    for ln in lanes {
        let at = ln.slot * g.stride()..(ln.slot + 1) * g.stride();
        let mut s: Vec<f64> = slab[at.clone()].iter().map(|&x| f64::from(x)).collect();
        recur(g, inp, &mut s, ln.rows.clone(), &mut y);
        if ln.fold {
            for (o, x) in slab[at].iter_mut().zip(&s) {
                *o = *x as f32;
            }
        }
    }
    (y, slab)
}

/// The device's tensors for the inputs.
struct Bound {
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: RecurrentPool,
    y: Tensor,
}

#[derive(Default)]
struct Seats<'a> {
    slots: &'a [i32],
    mask: Option<&'a [i32]>,
    commit: Option<&'a [i32]>,
    begin: Option<&'a [i32]>,
}

fn bind(gpu: &Gpu, g: Geo, inp: &Inputs, rows: usize, seats: &Seats) -> Bound {
    let (r, w) = (rows as u32, g.width() as u32);
    let mut state = RecurrentPool {
        slab: gpu.tensor(
            &inp.slab,
            (inp.slab.len() / g.stride()) as u32,
            g.stride() as u32,
            Dtype::F32,
        ),
        slot_ids: gpu.column(seats.slots),
        slot_stride_elems: g.stride() as i64,
        ..STATE
    };
    if let Some(m) = seats.mask {
        // One byte a lane.
        let bytes: Vec<u8> = m.iter().map(|&x| u8::from(x != 0)).collect();
        let mut words = vec![0i32; bytes.len().div_ceil(4)];
        for (i, &b) in bytes.iter().enumerate() {
            words[i / 4] |= i32::from(b) << (8 * (i % 4));
        }
        let t = gpu.ints(&words);
        state.write_state_mask = Tensor::new(t.ptr, bytes.len() as u32, 1, Dtype::U8);
    }
    if let Some(c) = seats.commit {
        state.commit_len = gpu.column(c);
    }
    if let Some(b) = seats.begin {
        state.begin_at = gpu.column(b);
    }
    Bound {
        mixed: gpu.tensor(&inp.mixed, r, 3 * w, Dtype::Bf16),
        f: gpu.tensor(&inp.f, r, w, Dtype::Bf16),
        b: gpu.tensor(&inp.b, r, g.heads as u32, Dtype::Bf16),
        dt_bias: gpu.tensor(&inp.dt_bias, 1, w, Dtype::F32),
        a_log: gpu.tensor(&inp.a_log, 1, g.heads as u32, Dtype::F32),
        state,
        y: gpu.zeros(r, w, Dtype::F32),
    }
}

fn step(gpu: &Gpu, ctx: &Ctx, g: Geo, bd: &mut Bound) {
    ssm::kda(
        ctx,
        bd.mixed,
        bd.f,
        bd.b,
        bd.dt_bias,
        bd.a_log,
        &bd.state,
        g.heads as u32,
        g.d as u32,
        EPS,
        g.floor,
        &mut bd.y,
    )
    .unwrap();
    gpu.sync();
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scan {
    Serial,
    Chunked,
}

fn chunked(gpu: &Gpu, ctx: &Ctx, g: Geo, bd: &mut Bound, indptr: &[i32], scan: Scan) {
    let run = match scan {
        Scan::Serial => ssm::kda_chunked_serial,
        Scan::Chunked => ssm::kda_chunked,
    };
    run(
        ctx,
        RaggedTensor {
            data: bd.mixed,
            indptr: gpu.column(indptr),
        },
        bd.f,
        bd.b,
        bd.dt_bias,
        bd.a_log,
        &bd.state,
        g.heads as u32,
        g.d as u32,
        EPS,
        g.floor,
        &mut bd.y,
    )
    .unwrap();
    gpu.sync();
}

/// The worst error relative to the plane's largest magnitude, and where.
fn worst(have: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(have.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    let mut w = (0f32, 0usize);
    for (i, (&a, &b)) in have.iter().zip(want).enumerate() {
        let e = (a - b).abs() / scale;
        let e = if e.is_nan() { f32::INFINITY } else { e };
        if e > w.0 {
            w = (e, i);
        }
    }
    w
}

fn check(what: &str, g: Geo, have: &[f32], want: &[f32], tol: f32) {
    let (e, at) = worst(have, want);
    let width = g.width();
    assert!(
        e <= tol,
        "{what}: row {} col {}: device {} host {} ({e} > {tol})",
        at / width,
        at % width,
        have[at],
        want[at]
    );
    eprintln!("{what}: worst error {e:.2e}");
}

/// The f32 kernels against the f64 recurrence.
const TIGHT: f32 = 1e-5;

/// The chunk passes multiply in bf16.
const LOOSE: f32 = 1.5e-2;

#[test]
fn the_step_answers_the_recurrence() {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    // 70 rows of 4 heads take a block a row and head; 3 rows the tiles.
    for (n, (g, rows)) in [
        (
            Geo {
                heads: 4,
                d: 128,
                floor: 0.0,
                steep: false,
            },
            70,
        ),
        (
            Geo {
                heads: 4,
                d: 128,
                floor: -5.0,
                steep: false,
            },
            3,
        ),
        (
            Geo {
                heads: 3,
                d: 64,
                floor: 0.0,
                steep: true,
            },
            5,
        ),
        (
            Geo {
                heads: 2,
                d: 96,
                floor: -5.0,
                steep: false,
            },
            140,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let inp = inputs(g, rows, rows, 10 + n as u64);
        let slots: Vec<i32> = (0..rows as i32).rev().collect();
        let mask: Vec<i32> = (0..rows as i32).map(|r| i32::from(r % 3 != 1)).collect();
        let lanes: Vec<Lane> = (0..rows)
            .map(|r| Lane {
                rows: r..r + 1,
                slot: slots[r] as usize,
                fold: mask[r] != 0,
            })
            .collect();
        let (y, slab) = reference(g, &inp, rows, &lanes);
        let seats = Seats {
            slots: &slots,
            mask: Some(&mask),
            ..Seats::default()
        };
        let mut bd = bind(&gpu, g, &inp, rows, &seats);
        step(&gpu, &gpu.ctx(), g, &mut bd);
        check(&format!("step {n}: y"), g, &gpu.read(&bd.y), &y, TIGHT);
        check(
            &format!("step {n}: state"),
            g,
            &gpu.read(&bd.state.slab),
            &slab,
            TIGHT,
        );
    }
}

#[test]
fn the_step_reads_its_window() {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    // Two heads take the tiles (a head over four blocks); GLM-5.3-Flash's 64
    // heads of 8 stated rows a block a whole head.
    for (n, (g, rows)) in [
        (
            Geo {
                heads: 2,
                d: 128,
                floor: 0.0,
                steep: false,
            },
            6usize,
        ),
        (
            Geo {
                heads: 64,
                d: 128,
                floor: -5.0,
                steep: false,
            },
            8,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        // The live rows are 2..rows - 1, their lanes 1..rows - 2; one live
        // lane has no seat and one does not fold.
        let live = rows - 3;
        let inp = inputs(g, rows, rows, 21 + n as u64);
        let mut slots: Vec<i32> = (0..rows as i32).rev().collect();
        slots[2] = -1;
        let mask: Vec<i32> = (0..rows).map(|l| i32::from(l != 3)).collect();
        let lanes: Vec<Lane> = (0..live)
            .filter(|&r| slots[r + 1] >= 0)
            .map(|r| Lane {
                rows: r + 2..r + 3,
                slot: slots[r + 1] as usize,
                fold: mask[r + 1] != 0,
            })
            .collect();
        let (y, slab) = reference(g, &inp, rows, &lanes);
        let seats = Seats {
            slots: &slots,
            mask: Some(&mask),
            ..Seats::default()
        };
        let mut bd = bind(&gpu, g, &inp, rows, &seats);
        let win = gpu.ints(&[live as i32, 2, live as i32, 1]);
        let ctx = gpu.ctx();
        ctx.arm_stage(win.ptr);
        step(&gpu, &ctx, g, &mut bd);
        check(
            &format!("windowed step {n}: y"),
            g,
            &gpu.read(&bd.y),
            &y,
            TIGHT,
        );
        check(
            &format!("windowed step {n}: state"),
            g,
            &gpu.read(&bd.state.slab),
            &slab,
            TIGHT,
        );
    }
}

struct Case {
    g: Geo,
    lens: Vec<usize>,
    begin: Option<Vec<i32>>,
    commit: Option<Vec<i32>>,
    mask: Option<Vec<i32>>,
}

impl Case {
    fn rows(&self) -> usize {
        self.lens.iter().sum()
    }
    fn indptr(&self) -> Vec<i32> {
        let mut p = vec![0i32];
        for &l in &self.lens {
            p.push(p.last().unwrap() + l as i32);
        }
        p
    }
    /// Request r at slot `lanes − 1 − r`, its span after the begin and the
    /// commit cap.
    fn lanes(&self, row0: usize) -> Vec<Lane> {
        let p = self.indptr();
        (0..self.lens.len())
            .map(|r| {
                let (t0, n) = (p[r] as usize, self.lens[r]);
                let b = self
                    .begin
                    .as_ref()
                    .map_or(0, |b| (b[r].max(0) as usize).min(n));
                let c = self
                    .commit
                    .as_ref()
                    .map_or(n - b, |c| (c[r] as usize).min(n - b));
                Lane {
                    rows: row0 + t0 + b..row0 + t0 + b + c,
                    slot: self.lens.len() - 1 - r,
                    fold: self.mask.as_ref().is_none_or(|m| m[r] != 0),
                }
            })
            .collect()
    }
    fn slots(&self) -> Vec<i32> {
        (0..self.lens.len() as i32).rev().collect()
    }
}

fn chunk_cases() -> Vec<Case> {
    let g = Geo {
        heads: 2,
        d: 128,
        floor: 0.0,
        steep: false,
    };
    vec![
        Case {
            g,
            lens: vec![70, 5, 130],
            begin: None,
            commit: None,
            mask: None,
        },
        Case {
            g: Geo {
                heads: 3,
                floor: -5.0,
                ..g
            },
            lens: vec![64, 1, 200, 129],
            begin: Some(vec![0, 0, 7, 64]),
            commit: Some(vec![64, 1, 150, 65]),
            mask: Some(vec![1, 0, 1, 1]),
        },
        Case {
            g: Geo { steep: true, ..g },
            lens: vec![150, 63],
            begin: None,
            commit: None,
            mask: None,
        },
    ]
}

fn chunked_answers(scan: Scan, tol: f32) {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    let mut cases = chunk_cases();
    if scan == Scan::Serial {
        // The token scan's own head widths.
        cases.push(Case {
            g: Geo {
                heads: 3,
                d: 64,
                floor: 0.0,
                steep: false,
            },
            lens: vec![33, 1, 90],
            begin: Some(vec![0, 0, 5]),
            commit: Some(vec![20, 1, 90]),
            mask: None,
        });
    }
    for (n, case) in cases.iter().enumerate() {
        let (g, rows) = (case.g, case.rows());
        let inp = inputs(g, rows, case.lens.len(), 30 + n as u64);
        let (y, slab) = reference(g, &inp, rows, &case.lanes(0));
        let slots = case.slots();
        let seats = Seats {
            slots: &slots,
            mask: case.mask.as_deref(),
            commit: case.commit.as_deref(),
            begin: case.begin.as_deref(),
        };
        let mut bd = bind(&gpu, g, &inp, rows, &seats);
        chunked(&gpu, &gpu.ctx(), g, &mut bd, &case.indptr(), scan);
        let (yd, sd) = (gpu.read(&bd.y), gpu.read(&bd.state.slab));
        check(&format!("{scan:?} {n}: y"), g, &yd, &y, tol);
        check(&format!("{scan:?} {n}: state"), g, &sd, &slab, tol);
        if scan == Scan::Chunked {
            let mut again = bind(&gpu, g, &inp, rows, &seats);
            chunked(&gpu, &gpu.ctx(), g, &mut again, &case.indptr(), scan);
            let (ya, sa) = (gpu.read(&again.y), gpu.read(&again.state.slab));
            assert!(
                yd.iter().zip(&ya).all(|(a, b)| a.to_bits() == b.to_bits())
                    && sd.iter().zip(&sa).all(|(a, b)| a.to_bits() == b.to_bits()),
                "case {n}: two chunked fires disagree"
            );
        }
    }
}

#[test]
fn the_serial_prefill_answers_the_recurrence() {
    chunked_answers(Scan::Serial, TIGHT);
}

#[test]
fn the_chunked_prefill_answers_the_recurrence() {
    chunked_answers(Scan::Chunked, LOOSE);
}

#[test]
fn the_prefill_reads_its_window() {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    // 200 stated rows and four stated lanes; live are three requests over
    // 150 rows from row 30, their lanes from 1.
    let case = Case {
        g: Geo {
            heads: 2,
            d: 128,
            floor: -5.0,
            steep: false,
        },
        lens: vec![70, 5, 75],
        begin: None,
        commit: None,
        mask: None,
    };
    let (g, rows, row0) = (case.g, 200, 30);
    let inp = inputs(g, rows, 4, 41);
    let mut indptr = case.indptr();
    indptr.push(*indptr.last().unwrap());
    let slots = vec![9, 2, 1, 0, 3];
    let lanes: Vec<Lane> = case
        .lanes(row0)
        .into_iter()
        .enumerate()
        .map(|(r, ln)| Lane {
            slot: slots[r + 1] as usize,
            ..ln
        })
        .collect();
    let (y, slab) = reference(g, &inp, rows, &lanes);
    for scan in [Scan::Serial, Scan::Chunked] {
        let seats = Seats {
            slots: &slots,
            ..Seats::default()
        };
        let mut bd = bind(&gpu, g, &inp, rows, &seats);
        let win = gpu.ints(&[150, row0 as i32, 3, 1]);
        let ctx = gpu.ctx();
        ctx.arm_stage(win.ptr);
        chunked(&gpu, &ctx, g, &mut bd, &indptr, scan);
        let tol = if scan == Scan::Serial { TIGHT } else { LOOSE };
        check(
            &format!("windowed {scan:?}: y"),
            g,
            &gpu.read(&bd.y),
            &y,
            tol,
        );
        check(
            &format!("windowed {scan:?}: state"),
            g,
            &gpu.read(&bd.state.slab),
            &slab,
            tol,
        );
    }
}
