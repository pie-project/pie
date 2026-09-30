//! The weight storage contract (`kernels_xla::pack`): banks landed the way
//! the engine lands them — codes one per element in a native narrow type,
//! mxfp4 pre-scaled to e5m2 — read by the dense, routed and gather kernels.
//! The device types come from `engine_xla::trace::storage`, as the engine's
//! tracer makes them.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use dtype::Dtype;
use engine_xla::bench::{assert_close, client, round_bf16};
use engine_xla::device::element_type;
use engine_xla::pjrt::Arg;
use engine_xla::trace::storage;
use kernels_xla::hlo::{Func, Val, bf16_bits};
use kernels_xla::linear::{moe, quant};
use kernels_xla::{Bank, Cx, Emit, Env, Tensor, layout, pack};

fn hash(i: usize, seed: u32) -> u32 {
    (i as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40503))
        .rotate_left(13)
        .wrapping_mul(0x9E37_79B1)
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| round_bf16(((hash(i, seed) >> 8) % 2000) as f32 / 1000.0 - 1.0))
        .collect()
}

fn bf16s(xs: &[f32]) -> Vec<u8> {
    xs.iter()
        .flat_map(|&x| bf16_bits(x).to_le_bytes())
        .collect()
}

/// Host arrays in their landed form, run through one module.
#[derive(Default)]
struct Native {
    planes: Vec<(Dtype, u32, u32, Vec<u8>)>,
}

impl Native {
    fn add(&mut self, dtype: Dtype, rows: u32, width: u32, host: Vec<u8>) -> Tensor {
        self.planes.push((dtype, rows, width, host));
        Tensor::new(self.planes.len() as u32 - 1, rows, width, dtype)
    }

    fn bf16(&mut self, rows: u32, width: u32, xs: &[f32]) -> Tensor {
        self.add(Dtype::Bf16, rows, width, bf16s(xs))
    }

    fn read(&self, t: Tensor) -> Vec<f32> {
        self.planes[t.buf as usize]
            .3
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    }

    fn run(
        &mut self,
        body: impl FnOnce(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> bool {
        let Some(client) = client() else {
            return false;
        };
        let tracer = Tracer {
            func: RefCell::new(Func::new("main")),
            env: RefCell::new(Roots {
                shapes: self.planes.iter().map(|p| (p.0, p.1, p.2)).collect(),
                ..Roots::default()
            }),
        };
        body(&tracer).unwrap_or_else(|e| panic!("{e}"));
        let Tracer { func, env } = tracer;
        let (func, env) = (func.into_inner(), env.into_inner());
        let written: Vec<u32> = env.written.iter().copied().collect();
        let results: Vec<Val> = written.iter().map(|b| env.current[b]).collect();
        let text = func.module("pack", &results);
        let client = client.lock().unwrap();
        let dev = client.devices()[0];
        let exe = client
            .compile(&text)
            .unwrap_or_else(|e| panic!("{e}\n{text}"));
        let uploads: Vec<_> = env
            .params
            .iter()
            .map(|&b| {
                let (d, r, w, ref host) = self.planes[b as usize];
                let ty = storage(d, r, w).unwrap();
                client
                    .upload(dev, host, element_type(ty.elem), &ty.dims)
                    .unwrap()
            })
            .collect();
        let (outs, done) = exe
            .execute(dev, uploads.iter().map(Arg::Keep).collect())
            .unwrap();
        done.wait().unwrap();
        for (b, out) in written.iter().zip(outs) {
            self.planes[*b as usize].3 = out.download().unwrap();
        }
        true
    }
}

#[derive(Default)]
struct Roots {
    shapes: Vec<(Dtype, u32, u32)>,
    current: HashMap<u32, Val>,
    params: Vec<u32>,
    written: BTreeSet<u32>,
}

impl Env for Roots {
    fn read(&mut self, f: &mut Func, t: Tensor) -> Result<Val, kernels_xla::Error> {
        let v = if let Some(&v) = self.current.get(&t.buf) {
            v
        } else {
            let (d, r, w) = self.shapes[t.buf as usize];
            let v = f.param(storage(d, r, w).expect("a storage form"), None);
            self.params.push(t.buf);
            self.current.insert(t.buf, v);
            v
        };
        // A handle may view its root as another same-width element (the
        // engine's tracer bitcasts, e.g. e8m0 scales read as u8).
        match storage(t.dtype, t.rows, t.width) {
            Some(ty) if ty.elem != f.elem(v) && ty.elem.bits() == f.elem(v).bits() => {
                Ok(f.bitcast(v, ty.elem)?)
            }
            _ => Ok(v),
        }
    }

    fn write(&mut self, _f: &mut Func, t: Tensor, v: Val) -> Result<(), kernels_xla::Error> {
        self.current.insert(t.buf, v);
        self.written.insert(t.buf);
        Ok(())
    }
}

struct Tracer {
    func: RefCell<Func>,
    env: RefCell<Roots>,
}

impl Emit for Tracer {
    fn emit(
        &self,
        body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<(), kernels_xla::Error> {
        let mut func = self.func.borrow_mut();
        let mut env = self.env.borrow_mut();
        let mut cx = Cx::new(&mut func, &mut *env);
        body(&mut cx)
    }
}

// ---------------------------------------------------------------- banks

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// A host bank of `rows` weight rows of `k`: its f32 weights and the
/// checkpoint's planes, packed low bits first.
struct HostBank {
    w: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    exps: Vec<u8>,
    dtype: Dtype,
    group: u32,
    bits: u32,
}

fn pack_codes(q: &[u32], bits: u32) -> Vec<u8> {
    let per = (8 / bits) as usize;
    q.chunks(per)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |a, (j, &v)| a | ((v as u8) << (j as u32 * bits)))
        })
        .collect()
}

fn affine(dtype: Dtype, rows: usize, k: usize, seed: u32) -> HostBank {
    let (group, bits) = match dtype {
        Dtype::U4g64 => (64, 4),
        Dtype::U4g32 => (32, 4),
        Dtype::U2g32 => (32, 2),
        Dtype::U8g64 => (64, 8),
        other => panic!("{other:?}"),
    };
    let g = group as usize;
    let q: Vec<u32> = (0..rows * k).map(|i| hash(i, seed) % (1 << bits)).collect();
    let scales: Vec<f32> = (0..rows * k / g)
        .map(|i| round_bf16(0.01 + (hash(i, seed + 1) % 50) as f32 / 1000.0))
        .collect();
    let biases: Vec<f32> = (0..rows * k / g)
        .map(|i| round_bf16((hash(i, seed + 2) % 400) as f32 / 1000.0 - 0.2))
        .collect();
    let w = (0..rows * k)
        .map(|i| scales[i / g] * q[i] as f32 + biases[i / g])
        .collect();
    HostBank {
        w,
        codes: pack_codes(&q, bits),
        scales,
        biases,
        exps: Vec::new(),
        dtype,
        group,
        bits,
    }
}

fn mxfp4(rows: usize, k: usize, seed: u32) -> HostBank {
    let q: Vec<u32> = (0..rows * k).map(|i| hash(i, seed) % 16).collect();
    let exps: Vec<u8> = (0..rows * k / 32)
        .map(|i| 122 + (hash(i, seed + 1) % 5) as u8)
        .collect();
    let w = (0..rows * k)
        .map(|i| E2M1[q[i] as usize] * 2f32.powi(i32::from(exps[i / 32]) - 127))
        .collect();
    HostBank {
        w,
        codes: pack_codes(&q, 4),
        scales: Vec::new(),
        biases: Vec::new(),
        exps,
        dtype: Dtype::Mxfp4,
        group: 32,
        bits: 4,
    }
}

impl HostBank {
    /// Lands the bank as the engine does, `lead` rows of `width` codes each
    /// (a routed bank: one row per expert); `prescale` lands mxfp4 as e5m2.
    fn land(&self, nat: &mut Native, lead: u32, prescale: bool) -> Bank {
        let codes_per_row = (self.w.len() / lead as usize) as u32;
        let factors = codes_per_row / self.group;
        let codes = if self.dtype == Dtype::Mxfp4 {
            if prescale {
                let w = pack::mxfp4_e5m2(&self.codes, &self.exps).expect("exact");
                nat.add(Dtype::E5m2, lead, codes_per_row, w)
            } else {
                nat.add(
                    Dtype::Mxfp4,
                    lead,
                    codes_per_row / 2,
                    pack::land(Dtype::Mxfp4, &self.codes).unwrap(),
                )
            }
        } else {
            nat.add(
                self.dtype,
                lead,
                codes_per_row,
                pack::land(self.dtype, &self.codes).unwrap(),
            )
        };
        let (scales, biases) = if self.dtype == Dtype::Mxfp4 {
            (nat.add(Dtype::E8m0, lead, factors, self.exps.clone()), None)
        } else {
            (
                nat.bf16(lead, factors, &self.scales),
                Some(nat.bf16(lead, factors, &self.biases)),
            )
        };
        Bank {
            codes,
            scales,
            biases,
            group: self.group,
            bits: self.bits,
        }
    }
}

fn gemm_ref(w: &[f32], n: usize, k: usize, x: &[f32], m: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            y[r * n + c] = round_bf16((0..k).map(|j| w[c * k + j] * x[r * k + j]).sum());
        }
    }
    y
}

fn check(got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
    assert_close(got, want, 0.01 * scale, 1e-2);
}

#[test]
fn dense_projections_read_native_codes_and_prescaled_weights() {
    let (n, k) = (48usize, 256usize);
    let mut banks: Vec<(HostBank, bool)> = [Dtype::U4g64, Dtype::U4g32, Dtype::U2g32, Dtype::U8g64]
        .into_iter()
        .enumerate()
        .map(|(i, d)| (affine(d, n, k, 10 + i as u32), false))
        .collect();
    banks.push((mxfp4(n, k, 20), false));
    banks.push((mxfp4(n, k, 21), true));
    // 3 rows run group-batched, 70 decode the weight once.
    for m in [3usize, 70] {
        let x = data(m * k, 30 + m as u32);
        let mut nat = Native::default();
        let xt = nat.bf16(m as u32, k as u32, &x);
        let mut outs = Vec::new();
        for (bank, pre) in &banks {
            let landed = bank.land(&mut nat, n as u32, *pre);
            let y = nat.bf16(m as u32, n as u32, &vec![0.0; m * n]);
            outs.push((landed, y));
        }
        let o = outs.clone();
        if !nat.run(|ctx| {
            for (bank, y) in &o {
                quant::matmul(ctx, xt, *bank, *y)?;
            }
            Ok(())
        }) {
            return;
        }
        for ((bank, _), (_, y)) in banks.iter().zip(&outs) {
            check(&nat.read(*y), &gemm_ref(&bank.w, n, k, &x, m));
        }
    }
}

#[test]
fn an_embedding_gathers_native_codes() {
    let (vocab, width) = (40usize, 128usize);
    let table = affine(Dtype::U4g64, vocab, width, 40);
    let ids: Vec<i32> = vec![3, 39, 0, 17, 50];
    let mut nat = Native::default();
    let it = nat.add(
        Dtype::I32,
        ids.len() as u32,
        1,
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    let bank = table.land(&mut nat, vocab as u32, false);
    let y = nat.bf16(
        ids.len() as u32,
        width as u32,
        &vec![0.0; ids.len() * width],
    );
    if !nat.run(|ctx| layout::embed_gather_mb_4bit(ctx, it, bank, vocab as u32, y)) {
        return;
    }
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&i| {
            (0..width).map(move |c| {
                if (i as usize) < vocab {
                    i as usize * width + c
                } else {
                    usize::MAX
                }
            })
        })
        .map(|at| {
            if at == usize::MAX {
                0.0
            } else {
                round_bf16(table.w[at])
            }
        })
        .collect();
    assert_close(&nat.read(y), &want, 1e-2, 1e-2);
}

/// `y[p] = W[routes[p]] · x_row(p) (+ bias)`; unrouted rows keep `prev`.
#[allow(clippy::too_many_arguments)]
fn select_ref(
    w: &[f32],
    e: usize,
    n: usize,
    k: usize,
    x: &[f32],
    routes: &[i32],
    top_k: usize,
    bias: Option<&[f32]>,
    prev: f32,
) -> Vec<f32> {
    let mut y = vec![prev; routes.len() * n];
    for (p, &r) in routes.iter().enumerate() {
        if r < 0 || r as usize >= e {
            continue;
        }
        let r = r as usize;
        for c in 0..n {
            let mut acc: f32 = (0..k)
                .map(|j| w[(r * n + c) * k + j] * x[(p / top_k) * k + j])
                .sum();
            if let Some(b) = bias {
                acc += b[r * n + c];
            }
            y[p * n + c] = round_bf16(acc);
        }
    }
    y
}

#[test]
fn routed_banks_read_native_codes_and_prescaled_weights_every_way() {
    let (e, n, k, t, tk) = (5usize, 24usize, 128usize, 3usize, 2usize);
    let banks = [
        (affine(Dtype::U4g64, e * n, k, 50), false),
        (affine(Dtype::U2g32, e * n, k, 51), false),
        (mxfp4(e * n, k, 52), false),
        (mxfp4(e * n, k, 53), true),
    ];
    let x = data(t * k, 54);
    let bias = data(e * n, 55);
    let mut routes: Vec<i32> = (0..t * tk)
        .map(|i| (hash(i, 56) % e as u32) as i32)
        .collect();
    routes[1] = -1;
    for path in ["all", "slice", "ragged"] {
        // SAFETY: only this test in this binary reads the variable.
        unsafe { std::env::set_var("PIE_XLA_MOE_PATH", path) };
        let mut nat = Native::default();
        let xt = nat.bf16(t as u32, k as u32, &x);
        let rt = nat.add(
            Dtype::I32,
            t as u32,
            tk as u32,
            routes.iter().flat_map(|r| r.to_le_bytes()).collect(),
        );
        let bt = nat.bf16(e as u32, n as u32, &bias);
        let mut outs = Vec::new();
        for (bank, pre) in &banks {
            let landed = bank.land(&mut nat, e as u32, *pre);
            let y = nat.bf16((t * tk) as u32, n as u32, &vec![7.0; t * tk * n]);
            outs.push((landed, y));
        }
        let o = outs.clone();
        let ran = nat.run(|ctx| {
            for (i, (bank, y)) in o.iter().enumerate() {
                if i % 2 == 0 {
                    moe::matmul_select_quant(ctx, xt, *bank, rt, *y)?;
                } else {
                    moe::matmul_select_bias(ctx, xt, *bank, bt, rt, *y)?;
                }
            }
            Ok(())
        });
        unsafe { std::env::remove_var("PIE_XLA_MOE_PATH") };
        if !ran {
            return;
        }
        for (i, ((bank, _), (_, y))) in banks.iter().zip(&outs).enumerate() {
            let b = (i % 2 == 1).then_some(&bias[..]);
            let want = select_ref(&bank.w, e, n, k, &x, &routes, tk, b, 7.0);
            check(&nat.read(*y), &want);
        }
    }
}
