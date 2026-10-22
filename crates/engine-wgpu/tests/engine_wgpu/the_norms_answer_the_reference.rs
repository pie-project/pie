//! The norm kernels on the first wgpu adapter (lavapipe through wgpu's Vulkan
//! backend will do) against an f64 host reference: each variant the IR's
//! `elemwise.rmsnorm` folds, over rows wider and narrower than a workgroup.

use engine_wgpu::api::DeviceBoot;
use engine_wgpu::{Buffer, Context, Handles, Pipelines, Sink, Slang};
use kernels_wgpu::Tensor;
use kernels_wgpu::elemwise::norm;
use poem_ir::Dtype;

fn bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn widen(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// A small deterministic stream in `[-2, 2)`.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

struct Gpu {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    held: Vec<Buffer>,
}

impl Gpu {
    fn open() -> Option<Gpu> {
        let device = Context::bind(&DeviceBoot::default(), 0.25).ok()?;
        Some(Gpu {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            held: Vec::new(),
        })
    }

    /// `values` as bf16 rows; returns the buffer's handle and the values the
    /// device holds (rounded).
    fn up(&mut self, values: &[f32]) -> (u32, Vec<f32>) {
        let halves: Vec<u16> = values.iter().map(|v| bf16(*v)).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        (handle, halves.into_iter().map(widen).collect())
    }

    /// `values` as f32 rows.
    fn up_f32(&mut self, values: &[f32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    fn zeros(&mut self, n: usize) -> u32 {
        let buf = Buffer::zeroed(&self.device, (n * 2) as u64).expect("a buffer");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    fn run(&self, f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_wgpu::Error>) {
        let frame = self.device.frame().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&sink).expect("the kernel fires");
        }
        frame.commit().expect("the frame lands");
    }

    fn run_slang(&self, f: impl FnOnce(&Slang<'_>) -> Result<(), kernels_slang::Error>) {
        let frame = self.device.frame().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&Slang(&sink)).expect("the kernel fires");
        }
        frame.commit().expect("the frame lands");
    }

    fn down(&self, handle: u32, n: usize) -> Vec<f32> {
        let bytes = self
            .handles
            .read(handle, (n * 2) as u64)
            .expect("read back");
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| widen(u16::from_le_bytes(*c)))
            .collect()
    }
    fn down_f32(&self, handle: u32, n: usize) -> Vec<f32> {
        let bytes = self
            .handles
            .read(handle, (n * 4) as u64)
            .expect("read back");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

/// `x / rms(x)` over each `run`-wide slice, times `scale(i)` at column `i`
/// of the row.
fn reference(
    x: &[f32],
    width: usize,
    run: usize,
    eps: f32,
    scale: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (r, row) in x.chunks_exact(width).enumerate() {
        for (g, slice) in row.chunks_exact(run).enumerate() {
            let ms = slice.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / run as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            for (i, v) in slice.iter().enumerate() {
                let col = g * run + i;
                out[r * width + col] = f64::from(*v) * inv * scale(col);
            }
        }
    }
    out
}

/// A slang case: its name, the run it norms over, the scale at a column.
type Case<'a> = (&'a str, usize, Box<dyn Fn(usize) -> f64 + 'a>);

fn close(got: &[f32], want: &[f64], what: &str) {
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        // bf16 output: half an ulp of the value, plus a floor near zero.
        let tol = w.abs() * 2f64.powi(-8) + 1e-3;
        let err = (f64::from(*g) - w).abs();
        assert!(
            err <= tol * 2.0,
            "{what}: got {g}, want {w} (err {err}, tol {tol})"
        );
        worst = worst.max(err / tol);
    }
    eprintln!("{what}: worst {worst:.3} of tolerance");
}

#[test]
fn the_norms_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no wgpu adapter");
        return;
    };
    let eps = 1e-6;
    for (rows, width, head) in [(3usize, 64usize, 16usize), (5, 4096, 128), (2, 2560, 256)] {
        let n = rows * width;
        let (x, xv) = gpu.up(&noise(rows as u64 * 31 + width as u64, n));
        let (w, wv) = gpu.up(&noise(width as u64, width));
        let (hw, hwv) = gpu.up(&noise(head as u64 + 7, head));
        let t = |buf: u32, width: usize| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        let shape = format!("{rows} x {width}");

        let y = gpu.zeros(n);
        gpu.run(|s| {
            norm::rmsnorm(
                s,
                t(x, width),
                Tensor::new(w, 1, width as u32, Dtype::Bf16),
                eps,
                t(y, width),
            )
        });
        close(
            &gpu.down(y, n),
            &reference(&xv, width, width, eps, |c| f64::from(wv[c])),
            &format!("rmsnorm {shape}"),
        );

        let y = gpu.zeros(n);
        gpu.run(|s| {
            norm::rmsnorm_plus_one(
                s,
                t(x, width),
                Tensor::new(w, 1, width as u32, Dtype::Bf16),
                eps,
                t(y, width),
            )
        });
        close(
            &gpu.down(y, n),
            &reference(&xv, width, width, eps, |c| 1.0 + f64::from(wv[c])),
            &format!("rmsnorm +1 {shape}"),
        );

        let y = gpu.zeros(n);
        gpu.run(|s| {
            norm::rmsnorm_per_head(
                s,
                t(x, width),
                Tensor::new(hw, 1, head as u32, Dtype::Bf16),
                head as u32,
                eps,
                t(y, width),
            )
        });
        close(
            &gpu.down(y, n),
            &reference(&xv, width, head, eps, |c| f64::from(hwv[c % head])),
            &format!("rmsnorm per head {shape}"),
        );

        let y = gpu.zeros(n);
        gpu.run(|s| {
            norm::rmsnorm_per_head_plus_one(
                s,
                t(x, width),
                Tensor::new(hw, 1, head as u32, Dtype::Bf16),
                head as u32,
                eps,
                t(y, width),
            )
        });
        close(
            &gpu.down(y, n),
            &reference(&xv, width, head, eps, |c| 1.0 + f64::from(hwv[c % head])),
            &format!("rmsnorm per head +1 {shape}"),
        );

        let y = gpu.zeros(n);
        gpu.run(|s| norm::rmsnorm_no_scale(s, t(x, width), head as u32, eps, t(y, width)));
        close(
            &gpu.down(y, n),
            &reference(&xv, width, head, eps, |_| 1.0),
            &format!("rmsnorm no scale {shape}"),
        );

        let y = gpu.zeros(n);
        gpu.run(|s| {
            norm::rmsnorm_grouped_plus_one(
                s,
                t(x, width),
                Tensor::new(w, 1, width as u32, Dtype::Bf16),
                head as u32,
                eps,
                t(y, width),
            )
        });
        close(
            &gpu.down(y, n),
            &reference(&xv, width, head, eps, |c| 1.0 + f64::from(wv[c])),
            &format!("rmsnorm grouped +1 {shape}"),
        );

        // The same norms from the one Slang tree (kernels-slang), on the
        // same device.
        use kernels_slang::elemwise::norm as slang;
        let st = |buf: u32, rows: usize, width: usize| {
            kernels_slang::Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16)
        };
        let cases: [Case<'_>; 6] = [
            ("rmsnorm", width, Box::new(|c| f64::from(wv[c]))),
            ("rmsnorm +1", width, Box::new(|c| 1.0 + f64::from(wv[c]))),
            (
                "rmsnorm per head",
                head,
                Box::new(|c| f64::from(hwv[c % head])),
            ),
            (
                "rmsnorm per head +1",
                head,
                Box::new(|c| 1.0 + f64::from(hwv[c % head])),
            ),
            (
                "rmsnorm grouped +1",
                head,
                Box::new(|c| 1.0 + f64::from(wv[c])),
            ),
            ("rmsnorm no scale", head, Box::new(|_| 1.0)),
        ];
        for (i, (what, run, scale)) in cases.iter().enumerate() {
            let y = gpu.zeros(n);
            let (xs, ys) = (st(x, rows, width), st(y, rows, width));
            gpu.run_slang(|s| match i {
                0 => slang::rmsnorm(s, xs, st(w, 1, width), eps, ys),
                1 => slang::rmsnorm_plus_one(s, xs, st(w, 1, width), eps, ys),
                2 => slang::rmsnorm_per_head(s, xs, st(hw, 1, head), head as u32, eps, ys),
                3 => slang::rmsnorm_per_head_plus_one(s, xs, st(hw, 1, head), head as u32, eps, ys),
                4 => slang::rmsnorm_grouped_plus_one(s, xs, st(w, 1, width), head as u32, eps, ys),
                _ => slang::rmsnorm_no_scale(s, xs, head as u32, eps, ys),
            });
            close(
                &gpu.down(y, n),
                &reference(&xv, width, *run, eps, scale),
                &format!("slang {what} {shape}"),
            );
        }
    }
}

/// `y = x / rms(x) * w * gate(z)` over each `vd`-wide head.
fn gated_reference(
    x: &[f32],
    z: &[f32],
    w: &[f32],
    vd: usize,
    eps: f32,
    sigmoid: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (h, head) in x.chunks_exact(vd).enumerate() {
        let ms = head.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / vd as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        for (i, v) in head.iter().enumerate() {
            let zr = f64::from(z[h * vd + i]);
            let sig = 1.0 / (1.0 + (-zr).exp());
            let gate = if sigmoid { sig } else { zr * sig };
            out[h * vd + i] = f64::from(*v) * inv * f64::from(w[i]) * gate;
        }
    }
    out
}

#[test]
fn the_gated_norms_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let eps = 1e-6;
    for (rows, heads, vd) in [(3usize, 2usize, 64usize), (4, 8, 128)] {
        let width = heads * vd;
        let n = rows * width;
        let xv = noise(rows as u64 * 17 + vd as u64, n);
        let x = gpu.up_f32(&xv);
        let (z, zv) = gpu.up(&noise(vd as u64 + 3, n));
        let wv = noise(vd as u64 + 5, vd);
        let w = gpu.up_f32(&wv);
        let f32t = |buf: u32, width: usize| Tensor::new(buf, rows as u32, width as u32, Dtype::F32);
        let bt = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
        let shape = format!("{rows} x {heads} x {vd}");
        for sigmoid in [false, true] {
            let want = gated_reference(&xv, &zv, &wv, vd, eps, sigmoid);
            let wt = Tensor::new(w, 1, vd as u32, Dtype::F32);
            let y = gpu.zeros(n);
            gpu.run(|s| {
                norm::rmsnorm_gated(s, f32t(x, width), bt(z), wt, vd as u32, eps, sigmoid, bt(y))
            });
            close(
                &gpu.down(y, n),
                &want,
                &format!("gated rms sigmoid={sigmoid} {shape}"),
            );
            let y = gpu.zeros(n);
            gpu.run_slang(|s| {
                kernels_slang::elemwise::gated_rms::rmsnorm_gated(
                    s,
                    st(f32t(x, width)),
                    st(bt(z)),
                    st(wt),
                    vd as u32,
                    eps,
                    sigmoid,
                    st(bt(y)),
                )
            });
            close(
                &gpu.down(y, n),
                &want,
                &format!("slang gated rms sigmoid={sigmoid} {shape}"),
            );
        }
        let want = gated_reference(&xv, &zv, &wv, vd, eps, true);
        let y = gpu.zeros(n);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::gated_rms::rmsnorm_gated_by(
                s,
                st(f32t(x, width)),
                st(bt(z)),
                st(Tensor::new(w, 1, vd as u32, Dtype::F32)),
                heads as u32,
                eps,
                st(bt(y)),
            )
        });
        close(
            &gpu.down(y, n),
            &want,
            &format!("slang gated rms by heads {shape}"),
        );
    }
}

/// The adds a norm leans on: `y += x` over a tensor and `out[r, c] +=
/// bias[c]`, bf16 and f32 (odd-sized f32, the pair's tail half there).
#[test]
fn the_sums_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    for (rows, width) in [(3usize, 64usize), (5, 1000)] {
        let n = rows * width;
        let (xv, yv) = (noise(width as u64, n), noise(width as u64 + 1, n));
        let bv = noise(width as u64 + 2, width);
        let bt = |buf: u32, rows: usize, width: usize| {
            Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16)
        };
        let (x, xr) = gpu.up(&xv);
        let (_, yr) = gpu.up(&yv);
        let (_, br) = gpu.up(&bv);
        let sum: Vec<f64> = xr
            .iter()
            .zip(&yr)
            .map(|(a, b)| f64::from(*a) + f64::from(*b))
            .collect();
        let biased: Vec<f64> = (0..n)
            .map(|i| f64::from(yr[i]) + f64::from(br[i % width]))
            .collect();
        let shape = format!("{rows} x {width}");
        for slang in [false, true] {
            let who = if slang { "slang " } else { "" };
            let (y, _) = gpu.up(&yv);
            let (yt, xt) = (bt(y, rows, width), bt(x, rows, width));
            if slang {
                gpu.run_slang(|s| {
                    kernels_slang::elemwise::residual_add::residual_add(s, st(xt), st(yt))
                });
            } else {
                gpu.run(|s| norm::residual_add(s, xt, yt));
            }
            close(
                &gpu.down(y, n),
                &sum,
                &format!("{who}residual add bf16 {shape}"),
            );
            let (y, _) = gpu.up(&yv);
            let (b, _) = gpu.up(&bv);
            let (yt, b) = (bt(y, rows, width), bt(b, 1, width));
            if slang {
                gpu.run_slang(|s| kernels_slang::elemwise::add_bias::add_bias(s, st(b), st(yt)));
            } else {
                gpu.run(|s| norm::add_bias(s, b, yt));
            }
            close(
                &gpu.down(y, n),
                &biased,
                &format!("{who}add bias bf16 {shape}"),
            );
        }
    }
    for (rows, width) in [(3usize, 7usize), (5, 1001)] {
        let n = rows * width;
        let (xv, yv) = (noise(width as u64 + 3, n), noise(width as u64 + 4, n));
        let bv = noise(width as u64 + 5, width);
        let ft = |buf: u32, rows: usize, width: usize| {
            Tensor::new(buf, rows as u32, width as u32, Dtype::F32)
        };
        let shape = format!("{rows} x {width}");
        let x = gpu.up_f32(&xv);
        let y = gpu.up_f32(&yv);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::residual_add::residual_add(
                s,
                st(ft(x, rows, width)),
                st(ft(y, rows, width)),
            )
        });
        let want: Vec<f64> = xv
            .iter()
            .zip(&yv)
            .map(|(a, b)| f64::from(*a) + f64::from(*b))
            .collect();
        close(
            &gpu.down_f32(y, n),
            &want,
            &format!("slang residual add f32 {shape}"),
        );
        let want: Vec<f64> = (0..n)
            .map(|i| f64::from(yv[i]) + f64::from(bv[i % width]))
            .collect();
        let y = gpu.up_f32(&yv);
        let b = gpu.up_f32(&bv);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::add_bias::add_bias(
                s,
                st(ft(b, 1, width)),
                st(ft(y, rows, width)),
            )
        });
        close(
            &gpu.down_f32(y, n),
            &want,
            &format!("slang add bias f32 {shape}"),
        );
        if width % 2 == 0 {
            continue;
        }
        // A bf16 bias under f32 rows; the bias is read an element at a time,
        // so it may be odd wide.
        let (b, br) = gpu.up(&bv);
        let want: Vec<f64> = (0..n)
            .map(|i| f64::from(yv[i]) + f64::from(br[i % width]))
            .collect();
        let y = gpu.up_f32(&yv);
        let bias = Tensor::new(b, 1, width as u32, Dtype::Bf16);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::add_bias::add_bias(s, st(bias), st(ft(y, rows, width)))
        });
        close(
            &gpu.down_f32(y, n),
            &want,
            &format!("slang add bf16 bias to f32 {shape}"),
        );
    }
}

/// `(x - mean) / sqrt(var + eps)` over each row, times `w` plus `b` when given.
fn layernorm_reference(
    x: &[f32],
    width: usize,
    eps: f32,
    affine: Option<(&[f32], &[f32])>,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / width as f64;
        let var = row
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / width as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for (c, v) in row.iter().enumerate() {
            let n = (f64::from(*v) - mean) * inv;
            out.push(match affine {
                Some((w, b)) => n * f64::from(w[c]) + f64::from(b[c]),
                None => n,
            });
        }
    }
    out
}

#[test]
fn the_layernorms_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let eps = 1e-5;
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    for (rows, width) in [(3usize, 64usize), (5, 1000), (2, 4096)] {
        let n = rows * width;
        let bt = |buf: u32, rows: usize| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        // Off-centre rows, so a missed mean shows.
        let xv: Vec<f32> = noise(width as u64 + 7, n).iter().map(|v| v + 1.5).collect();
        let (x, xr) = gpu.up(&xv);
        let (w, wr) = gpu.up(&noise(width as u64 + 8, width));
        let (b, br) = gpu.up(&noise(width as u64 + 9, width));
        let shape = format!("{rows} x {width}");
        let want = layernorm_reference(&xr, width, eps, Some((&wr, &br)));
        let y = gpu.zeros(n);
        gpu.run(|s| norm::layernorm(s, bt(x, rows), bt(w, 1), bt(b, 1), eps, bt(y, rows)));
        close(&gpu.down(y, n), &want, &format!("layernorm {shape}"));
        let y = gpu.zeros(n);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::layernorm::layernorm(
                s,
                st(bt(x, rows)),
                st(bt(w, 1)),
                st(bt(b, 1)),
                eps,
                st(bt(y, rows)),
            )
        });
        close(&gpu.down(y, n), &want, &format!("slang layernorm {shape}"));
        let want = layernorm_reference(&xr, width, eps, None);
        let y = gpu.zeros(n);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::layernorm::layernorm_no_scale(
                s,
                st(bt(x, rows)),
                eps,
                st(bt(y, rows)),
            )
        });
        close(
            &gpu.down(y, n),
            &want,
            &format!("slang layernorm no scale {shape}"),
        );
    }
    for (rows, width) in [(3usize, 7usize), (4, 1001)] {
        let n = rows * width;
        let ft = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::F32);
        let xv: Vec<f32> = noise(width as u64 + 10, n)
            .iter()
            .map(|v| v - 0.75)
            .collect();
        let x = gpu.up_f32(&xv);
        let y = gpu.zeros(2 * n);
        gpu.run_slang(|s| {
            kernels_slang::elemwise::layernorm::layernorm_no_scale(s, st(ft(x)), eps, st(ft(y)))
        });
        let want = layernorm_reference(&xv, width, eps, None);
        close(
            &gpu.down_f32(y, n),
            &want,
            &format!("slang layernorm no scale f32 {rows} x {width}"),
        );
    }
}

#[test]
fn standardize_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    for (rows, width) in [(3usize, 64usize), (5, 1000)] {
        let n = rows * width;
        let bt = |buf: u32, rows: usize| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        let xv = noise(width as u64 + 11, n);
        let (_, xr) = gpu.up(&xv);
        let (b, br) = gpu.up(&noise(width as u64 + 12, width));
        let (s, sr) = gpu.up(&noise(width as u64 + 13, width));
        let want: Vec<f64> = (0..n)
            .map(|i| (f64::from(xr[i]) - f64::from(br[i % width])) * f64::from(sr[i % width]))
            .collect();
        let shape = format!("{rows} x {width}");
        let (y, _) = gpu.up(&xv);
        gpu.run(|k| norm::standardize(k, bt(b, 1), bt(s, 1), bt(y, rows)));
        close(&gpu.down(y, n), &want, &format!("standardize {shape}"));
        let (y, _) = gpu.up(&xv);
        gpu.run_slang(|k| {
            kernels_slang::elemwise::standardize::standardize(
                k,
                st(bt(b, 1)),
                st(bt(s, 1)),
                st(bt(y, rows)),
            )
        });
        close(
            &gpu.down(y, n),
            &want,
            &format!("slang standardize {shape}"),
        );
    }
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// The in-place scalar forms, `x *= s` (stated or read) and `x = silu(x * s)`,
/// over even and odd counts: an odd bf16 tensor ends mid-word, and the half
/// past it must come back untouched.
#[test]
fn the_scalars_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let s = 0.3f32;
    let sb = f64::from(widen(bf16(s)));
    for (rows, width) in [(3usize, 64usize), (5, 1000), (3, 7)] {
        let n = rows * width;
        let bt = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        // One element past the tensor, a sentinel the kernels leave alone.
        let mut xv = noise(width as u64 + 14, n);
        xv.push(7.0);
        let (sbuf, sr) = gpu.up(&[-1.25]);
        let (_, xr) = gpu.up(&xv);
        let mut stated: Vec<f64> = xr.iter().map(|v| f64::from(*v) * sb).collect();
        let mut read: Vec<f64> = xr
            .iter()
            .map(|v| f64::from(*v) * f64::from(sr[0]))
            .collect();
        let mut silued: Vec<f64> = xr
            .iter()
            .map(|v| silu(f64::from(*v) * f64::from(s)))
            .collect();
        for want in [&mut stated, &mut read, &mut silued] {
            want[n] = 7.0;
        }
        let shape = format!("{rows} x {width}");
        let even = n % 2 == 0;
        for slang in [false, true] {
            if !slang && !even {
                continue;
            }
            let who = if slang { "slang " } else { "" };
            let (x, _) = gpu.up(&xv);
            if slang {
                gpu.run_slang(|k| kernels_slang::elemwise::scale::mul_scalar(k, s, st(bt(x))));
            } else {
                gpu.run(|k| norm::mul_scalar(k, s, bt(x)));
            }
            close(
                &gpu.down(x, n + 1),
                &stated,
                &format!("{who}mul scalar bf16 {shape}"),
            );
            let (x, _) = gpu.up(&xv);
            let sv = Tensor::new(sbuf, 1, 1, Dtype::Bf16);
            if slang {
                gpu.run_slang(|k| kernels_slang::elemwise::scale::scale(k, st(sv), st(bt(x))));
            } else {
                gpu.run(|k| norm::scale(k, sv, bt(x)));
            }
            close(
                &gpu.down(x, n + 1),
                &read,
                &format!("{who}scale bf16 {shape}"),
            );
            let (x, _) = gpu.up(&xv);
            if slang {
                gpu.run_slang(|k| {
                    kernels_slang::elemwise::silu_scaled::silu_scaled(k, s, st(bt(x)))
                });
            } else {
                gpu.run(|k| norm::silu_scaled(k, s, bt(x)));
            }
            close(
                &gpu.down(x, n + 1),
                &silued,
                &format!("{who}silu scaled bf16 {shape}"),
            );
        }
        if !even {
            let (y, yr) = gpu.up(&noise(width as u64 + 15, n + 1));
            let mut want: Vec<f64> = xr
                .iter()
                .zip(&yr)
                .map(|(a, b)| f64::from(*a) + f64::from(*b))
                .collect();
            want[n] = f64::from(yr[n]);
            let (x, _) = gpu.up(&xv);
            gpu.run_slang(|k| {
                kernels_slang::elemwise::residual_add::residual_add(k, st(bt(x)), st(bt(y)))
            });
            close(
                &gpu.down(y, n + 1),
                &want,
                &format!("slang residual add bf16 {shape}"),
            );
        }
        let ft = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::F32);
        let x = gpu.up_f32(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::scale::mul_scalar(k, s, st(ft(x))));
        let mut want: Vec<f64> = xv.iter().map(|v| f64::from(*v) * f64::from(s)).collect();
        want[n] = 7.0;
        close(
            &gpu.down_f32(x, n + 1),
            &want,
            &format!("slang mul scalar f32 {shape}"),
        );
        let x = gpu.up_f32(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::silu_scaled::silu_scaled(k, s, st(ft(x))));
        let mut want: Vec<f64> = xv
            .iter()
            .map(|v| silu(f64::from(*v) * f64::from(s)))
            .collect();
        want[n] = 7.0;
        close(
            &gpu.down_f32(x, n + 1),
            &want,
            &format!("slang silu scaled f32 {shape}"),
        );
    }
}

/// Each row the softmax-weighted sum of its candidates (the stacked blocks,
/// then the prefix), each candidate's logit `rms_norm(c) * w . proj`.
fn blend_reference(
    prefix: &[f32],
    stack: &[f32],
    n_blocks: usize,
    (w, proj): (&[f32], &[f32]),
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * hidden);
    for r in 0..rows {
        let cands: Vec<&[f32]> = (0..n_blocks)
            .map(|j| &stack[(j * rows + r) * hidden..][..hidden])
            .chain(std::iter::once(&prefix[r * hidden..][..hidden]))
            .collect();
        let logits: Vec<f64> = cands
            .iter()
            .map(|c| {
                let ms = c.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / hidden as f64;
                let inv = 1.0 / (ms + f64::from(eps)).sqrt();
                (0..hidden)
                    .map(|h| f64::from(c[h]) * inv * f64::from(w[h]) * f64::from(proj[h]))
                    .sum()
            })
            .collect();
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        for h in 0..hidden {
            out.push(
                cands
                    .iter()
                    .zip(&e)
                    .map(|(c, p)| p / sum * f64::from(c[h]))
                    .sum(),
            );
        }
    }
    out
}

#[test]
fn res_blend_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let eps = 1e-6;
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    for (rows, hidden, n_blocks) in [(3usize, 64usize, 0usize), (2, 1000, 3), (4, 512, 5)] {
        let n = rows * hidden;
        let t = |buf: u32, rows: usize| Tensor::new(buf, rows as u32, hidden as u32, Dtype::Bf16);
        let (prefix, pr) = gpu.up(&noise(hidden as u64 + 20, n));
        let (stack, sr) = gpu.up(&noise(
            hidden as u64 + 21,
            (n_blocks * rows).max(1) * hidden,
        ));
        // A small projection keeps the logits apart but not one-hot.
        let pv: Vec<f32> = noise(hidden as u64 + 23, hidden)
            .iter()
            .map(|v| v * 0.05)
            .collect();
        let (w, wr) = gpu.up(&noise(hidden as u64 + 22, hidden));
        let (proj, projr) = gpu.up(&pv);
        let want = blend_reference(&pr, &sr, n_blocks, (&wr, &projr), rows, hidden, eps);
        let blocks = vec![t(stack, rows); n_blocks];
        let shape = format!("{rows} x {hidden}, {n_blocks} blocks");
        let y = gpu.zeros(n);
        gpu.run(|k| {
            norm::res_blend(
                k,
                t(prefix, rows),
                &blocks,
                t(w, 1),
                eps,
                t(proj, 1),
                t(y, rows),
            )
        });
        close(&gpu.down(y, n), &want, &format!("res blend {shape}"));
        let y = gpu.zeros(n);
        let sblocks: Vec<_> = blocks.iter().map(|b| st(*b)).collect();
        gpu.run_slang(|k| {
            kernels_slang::elemwise::res_blend::res_blend(
                k,
                st(t(prefix, rows)),
                &sblocks,
                st(t(w, 1)),
                eps,
                st(t(proj, 1)),
                st(t(y, rows)),
            )
        });
        close(&gpu.down(y, n), &want, &format!("slang res blend {shape}"));
    }
}
