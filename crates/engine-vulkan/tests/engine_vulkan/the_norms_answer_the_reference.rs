//! The norm kernels on the first Vulkan device (a software one such as
//! lavapipe will do) against an f64 host reference: each variant the IR's
//! `elemwise.rmsnorm` folds, over rows wider and narrower than a workgroup.

use engine_vulkan::api::DeviceBoot;
use engine_vulkan::{Buffer, Context, Handles, Pipelines, Sink, Slang};
use kernels_vulkan::Tensor;
use kernels_vulkan::elemwise::norm;
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

    fn run(&self, f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_vulkan::Error>) {
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
        eprintln!("not asked: no Vulkan device");
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
