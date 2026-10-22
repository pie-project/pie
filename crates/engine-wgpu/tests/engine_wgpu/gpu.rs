//! The device the kernel tests fire on, the first wgpu adapter (lavapipe
//! through wgpu's Vulkan backend will do): rows up and down as bf16 or f32, a
//! frame for the hand-written kernels or the Slang ones, and the comparison
//! with an f64 host reference every test shares.

use engine_wgpu::api::DeviceBoot;
use engine_wgpu::{Buffer, Context, Handles, Pipelines, Sink, Slang};

pub fn bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn widen(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// A small deterministic stream in `[-2, 2)`.
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
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

pub struct Gpu {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    held: Vec<Buffer>,
}

impl Gpu {
    pub fn open() -> Option<Gpu> {
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
    pub fn up(&mut self, values: &[f32]) -> (u32, Vec<f32>) {
        let halves: Vec<u16> = values.iter().map(|v| bf16(*v)).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        (handle, halves.into_iter().map(widen).collect())
    }

    /// `values` as f32 rows.
    pub fn up_f32(&mut self, values: &[f32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    /// `values` as an i32 stream.
    pub fn up_i32(&mut self, values: &[i32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    /// `values` as raw 32-bit words.
    pub fn up_u32(&mut self, values: &[u32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    pub fn zeros(&mut self, n: usize) -> u32 {
        let buf = Buffer::zeroed(&self.device, (n * 2) as u64).expect("a buffer");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    pub fn run(&self, f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_wgpu::Error>) {
        let frame = self.device.frame().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&sink).expect("the kernel fires");
        }
        frame.commit().expect("the frame lands");
    }

    pub fn run_slang(&self, f: impl FnOnce(&Slang<'_>) -> Result<(), kernels_slang::Error>) {
        let frame = self.device.frame().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&Slang(&sink)).expect("the kernel fires");
        }
        frame.commit().expect("the frame lands");
    }

    pub fn down(&self, handle: u32, n: usize) -> Vec<f32> {
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
    pub fn down_f32(&self, handle: u32, n: usize) -> Vec<f32> {
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

pub fn close(got: &[f32], want: &[f64], what: &str) {
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

/// A landed output: its buffer, its element count, and whether it is f32.
pub type Out = (u32, usize, bool);

/// Fires a kernel into the outputs it is handed.
pub type Fires<'a> = &'a dyn Fn(&Gpu, &[Out]);

/// Run `hand` and `slang` on fresh copies of the outputs `fresh` makes, and
/// compare each landed output with its reference.
pub fn both(
    gpu: &mut Gpu,
    what: &str,
    fresh: &dyn Fn(&mut Gpu) -> Vec<Out>,
    want: &[Vec<f64>],
    hand: Fires<'_>,
    slang: Fires<'_>,
) {
    for (who, run) in [("", hand), ("slang ", slang)] {
        let outs = fresh(gpu);
        run(gpu, &outs);
        for ((buf, n, f32), want) in outs.iter().zip(want) {
            let got = if *f32 {
                gpu.down_f32(*buf, *n)
            } else {
                gpu.down(*buf, *n)
            };
            close(&got, want, &format!("{who}{what}"));
        }
    }
}
