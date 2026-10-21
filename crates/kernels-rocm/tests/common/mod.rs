//! What every device test shares: a device, a stream, a context over it
//! with a rocBLAS handle, buffers in bf16 / f32, and the host rounding the
//! device does. A test calls `Gpu::open()` and returns early on `None`, so
//! a machine with no HIP device passes vacuously.

#![allow(dead_code)]

use dtype::Dtype;
use kernels_rocm::hip::{self, blas};
use kernels_rocm::{Ctx, KvPool, RecurrentPool, Snapshots, Tensor};

pub struct Gpu {
    pub stream: hip::Stream,
    blas: blas::Handle,
    buffers: std::cell::RefCell<Vec<hip::Buffer>>,
}

impl Gpu {
    pub fn open() -> Option<Self> {
        if hip::device_count().ok()? == 0 {
            eprintln!("no HIP device; skipping");
            return None;
        }
        hip::set_device(0).ok()?;
        Some(Self {
            stream: hip::Stream::new().ok()?,
            blas: blas::Handle::new().ok()?,
            buffers: std::cell::RefCell::new(Vec::new()),
        })
    }

    pub fn ctx(&self) -> Ctx {
        // SAFETY: the stream and handle live as long as `self`, which every
        // test holds past its last fire.
        unsafe { Ctx::on(self.stream.raw()).with_blas(self.blas.raw()) }
    }

    /// Mean device microseconds per call of `f` over `reps` calls, after one
    /// warm-up call (which compiles what it fires).
    pub fn time(&self, reps: u32, mut f: impl FnMut()) -> f64 {
        f();
        let (start, stop) = (hip::Event::new().unwrap(), hip::Event::new().unwrap());
        start.record(&self.stream).unwrap();
        for _ in 0..reps {
            f();
        }
        stop.record(&self.stream).unwrap();
        f64::from(stop.since(&start).unwrap()) * 1e3 / f64::from(reps)
    }

    /// Mean device microseconds per call of `f`, with `reps` calls captured
    /// into one HIP graph and replayed: no host launch cost between them,
    /// as the engine runs a step. The first call (outside the capture)
    /// compiles what it fires.
    pub fn time_graph(&self, reps: u32, mut f: impl FnMut()) -> f64 {
        use kernels_rocm::hip::rt;
        f();
        self.sync();
        let s = self.stream.raw();
        let mut graph: rt::Graph = core::ptr::null_mut();
        let mut exec: rt::GraphExec = core::ptr::null_mut();
        // SAFETY: the stream is live and captured only here; the graph and
        // its instance are destroyed before returning.
        unsafe {
            assert_eq!(
                rt::hipStreamBeginCapture(s, rt::hipStreamCaptureModeThreadLocal),
                0
            );
            for _ in 0..reps {
                f();
            }
            assert_eq!(rt::hipStreamEndCapture(s, &raw mut graph), 0);
            assert_eq!(rt::hipGraphInstantiateWithFlags(&raw mut exec, graph, 0), 0);
            assert_eq!(rt::hipGraphLaunch(exec, s), 0);
        }
        self.sync();
        let (start, stop) = (hip::Event::new().unwrap(), hip::Event::new().unwrap());
        start.record(&self.stream).unwrap();
        // SAFETY: as above.
        unsafe { assert_eq!(rt::hipGraphLaunch(exec, s), 0) };
        stop.record(&self.stream).unwrap();
        let us = f64::from(stop.since(&start).unwrap()) * 1e3 / f64::from(reps);
        // SAFETY: as above.
        unsafe {
            rt::hipGraphExecDestroy(exec);
            rt::hipGraphDestroy(graph);
        }
        us
    }

    pub fn sync(&self) {
        self.stream.synchronize().unwrap();
    }

    fn keep(&self, buffer: hip::Buffer) -> u64 {
        let ptr = buffer.device_ptr();
        self.buffers.borrow_mut().push(buffer);
        ptr
    }

    /// A `[rows, width]` tensor holding `values` rounded to `dtype`.
    pub fn tensor(&self, values: &[f32], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        assert_eq!(values.len(), (rows * width) as usize);
        let ptr = match dtype {
            Dtype::Bf16 => self.keep(
                hip::Buffer::from_slice(&values.iter().map(|&v| bf16(v)).collect::<Vec<_>>())
                    .unwrap(),
            ),
            Dtype::F16 => self.keep(
                hip::Buffer::from_slice(&values.iter().map(|&v| f16(v)).collect::<Vec<_>>())
                    .unwrap(),
            ),
            Dtype::F32 => self.keep(hip::Buffer::from_slice(values).unwrap()),
            other => panic!("the harness uploads bf16, f16 and f32, not {other:?}"),
        };
        Tensor::new(ptr, rows, width, dtype)
    }

    /// A `[rows, width]` tensor of raw host elements tagged `dtype`, its
    /// buffer padded to whole 16 bytes.
    pub fn upload<T: Copy>(&self, host: &[T], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        // SAFETY: `T: Copy` elements read back as their bytes.
        let raw =
            unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), size_of_val(host)) };
        let mut bytes = raw.to_vec();
        bytes.resize(bytes.len().div_ceil(16).max(1) * 16, 0);
        Tensor::new(
            self.keep(hip::Buffer::from_slice(&bytes).unwrap()),
            rows,
            width,
            dtype,
        )
    }

    /// `t`'s bytes as raw `T`s.
    pub fn read_as<T: Copy + Default>(&self, t: &Tensor) -> Vec<T> {
        self.sync();
        let buffers = self.buffers.borrow();
        let held = buffers
            .iter()
            .find(|b| b.device_ptr() == t.ptr)
            .expect("a harness buffer");
        let mut all = held.read::<T>().unwrap();
        all.truncate((t.elements() * u64::from(t.dtype.bytes_ceil())) as usize / size_of::<T>());
        all
    }

    pub fn ints(&self, values: &[i32]) -> Tensor {
        self.ints_shaped(values, 1, values.len() as u32)
    }

    /// One i32 per row, as a position or index stream is shaped.
    pub fn column(&self, values: &[i32]) -> Tensor {
        self.ints_shaped(values, values.len() as u32, 1)
    }

    pub fn ints_shaped(&self, values: &[i32], rows: u32, width: u32) -> Tensor {
        assert_eq!(values.len(), (rows * width) as usize);
        let ptr = self.keep(hip::Buffer::from_slice(values).unwrap());
        Tensor::new(ptr, rows, width, Dtype::I32)
    }

    pub fn zeros(&self, rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let bytes = (rows as usize * width as usize * dtype.bytes_ceil() as usize).max(1);
        let ptr = self.keep(hip::Buffer::from_slice(&vec![0u8; bytes]).unwrap());
        Tensor::new(ptr, rows, width, dtype)
    }

    /// The tensor's elements, widened to f32.
    pub fn read(&self, t: &Tensor) -> Vec<f32> {
        self.sync();
        let n = t.elements() as usize;
        let buffers = self.buffers.borrow();
        let held = buffers
            .iter()
            .find(|b| b.device_ptr() == t.ptr)
            .expect("a harness buffer");
        match t.dtype {
            Dtype::Bf16 => held.read::<u16>().unwrap()[..n]
                .iter()
                .map(|&b| from_bf16(b))
                .collect(),
            Dtype::F16 => held.read::<u16>().unwrap()[..n]
                .iter()
                .map(|&h| from_f16(h))
                .collect(),
            Dtype::F32 => held.read::<f32>().unwrap()[..n].to_vec(),
            other => panic!("the harness reads bf16, f16 and f32, not {other:?}"),
        }
    }
}

/// A pool with every plane absent, for struct update.
pub const KV: KvPool = KvPool {
    keys: Tensor::ABSENT,
    values: Tensor::ABSENT,
    bf16_keys: Tensor::ABSENT,
    bf16_values: Tensor::ABSENT,
    key_scales: Tensor::ABSENT,
    value_scales: Tensor::ABSENT,
    page_indices: Tensor::ABSENT,
    page_indptr: Tensor::ABSENT,
    last_page_lens: Tensor::ABSENT,
    row_valid: Tensor::ABSENT,
    env_min: Tensor::ABSENT,
    env_max: Tensor::ABSENT,
    has_envelopes: false,
    page_size: 0,
    seq_stride: 0,
    head_stride: 0,
    layout: 0,
    scheme_byte: 0,
    block_size: 0,
    max_pages_per_request: 0,
    pages_in_batch: 0,
    live: [0, 0],
};

/// A recurrent pool writing every lane's state, nothing else seated.
pub const STATE: RecurrentPool = RecurrentPool {
    slab: Tensor::ABSENT,
    slot_ids: Tensor::ABSENT,
    slot_stride_elems: 0,
    conv_slab: Tensor::ABSENT,
    conv_stride: 0,
    write_state: true,
    write_state_mask: Tensor::ABSENT,
    commit_len: Tensor::ABSENT,
    begin_at: Tensor::ABSENT,
    fused_decay: false,
    snapshots: Snapshots::NONE,
};

/// f32 → bf16, round to nearest even, as the device stores.
pub fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// `v` as the device holds it in bf16.
pub fn round_bf16(v: f32) -> f32 {
    from_bf16(bf16(v))
}

/// f32 → f16 bits, round to nearest even, inside f16's range.
pub fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a < 6.103_515_6e-5 {
        return sign | (a / 5.960_464_5e-8).round_ties_even() as u16;
    }
    let e = ((b >> 23) & 0xff) + 15 - 127;
    let m = b & 0x7f_ffff;
    let mut r = (e << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

pub fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
    sign * if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

pub fn round_f16(v: f32) -> f32 {
    from_f16(f16(v))
}

/// Deterministic values in [-scale, scale): a test's inputs.
pub fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `n` bf16-exact values in [-scale, scale).
pub fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn gelu_tanh(g: f32) -> f32 {
    0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh())
}

/// The exclusive prefix sum of `lens`, from 0.
pub fn indptr_of(lens: &[u32]) -> Vec<i32> {
    let mut at = vec![0i32];
    for &l in lens {
        at.push(at.last().unwrap() + l as i32);
    }
    at
}

/// The value of OCP e4m3fn `code`.
pub fn e4m3_value(code: u8) -> f32 {
    let (e, m) = (i32::from((code >> 3) & 0xf), f32::from(code & 7));
    let a = if e == 0 {
        m * (-9f32).exp2()
    } else {
        (1.0 + m / 8.0) * ((e - 7) as f32).exp2()
    };
    if code & 0x80 != 0 { -a } else { a }
}

/// The nearest finite e4m3 code, a tie to the even code, saturating at 448
/// (no −0: zero is 0x00).
pub fn e4m3_code(x: f32) -> u8 {
    let a = x.abs();
    let mut best = 0u8;
    for c in 1..=0x7eu8 {
        let (dc, db) = ((e4m3_value(c) - a).abs(), (e4m3_value(best) - a).abs());
        if dc < db || (dc == db && c % 2 == 0) {
            best = c;
        }
    }
    if x < 0.0 && best != 0 {
        best | 0x80
    } else {
        best
    }
}

/// Every `have[i]` is within `ulps` bf16 steps of `want[i]`.
pub fn close_bf16(what: &str, have: &[f32], want: &[f32], ulps: f32) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        let step = (w.abs().max(1e-30)).log2().floor().exp2() / 128.0;
        assert!(
            (h - w).abs() <= ulps * step + 1e-6,
            "{what}[{i}] = {h}, want {w} (±{ulps} bf16 ulp)"
        );
    }
}

/// Every `have[i]` is within `tol` of `want[i]`: for outputs that cancel,
/// where the error is set by the inputs' scale and not the output's.
pub fn close_abs(what: &str, have: &[f32], want: &[f32], tol: f32) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        assert!((h - w).abs() <= tol, "{what}[{i}] = {h}, want {w} (±{tol})");
    }
}
