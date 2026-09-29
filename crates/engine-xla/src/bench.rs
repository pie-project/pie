//! A one-kernel test bench: host arrays in, one kernels-xla entry emitted as
//! its own module, run on the device, host arrays out.
//!
//! Every handle is its own root. A handle a kernel reads before writing is a
//! parameter; every handle it writes is a result, copied back over the host
//! array. Kernel tests compare those against a host reference.
//!
//! The device is opened once per process, under an exclusive `flock` on
//! `$TMPDIR/pie-xla-device.lock`: libtpu admits one process at a time, and
//! test binaries run concurrently.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;
use kernels_xla::hlo::{Func, Ty, Val, bf16_bits};
use kernels_xla::{Cx, Emit, Env, Tensor, elem_of};

use crate::pjrt::{Arg, Client, ElementType};

/// The process's device client, or `None` (tests skip) when no plugin loads.
pub fn client() -> Option<&'static Mutex<Client>> {
    static CLIENT: OnceLock<Option<Mutex<Client>>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let lock = lock_device();
            let api = match crate::pjrt::Api::discover(None) {
                Ok(api) => api,
                Err(e) => {
                    eprintln!("no PJRT plugin, skipping device tests: {e}");
                    return None;
                }
            };
            let client = Client::create(api).expect("the plugin loads but will not open a client");
            // The lock is held for the life of the process.
            std::mem::forget(lock);
            Some(Mutex::new(client))
        })
        .as_ref()
}

/// Takes the process-wide device lock (see the module doc); held while the
/// returned guard lives. Reentrant within a process: a test that holds it
/// and then opens a `Device` with `PIE_XLA_LOCK=1` takes it again instead of
/// waiting on itself (`flock` on a second open file would).
pub fn lock_device() -> DeviceLock {
    let mut held = HELD.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if held.0 == 0 {
        let path = std::env::temp_dir().join("pie-xla-device.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .expect("the device lock file opens");
        // SAFETY: flock on an fd we own.
        #[allow(unsafe_code)]
        let rc = unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_EX) };
        assert_eq!(rc, 0, "flock {}", path.display());
        held.1 = Some(file);
    }
    held.0 += 1;
    DeviceLock(())
}

static HELD: Mutex<(usize, Option<std::fs::File>)> = Mutex::new((0, None));

/// A hold on the device lock; the file lock goes when the last one drops.
#[derive(Debug)]
pub struct DeviceLock(());

impl Drop for DeviceLock {
    fn drop(&mut self) {
        let mut held = HELD.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        held.0 -= 1;
        if held.0 == 0 {
            held.1 = None;
        }
    }
}

/// The PJRT type a plain dtype lands as.
pub fn element_type(dtype: Dtype) -> ElementType {
    match dtype {
        Dtype::F32 => ElementType::F32,
        Dtype::F16 => ElementType::F16,
        Dtype::Bf16 => ElementType::Bf16,
        Dtype::E4m3 => ElementType::F8E4m3fn,
        Dtype::E5m2 => ElementType::F8E5m2,
        Dtype::E8m0 => ElementType::F8E8m0fnu,
        Dtype::I64 => ElementType::S64,
        Dtype::I32 => ElementType::S32,
        Dtype::I16 => ElementType::S16,
        Dtype::I8 => ElementType::S8,
        Dtype::U64 => ElementType::U64,
        Dtype::U32 => ElementType::U32,
        Dtype::U16 => ElementType::U16,
        Dtype::U8 | Dtype::Bool => ElementType::U8,
        other => panic!("{other:?} is packed and lands through its bank's planes"),
    }
}

struct Array {
    dtype: Dtype,
    rows: u32,
    width: u32,
    bytes: Vec<u8>,
}

#[derive(Default)]
pub struct Bench {
    arrays: Vec<Array>,
    /// The module text of the last `run`, for a failing test to print.
    pub last_module: String,
}

impl Bench {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A handle over raw little-endian bytes.
    pub fn raw(&mut self, dtype: Dtype, rows: u32, width: u32, bytes: Vec<u8>) -> Tensor {
        let ty = element_type(dtype);
        assert_eq!(
            bytes.len(),
            ty.bytes(rows as usize * width as usize),
            "{rows}x{width} {dtype:?}"
        );
        self.arrays.push(Array {
            dtype,
            rows,
            width,
            bytes,
        });
        Tensor::new(self.arrays.len() as u32 - 1, rows, width, dtype)
    }

    pub fn bf16(&mut self, rows: u32, width: u32, xs: &[f32]) -> Tensor {
        let bytes = xs.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect();
        self.raw(Dtype::Bf16, rows, width, bytes)
    }

    pub fn f32(&mut self, rows: u32, width: u32, xs: &[f32]) -> Tensor {
        let bytes = xs.iter().flat_map(|&x| x.to_le_bytes()).collect();
        self.raw(Dtype::F32, rows, width, bytes)
    }

    pub fn i32(&mut self, rows: u32, width: u32, xs: &[i32]) -> Tensor {
        let bytes = xs.iter().flat_map(|&x| x.to_le_bytes()).collect();
        self.raw(Dtype::I32, rows, width, bytes)
    }

    pub fn i64(&mut self, rows: u32, width: u32, xs: &[i64]) -> Tensor {
        let bytes = xs.iter().flat_map(|&x| x.to_le_bytes()).collect();
        self.raw(Dtype::I64, rows, width, bytes)
    }

    pub fn u8(&mut self, rows: u32, width: u32, xs: &[u8]) -> Tensor {
        self.raw(Dtype::U8, rows, width, xs.to_vec())
    }

    pub fn u32(&mut self, rows: u32, width: u32, xs: &[u32]) -> Tensor {
        let bytes = xs.iter().flat_map(|&x| x.to_le_bytes()).collect();
        self.raw(Dtype::U32, rows, width, bytes)
    }

    /// Zeroed `[rows, width]` of `dtype`, for an output.
    pub fn zeros(&mut self, dtype: Dtype, rows: u32, width: u32) -> Tensor {
        let n = element_type(dtype).bytes(rows as usize * width as usize);
        self.raw(dtype, rows, width, vec![0; n])
    }

    #[must_use]
    pub fn bytes(&self, t: Tensor) -> &[u8] {
        &self.arrays[t.buf as usize].bytes
    }

    /// `t` read back as f32, whatever float (or int) type it holds.
    #[must_use]
    pub fn read_f32(&self, t: Tensor) -> Vec<f32> {
        let a = &self.arrays[t.buf as usize];
        match a.dtype {
            Dtype::F32 => a
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::Bf16 => a
                .bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            Dtype::I32 => self.read_i32(t).into_iter().map(|x| x as f32).collect(),
            other => panic!("read_f32 of {other:?}"),
        }
    }

    #[must_use]
    pub fn read_i32(&self, t: Tensor) -> Vec<i32> {
        let a = &self.arrays[t.buf as usize];
        assert_eq!(a.dtype, Dtype::I32);
        a.bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Emits `body` once, compiles it, and runs it `iters` times on the
    /// same inputs; the mean seconds per run (`None` without a device). The
    /// host arrays are not updated.
    pub fn time(
        &mut self,
        iters: u32,
        body: impl FnOnce(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<Option<f64>, String> {
        self.time_with(iters, &[], false, body)
    }

    /// `time` as the engine runs a pool: every handle in `in_place` that the
    /// body reads and writes is donated to its result (`input_output_alias`),
    /// so XLA updates it in place, and each run consumes the previous run's
    /// result. The runs are enqueued back to back and waited for once, so
    /// the mean excludes the per-run host round trip.
    pub fn time_in_place(
        &mut self,
        iters: u32,
        in_place: &[Tensor],
        body: impl FnOnce(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<Option<f64>, String> {
        self.time_with(iters, in_place, true, body)
    }

    fn time_with(
        &mut self,
        iters: u32,
        in_place: &[Tensor],
        pipelined: bool,
        body: impl FnOnce(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<Option<f64>, String> {
        let Some(client) = client() else {
            return Ok(None);
        };
        let tracer = Tracer {
            func: RefCell::new(Func::new("main")),
            env: RefCell::new(Roots {
                shapes: self
                    .arrays
                    .iter()
                    .map(|a| (a.dtype, a.rows, a.width))
                    .collect(),
                ..Roots::default()
            }),
        };
        body(&tracer).map_err(|e| e.to_string())?;
        let Tracer { func, env } = tracer;
        let mut func = func.into_inner();
        let env = env.into_inner();
        let written: Vec<u32> = env.written.iter().copied().collect();
        let results: Vec<Val> = written.iter().map(|b| env.current[b]).collect();
        // param index -> result index, for the donated handles.
        let mut donated: Vec<(usize, usize)> = Vec::new();
        for (p, b) in env.params.iter().enumerate() {
            if in_place.iter().any(|t| t.buf == *b)
                && let Some(r) = written.iter().position(|w| w == b)
            {
                func.alias(p, r as u32);
                donated.push((p, r));
            }
        }
        let text = func.module("bench", &results);
        self.last_module = text.clone();
        let client = client.lock().map_err(|e| e.to_string())?;
        let dev = client.devices()[0];
        let exe = client.compile(&text).map_err(|e| e.to_string())?;
        let mut uploads: Vec<Option<crate::pjrt::Buffer>> = Vec::new();
        for &b in &env.params {
            let a = &self.arrays[b as usize];
            uploads.push(Some(
                client
                    .upload(dev, &a.bytes, element_type(a.dtype), &[i64::from(a.rows), i64::from(a.width)])
                    .map_err(|e| e.to_string())?,
            ));
        }
        let mut run = |wait: bool| -> Result<Option<crate::pjrt::Event>, String> {
            let mut args = Vec::with_capacity(uploads.len());
            let mut taken: Vec<Option<crate::pjrt::Buffer>> = Vec::with_capacity(uploads.len());
            for (p, u) in uploads.iter_mut().enumerate() {
                if donated.iter().any(|&(dp, _)| dp == p) {
                    taken.push(u.take());
                } else {
                    taken.push(None);
                }
            }
            for (p, t) in taken.into_iter().enumerate() {
                match t {
                    Some(b) => args.push(Arg::Donate(b)),
                    None => args.push(Arg::Keep(uploads[p].as_ref().expect("a kept param"))),
                }
            }
            let (outs, done) = exe.execute(dev, args).map_err(|e| e.to_string())?;
            let mut outs: Vec<Option<crate::pjrt::Buffer>> = outs.into_iter().map(Some).collect();
            for &(p, r) in &donated {
                uploads[p] = outs[r].take();
            }
            drop(outs);
            if wait {
                done.wait().map_err(|e| e.to_string())?;
                return Ok(None);
            }
            Ok(Some(done))
        };
        run(true)?;
        run(true)?;
        let started = std::time::Instant::now();
        let mut last = None;
        for i in 0..iters {
            last = run(!pipelined || i + 1 == iters)?;
        }
        if let Some(e) = last {
            e.wait().map_err(|e| e.to_string())?;
        }
        Ok(Some(started.elapsed().as_secs_f64() / f64::from(iters.max(1))))
    }

    /// Emits `body` as one module, runs it, and lands every handle it wrote.
    /// `Ok(false)` when no device is present (the test should return).
    pub fn run(
        &mut self,
        body: impl FnOnce(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<bool, String> {
        let Some(client) = client() else {
            return Ok(false);
        };
        let tracer = Tracer {
            func: RefCell::new(Func::new("main")),
            env: RefCell::new(Roots {
                shapes: self
                    .arrays
                    .iter()
                    .map(|a| (a.dtype, a.rows, a.width))
                    .collect(),
                ..Roots::default()
            }),
        };
        body(&tracer).map_err(|e| e.to_string())?;
        let Tracer { func, env } = tracer;
        let func = func.into_inner();
        let env = env.into_inner();
        let written: Vec<u32> = env.written.iter().copied().collect();
        let results: Vec<Val> = written.iter().map(|b| env.current[b]).collect();
        let text = func.module("bench", &results);
        self.last_module = text.clone();

        let client = client.lock().map_err(|e| e.to_string())?;
        let dev = client.devices()[0];
        let exe = client
            .compile(&text)
            .map_err(|e| format!("{e}\n--- module ---\n{text}"))?;
        let mut uploads = Vec::with_capacity(env.params.len());
        for &b in &env.params {
            let a = &self.arrays[b as usize];
            uploads.push(
                client
                    .upload(
                        dev,
                        &a.bytes,
                        element_type(a.dtype),
                        &[i64::from(a.rows), i64::from(a.width)],
                    )
                    .map_err(|e| e.to_string())?,
            );
        }
        let args = uploads.iter().map(Arg::Keep).collect();
        let (outs, done) = exe.execute(dev, args).map_err(|e| e.to_string())?;
        done.wait().map_err(|e| e.to_string())?;
        for (b, out) in written.iter().zip(outs) {
            self.arrays[*b as usize].bytes = out.download().map_err(|e| e.to_string())?;
        }
        Ok(true)
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
        if let Some(&v) = self.current.get(&t.buf) {
            return Ok(v);
        }
        let (dtype, rows, width) = self.shapes[t.buf as usize];
        let elem = elem_of("bench.read", dtype)?;
        let v = f.param(Ty::new(elem, &[i64::from(rows), i64::from(width)]), None);
        self.params.push(t.buf);
        self.current.insert(t.buf, v);
        Ok(v)
    }

    fn write(&mut self, f: &mut Func, t: Tensor, v: Val) -> Result<(), kernels_xla::Error> {
        let (dtype, rows, width) = self.shapes[t.buf as usize];
        let want = Ty::new(elem_of("bench.write", dtype)?, &[i64::from(rows), i64::from(width)]);
        if f.ty(v) != &want {
            return Err(kernels_xla::Error::Backend {
                op: "bench.write",
                detail: format!("wrote {} over a {want} handle", f.ty(v)),
            });
        }
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

/// Host bf16 rounding, for building references that round where the device
/// stores.
#[must_use]
pub fn round_bf16(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(x)) << 16)
}

/// Asserts `got` is within `atol + rtol·|want|` of `want` everywhere.
#[track_caller]
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "lengths differ");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let bound = atol + rtol * w.abs();
        if !(err <= bound) {
            panic!(
                "element {i}: got {g}, want {w} (err {err} > {bound}); first 8 got {:?} want {:?}",
                &got[..got.len().min(8)],
                &want[..want.len().min(8)]
            );
        }
    }
}
