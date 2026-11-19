//! The device the kernel tests fire on, the first device (a software
//! one such as lavapipe will do): rows up and down as bf16 or f32, a frame
//! the kernels fire into, and the comparison with an f64 host reference
//! every test shares.

use crate::backend::{ALLOWANCE, Buffer, Context, DeviceBoot, Handles, Pipelines, Sink};

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

/// The boot every suite load takes: the backend's default with a profiles
/// directory, so a load reads the device's measured `[rates]` (and keeps
/// what it measures) instead of the checked-in profiles alone, which leave
/// an unprofiled device (a phone's GPU) with no roofline rates. The
/// directory is `$PIE_HOME/profiles` where `PIE_HOME` is set, as `pie`'s
/// own loads use, and otherwise one directory per backend in the temp dir,
/// shared by every test of the run.
pub fn device_boot() -> DeviceBoot {
    let profiles = std::env::var_os("PIE_HOME").map_or_else(
        || std::env::temp_dir().join(format!("pie-{}-profiles", crate::backend::NAME)),
        |home| std::path::PathBuf::from(home).join("profiles"),
    );
    DeviceBoot {
        profiles: Some(profiles),
        ..DeviceBoot::default()
    }
}

/// How many parts the harness cuts the device into. A test that loads a
/// model takes one part, its share ([`share`]), and every load it makes
/// takes no more than that; one part is left to the kernel tests (a few
/// small buffers each) and the drivers. So at most `SHARES - 1` tests hold
/// a model at once, whatever the harness's thread count, and their loads
/// never collide: each sizes its pool within its share, not within what the
/// device has left when it loads (which a load beside it then takes).
pub const SHARES: u64 = 4;

/// The tests holding a share now, and the signal one gave its share back.
static HOLDERS: std::sync::Mutex<u64> = std::sync::Mutex::new(0);
static FREED: std::sync::Condvar = std::sync::Condvar::new();

/// A test's share, held by its thread until the test ends.
struct Held;

impl Drop for Held {
    fn drop(&mut self) {
        *HOLDERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) -= 1;
        FREED.notify_one();
    }
}

thread_local! {
    static HELD: std::cell::RefCell<Option<Held>> = const { std::cell::RefCell::new(None) };
}

/// The bytes of one share: a [`SHARES`]th of what the device states it has
/// to a load that states no budget (read once, as the first share is
/// taken, before any model stands), no more than the harness states a
/// test's load takes (`ALLOWANCE`); `None` where neither is stated.
fn share_bytes() -> Option<u64> {
    static BYTES: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let read = Context::bind(&device_boot(), engine::Allowance::WHOLE)
            .ok()
            .map(|device| engine_gpu::backend::Context::capacity(&device) / SHARES);
        match (read, ALLOWANCE.budget) {
            (Some(read), Some(stated)) => Some(read.min(stated)),
            (read, stated) => read.or(stated),
        }
    })
}

/// The device allowance of a load of the calling test: its share of the
/// device. The test takes its share (waiting while `SHARES - 1` other tests
/// hold theirs) where it holds none yet, and holds it until it ends; every
/// load it makes then takes the same share. Take it before
/// `model_logits::loading`, which a test holding a share waits on.
pub fn share() -> engine::Allowance {
    HELD.with(|held| {
        let mut held = held.borrow_mut();
        if held.is_none() {
            let mut holders = HOLDERS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while *holders >= SHARES - 1 {
                holders = FREED
                    .wait(holders)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            *holders += 1;
            *held = Some(Held);
        }
    });
    engine::Allowance {
        budget: share_bytes(),
    }
}

pub struct Gpu {
    pub device: Context,
    handles: Handles,
    pipelines: Pipelines,
    held: Vec<Buffer>,
}

impl Gpu {
    pub fn open() -> Option<Gpu> {
        let device = Context::bind(&device_boot(), ALLOWANCE).ok()?;
        Some(Gpu {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            held: Vec::new(),
        })
    }

    /// The narrowest subgroup the device runs (0 without subgroups).
    pub fn subgroup(&self) -> u32 {
        crate::backend::subgroup(&self.device)
    }

    /// Whether the device has cooperative matrices.
    pub fn coopmat(&self) -> bool {
        crate::backend::coopmat(&self.device)
    }

    /// `values` as bf16 rows; returns the buffer's handle and the values the
    /// device holds (rounded).
    pub fn up(&mut self, values: &[f32]) -> (u32, Vec<f32>) {
        let halves: Vec<u16> = values.iter().map(|v| bf16(*v)).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(
            &self.device,
            bytes.len() as u64,
            engine_gpu::books::owners::TEST,
        )
        .expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        (handle, halves.into_iter().map(widen).collect())
    }

    /// `values` as f32 rows.
    pub fn up_f32(&mut self, values: &[f32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(
            &self.device,
            bytes.len() as u64,
            engine_gpu::books::owners::TEST,
        )
        .expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    /// `values` as an i32 stream.
    pub fn up_i32(&mut self, values: &[i32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(
            &self.device,
            bytes.len() as u64,
            engine_gpu::books::owners::TEST,
        )
        .expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    /// `values` as raw 32-bit words.
    pub fn up_u32(&mut self, values: &[u32]) -> u32 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = Buffer::zeroed(
            &self.device,
            bytes.len() as u64,
            engine_gpu::books::owners::TEST,
        )
        .expect("a buffer");
        buf.write(0, &bytes).expect("written");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    pub fn zeros(&mut self, n: usize) -> u32 {
        let buf = Buffer::zeroed(
            &self.device,
            (n * 2) as u64,
            engine_gpu::books::owners::TEST,
        )
        .expect("a buffer");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("bound");
        self.held.push(buf);
        handle
    }

    pub fn run(&self, f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_slang::Error>) {
        let frame = self.device.frame().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&sink).expect("the kernel fires");
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

/// The tiers a kernel with a subgroup form is checked at: its subgroup form
/// is SPIR-V's.
pub fn tiers() -> &'static [kernels_slang::Tier] {
    if crate::backend::SPIRV {
        &[kernels_slang::Tier::Baseline, kernels_slang::Tier::Subgroup]
    } else {
        &[kernels_slang::Tier::Baseline]
    }
}

/// Whether an output of `columns` lands: on WGSL every output pair is one
/// bf16 word, so an odd count does not.
pub fn lands(columns: usize) -> bool {
    crate::backend::SPIRV || columns % 2 == 0
}

pub fn close(got: &[f32], want: &[f64], what: &str) {
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        // An exact match, an empty reading's infinite LSE included.
        if f64::from(*g) == *w {
            continue;
        }
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

/// Run `fires` on the outputs `fresh` makes, and compare each landed output
/// with its reference.
pub fn landed(
    gpu: &mut Gpu,
    what: &str,
    fresh: &dyn Fn(&mut Gpu) -> Vec<Out>,
    want: &[Vec<f64>],
    fires: Fires<'_>,
) {
    let outs = fresh(gpu);
    fires(gpu, &outs);
    for ((buf, n, f32), want) in outs.iter().zip(want) {
        let got = if *f32 {
            gpu.down_f32(*buf, *n)
        } else {
            gpu.down(*buf, *n)
        };
        close(&got, want, what);
    }
}
