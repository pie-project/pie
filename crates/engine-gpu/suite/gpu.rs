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
/// directory of this test process's own ([`profiles`]), so a load reads the
/// device's measured `[rates]` (and keeps what it measures) instead of the
/// checked-in profiles alone, which leave an unprofiled device (a phone's
/// GPU) with no roofline rates, and reads nothing an earlier run, or `pie`'s
/// own loads (`$PIE_HOME`), left there.
///
/// Every device a test opens is booted by this, so the calling test is
/// counted among the tests holding a device until it ends ([`sole`]); it
/// waits here while another test holds the process's devices alone.
pub fn device_boot() -> DeviceBoot {
    let profiles = profiles().to_path_buf();
    open(false);
    DeviceBoot {
        profiles: Some(profiles),
        ..DeviceBoot::default()
    }
}

/// `boot` for a load whose test fires the lanes and rows its budgets state
/// straight at the engine, as no runtime does: the load lays its fires to
/// those rows (`DeviceBoot::fires_stated`) in place of the fire-top rule's fire
/// top, which for a model with no weight to read is its narrowest rung (one
/// row, one lane), since every rung is then wholly efficient.
pub fn fires_stated(boot: DeviceBoot) -> DeviceBoot {
    DeviceBoot {
        fires_stated: true,
        ..boot
    }
}

/// The submit bound the process's rates step measures under ([`profiles`]):
/// the roomy one the bounded loads state (`runtime::fold`), so every key a
/// bounded load weighs by is measured, and no point is past it.
const MEASURED_UNDER: std::time::Duration = std::time::Duration::from_secs(2);

/// The profiles directory of this test process, fresh: what a load reads
/// from it is what this process measured, never what an earlier run left
/// (its kernel curves and picks, its rates), so a run answers as a clean
/// one does. The device's rates are measured into it once, as the first
/// test opens a device and before any other test opens one (every one waits
/// here), so the rates step times the device alone: a load's rates step
/// timed beside other tests' fires drops points past its bound and misses
/// keys. Each model's kernel curves are measured at its first load, and
/// kept for the process's later loads. Under the target's temp directory, a
/// directory a process; those of processes no longer running are removed
/// (where `/proc` says which run).
pub fn profiles() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        // A suite built on another machine (a phone's) has no such path:
        // `PIE_SUITE_TMPDIR` states one.
        let root = std::env::var_os("PIE_SUITE_TMPDIR")
            .map_or_else(|| env!("CARGO_TARGET_TMPDIR").into(), std::path::PathBuf::from)
            .join(format!("pie-{}-profiles", crate::backend::NAME));
        #[cfg(target_os = "linux")]
        for entry in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            let gone = entry
                .file_name()
                .to_str()
                .and_then(|pid| pid.parse::<u32>().ok())
                .is_some_and(|pid| !std::path::Path::new(&format!("/proc/{pid}")).exists());
            if gone {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
        let dir = root.join(std::process::id().to_string());
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the test process's profiles directory");
        let boot = DeviceBoot {
            profiles: Some(dir.clone()),
            submit_bound: Some(MEASURED_UNDER),
            ..DeviceBoot::default()
        };
        // No device: no rates, and no test that would read them runs.
        if let Err(why) =
            <crate::backend::Api as engine_gpu::serve::Serve>::open(&boot, ALLOWANCE, None)
        {
            eprintln!("the process's rates step: {why}");
        }
        dir
    })
}

/// How many parts the harness cuts the device into. A test that loads a
/// model takes one part, its share ([`share`]), and every load it makes
/// takes no more than that; one part is left to the kernel tests (a few
/// small buffers each) and the drivers. So at most `SHARES - 1` tests hold
/// a model at once, whatever the harness's thread count, and their loads
/// never collide: each sizes its pool within its share, not within what the
/// device has left when it loads (which a load beside it then takes).
pub const SHARES: u64 = 4;

/// The tests holding a share now, whether one of them holds the device
/// alone ([`alone`]), and the signal one gave its share back.
struct Holders {
    count: u64,
    alone: bool,
}

static HOLDERS: std::sync::Mutex<Holders> = std::sync::Mutex::new(Holders {
    count: 0,
    alone: false,
});
static FREED: std::sync::Condvar = std::sync::Condvar::new();

/// A test's share, held by its thread until the test ends; `alone` where
/// the test holds the whole device.
struct Held {
    alone: bool,
}

impl Drop for Held {
    fn drop(&mut self) {
        let mut holders = HOLDERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        holders.count -= 1;
        holders.alone &= !self.alone;
        // Waiters wait on different conditions (a free share, an empty
        // device): each looks again.
        FREED.notify_all();
    }
}

thread_local! {
    static HELD: std::cell::RefCell<Option<Held>> = const { std::cell::RefCell::new(None) };
    static OPENED: std::cell::RefCell<Option<Opened>> = const { std::cell::RefCell::new(None) };
}

/// The process's devices: a test holding one (or a share, under which it
/// may open one) reads this gate until it ends; a test holding them all
/// alone ([`sole`]) writes it.
static DEVICES: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// The calling test's hold on [`DEVICES`], kept by its thread until the
/// test ends.
enum Opened {
    Shared(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>),
    Sole(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>),
}

/// Takes the calling test's hold on the process's devices, unless it holds
/// one already: shared, or where `sole` every device alone.
fn open(sole: bool) {
    OPENED.with(|opened| {
        let mut opened = opened.borrow_mut();
        if let Some(held) = opened.as_ref() {
            assert!(
                matches!(held, Opened::Sole(_)) || !sole,
                "a test takes the process's devices alone before it opens one"
            );
            return;
        }
        *opened = Some(if sole {
            Opened::Sole(
                DEVICES
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        } else {
            Opened::Shared(
                DEVICES
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        });
    });
}

/// The calling test holds every device of the process alone until it ends:
/// it waits until no other test holds a device (or a share, under which it
/// may open one), and no test opens one meanwhile. For a test that reads
/// what the driver holds for the process (`Context::outside`): the driver
/// reports its usage a process, every device of it summed, so another
/// test's device opening, building pipelines, reserving or closing beside
/// this one's would read as this one's. Call this first, before any load.
pub fn sole() {
    open(true);
}

/// Takes the calling test's share, once `free` says the holders leave room
/// for it, unless it holds one already.
fn take(alone: bool, free: impl Fn(&Holders) -> bool) {
    HELD.with(|held| {
        let mut held = held.borrow_mut();
        if let Some(held) = held.as_ref() {
            assert!(
                held.alone || !alone,
                "a test takes the device alone before any load takes its share"
            );
            return;
        }
        let mut holders = HOLDERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !free(&holders) {
            holders = FREED
                .wait(holders)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        holders.count += 1;
        holders.alone = alone;
        *held = Some(Held { alone });
    });
}

/// The calling test holds the device alone until it ends: it waits until no
/// other test holds a share, and no test takes one meanwhile. For a test
/// that times the device's answers against its own host work (a lane's
/// fires against a compile on another thread), which a model test beside it
/// would stretch by its own fires and loads. The test's loads still take
/// its share ([`share`]); call this before them.
pub fn alone() {
    open(false);
    take(true, |holders| holders.count == 0);
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
/// hold theirs, or one holds the device [`alone`]) where it holds none yet, and holds it until it ends; every
/// load it makes then takes the same share. Take it before
/// `model_logits::loading`, which a test holding a share waits on.
pub fn share() -> engine::Allowance {
    open(false);
    take(false, |holders| !holders.alone && holders.count < SHARES - 1);
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

    /// `f` in a frame of its own, timed: the device's time across its work
    /// where it writes timestamps (a spanned frame), else the host's clock
    /// around the submit and the wait (never shorter), ns.
    pub fn timed(&self, f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_slang::Error>) -> f64 {
        let started = std::time::Instant::now();
        let frame = self.device.spanned().expect("a frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&sink).expect("the kernel fires");
        }
        let span = frame.commit_spanned().expect("the frame lands");
        let host = started.elapsed().as_nanos() as f64;
        span.map_or(host, |ns| ns as f64)
    }

    /// One arm's timestamps, laid as a load under a stated submit bound
    /// lays them (`Serve::lay_stamps`).
    pub fn stamps(&self) -> <crate::backend::Api as engine_gpu::serve::Serve>::Stamps {
        <crate::backend::Api as engine_gpu::serve::Serve>::lay_stamps(&self.device, 1)
            .expect("the timestamps lay")
    }

    /// `f` walked into a step's timed frame writing `stamps`, as a serving
    /// step walks under a stated submit bound (`Serve::walk_frame`), kept
    /// to replay where `keep`, committed as a step is, and its device time
    /// as the backend reads it once it lands (`crate::backend::fired_time`),
    /// ns: `None` where the backend reads none.
    pub fn fired(
        &self,
        stamps: &<crate::backend::Api as engine_gpu::serve::Serve>::Stamps,
        keep: bool,
        f: impl FnOnce(&Sink<'_>) -> Result<(), kernels_slang::Error>,
    ) -> Option<u64> {
        let frame = <crate::backend::Api as engine_gpu::serve::Serve>::walk_frame(
            &self.device,
            keep,
            Some(stamps),
        )
        .expect("a timed frame");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            f(&sink).expect("the kernel fires");
        }
        crate::backend::fired_time(frame, keep)
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
    /// Overwrites the buffer behind `handle` with `values`, raw 32-bit
    /// words from its first byte.
    pub fn write_u32(&self, handle: u32, values: &[u32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let binding = self.handles.get(handle).expect("a bound handle");
        let mut buffer = binding.buffer().clone();
        let at = binding.offset();
        drop(binding);
        buffer.write(at, &bytes).expect("written");
    }

    /// `n` raw 32-bit words.
    pub fn down_u32(&self, handle: u32, n: usize) -> Vec<u32> {
        let bytes = self
            .handles
            .read(handle, (n * 4) as u64)
            .expect("read back");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
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
/// A window's query CSR at int `at` of a table whose other ints are zero
/// (a scan blind to `at` reads empty requests there): `at` 0 is the CSR
/// alone, a later one a window's cut of the fire's packed table.
pub fn csr_at(gpu: &mut Gpu, indptr: &[i32], at: usize) -> kernels_slang::ssm::Csr {
    let mut table = vec![0i32; at];
    table.extend_from_slice(indptr);
    table.extend_from_slice(&[0; 3]);
    let buf = gpu.up_i32(&table);
    kernels_slang::ssm::Csr {
        table: kernels_slang::Tensor::new(buf, table.len() as u32, 1, poem_ir::Dtype::I32),
        at: at as u32,
        lanes: indptr.len() as u32 - 1,
    }
}

/// The ints a scan's CSR is placed at in the suite: alone, and as a
/// window's cut of a packed table.
pub const CSR_AT: [usize; 2] = [0, 5];

/// A fire's `rows` walked whole, and cut in two windows over the same
/// planes (the second from a third of the rows on), as a fire whose lanes
/// lie in two runs of a region takes them.
#[must_use]
pub fn row_cuts(rows: u32) -> Vec<Vec<kernels_slang::ssm::RowSpan>> {
    let whole = kernels_slang::ssm::RowSpan { row0: 0, rows };
    let cut = (rows / 3).max(1);
    let mut cuts = vec![vec![whole]];
    if rows > 1 {
        cuts.push(vec![
            kernels_slang::ssm::RowSpan { row0: 0, rows: cut },
            kernels_slang::ssm::RowSpan {
                row0: cut,
                rows: rows - cut,
            },
        ]);
    }
    cuts
}

/// A committed scan's dispatches over `lanes` lanes of a work plane holding
/// `batch` (the engine's `committed_batches`): each one's first lane, its
/// lanes, and the extended row its first lane starts at, past the rows the
/// lanes before it replay (`replay[lane0 + lane]`).
#[must_use]
pub fn committed_batches(
    lanes: u32,
    batch: u32,
    replay: &[i32],
    lane0: u32,
) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    let (mut first, mut row0) = (0u32, 0u32);
    while first < lanes {
        let n = batch.min(lanes - first);
        out.push((first, n, row0));
        for lane in first..first + n {
            row0 += replay[(lane0 + lane) as usize].max(0) as u32;
        }
        first += n;
    }
    out
}

/// A fire's requests (`indptr`) walked whole, and cut in two windows at a
/// lane: each window's first row and its own CSR, rebased to it, as a
/// window states them.
#[must_use]
pub fn lane_cuts(indptr: &[i32]) -> Vec<Vec<(u32, Vec<i32>)>> {
    let lanes = indptr.len() - 1;
    let mut cuts = vec![vec![(0, indptr.to_vec())]];
    if lanes > 1 {
        let at = lanes / 2;
        let first = indptr[at];
        cuts.push(vec![
            (0, indptr[..=at].to_vec()),
            (
                first as u32,
                indptr[at..].iter().map(|&b| b - first).collect(),
            ),
        ]);
    }
    cuts
}

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
    crate::backend::SPIRV || columns.is_multiple_of(2)
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
