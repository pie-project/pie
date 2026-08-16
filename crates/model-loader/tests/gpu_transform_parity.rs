//! The four rows this crate fires, fired, and compared byte for byte.
//!
//! `arena_transforms.rs` proves the executor OFFERS a transform to its
//! backing with no GPU in the build. This proves the other half, and it is
//! the half that had never been proved from inside this crate: that a real
//! [`CudaArena`] on a real device, reaching `kernels-cuda` through NVRTC
//! rather than an ahead-of-time archive, leaves behind the bytes the host
//! executor leaves behind.
//!
//! # Why bytes and not a build
//!
//! Cutting over from the archive crate's `kernels_cuda::ffi::pie_k_*` to
//! `kernels_cuda::api::*` moved two extents out of every argument list
//! and a stream out of a third — a launch that compiles is therefore no
//! evidence at all, because the rectangle the JIT derives from [`Dims`] is
//! precisely the thing the C symbols used to be handed and no longer are. A
//! `rows` supplied where the rule wanted a group count still compiles, still
//! launches, and writes a fraction of the tensor.
//!
//! `.wiki/fix/loader.md` records what a wrong answer on this path costs: a
//! host `Cast` that pivoted every element through `f64` ran at 0.25 GiB/s and
//! went unnoticed for a long time, and the device transforms that should have
//! replaced it were unreachable because two gates' intersection was empty.
//! These four kernels quantise and cast WEIGHTS at load time. Wrong bytes here
//! are a checkpoint that loads, runs, and is quietly wrong — there is no
//! later stage that can tell.
//!
//! # The reference
//!
//! The host executor, always, and on the same plan and the same file. It is
//! the implementation this crate builds without a toolkit, it is what
//! `PIE_LOADER_DEVICE_TRANSFORMS=0` selects, and a device answer nobody
//! compared against it is a claim rather than a result.
//!
//! # The one row no plan can name
//!
//! `quant::scale_rows_bf16` is unreachable from any compiled plan, and that is
//! a fact about the compiler rather than an omission here:
//! `passes::tile::cuda_kernel` requires `in_place`, `rewrites_in_place`
//! requires `dest.buffer == inputs[0]`, and `build.rs::transform_with` always
//! allocates a fresh destination — so the intersection is empty exactly as
//! `.wiki/fix/loader.md` describes for the pair of gates it found. It is
//! reported, not worked around. The row is still fired here, through a
//! hand-built [`TileMapOp`] that states what the compiler would state if it
//! could, because the launch has to be known-good on the day the gate opens.

#![cfg(feature = "cuda")]

use std::borrow::Cow;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

use cudarc::runtime::sys as rt;
use model_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use model_loader::contract::{Expr, ModelContract, TensorContract};
use model_loader::error::Error;
use model_loader::executor::Execution;
use model_loader::executor::arena::{ArenaBacking, ArenaSpan, TileMapOp};
use model_loader::executor::cuda::CudaArena;
use model_loader::executor::sink::MemorySink;
use model_loader::plan::passes::tile::CUDA_SCALE_ROWS_BF16;
use model_loader::plan::{
    CUDA_TILE_MAP_MASK, LoadPlan, StorageInstr, StorageTarget, compile as compile_load_plan,
};
use model_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

/// 64 rows of 128, which is four MXFP4 groups per row and enough rows that a
/// `RouteRows` grid sized from the wrong axis would leave most of the tensor
/// untouched rather than subtly wrong.
const ROWS: i64 = 64;
const COLS: i64 = 128;
const ELEMS: usize = (ROWS * COLS) as usize;

const F32_BYTES: u64 = ELEMS as u64 * 4;
const BF16_BYTES: u64 = ELEMS as u64 * 2;
const FACTOR_BYTES: u64 = COLS as u64 * 2;

const OFF_F32: u64 = 0;
const OFF_BF16: u64 = F32_BYTES;
const OFF_FACTORS: u64 = F32_BYTES + BF16_BYTES;
const FILE_BYTES: u64 = F32_BYTES + BF16_BYTES + FACTOR_BYTES;

/// `f32` to BF16 bits by round-to-nearest-even — the narrowing the loader's
/// own cast performs, spelled here so the fixture needs no dependency the test
/// target does not already have, and so the EXPECTED bytes are never produced
/// by the code under test.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// One file holding an F32 block, the same block in BF16, and a row of
/// per-column factors.
///
/// The plan carries every offset it reads, so the file is bytes and nothing
/// else — no container, no header. The values span the exponent range MXFP4's
/// per-block scale has to track and the range FP8-E4M3's per-channel absmax
/// has to find, so a kernel that ignored either would not survive the
/// comparison; and the F32 block is built from BF16-representable values plus
/// a deliberate low-mantissa perturbation, so `cast_fp32_to_bf16` is asked to
/// round rather than to truncate.
fn checkpoint() -> &'static Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        // Per PROCESS, not merely per test binary. `CARGO_TARGET_TMPDIR` is
        // one path per crate per target, so two concurrent
        // `cargo test -p model-loader --features cuda-13` runs — two agents,
        // two branches, a rebuild racing a run — address the same directory.
        // The publish below is atomic and the bytes are a pure function of
        // the constants above, so today both writers agree; the day `ROWS`
        // or `COLS` differs between the two builds, one run reads the other's
        // fixture, the plan's extents no longer describe the file, and the
        // parity assertion below reports a byte difference that is nothing to
        // do with the kernel. A wrong answer that names the wrong culprit is
        // worse than a crash, and the pid costs one line.
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("gpu-parity-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the fixture directory");

        let mut bytes = Vec::with_capacity(FILE_BYTES as usize);
        let mut lcg = 0x9e37_79b9u32;
        let mut next = move || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            lcg
        };

        let values: Vec<f32> = (0..ELEMS)
            .map(|i| {
                let r = next();
                // [-8, 8) with a mantissa that does not fit bf16's eight bits,
                // scaled by a per-row power of two so the exponent range is
                // real rather than nominal.
                let mag = ((r >> 8) as f32 / f32::from_bits(0x4b80_0000) - 0.5) * 16.0;
                mag * f32::from_bits(((127 + (i as u32 / 128 % 5)) << 23) - (1 << 23))
            })
            .collect();
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &values {
            bytes.extend_from_slice(&bf16_bits(*v).to_le_bytes());
        }
        for c in 0..COLS {
            // Factors near one, and never one: a scale that multiplied by the
            // wrong column would still be near the right magnitude, so the
            // comparison has to be bit-exact to catch it, which it is.
            let f = 0.5 + (c as f32) / (COLS as f32);
            bytes.extend_from_slice(&bf16_bits(f).to_le_bytes());
        }
        assert_eq!(bytes.len() as u64, FILE_BYTES);

        let staging = dir.join(format!("model.safetensors.{}", std::process::id()));
        std::fs::write(&staging, &bytes).expect("write the fixture checkpoint");
        std::fs::rename(&staging, dir.join("model.safetensors")).expect("publish it");
        dir
    })
}

/// The three tensors above, addressed.
fn metadata() -> CheckpointMetadata {
    let tensor =
        |id: u32, name: &str, off: u64, span: u64, shape: Vec<i64>, dtype: DType| RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: off,
            span_bytes: span,
            shape,
            encoding: Encoding::Raw(dtype),
        };
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: FILE_BYTES,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            tensor(0, "w32", OFF_F32, F32_BYTES, vec![ROWS, COLS], DType::F32),
            tensor(
                1,
                "w16",
                OFF_BF16,
                BF16_BYTES,
                vec![ROWS, COLS],
                DType::BF16,
            ),
            tensor(
                2,
                "s",
                OFF_FACTORS,
                FACTOR_BYTES,
                vec![1, COLS],
                DType::BF16,
            ),
        ],
    }
}

/// The device target, with the tile-map mask that lets a plan name a kernel.
fn cuda_target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::for_backend(BackendKind::Cuda, 0, 1)
    }
}

fn compile(contract: &ModelContract, target: StorageTarget) -> LoadPlan {
    compile_load_plan(&metadata(), contract, target).expect("the fixture compiles")
}

/// Every kernel row the plan names, in order.
fn kernels_named(plan: &LoadPlan) -> Vec<String> {
    plan.instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { transform, .. } => transform.kernel.clone(),
            _ => None,
        })
        .collect()
}

/// A device arena and everything it borrows, released in order on drop.
///
/// Its own type because the ordering matters and a test that got it wrong
/// would fail somewhere else: the arena's pinned staging and its events are
/// freed by its own `Drop`, and only then may the stream and the allocation
/// go. A `CudaArena` outliving the stream it was handed is precisely what
/// `CudaArena::new`'s safety contract forbids.
struct Device {
    arena: Option<CudaArena>,
    stream: rt::cudaStream_t,
    base: *mut c_void,
}

/// Run `f`, swallowing a panic and the message it would print.
///
/// cudarc is built `fallback-dynamic-loading`, so a missing `libcudart`
/// does not come back as an error -- it PANICS from inside the shim. A
/// guard that reads the count and matches on the result therefore never
/// runs on the machine it exists for, which is the one failure mode a
/// skip guard cannot have. Only the FIRST cudarc call is wrapped: past
/// it the library is known loaded, and catching panics any wider would
/// turn a real failure into a skip.
fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

/// Is there a CUDA device to run on, or should this test say why not?
///
/// `#![cfg(feature = "cuda")]` gates this file on a BUILD, and a build
/// that names CUDA is not a machine that has it. Without this the three
/// tests below called `cudaMalloc` on whatever came back and died on an
/// `assert_eq!(status, cudaSuccess, "cudaMalloc")` -- a red suite, on a
/// machine where the honest answer is "not here". CI builds this crate
/// with `cuda-13` on runners with no GPU.
fn device_or_skip(what: &str) -> bool {
    // The count lives INSIDE the closure: a `&mut i32` captured across a
    // catch is not `UnwindSafe`, and it does not need to be.
    let probe = quietly(|| {
        let mut count: i32 = 0;
        // SAFETY: `count` is a live local. This is the first cudarc call
        // the process makes, and the one that panics for a missing library.
        let status = unsafe { rt::cudaGetDeviceCount(&raw mut count) };
        (status, count)
    });
    let Some((status, count)) = probe else {
        eprintln!("skipping {what}: no CUDA runtime library on this machine");
        return false;
    };
    if status != rt::cudaError::cudaSuccess {
        eprintln!("skipping {what}: cudaGetDeviceCount: {status:?}");
        return false;
    }
    if count == 0 {
        eprintln!("skipping {what}: no CUDA device");
        return false;
    }
    true
}

impl Device {
    fn new(bytes: usize, max_write: usize) -> Self {
        let bytes = bytes.max(1);
        let mut base: *mut c_void = std::ptr::null_mut();
        // SAFETY: `base` is a live local; this is the allocation the arena
        // will address.
        let status = unsafe { rt::cudaMalloc(&raw mut base, bytes) };
        assert_eq!(status, rt::cudaError::cudaSuccess, "cudaMalloc");
        // The arena is a span of memory the caller owns, so a fresh
        // allocation holds whatever the last tenant left. Zeroing makes any
        // byte the transforms do NOT write compare equal to the host's
        // zero-initialized `Vec`, which is what lets the assertion below be
        // over the WHOLE arena rather than over a region the test chose.
        // SAFETY: `base` is the allocation just made, of `bytes` bytes.
        let status = unsafe { rt::cudaMemset(base, 0, bytes) };
        assert_eq!(status, rt::cudaError::cudaSuccess, "cudaMemset");

        let mut stream: rt::cudaStream_t = std::ptr::null_mut();
        // SAFETY: `stream` is a live local.
        let status = unsafe { rt::cudaStreamCreate(&raw mut stream) };
        assert_eq!(status, rt::cudaError::cudaSuccess, "cudaStreamCreate");

        // SAFETY: `base` is a live allocation of `bytes` bytes and `stream` is
        // a live stream; both outlive the arena, which `Drop` below enforces.
        let arena = unsafe { CudaArena::new(base, bytes, max_write, stream.cast()) }
            .expect("wrap the device allocation as an arena");
        Self {
            arena: Some(arena),
            stream,
            base,
        }
    }

    fn arena(&mut self) -> &mut CudaArena {
        self.arena.as_mut().expect("the arena is live")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        drop(self.arena.take());
        // SAFETY: the arena is gone, so nothing else refers to either handle.
        unsafe {
            rt::cudaStreamDestroy(self.stream);
            rt::cudaFree(self.base);
        }
    }
}

/// A [`CudaArena`] that counts what it was asked to run and what it ran.
///
/// Wraps rather than reimplements: every verb is the real arena's, so what is
/// measured is the device path and not a stand-in. The count is the assertion
/// that matters most — the whole reason this file exists is that
/// `run_tile_map` was called ZERO times for the entire life of the feature,
/// and every byte still compared equal because the host had silently done the
/// work.
struct Counting<'a> {
    inner: &'a mut CudaArena,
    offered: usize,
    ran: usize,
}

impl ArenaBacking for Counting<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        self.inner.read(offset, len)
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.inner.write(offset, bytes)
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        self.inner.fill(offset, len, byte)
    }

    fn runs_named_kernels(&self) -> bool {
        self.inner.runs_named_kernels()
    }

    // An offer that is not refused HAS run: the decline went away with the
    // `bool`, so the two counters differ only when a launch errors out.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.offered += 1;
        self.inner.run_tile_map(op)?;
        self.ran += 1;
        Ok(())
    }
}

/// What one device execution of a plan produced.
struct Ran {
    arena: Vec<u8>,
    tensors: std::collections::HashMap<String, Vec<u8>>,
    offered: usize,
    ran: usize,
}

fn on_device(plan: &LoadPlan) -> Ran {
    let arena_len = usize::try_from(plan.memory.arena_bytes()).expect("the arena fits");
    let max_write = usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX);
    let mut device = Device::new(arena_len, max_write);
    let mut sink = MemorySink::default();
    let (arena, offered, ran) = {
        let mut counting = Counting {
            inner: device.arena(),
            offered: 0,
            ran: 0,
        };
        Execution::new(plan, &checkpoint())
            .arena(&mut counting)
            .sink(&mut sink)
            .run()
            .expect("the plan executes on the device");
        counting.inner.finish().expect("the writes drain");
        let arena = counting
            .inner
            .read(0, arena_len)
            .expect("read the arena back")
            .into_owned();
        (arena, counting.offered, counting.ran)
    };
    Ran {
        arena,
        tensors: sink.tensors,
        offered,
        ran,
    }
}

/// Byte-for-byte, with a count rather than a boolean.
///
/// `assert_eq!` on two 40 KiB vectors prints both and says nothing useful. The
/// interesting number when a launch rectangle is wrong is HOW MANY bytes
/// differ and where the first one is — a grid built from the wrong axis leaves
/// a clean prefix, a block width that under-covers leaves a periodic
/// difference, and the two are told apart by the first offset alone.
fn same_bytes(what: &str, device: &[u8], host: &[u8]) -> usize {
    assert_eq!(
        device.len(),
        host.len(),
        "{what}: the two arenas are not the same size"
    );
    let differing = device.iter().zip(host).filter(|(d, h)| d != h).count();
    if differing != 0 {
        let first = device
            .iter()
            .zip(host)
            .position(|(d, h)| d != h)
            .expect("there is one");
        panic!(
            "{what}: the device transform produced different bytes than the \
             host — {differing} of {} differ, first at offset {first} \
             (device {:#04x}, host {:#04x})",
            device.len(),
            device[first],
            host[first],
        );
    }
    device.len()
}

/// THE PROPERTY, for the three rows a compiled plan can name.
///
/// Each case compiles a plan against the CUDA target, asserts the plan STATES
/// the row — because a plan that names nothing would pass a byte comparison
/// trivially, having run on the host both times — then executes it on the
/// device and against the host and compares the whole arena.
#[test]
fn a_named_row_fires_on_the_device_and_agrees_with_the_host() {
    if !device_or_skip("the named-row device/host parity") {
        return;
    }
    let quant = |scheme: QuantScheme, bits: u8, group: u32| {
        Encoding::Quant(QuantSpec {
            scheme,
            logical_dtype: DType::BF16,
            bits_per_element: bits,
            group_size: group,
            channel_axis: Some(Axis(1)),
        })
    };
    let cases: Vec<(&str, &str, ModelContract)> = vec![
        (
            "Cast f32 -> bf16",
            "quant::cast_fp32_to_bf16",
            ModelContract {
                alignment: 256,
                tensors: vec![TensorContract::new(
                    "out",
                    Expr::src("w32").cast(Encoding::Raw(DType::BF16)),
                    vec![ROWS, COLS],
                    Encoding::Raw(DType::BF16),
                )],
                groups: Vec::new(),
            },
        ),
        (
            "Encode bf16 -> MXFP4",
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
            ModelContract {
                alignment: 256,
                tensors: vec![TensorContract::new(
                    "out",
                    Expr::src("w16").cast(quant(QuantScheme::Mxfp4E2M1E8M0, 4, 32)),
                    vec![ROWS, COLS],
                    quant(QuantScheme::Mxfp4E2M1E8M0, 4, 32),
                )],
                groups: Vec::new(),
            },
        ),
        (
            "Encode bf16 -> FP8 E4M3",
            "quant::quantize_bf16_to_fp8_e4m3_per_channel",
            ModelContract {
                alignment: 256,
                tensors: vec![TensorContract::new(
                    "out",
                    Expr::src("w16").cast(quant(QuantScheme::Fp8E4M3, 8, 0)),
                    vec![ROWS, COLS],
                    quant(QuantScheme::Fp8E4M3, 8, 0),
                )],
                groups: Vec::new(),
            },
        ),
    ];

    let mut compared = 0usize;
    for (what, symbol, contract) in cases {
        let plan = compile(&contract, cuda_target());
        assert_eq!(
            kernels_named(&plan),
            vec![symbol.to_string()],
            "{what}: the plan must state the row before anything can check \
             that it ran"
        );

        let device = on_device(&plan);
        assert!(
            device.ran > 0,
            "{what}: the row the plan names never launched (offered \
             {}, ran {})",
            device.offered,
            device.ran
        );

        let host = Execution::new(&plan, &checkpoint())
            .run()
            .expect("the host executes the same plan");
        assert!(
            host.arena.iter().any(|b| *b != 0),
            "{what}: the reference is all zeros, so a device that wrote \
             nothing would have compared equal"
        );
        compared += same_bytes(what, &device.arena, &host.arena);
        for (name, host_bytes) in &host.tensors {
            let device_bytes = device
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("{what}: the device load published `{name}`"));
            compared += same_bytes(
                &format!("{what} / published `{name}`"),
                device_bytes,
                host_bytes,
            );
        }
        println!(
            "{what}: {symbol} — {} launches, bytes identical",
            device.ran
        );
    }
    assert!(compared > 0, "the comparison must have covered something");
    println!("{compared} bytes compared, all identical");
}

/// `quant::scale_rows_bf16`, which no compiled plan can reach.
///
/// The op is hand-built to state exactly what `passes::tile` would state if
/// `rewrites_in_place` could ever hold: a `Scale` whose destination IS its
/// source, blocked one factor per column. The reference is a real compiled
/// plan of the same algebra, executed on the host — the compiler chooses the
/// host for it because the destination is a fresh buffer, which is the same
/// gate that keeps the device row unreachable, so the two paths compute the
/// same function by different routes and that is the whole comparison.
#[test]
fn the_unreachable_scale_row_agrees_with_the_host_when_fired_by_hand() {
    if !device_or_skip("the hand-fired scale row") {
        return;
    }
    // The reference. `scales` is an internal so the factors reach the Scale as
    // an operand rather than as a published tensor.
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "scales",
                Expr::src("s"),
                vec![1, COLS],
                Encoding::Raw(DType::BF16),
            )
            .internal(),
            TensorContract::new(
                "out",
                Expr::src("w16").scale_per_block(Expr::out("scales")),
                vec![ROWS, COLS],
                Encoding::Raw(DType::BF16),
            ),
        ],
        groups: Vec::new(),
    };
    let plan = compile(&contract, cuda_target());
    assert!(
        kernels_named(&plan).is_empty(),
        "the compiler still cannot reach `{CUDA_SCALE_ROWS_BF16}` — if this \
         fires, `passes::tile::rewrites_in_place` has been given a way to \
         hold and this test should compile the plan instead of hand-building \
         the op"
    );
    let host = Execution::new(&plan, &checkpoint())
        .run()
        .expect("the host executes the reference");
    let expected = host
        .tensors
        .get("out")
        .expect("the reference publishes `out`");
    assert_eq!(expected.len(), BF16_BYTES as usize);

    // The operands, read off the fixture directly, so the input to the device
    // is the file's bytes and not something a plan chose.
    let file = std::fs::read(checkpoint().join("model.safetensors")).expect("read the fixture");
    let weight = &file[OFF_BF16 as usize..(OFF_BF16 + BF16_BYTES) as usize];
    let factors = &file[OFF_FACTORS as usize..(OFF_FACTORS + FACTOR_BYTES) as usize];

    let dst = ArenaSpan {
        offset: 0,
        len: weight.len(),
    };
    let factor_span = ArenaSpan {
        offset: weight.len(),
        len: factors.len(),
    };
    // One factor per column, every row sharing it — the blocking that makes
    // the host's odometer index `factors[c]`, which is the `l[c]` the kernel
    // reads. It rode on a `TransformSpec` the op no longer carries: a backing
    // is given the symbol the plan chose and the spans, and nothing it could
    // second-guess the choice from.
    let op = TileMapOp {
        kernel: CUDA_SCALE_ROWS_BF16,
        src: dst,
        dst,
        dst_scales: None,
        factors: Some(factor_span),
        shape: Some((ROWS as u32, COLS as u32)),
    };

    let mut device = Device::new(weight.len() + factors.len(), weight.len());
    let arena = device.arena();
    arena.write(0, weight).expect("stage the weight");
    arena
        .write(weight.len(), factors)
        .expect("stage the factors");
    arena
        .run_tile_map(&op)
        .expect("the arena declined the row it advertises");
    arena.finish().expect("the launch completes");
    let produced = arena
        .read(0, weight.len())
        .expect("read the result back")
        .into_owned();

    let n = same_bytes("Scale bf16 in place", &produced, expected);
    assert_ne!(
        produced, weight,
        "the factors are never one, so a row that did nothing would still \
         have compared equal to its input"
    );
    println!("Scale bf16 in place: {CUDA_SCALE_ROWS_BF16} — {n} bytes identical");
}

/// A refusal is loud.
///
/// The `pie_k_*` symbols returned `void`: a launch the archive declined — for
/// a collapsed rectangle, most often — produced no value to inspect, so the
/// load went on and published a tensor holding whatever the arena had. The
/// typed entry points return a `Result` and this crate propagates it, so the
/// same condition now fails the load with the row named.
///
/// A zero-row extent is the cheapest way to reach it and the one the archive's
/// own launchers guarded with `if (rows == 0 || cols == 0) return;` — the
/// JIT's `LaunchRule::eval` answers `Ungeometric::Empty` for exactly that, and
/// this asserts the answer travels all the way out rather than becoming a
/// silent `Ok(false)` that would send the transform to the host.
#[test]
fn a_launch_the_jit_refuses_fails_the_load() {
    if !device_or_skip("the refused-launch case") {
        return;
    }
    let span = ArenaSpan {
        offset: 0,
        len: BF16_BYTES as usize,
    };
    let op = TileMapOp {
        kernel: CUDA_SCALE_ROWS_BF16,
        src: span,
        dst: span,
        dst_scales: None,
        factors: Some(ArenaSpan {
            offset: BF16_BYTES as usize,
            len: FACTOR_BYTES as usize,
        }),
        // The collapse. Everything else is the op that works above.
        shape: Some((0, COLS as u32)),
    };

    let mut device = Device::new((BF16_BYTES + FACTOR_BYTES) as usize, 1 << 20);
    let refusal = device.arena().run_tile_map(&op);
    let Err(Error::Contract(why)) = refusal else {
        panic!("a rectangle with no rows must not be reported as a launch: {refusal:?}");
    };
    assert!(
        why.contains(CUDA_SCALE_ROWS_BF16),
        "a refusal names the row it refused, or nobody can act on it: {why}"
    );
    println!("refusal: {why}");
}
