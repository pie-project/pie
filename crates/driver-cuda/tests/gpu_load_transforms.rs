//! The loader's device transforms, on a device.
//!
//! Every CUDA load-time kernel in `model-loader` was unreachable — not
//! disabled, not slow, *unreachable* — in every plan the tree could compile.
//! `executor::cuda` implemented four launches, `plan::passes::tile` selected
//! between them per instruction, a golden plan recorded the selection, and
//! `ArenaBacking::run_tile_map` was called zero times
//! (`.wiki/fix/loader.md`).
//!
//! What made that possible is that nothing here existed. `arena_transforms.rs`
//! in the loader proves the executor OFFERS the transform, with no GPU in the
//! build; this proves a real `CudaArena` is asked to launch, launches, and
//! produces the bytes the host produces. The second half is why the file is in
//! this crate: the host executor is the reference implementation, and a device
//! answer nobody compared against it is a claim rather than a result.

#![cfg(all(feature = "cuda-13", feature = "abi"))]

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use model_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use model_loader::contract::{Expr, ModelContract, TensorContract};
use model_loader::error::Error;
use model_loader::executor::Execution;
use model_loader::executor::arena::{ArenaBacking, TileMapOp};
use model_loader::executor::cuda::CudaArena;
use model_loader::executor::sink::MemorySink;
use model_loader::plan::{
    CUDA_TILE_MAP_MASK, LoadPlan, StorageInstr, StorageTarget, compile as compile_load_plan,
};
use model_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

const ROWS: i64 = 64;
const COLS: i64 = 128;
const BYTES: u64 = (ROWS * COLS) as u64 * 2;

/// A BF16 weight on disk, and the directory holding it.
///
/// The plan carries every offset it reads, so the file is the bytes and
/// nothing else. The values are spread over the exponent range MXFP4's
/// per-block scale has to track, so a kernel that ignored the block absmax
/// would not survive the comparison below.
fn checkpoint() -> &'static Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu_load_transforms");
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        let mut bytes = Vec::with_capacity(BYTES as usize);
        let mut lcg = 0x9e37_79b9u32;
        for _ in 0..(ROWS * COLS) {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // A value in [-8, 8) with a fine mantissa, exactly representable
            // in bf16 by construction of `from_f32` rounding.
            let v = ((lcg >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 16.0;
            bytes.extend_from_slice(&bf16_bits(v).to_le_bytes());
        }
        let staging = dir.join(format!("model.safetensors.{}", std::process::id()));
        std::fs::write(&staging, &bytes).expect("write the fixture checkpoint");
        std::fs::rename(&staging, dir.join("model.safetensors")).expect("publish it");
        dir
    })
}

/// `f32` to BF16 bits, by round to nearest even — the same narrowing the
/// loader's own cast performs, spelled here so the fixture needs no dependency
/// beyond what this test target already has.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: BYTES,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "w".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: BYTES,
            shape: vec![ROWS, COLS],
            encoding: Encoding::Raw(DType::BF16),
        }],
    }
}

/// Runtime quantization: read a BF16 weight and store it in `scheme`.
///
/// The shape a driver actually asks for, and the shape that never once reached
/// a kernel — the operand is on a filesystem, a backing is handed arena
/// offsets, and nothing in the compiler moved the bytes across first.
fn plan_for(scheme: QuantScheme, bits: u8, group: u32) -> LoadPlan {
    let quant = Encoding::Quant(QuantSpec {
        scheme,
        logical_dtype: DType::BF16,
        bits_per_element: bits,
        group_size: group,
        channel_axis: Some(Axis(1)),
    });
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w_q",
            Expr::src("w").cast(quant.clone()),
            vec![ROWS, COLS],
            quant,
        )],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::for_backend(BackendKind::Cuda, 0, 1)
    };
    compile_load_plan(&metadata(), &contract, target).expect("the fixture compiles")
}

fn kernels_named(plan: &LoadPlan) -> Vec<String> {
    plan.instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { transform, .. } => transform.kernel.clone(),
            _ => None,
        })
        .collect()
}

/// A `CudaArena` that counts the transforms it is asked to run.
///
/// Wraps rather than reimplements: every verb is the real arena's, so what is
/// measured is the real device path and not a stand-in for it.
struct Counting {
    inner: CudaArena,
    offered: usize,
    ran: usize,
}

impl ArenaBacking for Counting {
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

/// Execute `plan` into device memory and read the arena back.
fn on_device(plan: &LoadPlan) -> (Vec<u8>, usize, usize) {
    let alloc = driver_cuda::device::Allocator::new();
    let arena_len = usize::try_from(plan.memory.arena_bytes())
        .expect("fits")
        .max(1);
    let buf = alloc.alloc(arena_len).expect("device memory for the arena");
    let stream = driver_cuda::device::OwnedStream::new(0).expect("a stream");
    // SAFETY: `buf` is a live allocation of `arena_len` bytes and `stream`
    // outlives this scope.
    let inner = unsafe {
        CudaArena::new(
            buf.as_ptr(),
            arena_len,
            usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX),
            stream.as_ref().as_raw().cast(),
        )
    }
    .expect("wrap the arena");
    let mut arena = Counting {
        inner,
        offered: 0,
        ran: 0,
    };
    let mut sink = MemorySink::default();
    Execution::new(plan, &checkpoint())
        .arena(&mut arena)
        .sink(&mut sink)
        .run()
        .expect("the plan executes on the device");
    arena.inner.finish().expect("the writes drain");
    let bytes = arena
        .inner
        .read(0, arena_len)
        .expect("read the arena back")
        .into_owned();
    (bytes, arena.offered, arena.ran)
}

/// THE PROPERTY. A plan that names a kernel reaches the arena that launches
/// it, and what the launch leaves behind is what the host would have left.
#[test]
fn a_named_kernel_runs_on_the_device_and_agrees_with_the_host() {
    for (scheme, bits, group, symbol) in [
        (
            QuantScheme::Mxfp4E2M1E8M0,
            4,
            32,
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
        ),
        (
            QuantScheme::Fp8E4M3,
            8,
            0,
            "quant::quantize_bf16_to_fp8_e4m3_per_channel",
        ),
    ] {
        let plan = plan_for(scheme, bits, group);
        assert_eq!(
            kernels_named(&plan),
            vec![symbol.to_string()],
            "{scheme:?}: the plan must state the row before anything can check \
             that it ran"
        );

        let (device, offered, ran) = on_device(&plan);
        assert!(
            ran > 0,
            "{scheme:?}: the kernel the plan names was never launched \
             (offered {offered})"
        );

        // The host is the reference. Same plan, same bytes on disk, arena
        // compared whole — including the staging region, which both paths
        // fill with the same operand.
        let host = Execution::new(&plan, &checkpoint())
            .run()
            .expect("the host executes it");
        assert_eq!(
            device.len(),
            host.arena.len(),
            "{scheme:?}: the two arenas are the same size"
        );
        assert_eq!(
            device, host.arena,
            "{scheme:?}: the device transform must produce the host's bytes, \
             or the plan does not determine execution"
        );
    }
}

/// The switch that turns the device path off selects a different route to the
/// same bytes — which is what makes it safe to turn off.
#[test]
fn host_transforms_only_produces_the_same_arena() {
    let plan = plan_for(QuantScheme::Mxfp4E2M1E8M0, 4, 32);
    let alloc = driver_cuda::device::Allocator::new();
    let arena_len = usize::try_from(plan.memory.arena_bytes())
        .expect("fits")
        .max(1);
    let buf = alloc.alloc(arena_len).expect("device memory");
    let stream = driver_cuda::device::OwnedStream::new(0).expect("a stream");
    // SAFETY: as above.
    let arena = unsafe {
        CudaArena::new(
            buf.as_ptr(),
            arena_len,
            usize::try_from(plan.target.max_tile_bytes).unwrap_or(usize::MAX),
            stream.as_ref().as_raw().cast(),
        )
    }
    .expect("wrap the arena");
    let mut arena = Counting {
        inner: arena.host_transforms_only(),
        offered: 0,
        ran: 0,
    };
    let mut sink = MemorySink::default();
    Execution::new(&plan, &checkpoint())
        .arena(&mut arena)
        .sink(&mut sink)
        .run()
        .expect("the plan executes");
    arena.inner.finish().expect("the writes drain");
    assert_eq!(arena.ran, 0, "the switch is off, so nothing launched");

    let bytes = arena
        .inner
        .read(0, arena_len)
        .expect("read back")
        .into_owned();
    let host = Execution::new(&plan, &checkpoint())
        .run()
        .expect("the host executes it");
    assert_eq!(bytes, host.arena);
}
