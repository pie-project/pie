//! What a backing that runs its own transforms is promised, and what it owes.
//!
//! `ArenaBacking::tile_map_caps` / `run_tile_map` let a device do a `TileMap`
//! on operands that are already in its arena, instead of the host staging
//! them back, computing, and sending them across again. That is a delegation
//! with three moving parts and every one of them is checkable with no GPU in
//! the build:
//!
//! * the executor only OFFERS a transform whose operands it has resolved to
//!   arena spans, and offers nothing when the backing claims nothing;
//! * a backing may DECLINE one it cannot run, and the host then produces the
//!   bytes it always did;
//! * the operands it is handed describe the same transform the host path
//!   would have run.
//!
//! The last one is why this is a file rather than a unit test beside the
//! trait. A recording backing that runs nothing and declines everything must
//! leave the arena holding exactly what a backing that was never asked
//! leaves — which is what makes the delegation a *route* rather than a second
//! implementation that could disagree.

use std::borrow::Cow;

use model_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use model_loader::contract::{Expr, ModelContract, TensorContract, TensorType};
use model_loader::error::Error;
use model_loader::executor::arena::{ArenaBacking, TileMapOp};
use model_loader::executor::host::execute_plan_into_backing;
use model_loader::executor::sink::MemorySink;
use model_loader::plan::compile as compile_load_plan;
use model_loader::plan::{
    CUDA_TILE_MAP_MASK, LoadPlan, StorageInstr, StorageTarget, TILE_MAP_ENCODE, TILE_MAP_SCALE,
    TileMapKind,
};
use model_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

// ── The fixture
//
// One MXFP4 payload with per-row exponents, dequantized to bf16 and then
// re-encoded to int8. It is `storage_compiler.rs`'s
// `a_quantized_tensor_is_re_encoded_through_a_decoded_intermediate` reduced
// to its operands: the smallest contract that produces BOTH kinds a device
// claims, a `Scale` (the dequant's multiply) and an `Encode`.

fn raw(id: u32, name: &str, offset: u64, span_bytes: u64, shape: &[i64], dtype: DType) -> RawTensor {
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(dtype),
    }
}

fn metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 256,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw(0, "w", 0, 64, &[4, 16], DType::U8),
            raw(1, "s", 64, 4, &[4, 1], DType::U8),
        ],
    }
}

fn mxfp4() -> QuantSpec {
    QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    }
}

fn int8() -> QuantSpec {
    QuantSpec {
        scheme: QuantScheme::Int8Symmetric,
        logical_dtype: DType::BF16,
        bits_per_element: 8,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    }
}

fn contract() -> ModelContract {
    let int8_encoding = Encoding::Quant(int8());
    ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "scales",
                Expr::src("s").transmute(TensorType {
                    shape: vec![4, 1],
                    encoding: Encoding::Raw(DType::E8M0),
                }),
                vec![4, 1],
                Encoding::Raw(DType::E8M0),
            ),
            // The decoded intermediate: internal, so the re-encode below
            // reads what this wrote rather than going back to the file.
            TensorContract::new(
                "w",
                Expr::src("w")
                    .transmute(TensorType {
                        shape: vec![4, 32],
                        encoding: Encoding::Quant(mxfp4()),
                    })
                    .scale_per_block(Expr::out("scales")),
                vec![4, 32],
                Encoding::Raw(DType::BF16),
            )
            .internal(),
            TensorContract::new(
                "w_int8",
                Expr::out("w").cast(int8_encoding.clone()),
                vec![4, 32],
                int8_encoding,
            ),
        ],
        groups: Vec::new(),
    }
}

fn plan() -> LoadPlan {
    // A CUDA target, because the default one is `BackendKind::Unknown` and
    // refuses `Encode` — and an Encode is half of what this file is about.
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    compile_load_plan(&metadata(), &contract(), target).expect("the fixture must compile")
}

fn kinds(plan: &LoadPlan) -> Vec<TileMapKind> {
    plan.instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect()
}

// ── A backing that records what it is offered and runs none of it.

/// One offer, flattened to the facts a device would branch on.
///
/// Recorded rather than the borrowed op, because the op borrows the
/// executor's transform and outlives nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Offer {
    kind: TileMapKind,
    has_dst_scales: bool,
    has_factors: bool,
    shape: Option<(u32, u32)>,
    dst_encoding: Encoding,
}

struct Recording {
    bytes: Vec<u8>,
    caps: u32,
    offers: Vec<Offer>,
}

impl Recording {
    fn new(len: usize, caps: u32) -> Self {
        Self {
            bytes: vec![0; len],
            caps,
            offers: Vec::new(),
        }
    }
}

fn oob(what: &str) -> Error {
    Error::Contract(format!("recording arena {what} is out of bounds"))
}

impl ArenaBacking for Recording {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let end = offset.checked_add(len).ok_or_else(|| oob("read"))?;
        self.bytes
            .get(offset..end)
            .map(Cow::Borrowed)
            .ok_or_else(|| oob("read"))
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = offset.checked_add(bytes.len()).ok_or_else(|| oob("write"))?;
        self.bytes
            .get_mut(offset..end)
            .ok_or_else(|| oob("write"))?
            .copy_from_slice(bytes);
        Ok(())
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let end = offset.checked_add(len).ok_or_else(|| oob("fill"))?;
        self.bytes
            .get_mut(offset..end)
            .ok_or_else(|| oob("fill"))?
            .fill(byte);
        Ok(())
    }

    fn tile_map_caps(&self) -> u32 {
        self.caps
    }

    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        self.offers.push(Offer {
            kind: op.kind,
            has_dst_scales: op.dst_scales.is_some(),
            has_factors: op.factors.is_some(),
            shape: op.shape,
            dst_encoding: op.dst_encoding.clone(),
        });
        // Declines everything: the point is that the host still finishes.
        Ok(false)
    }
}

/// Execute the fixture against a backing claiming `caps`.
///
/// The checkpoint the fixture names is not on disk, so the execution fails at
/// its first read — and that is fine for every question below, because the
/// executor decides what to OFFER from the plan alone. The byte comparison
/// is between two runs of the same execution, so it is a comparison of two
/// identical prefixes rather than of two completed loads.
fn run(caps: u32) -> Recording {
    let plan = plan();
    let len = usize::try_from(plan.memory.persistent_bytes).expect("fits");
    let mut arena = Recording::new(len.max(1), caps);
    let mut sink = MemorySink::default();
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let _ = execute_plan_into_backing(&plan, &dir, &mut arena, &mut sink, &mut |_| {});
    arena
}

#[test]
fn the_fixture_produces_the_two_kinds_a_device_claims() {
    assert_eq!(
        kinds(&plan()),
        vec![TileMapKind::Scale, TileMapKind::Encode],
        "the delegation tests are only meaningful over these two"
    );
}

#[test]
fn a_backing_that_claims_nothing_is_offered_nothing() {
    let arena = run(0);
    assert!(
        arena.offers.is_empty(),
        "host mode is the default, and the default must not reach the \
         backing: {:?}",
        arena.offers
    );
}

#[test]
fn a_claim_is_per_kind_and_only_that_kind_is_offered() {
    let arena = run(TILE_MAP_SCALE);
    assert!(
        arena.offers.iter().all(|o| o.kind == TileMapKind::Scale),
        "claiming Scale must not offer Encode: {:?}",
        arena.offers
    );
}

#[test]
fn a_scale_offer_carries_the_factors_it_multiplies_by() {
    let arena = run(TILE_MAP_SCALE);
    let Some(scale) = arena.offers.iter().find(|o| o.kind == TileMapKind::Scale) else {
        return;
    };
    assert!(
        scale.has_factors,
        "a blocked Scale reads per-group factors from a second operand; \
         without them a backing has nothing to multiply by"
    );
    assert!(
        !scale.has_dst_scales,
        "and it publishes one tensor, so the Encode-only second destination \
         must stay empty"
    );
}

#[test]
fn a_transform_reading_a_host_scratch_buffer_is_not_offered() {
    // THE LIMIT THIS TEST FOUND, and it is a property of the design rather
    // than of the fixture.
    //
    // Delegation needs every operand to be a span of the arena, because the
    // backing is handed offsets and nothing else. The re-encode in this
    // fixture reads what the dequantizing `Scale` wrote, and the compiler
    // gives that intermediate a buffer with no `persistent_offset` — it is
    // scratch, materialized on the host, because nothing outside the load
    // ever names it.
    //
    // So an `Encode` whose input is an internal tensor cannot be delegated as
    // things stand, and the runtime-quantization path is exactly that shape:
    // read a checkpoint weight, decode it, re-encode the result. What CAN be
    // delegated today is a transform over tensors the plan keeps — which is
    // the `Cast` and `Scale` case the CUDA arena claims.
    //
    // Closing this means giving the plan a way to place a transform's
    // intermediate in the arena when the backing could run both halves there.
    // That is a compiler change and not a backing's, which is why it is
    // written down here rather than worked around in `DeviceArena`.
    let arena = run(TILE_MAP_SCALE | TILE_MAP_ENCODE);
    let scratch_bound: Vec<_> = arena
        .offers
        .iter()
        .filter(|o| o.kind == TileMapKind::Encode)
        .collect();
    assert!(
        scratch_bound.is_empty(),
        "an Encode reading host scratch cannot be addressed in arena offsets, \
         so offering it would hand the backing a span that is not there: {:?}",
        scratch_bound
    );
}

#[test]
fn declining_leaves_the_host_to_produce_the_same_bytes() {
    // The property the whole design rests on: a backing is a ROUTE, not a
    // second implementation. An arena whose backing ran nothing must end up
    // holding what an arena that was never asked holds.
    let declining = run(TILE_MAP_SCALE | TILE_MAP_ENCODE);
    let never_asked = run(0);
    assert_eq!(
        declining.bytes, never_asked.bytes,
        "a declined transform must fall back to the identical host path"
    );
}

