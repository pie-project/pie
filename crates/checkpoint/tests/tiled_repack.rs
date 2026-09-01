//! **`Expr::Repack` REACHABLE AGAIN, AT THE ONE TARGET THAT MAY RUN IT**
//! (§J4b): the tiled affine relabelling `kernels_cuda::linear::tiled` reads,
//! compiled from a contract and executed on the host, held against a
//! hand-written inverse.
//!
//! The node has been in the algebra since the beginning and unreachable for
//! as long: no mask admitted it and no executor implemented it, so a
//! checkpoint that needed one compiled and then failed with nothing but a
//! kind to name. What revives it is a READER — the tiled GEMM and the tiled
//! decode point — and what keeps it honest is that the site is
//! `pie model import` and not a load: `CONVERT_TILE_MAP_MASK` carries the
//! bit and no device mask does.
//!
//! The gates:
//!
//! ```text
//! (a) the code repack is a RELABELLING: a host un-repack of the executor's
//!     own output recovers the plane bit for bit, and the band padding is
//!     zero codes
//! (b) the factor repack is the same rectangle in band order, and a band's
//!     tail is a zero factor
//! (c) a SERVING plan that states one is refused, with the tensor named
//! (d) the shapes the layout is not stated for are refused by the contract
//! ```

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, CUDA_TILE_MAP_MASK, StorageTarget};
use checkpoint::types::{
    CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, RepackLayout, TILED_BAND,
    TensorId,
};

/// The rows of the weight, which are the projection's output columns —
/// deliberately NOT a whole band, so the padding runs.
const ROWS: i64 = 100;
/// The contraction, which is two 64-wide steps and eight 16-wide mma tiles.
const K: i64 = 128;
/// The codes under one factor.
const GROUP: i64 = 64;

fn band_rows(rows: i64) -> i64 {
    let band = i64::from(TILED_BAND);
    (rows + band - 1) / band * band
}

fn affine_u4() -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: QuantScheme::MlxAffineU4,
        logical_dtype: DType::Bf16,
        bits_per_element: 4,
        group_size: GROUP as u32,
        channel_axis: None,
    })
}

/// A `[ROWS, K]` code plane whose nibble at `(row, kk)` is
/// `(row * 7 + kk * 3) % 16` — a value that says where it came from, so a
/// permutation that lands the right bytes in the wrong places is caught.
fn codes() -> Vec<u8> {
    let mut out = vec![0u8; (ROWS * K / 2) as usize];
    for row in 0..ROWS {
        for kk in 0..K {
            let code = ((row * 7 + kk * 3) % 16) as u8;
            let at = (row * K + kk) as usize;
            out[at / 2] |= code << (4 * (at % 2));
        }
    }
    out
}

/// A `[ROWS, K / GROUP]` bf16 factor plane whose element `(row, g)` is
/// `row * 4 + g`, for the same reason.
fn factors() -> Vec<u8> {
    let groups = (K / GROUP) as usize;
    let mut out = Vec::with_capacity(ROWS as usize * groups * 2);
    for row in 0..ROWS as usize {
        for g in 0..groups {
            let v = half::bf16::from_f32((row * 4 + g) as f32);
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
}

/// A one-tensor checkpoint holding `payload` under `name`.
fn source(dir: &std::path::Path, name: &str, payload: &[u8], shape: Vec<i64>, enc: Encoding)
-> Metadata {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.gguf"), payload).unwrap();
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.gguf".to_string(),
            size_bytes: payload.len() as u64,
            format: CheckpointFormat::Gguf,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: payload.len() as u64,
            shape,
            encoding: enc,
        }],
    }
}

/// Compile `out` against `mask` and run it on the host.
fn run(
    dir: &std::path::Path,
    metadata: &Metadata,
    out: TensorContract,
    mask: u32,
) -> Result<Vec<u8>, checkpoint::error::Error> {
    let name = out.name.clone();
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![out],
        groups: Vec::new(),
    };
    let plan = checkpoint::plan::compile(
        metadata,
        &contract,
        StorageTarget {
            tile_map_mask: mask,
            ..StorageTarget::default()
        },
    )?;
    Ok(checkpoint::executor::Execution::new(&plan, dir)
        .run()?
        .tensors
        .remove(&name)
        .unwrap())
}

/// **THE HOST UN-REPACK**, written from `linear/tiled.cuh`'s banner and not
/// from the executor's code, so that the two have to agree about what the
/// layout IS. It is `tests/tiled_matmul.rs`'s `unrepack`, in this crate's
/// vocabulary.
fn unrepack(repacked: &[u8], rows: i64, k: i64) -> Vec<u8> {
    let band = i64::from(TILED_BAND) as usize;
    let quad = 4usize;
    let bands = band_rows(rows) as usize / band;
    let k_tiles = (k as usize) / band;
    let quads = k_tiles / quad;
    let row_bytes = k as usize / 2;
    let mut out = vec![0u8; rows as usize * row_bytes];
    for b in 0..bands {
        for kt in 0..k_tiles {
            for lane in 0..32usize {
                let at = ((b * quads + kt / quad) * 32 + lane) * quad + kt % quad;
                let word = u32::from_le_bytes(repacked[at * 4..at * 4 + 4].try_into().unwrap());
                let col_of = lane / 4;
                let k_base = kt * band + 2 * (lane % 4);
                for s in 0..4usize {
                    let col = b * band + col_of + if s >= 2 { 8 } else { 0 };
                    for h in 0..2usize {
                        let kk = k_base + if s % 2 == 1 { 8 } else { 0 } + h;
                        let code = ((word >> (4 * (s + 4 * h))) & 0xF) as u8;
                        if col < rows as usize {
                            let flat = col * k as usize + kk;
                            out[flat / 2] |= code << (4 * (flat % 2));
                        } else {
                            assert_eq!(code, 0, "the band padding is not a zero code");
                        }
                    }
                }
            }
        }
    }
    out
}

// ─── (a) the code repack is a relabelling ──────────────────────────────────

#[test]
fn the_code_repack_is_a_relabelling_and_nothing_else() {
    let dir = std::env::temp_dir().join(format!("pie_tiled_codes_{}", std::process::id()));
    let plane = codes();
    let metadata = source(&dir, "w", &plane, vec![ROWS, K], affine_u4());
    let padded = band_rows(ROWS);
    let out = TensorContract::new(
        "w.tiled",
        Expr::src("w").repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(vec![padded, K], affine_u4()),
        ),
        vec![padded, K],
        affine_u4(),
    );
    let got = run(&dir, &metadata, out, CONVERT_TILE_MAP_MASK).expect("the repack compiles");
    assert_eq!(got.len(), (padded * K / 2) as usize);
    assert_eq!(
        unrepack(&got, ROWS, K),
        plane,
        "the un-repacked code plane is not the plane that went in"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ─── (b) the factor repack is the band transpose ───────────────────────────

#[test]
fn the_factor_repack_is_the_band_transpose() {
    let dir = std::env::temp_dir().join(format!("pie_tiled_factors_{}", std::process::id()));
    let plane = factors();
    let groups = K / GROUP;
    let metadata = source(
        &dir,
        "s",
        &plane,
        vec![ROWS, groups],
        Encoding::Raw(DType::Bf16),
    );
    let padded = band_rows(ROWS);
    let out = TensorContract::new(
        "s.tiled",
        Expr::src("s").repack(
            RepackLayout::TiledAffineFactor,
            TensorType::raw(vec![padded, groups], DType::Bf16),
        ),
        vec![padded, groups],
        Encoding::Raw(DType::Bf16),
    );
    let got = run(&dir, &metadata, out, CONVERT_TILE_MAP_MASK).expect("the repack compiles");
    assert_eq!(got.len(), (padded * groups * 2) as usize);
    let band = i64::from(TILED_BAND);
    for b in 0..padded / band {
        for g in 0..groups {
            for j in 0..band {
                let at = (((b * groups + g) * band + j) * 2) as usize;
                let row = b * band + j;
                let want = if row < ROWS {
                    let from = ((row * groups + g) * 2) as usize;
                    u16::from_le_bytes([plane[from], plane[from + 1]])
                } else {
                    0
                };
                assert_eq!(
                    u16::from_le_bytes([got[at], got[at + 1]]),
                    want,
                    "the factor at band {b} group {g} row {j} moved"
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ─── (c) a serving plan may not repack ─────────────────────────────────────

/// **THE SITE MOVED; THE LANGUAGE STAYED** (§M-3's pattern). A repack is a
/// pure relabelling, so nothing about it is unsafe to serve — what makes it
/// an import-time transform is that it is paid once per weight, and paying
/// it at every boot over a hundred gigabytes is minutes of rearranging bytes
/// the artifact could have held rearranged. So the CUDA mask refuses it, by
/// name.
#[test]
fn a_serving_plan_may_not_state_a_repack() {
    let dir = std::env::temp_dir().join(format!("pie_tiled_refuse_{}", std::process::id()));
    let plane = codes();
    let metadata = source(&dir, "w", &plane, vec![ROWS, K], affine_u4());
    let padded = band_rows(ROWS);
    let out = TensorContract::new(
        "w.tiled",
        Expr::src("w").repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(vec![padded, K], affine_u4()),
        ),
        vec![padded, K],
        affine_u4(),
    );
    let err = run(&dir, &metadata, out, CUDA_TILE_MAP_MASK)
        .expect_err("a CUDA plan carried a repack");
    let said = err.to_string();
    assert!(
        said.contains("TiledAffineU4Weight"),
        "the refusal does not name the layout: {said}"
    );
    assert!(
        said.contains("pie model import"),
        "the refusal does not name the command that runs it: {said}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ─── (d) the shapes the layout is not stated for ───────────────────────────

#[test]
fn the_shapes_this_layout_is_not_stated_for_are_refused() {
    let dir = std::env::temp_dir().join(format!("pie_tiled_ladder_{}", std::process::id()));
    let plane = codes();
    let metadata = source(&dir, "w", &plane, vec![ROWS, K], affine_u4());
    let padded = band_rows(ROWS);

    // A target that is not the next whole band: the kernel carves its grid
    // off the target's rows, so anything else launches blocks reading codes
    // nothing wrote.
    let short = TensorContract::new(
        "w.short",
        Expr::src("w").repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(vec![ROWS, K], affine_u4()),
        ),
        vec![ROWS, K],
        affine_u4(),
    );
    assert!(
        run(&dir, &metadata, short, CONVERT_TILE_MAP_MASK).is_err(),
        "an unpadded target compiled"
    );

    // A contraction that is not a whole 64-wide step.
    let narrow = TensorContract::new(
        "w.narrow",
        Expr::src("w").slice(1, 0, 32).repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(vec![padded, 32], affine_u4()),
        ),
        vec![padded, 32],
        affine_u4(),
    );
    assert!(
        run(&dir, &metadata, narrow, CONVERT_TILE_MAP_MASK).is_err(),
        "a 32-wide contraction compiled"
    );

    // A batched declaration: the tiled pair repacks a dense projection, and
    // a leading extent is a dimension the algebra could disagree about.
    let batched = TensorContract::new(
        "w.batched",
        Expr::src("w").repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(vec![1, padded, K], affine_u4()),
        ),
        vec![1, padded, K],
        affine_u4(),
    );
    assert!(
        run(&dir, &metadata, batched, CONVERT_TILE_MAP_MASK).is_err(),
        "a batched target compiled"
    );
    std::fs::remove_dir_all(&dir).ok();
}
