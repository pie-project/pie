//! Placing part of a blocked tensor, end to end through the real executor.
//!
//! Every GGUF import so far copied whole tensors, and offset zero is correct
//! under any addressing rule, so nothing here was ever exercised. The moment
//! a contract asks for SOME of a blocked tensor -- a tensor-parallel shard, or
//! llama.cpp's Q/K row order undone -- the element-to-byte conversion has to
//! know that a block carries its scale inside itself.
//!
//! These are end-to-end rather than unit tests because the failure they guard
//! against produced a result of exactly the right SIZE. `compile` returned a
//! plan, the executor ran it, and the bytes were wrong; only reading the
//! values back catches that.

use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use checkpoint::types::{
    CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

/// Four Q4_0 rows of 32 elements. Row `r` has scale `r + 1` and all-zero
/// codes, so every element of row `r` decodes to `-8 * (r + 1)` -- a value
/// that says which row it came from.
fn four_rows(dir: &std::path::Path) -> Metadata {
    let mut file = Vec::new();
    for r in 0..4u32 {
        let scale = half::f16::from_f32((r + 1) as f32);
        file.extend_from_slice(&scale.to_bits().to_le_bytes());
        file.extend_from_slice(&[0x00; 16]);
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.gguf"), &file).unwrap();
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.gguf".to_string(),
            size_bytes: 72,
            format: CheckpointFormat::Gguf,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "q".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 72,
            shape: vec![4, 32],
            encoding: Encoding::Quant(QuantSpec {
                scheme: QuantScheme::GgufQ4_0,
                logical_dtype: DType::Bf16,
                bits_per_element: 4,
                group_size: 32,
                channel_axis: None,
            }),
        }],
    }
}

/// The first value of each output row, which identifies the source row.
fn rows_of(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect::<Vec<_>>()
        .chunks_exact(32)
        .map(|row| row[0])
        .collect()
}

fn run(dir: &std::path::Path, metadata: &Metadata, out: TensorContract) -> Vec<u8> {
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
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .unwrap();
    checkpoint::executor::Execution::new(&plan, dir)
        .run()
        .unwrap()
        .tensors
        .remove(&name)
        .unwrap()
}

/// A band of blocked rows reads the rows it names.
///
/// Under the old bit-width rule row 1 began at byte 16 rather than 18, which
/// is inside row 0's codes. The band came back 36 bytes long -- exactly two
/// blocked rows, so every size check passed -- and full of the wrong data.
#[test]
fn a_band_of_blocked_rows_reads_those_rows() {
    let dir = std::env::temp_dir().join(format!("pie_blocked_band_{}", std::process::id()));
    let metadata = four_rows(&dir);
    let out = TensorContract::new(
        "band",
        Expr::src("q")
            .slice(0, 1, 2)
            .cast(Encoding::Raw(DType::Bf16)),
        vec![2, 32],
        Encoding::Raw(DType::Bf16),
    );
    assert_eq!(rows_of(&run(&dir, &metadata, out)), vec![-16.0, -24.0]);
    std::fs::remove_dir_all(&dir).ok();
}

/// The same band with the bytes left blocked, so nothing but addressing is
/// under test.
#[test]
fn a_band_of_blocked_rows_stays_block_aligned_without_a_decode() {
    let dir = std::env::temp_dir().join(format!("pie_blocked_raw_{}", std::process::id()));
    let mut metadata = four_rows(&dir);
    // Re-write the payload so each row's first byte names it.
    let mut file = Vec::new();
    for r in 0..4u8 {
        file.extend_from_slice(&[0xA0 + r, 0xBB]);
        file.extend_from_slice(&[r; 16]);
    }
    std::fs::write(dir.join("model.gguf"), &file).unwrap();
    let encoding = metadata.tensors[0].encoding.clone();
    metadata.tensors[0].shape = vec![4, 32];
    let out = TensorContract::new("band", Expr::src("q").slice(0, 1, 2), vec![2, 32], encoding);
    let got = run(&dir, &metadata, out);
    assert_eq!(got.len(), 36, "two blocked rows are 36 bytes");
    assert_eq!(&got[..2], &[0xA1, 0xBB], "the band starts at row 1's block");
    assert_eq!(&got[18..20], &[0xA2, 0xBB], "and continues at row 2's");
    std::fs::remove_dir_all(&dir).ok();
}

/// llama.cpp's Q/K row order, undone directly on the blocked payload.
///
/// This is the shape `crates/model/src/*/import.rs` needs for the `llama`
/// architecture: within each head GGUF interleaves the two rope halves, so
/// `gguf[2k]` is `hf[k]` and `gguf[2k + 1]` is `hf[hd/2 + k]`. Written as
/// `Concat` of two `Stride`s it needs no intermediate tensor and no second
/// decode -- the strided read moves whole blocked rows and the decode happens
/// once, on the way out.
///
/// Four rows standing in for one head of `hd = 4`: the expected order is
/// source rows 0, 2, 1, 3.
#[test]
fn a_strided_regroup_of_blocked_rows_keeps_the_blocks_whole() {
    let dir = std::env::temp_dir().join(format!("pie_blocked_perm_{}", std::process::id()));
    let metadata = four_rows(&dir);
    let expr = Expr::concat(
        0,
        vec![
            Expr::src("q").stride(0, 0, 2, 2),
            Expr::src("q").stride(0, 1, 2, 2),
        ],
    )
    .cast(Encoding::Raw(DType::Bf16));
    let out = TensorContract::new("hf_q", expr, vec![4, 32], Encoding::Raw(DType::Bf16));
    assert_eq!(
        rows_of(&run(&dir, &metadata, out)),
        vec![-8.0, -24.0, -16.0, -32.0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// One instance of a blocked stack is the instance it names.
///
/// The shape a GGUF mixture import needs: llama.cpp joins a
/// mixture's experts into one `[E, I, H]` tensor and the artifact holds `E`
/// separate `[I, H]` ones, so the ingest cuts a slab per expert. Every slab
/// but the first begins at a nonzero offset, which is where a blocked tensor
/// stops forgiving an addressing rule.
///
/// Four rows read as `[2, 2, 32]`. Instance 1 begins at element 64, which is
/// byte **36** and not byte 64 * 4 / 8 = 32; under the bit rule it would
/// start four bytes early, inside instance 0's last block, and come back the
/// right size and entirely wrong.
#[test]
fn one_instance_of_a_blocked_stack_is_the_instance_it_names() {
    let dir = std::env::temp_dir().join(format!("pie_blocked_unstack_{}", std::process::id()));
    let mut metadata = four_rows(&dir);
    metadata.tensors[0].shape = vec![2, 2, 32];
    let instance = |index: i64| {
        TensorContract::new(
            "expert",
            Expr::src("q")
                .slice(0, index, 1)
                .transmute(checkpoint::contract::TensorType::new(
                    vec![2, 32],
                    metadata.tensors[0].encoding.clone(),
                ))
                .cast(Encoding::Raw(DType::Bf16)),
            vec![2, 32],
            Encoding::Raw(DType::Bf16),
        )
    };
    assert_eq!(
        rows_of(&run(&dir, &metadata, instance(0))),
        vec![-8.0, -16.0]
    );
    assert_eq!(
        rows_of(&run(&dir, &metadata, instance(1))),
        vec![-24.0, -32.0],
        "instance 1 is rows 2 and 3, not a window four bytes short of them"
    );
    std::fs::remove_dir_all(&dir).ok();
}
