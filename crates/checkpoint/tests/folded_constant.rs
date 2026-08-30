//! Undoing a constant a checkpoint format folded in, end to end.
//!
//! llama.cpp publishes Gemma's rmsnorm weights as `w + 1`, because its own
//! kernel multiplies by the stored value; pie's kernel is `x * (1 + w)` and
//! its artifacts hold HuggingFace's `w`. Both files describe the same model
//! and only one matches the kernel, so an ingest that renamed and stopped
//! would produce a model that loads, serves and answers slightly wrong.
//!
//! Measured, not assumed: every norm of `gemma-3-270m-it-F16.gguf` is exactly
//! `+1.000000` from the same tensor in the safetensors release it was
//! converted from, and no other tensor differs by anything but F16 rounding.
//!
//! These are end-to-end through the real compiler and executor because the
//! interesting failures are all the right size: a bias that never ran, a bias
//! that ran twice, and a bias applied at the wrong width all produce a buffer
//! of exactly the expected length.

use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use checkpoint::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

/// One F32 row whose values are the four Gemma norms would look like after
/// llama.cpp folded the one in: `w + 1` for `w` in `{0, -0.5, 0.25, 3}`.
fn folded_norm(dir: &std::path::Path, dtype: DType) -> Metadata {
    let folded = [1.0f32, 0.5, 1.25, 4.0];
    let mut file = Vec::new();
    for value in folded {
        match dtype {
            DType::F32 => file.extend_from_slice(&value.to_le_bytes()),
            DType::Bf16 => {
                file.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            }
            other => panic!("no fixture for {other:?}"),
        }
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.gguf"), &file).unwrap();
    let span = u64::try_from(file.len()).unwrap();
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.gguf".to_string(),
            size_bytes: span,
            format: CheckpointFormat::Gguf,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "attn_norm".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: span,
            shape: vec![4],
            encoding: Encoding::Raw(dtype),
        }],
    }
}

fn floats(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        other => panic!("no reader for {other:?}"),
    }
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

/// The fold comes back out, in the operand's own dtype.
#[test]
fn a_folded_constant_is_subtracted_back_off() {
    let dir = std::env::temp_dir().join(format!("pie_bias_f32_{}", std::process::id()));
    let metadata = folded_norm(&dir, DType::F32);
    let out = TensorContract::new(
        "layer.0.input_layernorm",
        Expr::src("attn_norm").bias(-1.0),
        vec![4],
        Encoding::Raw(DType::F32),
    );
    assert_eq!(
        floats(&run(&dir, &metadata, out), DType::F32),
        vec![0.0, -0.5, 0.25, 3.0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A BF16 operand stays BF16, and the sum is rounded once.
///
/// Worth its own case because the arithmetic is done in `f32`: a bias that
/// widened the buffer, or that wrote `f32` bytes under a BF16 declaration,
/// would produce twice the bytes and be caught here rather than by whatever
/// read the tensor next.
#[test]
fn a_narrow_operand_keeps_its_width() {
    let dir = std::env::temp_dir().join(format!("pie_bias_bf16_{}", std::process::id()));
    let metadata = folded_norm(&dir, DType::Bf16);
    let out = TensorContract::new(
        "layer.0.input_layernorm",
        Expr::src("attn_norm").bias(-1.0),
        vec![4],
        Encoding::Raw(DType::Bf16),
    );
    let got = run(&dir, &metadata, out);
    assert_eq!(got.len(), 8, "four BF16 elements");
    assert_eq!(floats(&got, DType::Bf16), vec![0.0, -0.5, 0.25, 3.0]);
    std::fs::remove_dir_all(&dir).ok();
}

/// A bias composes with a placement rather than replacing one.
///
/// The half a shard takes is still the half it took, and the constant is
/// applied to that and not to the whole tensor -- which is the property that
/// lets one node cover a family whose norms are also sharded.
#[test]
fn a_bias_applies_to_the_band_it_is_written_over() {
    let dir = std::env::temp_dir().join(format!("pie_bias_band_{}", std::process::id()));
    let metadata = folded_norm(&dir, DType::F32);
    let out = TensorContract::new(
        "layer.0.input_layernorm",
        Expr::src("attn_norm").slice(0, 2, 2).bias(-1.0),
        vec![2],
        Encoding::Raw(DType::F32),
    );
    assert_eq!(
        floats(&run(&dir, &metadata, out), DType::F32),
        vec![0.25, 3.0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Zero is refused, because it is what an unset field reads as.
///
/// The same trap `Scale` guards: a node whose constant was forgotten is
/// indistinguishable from one that meant the identity, and the identity is
/// spelled by not writing the node.
#[test]
fn an_unset_constant_is_refused_rather_than_run_as_the_identity() {
    let dir = std::env::temp_dir().join(format!("pie_bias_zero_{}", std::process::id()));
    let metadata = folded_norm(&dir, DType::F32);
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "n",
            Expr::src("attn_norm").bias(0.0),
            vec![4],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let why = checkpoint::plan::compile(
        &metadata,
        &contract,
        StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .expect_err("a zero bias should not compile")
    .to_string();
    assert!(why.contains("Bias is zero"), "{why}");
    std::fs::remove_dir_all(&dir).ok();
}
