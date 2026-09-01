//! The gather lowering, end to end through the real executor.
//!
//! The affine fragment lowers to a copy list, and a copy list is stretches of
//! addresses. An element-granular permutation has no stretches to speak of:
//! the MLX vision patch-embed bank is a `[768, 1536]` stride-3 deinterleave,
//! which is 1.18M single-element runs. `contract::compile` used to refuse it —
//! "it needs a gather lowering, not a copy list" — and there was no gather
//! lowering to have. This is that lowering, checked the only way a byte
//! movement can be: by moving real bytes out of a real file and comparing
//! them, element for element, against the permutation worked out by hand.
//!
//! The shape is the real one on purpose. `plan::build`'s cap is 1<<20 runs, so
//! a toy tensor lowers to a copy list and would prove nothing about the path a
//! vision tower actually takes; the algebra-level cases live in `algebra.rs`,
//! where the cap can be set small enough to force the choice.

use checkpoint::contract::compile::{Lowering, compile};
use checkpoint::contract::infer::infer_type;
use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageInstr, StorageTarget};
use checkpoint::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

/// The vision tower's patch-embed bank: `hidden` rows of `C * T * P²` lanes.
/// Stored the way an MLX conv3d is — channels LAST, `[hidden, T, P, P, C]` —
/// which is the same element count as the channel-major bank the plan reads
/// and, until there was a gather, the same bytes read down the wrong axis.
const HIDDEN: i64 = 768;
const LANES: i64 = 1536;
const CHANNELS: i64 = 3;
const STORED: [i64; 5] = [HIDDEN, 2, 16, 16, CHANNELS];

/// Element `(r, k)` as a `bf16` bit pattern, distinct within a row.
///
/// The expectation is built by permuting these same bytes rather than by
/// re-deriving values, so a collision across rows cannot hide a mistake — but
/// mixing two rows up still changes the answer, because the pattern advances
/// with `r` too.
fn element(r: i64, k: i64) -> u16 {
    (r.wrapping_mul(2_654_435_761).wrapping_add(k.wrapping_mul(40_503)) & 0xffff) as u16
}

fn bank(dir: &std::path::Path) -> (Metadata, Vec<u8>) {
    let mut file = Vec::with_capacity((HIDDEN * LANES * 2) as usize);
    for r in 0..HIDDEN {
        for k in 0..LANES {
            file.extend_from_slice(&element(r, k).to_le_bytes());
        }
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), &file).unwrap();
    let span_bytes = file.len() as u64;
    (
        Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes: span_bytes,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "patch_embed.proj.weight".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes,
                shape: STORED.to_vec(),
                encoding: Encoding::Raw(DType::Bf16),
            }],
        },
        file,
    )
}

/// `out[.., c * per + j] = src[.., j * C + c]`: the channels-last conv bank an
/// MLX submission ships, read as the channel-major matmul bank the plan wants.
fn deinterleave() -> Vec<i64> {
    let per = LANES / CHANNELS;
    (0..CHANNELS)
        .flat_map(|c| (0..per).map(move |j| j * CHANNELS + c))
        .collect()
}

fn contract() -> ModelContract {
    ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "patch_embed",
            Expr::src("patch_embed.proj.weight")
                .transmute(TensorType::raw(
                    vec![HIDDEN, LANES],
                    DType::Bf16,
                ))
                .gather(1, deinterleave()),
            vec![HIDDEN, LANES],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    }
}

/// What the permutation means, worked out from the file's own bytes.
fn expected(file: &[u8]) -> Vec<u8> {
    let indices = deinterleave();
    let mut out = Vec::with_capacity(file.len());
    for r in 0..HIDDEN {
        for index in &indices {
            let at = ((r * LANES + index) * 2) as usize;
            out.extend_from_slice(&file[at..at + 2]);
        }
    }
    out
}

/// The whole point: the bytes come back permuted, exactly.
#[test]
fn a_channels_last_bank_gathers_into_a_channel_major_one() {
    let dir = std::env::temp_dir().join(format!("pie_gather_{}", std::process::id()));
    let (metadata, file) = bank(&dir);
    let contract = contract();
    let plan = checkpoint::plan::compile(
        &metadata,
        &contract,
        StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .unwrap();

    // One instruction, carrying one table of LANES indices — not 1.18M runs,
    // and not the 3069 rectangles the copy list folds to.
    let tables: Vec<_> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::GatherWrite { gather, .. } => Some(gather),
            _ => None,
        })
        .collect();
    assert_eq!(tables.len(), 1, "one gather");
    assert_eq!(tables[0].indices, deinterleave());
    assert_eq!(tables[0].block_bytes, 2, "one bf16 element per index");
    assert_eq!(tables[0].rows, HIDDEN as u64);
    assert_eq!(tables[0].src_row_bytes, (LANES * 2) as u64);
    assert!(
        !plan
            .instrs
            .iter()
            .any(|instr| matches!(instr, StorageInstr::ExtentWrite { .. })),
        "the gather replaces the copy list, it does not accompany it"
    );

    let out = checkpoint::executor::Execution::new(&plan, &dir)
        .run()
        .unwrap()
        .tensors
        .remove("patch_embed")
        .unwrap();
    let want = expected(&file);
    assert_eq!(out.len(), want.len());
    assert_eq!(out, want, "the gathered bank is not the permuted bank");
    std::fs::remove_dir_all(&dir).ok();
}

/// The lowering itself, without a plan around it: a gather over a whole
/// tensor becomes a table, and the table is the one the contract wrote.
#[test]
fn the_deinterleave_lowers_to_a_table_and_not_a_copy_list() {
    let (metadata, _) = {
        let dir = std::env::temp_dir().join(format!("pie_gather_low_{}", std::process::id()));
        let out = bank(&dir);
        std::fs::remove_dir_all(&dir).ok();
        out
    };
    let expr = contract().tensors[0].expr.clone();
    let (_, checked) = infer_type(&expr, &metadata).unwrap();

    // The cap is what routes it: a copy list is tried first, always.
    let Lowering::Gather(table) = compile(&expr, &checked, 1 << 20).unwrap() else {
        panic!("the deinterleave lowered to a copy list");
    };
    assert_eq!(table.indices, deinterleave());
    assert_eq!(table.block, 1, "the gather is on the innermost axis");
    assert_eq!(table.rows, HIDDEN);
    assert_eq!(table.src_row, LANES);
    assert_eq!(table.elements, HIDDEN * LANES);
    assert_eq!(table.dst_row(), LANES);

    // And it costs what a single pass over the bytes costs.
    assert_eq!(Lowering::Gather(table).cost(), 1);
}

/// An expression that is not a gather still gets the refusal, because there is
/// nothing else for it to be. The message names what it would take.
#[test]
fn a_fragmented_expression_that_is_not_a_gather_is_still_refused() {
    let (metadata, _) = {
        let dir = std::env::temp_dir().join(format!("pie_gather_ref_{}", std::process::id()));
        let out = bank(&dir);
        std::fs::remove_dir_all(&dir).ok();
        out
    };
    let expr = Expr::src("patch_embed.proj.weight")
        .transmute(TensorType::raw(vec![HIDDEN, LANES], DType::Bf16))
        .stride(1, 0, LANES / 2, 2);
    let (_, checked) = infer_type(&expr, &metadata).unwrap();
    let err = compile(&expr, &checked, 16).unwrap_err().to_string();
    assert!(err.contains("not a copy list"), "{err}");
}
