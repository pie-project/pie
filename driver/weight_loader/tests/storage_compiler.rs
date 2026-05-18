use pie_weight_loader::ir::{LayoutExpr, LayoutPlan};
use pie_weight_loader::source::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_weight_loader::storage::{StorageInstr, StorageTarget, TileMapKind};
use pie_weight_loader::storage_compiler::lower_layout_plan;
use pie_weight_loader::types::{
    CheckpointFormat, DType, Encoding, FileId, Layout, Sharding, TensorDecl, TensorId,
};

#[test]
fn buffer_join_tile_maps_carry_destination_offsets() {
    let mut plan = LayoutPlan::new();
    let a_decl = decl(0, "a", &[2], Encoding::Raw(DType::F32));
    let b_decl = decl(1, "b", &[2], Encoding::Raw(DType::F32));
    let a = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: a_decl.clone(),
    });
    let b = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: b_decl.clone(),
    });
    let a_cast_decl = decl(2, "a.cast", &[2], Encoding::Raw(DType::BF16));
    let b_cast_decl = decl(3, "b.cast", &[2], Encoding::Raw(DType::BF16));
    let a_cast = plan.push(LayoutExpr::Cast {
        input: a,
        dtype: DType::BF16,
        decl: a_cast_decl,
    });
    let b_cast = plan.push(LayoutExpr::Cast {
        input: b,
        dtype: DType::BF16,
        decl: b_cast_decl,
    });
    let joined_decl = decl(4, "joined", &[4], Encoding::Raw(DType::BF16));
    let joined = plan.push(LayoutExpr::Join {
        inputs: vec![a_cast, b_cast],
        axis: pie_weight_loader::types::Axis(0),
        decl: joined_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: joined,
        runtime_name: "joined".to_string(),
        decl: joined_decl,
    });
    plan.outputs.push(realized);

    let program = lower_layout_plan(&metadata(), &plan, StorageTarget::default()).unwrap();
    let reblocks: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Reblock,
                dest,
                ..
            } => dest.as_ref(),
            _ => None,
        })
        .collect();

    assert_eq!(reblocks.len(), 2);
    assert_eq!(reblocks[0].offset, 0);
    assert_eq!(reblocks[1].offset, 4);
    assert_eq!(reblocks[0].stride.element_bytes, 2);
    assert_eq!(reblocks[1].stride.element_bytes, 2);
}

fn metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 16,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw(0, "a", 0, &[2], DType::F32),
            raw(1, "b", 8, &[2], DType::F32),
        ],
    }
}

fn raw(id: u32, name: &str, offset: u64, shape: &[i64], dtype: DType) -> RawTensor {
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes: 8,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(dtype),
        layout: Layout::dense(1),
    }
}

fn decl(id: u32, name: &str, shape: &[i64], encoding: Encoding) -> TensorDecl {
    TensorDecl {
        id: TensorId(id),
        name: name.to_string(),
        shape: shape.to_vec(),
        encoding,
        layout: Layout::dense(1),
        sharding: Sharding::replicated(),
        alignment: 1,
    }
}
