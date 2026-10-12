use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::plan::{StorageInstr, StorageTarget, compile};
use checkpoint::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

const HEADS: i64 = 2;
const DIM: i64 = 4;
const ROW_BYTES: u64 = DIM as u64 * 4;

fn metadata() -> Metadata {
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: HEADS as u64 * ROW_BYTES,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "k_proj".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: HEADS as u64 * ROW_BYTES,
            shape: vec![HEADS, DIM],
            encoding: Encoding::Raw(DType::F32),
        }],
    }
}

fn read_offset(rank: u32, world: u32) -> u64 {
    // two kv heads over four ranks: each head is held by a group of two
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "k_proj",
            Expr::src("k_proj").shard_among(0, 2),
            vec![HEADS, DIM],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        tp_rank: rank,
        tp_size: world,
        ..StorageTarget::default()
    };
    let plan = compile(&metadata(), &contract, target)
        .unwrap_or_else(|why| panic!("rank {rank} of {world} does not land: {why}"));
    let reads: Vec<_> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::BulkExtentWrite { source, .. }
            | StorageInstr::ExtentWrite { source, .. } => Some(source),
            _ => None,
        })
        .collect();
    assert_eq!(reads.len(), 1, "rank {rank}: {reads:#?}");
    assert_eq!(reads[0].span_bytes, ROW_BYTES, "rank {rank} reads one head");
    reads[0].file_offset
}

#[test]
fn each_group_of_two_ranks_reads_its_own_head() {
    let offsets: Vec<u64> = (0..4).map(|rank| read_offset(rank, 4)).collect();
    assert_eq!(offsets, vec![0, 0, ROW_BYTES, ROW_BYTES]);
}
