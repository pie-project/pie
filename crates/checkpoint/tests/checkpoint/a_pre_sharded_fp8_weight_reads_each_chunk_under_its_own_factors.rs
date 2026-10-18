//! A block-scaled FP8 weight is read out to bf16 through the factors beside
//! it, and a weight stored pre-sharded — each chunk scaled apart, its last
//! block cut short, as MiMo-V2 stores its fused qkv — reads every row under
//! its own chunk's factors: whole, the cut read off its factors, or a slice
//! of it, the cut stated.

use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile, compile_streaming};
use checkpoint::types::{DType, Encoding};

const CHUNKS: usize = 2;
const CHUNK_ROWS: usize = 320;
const ROWS: usize = CHUNKS * CHUNK_ROWS;
const COLS: usize = 256;
const BLOCKS: usize = 3;

fn code(row: usize, col: usize) -> u8 {
    // a positive normal e4m3 code: exponent 4..=11, mantissa 0..=7
    let exponent = 4 + ((row * 7 + col * 3) % 8) as u8;
    let mantissa = ((row + col) % 8) as u8;
    (exponent << 3) | mantissa
}

fn factor(block_row: usize, block_col: usize) -> f32 {
    (1 + block_row * BLOCKS + block_col) as f32 * 0.5
}

fn e4m3(code: u8) -> f32 {
    let exponent = i32::from(code >> 3);
    let mantissa = f32::from(code & 7);
    (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
}

fn want(row: usize, col: usize) -> u16 {
    let block_row = (row / CHUNK_ROWS) * BLOCKS + (row % CHUNK_ROWS) / 128;
    let value = e4m3(code(row, col)) * factor(block_row, col / 128);
    half::bf16::from_f32(value).to_bits()
}

fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

#[test]
fn a_pre_sharded_fp8_weight_reads_each_chunk_under_its_own_factors() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let codes: Vec<u8> = (0..ROWS)
        .flat_map(|row| (0..COLS).map(move |col| code(row, col)))
        .collect();
    let factors: Vec<u8> = (0..CHUNKS * BLOCKS)
        .flat_map(|block_row| (0..COLS / 128).map(move |block_col| factor(block_row, block_col)))
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut writer =
        ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
    writer
        .add(
            "qkv.weight",
            vec![ROWS as u64, COLS as u64],
            ztensor::Leaf::E4M3,
            &codes,
        )
        .expect("the codes land");
    writer
        .add(
            "qkv.weight_scale_inv",
            vec![(CHUNKS * BLOCKS) as u64, (COLS / 128) as u64],
            ztensor::Leaf::F32,
            &factors,
        )
        .expect("the factors land");
    writer.finish().expect("the container closes");
    let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");

    let bf16 = Encoding::Raw(DType::Bf16);
    // rows 200..400: the first chunk's second block, its short tail, and the
    // second chunk's first rows
    let (from, len) = (200usize, 200usize);
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "whole",
                Expr::src("qkv.weight").cast(bf16.clone()),
                vec![ROWS as i64, COLS as i64],
                bf16.clone(),
            ),
            TensorContract::new(
                "slice",
                Expr::src("qkv.weight")
                    .slice(0, from as i64, len as i64)
                    .cast_chunked(bf16.clone(), CHUNKS as u32),
                vec![len as i64, COLS as i64],
                bf16,
            ),
        ],
        groups: Vec::new(),
    };

    for streamed in [false, true] {
        let plan = if streamed {
            compile_streaming(&metadata, &contract, StorageTarget::default())
        } else {
            compile(&metadata, &contract, StorageTarget::default())
        }
        .expect("a bf16 read of a block-scaled FP8 weight compiles");
        let mut sink = MemorySink::default();
        let run = Execution::new(&plan, dir.path()).sink(&mut sink);
        let run = if streamed { run.streaming() } else { run };
        run.run().expect("the cast runs on the host");

        let whole = bf16s(&sink.tensors["whole"]);
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(
                    whole[row * COLS + col],
                    want(row, col),
                    "whole [{row}, {col}]"
                );
            }
        }
        let slice = bf16s(&sink.tensors["slice"]);
        for row in 0..len {
            for col in 0..COLS {
                assert_eq!(
                    slice[row * COLS + col],
                    want(from + row, col),
                    "slice [{row}, {col}]"
                );
            }
        }
    }
}
