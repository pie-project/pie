use checkpoint::codec::fp8::fp8_e4m3_to_f32;
use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile_streaming};
use checkpoint::types::{DType, Encoding};

// 160 rows under two factors of 128: the second block is short, as a
// latent projection's 576 rows are under five.
const ROWS: usize = 160;
const COLS: usize = 256;
const FACTORS: [f32; 4] = [0.5, 2.0, 0.125, 3.0];

fn code(row: usize, col: usize) -> u8 {
    // Every e4m3 code but the two NaNs, spread over the plane.
    let code = ((row * 31 + col * 7) % 254) as u8;
    if code >= 0x7f { code + 1 } else { code }
}

fn write_checkpoint(dir: &std::path::Path) {
    let weight: Vec<u8> = (0..ROWS * COLS).map(|i| code(i / COLS, i % COLS)).collect();
    let factors: Vec<u8> = FACTORS.iter().flat_map(|f| f.to_le_bytes()).collect();
    let header = format!(
        "{{\"w.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{ROWS},{COLS}],\"data_offsets\":[0,{}]}},\
         \"w.weight_scale_inv\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[{},{}]}}}}",
        weight.len(),
        weight.len(),
        weight.len() + factors.len(),
    );
    let mut header = header.into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&weight);
    file.extend_from_slice(&factors);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

#[test]
fn a_block_scaled_fp8_plane_lands_as_bf16() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let metadata = parse_metadata(dir.path()).unwrap();
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src("w.weight").cast(Encoding::Raw(DType::Bf16)),
            vec![ROWS as i64, COLS as i64],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    };
    let plan = compile_streaming(&metadata, &contract, StorageTarget::default())
        .expect("a bf16 declaration over a block-scaled FP8 source compiles");
    let mut sink = MemorySink::default();
    Execution::new(&plan, dir.path())
        .streaming()
        .sink(&mut sink)
        .run()
        .expect("the FP8 plane is dequantized on the host");

    let landed = &sink.tensors["w"];
    assert_eq!(landed.len(), ROWS * COLS * 2, "the plane did not land bf16");
    for row in 0..ROWS {
        for col in 0..COLS {
            let factor = FACTORS[(row / 128) * 2 + col / 128];
            let want = half::bf16::from_f32(fp8_e4m3_to_f32(code(row, col)) * factor);
            let at = (row * COLS + col) * 2;
            let got = half::bf16::from_bits(u16::from_le_bytes([landed[at], landed[at + 1]]));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "w[{row}, {col}] is not its code times its block's factor"
            );
        }
    }
}
