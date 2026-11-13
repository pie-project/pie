//! A bf16 bank encoded to NVFP4 on the host lands the planes the NVFP4
//! codec writes: the codes, an e4m3 scale per 16 of them (`.scales`), and an
//! f32 tensor scale per row (`.global_scales`), attached as the bank's
//! scales and second companion.

use checkpoint::codec::nvfp4;
use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget, compile};
use checkpoint::types::{DType, Encoding, QuantScheme, QuantSpec};

#[test]
fn an_nvfp4_encode_lands_the_codec_planes() {
    let (experts, rows, cols) = (3usize, 40usize, 96usize);
    let value = |at: usize| ((at * 37 % 101) as f32 - 50.0) / (7.0 + (at / cols % 5) as f32);
    let bf16: Vec<half::bf16> = (0..experts * rows * cols)
        .map(|at| half::bf16::from_f32(value(at)))
        .collect();
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut writer = ztensor::Writer::create(dir.path().join("model.zt")).expect("it opens");
    let bytes: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = vec![experts as u64, rows as u64, cols as u64];
    writer
        .add("w.weight", shape, ztensor::Leaf::BF16, &bytes)
        .expect("the bank lands");
    writer.finish().expect("it closes");

    let spec = QuantSpec {
        scheme: QuantScheme::Nvfp4E2M1E4M3,
        logical_dtype: DType::Bf16,
        bits_per_element: 4,
        group_size: 16,
        channel_axis: None,
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src("w.weight").cast(Encoding::Quant(spec.clone())),
            vec![experts as i64, rows as i64, cols as i64],
            Encoding::Quant(spec),
        )],
        groups: Vec::new(),
    };
    let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");
    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    let plan = compile(&metadata, &contract, target).expect("the encode compiles");
    let attach = plan.attachments[0];
    let name = |id: checkpoint::types::TensorId| plan.tensors[id.0 as usize].name.clone();
    assert_eq!(name(attach.scale_tensor), "w.scales");
    let tensor = attach.zero_point_tensor.expect("a tensor scale plane");
    assert_eq!(name(tensor), "w.global_scales");
    assert_eq!(
        plan.tensors[tensor.0 as usize].shape,
        vec![experts as i64, rows as i64]
    );

    let storage = Execution::new(&plan, dir.path())
        .run()
        .expect("the encode runs on the host");
    let values: Vec<f32> = bf16.iter().map(|v| v.to_f32()).collect();
    let want = nvfp4::encode(&values, experts * rows, cols).expect("the codec encodes");
    assert_eq!(storage.tensors["w"], want.codes, "codes");
    assert_eq!(storage.tensors["w.scales"], want.scales, "e4m3 scales");
    let tensor_bytes: Vec<u8> = want.tensor.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        storage.tensors["w.global_scales"], tensor_bytes,
        "tensor scales"
    );
}
