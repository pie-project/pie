//! A contract carries a constant itself, and the plan lands it beside the
//! checkpoint's planes: both through the arena and streamed, and joined to a
//! plane the checkpoint holds.

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile, compile_streaming};
use checkpoint::types::{DType, Encoding};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn a_constant_lands_as_the_bytes_it_carries() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let held = [4.0f32, 5.0, 6.0];
    let mut writer =
        ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
    writer
        .add("held", vec![3], ztensor::Leaf::F32, &bytes(&held))
        .expect("the plane lands");
    writer.finish().expect("the container closes");
    let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");

    let stated = [1.0f32, -1.0, 0.5];
    let constant = || Expr::constant(TensorType::raw(vec![3], DType::F32), bytes(&stated));
    let raw = Encoding::Raw(DType::F32);
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new("alone", constant(), vec![3], raw.clone()),
            TensorContract::new(
                "joined",
                Expr::concat(0, vec![constant(), Expr::src("held")]),
                vec![6],
                raw,
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
        .expect("a contract carrying a constant compiles");
        let mut sink = MemorySink::default();
        let run = Execution::new(&plan, dir.path()).sink(&mut sink);
        let run = if streamed { run.streaming() } else { run };
        run.run().expect("the plan executes");
        assert_eq!(f32s(&sink.tensors["alone"]), stated, "streamed: {streamed}");
        assert_eq!(
            f32s(&sink.tensors["joined"]),
            [stated.as_slice(), held.as_slice()].concat(),
            "streamed: {streamed}"
        );
    }
}

#[test]
fn a_constant_short_of_its_shape_is_refused() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "short",
            Expr::constant(TensorType::raw(vec![3], DType::F32), vec![0; 8]),
            vec![3],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut writer =
        ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
    writer
        .add("unread", vec![1], ztensor::Leaf::F32, &[0u8; 4])
        .expect("the plane lands");
    writer.finish().expect("the container closes");
    let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");
    let why = compile(&metadata, &contract, StorageTarget::default())
        .expect_err("eight bytes are not three f32s");
    assert!(
        format!("{why}").contains("is 12 bytes and carries 8"),
        "{why}"
    );
}
