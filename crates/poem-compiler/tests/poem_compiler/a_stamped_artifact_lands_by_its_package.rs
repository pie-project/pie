use std::collections::BTreeMap;

use checkpoint::file::emit::{self, Object};
use checkpoint::serving::{self, Stamp};
use poem_compiler::catalog::repository;

const MODEL: &str = "qwen3-sim-bf16-kv-bf16";

#[test]
fn a_stamped_artifact_lands_by_its_package_and_a_raw_checkpoint_by_its_import() {
    let dir = std::env::temp_dir().join(format!("pie_contract_for_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let trace = repository()
        .parse(MODEL)
        .expect(MODEL)
        .trace(poem::Platform::Cerebras);

    let stamped = dir.join("stamped.zt");
    let stamp = Stamp {
        serving: serving::PROFILE.to_string(),
        backend: "cerebras".to_string(),
        deployment: MODEL.to_string(),
        layout_revision: serving::LAYOUT_REVISION,
        adapters_zeroed: true,
        package: "no_such_package".to_string(),
        package_digest: "0".repeat(64),
    };
    let norm = [0u8; 64];
    let objects = [Object::leaf("norm", vec![64], ztensor::Leaf::U8, &norm)];
    emit::write(
        &stamped,
        &stamp,
        &BTreeMap::new(),
        4096,
        &objects,
        |_, _, _| Ok(()),
    )
    .unwrap();
    let why = repository().contract_for(&trace, &stamped).unwrap_err();
    assert!(
        why.contains("imported with the package `no_such_package`"),
        "{why}"
    );

    let raw = dir.join("raw.zt");
    let writer = ztensor::Writer::create(&raw).unwrap();
    writer.finish().unwrap();
    let why = repository().contract_for(&trace, &raw).unwrap_err();
    assert!(why.contains("the import contract for"), "{why}");
    let _ = std::fs::remove_dir_all(&dir);
}
