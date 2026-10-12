use std::path::PathBuf;

use ztensor::{Leaf, Rule, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn file(name: &str, tensor: &str, bytes: &[u8]) -> PathBuf {
    let path = tmp(name);
    let mut w = Writer::create(&path).unwrap();
    w.add(tensor, [bytes.len() as u64], Leaf::U8, bytes)
        .unwrap();
    w.finish().unwrap();
    path
}

#[test]
fn merge_every_case() {
    tensors_are_one_name_space_that_remembers_its_files();
    offsets_stay_relative_to_their_own_file();
    a_name_in_two_files_is_refused();
    capabilities_are_the_holding_file_s();
    a_merged_set_has_no_manifest();
    an_absent_tensor_is_not_found();
}

fn tensors_are_one_name_space_that_remembers_its_files() {
    let a = vec![1u8; 64];
    let b = vec![2u8; 32];
    let second = file("merge-second.zt", "layer.1.weight", &b);
    let first = file("merge-first.zt", "layer.0.weight", &a);
    let src = Source::open_all(&[second.clone(), first.clone()]).unwrap();

    let names: Vec<&str> = src.names().collect();
    assert_eq!(names, vec!["layer.0.weight", "layer.1.weight"]);

    let at0 = src.tensor("layer.0.weight").unwrap().locate().unwrap();
    let at1 = src.tensor("layer.1.weight").unwrap().locate().unwrap();
    assert_eq!(src.store(at0.store).path(), first);
    assert_eq!(src.store(at1.store).path(), second);

    assert_eq!(src.tensor("layer.0.weight").unwrap().map().unwrap(), &a[..]);
    assert_eq!(src.tensor("layer.1.weight").unwrap().map().unwrap(), &b[..]);
}

fn offsets_stay_relative_to_their_own_file() {
    let a = file("merge-off-a.zt", "w.a", &[7u8; 100]);
    let b = file("merge-off-b.zt", "w.b", &[8u8; 100]);
    let src = Source::open_all(&[a, b]).unwrap();

    let at_a = src.tensor("w.a").unwrap().locate().unwrap();
    let at_b = src.tensor("w.b").unwrap().locate().unwrap();
    assert_eq!(at_a.offset, at_b.offset);
    assert_ne!(at_a.store, at_b.store);
    assert!(at_a.offset >= 64 * 1024, "not page-placed");
}

fn a_name_in_two_files_is_refused() {
    let a = file("merge-dup-a.zt", "shared", &[1u8; 8]);
    let b = file("merge-dup-b.zt", "shared", &[2u8; 8]);
    let err = Source::open_all(&[a, b]).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::NameCollision));
    let message = format!("{err}");
    assert!(message.contains("shared"), "unhelpful message: {message}");
    assert!(
        message.contains("merge-dup-a") && message.contains("merge-dup-b"),
        "the message must name both files: {message}"
    );
}

fn capabilities_are_the_holding_file_s() {
    let path = file("merge-caps.zt", "w", &[3u8; 128]);
    let alone = Source::open(&path).unwrap();
    let direct = alone.tensor("w").unwrap().caps();

    let merged = Source::open_all(&[path]).unwrap();
    assert_eq!(merged.tensor("w").unwrap().caps(), direct);
}

fn a_merged_set_has_no_manifest() {
    let a = file("merge-manifest-a.zt", "a", &[1u8; 8]);
    let b = file("merge-manifest-b.zt", "b", &[2u8; 8]);
    assert!(Source::open(&a).unwrap().provenance().as_root().is_some());
    assert!(Source::open_all(&[a, b])
        .unwrap()
        .provenance()
        .as_root()
        .is_none());
}

fn an_absent_tensor_is_not_found() {
    let src = Source::open_all(&[file("merge-missing.zt", "w", &[1u8; 8])]).unwrap();
    assert!(src.get("w").is_some());
    assert!(src.get("nope").is_none());
    assert!(src.tensor("nope").is_err());
}
