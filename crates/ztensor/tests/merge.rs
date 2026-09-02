//! Several self-describing files read as one.
//!
//! Every foreign snapshot arrives in this shape. A sharded safetensors
//! checkpoint is N complete files with an index beside them, and
//! it is not the shape `sharding.rs` covers. There, a root states each shard's
//! size and digest and opening the model checks them. Here nothing binds the
//! set but the caller's list, so nothing is verified and nothing pretends to
//! be: what a merge adds is one name space and the record of which file each
//! name came from.
//!
//! Both are the same `Source` type, because the difference is in how the
//! catalog was built, not in what a consumer can ask of it.

use std::path::PathBuf;

use ztensor::{Leaf, Rule, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Writes a one-tensor file and returns its path.
fn file(name: &str, tensor: &str, bytes: &[u8]) -> PathBuf {
    let path = tmp(name);
    let mut w = Writer::create(&path).unwrap();
    w.add(tensor, [bytes.len() as u64], Leaf::U8, bytes).unwrap();
    w.finish().unwrap();
    path
}

#[test]
fn tensors_are_one_name_space_that_remembers_its_files() {
    let a = vec![1u8; 64];
    let b = vec![2u8; 32];
    // Deliberately listed out of order.
    let second = file("merge-second.zt", "layer.1.weight", &b);
    let first = file("merge-first.zt", "layer.0.weight", &a);
    let src = Source::open_all(&[second.clone(), first.clone()]).unwrap();

    // Sorted across the whole set, so the order does not depend on how the
    // caller happened to list the files.
    let names: Vec<&str> = src.names().collect();
    assert_eq!(names, vec!["layer.0.weight", "layer.1.weight"]);

    // And each name still knows where it came from.
    let at0 = src.tensor("layer.0.weight").unwrap().locate().unwrap();
    let at1 = src.tensor("layer.1.weight").unwrap().locate().unwrap();
    assert_eq!(src.store(at0.store).path(), first);
    assert_eq!(src.store(at1.store).path(), second);

    assert_eq!(src.tensor("layer.0.weight").unwrap().map().unwrap(), &a[..]);
    assert_eq!(src.tensor("layer.1.weight").unwrap().map().unwrap(), &b[..]);
}

/// An offset belongs to the file that holds it, which is why the pair
/// (store, offset) a complete address and a bare offset a meaningless one.
#[test]
fn offsets_stay_relative_to_their_own_file() {
    let a = file("merge-off-a.zt", "w.a", &[7u8; 100]);
    let b = file("merge-off-b.zt", "w.b", &[8u8; 100]);
    let src = Source::open_all(&[a, b]).unwrap();

    let at_a = src.tensor("w.a").unwrap().locate().unwrap();
    let at_b = src.tensor("w.b").unwrap().locate().unwrap();
    // Both files were written the same way, so both payloads land at the same
    // offset, which is only coherent because the offset belongs to the file.
    assert_eq!(at_a.offset, at_b.offset);
    assert_ne!(at_a.store, at_b.store);
    assert!(at_a.offset >= 64 * 1024, "not page-placed");
}

/// A name in two files is a broken set, and there is no rule that would make
/// it whole: picking a winner would load half a model and say nothing.
#[test]
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

/// A merge adds no capability and takes none away: each file is read exactly
/// as it would be alone.
#[test]
fn capabilities_are_the_holding_file_s() {
    let path = file("merge-caps.zt", "w", &[3u8; 128]);
    let alone = Source::open(&path).unwrap();
    let direct = alone.tensor("w").unwrap().caps();

    let merged = Source::open_all(&[path]).unwrap();
    assert_eq!(merged.tensor("w").unwrap().caps(), direct);
}

/// A merged set is not one file, so it has no manifest. There is no single
/// document any of these files wrote.
#[test]
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

#[test]
fn an_absent_tensor_is_not_found() {
    let src = Source::open_all(&[file("merge-missing.zt", "w", &[1u8; 8])]).unwrap();
    assert!(src.get("w").is_some());
    assert!(src.get("nope").is_none());
    assert!(src.tensor("nope").is_err());
}
