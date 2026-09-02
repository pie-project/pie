//! Publishing: a reader never sees a half-written file.
//!
//! The spec puts durable publication in the transport's hands (Appendix B),
//! and it is right that the format does not mandate an fsync. But every
//! producer needs the same three steps: write beside the target, sync,
//! rename, and every producer was writing them again. So the library writes
//! them once.

use std::path::{Path, PathBuf};

use ztensor::{Leaf, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn siblings(path: &Path) -> Vec<String> {
    let dir = path.parent().unwrap();
    let stem = path.file_name().unwrap().to_string_lossy().into_owned();
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(&stem))
        .collect()
}

fn read_t(path: &Path) -> Vec<u8> {
    Source::open(path)
        .unwrap()
        .tensor("t")
        .unwrap()
        .bytes()
        .unwrap()
        .into_owned()
}

#[test]
fn nothing_is_at_the_path_until_finish() {
    let path = tmp("published.zt");
    let _ = std::fs::remove_file(&path);

    let mut w = Writer::publish(&path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    assert!(
        !path.exists(),
        "the target must not exist while the file is still being written"
    );
    assert!(
        siblings(&path).iter().any(|n| n.ends_with(".partial")),
        "the bytes have to be going somewhere"
    );

    w.finish().unwrap();
    assert!(path.exists());
    assert_eq!(read_t(&path), [1, 2, 3, 4]);
    assert!(
        !siblings(&path).iter().any(|n| n.ends_with(".partial")),
        "the partial file must not survive publication"
    );
}

/// The crash case, as far as a test can stage it: a writer that goes away
/// without finishing leaves neither a target nor a partial.
#[test]
fn dropping_an_unfinished_publisher_leaves_nothing() {
    let path = tmp("abandoned.zt");
    let _ = std::fs::remove_file(&path);
    {
        let mut w = Writer::publish(&path).unwrap();
        w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    }
    assert!(!path.exists());
    assert!(siblings(&path).is_empty(), "{:?}", siblings(&path));
}

#[test]
fn abandon_removes_the_partial() {
    let path = tmp("abandon-explicit.zt");
    let _ = std::fs::remove_file(&path);
    let mut w = Writer::publish(&path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.abandon();
    assert!(!path.exists());
    assert!(siblings(&path).is_empty());
}

/// Publishing over an existing file replaces it in one step: readers see the
/// old bytes or the new ones, never a mixture.
#[test]
fn publishing_replaces_atomically() {
    let path = tmp("replaced.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [1u64], Leaf::U8, &[1]).unwrap();
    w.finish().unwrap();

    let mut w = Writer::publish(&path).unwrap();
    w.add("t", [1u64], Leaf::U8, &[2]).unwrap();
    // The old file is still the whole truth at this point.
    assert_eq!(read_t(&path), [1]);
    w.finish().unwrap();
    assert_eq!(read_t(&path), [2]);
}

/// A plain `create` still writes in place, for the callers who do not want a
/// second file to appear in their directory.
#[test]
fn create_writes_in_place() {
    let path = tmp("in-place.zt");
    let _ = std::fs::remove_file(&path);
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [1u64], Leaf::U8, &[1]).unwrap();
    assert!(path.exists(), "create() writes where it says it does");
    w.finish().unwrap();
    assert!(siblings(&path).iter().all(|n| !n.ends_with(".partial")));
}
