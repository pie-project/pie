//! Adding to a finished `.zt` without rewriting it.
//!
//! The interesting cases are not "does it round-trip" but the ones where the
//! file ends up subtly wrong: an original byte moved, an alignment quietly
//! dropped, an append onto an append, a name already taken. The first two are
//! bugs this code shipped with, and both are invisible on the machine that
//! wrote the file.

use std::fs;
use std::path::PathBuf;

use ztensor::read::shard_identity;
use ztensor::{DigestAlgorithm, Error, Leaf, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn offset_of(src: &Source, name: &str) -> u64 {
    src.tensor(name).unwrap().locate().unwrap().offset
}

/// Spec §2.5: "Writers MUST NOT truncate or overwrite existing bytes."
///
/// The strong form of that is what this checks: the original file is a
/// *prefix* of the appended one, byte for byte, including its old manifest and
/// old footer. Anything weaker would let an append reclaim the old manifest,
/// which is what makes a crashed append unrecoverable.
#[test]
fn the_original_file_is_a_prefix_of_the_appended_one() {
    let path = tmp("append-basic.zt");
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("a", [4u64], Leaf::F32, &a).unwrap();
    w.finish().unwrap();

    let original = fs::read(&path).unwrap();
    let a_off = offset_of(&Source::open(&path).unwrap(), "a");

    let b = f32s(&[9.0; 8]);
    let mut w = Writer::append(&path).unwrap();
    w.add("b", [8u64], Leaf::F32, &b).unwrap();
    w.finish().unwrap();

    let after = fs::read(&path).unwrap();
    assert!(after.len() > original.len(), "an append only extends");
    assert_eq!(
        &after[..original.len()],
        &original[..],
        "every original byte, including the old manifest and footer, must survive"
    );

    let src = Source::open(&path).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["a", "b"]);
    assert_eq!(src.tensor("a").unwrap().map().unwrap(), &a[..]);
    assert_eq!(src.tensor("b").unwrap().map().unwrap(), &b[..]);
    assert_eq!(offset_of(&src, "a"), a_off);
}

/// A crashed append is undone by truncating back to the old length, which only
/// works because the old footer is still sitting there untouched.
#[test]
fn a_half_finished_append_is_undone_by_truncating() {
    let path = tmp("append-crash.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("a", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.finish().unwrap();
    let original = fs::read(&path).unwrap();

    // Simulate a crash: bytes written, no footer.
    let mut w = Writer::append(&path).unwrap();
    w.add("b", [4096u64], Leaf::F32, &vec![9u8; 16384]).unwrap();
    drop(w); // never finished

    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(original.len() as u64)
        .unwrap();

    assert_eq!(fs::read(&path).unwrap(), original);
    let src = Source::open(&path).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["a"]);
}

/// The alignment the file was written at is carried forward.
///
/// Without this a 64 KiB file silently becomes a mixed file: the tensors added
/// later land on 4 KiB boundaries and lose page exclusivity on any host whose
/// pages are larger than that, which is every Apple Silicon machine and some
/// ARM servers. The writer that made the file would never see it.
#[test]
fn an_append_keeps_the_files_alignment() {
    let path = tmp("append-align.zt");
    let mut w = Writer::create(&path).unwrap(); // canonical: 64 KiB
    w.add("a", [1024u64], Leaf::F32, &vec![1u8; 4096]).unwrap();
    w.finish().unwrap();

    let mut w = Writer::append(&path).unwrap();
    w.add("b", [1024u64], Leaf::F32, &vec![2u8; 4096]).unwrap();
    w.add("c", [1024u64], Leaf::F32, &vec![3u8; 4096]).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    for name in ["a", "b", "c"] {
        let at = offset_of(&src, name);
        assert_eq!(
            at % ztensor::format::ALIGN_CANONICAL,
            0,
            "{name} landed at {at}, off the file's 64 KiB grid"
        );
    }

    // And an explicit request still wins.
    let coarse = tmp("append-align-explicit.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&coarse)
        .unwrap();
    w.add("a", [16u64], Leaf::F32, &f32s(&[1.0; 16])).unwrap();
    w.finish().unwrap();
    let mut w = Writer::options()
        .canonical(false)
        .align(65536)
        .append(&coarse)
        .unwrap();
    w.add("b", [16u64], Leaf::F32, &f32s(&[2.0; 16])).unwrap();
    w.finish().unwrap();
    let src = Source::open(&coarse).unwrap();
    assert_eq!(offset_of(&src, "b") % 65536, 0);
}

/// The footer ends the file. A reader finds the manifest no other way, so
/// this is the property every append has to leave standing.
#[test]
fn the_footer_still_ends_the_file() {
    let path = tmp("append-shorter.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    // Many long names make the first manifest large.
    for i in 0..64 {
        let name = format!("a.very.long.tensor.name.that.pads.the.manifest.{i:04}");
        w.add(&name, [2u64], Leaf::F32, &f32s(&[1.0, 2.0])).unwrap();
    }
    w.finish().unwrap();
    let big_manifest = fs::metadata(&path).unwrap().len();

    let mut w = Writer::append(&path).unwrap();
    w.add("z", [1u64], Leaf::F32, &f32s(&[3.0])).unwrap();
    w.finish().unwrap();

    let bytes = fs::read(&path).unwrap();
    assert_eq!(
        &bytes[bytes.len() - 8..],
        &ztensor::format::MAGIC,
        "the footer must end the file"
    );
    let src = Source::open(&path).unwrap();
    assert_eq!(src.len(), 65);
    assert!(bytes.len() as u64 > big_manifest - 4096);
}

/// Appending twice: the second append reads what the first wrote.
#[test]
fn appends_compose() {
    let path = tmp("append-twice.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("one", [1u64], Leaf::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    for name in ["two", "three"] {
        let mut w = Writer::append(&path).unwrap();
        w.add(name, [1u64], Leaf::F32, &f32s(&[2.0])).unwrap();
        w.finish().unwrap();
    }

    let src = Source::open(&path).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["one", "three", "two"]);
    for name in ["one", "two", "three"] {
        src.tensor(name).unwrap().verify().unwrap();
    }
}

/// A shard table survives the trip, and a shard can be added by appending.
#[test]
fn a_shard_can_be_added_to_a_finished_file() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let shard = dir.join("append-shard-data.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("borrowed", [4u64], Leaf::F32, &f32s(&[7.0; 4])).unwrap();
    w.finish().unwrap();
    let id = shard_identity(&shard, DigestAlgorithm::Xxh3).unwrap();
    let object = ztensor::read::manifest_of(&shard)
        .unwrap()
        .unwrap()
        .object("borrowed")
        .unwrap()
        .clone();

    let root = dir.join("append-root.zt");
    let mut w = Writer::options().canonical(false).create(&root).unwrap();
    w.add("local", [1u64], Leaf::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let mut w = Writer::append(&root).unwrap();
    w.add_shard("data", &id).unwrap();
    w.link("borrowed", &object, "data").unwrap();
    w.finish().unwrap();

    let s = Source::options()
        .resolver({
            let shard = shard.clone();
            move |_: &str, _: &ztensor::Shard| Ok(shard.clone())
        })
        .open(&root)
        .unwrap();
    assert_eq!(s.names().collect::<Vec<_>>(), vec!["borrowed", "local"]);
    assert_eq!(
        s.tensor("borrowed").unwrap().map().unwrap(),
        &f32s(&[7.0; 4])[..]
    );
    s.verify_shards().unwrap();
}

#[test]
fn a_name_already_in_the_file_is_refused() {
    let path = tmp("append-dup.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("t", [1u64], Leaf::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let mut w = Writer::append(&path).unwrap();
    let err = w.add("t", [1u64], Leaf::F32, &f32s(&[2.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    w.abandon();

    // The failed attempt did not damage the file.
    assert_eq!(Source::open(&path).unwrap().len(), 1);
}

#[test]
fn canonical_form_cannot_be_appended_to() {
    let path = tmp("append-canonical.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [1u64], Leaf::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let err = Writer::options().append(&path).unwrap_err();
    let message = format!("{err}");
    assert!(message.contains("canonical(false)"), "{message}");
}

#[test]
fn a_data_shard_has_nothing_to_append_to() {
    let path = tmp("append-datashard.zt");
    // Manifest-less (spec §7.2). zTensor reads these but does not write them,
    // so the test builds one the way another producer would.
    let mut bytes = vec![0u8; 4160];
    bytes[..8].copy_from_slice(&ztensor::format::MAGIC);
    bytes[4096..4160].copy_from_slice(&[1u8; 64]);
    let mut footer = [0u8; 40];
    footer[24..28].copy_from_slice(&ztensor::format::VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::format::MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(&path, &bytes).unwrap();

    let err = Writer::append(&path).unwrap_err();
    assert!(format!("{err}").contains("no manifest"), "{err}");
}
