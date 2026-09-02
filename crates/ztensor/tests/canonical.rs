//! Deciding whether a file is canonical (spec §6.4).
//!
//! The check has to agree with the writer: whatever `Writer::create` produces
//! is canonical by definition, so any file it writes that the checker rejects
//! is a bug in the checker, not in the file. Everything else here breaks one
//! rule at a time and names the rule it broke.

use std::fs;
use std::path::PathBuf;

use ztensor::read::canonical_violations;
use ztensor::{DigestAlgorithm, Leaf, Term, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The writer defines canonical form, so its output must pass.
#[test]
fn what_the_writer_produces_is_canonical() {
    let path = tmp("canon-yes.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("a.bias", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.add("a.weight", [2u64, 4], Leaf::BF16, &[2u8; 16]).unwrap();
    // A group type written plane by plane is still one raw blob.
    let (codes, scales) = ([0x11u8; 32], [0x7fu8; 2]);
    w.object("q", |o| {
        o.shape([64u64])
            .term(Term::parse("g32_e2m1_e8m0_n").unwrap())
            .planes([&codes[..], &scales[..]])
    })
    .unwrap();
    // Tied weights: byte-identical, so they must share one blob.
    w.add("tied", [2u64, 4], Leaf::BF16, &[2u8; 16]).unwrap();
    w.add("z.last", [1u64], Leaf::U8, &[7]).unwrap();
    w.finish().unwrap();

    assert_eq!(
        canonical_violations(&path).unwrap(),
        Vec::<String>::new(),
        "the writer's own output must be canonical"
    );
}

/// Bytes between planes are part of the blob and must be zero (spec §5.1
/// rule 3): garbage in a gap is refused on verify, not smoothed over.
#[test]
fn garbage_between_planes_is_refused() {
    let path = tmp("canon-gap.zt");
    let term = Term::parse("g32_e2m1_e8m0_n").unwrap();
    let mut blob = vec![0u8; term.canonical_size(&[64]).unwrap() as usize];
    blob[40] = 0xff; // past the 32 code bytes, before the scales
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| o.shape([64u64]).term(term.clone()).bytes(&blob))
        .unwrap();
    w.finish().unwrap();

    let src = ztensor::Source::open(&path).unwrap();
    let err = src.tensor("q").unwrap().verify().unwrap_err();
    assert_eq!(err.rule(), Some(ztensor::Rule::LayoutData), "{err}");
}

/// An empty model is a degenerate but legal canonical file.
#[test]
fn an_empty_model_is_canonical() {
    let path = tmp("canon-empty.zt");
    Writer::create(&path).unwrap().finish().unwrap();
    assert_eq!(canonical_violations(&path).unwrap(), Vec::<String>::new());
}

fn violations_of(path: &PathBuf) -> String {
    canonical_violations(path).unwrap().join(" | ")
}

/// Rule 2: 4 KiB placement is legal, but it is not canonical.
#[test]
fn floor_alignment_breaks_rule_2() {
    let path = tmp("canon-floor.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    w.add("a", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.add("b", [4u64], Leaf::F32, &f32s(&[2.0; 4])).unwrap();
    w.finish().unwrap();

    let found = violations_of(&path);
    assert!(found.contains("rule 2"), "{found}");
}

/// Rule 4: block digests are not canonical, even at canonical placement.
#[test]
fn block_digests_break_rule_4() {
    let path = tmp("canon-blocks.zt");
    let mut w = Writer::options()
        .canonical(false)
        .blocks(8)
        .create(&path)
        .unwrap();
    w.add("a", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.finish().unwrap();

    let found = violations_of(&path);
    assert!(found.contains("rule 4"), "{found}");
    assert!(!found.contains("rule 2"), "placement was canonical: {found}");
}

/// Rule 6: a shard table.
#[test]
fn a_shard_table_breaks_rule_6() {
    let shard = tmp("canon-shard-data.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("t", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.finish().unwrap();
    let id = ztensor::read::shard_identity(&shard, DigestAlgorithm::Sha256).unwrap();

    let root = tmp("canon-sharded.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(65536)
        .create(&root)
        .unwrap();
    w.add("local", [1u64], Leaf::U8, &[1]).unwrap();
    w.add_shard("data", &id).unwrap();
    w.finish().unwrap();

    let found = violations_of(&root);
    assert!(found.contains("rule 6"), "{found}");
}

/// Rule 1: an appended file carries the manifest it superseded, and the space
/// that manifest occupies belongs to nothing.
#[test]
fn an_appended_file_breaks_rule_1() {
    let path = tmp("canon-appended.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("a", [4u64], Leaf::F32, &f32s(&[1.0; 4])).unwrap();
    w.finish().unwrap();
    assert_eq!(canonical_violations(&path).unwrap(), Vec::<String>::new());

    let mut w = Writer::append(&path).unwrap();
    w.add("b", [4u64], Leaf::F32, &f32s(&[2.0; 4])).unwrap();
    w.finish().unwrap();

    let found = violations_of(&path);
    assert!(found.contains("rule 1"), "{found}");
}

/// A data shard is not a canonical model, and saying so beats erroring.
#[test]
fn a_data_shard_is_not_canonical() {
    let path = tmp("canon-datashard.zt");
    // Manifest-less (spec §7.2), built here because zTensor only reads these.
    let mut bytes = vec![0u8; 4160];
    bytes[..8].copy_from_slice(&ztensor::format::MAGIC);
    bytes[4096..4160].copy_from_slice(&[1u8; 64]);
    let mut footer = [0u8; 40];
    footer[24..28].copy_from_slice(&ztensor::format::VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::format::MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(&path, &bytes).unwrap();

    let found = violations_of(&path);
    assert!(found.contains("rule 1"), "{found}");
}

/// Every conformance-valid file the writer produced is canonical, and the
/// hand-assembled ones mostly are not. This is the broad sweep that would
/// catch a checker that simply always returns "canonical".
#[test]
fn the_checker_disagrees_with_itself_on_different_files() {
    let canonical = tmp("canon-sweep-a.zt");
    let mut w = Writer::create(&canonical).unwrap();
    w.add("t", [8u64], Leaf::U8, &[1u8; 8]).unwrap();
    w.finish().unwrap();

    let not = tmp("canon-sweep-b.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&not)
        .unwrap();
    w.add("t", [8u64], Leaf::U8, &[1u8; 8]).unwrap();
    w.finish().unwrap();

    assert!(canonical_violations(&canonical).unwrap().is_empty());
    assert!(!canonical_violations(&not).unwrap().is_empty());
    let _ = fs::metadata(&canonical).unwrap();
}
