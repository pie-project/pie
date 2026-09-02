//! The container: round-trip, canonical determinism, blob sharing, and the
//! files that must be rejected.

use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::format::cbor;
use ztensor::format::cbor::Value;
use ztensor::format::{ALIGN_CANONICAL, MAGIC, VERSION};
use ztensor::{Error, Leaf, Rule, Source, Term, Verified, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn roundtrip_dense() {
    let path = tmp("roundtrip.zt");
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Vec<u8> = (0..10).collect(); // 5 bf16 elements
    let c: Vec<u8> = vec![7; 7];

    let mut w = Writer::create(&path).unwrap();
    w.add("a.weight", [2u64, 3], Leaf::F32, &a).unwrap();
    w.add("b.bias", [5u64], Leaf::BF16, &b).unwrap();
    w.add("c.mask", [7u64], Leaf::U8, &c).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    assert!(matches!(src.provenance(), ztensor::Provenance::Root(_)));
    assert_eq!(src.len(), 3);
    let aw = src.tensor("a.weight").unwrap();
    assert_eq!(aw.shape(), &[2, 3]);
    assert_eq!(aw.term(), Some(&Term::Leaf(Leaf::F32)));
    assert_eq!(aw.layout(), None);
    assert_eq!(aw.map().unwrap(), &a[..]);
    assert_eq!(src.tensor("b.bias").unwrap().map().unwrap(), &b[..]);
    assert_eq!(&*src.tensor("c.mask").unwrap().bytes().unwrap(), &c[..]);
    assert_eq!(aw.verify().unwrap(), Verified::Digest);

    // Canonical placement: every blob at a 64 KiB boundary.
    for tensor in src.tensors() {
        assert_eq!(tensor.locate().unwrap().offset % ALIGN_CANONICAL, 0);
    }
}

/// A source is one type whichever way it was opened, and `bytes()` is honest
/// about which half of the bargain it kept.
#[test]
fn an_indexed_source_locates_without_mapping() {
    let path = tmp("indexed.zt");
    let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut w = Writer::create(&path).unwrap();
    w.add("x", [4u64], Leaf::F32, &data).unwrap();
    w.finish().unwrap();

    let mapped = Source::open(&path).unwrap();
    let indexed = Source::options().map(false).open(&path).unwrap();

    // The same address either way.
    let here = mapped.tensor("x").unwrap().locate().unwrap();
    let there = indexed.tensor("x").unwrap().locate().unwrap();
    assert_eq!(here.offset, there.offset);
    assert_eq!(here.len, there.len);

    let caps = indexed.tensor("x").unwrap().caps();
    assert!(caps.locate, "an indexed file still knows where things are");
    assert!(!caps.map, "nothing is mapped, so nothing can be borrowed");
    assert!(!caps.evict);

    assert!(indexed.tensor("x").unwrap().map().is_err());
    let bytes = indexed.tensor("x").unwrap().bytes().unwrap();
    assert!(
        matches!(bytes, Cow::Owned(_)),
        "an unmapped file has to copy"
    );
    assert_eq!(&*bytes, &data[..]);
    assert!(matches!(
        mapped.tensor("x").unwrap().bytes().unwrap(),
        Cow::Borrowed(_)
    ));
}

#[test]
fn canonical_is_deterministic() {
    let write = |path: &PathBuf| {
        let mut w = Writer::create(path).unwrap();
        w.add("x", [4u64], Leaf::F32, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        w.add("y", [2u64], Leaf::U8, &[9, 9]).unwrap();
        w.finish().unwrap();
    };
    let p1 = tmp("det1.zt");
    let p2 = tmp("det2.zt");
    write(&p1);
    write(&p2);
    assert_eq!(fs::read(&p1).unwrap(), fs::read(&p2).unwrap());
}

/// Writing the same thing twice gives the same bytes outside canonical form
/// too, which is the path every sharded model takes.
///
/// Nothing in the non-canonical writer reads the clock or the environment, so
/// this holds by construction. It is pinned here because "holds by
/// construction" is a claim about code as it stands, and this is the path
/// sharding uses: canonical form is single-file by rule 6, so any model split
/// across files is written by the code this covers and by nothing else.
#[test]
fn non_canonical_writes_are_reproducible_too() {
    let write = |path: &PathBuf| {
        let mut w = Writer::options()
            .canonical(false)
            .align(4096)
            .blocks(8)
            .create(path)
            .unwrap();
        // Deliberately unsorted, plane-written, attribute-carrying, block
        // digested: the freedoms canonical form removes are exactly what is
        // exercised here.
        let later = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        w.object("z.later", |o| {
            o.shape([2u64, 2])
                .attr("note", "written first")
                .term(Leaf::F32)
                .bytes(&later)
        })
        .unwrap();
        let (codes, scales) = ([0xabu8; 32], [7u8; 2]);
        w.object("a.earlier", |o| {
            o.shape([64u64])
                .term(Term::parse("g32_e2m1_e8m0_n").unwrap())
                .planes([&codes[..], &scales[..]])
        })
        .unwrap();
        w.add("shared", [2u64], Leaf::U8, &[9, 9]).unwrap();
        w.finish().unwrap();
    };
    let p1 = tmp("nondet1.zt");
    let p2 = tmp("nondet2.zt");
    write(&p1);
    write(&p2);
    assert_eq!(
        fs::read(&p1).unwrap(),
        fs::read(&p2).unwrap(),
        "two identical non-canonical writes must agree byte for byte"
    );
}

#[test]
fn tied_weights_share_one_blob() {
    let path = tmp("tied.zt");
    let data = f32_bytes(&[42.0; 256]);
    let mut w = Writer::create(&path).unwrap();
    w.add("embed", [16u64, 16], Leaf::F32, &data).unwrap();
    w.add("lm_head", [16u64, 16], Leaf::F32, &data).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let e = src.tensor("embed").unwrap().locate().unwrap();
    let l = src.tensor("lm_head").unwrap().locate().unwrap();
    assert_eq!(e.offset, l.offset, "identical blobs must be shared");
    assert_eq!(src.tensor("embed").unwrap().map().unwrap(), &data[..]);
}

#[test]
fn zero_length_tensor() {
    let path = tmp("zero.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("empty", [0u64, 8], Leaf::F32, &[]).unwrap();
    w.finish().unwrap();
    let src = Source::open(&path).unwrap();
    assert_eq!(src.tensor("empty").unwrap().map().unwrap().len(), 0);
}

#[test]
fn canonical_requires_sorted_insertion() {
    let path = tmp("unsorted.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("b", [1u64], Leaf::U8, &[0]).unwrap();
    let err = w.add("a", [1u64], Leaf::U8, &[0]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

/// The alignment knob and the canonical-form switch are different questions,
/// and asking one while meaning the other is refused rather than obeyed.
#[test]
fn alignment_is_not_the_canonical_switch() {
    let err = Writer::options()
        .align(4096)
        .create(tmp("confused.zt"))
        .unwrap_err();
    let message = format!("{err}");
    assert!(
        message.contains("canonical(false)"),
        "the error should say how to mean it: {message}"
    );

    // Said properly, insertion order is free.
    let path = tmp("unsorted-ok.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    w.add("b", [1u64], Leaf::U8, &[0]).unwrap();
    w.add("a", [1u64], Leaf::U8, &[0]).unwrap();
    w.finish().unwrap();
    assert_eq!(Source::open(&path).unwrap().len(), 2);
}

/// Leaving canonical form gives up byte-reproducible placement, not placement.
///
/// It used to drop to the 4 KiB floor, which mattered because a sharded model
/// *cannot* be canonical (§6.4 rule 6): every sharded root silently lost
/// per-tensor page exclusivity on any host with pages above 4 KiB unless its
/// author knew to ask for the alignment back.
#[test]
fn a_non_canonical_writer_still_places_at_64_kib() {
    let path = tmp("noncanon-align.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("t", [4u64], Leaf::F32, &f32_bytes(&[1.0; 4])).unwrap();
    w.add("u", [4u64], Leaf::F32, &f32_bytes(&[2.0; 4])).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    for name in ["t", "u"] {
        let at = src.tensor(name).unwrap().locate().unwrap();
        assert_eq!(
            at.offset % ALIGN_CANONICAL,
            0,
            "{name} landed at {}, off the 64 KiB grid",
            at.offset
        );
    }

    // And asking for the floor still gets the floor.
    let floor = tmp("noncanon-floor.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&floor)
        .unwrap();
    w.add("t", [4u64], Leaf::F32, &f32_bytes(&[1.0; 4])).unwrap();
    w.finish().unwrap();
    assert_eq!(
        Source::open(&floor)
            .unwrap()
            .tensor("t")
            .unwrap()
            .locate()
            .unwrap()
            .offset,
        4096
    );
}

// =======================================================================
// Must-reject cases
// =======================================================================

fn write_small(path: &PathBuf) {
    let mut w = Writer::create(path).unwrap();
    w.add("t", [2u64], Leaf::U8, &[1, 2]).unwrap();
    w.finish().unwrap();
}

fn expect_reject(path: &PathBuf, rule: Rule) {
    match Source::open(path) {
        Err(e) => assert_eq!(e.rule(), Some(rule), "{e}"),
        Ok(_) => panic!("expected Reject({rule:?}), got a source"),
    }
}

#[test]
fn reject_bad_footer_magic() {
    let path = tmp("badfooter.zt");
    write_small(&path);
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    expect_reject(&path, Rule::FooterMagic);
}

/// A v2 file is a different format: refused by version, not misread.
#[test]
fn reject_previous_version() {
    let path = tmp("v2.zt");
    write_small(&path);
    let mut bytes = fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 16..n - 12].copy_from_slice(&2u32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    expect_reject(&path, Rule::Version);
}

#[test]
fn reject_corrupt_manifest() {
    let path = tmp("badmanifest.zt");
    write_small(&path);
    let mut bytes = fs::read(&path).unwrap();
    let n = bytes.len();
    let m_off = u64::from_le_bytes(bytes[n - 40..n - 32].try_into().unwrap()) as usize;
    bytes[m_off] ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    expect_reject(&path, Rule::ManifestHash);
}

#[test]
fn reject_truncated() {
    let path = tmp("truncated.zt");
    write_small(&path);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 10]).unwrap();
    assert!(Source::open(&path).unwrap_err().rule().is_some());
}

/// Assembles a file by hand: magic, blobs, a caller-supplied manifest value,
/// and a correct footer. Lets tests express structurally hostile manifests
/// that the writer would refuse to produce.
fn assemble(path: &PathBuf, data_len: u64, manifest: &Value) {
    let manifest_bytes = cbor::encode(manifest).unwrap();
    let m_off = (8 + data_len).div_ceil(4096) * 4096;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.resize(m_off as usize, 0);
    // fill the data region with a marker so blobs have content
    for b in bytes.iter_mut().take(m_off as usize).skip(8) {
        *b = 0xab;
    }
    bytes.extend_from_slice(&manifest_bytes);
    let mut footer = [0u8; 40];
    footer[0..8].copy_from_slice(&m_off.to_le_bytes());
    footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
    footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn obj(ty: &str, shape: &[u64], offset: u64, length: u64) -> Value {
    Value::Map(vec![
        (
            text("shape"),
            Value::Array(shape.iter().map(|&d| Value::Uint(d)).collect()),
        ),
        (text("type"), text(ty)),
        (
            text("blob"),
            Value::Map(vec![
                (text("offset"), Value::Uint(offset)),
                (text("length"), Value::Uint(length)),
            ]),
        ),
    ])
}

fn manifest_of(objs: Vec<(&str, Value)>) -> Value {
    Value::Map(vec![(
        text("objects"),
        Value::Map(objs.into_iter().map(|(n, o)| (text(n), o)).collect()),
    )])
}

#[test]
fn reject_partial_overlap() {
    let path = tmp("overlap.zt");
    // blob A: [4096, 12288), blob B: [8192, 8200), which is inside A.
    let m = manifest_of(vec![
        ("a", obj("f32", &[2048], 4096, 8192)),
        ("b", obj("f32", &[2], 8192, 8)),
    ]);
    assemble(&path, 12288 - 8, &m);
    expect_reject(&path, Rule::BlobOverlap);
}

#[test]
fn identical_refs_are_legal() {
    let path = tmp("aliased.zt");
    let m = manifest_of(vec![
        ("a", obj("f32", &[2], 4096, 8)),
        ("b", obj("f32", &[2], 4096, 8)),
    ]);
    assemble(&path, 4096 + 8 - 8, &m);
    let src = Source::open(&path).unwrap();
    assert_eq!(
        src.tensor("a").unwrap().map().unwrap(),
        src.tensor("b").unwrap().map().unwrap()
    );
}

#[test]
fn reject_misaligned_blob() {
    let path = tmp("misaligned.zt");
    let m = manifest_of(vec![("a", obj("f32", &[2], 4100, 8))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::BlobAlignment);
}

#[test]
fn reject_size_mismatch() {
    let path = tmp("badsize.zt");
    // f32 x [3] = 12 bytes, but the blob claims 8.
    let m = manifest_of(vec![("a", obj("f32", &[3], 4096, 8))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::Size);
}

#[test]
fn reject_unknown_leaf() {
    let path = tmp("badleaf.zt");
    let m = manifest_of(vec![("a", obj("f4", &[2], 4096, 1))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::Type);
}

/// `attributes()` hands over a whole map and `attr()` adds one entry. Mixing
/// them is refused rather than resolved: whichever silently won, the other
/// would lose object metadata with nothing said, which for a named layout is
/// the block geometry a reader needs to address bytes.
#[test]
fn attributes_are_given_one_way_or_the_other() {
    let path = tmp("attrs-mixed.zt");
    let mut w = Writer::create(&path).unwrap();
    let err = w
        .object("q", |o| {
            o.shape([4u64])
                .attributes(cbor::map([("bits", 4u64), ("group_size", 32u64)]))
                .attr("axis", 1u64)
                .term(Leaf::U8)
                .bytes(&[0u8; 4])
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    w.object("q", |o| {
        o.shape([4u64])
            .attributes(cbor::map([("bits", 4u64), ("group_size", 32u64)]))
            .term(Leaf::U8)
            .bytes(&[0u8; 4])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let attrs = src.tensor("q").unwrap().attributes().unwrap().clone();
    assert_eq!(attrs.get("bits").and_then(cbor::Value::as_u64), Some(4));
}

/// Two payloads on one object is a contradiction, not an override.
///
/// The builder methods return `Self`, so neither of them can refuse; the
/// second one records the conflict and the object refuses to build. Before
/// this, `.bytes(x).length(n)` compiled, dropped `x` silently, and produced a
/// file whose blob was never written.
#[test]
fn an_object_takes_one_payload() {
    let path = tmp("two-payloads.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    let data = [7u8; 4];

    let err = w
        .object("t", |o| o.shape([4u64]).term(Leaf::U8).bytes(&data).length(4))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("`bytes`") && message.contains("`length`"),
        "the error has to name both setters, got: {message}"
    );

    // Order is reported as given, and the object did not land.
    let err = w
        .object("t", |o| o.shape([4u64]).term(Leaf::U8).length(4).bytes(&data))
        .unwrap_err();
    assert!(
        err.to_string().contains("`length`, then `bytes`"),
        "got: {err}"
    );
    w.abandon();
}

/// A digest is for bytes this writer will not see. Supplying one for bytes it
/// does write could only agree with the computed digest or be wrong, so it is
/// refused rather than ignored.
#[test]
fn only_an_external_blob_takes_a_digest() {
    let path = tmp("digest-on-local.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    let err = w
        .object("t", |o| {
            o.shape([4u64])
                .term(Leaf::U8)
                .digest(ztensor::Digest::new(ztensor::DigestAlgorithm::Xxh3, vec![0; 8]))
                .bytes(&[0u8; 4])
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("only an external blob takes a digest"),
        "got: {err}"
    );
    w.abandon();
}
