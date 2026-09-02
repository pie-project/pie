//! What a tensor reports it can do, against what it actually does.
//!
//! Every [`Caps`] field is named after an operation and is meant to be that
//! operation's own precondition. The test that matters here is the one that
//! holds them together: for every tensor of every file, the report and the
//! outcome agree. A capability report that is a hand-written summary of the
//! real rules is a report that drifts, and the last one did: it demanded a
//! digest before admitting a tensor could be evicted, which eviction never
//! needed.

use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::format::cbor;
use ztensor::format::cbor::Value;
use ztensor::format::{ALIGN_CANONICAL, MAGIC, VERSION};
use ztensor::provide::page_size;
use ztensor::{Error, Leaf, Source, Term, Verified, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn canonical_file(name: &str) -> PathBuf {
    let path = tmp(name);
    let mut w = Writer::create(&path).unwrap();
    w.add("a.weight", [64u64, 64], Leaf::F32, &vec![3u8; 16384])
        .unwrap();
    w.add("b.weight", [128u64], Leaf::BF16, &vec![4u8; 256])
        .unwrap();
    w.finish().unwrap();
    path
}

#[test]
fn canonical_placement_reaches_every_capability() {
    let src = Source::open(canonical_file("caps.zt")).unwrap();
    for name in ["a.weight", "b.weight"] {
        let caps = src.tensor(name).unwrap().caps();
        assert!(caps.locate);
        assert!(caps.map);
        assert!(caps.verify);
        assert!(caps.alignment >= ALIGN_CANONICAL, "{caps:?}");
        // On any page size up to 64 KiB, canonical placement is exclusive.
        if page_size() <= ALIGN_CANONICAL {
            assert!(caps.evict, "{caps:?}");
        }
    }
}

#[test]
fn floor_alignment_still_pages_on_small_pages() {
    let path = tmp("floor.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    w.add("t", [16u64], Leaf::F32, &[7u8; 64]).unwrap();
    w.finish().unwrap();
    let src = Source::open(&path).unwrap();
    let caps = src.tensor("t").unwrap().caps();
    assert!(caps.map && caps.verify);
    assert_eq!(caps.alignment, 4096);
    if page_size() == 4096 {
        assert!(caps.evict);
    }
}

/// The anti-drift test: the report is the precondition, so it cannot disagree
/// with the operation. Run over files whose tensors land on both sides of
/// every predicate.
#[test]
fn the_report_and_the_outcome_agree() {
    let mut paths = vec![canonical_file("agree-canonical.zt"), no_digest_file()];

    let floor = tmp("agree-floor.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&floor)
        .unwrap();
    w.add("t", [16u64], Leaf::F32, &[7u8; 64]).unwrap();
    w.finish().unwrap();
    paths.push(floor);

    for path in &paths {
        for source in [
            Source::open(path).unwrap(),
            Source::options().map(false).open(path).unwrap(),
        ] {
            for tensor in source.tensors() {
                let name = tensor.name();
                let caps = tensor.caps();
                assert_eq!(caps.locate, tensor.locate().is_ok(), "{path:?} {name}: locate");
                assert_eq!(caps.map, tensor.map().is_ok(), "{path:?} {name}: map");
                assert_eq!(
                    caps.verify,
                    tensor.verify().unwrap() == Verified::Digest,
                    "{path:?} {name}: verify"
                );
                #[cfg(unix)]
                assert_eq!(caps.evict, tensor.evict().is_ok(), "{path:?} {name}: evict");
            }
        }
    }
}

/// A tensor with no digest is still evictable, and says so.
///
/// This is exactly what the old ordinal got wrong: it bundled integrity and
/// memory layout into one number, so this tensor reported one rung below the
/// operation it could perform.
#[test]
fn a_tensor_without_a_digest_is_still_evictable() {
    let path = no_digest_file();
    let src = Source::open(&path).unwrap();
    let caps = src.tensor("t").unwrap().caps();
    assert!(!caps.verify, "the fixture has no digest");
    assert_eq!(
        src.tensor("t").unwrap().verify().unwrap(),
        Verified::NoDigest
    );
    if page_size() <= ALIGN_CANONICAL {
        assert!(caps.evict, "nothing about eviction needs a digest");
        #[cfg(unix)]
        src.tensor("t").unwrap().evict().unwrap();
    }
}

#[cfg(unix)]
#[test]
fn evict_and_reread() {
    let src = Source::open(canonical_file("evict.zt")).unwrap();
    let tensor = src.tensor("a.weight").unwrap();
    let before = tensor.bytes().unwrap().into_owned();
    if tensor.caps().evict {
        tensor.prefetch().unwrap();
        tensor.evict().unwrap();
        // Evicted pages re-fault from the file: content is unchanged.
        assert_eq!(&*tensor.bytes().unwrap(), &before[..]);
        assert_eq!(tensor.verify().unwrap(), Verified::Digest);
    }
}

#[test]
fn an_absent_tensor_is_not_found() {
    let src = Source::open(canonical_file("nf.zt")).unwrap();
    assert!(matches!(src.tensor("nope"), Err(Error::NotFound(_))));
    assert!(src.get("nope").is_none());
    assert!(src.get("a.weight").is_some());
}

/// Hand-assembles a file whose single tensor carries no digest. The writer
/// always writes one, so this shape has to be built by hand.
fn no_digest_file() -> PathBuf {
    let path = tmp("no-digest.zt");
    let offset = ALIGN_CANONICAL;
    let len = 64u64;
    let text = |s: &str| Value::Text(s.to_string());
    let manifest = Value::Map(vec![(
        text("objects"),
        Value::Map(vec![(
            text("t"),
            Value::Map(vec![
                (text("shape"), Value::Array(vec![Value::Uint(16)])),
                (text("type"), text("f32")),
                (
                    text("blob"),
                    Value::Map(vec![
                        (text("offset"), Value::Uint(offset)),
                        (text("length"), Value::Uint(len)),
                    ]),
                ),
            ]),
        )]),
    )]);

    let manifest_bytes = cbor::encode(&manifest).unwrap();
    let m_off = (offset + len).div_ceil(ALIGN_CANONICAL) * ALIGN_CANONICAL;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.resize(m_off as usize, 0);
    for b in bytes
        .iter_mut()
        .take((offset + len) as usize)
        .skip(offset as usize)
    {
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
    fs::write(&path, &bytes).unwrap();
    path
}

/// A source is shareable across threads, because a loader that reads a
/// checkpoint from several of them is the ordinary case.
#[test]
fn a_source_can_be_shared_between_threads() {
    let path = canonical_file("threaded.zt");
    let src = std::sync::Arc::new(Source::open(&path).unwrap());
    let mut handles = Vec::new();
    for _ in 0..4 {
        let src = src.clone();
        handles.push(std::thread::spawn(move || {
            let tensor = src.tensor("a.weight").unwrap();
            assert_eq!(tensor.map().unwrap().len(), 16384);
            assert!(tensor.verify().unwrap().is_checked());
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Verifying a tensor covers every plane, and says so honestly when there is
/// nothing to cover.
///
/// A quantized tensor is one blob: codes, then scales. The digest is over the
/// whole blob, so a scale that rotted fails verification just as a code would.
/// The other trap is a tensor with no digest at all: it has nothing to vouch
/// for, and must not report a verification that never happened.
#[test]
fn verifying_a_tensor_covers_every_plane() {
    use ztensor::provide::{Catalog, Entry, Store};

    let path = tmp("verify-all-planes.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    let (codes, scales) = ([0xabu8; 32], [7u8; 2]);
    w.object("q", |o| {
        o.shape([64u64])
            .term(Term::parse("g32_e2m1_e8m0_n").unwrap())
            .planes([&codes[..], &scales[..]])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let q = src.tensor("q").unwrap();
    assert_eq!(q.verify().unwrap(), Verified::Digest);
    let planes = q.planes().unwrap();
    assert_eq!(planes[1].path, "gain");
    let scales_at = q.locate().unwrap().offset + planes[1].offset;
    let shape = q.shape().to_vec();
    let term = q.term().unwrap().clone();
    let at = q.locate().unwrap();
    drop(src);

    // Corrupt only the *scales*. Checking the codes alone would still pass.
    let mut raw = std::fs::read(&path).unwrap();
    raw[scales_at as usize] ^= 0xff;
    std::fs::write(&path, &raw).unwrap();

    let src = Source::open(&path).unwrap();
    let err = src.tensor("q").unwrap().verify().unwrap_err();
    assert_eq!(err.rule(), Some(ztensor::Rule::Digest), "{err}");

    // The same bytes described without a digest have nothing to vouch for.
    let mut catalog = Catalog::new();
    catalog.insert("bare", Entry::at(shape, term, at));
    let store = Store::index(&path, "zt").unwrap();
    let projected = Source::from_parts(vec![store], catalog).unwrap();
    assert_eq!(
        projected.tensor("bare").unwrap().verify().unwrap(),
        Verified::NoDigest,
        "a tensor with no digest must not report one was checked"
    );
}
