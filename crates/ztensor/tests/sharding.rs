//! Multi-file models (spec §7): the shard table, resolution, and the overlay
//! story, where a LoRA root references the base model's blobs directly.
//!
//! This is the shape a `Source` gets by *verification*: the root states each
//! shard's size and digest, so opening one checks that the files on disk are
//! the files the manifest meant. Contrast `merge.rs`, where nothing binds the
//! set and nothing pretends to.

use std::fs;
use std::path::PathBuf;

use ztensor::read::{shard_identity, ShardResolver};
use ztensor::{Digest, DigestAlgorithm, Error, Leaf, Rule, Shard, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn digest(hex: &str) -> Digest {
    let value = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    Digest::new(DigestAlgorithm::Xxh3, value)
}

/// Writes a data shard (spec §7.2): magic, one blob at `offset`, and a footer
/// whose manifest fields are all zero.
///
/// zTensor does not write these. It reads them, because other producers do, so
/// the tests build one the way such a producer would.
fn write_data_shard(path: &PathBuf, offset: u64, payload: &[u8]) {
    let end = offset as usize + payload.len();
    let mut bytes = vec![0u8; end];
    bytes[..8].copy_from_slice(&ztensor::format::MAGIC);
    bytes[offset as usize..end].copy_from_slice(payload);
    let mut footer = [0u8; 40];
    // offset, length and hash stay zero: that is what makes it a data shard.
    footer[24..28].copy_from_slice(&ztensor::format::VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::format::MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

/// Opens a root whose shards all live at one known path.
fn open_with_shard_at(root: &PathBuf, shard_path: PathBuf) -> ztensor::Result<Source> {
    Source::options()
        .resolver(move |_name: &str, _shard: &Shard| Ok(shard_path.clone()))
        .open(root)
}

fn object_of(path: &PathBuf, name: &str) -> ztensor::Object {
    ztensor::read::manifest_of(path)
        .unwrap()
        .unwrap()
        .object(name)
        .unwrap()
        .clone()
}

/// Base model + LoRA overlay: the LoRA file stores only its deltas and
/// references the base's blobs through the shard table.
#[test]
fn lora_overlay() {
    let base_path = tmp("overlay-base.zt");
    let base_data: Vec<u8> = (0..1024u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let mut w = Writer::create(&base_path).unwrap();
    w.add("base.weight", [32u64, 32], Leaf::F32, &base_data)
        .unwrap();
    w.finish().unwrap();
    let base = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();

    let lora_path = tmp("overlay-lora.zt");
    let delta = vec![7u8; 256];
    let base_object = object_of(&base_path, "base.weight");

    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&lora_path)
        .unwrap();
    w.add_shard("base", &base).unwrap();
    w.link("base.weight", &base_object, "base").unwrap();
    w.add("base.weight.lora_a", [64u64], Leaf::F32, &delta)
        .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&lora_path, base_path.clone()).unwrap();

    // Cross-file borrow: the base tensor's bytes come from base.zt.
    assert_eq!(
        model.tensor("base.weight").unwrap().map().unwrap(),
        &base_data[..]
    );
    assert_eq!(
        &*model.tensor("base.weight.lora_a").unwrap().bytes().unwrap(),
        &delta[..]
    );

    // The address names the file it came from, which is what a store id is
    // for: two tensors of one model, living in two different files.
    let base_at = model.tensor("base.weight").unwrap().locate().unwrap();
    let lora_at = model.tensor("base.weight.lora_a").unwrap().locate().unwrap();
    assert_ne!(base_at.store, lora_at.store);
    assert_eq!(model.store(base_at.store).path(), base_path);
    assert_eq!(model.store(lora_at.store).path(), lora_path);

    // The base is itself a manifest-carrying container, so its occupancy is
    // known and page exclusivity is a fact rather than a guess, so a tensor in
    // another file is as evictable as one at home.
    let caps = model.tensor("base.weight").unwrap().caps();
    assert!(caps.map && caps.locate && caps.verify);
    if ztensor::provide::page_size() <= ztensor::format::ALIGN_CANONICAL {
        assert!(caps.evict, "{caps:?}");
    }

    // The digest carried over by link verifies against the base's bytes.
    assert!(model
        .tensor("base.weight")
        .unwrap()
        .verify()
        .unwrap()
        .is_checked());
    model.verify_shards().unwrap();
}

#[test]
fn positional_shards() {
    // A manifest-less shard, built here because zTensor no longer writes one:
    // this is the shape some *other* producer hands you, and reading it is
    // what has to keep working.
    let shard_path = tmp("posmodel-00001.zt");
    let payload = vec![9u8; 8192];
    let offset = 4096u64;
    write_data_shard(&shard_path, offset, &payload);
    let identity = shard_identity(&shard_path, DigestAlgorithm::Xxh3).unwrap();

    // The data shard alone is a valid manifest-less file.
    assert_eq!(
        Source::open(&shard_path).unwrap().provenance(),
        ztensor::Provenance::DataShard
    );
    assert!(Source::open(&shard_path).unwrap().is_empty());

    let root_path = tmp("posmodel.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("00001", &identity).unwrap();
    let at = offset..offset + payload.len() as u64;
    w.object("t", |o| {
        o.shape([8192u64])
            .term(Leaf::U8)
            .digest(DigestAlgorithm::Xxh3.digest(&payload))
            .external("00001", at)
    })
    .unwrap();
    w.finish().unwrap();

    // The positional convention: posmodel.zt -> posmodel-00001.zt
    let model = Source::open(&root_path).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    assert!(model.tensor("t").unwrap().verify().unwrap().is_checked());
    model.verify_shards().unwrap();

    // The contrast with `lora_overlay`: a data shard states no occupancy, so
    // nothing can prove this blob has its pages to itself, and eviction is
    // refused rather than assumed safe.
    let caps = model.tensor("t").unwrap().caps();
    assert!(caps.map && caps.locate);
    assert!(
        !caps.evict,
        "a manifest-less shard cannot prove exclusivity"
    );
}

#[test]
fn shard_size_mismatch_rejected() {
    let base_path = tmp("mismatch-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let mut identity = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();
    let base_object = object_of(&base_path, "t");
    identity.size += 4096; // a claim the file on disk does not meet

    let root_path = tmp("mismatch-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("base", &identity).unwrap();
    w.link("t", &base_object, "base").unwrap();
    w.finish().unwrap();

    let err = open_with_shard_at(&root_path, base_path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
}

#[test]
fn shard_digest_mismatch_caught_by_deep_verify() {
    let base_path = tmp("digest-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [256u64], Leaf::U8, &[5u8; 256]).unwrap();
    w.finish().unwrap();
    let identity = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();
    let base_object = object_of(&base_path, "t");

    // Corrupt one data byte: size and footer stay valid.
    let mut bytes = fs::read(&base_path).unwrap();
    bytes[65536] ^= 0xff;
    let corrupted = tmp("digest-base-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();

    let root_path = tmp("digest-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("base", &identity).unwrap();
    w.link("t", &base_object, "base").unwrap();
    w.finish().unwrap();

    // The cheap rungs pass: the size is right and the frame parses.
    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
    // The object's own digest catches it independently.
    assert!(model.tensor("t").unwrap().verify().is_err());
}

#[test]
fn canonical_is_single_file() {
    let mut w = Writer::create(tmp("canon-shard.zt")).unwrap();
    let err = w
        .add_shard(
            "base",
            &Shard {
                size: 4096,
                digest: digest("0011223344556677"),
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn a_single_file_is_the_degenerate_case() {
    let path = tmp("single.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let model = Source::open(&path).unwrap(); // no shards: the resolver is never used
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &[1, 2, 3, 4]);
    model.verify_shards().unwrap(); // vacuous
}

/// A name is a label, so the resolver is free to ignore it and match on
/// identity instead, which is the only thing that survives a rename.
#[test]
fn shards_found_by_identity_after_a_rename() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("byid");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let base_path = dir.join("original-name.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let identity = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();
    let base_object = object_of(&base_path, "t");

    let root_path = dir.join("root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("weights", &identity).unwrap();
    w.link("t", &base_object, "weights").unwrap();
    w.finish().unwrap();

    // Rename the shard: no convention that consults a name can find it now.
    let renamed = dir.join("something-else-entirely.zt");
    fs::rename(&base_path, &renamed).unwrap();
    assert!(Source::open(&root_path).is_err(), "positional must miss it");

    let model = Source::options()
        .resolver(ztensor::read::DirectoryResolver::scan(&dir).unwrap())
        .open(&root_path)
        .unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &[1, 2, 3, 4]);
    assert_eq!(
        model
            .store(model.tensor("t").unwrap().locate().unwrap().store)
            .path(),
        renamed
    );
    model.verify_shards().unwrap();
}

/// The names a producer may choose are constrained so that a resolver can
/// spend one as a path component without sanitizing it first.
#[test]
fn a_shard_name_cannot_be_a_path() {
    let identity = Shard {
        size: 1 << 20,
        digest: digest("0011223344556677"),
    };
    for name in [
        "../etc/passwd",
        "sub/dir",
        "",
        ".hidden",
        "a b",
        &"x".repeat(65),
    ] {
        let mut w = Writer::options()
            .canonical(false)
            .create(tmp("badname.zt"))
            .unwrap();
        // The writer reports reader rules as InvalidInput, carrying the
        // rule's own message. See `writer::invalid`.
        let err = w.add_shard(name, &identity).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{name:?} was accepted, or reported as {err:?}"
        );
        w.abandon();
    }
}

/// Registering the same name twice is fine if it means the same file, and an
/// error if it does not. Silently keeping one of two identities would make
/// the manifest a claim nobody checked.
#[test]
fn a_name_means_one_shard() {
    let identity = Shard {
        size: 1 << 20,
        digest: digest("0011223344556677"),
    };
    let mut w = Writer::options()
        .canonical(false)
        .create(tmp("dupname.zt"))
        .unwrap();
    w.add_shard("base", &identity).unwrap();
    w.add_shard("base", &identity).unwrap();

    let other = Shard {
        size: 1 << 21,
        digest: digest("8899aabbccddeeff"),
    };
    assert!(w.add_shard("base", &other).is_err());
    w.abandon();
}

#[test]
fn resolver_trait_objects() {
    // The CAS path shape (no file IO, just the mapping).
    let cas = ztensor::read::cas("/store");
    let shard = Shard {
        size: 4096,
        digest: digest("00ff00ff00ff00ff"),
    };
    let path = cas.resolve("anything", &shard).unwrap();
    assert_eq!(path, PathBuf::from("/store/blobs/xxh3/00ff00ff00ff00ff"));
}

/// A sha256 shard identity, produced and then checked.
///
/// §6.6 makes this the basis of signing: a root whose shard digests are
/// cryptographic commits to every shard byte, so one signature over the root
/// covers the model. That only holds if the digest can be *verified*, which is
/// why generating one without being able to check it would be worse than not
/// generating it at all.
///
/// The shard is an ordinary `.zt`, which is how shards are made: it keeps the
/// occupancy and the per-object digests that a manifest-less one throws away.
#[test]
fn a_sha256_shard_identity_round_trips() {
    let shard_path = tmp("sha-shard.zt");
    let payload = vec![3u8; 4096];
    let mut w = Writer::create(&shard_path).unwrap();
    w.add("borrowed", [4096u64], Leaf::U8, &payload).unwrap();
    w.finish().unwrap();
    let offset = Source::open(&shard_path)
        .unwrap()
        .tensor("borrowed")
        .unwrap()
        .locate()
        .unwrap()
        .offset;
    let from_writer = ztensor::read::shard_identity(&shard_path, DigestAlgorithm::Sha256).unwrap();

    assert_eq!(from_writer.digest.algorithm, "sha256");
    assert_eq!(from_writer.digest.algorithm().unwrap(), DigestAlgorithm::Sha256);
    assert_eq!(from_writer.digest.value.len(), 32);
    assert!(from_writer.digest.to_string().starts_with("sha256:"));
    assert_eq!(from_writer.digest.to_string().len(), "sha256:".len() + 64);

    // Asking twice gives the same answer.
    let scanned = ztensor::read::shard_identity(&shard_path, DigestAlgorithm::Sha256).unwrap();
    assert_eq!(scanned, from_writer);
    // And the default is still xxh3, over the same bytes.
    let default = ztensor::read::shard_identity(&shard_path, DigestAlgorithm::Xxh3).unwrap();
    assert_eq!(default.digest.algorithm, "xxh3");
    assert_eq!(default.size, from_writer.size);

    // A root that records it, and a deep verify that checks it.
    let root_path = tmp("sha-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("data", &from_writer).unwrap();
    let at = offset..offset + payload.len() as u64;
    w.object("t", |o| o.shape([4096u64]).term(Leaf::U8).external("data", at))
        .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&root_path, shard_path.clone()).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    model.verify_shards().unwrap();

    // A corrupted shard is caught by the sha256 path, not waved through.
    let mut bytes = fs::read(&shard_path).unwrap();
    bytes[4096] ^= 0xff;
    let corrupted = tmp("sha-shard-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();
    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
}

/// An object digest in any algorithm this build knows is checked, not waved
/// through.
///
/// The writer always writes `xxh3` digests, which is what canonical form
/// requires (§6.4 rule 4) and what corruption detection wants. Verification is
/// the other half: a file from elsewhere may use `sha256`, and reading it must
/// check that digest rather than report "unsupported" or, worse, skip it.
#[test]
fn a_sha256_object_digest_is_verified() {
    use xxhash_rust::xxh3::xxh3_64;
    use ztensor::format::cbor::{self, Value};
    let data = vec![0xabu8; 256];
    let text = |s: &str| Value::Text(s.to_string());

    // Builds a file whose one object claims `digest`, over filler bytes that
    // the real payload matches.
    let build = |name: &str, digest: Digest| -> PathBuf {
        let manifest = Value::Map(vec![(
            text("objects"),
            Value::Map(vec![(
                text("t"),
                Value::Map(vec![
                    (text("shape"), Value::Array(vec![Value::Uint(256)])),
                    (text("type"), text("u8")),
                    (
                        text("blob"),
                        Value::Map(vec![
                            (text("offset"), Value::Uint(4096)),
                            (text("length"), Value::Uint(256)),
                            (
                                text("digest"),
                                Value::Map(vec![
                                    (text("algorithm"), text(&digest.algorithm)),
                                    (text("value"), Value::Bytes(digest.value)),
                                ]),
                            ),
                        ]),
                    ),
                ]),
            )]),
        )]);
        let encoded = cbor::encode(&manifest).unwrap();
        let mut bytes = vec![0u8; 8192];
        bytes[..8].copy_from_slice(&ztensor::format::MAGIC);
        bytes[4096..4096 + data.len()].copy_from_slice(&data);
        bytes.extend_from_slice(&encoded);
        let mut footer = [0u8; 40];
        footer[0..8].copy_from_slice(&8192u64.to_le_bytes());
        footer[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        footer[16..24].copy_from_slice(&xxh3_64(&encoded).to_le_bytes());
        footer[24..28].copy_from_slice(&ztensor::format::VERSION.to_le_bytes());
        footer[32..40].copy_from_slice(&ztensor::format::MAGIC);
        bytes.extend_from_slice(&footer);
        let path = tmp(name);
        fs::write(&path, &bytes).unwrap();
        path
    };

    let good = build("sha-obj-ok.zt", DigestAlgorithm::Sha256.digest(&data));
    let src = Source::open(&good).unwrap();
    assert_eq!(src.tensor("t").unwrap().digest().unwrap().algorithm, "sha256");
    assert!(
        src.tensor("t").unwrap().verify().unwrap().is_checked(),
        "a sha256 digest must be checked, not skipped"
    );

    // The same file, claiming the digest of different bytes.
    let bad = build("sha-obj-bad.zt", DigestAlgorithm::Sha256.digest(&[0u8; 256]));
    let err = Source::open(&bad)
        .unwrap()
        .tensor("t")
        .unwrap()
        .verify()
        .unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Digest), "{err}");
}

/// The digests are the ones the rest of the world computes.
///
/// A digest that is merely self-consistent would pass every round-trip test in
/// this file and still be useless: the point of `sha256` here is that a tool
/// which has never seen this crate can check a signature over it.
#[test]
fn the_digests_match_the_published_vectors() {
    assert_eq!(
        DigestAlgorithm::Sha256.digest(b"").to_string(),
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        DigestAlgorithm::Sha256.digest(b"abc").to_string(),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A whole-file digest is computed a chunk at a time, so that path has to
    // produce a real sha256 too, not just a self-consistent one.
    let long = vec![0x5au8; 1 << 20];
    let path = tmp("sha-chunked.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [long.len() as u64], Leaf::U8, &long).unwrap();
    w.finish().unwrap();

    let streamed = ztensor::read::shard_identity(&path, DigestAlgorithm::Sha256).unwrap();
    let whole_file = fs::read(&path).unwrap();
    assert_eq!(
        streamed.digest,
        DigestAlgorithm::Sha256.digest(&whole_file),
        "the chunked digest must equal the one-shot digest of the same bytes"
    );
    assert!(streamed.digest.matches(&whole_file).unwrap());
    assert_eq!(streamed.size, whole_file.len() as u64);
}

/// The by-identity resolver has to match whatever algorithm the root used.
///
/// It used to hash every candidate with one fixed algorithm and compare digest
/// strings, so a root whose shards are `sha256` could never match: the very
/// tables that signing needs (§6.6) were the ones it could not resolve. Sizes
/// are indexed instead, and the digest is computed in `resolve` in the
/// algorithm the shard asked for.
#[test]
fn the_directory_resolver_matches_a_sha256_shard_table() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dr-sha");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let shard = dir.join("some-name.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("t", [4u64], Leaf::U8, &[5u8; 4]).unwrap();
    w.finish().unwrap();
    let object = object_of(&shard, "t");

    for algo in [DigestAlgorithm::Sha256, DigestAlgorithm::Xxh3] {
        let root = dir.join("root.zt");
        let mut w = Writer::options().canonical(false).create(&root).unwrap();
        w.add_shard("s", &shard_identity(&shard, algo).unwrap())
            .unwrap();
        w.link("t", &object, "s").unwrap();
        w.finish().unwrap();

        // The name in the table says nothing about the file name on disk.
        let model = Source::options()
            .resolver(ztensor::read::DirectoryResolver::scan(&dir).unwrap())
            .open(&root)
            .unwrap();
        assert_eq!(
            &*model.tensor("t").unwrap().bytes().unwrap(),
            &[5u8; 4],
            "{algo:?} shard table"
        );
        model.verify_shards().unwrap();
        fs::remove_file(&root).unwrap();
    }
}

/// Shards of the *same size* are told apart, and each candidate is hashed once.
///
/// This is what a real sharded checkpoint looks like: a producer splitting at a
/// size limit emits files that are byte-for-byte the same length, so size
/// narrows nothing and every shard's identity comes down to its digest. The
/// resolver used to rehash the whole same-size bucket for every shard, which
/// makes an n-shard model cost O(n^2) whole-file hashes — invisible in a test
/// with two small files, and hours on a real checkpoint.
#[test]
fn equal_sized_shards_are_resolved_by_content() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("equalsize");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // Same shape, same type, same tensor name: identical layout, so the two
    // files differ only in their payload bytes and therefore only in digest.
    let mut ids = Vec::new();
    let mut objects = Vec::new();
    for (index, fill) in [1u8, 2].into_iter().enumerate() {
        let path = dir.join(format!("part-{index}.zt"));
        let mut w = Writer::create(&path).unwrap();
        w.add("t", [64u64], Leaf::U8, &[fill; 64]).unwrap();
        w.finish().unwrap();
        ids.push(shard_identity(&path, DigestAlgorithm::Xxh3).unwrap());
        objects.push(object_of(&path, "t"));
    }
    assert_eq!(ids[0].size, ids[1].size, "the shards must be the same size");
    assert_ne!(ids[0].digest, ids[1].digest);

    let root_path = dir.join("root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    for (index, id) in ids.iter().enumerate() {
        w.add_shard(format!("s{index}"), id).unwrap();
        w.link(format!("t{index}"), &objects[index], &format!("s{index}"))
            .unwrap();
    }
    w.finish().unwrap();

    // Rename both so nothing positional can help.
    fs::rename(dir.join("part-0.zt"), dir.join("zzz.zt")).unwrap();
    fs::rename(dir.join("part-1.zt"), dir.join("aaa.zt")).unwrap();

    let model = Source::options()
        .resolver(ztensor::read::DirectoryResolver::scan(&dir).unwrap())
        .open(&root_path)
        .unwrap();
    assert_eq!(&*model.tensor("t0").unwrap().bytes().unwrap(), &[1u8; 64][..]);
    assert_eq!(&*model.tensor("t1").unwrap().bytes().unwrap(), &[2u8; 64][..]);
    model.verify_shards().unwrap();
}

/// A named resolver has to survive being installed, which is the only thing
/// anyone does with one.
///
/// `positional` and `cas` return `impl ShardResolver`, and in edition 2021 an
/// opaque return type captures the function's type parameters — so without an
/// explicit `'static` the resolver borrowed from its `impl AsRef<Path>`
/// argument and `Options::resolver`, which needs `'static`, rejected it. Both
/// closures own everything they capture; the bound says so.
#[test]
fn a_named_resolver_outlives_the_path_it_was_built_from() {
    let installed = {
        let root = tmp("transient-root.zt");
        Source::options()
            .resolver(ztensor::read::positional(&root))
            .map(false)
    };
    // Built inside the scope, still usable outside it.
    assert!(installed.open(tmp("no-such-root.zt")).is_err());

    let dir = PathBuf::from("/store");
    let _ = Source::options().resolver(ztensor::read::cas(&dir));
}

/// An empty external range is a zero-length tensor, which the format allows.
/// A backwards one is a typo, and subtracting it would wrap.
#[test]
fn an_external_range_may_be_empty_but_not_backwards() {
    let shard_path = tmp("range-shard.zt");
    let mut w = Writer::create(&shard_path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1u8; 4]).unwrap();
    w.finish().unwrap();
    let id = shard_identity(&shard_path, DigestAlgorithm::Xxh3).unwrap();

    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(tmp("range-root.zt"))
        .unwrap();
    w.add_shard("s", &id).unwrap();

    // An empty range is the point of this half of the test.
    #[allow(clippy::reversed_empty_ranges)]
    let empty = 4096..4096;
    w.object("empty", |o| o.shape([0u64]).term(Leaf::U8).external("s", empty))
        .unwrap();

    #[allow(clippy::reversed_empty_ranges)]
    let backwards = 8192..4096;
    let err = w
        .object("backwards", |o| {
            o.shape([4u64]).term(Leaf::U8).external("s", backwards)
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("ends before it starts"),
        "got: {err}"
    );
    w.abandon();
}
