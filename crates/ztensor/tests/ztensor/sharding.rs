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

fn write_data_shard(path: &PathBuf, offset: u64, payload: &[u8]) {
    let end = offset as usize + payload.len();
    let mut bytes = vec![0u8; end];
    bytes[..8].copy_from_slice(&ztensor::format::MAGIC);
    bytes[offset as usize..end].copy_from_slice(payload);
    let mut footer = [0u8; 40];
    footer[24..28].copy_from_slice(&ztensor::format::VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::format::MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

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

#[test]
fn sharding_every_case() {
    lora_overlay();
    positional_shards();
    shard_size_mismatch_rejected();
    shard_digest_mismatch_caught_by_deep_verify();
    canonical_is_single_file();
    a_single_file_is_the_degenerate_case();
    shards_found_by_identity_after_a_rename();
    a_shard_name_cannot_be_a_path();
    a_name_means_one_shard();
    resolver_trait_objects();
    a_sha256_shard_identity_round_trips();
    a_sha256_object_digest_is_verified();
    the_digests_match_the_published_vectors();
    the_directory_resolver_matches_a_sha256_shard_table();
    equal_sized_shards_are_resolved_by_content();
    a_named_resolver_outlives_the_path_it_was_built_from();
    an_external_range_may_be_empty_but_not_backwards();
}

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

    assert_eq!(
        model.tensor("base.weight").unwrap().map().unwrap(),
        &base_data[..]
    );
    assert_eq!(
        &*model.tensor("base.weight.lora_a").unwrap().bytes().unwrap(),
        &delta[..]
    );

    let base_at = model.tensor("base.weight").unwrap().locate().unwrap();
    let lora_at = model
        .tensor("base.weight.lora_a")
        .unwrap()
        .locate()
        .unwrap();
    assert_ne!(base_at.store, lora_at.store);
    assert_eq!(model.store(base_at.store).path(), base_path);
    assert_eq!(model.store(lora_at.store).path(), lora_path);

    let caps = model.tensor("base.weight").unwrap().caps();
    assert!(caps.map && caps.locate && caps.verify);
    if ztensor::provide::page_size() <= ztensor::format::ALIGN_CANONICAL {
        assert!(caps.evict, "{caps:?}");
    }

    assert!(model
        .tensor("base.weight")
        .unwrap()
        .verify()
        .unwrap()
        .is_checked());
    model.verify_shards().unwrap();
}

fn positional_shards() {
    let shard_path = tmp("posmodel-00001.zt");
    let payload = vec![9u8; 8192];
    let offset = 4096u64;
    write_data_shard(&shard_path, offset, &payload);
    let identity = shard_identity(&shard_path, DigestAlgorithm::Xxh3).unwrap();

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

    let model = Source::open(&root_path).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    assert!(model.tensor("t").unwrap().verify().unwrap().is_checked());
    model.verify_shards().unwrap();

    let caps = model.tensor("t").unwrap().caps();
    assert!(caps.map && caps.locate);
    assert!(
        !caps.evict,
        "a manifest-less shard cannot prove exclusivity"
    );
}

fn shard_size_mismatch_rejected() {
    let base_path = tmp("mismatch-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let mut identity = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();
    let base_object = object_of(&base_path, "t");
    identity.size += 4096;

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

fn shard_digest_mismatch_caught_by_deep_verify() {
    let base_path = tmp("digest-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [256u64], Leaf::U8, &[5u8; 256]).unwrap();
    w.finish().unwrap();
    let identity = shard_identity(&base_path, DigestAlgorithm::Xxh3).unwrap();
    let base_object = object_of(&base_path, "t");

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

    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
    assert!(model.tensor("t").unwrap().verify().is_err());
}

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

fn a_single_file_is_the_degenerate_case() {
    let path = tmp("single.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let model = Source::open(&path).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &[1, 2, 3, 4]);
    model.verify_shards().unwrap();
}

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
        let err = w.add_shard(name, &identity).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{name:?} was accepted, or reported as {err:?}"
        );
        w.abandon();
    }
}

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

fn resolver_trait_objects() {
    let cas = ztensor::read::cas("/store");
    let shard = Shard {
        size: 4096,
        digest: digest("00ff00ff00ff00ff"),
    };
    let path = cas.resolve("anything", &shard).unwrap();
    assert_eq!(path, PathBuf::from("/store/blobs/xxh3/00ff00ff00ff00ff"));
}

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
    assert_eq!(
        from_writer.digest.algorithm().unwrap(),
        DigestAlgorithm::Sha256
    );
    assert_eq!(from_writer.digest.value.len(), 32);
    assert!(from_writer.digest.to_string().starts_with("sha256:"));
    assert_eq!(from_writer.digest.to_string().len(), "sha256:".len() + 64);

    let scanned = ztensor::read::shard_identity(&shard_path, DigestAlgorithm::Sha256).unwrap();
    assert_eq!(scanned, from_writer);
    let default = ztensor::read::shard_identity(&shard_path, DigestAlgorithm::Xxh3).unwrap();
    assert_eq!(default.digest.algorithm, "xxh3");
    assert_eq!(default.size, from_writer.size);

    let root_path = tmp("sha-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("data", &from_writer).unwrap();
    let at = offset..offset + payload.len() as u64;
    w.object("t", |o| {
        o.shape([4096u64]).term(Leaf::U8).external("data", at)
    })
    .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&root_path, shard_path.clone()).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    model.verify_shards().unwrap();

    let mut bytes = fs::read(&shard_path).unwrap();
    bytes[4096] ^= 0xff;
    let corrupted = tmp("sha-shard-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();
    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
}

fn a_sha256_object_digest_is_verified() {
    use xxhash_rust::xxh3::xxh3_64;
    use ztensor::format::cbor::{self, Value};
    let data = vec![0xabu8; 256];
    let text = |s: &str| Value::Text(s.to_string());

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
    assert_eq!(
        src.tensor("t").unwrap().digest().unwrap().algorithm,
        "sha256"
    );
    assert!(
        src.tensor("t").unwrap().verify().unwrap().is_checked(),
        "a sha256 digest must be checked, not skipped"
    );

    let bad = build(
        "sha-obj-bad.zt",
        DigestAlgorithm::Sha256.digest(&[0u8; 256]),
    );
    let err = Source::open(&bad)
        .unwrap()
        .tensor("t")
        .unwrap()
        .verify()
        .unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Digest), "{err}");
}

fn the_digests_match_the_published_vectors() {
    assert_eq!(
        DigestAlgorithm::Sha256.digest(b"").to_string(),
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        DigestAlgorithm::Sha256.digest(b"abc").to_string(),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
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

fn equal_sized_shards_are_resolved_by_content() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("equalsize");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

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

    fs::rename(dir.join("part-0.zt"), dir.join("zzz.zt")).unwrap();
    fs::rename(dir.join("part-1.zt"), dir.join("aaa.zt")).unwrap();

    let model = Source::options()
        .resolver(ztensor::read::DirectoryResolver::scan(&dir).unwrap())
        .open(&root_path)
        .unwrap();
    assert_eq!(
        &*model.tensor("t0").unwrap().bytes().unwrap(),
        &[1u8; 64][..]
    );
    assert_eq!(
        &*model.tensor("t1").unwrap().bytes().unwrap(),
        &[2u8; 64][..]
    );
    model.verify_shards().unwrap();
}

fn a_named_resolver_outlives_the_path_it_was_built_from() {
    let installed = {
        let root = tmp("transient-root.zt");
        Source::options()
            .resolver(ztensor::read::positional(&root))
            .map(false)
    };
    assert!(installed.open(tmp("no-such-root.zt")).is_err());

    let dir = PathBuf::from("/store");
    let _ = Source::options().resolver(ztensor::read::cas(&dir));
}

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

    #[allow(clippy::reversed_empty_ranges)]
    let empty = 4096..4096;
    w.object("empty", |o| {
        o.shape([0u64]).term(Leaf::U8).external("s", empty)
    })
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
