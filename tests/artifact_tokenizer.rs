//! The seam between the two halves of `pie.tokenizer/1`.
//!
//! `runtime/tokenizer` proves a tokenizer survives serialization, and
//! `model-loader` proves a metadata object survives an artifact. Neither proves
//! the join: that the objects `to_canonical` produces, written under
//! `__meta__/` and read back out of a `.zt` file by name, rebuild the same
//! tokenizer. That join is the whole load path a served artifact uses, and it
//! is where an off-by-one in an offset or a name that disagrees between writer
//! and reader would land.

use std::collections::HashMap;

use model_loader::checkpoint::meta;
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::checkpoint::write::CheckpointWriter;
use model_loader::types::{DType, Encoding, TensorDecl, TensorId, Visibility};
use tokenizer::Tokenizer;
use tokenizer::canonical::CanonicalTokenizer;

/// Pulls every `__meta__/tokenizer/*` object out of an artifact and rebuilds
/// the tokenizer — the read path a served artifact runs.
fn read_tokenizer(artifact: &std::path::Path) -> anyhow::Result<Tokenizer> {
    let metadata = parse_metadata(artifact)?;
    let canonical = CanonicalTokenizer::from_objects(|path| metadata.get(path).cloned())?;
    Tokenizer::from_canonical(&canonical)
}

fn parse_metadata(artifact: &std::path::Path) -> anyhow::Result<HashMap<String, Vec<u8>>> {
    let checkpoint = parse_checkpoint_metadata(artifact)
        .map_err(|err| anyhow::anyhow!("reading {}: {err}", artifact.display()))?;
    let mut objects = HashMap::new();
    for object in checkpoint.meta_objects() {
        let file = checkpoint
            .files
            .iter()
            .find(|file| file.id == object.file_id)
            .expect("every object names a file of the checkpoint");
        let bytes = std::fs::read(&file.path)?;
        let at = object.file_offset as usize;
        let end = at + object.span_bytes as usize;
        let path = object
            .name
            .strip_prefix(meta::META_PREFIX)
            .expect("meta_objects yields prefixed names")
            .to_string();
        objects.insert(path, bytes[at..end].to_vec());
    }
    Ok(objects)
}

/// Write a tokenizer into an artifact beside a weight; read it back; it
/// tokenizes identically.
#[test]
fn a_tokenizer_survives_the_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("model.zt");

    let vocab: Vec<String> = ["a", "b", "c", "ab", "abc", " ", "<pad>"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let original = Tokenizer::from_vocab(&vocab);
    let canonical = original.to_canonical().unwrap();

    // Weights and metadata go in together, in one ascending name order —
    // `__meta__/` (0x5F) sorts before a lowercase weight name.
    let mut writer = CheckpointWriter::create(&artifact, &Default::default()).unwrap();
    for (path, bytes) in canonical.objects() {
        writer.add_meta(path, bytes).unwrap();
    }
    writer
        .add_tensor(
            &TensorDecl {
                id: TensorId(0),
                name: "model.embed.weight".to_string(),
                shape: vec![4],
                encoding: Encoding::Raw(DType::U8),
                alignment: 1,
                visibility: Visibility::default(),
            },
            &[1u8, 2, 3, 4],
        )
        .unwrap();
    writer.finish().unwrap();

    let rebuilt = read_tokenizer(&artifact).unwrap();
    assert_eq!(rebuilt.vocab_size(), original.vocab_size());
    for text in ["abc", "a b c", "cab", ""] {
        let ids = original.encode(text);
        assert_eq!(rebuilt.encode(text), ids, "encoding {text:?}");
        assert_eq!(rebuilt.decode(&ids, false), original.decode(&ids, false));
    }

    // The weight is still a weight, and the tokenizer objects are still not.
    let checkpoint = parse_checkpoint_metadata(&artifact).unwrap();
    let weights: Vec<&str> = checkpoint.weights().map(|t| t.name.as_str()).collect();
    assert_eq!(weights, ["model.embed.weight"]);
    assert_eq!(checkpoint.meta_objects().count(), 5);
}

/// An artifact missing one of its tokenizer objects is refused by name rather
/// than silently rebuilding a tokenizer that behaves differently.
#[test]
fn an_incomplete_tokenizer_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("model.zt");

    let vocab: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let canonical = Tokenizer::from_vocab(&vocab).to_canonical().unwrap();

    let mut writer = CheckpointWriter::create(&artifact, &Default::default()).unwrap();
    for (path, bytes) in canonical.objects() {
        if path == tokenizer::canonical::MERGE_TABLE {
            continue;
        }
        writer.add_meta(path, bytes).unwrap();
    }
    writer.finish().unwrap();

    match read_tokenizer(&artifact) {
        Ok(_) => panic!("an artifact with no merge table produced a tokenizer"),
        Err(err) => assert!(
            err.to_string().contains(tokenizer::canonical::MERGE_TABLE),
            "unexpected error: {err}"
        ),
    }
}

/// The checkpoint's own config survives the artifact, byte for byte.
///
/// # What this used to check, and why the claim got simpler
///
/// It compared a normalized `pie.model/1` descriptor — 136 fields — against
/// `hf_config_dump/golden/*.json`, proving the Rust normalizer agreed with
/// `config.cpp` after a round trip through a `.zt`. Both sides of that
/// comparison are gone: the C++ normalizer was deleted, and the descriptor
/// with it, because a normalized config was identity crossing as a document.
/// Identity is now a manifest match against the tensors, which are already in
/// the artifact and need no separate copy.
///
/// What a config is still needed for is the one thing tensors cannot say — the
/// declared quantization, since a group size is not an extent of anything — so
/// what the artifact carries is the checkpoint's own `config.json`, unread and
/// unaltered. That makes the property checkable without a golden: the bytes
/// are the same bytes.
///
/// Point `PIE_TEST_ARTIFACT` at a converted artifact, and optionally
/// `PIE_TEST_CONFIG` at the `config.json` it was converted from; a no-op
/// without the first.
#[test]
fn a_converted_artifact_carries_the_checkpoints_own_config() {
    let Ok(artifact) = std::env::var("PIE_TEST_ARTIFACT") else {
        return;
    };

    let objects = parse_metadata(std::path::Path::new(&artifact)).unwrap();
    let raw = objects
        .get(model::encoding::CONFIG_OBJECT)
        .unwrap_or_else(|| {
            panic!(
                "the artifact carries no {}; it was written before the config was \
                 carried verbatim, or from a source that had none",
                model::encoding::CONFIG_OBJECT
            )
        });

    // Valid JSON, because an artifact must never carry an object no reader can
    // open — `pie model import` checks this at write time and this is the
    // other end of that check.
    let parsed: serde_json::Value =
        serde_json::from_slice(raw).expect("the carried config is not JSON");
    assert!(
        parsed.is_object(),
        "a config.json that is not an object cannot be a config"
    );

    // And it says something `Encoding` can read, which is the only reason it
    // is carried at all.
    let text = std::str::from_utf8(raw).expect("the carried config is not UTF-8");
    let encoding =
        model::encoding::Encoding::from_config_json(text).expect("Encoding cannot read it");

    if let Ok(source) = std::env::var("PIE_TEST_CONFIG") {
        let want = std::fs::read(&source).expect("cannot read PIE_TEST_CONFIG");
        assert_eq!(
            raw, &want,
            "the artifact's config differs from {source}; verbatim means verbatim, \
             and a config that was rewritten on the way in is one the driver reads \
             differently from the checkpoint's author"
        );
    }

    eprintln!(
        "{artifact}: carries {} ({} bytes, quant method {:?})",
        model::encoding::CONFIG_OBJECT,
        raw.len(),
        encoding.method
    );
}

/// The same read path against an artifact `pie model import` actually wrote,
/// checked against the `tokenizer.json` it was built from.
///
/// Hermetic tests use `from_vocab`, which is the fixture pipeline: no merges,
/// no regex splitters, no added tokens. Point `PIE_TEST_ARTIFACT` at a
/// converted `.zt` and `PIE_TEST_TOKENIZER` at its source tokenizer to check a
/// production vocabulary end to end; the test is a no-op otherwise.
#[test]
fn a_converted_artifact_serves_its_source_tokenizer() {
    let (Ok(artifact), Ok(source)) = (
        std::env::var("PIE_TEST_ARTIFACT"),
        std::env::var("PIE_TEST_TOKENIZER"),
    ) else {
        return;
    };

    let expected = Tokenizer::from_file(std::path::Path::new(&source))
        .unwrap_or_else(|err| panic!("loading {source}: {err:#}"));
    let actual = read_tokenizer(std::path::Path::new(&artifact))
        .unwrap_or_else(|err| panic!("reading {artifact}: {err:#}"));

    assert_eq!(actual.vocab_size(), expected.vocab_size());
    assert_eq!(actual.special_token_ids(), expected.special_token_ids());
    for text in [
        "Hello, world!",
        "日本語テスト 中文测试 한국어",
        "fn main() { let x = 42; }",
        "<|im_start|>system\nHello<|im_end|>",
        "  spaces   and\ttabs  ",
        "🎵",
    ] {
        let ids = expected.encode(text);
        assert_eq!(actual.encode(text), ids, "encoding {text:?}");
        assert_eq!(
            actual.decode(&ids, false),
            expected.decode(&ids, false),
            "decoding {text:?}"
        );
    }
    eprintln!(
        "{artifact}: tokenizer matches {source} ({} tokens)",
        actual.vocab_size()
    );
}
