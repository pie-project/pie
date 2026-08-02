//! The seam between the two halves of `pie.tokenizer/1`.
//!
//! `runtime/tokenizer` proves a tokenizer survives serialization, and
//! `pie-loader` proves a metadata object survives an artifact. Neither proves
//! the join: that the objects `to_canonical` produces, written under
//! `__meta__/` and read back out of a `.zt` file by name, rebuild the same
//! tokenizer. That join is the whole load path a served artifact uses, and it
//! is where an off-by-one in an offset or a name that disagrees between writer
//! and reader would land.

use std::collections::HashMap;

use pie_loader::checkpoint::meta;
use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::CheckpointWriter;
use pie_loader::types::{DType, Encoding, TensorDecl, TensorId, Visibility};
use pie_tokenizer::Tokenizer;
use pie_tokenizer::canonical::CanonicalTokenizer;

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
        if path == pie_tokenizer::canonical::MERGE_TABLE {
            continue;
        }
        writer.add_meta(path, bytes).unwrap();
    }
    writer.finish().unwrap();

    match read_tokenizer(&artifact) {
        Ok(_) => panic!("an artifact with no merge table produced a tokenizer"),
        Err(err) => assert!(
            err.to_string()
                .contains(pie_tokenizer::canonical::MERGE_TABLE),
            "unexpected error: {err}"
        ),
    }
}

/// The `pie.model/1` descriptor survives the artifact, and says what the C++
/// normalizer says.
///
/// The differential in `pie-model-config` proves the normalizer agrees with
/// `config.cpp`; this proves the agreement survives being written into a `.zt`
/// and read back out — and that `head_dim_kernel`, which is a property of a
/// driver build rather than of a checkpoint, is *not* along for the ride.
///
/// Point `PIE_TEST_ARTIFACT` at a converted artifact and `PIE_TEST_GOLDEN` at
/// the matching `driver/cuda/tests/hf_config_dump/golden/*.json`; a no-op
/// otherwise.
#[test]
fn a_converted_artifact_carries_the_normalized_model_config() {
    let (Ok(artifact), Ok(golden)) = (
        std::env::var("PIE_TEST_ARTIFACT"),
        std::env::var("PIE_TEST_GOLDEN"),
    ) else {
        return;
    };

    let objects = parse_metadata(std::path::Path::new(&artifact)).unwrap();
    let raw = objects
        .get("model/descriptor")
        .expect("the artifact carries no model descriptor");
    let actual: serde_json::Value = serde_json::from_slice(raw).unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();

    assert_eq!(actual["version"], "pie.model/1");
    assert!(
        actual.get("head_dim_kernel").is_none(),
        "head_dim_kernel is a driver-build fact and must not be in the artifact"
    );

    let expected = expected.as_object().unwrap();
    let actual_map = actual.as_object().unwrap();
    for (key, want) in expected {
        if key == "head_dim_kernel" {
            continue;
        }
        let got = actual_map
            .get(key)
            .unwrap_or_else(|| panic!("descriptor is missing {key}"));
        // Numbers are f32 on both sides; compare at that width rather than as
        // printed text.
        match (got.as_f64(), want.as_f64()) {
            (Some(a), Some(b)) => assert_eq!(a as f32, b as f32, "{key}"),
            _ => assert_eq!(got, want, "{key}"),
        }
    }
    eprintln!(
        "{artifact}: descriptor matches {golden} ({} fields)",
        expected.len() - 1
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
