//! The seam between the two halves of `pie.tokenizer/1`.
//!
//! `runtime/tokenizer` proves a tokenizer survives serialization, and
//! `checkpoint` proves a metadata object survives an artifact. Neither proves
//! the join: that the objects `to_canonical` produces, written under
//! `__meta__/` and read back out of a `.zt` file by name, rebuild the same
//! tokenizer. That join is the whole load path a served artifact uses, and it
//! is where an off-by-one in an offset or a name that disagrees between writer
//! and reader would land.

use std::collections::HashMap;

use checkpoint::file::meta;
use checkpoint::file::read;
use checkpoint::file::write::Writer;
use checkpoint::types::{DType, Encoding, TensorDecl, TensorId, Visibility};
use tokenizer::Tokenizer;
use tokenizer::canonical::CanonicalTokenizer;
// The object the checkpoint's own `config.json` is carried under, from the
// party that reads it back — see the note beside `pie model import`'s copy of
// this import.

/// Pulls every `__meta__/tokenizer/*` object out of an artifact and rebuilds
/// the tokenizer — the read path a served artifact runs.
fn read_tokenizer(artifact: &std::path::Path) -> anyhow::Result<Tokenizer> {
    let metadata = parse_metadata(artifact)?;
    let canonical = CanonicalTokenizer::from_objects(|path| metadata.get(path).cloned())?;
    Tokenizer::from_canonical(&canonical)
}

fn parse_metadata(artifact: &std::path::Path) -> anyhow::Result<HashMap<String, Vec<u8>>> {
    let checkpoint = read::parse_metadata(artifact)
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
    let mut writer = Writer::create(&artifact, &Default::default()).unwrap();
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
    let checkpoint = read::parse_metadata(&artifact).unwrap();
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

    let mut writer = Writer::create(&artifact, &Default::default()).unwrap();
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

