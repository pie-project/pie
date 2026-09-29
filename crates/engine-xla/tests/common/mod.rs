//! Loading a model for a device test from `PIE_XLA_SNAPSHOT` + `PIE_XLA_SKU`
//! (a Hugging Face snapshot) or `PIE_XLA_ARTIFACT` (an imported `.zt`).

#![allow(dead_code)]

use std::path::PathBuf;

use checkpoint::contract::ModelContract;
use model_dsl::Platform;

pub struct Model {
    pub checkpoint: PathBuf,
    pub sku: &'static models::Sku,
    pub contract: ModelContract,
    pub tokenizer: Option<PathBuf>,
}

pub fn model() -> Option<Model> {
    match (
        std::env::var("PIE_XLA_ARTIFACT"),
        std::env::var("PIE_XLA_SNAPSHOT"),
        std::env::var("PIE_XLA_SKU"),
    ) {
        (Ok(artifact), snapshot, _) => {
            let artifact = PathBuf::from(artifact);
            let stamp = checkpoint::file::serve::stamp_of(&artifact)
                .expect("the artifact reads")
                .expect("the artifact carries a serving stamp");
            let sku = models::sku(&stamp.sku).unwrap_or_else(|| panic!("no SKU {}", stamp.sku));
            let trace = (sku.trace)(Platform::Xla);
            let source = ztensor_compat::index(&artifact).expect("the artifact opens");
            let contract =
                checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, Platform::Xla)
                    .unwrap_or_else(|why| panic!("the artifact holds every plane: {why}"));
            Some(Model {
                checkpoint: artifact,
                sku,
                contract,
                tokenizer: snapshot
                    .ok()
                    .map(|s| PathBuf::from(s).join("tokenizer.json")),
            })
        }
        (_, Ok(snapshot), Ok(name)) => {
            let snapshot = PathBuf::from(snapshot);
            let sku = models::sku(&name).unwrap_or_else(|| panic!("no SKU {name}"));
            let mut shards: Vec<PathBuf> = std::fs::read_dir(&snapshot)
                .expect("the snapshot lists")
                .filter_map(|e| {
                    let path = e.ok()?.path();
                    path.to_str()?.ends_with(".safetensors").then_some(path)
                })
                .collect();
            shards.sort();
            let source = ztensor_compat::index_all(&shards).expect("the snapshot opens");
            let contract = sku
                .contract(&source, Platform::Xla)
                .unwrap_or_else(|why| panic!("{name}'s import reads the snapshot: {why}"));
            Some(Model {
                tokenizer: Some(snapshot.join("tokenizer.json")),
                checkpoint: snapshot,
                sku,
                contract,
            })
        }
        _ => None,
    }
}

/// The model's tokenizer: the snapshot's `tokenizer.json`, else the
/// canonical tokenizer an imported artifact carries in its metadata.
pub fn tokenizer(m: &Model) -> tokenizer::Tokenizer {
    if let Some(path) = m.tokenizer.as_ref().filter(|p| p.exists()) {
        return tokenizer::Tokenizer::from_file(path).expect("the tokenizer reads");
    }
    use std::io::{Read, Seek, SeekFrom};
    let checkpoint = checkpoint::file::read::parse_metadata(&m.checkpoint)
        .expect("the artifact's metadata reads");
    let mut objects = std::collections::HashMap::new();
    for object in checkpoint.meta_objects() {
        let file = checkpoint
            .files
            .iter()
            .find(|file| file.id == object.file_id)
            .expect("every object names a file of the checkpoint");
        let mut f = std::fs::File::open(&file.path).expect("the artifact opens");
        f.seek(SeekFrom::Start(object.file_offset)).expect("seek");
        let mut bytes = vec![0u8; object.span_bytes as usize];
        f.read_exact(&mut bytes).expect("the object reads");
        let path = object
            .name
            .strip_prefix(checkpoint::file::meta::META_PREFIX)
            .expect("meta_objects yields prefixed names")
            .to_string();
        objects.insert(path, bytes);
    }
    let canonical =
        tokenizer::canonical::CanonicalTokenizer::from_objects(|path| objects.get(path).cloned())
            .expect("the artifact carries a tokenizer");
    tokenizer::Tokenizer::from_canonical(&canonical).expect("the tokenizer rebuilds")
}
