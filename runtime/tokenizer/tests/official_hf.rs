mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::TEXTS;
use pie_tokenizer::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

struct Model {
    id: &'static str,
    fixture: &'static str,
    revision: &'static str,
}

const MODELS: &[Model] = &[
    Model {
        id: "Qwen/Qwen3-0.6B",
        fixture: "qwen3",
        revision: "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    Model {
        id: "Qwen/Qwen3.6-27B",
        fixture: "qwen36",
        revision: "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    },
    Model {
        id: "deepseek-ai/DeepSeek-V4-Pro",
        fixture: "deepseek-v4",
        revision: "b5968e9190ef611bbf34a7229255be88a0e937c1",
    },
    Model {
        id: "google/gemma-4-E4B-it",
        fixture: "gemma4",
        revision: "fee6332c1abaafb77f6f9624236c63aa2f1d0187",
    },
    Model {
        id: "zai-org/GLM-5.2",
        fixture: "glm52",
        revision: "b4734de4facf877f85769a911abafc5283eab3d9",
    },
    Model {
        id: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        fixture: "nemotron3",
        revision: "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    },
    Model {
        id: "microsoft/Phi-3-mini-4k-instruct",
        fixture: "phi3",
        revision: "f39ac1d28e925b323eae81227eaba4464caced4e",
    },
];

#[test]
#[ignore = "requires official tokenizer snapshots"]
fn official_hf_models_match_exactly() {
    for model in MODELS {
        assert_model_matches_exactly(&tokenizer_path(model), model.id);
    }
}

/// Same parity check for Phi-3 alone, gated on the snapshot's presence so it
/// runs in a default `cargo test` wherever the model is already cached.
#[test]
fn official_phi3_matches_exactly_when_cached() {
    let model = MODELS
        .iter()
        .find(|model| model.fixture == "phi3")
        .expect("phi3 model entry");
    let path = PathBuf::from(std::env::var_os("HOME").expect("HOME is not set"))
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model.id.replace('/', "--")))
        .join("snapshots")
        .join(model.revision)
        .join("tokenizer.json");
    if !path.is_file() {
        eprintln!("skipping: {} is not cached", model.id);
        return;
    }
    assert_model_matches_exactly(&path, model.id);
}

fn assert_model_matches_exactly(path: &Path, id: &str) {
    let pie = Arc::new(
        Tokenizer::from_file(path)
            .unwrap_or_else(|error| panic!("loading {id} with Pie: {error:#}")),
    );
    let hf = HfTokenizer::from_file(path)
        .unwrap_or_else(|error| panic!("loading {id} with HF: {error}"));

    assert_eq!(
        pie.vocab_size(),
        hf.get_vocab_size(true),
        "{id} vocabulary size"
    );
    for &text in TEXTS {
        let pie_ids = pie.encode(text);
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(pie_ids, hf_ids, "{id} encoding {text:?}");
        assert_eq!(
            pie.decode(&hf_ids, false),
            hf.decode(&hf_ids, false).unwrap(),
            "{id} HF→Pie decode {text:?}"
        );
        assert_eq!(
            pie.decode(&pie_ids, false),
            hf.decode(&pie_ids, false).unwrap(),
            "{id} Pie→HF decode {text:?}"
        );
        assert_eq!(
            pie.decode(&hf_ids, true),
            hf.decode(&hf_ids, true).unwrap(),
            "{id} special-token filtering {text:?}"
        );
        let mut decoder = pie.decoder(false);
        let mut incremental = String::new();
        for token in &hf_ids {
            incremental.push_str(&decoder.feed(std::slice::from_ref(token)));
        }
        incremental.push_str(&decoder.finish());
        assert_eq!(
            incremental,
            pie.decode(&hf_ids, false),
            "{id} incremental decode {text:?}"
        );
    }
}

fn tokenizer_path(model: &Model) -> PathBuf {
    if let Some(root) = std::env::var_os("PIE_TOKENIZER_FIXTURES_DIR") {
        let path = Path::new(&root).join(model.fixture).join("tokenizer.json");
        assert!(
            path.is_file(),
            "missing {} fixture at {}",
            model.id,
            path.display()
        );
        return path;
    }

    let cache = PathBuf::from(std::env::var_os("HOME").expect("HOME is not set"))
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model.id.replace('/', "--")))
        .join("snapshots")
        .join(model.revision)
        .join("tokenizer.json");
    assert!(
        cache.is_file(),
        "{} revision {} is not cached; set PIE_TOKENIZER_FIXTURES_DIR",
        model.id,
        model.revision
    );
    cache
}
