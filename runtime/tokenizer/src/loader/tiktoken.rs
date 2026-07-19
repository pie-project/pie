//! Native tiktoken rank-file loader.
//!
//! A `.tiktoken` rank file contains token bytes and merge ranks, but not the
//! regex required to split input before BPE. We resolve that missing behavior
//! from the official tokenizer class declared by the sibling
//! `tokenizer_config.json`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, Pipeline, Tokenizer};

const KIMI_TIKTOKEN_REGEX: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const KIMI_RESERVED_SPECIAL_TOKENS: u32 = 256;

#[derive(Clone, Copy)]
struct TiktokenProfile {
    split_regex: &'static str,
    reserved_special_tokens: u32,
}

const KIMI_PROFILE: TiktokenProfile = TiktokenProfile {
    split_regex: KIMI_TIKTOKEN_REGEX,
    reserved_special_tokens: KIMI_RESERVED_SPECIAL_TOKENS,
};

/// Load a tiktoken rank file using its official tokenizer profile.
pub fn from_file(path: &Path) -> Result<Tokenizer> {
    let (config, profile) = load_config(path)?;
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut map: HashMap<u32, Vec<u8>> = HashMap::new();

    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let bytes_b64 = parts.next().ok_or_else(|| {
            anyhow::anyhow!("{}:{}: missing token bytes", path.display(), line_no + 1)
        })?;
        let rank = parts
            .next()
            .ok_or_else(|| {
                anyhow::anyhow!("{}:{}: missing token rank", path.display(), line_no + 1)
            })?
            .parse::<u32>()
            .with_context(|| format!("{}:{}: invalid token rank", path.display(), line_no + 1))?;
        if parts.next().is_some() {
            bail!(
                "{}:{}: expected exactly token bytes and rank",
                path.display(),
                line_no + 1
            );
        }
        let bytes = BASE64_STANDARD.decode(bytes_b64).with_context(|| {
            format!(
                "{}:{}: invalid base64 token bytes",
                path.display(),
                line_no + 1
            )
        })?;
        if map.insert(rank, bytes).is_some() {
            bail!(
                "{}:{}: duplicate token rank {rank}",
                path.display(),
                line_no + 1
            );
        }
    }

    if map.is_empty() {
        bail!("{}: empty tiktoken rank file", path.display());
    }
    let base_vocab_size = u32::try_from(map.len()).context("tiktoken vocabulary too large")?;
    let max_rank = map.keys().copied().max().unwrap_or(0);
    if max_rank.checked_add(1) != Some(base_vocab_size) {
        bail!(
            "{}: tiktoken ranks must be contiguous from 0 ({} entries, max rank {max_rank})",
            path.display(),
            map.len()
        );
    }

    let mut bpe = BpeTable::from_decoder_map(map)?;
    ensure!(
        bpe.has_unique_token_bytes(),
        "tiktoken ranks contain duplicate token bytes"
    );
    ensure!(
        bpe.has_all_byte_atoms(),
        "tiktoken profile requires all 256 byte atoms"
    );
    let added_tokens =
        config.into_added_tokens(base_vocab_size, profile.reserved_special_tokens)?;
    for token in &added_tokens {
        bpe.insert_added(token.content.as_bytes().to_vec(), token.id)?;
    }
    let split_regex =
        fancy_regex::Regex::new(profile.split_regex).context("compiling tiktoken split regex")?;

    Tokenizer::new(
        bpe,
        Pipeline::ByteLevelRegex {
            nfc: false,
            splitters: vec![split_regex],
            bpe_mode: BpeMode::PreferWholeToken,
        },
        added_tokens,
    )
}

#[derive(serde::Deserialize)]
struct TiktokenTokenizerConfig {
    #[serde(default)]
    tokenizer_class: Option<String>,
    #[serde(default)]
    auto_map: HashMap<String, Vec<Option<String>>>,
    #[serde(default)]
    added_tokens_decoder: HashMap<String, TiktokenConfigToken>,
}

#[derive(serde::Deserialize)]
struct TiktokenConfigToken {
    content: String,
    #[serde(default)]
    special: bool,
}

impl TiktokenTokenizerConfig {
    fn into_added_tokens(
        self,
        base_vocab_size: u32,
        reserved_special_tokens: u32,
    ) -> Result<Vec<AddedToken>> {
        let end = base_vocab_size
            .checked_add(reserved_special_tokens)
            .context("tiktoken special-token range overflows u32")?;
        let mut configured = HashMap::with_capacity(self.added_tokens_decoder.len());

        for (id, token) in self.added_tokens_decoder {
            let id = id
                .parse::<u32>()
                .with_context(|| format!("invalid added-token ID {id:?}"))?;
            if !(base_vocab_size..end).contains(&id) {
                bail!(
                    "tiktoken added-token ID {id} is outside reserved range \
                     {base_vocab_size}..{end}"
                );
            }
            if configured.insert(id, token).is_some() {
                bail!("duplicate tiktoken added-token ID {id}");
            }
        }

        let mut contents = HashMap::with_capacity(reserved_special_tokens as usize);
        let mut tokens = Vec::with_capacity(reserved_special_tokens as usize);
        for id in base_vocab_size..end {
            let (content, special) = match configured.remove(&id) {
                Some(token) => (token.content, token.special),
                None => (format!("<|reserved_token_{id}|>"), false),
            };
            if let Some(previous_id) = contents.insert(content.clone(), id) {
                bail!(
                    "duplicate tiktoken added-token content {content:?} for IDs \
                     {previous_id} and {id}"
                );
            }
            tokens.push(AddedToken {
                id,
                content,
                special,
            });
        }
        Ok(tokens)
    }
}

fn load_config(path: &Path) -> Result<(TiktokenTokenizerConfig, TiktokenProfile)> {
    let dir = path
        .parent()
        .context("tiktoken model path has no parent directory")?;
    let config_path = dir.join("tokenizer_config.json");
    let data = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: TiktokenTokenizerConfig = serde_json::from_slice(&data)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    let implementation = config
        .auto_map
        .get("AutoTokenizer")
        .and_then(|entry| entry.first())
        .and_then(Option::as_deref);

    let profile = match (config.tokenizer_class.as_deref(), implementation) {
        (Some("TikTokenTokenizer"), Some("tokenization_kimi.TikTokenTokenizer")) => KIMI_PROFILE,
        _ => {
            bail!(
                "unsupported tiktoken tokenizer in {}: rank files require a \
                 format-specific split regex; recognized profiles: Kimi K2/K2.5",
                config_path.display()
            )
        }
    };
    Ok((config, profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_loader_matches_split_and_special_token_contract() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiktoken.model");
        let mut rank_bytes = (0u16..=255)
            .map(|byte| vec![byte as u8])
            .collect::<Vec<_>>();
        rank_bytes.extend([b"12".to_vec(), b"123".to_vec(), b"1234".to_vec()]);
        let mut model = String::new();
        for (rank, bytes) in rank_bytes.iter().enumerate() {
            model.push_str(&format!(
                "{} {rank}\n",
                BASE64_STANDARD.encode(bytes.as_slice())
            ));
        }
        std::fs::write(&model_path, model).unwrap();

        let base_vocab_size = rank_bytes.len() as u32;
        let mut added_tokens = serde_json::Map::new();
        added_tokens.insert(
            base_vocab_size.to_string(),
            serde_json::json!({"content": "<|im_user|>", "special": true}),
        );
        added_tokens.insert(
            (base_vocab_size + 1).to_string(),
            serde_json::json!({"content": "<|tool|>", "special": false}),
        );
        added_tokens.insert(
            (base_vocab_size + 255).to_string(),
            serde_json::json!({"content": "[PAD]", "special": true}),
        );
        let config = serde_json::json!({
            "tokenizer_class": "TikTokenTokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]
            },
            "added_tokens_decoder": added_tokens
        });
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        let tokenizer = from_file(&model_path).unwrap();
        assert_eq!(tokenizer.get_split_regex(), KIMI_TIKTOKEN_REGEX);
        assert_eq!(
            tokenizer.vocab_size(),
            (base_vocab_size + KIMI_RESERVED_SPECIAL_TOKENS) as usize
        );
        assert_eq!(
            tokenizer.decoded_token_bytes(base_vocab_size),
            None,
            "special tokens must be excluded from grammar bytes"
        );
        assert_eq!(
            tokenizer.token_to_id(&format!("<|reserved_token_{}|>", base_vocab_size + 2)),
            Some(base_vocab_size + 2)
        );

        assert_eq!(tokenizer.encode("1234"), vec![257, b'4' as u32]);
        assert_eq!(
            tokenizer.encode("<|im_user|>1234<|tool|>"),
            vec![base_vocab_size, 257, b'4' as u32, base_vocab_size + 1]
        );
        assert_eq!(
            tokenizer.decode(&[base_vocab_size, b'h' as u32, base_vocab_size + 1], false),
            "<|im_user|>h<|tool|>"
        );
        assert_eq!(
            tokenizer.decode(&[base_vocab_size, b'h' as u32, base_vocab_size + 1], true),
            "h<|tool|>"
        );
    }

    #[test]
    fn rejects_unknown_format_without_split_regex() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("vocab.tiktoken");
        std::fs::write(&model_path, format!("{} 0\n", BASE64_STANDARD.encode("a"))).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
              "tokenizer_class": "OtherTokenizer",
              "auto_map": {
                "AutoTokenizer": ["tokenization_other.OtherTokenizer", null]
              }
            }"#,
        )
        .unwrap();

        let error = crate::loader::from_file(&model_path).err().unwrap();
        assert!(error.to_string().contains("unsupported tiktoken tokenizer"));
    }
}
