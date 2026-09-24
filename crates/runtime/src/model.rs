use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

use models::template::Instruct;
use tokenizer::Tokenizer;

static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    pub config: Vec<u8>,
}

pub struct Row {
    pub id: &'static str,
    pub layers: u32,
    pub vocab: u32,
    pub arch: &'static str,
}

pub const ROWS: &[Row] = &[
    Row {
        id: "dsv4-base-bf16-kv-bf16",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-base-bf16-kv-bf16-tp2",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-bf16-kv-bf16",
        layers: 43,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-u4g64-u2g64-kv-bf16",
        layers: 5,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv41-flash-u4g64-kv-bf16",
        layers: 40,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv41-flash-bf16-mxfp4-kv-bf16",
        layers: 40,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv41-flash-mini-bf16-mxfp4-kv-bf16",
        layers: 8,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-mini-bf16-kv-bf16",
        layers: 5,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        layers: 5,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-full-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        layers: 43,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-full-u4g64-u2g64-kv-bf16",
        layers: 43,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "gemma4-e4b-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-e4b-bf16-kv-bf16-tp2",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-e4b-eagle-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-e4b-vision-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "diffusiongemma-26b-a4b-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "diffusiongemma-26b-a4b-u8g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "diffusiongemma-26b-a4b-u8g64-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "diffusiongemma-26b-a4b-u4g64-u8g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "diffusiongemma-26b-a4b-u8g64-u4g64-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "diffusiongemma-26b-a4b-bf16-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "diffusion_gemma",
    },
    Row {
        id: "gemma4-26b-a4b-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-26b-a4b-mtp-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-26b-a4b-dflash-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-26b-a4b-vision-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-mtp-u4g64-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-u4g64-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-vision-u4g64-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16-tp2",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "glm5-a12b-bf16-kv-bf16",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    Row {
        id: "glm5-a12b-bf16-kv-bf16-tp2",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    Row {
        id: "glm53-flash-mini-u4g64-u4g64-kv-bf16",
        layers: 8,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mini-u4g64-u4g64-kv-bf16-tp2",
        layers: 8,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mtp-u8g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mtp-vision-u8g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-vision-u8g64-u2g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-u8g64-u2g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mtp-vision-u4g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "gptoss-20b-bf16-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-20b-u4g64-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-20b-dflash-u4g64-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-20b-mini-bf16-mxfp4-kv-bf16",
        layers: 5,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-20b-mini-bf16-mxfp4-kv-bf16-tp2",
        layers: 5,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "kimik3-mini-bf16-mxfp4-kv-bf16",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    Row {
        id: "kimik3-mini-bf16-mxfp4-kv-bf16-tp2",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16-tp2",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    Row {
        id: "qwen36-27b-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-mtp-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-dflash-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-dflash2-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-dspark-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-mtp-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-mtp-u4g64-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-dflash-u4g64-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-u4g64-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-mini-u4g64-kv-bf16",
        layers: 5,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-mini-u4g64-kv-bf16-tp2",
        layers: 5,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-mini64-u4g64-kv-bf16",
        layers: 5,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-u4g64-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-tiny-u4g64-kv-bf16",
        layers: 4,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d2b-u4g64-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d4b-u4g64-kv-bf16",
        layers: 32,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d9b-u4g64-kv-bf16",
        layers: 32,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d9b-dflash-u4g64-kv-bf16",
        layers: 32,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16-tp2",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-flash-next-u4g64-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-bf16-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-u4g64-u2g128-kv-bf16",
        layers: 4,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-mtp-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-mtp-vision-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-vision-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen35-d3b-bf16-kv-bf16",
        layers: 24,
        vocab: 151_936,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-bf16-kv-bf16-tp2",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-vision-eagle-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-vision-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-vision-u4g64-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-vision-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-vision-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-vision-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-vision-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-eagle-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "z-image-turbo-bf16-kv-bf16",
        layers: 35,
        vocab: 151_936,
        arch: "z_image",
    },
    Row {
        id: "z-image-turbo-u4g64-kv-bf16",
        layers: 35,
        vocab: 151_936,
        arch: "z_image",
    },
    Row {
        id: "z-image-mini-bf16-kv-bf16",
        layers: 6,
        vocab: 0,
        arch: "z_image",
    },
    Row {
        id: "flux2-klein-4b-bf16-kv-bf16",
        layers: 27,
        vocab: 151_936,
        arch: "flux_2",
    },
    Row {
        id: "flux2-klein-4b-u4g64-kv-bf16",
        layers: 27,
        vocab: 151_936,
        arch: "flux_2",
    },
    Row {
        id: "flux2-mini-bf16-kv-bf16",
        layers: 4,
        vocab: 0,
        arch: "flux_2",
    },
    Row {
        id: "hunyuanimage3-80b-a13b-bf16-u8g64-kv-bf16",
        layers: 32,
        vocab: 133_120,
        arch: "hunyuan_image_3_moe",
    },
    Row {
        id: "hunyuanimage3-80b-a13b-bf16-u8g64-kv-bf16-tp4",
        layers: 32,
        vocab: 133_120,
        arch: "hunyuan_image_3_moe",
    },
    Row {
        id: "hunyuanimage3-80b-a13b-bf16-u4g64-kv-bf16-tp4",
        layers: 32,
        vocab: 133_120,
        arch: "hunyuan_image_3_moe",
    },
    Row {
        id: "hunyuanimage3-mini-bf16-kv-bf16",
        layers: 2,
        vocab: 133_120,
        arch: "hunyuan_image_3_moe",
    },
    Row {
        id: "minimax-h3-fl2va-bf16-kv-bf16",
        layers: 50,
        vocab: 151_936,
        arch: "minimax_h3",
    },
    Row {
        id: "minimax-h3-fl2va-bf16-kv-bf16-tp2",
        layers: 50,
        vocab: 151_936,
        arch: "minimax_h3",
    },
    Row {
        id: "minimax-h3-fl2va-bf16-kv-bf16-tp4",
        layers: 50,
        vocab: 151_936,
        arch: "minimax_h3",
    },
    Row {
        id: "minimax-h3-mini-bf16-kv-bf16",
        layers: 3,
        vocab: 0,
        arch: "minimax_h3",
    },
    Row {
        id: "wan22-ti2v-5b-bf16-kv-bf16",
        layers: 24,
        vocab: 256_384,
        arch: "wan_2",
    },
    Row {
        id: "wan22-ti2v-5b-u4g64-kv-bf16",
        layers: 24,
        vocab: 256_384,
        arch: "wan_2",
    },
    Row {
        id: "wan22-mini-d128-bf16-kv-bf16",
        layers: 2,
        vocab: 0,
        arch: "wan_2",
    },
    Row {
        id: "wan22-mini-nano-bf16-kv-bf16",
        layers: 2,
        vocab: 0,
        arch: "wan_2",
    },
    Row {
        id: "ltx25-bf16-kv-bf16",
        layers: 48,
        vocab: 0,
        arch: "ltx_2",
    },
    Row {
        id: "ltx25-u4g64-kv-bf16",
        layers: 48,
        vocab: 0,
        arch: "ltx_2",
    },
    Row {
        id: "ltx25-mini-bf16-kv-bf16",
        layers: 2,
        vocab: 0,
        arch: "ltx_2",
    },
    Row {
        id: "mini-dit-bf16-kv-bf16",
        layers: 3,
        vocab: 0,
        arch: "mini_dit",
    },
    Row {
        id: "mini-dit-bf16-kv-bf16-tp2",
        layers: 3,
        vocab: 0,
        arch: "mini_dit",
    },
    Row {
        id: "mini-dit-bf16-kv-bf16-tp4",
        layers: 3,
        vocab: 0,
        arch: "mini_dit",
    },
    Row {
        id: "muse-glimmer-30b-bf16-kv-bf16",
        layers: 52,
        vocab: 202_048,
        arch: "muse_glimmer",
    },
    Row {
        id: "muse-glimmer-30b-bf16-kv-bf16-tp2",
        layers: 52,
        vocab: 202_048,
        arch: "muse_glimmer",
    },
    Row {
        id: "muse-glimmer-30b-u4g64-kv-bf16",
        layers: 52,
        vocab: 202_048,
        arch: "muse_glimmer",
    },
    Row {
        id: "muse-glimmer-30b-mini-l8-bf16-kv-bf16",
        layers: 8,
        vocab: 202_048,
        arch: "muse_glimmer",
    },
    Row {
        id: "inkling-bf16-kv-bf16",
        layers: 66,
        vocab: 200_058,
        arch: "inkling",
    },
    Row {
        id: "inkling-mini-l7-e8-bf16-kv-bf16",
        layers: 7,
        vocab: 200_058,
        arch: "inkling",
    },
];

#[must_use]
pub fn row(id: &str) -> Option<&'static Row> {
    ROWS.iter().find(|row| row.id == id)
}

#[must_use]
pub fn ids() -> Vec<&'static str> {
    ROWS.iter().map(|row| row.id).collect()
}

#[must_use]
pub fn nearest_ids(id: &str, take: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = ids()
        .into_iter()
        .map(|k| (edit_distance(id, k), k))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(take).map(|(_, k)| k).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn compiled_tokenizer(metadata: &ModelMetadata) -> Option<Result<Tokenizer>> {
    let objects = metadata.tokenizer.as_ref()?;
    Some((|| {
        let canonical = tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            objects
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })?;
        Tokenizer::from_canonical(&canonical)
    })())
}

fn loaded_row(model_id: &str) -> Result<&'static Row> {
    row(model_id).ok_or_else(|| {
        anyhow!(
            "the engine loaded {model_id:?}, which this build's model catalog \
             does not contain; nearest ids: {:?}",
            nearest_ids(model_id, 3)
        )
    })
}

pub fn register(
    name: String,
    model_id: &str,
    kv_page_size: u32,
    rs: RsCaps,
    eta: EtaCaps,
    tokenizer_path: PathBuf,
    metadata: &ModelMetadata,
) -> Result<()> {
    let row = loaded_row(model_id)?;
    let num_layers = row.layers;
    let vocab_size = row.vocab;
    let tokenizer = match compiled_tokenizer(metadata) {
        Some(compiled) => compiled?,
        None => Tokenizer::from_file(&tokenizer_path)?,
    };
    let tokenizer = Arc::new(tokenizer);
    match models::tokenizer::contract_of(row.id) {
        Some(contract) => contract
            .verify(&tokenizer)
            .map_err(|fault| anyhow!("`{}` refuses this artifact's tokenizer: {fault}", row.id))?,
        None => {
            return Err(anyhow!(
                "this build serves {:?} but ships no tokenizer contract for \
                 it; `models::tokenizer::contracts()` has no row under that \
                 SKU",
                row.id
            ));
        }
    }
    let instruct = match models::template::template_of(row.id) {
        Some(make) => make(tokenizer.clone()),
        None => {
            return Err(anyhow!(
                "this build serves {:?} but ships no chat template for it; \
                 `models::template::templates()` has no row under that SKU",
                row.id
            ));
        }
    };

    let catalog_row = models::sku(row.id).ok_or_else(|| {
        anyhow!(
            "this build serves {:?} but its model catalog states no classifier \
             for it; a lane's fact word cannot be computed",
            row.id
        )
    })?;
    let classify = catalog_row.classify;
    let diffusion = catalog_row.diffusion;
    let generative = catalog_row.generative.clone();
    if let Some(generative) = &generative {
        validate_generative(generative).map_err(|fault| {
            anyhow!(
                "`{}` states generative facts this runtime refuses: {fault}",
                row.id
            )
        })?;
    }

    let model = Arc::new(Model {
        name,
        arch_name: row.arch,
        instruct,
        classify,
        kv_page_size,
        rs_caps: rs,
        eta_caps: eta,
        tokenizer,
        vocab: OnceLock::new(),
        vocab_size,
        num_layers,
        diffusion,
        generative,
    });
    MODEL.set(model).map_err(|_| {
        anyhow!("a model is already registered; the runtime serves exactly one model")
    })?;
    Ok(())
}

pub fn validate_generative(generative: &models::Generative) -> Result<(), String> {
    let mut velocity_width = None;
    for (at, reading) in generative.readings.iter().enumerate() {
        if usize::from(reading.index) != at {
            return Err(format!(
                "reading `{}` sits at position {at} but states index {}; readings are \
                 listed in index order, dense from 0",
                reading.name, reading.index
            ));
        }
        if reading.name.is_empty() {
            return Err(format!("reading {at} has an empty name"));
        }
        if generative.readings[..at]
            .iter()
            .any(|r| r.name == reading.name)
        {
            return Err(format!("reading `{}` is declared twice", reading.name));
        }
        if reading.readout_width == 0 {
            return Err(format!(
                "reading `{}` states a zero-width readout",
                reading.name
            ));
        }
        if reading.readout == models::ReadoutKind::Velocity {
            match velocity_width {
                None => velocity_width = Some(reading.readout_width),
                Some(width) if width != reading.readout_width => {
                    return Err(format!(
                        "reading `{}` reads a velocity {} wide beside another reading's {width}; \
                         the eta profile carries one velocity width",
                        reading.name, reading.readout_width
                    ));
                }
                Some(_) => {}
            }
        }
        for (i, port) in reading.ports.iter().enumerate() {
            if port.name.is_empty() {
                return Err(format!(
                    "reading `{}` port {i} has an empty name",
                    reading.name
                ));
            }
            if reading.ports[..i].iter().any(|p| p.name == port.name) {
                return Err(format!(
                    "reading `{}` declares port `{}` twice",
                    reading.name, port.name
                ));
            }
            if port.width == 0 {
                return Err(format!(
                    "reading `{}` port `{}` states a zero width",
                    reading.name, port.name
                ));
            }
            if port.kind == models::PortKind::AxisPositions && port.width > 4 {
                return Err(format!(
                    "reading `{}` port `{}` states {} axes; a positions port carries 1..=4",
                    reading.name, port.name, port.width
                ));
            }
        }
        if let Some(convention) = &reading.positions {
            let axes = reading
                .ports
                .iter()
                .find(|port| port.kind == models::PortKind::AxisPositions)
                .map(|port| port.width);
            let Some(axes) = axes else {
                return Err(format!(
                    "reading `{}` states a position convention but declares no \
                     axis-positions port",
                    reading.name
                ));
            };
            if convention.axes.len() != axes as usize {
                return Err(format!(
                    "reading `{}` states {} axis roles for a {axes}-wide positions port",
                    reading.name,
                    convention.axes.len()
                ));
            }
            if convention.text_axis >= axes {
                return Err(format!(
                    "reading `{}` numbers its text rows on axis {} of a {axes}-axis \
                     positions port",
                    reading.name, convention.text_axis
                ));
            }
        }
        if !reading.takes_tokens
            && !reading.ports.iter().any(|port| {
                matches!(
                    port.kind,
                    models::PortKind::Latents
                        | models::PortKind::Voxels
                        | models::PortKind::Context
                )
            })
        {
            return Err(format!(
                "reading `{}` embeds no tokens and declares no latents, context or voxels port; \
                 nothing states its lane's row count",
                reading.name
            ));
        }
    }
    Ok(())
}

pub fn velocity_facts(readings: &[models::ReadingFact]) -> (bool, u32) {
    readings
        .iter()
        .find(|reading| reading.readout == models::ReadoutKind::Velocity)
        .map_or((false, 0), |reading| (true, reading.readout_width))
}

pub fn pixels_facts(readings: &[models::ReadingFact]) -> (bool, u32) {
    let mut widths = readings
        .iter()
        .filter(|reading| reading.readout == models::ReadoutKind::Pixels)
        .map(|reading| reading.readout_width);
    let Some(first) = widths.next() else {
        return (false, 0);
    };
    (
        true,
        if widths.all(|width| width == first) {
            first
        } else {
            0
        },
    )
}

pub fn model() -> &'static Arc<Model> {
    MODEL.get().expect("model accessed before registration")
}

pub fn media_pad() -> Option<u32> {
    static PAD: OnceLock<Option<u32>> = OnceLock::new();
    *PAD.get_or_init(|| {
        use crate::inferlet::host::media::multimodal;
        let m = model();
        let arch = m.arch_name();
        let spelling = models::media::vision_front_end(arch)
            .map(|fe| fe.delimiters().placeholder)
            .or_else(|| {
                multimodal::audio_arch_supported(arch).then(multimodal::audio_placeholder)
            })?;
        match m.tokenize(spelling)[..] {
            [id] => Some(id),
            _ => None,
        }
    })
}

pub struct Model {
    name: String,
    arch_name: &'static str,
    instruct: Arc<dyn Instruct>,
    classify: models::ClassifyFn,
    kv_page_size: u32,
    rs_caps: RsCaps,
    eta_caps: EtaCaps,
    tokenizer: Arc<Tokenizer>,
    vocab: OnceLock<(Vec<u32>, Vec<Vec<u8>>)>,
    vocab_size: u32,
    num_layers: u32,
    diffusion: Option<models::Diffusion>,
    generative: Option<models::Generative>,
}

#[derive(Debug, Clone, Copy)]
pub struct RsCaps {
    pub state_size: u64,
    pub buffer_page_size: u32,
    pub fold_granularity: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EtaCaps {
    pub has_mtp_logits: bool,
    pub mtp_depth: u32,
    pub draft_block: u32,
    pub draft_mask_token: u32,
    pub draft_bidirectional: bool,
    pub draft_proposals_from: u32,
    pub has_value_head: bool,
    pub has_kv_envelopes: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("name", &self.name).finish()
    }
}

impl Model {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arch_name(&self) -> &'static str {
        self.arch_name
    }

    pub fn instruct(&self) -> &dyn Instruct {
        &*self.instruct
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, false)
    }

    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.vocab().clone()
    }

    fn vocab(&self) -> &(Vec<u32>, Vec<Vec<u8>>) {
        self.vocab.get_or_init(|| {
            let size = self.tokenizer.vocab_size();
            let mut ids = Vec::with_capacity(size);
            let mut bytes = Vec::with_capacity(size);
            for id in 0..size as u32 {
                if let Some(tok_bytes) = self.tokenizer.id_to_token(id) {
                    ids.push(id);
                    bytes.push(tok_bytes);
                }
            }
            (ids, bytes)
        })
    }

    pub fn token_bytes(&self, tokens: &[u32]) -> Vec<Vec<u8>> {
        tokens
            .iter()
            .map(|id| self.tokenizer.id_to_token(*id).unwrap_or_default())
            .collect()
    }

    pub fn tokens_with_prefix(&self, prefix: &[u8]) -> Vec<u32> {
        self.tokenizer.ids_with_prefix(prefix)
    }

    pub fn get_split_regex(&self) -> String {
        self.tokenizer.get_split_regex()
    }

    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_special_tokens()
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn word(
        &self,
        query_len: u32,
        custom_mask: bool,
        adapter: bool,
        drafts: bool,
        captures_scores: bool,
        media: bool,
        block_draft: bool,
        denoise: bool,
        stream: models::Stream,
        reading: u8,
    ) -> u64 {
        (self.classify)(
            &models::Request::new(query_len, custom_mask)
                .adapted(adapter)
                .drafting(drafts)
                .capturing_scores(captures_scores)
                .with_media(media)
                .drafting_a_block(block_draft)
                .denoising(denoise)
                .on_stream(stream)
                .in_reading(reading),
        )
    }

    pub fn diffusion(&self) -> Option<models::Diffusion> {
        self.diffusion
    }

    pub fn generative(&self) -> Option<&models::Generative> {
        self.generative.as_ref()
    }

    pub fn readings(&self) -> &[models::ReadingFact] {
        self.generative
            .as_ref()
            .map_or(&[], |generative| generative.readings.as_slice())
    }

    pub fn sole_reading(&self) -> Option<&models::ReadingFact> {
        match self.readings() {
            [only] => Some(only),
            _ => None,
        }
    }

    pub fn kv_page_size(&self) -> u32 {
        self.kv_page_size
    }

    pub fn num_layers(&self) -> u32 {
        self.num_layers
    }

    pub fn rs_caps(&self) -> RsCaps {
        self.rs_caps
    }

    pub fn eta_caps(&self) -> EtaCaps {
        self.eta_caps
    }
}
