pub mod adapter;
pub mod deepseek_v4;
pub mod drafter;
pub mod flux_2;
pub mod gemma_4;
pub mod gemma_4_diffusion;
pub mod glm_5;
pub mod glm_5_next;
pub mod gpt_oss;
pub mod hunyuan_image_3;
pub mod inkling;
pub mod kimi_k3;
pub mod ltx_2;
pub mod media;
pub mod mini_dit;
pub mod minimax_h3;
pub mod muse_glimmer;
pub mod numpy;
pub mod published;
pub mod qwen_3;
pub mod qwen_4;
pub mod template;
pub mod tokenizer;
pub mod wan_2;
pub mod z_image;

use std::sync::LazyLock;

use checkpoint::contract::ModelContract;
use model_dsl::Dtype;

pub use model_dsl::{ClassifyFn, Platform, Request, Stream, biases_name, scales_name};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub text: &'static str,
    pub weights: &'static [Dtype],
    pub kv: Dtype,
    pub tp: u32,
}

impl Recipe {
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = self.text.to_string();
        for dtype in self.weights {
            name.push('-');
            name.push_str(&word(*dtype));
        }
        name.push_str("-kv-");
        name.push_str(&word(self.kv));
        if self.tp > 1 {
            name.push_str(&format!("-tp{}", self.tp));
        }
        name
    }
}

#[must_use]
pub fn word(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

pub struct Sku {
    pub name: String,
    pub recipe: Recipe,
    pub trace: model_dsl::TraceFn,
    pub classify: ClassifyFn,
    pub import: ImportFn,
    pub template:
        fn(std::sync::Arc<::tokenizer::Tokenizer>) -> std::sync::Arc<dyn template::Instruct>,
    pub tokenizer: &'static tokenizer::Contract,
    pub diffusion: Option<Diffusion>,
    pub generative: Option<Generative>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generative {
    pub readings: Vec<ReadingFact>,
    pub latent: Option<LatentSpace>,
    pub schedule: Option<ScheduleFact>,
    pub max_rows: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadingFact {
    pub name: &'static str,
    pub index: u8,
    pub has_kv: bool,
    pub takes_tokens: bool,
    pub streams: Vec<Stream>,
    pub ports: Vec<PortFact>,
    pub positions: Option<PositionConvention>,
    pub readout: ReadoutKind,
    pub readout_width: u32,
}

impl ReadingFact {
    #[must_use]
    pub fn port(&self, name: &str) -> Option<(u8, &PortFact)> {
        self.ports_indexed().find(|(_, port)| port.name == name)
    }

    pub fn ports_indexed(&self) -> impl Iterator<Item = (u8, &PortFact)> + '_ {
        let mut seen = [0u8; 5];
        self.ports.iter().map(move |port| {
            let slot = match port.kind {
                PortKind::Latents => 0,
                PortKind::LaneVector => 1,
                PortKind::Context => 2,
                PortKind::AxisPositions => 3,
                PortKind::Voxels => 4,
            };
            let index = port.at.unwrap_or(seen[slot]);
            seen[slot] = index.saturating_add(1);
            (index, port)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortFact {
    pub name: &'static str,
    pub kind: PortKind,
    pub width: u32,
    pub streams: Vec<Stream>,
    pub at: Option<u8>,
    pub rows: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortKind {
    Latents,
    LaneVector,
    Context,
    AxisPositions,
    Voxels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisRole {
    Time,
    Height,
    Width,
    Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionConvention {
    pub axes: Vec<AxisRole>,
    pub text_axis: u32,
    pub text_origin: u32,
    pub image_follows_text: bool,
    pub reference_stride: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadoutKind {
    Logits,
    Velocity,
    Hidden,
    Pixels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentSpace {
    pub channels: u32,
    pub patch_t: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    pub spatial_compression: u32,
    pub temporal_compression: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleFact {
    pub kind: ScheduleKind,
    pub shift: f32,
    pub train_steps: u32,
    pub boundary: Option<f32>,
    pub pinned_sigmas: Vec<f32>,
    pub stream_shifts: Vec<(Stream, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Flow,
    Epsilon,
    V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Diffusion {
    pub canvas: u32,
    pub hidden: u32,
    pub self_cond_taps: u32,
}

impl Sku {
    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        (self.import)(src, self.recipe.tp, platform)
    }
}

pub type ImportFn =
    fn(&ztensor::Source, u32, Platform) -> Result<ModelContract, checkpoint_dsl::Error>;

#[macro_export]
macro_rules! skus {
    ($( ($text:literal, $tp:literal, [$($w:expr),+ $(,)?], $kv:expr, $trace:path, $template:expr, $tokenizer:expr, $m:expr $(,)?) ),+ $(,)?) => {
        vec![ $( {
            const RECIPE: $crate::Recipe = $crate::Recipe {
                text: $text,
                weights: &[$($w),+],
                kv: $kv,
                tp: $tp,
            };
            $crate::Sku {
                name: RECIPE.name(),
                recipe: RECIPE,
                trace: |platform: $crate::Platform| {
                    $trace(&RECIPE.name(), &($m)(RECIPE.tp), platform)
                },
                classify: |request: &$crate::Request| {
                    model_dsl::word_of(|| ($m)(RECIPE.tp), request)
                },
                import: |src: &ztensor::Source, tp: u32, platform: $crate::Platform| {
                    ($m)(tp).import(src, platform)
                },
                template: $template,
                tokenizer: $tokenizer,
                diffusion: None,
                generative: None,
            }
        } ),+ ]
    };
}

static SKUS: LazyLock<Vec<Sku>> = LazyLock::new(|| {
    [
        deepseek_v4::skus(),
        flux_2::skus(),
        gemma_4::skus(),
        gemma_4_diffusion::skus(),
        glm_5::skus(),
        glm_5_next::skus(),
        gpt_oss::skus(),
        hunyuan_image_3::skus(),
        inkling::skus(),
        kimi_k3::skus(),
        muse_glimmer::skus(),
        qwen_3::skus(),
        qwen_4::skus(),
        z_image::skus(),
        wan_2::skus(),
        minimax_h3::skus(),
        ltx_2::skus(),
        mini_dit::skus(),
    ]
    .into_iter()
    .flatten()
    .collect()
});

pub fn skus() -> impl Iterator<Item = &'static Sku> {
    SKUS.iter()
}

#[must_use]
pub fn sku(name: &str) -> Option<&'static Sku> {
    skus().find(|sku| sku.name == name)
}

pub fn fits<'a>(
    src: &'a ztensor::Source,
    platform: Platform,
) -> impl Iterator<Item = (&'static Sku, Result<ModelContract, checkpoint_dsl::Error>)> + 'a {
    skus()
        .filter(|sku| sku.recipe.tp == 1)
        .map(move |sku| (sku, sku.contract(src, platform)))
}

pub(crate) fn dense(banks: Dtype) -> Dtype {
    model_dsl::compute_dtype(banks)
        .unwrap_or_else(|| panic!("`{banks:?}` is not a weight representation a family declares"))
}

pub fn identify(src: &ztensor::Source, platform: Platform) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    for (sku, read) in fits(src, platform) {
        match read {
            Ok(contract) => match requantizes(&contract) {
                None => return Ok(&sku.name),
                Some(plane) => misses.push((
                    &sku.name,
                    format!(
                        "reads this checkpoint only by re-quantizing `{plane}` from the form \
                         it is stored in; a second quantization is taken by `--sku`, not by \
                         identification"
                    ),
                )),
            },
            Err(why) => misses.push((&sku.name, why.to_string())),
        }
    }
    Err(Unmatched { misses })
}

pub fn requantizes(contract: &checkpoint::contract::ModelContract) -> Option<String> {
    use checkpoint::types::Encoding;
    contract.tensors.iter().find_map(|stored| {
        let name = stored.name.strip_suffix(".stored")?;
        if !matches!(stored.encoding, Encoding::Quant(_)) {
            return None;
        }
        let published = contract.tensors.iter().find(|t| t.name == name)?;
        matches!(published.encoding, Encoding::Quant(_)).then(|| name.to_string())
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmatched {
    pub misses: Vec<(&'static str, String)>,
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "this checkpoint matches no SKU this build ships")?;
        for (sku, why) in &self.misses {
            write!(f, "\n  {sku}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unmatched {}
