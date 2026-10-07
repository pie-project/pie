pub mod adapter;
pub mod catalog;
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
use poem_dsl::Dtype;

pub use poem_dsl::{ClassifyFn, Platform, Request, Stream, biases_name, scales_name};

#[must_use]
pub fn word(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// The dtype `word` spells, the inverse of [`word`].
#[must_use]
pub fn dtype_of(word: &str) -> Option<Dtype> {
    Dtype::ALL
        .iter()
        .copied()
        .find(|dtype| self::word(*dtype) == word)
}

/// One deployment of one model, under the name it is served by.
#[derive(Clone)]
pub struct Deployment {
    pub name: String,
    pub entry: &'static catalog::Entry,
    pub deploy: catalog::Deploy,
    pub classify: ClassifyFn,
    pub template: catalog::TemplateFn,
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

impl Deployment {
    #[must_use]
    pub fn of(entry: &'static catalog::Entry, deploy: catalog::Deploy) -> Deployment {
        Deployment {
            name: entry.name(&deploy),
            entry,
            classify: entry.classify,
            template: entry.template,
            tokenizer: entry.tokenizer,
            diffusion: (entry.diffusion)(&deploy),
            generative: (entry.generative)(&deploy),
            deploy,
        }
    }

    /// The deployment `name` spells.
    #[must_use]
    pub fn parse(name: &str) -> Option<Deployment> {
        catalog::parse(name).map(|(entry, deploy)| Deployment::of(entry, deploy))
    }

    /// Whether the model serves this deployment on `platform`.
    pub fn check(&self, platform: Platform) -> Result<(), catalog::Refused> {
        self.entry.check(&self.deploy, platform)
    }

    /// The trace each of the deployment's ranks runs.
    pub fn try_trace(&self, platform: Platform) -> Result<poem_ir::Trace, catalog::Refused> {
        self.entry.trace(&self.deploy, platform)
    }

    /// The trace each of the deployment's ranks runs, for a deployment the
    /// catalog lists, which always builds.
    #[must_use]
    pub fn trace(&self, platform: Platform) -> poem_ir::Trace {
        self.try_trace(platform)
            .unwrap_or_else(|why| panic!("`{}` does not build: {why}", self.name))
    }

    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        (self.entry.import)(&self.deploy, src, platform)
    }
}

/// An import reads a whole checkpoint, which one rank of a split row does not
/// land: its share is banded out of a stamped artifact instead.
pub fn whole(tp: u32) -> Result<(), checkpoint_dsl::Error> {
    if tp == 1 {
        return Ok(());
    }
    Err(checkpoint_dsl::Error::Illegible {
        name: String::new(),
        detail: format!(
            "an import states the WHOLE checkpoint and this contract is built for {tp} \
             ranks; nothing has banded the file it is reading"
        ),
    })
}

static FAMILIES: LazyLock<Vec<Vec<catalog::Entry>>> = LazyLock::new(|| {
    vec![
        deepseek_v4::entries(),
        flux_2::entries(),
        gemma_4::entries(),
        gemma_4_diffusion::entries(),
        glm_5::entries(),
        glm_5_next::entries(),
        gpt_oss::entries(),
        hunyuan_image_3::entries(),
        inkling::entries(),
        kimi_k3::entries(),
        muse_glimmer::entries(),
        qwen_3::entries(),
        qwen_4::entries(),
        z_image::entries(),
        wan_2::entries(),
        minimax_h3::entries(),
        ltx_2::entries(),
        mini_dit::entries(),
    ]
});

/// Every model of the catalog, published and fixture alike.
pub fn entries() -> impl Iterator<Item = &'static catalog::Entry> {
    FAMILIES.iter().flatten()
}

#[must_use]
pub fn entry(id: &str) -> Option<&'static catalog::Entry> {
    entries().find(|entry| entry.id == id)
}

/// Every deployment the catalog lists, family by family in the order each
/// family lists them.
static DEPLOYMENTS: LazyLock<Vec<Deployment>> = LazyLock::new(|| {
    FAMILIES
        .iter()
        .flat_map(|family| {
            let mut rows: Vec<(&'static catalog::Entry, &catalog::Row)> = family
                .iter()
                .flat_map(|entry| entry.rows.iter().map(move |row| (entry, row)))
                .collect();
            rows.sort_by_key(|(_, row)| row.seq);
            rows.into_iter()
                .map(|(entry, row)| Deployment::of(entry, row.deploy.clone()))
        })
        .collect()
});

/// Every deployment the catalog lists, fixtures' included.
pub fn deployments() -> impl Iterator<Item = &'static Deployment> {
    DEPLOYMENTS.iter()
}

/// The deployment the catalog lists under `name`.
#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    deployments().find(|deployment| deployment.name == name)
}

pub fn fits<'a>(
    src: &'a ztensor::Source,
    platform: Platform,
) -> impl Iterator<
    Item = (
        &'static Deployment,
        Result<ModelContract, checkpoint_dsl::Error>,
    ),
> + 'a {
    let mut candidates: Vec<&'static Deployment> = deployments()
        .filter(|d| d.deploy.tp == 1 && !d.entry.fixture)
        .collect();
    // A checkpoint is served with every part and drafter it carries unless a
    // config leaves them off, so the richest deployment that fits is tried
    // first; the catalog's order breaks ties.
    candidates.sort_by_key(|d| {
        std::cmp::Reverse(d.deploy.parts.len() + usize::from(d.deploy.drafter.is_some()))
    });
    candidates
        .into_iter()
        .map(move |d| (d, d.contract(src, platform)))
}

pub(crate) fn dense(banks: Dtype) -> Dtype {
    poem_dsl::compute_dtype(banks)
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
                         it is stored in; a second quantization is taken by `--deployment`, not by \
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
