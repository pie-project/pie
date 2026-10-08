pub mod adapter;
pub mod catalog;
pub mod deepseek_v4;
pub mod drafter;
pub mod flux_2;
pub mod gemma_4;
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
pub mod star;
pub mod template;
pub mod tokenizer;
pub mod wan_2;
pub mod z_image;

use std::sync::LazyLock;

use checkpoint::contract::ModelContract;
use poem::Dtype;

pub use poem::generative::{
    AxisRole, Diffusion, Generative, LatentSpace, PortFact, PortKind, PositionConvention,
    ReadingFact, ReadoutKind, ScheduleFact, ScheduleKind,
};
pub use poem::{Platform, Request, Stream, biases_name, scales_name};

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
    pub template: catalog::TemplateFn,
    pub tokenizer: &'static tokenizer::Contract,
    pub diffusion: Option<Diffusion>,
    pub generative: Option<Generative>,
}

impl Deployment {
    #[must_use]
    pub fn of(entry: &'static catalog::Entry, deploy: catalog::Deploy) -> Deployment {
        Deployment {
            name: entry.name(&deploy),
            entry,
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
    ) -> Result<ModelContract, poem::import::Error> {
        (self.entry.import)(&self.deploy, src, platform)
    }
}

/// An import reads a whole checkpoint, which one rank of a split row does not
/// land: its share is banded out of a stamped artifact instead.
pub fn whole(tp: u32) -> Result<(), poem::import::Error> {
    if tp == 1 {
        return Ok(());
    }
    Err(poem::import::Error::Illegible {
        name: String::new(),
        detail: format!(
            "an import states the WHOLE checkpoint and this contract is built for {tp} \
             ranks; nothing has banded the file it is reading"
        ),
    })
}

static FAMILIES: LazyLock<Vec<Vec<catalog::Entry>>> = LazyLock::new(|| {
    let mut families = vec![
        deepseek_v4::entries(),
        flux_2::entries(),
        glm_5_next::entries(),
        hunyuan_image_3::entries(),
        kimi_k3::entries(),
        qwen_3::entries(),
        qwen_4::entries(),
        z_image::entries(),
        wan_2::entries(),
        minimax_h3::entries(),
        ltx_2::entries(),
    ];
    families.extend(star::families());
    families
});

/// Every model of the catalog, whole and miniature alike.
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

/// Every deployment the catalog lists, miniatures' included. Each runs on
/// one rank; [`splits`] are the same deployments across more.
pub fn deployments() -> impl Iterator<Item = &'static Deployment> {
    DEPLOYMENTS.iter()
}

/// The rank counts [`splits`] tries each listed deployment at.
pub const RANKS: [u32; 3] = [2, 4, 8];

/// Every listed deployment at every count of [`RANKS`] its model splits it
/// across, a deployment the compiler's sharding pass refuses left out.
static SPLITS: LazyLock<Vec<Deployment>> = LazyLock::new(|| {
    DEPLOYMENTS
        .iter()
        .flat_map(|whole| {
            RANKS.into_iter().filter_map(|tp| {
                let deploy = catalog::Deploy {
                    tp,
                    ..whole.deploy.clone()
                };
                let split = Deployment::of(whole.entry, deploy);
                split.check(Platform::Cuda).is_ok().then_some(split)
            })
        })
        .collect()
});

/// Every listed deployment split across the ranks of [`RANKS`] it splits
/// across.
pub fn splits() -> impl Iterator<Item = &'static Deployment> {
    SPLITS.iter()
}

/// The deployment `name` names: one the catalog lists, or one of its
/// [`splits`].
#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    deployments()
        .find(|deployment| deployment.name == name)
        .or_else(|| splits().find(|deployment| deployment.name == name))
}

pub fn fits<'a>(
    src: &'a ztensor::Source,
    platform: Platform,
) -> impl Iterator<
    Item = (
        &'static Deployment,
        Result<ModelContract, poem::import::Error>,
    ),
> + 'a {
    let mut candidates: Vec<&'static Deployment> = deployments().collect();
    // A miniature reads a prefix of its whole model's planes and would claim
    // the whole checkpoint too, so whole models are tried first. A checkpoint is
    // served with every part and drafter it carries unless a config leaves
    // them off, so the richest deployment that fits is tried first; the
    // catalog's order breaks ties.
    candidates.sort_by_key(|d| {
        (
            d.entry.mini,
            std::cmp::Reverse(d.deploy.parts.len() + usize::from(d.deploy.drafter.is_some())),
        )
    });
    candidates
        .into_iter()
        .map(move |d| (d, d.contract(src, platform)))
}

pub(crate) fn dense(banks: Dtype) -> Dtype {
    poem::compute_dtype(banks)
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
