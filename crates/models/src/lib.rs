pub mod adapter;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod media;
pub mod qwen_3;
pub mod qwen_4;
pub mod template;
pub mod tokenizer;

use std::sync::LazyLock;

use checkpoint::contract::ModelContract;
use model_dsl::Dtype;

pub use model_dsl::{ClassifyFn, Platform, Request, biases_name, scales_name};

/// What a SKU is: the text it serves, the numeric forms its weight banks are
/// stored in (dense first, then routed experts when they differ), the kv
/// dtype, and the tensor-parallel width it is traced and imported at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub text: &'static str,
    pub weights: &'static [Dtype],
    pub kv: Dtype,
    pub tp: u32,
}

impl Recipe {
    /// `<text>-<weights..>-kv-<kv>[-tp<n>]`: the name a deployment keys on.
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

/// A dtype as a SKU name spells it: the variant, lowercased.
#[must_use]
pub fn word(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// One shipping SKU, stated once: its recipe and how it traces, classifies,
/// imports, chats and tokenizes.
pub struct Sku {
    pub name: String,
    pub recipe: Recipe,
    pub trace: model_dsl::TraceFn,
    pub classify: ClassifyFn,
    pub import: ImportFn,
    pub template: fn(std::sync::Arc<::tokenizer::Tokenizer>) -> std::sync::Arc<dyn template::Instruct>,
    pub tokenizer: &'static tokenizer::Contract,
}

impl Sku {
    /// This SKU's reading of a checkpoint, at its own width.
    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        (self.import)(src, self.recipe.tp, platform)
    }
}

/// A family's reading of a checkpoint, for a stated world width and platform.
pub type ImportFn =
    fn(&ztensor::Source, u32, Platform) -> Result<ModelContract, checkpoint_dsl::Error>;

/// One family's SKU table. `$m` is `|tp: u32| Model::…(…, tp)`; the same
/// closure traces, classifies and imports the row, so the three cannot
/// disagree about the model.
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
            }
        } ),+ ]
    };
}

static SKUS: LazyLock<Vec<Sku>> = LazyLock::new(|| {
    [
        deepseek_v4::skus(),
        gemma_4::skus(),
        glm_5::skus(),
        gpt_oss::skus(),
        kimi_k3::skus(),
        qwen_3::skus(),
        qwen_4::skus(),
    ]
    .into_iter()
    .flatten()
    .collect()
});

/// Every SKU this build ships, in identification order.
pub fn skus() -> impl Iterator<Item = &'static Sku> {
    SKUS.iter()
}

#[must_use]
pub fn sku(name: &str) -> Option<&'static Sku> {
    skus().find(|sku| sku.name == name)
}

/// Every one-rank SKU's reading of `src`, in identification order: the
/// first that reads is what the checkpoint is.
pub fn fits<'a>(
    src: &'a ztensor::Source,
    platform: Platform,
) -> impl Iterator<Item = (&'static Sku, Result<ModelContract, checkpoint_dsl::Error>)> + 'a {
    skus()
        .filter(|sku| sku.recipe.tp == 1)
        .map(move |sku| (sku, sku.contract(src, platform)))
}

/// The dtype the planes beside a bank of `banks` are stated in (e.g. a
/// layernorm next to a quantized weight, which is never itself quantized).
pub(crate) fn dense(banks: Dtype) -> Dtype {
    model_dsl::compute_dtype(banks)
        .unwrap_or_else(|| panic!("`{banks:?}` is not a weight representation a family declares"))
}

pub fn identify(src: &ztensor::Source, platform: Platform) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    for (sku, read) in fits(src, platform) {
        match read {
            Ok(_) => return Ok(&sku.name),
            Err(why) => misses.push((&sku.name, why.to_string())),
        }
    }
    Err(Unmatched { misses })
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
