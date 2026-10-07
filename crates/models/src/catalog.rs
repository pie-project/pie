//! The catalog: each model once, at its whole size, with the parts its
//! checkpoint can carry and the drafters it pairs with; and a deployment,
//! which picks the precision, the kv dtype, the rank count, the parts it
//! serves and the drafter, and is checked against the model.

use std::sync::Arc;

use checkpoint::contract::ModelContract;
use poem_dsl::{ClassifyFn, Dtype, Platform, Trace};

use crate::{Diffusion, Generative, Recipe, template, tokenizer};

/// A part a checkpoint may carry besides the text trunk, which a deployment
/// may leave off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Part {
    Vision,
    SelfCond,
}

/// What proposes draft tokens for the trunk to verify: its own multi-token
/// head (`Mtp`) or a published draft model the checkpoint was imported with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Drafter {
    Mtp,
    DFlash,
    DFlash2,
    DSpark,
    Eagle,
}

/// How one model is served.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Deploy {
    /// The weight precision: one dtype, or the banks of a mixed recipe in the
    /// order the model states them.
    pub weights: Vec<Dtype>,
    pub kv: Dtype,
    pub tp: u32,
    pub parts: Vec<Part>,
    pub drafter: Option<Drafter>,
}

impl Deploy {
    #[must_use]
    pub fn has(&self, part: Part) -> bool {
        self.parts.contains(&part)
    }

    #[must_use]
    pub fn drafts_with(&self, drafter: Drafter) -> bool {
        self.drafter == Some(drafter)
    }

    /// The one weight dtype of a single-precision deployment.
    pub fn dtype(&self) -> Result<Dtype, Refused> {
        match self.weights[..] {
            [dtype] => Ok(dtype),
            _ => Err(Refused(format!(
                "the weights are one dtype here, not {:?}",
                self.weights
            ))),
        }
    }
}

/// Why a model does not serve a deployment.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Refused(pub String);

impl Refused {
    #[must_use]
    pub fn unsupported(what: &str, deploy: &Deploy) -> Refused {
        Refused(format!("{what} does not ship {deploy:?}"))
    }
}

pub type TraceFn = fn(&str, &Deploy, Platform) -> Result<Trace, Refused>;
pub type ImportFn =
    fn(&Deploy, &ztensor::Source, Platform) -> Result<ModelContract, checkpoint_dsl::Error>;
pub type TemplateFn = fn(Arc<::tokenizer::Tokenizer>) -> Arc<dyn template::Instruct>;

/// One model of the catalog.
pub struct Entry {
    pub id: &'static str,
    /// A cut-down model the tests trace and serve, not a published one.
    pub fixture: bool,
    pub parts: &'static [Part],
    pub drafters: &'static [Drafter],
    pub trace: TraceFn,
    pub import: ImportFn,
    pub classify: ClassifyFn,
    pub template: TemplateFn,
    pub tokenizer: &'static tokenizer::Contract,
    pub diffusion: fn(&Deploy) -> Option<Diffusion>,
    pub generative: fn(&Deploy) -> Option<Generative>,
    /// The deployments the catalog has always named, by the row name each
    /// was served under, with each row's place in its family's list.
    pub rows: Vec<Row>,
}

pub struct Row {
    pub seq: u32,
    pub recipe: Recipe,
    pub deploy: Deploy,
}

impl Entry {
    /// Whether this model serves `deploy` on `platform`: the parts and the
    /// drafter are ones it has, the model builds at that precision, and its
    /// trace splits across the ranks.
    pub fn check(&self, deploy: &Deploy, platform: Platform) -> Result<(), Refused> {
        if let Some(part) = deploy.parts.iter().find(|p| !self.parts.contains(p)) {
            return Err(Refused(format!("`{}` carries no {part:?} part", self.id)));
        }
        if let Some(drafter) = deploy.drafter
            && !self.drafters.contains(&drafter)
        {
            return Err(Refused(format!(
                "`{}` pairs with no {drafter:?} drafter",
                self.id
            )));
        }
        if deploy.tp == 0 {
            return Err(Refused("a deployment runs on at least one rank".into()));
        }
        let trace = (self.trace)(&self.name(deploy), deploy, platform)?;
        poem_compiler::shard::shard(trace, deploy.tp)
            .map(|_| ())
            .map_err(|why| Refused(why.to_string()))
    }

    /// The trace each rank of `deploy` runs.
    pub fn trace(&self, deploy: &Deploy, platform: Platform) -> Result<Trace, Refused> {
        let trace = (self.trace)(&self.name(deploy), deploy, platform)?;
        poem_compiler::shard::shard(trace, deploy.tp).map_err(|why| Refused(why.to_string()))
    }

    /// The name a deployment of this model is served under: the row the
    /// catalog has always called it, or one spelled the same way.
    #[must_use]
    pub fn name(&self, deploy: &Deploy) -> String {
        if let Some(row) = self.rows.iter().find(|row| row.deploy == *deploy) {
            return row.recipe.name();
        }
        let mut name = self.id.to_string();
        for part in &deploy.parts {
            name.push('-');
            name.push_str(&format!("{part:?}").to_lowercase());
        }
        if let Some(drafter) = deploy.drafter {
            name.push('-');
            name.push_str(&format!("{drafter:?}").to_lowercase());
        }
        for dtype in &deploy.weights {
            name.push('-');
            name.push_str(&crate::word(*dtype));
        }
        name.push_str("-kv-");
        name.push_str(&crate::word(deploy.kv));
        if deploy.tp > 1 {
            name.push_str(&format!("-tp{}", deploy.tp));
        }
        name
    }
}

/// A catalog entry: its id, parts, drafters and family hooks, the function
/// that builds its model for a deployment, and the rows it was named by.
#[macro_export]
macro_rules! entry {
    (
        id: $id:literal,
        fixture: $fixture:literal,
        parts: [$($part:ident),* $(,)?],
        drafters: [$($drafter:ident),* $(,)?],
        template: $template:expr,
        tokenizer: $tokenizer:expr,
        diffusion: $diffusion:expr,
        generative: $generative:expr,
        build: |$d:ident| -> $model:ty $body:block,
        rows: [ $( ($seq:literal, $text:literal, $tp:literal, [$($w:expr),+ $(,)?], $kv:expr, [$($rpart:ident),* $(,)?], $rdrafter:expr $(,)?) ),* $(,)? ] $(,)?
    ) => {{
        fn build($d: &$crate::catalog::Deploy) -> Result<$model, $crate::catalog::Refused> $body
        $crate::catalog::Entry {
            id: $id,
            fixture: $fixture,
            parts: &[$($crate::catalog::Part::$part),*],
            drafters: &[$($crate::catalog::Drafter::$drafter),*],
            trace: |name, deploy, platform| {
                Ok(poem_dsl::trace_hybrid(name, &build(deploy)?, platform))
            },
            import: |deploy, src, platform| {
                $crate::whole(deploy.tp)?;
                build(deploy)
                    .map_err(|why| checkpoint_dsl::Error::Illegible {
                        name: $id.to_string(),
                        detail: why.to_string(),
                    })?
                    .import(src, platform)
            },
            classify: |request| {
                poem_dsl::word_of(
                    || -> $model { unreachable!("a classifier reads the facts' type only") },
                    request,
                )
            },
            template: $template,
            tokenizer: $tokenizer,
            diffusion: |deploy| {
                let diffusion: Option<fn(&$model) -> $crate::Diffusion> = $diffusion;
                diffusion.and_then(|f| build(deploy).ok().map(|m| f(&m)))
            },
            generative: |deploy| {
                let generative: Option<fn(&$model) -> $crate::Generative> = $generative;
                generative.and_then(|g| build(deploy).ok().map(|m| g(&m)))
            },
            rows: vec![ $( $crate::catalog::Row {
                seq: $seq,
                recipe: $crate::Recipe {
                    text: $text,
                    weights: &[$($w),+],
                    kv: $kv,
                    tp: $tp,
                },
                deploy: $crate::catalog::Deploy {
                    weights: vec![$($w),+],
                    kv: $kv,
                    tp: $tp,
                    parts: vec![$($crate::catalog::Part::$rpart),*],
                    drafter: $rdrafter,
                },
            } ),* ],
        }
    }};
}
