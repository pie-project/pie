//! The catalog: each model once, at its whole size, with the parts its
//! checkpoint can carry and the drafters it pairs with; and a deployment,
//! which picks the precision, the kv dtype, the rank count, the parts it
//! serves and the drafter, and is checked against the model.

use std::sync::Arc;

use checkpoint::contract::ModelContract;
use poem_dsl::{Dtype, Platform, Trace};

use crate::{Diffusion, Generative, template, tokenizer};

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
    /// A miniature: a published model's widths at fewer layers or experts,
    /// cut from its checkpoint, which kernel bring-up, parity and tuning
    /// serve at production shapes on one device. It serves like any model;
    /// identification tries it after every whole one, since it reads a
    /// prefix of its whole model's planes.
    pub mini: bool,
    pub parts: &'static [Part],
    pub drafters: &'static [Drafter],
    pub trace: TraceFn,
    pub import: ImportFn,
    pub template: TemplateFn,
    pub tokenizer: &'static tokenizer::Contract,
    pub diffusion: fn(&Deploy) -> Option<Diffusion>,
    pub generative: fn(&Deploy) -> Option<Generative>,
    /// The deployments the catalog lists for this model, with each one's
    /// place in its family's list (identification tries them in that order).
    pub rows: Vec<Row>,
}

pub struct Row {
    pub seq: u32,
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

    /// The name a deployment of this model is served under:
    /// `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-tp{n}]`.
    #[must_use]
    pub fn name(&self, deploy: &Deploy) -> String {
        let mut name = self.id.to_string();
        let mut parts = deploy.parts.clone();
        parts.sort_unstable();
        for part in parts {
            name.push('-');
            name.push_str(part.word());
        }
        if let Some(drafter) = deploy.drafter {
            name.push('-');
            name.push_str(drafter.word());
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

    /// The deployment `rest` names, `rest` being a name with this model's id
    /// and its dash taken off the front.
    fn parse(&self, rest: &str) -> Option<Deploy> {
        let mut words = rest.split('-').peekable();
        let mut parts = Vec::new();
        while let Some(part) = words.peek().and_then(|w| Part::of(w)) {
            parts.push(part);
            words.next();
        }
        let drafter = words.peek().and_then(|w| Drafter::of(w));
        if drafter.is_some() {
            words.next();
        }
        let mut weights = Vec::new();
        loop {
            let word = words.next()?;
            if word == "kv" {
                break;
            }
            weights.push(crate::dtype_of(word)?);
        }
        let kv = crate::dtype_of(words.next()?)?;
        let tp = match words.next() {
            None => 1,
            Some(word) => word.strip_prefix("tp")?.parse().ok().filter(|tp| *tp > 1)?,
        };
        if words.next().is_some() || weights.is_empty() {
            return None;
        }
        Some(Deploy {
            weights,
            kv,
            tp,
            parts,
            drafter,
        })
    }
}

impl Part {
    const ALL: [Part; 2] = [Part::Vision, Part::SelfCond];

    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Part::Vision => "vision",
            Part::SelfCond => "selfcond",
        }
    }

    #[must_use]
    pub fn of(word: &str) -> Option<Part> {
        Part::ALL.into_iter().find(|part| part.word() == word)
    }
}

impl Drafter {
    const ALL: [Drafter; 5] = [
        Drafter::Mtp,
        Drafter::DFlash,
        Drafter::DFlash2,
        Drafter::DSpark,
        Drafter::Eagle,
    ];

    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Drafter::Mtp => "mtp",
            Drafter::DFlash => "dflash",
            Drafter::DFlash2 => "dflash2",
            Drafter::DSpark => "dspark",
            Drafter::Eagle => "eagle",
        }
    }

    #[must_use]
    pub fn of(word: &str) -> Option<Drafter> {
        Drafter::ALL
            .into_iter()
            .find(|drafter| drafter.word() == word)
    }
}

/// The model and deployment `name` spells, if any model of the catalog is
/// the one it names.
#[must_use]
pub fn parse(name: &str) -> Option<(&'static Entry, Deploy)> {
    let mut entries: Vec<&'static Entry> = crate::entries().collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.id.len()));
    entries.into_iter().find_map(|entry| {
        let rest = name.strip_prefix(entry.id)?.strip_prefix('-')?;
        entry.parse(rest).map(|deploy| (entry, deploy))
    })
}

/// A catalog entry: its id, parts, drafters and family hooks, the function
/// that builds its model for a deployment, and the rows it was named by.
#[macro_export]
macro_rules! entry {
    (
        id: $id:literal,
        mini: $mini:literal,
        parts: [$($part:ident),* $(,)?],
        drafters: [$($drafter:ident),* $(,)?],
        template: $template:expr,
        tokenizer: $tokenizer:expr,
        diffusion: $diffusion:expr,
        generative: $generative:expr,
        build: |$d:ident| -> $model:ty $body:block,
        rows: [ $( ($seq:literal, [$($w:expr),+ $(,)?], $kv:expr, [$($rpart:ident),* $(,)?], $rdrafter:expr $(,)?) ),* $(,)? ] $(,)?
    ) => {{
        fn build($d: &$crate::catalog::Deploy) -> Result<$model, $crate::catalog::Refused> $body
        $crate::catalog::Entry {
            id: $id,
            mini: $mini,
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
                deploy: $crate::catalog::Deploy {
                    weights: vec![$($w),+],
                    kv: $kv,
                    tp: 1,
                    parts: vec![$($crate::catalog::Part::$rpart),*],
                    drafter: $rdrafter,
                },
            } ),* ],
        }
    }};
}
