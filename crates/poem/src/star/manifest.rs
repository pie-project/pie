//! What a package states of itself in `package.star`: the models it holds and
//! the deployments of them it lists, each named by the grammar every
//! deployment is named by.

use std::fmt;

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{FrozenModule, GlobalsBuilder, Module};
use starlark::starlark_simple_value;
use starlark::values::list::{ListRef, UnpackList};
use starlark::values::none::NoneOr;
use starlark::values::{
    NoSerialize, StarlarkPagableUnsupported, StarlarkValue, UnpackValue, Value,
};
use starlark_derive::{starlark_module, starlark_value};

use crate::Dtype;
use crate::star::run::Deploy;
use crate::star::values::{DtypeValue, word};

/// One model a package holds.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    ProvidesStaticType,
    NoSerialize,
    StarlarkPagableUnsupported,
    Allocative,
)]
pub struct Model {
    pub id: String,
    /// A miniature: a published model's widths at fewer layers or experts,
    /// which reads a prefix of its whole model's checkpoint.
    pub mini: bool,
    /// The parts its checkpoint may carry besides the text trunk, in the
    /// order a deployment's name spells them.
    pub parts: Vec<String>,
    /// The drafters it pairs with.
    pub drafters: Vec<String>,
    /// The chat template its turns are written in, by name.
    pub template: String,
    /// The tokenizer its checkpoint carries, by name.
    pub tokenizer: String,
    /// The architecture its media front-ends are chosen by.
    pub arch: String,
    /// Its depth and vocabulary, as the runtime states them to an inferlet.
    pub layers: u32,
    pub vocab: u32,
}

starlark_simple_value!(Model);

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model({:?})", self.id)
    }
}

#[starlark_value(type = "model")]
impl<'v> StarlarkValue<'v> for Model {}

/// One deployment a package lists.
#[derive(Clone, Debug, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct Listed {
    pub model: String,
    #[allocative(skip)]
    pub deploy: Deploy,
}

starlark_simple_value!(Listed);

impl fmt::Display for Listed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "deployment({:?})", self.model)
    }
}

#[starlark_value(type = "deployment")]
impl<'v> StarlarkValue<'v> for Listed {}

/// A drafter published apart from the model it drafts for: its `head`
/// repository drafts as `drafter` for the `target` repository, served as the
/// listed `deployment`.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    ProvidesStaticType,
    NoSerialize,
    StarlarkPagableUnsupported,
    Allocative,
)]
pub struct Published {
    pub target: String,
    pub head: String,
    pub drafter: String,
    pub deployment: String,
}

starlark_simple_value!(Published);

impl fmt::Display for Published {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "published({:?}, {:?})", self.target, self.head)
    }
}

#[starlark_value(type = "published")]
impl<'v> StarlarkValue<'v> for Published {}

/// What a package states of itself: its models, the deployments it lists, in
/// the order an import tries them, and the drafters published for them.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    pub models: Vec<Model>,
    pub deployments: Vec<(String, Deploy)>,
    pub published: Vec<Published>,
}

impl Manifest {
    /// The manifest `package.star` states: its `MODELS` and `DEPLOYMENTS`.
    pub(crate) fn of(package: &str, frozen: &FrozenModule) -> anyhow::Result<Manifest> {
        Module::with_temp_heap(|module| {
            let heap = module.heap();
            let list = |name: &str| -> anyhow::Result<Vec<Value<'_>>> {
                let value = frozen
                    .get(name)
                    .map_err(|_| anyhow::anyhow!("`{package}/package.star` states no `{name}`"))?
                    .add_to_heap(heap);
                let items = ListRef::from_value(value).ok_or_else(|| {
                    anyhow::anyhow!("`{package}/package.star`: `{name}` is no list")
                })?;
                Ok(items.iter().collect())
            };
            let mut manifest = Manifest::default();
            for value in list("MODELS")? {
                let model = value.downcast_ref::<Model>().ok_or_else(|| {
                    anyhow::anyhow!(
                        "`{package}/package.star`: `MODELS` holds {}, not a model",
                        value.get_type()
                    )
                })?;
                if manifest.model(&model.id).is_some() {
                    anyhow::bail!("`{package}/package.star` states `{}` twice", model.id);
                }
                manifest.models.push(model.clone());
            }
            for value in list("DEPLOYMENTS")? {
                let listed = value.downcast_ref::<Listed>().ok_or_else(|| {
                    anyhow::anyhow!(
                        "`{package}/package.star`: `DEPLOYMENTS` holds {}, not a deployment",
                        value.get_type()
                    )
                })?;
                let model = manifest.model(&listed.model).ok_or_else(|| {
                    anyhow::anyhow!(
                        "`{package}/package.star` lists a deployment of `{}`, which it states \
                         no model of",
                        listed.model
                    )
                })?;
                model.admits(&listed.deploy)?;
                manifest
                    .deployments
                    .push((listed.model.clone(), listed.deploy.clone()));
            }
            if frozen.get("PUBLISHED").is_ok() {
                for value in list("PUBLISHED")? {
                    let published = value.downcast_ref::<Published>().ok_or_else(|| {
                        anyhow::anyhow!(
                            "`{package}/package.star`: `PUBLISHED` holds {}, not a published \
                             drafter",
                            value.get_type()
                        )
                    })?;
                    let listed = manifest.deployments.iter().any(|(id, d)| {
                        manifest
                            .model(id)
                            .is_some_and(|m| m.name(d) == published.deployment)
                    });
                    if !listed {
                        anyhow::bail!(
                            "`{package}/package.star` publishes a drafter for `{}`, which it \
                             lists no deployment of",
                            published.deployment
                        );
                    }
                    manifest.published.push(published.clone());
                }
            }
            Ok(manifest)
        })
    }

    /// The model `id`.
    #[must_use]
    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// The model and deployment `name` spells, if it names one of this
    /// package's models.
    #[must_use]
    pub fn parse(&self, name: &str) -> Option<(&Model, Deploy)> {
        let mut models: Vec<&Model> = self.models.iter().collect();
        models.sort_by_key(|m| std::cmp::Reverse(m.id.len()));
        models.into_iter().find_map(|model| {
            let rest = name.strip_prefix(model.id.as_str())?.strip_prefix('-')?;
            model.parse(rest).map(|deploy| (model, deploy))
        })
    }
}

impl Model {
    /// Whether the model has the parts and the drafter `deploy` serves.
    pub fn admits(&self, deploy: &Deploy) -> anyhow::Result<()> {
        if let Some(part) = deploy.parts.iter().find(|p| !self.parts.contains(p)) {
            anyhow::bail!("`{}` carries no {part} part", self.id);
        }
        if let Some(drafter) = &deploy.drafter
            && !self.drafters.contains(drafter)
        {
            anyhow::bail!("`{}` pairs with no {drafter} drafter", self.id);
        }
        if deploy.tp == 0 {
            anyhow::bail!("a deployment runs on at least one rank");
        }
        if deploy.weights.is_empty() {
            anyhow::bail!("a deployment of `{}` states no weight precision", self.id);
        }
        Ok(())
    }

    /// The name a deployment of this model is served under:
    /// `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-tp{n}]`, its parts in
    /// the order the model states them.
    #[must_use]
    pub fn name(&self, deploy: &Deploy) -> String {
        let mut name = self.id.clone();
        for part in self.parts.iter().filter(|p| deploy.parts.contains(p)) {
            name.push('-');
            name.push_str(part);
        }
        if let Some(drafter) = &deploy.drafter {
            name.push('-');
            name.push_str(drafter);
        }
        for dtype in &deploy.weights {
            name.push('-');
            name.push_str(&word(*dtype));
        }
        name.push_str("-kv-");
        name.push_str(&word(deploy.kv));
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
        while let Some(part) = words.peek().filter(|w| self.parts.iter().any(|p| p == *w)) {
            parts.push((*part).to_string());
            words.next();
        }
        let drafter = words
            .peek()
            .filter(|w| self.drafters.iter().any(|d| d == *w))
            .map(|w| (*w).to_string());
        if drafter.is_some() {
            words.next();
        }
        let mut weights = Vec::new();
        loop {
            let word = words.next()?;
            if word == "kv" {
                break;
            }
            weights.push(dtype_of(word)?);
        }
        let kv = dtype_of(words.next()?)?;
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

fn dtype_of(spelled: &str) -> Option<Dtype> {
    Dtype::ALL.iter().copied().find(|d| word(*d) == spelled)
}

#[starlark_module]
pub(crate) fn manifest(builder: &mut GlobalsBuilder) {
    /// The model `id`: whether it is a miniature, the parts and drafters it
    /// may serve with, and the template and tokenizer it is spoken through.
    fn model(
        #[starlark(require = pos)] id: String,
        #[starlark(require = named)] template: String,
        #[starlark(require = named)] tokenizer: String,
        #[starlark(require = named, default = false)] mini: bool,
        #[starlark(require = named, default = UnpackList::default())] parts: UnpackList<String>,
        #[starlark(require = named, default = UnpackList::default())] drafters: UnpackList<String>,
        #[starlark(require = named, default = String::new())] arch: String,
        #[starlark(require = named, default = 0)] layers: u32,
        #[starlark(require = named, default = 0)] vocab: u32,
    ) -> anyhow::Result<Model> {
        Ok(Model {
            id,
            mini,
            parts: parts.items,
            drafters: drafters.items,
            template,
            tokenizer,
            arch,
            layers,
            vocab,
        })
    }

    /// A drafter published apart from its model: `head` drafts as `drafter`
    /// for `target`, served as the listed `deployment`.
    fn published(
        #[starlark(require = named)] target: String,
        #[starlark(require = named)] head: String,
        #[starlark(require = named)] drafter: String,
        #[starlark(require = named)] deployment: String,
    ) -> anyhow::Result<Published> {
        Ok(Published {
            target,
            head,
            drafter,
            deployment,
        })
    }

    /// A deployment of the model `model`: its weights at `weights` (one
    /// dtype, or a mixed recipe's banks in order), its kv at `kv`, with the
    /// parts and the drafter it serves.
    fn deployment<'v>(
        #[starlark(require = pos)] model: String,
        #[starlark(require = named)] weights: Value<'v>,
        #[starlark(require = named)] kv: &DtypeValue,
        #[starlark(require = named, default = UnpackList::default())] parts: UnpackList<String>,
        #[starlark(require = named, default = NoneOr::None)] drafter: NoneOr<String>,
    ) -> anyhow::Result<Listed> {
        let weights = match weights.downcast_ref::<DtypeValue>() {
            Some(one) => vec![one.0],
            None => UnpackList::<Value<'v>>::unpack_value_err(weights)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .items
                .into_iter()
                .map(|w| {
                    w.downcast_ref::<DtypeValue>()
                        .map(|d| d.0)
                        .ok_or_else(|| anyhow::anyhow!("weights are dtypes, not {}", w.get_type()))
                })
                .collect::<anyhow::Result<_>>()?,
        };
        Ok(Listed {
            model,
            deploy: Deploy {
                weights,
                kv: kv.0,
                tp: 1,
                parts: parts.items,
                drafter: drafter.into_option(),
            },
        })
    }
}
