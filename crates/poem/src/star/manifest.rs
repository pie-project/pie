//! What a package states of itself in its `MODELS`, `DEPLOYMENTS` and
//! `PUBLISHED`: its models, the
//! deployments it lists, and each model's template and tokenizer, as data.

use std::collections::BTreeMap;
use std::fmt;

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{FrozenModule, GlobalsBuilder, Module};
use starlark::starlark_simple_value;
use starlark::values::ValueLike;
use starlark::values::dict::UnpackDictEntries;
use starlark::values::list::{ListRef, UnpackList};
use starlark::values::none::NoneOr;
use starlark::values::{NoSerialize, StarlarkValue, UnpackValue, Value};
use starlark_derive::{starlark_module, starlark_value};

use crate::star::run::Deploy;
use crate::star::values::{DtypeValue, precision, precision_word, word};
use crate::{Dtype, Precision};

/// A template: a format the runtime ships, with the settings it reads.
#[derive(Clone, Debug, Default, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
pub struct Template {
    pub format: String,
    pub thinking: bool,
    pub preserve_thinking: bool,
    pub reasoning_effort: bool,
    pub tools: bool,
    pub xml_tools: bool,
    pub generation_suffix: String,
    pub identity: String,
    pub stop: Vec<String>,
    pub bos: Option<String>,
    pub eos: Option<String>,
}

starlark_simple_value!(Template);

impl fmt::Display for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "template({:?})", self.format)
    }
}

#[starlark_value(type = "template")]
impl<'v> StarlarkValue<'v> for Template {}

/// What a model asks of its tokenizer: markers the vocabulary holds, markers
/// pinned at an id, and the markers each part adds when served.
#[derive(Clone, Debug, Default, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
pub struct Tokenizer {
    pub markers: Vec<Vec<String>>,
    pub pinned: Vec<(String, u32)>,
    pub parts: BTreeMap<String, Vec<Vec<String>>>,
}

starlark_simple_value!(Tokenizer);

impl fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer({} marker groups)", self.markers.len())
    }
}

#[starlark_value(type = "tokenizer")]
impl<'v> StarlarkValue<'v> for Tokenizer {}

impl Tokenizer {
    #[must_use]
    pub fn markers_for(&self, parts: &[String]) -> Vec<Vec<String>> {
        let mut markers = self.markers.clone();
        for part in parts {
            if let Some(groups) = self.parts.get(part) {
                markers.extend(groups.iter().cloned());
            }
        }
        markers
    }
}

/// One model a package holds.
#[derive(Clone, Debug, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
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
    /// Whether a deployment may tap it: end its forward at a named
    /// intermediate and read that out instead (`-tap.{key}`).
    pub taps: bool,
    pub template: Template,
    pub tokenizer: Tokenizer,
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
#[derive(Clone, Debug, ProvidesStaticType, NoSerialize, Allocative)]
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
#[derive(Clone, Debug, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
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
    /// The manifest a package states: its `MODELS` and `DEPLOYMENTS`.
    pub(crate) fn of(package: &str, frozen: &FrozenModule) -> anyhow::Result<Manifest> {
        Module::with_temp_heap(|module| {
            let heap = module.heap();
            let list = |name: &str| -> anyhow::Result<Vec<Value<'_>>> {
                let value = frozen
                    .get(name)
                    .map_err(|_| anyhow::anyhow!("the package `{package}` states no `{name}`"))?;
                let value = heap.access_owned_frozen_value(&value);
                let items = ListRef::from_value(value).ok_or_else(|| {
                    anyhow::anyhow!("the package `{package}`: `{name}` is no list")
                })?;
                Ok(items.iter().collect())
            };
            let mut manifest = Manifest::default();
            for value in list("MODELS")? {
                let model = value.downcast_ref::<Model>().ok_or_else(|| {
                    anyhow::anyhow!(
                        "the package `{package}`: `MODELS` holds {}, not a model",
                        value.get_type()
                    )
                })?;
                if manifest.model(&model.id).is_some() {
                    anyhow::bail!("the package `{package}` states `{}` twice", model.id);
                }
                manifest.models.push(model.clone());
            }
            for value in list("DEPLOYMENTS")? {
                let listed = value.downcast_ref::<Listed>().ok_or_else(|| {
                    anyhow::anyhow!(
                        "the package `{package}`: `DEPLOYMENTS` holds {}, not a deployment",
                        value.get_type()
                    )
                })?;
                let model = manifest.model(&listed.model).ok_or_else(|| {
                    anyhow::anyhow!(
                        "the package `{package}` lists a deployment of `{}`, which it states \
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
                            "the package `{package}`: `PUBLISHED` holds {}, not a published \
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
                            "the package `{package}` publishes a drafter for `{}`, which it \
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
        if let Some(tap) = &deploy.tap
            && !self.taps
        {
            anyhow::bail!(
                "`{}` reads out no intermediate; it takes no tap {tap}",
                self.id
            );
        }
        if deploy.tp == 0 || deploy.ep == 0 {
            anyhow::bail!("a deployment runs on at least one rank");
        }
        if deploy.tp > 1 && deploy.ep > 1 {
            anyhow::bail!("a deployment splits its ranks by tensor or by expert, not both yet");
        }
        if deploy.weights.is_empty() {
            anyhow::bail!("a deployment of `{}` states no weight precision", self.id);
        }
        Ok(())
    }

    /// The name a deployment of this model is served under:
    /// `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-yarn{f}][-tap.{key}][-tp{n}][-ep{n}]`, its parts in
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
        for stated in &deploy.weights {
            name.push('-');
            name.push_str(&precision_word(*stated));
        }
        name.push_str("-kv-");
        name.push_str(&word(deploy.kv));
        if let Some(factor) = deploy.yarn {
            name.push_str(&format!("-yarn{factor}"));
        }
        if let Some(tap) = &deploy.tap {
            name.push_str(&format!("-tap.{tap}"));
        }
        if deploy.tp > 1 {
            name.push_str(&format!("-tp{}", deploy.tp));
        }
        if deploy.ep > 1 {
            name.push_str(&format!("-ep{}", deploy.ep));
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
            weights.push(precision_of(word)?);
        }
        let kv = dtype_of(words.next()?)?;
        let yarn = match words.peek().and_then(|w| w.strip_prefix("yarn")) {
            Some(factor) => {
                let factor = factor.parse().ok().filter(|f| *f > 1)?;
                words.next();
                Some(factor)
            }
            None => None,
        };
        let tap = match words.peek().and_then(|w| w.strip_prefix("tap.")) {
            Some(key) => {
                let key = Some(key.to_string()).filter(|k| !k.is_empty())?;
                words.next();
                Some(key)
            }
            None => None,
        };
        let mut ranks = |prefix: &str| match words.peek().and_then(|w| w.strip_prefix(prefix)) {
            Some(n) => {
                let n = n.parse().ok().filter(|n| *n > 1)?;
                words.next();
                Some(n)
            }
            None => Some(1),
        };
        let tp = ranks("tp")?;
        let ep = ranks("ep")?;
        if words.next().is_some() || weights.is_empty() {
            return None;
        }
        Some(Deploy {
            weights,
            kv,
            tp,
            ep,
            parts,
            drafter,
            yarn,
            tap,
        })
    }
}

#[must_use]
pub fn dtype_of(spelled: &str) -> Option<Dtype> {
    Dtype::ALL.iter().copied().find(|d| word(*d) == spelled)
}

/// The precision `spelled` names: a dtype, or a dtype whose products quantize
/// their activations (`nvfp4a4`).
#[must_use]
pub fn precision_of(spelled: &str) -> Option<Precision> {
    dtype_of(spelled).map(Precision::of).or_else(|| {
        Precision::ACTED
            .into_iter()
            .find(|p| precision_word(*p) == spelled)
    })
}

#[starlark_module]
pub(crate) fn manifest(builder: &mut GlobalsBuilder) {
    /// The template `format` with the settings it reads: ChatML `thinking`,
    /// `preserve_thinking`, `reasoning_effort`, `tools`, `xml_tools`, `generation_suffix`, `stop`; `lines`
    /// `stop`, `bos`, `eos`; `raw` `stop`; `minimax2` `identity`.
    fn template(
        #[starlark(require = pos)] format: String,
        #[starlark(require = named, default = false)] thinking: bool,
        #[starlark(require = named, default = false)] preserve_thinking: bool,
        #[starlark(require = named, default = false)] reasoning_effort: bool,
        #[starlark(require = named, default = false)] tools: bool,
        #[starlark(require = named, default = false)] xml_tools: bool,
        #[starlark(require = named, default = String::new())] generation_suffix: String,
        #[starlark(require = named, default = String::new())] identity: String,
        #[starlark(require = named, default = UnpackList::default())] stop: UnpackList<String>,
        #[starlark(require = named, default = NoneOr::None)] bos: NoneOr<String>,
        #[starlark(require = named, default = NoneOr::None)] eos: NoneOr<String>,
    ) -> anyhow::Result<Template> {
        Ok(Template {
            format,
            thinking,
            preserve_thinking,
            reasoning_effort,
            tools,
            xml_tools,
            generation_suffix,
            identity,
            stop: stop.items,
            bos: bos.into_option(),
            eos: eos.into_option(),
        })
    }

    /// `markers` the vocabulary must hold, `pinned` at the id given, and the
    /// marker groups each of `parts` adds when served.
    fn tokenizer<'v>(
        #[starlark(require = named, default = UnpackList::default())] markers: UnpackList<
            UnpackList<String>,
        >,
        #[starlark(require = named, default = UnpackDictEntries::default())]
        pinned: UnpackDictEntries<String, u32>,
        #[starlark(require = named, default = UnpackDictEntries::default())]
        parts: UnpackDictEntries<String, UnpackList<UnpackList<String>>>,
    ) -> anyhow::Result<Tokenizer> {
        Ok(Tokenizer {
            markers: markers.items.into_iter().map(|g| g.items).collect(),
            pinned: pinned.entries,
            parts: parts
                .entries
                .into_iter()
                .map(|(part, groups)| (part, groups.items.into_iter().map(|g| g.items).collect()))
                .collect(),
        })
    }

    /// The model `id`: whether it is a miniature, the parts and drafters it
    /// may serve with, and the template and tokenizer it is spoken through.
    fn model(
        #[starlark(require = pos)] id: String,
        #[starlark(require = named)] template: &Template,
        #[starlark(require = named)] tokenizer: &Tokenizer,
        #[starlark(require = named, default = false)] mini: bool,
        #[starlark(require = named, default = UnpackList::default())] parts: UnpackList<String>,
        #[starlark(require = named, default = UnpackList::default())] drafters: UnpackList<String>,
        #[starlark(require = named, default = false)] taps: bool,
        #[starlark(require = named, default = String::new())] arch: String,
        #[starlark(require = named, default = 0)] layers: u32,
        #[starlark(require = named, default = 0)] vocab: u32,
    ) -> anyhow::Result<Model> {
        Ok(Model {
            id,
            mini,
            parts: parts.items,
            drafters: drafters.items,
            taps,
            template: template.clone(),
            tokenizer: tokenizer.clone(),
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
    /// dtype, or a mixed recipe's banks in order, each a dtype or a precision
    /// that quantizes its products' activations), its kv at `kv`, with the
    /// parts and the drafter it serves.
    fn deployment<'v>(
        #[starlark(require = pos)] model: String,
        #[starlark(require = named)] weights: Value<'v>,
        #[starlark(require = named)] kv: &DtypeValue,
        #[starlark(require = named, default = UnpackList::default())] parts: UnpackList<String>,
        #[starlark(require = named, default = NoneOr::None)] drafter: NoneOr<String>,
    ) -> anyhow::Result<Listed> {
        let weights = match precision(weights) {
            Ok(one) => vec![one],
            Err(_) => UnpackList::<Value<'v>>::unpack_value_err(weights)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .items
                .into_iter()
                .map(|w| {
                    precision(w)
                        .map_err(|_| anyhow::anyhow!("weights are dtypes, not {}", w.get_type()))
                })
                .collect::<anyhow::Result<_>>()?,
        };
        Ok(Listed {
            model,
            deploy: Deploy {
                weights,
                kv: kv.0,
                tp: 1,
                ep: 1,
                parts: parts.items,
                drafter: drafter.into_option(),
                yarn: None,
                tap: None,
            },
        })
    }
}
