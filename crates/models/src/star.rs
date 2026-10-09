//! The models written as Starlark packages: every package under the
//! repository's `models/`, embedded, and the catalog entries its
//! `package.poem` states.

use std::sync::LazyLock;

use crate::catalog::{Deploy, Drafter, Entry, Part, Refused};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/packages.rs"));
}

static PACKAGES: LazyLock<Vec<poem::star::Package>> = LazyLock::new(|| {
    embedded::PACKAGES
        .iter()
        .map(|(name, files)| {
            poem::star::Package::new(name, files).unwrap_or_else(|why| {
                panic!("the embedded package `{name}` does not load: {why:#}")
            })
        })
        .collect()
});

/// Every package this build embeds.
pub fn packages() -> &'static [poem::star::Package] {
    &PACKAGES
}

/// The package holding the model `id`, if a package holds it.
#[must_use]
pub fn package_of(id: &str) -> Option<&'static poem::star::Package> {
    packages()
        .iter()
        .find(|package| package.manifest().model(id).is_some())
}

/// A catalog deployment as a package reads it.
#[must_use]
pub fn deploy(d: &Deploy) -> poem::star::Deploy {
    poem::star::Deploy {
        weights: d.weights.clone(),
        kv: d.kv,
        tp: d.tp,
        parts: d.parts.iter().map(|p| p.word().to_string()).collect(),
        drafter: d.drafter.map(|d| d.word().to_string()),
    }
}

/// A package's deployment as the catalog states it, if the catalog names
/// its parts and drafter.
pub fn catalog(d: &poem::star::Deploy) -> Result<Deploy, Refused> {
    Ok(Deploy {
        weights: d.weights.clone(),
        kv: d.kv,
        tp: d.tp,
        parts: d.parts.iter().map(|p| part(p)).collect::<Result<_, _>>()?,
        drafter: d.drafter.as_deref().map(drafter).transpose()?,
    })
}

fn part(word: &str) -> Result<Part, Refused> {
    Part::of(word).ok_or_else(|| Refused(format!("no part is called `{word}`")))
}

fn drafter(word: &str) -> Result<Drafter, Refused> {
    Drafter::of(word).ok_or_else(|| Refused(format!("no drafter is called `{word}`")))
}

/// The catalog entries of `package`, one family: each model it states,
/// with the deployments it lists of it in the package's order.
pub fn family(package: &'static poem::star::Package) -> Vec<Entry> {
    let manifest = package.manifest();
    let fail = |why: String| -> ! { panic!("the package `{}`: {why}", package.name()) };
    manifest
        .models
        .iter()
        .map(|model| {
            let id = model.id.as_str();
            Entry {
                id,
                mini: model.mini,
                parts: model
                    .parts
                    .iter()
                    .map(|p| part(p).unwrap_or_else(|why| fail(why.0)))
                    .collect(),
                drafters: model
                    .drafters
                    .iter()
                    .map(|d| drafter(d).unwrap_or_else(|why| fail(why.0)))
                    .collect(),
                trace: Box::new(move |name, d, platform| {
                    package
                        .trace(id, &deploy(d), name, platform)
                        .map_err(|why| Refused(format!("{why:#}")))
                }),
                import: Box::new(move |d, src, platform| {
                    crate::whole(d.tp)?;
                    package.import(id, &deploy(d), src, platform)
                }),
                template: crate::template::named(&model.template).unwrap_or_else(|| {
                    fail(format!(
                        "`{id}` is spoken through no template `{}`",
                        model.template
                    ))
                }),
                tokenizer: {
                    for vision in [false, true] {
                        if crate::tokenizer::named(&model.tokenizer, vision).is_none() {
                            fail(format!("`{id}` names no tokenizer `{}`", model.tokenizer));
                        }
                    }
                    model.tokenizer.as_str()
                },
                arch: model.arch.as_str(),
                layers: model.layers,
                vocab: model.vocab,
                diffusion: Box::new(move |d| {
                    package
                        .diffusion(id, &deploy(d))
                        .unwrap_or_else(|why| panic!("`{id}` states no canvas: {why:#}"))
                }),
                generative: Box::new(move |d| {
                    package
                        .generative(id, &deploy(d))
                        .unwrap_or_else(|why| panic!("`{id}` states no generative facts: {why:#}"))
                }),
            }
        })
        .collect()
}

/// The catalog entries of every embedded package, family by family.
pub fn families() -> Vec<Vec<Entry>> {
    packages().iter().map(family).collect()
}

/// The package holding the model `id` with `source` appended to its `file`,
/// stating `function` in place of the one the file states, which stays as
/// `whole_<function>`: a part of a model traced or read alone, by the
/// package's own functions.
pub fn replacing(
    id: &str,
    file: &str,
    function: &str,
    source: &str,
) -> Result<poem::star::Package, String> {
    let package = package_of(id).ok_or_else(|| format!("no package holds `{id}`"))?;
    let prefix = format!("{}files/", poem::star::ATTRIBUTE);
    let stated = format!("def {function}(");
    let files: Vec<(String, String)> = package
        .attributes()
        .into_iter()
        .filter_map(|(key, text)| {
            let name = key.strip_prefix(&prefix)?.to_string();
            let text = if name == file {
                text.replace(&stated, &format!("def whole_{function}(")) + "\n" + source
            } else {
                text
            };
            Some((name, text))
        })
        .collect();
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(f, s)| (f.as_str(), s.as_str()))
        .collect();
    poem::star::Package::new(package.name(), &files).map_err(|why| format!("{why:#}"))
}

/// A deployment at weights `w` and kv `kv` on one rank, with no part or
/// drafter.
#[must_use]
pub fn one_rank(w: poem::Dtype, kv: poem::Dtype) -> poem::star::Deploy {
    poem::star::Deploy {
        weights: vec![w],
        kv,
        tp: 1,
        parts: Vec::new(),
        drafter: None,
    }
}

/// The trace of the model `id`, listed or not, at weights `w` and kv `kv`
/// on one rank, named `id`: the small geometries the engines' tests serve.
#[must_use]
pub fn trace_of(
    id: &str,
    w: poem::Dtype,
    kv: poem::Dtype,
    platform: poem::Platform,
) -> poem::Trace {
    let package = package_of(id).unwrap_or_else(|| panic!("no package holds `{id}`"));
    let deploy = one_rank(w, kv);
    package
        .trace(id, &deploy, id, platform)
        .unwrap_or_else(|why| panic!("`{id}` does not trace: {why:#}"))
}

/// The contract reading `src` into the model `id`, listed or not, at weights
/// `w` and kv `kv` on one rank.
pub fn import_of(
    id: &str,
    w: poem::Dtype,
    kv: poem::Dtype,
    src: &ztensor::Source,
    platform: poem::Platform,
) -> Result<checkpoint::contract::ModelContract, poem::import::Error> {
    let package = package_of(id).unwrap_or_else(|| panic!("no package holds `{id}`"));
    let deploy = one_rank(w, kv);
    package.import(id, &deploy, src, platform)
}
