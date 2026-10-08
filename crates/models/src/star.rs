//! The models written as Starlark packages: every package under the
//! repository's `models/`, embedded, and the catalog entries its
//! `package.star` states.

use std::sync::LazyLock;

use crate::catalog::{Deploy, Drafter, Entry, Part, Refused, Row};

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
fn family(package: &'static poem::star::Package) -> Vec<Entry> {
    let manifest = package.manifest();
    let fail = |why: String| -> ! { panic!("the package `{}`: {why}", package.name()) };
    manifest
        .models
        .iter()
        .map(|model| {
            let id = model.id.as_str();
            let rows = manifest
                .deployments
                .iter()
                .enumerate()
                .filter(|(_, (of, _))| of == id)
                .map(|(seq, (_, d))| Row {
                    seq: seq as u32,
                    deploy: catalog(d).unwrap_or_else(|why| fail(why.0)),
                })
                .collect();
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
                tokenizer: crate::tokenizer::named(&model.tokenizer).unwrap_or_else(|| {
                    fail(format!("`{id}` names no tokenizer `{}`", model.tokenizer))
                }),
                diffusion: Box::new(|_| None),
                generative: Box::new(|_| None),
                rows,
            }
        })
        .collect()
}

/// The catalog entries of every embedded package, family by family.
pub fn families() -> Vec<Vec<Entry>> {
    packages().iter().map(family).collect()
}
