use std::collections::BTreeMap;

use crate::Capability;
use crate::preproc::{Malformed, Variant};

include!(concat!(env!("OUT_DIR"), "/sources.rs"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missing {
    NoVariant {
        entrypoint: String,
        tier: Capability,
    },
    Unexpandable {
        file: String,
        why: Malformed,
    },
}

impl std::fmt::Display for Missing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoVariant { entrypoint, tier } => {
                write!(f, "no `{}` variant of `{entrypoint}`", tier.tag())
            }
            Self::Unexpandable { file, why } => write!(f, "{file}: {why}"),
        }
    }
}

impl std::error::Error for Missing {}

#[must_use]
pub fn source(path: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find(|(name, _)| *name == path)
        .map(|(_, text)| *text)
}

#[must_use]
pub fn declared() -> Vec<(&'static str, Variant)> {
    let mut out = Vec::new();
    for (path, text) in SOURCES {
        let variants = crate::preproc::instantiations(text)
            .unwrap_or_else(|why| panic!("`{path}` was parsed at build time: {why}"));
        out.extend(variants.into_iter().map(|v| (*path, v)));
    }
    out
}

pub fn at(file: &str, entrypoint: &str, tier: Capability) -> Result<String, Missing> {
    let text = source(file).ok_or_else(|| Missing::NoVariant {
        entrypoint: entrypoint.to_owned(),
        tier,
    })?;
    let variant = crate::preproc::instantiations(text)
        .unwrap_or_else(|why| panic!("`{file}` was parsed at build time: {why}"))
        .into_iter()
        .find(|v| v.entrypoint == entrypoint && v.tier == tier)
        .ok_or_else(|| Missing::NoVariant {
            entrypoint: entrypoint.to_owned(),
            tier,
        })?;
    expand_variant(file, text, &variant, tier)
}

pub fn entrypoint_source(entrypoint: &str, tier: Capability) -> Result<String, Missing> {
    let found = declared()
        .into_iter()
        .find(|(_, v)| v.entrypoint == entrypoint && v.tier == tier);

    let Some((file, variant)) = found else {
        return Err(Missing::NoVariant {
            entrypoint: entrypoint.to_owned(),
            tier,
        });
    };

    let mut defines: BTreeMap<String, String> = tier
        .defines()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    defines.extend(variant.defines.clone());

    crate::preproc::expand(
        source(file).expect("the path came from SOURCES"),
        &defines,
        &|path| source(path).map(ToOwned::to_owned),
    )
    .map_err(|why| Missing::Unexpandable {
        file: file.to_owned(),
        why,
    })
}

fn expand_variant(
    file: &str,
    text: &'static str,
    variant: &Variant,
    tier: Capability,
) -> Result<String, Missing> {
    let mut defines: BTreeMap<String, String> = tier
        .defines()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    defines.extend(variant.defines.clone());

    crate::preproc::expand(text, &defines, &|path| {
        source(path).map(ToOwned::to_owned)
    })
    .map_err(|why| Missing::Unexpandable {
        file: file.to_owned(),
        why,
    })
}

