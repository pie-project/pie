use std::collections::BTreeMap;

use crate::capability::Capability;
use crate::preproc::{Malformed, Variant};

include!(concat!(env!("OUT_DIR"), "/sources.rs"));
include!(concat!(env!("OUT_DIR"), "/census.rs"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expanded {
    pub wgsl: String,

    pub file: &'static str,

    pub workgroup: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missing {
    NoVariant {
        entrypoint: String,
        tier: Capability,
    },

    Unexpandable {
        file: &'static str,
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
pub fn text(path: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find(|(name, _)| *name == path)
        .map(|(_, body)| *body)
}

#[must_use]
pub fn source(entrypoint: &str) -> Option<Expanded> {
    match at(entrypoint, Capability::Baseline) {
        Ok(expanded) => Some(expanded),
        Err(Missing::NoVariant { .. }) => None,
        Err(why) => panic!("`{entrypoint}` was expanded at build time: {why}"),
    }
}

pub fn at(entrypoint: &str, tier: Capability) -> Result<Expanded, Missing> {
    let nothing = || Missing::NoVariant {
        entrypoint: entrypoint.to_owned(),
        tier,
    };

    let (file, variant) = declared()
        .into_iter()
        .find(|(_, v)| v.entrypoint == entrypoint && v.tier == tier)
        .ok_or_else(nothing)?;

    let mut defines: BTreeMap<String, String> = tier
        .defines()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    defines.extend(variant.defines);

    let wgsl = crate::preproc::expand(
        text(file).expect("the path came from SOURCES"),
        &defines,
        &{ |path: &str| text(path).map(ToOwned::to_owned) },
    )
    .map_err(|why| Missing::Unexpandable { file, why })?;

    let workgroup = workgroup_of(&wgsl, &defines);
    Ok(Expanded {
        wgsl,
        file,
        workgroup,
    })
}

#[must_use]
pub fn declared() -> Vec<(&'static str, Variant)> {
    let mut out = Vec::new();
    for (path, body) in SOURCES {
        let variants = crate::preproc::instantiations(body)
            .unwrap_or_else(|why| panic!("`{path}` was parsed at build time: {why}"));
        out.extend(variants.into_iter().map(|v| (*path, v)));
    }
    out
}

#[must_use]
pub fn census() -> &'static [&'static str] {
    CENSUS
}

fn workgroup_of(wgsl: &str, defines: &BTreeMap<String, String>) -> [u32; 3] {
    let mut group = [1u32; 3];
    let Some(head) = wgsl.split_once("@workgroup_size(") else {
        return group;
    };
    let Some((args, _)) = head.1.split_once(')') else {
        return group;
    };
    for (axis, word) in args.split(',').take(3).enumerate() {
        let word = word.trim();
        let word = defines.get(word).map_or(word, String::as_str).trim();
        let word = word.trim_end_matches(['u', 'i']);
        if let Ok(v) = word.parse::<u32>() {
            group[axis] = v;
        }
    }
    group
}
