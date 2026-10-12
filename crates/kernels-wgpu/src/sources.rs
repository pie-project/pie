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

/// Bytes of `var<workgroup>` an expanded shader declares, array lengths
/// given as literals or module `const`s.
#[must_use]
pub fn workgroup_bytes(wgsl: &str) -> u64 {
    let consts: std::collections::BTreeMap<&str, u64> = wgsl
        .lines()
        .filter_map(|l| {
            let rest = l.trim().strip_prefix("const ")?;
            let (name, value) = rest.split_once('=')?;
            let value = value
                .trim()
                .trim_end_matches(';')
                .trim()
                .trim_end_matches('u');
            Some((name.trim().split(':').next()?.trim(), value.parse().ok()?))
        })
        .collect();
    fn size(ty: &str, consts: &std::collections::BTreeMap<&str, u64>) -> u64 {
        let ty = ty.trim();
        if let Some(inner) = ty.strip_prefix("array<").and_then(|t| t.strip_suffix('>')) {
            let Some((elem, len)) = inner.rsplit_once(',') else {
                return 0;
            };
            let len = len.trim().trim_end_matches('u');
            let n = len
                .parse()
                .ok()
                .or_else(|| consts.get(len).copied())
                .unwrap_or(0);
            return n * size(elem, consts);
        }
        let lanes = if ty.starts_with("vec2") {
            2
        } else if ty.starts_with("vec3") || ty.starts_with("vec4") {
            4
        } else {
            1
        };
        let scalar = if ty.contains("f16") { 2 } else { 4 };
        lanes * scalar
    }
    wgsl.lines()
        .filter_map(|l| l.trim().strip_prefix("var<workgroup>"))
        .filter_map(|rest| rest.split_once(':'))
        .map(|(_, ty)| size(ty.trim().trim_end_matches(';'), &consts))
        .sum()
}

#[cfg(test)]
mod workgroup_tests {
    #[test]
    fn workgroup_bytes_count_arrays_consts_and_scalars() {
        let wgsl = "const PIE_XS_WORDS = 4096u;\nvar<workgroup> xs_shared: array<u32, PIE_XS_WORDS>;\nvar<workgroup> xs_staged: u32;\nvar<workgroup> p: array<array<f32, 32>, 16>;\n";
        assert_eq!(super::workgroup_bytes(wgsl), 4096 * 4 + 4 + 16 * 32 * 4);
    }
}
