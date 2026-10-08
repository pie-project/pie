//! The families written as Starlark packages: their files embedded, and a
//! deployment of one traced or read through the package.

use checkpoint::contract::ModelContract;
use poem_dsl::{Platform, Trace};

use crate::catalog::{Deploy, Refused};

/// A package embedded from `star/<name>/`.
macro_rules! package {
    ($name:literal) => {
        std::sync::LazyLock::new(|| {
            poem_star::Package::new(
                $name,
                &[
                    (
                        "model.star",
                        include_str!(concat!("../star/", $name, "/model.star")),
                    ),
                    (
                        "forward.star",
                        include_str!(concat!("../star/", $name, "/forward.star")),
                    ),
                    (
                        "formats.star",
                        include_str!(concat!("../star/", $name, "/formats.star")),
                    ),
                ],
            )
            .unwrap_or_else(|why| panic!("the embedded package `{}` does not load: {why:#}", $name))
        })
    };
}

pub static MUSE_GLIMMER: std::sync::LazyLock<poem_star::Package> = package!("muse_glimmer");

fn deploy(d: &Deploy) -> poem_star::Deploy {
    poem_star::Deploy {
        weights: d.weights.clone(),
        kv: d.kv,
        tp: d.tp,
        parts: d.parts.iter().map(|p| p.word().to_string()).collect(),
        drafter: d.drafter.map(|d| d.word().to_string()),
    }
}

/// The trace of `id`'s deployment `d` of `package`, named `name`.
pub fn trace(
    package: &poem_star::Package,
    id: &str,
    name: &str,
    d: &Deploy,
    platform: Platform,
) -> Result<Trace, Refused> {
    package
        .trace(id, &deploy(d), name, platform)
        .map_err(|why| Refused(format!("{why:#}")))
}

/// The contract reading `src` into `id`'s deployment `d` of `package`.
pub fn import(
    package: &poem_star::Package,
    id: &str,
    d: &Deploy,
    src: &ztensor::Source,
    platform: Platform,
) -> Result<ModelContract, checkpoint_dsl::Error> {
    crate::whole(d.tp)?;
    package.import(id, &deploy(d), src, platform)
}
