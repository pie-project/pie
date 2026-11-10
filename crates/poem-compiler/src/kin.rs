//! What the catalog makes of a checkpoint none of its deployments reads: the
//! families whose formats recognize it, the model of each it is nearest and
//! where it departs from that model's numbers, and the package a family that
//! adopts writes for it. Nothing here names a model.

use std::sync::Arc;

use checkpoint::contract::ModelContract;
use poem::Platform;
use poem::import::format::{Key, Stated};
use poem::star::{Adoption, Package};

use crate::catalog::{Catalog, Deployment, Refused};

/// A package whose formats recognize a checkpoint: the format that did, and
/// the package's model the checkpoint is nearest.
pub struct Kin {
    pub package: Arc<Package>,
    pub format: String,
    pub nearest: Nearest,
    /// Whether the package says how to adopt a checkpoint (`adopt`).
    pub adopts: bool,
}

/// A model of a package, and each number it is built with that a checkpoint
/// states otherwise.
pub struct Nearest {
    pub id: String,
    pub agreeing: Vec<Agreement>,
    pub departing: Vec<Agreement>,
}

/// A configuration value a model is built with, and what a checkpoint states.
#[derive(Clone, Debug, PartialEq)]
pub struct Agreement {
    pub key: String,
    pub built: f64,
    pub at_least: bool,
    /// Whether the checkpoint must state it.
    pub required: bool,
    pub stated: Option<f64>,
}

impl Agreement {
    fn of(src: &ztensor::Source, stated: &Stated) -> Option<Agreement> {
        let Key::Config(path) = &stated.key else {
            return None;
        };
        Some(Agreement {
            key: path.clone(),
            built: stated.want,
            at_least: stated.at_least,
            required: stated.required,
            stated: poem::import::format::stated(src, path),
        })
    }

    /// Whether the checkpoint states another value than the model is built with.
    #[must_use]
    pub fn departs(&self) -> bool {
        match self.stated {
            None => self.required,
            Some(found) if self.at_least => found < self.built,
            Some(found) => {
                found != self.built
                    && (found - self.built).abs() > 1e-6 * self.built.abs().max(found.abs())
            }
        }
    }
}

/// A checkpoint as an adoption reads it: its tensors, its `config.json`, the
/// special tokens its tokenizer names, and where it came from.
pub struct Checkpoint<'a> {
    pub src: &'a ztensor::Source,
    pub config: &'a serde_json::Value,
    pub special: &'a serde_json::Value,
    pub origin: &'a str,
}

/// A package a family wrote for a checkpoint, and each of its deployments
/// with whether it reads the checkpoint.
pub struct Adopted {
    pub adoption: Adoption,
    pub deployments: Vec<(Deployment, Result<ModelContract, String>)>,
}

impl Catalog {
    /// The packages whose formats recognize `src`, nearest first: each with
    /// its model whose numbers `src` departs from least.
    #[must_use]
    pub fn kin(&self, src: &ztensor::Source, platform: Platform) -> Vec<Kin> {
        let mut kin: Vec<Kin> = Vec::new();
        for package in self.packages() {
            let manifest = package.manifest();
            let mut best: Option<(String, Nearest)> = None;
            for model in manifest.models.iter().filter(|m| !m.mini) {
                let Some((_, deploy)) = manifest
                    .deployments
                    .iter()
                    .find(|(id, d)| *id == model.id && d.parts.is_empty() && d.drafter.is_none())
                else {
                    continue;
                };
                let Ok(recognized) = package.recognizing(&model.id, deploy, src, platform) else {
                    continue;
                };
                for format in recognized {
                    let (departing, agreeing): (Vec<Agreement>, Vec<Agreement>) = format
                        .states
                        .iter()
                        .filter_map(|s| Agreement::of(src, s))
                        .partition(Agreement::departs);
                    let nearer = best.as_ref().is_none_or(|(_, b)| {
                        (departing.len(), std::cmp::Reverse(agreeing.len()))
                            < (b.departing.len(), std::cmp::Reverse(b.agreeing.len()))
                    });
                    if nearer {
                        best = Some((
                            format.format,
                            Nearest {
                                id: model.id.clone(),
                                agreeing,
                                departing,
                            },
                        ));
                    }
                }
            }
            if let Some((format, nearest)) = best {
                kin.push(Kin {
                    package: Arc::clone(package),
                    format,
                    nearest,
                    adopts: package.adopts(),
                });
            }
        }
        // Nearest first: a family that adopts before one that lists its
        // models by hand, then the most numbers the checkpoint agrees with,
        // net of those it departs from; a format that states none says least.
        kin.sort_by_key(|k| {
            let (agree, depart) = (k.nearest.agreeing.len(), k.nearest.departing.len());
            (agree + depart == 0, !k.adopts, depart as i64 - agree as i64)
        });
        kin
    }

    /// `kin`'s package adopting `checkpoint` as the model `id` in the
    /// package `name`: what it wrote, and which of its deployments read it.
    pub fn adopt(
        &self,
        kin: &Kin,
        name: &str,
        id: &str,
        checkpoint: &Checkpoint<'_>,
        platform: Platform,
    ) -> Result<Adopted, Refused> {
        let src = checkpoint.src;
        if self.package(name).is_some() {
            return Err(Refused(format!(
                "the catalog already holds a package `{name}`; delete it or name another"
            )));
        }
        if self.model(id).is_some() {
            return Err(Refused(format!(
                "the catalog already holds a model `{id}`; name the adoption another"
            )));
        }
        let adoption = kin
            .package
            .adopt(
                name,
                id,
                checkpoint.config,
                checkpoint.special,
                checkpoint.origin,
            )
            .map_err(|why| Refused(format!("{why:#}")))?;
        let package = Arc::new(
            Package::new(
                name,
                &adoption.package.files().collect::<Vec<(&str, &str)>>(),
            )
            .map_err(|why| Refused(format!("the adopted package does not load: {why:#}")))?,
        );
        let manifest = package.manifest();
        let deployments = manifest
            .deployments
            .iter()
            .filter_map(|(model, deploy)| {
                let model = manifest.model(model)?;
                let deployment = Deployment::of(Arc::clone(&package), model, deploy.clone());
                let read = deployment
                    .contract(src, platform)
                    .map_err(|why| why.to_string());
                Some((deployment, read))
            })
            .collect();
        Ok(Adopted {
            adoption,
            deployments,
        })
    }
}
