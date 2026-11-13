//! A serving artifact made from a source container, for tests of any
//! backend: what `pie model import` writes, without the model package's
//! own import or the hub.
//!
//! A deployment loads from a serving artifact, whose planes are stored as
//! they are served: a load converts nothing, so a quantized deployment
//! cannot be served from a float container. [`import`] runs the encode the
//! import runs (the contract's own expressions, compiled for the converting
//! target) and writes its planes as the artifact's objects, a packed
//! weight's codes, scales and zero points grouped as one object.

use std::collections::BTreeMap;
use std::path::Path;

use crate::contract::ModelContract;
use crate::error::Error;
use crate::executor::Execution;
use crate::executor::sink::MemorySink;
use crate::file::write::Writer;
use crate::file::zt;
use crate::plan::{CONVERT_TILE_MAP_MASK, StorageTarget, compile_streaming};
use crate::serving::Stamp;

/// Writes at `out` the serving artifact `contract` reads `source` into,
/// stamped `stamp`.
///
/// `source` is a container whose tensors the contract's expressions name (a
/// test's seeded planes: a packed weight stored as the floats it is
/// quantized from, at its unpacked shape). `contract` is the deployment's,
/// for its backend, so the planes land in that backend's served form.
/// `tiled` names the packed weights whose codes the backend lays in tiles.
/// The whole model passes through memory.
pub fn import(
    source: &Path,
    contract: &ModelContract,
    stamp: &Stamp,
    tiled: &[&str],
    out: &Path,
) -> Result<(), Error> {
    let metadata = zt::parse(source)?;
    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        ..StorageTarget::default()
    };
    let plan = compile_streaming(&metadata, contract, target)?;
    let mut sink = MemorySink::default();
    Execution::new(&plan, source.parent().unwrap_or(Path::new(".")))
        .streaming()
        .sink(&mut sink)
        .run()?;

    let name_of = |id| {
        plan.tensors
            .iter()
            .find(|decl| decl.id == id)
            .map(|decl| decl.name.clone())
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "the plan attaches tensor {id:?} and declares no such tensor"
                ))
            })
    };
    let mut writer = Writer::create_serving(out, &BTreeMap::new(), stamp.clone())?;
    for attachment in &plan.attachments {
        let object = name_of(attachment.tensor)?;
        let mut planes = vec![object.clone(), name_of(attachment.scale_tensor)?];
        if let Some(zero) = attachment.zero_point_tensor {
            planes.push(name_of(zero)?);
        }
        let is_tiled = tiled.contains(&object.as_str());
        writer.group(object, planes, is_tiled)?;
    }
    // A grouped object's planes are written together, codes first.
    let mut public: Vec<_> = plan
        .tensors
        .iter()
        .filter(|decl| decl.visibility.is_public())
        .collect();
    public.sort_by_key(|decl| writer.order_key(&decl.name));
    for decl in public {
        let bytes = sink.tensors.get(&decl.name).ok_or_else(|| {
            Error::Checkpoint(format!(
                "the encode published no plane named {:?}",
                decl.name
            ))
        })?;
        writer.add_tensor(decl, bytes)?;
    }
    writer.finish()
}
