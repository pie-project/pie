//! **THE `.zt` AS A SPILL SOURCE** (§M-4d) — the boot reading its planes out
//! of the one file, instead of out of a second one beside it.
//!
//! Since §M-4b the artifact `pie model import` writes IS the serving file: it
//! holds every plane of the trace, under this SKU's own names, in the order a
//! boot reads them, with a digest table per plane and a stamp saying what
//! deployment it was written for. Everything `weight_cache/tier.rs` exists to
//! provide, the checkpoint now provides itself.
//!
//! # What this type is, and what it is not
//!
//! It is the ADDRESSING adapter and nothing else. `experts::Spill` asks for a
//! plane by the param's ordinal in the trace; a serving artifact addresses an
//! object by NAME, because a name is what survives being written to a file
//! that another build will open. So this holds the trace's names in ordinal
//! order and does one lookup.
//!
//! It is not a second reader. `checkpoint::file::serve::Artifact` is the
//! reader — it opens without hashing, verifies the prefix a boot is about to
//! serve, and maps nothing it was not asked for — and this borrows from it.
//!
//! # Why the names are carried and not re-derived
//!
//! `Spill::plane` is called once per plane of the load, and a lookup that
//! walked the trace each time would be quadratic in the plane count. The
//! catalog's widest row has 2030 params.

use std::path::{Path, PathBuf};

use checkpoint::file::serve::Artifact;
use model_ir::Trace;

/// A serving artifact, plus the ordinal-to-name translation `Spill` needs.
///
/// `Debug` prints the path and the plane count and NOT the names: `Spill`
/// derives it, a refusal that formats a spill would otherwise put two
/// thousand plane names in one line, and the two facts an operator reading
/// such a line wants are which file and how much of it.
pub struct Serving {
    artifact: Artifact,
    names: Vec<String>,
    path: PathBuf,
}

impl std::fmt::Debug for Serving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Serving")
            .field("path", &self.path)
            .field("planes", &self.names.len())
            .finish()
    }
}

impl Serving {
    /// **Open `path` as this trace's serving artifact**, or say there is none.
    ///
    /// `None` for a file that is not a serving artifact at all, which is an
    /// ordinary checkpoint and is the road every load had before §M-4. A file
    /// that CLAIMS to be one and does not read back is not silently one of
    /// those — `serve::stamp_of` keeps those two apart, and the load has
    /// already refused on it at the door in `api.rs` before this is reached.
    ///
    /// The stamp is not re-checked here. It was checked before any plane
    /// landed, which is the whole of §M-4c, and checking it twice would
    /// invite the two checks to disagree.
    #[must_use]
    pub fn open(path: &Path, trace: &Trace) -> Option<Serving> {
        let artifact = Artifact::open(path).ok()?;
        Some(Serving {
            artifact,
            names: trace.params.iter().map(|param| param.name.clone()).collect(),
            path: path.to_path_buf(),
        })
    }

    /// Which file these planes come out of.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// **One plane's bytes**, borrowed from the mapping, or `None` for a name
    /// this artifact does not carry.
    ///
    /// The part is `data`: a checkpoint's streamed object is single-part by
    /// construction — `file/write.rs`'s `begin_tensor` declares one part and
    /// names it that — and a split-plane bank is two `Trace::params` rows
    /// here, so each is its own object rather than two parts of one.
    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let name = self.names.get(id as usize)?;
        self.artifact.part(name, "data").ok()
    }

    /// The artifact underneath, for a caller that needs the reader rather than
    /// the addressing.
    #[must_use]
    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::file::emit::{self, Object, Part, Payload};
    use checkpoint::serving::Stamp;
    use std::collections::BTreeMap;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cs_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A trace of three named params, in an order that is not name order — a
    /// serving artifact's is the boot's read order and this must not depend
    /// on the alphabet.
    fn trace(names: &[&str]) -> Trace {
        Trace {
            name: "qwen_3".to_string(),
            platform: model_ir::Platform::Cuda,
            params: names
                .iter()
                .map(|name| model_ir::Param {
                    name: (*name).to_string(),
                    shape: vec![4096],
                    shard: model_ir::Shard::Replicated,
                    dtype: model_ir::Dtype::U8,
                    source: model_ir::ParamSource::default(),
                })
                .collect(),
            caches: Vec::new(),
            values: Vec::new(),
            nodes: Vec::new(),
            seams: Vec::new(),
        }
    }

    /// **THE ORDINAL IS THE TRACE'S AND THE NAME IS THE FILE'S**, and this is
    /// the whole of what this adapter does.
    ///
    /// The fixture writes the planes in a DIFFERENT order from the trace's, so
    /// a lookup that used the file's position instead of the name — which is
    /// what both indices this replaces do — comes back with the wrong bytes
    /// rather than with none. Same lengths on purpose, so the wrong answer
    /// would be the right SIZE.
    #[test]
    fn a_plane_is_found_by_its_name_and_not_by_where_it_sits() {
        let dir = tmp("byname");
        let path = dir.join("m.zt");
        let bytes = |seed: u8| vec![seed; 4096];
        let (embed, head, norm) = (bytes(1), bytes(2), bytes(3));
        let objects: Vec<Object<'_>> = [("head", &head), ("norm", &norm), ("embed", &embed)]
            .into_iter()
            .map(|(name, data)| Object {
                name,
                shape: vec![4096],
                layout: "dense",
                attributes: None,
                parts: vec![Part {
                    name: "data",
                    dtype: ztensor::DType::U8,
                    logical: None,
                    payload: Payload::Whole(data),
                }],
            })
            .collect();
        emit::write(
            &path,
            &Stamp::of("cuda", 1, "qwen_3", "bf16", None),
            &BTreeMap::new(),
            4096,
            &objects,
            |o, p, _| panic!("{o}/{p} is not streamed"),
        )
        .unwrap();

        // The trace's order is embed, norm, head — none of which is the
        // file's, and none of which is alphabetical.
        let trace = trace(&["embed", "norm", "head"]);
        let serving = Serving::open(&path, &trace).expect("a serving artifact opens");
        assert_eq!(serving.plane(0), Some(&embed[..]), "param 0 is `embed`");
        assert_eq!(serving.plane(1), Some(&norm[..]), "param 1 is `norm`");
        assert_eq!(serving.plane(2), Some(&head[..]), "param 2 is `head`");
        assert_eq!(serving.plane(3), None, "there is no param 3");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A plane the trace declares and the artifact does not hold answers
    /// `None` — the hole `Spill::remedy` has a sentence for — rather than
    /// panicking or returning somebody else's bytes.
    #[test]
    fn a_name_the_artifact_does_not_hold_is_a_hole_and_not_a_neighbour() {
        let dir = tmp("hole");
        let path = dir.join("m.zt");
        let data = vec![9u8; 4096];
        emit::write(
            &path,
            &Stamp::of("cuda", 1, "qwen_3", "bf16", None),
            &BTreeMap::new(),
            4096,
            &[Object {
                name: "embed",
                shape: vec![4096],
                layout: "dense",
                attributes: None,
                parts: vec![Part {
                    name: "data",
                    dtype: ztensor::DType::U8,
                    logical: None,
                    payload: Payload::Whole(&data),
                }],
            }],
            |o, p, _| panic!("{o}/{p} is not streamed"),
        )
        .unwrap();

        let serving =
            Serving::open(&path, &trace(&["embed", "gained_since"])).expect("it opens");
        assert_eq!(serving.plane(0), Some(&data[..]));
        assert_eq!(serving.plane(1), None, "a trace that gained a plane");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An ordinary checkpoint is not a spill source, and saying so is how the
    /// boot keeps its older roads: `Serving::open` answering `None` is what
    /// sends it to the tier file and then to the resident snapshot.
    #[test]
    fn an_ordinary_checkpoint_is_not_a_serving_artifact() {
        let dir = tmp("plain");
        let path = dir.join("plain.zt");
        let mut writer =
            checkpoint::file::write::Writer::create(&path, &BTreeMap::new()).unwrap();
        let decl = checkpoint::types::TensorDecl {
            id: checkpoint::types::TensorId(0),
            name: "embed".to_string(),
            shape: vec![16],
            encoding: checkpoint::types::Encoding::Raw(checkpoint::types::DType::U8),
            alignment: 256,
            visibility: checkpoint::types::Visibility::default(),
        };
        writer.add_tensor(&decl, &[0u8; 16]).unwrap();
        writer.finish().unwrap();
        assert!(Serving::open(&path, &trace(&["embed"])).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
