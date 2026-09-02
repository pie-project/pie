//! Addressing adapter that reads model planes out of a serving `.zt`
//! checkpoint by trace param name, rather than by file position.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::file::serve::Artifact;
use checkpoint::serving::BlockAlgorithm;
use model_ir::Trace;

/// A serving artifact, plus the ordinal-to-name translation `Spill` needs.
///
/// `Debug` prints the path and the plane count and not the names, since a
/// spill refusal formatting this would otherwise put thousands of names in
/// one line.
/// One mapping of a serving artifact, shared by every seat that reads it.
#[derive(Clone)]
pub struct Serving {
    artifact: Arc<Artifact>,
    names: Arc<[String]>,
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
    /// Opens `path` as this trace's serving artifact, or `None` if it isn't
    /// one. The stamp is checked once, before any plane lands; not
    /// re-checked here.
    #[must_use]
    pub fn open(path: &Path, trace: &Trace) -> Option<Serving> {
        let artifact = Artifact::open(path).ok()?;
        Some(Serving {
            artifact: Arc::new(artifact),
            names: trace.params.iter().map(|param| param.name.clone()).collect(),
            path: path.to_path_buf(),
        })
    }

    /// Which file these planes come out of.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// One plane's bytes, borrowed from the mapping, or `None` if this
    /// artifact does not carry that name. Each plane is its own object with
    /// a single `data` part.
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

    /// The trace's name for this ordinal, or `None` past the end — so a
    /// refusal can name the plane.
    #[must_use]
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    /// One plane's bytes at the padded extent a deferred tier seat reads
    /// (the declared length rounded up), or `None` if not carried that far.
    ///
    /// The last blob in the file has no padding behind it, so it can never
    /// serve a deferred seat.
    #[must_use]
    pub fn plane_padded(&self, id: u32, reserved: u64) -> Option<&[u8]> {
        let name = self.names.get(id as usize)?;
        self.artifact.span(name, "data", reserved).ok()
    }

    /// Verifies the planes a seat will serve, before a kernel is pointed at
    /// one. Each name is checked against its own stated block table.
    pub fn verify_planes(&self, params: &[u32]) -> Result<(), String> {
        let mut names = Vec::with_capacity(params.len());
        for id in params {
            let Some(name) = self.name(*id) else {
                return Err(format!(
                    "this deployment asks for param {id} and the trace it was opened with \
                     names only {}",
                    self.names.len()
                ));
            };
            names.push(name);
        }
        self.artifact
            .verify(&names)
            .map_err(|why| format!("{}: {why}", self.path.display()))
    }
}

/// One T1 plane as the background fill needs it: everything owned, nothing
/// borrowed, since the fill runs on a thread that reopens the file by path.
pub struct Landing {
    /// Where this plane's blob starts in the file.
    pub at: u64,
    /// How many bytes to read: the seat's padded extent.
    pub copy: u64,
    /// How many of those the block table covers. Shorter than `copy`; the
    /// gap is unhashed alignment padding.
    pub hashed: u64,
    /// Where they land in the page-locked image (`Plan::host_layout`'s offset).
    pub into: u64,
    /// This plane's own block digests, concatenated.
    pub digests: Vec<u8>,
}

/// The whole background fill, stated before a thread is spawned for it.
pub struct Refill {
    pub path: PathBuf,
    pub block_bytes: u64,
    pub algorithm: BlockAlgorithm,
    pub planes: Vec<Landing>,
}

impl Serving {
    /// Whether this artifact carries every plane of `layout` out to the
    /// extent a seat would read. A deferred seat needs the whole T1 set
    /// (unlike a spill, which needs only its own subset), so this is
    /// checked up front rather than left to fail inside `Tier::open`.
    ///
    /// # Errors
    ///
    /// Names the first plane that does not resolve, and at what extent.
    pub fn covers(&self, layout: &[(u64, u64, u64, u64)]) -> Result<(), String> {
        for (param, _, _, reserved) in layout.iter().copied() {
            let id = u32::try_from(param).unwrap_or(u32::MAX);
            if self.plane_padded(id, reserved).is_none() {
                let plane = self
                    .name(id)
                    .map_or_else(|| format!("param {param}"), |name| format!("`{name}`"));
                return Err(format!("it does not carry {plane} out to {reserved} bytes"));
            }
        }
        Ok(())
    }

    /// What the background fill would have to do, or why it cannot. A fill
    /// that can't be armed falls back to serving out of the mapping.
    /// `layout` is `Plan::host_layout`'s quads: `(param, into, _, reserved)`.
    ///
    /// # Errors
    ///
    /// An unknown param, an uncarried plane, an unreadable block table, or a
    /// plane in a shard (the fill reads the file by path with positioned
    /// reads, and a sharded artifact's blobs live in sibling files).
    pub fn refill(&self, layout: &[(u64, u64, u64, u64)]) -> Result<Refill, String> {
        let spans = self.artifact.spans();
        let padded = self.artifact.padded_spans();
        let mut planes = Vec::with_capacity(layout.len());
        for (param, into, _, reserved) in layout.iter().copied() {
            let id = u32::try_from(param).unwrap_or(u32::MAX);
            let Some(name) = self.name(id) else {
                return Err(format!("param {param} is past the end of this trace"));
            };
            let Some(which) = spans
                .iter()
                .position(|span| span.object == name && span.part == "data")
            else {
                return Err(format!("`{name}` is not a serving part of this artifact"));
            };
            let span = spans[which];
            if let Some(shard) = span.shard {
                return Err(format!(
                    "`{name}` lives in shard `{shard}`, and the fill reads the artifact by \
                     path with positioned reads"
                ));
            }
            let room = padded.get(which).copied().unwrap_or(span.length);
            let copy = reserved.min(room);
            // the block table covers the whole decoded size, so a `copy`
            // shorter than it would hash a prefix against a whole-plane digest.
            if copy < span.length {
                return Err(format!(
                    "`{name}` is {} bytes in the file and this plan reserves {copy} for it, \
                     and a table that covers the whole plane cannot answer for a prefix",
                    span.length,
                ));
            }
            let digests = self
                .artifact
                .blocks(name, "data")
                .map_err(|why| format!("`{name}` states no readable block table: {why}"))?
                .as_bytes()
                .to_vec();
            planes.push(Landing {
                at: span.offset,
                copy,
                hashed: span.length.min(copy),
                into,
                digests,
            });
        }
        Ok(Refill {
            path: self.path.clone(),
            block_bytes: self.artifact.stamp().block_bytes,
            algorithm: self.artifact.stamp().block_algorithm,
            planes,
        })
    }
}

/// Thread count for reading behind a deferred seat: about the disk's queue
/// depth, not this file (mirrors the tier road's measured `TIER_READERS`).
const READERS: usize = 8;

/// A destination pointer a scope thread may carry. Sound to move because
/// `read_into`'s landings are disjoint windows on a mapping the caller has
/// handed to nobody else.
struct Carried(*mut u8);
// SAFETY: see the type doc and `read_into`'s contract.
unsafe impl Send for Carried {}
// SAFETY: the work list is shared by reference and each lane touches only
// its own slice of it.
unsafe impl Sync for Carried {}

/// Reads the whole fill into `into`, hashing each block as its bytes land.
/// One work item per block rather than per plane, since plane sizes are
/// very uneven. A landing's `copy` (padded, read) and `hashed` (block-table
/// extent) can differ: the gap is file padding, read but not hashed.
///
/// # Safety
///
/// `into` must be valid for writes over every landing's `into..into + copy`,
/// those windows must be disjoint, and nothing else may name a byte of them
/// for the duration.
///
/// # Errors
///
/// A filesystem error first, then the first block whose digest disagrees.
pub unsafe fn read_into(refill: &Refill, into: *mut u8) -> Result<(), String> {
    use std::os::unix::fs::FileExt;

    let file = std::fs::File::open(&refill.path)
        .map_err(|why| format!("{}: {why}", refill.path.display()))?;
    let width = refill.algorithm.width();
    // (file offset, length, destination, digest to check — none for padding)
    let mut work: Vec<(u64, u64, Carried, Option<&[u8]>)> = Vec::new();
    for plane in &refill.planes {
        let blocks = checkpoint::serving::block_count(plane.hashed, refill.block_bytes);
        for which in 0..blocks {
            let Some(span) = checkpoint::serving::block_span(plane.hashed, refill.block_bytes, which)
            else {
                continue;
            };
            let at = usize::try_from(which).unwrap_or(usize::MAX).saturating_mul(width);
            // SAFETY: span.start < plane.hashed <= plane.copy, and the
            // caller guarantees `into` valid over the landing's whole width.
            let dst = Carried(unsafe {
                into.add(
                    usize::try_from(plane.into.saturating_add(span.start)).unwrap_or(usize::MAX),
                )
            });
            work.push((
                plane.at.saturating_add(span.start),
                span.end.saturating_sub(span.start),
                dst,
                plane.digests.get(at..at.saturating_add(width)),
            ));
        }
        // The padding, read and not hashed.
        if plane.copy > plane.hashed {
            // SAFETY: as above; this window starts where the last block
            // ended, so it overlaps none of them.
            let dst = Carried(unsafe {
                into.add(
                    usize::try_from(plane.into.saturating_add(plane.hashed)).unwrap_or(usize::MAX),
                )
            });
            work.push((
                plane.at.saturating_add(plane.hashed),
                plane.copy - plane.hashed,
                dst,
                None,
            ));
        }
    }

    let found: Vec<Result<(), String>> = std::thread::scope(|scope| {
        let width = READERS.min(work.len().max(1));
        let per = work.len().div_ceil(width.max(1));
        let mut reading = Vec::with_capacity(width);
        for lane in 0..width {
            let mine = work.get(lane * per..((lane + 1) * per).min(work.len())).unwrap_or(&[]);
            let file = &file;
            let algorithm = refill.algorithm;
            reading.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(mine.len());
                for (at, len, dst, stated) in mine {
                    // SAFETY: caller's clause, plus the disjointness the
                    // walk above preserves — no two items alias.
                    let bytes = unsafe {
                        core::slice::from_raw_parts_mut(dst.0, usize::try_from(*len).unwrap_or(0))
                    };
                    out.push(match file.read_exact_at(bytes, *at) {
                        Err(why) => Err(format!("read at {at}: {why}")),
                        Ok(()) => match stated {
                            None => Ok(()),
                            Some(stated) if algorithm.digest(bytes) == *stated => Ok(()),
                            Some(_) => Err(String::new()),
                        },
                    });
                }
                out
            }));
        }
        reading
            .into_iter()
            .flat_map(|thread| {
                thread.join().unwrap_or_else(|_| vec![Err("a read worker panicked".to_string())])
            })
            .collect()
    });

    let mut rotten = false;
    for outcome in found {
        match outcome {
            Ok(()) => {}
            // empty string is the digest-disagreement sentinel, held back so
            // a filesystem error anywhere in the run is reported first.
            Err(why) if why.is_empty() => rotten = true,
            Err(why) => return Err(why),
        }
    }
    match rotten {
        true => Err(format!(
            "{}: a block does not hash to what this artifact states for it",
            refill.path.display()
        )),
        false => Ok(()),
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

    /// Three named params in non-alphabetical order: serving order is the
    /// boot's read order, not name order.
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

    /// A plane is found by name, not by file position — the fixture writes
    /// planes in a different, same-length order from the trace's.
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
            &Stamp::of("cuda", "qwen_3"),
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

    /// An ordinary checkpoint is not a serving artifact; `Serving::open`
    /// returns `None`, sending the boot to the tier file instead.
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
