//! Reading a `pie.serving/1` artifact. [`Artifact`] is a serving artifact,
//! open and mapped, answering what this file is, which objects it carries in
//! which order, where one lies, and — only when asked — whether the bytes
//! match what was written. [`read_head`] answers "what is this file for"
//! without mapping.

use std::ops::Range;
use std::path::{Path, PathBuf};

use ztensor::{DigestAlgorithm, Manifest, Plane};

use crate::error::Error;
use crate::serving::{self, Blocks, Mismatch, PROFILE, Span, Stamp};
use crate::term::{blob_planes, plane_of};

/// Concurrent readers on the verify and fill paths.
pub const READERS: usize = 8;

/// A serving artifact, open and mapped. A `&[u8]` handed out here points
/// into the mapping.
pub struct Artifact {
    path: PathBuf,
    stamp: Stamp,
    source: ztensor::Source,
}

impl std::fmt::Debug for Artifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Artifact")
            .field("path", &self.path)
            .field("stamp", &self.stamp)
            .finish_non_exhaustive()
    }
}

/// One plane of an object, located: the object it is in, where that
/// object's blob starts in the file, and the plane's range within the blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    pub object: String,
    /// The blob's file offset; the plane lies at `at + plane.offset`.
    pub at: u64,
    pub plane: Plane,
}

impl Artifact {
    /// Open the serving artifact at `path` and map it. Reads the manifest and
    /// then the stamp; hashes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a container version or a `pie.serving/<n>`
    /// this build does not implement; [`Error::Checkpoint`] for a file that
    /// is not a serving artifact at all, or whatever the container refuses.
    pub fn open(path: &Path) -> Result<Artifact, Error> {
        let source = ztensor::Source::options()
            .vocabulary(&serving::vocabulary())
            .open(path)
            .map_err(Error::from)?;
        let attributes = source.attributes().ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} states no file attributes, so it carries no {PROFILE:?} block \
                 and is not a serving artifact",
                path.display(),
            ))
        })?;
        let stamp = Stamp::decode(attributes).map_err(|why| refuse(path, why))?;
        if source.provenance().as_root().is_none() {
            return Err(Error::Checkpoint(format!(
                "{} carries no manifest of its own, so nothing in it says where a \
                 plane lies",
                path.display(),
            )));
        }
        Ok(Artifact {
            path: path.to_path_buf(),
            stamp,
            source,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        self.source
            .provenance()
            .as_root()
            .expect("open refused a source with no manifest")
    }

    #[must_use]
    pub fn source(&self) -> &ztensor::Source {
        &self.source
    }

    /// How far the mapping extends: the file's length.
    #[must_use]
    pub fn mapped_len(&self) -> u64 {
        self.source.store(ztensor::StoreId(0)).len()
    }

    pub fn check(&self, deployment: &Stamp) -> Result<(), Mismatch> {
        self.stamp.check(deployment)
    }

    /// The serving sequence, recovered from the manifest's offsets.
    #[must_use]
    pub fn sequence(&self) -> Vec<&str> {
        serving::sequence(self.manifest())
    }

    /// Every serving object, in sequence order.
    #[must_use]
    pub fn spans(&self) -> Vec<Span<'_>> {
        serving::spans(self.manifest())
    }

    #[must_use]
    pub fn alignment(&self) -> u64 {
        serving::alignment(&self.spans())
    }

    /// One object's whole blob, borrowed from the mapping.
    pub fn object(&self, name: &str) -> Result<&[u8], Error> {
        self.source
            .tensor(name)
            .and_then(|tensor| tensor.map())
            .map_err(Error::from)
    }

    /// Where a plane named as a trace names it lies: the object holding it,
    /// that object's blob offset in the file, and the plane's range in the
    /// blob. `w` is object `w`'s codes, `w.scales` its gain plane,
    /// `w.biases` its offset plane — unless the file holds an object under
    /// the whole name, which wins. A blob in a shard is refused: the artifact
    /// is served out of its one mapped file.
    pub fn locate(&self, name: &str) -> Result<Located, Error> {
        for (object, path) in plane_of(name) {
            let Some(tensor) = self.source.get(&object) else {
                continue;
            };
            let planes = blob_planes(
                &object,
                tensor.layout(),
                tensor.term(),
                tensor.shape(),
                tensor.nbytes(),
            )?;
            let lone = path == "code" && planes.len() == 1;
            let Some(plane) = planes.into_iter().find(|plane| lone || plane.path == path) else {
                continue;
            };
            let at = tensor.locate().map_err(Error::from)?;
            if at.store.0 != 0 {
                return Err(Error::Checkpoint(format!(
                    "{object:?} lives in a shard of the serving artifact {}, which is served \
                     out of its one mapped file",
                    self.path.display()
                )));
            }
            return Ok(Located {
                object,
                at: at.offset,
                plane,
            });
        }
        Err(Error::Checkpoint(format!(
            "the serving artifact {} carries no plane named {name:?}",
            self.path.display()
        )))
    }

    /// One plane's bytes, by the name a trace gives it.
    pub fn plane(&self, name: &str) -> Result<&[u8], Error> {
        let located = self.locate(name)?;
        let blob = self.object(&located.object)?;
        blob.get(located.plane.range()).ok_or_else(|| {
            Error::Checkpoint(format!(
                "{name:?} lies at {:?} of {:?}, past its {} bytes",
                located.plane.range(),
                located.object,
                blob.len()
            ))
        })
    }

    /// One object's block digests.
    pub fn blocks(&self, object: &str) -> Result<Blocks<'_>, Error> {
        let entry = self.manifest().object(object).map_err(Error::from)?;
        Blocks::of(object, entry)
    }

    /// Hash the blocks of these objects and compare them to their digests.
    /// Also warms the page cache.
    pub fn verify(&self, objects: &[&str]) -> Result<(), Error> {
        let mut work = Vec::new();
        for object in objects {
            self.gather(object, &mut work)?;
        }
        self.hash(&work)
    }

    /// Hash every block of every serving object, and each object's own
    /// digest too.
    pub fn verify_all(&self) -> Result<(), Error> {
        let sequence = self.sequence();
        self.verify(&sequence)?;
        for name in &sequence {
            let tensor = self.source.tensor(name).map_err(Error::from)?;
            tensor.verify().map_err(Error::from)?;
        }
        Ok(())
    }

    fn gather<'a>(&'a self, object: &'a str, into: &mut Vec<Work<'a>>) -> Result<(), Error> {
        if !serving::is_serving(object) {
            return Err(Error::Checkpoint(format!(
                "{object:?} is a metadata object, which is not served"
            )));
        }
        let blocks = self.blocks(object)?;
        let bytes = self.object(object)?;
        if bytes.len() as u64 != blocks.size() {
            return Err(Error::Checkpoint(format!(
                "the serving artifact {} maps {} bytes for {object:?} and its block table \
                 covers {}",
                self.path.display(),
                bytes.len(),
                blocks.size()
            )));
        }
        for (span, stated) in blocks.iter() {
            let from = usize::try_from(span.start).unwrap_or(usize::MAX);
            let upto = usize::try_from(span.end).unwrap_or(usize::MAX);
            into.push(Work {
                object,
                algorithm: blocks.algorithm(),
                which: span.start / blocks.block_bytes(),
                span,
                bytes: &bytes[from..upto],
                stated,
            });
        }
        Ok(())
    }

    fn hash(&self, work: &[Work<'_>]) -> Result<(), Error> {
        let width = READERS.min(work.len().max(1));
        let per = work.len().div_ceil(width);
        let found: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let hashing: Vec<_> = work
                .chunks(per.max(1))
                .map(|mine| {
                    scope.spawn(move || {
                        mine.iter()
                            .map(|item| item.algorithm.digest(item.bytes).value)
                            .collect::<Vec<Vec<u8>>>()
                    })
                })
                .collect();
            let mut found = Vec::with_capacity(work.len());
            for thread in hashing {
                found.extend(thread.join().map_err(|_| {
                    Error::Checkpoint(format!(
                        "a hash lane panicked verifying the serving artifact {}",
                        self.path.display()
                    ))
                })?);
            }
            Ok::<_, Error>(found)
        })?;
        if found.len() != work.len() {
            return Err(Error::Internal(format!(
                "{} blocks were hashed for {} of work",
                found.len(),
                work.len()
            )));
        }
        for (item, found) in work.iter().zip(&found) {
            if found.as_slice() != item.stated {
                return Err(Error::Checkpoint(item.refusal(self.path.as_path(), found)));
            }
        }
        Ok(())
    }
}

struct Work<'a> {
    object: &'a str,
    algorithm: DigestAlgorithm,
    which: u64,
    span: Range<u64>,
    bytes: &'a [u8],
    stated: &'a [u8],
}

impl Work<'_> {
    fn refusal(&self, path: &Path, found: &[u8]) -> String {
        format!(
            "the serving artifact {} states {} {} for block {} of {:?} (bytes {}..{}), \
             whose bytes hash to {}",
            path.display(),
            self.algorithm.as_str(),
            hex(self.stated),
            self.which,
            self.object,
            self.span.start,
            self.span.end,
            hex(found),
        )
    }
}

/// Read a serving artifact's stamp without mapping it: `Ok(None)` is an
/// ordinary checkpoint (or not a container at all), `Ok(Some)` a serving
/// artifact this build reads, `Err` a file that claims to be one and is not.
pub fn stamp_of(path: &Path) -> Result<Option<Stamp>, Error> {
    let Ok(Some(manifest)) = ztensor::read::manifest_of(path) else {
        return Ok(None);
    };
    let Some(attributes) = manifest.attributes.as_ref() else {
        return Ok(None);
    };
    if serving::stated_profile(attributes).is_none() {
        return Ok(None);
    }
    Stamp::decode(attributes).map(Some)
}

pub fn read_head(path: &Path) -> Result<(Stamp, Manifest), Error> {
    let manifest = ztensor::read::manifest_of(path)
        .map_err(Error::from)?
        .ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} carries no manifest, so nothing in it says what it is for",
                path.display(),
            ))
        })?;
    let attributes = manifest.attributes.as_ref().ok_or_else(|| {
        Error::Checkpoint(format!(
            "{} states no file attributes, so it carries no {PROFILE:?} block and is \
             not a serving artifact",
            path.display(),
        ))
    })?;
    let stamp = Stamp::decode(attributes).map_err(|why| refuse(path, why))?;
    Ok((stamp, manifest))
}

fn refuse(path: &Path, why: Error) -> Error {
    let say = |what: String| {
        format!(
            "{what} This file is how this machine holds the model, not a cache of a \
             boot, so nothing here rewrites it and nothing here deletes it — run `{}` \
             to write it again.",
            serving::rebuild(Some(&path.to_string_lossy())),
        )
    };
    match why {
        Error::Unsupported(what) => Error::Unsupported(say(what)),
        Error::Checkpoint(what) => Error::Checkpoint(say(what)),
        other => other,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
