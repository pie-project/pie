use std::borrow::Cow;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ztensor::{DigestAlgorithm, Manifest, Plane};

use crate::error::Error;
use crate::serving::{self, Blocks, Mismatch, PROFILE, Span, Stamp};
use crate::term::{blob_planes, plane_of};

pub const READERS: usize = 8;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    pub object: String,
    pub at: u64,
    pub plane: Plane,
}

impl Artifact {
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

    #[must_use]
    pub fn mapped_len(&self) -> u64 {
        self.source.store(ztensor::StoreId(0)).len()
    }

    pub fn check(&self, deployment: &Stamp) -> Result<(), Mismatch> {
        self.stamp.check(deployment)
    }

    #[must_use]
    pub fn sequence(&self) -> Vec<&str> {
        serving::sequence(self.manifest())
    }

    #[must_use]
    pub fn spans(&self) -> Vec<Span<'_>> {
        serving::spans(self.manifest())
    }

    #[must_use]
    pub fn alignment(&self) -> u64 {
        serving::alignment(&self.spans())
    }

    /// The zero-copy view of an object: the mapped file, or the chunk of a
    /// memory mount it lies in. An object straddling two chunks has no such
    /// view and is refused; `object_bytes` reads it either way.
    pub fn object(&self, name: &str) -> Result<&[u8], Error> {
        match self.object_bytes(name)? {
            Cow::Borrowed(bytes) => Ok(bytes),
            Cow::Owned(_) => Err(Error::Checkpoint(format!(
                "{name:?} lies across two chunks of the memory mount {}, which has no \
                 zero-copy view of it",
                self.path.display()
            ))),
        }
    }

    /// An object's bytes: borrowed where a zero-copy view exists, a copy
    /// where it lies across two chunks of a memory mount.
    pub fn object_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>, Error> {
        self.source
            .tensor(name)
            .and_then(|tensor| tensor.bytes())
            .map_err(Error::from)
    }

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

    pub fn blocks(&self, object: &str) -> Result<Blocks<'_>, Error> {
        let entry = self.manifest().object(object).map_err(Error::from)?;
        Blocks::of(object, entry)
    }

    pub fn verify(&self, objects: &[&str]) -> Result<(), Error> {
        let mut blobs = Vec::with_capacity(objects.len());
        for object in objects {
            if !serving::is_serving(object) {
                return Err(Error::Checkpoint(format!(
                    "{object:?} is a metadata object, which is not served"
                )));
            }
            blobs.push(self.object_bytes(object)?);
        }
        let mut work = Vec::new();
        for (object, bytes) in objects.iter().zip(&blobs) {
            self.gather(object, bytes, &mut work)?;
        }
        self.hash(&work)
    }

    pub fn verify_all(&self) -> Result<(), Error> {
        let sequence = self.sequence();
        self.verify(&sequence)?;
        for name in &sequence {
            let tensor = self.source.tensor(name).map_err(Error::from)?;
            tensor.verify().map_err(Error::from)?;
        }
        Ok(())
    }

    fn gather<'a>(
        &'a self,
        object: &'a str,
        bytes: &'a [u8],
        into: &mut Vec<Work<'a>>,
    ) -> Result<(), Error> {
        let blocks = self.blocks(object)?;
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
        // Bytes already in memory (or a platform with one thread) are hashed
        // in place; the reader lanes exist to keep a disk busy.
        let found =
            if cfg!(target_arch = "wasm32") || self.source.store(ztensor::StoreId(0)).is_memory() {
                work.iter()
                    .map(|item| item.algorithm.digest(item.bytes).value)
                    .collect()
            } else {
                self.hash_across_readers(work)?
            };
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

    fn hash_across_readers(&self, work: &[Work<'_>]) -> Result<Vec<Vec<u8>>, Error> {
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
        Ok(found)
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
