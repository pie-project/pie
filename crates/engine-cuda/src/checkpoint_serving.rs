#![allow(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::file::serve::Artifact;
use checkpoint::serving::{DigestAlgorithm, Digesting};
use poem_ir::Trace;

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
    #[must_use]
    pub fn open(path: &Path, trace: &Trace) -> Option<Serving> {
        let artifact = Artifact::open(path).ok()?;
        Some(Serving {
            artifact: Arc::new(artifact),
            names: trace
                .params
                .iter()
                .map(|param| param.name.clone())
                .collect(),
            path: path.to_path_buf(),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let name = self.names.get(id as usize)?;
        self.artifact.plane(name).ok()
    }

    /// Whether the file holds each of `params` as the trace reads it: a
    /// plane of exactly `bytes[param]` bytes, which a pool slot can be
    /// filled from where it lies.
    #[must_use]
    pub fn holds(&self, params: &[usize], bytes: &[u64]) -> bool {
        params.iter().all(|&param| {
            u32::try_from(param)
                .ok()
                .and_then(|id| self.plane(id))
                .is_some_and(|plane| Some(&(plane.len() as u64)) == bytes.get(param))
        })
    }

    #[must_use]
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    pub fn verify_planes(&self, params: &[u32]) -> Result<(), String> {
        let mut objects = BTreeSet::new();
        for id in params {
            let Some(name) = self.name(*id) else {
                return Err(format!(
                    "this deployment asks for param {id} and the trace it was opened with \
                     names only {}",
                    self.names.len()
                ));
            };
            let located = self
                .artifact
                .locate(name)
                .map_err(|why| format!("{}: {why}", self.path.display()))?;
            objects.insert(located.object);
        }
        let borrowed: Vec<&str> = objects.iter().map(String::as_str).collect();
        self.artifact
            .verify(&borrowed)
            .map_err(|why| format!("{}: {why}", self.path.display()))
    }
}

pub struct PlaneLanding {
    pub offset: u64,
    pub len: u64,
    pub into: u64,
    pub reserved: u64,
}

pub struct Landing {
    pub object: String,
    pub at: u64,
    pub algorithm: DigestAlgorithm,
    pub blocks: Vec<(Range<u64>, Vec<u8>)>,
    pub planes: Vec<PlaneLanding>,
}

pub struct Landings {
    pub path: PathBuf,
    pub landings: Vec<Landing>,
}

impl Serving {
    pub fn covers(&self, layout: &[(u64, u64, u64, u64)]) -> Result<(), String> {
        self.refill(layout).map(drop)
    }

    pub fn refill(&self, layout: &[(u64, u64, u64, u64)]) -> Result<Landings, String> {
        let mut by_object: BTreeMap<String, Landing> = BTreeMap::new();
        for (param, into, bytes, reserved) in layout.iter().copied() {
            let id = u32::try_from(param).unwrap_or(u32::MAX);
            let Some(name) = self.name(id) else {
                return Err(format!("param {param} is past the end of this trace"));
            };
            let located = self.artifact.locate(name).map_err(|why| why.to_string())?;
            if located.plane.len != bytes {
                return Err(format!(
                    "`{name}` is {} bytes and this plan declares {bytes}; the file is another \
                     deployment's",
                    located.plane.len
                ));
            }
            if bytes > reserved {
                return Err(format!(
                    "`{name}` is {bytes} bytes and this plan reserves {reserved} for it"
                ));
            }
            let landing = match by_object.get_mut(&located.object) {
                Some(landing) => landing,
                None => {
                    let blocks = self.artifact.blocks(&located.object).map_err(|why| {
                        format!("`{name}` states no readable block digests: {why}")
                    })?;
                    by_object.insert(
                        located.object.clone(),
                        Landing {
                            object: located.object.clone(),
                            at: located.at,
                            algorithm: blocks.algorithm(),
                            blocks: blocks
                                .iter()
                                .map(|(span, digest)| (span, digest.to_vec()))
                                .collect(),
                            planes: Vec::new(),
                        },
                    );
                    by_object.get_mut(&located.object).expect("just inserted")
                }
            };
            landing.planes.push(PlaneLanding {
                offset: located.plane.offset,
                len: located.plane.len,
                into,
                reserved,
            });
        }
        Ok(Landings {
            path: self.path.clone(),
            landings: by_object.into_values().collect(),
        })
    }
}

const READERS: usize = 8;

struct Carried(*mut u8);
// SAFETY: see the type doc and `read_into`'s contract.
unsafe impl Send for Carried {}
// SAFETY: as above.
unsafe impl Sync for Carried {}

struct Work<'a> {
    object: &'a str,
    which: usize,
    at: u64,
    segments: Vec<(u64, Option<Carried>)>,
    algorithm: DigestAlgorithm,
    stated: &'a [u8],
}

impl Work<'_> {
    fn refusal(&self, path: &Path, found: &[u8]) -> String {
        format!(
            "the serving artifact {} states {} {} for block {} of {:?}, whose bytes hash to {}",
            path.display(),
            self.algorithm.as_str(),
            hex(self.stated),
            self.which,
            self.object,
            hex(found),
        )
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

enum Failed {
    Read(String),
    Rotten(String),
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, at)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], at + done as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "the spill file ends inside a block",
                ));
            }
            Ok(n) => done += n,
            Err(why) if why.kind() == std::io::ErrorKind::Interrupted => {}
            Err(why) => return Err(why),
        }
    }
    Ok(())
}

/// # Safety
///
/// `into` must be valid for writes over every plane's reservation,
/// `plane.into .. plane.into + plane.reserved`.
pub unsafe fn read_into(refill: &Landings, into: *mut u8) -> Result<(), String> {
    let file = std::fs::File::open(&refill.path)
        .map_err(|why| format!("{}: {why}", refill.path.display()))?;
    let mut work: Vec<Work<'_>> = Vec::new();
    for landing in &refill.landings {
        for plane in &landing.planes {
            if plane.reserved > plane.len {
                let tail = usize::try_from(plane.reserved - plane.len).unwrap_or(0);
                // SAFETY: the caller guarantees `into` valid over the plane's
                // whole reservation.
                unsafe {
                    std::ptr::write_bytes(
                        into.add(usize::try_from(plane.into + plane.len).unwrap_or(usize::MAX)),
                        0,
                        tail,
                    );
                }
            }
        }
        let mut planes: Vec<&PlaneLanding> = landing.planes.iter().collect();
        planes.sort_by_key(|plane| plane.offset);
        for (which, (span, stated)) in landing.blocks.iter().enumerate() {
            let mut segments = Vec::new();
            let mut cursor = span.start;
            for plane in &planes {
                let plane_end = plane.offset + plane.len;
                if plane_end <= cursor || plane.offset >= span.end {
                    continue;
                }
                if plane.offset > cursor {
                    segments.push((plane.offset - cursor, None));
                    cursor = plane.offset;
                }
                let upto = plane_end.min(span.end);
                // SAFETY: `cursor - plane.offset < plane.len`, inside the
                // plane's reservation the caller vouches for.
                let dst = Carried(unsafe {
                    into.add(
                        usize::try_from(plane.into + (cursor - plane.offset)).unwrap_or(usize::MAX),
                    )
                });
                segments.push((upto - cursor, Some(dst)));
                cursor = upto;
            }
            if cursor < span.end {
                segments.push((span.end - cursor, None));
            }
            work.push(Work {
                object: &landing.object,
                which,
                at: landing.at + span.start,
                segments,
                algorithm: landing.algorithm,
                stated,
            });
        }
    }

    let found: Vec<Result<(), Failed>> = std::thread::scope(|scope| {
        let width = READERS.min(work.len().max(1));
        let per = work.len().div_ceil(width.max(1));
        let mut reading = Vec::with_capacity(width);
        for lane in 0..width {
            let mine = work
                .get(lane * per..((lane + 1) * per).min(work.len()))
                .unwrap_or(&[]);
            let file = &file;
            let path = &refill.path;
            reading.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(mine.len());
                let mut scratch = Vec::new();
                for item in mine {
                    let mut hasher = Digesting::new(item.algorithm);
                    let mut at = item.at;
                    let mut outcome = Ok(());
                    for (len, dst) in &item.segments {
                        let len = usize::try_from(*len).unwrap_or(0);
                        let bytes: &mut [u8] = match dst {
                            // SAFETY: caller's clause, plus the disjointness
                            // the walk above preserves — no two items alias.
                            Some(dst) => unsafe { core::slice::from_raw_parts_mut(dst.0, len) },
                            None => {
                                scratch.resize(len, 0);
                                &mut scratch[..len]
                            }
                        };
                        if let Err(why) = read_exact_at(file, bytes, at) {
                            outcome = Err(Failed::Read(format!("read at {at}: {why}")));
                            break;
                        }
                        hasher.update(bytes);
                        at += len as u64;
                    }
                    out.push(outcome.and_then(|()| {
                        let found = hasher.finish();
                        match found == item.stated {
                            true => Ok(()),
                            false => Err(Failed::Rotten(item.refusal(path, &found))),
                        }
                    }));
                }
                out
            }));
        }
        reading
            .into_iter()
            .flat_map(|thread| {
                thread.join().unwrap_or_else(|_| {
                    vec![Err(Failed::Read("a read worker panicked".to_string()))]
                })
            })
            .collect()
    });

    let mut rotten = None;
    for outcome in found {
        match outcome {
            Ok(()) => {}
            Err(Failed::Rotten(why)) => {
                rotten.get_or_insert(why);
            }
            Err(Failed::Read(why)) => return Err(why),
        }
    }
    rotten.map_or(Ok(()), Err)
}
