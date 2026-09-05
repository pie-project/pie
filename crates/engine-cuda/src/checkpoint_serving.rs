//! Addressing adapter that reads model planes out of a serving `.zt`
//! checkpoint by trace param name, rather than by file position.
//!
//! A trace names planes (`w`, `w.scales`, `w.biases`); the artifact holds
//! objects (`w`, one blob of three planes). [`Serving`] resolves the one to
//! the other through the artifact, and the deferred fill reads whole objects
//! so that every block digest is checked against the bytes it covers.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::file::serve::Artifact;
use checkpoint::serving::{DigestAlgorithm, Digesting};
use model_ir::Trace;

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
    /// one. The stamp is checked once, before any plane lands; not re-checked here.
    #[must_use]
    pub fn open(path: &Path, trace: &Trace) -> Option<Serving> {
        let artifact = Artifact::open(path).ok()?;
        Some(Serving {
            artifact: Arc::new(artifact),
            names: trace.params.iter().map(|param| param.name.clone()).collect(),
            path: path.to_path_buf(),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// One plane's bytes, borrowed from the mapping, or `None` if this
    /// artifact does not carry that name.
    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let name = self.names.get(id as usize)?;
        self.artifact.plane(name).ok()
    }

    /// The trace's name for this ordinal, or `None` past the end.
    #[must_use]
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    /// A plane for a seat that hands out a pointer `reserved` bytes wide:
    /// the plane's own bytes, answered only when they fit the reservation
    /// and the mapping extends `reserved` bytes past the plane's start. No
    /// kernel reads past the plane; what lies beyond it (the next plane, or
    /// the file's tail) only keeps the pointer inside the mapping.
    #[must_use]
    pub fn plane_reserved(&self, id: u32, reserved: u64) -> Option<&[u8]> {
        let name = self.names.get(id as usize)?;
        let located = self.artifact.locate(name).ok()?;
        let start = located.at + located.plane.offset;
        if located.plane.len > reserved || start.checked_add(reserved)? > self.artifact.mapped_len() {
            return None;
        }
        let blob = self.artifact.object(&located.object).ok()?;
        blob.get(located.plane.range())
    }

    /// Verifies the objects holding these planes, before a kernel is pointed
    /// at one. Each object is checked against its own block digests.
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

/// One plane of a T1 object as the background fill needs it.
pub struct PlaneLanding {
    /// Where the plane starts within its object's blob.
    pub offset: u64,
    /// The plane's length: what the block digests cover.
    pub len: u64,
    /// Where it lands in the page-locked image.
    pub into: u64,
    /// What the seat reserved for it; the tail past `len` is zeroed.
    pub reserved: u64,
}

/// One T1 object: where its blob is, its blocks, and which of its planes
/// the seat holds.
pub struct Landing {
    pub object: String,
    /// Where the blob starts in the file.
    pub at: u64,
    pub algorithm: DigestAlgorithm,
    /// Every block of the blob: its blob-local range and its stated digest.
    pub blocks: Vec<(Range<u64>, Vec<u8>)>,
    pub planes: Vec<PlaneLanding>,
}

/// The whole background fill, stated before a thread is spawned for it.
pub struct Landings {
    pub path: PathBuf,
    pub landings: Vec<Landing>,
}

impl Serving {
    /// Whether this artifact can fill `layout`: every plane carried at the
    /// length the plan declares, fitting its reservation, in an object whose
    /// blocks are stated. `layout` is `Plan::host_layout`'s quads:
    /// `(param, into, bytes, reserved)`.
    pub fn covers(&self, layout: &[(u64, u64, u64, u64)]) -> Result<(), String> {
        self.refill(layout).map(drop)
    }

    /// What the background fill would have to do, or why it cannot.
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
                    let blocks = self
                        .artifact
                        .blocks(&located.object)
                        .map_err(|why| format!("`{name}` states no readable block digests: {why}"))?;
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

/// Thread count for reading behind a deferred seat.
const READERS: usize = 8;

/// A destination pointer a scope thread may carry. Sound to move and share
/// because `read_into`'s landings are disjoint windows on a mapping the
/// caller has handed to nobody else, and each lane touches only its own
/// items.
struct Carried(*mut u8);
// SAFETY: see the type doc and `read_into`'s contract.
unsafe impl Send for Carried {}
// SAFETY: as above.
unsafe impl Sync for Carried {}

/// One block of one object: the file window, and where each piece of it
/// goes (a seated plane's bytes to the image, the rest to a scratch buffer
/// so the digest still covers the whole block).
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

/// Why one block did not land.
enum Failed {
    Read(String),
    Rotten(String),
}

/// Reads the whole fill into `into`, hashing each block as its bytes land.
/// One work item per block rather than per plane, since plane sizes are
/// very uneven. Each seated plane's reservation past its length is zeroed.
///
/// # Safety
///
/// `into` must be valid for writes over every plane's `into..into +
/// reserved`, those windows must be disjoint, and nothing else may name a
/// byte of them for the duration.
///
/// # Errors
///
/// A filesystem error first, then the first block whose digest disagrees.
pub unsafe fn read_into(refill: &Landings, into: *mut u8) -> Result<(), String> {
    use std::os::unix::fs::FileExt;

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
                    into.add(usize::try_from(plane.into + (cursor - plane.offset)).unwrap_or(usize::MAX))
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
            let mine = work.get(lane * per..((lane + 1) * per).min(work.len())).unwrap_or(&[]);
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
                        if let Err(why) = file.read_exact_at(bytes, at) {
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
                thread
                    .join()
                    .unwrap_or_else(|_| vec![Err(Failed::Read("a read worker panicked".to_string()))])
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

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::file::emit::{self, Object, Payload};
    use checkpoint::serving::Stamp;
    use ztensor::{Leaf, Term};

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cs_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

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
            drafter: None,
        }
    }

    fn leaves<'a>(planes: &'a [(&'a str, Vec<u8>)]) -> Vec<Object<'a>> {
        planes
            .iter()
            .map(|(name, data)| Object::leaf(name, vec![data.len() as u64], Leaf::U8, data))
            .collect()
    }

    /// A plane is found by name, not by file position.
    #[test]
    fn a_plane_is_found_by_its_name_and_not_by_where_it_sits() {
        let dir = tmp("byname");
        let path = dir.join("m.zt");
        let planes = vec![("head", vec![2u8; 4096]), ("norm", vec![3u8; 4096]), ("embed", vec![1u8; 4096])];
        emit::write(&path, &Stamp::of("cuda", "qwen_3"), &BTreeMap::new(), 4096, &leaves(&planes), |o, p, _| {
            panic!("{o}/{p} is not streamed")
        })
        .unwrap();

        let trace = trace(&["embed", "norm", "head"]);
        let serving = Serving::open(&path, &trace).expect("a serving artifact opens");
        assert_eq!(serving.plane(0), Some(&planes[2].1[..]));
        assert_eq!(serving.plane(1), Some(&planes[1].1[..]));
        assert_eq!(serving.plane(2), Some(&planes[0].1[..]));
        assert_eq!(serving.plane(3), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A seat reserves more than the plane is long; the plane is answered at
    /// its own length as long as the mapping runs on past it.
    #[test]
    fn a_reservation_wider_than_the_plane_is_answered_inside_the_mapping() {
        let dir = tmp("reserved");
        let path = dir.join("m.zt");
        let planes = vec![("short", vec![5u8; 100]), ("long", vec![6u8; 4096])];
        emit::write(&path, &Stamp::of("cuda", "qwen_3"), &BTreeMap::new(), 4096, &leaves(&planes), |o, p, _| {
            panic!("{o}/{p} is not streamed")
        })
        .unwrap();
        let serving = Serving::open(&path, &trace(&["short", "long"])).expect("it opens");
        assert_eq!(serving.plane_reserved(0, 256), Some(&planes[0].1[..]));
        assert_eq!(serving.plane_reserved(0, 64), None, "a plane longer than its seat");
        assert_eq!(serving.plane_reserved(1, 1 << 40), None, "a reservation past the file");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An affine weight's three trace planes resolve into one object, and the
    /// fill rebuilds each plane's image from that object's blocks — catching
    /// a byte that changed in any of them.
    #[test]
    fn the_fill_rebuilds_the_image_from_one_object_and_sees_a_changed_byte() {
        let dir = tmp("refill");
        let path = dir.join("m.zt");
        let codes = vec![0x5Au8; 2 * 128 / 2];
        let scales = vec![1u8; 8];
        let biases = vec![2u8; 8];
        emit::write(
            &path,
            &Stamp::of("cuda", "qwen_3"),
            &BTreeMap::new(),
            4096,
            &[Object {
                name: "w",
                shape: vec![2, 128],
                term: Some(Term::parse("g64_u4_bf16_b_bf16").unwrap()),
                layout: None,
                attributes: None,
                planes: vec![Payload::Whole(&codes), Payload::Whole(&scales), Payload::Whole(&biases)],
            }],
            |o, p, _| panic!("{o}/{p} is not streamed"),
        )
        .unwrap();
        let trace = trace(&["w", "w.scales", "w.biases"]);
        let serving = Serving::open(&path, &trace).expect("a serving artifact opens");
        assert_eq!(serving.plane(1), Some(&scales[..]));
        serving.verify_planes(&[0, 1, 2]).unwrap();

        // Codes at 0 (256 reserved), biases at 256 (64 reserved); scales not seated.
        let layout = vec![(0u64, 0u64, 128u64, 256u64), (2, 256, 8, 64)];
        serving.covers(&layout).unwrap();
        let why = serving.covers(&[(0, 0, 64, 256)]).unwrap_err();
        assert!(why.contains("declares 64"), "{why}");
        let refill = serving.refill(&layout).expect("a described fill");
        assert_eq!(refill.landings.len(), 1);
        assert_eq!(refill.landings[0].planes.len(), 2);

        let mut image = vec![0xAAu8; 320];
        // SAFETY: the landings tile `image`, which is this thread's own.
        unsafe { read_into(&refill, image.as_mut_ptr()) }.expect("the fill reads and verifies");
        assert_eq!(&image[..128], &codes[..]);
        assert!(image[128..256].iter().all(|b| *b == 0), "the reservation's tail is zeroed");
        assert_eq!(&image[256..264], &biases[..]);

        let mut raw = std::fs::read(&path).unwrap();
        let at = raw.windows(8).position(|w| w == &scales[..]).expect("the scales are in the file");
        raw[at + 3] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();
        let why = unsafe { read_into(&refill, image.as_mut_ptr()) }.expect_err("a rotted block must not pass");
        assert!(why.contains("block 0 of \"w\""), "{why}");
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
