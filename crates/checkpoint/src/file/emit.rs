//! Writing a `pie.serving/1` artifact from objects in hand or streamed:
//! objects in caller order, a [`Stamp`] under [`serving::PROFILE`], and block
//! digests on every blob. Deleting the stamp leaves an ordinary checkpoint.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::format::cbor::Value;
use ztensor::Term;

use crate::error::Error;
use crate::serving::{self, Stamp};
use crate::term::blob_planes;

/// The placement alignment this build writes at: payloads are read into
/// page-locked memory on huge-page boundaries.
pub const SERVING_ALIGN: u64 = 2 << 20;

/// Free space a write must leave behind after the artifact fits.
pub const MARGIN: u64 = 256 << 20;

/// Where one plane's bytes come from.
pub enum Payload<'a> {
    /// The bytes, in hand. Shares a blob with any identical object already
    /// written.
    Whole(&'a [u8]),
    /// The plane's size. The bytes arrive through [`write`]'s `fill`.
    Streamed(u64),
}

impl Payload<'_> {
    #[must_use]
    pub fn len(&self) -> u64 {
        match self {
            Payload::Whole(bytes) => bytes.len() as u64,
            Payload::Streamed(length) => *length,
        }
    }
}

/// One object of a serving artifact: its type, its layout, and one payload
/// per plane in canonical order.
pub struct Object<'a> {
    pub name: &'a str,
    pub shape: Vec<u64>,
    pub term: Option<Term>,
    /// A named layout, or `None` for the canonical one.
    pub layout: Option<String>,
    pub attributes: Option<Value>,
    /// One per plane of the type (one for a leaf or a named layout).
    pub planes: Vec<Payload<'a>>,
}

impl<'a> Object<'a> {
    /// A leaf object of `bytes`.
    #[must_use]
    pub fn leaf(name: &'a str, shape: Vec<u64>, leaf: ztensor::Leaf, bytes: &'a [u8]) -> Object<'a> {
        Object {
            name,
            shape,
            term: Some(Term::Leaf(leaf)),
            layout: None,
            attributes: None,
            planes: vec![Payload::Whole(bytes)],
        }
    }
}

/// Write the serving artifact at `path`, atomically.
///
/// `objects` are written in the order given, except metadata objects
/// (`__meta__/…`) go first regardless of position. `fill` delivers every
/// [`Payload::Streamed`] plane's bytes, once per plane, in canonical order,
/// and must deliver exactly [`Payload::len`] bytes. An object may not mix
/// the two kinds.
pub fn write(
    path: &Path,
    stamp: &Stamp,
    provenance: &BTreeMap<String, String>,
    align: u64,
    objects: &[Object<'_>],
    fill: impl FnMut(&str, usize, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<PathBuf, Error> {
    let directory = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(directory).map_err(|why| {
        Error::Checkpoint(format!("cannot create {}: {why}", directory.display()))
    })?;
    refuse_for_space(directory, path, objects)?;

    let temp = partial_path(path);
    if let Err(why) = emit(&temp, stamp, provenance, align, objects, fill) {
        let _ = std::fs::remove_file(&temp);
        return Err(why);
    }
    if let Err(why) = check_tiling(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(why);
    }
    std::fs::rename(&temp, path).map_err(|why| {
        let _ = std::fs::remove_file(&temp);
        Error::Checkpoint(format!("publishing {}: {why}", path.display()))
    })?;
    Ok(path.to_path_buf())
}

fn emit(
    target: &Path,
    stamp: &Stamp,
    provenance: &BTreeMap<String, String>,
    align: u64,
    objects: &[Object<'_>],
    mut fill: impl FnMut(&str, usize, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut writer = ztensor::Writer::options()
        .canonical(false)
        .align(align)
        .blocks(serving::BLOCK_BYTES)
        .vocabulary(&serving::vocabulary())
        .create(target)
        .map_err(Error::from)?;

    let metadata = objects.iter().filter(|it| !serving::is_serving(it.name));
    let served = objects.iter().filter(|it| serving::is_serving(it.name));
    for object in metadata.chain(served) {
        if let Err(why) = add(&mut writer, object, &mut fill) {
            writer.abandon();
            return Err(why);
        }
    }
    let attributes = match super::write::serving_attributes(stamp, provenance) {
        Ok(attributes) => attributes,
        Err(why) => {
            writer.abandon();
            return Err(why);
        }
    };
    writer.set_attributes(attributes);
    writer.finish().map_err(Error::from)?;
    std::fs::File::open(target)
        .and_then(|file| file.sync_all())
        .map_err(|why| Error::Checkpoint(format!("syncing {}: {why}", target.display())))
}

/// Where a streamed plane's bytes go.
pub struct Chunks<'w> {
    writer: &'w mut ztensor::Writer,
    sink: &'w mut ztensor::Sink,
    written: u64,
    expect: u64,
}

impl Chunks<'_> {
    pub fn put(&mut self, chunk: &[u8]) -> Result<(), Error> {
        self.written = self.written.saturating_add(chunk.len() as u64);
        if self.written > self.expect {
            return Err(Error::Checkpoint(format!(
                "a streamed plane was declared {} bytes and has been handed {}",
                self.expect, self.written,
            )));
        }
        self.sink.write(self.writer, chunk).map_err(Error::from)
    }
}

/// Where an object's planes lie, checked against its payloads.
fn laid(object: &Object<'_>) -> Result<Vec<(u64, u64)>, Error> {
    let bad = |detail: String| Error::Checkpoint(format!("serving object {:?} {detail}", object.name));
    if object.planes.is_empty() {
        return Err(bad("has no planes, so it has nothing to serve".into()));
    }
    let expected: Vec<(u64, u64)> = blob_planes(
        object.name,
        object.layout.as_deref(),
        object.term.as_ref(),
        &object.shape,
        object.planes[0].len(),
    )?
    .into_iter()
    .map(|plane| (plane.offset, plane.len))
    .collect();
    if expected.len() != object.planes.len() {
        return Err(bad(format!(
            "has {} payloads and its type lays out {} planes",
            object.planes.len(),
            expected.len()
        )));
    }
    for (at, ((_, len), payload)) in expected.iter().zip(&object.planes).enumerate() {
        if payload.len() != *len {
            return Err(bad(format!(
                "plane {at} is {} bytes under its type and {} were given",
                len,
                payload.len()
            )));
        }
    }
    Ok(expected)
}

fn add(
    writer: &mut ztensor::Writer,
    object: &Object<'_>,
    fill: &mut impl FnMut(&str, usize, &mut Chunks<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let planes = laid(object)?;
    let streamed = object
        .planes
        .iter()
        .filter(|payload| matches!(payload, Payload::Streamed(_)))
        .count();
    if streamed != 0 && streamed != object.planes.len() {
        return Err(Error::Checkpoint(format!(
            "serving object {:?} mixes planes in hand with streamed planes; give the \
             object one kind or the other",
            object.name,
        )));
    }
    fn describe<'d>(object: &Object<'_>, mut o: ztensor::ObjectBuilder<'d>) -> ztensor::ObjectBuilder<'d> {
        o = o.shape(object.shape.clone());
        if let Some(term) = &object.term {
            o = o.term(term.clone());
        }
        if let Some(layout) = &object.layout {
            o = o.layout(layout.clone());
        }
        if let Some(attributes) = &object.attributes {
            o = o.attributes(attributes.clone());
        }
        o
    }
    let total = planes.last().map_or(0, |(at, len)| at + len);
    if streamed == 0 {
        let whole: Vec<&[u8]> = object
            .planes
            .iter()
            .map(|payload| match payload {
                Payload::Whole(bytes) => *bytes,
                Payload::Streamed(_) => unreachable!("counted above"),
            })
            .collect();
        // The container lays planes out only under the canonical layout; a
        // named layout takes the blob whole, so the tiled planes are laid
        // here and a gguf array is its one plane already.
        let blob: Vec<u8> = match (&object.layout, whole.as_slice()) {
            (None, _) => {
                return writer
                    .object(object.name, |o| describe(object, o).planes(whole))
                    .map_err(Error::from);
            }
            (Some(_), [one]) => {
                return writer
                    .object(object.name, |o| describe(object, o).bytes(one))
                    .map_err(Error::from);
            }
            (Some(_), _) => {
                let mut blob = vec![0u8; usize::try_from(total).map_err(|_| {
                    Error::Checkpoint(format!("{:?} does not fit in memory", object.name))
                })?];
                for ((at, _), bytes) in planes.iter().zip(&whole) {
                    let at = *at as usize;
                    blob[at..at + bytes.len()].copy_from_slice(bytes);
                }
                blob
            }
        };
        return writer
            .object(object.name, |o| describe(object, o).bytes(&blob))
            .map_err(Error::from);
    }
    let mut sink = writer
        .stream(object.name, |o| describe(object, o).length(total))
        .map_err(Error::from)?;
    let mut cursor = 0u64;
    for (index, (offset, expect)) in planes.iter().copied().enumerate() {
        super::write::pad_to(&mut sink, writer, &mut cursor, offset)?;
        let mut chunks = Chunks {
            writer,
            sink: &mut sink,
            written: 0,
            expect,
        };
        fill(object.name, index, &mut chunks)?;
        if chunks.written != expect {
            return Err(Error::Checkpoint(format!(
                "{}'s plane {index} was declared {expect} bytes and {} arrived",
                object.name, chunks.written,
            )));
        }
        cursor += expect;
    }
    sink.close(writer).map_err(Error::from)
}

fn partial_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|it| it.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact.zt".to_string());
    path.with_file_name(format!(".{name}.{}.part", std::process::id()))
}

fn refuse_for_space(directory: &Path, path: &Path, objects: &[Object<'_>]) -> Result<(), Error> {
    let total: u64 = objects
        .iter()
        .flat_map(|object| object.planes.iter())
        .map(Payload::len)
        .sum();
    let need = total.saturating_add(MARGIN);
    let free = available_bytes(directory)?;
    if free < need {
        return Err(Error::Checkpoint(format!(
            "{} has {:.1} GiB free and the serving artifact {} wants {:.1} GiB \
             ({} GiB of planes plus a {} GiB margin); point the model store at a disk \
             with more space",
            directory.display(),
            free as f64 / (1u64 << 30) as f64,
            path.display(),
            need as f64 / (1u64 << 30) as f64,
            total >> 30,
            MARGIN >> 30,
        )));
    }
    Ok(())
}

fn available_bytes(directory: &Path) -> Result<u64, Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(directory.as_os_str().as_bytes()).map_err(|_| {
        Error::Checkpoint(format!(
            "{} is not a path this platform can state",
            directory.display()
        ))
    })?;
    // SAFETY: `path` is a NUL-terminated C string that outlives the call, and
    // `stat` is a plain out-parameter the call fully initializes on success.
    let mut stat = unsafe { std::mem::zeroed::<libc::statvfs>() };
    let rc = unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) };
    if rc != 0 {
        return Err(Error::Checkpoint(format!(
            "{}: cannot read the filesystem's free space",
            directory.display()
        )));
    }
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

fn check_tiling(temp: &Path, path: &Path) -> Result<(), Error> {
    let manifest = ztensor::read::manifest_of(temp)
        .map_err(Error::from)?
        .ok_or_else(|| {
            Error::Internal(format!(
                "{} was written without a manifest",
                path.display()
            ))
        })?;
    let spans = serving::spans(&manifest);
    if let Some(fault) = serving::tiling_fault(&spans, serving::alignment(&spans)) {
        return Err(Error::Checkpoint(format!(
            "the serving artifact {} {fault}",
            path.display(),
        )));
    }
    Ok(())
}

