//! Writing a checkpoint: the output side of `convert`. The one writer the
//! loader has; `.zt` is the one format it writes.
//!
//! A quantized weight is one object whose blob holds its planes in canonical
//! order (codes, then scales, then biases). The planes arrive as the separate
//! declarations a plan produces, so a caller states the grouping first
//! ([`Writer::group`]) and then adds the planes in that order; the writer
//! streams them into one blob with the canonical padding between.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::format::cbor::{self, Value};
use ztensor::{Leaf, Plane, Term};

use crate::error::Error;
use crate::serving::{self, Stamp};
use crate::term::{blob_planes, gguf_name, term_of, MMA_TILED};
use crate::types::{Encoding, TensorDecl};

/// One tensor of the file: what to call it, and the bytes as stored.
pub struct WriteTensor<'a> {
    pub decl: &'a TensorDecl,
    pub bytes: &'a [u8],
}

/// What an object says about itself: its type, and a named layout with its
/// attributes when the bytes do not lie canonically.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Described {
    pub term: Option<Term>,
    pub layout: Option<String>,
    pub attributes: Option<Value>,
}

/// The description of the object a declaration heads. A gguf block array
/// is its own named layout; everything else is a term in canonical layout.
pub(crate) fn object_of(decl: &TensorDecl, tiled: bool) -> Result<Described, Error> {
    let name = &decl.name;
    if let Encoding::Quant(spec) = &decl.encoding
        && let Some(kind) = gguf_name(spec.scheme)
    {
        let row = ztensor::vocab::gguf::row_of(kind).ok_or_else(|| {
            Error::Internal(format!("gguf type {kind:?} has no registry row"))
        })?;
        return Ok(Described {
            term: row.term(),
            layout: Some(row.layout_id()),
            attributes: Some(cbor::map([
                ("elems_per_block", row.elems_per_block),
                ("block_bytes", row.block_bytes),
            ])),
        });
    }
    let Some(term) = term_of(&decl.encoding) else {
        return Err(Error::Checkpoint(format!(
            "tensor {name}: {:?} has no type this container can state",
            decl.encoding
        )));
    };
    if tiled {
        return Ok(Described {
            term: Some(term),
            layout: Some(MMA_TILED.to_string()),
            attributes: Some(cbor::map([
                ("band", u64::from(dtype::TILED_BAND)),
                ("step", u64::from(dtype::TILED_STEP)),
            ])),
        });
    }
    Ok(Described {
        term: Some(term),
        layout: None,
        attributes: None,
    })
}

/// Writes a checkpoint one tensor at a time, payloads in chunks.
///
/// Canonical form requires objects in ascending name order; [`write_zt`]
/// sorts for its caller, this type trusts its caller to add in order.
pub struct Writer {
    writer: Option<ztensor::Writer>,
    open: Option<Open>,
    /// Object name -> the plane names it holds, in canonical order.
    groups: BTreeMap<String, Group>,
    /// Plane name -> the object it belongs to.
    member_of: BTreeMap<String, String>,
    sharding: Option<Sharding>,
    metadata: BTreeMap<String, String>,
    serving: Option<Stamp>,
}

struct Group {
    planes: Vec<String>,
    tiled: bool,
}

/// One object being streamed: its sink, its planes, and how far along it is.
struct Open {
    name: String,
    sink: ztensor::Sink,
    /// `(plane name, where it lies)` in canonical order.
    planes: Vec<(String, Plane)>,
    /// Bytes of the blob written so far.
    cursor: u64,
    at: At,
}

enum At {
    /// Plane `next` is due; the object closes with its last plane, so
    /// `next` always names one.
    Between { next: usize },
    /// Plane `plane` is open and has received `written` bytes.
    Inside { plane: usize, written: u64 },
}

struct Sharding {
    root: PathBuf,
    attributes: BTreeMap<String, String>,
    max_bytes: u64,
    current_bytes: u64,
    index: u32,
    done: Vec<(String, PathBuf)>,
    meta: Vec<(String, Vec<u8>)>,
}

fn write_meta_object(writer: &mut ztensor::Writer, name: &str, bytes: &[u8]) -> Result<(), Error> {
    writer
        .object(name, |o| o.shape(vec![bytes.len() as u64]).term(Leaf::U8).bytes(bytes))
        .map_err(Error::from)
}

fn shard_name(index: u32) -> String {
    format!("{index:05}")
}

fn shard_path(root: &Path, index: u32) -> PathBuf {
    let stem = root
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    root.parent()
        .unwrap_or(Path::new("."))
        .join(format!("{stem}-{}.zt", shard_name(index)))
}

fn text_map(entries: &BTreeMap<String, String>) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(k, v)| (Value::Text(k.clone()), Value::Text(v.clone())))
            .collect(),
    )
}

impl Writer {
    /// Opens a checkpoint at `path`; `metadata` lands in the file's
    /// attributes. Publication is atomic.
    pub fn create(path: &Path, metadata: &BTreeMap<String, String>) -> Result<Self, Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        Self::opened(path, metadata, None)
    }

    /// [`create`](Self::create), for a file that is to be a serving artifact
    /// as well as a checkpoint: objects in the caller's (the boot's) order,
    /// block digests on every blob, and the stamp under [`serving::PROFILE`].
    pub fn create_serving(
        path: &Path,
        metadata: &BTreeMap<String, String>,
        stamp: Stamp,
    ) -> Result<Self, Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        Self::opened(path, metadata, Some(stamp))
    }

    /// [`create_serving`](Self::create_serving) for an artifact that already
    /// exists: the objects added through this writer land after the ones the
    /// file holds, and on [`finish`](Self::finish) the file is restamped with
    /// `stamp`; the provenance is kept with `metadata`'s keys written over it.
    ///
    /// **NOT ATOMIC.** The container is extended in place, so a run that dies
    /// between the first appended byte and the new footer leaves a file whose
    /// footer is not at its end; the caller holds the length the file had and
    /// truncates back to it. What that buys: no second copy of the artifact.
    pub fn append_serving(
        path: &Path,
        metadata: &BTreeMap<String, String>,
        stamp: Stamp,
    ) -> Result<Self, Error> {
        let mut carried = BTreeMap::new();
        {
            let source = ztensor::Source::open(path).map_err(Error::from)?;
            if let Some(Value::Map(entries)) = source.attributes() {
                for (key, value) in entries {
                    if let (Value::Text(key), Value::Text(value)) = (key, value)
                        && !key.starts_with(serving::PROFILE_FAMILY)
                    {
                        carried.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        for (key, value) in metadata {
            carried.insert(key.clone(), value.clone());
        }
        let writer = ztensor::Writer::options()
            .canonical(false)
            .blocks(serving::BLOCK_BYTES)
            .append(path)
            .map_err(Error::from)?;
        Ok(Self {
            writer: Some(writer),
            open: None,
            groups: BTreeMap::new(),
            member_of: BTreeMap::new(),
            sharding: None,
            metadata: carried,
            serving: Some(stamp),
        })
    }

    fn opened(
        path: &Path,
        metadata: &BTreeMap<String, String>,
        stamp: Option<Stamp>,
    ) -> Result<Self, Error> {
        let mut writer = match &stamp {
            Some(_) => ztensor::Writer::options()
                .canonical(false)
                .blocks(serving::BLOCK_BYTES)
                .publish(path)
                .map_err(Error::from)?,
            None => ztensor::Writer::publish(path).map_err(Error::from)?,
        };
        if !metadata.is_empty() {
            writer.set_attributes(text_map(metadata));
        }
        Ok(Self {
            writer: Some(writer),
            open: None,
            groups: BTreeMap::new(),
            member_of: BTreeMap::new(),
            sharding: None,
            metadata: metadata.clone(),
            serving: stamp,
        })
    }

    /// Opens a checkpoint that spills into shards once one file passes
    /// `max_shard_bytes`. The output is a root `.zt` beside `<stem>-00001.zt`,
    /// …; a tensor is never split across shards.
    pub fn create_sharded(
        root: &Path,
        metadata: &BTreeMap<String, String>,
        max_shard_bytes: u64,
    ) -> Result<Self, Error> {
        if max_shard_bytes == 0 {
            return Err(Error::Checkpoint(
                "a shard size of zero would put every tensor in its own file".into(),
            ));
        }
        if let Some(parent) = root.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        let first = shard_path(root, 1);
        let writer = ztensor::Writer::publish(&first).map_err(Error::from)?;
        Ok(Self {
            writer: Some(writer),
            open: None,
            groups: BTreeMap::new(),
            member_of: BTreeMap::new(),
            metadata: metadata.clone(),
            serving: None,
            sharding: Some(Sharding {
                root: root.to_path_buf(),
                attributes: metadata.clone(),
                max_bytes: max_shard_bytes,
                current_bytes: 0,
                index: 1,
                done: Vec::new(),
                meta: Vec::new(),
            }),
        })
    }

    /// States that the declarations named `planes` are the planes of one
    /// object called `object`, in canonical order: the codes first, then the
    /// gain (scales), then the offset (biases). `tiled` writes the object
    /// under [`MMA_TILED`]. The planes must then be added consecutively in
    /// that order.
    pub fn group(
        &mut self,
        object: impl Into<String>,
        planes: impl IntoIterator<Item = String>,
        tiled: bool,
    ) -> Result<(), Error> {
        let object = object.into();
        let planes: Vec<String> = planes.into_iter().collect();
        if self.groups.contains_key(&object) {
            return Err(Error::Checkpoint(format!(
                "object {object} was grouped twice"
            )));
        }
        if planes.first() != Some(&object) {
            return Err(Error::Checkpoint(format!(
                "object {object}: the first plane is the object's own codes and must carry \
                 its name, got {:?}",
                planes.first()
            )));
        }
        for plane in &planes {
            if let Some(other) = self.member_of.get(plane) {
                return Err(Error::Checkpoint(format!(
                    "plane {plane} is already a plane of {other}"
                )));
            }
            self.member_of.insert(plane.clone(), object.clone());
        }
        self.groups.insert(object, Group { planes, tiled });
        Ok(())
    }

    /// The order a declaration takes in a sorted write: its object's name,
    /// then its position among that object's planes.
    #[must_use]
    pub fn order_key(&self, name: &str) -> (String, usize) {
        match self.member_of.get(name) {
            Some(object) => {
                let at = self.groups[object]
                    .planes
                    .iter()
                    .position(|plane| plane == name)
                    .expect("a member's object lists it");
                (object.clone(), at)
            }
            None => (name.to_string(), 0),
        }
    }

    /// Declares a tensor and opens it for writing. Its payload is exactly
    /// `nbytes` bytes, delivered by [`write`](Self::write).
    pub fn begin_tensor(&mut self, decl: &TensorDecl, nbytes: u64) -> Result<(), Error> {
        crate::file::meta::reject_reserved(&decl.name)?;
        if let Some(open) = &mut self.open {
            let next = match open.at {
                At::Inside { plane, .. } => {
                    return Err(Error::Checkpoint(format!(
                        "tensor {} was begun while {} is still open",
                        decl.name, open.planes[plane].0
                    )));
                }
                At::Between { next } => next,
            };
            let (expected, plane) = &open.planes[next];
            if decl.name != *expected {
                return Err(Error::Checkpoint(format!(
                    "object {}: plane {expected} is expected next and {} was begun; an \
                     object's planes are added consecutively in canonical order",
                    open.name, decl.name
                )));
            }
            if nbytes != plane.len {
                return Err(Error::Checkpoint(format!(
                    "object {}: plane {expected} is {} bytes under its type and {nbytes} \
                     were declared",
                    open.name, plane.len
                )));
            }
            if term_of(&decl.encoding).as_ref().and_then(Term::leaf) != Some(plane.leaf) {
                return Err(Error::Checkpoint(format!(
                    "object {}: plane {expected} holds `{}` under its type and was declared \
                     {:?}",
                    open.name, plane.leaf, decl.encoding
                )));
            }
            let writer = self.writer.as_mut().expect("writer present");
            pad_to(&mut open.sink, writer, &mut open.cursor, plane.offset)?;
            open.at = At::Inside {
                plane: next,
                written: 0,
            };
            return Ok(());
        }

        if let Some(object) = self.member_of.get(&decl.name).cloned()
            && object != decl.name
        {
            return Err(Error::Checkpoint(format!(
                "plane {} belongs to {object} and was begun before it; an object's \
                 planes follow its codes",
                decl.name
            )));
        }
        let (planes, tiled) = match self.groups.get(&decl.name) {
            Some(group) => (group.planes.clone(), group.tiled),
            None => (vec![decl.name.clone()], false),
        };
        let described = object_of(decl, tiled)?;
        let shape: Vec<u64> = decl
            .shape
            .iter()
            .map(|&d| {
                u64::try_from(d).map_err(|_| {
                    Error::Checkpoint(format!("tensor {} has negative extent {d}", decl.name))
                })
            })
            .collect::<Result<_, _>>()?;
        let expected = blob_planes(
            &decl.name,
            described.layout.as_deref(),
            described.term.as_ref(),
            &shape,
            nbytes,
        )?;
        if expected.len() != planes.len() {
            return Err(Error::Checkpoint(format!(
                "object {}: its type has {} planes and {} were grouped ({planes:?}); a \
                 quantized weight's scales and biases are written with its codes",
                decl.name,
                expected.len(),
                planes.len()
            )));
        }
        if expected[0].len != nbytes {
            return Err(Error::Checkpoint(format!(
                "object {}: its codes are {} bytes under its type and {nbytes} were declared",
                decl.name, expected[0].len
            )));
        }
        let total = expected.last().map_or(0, |plane| plane.offset + plane.len);
        let laid: Vec<(String, Plane)> = planes.into_iter().zip(expected).collect();
        self.roll_if_full(total)?;
        let name = decl.name.clone();
        let sink = self
            .writer()
            .stream(&decl.name, move |mut o| {
                o = o.shape(shape);
                if let Some(term) = described.term {
                    o = o.term(term);
                }
                if let Some(layout) = described.layout {
                    o = o.layout(layout);
                }
                if let Some(attributes) = described.attributes {
                    o = o.attributes(attributes);
                }
                o.length(total)
            })
            .map_err(Error::from)?;
        self.open = Some(Open {
            name,
            sink,
            planes: laid,
            cursor: 0,
            at: At::Inside {
                plane: 0,
                written: 0,
            },
        });
        if let Some(sharding) = &mut self.sharding {
            sharding.current_bytes = sharding.current_bytes.saturating_add(total);
        }
        Ok(())
    }

    /// Appends bytes to the open tensor.
    pub fn write(&mut self, chunk: &[u8]) -> Result<(), Error> {
        let Some(Open {
            name,
            sink,
            planes,
            cursor,
            at: At::Inside { plane, written },
        }) = &mut self.open
        else {
            return Err(Error::Checkpoint("no tensor is open".into()));
        };
        let (plane_name, plane) = &planes[*plane];
        if *written + chunk.len() as u64 > plane.len {
            return Err(Error::Checkpoint(format!(
                "object {name}: plane {plane_name} was declared {} bytes and has been handed {}",
                plane.len,
                *written + chunk.len() as u64
            )));
        }
        let writer = self.writer.as_mut().expect("writer present");
        sink.write(writer, chunk).map_err(Error::from)?;
        *written += chunk.len() as u64;
        *cursor += chunk.len() as u64;
        Ok(())
    }

    /// Closes the open tensor, which must have received its whole payload.
    /// The object closes with its last plane.
    pub fn end_tensor(&mut self) -> Result<(), Error> {
        let Some(open) = &mut self.open else {
            return Err(Error::Checkpoint("no tensor is open".into()));
        };
        let At::Inside { plane, written } = open.at else {
            return Err(Error::Checkpoint("no tensor is open".into()));
        };
        let (name, laid) = &open.planes[plane];
        if written != laid.len {
            return Err(Error::Checkpoint(format!(
                "object {}: plane {name} was declared {} bytes and {written} arrived",
                open.name, laid.len
            )));
        }
        if plane + 1 < open.planes.len() {
            open.at = At::Between { next: plane + 1 };
            return Ok(());
        }
        let open = self.open.take().expect("open");
        let writer = self.writer.as_mut().expect("writer present");
        open.sink.close(writer).map_err(Error::from)
    }

    /// Adds a tensor whose payload is already in memory.
    pub fn add_tensor(&mut self, decl: &TensorDecl, bytes: &[u8]) -> Result<(), Error> {
        self.begin_tensor(decl, bytes.len() as u64)?;
        self.write(bytes)?;
        self.end_tensor()
    }

    /// Adds a metadata object at `path` under the reserved namespace, stored
    /// as a `u8` object so it versions with the weights under one manifest.
    pub fn add_meta(&mut self, path: &str, bytes: &[u8]) -> Result<(), Error> {
        self.nothing_open("a metadata object")?;
        if let Some(sharding) = &mut self.sharding {
            sharding.meta.push((path.to_string(), bytes.to_vec()));
            return Ok(());
        }
        let name = crate::file::meta::meta_name(path);
        write_meta_object(self.writer(), &name, bytes)
    }

    fn nothing_open(&self, what: &str) -> Result<(), Error> {
        if let Some(open) = &self.open {
            let plane = match open.at {
                At::Between { next } | At::Inside { plane: next, .. } => next,
            };
            return Err(Error::Checkpoint(format!(
                "{what} was added while object {} still waits for its plane {}",
                open.name, open.planes[plane].0
            )));
        }
        Ok(())
    }

    fn roll_if_full(&mut self, nbytes: u64) -> Result<(), Error> {
        let Some(sharding) = &self.sharding else {
            return Ok(());
        };
        if sharding.current_bytes == 0
            || sharding.current_bytes.saturating_add(nbytes) <= sharding.max_bytes
        {
            return Ok(());
        }
        let writer = self.writer.take().expect("writer present");
        writer.finish().map_err(Error::from)?;

        let sharding = self.sharding.as_mut().expect("sharding present");
        sharding.done.push((
            shard_name(sharding.index),
            shard_path(&sharding.root, sharding.index),
        ));
        sharding.index += 1;
        sharding.current_bytes = 0;
        let next = shard_path(&sharding.root, sharding.index);
        self.writer = Some(ztensor::Writer::publish(&next).map_err(Error::from)?);
        Ok(())
    }

    fn finish_sharded(mut self, sharding: Sharding) -> Result<(), Error> {
        let writer = self.writer.take().expect("writer present");
        writer.finish().map_err(Error::from)?;
        let mut shards = sharding.done;
        shards.push((
            shard_name(sharding.index),
            shard_path(&sharding.root, sharding.index),
        ));

        let mut root = ztensor::Writer::options()
            .canonical(false)
            .publish(&sharding.root)
            .map_err(Error::from)?;
        if let Some(stamp) = self.serving.take() {
            root.set_attributes(serving_attributes(&stamp, &sharding.attributes)?);
        } else if !sharding.attributes.is_empty() {
            root.set_attributes(text_map(&sharding.attributes));
        }
        for (name, path) in &shards {
            let identity = ztensor::read::shard_identity(path, ztensor::DigestAlgorithm::Xxh3)
                .map_err(Error::from)?;
            root.add_shard(name.clone(), &identity)
                .map_err(Error::from)?;
            let manifest = ztensor::read::manifest_of(path)
                .map_err(Error::from)?
                .ok_or_else(|| {
                    Error::Checkpoint(format!("shard {} carries no manifest", path.display()))
                })?;
            for (object_name, object) in &manifest.objects {
                root.link(object_name.clone(), object, name)
                    .map_err(Error::from)?;
            }
        }
        for (path, bytes) in &sharding.meta {
            let name = crate::file::meta::meta_name(path);
            write_meta_object(&mut root, &name, bytes)?;
        }
        root.finish().map_err(Error::from)?;
        Ok(())
    }

    /// Closes the manifest and moves the file into place.
    pub fn finish(mut self) -> Result<(), Error> {
        self.nothing_open("finish")?;
        if let Some(sharding) = self.sharding.take() {
            return self.finish_sharded(sharding);
        }
        let serving = self.serving.take();
        let metadata = std::mem::take(&mut self.metadata);
        let mut writer = self.writer.take().expect("writer present");
        if let Some(stamp) = serving {
            writer.set_attributes(serving_attributes(&stamp, &metadata)?);
        }
        writer.finish().map_err(Error::from)?;
        Ok(())
    }

    fn writer(&mut self) -> &mut ztensor::Writer {
        self.writer.as_mut().expect("writer present")
    }
}

/// Zero bytes from `cursor` to `offset`: the canonical padding between one
/// plane and the next.
pub(crate) fn pad_to(
    sink: &mut ztensor::Sink,
    writer: &mut ztensor::Writer,
    cursor: &mut u64,
    offset: u64,
) -> Result<(), Error> {
    const ZEROS: [u8; 64] = [0u8; 64];
    while *cursor < offset {
        let n = (offset - *cursor).min(ZEROS.len() as u64) as usize;
        sink.write(writer, &ZEROS[..n]).map_err(Error::from)?;
        *cursor += n as u64;
    }
    Ok(())
}

/// The file attributes of a serving artifact: the stamp under its own key,
/// and the flat provenance beside it. A provenance key that collides with
/// the profile's is refused rather than resolved.
pub(crate) fn serving_attributes(
    stamp: &Stamp,
    metadata: &BTreeMap<String, String>,
) -> Result<Value, Error> {
    let Value::Map(mut entries) = stamp.encode() else {
        return Err(Error::Internal(
            "the stamp encoded to something that is not a map".to_string(),
        ));
    };
    for (key, value) in metadata {
        if key.starts_with(serving::PROFILE_FAMILY) {
            return Err(Error::Checkpoint(format!(
                "the provenance key {key:?} is the key the stamp itself is written \
                 under, so this artifact would carry two answers for it"
            )));
        }
        entries.push((Value::Text(key.clone()), Value::Text(value.clone())));
    }
    Ok(Value::Map(entries))
}

/// Writes `tensors` as one canonical `.zt` file at `path`, ordered by name.
pub fn write_zt(
    path: &Path,
    metadata: &BTreeMap<String, String>,
    tensors: &[WriteTensor<'_>],
) -> Result<(), Error> {
    write_zt_grouped(path, metadata, tensors, &[])
}

/// [`write_zt`] with `groups` stating which declarations are one object's
/// planes (`(object, [codes, scales, biases])`); the file is ordered by
/// object name and plane.
pub fn write_zt_grouped(
    path: &Path,
    metadata: &BTreeMap<String, String>,
    tensors: &[WriteTensor<'_>],
    groups: &[(String, Vec<String>)],
) -> Result<(), Error> {
    let mut writer = Writer::create(path, metadata)?;
    for (object, planes) in groups {
        writer.group(object.clone(), planes.iter().cloned(), false)?;
    }
    let mut ordered: Vec<&WriteTensor<'_>> = tensors.iter().collect();
    ordered.sort_by_cached_key(|t| writer.order_key(&t.decl.name));
    for tensor in ordered {
        writer.add_tensor(tensor.decl, tensor.bytes)?;
    }
    writer.finish()
}

