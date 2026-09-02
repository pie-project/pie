//! Reading tensors.
//!
//! A [`Source`] is a catalog over one or more [`Store`]s. One `.zt` file, a
//! `.zt` root plus its shards, a foreign checkpoint and a set of unrelated
//! files all come back as this same type. Only the way the catalog was built
//! differs.
//!
//! There are three ways to get at a tensor's bytes:
//!
//! - [`bytes`](Tensor::bytes) gives the best the source can do, as a
//!   `Cow<[u8]>` that says whether it borrowed or copied.
//! - [`map`](Tensor::map) gives a borrow or an error, never a hidden copy.
//! - [`locate`](Tensor::locate) gives the address, so the caller can do its
//!   own I/O with io_uring, cuFile or a staged host-to-device copy.
//!
//! A tensor is one blob. Where its type has several planes, [`Tensor::planes`]
//! says where each lies inside those bytes.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result, Rule};
use crate::format::cbor::Value;
use crate::format::validate;
use crate::format::{Blocks, Digest, DigestAlgorithm, Hasher, Manifest, Plane, Shard, Term};
use crate::provide::catalog::{Catalog, Entry, Location, Payload};
use crate::provide::store::{Store, StoreId};
use crate::vocab::Vocabulary;

// =======================================================================
// capabilities
// =======================================================================

/// What can be done with one tensor's bytes.
///
/// Each field is named after the operation it gates, and is computed from
/// that operation's own precondition, so the report cannot disagree with the
/// behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    /// [`Tensor::map`] will succeed: the bytes are raw and the file is mapped.
    pub map: bool,
    /// [`Tensor::locate`] will succeed: the decoded bytes are exactly one
    /// range of one file.
    pub locate: bool,
    /// [`Tensor::evict`] will succeed: no other blob shares an OS page with
    /// this one.
    pub evict: bool,
    /// [`Tensor::verify`] will check a digest rather than report that there
    /// is none to check.
    pub verify: bool,
    /// Largest power of two dividing the blob's file offset.
    pub alignment: u64,
}

/// The outcome of a successful [`Tensor::verify`].
///
/// A digest *mismatch* is not here: that is a rejected file, and it comes back
/// as `Err(Reject { rule: Rule::Digest, .. })`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verified {
    /// A digest was checked and matched.
    Digest,
    /// There is no digest; content rules (if any) passed.
    NoDigest,
}

impl Verified {
    pub fn is_checked(self) -> bool {
        self == Verified::Digest
    }
}

/// Where a [`Source`]'s description came from, and therefore who is making the
/// claim that its tensors are where it says they are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provenance<'a> {
    /// A `.zt` root: the file states its own structure, hash-checked on open,
    /// and every object may carry a digest.
    Root(&'a Manifest),
    /// A `.zt` data shard (spec §7.2): a container holding bytes and no
    /// manifest.
    DataShard,
    /// A foreign format, or several sources merged. The description was built
    /// by whoever opened the files.
    Projection,
}

impl<'a> Provenance<'a> {
    pub fn as_root(self) -> Option<&'a Manifest> {
        match self {
            Provenance::Root(manifest) => Some(manifest),
            _ => None,
        }
    }
}

// =======================================================================
// shard resolution
// =======================================================================

/// Resolves a shard name + identity to a file path. A name is a label the
/// producer chose, not a location: turning it into bytes is entirely the
/// transport's concern (spec §7.1, Appendix B).
pub trait ShardResolver {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf>;
}

impl<F: Fn(&str, &Shard) -> Result<PathBuf>> ShardResolver for F {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf> {
        self(name, shard)
    }
}

/// The positional convention (Appendix B), and the default: root
/// `<dir>/<stem>.zt` maps a shard named `n` to `<dir>/<stem>-<n>.zt`.
pub fn positional(root: impl AsRef<Path>) -> impl ShardResolver + 'static {
    let root = root.as_ref();
    let dir = root.parent().unwrap_or(Path::new(".")).to_path_buf();
    let stem = root
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    move |name: &str, _: &Shard| Ok(dir.join(format!("{stem}-{name}.zt")))
}

/// The content-addressed convention (Appendix B): a shard with digest
/// `algo:hex` lives at `<store>/blobs/<algo>/<hex>`.
pub fn cas(store: impl AsRef<Path>) -> impl ShardResolver + 'static {
    let store = store.as_ref().to_path_buf();
    move |_: &str, shard: &Shard| {
        Ok(store
            .join("blobs")
            .join(&shard.digest.algorithm)
            .join(shard.digest.hex()))
    }
}

/// Finds shards by identity: scans a directory once and matches each file by
/// size and whole-file digest, ignoring what anything is called.
pub struct DirectoryResolver {
    by_size: BTreeMap<u64, Vec<PathBuf>>,
    digests: std::sync::Mutex<HashMap<(u64, DigestAlgorithm), BTreeMap<Vec<u8>, PathBuf>>>,
}

impl DirectoryResolver {
    /// Indexes every `.zt` file directly inside `dir` by size. Reads no
    /// tensor bytes; hashing happens in `resolve`, once per bucket.
    pub fn scan(dir: impl AsRef<Path>) -> Result<Self> {
        let mut by_size: BTreeMap<u64, Vec<PathBuf>> = BTreeMap::new();
        for entry in std::fs::read_dir(dir.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "zt") {
                if let Ok(meta) = entry.metadata() {
                    by_size.entry(meta.len()).or_default().push(path);
                }
            }
        }
        for paths in by_size.values_mut() {
            paths.sort();
        }
        Ok(Self {
            by_size,
            digests: Default::default(),
        })
    }
}

impl ShardResolver for DirectoryResolver {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf> {
        let algo = shard.digest.algorithm()?;
        let mut cache = self.digests.lock().unwrap_or_else(|e| e.into_inner());
        let bucket = cache.entry((shard.size, algo)).or_insert_with(|| {
            let mut found = BTreeMap::new();
            for path in self.by_size.get(&shard.size).into_iter().flatten() {
                if let Ok(id) = shard_identity(path, algo) {
                    found.entry(id.digest.value).or_insert_with(|| path.clone());
                }
            }
            found
        });
        bucket.get(&shard.digest.value).cloned().ok_or_else(|| {
            Error::NotFound(format!(
                "shard {name:?} ({} bytes, {}) is not in the scanned directory",
                shard.size, shard.digest
            ))
        })
    }
}

// =======================================================================
// opening
// =======================================================================

/// How to open a source.
pub struct Options {
    vocab: Option<Arc<Vocabulary>>,
    resolver: Option<Box<dyn ShardResolver>>,
    map: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            vocab: None,
            resolver: None,
            map: true,
        }
    }
}

impl Options {
    /// The profiles the reader should know. Defaults to
    /// [`Vocabulary::standard`].
    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(Arc::new(vocab.clone()));
        self
    }

    /// How to find shard files. Defaults to [`positional`].
    pub fn resolver(mut self, resolver: impl ShardResolver + 'static) -> Self {
        self.resolver = Some(Box::new(resolver));
        self
    }

    /// Map the files (the default). With `false`, files are opened but not
    /// mapped: metadata and addresses are available, borrowed reads are not.
    pub fn map(mut self, map: bool) -> Self {
        self.map = map;
        self
    }

    pub(crate) fn vocabulary_arc(&self) -> Arc<Vocabulary> {
        self.vocab.clone().unwrap_or_else(Vocabulary::shared)
    }

    fn open_store(&self, path: &Path, format: &'static str) -> Result<Store> {
        if self.map {
            Store::map(path, format)
        } else {
            Store::index(path, format)
        }
    }

    /// Opens a `.zt` file, following its shard table if it has one.
    pub fn open(self, path: impl AsRef<Path>) -> Result<Source> {
        let path = path.as_ref();
        let vocab = self.vocabulary_arc();
        let root = self.open_store(path, "zt")?;
        let parsed = validate::store(&root, &vocab)?;
        let root = root.with_occupied(parsed.occupied);

        let Some((manifest, _)) = parsed.manifest else {
            return Ok(Source {
                stores: vec![root],
                catalog: Catalog::new(),
                manifest: None,
                data_shard: true,
                vocab,
            });
        };

        let mut stores = vec![root];
        let mut store_of: BTreeMap<&str, StoreId> = BTreeMap::new();

        let default;
        let resolver: &dyn ShardResolver = match &self.resolver {
            Some(r) => r.as_ref(),
            None => {
                default = positional(path);
                &default
            }
        };

        for (name, shard) in &manifest.shards {
            let shard_path = resolver.resolve(name, shard)?;
            let store = self.open_store(&shard_path, "zt")?;
            if store.len() != shard.size {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!(
                        "shard {name:?}: {} is {} bytes, the root expects {}",
                        shard_path.display(),
                        store.len(),
                        shard.size
                    ),
                ));
            }
            let parsed = validate::store(&store, &vocab)
                .map_err(|e| Error::reject(Rule::ShardIdentity, format!("shard {name:?}: {e}")))?;
            store_of.insert(name.as_str(), StoreId(stores.len() as u32));
            stores.push(store.with_occupied(parsed.occupied));
        }

        let catalog = resolve_manifest(&manifest, &store_of)?;
        Ok(Source {
            stores,
            catalog,
            manifest: Some(manifest),
            data_shard: false,
            vocab,
        })
    }

    /// Opens several `.zt` files as one name space.
    pub fn open_all(self, paths: &[impl AsRef<Path>]) -> Result<Source> {
        let vocab = self.vocabulary_arc();
        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            let opts = Options {
                vocab: Some(vocab.clone()),
                resolver: None,
                map: self.map,
            };
            sources.push(opts.open(path.as_ref())?);
        }
        Source::merge(sources)
    }

    /// Builds a source from a projection's own stores and catalog, under this
    /// vocabulary.
    pub fn from_parts(self, stores: Vec<Store>, catalog: Catalog) -> Result<Source> {
        for (name, entry) in catalog.iter() {
            if entry.store().0 as usize >= stores.len() {
                return Err(Error::InvalidInput(format!(
                    "{name:?} addresses store {} of {}",
                    entry.store(),
                    stores.len()
                )));
            }
        }
        Ok(Source {
            stores,
            catalog,
            manifest: None,
            data_shard: false,
            vocab: self.vocabulary_arc(),
        })
    }
}

/// Turns a manifest's blob references into addresses.
fn resolve_manifest(manifest: &Manifest, store_of: &BTreeMap<&str, StoreId>) -> Result<Catalog> {
    let mut catalog = Catalog::new();
    catalog.set_attributes(manifest.attributes.clone());
    for (name, obj) in &manifest.objects {
        let blob = &obj.blob;
        let store = match &blob.shard {
            None => StoreId(0),
            Some(s) => *store_of.get(s.as_str()).ok_or_else(|| {
                Error::reject(
                    Rule::ShardRef,
                    format!("{name:?}: shard {s:?} not resolved"),
                )
            })?,
        };
        let at = Location {
            store,
            offset: blob.offset,
            len: blob.length,
        };
        let payload = match &blob.encoding {
            None => Payload::At(at),
            Some(encoding) => Payload::Encoded {
                at,
                encoding: encoding.clone(),
                decoded_len: blob.decoded_size(),
            },
        };
        catalog.insert(
            name.clone(),
            Entry {
                shape: obj.shape.clone(),
                term: obj.term.clone(),
                layout: obj.layout.clone(),
                attributes: obj.attributes.clone(),
                payload,
                digest: blob.digest.clone(),
                blocks: blob.blocks.clone(),
            },
        );
    }
    Ok(catalog)
}

// =======================================================================
// Source
// =======================================================================

pub struct Source {
    stores: Vec<Store>,
    catalog: Catalog,
    manifest: Option<Manifest>,
    data_shard: bool,
    vocab: Arc<Vocabulary>,
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("stores", &self.stores.len())
            .field("tensors", &self.catalog.len())
            .field("data_shard", &self.data_shard)
            .finish()
    }
}

impl Source {
    /// Opens a `.zt` file, including a sharded model. Shards are found with
    /// the positional convention.
    pub fn open(path: impl AsRef<Path>) -> Result<Source> {
        Options::default().open(path)
    }

    /// Opens several `.zt` files as one name space.
    pub fn open_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
        Options::default().open_all(paths)
    }

    /// Every way of opening that is not one of the two above.
    pub fn options() -> Options {
        Options::default()
    }

    /// Builds a source directly from a projection's stores and catalog.
    pub fn from_parts(stores: Vec<Store>, catalog: Catalog) -> Result<Source> {
        Options::default().from_parts(stores, catalog)
    }

    /// Reads several sources as one name space. Names must not collide.
    pub fn merge(sources: Vec<Source>) -> Result<Source> {
        let vocab = sources
            .first()
            .map(|s| s.vocab.clone())
            .unwrap_or_else(Vocabulary::shared);
        let mut stores: Vec<Store> = Vec::new();
        let mut merged = Catalog::new();
        let mut attributes: Option<Value> = None;

        for source in sources {
            let base = stores.len() as u32;
            let Source {
                stores: mut part_stores,
                mut catalog,
                ..
            } = source;
            catalog.rebase(|id| StoreId(id.0 + base));
            if attributes.is_none() {
                attributes = catalog.attributes().cloned();
            }
            stores.append(&mut part_stores);
            for (name, entry) in catalog.into_iter_sorted() {
                if let Some(previous) = merged.get(&name) {
                    let here = stores[entry.store().0 as usize].path();
                    let there = stores[previous.store().0 as usize].path();
                    return Err(Error::reject(
                        Rule::NameCollision,
                        format!(
                            "tensor {name:?} is in both {} and {}",
                            there.display(),
                            here.display()
                        ),
                    ));
                }
                merged.insert(name, entry);
            }
        }
        merged.set_attributes(attributes);
        Ok(Source {
            stores,
            catalog: merged,
            manifest: None,
            data_shard: false,
            vocab,
        })
    }

    pub fn len(&self) -> usize {
        self.catalog.len()
    }

    pub fn is_empty(&self) -> bool {
        self.catalog.is_empty()
    }

    /// Tensor names, sorted, across every file of this source.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.catalog.names()
    }

    pub fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> {
        self.catalog.iter().map(|(name, entry)| Tensor {
            src: self,
            name,
            entry,
        })
    }

    /// One tensor by name, or [`Error::NotFound`].
    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        self.get(name)
            .ok_or_else(|| Error::NotFound(format!("tensor {name:?}")))
    }

    /// One tensor by name, for a caller to whom absence is not an error.
    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let (name, entry) = self.catalog.get_key_value(name)?;
        Some(Tensor {
            src: self,
            name,
            entry,
        })
    }

    /// File-level attributes.
    pub fn attributes(&self) -> Option<&Value> {
        self.catalog.attributes()
    }

    /// Who is claiming that this source's tensors are where it says they are.
    pub fn provenance(&self) -> Provenance<'_> {
        match (&self.manifest, self.data_shard) {
            (Some(manifest), _) => Provenance::Root(manifest),
            (None, true) => Provenance::DataShard,
            (None, false) => Provenance::Projection,
        }
    }

    /// Every file this source reads from: `stores()[k]` is `StoreId(k)`.
    pub fn stores(&self) -> &[Store] {
        &self.stores
    }

    /// The file an address names.
    ///
    /// # Panics
    ///
    /// If `id` did not come from this `Source`.
    pub fn store(&self, id: StoreId) -> &Store {
        &self.stores[id.0 as usize]
    }

    /// Deep shard verification: the whole-file digest of every shard against
    /// the root's shard table. Only a `.zt` root has one.
    pub fn verify_shards(&self) -> Result<()> {
        let Some(manifest) = &self.manifest else {
            return Ok(());
        };
        if self.stores.len() != manifest.shards.len() + 1 {
            return Err(Error::reject(
                Rule::ShardIdentity,
                format!(
                    "{} shards resolved for a table of {}",
                    self.stores.len().saturating_sub(1),
                    manifest.shards.len()
                ),
            ));
        }
        for (position, (name, shard)) in manifest.shards.iter().enumerate() {
            let store = &self.stores[position + 1];
            if hash_store(store, shard.digest.algorithm()?)? != shard.digest {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!("shard {name:?}: digest mismatch"),
                ));
            }
        }
        Ok(())
    }
}

// =======================================================================
// handles
// =======================================================================

/// One named tensor of a [`Source`]. Holding one has read nothing.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    src: &'a Source,
    name: &'a str,
    entry: &'a Entry,
}

impl<'a> Tensor<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn shape(&self) -> &'a [u64] {
        &self.entry.shape
    }

    /// The type. `None` only under a layout that defines the values itself.
    pub fn term(&self) -> Option<&'a Term> {
        self.entry.term.as_ref()
    }

    /// The named layout, or `None` for the canonical layout.
    pub fn layout(&self) -> Option<&'a str> {
        self.entry.layout.as_deref()
    }

    pub fn attributes(&self) -> Option<&'a Value> {
        self.entry.attributes.as_ref()
    }

    pub fn num_elements(&self) -> Result<u64> {
        self.entry.num_elements()
    }

    /// The planes inside this tensor's bytes, under the canonical layout.
    pub fn planes(&self) -> Result<Vec<Plane>> {
        self.entry.planes()
    }

    pub fn digest(&self) -> Option<&'a Digest> {
        self.entry.digest.as_ref()
    }

    pub fn blocks(&self) -> Option<&'a Blocks> {
        self.entry.blocks.as_ref()
    }

    /// Decoded byte size.
    pub fn nbytes(&self) -> u64 {
        self.entry.payload.decoded_len()
    }

    pub fn payload(&self) -> &'a Payload {
        &self.entry.payload
    }

    /// The file these bytes live in.
    pub fn store(&self) -> &'a Store {
        self.src.store(self.entry.payload.store())
    }

    pub fn entry(&self) -> &'a Entry {
        self.entry
    }

    fn addressable(&self) -> Option<Location> {
        self.entry.payload.location()
    }

    fn mappable(&self) -> Option<Location> {
        let at = self.addressable()?;
        self.src.store(at.store).is_mapped().then_some(at)
    }

    fn evictable(&self) -> Option<Location> {
        if !cfg!(unix) {
            return None;
        }
        let at = self.mappable()?;
        self.src
            .store(at.store)
            .page_exclusive(at.offset, at.len)
            .then_some(at)
    }

    /// What can be done with these bytes.
    pub fn caps(&self) -> Caps {
        Caps {
            map: self.mappable().is_some(),
            locate: self.addressable().is_some(),
            evict: self.evictable().is_some(),
            verify: self.entry.digest.is_some(),
            alignment: self.addressable().map(|at| at.alignment()).unwrap_or(1),
        }
    }

    /// The address of the decoded bytes: exactly this range of this file.
    pub fn locate(&self) -> Result<Location> {
        self.addressable().ok_or_else(|| {
            Error::Unsupported(format!(
                "{:?} has no address; its bytes are {}",
                self.name,
                self.shape_of_payload()
            ))
        })
    }

    /// Borrowed bytes. Errors rather than copying.
    pub fn map(&self) -> Result<&'a [u8]> {
        let Some(at) = self.mappable() else {
            let detail = if self.addressable().is_some() {
                "its file was opened without mapping".to_string()
            } else {
                format!("its bytes are {}", self.shape_of_payload())
            };
            return Err(Error::Unsupported(format!(
                "{:?}: no zero-copy view; {detail}",
                self.name
            )));
        };
        Ok(self
            .src
            .store(at.store)
            .slice(at.offset, at.len)?
            .expect("mappable stores are mapped"))
    }

    /// Decoded bytes, the best way this source can produce them.
    pub fn bytes(&self) -> Result<Cow<'a, [u8]>> {
        match &self.entry.payload {
            Payload::At(at) => {
                let store = self.src.store(at.store);
                match store.slice(at.offset, at.len)? {
                    Some(slice) => Ok(Cow::Borrowed(slice)),
                    None => Ok(Cow::Owned(store.read(at.offset, at.len)?)),
                }
            }
            Payload::Encoded {
                at,
                encoding,
                decoded_len,
            } => {
                let stored = self.src.store(at.store).read(at.offset, at.len)?;
                let profile = self.src.vocab.encoding(encoding).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{:?}: encoding profile {encoding:?} is not registered",
                        self.name
                    ))
                })?;
                Ok(Cow::Owned(profile.decode(&stored, *decoded_len)?))
            }
            Payload::Opaque {
                store,
                key,
                decoded_len,
            } => {
                let store = self.src.store(*store);
                let decoder = store.decoder().ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{:?}: opaque payload with no decoder attached",
                        self.name
                    ))
                })?;
                Ok(Cow::Owned(decoder.decode(*key, *decoded_len)?))
            }
        }
    }

    /// Checks this tensor's digest (if it has one) and, under the canonical
    /// layout, the content rules of its leaves.
    ///
    /// A mismatch comes back as `Err(Reject { rule: Digest, .. })`.
    /// `Ok(NoDigest)` means there was no digest to check.
    pub fn verify(&self) -> Result<Verified> {
        let canonical = self.entry.layout.is_none();
        if self.entry.digest.is_none() && !canonical {
            return Ok(Verified::NoDigest);
        }
        let bytes = self.bytes()?;
        if let (true, Some(term)) = (canonical, &self.entry.term) {
            term.check_bytes(&self.entry.shape, &bytes)
                .map_err(|e| e.at(self.name))?;
        }
        match &self.entry.digest {
            None => Ok(Verified::NoDigest),
            Some(digest) => {
                if !digest.matches(&bytes)? {
                    return Err(Error::reject(
                        Rule::Digest,
                        format!("digest mismatch for {:?}", self.name),
                    ));
                }
                Ok(Verified::Digest)
            }
        }
    }

    /// Checks block `which` of the decoded bytes against its block digest.
    /// What a streaming consumer calls per window instead of [`verify`](Self::verify).
    pub fn verify_block(&self, which: u64, bytes: &[u8]) -> Result<()> {
        let (Some(blocks), Some(digest)) = (&self.entry.blocks, &self.entry.digest) else {
            return Err(Error::Unsupported(format!(
                "{:?} carries no block digests",
                self.name
            )));
        };
        let expected = blocks
            .digests
            .get(which as usize)
            .ok_or_else(|| Error::NotFound(format!("{:?}: block {which}", self.name)))?;
        let span = blocks.span(which, self.nbytes()).unwrap_or(0..0);
        if bytes.len() as u64 != span.end - span.start {
            return Err(Error::InvalidInput(format!(
                "{:?}: block {which} is {} bytes, {} given",
                self.name,
                span.end - span.start,
                bytes.len()
            )));
        }
        if digest.algorithm()?.digest(bytes).value != *expected {
            return Err(Error::reject(
                Rule::Digest,
                format!("digest mismatch for {:?} block {which}", self.name),
            ));
        }
        Ok(())
    }

    /// Hints the OS to prefetch these pages.
    pub fn prefetch(&self) -> Result<()> {
        let Some(at) = self.mappable() else {
            return Ok(());
        };
        self.src.store(at.store).prefetch(at.offset, at.len)
    }

    /// Drops these pages from the page cache. Requires page exclusivity.
    pub fn evict(&self) -> Result<()> {
        let at = self.evictable().ok_or_else(|| {
            Error::Unsupported(format!(
                "{:?}: not evictable; {}",
                self.name,
                if !cfg!(unix) {
                    "dropping page cache is a unix facility"
                } else if self.mappable().is_some() {
                    "it shares an OS page with another blob"
                } else {
                    "its bytes are not a mapped range"
                }
            ))
        })?;
        self.src.store(at.store).evict(at.offset, at.len)
    }

    fn shape_of_payload(&self) -> &'static str {
        match &self.entry.payload {
            Payload::At(_) => "a raw range",
            Payload::Encoded { .. } => "stored under an encoding profile",
            Payload::Opaque { .. } => "produced by the format's own reader",
        }
    }
}

/// The identity of a `.zt` container: its size and whole-file digest. This is
/// exactly what [`Writer::add_shard`](crate::Writer::add_shard) records.
pub fn shard_identity(path: impl AsRef<Path>, algo: DigestAlgorithm) -> Result<Shard> {
    let store = Store::index(path.as_ref(), "zt")?;
    validate::store(&store, &Vocabulary::shared())?;
    Ok(Shard {
        size: store.len(),
        digest: hash_store(&store, algo)?,
    })
}

/// The whole-file digest, a megabyte at a time.
fn hash_store(store: &Store, algo: DigestAlgorithm) -> Result<Digest> {
    let mut hasher = Hasher::new(algo);
    let mut at = 0u64;
    while at < store.len() {
        let n = (store.len() - at).min(1 << 20);
        hasher.update(&store.read(at, n)?);
        at += n;
    }
    Ok(hasher.finish())
}

/// Reads and validates one container's manifest, resolving nothing. `None`
/// is a data shard (§7.2).
pub fn manifest_of(path: impl AsRef<Path>) -> Result<Option<Manifest>> {
    let store = Store::index(path.as_ref(), "zt")?;
    let parsed = validate::store(&store, &Vocabulary::shared())?;
    Ok(parsed.manifest.map(|(manifest, _)| manifest))
}

/// Checks a file against canonical form (spec §6.4) and returns every rule it
/// breaks, in rule order. An empty list means the file is canonical.
pub fn canonical_violations(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let store = Store::index(path.as_ref(), "zt")?;
    let Some((manifest, placement)) = validate::store(&store, &Vocabulary::shared())?.manifest
    else {
        return Ok(vec![
            "rule 1: a data shard carries no manifest, so it is not a canonical model".into(),
        ]);
    };
    Ok(validate::canonical_violations(&manifest, &placement))
}
