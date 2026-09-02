//! Opening a tensor file of any supported format.
//!
//! Detection is by magic bytes wherever the format has them; ONNX (protobuf,
//! magic-less) falls back to the file extension. Formats whose feature is
//! disabled are reported as such, never misdetected as something else.
//!
//! What comes back is an ordinary [`Source`]. There is no per-format type to
//! learn: a projected safetensors file and a canonical `.zt` file answer the
//! same questions, and differ only in what they can honestly say yes to.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ztensor::{Error, Result, Source, Store, Vocabulary};

/// Every label [`detect`] can return.
///
/// Enumerable because a consumer usually has a table keyed by these, such as a
/// display name or an enum of its own. A table that is missing a row says
/// nothing when the list grows. Checking against this turns that silence into
/// a failing test.
pub const FORMATS: &[&str] = &["gguf", "hdf5", "npz", "onnx", "pt", "safetensors", "zt"];

/// Sniffs the format of a tensor file. Returns one of [`FORMATS`].
pub fn detect(path: impl AsRef<Path>) -> Result<&'static str> {
    let path = path.as_ref();
    // A single read() may return fewer bytes than asked for; fill the buffer
    // so a short read cannot cause a mis-detection.
    let mut file = File::open(path)?;
    let mut head = [0u8; 9];
    let mut n = 0;
    while n < head.len() {
        match file.read(&mut head[n..])? {
            0 => break,
            got => n += got,
        }
    }
    let head = &head[..n];

    if head.len() >= 8 && head[..8] == ztensor::format::MAGIC {
        return Ok("zt");
    }
    if head.starts_with(b"GGUF") {
        return Ok("gguf");
    }
    if head.len() >= 8 && &head[..8] == b"\x89HDF\r\n\x1a\n" {
        return Ok("hdf5");
    }
    if head.starts_with(b"PK\x03\x04") {
        #[cfg(any(feature = "pickle", feature = "npz"))]
        {
            let is_pt = zip::ZipArchive::new(File::open(path)?)
                .ok()
                .map(|z| z.file_names().any(|n| n.ends_with("data.pkl")))
                .unwrap_or(false);
            return Ok(if is_pt { "pt" } else { "npz" });
        }
        #[cfg(not(any(feature = "pickle", feature = "npz")))]
        return Err(Error::Unsupported(
            "zip-container formats (.pt/.npz) are not compiled in".into(),
        ));
    }
    if head.len() >= 9 && head[8] == b'{' {
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        if header_len > 0 && header_len < (100 << 20) {
            return Ok("safetensors");
        }
    }
    if path.extension().is_some_and(|e| e == "onnx") {
        return Ok("onnx");
    }
    Err(Error::Unsupported(format!(
        "cannot detect the format of {}",
        path.display()
    )))
}

/// How to open. Mirrors [`ztensor::read::Options`], minus shard resolution:
/// no foreign format has a shard table.
#[derive(Clone)]
pub struct Open {
    vocab: Option<Vocabulary>,
    map: bool,
}

impl Default for Open {
    fn default() -> Self {
        Self {
            vocab: None,
            map: true,
        }
    }
}

/// Opening options: a vocabulary to read with, and whether to map.
pub fn options() -> Open {
    Open::default()
}

impl Open {
    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(vocab.clone());
        self
    }

    /// Map the files (the default). With `false`, files are opened but not
    /// mapped: metadata and addresses are available, borrowed reads are not.
    pub fn map(mut self, map: bool) -> Self {
        self.map = map;
        self
    }

    /// Opens one file of any supported format.
    pub fn open(self, path: impl AsRef<Path>) -> Result<Source> {
        let path = path.as_ref();
        let format = detect(path)?;

        if format == "zt" {
            let mut opts = ztensor::Source::options().map(self.map);
            if let Some(vocab) = &self.vocab {
                opts = opts.vocabulary(vocab);
            }
            return opts.open(path);
        }

        let store = if self.map {
            Store::map(path, format)?
        } else {
            Store::index(path, format)?
        };

        #[allow(unused_variables)]
        let projection = match format {
            #[cfg(feature = "safetensors")]
            "safetensors" => crate::safetensors::project(&store)?,
            #[cfg(feature = "gguf")]
            "gguf" => crate::gguf::project(&store)?,
            #[cfg(feature = "npz")]
            "npz" => crate::npz::project(&store)?,
            #[cfg(feature = "pickle")]
            "pt" => crate::pt::project(&store)?,
            #[cfg(feature = "hdf5")]
            "hdf5" => crate::hdf5::project(&store)?,
            #[cfg(feature = "onnx")]
            "onnx" => crate::onnx::project(&store)?,
            other => {
                return Err(Error::Unsupported(format!(
                    "{other} support is not compiled in (enable the matching \
                     ztensor-compat feature)"
                )))
            }
        };
        projection.into_source(store, self.vocab.as_ref())
    }

    /// Opens several files as one name space.
    ///
    /// What a sharded snapshot is: `model-00001-of-00003.safetensors` and its
    /// siblings are each a whole file that describes itself, and the index
    /// beside them is a naming convention outside the format. So this is a
    /// list of paths and nothing more. The caller decides which files belong
    /// together, because the files themselves never said.
    ///
    /// The set may mix formats. Nothing here requires them to match: what
    /// makes it a model is that the names do not collide, which the merge
    /// checks.
    pub fn open_all(self, paths: &[impl AsRef<Path>]) -> Result<Source> {
        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            sources.push(self.clone().open(path.as_ref())?);
        }
        Source::merge(sources)
    }
}

/// Opens a tensor file of any supported format.
pub fn open(path: impl AsRef<Path>) -> Result<Source> {
    options().open(path)
}

/// Opens without mapping: metadata and addresses only.
pub fn index(path: impl AsRef<Path>) -> Result<Source> {
    options().map(false).open(path)
}

/// Opens several files as one name space. See [`Open::open_all`].
pub fn open_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
    options().open_all(paths)
}

/// Indexes several files as one name space, mapping none of them.
pub fn index_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
    options().map(false).open_all(paths)
}
