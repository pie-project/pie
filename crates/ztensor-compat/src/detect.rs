use std::fs::File;
use std::io::Read;
use std::path::Path;

use ztensor::{Error, Result, Source, Store, Vocabulary};

pub const FORMATS: &[&str] = &["gguf", "hdf5", "npz", "onnx", "pt", "safetensors", "zt"];

pub fn detect(path: impl AsRef<Path>) -> Result<&'static str> {
    let path = path.as_ref();
    if let Some(len) = ztensor::memfs::len(path) {
        let mut head = [0u8; 9];
        let take = usize::try_from(len.min(9)).unwrap_or(9);
        if ztensor::memfs::read(path, 0, &mut head[..take]).is_none() {
            return Err(Error::Unsupported(format!(
                "{}: the mounted bytes could not be read",
                path.display()
            )));
        }
        return detect_head(path, &head[..take], || {
            Err(Error::Unsupported(format!(
                "{}: zip-container formats (.pt/.npz) are read from a file, not a memory mount",
                path.display()
            )))
        });
    }
    let mut file = File::open(path)?;
    let mut head = [0u8; 9];
    let mut n = 0;
    while n < head.len() {
        match file.read(&mut head[n..])? {
            0 => break,
            got => n += got,
        }
    }
    detect_head(path, &head[..n], || zip_is_pt(path))
}

#[cfg(any(feature = "pickle", feature = "npz"))]
fn zip_is_pt(path: &Path) -> Result<bool> {
    Ok(zip::ZipArchive::new(File::open(path)?)
        .ok()
        .map(|z| z.file_names().any(|n| n.ends_with("data.pkl")))
        .unwrap_or(false))
}

#[cfg(not(any(feature = "pickle", feature = "npz")))]
fn zip_is_pt(_path: &Path) -> Result<bool> {
    Ok(false)
}

fn detect_head(
    path: &Path,
    head: &[u8],
    zip_is_pt: impl FnOnce() -> Result<bool>,
) -> Result<&'static str> {
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
        return Ok(if zip_is_pt()? { "pt" } else { "npz" });
        #[cfg(not(any(feature = "pickle", feature = "npz")))]
        {
            let _ = zip_is_pt;
            return Err(Error::Unsupported(
                "zip-container formats (.pt/.npz) are not compiled in".into(),
            ));
        }
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

pub fn options() -> Open {
    Open::default()
}

impl Open {
    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(vocab.clone());
        self
    }

    pub fn map(mut self, map: bool) -> Self {
        self.map = map;
        self
    }

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

    pub fn open_all(self, paths: &[impl AsRef<Path>]) -> Result<Source> {
        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            sources.push(self.clone().open(path.as_ref())?);
        }
        Source::merge(sources)
    }
}

pub fn open(path: impl AsRef<Path>) -> Result<Source> {
    options().open(path)
}

pub fn index(path: impl AsRef<Path>) -> Result<Source> {
    options().map(false).open(path)
}

pub fn open_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
    options().open_all(paths)
}

pub fn index_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
    options().map(false).open_all(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn detect_every_case() {
        a_mounted_artifact_is_detected_and_indexed_in_memory();
    }

    fn a_mounted_artifact_is_detected_and_indexed_in_memory() {
        let dir =
            std::env::temp_dir().join(format!("ztensor_compat_detect_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("model.zt");
        let mut writer = ztensor::Writer::create(&real).unwrap();
        writer
            .add("w", vec![4], ztensor::Leaf::U8, &[1u8, 2, 3, 4])
            .unwrap();
        writer.finish().unwrap();
        let bytes = std::fs::read(&real).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        let fake = Path::new("/nowhere/on/this/machine/compat.zt");
        assert!(detect(fake).is_err());
        ztensor::memfs::mount(fake, Arc::from(bytes.as_slice()));
        assert_eq!(detect(fake).unwrap(), "zt");
        let source = index(fake).unwrap();
        assert_eq!(source.tensor("w").unwrap().map().unwrap(), &[1, 2, 3, 4]);
        assert!(source.store(ztensor::StoreId(0)).is_memory());

        let zipped = Path::new("/nowhere/on/this/machine/archive.npz");
        ztensor::memfs::mount(zipped, Arc::from(&b"PK\x03\x04rest"[..]));
        let err = detect(zipped).unwrap_err().to_string();
        assert!(
            err.contains("mount") || err.contains("not compiled in"),
            "{err}"
        );

        ztensor::memfs::unmount(fake);
        ztensor::memfs::unmount(zipped);
    }
}
