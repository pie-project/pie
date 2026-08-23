//! A cached HF safetensors snapshot, header-parsed and addressed.
//!
//! LIFTED OUT OF `bin/baker_load.rs`, unchanged, because it now has two
//! callers. `baker_load` joins a plan's `params` against what this reader
//! feeds [`crate::produce::produce`]; `baker-smoke` uploads the same
//! production to a device and fires it. Two readers that had to agree
//! byte-for-byte would be the bug this tree keeps writing comments about --
//! the whole point of the join is that the bytes are the RIGHT bytes, and a
//! second, drifting `canon()` would retire that proof silently.
//!
//! The naming convention lives here rather than in `produce` on purpose:
//! the interpreter reads its checkpoint through a `&dyn Fn(&str) ->
//! Option<HostTensor>` so that safetensors, GGUF and zt all reduce to one
//! verb set. Whoever opens the file owns the spelling.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use crate::produce::{Dtype, HostTensor};

/// One tensor's home in the snapshot.
struct Part {
    file: usize,
    begin: u64,
    end: u64,
    shape: Vec<u64>,
    dtype: Dtype,
}

/// A cached HF snapshot, header-parsed and addressed. Nothing is read
/// until a production row asks for it, which is what keeps a harness's
/// peak memory the produced tensors rather than the checkpoint plus them.
pub struct Snapshot {
    /// The snapshot directory that was opened.
    pub dir: PathBuf,
    files: Vec<File>,
    index: BTreeMap<String, Part>,
    taken: std::cell::RefCell<BTreeSet<String>>,
}

impl Snapshot {
    /// Open the newest snapshot under `~/.cache/huggingface/hub/<cache_dir>`
    /// that carries either a single `model.safetensors` or a shard index.
    #[must_use]
    pub fn open(cache_dir: &str) -> Option<Snapshot> {
        let home = std::env::var_os("HOME")?;
        let snaps =
            PathBuf::from(home).join(format!(".cache/huggingface/hub/{cache_dir}/snapshots"));
        let dir = std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(Result::ok)
            .find_map(|e| {
                let d = e.path();
                (d.join("model.safetensors").is_file()
                    || d.join("model.safetensors.index.json").is_file())
                .then_some(d)
            })?;
        Snapshot::at(dir)
    }

    /// The same reader over an explicit snapshot directory.
    #[must_use]
    pub fn at(dir: PathBuf) -> Option<Snapshot> {
        let paths: Vec<PathBuf> = if dir.join("model.safetensors.index.json").is_file() {
            let idx: serde_json::Value = serde_json::from_slice(
                &std::fs::read(dir.join("model.safetensors.index.json")).ok()?,
            )
            .ok()?;
            let mut shards: Vec<String> = idx["weight_map"]
                .as_object()?
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            shards.into_iter().map(|f| dir.join(f)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };

        let mut files = Vec::new();
        let mut index: BTreeMap<String, Part> = BTreeMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let f = File::open(p).ok()?;
            let mut len = [0u8; 8];
            f.read_exact_at(&mut len, 0).ok()?;
            let header_len = u64::from_le_bytes(len);
            let mut header = vec![0u8; header_len as usize];
            f.read_exact_at(&mut header, 8).ok()?;
            let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
            let payload = 8 + header_len;
            for (name, meta) in header.as_object()? {
                if name == "__metadata__" {
                    continue;
                }
                let Some(dtype) = meta["dtype"].as_str().and_then(Dtype::parse) else {
                    continue;
                };
                let offs = meta["data_offsets"].as_array()?;
                let part = Part {
                    file: fi,
                    begin: payload + offs[0].as_u64()?,
                    end: payload + offs[1].as_u64()?,
                    shape: meta["shape"]
                        .as_array()?
                        .iter()
                        .filter_map(serde_json::Value::as_u64)
                        .collect(),
                    dtype,
                };
                let canon = canon(name);
                if let Some(old) = index.insert(canon.clone(), part) {
                    panic!(
                        "two checkpoint tensors normalize to `{canon}` (one in shard {}, one in {fi})",
                        old.file
                    );
                }
            }
            files.push(f);
        }
        Some(Snapshot {
            dir,
            files,
            index,
            taken: std::cell::RefCell::new(BTreeSet::new()),
        })
    }

    /// How many tensors the header indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the header indexed nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// How many shard files back this snapshot.
    #[must_use]
    pub fn shards(&self) -> usize {
        self.files.len()
    }

    /// Read one tensor by its canonical name, recording that it was taken.
    #[must_use]
    pub fn read(&self, name: &str) -> Option<HostTensor> {
        let p = self.index.get(name)?;
        self.taken.borrow_mut().insert(name.to_string());
        let mut bytes = vec![0u8; (p.end - p.begin) as usize];
        self.files[p.file].read_exact_at(&mut bytes, p.begin).ok()?;
        Some(HostTensor::new(p.shape.iter().copied(), p.dtype, bytes))
    }

    /// How many distinct tensors [`Snapshot::read`] has answered.
    #[must_use]
    pub fn taken(&self) -> usize {
        self.taken.borrow().len()
    }

    /// The tensors no production row has asked for, in index order.
    #[must_use]
    pub fn untaken(&self) -> Vec<&str> {
        let taken = self.taken.borrow();
        self.index
            .keys()
            .filter(|k| !taken.contains(*k))
            .map(String::as_str)
            .collect()
    }
}

/// HF's own spelling to the one an import table uses.
///
/// The tables say `layer.3.self_attn.q_proj`; a `Qwen3_5ForConditionalGeneration`
/// checkpoint says `model.language_model.layers.3.self_attn.q_proj.weight`.
/// Three rules cover every family read here, and they are the checkpoint
/// format's conventions rather than any model's: the wrapper prefixes a
/// multimodal release puts the text tower under, `layers` -> `layer`, and
/// the `.weight` leaf a `nn.Linear` adds. A tensor that is not a `.weight`
/// -- gpt-oss's `.bias` rows, gemma's `layer_scalar` -- keeps its leaf,
/// which is exactly what the tables spell.
#[must_use]
pub fn canon(hf: &str) -> String {
    let mut s = hf;
    for p in ["model.language_model.", "model.text_model.", "model."] {
        if let Some(rest) = s.strip_prefix(p) {
            s = rest;
            break;
        }
    }
    if let Some(rest) = s.strip_prefix("language_model.") {
        s = rest;
    }
    let s = s.strip_suffix(".weight").unwrap_or(s);
    match s.strip_prefix("layers.") {
        Some(rest) => format!("layer.{rest}"),
        None => s.to_string(),
    }
}
