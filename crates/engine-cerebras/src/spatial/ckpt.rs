//! The weights the wafer model places, by the model's (poem) parameter names
//! (`embed`, `head`, `final_norm`, `layers.{l}.q`, `.k`, `.v`, `.o`,
//! `.q_norm`, `.k_norm`, `.attn_norm`, `.mlp_norm`, `.gate`, `.up`, `.down`).
//!
//! Serving reads them through PIE's checkpoint path ([`Ckpt::landed`]): the
//! load's contract run over its checkpoint, every plane landed on the host
//! as the trace declares it (packed `qkv` and `gate_up` planes are read in
//! their parts). A Hugging Face directory ([`Ckpt::open`]: `config.json` and
//! `model.safetensors`, its shards by `model.safetensors.index.json`, or a
//! `weights.npz` of bf16 bits) is read by the transformers names the poem
//! maps (`TRANSFORMERS`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    Bf16,
    F16,
    F32,
    /// Bytes as stored (gpt-oss's MXFP4 codes and E8M0 scales): read raw
    /// by [`Ckpt::hf_u8`]; widened, each byte's value.
    U8,
}

impl Dtype {
    fn bytes(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    /// Which of the checkpoint's files (shards) holds it.
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    offset: u64,
}

impl Entry {
    fn elems(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A tensor widened to `f32`, row-major.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Refuses a tensor `name` whose shape is not the program's `shape`.
    pub fn expect_shape(&self, name: &str, shape: &[usize]) -> R<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(format!(
                "{name} is {:?}; the program wants {shape:?}",
                self.shape
            ))
        }
    }
}

pub struct Ckpt {
    inner: Inner,
    config: serde_json::Value,
}

enum Inner {
    /// A Hugging Face directory, read lazily by transformers name: each
    /// file one path, or its pieces in order (`<file>.partNN`, as the CS-3
    /// session stages files under 2 GiB).
    Files {
        paths: Vec<Vec<PathBuf>>,
        entries: HashMap<String, Entry>,
    },
    /// A test checkpoint: every named tensor of its shape, its values a
    /// function of its name and index.
    #[cfg(test)]
    Synthetic { shapes: HashMap<String, Vec<usize>> },
    /// The planes PIE's checkpoint executor landed, by plan param name.
    Landed {
        planes: HashMap<String, Plane>,
        parts: Parts,
    },
}

/// One landed plane: its trace dtype, shape and bytes.
struct Plane {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// The row spans of the packed planes' parts: `qkv` = q | k | v, `gate_up`
/// = gate | up.
#[derive(Debug, Clone, Copy)]
struct Parts {
    q: usize,
    kv: usize,
    inter: usize,
    /// Qwen3.5's packed planes: `gdn.in_qkvz` = q, k, v (`qkv` rows) | z
    /// (`vw`), `gdn.in_ba` = b | a (`vh` rows each), `attn.qg_proj_k_proj_v_proj`
    /// = q and gate (`qg`) | k | v (`kv`); 0 on a model without them.
    qkv: usize,
    vw: usize,
    vh: usize,
    qg: usize,
}

/// How the poem names Qwen 3's transformers tensors (models/qwen3_dense.poem's
/// `TRANSFORMERS`).
const TRANSFORMERS: &[(&str, &str)] = &[
    ("embed", "model.embed_tokens.weight"),
    ("final_norm", "model.norm.weight"),
    ("head", "lm_head.weight"),
    ("attn_norm", "input_layernorm.weight"),
    ("q", "self_attn.q_proj.weight"),
    ("k", "self_attn.k_proj.weight"),
    ("v", "self_attn.v_proj.weight"),
    ("o", "self_attn.o_proj.weight"),
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
    ("mlp_norm", "post_attention_layernorm.weight"),
    ("gate", "mlp.gate_proj.weight"),
    ("up", "mlp.up_proj.weight"),
    ("down", "mlp.down_proj.weight"),
];

/// How the poem names Qwen3.5's transformers tensors past `layers.{l}.`
/// (models/qwen_3.poem's `safetensors`), its packed planes by their parts.
const QWEN35: &[(&str, &str)] = &[
    ("mixer_norm", "input_layernorm.weight"),
    ("mlp_norm", "post_attention_layernorm.weight"),
    ("mlp.gate", "mlp.gate_proj.weight"),
    ("mlp.up", "mlp.up_proj.weight"),
    ("mlp.down", "mlp.down_proj.weight"),
    ("gdn.in_qkv", "linear_attn.in_proj_qkv.weight"),
    ("gdn.in_z", "linear_attn.in_proj_z.weight"),
    ("gdn.in_b", "linear_attn.in_proj_b.weight"),
    ("gdn.in_a", "linear_attn.in_proj_a.weight"),
    ("gdn.conv", "linear_attn.conv1d.weight"),
    ("gdn.dt_bias", "linear_attn.dt_bias"),
    ("gdn.a_log", "linear_attn.A_log"),
    ("gdn.norm", "linear_attn.norm.weight"),
    ("gdn.out_proj", "linear_attn.out_proj.weight"),
    ("attn.qg_proj", "self_attn.q_proj.weight"),
    ("attn.k_proj", "self_attn.k_proj.weight"),
    ("attn.v_proj", "self_attn.v_proj.weight"),
    ("attn.o_proj", "self_attn.o_proj.weight"),
    ("attn.q_norm", "self_attn.q_norm.weight"),
    ("attn.k_norm", "self_attn.k_norm.weight"),
];

/// Where a checkpoint keeps its language model: Qwen3.5's under
/// `model.language_model.`, a text-only one's under `model.`, a stripped
/// `weights.npz`'s at the top.
const TRUNKS: &[&str] = &["model.language_model.", "model.", ""];

/// A poem parameter name's transformers tensor, as Qwen3.5's checkpoints
/// spell it: one name for each trunk.
fn qwen35_names(name: &str) -> Vec<String> {
    let rest = match name {
        "embed" => Some("embed_tokens.weight".to_string()),
        "final_norm" => Some("norm.weight".to_string()),
        _ => name.strip_prefix("layers.").and_then(|rest| {
            let (l, part) = rest.split_once('.')?;
            l.parse::<usize>().ok()?;
            let hf = QWEN35.iter().find(|e| e.0 == part)?.1;
            Some(format!("layers.{l}.{hf}"))
        }),
    };
    rest.map_or_else(Vec::new, |r| {
        TRUNKS.iter().map(|t| format!("{t}{r}")).collect()
    })
}

/// A poem parameter name's transformers tensor.
fn transformers_name(name: &str) -> Option<String> {
    let find = |k: &str| TRANSFORMERS.iter().find(|e| e.0 == k).map(|e| e.1);
    match name.strip_prefix("layers.") {
        Some(rest) => {
            let (l, part) = rest.split_once('.')?;
            l.parse::<usize>().ok()?;
            Some(format!("model.layers.{l}.{}", find(part)?))
        }
        None => find(name).map(String::from),
    }
}

type R<T> = std::result::Result<T, String>;

/// A tensor as stored: its dtype, shape and bytes.
type Raw = (Dtype, Vec<usize>, Vec<u8>);

fn u16_at(b: &[u8], at: usize) -> u64 {
    u64::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> u64 {
    u64::from(u32::from_le_bytes(
        b[at..at + 4].try_into().expect("4 bytes"),
    ))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn read_at(file: &mut File, offset: u64, len: usize) -> R<Vec<u8>> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|e| format!("reading {len} bytes at {offset}: {e}"))?;
    Ok(buf)
}

/// A tied head: as declared, else an `lm_head.weight` (if any) the same as
/// the embedding.
fn tied(flag: Option<bool>, lm_head_same: Option<bool>) -> bool {
    flag.unwrap_or(lm_head_same != Some(false))
}

impl Ckpt {
    pub fn open(dir: &Path) -> R<Ckpt> {
        let config_path = dir.join("config.json");
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| format!("{}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
        let index = dir.join("model.safetensors.index.json");
        let (paths, entries) = if dir.join("model.safetensors").exists() {
            let path = dir.join("model.safetensors");
            let entries = safetensors(std::slice::from_ref(&path), 0)?;
            (vec![vec![path]], entries)
        } else if index.exists() {
            let map: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&index)
                    .map_err(|e| format!("{}: {e}", index.display()))?,
            )
            .map_err(|e| format!("{}: {e}", index.display()))?;
            let mut files: Vec<&str> = map["weight_map"]
                .as_object()
                .ok_or_else(|| format!("{}: no weight_map", index.display()))?
                .values()
                .filter_map(|v| v.as_str())
                .collect();
            files.sort_unstable();
            files.dedup();
            let mut entries = HashMap::new();
            let paths: Vec<Vec<PathBuf>> = files.iter().map(|f| vec![dir.join(f)]).collect();
            for (i, path) in paths.iter().enumerate() {
                entries.extend(safetensors(path, i)?);
            }
            (paths, entries)
        } else if let Some(paths) = shards(dir)? {
            let mut entries = HashMap::new();
            for (i, pieces) in paths.iter().enumerate() {
                entries.extend(safetensors(pieces, i)?);
            }
            (paths, entries)
        } else {
            let path = dir.join("weights.npz");
            let entries = npz(&path)?;
            (vec![vec![path]], entries)
        };
        Ok(Ckpt {
            inner: Inner::Files { paths, entries },
            config,
        })
    }

    /// The load's checkpoint through PIE's checkpoint path: `contract`
    /// (the runtime's for `trace`) run over `path` (an imported artifact or
    /// a checkpoint directory), every `Checkpoint` param of `trace` landed
    /// on the host as the trace declares it.
    pub fn landed(
        trace: &poem_ir::Trace,
        contract: &checkpoint::contract::ModelContract,
        path: &Path,
    ) -> R<Ckpt> {
        use checkpoint::executor::Execution;
        use checkpoint::plan::{StorageTarget, compile_streaming};
        use checkpoint::types::BackendKind;
        let spec = super::model::ModelSpec::read(trace).map_err(|e| e.to_string())?;
        // The metadata names files under the path: one the executor can
        // open from anywhere.
        let path: &Path =
            &std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (metadata, snapshot) = if path.is_dir() {
            (
                checkpoint::file::read::parse_metadata(path).map_err(|e| e.to_string())?,
                path,
            )
        } else {
            (
                checkpoint::file::zt::parse(path).map_err(|e| e.to_string())?,
                path.parent().unwrap_or(Path::new(".")),
            )
        };
        let target = StorageTarget::for_backend(BackendKind::Cerebras, 0, 1);
        let streaming =
            compile_streaming(&metadata, contract, target).map_err(|e| e.to_string())?;
        let mut sink = Collect {
            trace,
            planes: HashMap::new(),
        };
        Execution::new(&streaming, snapshot)
            .streaming()
            .sink(&mut sink)
            .run()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let gdn = spec.hybrid.as_ref().map(|h| h.gdn);
        let parts = Parts {
            q: spec.q_heads * spec.head_dim,
            kv: spec.kv_heads * spec.head_dim,
            inter: spec.inter,
            qkv: gdn.map_or(0, |g| g.conv_channels()),
            vw: gdn.map_or(0, |g| g.v_heads * g.v_dim),
            vh: gdn.map_or(0, |g| g.v_heads),
            qg: 2 * spec.q_heads * spec.head_dim,
        };
        Ok(Ckpt {
            inner: Inner::Landed {
                planes: sink.planes,
                parts,
            },
            config: serde_json::Value::Null,
        })
    }

    /// A test checkpoint of `shapes` (param name, shape), its values
    /// deterministic: what a host-protocol check places.
    #[cfg(test)]
    pub fn synthetic(shapes: impl IntoIterator<Item = (String, Vec<usize>)>) -> Ckpt {
        Ckpt {
            inner: Inner::Synthetic {
                shapes: shapes.into_iter().collect(),
            },
            config: serde_json::Value::Null,
        }
    }

    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// Whether the logits read the embedding (a tied LM head), which is all
    /// the wafer programs compute: `tie_word_embeddings` says (true ties
    /// whatever `lm_head.weight` holds, as Hugging Face loads it); without
    /// it, an `lm_head.weight` that differs from `embed_tokens` is untied.
    /// A landed checkpoint states it by whether the trace has a `head`.
    pub fn tied_head(&self) -> R<bool> {
        if let Inner::Landed { .. } = self.inner {
            return Ok(!self.has("head"));
        }
        let flag = self.config["tie_word_embeddings"].as_bool();
        let same = if self.has("head") && flag.is_none() {
            let (head, emb) = (self.get("head")?, self.get("embed")?);
            Some(head.shape == emb.shape && head.data == emb.data)
        } else {
            None
        };
        Ok(tied(flag, same))
    }

    /// Whether the checkpoint holds the (poem) param `name`.
    pub fn has(&self, name: &str) -> bool {
        self.raw(name).is_ok_and(|r| r.is_some())
    }

    /// `name`'s dtype, shape and bytes (`None`: no such param).
    fn raw(&self, name: &str) -> R<Option<Raw>> {
        match &self.inner {
            Inner::Files { paths, entries } => {
                let found = transformers_name(name)
                    .and_then(|n| entries.get(&n))
                    .or_else(|| qwen35_names(name).iter().find_map(|n| entries.get(n)));
                let Some(e) = found else {
                    return Ok(None);
                };
                let raw = read_pieces(&paths[e.file], e.offset, e.elems() * e.dtype.bytes())?;
                Ok(Some((e.dtype, e.shape.clone(), raw)))
            }
            #[cfg(test)]
            Inner::Synthetic { shapes } => Ok(shapes.get(name).map(|shape| {
                let n: usize = shape.iter().product();
                let seed = u64::from_le_bytes(
                    blake3::hash(name.as_bytes()).as_bytes()[..8]
                        .try_into()
                        .expect("eight bytes"),
                );
                let bytes = (0..n as u64)
                    .flat_map(|i| {
                        let h = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let h = (h ^ (h >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        (((h >> 40) % 2048) as f32 / 1024.0 - 1.0).to_le_bytes()
                    })
                    .collect();
                (Dtype::F32, shape.clone(), bytes)
            })),
            Inner::Landed { planes, parts } => {
                if let Some(p) = planes.get(name) {
                    return Ok(Some((p.dtype, p.shape.clone(), p.bytes.clone())));
                }
                // A part of a packed plane: rows [lo, hi) of `packed`.
                let (layer, part) = match name.rsplit_once('.') {
                    Some(x) => x,
                    None => return Ok(None),
                };
                let pk = parts;
                let (packed, lo, hi) = match part {
                    "in_qkv" => ("in_qkvz", 0, pk.qkv),
                    "in_z" => ("in_qkvz", pk.qkv, pk.qkv + pk.vw),
                    "in_b" => ("in_ba", 0, pk.vh),
                    "in_a" => ("in_ba", pk.vh, 2 * pk.vh),
                    "qg_proj" => ("qg_proj_k_proj_v_proj", 0, pk.qg),
                    "k_proj" => ("qg_proj_k_proj_v_proj", pk.qg, pk.qg + pk.kv),
                    "v_proj" => ("qg_proj_k_proj_v_proj", pk.qg + pk.kv, pk.qg + 2 * pk.kv),
                    "q" => ("qkv", 0, parts.q),
                    "k" => ("qkv", parts.q, parts.q + parts.kv),
                    "v" => ("qkv", parts.q + parts.kv, parts.q + 2 * parts.kv),
                    "gate" => ("gate_up", 0, parts.inter),
                    "up" => ("gate_up", parts.inter, 2 * parts.inter),
                    _ => return Ok(None),
                };
                let Some(p) = planes.get(&format!("{layer}.{packed}")) else {
                    return Ok(None);
                };
                let rows = p.shape.first().copied().unwrap_or(0);
                if hi > rows {
                    return Err(format!(
                        "{layer}.{packed} has {rows} rows; its {part} is rows {lo}..{hi}"
                    ));
                }
                let row = p.bytes.len() / rows.max(1);
                let mut shape = p.shape.clone();
                shape[0] = hi - lo;
                Ok(Some((p.dtype, shape, p.bytes[lo * row..hi * row].to_vec())))
            }
        }
    }

    fn need(&self, name: &str) -> R<(Dtype, Vec<usize>, Vec<u8>)> {
        self.raw(name)?
            .ok_or_else(|| format!("the checkpoint has no param {name}"))
    }

    /// A tensor's bf16 bits, row-major, and its shape: a bf16 tensor's
    /// untouched, a wider one's rounded to nearest even.
    pub fn bits(&self, name: &str) -> R<(Vec<usize>, Vec<u16>)> {
        let (dtype, shape, raw) = self.need(name)?;
        let bits = match dtype {
            Dtype::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
            _ => widen(dtype, &raw).iter().map(|&x| bf16_bits(x)).collect(),
        };
        Ok((shape, bits))
    }

    pub fn get(&self, name: &str) -> R<Tensor> {
        let (dtype, shape, raw) = self.need(name)?;
        Ok(Tensor {
            shape,
            data: widen(dtype, &raw),
        })
    }

    /// A Hugging Face directory's tensor by its name past the language
    /// model's trunk (`model.language_model.`, `model.` or none), widened
    /// to f32: what serve_g4 places (Gemma 4's own names).
    pub fn hf(&self, leaf: &str) -> R<Tensor> {
        let (dtype, shape, raw) = self.hf_raw(leaf, None)?;
        Ok(Tensor {
            shape,
            data: widen(dtype, &raw),
        })
    }

    /// [`Ckpt::hf`]'s tensor as bf16 bits (a bf16 tensor's untouched).
    pub fn hf_bits(&self, leaf: &str) -> R<(Vec<usize>, Vec<u16>)> {
        let (dtype, shape, raw) = self.hf_raw(leaf, None)?;
        Ok((shape, bf16_bits_of(dtype, &raw)))
    }

    /// Rows `lo..hi` of [`Ckpt::hf`]'s tensor `leaf` as bf16 bits (a bf16
    /// tensor's untouched), and the whole tensor's shape: a large table
    /// read a band of rows at a time.
    pub fn hf_rows_bits(&self, leaf: &str, lo: usize, hi: usize) -> R<(Vec<usize>, Vec<u16>)> {
        let (dtype, shape, raw) = self.hf_raw(leaf, Some((lo, hi)))?;
        Ok((shape, bf16_bits_of(dtype, &raw)))
    }

    /// [`Ckpt::hf`]'s tensor's bytes as stored, of a `U8` tensor (MXFP4
    /// blocks and scales).
    pub fn hf_u8(&self, leaf: &str) -> R<(Vec<usize>, Vec<u8>)> {
        let (dtype, shape, raw) = self.hf_raw(leaf, None)?;
        if dtype != Dtype::U8 {
            return Err(format!("{leaf} is {dtype:?}, not U8"));
        }
        Ok((shape, raw))
    }

    fn hf_raw(&self, leaf: &str, rows: Option<(usize, usize)>) -> R<Raw> {
        let Inner::Files { paths, entries } = &self.inner else {
            return Err(format!(
                "{leaf}: this placement reads a Hugging Face directory (`spatial_ckpt`), \
                 not the landed planes"
            ));
        };
        let e = TRUNKS
            .iter()
            .find_map(|t| entries.get(&format!("{t}{leaf}")))
            .ok_or_else(|| format!("the checkpoint has no tensor {leaf}"))?;
        let row = e.shape.iter().skip(1).product::<usize>() * e.dtype.bytes();
        let n = e.shape.first().copied().unwrap_or(1);
        let (lo, hi) = rows.unwrap_or((0, n));
        if lo > hi || hi > n {
            return Err(format!("{leaf}: rows {lo}..{hi} of {n}"));
        }
        let raw = read_pieces(
            &paths[e.file],
            e.offset + (lo * row) as u64,
            (hi - lo) * row,
        )?;
        Ok((e.dtype, e.shape.clone(), raw))
    }
}

/// Raw tensor bytes as bf16 bits (a bf16 tensor's untouched).
fn bf16_bits_of(dtype: Dtype, raw: &[u8]) -> Vec<u16> {
    match dtype {
        Dtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
        _ => widen(dtype, raw).iter().map(|&x| bf16_bits(x)).collect(),
    }
}

/// `len` bytes at `offset` of a file stored as `pieces` in order.
fn read_pieces(pieces: &[PathBuf], mut offset: u64, len: usize) -> R<Vec<u8>> {
    let mut out = Vec::with_capacity(len);
    for path in pieces {
        if out.len() == len {
            break;
        }
        let size = std::fs::metadata(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        if offset >= size {
            offset -= size;
            continue;
        }
        let n = (len - out.len()).min((size - offset) as usize);
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        out.extend(read_at(&mut file, offset, n)?);
        offset = 0;
    }
    if out.len() != len {
        return Err(format!(
            "{}: {len} bytes past its end",
            pieces
                .first()
                .map_or(String::new(), |p| p.display().to_string())
        ));
    }
    Ok(out)
}

/// A directory's `*.safetensors` files without an index, each whole or as
/// its pieces `<file>.partNN` (`None`: there are none).
fn shards(dir: &Path) -> R<Option<Vec<Vec<PathBuf>>>> {
    let mut files: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    let listing = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in listing.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") {
            files.insert(name, vec![entry.path()]);
        } else if let Some((file, part)) = name.rsplit_once(".part")
            && file.ends_with(".safetensors")
            && part.len() == 2
            && part.bytes().all(|b| b.is_ascii_digit())
        {
            let pieces = files.entry(file.to_string()).or_default();
            // A whole file wins over its pieces.
            if !(pieces.len() == 1 && pieces[0].file_name() == Some(file.as_ref())) {
                pieces.push(entry.path());
            }
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        files
            .into_values()
            .map(|mut p| {
                p.sort();
                p
            })
            .collect(),
    ))
}

/// Collects the executor's planes by plan param name.
struct Collect<'a> {
    trace: &'a poem_ir::Trace,
    planes: HashMap<String, Plane>,
}

impl checkpoint::executor::sink::TensorSink for Collect<'_> {
    fn publish(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> std::result::Result<(), checkpoint::error::Error> {
        let param = self
            .trace
            .params
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| {
                checkpoint::error::Error::Contract(format!(
                    "the load contract publishes `{name}`, which the trace does not name"
                ))
            })?;
        let dtype = match param.dtype {
            poem_ir::Dtype::Bf16 => Dtype::Bf16,
            poem_ir::Dtype::F16 => Dtype::F16,
            poem_ir::Dtype::F32 => Dtype::F32,
            other => {
                return Err(checkpoint::error::Error::Contract(format!(
                    "`{name}` lands as {other:?}; the wafer places bf16, f16 or f32 weights"
                )));
            }
        };
        let shape: Vec<usize> = param.shape.iter().map(|&d| d as usize).collect();
        if bytes.len() != shape.iter().product::<usize>() * dtype.bytes() {
            return Err(checkpoint::error::Error::Contract(format!(
                "`{name}` lands {} bytes for {shape:?} {dtype:?}",
                bytes.len()
            )));
        }
        self.planes.insert(
            name.to_string(),
            Plane {
                dtype,
                shape,
                bytes: bytes.to_vec(),
            },
        );
        Ok(())
    }
}

fn widen(dtype: Dtype, raw: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        Dtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        Dtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::U8 => raw.iter().map(|&b| f32::from(b)).collect(),
    }
}

/// f32 to bf16 bits, round to nearest even (numpy's `(u + 0x7FFF + lsb) >> 16`).
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: renormalize.
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn safetensors(pieces: &[PathBuf], shard: usize) -> R<HashMap<String, Entry>> {
    let path = &pieces[0];
    let n = u64_at(&read_pieces(pieces, 0, 8)?, 0);
    let header: serde_json::Value = serde_json::from_slice(&read_pieces(pieces, 8, n as usize)?)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    let base = 8 + n;
    let mut entries = HashMap::new();
    for (name, v) in header.as_object().into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match v["dtype"].as_str() {
            Some("BF16") => Dtype::Bf16,
            Some("F16") => Dtype::F16,
            Some("F32") => Dtype::F32,
            Some("U8") => Dtype::U8,
            other => {
                return Err(format!(
                    "{name}: dtype {other:?} is not a float the wafer takes"
                ));
            }
        };
        let shape = v["shape"]
            .as_array()
            .map(|s| {
                s.iter()
                    .filter_map(|d| d.as_u64())
                    .map(|d| d as usize)
                    .collect()
            })
            .unwrap_or_default();
        let start = v["data_offsets"][0]
            .as_u64()
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        entries.insert(
            name.clone(),
            Entry {
                file: shard,
                dtype,
                shape,
                offset: base + start,
            },
        );
    }
    Ok(entries)
}

/// The members of an uncompressed zip of `.npy` files (`np.savez`), zip64
/// extras included.
fn npz(path: &Path) -> R<HashMap<String, Entry>> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let tail_len = size.min(65536 + 22);
    let tail = read_at(&mut file, size - tail_len, tail_len as usize)?;
    let eocd = (0..=tail.len() - 22)
        .rev()
        .find(|&i| u32_at(&tail, i) == 0x0605_4b50)
        .ok_or_else(|| format!("{}: not a zip", path.display()))?;
    let mut count = u16_at(&tail, eocd + 10);
    let mut cd_size = u32_at(&tail, eocd + 12);
    let mut cd_offset = u32_at(&tail, eocd + 16);
    if cd_offset == 0xffff_ffff || count == 0xffff {
        // The zip64 end record, found through its locator just before.
        let loc = eocd
            .checked_sub(20)
            .filter(|&l| u32_at(&tail, l) == 0x0706_4b50)
            .ok_or_else(|| format!("{}: zip64 locator missing", path.display()))?;
        let rec = read_at(&mut file, u64_at(&tail, loc + 8), 56)?;
        count = u64_at(&rec, 32);
        cd_size = u64_at(&rec, 40);
        cd_offset = u64_at(&rec, 48);
    }
    let cd = read_at(&mut file, cd_offset, cd_size as usize)?;
    let mut entries = HashMap::new();
    let mut at = 0usize;
    for _ in 0..count {
        if u32_at(&cd, at) != 0x0201_4b50 {
            return Err(format!("{}: bad central directory", path.display()));
        }
        let method = u16_at(&cd, at + 10);
        let name_len = u16_at(&cd, at + 28) as usize;
        let extra_len = u16_at(&cd, at + 30) as usize;
        let comment_len = u16_at(&cd, at + 32) as usize;
        let name = String::from_utf8_lossy(&cd[at + 46..at + 46 + name_len]).into_owned();
        let mut sizes = [u32_at(&cd, at + 24), u32_at(&cd, at + 20)];
        let mut local = u32_at(&cd, at + 42);
        // Zip64 extra: the fields whose 32-bit slots are saturated, in order.
        let extra = &cd[at + 46 + name_len..at + 46 + name_len + extra_len];
        let mut e = 0usize;
        while e + 4 <= extra.len() {
            let id = u16_at(extra, e);
            let len = u16_at(extra, e + 2) as usize;
            if id == 1 {
                let mut f = e + 4;
                for s in &mut sizes {
                    if *s == 0xffff_ffff {
                        *s = u64_at(extra, f);
                        f += 8;
                    }
                }
                if local == 0xffff_ffff {
                    local = u64_at(extra, f);
                }
            }
            e += 4 + len;
        }
        at += 46 + name_len + extra_len + comment_len;
        if method != 0 {
            return Err(format!(
                "{}: {name} is compressed; save with np.savez",
                path.display()
            ));
        }
        let head = read_at(&mut file, local, 30)?;
        let data = local + 30 + u16_at(&head, 26) + u16_at(&head, 28);
        let (dtype, shape, header_len) =
            npy_header(&read_at(&mut file, data, 512.min(sizes[0] as usize))?)
                .map_err(|e| format!("{}: {name}: {e}", path.display()))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        entries.insert(
            key,
            Entry {
                file: 0,
                dtype,
                shape,
                offset: data + header_len,
            },
        );
    }
    Ok(entries)
}

/// An `.npy` header: the dtype (`uint16` bits are bf16), the shape, and the
/// header's length.
fn npy_header(b: &[u8]) -> R<(Dtype, Vec<usize>, u64)> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not an .npy".into());
    }
    let (len, start) = if b[6] == 1 {
        (u16_at(b, 8) as usize, 10)
    } else {
        (u32_at(b, 8) as usize, 12)
    };
    let text = std::str::from_utf8(b.get(start..start + len).ok_or("npy header too long")?)
        .map_err(|e| e.to_string())?;
    let field = |key: &str| -> R<&str> {
        let at = text
            .find(key)
            .ok_or_else(|| format!("no {key} in {text}"))?
            + key.len();
        Ok(text[at..].trim_start_matches([' ', ':']))
    };
    let descr = field("'descr'")?;
    let dtype = if descr.starts_with("'<u2'") || descr.starts_with("'<i2'") {
        Dtype::Bf16
    } else if descr.starts_with("'<f2'") {
        Dtype::F16
    } else if descr.starts_with("'<f4'") {
        Dtype::F32
    } else {
        return Err(format!("dtype {descr}"));
    };
    if field("'fortran_order'")?.starts_with("True") {
        return Err("fortran order".into());
    }
    let shape_text = field("'shape'")?;
    let close = shape_text.find(')').ok_or("shape")?;
    let shape = shape_text[1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<R<Vec<_>>>()?;
    Ok((dtype, shape, (start + len) as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poem_names_map_to_transformers() {
        assert_eq!(
            transformers_name("embed").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            transformers_name("layers.7.gate").as_deref(),
            Some("model.layers.7.mlp.gate_proj.weight")
        );
        assert_eq!(
            transformers_name("layers.0.attn_norm").as_deref(),
            Some("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(transformers_name("layers.x.q"), None);
        assert_eq!(transformers_name("layers.0.qkv"), None);
    }

    /// Packed planes read in parts: `qkv` = q | k | v rows, `gate_up` = gate
    /// | up.
    #[test]
    fn packed_planes_read_in_parts() {
        let plane = |rows: usize, cols: usize| Plane {
            dtype: Dtype::Bf16,
            shape: vec![rows, cols],
            bytes: (0..rows * cols)
                .flat_map(|i| (i as u16).to_le_bytes())
                .collect(),
        };
        let planes = HashMap::from([
            ("layers.0.qkv".to_string(), plane(8, 2)),
            ("layers.0.gate_up".to_string(), plane(6, 2)),
        ]);
        let c = Ckpt {
            inner: Inner::Landed {
                planes,
                parts: Parts {
                    q: 4,
                    kv: 2,
                    inter: 3,
                    qkv: 0,
                    vw: 0,
                    vh: 0,
                    qg: 8,
                },
            },
            config: serde_json::Value::Null,
        };
        assert_eq!(
            c.bits("layers.0.q").unwrap(),
            (vec![4, 2], (0..8).collect())
        );
        assert_eq!(
            c.bits("layers.0.k").unwrap(),
            (vec![2, 2], (8..12).collect())
        );
        assert_eq!(
            c.bits("layers.0.v").unwrap(),
            (vec![2, 2], (12..16).collect())
        );
        assert_eq!(
            c.bits("layers.0.up").unwrap(),
            (vec![3, 2], (6..12).collect())
        );
        assert!(c.has("layers.0.gate") && !c.has("layers.1.q") && !c.has("head"));
        assert!(c.tied_head().unwrap());
    }

    #[test]
    fn untied_heads_are_seen() {
        assert!(super::tied(None, None));
        assert!(super::tied(Some(true), None));
        assert!(super::tied(None, Some(true)));
        assert!(!super::tied(Some(false), None));
        assert!(!super::tied(None, Some(false)));
    }
}
