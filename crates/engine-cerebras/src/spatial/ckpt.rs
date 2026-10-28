//! The checkpoint the wafer model places: a directory with `config.json` and
//! either `model.safetensors` (a Hugging Face export) or `weights.npz` (the
//! bring-up's bf16 bits as `uint16` arrays, stored uncompressed). Tensors are
//! read one at a time and widened to `f32`; nothing else is kept in memory.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    Bf16,
    F16,
    F32,
}

impl Dtype {
    fn bytes(self) -> usize {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
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

pub struct Ckpt {
    path: PathBuf,
    entries: HashMap<String, Entry>,
    config: serde_json::Value,
}

type R<T> = std::result::Result<T, String>;

fn u16_at(b: &[u8], at: usize) -> u64 {
    u64::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> u64 {
    u64::from(u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes")))
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

impl Ckpt {
    pub fn open(dir: &Path) -> R<Ckpt> {
        let config_path = dir.join("config.json");
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| format!("{}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
        let (path, entries) = if dir.join("model.safetensors").exists() {
            let path = dir.join("model.safetensors");
            let entries = safetensors(&path)?;
            (path, entries)
        } else {
            let path = dir.join("weights.npz");
            let entries = npz(&path)?;
            (path, entries)
        };
        Ok(Ckpt {
            path,
            entries,
            config,
        })
    }

    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    pub fn get(&self, name: &str) -> R<Tensor> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("{} has no tensor {name}", self.path.display()))?;
        let mut file = File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let raw = read_at(&mut file, e.offset, e.elems() * e.dtype.bytes())?;
        let data = match e.dtype {
            Dtype::Bf16 => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            Dtype::F16 => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };
        Ok(Tensor {
            shape: e.shape.clone(),
            data,
        })
    }
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

fn safetensors(path: &Path) -> R<HashMap<String, Entry>> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = u64_at(&read_at(&mut file, 0, 8)?, 0);
    let header: serde_json::Value = serde_json::from_slice(&read_at(&mut file, 8, n as usize)?)
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
            other => return Err(format!("{name}: dtype {other:?} is not a float the wafer takes")),
        };
        let shape = v["shape"]
            .as_array()
            .map(|s| s.iter().filter_map(|d| d.as_u64()).map(|d| d as usize).collect())
            .unwrap_or_default();
        let start = v["data_offsets"][0]
            .as_u64()
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        entries.insert(
            name.clone(),
            Entry {
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
        let (dtype, shape, header_len) = npy_header(&read_at(&mut file, data, 512.min(sizes[0] as usize))?)
            .map_err(|e| format!("{}: {name}: {e}", path.display()))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        entries.insert(
            key,
            Entry {
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
        let at = text.find(key).ok_or_else(|| format!("no {key} in {text}"))? + key.len();
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
