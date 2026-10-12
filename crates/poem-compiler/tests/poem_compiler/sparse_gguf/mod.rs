//! GGUFs that state metadata and name tensors without holding them: enough
//! for an import run in recording mode to say what it would read.

use std::path::{Path, PathBuf};

/// A GGUF metadata value.
pub enum Kv {
    U32(u32),
    Bool(bool),
    Str(String),
    /// An `int32` array.
    I32s(Vec<i32>),
}

/// Writes a GGUF at `path` stating `kvs`, naming each of `names` as a
/// one-value f32 tensor.
fn write(path: &Path, kvs: &[(&str, Kv)], names: &[String]) {
    fn s(out: &mut Vec<u8>, x: &str) {
        out.extend_from_slice(&(x.len() as u64).to_le_bytes());
        out.extend_from_slice(x.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(names.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value) in kvs {
        s(&mut out, key);
        match value {
            Kv::U32(n) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&n.to_le_bytes());
            }
            Kv::Bool(b) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*b));
            }
            Kv::Str(text) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                s(&mut out, text);
            }
            Kv::I32s(xs) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                for x in xs {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    for (i, name) in names.iter().enumerate() {
        s(&mut out, name);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(i as u64 * 32).to_le_bytes());
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    let len = out.len() as u64 + names.len() as u64 * 32;
    std::fs::write(path, &out).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(len)
        .unwrap();
}

/// A scratch directory, removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("pie-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The reads the deployment `id` (weights `w`, one rank) asks of a GGUF
/// stating `kvs`: grown, name by name, until its import asks for no tensor
/// the GGUF does not name.
pub fn reads(dir: &Path, id: &str, w: poem::Dtype, kvs: &[(&str, Kv)]) -> Vec<String> {
    let package = poem_compiler::catalog::package_of(id).expect("a package holds the model");
    let deploy = poem::star::Deploy {
        weights: vec![w],
        kv: poem::Dtype::Bf16,
        tp: 1,
        parts: Vec::new(),
        drafter: None,
    };
    let mut names: Vec<String> = Vec::new();
    loop {
        let path = dir.join(format!("{id}-{}.gguf", names.len()));
        write(&path, kvs, &names);
        let src = ztensor_compat::index(&path).unwrap();
        let (read, log) =
            poem::import::recording(|| package.import(id, &deploy, &src, poem::Platform::Metal));
        match read {
            Ok(_) => return log,
            Err(poem::import::Error::Missing(name)) => {
                let first = name.split("` or `").next().unwrap().to_string();
                assert!(
                    !names.contains(&first),
                    "`{first}` is named and still missing"
                );
                names.push(first);
            }
            Err(why) => panic!("`{id}` does not read the GGUF: {why}"),
        }
    }
}
