//! The built program as the engine runs it: what the host finds on the wafer
//! ([`HostLayout`]), every copy onto it in placement order ([`Binding`]), every PE's rcfg words,
//! and the model facts the host's math needs. Nothing here names a family: a program built from
//! any plan is run the same way.
//!
//! On disk, a directory (what `split-geo` writes):
//! - `program.json`: [`Header`] (the format, the model facts, the meta it was built from);
//! - `host.json`: [`HostLayout`];
//! - `bindings.json`: the [`Binding`]s;
//! - `rcfg.npy` (u32, `[h][w][words]`, zero-padded to the longest; idle PEs all zero) and
//!   `rcfg_n.npy` (u32, `[h][w]`: each PE's word count).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::emit::host::HostLayout;
use super::emit::meta::Meta;
use super::emit::rcfg;
use super::floor::{Coord, Grid};
use super::npy::{self, Npy};
use super::program::Program;
use super::weights::Binding;

type R<T> = Result<T, String>;

/// The artifact format [`Artifact::read`] takes.
pub const FORMAT: u32 = 1;

/// The model facts the host's math needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// The checkpoint's vocabulary and hidden size (the head pads past the one, the placed hidden
    /// past the other).
    pub vocab: u32,
    pub hidden: u32,
    /// The final norm's eps (the host scales `lgt` by its 1/rms).
    pub eps: f32,
    /// `lgt`'s prescale (the prep's `HSCALE`: the band head's 2^15 logits; 1 without one).
    pub lgt_scale: f32,
}

/// `program.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub format: u32,
    pub model: Model,
    /// The prep's meta (overrides applied) the program was built from; `None` for a program
    /// built from a plan file. A host opening a prep checks it against the prep's.
    pub meta: Option<BTreeMap<String, serde_json::Value>>,
}

/// A built program, as the engine runs it.
pub struct Artifact {
    pub header: Header,
    pub host: HostLayout,
    pub bindings: Vec<Binding>,
    /// Every PE's rcfg words (`None`: the PE's kernel reads none).
    pub rcfg: Grid<Option<Vec<u32>>>,
}

fn json(dir: &Path, name: &str, v: &impl Serialize) -> R<()> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("{name}: {e}"))?;
    let p = dir.join(name);
    std::fs::write(&p, format!("{s}\n")).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(dir: &Path, name: &str) -> R<T> {
    let p = dir.join(name);
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

impl Artifact {
    /// The program `meta` states ([`Program::from_meta`]), as the engine runs it.
    pub fn build(meta: &Meta) -> R<Artifact> {
        Artifact::of(&Program::from_meta(meta)?, Some(meta))
    }

    /// `prog` as the engine runs it; `meta`: the prep's meta it was built from, if any.
    pub fn of(prog: &Program, meta: Option<&Meta>) -> R<Artifact> {
        let s = &prog.plan.shape;
        Ok(Artifact {
            header: Header {
                format: FORMAT,
                model: Model {
                    vocab: s.vocab,
                    hidden: s.hidden,
                    eps: s.norm.eps,
                    lgt_scale: meta.and_then(|m| m.float("HSCALE")).unwrap_or(1.0) as f32,
                },
                meta: meta.map(|m| m.map.clone()),
            },
            host: HostLayout::of(prog)?,
            bindings: prog.bindings()?,
            rcfg: rcfg::all(prog)?,
        })
    }

    /// Writes the whole artifact into `dir`.
    pub fn write(&self, dir: &Path) -> R<()> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        self.write_header(dir)?;
        self.write_host(dir)?;
        self.write_bindings(dir)?;
        self.write_rcfg(dir)
    }

    pub fn write_header(&self, dir: &Path) -> R<()> {
        json(dir, "program.json", &self.header)
    }

    pub fn write_host(&self, dir: &Path) -> R<()> {
        json(dir, "host.json", &self.host)
    }

    pub fn write_bindings(&self, dir: &Path) -> R<()> {
        json(dir, "bindings.json", &self.bindings)
    }

    /// `rcfg.npy` and `rcfg_n.npy`.
    pub fn write_rcfg(&self, dir: &Path) -> R<()> {
        let g = &self.rcfg;
        let (w, h) = (g.w as usize, g.h as usize);
        let n = g
            .iter()
            .filter_map(|(_, v)| v.as_ref().map(Vec::len))
            .max()
            .unwrap_or(0);
        let mut data = vec![0u32; w * h * n];
        let mut lens = vec![0u32; w * h];
        for (c, v) in g.iter() {
            if let Some(v) = v {
                let at = c.y as usize * w + c.x as usize;
                data[at * n..at * n + v.len()].copy_from_slice(v);
                lens[at] = v.len() as u32;
            }
        }
        for (name, bytes) in [
            ("rcfg.npy", npy::u32_bytes(&[h, w, n], &data)),
            ("rcfg_n.npy", npy::u32_bytes(&[h, w], &lens)),
        ] {
            let p = dir.join(name);
            std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        Ok(())
    }

    /// Reads an artifact [`Artifact::write`] wrote.
    pub fn read(dir: &Path) -> R<Artifact> {
        let header: Header = read_json(dir, "program.json")?;
        if header.format != FORMAT {
            return Err(format!(
                "{}: artifact format {}, this build reads {FORMAT} (rebuild it with split-geo)",
                dir.display(),
                header.format
            ));
        }
        let n = Npy::open(&dir.join("rcfg_n.npy"))?;
        let a = Npy::open(&dir.join("rcfg.npy"))?;
        let (h, w) = match n.shape[..] {
            [h, w] => (h, w),
            _ => return Err(format!("rcfg_n.npy: shape {:?}", n.shape)),
        };
        if a.shape.len() != 3 || a.shape[0] != h || a.shape[1] != w {
            return Err(format!(
                "rcfg.npy: shape {:?}, rcfg_n {:?}",
                a.shape, n.shape
            ));
        }
        let mut lens = Vec::with_capacity(w * h);
        n.words(0, w * h, &mut lens)?;
        let words = a.shape[2];
        let mut all = Vec::with_capacity(w * h * words);
        a.words(0, w * h * words, &mut all)?;
        let mut rcfg = Grid::filled(w as u32, h as u32, None);
        for (i, &len) in lens.iter().enumerate() {
            if len > 0 {
                let at = i * words;
                *rcfg.get_mut(Coord::new((i % w) as u32, (i / w) as u32)) =
                    Some(all[at..at + len as usize].to_vec());
            }
        }
        Ok(Artifact {
            header,
            host: read_json(dir, "host.json")?,
            bindings: read_json(dir, "bindings.json")?,
            rcfg,
        })
    }
}
