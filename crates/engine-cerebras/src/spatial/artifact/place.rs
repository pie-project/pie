//! The placer: an artifact's bindings ([`Artifact::bindings`]) as device copies, in order. A
//! binding's words come from a staged array of the prep (`<prefix><name>.npy`, a block or a
//! [`Slice`] of it) or the artifact's rcfg words.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kernels_cerebras::spatial::artifact::Artifact;
use kernels_cerebras::spatial::floor::{Coord, Rect};
use kernels_cerebras::spatial::npy::Npy;
use kernels_cerebras::spatial::weights::Source;

use crate::spatial::device::Device;

type R<T> = Result<T, String>;

/// The staged arrays a placement reads, opened once each.
struct Staged<'a> {
    dir: &'a Path,
    prefix: &'a str,
    open: HashMap<String, Npy>,
}

impl Staged<'_> {
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}{name}.npy", self.prefix))
    }

    fn has(&self, name: &str) -> bool {
        self.open.contains_key(name) || self.path(name).exists()
    }

    fn get(&mut self, name: &str) -> R<&Npy> {
        if !self.open.contains_key(name) {
            let n = Npy::open(&self.path(name))?;
            self.open.insert(name.to_string(), n);
        }
        Ok(&self.open[name])
    }
}

/// A source's words (`None`: an optional slice whose file is not staged).
fn words(s: &Source, rect: Rect, staged: &mut Staged, art: &Artifact) -> R<Option<Vec<u32>>> {
    Ok(Some(match s {
        Source::Staged { name, index, words } => {
            let a = staged.get(name)?;
            let stride = a.stride0();
            let mut v = Vec::new();
            a.words(
                index.0 as usize * stride,
                (index.1 - index.0) as usize * stride,
                &mut v,
            )?;
            let pes = (rect.w * rect.h) as usize;
            if v.len() != pes * *words as usize {
                return Err(format!(
                    "{name}[{}..{}]: {} words, not {words} a PE over {pes} PEs",
                    index.0,
                    index.1,
                    v.len()
                ));
            }
            v
        }
        Source::Slice(sl) => {
            if sl.optional && !staged.has(&sl.name) {
                return Ok(None);
            }
            sl.words(staged.get(&sl.name)?)?
        }
        Source::Rcfg => {
            let mut v = Vec::new();
            let mut width = None;
            for y in rect.y..rect.y + rect.h {
                for x in rect.x..rect.x + rect.w {
                    let w = art
                        .rcfg
                        .get(Coord::new(x, y))
                        .as_ref()
                        .ok_or_else(|| format!("rcfg at ({x}, {y}): the PE reads none"))?;
                    if *width.get_or_insert(w.len()) != w.len() {
                        return Err(format!(
                            "rcfg words of unequal length in one copy (at ({x}, {y}))"
                        ));
                    }
                    v.extend_from_slice(w);
                }
            }
            v
        }
    }))
}

/// Places every binding of `art` on `d`, in order, from the prep `<prep>/<prefix>*.npy`. An
/// empty rectangle, or an optional slice not staged, is no copy.
pub fn place(d: &mut dyn Device, art: &Artifact, prep: &Path, prefix: &str) -> R<()> {
    let mut staged = Staged {
        dir: prep,
        prefix,
        open: HashMap::new(),
    };
    for b in &art.bindings {
        let pes = (b.rect.w * b.rect.h) as usize;
        if pes == 0 {
            continue;
        }
        let Some(v) = words(&b.source, b.rect, &mut staged, art)? else {
            continue;
        };
        if v.is_empty() || v.len() % pes != 0 {
            return Err(format!(
                "placing {}: {} words over {pes} PEs",
                b.sym,
                v.len()
            ));
        }
        let r = super::rect(&b.rect);
        let n = v.len() / pes;
        if b.b16 {
            d.put16(&b.sym, r, n, &v)?;
        } else {
            d.put(&b.sym, r, n, &v)?;
        }
    }
    Ok(())
}
