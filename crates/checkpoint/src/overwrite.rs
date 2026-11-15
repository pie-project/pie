use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::contract::{Expr, ModelContract};
use crate::error::Error;
use crate::executor::Execution;
use crate::executor::arena::ArenaBacking;
use crate::executor::sink::TensorSink;
use crate::file::read::parse_metadata;
use crate::file::zt;
use crate::plan::{LoadPlan, StorageTarget, compile};
use crate::types::{DType, Encoding, encoding_nbytes};

#[derive(Debug, Clone, Copy)]
pub struct Held<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub bytes: u64,
}

#[derive(Debug)]
pub struct Overwrite {
    snapshot: PathBuf,
    over: BTreeMap<String, usize>,
    held: usize,
    verbatim: Vec<Verbatim>,
    rest: Option<LoadPlan>,
}

#[derive(Debug)]
struct Verbatim {
    name: String,
    pieces: Vec<(PathBuf, u64, u64)>,
    bytes: u64,
}

impl Overwrite {
    pub fn plan(
        path: &Path,
        contract: &ModelContract,
        target: StorageTarget,
        held: &[Option<Held<'_>>],
    ) -> Result<Overwrite, Error> {
        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };
        let plan = compile(&metadata, contract, target)?;
        let by_name: BTreeMap<&str, usize> = held
            .iter()
            .enumerate()
            .filter_map(|(at, plane)| plane.map(|plane| (plane.name, at)))
            .collect();
        let refused = |why: String| Error::Contract(format!("{}: {why}", path.display()));
        let mut over = BTreeMap::new();
        for decl in plan.tensors.iter().filter(|d| d.visibility.is_public()) {
            let Some((at, plane)) = by_name
                .get(decl.name.as_str())
                .and_then(|&at| held[at].map(|plane| (at, plane)))
            else {
                return Err(refused(format!(
                    "lands `{}`, which is no plane the loaded model holds",
                    decl.name
                )));
            };
            let bytes = encoding_nbytes(&decl.shape, &decl.encoding);
            let dense = matches!(
                decl.encoding,
                Encoding::Raw(DType::F32 | DType::F16 | DType::Bf16)
            );
            let shaped = !dense
                || decl.shape.len() != plane.shape.len()
                || decl
                    .shape
                    .iter()
                    .zip(plane.shape)
                    .all(|(&lands, &holds)| u64::try_from(lands).ok() == Some(holds));
            if bytes != Some(plane.bytes) || !shaped {
                return Err(refused(format!(
                    "lands `{}` as {:?} {:?} ({} bytes) over a held {:?} of {} bytes: a \
                     checkpoint of another architecture, size or dtype than the loaded one",
                    decl.name,
                    decl.shape,
                    decl.encoding,
                    bytes.unwrap_or(0),
                    plane.shape,
                    plane.bytes,
                )));
            }
            over.insert(decl.name.clone(), at);
        }
        let landed: std::collections::BTreeSet<usize> = over.values().copied().collect();
        if let Some(missing) = held
            .iter()
            .enumerate()
            .find_map(|(at, plane)| plane.filter(|_| !landed.contains(&at)))
        {
            return Err(refused(format!(
                "lands no `{}`, which the loaded model holds",
                missing.name
            )));
        }
        let (verbatim, rest) = split(&metadata, contract, &plan, snapshot);
        let rest = if verbatim.is_empty() {
            Some(plan)
        } else if rest.tensors.is_empty() {
            None
        } else {
            Some(compile(&metadata, &rest, plan.target)?)
        };
        Ok(Overwrite {
            snapshot: snapshot.to_path_buf(),
            over,
            held: held.len(),
            verbatim,
            rest,
        })
    }

    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.rest
            .as_ref()
            .map_or(0, |rest| rest.memory.arena_bytes())
    }

    #[must_use]
    pub fn snapshot(&self) -> &Path {
        &self.snapshot
    }

    pub fn run(
        &self,
        arena: Option<&mut dyn ArenaBacking>,
        write: impl FnMut(usize, &[u8]) -> Result<(), Error>,
    ) -> Result<(u32, u64), Error> {
        let mut sink = Over {
            over: &self.over,
            write,
            landed: vec![false; self.held],
            planes: 0,
            bytes: 0,
        };
        let widest = self.verbatim.iter().map(|plane| plane.bytes).max();
        let mut staged = vec![0u8; usize::try_from(widest.unwrap_or(0)).unwrap_or(usize::MAX)];
        for plane in &self.verbatim {
            let staged = &mut staged[..usize::try_from(plane.bytes).unwrap_or(usize::MAX)];
            let mut at = 0usize;
            for (file, offset, bytes) in &plane.pieces {
                let bytes = usize::try_from(*bytes).unwrap_or(usize::MAX);
                read_at(file, *offset, &mut staged[at..at + bytes])?;
                at += bytes;
            }
            sink.publish(&plane.name, staged)?;
        }
        drop(staged);
        if let Some(rest) = &self.rest {
            let execution = Execution::new(rest, &self.snapshot);
            match arena {
                Some(arena) => execution.arena(arena),
                None => execution,
            }
            .sink(&mut sink)
            .run()?;
        }
        if sink.planes as usize != self.over.len() {
            return Err(Error::Internal(format!(
                "the plan landed {} of its {} planes",
                sink.planes,
                self.over.len()
            )));
        }
        Ok((sink.planes, sink.bytes))
    }
}

fn split(
    metadata: &crate::file::Metadata,
    contract: &ModelContract,
    plan: &LoadPlan,
    snapshot: &Path,
) -> (Vec<Verbatim>, ModelContract) {
    let produced: std::collections::BTreeSet<&str> = contract
        .tensors
        .iter()
        .flat_map(|tensor| tensor.expr.outputs())
        .collect();
    let decoded: std::collections::BTreeSet<&str> = contract
        .tensors
        .iter()
        .filter(|tensor| !matches!(tensor.expr, Expr::Src(_)))
        .flat_map(|tensor| {
            tensor
                .zero_points
                .as_deref()
                .into_iter()
                .chain(tensor.scales.as_ref().map(|scales| scales.of.as_str()))
        })
        .collect();
    let mut verbatim = Vec::new();
    let mut rest = Vec::new();
    for tensor in &contract.tensors {
        let joined = match &tensor.expr {
            Expr::Src(from) => Some(vec![from.as_str()]),
            Expr::Concat { axis, parts } if axis.0 == 0 => parts
                .iter()
                .map(|part| match part {
                    Expr::Src(from) => Some(from.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        };
        let found = joined
            .filter(|_| {
                tensor.visibility.is_public()
                    && contract.groups.is_empty()
                    && tensor.scales.is_none()
                    && tensor.zero_points.is_none()
                    && !produced.contains(tensor.name.as_str())
                    && !decoded.contains(tensor.name.as_str())
            })
            .and_then(|from| verbatim_plane(metadata, plan, snapshot, &tensor.name, &from));
        match found {
            Some(plane) => verbatim.push(plane),
            None => rest.push(tensor.clone()),
        }
    }
    let rest = ModelContract {
        alignment: contract.alignment,
        tensors: rest,
        groups: contract.groups.clone(),
    };
    (verbatim, rest)
}

fn verbatim_plane(
    metadata: &crate::file::Metadata,
    plan: &LoadPlan,
    snapshot: &Path,
    name: &str,
    from: &[&str],
) -> Option<Verbatim> {
    let decl = plan.tensors.iter().find(|decl| decl.name == name)?;
    let dense = matches!(
        decl.encoding,
        Encoding::Raw(DType::F32 | DType::F16 | DType::Bf16)
    );
    let (&rows, row) = decl.shape.split_first()?;
    if !dense {
        return None;
    }
    let mut pieces = Vec::with_capacity(from.len());
    let mut joined = 0i64;
    for source in from {
        let raw = metadata.tensor_by_name(source)?;
        let (&leads, rest) = raw.shape.split_first()?;
        let bytes = encoding_nbytes(&raw.shape, &raw.encoding)?;
        if raw.encoding != decl.encoding || rest != row || raw.span_bytes != bytes {
            return None;
        }
        let file = metadata.files.iter().find(|file| file.id == raw.file_id)?;
        pieces.push((snapshot.join(&file.path), raw.file_offset, bytes));
        joined += leads;
    }
    let bytes = encoding_nbytes(&decl.shape, &decl.encoding)?;
    if joined != rows || pieces.iter().map(|piece| piece.2).sum::<u64>() != bytes {
        return None;
    }
    Some(Verbatim {
        name: name.to_string(),
        pieces,
        bytes,
    })
}

fn read_at(path: &Path, offset: u64, into: &mut [u8]) -> Result<(), Error> {
    let failed = |why: std::io::Error| Error::Checkpoint(format!("read {}: {why}", path.display()));
    let file = std::fs::File::open(path).map_err(failed)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        const PIECE: usize = 32 << 20;
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(16);
        let piece = into.len().div_ceil(workers).max(PIECE);
        std::thread::scope(|scope| {
            let reads: Vec<_> = into
                .chunks_mut(piece)
                .enumerate()
                .map(|(at, buf)| {
                    let file = &file;
                    scope.spawn(move || file.read_exact_at(buf, offset + (at * piece) as u64))
                })
                .collect();
            reads.into_iter().try_for_each(|read| {
                read.join()
                    .unwrap_or_else(|_| Err(std::io::Error::other("a read panicked")))
            })
        })
        .map_err(failed)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file;
        file.seek(SeekFrom::Start(offset)).map_err(failed)?;
        file.read_exact(into).map_err(failed)
    }
}

struct Over<'a, W> {
    over: &'a BTreeMap<String, usize>,
    write: W,
    landed: Vec<bool>,
    planes: u32,
    bytes: u64,
}

impl<W: FnMut(usize, &[u8]) -> Result<(), Error>> TensorSink for Over<'_, W> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        let Some(&at) = self.over.get(name) else {
            return Err(Error::Internal(format!(
                "the plan published `{name}`, which its checked planes do not name"
            )));
        };
        if std::mem::replace(&mut self.landed[at], true) {
            return Err(Error::Internal(format!(
                "the plan published `{name}` twice"
            )));
        }
        (self.write)(at, bytes)?;
        self.planes += 1;
        self.bytes += bytes.len() as u64;
        Ok(())
    }
}
