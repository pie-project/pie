use std::collections::HashMap;

use super::Expr;
use super::infer::Checked;
use crate::error::{Error, OrOverflow};
use crate::extent::{Dim, Rect};
use crate::types::Encoding;

const MAX_RANK: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Leaf {
    Checkpoint(String),
    Contract(String),
}

impl Leaf {
    pub fn name(&self) -> &str {
        match self {
            Leaf::Checkpoint(name) | Leaf::Contract(name) => name,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub source: RunSource,
    pub dst_elem: i64,
    pub len: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunSource {
    Leaf { leaf: usize, src_elem: i64 },
    Zero,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lowering {
    Copy(CopyList),
    Gather(GatherList),
}

impl Lowering {
    pub fn leaves(&self) -> &[Leaf] {
        match self {
            Lowering::Copy(copies) => &copies.leaves,
            Lowering::Gather(gather) => &gather.leaves,
        }
    }

    pub fn elements(&self) -> i64 {
        match self {
            Lowering::Copy(copies) => copies.elements,
            Lowering::Gather(gather) => gather.elements,
        }
    }

    pub fn cost(&self) -> usize {
        match self {
            Lowering::Copy(copies) => copies.cost(),
            Lowering::Gather(_) => 1,
        }
    }

    pub fn as_copy(&self) -> Option<&CopyList> {
        match self {
            Lowering::Copy(copies) => Some(copies),
            Lowering::Gather(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherList {
    pub leaves: Vec<Leaf>,
    pub leaf: usize,
    pub indices: Vec<i64>,
    pub block: i64,
    pub rows: i64,
    pub src_row: i64,
    pub elements: i64,
}

impl GatherList {
    pub fn dst_row(&self) -> i64 {
        self.indices.len() as i64 * self.block
    }

    pub fn source_elements(&self) -> i64 {
        self.rows * self.src_row
    }

    pub fn byte_geometry(&self, encoding: &Encoding) -> Result<GatherBytes, Error> {
        let scale = ByteScale::of(encoding);
        Ok(GatherBytes {
            block_bytes: scale.extent(self.block, "gather block")?,
            rows: u64::try_from(self.rows)
                .map_err(|_| Error::Internal("gather has a negative row count".to_string()))?,
            src_row_bytes: scale.extent(self.src_row, "gather source row")?,
        })
    }

    pub fn byte_rects(&self, encoding: &Encoding) -> Result<Vec<Rect>, Error> {
        let scale = ByteScale::of(encoding);
        let block = scale.extent(self.block, "gather block")?;
        let src_row = scale.stride(self.src_row, "gather source row")?;
        let dst_row = scale.stride(self.dst_row(), "gather destination row")?;
        self.indices
            .iter()
            .enumerate()
            .map(|(at, index)| {
                let mut dims = Vec::with_capacity(2);
                if self.rows > 1 {
                    dims.push(Dim {
                        count: self.rows,
                        src_stride: src_row,
                        dst_stride: dst_row,
                    });
                }
                dims.push(Dim {
                    count: block as i64,
                    src_stride: 1,
                    dst_stride: 1,
                });
                Ok(Rect {
                    leaf: self.leaf,
                    src_offset: scale.offset(
                        index
                            .checked_mul(self.block)
                            .or_overflow("gather source offset overflows")?,
                        "gather source",
                    )?,
                    dst_offset: scale.offset(
                        (at as i64)
                            .checked_mul(self.block)
                            .or_overflow("gather destination offset overflows")?,
                        "gather destination",
                    )?,
                    dims,
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherBytes {
    pub block_bytes: u64,
    pub rows: u64,
    pub src_row_bytes: u64,
}

impl GatherBytes {
    pub fn source_bytes(&self) -> u64 {
        self.rows.saturating_mul(self.src_row_bytes)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CopyList {
    pub leaves: Vec<Leaf>,
    pub runs: Vec<Run>,
    pub elements: i64,
}

impl CopyList {
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    pub fn mean_run_elements(&self) -> i64 {
        if self.runs.is_empty() {
            0
        } else {
            self.elements / self.runs.len() as i64
        }
    }

    pub fn pieces(&self) -> Vec<Piece> {
        fold(self.runs.iter().map(seed).collect())
    }

    pub fn cost(&self) -> usize {
        self.copy_pieces().len() + usize::from(self.needs_zero_fill())
    }

    pub fn copy_pieces(&self) -> Vec<Piece> {
        fold(
            self.runs
                .iter()
                .filter(|run| run.source != RunSource::Zero)
                .map(seed)
                .collect(),
        )
    }

    pub fn needs_zero_fill(&self) -> bool {
        self.runs.iter().any(|run| run.source == RunSource::Zero)
    }

    pub fn byte_runs(&self, encoding: &Encoding) -> Result<Vec<ByteRun>, Error> {
        let scale = ByteScale::of(encoding);
        self.runs
            .iter()
            .map(|run| {
                Ok(ByteRun {
                    source: match run.source {
                        RunSource::Leaf { leaf, src_elem } => ByteRunSource::Leaf {
                            leaf,
                            src_offset: scale.offset(src_elem, "run source")?,
                        },
                        RunSource::Zero => ByteRunSource::Zero,
                    },
                    dst_offset: scale.offset(run.dst_elem, "run destination")?,
                    len: scale.offset(run.len, "run length")?,
                })
            })
            .collect()
    }

    pub fn byte_pieces(&self, encoding: &Encoding) -> Result<Vec<Rect>, Error> {
        let scale = ByteScale::of(encoding);
        self.copy_pieces()
            .into_iter()
            .map(|piece| {
                let RunSource::Leaf { leaf, src_elem } = piece.source else {
                    unreachable!("copy_pieces excludes holes");
                };
                let dims = piece
                    .dims
                    .iter()
                    .enumerate()
                    .map(|(level, dim)| {
                        if level + 1 == piece.dims.len() {
                            debug_assert_eq!((dim.src_stride, dim.dst_stride), (1, 1));
                            return Ok(Dim {
                                count: scale.extent(dim.count, "piece length")? as i64,
                                src_stride: 1,
                                dst_stride: 1,
                            });
                        }
                        Ok(Dim {
                            count: dim.count,
                            src_stride: scale.stride(dim.src_stride, "source stride")?,
                            dst_stride: scale.stride(dim.dst_stride, "destination stride")?,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Rect {
                    leaf,
                    src_offset: scale.offset(src_elem, "piece source")?,
                    dst_offset: scale.offset(piece.dst_elem, "piece destination")?,
                    dims,
                })
            })
            .collect()
    }
}

enum ByteScale {
    Bits(i64),
    Blocked { elems: i64, bytes: i64 },
}

impl ByteScale {
    fn of(encoding: &Encoding) -> Self {
        if let Encoding::Quant(spec) = encoding
            && let Some((elems, bytes)) = spec.scheme.block_layout()
        {
            return Self::Blocked {
                elems: i64::try_from(elems).unwrap_or(0),
                bytes: i64::try_from(bytes).unwrap_or(0),
            };
        }
        Self::Bits(i64::from(bits_per_element(encoding)))
    }

    fn scaled(&self, elems: i64, what: &str) -> Result<i64, Error> {
        let (bits, block) = match *self {
            Self::Bits(bits) => (bits, None),
            Self::Blocked {
                elems: per,
                bytes: cost,
            } => (0, Some((per, cost))),
        };
        if let Some((per, cost)) = block {
            if per == 0 {
                return Err(Error::Internal(
                    "blocked encoding with no block".to_string(),
                ));
            }
            if elems % per != 0 {
                return Err(Error::Contract(format!(
                    "{what} of {elems} elements does not land on a {per}-element \
                     block boundary; a blocked payload carries its scale inside \
                     the block, so a partial block has no byte address"
                )));
            }
            return (elems / per)
                .checked_mul(cost)
                .or_overflow("byte offset overflows");
        }
        let total = elems
            .checked_mul(bits)
            .or_overflow("byte offset overflows")?;
        if total % 8 != 0 {
            return Err(Error::Contract(format!(
                "{what} of {elems} elements is not byte-aligned under a {bits} -bit encoding"
            )));
        }
        Ok(total / 8)
    }

    fn offset(&self, elems: i64, what: &str) -> Result<u64, Error> {
        u64::try_from(self.scaled(elems, what)?).map_err(|_| {
            Error::Internal(format!(
                "{what} lowered to a negative byte offset from {elems} elements"
            ))
        })
    }

    fn extent(&self, elems: i64, what: &str) -> Result<u64, Error> {
        self.offset(elems, what)
    }

    fn stride(&self, elems: i64, what: &str) -> Result<i64, Error> {
        self.scaled(elems, what)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRun {
    pub source: ByteRunSource,
    pub dst_offset: u64,
    pub len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteRunSource {
    Leaf { leaf: usize, src_offset: u64 },
    Zero,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub source: RunSource,
    pub dst_elem: i64,
    pub dims: Vec<Dim>,
}

impl Piece {
    pub fn elements(&self) -> i64 {
        self.dims.iter().map(|dim| dim.count).product()
    }
}

fn seed(run: &Run) -> Piece {
    Piece {
        source: run.source,
        dst_elem: run.dst_elem,
        dims: vec![Dim {
            count: run.len,
            src_stride: i64::from(run.source != RunSource::Zero),
            dst_stride: 1,
        }],
    }
}

fn fold(mut items: Vec<Piece>) -> Vec<Piece> {
    while let Some(folded) = fold_once(&items) {
        items = folded;
    }
    items
}

fn fold_once(items: &[Piece]) -> Option<Vec<Piece>> {
    let mut out: Vec<Piece> = Vec::new();
    let mut changed = false;
    let mut at = 0;
    while at < items.len() {
        let head = &items[at];
        let mut end = at + 1;
        let mut src_stride = 0;
        let mut dst_stride = 0;
        if let Some(next) = items.get(at + 1)
            && let Some(strides) = step_between(head, next)
            && strides.1 == head.elements()
            && strides.0 >= 0
        {
            (src_stride, dst_stride) = strides;
            end = at + 2;
            while let Some(further) = items.get(end) {
                if step_between(&items[end - 1], further) != Some((src_stride, dst_stride)) {
                    break;
                }
                end += 1;
            }
        }
        if end - at >= 2 {
            let mut dims = Vec::with_capacity(head.dims.len() + 1);
            dims.push(Dim {
                count: (end - at) as i64,
                src_stride,
                dst_stride,
            });
            dims.extend_from_slice(&head.dims);
            out.push(Piece {
                source: head.source,
                dst_elem: head.dst_elem,
                dims,
            });
            changed = true;
        } else {
            out.push(head.clone());
        }
        at = end;
    }
    changed.then_some(out)
}

fn step_between(a: &Piece, b: &Piece) -> Option<(i64, i64)> {
    if a.dims != b.dims {
        return None;
    }
    let src_stride = match (a.source, b.source) {
        (
            RunSource::Leaf { leaf, src_elem },
            RunSource::Leaf {
                leaf: next_leaf,
                src_elem: next_elem,
            },
        ) if leaf == next_leaf => next_elem - src_elem,
        (RunSource::Zero, RunSource::Zero) => 0,
        _ => return None,
    };
    Some((src_stride, b.dst_elem - a.dst_elem))
}

pub fn bits_per_element(encoding: &Encoding) -> u32 {
    match encoding {
        Encoding::Raw(dtype) => u32::try_from(dtype.bytes_ceil()).unwrap_or(0) * 8,
        Encoding::Quant(spec) => u32::from(spec.normalized_bits()),
    }
}

pub fn compile(expr: &Expr, checked: &Checked, max_runs: usize) -> Result<Lowering, Error> {
    let mut builder = Builder {
        checked,
        nodes: Vec::new(),
        leaves: Vec::new(),
        leaf_index: HashMap::new(),
    };
    let root = builder.build(expr)?;
    builder.lower(root, max_runs)
}

struct Node {
    kind: Kind,
    shape: Vec<i64>,
    strides: Vec<i64>,
    elements: i64,
}

enum Kind {
    Leaf(usize),
    Slice {
        src: usize,
        axis: usize,
        start: i64,
        whole: bool,
    },
    Stride {
        src: usize,
        axis: usize,
        start: i64,
        step: i64,
    },
    Gather {
        src: usize,
        axis: usize,
        indices: Vec<i64>,
    },
    Concat {
        axis: usize,
        parts: Vec<(i64, usize)>,
    },
    Transmute {
        src: usize,
    },
    Fill,
}

struct Builder<'a> {
    checked: &'a Checked,
    nodes: Vec<Node>,
    leaves: Vec<Leaf>,
    leaf_index: HashMap<Leaf, usize>,
}

impl Builder<'_> {
    fn push(&mut self, kind: Kind, shape: Vec<i64>) -> Result<usize, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::Contract(format!(
                "rank {} exceeds the supported maximum of {MAX_RANK}",
                shape.len()
            )));
        }
        let mut strides = vec![1_i64; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .or_overflow("shape overflows i64")?;
        }
        let elements = match shape.first() {
            Some(first) => first
                .checked_mul(strides[0])
                .or_overflow("element count overflows i64")?,
            None => 1,
        };
        self.nodes.push(Node {
            kind,
            shape,
            strides,
            elements,
        });
        Ok(self.nodes.len() - 1)
    }

    fn intern(&mut self, leaf: Leaf) -> usize {
        if let Some(found) = self.leaf_index.get(&leaf) {
            return *found;
        }
        let index = self.leaves.len();
        self.leaves.push(leaf.clone());
        self.leaf_index.insert(leaf, index);
        index
    }

    fn build(&mut self, expr: &Expr) -> Result<usize, Error> {
        match expr {
            Expr::Src(name) => {
                let shape = self
                    .checked
                    .source(name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "checkpoint tensor '{name}' was not resolved by the type checker"
                        ))
                    })?
                    .shape
                    .clone();
                let leaf = self.intern(Leaf::Checkpoint(name.clone()));
                self.push(Kind::Leaf(leaf), shape)
            }
            Expr::Out(name) => {
                let shape = self
                    .checked
                    .output(name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "contract '{name}' was not resolved by the type checker"
                        ))
                    })?
                    .shape
                    .clone();
                let leaf = self.intern(Leaf::Contract(name.clone()));
                self.push(Kind::Leaf(leaf), shape)
            }
            Expr::Slice {
                src,
                axis,
                start,
                len,
            } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                let operand_extent = axis_extent(&shape, axis, "Slice")?;
                shape[axis] = *len;
                let whole = *start == 0 && *len == operand_extent;
                self.push(
                    Kind::Slice {
                        src: inner,
                        axis,
                        start: *start,
                        whole,
                    },
                    shape,
                )
            }
            Expr::Stride {
                src,
                axis,
                start,
                len,
                step,
            } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                axis_extent(&shape, axis, "Stride")?;
                shape[axis] = *len;
                self.push(
                    Kind::Stride {
                        src: inner,
                        axis,
                        start: *start,
                        step: *step,
                    },
                    shape,
                )
            }
            Expr::Gather { src, axis, indices } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                axis_extent(&shape, axis, "Gather")?;
                shape[axis] = indices.len() as i64;
                self.push(
                    Kind::Gather {
                        src: inner,
                        axis,
                        indices: indices.clone(),
                    },
                    shape,
                )
            }
            Expr::Concat { axis, parts } => {
                let axis = usize::from(axis.0);
                let mut placed = Vec::with_capacity(parts.len());
                let mut offset = 0_i64;
                let mut shape = Vec::new();
                for part in parts {
                    let inner = self.build(part)?;
                    if placed.is_empty() {
                        shape = self.nodes[inner].shape.clone();
                    }
                    placed.push((offset, inner));
                    offset += axis_extent(&self.nodes[inner].shape, axis, "Concat")?;
                }
                if placed.is_empty() {
                    return Err(Error::Internal(
                        "empty Concat escaped the type checker".to_string(),
                    ));
                }
                shape[axis] = offset;
                self.push(
                    Kind::Concat {
                        axis,
                        parts: placed,
                    },
                    shape,
                )
            }
            Expr::Transmute { src, to } => match src.as_ref() {
                Expr::Src(name) => {
                    let leaf = self.intern(Leaf::Checkpoint(name.clone()));
                    self.push(Kind::Leaf(leaf), to.shape.clone())
                }
                Expr::Out(name) => {
                    let leaf = self.intern(Leaf::Contract(name.clone()));
                    self.push(Kind::Leaf(leaf), to.shape.clone())
                }
                _ => {
                    let inner = self.build(src)?;
                    self.push(Kind::Transmute { src: inner }, to.shape.clone())
                }
            },
            Expr::Fill { ty, .. } => self.push(Kind::Fill, ty.shape.clone()),
            Expr::Repack { .. }
            | Expr::Cast { .. }
            | Expr::Scale { .. }
            | Expr::Bias { .. }
            | Expr::Unary { .. } => Err(Error::Contract(format!(
                "{} needs a kernel and cannot be lowered to byte runs",
                expr.node_name()
            ))),
            Expr::Shard { .. } => Err(Error::Internal(
                "Shard reached lowering; Resolver::specialize rewrites it into \
                 this rank's Slice, and byte offsets cannot be symbolic"
                    .to_string(),
            )),
            Expr::SrcIndexed(template) => Err(Error::Internal(format!(
                "SrcIndexed('{template}') reached lowering; Resolver::specialize \
                 substitutes this instance's index, and a tensor name cannot be \
                 symbolic"
            ))),
            Expr::Select { .. } => Err(Error::Internal(
                "Select reached lowering; Resolver::specialize rewrites it into \
                 this instance's Slice, and byte offsets cannot be symbolic"
                    .to_string(),
            )),
        }
    }

    fn lower(&mut self, root: usize, max_runs: usize) -> Result<Lowering, Error> {
        if let Some(runs) = self.walk(root, max_runs)? {
            return Ok(Lowering::Copy(CopyList {
                leaves: std::mem::take(&mut self.leaves),
                runs,
                elements: self.nodes[root].elements,
            }));
        }
        let Some(gather) = self.gather(root)? else {
            return Err(Error::Contract(format!(
                "expression breaks into more than {max_runs} contiguous \
                 stretches; only a Gather over a whole tensor has a lowering \
                 that is not a copy list, and this is not one"
            )));
        };
        Ok(gather)
    }

    fn gather(&mut self, root: usize) -> Result<Option<Lowering>, Error> {
        let Kind::Gather { src, axis, indices } = &self.nodes[root].kind else {
            return Ok(None);
        };
        let (src, axis, indices) = (*src, *axis, indices.clone());
        let Kind::Leaf(leaf) = self.nodes[src].kind else {
            return Ok(None);
        };
        let node = &self.nodes[root];
        let block = node.strides[axis];
        let dst_row = node.shape[axis]
            .checked_mul(block)
            .or_overflow("gather destination row overflows i64")?;
        if dst_row <= 0 {
            return Ok(None);
        }
        let src_row = self.nodes[src].shape[axis]
            .checked_mul(block)
            .or_overflow("gather source row overflows i64")?;
        Ok(Some(Lowering::Gather(GatherList {
            leaves: std::mem::take(&mut self.leaves),
            leaf,
            indices,
            block,
            rows: node.elements / dst_row,
            src_row,
            elements: node.elements,
        })))
    }

    fn walk(&mut self, root: usize, max_runs: usize) -> Result<Option<Vec<Run>>, Error> {
        let total = self.nodes[root].elements;
        let mut runs: Vec<Run> = Vec::new();
        let mut flat = 0_i64;
        while flat < total {
            let mut coord = Coord::default();
            unflatten(flat, &self.nodes[root].shape, &mut coord);
            let (found, span) = self.step(root, &coord, flat)?;
            let span = span.clamp(1, total - flat);
            let source = match found {
                Some((leaf, src_elem)) => RunSource::Leaf { leaf, src_elem },
                None => RunSource::Zero,
            };
            match runs.last_mut() {
                Some(last) if adjacent(last, source, flat) => last.len += span,
                _ => {
                    if runs.len() >= max_runs {
                        return Ok(None);
                    }
                    runs.push(Run {
                        source,
                        dst_elem: flat,
                        len: span,
                    });
                }
            }
            flat += span;
        }
        Ok(Some(runs))
    }

    fn step(
        &self,
        index: usize,
        coord: &Coord,
        flat: i64,
    ) -> Result<(Option<(usize, i64)>, i64), Error> {
        let node = &self.nodes[index];
        let remaining = node.elements - flat;
        match &node.kind {
            Kind::Leaf(leaf) => Ok((Some((*leaf, flat)), remaining)),
            Kind::Slice {
                src,
                axis,
                start,
                whole,
            } => {
                let mut inner = *coord;
                inner.dims[*axis] = start + coord.dims[*axis];
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                let limit = if *whole || *axis == 0 {
                    i64::MAX
                } else {
                    distance_to(node, coord, flat, *axis - 1, coord.dims[*axis - 1] + 1)
                };
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Stride {
                src,
                axis,
                start,
                step,
            } => {
                let mut inner = *coord;
                inner.dims[*axis] = start + coord.dims[*axis] * step;
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                let limit = distance_to(node, coord, flat, *axis, coord.dims[*axis] + 1);
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Gather { src, axis, indices } => {
                let at = coord.dims[*axis] as usize;
                let mut inner = *coord;
                inner.dims[*axis] = *indices.get(at).ok_or_else(|| {
                    Error::Internal(format!("Gather has no index for position {at}"))
                })?;
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                let mut end = at + 1;
                while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                    end += 1;
                }
                let limit = distance_to(node, coord, flat, *axis, end as i64);
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Concat { axis, parts } => {
                let at = coord.dims[*axis];
                let (offset, part) = parts
                    .iter()
                    .rev()
                    .find(|(offset, _)| *offset <= at)
                    .copied()
                    .ok_or_else(|| {
                        Error::Internal(format!("Concat has no part covering index {at}"))
                    })?;
                let mut inner = *coord;
                inner.dims[*axis] = at - offset;
                let inner_flat = flatten(&inner, &self.nodes[part].shape);
                let (found, span) = self.step(part, &inner, inner_flat)?;
                let part_end = offset + self.nodes[part].shape[*axis];
                let limit = distance_to(node, coord, flat, *axis, part_end);
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Transmute { src } => {
                let mut inner = Coord::default();
                unflatten(flat, &self.nodes[*src].shape, &mut inner);
                let (found, span) = self.step(*src, &inner, flat)?;
                Ok((found, span.min(remaining)))
            }
            Kind::Fill => Ok((None, remaining)),
        }
    }
}

fn axis_extent(shape: &[i64], axis: usize, node: &str) -> Result<i64, Error> {
    shape.get(axis).copied().ok_or_else(|| {
        Error::Internal(format!(
            "{node} axis {axis} escaped the type checker (operand rank {})",
            shape.len()
        ))
    })
}

fn adjacent(last: &Run, source: RunSource, flat: i64) -> bool {
    if last.dst_elem + last.len != flat {
        return false;
    }
    match (last.source, source) {
        (RunSource::Zero, RunSource::Zero) => true,
        (
            RunSource::Leaf { leaf, src_elem },
            RunSource::Leaf {
                leaf: next,
                src_elem: next_elem,
            },
        ) => leaf == next && src_elem + last.len == next_elem,
        _ => false,
    }
}

fn distance_to(node: &Node, coord: &Coord, flat: i64, axis: usize, target: i64) -> i64 {
    let stride = node.strides[axis];
    (target - coord.dims[axis]) * stride - flat.rem_euclid(stride)
}

#[derive(Clone, Copy)]
struct Coord {
    dims: [i64; MAX_RANK],
}

impl Default for Coord {
    fn default() -> Self {
        Self {
            dims: [0; MAX_RANK],
        }
    }
}

fn unflatten(flat: i64, shape: &[i64], out: &mut Coord) {
    let mut rest = flat;
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis].max(1);
        out.dims[axis] = rest % extent;
        rest /= extent;
    }
}

fn flatten(coord: &Coord, shape: &[i64]) -> i64 {
    let mut flat = 0_i64;
    for (axis, extent) in shape.iter().enumerate() {
        flat = flat * extent + coord.dims[axis];
    }
    flat
}
