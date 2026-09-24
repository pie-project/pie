use std::collections::HashMap;

use model_ir::{Attention, Dim, Dtype, Operands, Operation, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::device::elastic::Arena;
use crate::error::{Fault, Result};
use crate::store::Pools;
use crate::store::kv::Paging;

pub const PLANE_DTYPE: Dtype = Dtype::Bf16;

const ELEMENT: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pub layer: u32,
    pub at: u64,
    pub width: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Planes(HashMap<u32, Plane>);

impl Planes {
    #[must_use]
    pub fn of(&self, id: ValueId) -> Option<Plane> {
        self.0.get(&id.0).copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Debug)]
pub struct Buffers {
    arena: Arena,
    committed_pages: u32,
    planes: Planes,
    page_tokens: u32,
    slots: u32,
    per_page: u64,
    layers: u32,
    ext_row_bytes: u64,
}

impl Buffers {
    pub fn reserve(trace: &Trace, paging: Paging, pools: &Pools) -> Result<Option<Buffers>> {
        let Some((planes, per_token, layers)) = read(trace, paging.page_size)? else {
            return Ok(None);
        };
        let per_page = u64::from(paging.page_size) * per_token;
        let ext_row_bytes = ext_row_bytes(trace)?;
        let bytes = per_page
            .saturating_mul(u64::from(paging.slots))
            .saturating_mul(u64::from(layers))
            .saturating_mul(ELEMENT);
        Ok(Some(Buffers {
            arena: pools.reserve_arena(bytes, "rs buffered activations")?,
            committed_pages: 0,
            planes,
            page_tokens: paging.page_size,
            slots: paging.slots,
            per_page,
            layers,
            ext_row_bytes,
        }))
    }
}

fn ext_row_bytes(trace: &Trace) -> Result<u64> {
    let mut defined: HashMap<u32, usize> = HashMap::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        for out in outs {
            defined.insert(out.0, at);
        }
    }
    let row_bytes = |id: ValueId| -> Result<u64> {
        let decl = trace.values.get(id.0 as usize);
        let Some(Ty::Tensor { shape, dtype }) = decl.map(|d| &d.ty) else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let Some(Dim::Const(width)) = shape.last() else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let elem = model_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
            what: format!("value {} has a dtype with no element size", id.0),
        })?;
        Ok(*width * elem)
    };
    let mut widest = 0u64;
    for node in &trace.nodes {
        let total = match &node.op {
            Operation::Attention(Attention::SsmGatedDeltaChunked {
                qkv, z, gates, y, ..
            }) => {
                let mut total =
                    row_bytes(*qkv)? + row_bytes(*z)? + row_bytes(*gates)? + row_bytes(*y)?;
                if let Some(Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. })) =
                    defined.get(&qkv.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                if let Some(Operation::Attention(Attention::SsmGdnPrep { ba, .. })) =
                    defined.get(&gates.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*ba)?;
                }
                total
            }
            Operation::Attention(Attention::SsmKdaChunked { mixed, f, b, y, .. }) => {
                let mut total =
                    row_bytes(*mixed)? + row_bytes(*f)? + row_bytes(*b)? + row_bytes(*y)?;
                if let Some(Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. })) =
                    defined.get(&mixed.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                total
            }
            _ => continue,
        };
        widest = widest.max(total);
    }
    Ok(widest)
}

/// The planes a recurrence reads besides the conv's rows: the gated-delta
/// pair's `[b | a]` stands behind an `SsmGdnPrep`, the KDA pair's forget and
/// beta projections are read by the scan itself.
#[derive(Clone, Copy)]
enum Reads {
    GatedDelta {
        qkv: ValueId,
        gates: ValueId,
    },
    Kda {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
    },
}

pub fn read(trace: &Trace, page_tokens: u32) -> Result<Option<(Planes, u64, u32)>> {
    {
        let paging = page_tokens;
        let mut defined: HashMap<u32, usize> = HashMap::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            node.op.outputs(&mut outs);
            for out in outs {
                defined.insert(out.0, at);
            }
        }

        let mut planes: HashMap<u32, Plane> = HashMap::new();
        let mut per_token: Option<u64> = None;
        let mut layers = 0u32;
        let mut decode_layers = 0u32;
        for (at, node) in trace.nodes.iter().enumerate() {
            let (reads, chunked) = match &node.op {
                Operation::Attention(Attention::SsmGatedDeltaChunked { qkv, gates, .. }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                    },
                    true,
                ),
                Operation::Attention(Attention::SsmGatedDelta { qkv, gates, .. }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                    },
                    false,
                ),
                Operation::Attention(Attention::SsmKdaChunked { mixed, f, b, .. }) => (
                    Reads::Kda {
                        mixed: *mixed,
                        f: *f,
                        b: *b,
                    },
                    true,
                ),
                Operation::Attention(Attention::SsmKdaStep { mixed, f, b, .. }) => (
                    Reads::Kda {
                        mixed: *mixed,
                        f: *f,
                        b: *b,
                    },
                    false,
                ),
                _ => continue,
            };
            let written = |id: ValueId| defined.get(&id.0).map(|n| (*n, &trace.nodes[*n].op));
            let conv = match reads {
                Reads::GatedDelta { qkv, .. } | Reads::Kda { mixed: qkv, .. } => written(qkv),
            }
            .and_then(|(n, op)| match op {
                Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. }) if chunked => {
                    Some((n, *x))
                }
                Operation::Attention(Attention::SsmCausalConv1d { x, .. }) if !chunked => {
                    Some((n, *x))
                }
                _ => None,
            });
            // The planes read beside the conv's rows, each with the node that
            // writes it when one does: the gate prep's `[b | a]`, or the KDA
            // scan's forget and beta projections.
            let others: Option<Vec<(&str, ValueId, Option<usize>)>> = match reads {
                Reads::GatedDelta { gates, .. } => written(gates).and_then(|(n, op)| match op {
                    Operation::Attention(Attention::SsmGdnPrep { ba, .. }) => {
                        Some(vec![("the gate prep's `[b | a]`", *ba, Some(n))])
                    }
                    _ => None,
                }),
                Reads::Kda { f, b, .. } => Some(vec![
                    ("the forget projection's rows", f, None),
                    ("the beta projection's rows", b, None),
                ]),
            };
            let (Some((conv_at, x)), Some(others)) = (conv, others) else {
                if !chunked {
                    continue;
                }
                return Err(Fault::Unbound {
                    what: match reads {
                        Reads::GatedDelta { .. } => format!(
                            "the chunked recurrence at node {at}, whose `qkv` and `gates` are \
                             not written by a chunked conv and a gate prep — this shell buffers \
                             the two in-projection planes those ops read and knows no third \
                             shape"
                        ),
                        Reads::Kda { .. } => format!(
                            "the chunked KDA recurrence at node {at}, whose `mixed` is not \
                             written by a chunked conv — this shell buffers the conv's rows \
                             beside the forget and beta projections and knows no other shape"
                        ),
                    },
                });
            };
            if let Some(late) = core::iter::once(conv_at)
                .chain(others.iter().filter_map(|(_, _, wrote)| *wrote))
                .find(|wrote| *wrote >= at)
            {
                return Err(Fault::Unbound {
                    what: format!(
                        "the recurrence at node {at} reads a plane written at node {late}, \
                         which does not stand before it"
                    ),
                });
            }
            let page = u64::from(paging);
            let layer = if chunked { layers } else { decode_layers };
            let mut here = 0u64;
            for (what, id, _) in core::iter::once(("the conv's rows", x, None)).chain(others) {
                let width = width_of(trace, id).ok_or_else(|| unsized_plane(what, id))?;
                planes.insert(
                    id.0,
                    Plane {
                        layer,
                        at: page * here,
                        width,
                    },
                );
                here += width;
            }
            match per_token {
                None => per_token = Some(here),
                Some(first) if first == here => {}
                Some(first) => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "recurrent layer {layer} buffers {here} elements a token where \
                             layer 0 buffers {first} — this pool cuts one page-slot stride \
                             for every layer"
                        ),
                    });
                }
            }
            if chunked {
                layers += 1;
            } else {
                decode_layers += 1;
            }
        }
        if decode_layers != 0 && decode_layers != layers {
            return Err(Fault::Unbound {
                what: format!(
                    "the plan runs {decode_layers} step recurrence(s) and {layers} chunked \
                     one(s); the buffered planes pair the two chains layer by layer"
                ),
            });
        }
        let Some(per_token) = per_token else {
            return Ok(None);
        };
        Ok(Some((Planes(planes), per_token, layers)))
    }
}

impl Buffers {
    #[must_use]
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arena.committed_bytes()
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.per_page
            .saturating_mul(u64::from(self.slots))
            .saturating_mul(u64::from(self.layers))
            .saturating_mul(ELEMENT)
    }

    pub fn ensure(&mut self, pools: &mut Pools, pages: u32) -> Result<()> {
        if pages <= self.committed_pages {
            return Ok(());
        }
        if pages > self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(pages),
                have: u64::from(self.slots),
            });
        }
        let bytes = u64::from(pages)
            .saturating_mul(u64::from(self.layers))
            .saturating_mul(self.per_page)
            .saturating_mul(ELEMENT);
        pools.commit_arena(&mut self.arena, bytes)?;
        self.committed_pages = pages;
        Ok(())
    }

    #[must_use]
    pub fn ext_row_bytes(&self) -> u64 {
        self.ext_row_bytes
    }

    pub fn row(&self, plane: Plane, page_slot: u32, row: u32) -> Result<u64> {
        if page_slot >= self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.slots),
            });
        }
        if row >= self.page_tokens {
            return Err(Fault::Ceiling {
                what: "rs buffer page tokens",
                need: u64::from(row) + 1,
                have: u64::from(self.page_tokens),
            });
        }
        if page_slot >= self.committed_pages {
            return Err(Fault::Ceiling {
                what: "rs buffer pages committed before this fire",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.committed_pages),
            });
        }
        let block = (u64::from(page_slot) * u64::from(self.layers) + u64::from(plane.layer))
            * self.per_page;
        Ok(self.arena.base() + (block + plane.at + u64::from(row) * plane.width) * ELEMENT)
    }
}

fn unsized_plane(what: &str, id: ValueId) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what} at value {}, whose row is not a constant width of {PLANE_DTYPE:?} this \
             shell can reserve a buffered page for",
            id.0
        ),
    }
}

fn width_of(trace: &Trace, id: ValueId) -> Option<u64> {
    let decl = trace.values.get(id.0 as usize)?;
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return None;
    };
    if *dtype != PLANE_DTYPE {
        return None;
    }
    match shape.last()? {
        Dim::Const(width) => Some(*width),
        _ => None,
    }
}

#[derive(Debug)]
pub struct Predicate {
    region: Buffer,
    at_one: u64,
    at_zero: u64,
    at_commits: u64,
    at_indptr: u64,
    at_mask: u64,
    at_len: u64,
    lanes: u32,
}

impl Predicate {
    pub fn reserve(max_lanes: u32) -> Result<Predicate> {
        let lanes = u64::from(max_lanes);
        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(256);
            here
        };
        let at_one = take(4);
        let at_zero = take(4);
        let at_commits = take(lanes * 8);
        let at_indptr = take((lanes + 1) * 4);
        let at_mask = take(lanes);
        let at_len = take(lanes * 4);
        let mut region = Buffer::zeroed(usize::try_from(at).unwrap_or(usize::MAX))?;
        region.write(at_one, &1u32.to_le_bytes())?;
        let identity: Vec<u8> = (0..=max_lanes)
            .flat_map(|l| (l as i32).to_le_bytes())
            .collect();
        region.write(at_indptr, &identity)?;
        Ok(Predicate {
            region,
            at_one,
            at_zero,
            at_commits,
            at_indptr,
            at_mask,
            at_len,
            lanes: max_lanes,
        })
    }

    #[must_use]
    pub fn always(&self) -> u64 {
        self.region.ptr() + self.at_one
    }

    #[must_use]
    pub fn never(&self) -> u64 {
        self.region.ptr() + self.at_zero
    }

    pub fn write(
        &mut self,
        stream: *mut core::ffi::c_void,
        commits: &[u64],
        lens: &[i32],
    ) -> Result<()> {
        if commits.len() as u64 > u64::from(self.lanes) {
            return Err(Fault::Ceiling {
                what: "rs fold predicates",
                need: commits.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let words: Vec<u8> = commits.iter().flat_map(|c| c.to_le_bytes()).collect();
        let at_commits = self.at_commits;
        let at_len = self.at_len;
        self.region.stage(stream, at_commits, &words)?;
        let lens: Vec<u8> = lens.iter().flat_map(|n| n.to_le_bytes()).collect();
        self.region.stage(stream, at_len, &lens)
    }

    #[must_use]
    pub fn commits(&self) -> u64 {
        self.region.ptr() + self.at_commits
    }

    #[must_use]
    pub fn indptr(&self) -> u64 {
        self.region.ptr() + self.at_indptr
    }

    #[must_use]
    pub fn mask(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_mask, lanes, 1, Dtype::U8)
    }

    #[must_use]
    pub fn commit_len(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_len, lanes, 1, Dtype::I32)
    }

    pub fn read_mask(&self, lanes: u32) -> Result<Vec<u8>> {
        let mut out = vec![0u8; lanes as usize];
        self.region.read(self.at_mask, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use model_ir::{Def, Guard, Node, Platform, ValueDecl};

    use super::*;

    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 32;
    const PAGE: u32 = 16;

    fn row(width: u64, dtype: Dtype) -> ValueDecl {
        ValueDecl {
            def: Def::Op(0),
            ty: Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(width)],
                dtype,
            },
        }
    }

    /// One KDA layer as `glm5_next` and `kimi_k3` state it: the conv's rows
    /// (0) through the chunked conv into `mixed` (3), which the scan reads
    /// beside its forget (4) and beta (5) projections, landing `y` (9).
    fn kda_values() -> Vec<ValueDecl> {
        let wide = HEADS * HEAD_DIM;
        vec![
            row(3 * wide, Dtype::Bf16),
            row(4, Dtype::Bf16),
            row(4, Dtype::Bf16),
            row(3 * wide, Dtype::Bf16),
            row(wide, Dtype::Bf16),
            row(HEADS, Dtype::Bf16),
            row(wide, Dtype::F32),
            row(HEADS, Dtype::F32),
            row(wide, Dtype::F32),
            row(wide, Dtype::F32),
        ]
    }

    fn kda_chunked(mixed: u32) -> Operation {
        Operation::Attention(Attention::SsmKdaChunked {
            mixed: ValueId(mixed),
            f: ValueId(4),
            b: ValueId(5),
            dt_bias: ValueId(6),
            a_log: ValueId(7),
            state: ValueId(8),
            heads: HEADS as u32,
            head_dim: HEAD_DIM as u32,
            norm_eps: 1e-6,
            gate_floor: 0.0,
            y: ValueId(9),
        })
    }

    fn plan(values: Vec<ValueDecl>, nodes: Vec<Operation>) -> Trace {
        Trace {
            name: "rs".to_string(),
            platform: Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values,
            nodes: nodes
                .into_iter()
                .map(|op| Node {
                    op,
                    guard: Guard::Always,
                    layer: None,
                })
                .collect(),
            seams: Vec::new(),
            drafter: None,
        }
    }

    /// A KDA layer buffers three planes a token — the conv's rows, then the
    /// forget and beta projections — laid page by page behind one another,
    /// and its extended run stages those beside `mixed` and `y`.
    #[test]
    fn a_kda_recurrence_buffers_its_conv_rows_beside_its_two_projections() {
        let wide = HEADS * HEAD_DIM;
        let trace = plan(
            kda_values(),
            vec![
                Operation::Attention(Attention::SsmCausalConv1dChunked {
                    x: ValueId(0),
                    weight: ValueId(1),
                    state: ValueId(2),
                    conv_width: 4,
                    dilation: 1,
                    y: ValueId(3),
                }),
                kda_chunked(3),
            ],
        );
        let (planes, per_token, layers) = read(&trace, PAGE)
            .expect("the KDA pair reads clean")
            .expect("the KDA pair reserves buffered planes");
        assert_eq!((layers, per_token), (1, 3 * wide + wide + HEADS));
        assert_eq!(planes.len(), 3);
        let page = u64::from(PAGE);
        let plane = |at: u64, width: u64| {
            Some(Plane {
                layer: 0,
                at,
                width,
            })
        };
        assert_eq!(planes.of(ValueId(0)), plane(0, 3 * wide));
        assert_eq!(planes.of(ValueId(4)), plane(page * 3 * wide, wide));
        assert_eq!(planes.of(ValueId(5)), plane(page * 4 * wide, HEADS));
        assert_eq!(
            ext_row_bytes(&trace).expect("the extended run's rows size"),
            2 * (3 * wide) + 2 * (3 * wide) + 2 * wide + 2 * HEADS + 4 * wide
        );
    }

    /// A chunked KDA scan whose `mixed` no chunked conv writes has no plane
    /// this shell knows to buffer, and the refusal names the scan.
    #[test]
    fn a_kda_recurrence_without_its_conv_is_refused_by_name() {
        let trace = plan(kda_values(), vec![kda_chunked(0)]);
        let why = read(&trace, PAGE).expect_err("no conv writes `mixed`");
        assert!(
            why.to_string().contains("chunked KDA recurrence at node 0"),
            "{why}"
        );
        assert!(
            read(&plan(Vec::new(), Vec::new()), PAGE)
                .expect("an empty plan reads clean")
                .is_none()
        );
    }
}
