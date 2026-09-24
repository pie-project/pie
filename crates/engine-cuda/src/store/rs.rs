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
        let Operation::Attention(Attention::SsmGatedDeltaChunked {
            qkv, z, gates, y, ..
        }) = &node.op
        else {
            continue;
        };
        let mut total = row_bytes(*qkv)? + row_bytes(*z)? + row_bytes(*gates)? + row_bytes(*y)?;
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
        widest = widest.max(total);
    }
    Ok(widest)
}

/// Why a plan `read` reserves nothing for still has a recurrence a lane may ask
/// to buffer: the pair this shell buffers is the gated-delta one, and a plan
/// whose chunked recurrence is KDA (as `glm5_next` runs it) is named as such,
/// so the refusal says what is missing rather than that nothing was declared.
pub fn unbuffered(trace: &Trace) -> Option<String> {
    trace.nodes.iter().enumerate().find_map(|(at, node)| {
        matches!(
            node.op,
            Operation::Attention(Attention::SsmKdaChunked { .. })
        )
        .then(|| {
            format!(
                "its chunked recurrence at node {at} is KDA (`SsmKdaChunked`), and this shell \
                 buffers the gated-delta pair only (`SsmCausalConv1dChunked` into \
                 `SsmGatedDeltaChunked` with an `SsmGdnPrep`)"
            )
        })
    })
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
            let (qkv, gates, chunked) = match &node.op {
                Operation::Attention(Attention::SsmGatedDeltaChunked { qkv, gates, .. }) => {
                    (*qkv, *gates, true)
                }
                Operation::Attention(Attention::SsmGatedDelta { qkv, gates, .. }) => {
                    (*qkv, *gates, false)
                }
                _ => continue,
            };
            let conv = defined
                .get(&qkv.0)
                .copied()
                .and_then(|n| match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. })
                        if chunked =>
                    {
                        Some((n, *x))
                    }
                    Operation::Attention(Attention::SsmCausalConv1d { x, .. }) if !chunked => {
                        Some((n, *x))
                    }
                    _ => None,
                });
            let prep = defined
                .get(&gates.0)
                .copied()
                .and_then(|n| match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmGdnPrep { ba, .. }) => Some((n, *ba)),
                    _ => None,
                });
            let (Some((conv_at, x)), Some((prep_at, ba))) = (conv, prep) else {
                if !chunked {
                    continue;
                }
                return Err(Fault::Unbound {
                    what: format!(
                        "the chunked recurrence at node {at}, whose `qkv` and `gates` are not \
                         written by a chunked conv and a gate prep — this shell buffers the \
                         two in-projection planes those ops read and knows no third shape"
                    ),
                });
            };
            if conv_at >= at || prep_at >= at {
                return Err(Fault::Unbound {
                    what: format!(
                        "the recurrence at node {at} reads planes written at nodes \
                         {conv_at} and {prep_at}, which do not both stand before it"
                    ),
                });
            }
            let qkv_width =
                width_of(trace, x).ok_or_else(|| unsized_plane("the conv's rows", x))?;
            let ba_width = width_of(trace, ba)
                .ok_or_else(|| unsized_plane("the gate prep's `[b | a]`", ba))?;
            let page = u64::from(paging);
            let layer = if chunked { layers } else { decode_layers };
            planes.insert(
                x.0,
                Plane {
                    layer,
                    at: 0,
                    width: qkv_width,
                },
            );
            planes.insert(
                ba.0,
                Plane {
                    layer,
                    at: page * qkv_width,
                    width: ba_width,
                },
            );
            let here = qkv_width + ba_width;
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
            "{what} at value {}, whose row width is not a constant this shell can reserve a \
             buffered page for",
            id.0
        ),
    }
}

fn width_of(trace: &Trace, id: ValueId) -> Option<u64> {
    let decl = trace.values.get(id.0 as usize)?;
    let Ty::Tensor { shape, .. } = &decl.ty else {
        return None;
    };
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
    use model_ir::{Guard, Node, Platform};

    use super::*;

    fn plan(nodes: Vec<Operation>) -> Trace {
        Trace {
            name: "rs".to_string(),
            platform: Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
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

    /// A KDA plan reserves no buffered planes, and the refusal a lane's
    /// recurrent verb then meets names the recurrence it has rather than
    /// saying none was declared.
    #[test]
    fn a_kda_recurrence_is_named_as_the_one_this_shell_does_not_buffer() {
        let kda = plan(vec![Operation::Attention(Attention::SsmKdaChunked {
            mixed: ValueId(0),
            f: ValueId(1),
            b: ValueId(2),
            dt_bias: ValueId(3),
            a_log: ValueId(4),
            state: ValueId(5),
            heads: 4,
            head_dim: 32,
            norm_eps: 1e-6,
            gate_floor: 0.0,
            y: ValueId(6),
        })]);
        assert!(
            read(&kda, 16)
                .expect("a plan with no gated-delta pair reads clean")
                .is_none()
        );
        let why = unbuffered(&kda).expect("the KDA recurrence is named");
        assert!(why.contains("node 0 is KDA"), "{why}");
        assert_eq!(unbuffered(&plan(Vec::new())), None);
    }
}
