use std::collections::HashMap;

use poem_ir::{Dim, Dtype, Operands, Operation, Ssm, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::RsScratch;
use crate::store::Pools;

pub const PLANE_DTYPE: Dtype = Dtype::Bf16;

const ELEMENT: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pub layer: u32,
    pub at: u64,
    pub width: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Planes(
    HashMap<u32, Plane>,
    std::collections::HashSet<u32>,
    HashMap<u32, StepGates>,
    HashMap<u32, Corrected>,
    HashMap<u32, (Plane, u32, u32)>,
);

/// A decode step's gate prep, by its output: the step's `q k v` (whose keys
/// and queries its buffered rows' attention weights are dotted against) and
/// its key heads' count and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepGates {
    pub qkv: ValueId,
    pub k_heads: u32,
    pub k_dim: u32,
}

/// A gated-delta or KDA layer's corrected planes
/// (`ChunkProfile::rs_corrected`): each buffered row's keys (the conv
/// output's key heads, bf16) and its corrected values (`u = beta (v - g
/// S^T k)`, the value its delta rule writes, f32: `width` counts bf16
/// elements, two a value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Corrected {
    pub k: Plane,
    pub u: Plane,
}

impl Planes {
    #[must_use]
    pub fn of(&self, id: ValueId) -> Option<Plane> {
        self.0.get(&id.0).copied()
    }

    /// Every buffered plane once, by layer and place: the read values', a
    /// gated-delta or KDA layer's keys and corrected values, and a Mamba-2
    /// layer's `x | B`.
    #[must_use]
    pub fn all(&self) -> Vec<Plane> {
        let mut planes: Vec<Plane> = self
            .0
            .values()
            .copied()
            .chain(self.3.values().flat_map(|c| [c.k, c.u]))
            .chain(self.4.values().map(|(k, ..)| *k))
            .collect();
        planes.sort_by_key(|plane| (plane.layer, plane.at, plane.width));
        planes.dedup_by_key(|plane| (plane.layer, plane.at, plane.width));
        planes
    }

    /// Whether `id` is a lone conv's rows, a layer of its own.
    #[must_use]
    pub fn lone(&self, id: ValueId) -> bool {
        self.1.contains(&id.0)
    }

    /// Whether `gates` is a decode step's gate prep output: its recurrence
    /// lands the step's `[b | a]` in the buffer itself (`dispatch::ssm`).
    /// (A plane's value may be read by both forms of a layer, the two arms
    /// of one branch; the gates are each form's own.)
    #[must_use]
    pub fn step_gates(&self, gates: ValueId) -> Option<StepGates> {
        self.2.get(&gates.0).copied()
    }

    /// The corrected planes of the gated-delta or KDA recurrence landing
    /// `y`.
    #[must_use]
    pub fn corrected(&self, y: ValueId) -> Option<Corrected> {
        self.3.get(&y.0).copied()
    }

    /// Where a decode conv landing `y` (a recurrence's convolved rows) lands
    /// the channels of its row the recurrence reads a buffered row by (a
    /// gated-delta or KDA layer's keys, a Mamba-2 layer's `x | B`): their
    /// plane, the channel of `y` they start at, and their count.
    #[must_use]
    pub fn keys_of(&self, y: ValueId) -> Option<(Plane, u32, u32)> {
        self.4.get(&y.0).copied()
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

/// The extended run's scratch: a run of the activation's span, a fire's
/// rows and the buffered rows its lanes replay ahead of them, a row each
/// at the widest recurrence's width. The contract's floor is a rung's
/// tokens of each ([`::engine::fire::rung_holds`]); this engine lays the
/// widest bucket's of each ([`Buffers::rows_ext`]), whatever the fire's
/// bucket. Its reach is that ([`Buffers::lay_reach`]), backed when the
/// first fire replays.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    /// Where it starts in the activation's span.
    at: u64,
    reach: u64,
    held: u64,
}

/// The rs buffer: the rows a lane lands past its state before they fold. Its
/// pages are slots of the state pool's rs list ([`Pools::rs_span`]): a page
/// holds `tokens` rows of every recurrent layer's planes, a layer's rows a
/// run of the slot, each buffered plane `tokens` rows of it.
#[derive(Debug)]
pub struct Buffers {
    scratch: Scratch,
    planes: Planes,
    /// The rows a page holds.
    tokens: u32,
    /// Where each buffered layer's run starts in a page's slot.
    rows: Vec<u64>,
    ext_row_bytes: u64,
    /// The room a gated-delta decode step splits a lane's carried rows in
    /// (`kernels_cuda::ssm::carry_split_bytes`), zeroed at the load: its
    /// counters return to zero after each step.
    split: Option<crate::device::Buffer>,
}

impl Buffers {
    /// The device bytes the extended run's scratch has backed.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch.held
    }

    /// The buffered planes of `trace`, laid as the pool's rs list lays its
    /// pages, and the scratch in the activation's span, widened by what a
    /// fire asks: the widest bucket's `max_rows` rows and as many replayed
    /// ([`Buffers::rows_ext`]).
    pub fn reserve(trace: &Trace, max_rows: u32, pools: &Pools) -> Result<Option<Buffers>> {
        let layout = pools.rs_layout();
        let tokens = layout.as_ref().map_or(1, |(tokens, _)| (*tokens).max(1));
        let Some((planes, widths)) = read(trace, tokens)? else {
            return Ok(None);
        };
        // The list's runs, a buffered layer's each, in the layers' order;
        // the runs after them buffer what this engine does not (an n-gram
        // window's ids).
        let held = layout.map_or_else(Vec::new, |(_, runs)| runs);
        let fits = held.len() >= widths.len()
            && held
                .iter()
                .zip(&widths)
                .all(|(&(_, bytes), width)| bytes == width * ELEMENT);
        if !fits {
            return Err(Fault::program(
                "store::rs",
                format!(
                    "the plan buffers {} layer(s) of {:?} bytes a token, and the chunk recipe's \
                     rs list lays runs of {held:?}",
                    widths.len(),
                    widths.iter().map(|w| w * ELEMENT).collect::<Vec<_>>()
                ),
            ));
        }
        let rows: Vec<u64> = held
            .iter()
            .take(widths.len())
            .map(|&(offset, _)| offset)
            .collect();
        // A beam's fork copies a page's rows plane by plane
        // (`Buffers::page_planes`, BEAM-PAGE-LISTS.md §11.2): the planes must
        // tile each buffered layer's row, or a fork leaves a plane's rows
        // stale.
        for (layer, &width) in widths.iter().enumerate() {
            let spanned: u64 = planes
                .all()
                .iter()
                .filter(|plane| plane.layer as usize == layer)
                .map(|plane| plane.width)
                .sum();
            if spanned != width {
                return Err(Fault::program(
                    "store::rs",
                    format!(
                        "recurrent layer {layer} buffers {width} elements a token and its \
                         planes span {spanned}: a beam's fork would copy only those"
                    ),
                ));
            }
        }
        let ext_row_bytes = ext_row_bytes(trace)?;
        let reach = RsScratch::need(Self::rows_ext(max_rows), ext_row_bytes);
        #[cfg(feature = "cuda")]
        let split = kernels_cuda::linear::gemm::multiprocessors()
            .map(|sms| crate::device::Buffer::zeroed(kernels_cuda::ssm::carry_split_bytes(sms) as usize))
            .transpose()?;
        #[cfg(not(feature = "cuda"))]
        let split = None;
        Ok(Some(Buffers {
            scratch: Scratch {
                at: 0,
                reach,
                held: 0,
            },
            planes,
            tokens,
            rows,
            ext_row_bytes,
            split,
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
        let elem = poem_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
            what: format!("value {} has a dtype with no element size", id.0),
        })?;
        Ok(*width * elem)
    };
    let lone: std::collections::HashSet<usize> = poem_compiler::rs::buffered_convs(trace, true)
        .filter(|conv| conv.feeds == poem_compiler::rs::Feeds::Alone)
        .map(|conv| conv.node)
        .collect();
    // A recurrence's operands and a lone conv's are laid side by side in the
    // extended run (`run::RsScratch` keeps the two kinds apart): the widest
    // of each.
    let mut widest = 0u64;
    let mut widest_lone = 0u64;
    for (at, node) in trace.nodes.iter().enumerate() {
        let total = match &node.op {
            // A lone conv runs extended over its own rows when a lane replays.
            Operation::Ssm(
                Ssm::CausalConv1dChunked { x, y, .. } | Ssm::CausalConv1d { x, y, .. },
            ) if lone.contains(&at) => {
                widest_lone = widest_lone.max(row_bytes(*x)? + row_bytes(*y)?);
                continue;
            }
            // A one-token class's recurrence pairs a chunked one at its
            // widths; its decode step reads its carried rows itself
            // (`dispatch::ssm`), its conv's carried outputs a plane of the
            // run.
            Operation::Ssm(
                Ssm::GatedDeltaChunked {
                    qkv, z, gates, y, ..
                }
                | Ssm::GatedDelta {
                    qkv, z, gates, y, ..
                },
            ) => {
                let mut total =
                    row_bytes(*qkv)? + row_bytes(*z)? + row_bytes(*gates)? + row_bytes(*y)?;
                if let Some(Operation::Ssm(
                    Ssm::CausalConv1dChunked { x, .. } | Ssm::CausalConv1d { x, .. },
                )) = defined.get(&qkv.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                if let Some(Operation::Ssm(Ssm::GdnGates { ba, .. })) =
                    defined.get(&gates.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*ba)?;
                }
                total
            }
            Operation::Ssm(
                Ssm::KdaChunked { mixed, f, b, y, .. } | Ssm::Kda { mixed, f, b, y, .. },
            ) => {
                let mut total =
                    row_bytes(*mixed)? + row_bytes(*f)? + row_bytes(*b)? + row_bytes(*y)?;
                if let Some(Operation::Ssm(
                    Ssm::CausalConv1dChunked { x, .. } | Ssm::CausalConv1d { x, .. },
                )) = defined.get(&mixed.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                total
            }
            Operation::Ssm(
                Ssm::Mamba2Chunked { xbc, dt, z, y, .. } | Ssm::Mamba2 { xbc, dt, z, y, .. },
            ) => {
                let mut total =
                    row_bytes(*xbc)? + row_bytes(*dt)? + row_bytes(*z)? + row_bytes(*y)?;
                if let Some(Operation::Ssm(
                    Ssm::CausalConv1dChunked { x, .. } | Ssm::CausalConv1d { x, .. },
                )) = defined.get(&xbc.0).map(|n| &trace.nodes[*n].op)
                {
                    total += row_bytes(*x)?;
                }
                total
            }
            _ => continue,
        };
        widest = widest.max(total);
    }
    Ok(widest + widest_lone)
}

/// The planes a recurrence reads besides the conv's rows: the gated-delta
/// pair's `[b | a]` stands behind an `SsmGdnPrep`, the KDA pair's forget and
/// beta projections are read by the scan itself, as a Mamba-2 scan reads its
/// step sizes.
#[derive(Clone, Copy)]
enum Reads {
    Mamba2 {
        xbc: ValueId,
        dt: ValueId,
        /// The elements of its `x | B`: its `xbc` but the `C`.
        xb: u64,
    },
    GatedDelta {
        qkv: ValueId,
        gates: ValueId,
        /// Its output, which names its corrected planes, and the elements
        /// a row of its keys and of its values.
        y: ValueId,
        keys: u64,
        values: u64,
        key_head: u32,
    },
    Kda {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        /// Its output, which names its corrected planes, and the elements
        /// a row of its keys (and of its values).
        y: ValueId,
        width: u64,
    },
}

/// The buffered planes of `trace` and the elements a token each buffered
/// layer lays, in the order the chunk recipe's rs list lays their runs: the
/// recurrences and the lone convs (`poem_compiler::rs::buffered_convs`,
/// `Feeds::Alone`, Qwen4's PLE conv) in node order. A recurrence buffers its
/// conv's rows and the planes it reads beside them; a lone conv its own
/// input. A layer's prompt and step forms pair in that order, at one width.
pub fn read(trace: &Trace, page_tokens: u32) -> Result<Option<(Planes, Vec<u64>)>> {
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
        let mut step_gates = HashMap::new();
        let mut corrected = HashMap::new();
        let mut keys_of = HashMap::new();
        // The elements a token of each layer, prompt forms and step forms.
        let mut widths: Vec<u64> = Vec::new();
        let mut decode_widths: Vec<u64> = Vec::new();
        let mut layers = 0u32;
        let mut decode_layers = 0u32;
        // CUDA replays Mamba-2, so a conv feeding its scan is buffered.
        let lone: HashMap<usize, (ValueId, bool)> = poem_compiler::rs::buffered_convs(trace, true)
            .filter(|conv| conv.feeds == poem_compiler::rs::Feeds::Alone)
            .map(|conv| (conv.node, (conv.x, conv.chunked)))
            .collect();
        for (at, node) in trace.nodes.iter().enumerate() {
            if let Some(&(x, chunked)) = lone.get(&at) {
                let width =
                    width_of(trace, x).ok_or_else(|| unsized_plane("a lone conv's rows", x))?;
                let layer = if chunked { layers } else { decode_layers };
                planes.insert(
                    x.0,
                    Plane {
                        layer,
                        at: 0,
                        width,
                    },
                );
                if chunked {
                    widths.push(width);
                    layers += 1;
                } else {
                    decode_widths.push(width);
                    decode_layers += 1;
                }
                continue;
            }
            // The fused decode step reads the planes its primitives would:
            // the conv's rows (the split it lands), the gate prep's
            // `[b | a]`, and the recurrence's keys and corrected values.
            if let Operation::Fused(poem_ir::Fused::GdnStep {
                qkv,
                ba,
                conv,
                gates,
                y,
                k_heads,
                k_dim,
                v_heads,
                v_dim,
                ..
            }) = &node.op
            {
                let page = u64::from(paging);
                let mut here = 0u64;
                for (what, id) in [
                    ("the conv's rows", *qkv),
                    ("the gate prep's `[b | a]`", *ba),
                ] {
                    let width = width_of(trace, id).ok_or_else(|| unsized_plane(what, id))?;
                    planes.insert(
                        id.0,
                        Plane {
                            layer: decode_layers,
                            at: page * here,
                            width,
                        },
                    );
                    here += width;
                }
                let keys = u64::from(k_heads * k_dim);
                let values = u64::from(v_heads * v_dim);
                let k = Plane {
                    layer: decode_layers,
                    at: page * here,
                    width: keys,
                };
                here += keys;
                let u = Plane {
                    layer: decode_layers,
                    at: page * here,
                    width: 2 * values,
                };
                here += 2 * values;
                corrected.insert(y.0, Corrected { k, u });
                let count = u32::try_from(keys).unwrap_or(u32::MAX);
                keys_of.insert(conv.0, (k, count, count));
                step_gates.insert(
                    gates.0,
                    StepGates {
                        qkv: *conv,
                        k_heads: *k_heads,
                        k_dim: *k_dim,
                    },
                );
                decode_widths.push(here);
                decode_layers += 1;
                continue;
            }
            let (reads, chunked) = match &node.op {
                Operation::Ssm(Ssm::GatedDeltaChunked {
                    qkv,
                    gates,
                    y,
                    k_heads,
                    k_dim,
                    v_heads,
                    v_dim,
                    ..
                }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                        y: *y,
                        keys: u64::from(k_heads * k_dim),
                        values: u64::from(v_heads * v_dim),
                        key_head: *k_dim,
                    },
                    true,
                ),
                Operation::Ssm(Ssm::GatedDelta {
                    qkv,
                    gates,
                    y,
                    k_heads,
                    k_dim,
                    v_heads,
                    v_dim,
                    ..
                }) => (
                    Reads::GatedDelta {
                        qkv: *qkv,
                        gates: *gates,
                        y: *y,
                        keys: u64::from(k_heads * k_dim),
                        values: u64::from(v_heads * v_dim),
                        key_head: *k_dim,
                    },
                    false,
                ),
                Operation::Ssm(
                    Ssm::KdaChunked {
                        mixed,
                        f,
                        b,
                        y,
                        heads,
                        head_dim,
                        ..
                    }
                    | Ssm::Kda {
                        mixed,
                        f,
                        b,
                        y,
                        heads,
                        head_dim,
                        ..
                    },
                ) => (
                    Reads::Kda {
                        mixed: *mixed,
                        f: *f,
                        b: *b,
                        y: *y,
                        width: u64::from(heads * head_dim),
                    },
                    matches!(node.op, Operation::Ssm(Ssm::KdaChunked { .. })),
                ),
                Operation::Ssm(
                    Ssm::Mamba2Chunked {
                        xbc,
                        dt,
                        groups,
                        state_dim,
                        ..
                    }
                    | Ssm::Mamba2 {
                        xbc,
                        dt,
                        groups,
                        state_dim,
                        ..
                    },
                ) => {
                    let xb = width_of(trace, *xbc)
                        .ok_or_else(|| unsized_plane("a Mamba-2 layer's `xbc`", *xbc))?
                        .saturating_sub(u64::from(groups * state_dim));
                    (
                        Reads::Mamba2 {
                            xbc: *xbc,
                            dt: *dt,
                            xb,
                        },
                        matches!(node.op, Operation::Ssm(Ssm::Mamba2Chunked { .. })),
                    )
                }
                _ => continue,
            };
            let written = |id: ValueId| defined.get(&id.0).map(|n| (*n, &trace.nodes[*n].op));
            let conv = match reads {
                Reads::GatedDelta { qkv, .. }
                | Reads::Kda { mixed: qkv, .. }
                | Reads::Mamba2 { xbc: qkv, .. } => written(qkv),
            }
            .and_then(|(n, op)| match op {
                Operation::Ssm(Ssm::CausalConv1dChunked { x, .. }) if chunked => Some((n, *x)),
                Operation::Ssm(Ssm::CausalConv1d { x, .. }) if !chunked => Some((n, *x)),
                _ => None,
            });
            // The planes read beside the conv's rows, each with the node that
            // writes it when one does: the gate prep's `[b | a]`, or the KDA
            // scan's forget and beta projections.
            let others: Option<Vec<(&str, ValueId, Option<usize>)>> = match reads {
                Reads::GatedDelta { gates, .. } => written(gates).and_then(|(n, op)| match op {
                    Operation::Ssm(Ssm::GdnGates { ba, .. }) => {
                        Some(vec![("the gate prep's `[b | a]`", *ba, Some(n))])
                    }
                    _ => None,
                }),
                Reads::Kda { f, b, .. } => Some(vec![
                    ("the forget projection's rows", f, None),
                    ("the beta projection's rows", b, None),
                ]),
                Reads::Mamba2 { dt, .. } => {
                    Some(vec![("the step sizes", dt, written(dt).map(|(n, _)| n))])
                }
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
                        Reads::Mamba2 { .. } => format!(
                            "the chunked Mamba-2 recurrence at node {at}, whose `xbc` is not \
                             written by a chunked conv — this shell buffers the conv's rows \
                             beside the step sizes and knows no other shape"
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
            if let (
                Reads::GatedDelta {
                    gates,
                    qkv,
                    keys,
                    key_head,
                    ..
                },
                false,
            ) = (reads, chunked)
            {
                step_gates.insert(
                    gates.0,
                    StepGates {
                        qkv,
                        k_heads: u32::try_from(keys).unwrap_or(0) / key_head.max(1),
                        k_dim: key_head,
                    },
                );
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
            // The keys and corrected values after them (a Mamba-2 layer's
            // `x | B`), as the recipe's rs row lays them (`rs_corrected`):
            // with the conv output they are read from, the channel they
            // start at in it, and their count.
            let narrow = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
            let (conv_y, keys, values, k_off) = match reads {
                Reads::GatedDelta {
                    qkv,
                    keys,
                    values,
                    y,
                    ..
                } => (qkv, keys, Some((y, values)), keys),
                Reads::Kda {
                    mixed, width, y, ..
                } => (mixed, width, Some((y, width)), width),
                Reads::Mamba2 { xbc, xb, .. } => (xbc, xb, None, 0),
            };
            let k = Plane {
                layer,
                at: page * here,
                width: keys,
            };
            here += keys;
            keys_of.insert(conv_y.0, (k, narrow(k_off), narrow(keys)));
            if let Some((y, values)) = values {
                let u = Plane {
                    layer,
                    at: page * here,
                    width: 2 * values,
                };
                here += 2 * values;
                corrected.insert(y.0, Corrected { k, u });
            }
            if chunked {
                widths.push(here);
                layers += 1;
            } else {
                decode_widths.push(here);
                decode_layers += 1;
            }
        }
        if decode_layers != 0 && decode_widths != widths {
            return Err(Fault::Unbound {
                what: format!(
                    "the plan's step forms buffer {decode_widths:?} elements a token, layer by \
                     layer, and its chunked forms {widths:?}; the buffered planes pair the two \
                     chains layer by layer, at one width"
                ),
            });
        }
        if widths.is_empty() {
            return Ok(None);
        }
        let lone_rows = lone.values().map(|(x, _)| x.0).collect();
        Ok(Some((Planes(planes, lone_rows, step_gates, corrected, keys_of), widths)))
    }
}

impl Buffers {
    #[must_use]
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    /// The extended rows a fire's run holds, under a widest bucket of
    /// `max_rows` tokens: `2 * max_rows`, its own rows and the buffered
    /// rows it replays, that many of each at most, whatever its bucket. A
    /// fire composes at the bucket of its own rows while it may carry up
    /// to the largest rung's tokens, and the body recorded at that bucket
    /// bakes where each plane of the run lies: so every body lays its
    /// planes over the whole run, more than the contract's floor of the
    /// rung's tokens of each.
    #[must_use]
    pub fn rows_ext(max_rows: u32) -> u32 {
        max_rows.saturating_mul(2)
    }

    /// Backs the scratch for an extended run of `rows_ext` rows, in place.
    pub fn ensure_scratch(&mut self, pools: &mut Pools, rows_ext: u32) -> Result<()> {
        let bytes = RsScratch::need(rows_ext, self.ext_row_bytes);
        if bytes <= self.scratch.held {
            return Ok(());
        }
        if bytes > self.scratch.reach {
            return Err(Fault::Ceiling {
                what: "bytes of rs extended-run scratch",
                need: bytes,
                have: self.scratch.reach,
            });
        }
        // A size-only walk backs nothing (`crate::sizing`), and walks
        // before the scratch has its run: what it asked is the reach, laid
        // and backed with the span ([`Buffers::lay_reach`]).
        if !crate::sizing::active() {
            pools.hold_activation_at(self.scratch.at, bytes, true)?;
        }
        self.scratch.held = bytes;
        Ok(())
    }

    /// Gives the scratch its run of the activation's span, widened by its
    /// reach: where a size-only walk fired the load's compositions
    /// (`walked`), what one of them asked, a fire's extended run being the
    /// widest bucket's whatever its rows; else the same.
    pub fn lay_reach(&mut self, pools: &mut Pools, walked: bool) -> Result<()> {
        if walked {
            self.scratch.reach = self.scratch.held;
        }
        self.scratch.at = pools.widen_activation(self.scratch.reach)?;
        Ok(())
    }

    /// Backs what the scratch held where the range now lies (a fit
    /// reserves it anew).
    pub fn lay_scratch(&self, pools: &mut Pools) -> Result<()> {
        if self.scratch.held == 0 {
            return Ok(());
        }
        pools.hold_activation_at(self.scratch.at, self.scratch.held, true)
    }

    /// The scratch's base and the bytes of it backed.
    #[must_use]
    pub fn scratch(&self, pools: &Pools) -> (u64, u64) {
        (pools.activation_base() + self.scratch.at, self.scratch.held)
    }

    /// A row of every extended operand of one recurrent layer, in bytes.
    #[must_use]
    pub fn ext_row_bytes(&self) -> u64 {
        self.ext_row_bytes
    }

    /// Where a decode step splits a lane's carried rows (0: it splits none).
    #[must_use]
    pub fn split_room(&self) -> u64 {
        self.split.as_ref().map_or(0, crate::device::Buffer::ptr)
    }

    /// Whether a decode step carrying `carried` rows splits them over the
    /// spare blocks: it spans more than one of the carried step's chunks,
    /// on a load with room to split (CUDA's; elsewhere none splits).
    #[must_use]
    pub fn splits(&self, carried: u32) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.split.is_some() && carried > kernels_cuda::ssm::CARRY_CHUNK
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = carried;
            false
        }
    }

    /// The rows an rs buffer page holds.
    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.tokens
    }

    /// Where `plane`'s rows lie in an rs buffer page's slot.
    pub fn place(&self, plane: Plane) -> Result<kernels_cuda::runtime::channel::RsPlane> {
        let run = self
            .rows
            .get(plane.layer as usize)
            .copied()
            .ok_or(Fault::Ceiling {
                what: "recurrent layers the rs buffer holds rows of",
                need: u64::from(plane.layer) + 1,
                have: self.rows.len() as u64,
            })?;
        Ok(kernels_cuda::runtime::channel::RsPlane {
            off: run + plane.at * ELEMENT,
            stride: plane.width * ELEMENT,
        })
    }

    /// Every buffered plane as a page copy spans it: its rows' place in a
    /// page's slot and the bytes a token's row takes (a beam's fork copies a
    /// page's first rows of each, `BEAM-PAGE-LISTS.md` §11).
    pub fn page_planes(&self) -> Result<Vec<kernels_cuda::runtime::channel::PagePlane>> {
        self.planes
            .all()
            .into_iter()
            .map(|plane| {
                let place = self.place(plane)?;
                Ok(kernels_cuda::runtime::channel::PagePlane {
                    offset: place.off,
                    cell: place.stride,
                })
            })
            .collect()
    }

    /// The address of rs buffer page `page`'s slot: a page's rows of a
    /// plane lie at [`Buffers::place`] within it.
    pub fn slot(&self, pools: &Pools, page: u32) -> Result<u64> {
        pools.rs_span(page, 0, 0)
    }

    /// The address of `plane`'s row `row` of rs buffer page `page`.
    pub fn row(&self, pools: &Pools, plane: Plane, page: u32, row: u32) -> Result<u64> {
        if row >= self.tokens {
            return Err(Fault::Ceiling {
                what: "rs buffer page tokens",
                need: u64::from(row) + 1,
                have: u64::from(self.tokens),
            });
        }
        let run = self
            .rows
            .get(plane.layer as usize)
            .copied()
            .ok_or(Fault::Ceiling {
                what: "recurrent layers the rs buffer holds rows of",
                need: u64::from(plane.layer) + 1,
                have: self.rows.len() as u64,
            })?;
        pools.rs_span(
            page,
            run + (plane.at + u64::from(row) * plane.width) * ELEMENT,
            plane.width * ELEMENT,
        )
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

/// The snapshot rows one lane lands in a fire (`Lane::rs_snapshots`), and
/// the chunks every fire's scans keep for them, so a body recorded with none
/// replays with up to this many.
pub const SNAPSHOTS: u32 = 16;

/// The snapshots this engine lands in a lane: [`SNAPSHOTS`] where every op
/// that reads a state slot is a recurrent scan or window whose chunked arm
/// lands them (a single-token arm never meets one), else none. On CUDA the
/// gated delta rule's prefill lands them on the two-launch scan, where
/// `gdn_lands` says this device takes it at a key width.
#[must_use]
pub fn snapshots(trace: &Trace, gdn_lands: impl Fn(u32) -> bool) -> u32 {
    let state = |id: &ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|value| &value.def),
            Some(poem_ir::Def::Cache(row))
                if matches!(trace.caches.get(*row as usize), Some(poem_ir::CacheRow::State { .. }))
        )
    };
    let mut reads = Vec::new();
    let lands = trace.nodes.iter().all(|node| {
        reads.clear();
        node.op.inputs(&mut reads);
        !reads.iter().any(state) || lands_snapshots(&node.op, &gdn_lands)
    });
    if lands { SNAPSHOTS } else { 0 }
}

#[cfg(feature = "rocm")]
fn lands_snapshots(op: &Operation, _gdn_lands: &impl Fn(u32) -> bool) -> bool {
    matches!(
        op,
        Operation::Ssm(
            Ssm::CausalConv1d { .. }
                | Ssm::CausalConv1dChunked { .. }
                | Ssm::ShortConv { .. }
                | Ssm::ShortConvChunked { .. }
                | Ssm::GatedDelta { .. }
                | Ssm::GatedDeltaChunked { .. }
                | Ssm::Kda { .. }
                | Ssm::KdaChunked { .. }
        ) | Operation::Gather(
            poem_ir::Gather::NgramIds { .. } | poem_ir::Gather::NgramIdsChunked { .. }
        ) | Operation::Fused(poem_ir::Fused::GdnStep { .. })
    )
}

/// CUDA's arms that land a snapshot: the causal conv's row-tiled commit
/// (a reach of at most 32 rows) and the gated delta rule's scans.
#[cfg(not(feature = "rocm"))]
fn lands_snapshots(op: &Operation, gdn_lands: &impl Fn(u32) -> bool) -> bool {
    match op {
        Operation::Ssm(Ssm::CausalConv1dChunked {
            conv_width,
            dilation,
            ..
        }) => conv_width.saturating_sub(1).saturating_mul(*dilation) <= 32,
        Operation::Ssm(Ssm::GatedDeltaChunked { k_dim, v_dim, .. }) => {
            *k_dim == 128 && *v_dim == 128 && gdn_lands(*k_dim)
        }
        Operation::Ssm(Ssm::CausalConv1d { .. } | Ssm::GatedDelta { .. })
        | Operation::Fused(poem_ir::Fused::GdnStep { .. }) => true,
        _ => false,
    }
}

/// A fire's snapshots in fire-lane order: lane `l`'s are entries
/// `indptr[l]..indptr[l + 1]` of `rows` and `slots`.
#[derive(Debug, Default, Clone)]
pub struct SnapTable {
    pub indptr: Vec<i32>,
    pub rows: Vec<i32>,
    pub slots: Vec<i32>,
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
    at_snap: Option<[u64; 3]>,
    /// The lane bounds the last fire left nonzero, which the next fire that
    /// takes no snapshot zeroes; a run of fires taking none stages nothing.
    snap_dirty: usize,
    lanes: u32,
}

impl Predicate {
    /// `ids` ids name a record a snapshot lands in (`Pools::ids_a_lane`).
    pub fn reserve(max_lanes: u32, snapshots: bool, ids: u32) -> Result<Predicate> {
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
        let at_snap = snapshots.then(|| {
            let entries = lanes * u64::from(SNAPSHOTS);
            [
                take((lanes + 1) * 4),
                take(entries * 4),
                take(entries * 4 * u64::from(ids.max(1))),
            ]
        });
        // A run of the activation's span, laid by the load
        // ([`Predicate::lay`]).
        let region = Buffer::lent(usize::try_from(at).unwrap_or(usize::MAX));
        Ok(Predicate {
            region,
            at_one,
            at_zero,
            at_commits,
            at_indptr,
            at_mask,
            at_len,
            at_snap,
            snap_dirty: 0,
            lanes: max_lanes,
        })
    }

    /// The region's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.region.bytes() as u64
    }

    /// The region lies at `base` from here on: a run of the activation's
    /// span the load holds backed and zeroed, of [`Predicate::bytes`]. Its
    /// constants are written there: the one a fold always takes, and the
    /// lanes' identity bounds.
    pub fn lay(&mut self, base: u64) -> Result<()> {
        self.region.lay(base);
        self.region.write(self.at_one, &1u32.to_le_bytes())?;
        let identity: Vec<u8> = (0..=self.lanes)
            .flat_map(|l| (l as i32).to_le_bytes())
            .collect();
        self.region.write(self.at_indptr, &identity)?;
        // What a fire left of its snapshots' bounds went with the old run.
        self.snap_dirty = 0;
        Ok(())
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

    /// Whether this load lands snapshots: its table's planes are reserved.
    #[must_use]
    pub fn lands_snapshots(&self) -> bool {
        self.at_snap.is_some()
    }

    /// The fire's snapshot table, staged on every fire of an engine that
    /// lands them (its lanes' bounds, zero where none takes one), so every
    /// launch reads it where a body recorded it.
    pub fn snapshots(
        &mut self,
        stream: *mut core::ffi::c_void,
        table: &SnapTable,
    ) -> Result<kernels_cuda::Snapshots> {
        let Some(at_snap) = self.at_snap else {
            return Ok(kernels_cuda::Snapshots::NONE);
        };
        let entries = u64::from(self.lanes) * u64::from(SNAPSHOTS);
        if table.rows.len() as u64 > entries {
            return Err(Fault::Ceiling {
                what: "rs snapshots",
                need: table.rows.len() as u64,
                have: entries,
            });
        }
        if !table.rows.is_empty() {
            for (at, values) in at_snap
                .into_iter()
                .zip([&table.indptr, &table.rows, &table.slots])
            {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.region.stage(stream, at, &bytes)?;
            }
            self.snap_dirty = table.indptr.len();
        } else if self.snap_dirty > 0 {
            self.region
                .stage(stream, at_snap[0], &vec![0u8; self.snap_dirty * 4])?;
            self.snap_dirty = 0;
        }
        let planes =
            at_snap.map(|at| kernels_cuda::Tensor::new(self.region.ptr() + at, 1, 1, Dtype::I32));
        let [indptr, rows, slots] = planes;
        Ok(kernels_cuda::Snapshots {
            indptr,
            rows,
            slots,
            count: SNAPSHOTS.max(table.rows.len() as u32),
        })
    }
}

#[cfg(test)]
mod tests {
    use poem_ir::{Def, Guard, Node, Platform, ValueDecl};

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
        Operation::Ssm(Ssm::KdaChunked {
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
            facts: Default::default(),
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
            outputs: Vec::new(),
            drafter: None,
            inputs: Vec::new(),
        }
    }

    /// A KDA layer buffers the conv's rows, then the forget and beta
    /// projections, laid page by page behind one another, and after them its
    /// keys and corrected values (`rs_corrected`: the rows its decode reads
    /// as a linear attention); its extended run stages the first three
    /// beside `mixed` and `y`.
    #[test]
    fn a_kda_recurrence_buffers_its_conv_rows_beside_its_two_projections() {
        let wide = HEADS * HEAD_DIM;
        let trace = plan(
            kda_values(),
            vec![
                Operation::Ssm(Ssm::CausalConv1dChunked {
                    x: ValueId(0),
                    weight: ValueId(1),
                    bias: None,
                    state: ValueId(2),
                    conv_width: 4,
                    dilation: 1,
                    y: ValueId(3),
                }),
                kda_chunked(3),
            ],
        );
        let (planes, widths) = read(&trace, PAGE)
            .expect("the KDA pair reads clean")
            .expect("the KDA pair reserves buffered planes");
        assert_eq!(widths, vec![3 * wide + wide + HEADS + wide + 2 * wide]);
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

    /// A causal conv no recurrence reads (Qwen4's PLE conv), whose state is
    /// a cache row, buffers its own input as a layer of its own, after the
    /// recurrence before it and at its own width: a buffered fire leaves its
    /// state as it was, and a fold replays its rows.
    #[test]
    fn a_lone_conv_buffers_its_own_rows_as_a_layer_of_its_own() {
        let wide = HEADS * HEAD_DIM;
        let mut values = kda_values();
        let lone = values.len() as u32;
        values.push(row(2 * wide, Dtype::Bf16));
        values.push(row(4, Dtype::Bf16));
        values.push(ValueDecl {
            def: Def::Cache(0),
            ty: Ty::Tensor {
                shape: vec![Dim::Const(3), Dim::Const(2 * wide)],
                dtype: Dtype::Bf16,
            },
        });
        values.push(row(2 * wide, Dtype::Bf16));
        let conv = |x: u32, state: u32, y: u32| {
            Operation::Ssm(Ssm::CausalConv1dChunked {
                x: ValueId(x),
                weight: ValueId(1),
                bias: None,
                state: ValueId(state),
                conv_width: 4,
                dilation: 3,
                y: ValueId(y),
            })
        };
        let trace = plan(
            values,
            vec![
                conv(0, 2, 3),
                kda_chunked(3),
                conv(lone, lone + 2, lone + 3),
            ],
        );
        let (planes, widths) = read(&trace, PAGE)
            .expect("a KDA layer and a lone conv read clean")
            .expect("both buffer planes");
        assert_eq!(widths, vec![3 * wide + wide + HEADS + wide + 2 * wide, 2 * wide]);
        assert_eq!(
            planes.of(ValueId(lone)),
            Some(Plane {
                layer: 1,
                at: 0,
                width: 2 * wide,
            })
        );
        assert!(ext_row_bytes(&trace).expect("the extended run's rows size") >= 2 * (2 * wide) * 2);
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
