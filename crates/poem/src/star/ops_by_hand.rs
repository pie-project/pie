//! The bindings `tools/bind_ops.py` cannot write, by hand: an op whose
//! parameters or result are of types it does not spell (`norm_gate`'s
//! optional pairs, `conv3d`'s shape, `split_qkv_padded`'s arrays, the
//! `bool`-valued input methods), or whose packages' spelling keeps an order
//! or a default its signature does not (the causal convolutions take
//! `state` and `conv_width` before their optional `bias`). Each is named in
//! the generator's `BY_HAND`, so it is generated nowhere else.

use crate::ops;
use crate::star::bind::spelled::*;
use crate::star::bind::{Op, result};
use crate::star::forward::dsl;

pub(crate) static OPS: &[Op] = &[
    Op {
        module: "ssm",
        name: "causal_conv1d",
        params: &["x", "weight", "state", "conv_width", "bias"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let weight: Weight = a.next()?;
            let state: ValueId = a.next()?;
            let conv_width: u32 = a.next()?;
            let bias: Option<Weight> = a.next()?;
            result(
                heap,
                dsl("ssm.causal_conv1d", || {
                    ops::ssm::causal_conv1d(&x, &weight, bias.as_ref(), state, conv_width)
                })?,
            )
        },
    },
    Op {
        module: "ssm",
        name: "causal_conv1d_dilated",
        params: &["x", "weight", "state", "conv_width", "dilation"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let weight: Weight = a.next()?;
            let state: ValueId = a.next()?;
            let conv_width: u32 = a.next()?;
            let dilation: u32 = a.next()?;
            result(
                heap,
                dsl("ssm.causal_conv1d_dilated", || {
                    ops::ssm::causal_conv1d_dilated(&x, &weight, None, state, conv_width, dilation)
                })?,
            )
        },
    },
    Op {
        module: "ssm",
        name: "causal_conv1d_chunked",
        params: &["x", "weight", "state", "conv_width", "bias"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let weight: Weight = a.next()?;
            let state: ValueId = a.next()?;
            let conv_width: u32 = a.next()?;
            let bias: Option<Weight> = a.next()?;
            result(
                heap,
                dsl("ssm.causal_conv1d_chunked", || {
                    ops::ssm::causal_conv1d_chunked(&x, &weight, bias.as_ref(), state, conv_width)
                })?,
            )
        },
    },
    Op {
        module: "ssm",
        name: "causal_conv1d_chunked_dilated",
        params: &["x", "weight", "state", "conv_width", "dilation"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let weight: Weight = a.next()?;
            let state: ValueId = a.next()?;
            let conv_width: u32 = a.next()?;
            let dilation: u32 = a.next()?;
            result(
                heap,
                dsl("ssm.causal_conv1d_chunked_dilated", || {
                    ops::ssm::causal_conv1d_chunked_dilated(
                        &x, &weight, None, state, conv_width, dilation,
                    )
                })?,
            )
        },
    },
    Op {
        module: "elemwise",
        name: "norm_gate",
        params: &[
            "x",
            "scale",
            "res",
            "gate",
            "eps",
            "temperature",
            "next",
            "next_plus_one",
        ],
        call: |a, heap| {
            let x: Value = a.next()?;
            let scale: Option<Weight> = a.next()?;
            let res: Option<Value> = a.next()?;
            let gate: Option<Weight> = a.next()?;
            let eps: f32 = a.next()?;
            let temperature: f32 = a.next()?;
            let next: Option<(Weight, f32)> = a.next()?;
            let plus_one: Option<bool> = a.next()?;
            let joined = match (&res, &gate) {
                (Some(stream), Some(gate)) => Some((stream, gate)),
                (None, None) => None,
                _ => anyhow::bail!("`elemwise.norm_gate` takes a residual and its gate together"),
            };
            let next = next
                .as_ref()
                .map(|(w, e)| (w, *e, plus_one.unwrap_or(true)));
            let (y, z) = dsl("elemwise.norm_gate", || {
                ops::elemwise::norm_gate(&x, scale.as_ref(), joined, eps, temperature, next)
            })?;
            match z {
                Some(z) => result(heap, (y, z)),
                None => result(heap, y),
            }
        },
    },
    Op {
        module: "gather",
        name: "split_qkv_padded",
        params: &[
            "packed", "q_heads", "kv_heads", "qk_width", "v_width", "head_dim",
        ],
        call: |a, heap| {
            let packed: Value = a.next()?;
            let heads: [u32; 2] = [a.next()?, a.next()?];
            let widths: [u32; 2] = [a.next()?, a.next()?];
            let head_dim: u32 = a.next()?;
            result(
                heap,
                dsl("gather.split_qkv_padded", || {
                    ops::gather::split_qkv_padded(&packed, heads, widths, head_dim)
                })?,
            )
        },
    },
    Op {
        module: "gather",
        name: "gather_rows",
        params: &["x", "rows"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let rows: Value = a.next()?;
            result(
                heap,
                dsl("gather.rows", || ops::gather::gather_rows(&x, &rows))?,
            )
        },
    },
    Op {
        module: "conv",
        name: "conv3d",
        params: &["x", "grid", "w", "bias", "conv", "cache"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let grid: Value = a.next()?;
            let w: Weight = a.next()?;
            let bias: Option<Weight> = a.next()?;
            let conv: ConvShape = a.next()?;
            let cache: Option<ValueId> = a.next()?;
            result(
                heap,
                dsl("conv.conv3d", || {
                    ops::conv::conv3d(&x, &grid, &w, bias.as_ref(), conv, cache)
                })?,
            )
        },
    },
    Op {
        module: "conv",
        name: "upsample_nearest",
        params: &["x", "grid", "factor", "keep_first_frame"],
        call: |a, heap| {
            let x: Value = a.next()?;
            let grid: Value = a.next()?;
            let factor: [u32; 3] = a.next()?;
            let keep_first_frame: bool = a.next()?;
            result(
                heap,
                dsl("conv.upsample", || {
                    ops::conv::upsample_nearest(&x, &grid, factor, keep_first_frame)
                })?,
            )
        },
    },
];

pub(crate) static INPUTS: &[Op] = &[
    Op {
        module: "inputs",
        name: "route_hints",
        params: &["inputs"],
        call: |a, heap| {
            let inputs: Input = a.next()?;
            Ok(heap.alloc(inputs.route_hints()))
        },
    },
    Op {
        module: "inputs",
        name: "readout_attention",
        params: &["inputs"],
        call: |a, heap| {
            let inputs: Input = a.next()?;
            Ok(heap.alloc(inputs.readout_attention()))
        },
    },
];
