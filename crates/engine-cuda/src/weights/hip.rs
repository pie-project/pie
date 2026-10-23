//! What a ROCm load does to the planes it lands, past what the CUDA one
//! does: e4m3 made canonical for the FP8 matrix core (it reads e4m3fnuz),
//! and fp8b32 / fp8b128 projections and mxfp4 expert banks put in the
//! matrix core's order where they lie.

use std::collections::BTreeMap;

use poem_ir::{Dtype, Trace};

use super::{Pairing, Place, address};
use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::WeightRow;

/// A GGUF K-quant bank binds as stored, and rocm has no kernel that reads one.
pub(super) fn refuse_k_quants(trace: &Trace) -> Result<()> {
    match trace.params.iter().find(|param| {
        matches!(
            param.dtype,
            Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k
        )
    }) {
        Some(param) => Err(Fault::Param {
            name: param.name.clone(),
            why: "is a GGUF K-quant bank (q2_k..q6_k), which rocm has no products for; \
                  serve a bf16, u4g64 or mxfp4 deployment of this model",
        }),
        None => Ok(()),
    }
}

/// Every e4m3 plane beside f32 block factors (an `e4m3tile128` projection
/// or bank) put in the FP8 matrix core's order and made canonical for it
/// (it reads e4m3fnuz): negative zero as zero (fnuz's NaN; the value it
/// is), and a plane holding a NaN code (0x7f, 0xff: numbers to fnuz)
/// refused. One staging buffer, the widest plane's, carries each plane out
/// of place and back.
pub(super) fn canonical_fp8(
    trace: &Trace,
    index: &BTreeMap<&str, usize>,
    places: &[Place],
    store: &Buffer,
    tier: Option<&crate::experts::Tier>,
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    // SAFETY: `stream` is the load's stream, live for the load.
    let ctx = unsafe { kernels_cuda::Ctx::on(stream.cast()) };
    let mut planes = Vec::new();
    for (at, param) in trace.params.iter().enumerate() {
        if param.dtype != Dtype::E4m3 {
            continue;
        }
        let factors = format!("{}{}", param.name, poem_ir::SCALES);
        let Some(&scales) = index.get(factors.as_str()) else {
            continue;
        };
        if trace.params[scales].dtype != Dtype::F32 {
            continue;
        }
        if tier.is_some_and(|tier| tier.handles(at).is_some() || tier.group_handles(at).is_some()) {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is an fp8 bank the expert tier seats from host memory; the FP8 matrix core \
                      reads its codes reordered and made canonical on the device, so it must be resident",
            });
        }
        let (Some(&k), true) = (param.shape.last(), param.shape.len() >= 2) else {
            continue;
        };
        let rows: u64 = param.shape[..param.shape.len() - 1].iter().product();
        planes.push((at, rows, k));
    }
    if planes.is_empty() {
        return Ok(());
    }
    let widest = planes
        .iter()
        .map(|(_, rows, k)| rows * k)
        .max()
        .unwrap_or(0);
    let staging = Buffer::zeroed(usize::try_from(widest).unwrap_or(usize::MAX))?;
    let nans = Buffer::zeroed(4)?;
    for &(at, rows, k) in &planes {
        let plane = address(tier, store, places[at].offset, at)?;
        kernels_cuda::linear::act_quant::e4m3fn_shuffle(
            &ctx,
            "linear.matmul",
            plane,
            staging.ptr(),
            rows,
            k,
            nans.ptr(),
        )?;
        crate::device::copy_d2d(
            stream,
            plane,
            staging.ptr(),
            usize::try_from(rows * k).unwrap_or(usize::MAX),
        )?;
    }
    crate::device::ctx::sync(stream)?;
    let mut count = [0u8; 4];
    nans.read(0, &mut count)?;
    if u32::from_le_bytes(count) != 0 {
        return Err(Fault::Param {
            name: trace.params[planes[0].0].name.clone(),
            why: "is one of the fp8 planes, which together hold NaN codes (0x7f, 0xff); the FP8 \
                  matrix core reads e4m3fnuz, where they are numbers, so this load refuses them",
        });
    }
    Ok(())
}

/// Each `e4m3g128` projection's per-row factors (`[n, k / 128]`) folded to
/// one a 64-row group, the FP8 kernels' finest: a release's 128-row blocks
/// laid as rows whose blocks restart on 64-row bounds share them. A
/// projection whose rows differ within a group is refused.
pub(super) fn group_row_factors(
    trace: &Trace,
    index: &BTreeMap<&str, usize>,
    table: &mut [Option<WeightRow>],
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    const GROUP: usize = 64;
    for param in &trace.params {
        if param.dtype != Dtype::E4m3 {
            continue;
        }
        let factors = format!("{}{}", param.name, poem_ir::SCALES);
        let Some(&at) = index.get(factors.as_str()) else {
            continue;
        };
        let (Some(&n), Some(t)) = (param.shape.first(), table.get(at).copied().flatten()) else {
            continue;
        };
        let WeightRow::Dense(t) = t else { continue };
        if param.shape.len() != 2 || t.dtype != Dtype::F32 || n <= 128 || u64::from(t.rows) != n {
            continue;
        }
        let (rows, cols) = (t.rows as usize, t.width as usize);
        let mut host = vec![0f32; rows * cols];
        crate::device::ctx::sync(stream)?;
        crate::device::copy_any(stream, host.as_mut_ptr() as u64, t.ptr, host.len() * 4)?;
        crate::device::ctx::sync(stream)?;
        let groups: Vec<&[f32]> = host.chunks(GROUP * cols).collect();
        if groups
            .iter()
            .any(|g| g.chunks(cols).any(|row| row != &g[..cols]))
        {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is an e4m3g128 projection whose per-row factors differ within a 64-row \
                      group; rocm's FP8 kernels read one factor a 64-row group at the finest",
            });
        }
        let folded: Vec<f32> = groups
            .iter()
            .flat_map(|g| g[..cols].iter().copied())
            .collect();
        crate::device::copy_any(stream, t.ptr, folded.as_ptr() as u64, folded.len() * 4)?;
        crate::device::ctx::sync(stream)?;
        table[at] = Some(WeightRow::Dense(kernels_cuda::Tensor::new(
            t.ptr,
            u32::try_from(groups.len()).unwrap_or(u32::MAX),
            t.width,
            t.dtype,
        )));
    }
    Ok(())
}

/// The fp8b32 and fp8b128 projections that land in the FP8 matrix core's
/// order: `[n, k]` e4m3 planes (n a multiple of 16, k of 128) beside ue8m0
/// exponents a 32x32 (128x128) block, held on the device, that only W8A8
/// products of bf16 rows under e4m3 a 1x32 (1x128) block read; with their
/// exponents and block side.
fn fp8b_planes(trace: &Trace, tier: Option<&crate::experts::Tier>) -> Vec<(usize, usize, u64)> {
    use poem_ir::{ActQuant, Def, Linear, Operands, Operation, Ty};
    let weight = |id: poem_ir::ValueId| match trace.values.get(id.0 as usize).map(|decl| &decl.def)
    {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    };
    let bf16 = |id: poem_ir::ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|decl| &decl.ty),
            Some(Ty::Tensor {
                dtype: Dtype::Bf16,
                ..
            })
        )
    };
    let mut reads = vec![0u32; trace.params.len()];
    let mut products = vec![0u32; trace.params.len()];
    let mut exps_of = vec![(0usize, 0u64); trace.params.len()];
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        for &id in &ins {
            if let Some(w) = weight(id) {
                reads[w] += 1;
            }
        }
        if let Operation::Linear(Linear::Matmul {
            act,
            w,
            quant: Some(ActQuant::E4m3 {
                gran: side @ (32 | 128),
            }),
            input_scales: None,
            scales: Some(scales),
            logits: false,
            ..
        }) = &node.op
            && let (Some(w), Some(s)) = (weight(*w), weight(*scales))
            && bf16(*act)
        {
            let (param, exps) = (&trace.params[w], &trace.params[s]);
            let side = u64::from(*side);
            if param.dtype == Dtype::E4m3
                && exps.dtype == Dtype::E8m0
                && param.shape.len() == 2
                && param.shape[0] % 16 == 0
                && param.shape[1] % 128 == 0
                && exps.shape == [param.shape[0].div_ceil(side), param.shape[1] / side]
            {
                products[w] += 1;
                exps_of[w] = (s, side);
            }
        }
    }
    (0..trace.params.len())
        .filter(|&at| {
            products[at] > 0
                && products[at] == reads[at]
                && !tier.is_some_and(|tier| {
                    tier.handles(at).is_some() || tier.group_handles(at).is_some()
                })
        })
        .map(|at| (at, exps_of[at].0, exps_of[at].1))
        .collect()
}

/// Puts `fp8b_planes` in the FP8 matrix core's order where they lie, made
/// canonical (a NaN code refused), through one staging plane, and rebinds
/// their rows as ordered; an fp8b128 plane's exponents rebind as the f32
/// factors 2^(e − 127) the matrix core's 128-block products fold under
/// (into `held`).
pub(super) fn order_fp8b_planes(
    trace: &Trace,
    places: &[Place],
    store: &Buffer,
    tier: Option<&crate::experts::Tier>,
    table: &mut [Option<WeightRow>],
    held: &mut Vec<Buffer>,
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    let found = fp8b_planes(trace, tier);
    let Some(widest) = found
        .iter()
        .map(|&(at, _, _)| trace.params[at].shape.iter().product::<u64>())
        .max()
    else {
        return Ok(());
    };
    for &(_, exps, side) in &found {
        if side != 128 {
            continue;
        }
        let Some(Some(WeightRow::Dense(t))) = table.get(exps).copied() else {
            continue;
        };
        let mut bytes = vec![0u8; t.rows as usize * t.width as usize];
        crate::device::ctx::sync(stream)?;
        crate::device::copy_any(stream, bytes.as_mut_ptr() as u64, t.ptr, bytes.len())?;
        crate::device::ctx::sync(stream)?;
        // 2^(e − 127); e = 0 is 2^-127, an f32 denormal.
        let factors: Vec<f32> = bytes
            .iter()
            .map(|&e| {
                f32::from_bits(if e == 0 {
                    0x40_0000
                } else {
                    u32::from(e) << 23
                })
            })
            .collect();
        let buffer = Buffer::zeroed(factors.len() * 4)?;
        crate::device::copy_any(
            stream,
            buffer.ptr(),
            factors.as_ptr() as u64,
            factors.len() * 4,
        )?;
        crate::device::ctx::sync(stream)?;
        table[exps] = Some(WeightRow::Dense(kernels_cuda::Tensor::new(
            buffer.ptr(),
            t.rows,
            t.width,
            Dtype::F32,
        )));
        held.push(buffer);
    }
    let planes: Vec<usize> = found.iter().map(|&(at, _, _)| at).collect();
    // SAFETY: `stream` is the load's stream, live for the load.
    let ctx = unsafe { kernels_cuda::Ctx::on(stream.cast()) };
    let staging = Buffer::zeroed(usize::try_from(widest).unwrap_or(usize::MAX))?;
    let nans = Buffer::zeroed(4)?;
    for &at in &planes {
        let (rows, k) = (trace.params[at].shape[0], trace.params[at].shape[1]);
        let plane = address(tier, store, places[at].offset, at)?;
        kernels_cuda::linear::act_quant::e4m3fn_shuffle(
            &ctx,
            "linear.matmul",
            plane,
            staging.ptr(),
            rows,
            k,
            nans.ptr(),
        )?;
        crate::device::copy_d2d(
            stream,
            plane,
            staging.ptr(),
            usize::try_from(rows * k).unwrap_or(usize::MAX),
        )?;
        if let Some(Some(WeightRow::Dense(handle))) = table.get(at).copied() {
            table[at] = Some(WeightRow::Ordered(handle));
        }
    }
    crate::device::ctx::sync(stream)?;
    let mut count = [0u8; 4];
    nans.read(0, &mut count)?;
    if u32::from_le_bytes(count) != 0 {
        return Err(Fault::Param {
            name: trace.params[planes[0]].name.clone(),
            why: "is one of the fp8b32 / fp8b128 planes, which together hold NaN codes (0x7f, 0xff); the FP8 \
                  matrix core reads e4m3fnuz, where they are numbers, so this load refuses them",
        });
    }
    Ok(())
}

/// The mxfp4 expert banks that may land in the matrix core's lane order:
/// `[experts, n, ..]` banks (n a multiple of 16) under e8m0
/// exponents alone, held on the device, that only routed selects of bf16
/// rows read (`MoeMatmulSelectQuant`, or its e4m3-quantized twin).
fn lane_banks(
    trace: &Trace,
    tier: Option<&crate::experts::Tier>,
    pairings: &BTreeMap<&str, Pairing>,
) -> Vec<(usize, usize)> {
    use poem_ir::{ActQuant, Def, Fused, Linear, Operands, Operation, Ty};
    let weight = |id: poem_ir::ValueId| match trace.values.get(id.0 as usize).map(|decl| &decl.def)
    {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    };
    let bf16 = |id: poem_ir::ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|decl| &decl.ty),
            Some(Ty::Tensor {
                dtype: Dtype::Bf16,
                ..
            })
        )
    };
    let mut reads = vec![0u32; trace.params.len()];
    let mut selects = vec![0u32; trace.params.len()];
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        for &id in &ins {
            if let Some(w) = weight(id) {
                reads[w] += 1;
            }
        }
        let legs: Vec<&Linear> = match &node.op {
            Operation::Linear(op) => vec![op],
            Operation::Fused(Fused::MoeExperts { select, down, .. }) => vec![&**select, &**down],
            Operation::Fused(Fused::MoeSelectGlu { select, .. }) => vec![&**select],
            _ => Vec::new(),
        };
        for leg in legs {
            let (x, bank) = match leg {
                Linear::MoeSelect {
                    x,
                    bank,
                    bias: None,
                    quantized: true,
                    quant: None,
                    ..
                } => (*x, *bank),
                Linear::MoeSelect {
                    x,
                    bank,
                    quant: Some(ActQuant::E4m3 { .. }),
                    bias: None,
                    quantized: true,
                    ..
                } => (*x, *bank),
                _ => continue,
            };
            if let Some(w) = weight(bank)
                && bf16(x)
            {
                selects[w] += 1;
            }
        }
    }
    trace
        .params
        .iter()
        .enumerate()
        .filter_map(|(at, param)| {
            let pairing = pairings.get(param.name.as_str())?;
            (param.dtype == Dtype::Mxfp4
                && pairing.biases.is_none()
                && param.shape.len() >= 3
                && param.shape[1] % 16 == 0
                && selects[at] > 0
                && selects[at] == reads[at]
                && !tier.is_some_and(|tier| {
                    [at, pairing.scales].iter().any(|&p| {
                        tier.handles(p).is_some()
                            || tier.group_handles(p).is_some()
                            || tier.offloaded_at(p).is_some()
                    })
                }))
            .then_some((at, pairing.scales))
        })
        .collect()
}

/// Puts `lane_banks` in the matrix core's lane order where they lie, codes
/// and exponents through one staging span, and marks their rows.
pub(super) fn order_mxfp4_banks(
    trace: &Trace,
    places: &[Place],
    store: &Buffer,
    tier: Option<&crate::experts::Tier>,
    pairings: &BTreeMap<&str, Pairing>,
    table: &mut [Option<WeightRow>],
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    // Each bank's experts, rows (n) and K, from its codes' row: n · K / 2
    // bytes an expert. K a whole number of 128-code units.
    let banks: Vec<(usize, usize, u64, u64, u64)> = lane_banks(trace, tier, pairings)
        .into_iter()
        .filter_map(|(at, scales)| {
            let Some(Some(WeightRow::Planes { codes, .. })) = table.get(at) else {
                return None;
            };
            let n = trace.params[at].shape[1];
            let k = 2 * u64::from(codes.width) / n;
            (2 * u64::from(codes.width) == n * k && k.is_multiple_of(128)).then_some((
                at,
                scales,
                u64::from(codes.rows),
                n,
                k,
            ))
        })
        .collect();
    // A run of whole experts at a time through one staging span of at most
    // `STAGED` bytes (an expert's strips never cross into the next), so the
    // load holds no bank-sized copy.
    const STAGED: u64 = 256 << 20;
    let Some(widest) = banks.iter().map(|&(_, _, _, n, k)| n * k).max() else {
        return Ok(());
    };
    // SAFETY: `stream` is the load's stream, live for the load.
    let ctx = unsafe { kernels_cuda::Ctx::on(stream.cast()) };
    // An expert's codes (half a byte an element) then exponents (one a 32).
    let per_expert = widest / 2 + widest / 32;
    let run = (STAGED / per_expert).max(1);
    let codes_span = (run * widest / 2).next_multiple_of(16);
    let staging =
        Buffer::zeroed(usize::try_from(codes_span + run * widest / 32).unwrap_or(usize::MAX))?;
    for &(at, scales, experts, n, k) in &banks {
        let codes = address(tier, store, places[at].offset, at)?;
        let exps = address(tier, store, places[scales].offset, scales)?;
        let (staged_codes, staged_exps) = (staging.ptr(), staging.ptr() + codes_span);
        let mut first = 0;
        while first < experts {
            let take = run.min(experts - first);
            let rows = take * n;
            let (c, e) = (codes + first * n * k / 2, exps + first * n * k / 32);
            kernels_cuda::linear::moe::mxfp4_lanes(
                &ctx,
                "linear.moe_select",
                c,
                staged_codes,
                e,
                staged_exps,
                rows,
                k,
            )?;
            for (to, from, bytes) in [
                (c, staged_codes, rows * k / 2),
                (e, staged_exps, rows * k / 32),
            ] {
                crate::device::copy_d2d(
                    stream,
                    to,
                    from,
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                )?;
            }
            first += take;
        }
        if let Some(Some(WeightRow::Planes { lanes, .. })) = table.get_mut(at) {
            *lanes = true;
        }
    }
    crate::device::ctx::sync(stream)
}
