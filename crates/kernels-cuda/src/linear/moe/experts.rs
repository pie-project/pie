#[cfg(feature = "cuda")]
use super::*;
#[cfg(feature = "cuda")]
use crate::error::Error;
#[cfg(feature = "cuda")]
use crate::jit::Ctx;
use crate::tensor::Tensor;
#[cfg(feature = "cuda")]
use dtype::Dtype;

/// A gate-up bank a fused select reads (bf16, mxfp4 with an optional bias,
/// nvfp4, or affine u4): each expert's gate rows, then its up rows.
#[derive(Clone, Copy, Debug)]
pub enum GluBank {
    Bf16(Tensor),
    Mxfp4 {
        codes: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    },
    Nvfp4 {
        codes: Tensor,
        scales: Tensor,
        tensor_scales: Tensor,
    },
    Affine {
        codes: Tensor,
        scales: Tensor,
        zeros: Tensor,
    },
    /// An NVFP4 bank against NVFP4 activations (W4A4): its scales laid in
    /// atoms, a tensor scale a row, an input scale an expert (the global its
    /// activations quantize under). Only [`experts`] takes it, both legs so.
    Nvfp4A4 {
        codes: Tensor,
        atoms: Tensor,
        tensor_scales: Tensor,
        input_scales: Tensor,
    },
    /// An MXFP4 bank against e4m3 activations in 1x128 blocks (W4A8): its
    /// ue8m0 scales laid in atoms. Only [`experts`] takes it, both legs so.
    Mxfp4A8 {
        codes: Tensor,
        atoms: Tensor,
    },
    /// A 128x128-block FP8 bank against e4m3 activations in 1x128 blocks
    /// (W8A8): e4m3 codes, their ue8m0 exponents laid in atoms a weight row
    /// an atom row. Only [`experts`] takes it, both legs so.
    Fp8A8 {
        codes: Tensor,
        atoms: Tensor,
    },
}

/// A bank's tcgen05 operand, its experts, its bias and weight-scale planes
/// and the dtype `tc5_worth` reads, for a product `n` wide of `x`; `None`
/// for an affine bank whose group the GEMM does not take.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(super) fn bank_operand(
    x: Tensor,
    bank: GluBank,
    n: u32,
) -> Option<(crate::linear::gemm_sm100::Operand, u32, u64, u64, Dtype)> {
    use crate::linear::gemm_sm100;
    Some(match bank {
        GluBank::Bf16(b) => (
            gemm_sm100::Operand::plain(b.ptr, gemm_sm100::Elem::Bf16),
            b.rows,
            0,
            0,
            b.dtype,
        ),
        GluBank::Mxfp4 {
            codes,
            scales,
            bias,
        } => (
            gemm_sm100::Operand::scaled(codes.ptr, gemm_sm100::Elem::E2m1, scales.ptr, 32),
            codes.rows,
            bias.map_or(0, |b| b.ptr),
            0,
            Dtype::Bf16,
        ),
        GluBank::Nvfp4 {
            codes,
            scales,
            tensor_scales,
        } => (
            gemm_sm100::Operand::scaled(codes.ptr, gemm_sm100::Elem::E2m1, scales.ptr, 16),
            codes.rows,
            0,
            tensor_scales.ptr,
            Dtype::Bf16,
        ),
        GluBank::Affine {
            codes,
            scales,
            zeros,
        } => {
            let group = affine_u4_group(x, codes, scales, n)?;
            (
                gemm_sm100::Operand::affine(codes.ptr, scales.ptr, zeros.ptr, group),
                codes.rows,
                0,
                0,
                Dtype::Bf16,
            )
        }
        // Its activations quantize first: no bf16 operand reads it.
        GluBank::Nvfp4A4 { .. } | GluBank::Mxfp4A8 { .. } | GluBank::Fp8A8 { .. } => {
            return None;
        }
    })
}

/// `y[route] = glu(x[route or its token] . W[routes[route]]^T)`, the GLU in
/// the grouped tcgen05 GEMM's epilogue. `Ok(false)` where the fire does not
/// take it: the caller runs the select and the GLU apart.
#[cfg(feature = "cuda")]
pub fn select_glu(
    ctx: &Ctx,
    x: Tensor,
    bank: GluBank,
    routes: Tensor,
    y: &mut Tensor,
    glu: crate::linear::gemm_sm100::Glu,
    experts: ExpertTable,
) -> Result<bool, Error> {
    use crate::linear::gemm_sm100;
    const OP: &str = "fused.moe_select_glu";
    let fan = selected(OP, x, routes, y)?;
    // A narrow mxfp4 fire reads its bank faster on the register-dequant
    // kernel, which lands the GLU too.
    if let GluBank::Mxfp4 {
        codes,
        scales,
        bias,
    } = bank
        && mxfp4_mma_spans(ctx, fan.route_count, codes.rows)
        && !experts.streams()
        && x.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
    {
        select_mxfp4_mma(
            ctx,
            OP,
            x,
            codes,
            scales,
            bias,
            routes,
            y,
            &fan,
            Some(glu),
            None,
            false,
        )?;
        return Ok(true);
    }
    // A narrow NVFP4 (W4A16) fire likewise, where the profile says.
    if let GluBank::Nvfp4 {
        codes,
        scales,
        tensor_scales,
    } = bank
        && nvfp4_mma_worth(ctx, x, y, experts, &fan)
    {
        select_mxfp4_mma(
            ctx,
            OP,
            x,
            codes,
            scales,
            None,
            routes,
            y,
            &fan,
            Some(glu),
            Some(tensor_scales),
            false,
        )?;
        return Ok(true);
    }
    let Some((w, rows, bias, weight_scale, dtype)) = bank_operand(x, bank, 2 * y.width) else {
        return Ok(false);
    };
    if !tc5_worth(
        ctx,
        x,
        (dtype, rows),
        y,
        experts,
        routes.rows,
        fan.route_count,
    ) {
        return Ok(false);
    }
    select_tc5(
        ctx,
        OP,
        x,
        w,
        rows,
        routes,
        y,
        gemm_sm100::Epilogue {
            glu: Some(glu),
            bias,
            weight_scale,
            ..gemm_sm100::Epilogue::STORE
        },
    )?;
    Ok(true)
}

/// A routed MLP's operands: the activations, the gate-up and down banks
/// and the GLU between them, the routes and their weights, the GLU's width.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub struct RoutedMlp {
    pub x: Tensor,
    pub up: GluBank,
    pub glu: crate::linear::gemm_sm100::Glu,
    pub down: GluBank,
    pub routes: Tensor,
    pub weights: Tensor,
    pub inter: u32,
}

/// A routed MLP on the grouped tcgen05 GEMM, its rows kept by expert from
/// the gate-up select to the down select: routes laid out once, the GLU
/// landing from the gate-up epilogue into the down operand, the weighted
/// sum reading the down product in place. `norm`: each route's down row
/// normed under a scale first (`norm_gate`'s), but on W4A8 banks.
/// `Ok(false)`: run the four apart.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn experts(
    ctx: &Ctx,
    m: RoutedMlp,
    y: &mut Tensor,
    up_experts: ExpertTable,
    down_experts: ExpertTable,
    gate: Option<(Tensor, Tensor)>,
    norm: Option<(Tensor, f32)>,
) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, gemm_sm100};
    const OP: &str = "fused.moe_experts";
    if let Some((shared, g)) = gate
        && (shared.dtype != Dtype::Bf16
            || g.dtype != Dtype::Bf16
            || shared.rows != y.rows
            || shared.width != y.width
            || g.rows != y.rows
            || g.width == 0)
    {
        return Ok(false);
    }
    const GROUPED_X: &str = "linear.moe_tc5_x";
    const GROUPED_H: &str = "linear.moe_tc5_h";
    const GROUPED_Y: &str = "linear.moe_tc5_y";
    let count = routes.rows * routes.width;
    if let (GluBank::Nvfp4A4 { .. }, GluBank::Nvfp4A4 { .. }) = (up, down) {
        return experts_w4a4(ctx, m, y, gate, norm);
    }
    if let (GluBank::Mxfp4A8 { .. }, GluBank::Mxfp4A8 { .. })
    | (GluBank::Fp8A8 { .. }, GluBank::Fp8A8 { .. }) = (up, down)
    {
        if norm.is_some() {
            return Ok(false);
        }
        return experts_w4a8_as(ctx, m, y, W4a8Sum::Weighted(gate));
    }
    let hidden = Tensor::new(0, count, inter, Dtype::Bf16);
    let routed = Tensor::new(0, count, y.width, Dtype::Bf16);
    // An mxfp4 fire the register-dequant kernel takes stays on it.
    let mma = |bank: GluBank| match bank {
        GluBank::Mxfp4 { codes, .. } => mxfp4_mma_spans(ctx, count, codes.rows),
        GluBank::Nvfp4 { .. } => nvfp4_mma_spans(ctx, count),
        _ => false,
    };
    if count == 0
        || norm.is_some_and(|(s, _)| s.dtype != Dtype::Bf16 || s.rows * s.width != y.width)
        || mma(up)
        || mma(down)
        || weights.dtype != Dtype::F32
        || weights.rows != routes.rows
        || weights.width != routes.width
        || y.rows != routes.rows
        || !y.width.is_multiple_of(8)
    {
        return Ok(false);
    }
    let (Some(u), Some(d)) = (
        bank_operand(x, up, 2 * inter),
        bank_operand(hidden, down, y.width),
    ) else {
        return Ok(false);
    };
    if !tc5_worth(ctx, x, (u.4, u.1), &hidden, up_experts, routes.rows, count)
        || !tc5_worth(
            ctx,
            hidden,
            (d.4, d.1),
            &routed,
            down_experts,
            routes.rows,
            count,
        )
        || u.1 != d.1
    {
        return Ok(false);
    }
    let experts = u.1;
    let mut run = || -> Result<(), Error> {
        let layout = act_quant::lay_out(ctx, OP, routes, experts, gemm_sm100::GROUP_ALIGN, true)?;
        let k = x.width;
        let rows = layout.rows as usize;
        let hs = ctx.scratch_stream(OP, GROUPED_H, rows * inter as usize * 2)? as u64;
        let grouped = gemm_sm100::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        };
        let up_epi = gemm_sm100::Epilogue {
            glu: Some(glu),
            bias: u.2,
            weight_scale: u.3,
            live_rows: layout.live_rows(),
            ..gemm_sm100::Epilogue::STORE
        };
        let xs = ctx.scratch_stream(OP, GROUPED_X, rows * k as usize * 2)? as u64;
        act_quant::scatter_rows(ctx, OP, x, routes.width, layout.pos, count, xs)?;
        gemm_sm100::grouped_gemm_epi(
            ctx,
            gemm_sm100::Operand::plain(xs, gemm_sm100::Elem::Bf16),
            u.0,
            hs,
            grouped,
            [inter, k],
            1.0,
            up_epi,
            count,
        )?;
        let ys = ctx.scratch_stream(OP, GROUPED_Y, rows * y.width as usize * 2)? as u64;
        gemm_sm100::grouped_gemm_epi(
            ctx,
            gemm_sm100::Operand::plain(hs, gemm_sm100::Elem::Bf16),
            d.0,
            ys,
            grouped,
            [y.width, inter],
            1.0,
            gemm_sm100::Epilogue {
                bias: d.2,
                weight_scale: d.3,
                live_rows: layout.live_rows(),
                ..gemm_sm100::Epilogue::STORE
            },
            count,
        )?;
        act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, norm, y)
    };
    run()?;
    Ok(true)
}

/// The most routes an expert on average whose W4A4 legs run swapped
/// (GLM-5.3's experts, measured).
#[cfg(feature = "cuda")]
pub(super) const SWAPPED_UP_ROUTES: u32 = 64;

/// A W4A8 routed MLP of this many routes an expert or fewer runs its
/// gate-up swapped, in 64-row Q tiles (DeepSeek-V4-Flash, measured).
#[cfg(feature = "cuda")]
pub(super) const SWAPPED_W4A8_UP_ROUTES: u32 = 32;

/// [`experts`] on two W4A4 banks: the GLU lands as the down GEMM's NVFP4
/// operand (rounded through bf16 first, as apart). Under
/// `kernel.moe.w4a4_fused_from` tokens the four run apart; a narrow fire
/// lays its routes out contiguous all the same.
#[cfg(feature = "cuda")]
pub(super) fn experts_w4a4(
    ctx: &Ctx,
    m: RoutedMlp,
    y: &mut Tensor,
    gate: Option<(Tensor, Tensor)>,
    norm: Option<(Tensor, f32)>,
) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, gemm_sm100};
    const OP: &str = "fused.moe_experts";
    const W4A4_X: &str = "linear.moe_tc5_x4";
    const W4A4_XS: &str = "linear.moe_tc5_x4_atoms";
    const W4A4_H: &str = "linear.moe_tc5_h4";
    const W4A4_HS: &str = "linear.moe_tc5_h4_atoms";
    const W4A4_Y: &str = "linear.moe_tc5_y";
    const UP_ALPHA: &str = "linear.moe_tc5_up_alpha";
    const DOWN_ALPHA: &str = "linear.moe_tc5_down_alpha";
    let (
        GluBank::Nvfp4A4 {
            codes: up_codes,
            atoms: up_atoms,
            tensor_scales: up_ts,
            input_scales: up_is,
        },
        GluBank::Nvfp4A4 {
            codes: down_codes,
            atoms: down_atoms,
            tensor_scales: down_ts,
            input_scales: down_is,
        },
    ) = (up, down)
    else {
        return Ok(false);
    };
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let experts = up_codes.rows;
    let (k, n) = (x.width, y.width);
    // Only SwiGLU's and relu2's epilogues land what the four apart would.
    if count == 0
        || norm.is_some_and(|(s, _)| s.dtype != Dtype::Bf16 || s.rows * s.width != y.width)
        || tokens < ctx.tuning().moe_w4a4_fused_from
        || !matches!(glu, gemm_sm100::Glu::Swiglu | gemm_sm100::Glu::Relu2)
        || !ctx.tuning().moe_tc5
        || !gemm_sm100::supported()
        || x.dtype != Dtype::Bf16
        || x.rows != tokens
        || down_codes.rows != experts
        || up_is.rows * up_is.width < experts
        || down_is.rows * down_is.width < experts
        || !k.is_multiple_of(128)
        || !inter.is_multiple_of(128)
        || !n.is_multiple_of(64)
        || weights.dtype != Dtype::F32
        || weights.rows != tokens
        || weights.width != routes.width
        || y.rows != tokens
        || !n.is_multiple_of(8)
    {
        return Ok(false);
    }
    let mut run = || -> Result<(), Error> {
        let e = experts as usize;
        let up_alpha = ctx.scratch_stream(OP, UP_ALPHA, e * 4)? as u64;
        let down_alpha = ctx.scratch_stream(OP, DOWN_ALPHA, e * 4)? as u64;
        let alphas = [
            (
                act_quant::AlphaFrom {
                    input_scales: up_is.ptr,
                    tensor_scales: up_ts.ptr,
                    rows: if glu == gemm_sm100::Glu::Relu2 {
                        inter
                    } else {
                        2 * inter
                    },
                },
                up_alpha,
            ),
            (
                act_quant::AlphaFrom {
                    input_scales: down_is.ptr,
                    tensor_scales: down_ts.ptr,
                    rows: n,
                },
                down_alpha,
            ),
        ];
        // An expert's rows start on a 64-row half block: a decode's few rows
        // an expert fill half the tiles a 128-row block would pad; contiguous,
        // as the gate-up's NVFP4 epilogue lands its atoms.
        let align = gemm_sm100::GROUP_ALIGN;
        // A narrow fire lays its routes out and quantizes their rows in one
        // launch where the profile says (`kernel.moe.w4a4_fused_layout`).
        let fused = ctx.tuning().moe_w4a4_fused_layout
            && count <= act_quant::FUSED_ROUTES
            && experts <= act_quant::FUSED_EXPERTS;
        let rows = act_quant::contiguous_rows(count, experts, align) as usize;
        let xq = ctx.scratch_stream(OP, W4A4_X, rows * k as usize / 2)? as u64;
        let x_atom_bytes = gemm_sm100::atom_bytes(rows, k as usize, 16);
        let xs = act_quant::atoms_of(ctx, OP, W4A4_XS, x_atom_bytes)?;
        let layout = if fused {
            act_quant::lay_out_quant_nvfp4(
                ctx, OP, routes, experts, align, alphas, x, up_is, xq, xs,
            )?
        } else {
            let layout = act_quant::lay_out_with(ctx, OP, routes, experts, align, None, &alphas)?;
            act_quant::quantize(
                ctx,
                OP,
                act_quant::Format::Nvfp4,
                act_quant::Source {
                    x,
                    rows: count,
                    div: routes.width,
                    routes: Some(routes),
                    input_scales: Some(up_is),
                    pos: layout.pos,
                    m_max: 0,
                },
                xq,
                xs,
                x_atom_bytes,
            )?;
            layout
        };
        let rows = layout.rows as usize;
        let hq = ctx.scratch_stream(OP, W4A4_H, rows * inter as usize / 2)? as u64;
        let hs = ctx.scratch_stream(
            OP,
            W4A4_HS,
            gemm_sm100::atom_bytes(rows, inter as usize, 16),
        )? as u64;
        let grouped_up = gemm_sm100::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: up_alpha,
        };
        // A decode's few rows an expert run the gate-up swapped: a thread
        // a weight row, its epilogue over the expert's rows alone (unswapped,
        // a thread a row, the whole 128-row block's).
        let sms = ctx.multiprocessors().unwrap_or(148);
        // A Q tile of 32 rows where every expert's fit one (the fused
        // layout states its rows the fire's tokens at most), else 64.
        let swapped = gemm_sm100::Plan {
            swap: true,
            pair: false,
            bn: if fused && routes.rows <= 32 { 32 } else { 64 },
            stages: 0,
            splits: 1,
        };
        let plan = if count <= SWAPPED_UP_ROUTES * experts {
            swapped
        } else {
            gemm_sm100::Plan::grouped(&grouped_up, inter, k, count, sms)
        };
        gemm_sm100::grouped_gemm_epi_with(
            ctx,
            gemm_sm100::Operand::scaled(xq, gemm_sm100::Elem::E2m1, xs, 16),
            gemm_sm100::Operand::scaled(up_codes.ptr, gemm_sm100::Elem::E2m1, up_atoms.ptr, 16),
            hq,
            grouped_up,
            [inter, k],
            1.0,
            gemm_sm100::Epilogue {
                glu: Some(glu),
                out: gemm_sm100::Out::Nvfp4 {
                    data: hq,
                    scales: hs,
                    gscale: 1.0,
                },
                live_rows: layout.live_rows(),
                input_scales: down_is.ptr,
                ..gemm_sm100::Epilogue::STORE
            },
            plan,
        )?;
        let ys = ctx.scratch_stream(OP, W4A4_Y, rows * n as usize * 2)? as u64;
        let grouped_down = gemm_sm100::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: down_alpha,
        };
        let down_plan = if count <= SWAPPED_UP_ROUTES * experts {
            swapped
        } else {
            gemm_sm100::Plan::grouped(&grouped_down, n, inter, count, sms)
        };
        gemm_sm100::grouped_gemm_epi_with(
            ctx,
            gemm_sm100::Operand::scaled(hq, gemm_sm100::Elem::E2m1, hs, 16),
            gemm_sm100::Operand::scaled(down_codes.ptr, gemm_sm100::Elem::E2m1, down_atoms.ptr, 16),
            ys,
            grouped_down,
            [n, inter],
            1.0,
            gemm_sm100::Epilogue {
                live_rows: layout.live_rows(),
                weights_ahead: true,
                ..gemm_sm100::Epilogue::STORE
            },
            down_plan,
        )?;
        act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, norm, y)
    };
    run()?;
    Ok(true)
}

/// How a W4A8 routed MLP's routes fold into a token's row.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) enum W4a8Sum {
    /// Each route under its weight (`moe_weighted_sum`), under a gate plus
    /// the shared row at its sigmoid.
    Weighted(Option<(Tensor, Tensor)>),
    /// Each route under its weight before the down leg (the GLU's rows
    /// scaled before they quantize), then summed plainly and the shared row
    /// added (`moe_sum_add`): DeepSeek's.
    WeightedFirst(Tensor),
}

/// [`experts`]' W4A8 routed MLP the DeepSeek way: route weights scale the
/// GLU's rows before the down leg quantizes them, and the routes sum plainly
/// with the shared row added. `Ok(false)`: run the four apart.
#[cfg(feature = "cuda")]
pub fn experts_weighted_first(
    ctx: &Ctx,
    m: RoutedMlp,
    shared: Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    if shared.dtype != Dtype::Bf16 || shared.rows != y.rows || shared.width != y.width {
        return Ok(false);
    }
    experts_w4a8_as(ctx, m, y, W4a8Sum::WeightedFirst(shared))
}

/// [`experts`] on two W4A8 banks (or two block-FP8 ones) (MXFP4 codes against e4m3 activations in
/// 1x128 blocks): the GLU lands as the down GEMM's e4m3 operand. Under
/// `kernel.moe.w4a8_fused_from` tokens, or masked, the four run apart; but
/// a block-FP8 pair's masked fire (a decode's) runs fused on the masked
/// layout, laid out once: the gate-up lands bf16 and one pass takes its
/// SwiGLU into the down GEMM's e4m3 operand.
#[cfg(feature = "cuda")]
pub(super) fn experts_w4a8_as(
    ctx: &Ctx,
    m: RoutedMlp,
    y: &mut Tensor,
    sum: W4a8Sum,
) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, gemm_sm100};
    const OP: &str = "fused.moe_experts";
    const W4A8_X: &str = "linear.moe_tc5_x8";
    const W4A8_XS: &str = "linear.moe_tc5_x8_atoms";
    const W4A8_H: &str = "linear.moe_tc5_h8";
    const W4A8_HS: &str = "linear.moe_tc5_h8_atoms";
    const W4A8_Y: &str = "linear.moe_tc5_y";
    const ROW_WEIGHTS: &str = "linear.moe_tc5_row_weights";
    const GRAN: u32 = 128;
    // The weights' element and their scale's span: MXFP4 codes a 32 under
    // ue8m0, or e4m3 a 128 (a block-FP8 bank's rows) under ue8m0.
    let (up_codes, up_atoms, down_codes, down_atoms, w_elem, w_gran) = match (up, down) {
        (
            GluBank::Mxfp4A8 {
                codes: up_codes,
                atoms: up_atoms,
            },
            GluBank::Mxfp4A8 {
                codes: down_codes,
                atoms: down_atoms,
            },
        ) => (
            up_codes,
            up_atoms,
            down_codes,
            down_atoms,
            gemm_sm100::Elem::E2m1,
            32,
        ),
        (
            GluBank::Fp8A8 {
                codes: up_codes,
                atoms: up_atoms,
            },
            GluBank::Fp8A8 {
                codes: down_codes,
                atoms: down_atoms,
            },
        ) => (
            up_codes,
            up_atoms,
            down_codes,
            down_atoms,
            gemm_sm100::Elem::E4m3,
            128,
        ),
        _ => return Ok(false),
    };
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let experts = up_codes.rows;
    let (k, n) = (x.width, y.width);
    // SwiGLU, its clamp and SiTU land from the epilogue as the apart GLU
    // does.
    let glu_taken = matches!(
        glu,
        gemm_sm100::Glu::Swiglu
            | gemm_sm100::Glu::SwigluClamp { .. }
            | gemm_sm100::Glu::Situ { .. }
    );
    let masked = w_gran == 128
        && count <= act_quant::MASKED_FIRE
        && glu == gemm_sm100::Glu::Swiglu
        && matches!(sum, W4a8Sum::Weighted(_));
    if (count <= act_quant::MASKED_FIRE || tokens < ctx.tuning().moe_w4a8_fused_from) && !masked
        || !glu_taken
        || !ctx.tuning().moe_tc5
        || !gemm_sm100::supported()
        || x.dtype != Dtype::Bf16
        || x.rows != tokens
        || down_codes.rows != experts
        || !k.is_multiple_of(GRAN)
        || !inter.is_multiple_of(GRAN)
        || !n.is_multiple_of(64)
        || weights.dtype != Dtype::F32
        || weights.rows != tokens
        || weights.width != routes.width
        || y.rows != tokens
    {
        return Ok(false);
    }
    let mut run = || -> Result<(), Error> {
        let layout = if masked {
            act_quant::lay_out_with(
                ctx,
                OP,
                routes,
                experts,
                act_quant::BLOCK_ROWS,
                Some(false),
                &[],
            )?
        } else {
            act_quant::lay_out(ctx, OP, routes, experts, act_quant::BLOCK_ROWS, false)?
        };
        let rows = layout.rows as usize;
        let m_max = layout.m_max as usize;
        // A masked layout's scale atoms: each expert's whole 128-row blocks.
        let atoms_for = |cols: u32| {
            if m_max > 0 {
                gemm_sm100::masked_atom_bytes(experts as usize, m_max, cols as usize, GRAN as usize)
            } else {
                gemm_sm100::atom_bytes(rows, cols as usize, GRAN as usize)
            }
        };
        let fmt = act_quant::Format::E4m3 { gran: GRAN };
        let xq = ctx.scratch_stream(OP, W4A8_X, rows * k as usize)? as u64;
        let x_atom_bytes = atoms_for(k);
        let xs = act_quant::atoms_of(ctx, OP, W4A8_XS, x_atom_bytes)?;
        act_quant::quantize(
            ctx,
            OP,
            fmt,
            act_quant::Source {
                x,
                rows: count,
                div: routes.width,
                routes: None,
                input_scales: None,
                pos: layout.pos,
                m_max: layout.m_max,
            },
            xq,
            xs,
            x_atom_bytes,
        )?;
        let hq = ctx.scratch_stream(OP, W4A8_H, rows * inter as usize)? as u64;
        // A masked layout's padding rows keep their atoms: a zeroed slab.
        let hs = if masked {
            act_quant::atoms_of(ctx, OP, W4A8_HS, atoms_for(inter))?
        } else {
            ctx.scratch_stream(OP, W4A8_HS, atoms_for(inter))? as u64
        };
        // DeepSeek's weights scale the GLU's rows, each at its grouped row.
        let row_scale = match sum {
            W4a8Sum::Weighted(_) => 0,
            W4a8Sum::WeightedFirst(_) => {
                let at = ctx.scratch_stream(OP, ROW_WEIGHTS, rows * 4)? as u64;
                act_quant::scatter_f32(ctx, OP, weights.ptr, layout.pos, count, at)?;
                at
            }
        };
        let grouped = gemm_sm100::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        };
        if masked {
            const W4A8_U: &str = "linear.moe_tc5_u";
            let us = ctx.scratch_stream(OP, W4A8_U, rows * 2 * inter as usize * 2)? as u64;
            gemm_sm100::grouped_gemm_epi(
                ctx,
                gemm_sm100::Operand::scaled(xq, gemm_sm100::Elem::E4m3, xs, GRAN),
                gemm_sm100::Operand::scaled(up_codes.ptr, w_elem, up_atoms.ptr, w_gran),
                us,
                grouped,
                [2 * inter, k],
                1.0,
                gemm_sm100::Epilogue::STORE,
                count,
            )?;
            act_quant::swiglu_quant(ctx, OP, us, layout.rows, inter, layout.m_max, hq, hs)?;
        } else {
            // A few rows an expert run the gate-up swapped (bank on P, rows on 64-row
            // Q tiles): the same products at a quarter the MMA work.
            let sms = ctx.multiprocessors().unwrap_or(148);
            let up_plan = if count <= SWAPPED_W4A8_UP_ROUTES * experts {
                gemm_sm100::Plan {
                    swap: true,
                    pair: false,
                    bn: 64,
                    stages: 0,
                    splits: 1,
                }
            } else {
                gemm_sm100::Plan::grouped(&grouped, inter, k, count, sms)
            };
            gemm_sm100::grouped_gemm_epi_with(
                ctx,
                gemm_sm100::Operand::scaled(xq, gemm_sm100::Elem::E4m3, xs, GRAN),
                gemm_sm100::Operand::scaled(up_codes.ptr, w_elem, up_atoms.ptr, w_gran),
                hq,
                grouped,
                [inter, k],
                1.0,
                gemm_sm100::Epilogue {
                    glu: Some(glu),
                    out: gemm_sm100::Out::Fp8Block {
                        data: hq,
                        scales: hs,
                    },
                    row_scale,
                    live_rows: layout.live_rows(),
                    ..gemm_sm100::Epilogue::STORE
                },
                up_plan,
            )?;
        }
        let ys = ctx.scratch_stream(OP, W4A8_Y, rows * n as usize * 2)? as u64;
        gemm_sm100::grouped_gemm_epi(
            ctx,
            gemm_sm100::Operand::scaled(hq, gemm_sm100::Elem::E4m3, hs, GRAN),
            gemm_sm100::Operand::scaled(down_codes.ptr, w_elem, down_atoms.ptr, w_gran),
            ys,
            grouped,
            [n, inter],
            1.0,
            gemm_sm100::Epilogue {
                live_rows: layout.live_rows(),
                ..gemm_sm100::Epilogue::STORE
            },
            count,
        )?;
        match sum {
            W4a8Sum::Weighted(gate) => {
                act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, None, y)
            }
            W4a8Sum::WeightedFirst(shared) => {
                act_quant::sum_gather(ctx, OP, ys, layout.pos, routes.width, shared, y)
            }
        }
    };
    run()?;
    Ok(true)
}
