use super::select_fp4::{matmul_select_mlxu4, matmul_select_mxfp4};
use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;
use dtype::Dtype;

pub(super) const GEMV_WARPS: u32 = 4;

pub(super) const MAX_GRID_Y: u32 = 65_535;

pub(crate) struct Selected {
    pub(crate) route_count: u32,

    pub(crate) top_k: i32,

    pub(crate) by_token: bool,
}

pub(crate) fn selected(
    op: &'static str,
    x: Tensor,
    routes: Tensor,
    y: &Tensor,
) -> Result<Selected, Error> {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` walks i32 routes");
    let top_k = nonzero(op, "the routed fan-out", routes.width)?;
    let route_count = routes.rows.checked_mul(top_k).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the route run will not launch: {} tokens x {top_k} fan-out",
                routes.rows
            ),
        )
    })?;
    debug_assert_eq!(y.rows, route_count, "the result lands one row per route");
    let by_token = if x.rows == route_count {
        false
    } else if x.rows == routes.rows {
        true
    } else {
        return Err(refuse(
            op,
            format!(
                "the activation's {} rows are neither the fire's tokens nor its routes",
                x.rows
            ),
        ));
    };
    Ok(Selected {
        route_count,
        top_k: stated(op, top_k)?,
        by_token,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTable {
    pub table: u64,
    pub hits: u64,
}

impl ExpertTable {
    pub const RESIDENT: ExpertTable = ExpertTable { table: 0, hits: 0 };

    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupSeat {
    pub cell: u64,
    pub hits: u64,
}

impl GroupSeat {
    pub const RESIDENT: GroupSeat = GroupSeat { cell: 0, hits: 0 };

    #[must_use]
    pub const fn streams(&self) -> bool {
        self.cell != 0
    }
}

pub fn matmul_select(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select";

    #[cfg(feature = "cuda")]
    {
        let fan = selected(OP, x, routes, y)?;
        if tc5_worth(
            ctx,
            x,
            (bank.dtype, bank.rows),
            y,
            experts,
            routes.rows,
            fan.route_count,
        ) {
            return select_tc5(
                ctx,
                OP,
                x,
                crate::linear::tc5::Operand::plain(bank.ptr, crate::linear::tc5::Elem::Bf16),
                bank.rows,
                routes,
                y,
                crate::linear::tc5::Epilogue::STORE,
            );
        }
        // Where there are no tcgen05 GEMMs, a batch whose experts fill their
        // tiles (past 12 routes an expert, `BF16_GEMM_FROM` routes a fire) on
        // the tiled mma.sync GEMM. Fewer routes an expert take the pipelined
        // select or the per-route GEMV below, which measured ahead of the
        // register-held kernel there (Gemma 4 26B-A4B's 128 experts at 64
        // rows, 196 against 146 us a select; every decode measured).
        let tiled = fan.route_count >= BF16_GEMM_FROM
            && x.width.is_multiple_of(64)
            && fan.route_count.div_ceil(bank.rows.max(1)) > 12;
        if !crate::linear::tc5::supported()
            && tiled
            && bank.dtype == Dtype::Bf16
            && !experts.streams()
            && select_bf16_held(ctx, OP, x, bank.ptr, bank.rows, routes, y, None)?
        {
            return Ok(());
        }
        if let Some(count) = grouped_worth(ctx, x, bank, y, &fan, experts) {
            return select_grouped(ctx, OP, x, bank, routes, y, count);
        }
        // Bf16 experts under a captured graph have no grouped path (tcgen05 is
        // Blackwell's, cuBLAS is not graph-capturable); the per-route GEMV
        // re-reads each expert once a route. The grouped mma.sync kernel reads
        // it once a slice.
        if bf16_mma_spans(ctx, fan.route_count)
            && x.dtype == Dtype::Bf16
            && bank.dtype == Dtype::Bf16
            && !experts.streams()
            && x.width.is_multiple_of(64)
            && y.width.is_multiple_of(8)
            && bank.rows >= 1
            && u64::from(bank.width) == u64::from(x.width) * u64::from(y.width)
        {
            // On Hopper, wgmma over the same staging (sm_90a).
            if hopper_wgmma() && y.width.is_multiple_of(128) {
                return select_bf16_wgmma(ctx, OP, x, bank, routes, y, &fan, bank.rows);
            }
            return select_bf16_mma(ctx, OP, x, bank, routes, y, &fan, bank.rows);
        }
    }
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

/// Whether this device is Hopper (sm_90), whose arch-specific target has
/// wgmma: the routed W8A8 select's tensor-core GEMM there.
#[cfg(feature = "cuda")]
pub(crate) fn hopper_wgmma() -> bool {
    crate::jit::cache::arch() == Some("sm_90")
}

/// Whether a bf16 select runs on the grouped tcgen05 GEMM: a resident bank
/// of `count` experts at widths its tiles take, from `kernel.moe.tc5_from`
/// tokens or two routes an expert (fewer keep the one-launch GEMV, which
/// reads an expert once a route).
#[cfg(feature = "cuda")]
pub(crate) fn tc5_worth(
    ctx: &Ctx,
    x: Tensor,
    (bank, count): (Dtype, u32),
    y: &Tensor,
    experts: ExpertTable,
    tokens: u32,
    routes: u32,
) -> bool {
    ctx.tuning().moe_tc5
        && (tokens >= ctx.tuning().moe_tc5_from || routes >= 2 * count || routes > MAX_GRID_Y)
        && x.dtype == Dtype::Bf16
        && bank == Dtype::Bf16
        && !experts.streams()
        && x.width.is_multiple_of(64)
        && y.width.is_multiple_of(64)
        && crate::linear::tc5::supported()
}

/// `y[route] = x[route or its token] . W[routes[route]]^T` on one grouped
/// tcgen05 GEMM: routes laid out by expert, rows scattered in, the product
/// gathered back to route order.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_tc5(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    w: crate::linear::tc5::Operand,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    epi: crate::linear::tc5::Epilogue,
) -> Result<(), Error> {
    use crate::linear::{act_quant, tc5};
    const GROUPED_X: &str = "linear.moe_tc5_x";
    const GROUPED_Y: &str = "linear.moe_tc5_y";
    let fan = selected(op, x, routes, y)?;
    let count = fan.route_count;
    if count == 0 {
        return Ok(());
    }
    let (n, k) = (y.width, x.width);
    // The routers this select serves name an expert once a token.
    let layout = act_quant::lay_out(ctx, op, routes, experts, tc5::GROUP_ALIGN, true)?;
    let div = if fan.by_token { routes.width } else { 1 };
    let xs = ctx.scratch_stream(op, GROUPED_X, layout.rows as usize * k as usize * 2)? as u64;
    act_quant::scatter_rows(ctx, op, x, div, layout.pos, count, xs)?;
    let ys = ctx.scratch_stream(op, GROUPED_Y, layout.rows as usize * n as usize * 2)? as u64;
    tc5::grouped_gemm_epi(
        ctx,
        tc5::Operand::plain(xs, tc5::Elem::Bf16),
        w,
        ys,
        tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        },
        [n, k],
        1.0,
        // An expert's padding rows land nowhere.
        crate::linear::tc5::Epilogue {
            live_rows: layout.live_rows(),
            ..epi
        },
        count,
    )?;
    act_quant::gather_back(ctx, op, ys, layout.pos, routes.width, y)
}

#[cfg(feature = "cuda")]
pub(super) const fn block_rows(per_expert: u32) -> u32 {
    let want = per_expert.next_power_of_two();
    if want < 16 {
        16
    } else if want > 128 {
        128
    } else {
        want
    }
}

#[cfg(feature = "cuda")]
pub(super) const GROUP_WORKSPACE_CAP: u64 = 512 << 20;

#[cfg(feature = "cuda")]
pub(super) fn grouped_worth(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    y: &Tensor,
    fan: &Selected,
    experts: ExpertTable,
) -> Option<u32> {
    if !ctx.tuning().moe_grouped {
        return None;
    }
    if experts.streams() || !matches!(ctx.stage(), ArgValue::Ptr(0)) {
        return None;
    }
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F16) || bank.dtype != x.dtype {
        return None;
    }
    let count = bank.rows;
    if count == 0 || count > MAX_EXPERTS {
        return None;
    }
    if u64::from(bank.width) != u64::from(x.width) * u64::from(y.width) {
        return None;
    }
    let per_expert = fan.route_count.div_ceil(count);
    if per_expert < ctx.tuning().moe_group_at {
        return None;
    }
    let block = block_rows(per_expert);
    let blocks = u64::from(count) + u64::from(fan.route_count.div_ceil(block));
    let rows = blocks * u64::from(block);
    let staged = rows
        .checked_mul(u64::from(x.width) + u64::from(y.width))?
        .checked_mul(x.dtype.bytes_ceil())?;
    (staged <= GROUP_WORKSPACE_CAP).then_some(count)
}

#[cfg(feature = "cuda")]
pub(super) fn select_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
) -> Result<(), Error> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmBatchedEx, cublasHandle_t,
        cublasOperation_t, cublasStatus_t, cudaDataType,
    };

    let (t, scalar) = match x.dtype {
        Dtype::Bf16 => ("::pie::bf16", cudaDataType::CUDA_R_16BF),
        Dtype::F16 => ("::pie::f16", cudaDataType::CUDA_R_16F),
        other => {
            return Err(refuse(
                op,
                format!("the grouped leg hands cuBLAS a 16-bit bank and this one is {other:?}"),
            ));
        }
    };
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();

    let fan = selected(op, x, routes, y)?;
    let per_expert = fan.route_count.div_ceil(experts);
    let block = block_rows(per_expert);
    let blocks = experts + fan.route_count.div_ceil(block);
    let rows = blocks.checked_mul(block).ok_or_else(|| {
        refuse(
            op,
            format!("{blocks} blocks of {block} rows overflow the fire"),
        )
    })?;

    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let route_count = stated(op, fan.route_count)?;
    let aligned_rows = stated(op, rows)?;
    let block_size = stated(op, block)?;
    let max_blocks = stated(op, blocks)?;
    let expert_count = stated(op, experts)?;

    let elem = x.dtype.bytes_ceil();
    let slab = |name: &'static str, bytes: u64| -> Result<u64, Error> {
        usize::try_from(bytes)
            .map_err(|_| {
                refuse(
                    op,
                    format!("{name} wants {bytes} bytes, past this host's usize"),
                )
            })
            // One slab a stream: the layout lives within this op, so
            // regions captured apart share it rather than holding one each.
            .and_then(|bytes| ctx.scratch_stream(op, name, bytes))
            .map(|ptr| ptr as usize as u64)
    };
    let sorted = slab("linear.moe_group_sorted", u64::from(rows) * 4)?;
    let expert_ids = slab("linear.moe_group_experts", u64::from(blocks) * 4)?;
    let staged_in = slab(
        "linear.moe_group_in",
        u64::from(rows) * u64::from(x.width) * elem,
    )?;
    let staged_out = slab(
        "linear.moe_group_out",
        u64::from(rows) * u64::from(y.width) * elem,
    )?;
    let ptrs = slab("linear.moe_group_ptrs", u64::from(blocks) * 3 * 8)?;
    let w_ptrs = ptrs;
    let act_ptrs = ptrs + u64::from(blocks) * 8;
    let out_ptrs = ptrs + u64::from(blocks) * 16;

    ctx.fire(
        op,
        Fire::at(ROUTE, "::pie::linear::moe_align_decode<::pie::i32>")
            .apply(Launch::grid([1, 1, 1], [BLOCK, 1, 1]).smem((3 * experts + 34) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(expert_ids),
            ArgValue::ABSENT,
            route_count.arg(),
            expert_count.arg(),
            block_size.arg(),
            max_blocks.arg(),
            ArgValue::ABSENT,
        ],
    )?;

    let gather_fan = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(
            ROUTE,
            symbol(&format!("::pie::linear::gather_moe_aligned_inputs<{t}>")),
        )
        .apply(Launch::grid(
            [rows, x.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(staged_in),
            route_count.arg(),
            aligned_rows.arg(),
            gather_fan.arg(),
            k.arg(),
            (-1i32).arg(),
            stated(op, x.rows)?.arg(),
        ],
    )?;

    ctx.fire(
        op,
        Fire::at(
            ROUTE,
            symbol(&format!("::pie::linear::build_moe_leg_ptrs<{t}>")),
        )
        .apply(Launch::flat(blocks, BLOCK)),
        &[
            ArgValue::Ptr(expert_ids),
            bank.arg(),
            ArgValue::Ptr(staged_in),
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(w_ptrs),
            ArgValue::Ptr(act_ptrs),
            ArgValue::Ptr(out_ptrs),
            max_blocks.arg(),
            block_size.arg(),
            k.arg(),
            n.arg(),
        ],
    )?;

    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: `handle` is this context's live cuBLAS handle bound to its
    // stream; the pointer arrays hold `blocks` entries in slabs this context
    // owns for the fire's life.
    let status = unsafe {
        cublasGemmBatchedEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            block_size,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w_ptrs as usize as *const *const std::ffi::c_void,
            scalar,
            k,
            act_ptrs as usize as *const *const std::ffi::c_void,
            scalar,
            k,
            std::ptr::from_ref(&beta).cast(),
            out_ptrs as usize as *const *mut std::ffi::c_void,
            scalar,
            n,
            max_blocks,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(refuse(
            op,
            format!(
                "`cublasGemmBatchedEx` answered {status:?} at {max_blocks} blocks of \
                 M={block_size} N={n} K={k}"
            ),
        ));
    }

    ctx.fire(
        op,
        Fire::at(
            COMBINE,
            symbol(&format!("::pie::linear::reorder_moe_aligned_output<{t}>")),
        )
        .apply(Launch::grid(
            [rows, y.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(sorted),
            y.arg(),
            route_count.arg(),
            aligned_rows.arg(),
            n.arg(),
            (-1i32).arg(),
            0i32.arg(),
            ArgValue::ABSENT,
        ],
    )
}

pub(crate) fn select_gemv(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const VEC_WIDTH: u32 = 8;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(op, x, routes, y)?;
    if fan.route_count == 0 {
        return Ok(());
    }
    // A run past grid.y's 65535 rows continues on grid.z; every run within
    // it launches as it did.
    let (grid_y, grid_z) = if fan.route_count > MAX_GRID_Y {
        (MAX_GRID_Y, fan.route_count.div_ceil(MAX_GRID_Y))
    } else {
        (fan.route_count, 1)
    };
    if x.width == 0 || !x.width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {VEC_WIDTH}-element float4 loads",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let form = if fan.by_token { "by_token" } else { "by_route" };
    // Four weight rows a warp, their loads in flight together: Qwen3.6's
    // layer at 8 decode tokens (gate-up K 2048, down K 512) 55 + 37 us at one
    // row a warp, 40 + 21 at four (two 44 + 25, eight 45 + 24).
    const ROWS: u32 = 4;
    let entry = format!("::pie::linear::moe_matmul_select_gemv_{form}<{t}, {ROWS}>");
    ctx.fire(
        op,
        Fire::at(SELECT, symbol(&entry)).apply(
            Launch::grid(
                [y.width.div_ceil(GEMV_WARPS * ROWS), grid_y, grid_z],
                [WARP, GEMV_WARPS, 1],
            )
            .programmatic(ctx.tuning().pdl),
        ),
        &[
            routes.arg(),
            x.arg(),
            bank.arg(),
            y.arg(),
            fan.top_k.arg(),
            k.arg(),
            n.arg(),
            (i64::from(n) * i64::from(k)).arg(), // the bank's expert stride
            ArgValue::Ptr(experts.table),
            ArgValue::Ptr(experts.hits),
            stated(op, fan.route_count)?.arg(),
            ctx.stage(),
        ],
    )
}

pub(crate) const EXPERT_CAP: u32 = 4096;

pub(crate) const GROUPED_TILE_N: u32 = 128;

pub(crate) const GROUPED_BLOCK: u32 = 256;

pub(super) const WMMA_ROUTES: u32 = 32;

/// A select's GLU as its kernels' `kGlu` code and the GLU's two
/// constants (`linear/glu.cuh`); the register-dequant selects land GLUs
/// alone, not relu2.
#[cfg(feature = "cuda")]
pub(super) fn glu_code(
    op: &'static str,
    glu: Option<crate::linear::tc5::Glu>,
) -> Result<(i32, f32, f32), Error> {
    use crate::linear::tc5::Glu;
    Ok(match glu {
        None => (0, 0.0, 1.0),
        Some(Glu::Swiglu) => (1, 0.0, 1.0),
        Some(Glu::SwigluClamp { limit }) => (2, limit, 1.0),
        Some(Glu::SwigluClampAlpha { limit, alpha }) => (3, limit, alpha),
        Some(Glu::GegluTanh) => (4, 0.0, 1.0),
        Some(Glu::Situ { beta, cap }) => (5, beta, cap),
        Some(Glu::Relu2) => {
            return Err(refuse(
                op,
                "the register-dequant select lands GLUs, not relu2",
            ));
        }
    })
}

/// How a W8A16 or bf16 select's fire is cut: an expert's routes in slices
/// of `8 nb` (a tiled GEMM's 64 where `tiled`), and on the register-held
/// kernel its m16 tiles of weight rows a warp and its K split.
#[cfg(feature = "cuda")]
pub(super) struct SelectPlan {
    pub(super) nb: u32,
    pub(super) tiled: bool,
    pub(super) tiles: u32,
    pub(super) ksplit: u32,
    pub(super) order: RouteOrder,
}

/// The [`SelectPlan`] of a select of `fan` over `experts` landing `y`
/// (`glu` where it lands a GLU), on the tiled GEMM where `gemm` and its
/// experts average more than 12 routes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_plan(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    y: &Tensor,
    glu: bool,
    gemm: bool,
    fill: u32,
) -> Result<SelectPlan, Error> {
    let nb: u32 = match fan.route_count.div_ceil(experts) {
        0..=4 => 1,
        5..=12 => 2,
        _ if !gemm => 4,
        _ => 8,
    };
    let order = route_order_sliced(
        ctx,
        op,
        routes,
        fan,
        experts,
        8 * nb,
        true,
        if glu || KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A decode's few CTAs: half the rows a warp (twice the CTAs), then K
    // split across groups of a CTA's warps, toward `fill` CTAs' warps an SM.
    let sms = ctx.multiprocessors().unwrap_or(108);
    let tiles: u32 = if !glu && order.work_cap * y.width.div_ceil(128) < 2 * sms {
        1
    } else {
        2
    };
    let cols = if glu { 64 } else { 64 * tiles };
    let ctas = (order.work_cap * y.width.div_ceil(cols)).max(1);
    let ksplit = if nb == 1 {
        (fill * sms)
            .div_ceil(ctas)
            .next_power_of_two()
            .clamp(1, MMA_KSPLIT_MAX)
    } else {
        1
    };
    Ok(SelectPlan {
        nb,
        tiled: nb == 8,
        tiles,
        ksplit,
        order,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_bias(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_bias";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, Some(bias), routes, y, seat)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_quant(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, seat),
        Some(biases) => matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, seat),
    }
}
