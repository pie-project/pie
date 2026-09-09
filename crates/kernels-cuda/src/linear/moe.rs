use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const GEMV_WARPS: u32 = 4;

const MAX_EXPERTS: u32 = 512;

const MAX_GRID_Y: u32 = 65_535;

const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
}

fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

fn ranked_planes(op: &'static str, logits: Tensor, top_k: u32, routes: &Tensor, weights: &Tensor) {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` lands i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{op}` lands f32 route weights");
    debug_assert!(
        routes.rows == logits.rows && weights.rows == logits.rows,
        "a routed result lands one row per token row"
    );
    debug_assert!(
        routes.width == top_k && weights.width == top_k,
        "a routed result is the fan-out the statement states"
    );
}

fn router_extents(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
) -> Result<(i32, i32), Error> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    nonzero(op, "rows", logits.rows)?;
    nonzero(op, "the fan-out this router states", top_k)?;
    if experts > MAX_EXPERTS {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {MAX_EXPERTS} scores this router \
                 stages in shared memory"
            ),
        ));
    }
    Ok((stated(op, experts)?, stated(op, top_k)?))
}

pub fn topk_softmax(
    ctx: &Ctx,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_softmax";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_softmax<{t}>")),
        )
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(),
            ctx.stage(),
        ],
    )
}

pub fn topk_softmax_scaled(
    ctx: &Ctx,
    logits: Tensor,
    scale: Tensor,
    experts: u32,
    top_k: u32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_softmax_scaled";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        scale.dtype, logits.dtype,
        "the gain is read at the router's own width"
    );
    debug_assert_eq!(
        scale.rows * scale.width,
        experts,
        "the gain is indexed by expert, so it holds one entry per expert"
    );
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_softmax_scaled<{t}>")),
        )
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            ArgValue::ABSENT,
            scale.arg(),
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn ranked_router(
    ctx: &Ctx,
    op: &'static str,
    file: &'static str,
    entrypoint: &'static str,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    ranked_planes(op, logits, top_k, routes, weights);
    let (e, k) = router_extents(op, logits, experts, top_k)?;
    ctx.fire(
        op,
        Fire::at(file, entrypoint).apply(rms(logits.rows)),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            scaling.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid(
    ctx: &Ctx,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sigmoid";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_router(
        ctx,
        OP,
        FILE,
        symbol(&format!("::pie::linear::moe_topk_sigmoid<{t}>")),
        logits,
        correction_bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid_sink(
    ctx: &Ctx,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    global_scale: Option<Tensor>,
    experts: u32,
    top_k: u32,
    sink: u32,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sigmoid_sink";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = top_k
        .checked_add(sink)
        .ok_or_else(|| refuse(OP, "the fan-out does not count"))?;
    ranked_planes(OP, logits, fan, routes, weights);
    nonzero(OP, "rows", logits.rows)?;
    nonzero(OP, "the fan-out this router states", top_k)?;
    let width = experts
        .checked_add(sink)
        .ok_or_else(|| refuse(OP, "the expert count does not count"))?;
    if logits.width != width {
        return Err(refuse(
            OP,
            format!(
                "the router's row is {} wide and the statement names {experts} routed + {sink} sink experts",
                logits.width
            ),
        ));
    }
    if width > MAX_EXPERTS {
        return Err(refuse(
            OP,
            format!(
                "the expert count is {width}, above the {MAX_EXPERTS} scores this router \
                 stages in shared memory"
            ),
        ));
    }
    if let Some(bias) = &correction_bias {
        debug_assert_eq!(bias.dtype, Dtype::F32, "`{OP}` reads an f32 correction bias");
    }
    if let Some(scale) = &global_scale {
        debug_assert_eq!(scale.dtype, Dtype::F32, "`{OP}` reads an f32 global scale");
    }
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_topk_sigmoid_sink<{t}>")))
            .apply(rms(logits.rows)),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            global_scale.map_or(ArgValue::ABSENT, |scale| scale.arg()),
            stated(OP, experts)?.arg(),
            stated(OP, sink)?.arg(),
            stated(OP, top_k)?.arg(),
            scaling.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sqrt_softplus(
    ctx: &Ctx,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sqrt_softplus";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 correction bias"
    );
    ranked_router(
        ctx,
        OP,
        FILE,
        symbol(&format!("::pie::linear::moe_topk_sqrt_softplus<{t}>")),
        logits,
        Some(bias),
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

struct Selected {
    route_count: u32,

    top_k: i32,

    by_token: bool,
}

fn selected(op: &'static str, x: Tensor, routes: Tensor, y: &Tensor) -> Result<Selected, Error> {
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
        if let Some(count) = grouped_worth(ctx, x, bank, y, &fan, experts) {
            return select_grouped(ctx, OP, x, bank, routes, y, count);
        }
    }
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

#[cfg(feature = "cuda")]
const fn block_rows(per_expert: u32) -> u32 {
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
const GROUP_AT: u32 = 4;

#[cfg(feature = "cuda")]
const GROUP_WORKSPACE_CAP: u64 = 512 << 20;

#[cfg(feature = "cuda")]
fn grouped_worth(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    y: &Tensor,
    fan: &Selected,
    experts: ExpertTable,
) -> Option<u32> {
    if std::env::var_os("PIE_NO_MOE_GROUP").is_some() {
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
    if per_expert < GROUP_AT {
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
fn select_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
) -> Result<(), Error> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmBatchedEx,
        cublasHandle_t, cublasOperation_t, cublasStatus_t, cudaDataType,
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
    let rows = blocks
        .checked_mul(block)
        .ok_or_else(|| refuse(op, format!("{blocks} blocks of {block} rows overflow the fire")))?;

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
            .map_err(|_| refuse(op, format!("{name} wants {bytes} bytes, past this host's usize")))
            .and_then(|bytes| ctx.scratch(op, name, bytes))
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
        Fire::at(FILE, "::pie::linear::moe_align_decode<::pie::i32>").apply(
            Launch::grid([1, 1, 1], [BLOCK, 1, 1]).smem((3 * experts + 34) * 4),
        ),
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
            FILE,
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
            FILE,
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
    // SAFETY: `handle` is this context's live cuBLAS handle, already bound
    // to its stream by the shell. The three pointer arrays hold `blocks`
    // entries each, written by the launch above into slabs this context
    // owns and holds for the fire's life.
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
            FILE,
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
    if fan.route_count > MAX_GRID_Y {
        return Err(refuse(
            op,
            format!(
                "the route run is {}, above the {MAX_GRID_Y} rows this GEMV puts on the \
                 grid's y axis; the aligned batched leg is what a wider fire wants",
                fan.route_count
            ),
        ));
    }
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
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_matmul_select_gemv_{form}<{t}>"
            )),
        )
        .apply(Launch::grid(
            [y.width.div_ceil(GEMV_WARPS), fan.route_count, 1],
            [WARP, GEMV_WARPS, 1],
        )),
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
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn matmul_select_mxfp4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const MXFP4_BLOCK: u32 = 32;

    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert!(
        bias.is_none_or(|bias| bias.dtype == x.dtype),
        "the expert bias rides the activation's dtype"
    );
    let fan = selected(op, x, routes, y)?;
    if x.width == 0 || !x.width.is_multiple_of(MXFP4_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {MXFP4_BLOCK}-code mxfp4 blocks",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    const ROUTE_ORDER: &str = "moe_route_order";
    const ORDER_BLOCK: u32 = 1024;
    const EXPERT_CAP: u32 = 4096;
    const GROUPED_FROM: u32 = 16;
    const GROUPED_TILE_N: u32 = 128;
    const GROUPED_BLOCK: u32 = 256;
    const WMMA_ROUTES: u32 = 32;

    let experts = codes.rows.clamp(1, EXPERT_CAP);
    let order_words = fan.route_count as usize;
    let offsets_at = order_words.next_multiple_of(64);
    let work_cap = fan.route_count.div_ceil(WMMA_ROUTES) + experts;
    let work_at = (offsets_at + experts as usize + 2).next_multiple_of(64);
    let order = ctx.scratch(
        op,
        ROUTE_ORDER,
        (work_at + work_cap as usize) * core::mem::size_of::<i32>(),
    )? as usize as u64;
    let offsets = order + (offsets_at * core::mem::size_of::<i32>()) as u64;
    let work = order + (work_at * core::mem::size_of::<i32>()) as u64;
    ctx.fire(
        op,
        Fire::at("linear/quant.cuh", symbol("::pie::linear::moe_route_order"))
            .apply(Launch::grid([1, 1, 1], [ORDER_BLOCK, 1, 1]).smem(2 * (experts + 1) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            stated(op, work_cap)?.arg(),
            WMMA_ROUTES.arg(),
            fan.top_k.arg(),
            stated(op, fan.route_count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
        ],
    )?;
    if fan.route_count >= GROUPED_FROM && std::env::var_os("PIE_NO_MXFP4_GROUP").is_none() {
        let wmma = x.dtype == Dtype::Bf16 && std::env::var_os("PIE_MXFP4_NO_WMMA").is_none();
        let entry = if wmma {
            "::pie::linear::moe_matmul_select_mxfp4_wmma".to_string()
        } else {
            format!("::pie::linear::moe_matmul_select_mxfp4_grouped<{t}>")
        };
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at("linear/quant.cuh", symbol(&entry)).apply(Launch::grid(
                [
                    if wmma { work_cap } else { experts },
                    y.width.div_ceil(GROUPED_TILE_N),
                    1,
                ],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.cuh",
            symbol(&format!(
                "::pie::linear::moe_matmul_select_mxfp4<{t}, ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [fan.route_count, y.width.div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
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

#[allow(clippy::too_many_arguments)]
fn matmul_select_mlxu4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    debug_assert_eq!(biases.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    let fan = selected(op, x, routes, y)?;
    let k = stated(op, nonzero(op, "K, the bank's contracted width", x.width)?)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let elems = u64::from(x.width) * u64::from(y.width);
    let code_bits = u64::from(codes.width) * 8;
    let bits: u32 = match code_bits {
        b if b == elems * 8 => 8,
        b if b == elems * 4 => 4,
        b if b == elems * 2 => 2,
        _ => {
            return Err(refuse(
                op,
                format!(
                    "a {}-byte expert code row stores {elems} codes at neither two, four nor eight bits",
                    codes.width
                ),
            ));
        }
    };
    let factor_bytes = u64::from(scales.width);
    let groups_per_row = if factor_bytes > 0 && factor_bytes % (2 * u64::from(y.width)) == 0 {
        factor_bytes / (2 * u64::from(y.width))
    } else {
        0
    };
    let group = if groups_per_row > 0 && u64::from(x.width) % groups_per_row == 0 {
        u64::from(x.width) / groups_per_row
    } else {
        0
    };
    let group = u32::try_from(group).unwrap_or(0);
    if !matches!(group, 32 | 64 | 128) || !group.is_multiple_of(32 / bits) {
        return Err(refuse(
            op,
            format!(
                "{groups_per_row} factors over a {}-wide row is not a 32-, 64- or 128-code affine group",
                x.width
            ),
        ));
    }
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    const ROUTE_ORDER: &str = "moe_route_order";
    const ORDER_BLOCK: u32 = 1024;
    const EXPERT_CAP: u32 = 4096;
    const GROUPED_FROM: u32 = 16;
    const GROUPED_TILE_N: u32 = 128;
    const GROUPED_BLOCK: u32 = 256;
    const WMMA_ROUTES: u32 = 32;
    let experts = codes.rows.clamp(1, EXPERT_CAP);
    let order_words = fan.route_count as usize;
    let offsets_at = order_words.next_multiple_of(64);
    let work_cap = fan.route_count.div_ceil(WMMA_ROUTES) + experts;
    let work_at = (offsets_at + experts as usize + 2).next_multiple_of(64);
    let order = ctx.scratch(
        op,
        ROUTE_ORDER,
        (work_at + work_cap as usize) * core::mem::size_of::<i32>(),
    )? as usize as u64;
    let offsets = order + (offsets_at * core::mem::size_of::<i32>()) as u64;
    let work = order + (work_at * core::mem::size_of::<i32>()) as u64;
    ctx.fire(
        op,
        Fire::at("linear/quant.cuh", symbol("::pie::linear::moe_route_order"))
            .apply(Launch::grid([1, 1, 1], [ORDER_BLOCK, 1, 1]).smem(2 * (experts + 1) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            stated(op, work_cap)?.arg(),
            WMMA_ROUTES.arg(),
            fan.top_k.arg(),
            stated(op, fan.route_count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
        ],
    )?;
    if fan.route_count >= GROUPED_FROM {
        let entry = if x.dtype == Dtype::Bf16 {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_wmma<::pie::i32({bits}), ::pie::i32({group})>"
            )
        } else {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_grouped<{t}, ::pie::i32({bits}), \
                 ::pie::i32({group})>"
            )
        };
        let wmma = x.dtype == Dtype::Bf16;
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at("linear/quant.cuh", symbol(&entry)).apply(Launch::grid(
                [if wmma { work_cap } else { experts }, y.width.div_ceil(GROUPED_TILE_N), 1],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.cuh",
            symbol(&format!(
                "::pie::linear::moe_matmul_select_mlxu4<{t}, ::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [fan.route_count, y.width.div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            routes.arg(),
            ArgValue::Ptr(order),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

pub fn weighted_sum(
    ctx: &Ctx,
    routed: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) {
        return Err(refuse(
            OP,
            format!(
                "the routed rectangle's {} rows do not fold into the {} token rows",
                routed.rows, y.rows
            ),
        ));
    }
    let top_k = routed.rows / y.rows;
    debug_assert_eq!(
        routed.width, y.width,
        "the routed row's width, which the fold does not change"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == top_k,
        "the weight plane is one weight per route"
    );
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_weighted_sum<{t}>")),
        )
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn bias_sum(
    ctx: &Ctx,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_bias_sum";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "the bias lands on the activation's own rectangle"
    );
    debug_assert_eq!(
        bias.width, y.width,
        "an expert's bias row is the width it is added to"
    );
    debug_assert_eq!(
        routes.rows, y.rows,
        "the route plane is one row per token row"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == routes.width,
        "the weight plane is one weight per route"
    );
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_bias_sum<{t}>")))
            .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            x.arg(),
            bias.arg(),
            routes.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn sigmoid_gate_add(
    ctx: &Ctx,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        routed.rows == y.rows && routed.width == y.width,
        "the routed row is the combined row"
    );
    debug_assert!(
        shared.rows == y.rows && shared.width == y.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert_eq!(gate.rows, y.rows, "the gate column is one scalar per row");
    ctx.fire(
        OP,
        Fire::at(
            "linear/glu.cuh",
            symbol(&format!("::pie::linear::moe_sigmoid_gate_add<{t}>")),
        )
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            gate.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
            ctx.stage(),
        ],
    )
}
