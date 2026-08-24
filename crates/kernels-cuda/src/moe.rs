use crate::jit::abi::Bank;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch, Root};
use kernels::points::{Form, Repr};
use kernels::routine::{Const, In, Out};
use kernels::{Bind, Fire};

use kernels::Refusal;


const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const MOE_VEC_WIDTH: i32 = 8;

const GEMV_WARPS: i32 = 4;

pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

#[must_use]
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

#[must_use]
const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
}

const MAX_EXPERTS: i32 = 512;

fn too_many_experts(e: i32) -> Refusal {
    Refusal::Wide {
        what: "num_experts, which the router stages in shared memory",
        at: i64::from(e),
        max: i64::from(MAX_EXPERTS),
    }
}

/// The two RANKED routers, which are one launch under two symbols.
///
/// `moe.topk_sigmoid` and `moe.topk_sqrt_softplus` differ in the `.cuh` they
/// are compiled from and in whether the statement carries a correction bias,
/// and in nothing else: the same rectangle pair says how many experts there
/// are and how many of them a token keeps, the same shared-memory bound caps
/// the expert count, and the same eight arguments land. That is why this is
/// a function rather than two transcriptions of itself in the block above.
#[allow(clippy::too_many_arguments)]
fn ranked_router<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    root: &'static str,
    symbol: &str,
    logits: In<Tensor<T>>,
    routes: Out<Tensor<i32>>,
    weights: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<(), Refusal> {
    let rect = logits.all("num_experts")?;
    let routed = routes.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(too_many_experts(e));
    }
    ctx.fire(
        Fire::at(root, crate::jit::symbol(symbol)).apply(rms(rect.rows.unsigned_abs())),
        &[
            rect.ptr.arg(),
            routed.ptr.arg(),
            weights.arg(),
            correction_bias.arg(),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

/// The `Moe` family, claimed. Every point in it, and every body is the launch
/// itself.
///
/// THE THREE ROUTERS DROP THEIR STATED `experts` AND `top_k`: the statement
/// states them because its two results are sized from them, and the launch
/// reads the same two numbers back off the rectangles it was given — the
/// logits' width and the fanout's. Two of the three are [`ranked_router`],
/// which is one launch under two symbols; `topk_softmax` reads a different
/// `.cuh` with two null bias planes in its argument run.
#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn topk_softmax<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        let rect = logits.all("num_experts")?;
        let routed = routes.all("the routed fanout")?;
        let (num_experts, k) = (rect.width, routed.width);
        if num_experts > MAX_EXPERTS {
            return Err(too_many_experts(num_experts));
        }
        self.fire(
            Fire::at(
                "moe/topk_softmax.cuh",
                crate::jit::symbol(&format!("::pie::moe::topk_softmax<{}>", T::CPP)),
            )
            .apply(router_lane(rect.rows.unsigned_abs())),
            &[
                rect.ptr.arg(),
                core::ptr::null::<bf16>().arg(),
                core::ptr::null::<bf16>().arg(),
                routed.ptr.arg(),
                weights.arg(),
                num_experts.arg(),
                k.arg(),
                0_i32.arg(),
            ],
        )
    }

    fn topk_sigmoid<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        ranked_router(
            self,
            "moe/topk_sigmoid.cuh",
            &format!("::pie::moe::topk_sigmoid<{}>", T::CPP),
            logits,
            routes,
            weights,
            // The launch takes an OPTIONAL correction bias; no statement of
            // this point carries one — `topk_sqrt_softplus` is the router
            // that does, and it states the bias in its own declaration.
            None,
            renormalize,
            scaling,
        )
    }

    fn topk_sqrt_softplus<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        bias: Const<Tensor<f32>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        ranked_router(
            self,
            "moe/dsv4_routing.cuh",
            &format!("::pie::moe::topk_sqrtsoftplus<{}>", T::CPP),
            logits,
            routes,
            weights,
            Some(bias),
            renormalize,
            scaling,
        )
    }

    fn weighted_sum<T: kernels::points::Scalar>(
        &self,
        routed: In<Tensor<T>>,
        weights: In<Tensor<f32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let fan = weights.all("the routed fanout")?;
        self.fire(
            Fire::at(
                "moe/moe_dispatch.cuh",
                crate::jit::symbol(&format!(
                    "::pie::moe::token_batched_weighted_sum<{}>",
                    T::CPP
                )),
            )
            .apply(elementwise_rows(
                y.rows.unsigned_abs(),
                y.width.unsigned_abs(),
            )),
            &[
                y.arg(),
                routed.ptr.arg(),
                fan.ptr.arg(),
                fan.width.arg(),
                y.width.arg(),
            ],
        )
    }

    /// The routed GEMM against a DENSE `[E, N, K]` stack: `y[r] = x[r] @
    /// bank[routes[r]]`, one dot per route.
    ///
    /// A BODY, and the routine it replaces was never its launcher — which is
    /// why `moe_grouped_gemm` is DELETED rather than marked inlined. The
    /// canon row named it; it is the WMMA arm of the legacy aligned leg, and
    /// its own `supported` gate refused every `K` above 512 ("above which
    /// cuBLAS wins", `m` exactly one 16-row fragment, `n` in whole 64-wide
    /// tiles). Every SKU that states this point contracts far deeper: a3b's
    /// gate/up leg is `K = 2048` and its down leg `K = 768`, dsv4's and
    /// kimi's deeper still. So the legacy driver's `fire::moe_grouped` took
    /// the cuBLAS branch at every shipping shape and this kernel never ran
    /// for any of them — the row named a launcher that would have refused.
    /// What ran the same arithmetic per route is `moe_decode_gemv_body`, and
    /// that is what this fires. Nothing else in the tree reached the routine
    /// or its gate, measured; the legacy eDSL names it only as a string.
    ///
    /// # `act_div`, measured off the rectangles
    ///
    /// [`Moe::matmul_select_bias`]'s note applies verbatim and is not
    /// repeated: a text says this point twice, the gate/up leg hands it a
    /// PER-TOKEN `x` and the down leg hands it the gate/up leg's own already
    /// fanned-out result. The ratio `y.rows / x.rows` is the divisor, read
    /// here rather than stated anywhere, and it picks between the kernel's
    /// two instantiations — `ActByToken` is a template parameter, so the
    /// choice is a symbol and not an argument.
    ///
    /// # No staging, and the debt that leaves
    ///
    /// The aligned leg the legacy prefill built — `moe_align_decode` →
    /// `gather_moe_aligned_inputs` → `build_moe_ptrs_aligned` → a batched
    /// cuBLAS call → `reorder_moe_aligned_output` — exists to turn many
    /// tokens' routes into one GEMM per expert, and it buys back the weight
    /// re-read this GEMV pays once per route. It is a THROUGHPUT arm, not a
    /// correctness one, and by W10's rule its five scratch rectangles belong
    /// inside this fire (`Ctx::scratch`) rather than in a plan. Until it is
    /// measured back in, a fire wide enough to want it is the one this body
    /// refuses by name below: `route_count` is the grid's `y` extent and CUDA
    /// caps that at 65535.
    fn matmul_select<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        bank: Const<Tensor<T>>,
        routes: In<Tensor<i32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        /// `gridDim.y`'s hardware bound. `blockIdx.y` IS the route here.
        const MAX_GRID_Y: i32 = 65_535;

        let dst = y.all("N, the bank's output width")?;
        let act = x.all("K, the activation's width")?;
        let fan = routes.all("the routed fanout")?;
        let (n, k) = (dst.width, act.width);
        let top_k = fan.width;

        if top_k <= 0 {
            return Err(Refusal::Empty {
                what: "the routed fanout",
            });
        }
        let route_count = fan.rows.saturating_mul(top_k);
        if dst.rows != route_count {
            return Err(Refusal::Narrow {
                what: "the result's rows against one row per route",
                at: i64::from(dst.rows),
            });
        }
        if route_count > MAX_GRID_Y {
            return Err(Refusal::Wide {
                what: "the route run, which this GEMV puts on the grid's y axis; the \
                       aligned batched leg is what a wider fire wants",
                at: i64::from(route_count),
                max: i64::from(MAX_GRID_Y),
            });
        }
        let by_token = if act.rows == route_count {
            false
        } else if act.rows.saturating_mul(top_k) == route_count {
            true
        } else {
            return Err(Refusal::Narrow {
                what: "the activation's rows, which are the fire's tokens or its routes and \
                       neither here",
                at: i64::from(act.rows),
            });
        };
        // Both operand rows are walked as `float4`: eight elements a lane,
        // no tail. The bank's row is `K` deep and the activation's is the
        // same `K`, so one divisibility covers both.
        if k <= 0 || k % MOE_VEC_WIDTH != 0 {
            return Err(Refusal::Narrow {
                what: "K, in whole float4 loads of 8",
                at: i64::from(k),
            });
        }
        if n <= 0 {
            return Err(Refusal::Empty {
                what: "N, the bank's output width",
            });
        }
        let form = if by_token { "by_token" } else { "by_route" };
        self.fire(
            Fire::at(
                "moe/moe_dispatch.cuh",
                crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_{form}<{}>", T::CPP)),
            )
            .apply(Launch::grid(
                [
                    n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    route_count.unsigned_abs(),
                    1,
                ],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            )),
            &[
                fan.ptr.arg(),
                act.ptr.arg(),
                bank.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                k.arg(),
                n.arg(),
                (i64::from(n) * i64::from(k)).arg(),
            ],
        )
    }

    /// `y = routed + shared * sigmoid(gate)`: the shared expert joining the
    /// routed sum through the `[tokens, 1]` gate column the statement
    /// already projected.
    ///
    /// A BODY, and the routine it replaces answers a DIFFERENT question.
    /// `sigmoid_dot_scalar_gate_add` takes the gate's WEIGHT and computes
    /// the dot itself against a pre-norm row this statement no longer names,
    /// then adds in place; the point states the column and a separate
    /// result, which is what all three shader planes take. The `__global__`
    /// here is `sigmoid_scalar_gate_add`, whose `stride` is the pitch
    /// between two rows' gate values — `1` for a dense column, and whatever
    /// the column's own rectangle says when it is one column of a wider one.
    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<Tensor<T>>,
        shared: In<Tensor<T>>,
        gate: In<Tensor<T>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let dst = y.all("the combined row's width")?;
        let sum = routed.over(dst.rows, "the routed row's width")?;
        let side = shared.over(dst.rows, "the shared expert's row width")?;
        if sum.width != dst.width || side.width != dst.width {
            return Err(Refusal::Narrow {
                what: "the two rows this combine adds, which are the result's width",
                at: i64::from(sum.width.min(side.width)),
            });
        }
        let col = gate.over(dst.rows, "the gate column")?;
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::sigmoid_scalar_gate_add<{}>", T::CPP)),
            )
            .apply(elementwise_rows(
                dst.rows.unsigned_abs(),
                dst.width.unsigned_abs(),
            )),
            &[
                dst.ptr.arg(),
                sum.ptr.arg(),
                side.ptr.arg(),
                col.ptr.arg(),
                dst.width.arg(),
                (*col.stride).arg(),
            ],
        )
    }

    /// The routed GEMM with the expert's own bias row — gpt-oss's, and the
    /// last point that SKU was missing.
    ///
    /// A BODY AND NOT A DELEGATION, which makes it the second one in this
    /// block (`kv_append_shared`'s is the other). Every routine below fires
    /// something a legacy text already stated; nothing in this tree fires
    /// THIS statement, because nothing in this tree stated it. The two MXFP4
    /// GEMVs in `crate::quant` come closest and neither is it: both fuse an
    /// activation into the epilogue and one hard-wires its operand indexing.
    /// So the launcher is here and the `__global__` is
    /// `quant/dequant_fp4.cuh`'s `mxfp4_matmul_select_bias`, beside the two
    /// it is a de-fused sibling of.
    ///
    /// # What the body reads and what it refuses
    ///
    /// `n` and `k` come off the rectangles, never off a param: the result's
    /// width is `N` and the activation's is `K`, and the bank's `[E, N, K]`
    /// is those two with the expert fan the router already chose. The one
    /// number that is neither is `act_div` — see below.
    ///
    /// # `act_div`: the same statement, two operand shapes
    ///
    /// A text says this point twice and hands it two different rows. The
    /// gate/up leg's `x` is PER TOKEN (`[tokens, K]`); the down leg's is the
    /// gate/up leg's own result, already fanned out (`[routes, K]`). The
    /// declaration cannot tell them apart and does not have to: the result is
    /// `[routes, N]` either way, so the ratio `y.rows / x.rows` IS the
    /// divisor that turns a route index into an activation row, and it is
    /// read here rather than stated anywhere. A ratio that is neither `1` nor
    /// the route width is refused by name — it would mean the two rectangles
    /// came from different fires.
    ///
    /// # No staging, and that is the finding
    ///
    /// The MXFP4 leg this replaces needed per-expert POINTER ARRAYS
    /// (`packed_ptrs[e]`, `scale_ptrs[e]`, `bias_ptrs[e]`), carved at load
    /// beside every bank because no statement names them, and one bug class
    /// of its own — a bank's BASE bound where an array of bases belongs reads
    /// eight bytes of weight data as an address. A canonical bank is
    /// `[E, ...]` contiguous, so the address is `e * n * (k / 2)` and the
    /// kernel computes it from numbers it already has. There is no
    /// `Ctx::scratch` in this body because there is nothing to stage.
    fn matmul_select_bias<T: kernels::points::Scalar, R: Repr>(
        &self,
        x: In<Tensor<T>>,
        bank: Const<Bank<R>>,
        bias: Const<Tensor<T>>,
        routes: In<Tensor<i32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        match R::FORM {
            Form::Mxfp4 => mxfp4_matmul_select_bias(self, x, bank, bias, routes, y),
        }
    }
}

/// `moe.matmul_select_bias` at the MXFP4 repr. See the claim above.
fn mxfp4_matmul_select_bias<T: kernels::points::Scalar, R: Repr>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    bank: Const<Bank<R>>,
    bias: Const<Tensor<T>>,
    routes: In<Tensor<i32>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    /// One warp per (route, slab of four output rows), 128 threads a block —
    /// `crate::quant::routed_qmv_quad`'s geometry, which is the shape the
    /// two decode GEMVs were measured at.
    const ROWS_PER_WARP: i32 = 4;
    const DECODE_BLOCK: u32 = 128;

    let dst = y.all("N, the bank's output width")?;
    let act = x.all("K, the activation's width")?;
    let fan = routes.all("the routed fanout")?;
    let (n, k) = (dst.width, act.width);
    let top_k = fan.width;

    if top_k <= 0 {
        return Err(Refusal::Empty {
            what: "the routed fanout",
        });
    }
    // The route run is the router's whole rectangle, flattened: one matmul
    // per route, and the result carries exactly that many rows.
    let route_count = fan.rows.saturating_mul(top_k);
    if dst.rows != route_count {
        return Err(Refusal::Narrow {
            what: "the result's rows against one row per route",
            at: i64::from(dst.rows),
        });
    }
    // See the claim's `act_div` note: the ratio is measured, never stated.
    let act_div = if act.rows == route_count {
        1
    } else if act.rows.saturating_mul(top_k) == route_count {
        top_k
    } else {
        return Err(Refusal::Narrow {
            what: "the activation's rows, which are the fire's tokens or its routes and \
                   neither here",
            at: i64::from(act.rows),
        });
    };
    // MXFP4's own geometry: 32 codes to one E8M0 byte, two codes to a byte,
    // eight to the 32-bit word the unpacker reads.
    if k <= 0 || k % 32 != 0 {
        return Err(Refusal::Narrow {
            what: "K, in whole 32-code MXFP4 blocks",
            at: i64::from(k),
        });
    }
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "N, the bank's output width",
        });
    }
    let planes = bank.get();
    if planes.codes.is_null() || planes.scales.is_null() {
        return Err(Refusal::Null {
            what: "an MXFP4 bank plane; a bank slot binds its codes AND its block scales",
        });
    }

    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP.unsigned_abs();
    ctx.fire(
        Fire::at(
            "quant/dequant_fp4.cuh",
            crate::jit::symbol(&format!(
                "::pie::quant::mxfp4_matmul_select_bias<{}, ::pie::i32({ROWS_PER_WARP})>",
                T::CPP
            )),
        )
        .apply(Launch::grid(
            [route_count.unsigned_abs(), n.unsigned_abs().div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            act.ptr.arg(),
            fan.ptr.arg(),
            planes.codes.arg(),
            planes.scales.arg(),
            bias.arg(),
            dst.ptr.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
        ],
    )
}

#[must_use]
pub fn moe_aligned_block(routes: i32, num_experts: i32) -> i32 {
    if num_experts <= 0 {
        return MOE_ALIGNED_BLOCK_MIN;
    }
    let per_expert = routes / num_experts;
    let mut block = MOE_ALIGNED_BLOCK_MIN;
    while block * 2 <= MOE_ALIGNED_BLOCK_MAX && block * 2 <= per_expert {
        block *= 2;
    }
    block
}

pub static EXPERT_OFFSETS_ROOT: Root = Root::new("moe/expert_offsets.cuh");
