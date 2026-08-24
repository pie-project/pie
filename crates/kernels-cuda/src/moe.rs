use crate::jit::Abi;
use crate::jit::abi::Bank;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch, Root, aligned16};
use kernels::points::{Form, Repr};
use kernels::routine::{Const, In, InOut, Out, Region, Stride};
use kernels::{Bind, Fire};
use kernels_macros::routine;

use kernels::Refusal;

use core::ffi::c_void;

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const SORT_BLOCK: u32 = 1024;

const DISPATCH_BLOCK: u32 = 256;

const MOE_VEC_WIDTH: i32 = 8;

const GEMV_WARPS: i32 = 4;

pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
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

fn per_route<P, Q>(dst: Region<P>, routes: Region<Q>) -> Result<i32, Refusal> {
    if dst.width % routes.width != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of routes",
            at: i64::from(dst.width),
        });
    }
    Ok(dst.width / routes.width)
}

fn routed_rows<P>(out_rows: i32, out_width: i32, aligned: Region<P>) -> Result<i32, Refusal> {
    let routes = out_rows.saturating_mul(out_width) / aligned.width;
    if routes <= 0 {
        return Err(Refusal::Empty {
            what: "the routed row count",
        });
    }
    Ok(routes)
}

/// The `Moe` family, claimed. Every point in it. The three routers and
/// `weighted_sum` are delegations to the routine below that already fires
/// the point — the router's own `experts` and `top_k` ride the declaration
/// because the statement states them and the results are sized from them,
/// while the routines read the same two numbers back off the rectangles
/// they were given. The two routed GEMMs and the shared-expert combine are
/// BODIES: no routine in this file is their launcher, and each says why.
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
        topk_softmax(self, logits, routes, weights)
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
        topk_sigmoid(
            self,
            logits,
            routes,
            weights,
            // The routine takes an OPTIONAL correction bias; no statement of
            // this point carries one — `topk_sqrt_softplus` is the router
            // that does, and it states the bias in its own declaration.
            None,
            Const::new(renormalize),
            Const::new(scaling),
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
        topk_sqrtsoftplus(
            self,
            logits,
            routes,
            weights,
            Some(bias),
            Const::new(renormalize),
            Const::new(scaling),
        )
    }

    fn weighted_sum<T: kernels::points::Scalar>(
        &self,
        routed: In<Tensor<T>>,
        weights: In<Tensor<f32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        token_batched_weighted_sum(self, y, routed, weights)
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

#[routine(bf16, canon = "moe.topk_sigmoid")]
pub fn topk_sigmoid<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>,
) -> Result<(), Refusal> {
    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/topk_sigmoid.cuh",
            crate::jit::symbol(&format!("::pie::moe::topk_sigmoid<{}>", T::CPP)),
        )
        .apply(rms(rect.rows.unsigned_abs())),
        &[
            rect.ptr.arg(),
            routed.ptr.arg(),
            topk_w.arg(),
            correction_bias.arg(),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

#[routine(bf16, canon = "moe.topk_sqrt_softplus")]
pub fn topk_sqrtsoftplus<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>,
) -> Result<(), Refusal> {
    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/dsv4_routing.cuh",
            crate::jit::symbol(&format!("::pie::moe::topk_sqrtsoftplus<{}>", T::CPP)),
        )
        .apply(rms(rect.rows.unsigned_abs())),
        &[
            rect.ptr.arg(),
            routed.ptr.arg(),
            topk_w.arg(),
            correction_bias.arg(),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

#[routine]
pub fn hash_route_lookup(
    ctx: &Ctx<'_>,
    token_ids: In<Tensor<i32>>,
    tid2eid: Const<Tensor<i64>>,
    logits: In<Tensor<bf16>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    vocab_size: Const<i32>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>,
) -> Result<(), Refusal> {
    let vocab_size = *vocab_size;
    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

    const DSV4_BLOCK: u32 = 256;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (tokens, num_experts, top_k) = (token_ids.rows, rect.width, routed.width);
    ctx.fire(
        Fire::at(
            "moe/dsv4_routing.cuh",
            "::pie::moe::hash_route_lookup<::pie::bf16>",
        )
        .apply(Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK)),
        &[
            token_ids.arg(),
            tid2eid.arg(),
            rect.ptr.arg(),
            routed.ptr.arg(),
            topk_w.arg(),
            tokens.arg(),
            vocab_size.arg(),
            num_experts.arg(),
            top_k.arg(),
            renormalize.arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

#[routine(bf16, canon = "moe.topk_softmax")]
pub fn topk_softmax<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (num_experts, k) = (rect.width, routed.width);
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(
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
            topk_w.arg(),
            num_experts.arg(),
            k.arg(),
            0_i32.arg(),
        ],
    )
}

#[routine]
pub fn topk_sigmoid_bias_fp32(
    ctx: &Ctx<'_>,
    logits: In<Tensor<f32>>,
    correction_bias: Const<Tensor<f32>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    normalize: Const<bool>,
    routed_scaling_factor: Const<f32>,
) -> Result<(), Refusal> {
    let normalize = *normalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (num_experts, k) = (rect.width, routed.width);
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/topk_softmax.cuh",
            "::pie::moe::topk_sigmoid_bias<::pie::moe::f32>",
        )
        .apply(router_lane(rect.rows.unsigned_abs())),
        &[
            rect.ptr.arg(),
            correction_bias.arg(),
            routed.ptr.arg(),
            topk_w.arg(),
            num_experts.arg(),
            k.arg(),
            i32::from(normalize).arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

#[routine(bf16, out(topk_w = like(topk_w)))]
pub fn apply_per_expert_scale<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    topk_w: InOut<Tensor<f32>>,
    per_expert_scale: Const<Tensor<T>>,
) -> Result<(), Refusal> {
    let total = topk_w.rows.saturating_mul(topk_w.width);
    ctx.fire(
        Fire::at(
            "moe/topk_softmax.cuh",
            crate::jit::symbol(&format!("::pie::moe::apply_per_expert_scale<{}>", T::CPP)),
        )
        .apply(elementwise(total.unsigned_abs())),
        &[
            topk_idx.arg(),
            topk_w.arg(),
            per_expert_scale.arg(),
            total.arg(),
        ],
    )
}

#[routine(bf16)]
pub fn moe_gate_up_decode_gemv<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    norm_x: In<Tensor<T>>,
    gate_up_base: Const<Tensor<T>>,
    expert_gate_up: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let routed = topk_idx.all("the route width")?;
    let dst = expert_gate_up.all("the routed destination's width")?;
    let i_moe = per_route(dst, routed)?;
    let src = norm_x.all("H, the hidden size")?;
    let (num_tokens, top_k, h) = (routed.rows, routed.width, src.width);
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    if h % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow {
            what: "H, in whole float4 loads of 8",
            at: i64::from(h),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_by_token<{}>", T::CPP)),
        )
        .apply(Launch::grid(
            [
                n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                routes.unsigned_abs(),
                1,
            ],
            [WARP, GEMV_WARPS.unsigned_abs(), 1],
        )),
        &[
            routed.ptr.arg(),
            src.ptr.arg(),
            gate_up_base.arg(),
            dst.ptr.arg(),
            top_k.arg(),
            h.arg(),
            n.arg(),
            (i64::from(n) * i64::from(h)).arg(),
        ],
    )
}

#[routine(bf16)]
pub fn moe_down_decode_gemv<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    expert_act: In<Tensor<T>>,
    down_base: Const<Tensor<T>>,
    expert_out: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let routed = topk_idx.all("the route width")?;
    let dst = expert_out.all("the routed destination's width")?;
    let h = per_route(dst, routed)?;
    let act = expert_act.all("I_moe, the per-expert intermediate size")?;
    let (num_tokens, top_k, i_moe) = (routed.rows, routed.width, act.width);
    let routes = num_tokens * top_k;
    if i_moe % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow {
            what: "I_moe, in whole float4 loads of 8",
            at: i64::from(i_moe),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_by_route<{}>", T::CPP)),
        )
        .apply(Launch::grid(
            [
                h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                routes.unsigned_abs(),
                1,
            ],
            [WARP, GEMV_WARPS.unsigned_abs(), 1],
        )),
        &[
            routed.ptr.arg(),
            act.ptr.arg(),
            down_base.arg(),
            dst.ptr.arg(),
            top_k.arg(),
            i_moe.arg(),
            h.arg(),
            (i64::from(h) * i64::from(i_moe)).arg(),
        ],
    )
}

#[routine]
pub fn transpose_expert_scales_u8(
    ctx: &Ctx<'_>,
    src: Const<Tensor<u8>>,
    dst: Out<Tensor<u8>>,
    num_experts: Const<i32>,
    n: Const<i32>,
    k_groups: Const<i32>,
) -> Result<(), Refusal> {
    let num_experts = *num_experts;
    let n = *n;
    let k_groups = *k_groups;
    const BX: u32 = 32;
    const BY: u32 = 8;
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            "::pie::moe::transpose_expert_scales<::pie::u8>",
        )
        .apply(Launch::grid(
            [
                k_groups.unsigned_abs().div_ceil(BX),
                n.unsigned_abs().div_ceil(BY),
                num_experts.unsigned_abs(),
            ],
            [BX, BY, 1],
        )),
        &[src.arg(), dst.ptr.arg(), n.arg(), k_groups.arg()],
    )
}

#[routine(whole, untraced, driver)]
pub fn build_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    gate_up_base: Const<Tensor<bf16>>,
    down_base: Const<Tensor<bf16>>,
    aligned_in: In<Tensor<bf16>>,
    aligned_gate_up: Out<Tensor<bf16>>,
    aligned_act: Out<Tensor<bf16>>,
    aligned_out: Out<Tensor<bf16>>,
    // BARE, AND THE TYPE SYSTEM IS THE REASON. A mark carries its operand as
    // `Tensor<E>`, and `E` must be an `Elem`; `*const bf16` is not one, so a
    // pointer-to-pointer ARRAY has no carrier a mark could wrap. That is not a
    // gap in this signature — `#[routine]` names the bare pointer *"THE STATED
    // ABSENCE"* and admits it precisely here, on a row the trace never binds.
    //
    // The three rectangles above ARE marks because a staging buffer is a
    // `Tensor<bf16>`; these six are addresses of addresses, which no rectangle
    // describes.
    a_gu_ptrs: *mut *const bf16,
    b_gu_ptrs: *mut *const bf16,
    c_gu_ptrs: *mut *mut bf16,
    a_dn_ptrs: *mut *const bf16,
    b_dn_ptrs: *mut *const bf16,
    c_dn_ptrs: *mut *mut bf16,
    // Null when the text has no shared expert; the rewrite below makes it safe.
    shared_gate_up_base: Const<Tensor<bf16>>,
    shared_down_base: Const<Tensor<bf16>>,
    max_blocks: i32,
    block_size: i32,
    routed_blocks: i32,
) -> Result<(), Refusal> {
    let (shared_gate_up_base, shared_down_base) = (shared_gate_up_base.v, shared_down_base.v);

    let aligned_rows = max_blocks.saturating_mul(block_size);
    let hidden = aligned_out.over(aligned_rows, "H, the hidden size")?;
    let inter = aligned_act.over(aligned_rows, "I_moe, the per-expert intermediate size")?;
    let (h, i_moe) = (hidden.width, inter.width);
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            "::pie::moe::build_moe_ptrs_aligned<::pie::bf16>",
        )
        .apply(Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK)),
        &[
            expert_ids.arg(),
            gate_up_base.arg(),
            down_base.arg(),
            aligned_in.arg(),
            aligned_gate_up.arg(),
            inter.ptr.arg(),
            hidden.ptr.arg(),
            a_gu_ptrs.arg(),
            b_gu_ptrs.arg(),
            c_gu_ptrs.arg(),
            a_dn_ptrs.arg(),
            b_dn_ptrs.arg(),
            c_dn_ptrs.arg(),
            max_blocks.arg(),
            block_size.arg(),
            h.arg(),
            i_moe.arg(),
            routed_blocks.arg(),
            shared_gate_up_base.arg(),
            shared_down_base.arg(),
        ],
    )
}

#[routine(bf16, whole)]
pub fn reorder_moe_aligned_output<T>(
    ctx: &Ctx<'_>,
    aligned_out: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    route_out: Out<Tensor<T>>,
) -> Result<(), Refusal>
where
    *const T: Abi + kernels::Bind<crate::jit::ArgValue>,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Read: Into<*const T>,
    <T as kernels::Elem>::Write: Into<*mut T>,
{
    fn ptr_of<T>(p: impl Into<*const T>) -> *const c_void {
        p.into().cast()
    }

    fn ptr_of_mut<T>(p: impl Into<*mut T>) -> *const c_void {
        p.into().cast_const().cast()
    }

    #[must_use]
    fn moe_vectorizable(a: *const c_void, b: *const c_void, hidden: i32) -> bool {
        hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
    }

    let aligned = aligned_out.all("the aligned rectangle's width")?;
    let num_routes = routed_rows(route_out.rows, route_out.width, aligned)?;
    let (aligned_rows, hidden, num_tokens) = (sorted_route_ids.rows, aligned.width, route_out.rows);

    let shared_row_begin = -1;
    let vectorizable = moe_vectorizable(ptr_of(aligned.ptr), ptr_of_mut(route_out.ptr), hidden);
    let width = if vectorizable {
        hidden / MOE_VEC_WIDTH
    } else {
        hidden
    };
    let instantiation = if vectorizable {
        format!("::pie::moe::reorder_moe_aligned_output_vec<{}>", T::CPP)
    } else {
        format!("::pie::moe::reorder_moe_aligned_output<{}>", T::CPP)
    };
    ctx.fire(
        Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&instantiation)).apply(Launch::grid(
            [
                aligned_rows.unsigned_abs(),
                width.unsigned_abs().div_ceil(DISPATCH_BLOCK),
                1,
            ],
            [DISPATCH_BLOCK, 1, 1],
        )),
        &[
            aligned.ptr.arg(),
            sorted_route_ids.arg(),
            route_out.arg(),
            num_routes.arg(),
            aligned_rows.arg(),
            width.arg(),
            shared_row_begin.arg(),
            num_tokens.arg(),
            core::ptr::null_mut::<T>().arg(),
        ],
    )
}

#[routine(whole)]
pub fn moe_align_decode(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    expert_ids: Out<Tensor<i32>>,
    route_to_aligned_row: Out<Tensor<i32>>,
    num_experts: Const<i32>,
    block_size: Const<i32>,
    max_blocks: Const<i32>,
) -> Result<(), Refusal> {
    #[must_use]
    const fn router_sort(n_experts: u32) -> Launch {
        Launch::per_row(1, SORT_BLOCK).smem((3 * n_experts + 34) * FLOAT)
    }

    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            "::pie::moe::moe_align_decode<::pie::i32>",
        )
        .apply(router_sort(num_experts.unsigned_abs())),
        &[
            topk_idx.arg(),
            sorted_route_ids.arg(),
            expert_ids.arg(),
            route_to_aligned_row.arg(),
            num_routes.arg(),
            num_experts.arg(),
            block_size.arg(),
            max_blocks.arg(),
            core::ptr::null_mut::<i32>().arg(),
        ],
    )
}

#[routine(whole)]
pub fn moe_bucket_exact(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    route_to_sorted_row: Out<Tensor<i32>>,
    counts_out: Out<Tensor<i32>>,
) -> Result<(), Refusal> {
    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let counts = counts_out.all("num_experts")?;
    let num_experts = counts.width;
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            "::pie::moe::moe_bucket_exact<::pie::i32>",
        )
        .apply(
            Launch::grid([1, 1, 1], [SORT_BLOCK, 1, 1])
                .smem((3 * num_experts.unsigned_abs() + 1) * FLOAT),
        ),
        &[
            topk_idx.arg(),
            sorted_route_ids.arg(),
            route_to_sorted_row.arg(),
            counts.ptr.arg(),
            num_routes.arg(),
            num_experts.arg(),
        ],
    )
}

#[routine(bf16, whole)]
pub fn gather_moe_aligned_inputs<T>(
    ctx: &Ctx<'_>,
    norm_x: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    aligned_in: Out<Tensor<T>>,
    top_k: Const<i32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    let top_k = *top_k;

    let tokens = *tokens;
    let (aligned_rows, hidden) = (sorted_route_ids.rows, aligned_in.width);

    let num_tokens = tokens;
    let num_routes = num_tokens.saturating_mul(top_k);
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!(
                "::pie::moe::gather_moe_aligned_inputs<{}>",
                T::CPP
            )),
        )
        .apply(elementwise_rows(
            aligned_rows.unsigned_abs(),
            hidden.unsigned_abs(),
        )),
        &[
            norm_x.arg(),
            sorted_route_ids.arg(),
            aligned_in.arg(),
            num_routes.arg(),
            aligned_rows.arg(),
            top_k.arg(),
            hidden.arg(),
            (-1i32).arg(),
            num_tokens.arg(),
        ],
    )
}

#[routine(bf16, canon = "moe.weighted_sum")]
pub fn token_batched_weighted_sum<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,
) -> Result<(), Refusal> {
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!(
                "::pie::moe::token_batched_weighted_sum<{}>",
                T::CPP
            )),
        )
        .apply(elementwise_rows(
            out.rows.unsigned_abs(),
            out.width.unsigned_abs(),
        )),
        &[
            out.arg(),
            src.ptr.arg(),
            fan.ptr.arg(),
            top_k.arg(),
            out.width.arg(),
        ],
    )
}

#[routine(bf16, out(out = like(out)))]
pub fn token_batched_weighted_sum_add<T>(
    ctx: &Ctx<'_>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,
    out: InOut<Tensor<T>>,
) -> Result<(), Refusal> {
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!(
                "::pie::moe::token_batched_weighted_sum_add<{}>",
                T::CPP
            )),
        )
        .apply(elementwise_rows(
            out.rows.unsigned_abs(),
            out.width.unsigned_abs(),
        )),
        &[
            out.arg(),
            src.ptr.arg(),
            fan.ptr.arg(),
            top_k.arg(),
            out.width.arg(),
        ],
    )
}

#[routine(bf16, internal)]
pub fn scalar_weighted_add<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    weight: Const<f32>,
) -> Result<(), Refusal> {
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!("::pie::moe::scalar_weighted_add<{}>", T::CPP)),
        )
        .apply(elementwise(n.unsigned_abs())),
        &[out.arg(), src.ptr.arg(), weight.arg(), n.arg()],
    )
}

#[routine(bf16, whole, out(out = like(out)))]
pub fn add_moe_route_bias<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    bias: Const<Tensor<T>>,
    topk_idx: In<Tensor<i32>>,
    out_stride: Const<i32>,
) -> Result<(), Refusal> {
    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let dst = out.all("the bias column count")?;

    let out_stride = Stride(*out_stride);

    if dst.width > out_stride.0 {
        return Err(Refusal::Wide {
            what: "the bias column count against the destination's row pitch",
            at: i64::from(dst.width),
            max: i64::from(out_stride.0),
        });
    }
    ctx.fire(
        Fire::at(
            "moe/moe_dispatch.cuh",
            crate::jit::symbol(&format!("::pie::moe::add_moe_route_bias<{}>", T::CPP)),
        )
        .apply(rms(num_routes.unsigned_abs())),
        &[
            dst.ptr.arg(),
            bias.arg(),
            topk_idx.arg(),
            num_routes.arg(),
            dst.width.arg(),
            out_stride.arg(),
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

#[routine(untraced)]
pub fn flashinfer_cutlass_moe_bf16(
    _ctx: &Ctx<'_>,
    _x: In<Tensor<bf16>>,
    _experts: In<Tensor<c_void>>,
    _weights: Const<Tensor<c_void>>,
    _out: Out<Tensor<bf16>>,
    _tokens: i32,
    _hidden: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the fused CUTLASS MoE leg, retired with its instantiation seam \
               rather than carried: the aligned leg is the only leg left, and \
               `moe_cutlass_max_rows = 0` is what selects it",
    })
}
