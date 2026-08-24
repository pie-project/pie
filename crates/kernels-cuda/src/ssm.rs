#![allow(clippy::too_many_arguments)]

use crate::jit::Abi;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use crate::jit::{Ctx, Launch};
use crate::views::RecurrentState;
use kernels::Refusal;
use kernels::plane::{Cache, Const, In, Out};
use kernels::raises::Struct;
use kernels::{Bind, Fire};

use core::ffi::c_void;

const RULE_BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}

#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    const SINK_BLOCK_MIN: u32 = WARP;

    const SINK_BLOCK_MAX: u32 = 128;

    Launch::grid(
        [rows, heads, 1],
        [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
    )
}

#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch::grid([rows, heads, 1], [RULE_BLOCK, 1, 1])
}

#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    const SCAN_WARPS: u32 = 4;

    Launch::grid(
        [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        [SCAN_WARPS * WARP, 1, 1],
    )
}

#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

const GDN_BLOCK: u32 = 128;

fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if T::CPP == <bf16 as kernels::Elem>::CPP {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

fn recurrent(
    rsv: In<Struct<RecurrentState>>,
) -> Result<&'static crate::views::RecurrentView, Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    Ok(unsafe { &*rsv.ptr })
}

fn conv_shape<T: kernels::Elem>(x: In<Tensor<T>>, conv_width: u32) -> Result<(i32, i32), Refusal> {
    let rect = x.all("the conv's channel count")?;
    let k = i32::try_from(conv_width).map_err(|_| Refusal::Wide {
        what: "the conv width this statement states",
        at: i64::from(conv_width),
        max: i64::from(i32::MAX),
    })?;
    Ok((rect.width, k))
}

#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    fn causal_conv1d<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d at an element other than bf16")?;
        let (c, k) = conv_shape(x, conv_width)?;
        let rsv = recurrent(state.raised())?;
        let r = x.rows;
        self.fire(
            Fire::at(
                "ssm/causal_conv1d.cuh",
                "::pie::ssm::causal_conv1d_update_batched<::pie::bf16>",
            )
            .apply(Launch::grid(
                [c.unsigned_abs().div_ceil(RULE_BLOCK), r.unsigned_abs(), 1],
                [RULE_BLOCK, 1, 1],
            )),
            &[
                x.ptr.cast::<bf16>().arg(),
                weight.v.cast::<bf16>().arg(),
                MaybeConst::<bf16>::none().arg(),
                rsv.conv_slab.arg(),
                rsv.slot_ids.arg(),
                rsv.conv_stride.arg(),
                y.ptr.cast::<bf16>().arg(),
                r.arg(),
                c.arg(),
                k.arg(),
            ],
        )
    }

    fn causal_conv1d_chunked<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        weight: Const<Tensor<T>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d_chunked at an element other than bf16")?;
        let (c, k) = conv_shape(x, conv_width)?;
        let rsv = recurrent(state.raised())?;

        let r = indptr.rows;

        const CONV_CHANNEL_TILE_FROM: i32 = 8;
        const CONV_TILE: u32 = 128;
        const CONV_PER_CHANNEL_BLOCK: u32 = 64;
        let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
        let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
            (
                "::pie::ssm::causal_conv1d_prefill_batched_channel_tile<::pie::bf16>",
                Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
            )
        } else {
            (
                "::pie::ssm::causal_conv1d_prefill_batched<::pie::bf16>",
                Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
            )
        };
        self.fire(
            Fire::at("ssm/causal_conv1d.cuh", instantiation).apply(launch),
            &[
                x.ptr.cast::<bf16>().arg(),
                weight.v.cast::<bf16>().arg(),
                MaybeConst::<bf16>::none().arg(),
                y.ptr.cast::<bf16>().arg(),
                rsv.conv_slab.arg(),
                rsv.slot_ids.arg(),
                (indptr.ptr as *const u32).arg(),
                rsv.conv_stride.arg(),
                c.arg(),
                k.arg(),
                true.arg(),
                MaybeConst::<u8>::none().arg(),
                MaybeConst::<i32>::none().arg(),
            ],
        )
    }

    fn gdn_prep<T: kernels::points::Scalar>(
        &self,
        ba: In<Tensor<T>>,
        dt_bias: Const<Tensor<T>>,
        a_log: Const<Tensor<f32>>,
        gates: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.gdn_prep at an element other than bf16")?;
        let src = ba.all("the `[b | a]` projection")?;
        let dst = gates.all("the fused `[g_log | beta]` row")?;
        if src.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(src.width),
            });
        }
        let v_h = src.width / 2;

        if dst.width != src.width || dst.rows != src.rows {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(dst.width),
            });
        }
        const PREP_BLOCK: u32 = 256;
        self.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_ba_gates<::pie::bf16>",
            )
            .apply(Launch::grid(
                [
                    src.rows.unsigned_abs(),
                    v_h.unsigned_abs().div_ceil(PREP_BLOCK),
                    1,
                ],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                ba.ptr.cast::<bf16>().arg(),
                a_log.v.arg(),
                dt_bias.v.cast::<bf16>().arg(),
                gates.ptr.arg(),
                src.rows.arg(),
                v_h.arg(),
            ],
        )
    }

    fn gated_delta<T: kernels::points::Scalar>(
        &self,
        qkv: In<Tensor<T>>,
        z: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta at an element other than bf16")?;

        let shape = GdnShape::of(qkv, gates, k_heads, v_heads, k_dim, v_dim)?;

        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(shape.v_h) * i64::from(shape.v_d) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the stated value heads",
                at: i64::from(result.width),
            });
        }
        if state.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent view this statement names",
            });
        }
        let staged = shape.stage(self, qkv.ptr.cast::<bf16>(), gates.ptr)?;
        shape.step(self, &staged, state.ptr, y)
    }

    fn gated_delta_chunked<T: kernels::points::Scalar>(
        &self,
        qkv: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        z: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta_chunked at an element other than bf16")?;

        let window = Chunked::of(qkv, indptr, gates, k_heads, v_heads, k_dim, v_dim)?;
        let shape = window.g;

        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(shape.v_h) * i64::from(shape.v_d) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the stated value heads",
                at: i64::from(result.width),
            });
        }
        if state.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent view this statement names",
            });
        }
        let rsv = unsafe { &*state.ptr };
        let staged = shape.stage(self, qkv.ptr.cast::<bf16>(), gates.ptr)?;
        window.scan(self, &staged, rsv, y.ptr)
    }

    fn kda_step<T: kernels::points::Scalar>(
        &self,
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_step at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, heads, head_dim, norm_eps, y)?;
        let rsv = raised(state)?;
        let staged = shape.stage(self, mixed, f, b, dt_bias, a_log)?;
        kda_recurrent_step_batched(
            self,
            plane(staged.q_norm, shape.n, shape.w),
            plane(staged.k_norm, shape.n, shape.w),
            plane(staged.v, shape.n, shape.w),
            plane(staged.gate, shape.n, shape.w),
            plane(staged.beta, shape.n, shape.h),
            Out {
                ptr: y.ptr,
                rows: shape.n,
                width: shape.w,
            },
            Const::new(shape.h),
            Const::new(shape.d),
            rsv,
        )
    }

    fn kda_chunked<T: kernels::points::Scalar>(
        &self,
        mixed: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_chunked at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, heads, head_dim, norm_eps, y)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        let rsv = raised(state)?;
        let staged = shape.stage(self, mixed, f, b, dt_bias, a_log)?;
        let q_norm = plane(staged.q_norm, shape.n, shape.w);
        let k_norm = plane(staged.k_norm, shape.n, shape.w);
        let v = plane(staged.v, shape.n, shape.w);
        let gate = plane(staged.gate, shape.n, shape.w);
        let beta = plane(staged.beta, shape.n, shape.h);
        let out: Out<Tensor<f32>> = Out {
            ptr: y.ptr,
            rows: shape.n,
            width: shape.w,
        };
        let (h, d) = (shape.h, shape.d);
        let rsv = recurrent(rsv)?;

        let r = indptr.rows;
        let qo_indptr = indptr.ptr as *const u32;
        const KDA_PREFILL_MAX_WARPS: i32 = 32;
        self.fire(
            Fire::at("ssm/kda.cuh", "::pie::ssm::kda_prefill_batched").apply(
                Launch::grid(
                    [r.unsigned_abs(), h.unsigned_abs(), 1],
                    [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
                )
                .smem(kda_shmem(d.unsigned_abs())),
            ),
            &[
                q_norm.arg(),
                k_norm.arg(),
                v.arg(),
                gate.arg(),
                beta.arg(),
                rsv.slab.arg(),
                rsv.slot_ids.arg(),
                qo_indptr.arg(),
                rsv.slot_stride_elems.arg(),
                out.arg(),
                h.arg(),
                d.arg(),
            ],
        )
    }
}

fn raised(
    state: Cache<kernels::raises::Struct<RecurrentState>>,
) -> Result<In<Struct<RecurrentState>>, Refusal> {
    if state.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    Ok(state.raised())
}

const fn plane(ptr: *mut f32, rows: i32, width: i32) -> In<Tensor<f32>> {
    In {
        ptr: ptr.cast_const(),
        rows,
        width,
    }
}

#[derive(Clone, Copy)]
struct Kda {
    n: i32,
    h: i32,
    d: i32,

    w: i32,
    eps: f32,
}

#[derive(Clone, Copy)]
struct KdaStaged {
    q_norm: *mut f32,
    k_norm: *mut f32,
    v: *mut f32,
    gate: *mut f32,
    beta: *mut f32,
}

impl Kda {
    fn of<T: kernels::points::Scalar>(
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<Kda, Refusal> {
        fn stated(n: u32, what: &'static str) -> Result<i32, Refusal> {
            match i32::try_from(n) {
                Ok(0) => Err(Refusal::Empty { what }),
                Ok(n) => Ok(n),
                Err(_) => Err(Refusal::Wide {
                    what,
                    at: i64::from(n),
                    max: i64::from(i32::MAX),
                }),
            }
        }
        let h = stated(heads, "the KDA heads this statement states")?;
        let d = stated(head_dim, "the KDA head width this statement states")?;
        let wide = i64::from(h) * i64::from(d);
        let w = i32::try_from(wide).map_err(|_| Refusal::Wide {
            what: "the KDA plane the two stated head numbers multiply out to",
            at: wide,
            max: i64::from(i32::MAX),
        })?;

        let packed = mixed.all("the post-convolution `[q | k | v]` row")?;
        if i64::from(packed.width) != 3 * i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the post-convolution `[q | k | v]` row, against the two stated head numbers",
                at: i64::from(packed.width),
            });
        }
        let n = packed.rows;

        let forget = f.all("the forget projection this statement hands over")?;
        if i64::from(forget.width) != i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the forget projection's row, against the two stated head numbers",
                at: i64::from(forget.width),
            });
        }
        let beta = b.all("the beta projection this statement hands over")?;
        if i64::from(beta.width) != i64::from(h) {
            return Err(Refusal::Narrow {
                what: "the beta projection's row, against the stated head count",
                at: i64::from(beta.width),
            });
        }
        if forget.rows != n || beta.rows != n {
            return Err(Refusal::Narrow {
                what: "the forget and beta projections do not carry the packed row's rows",
                at: i64::from(forget.rows.min(beta.rows)),
            });
        }

        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the two stated head numbers",
                at: i64::from(result.width),
            });
        }
        Ok(Kda {
            n,
            h,
            d,
            w,
            eps: norm_eps,
        })
    }

    fn plane_elems(self, width: i32, what: &'static str) -> Result<usize, Refusal> {
        let elems = i64::from(self.n) * i64::from(width);
        usize::try_from(elems).map_err(|_| Refusal::Wide {
            what,
            at: elems,
            max: i64::MAX,
        })
    }

    fn stage<T: kernels::points::Scalar>(
        self,
        ctx: &Ctx<'_>,
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
    ) -> Result<KdaStaged, Refusal> {
        const PREP_BLOCK: u32 = 128;

        const PLANES: u32 = 3;

        let Kda { n, h, d, w, eps } = self;
        let wide = self.plane_elems(w, "the `[q | k | v]` plane this fire stages")?;
        let decay = self.plane_elems(h, "the beta column this fire stages")?;

        let qkv = ctx
            .scratch("ssm::kda_qkv", 3 * wide * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let gb = ctx
            .scratch(
                "ssm::kda_gates",
                (wide + decay) * core::mem::size_of::<f32>(),
            )?
            .cast::<f32>();
        let staged = KdaStaged {
            q_norm: qkv,
            k_norm: unsafe { qkv.add(wide) },
            v: unsafe { qkv.add(2 * wide) },
            gate: gb,
            beta: unsafe { gb.add(wide) },
        };

        ctx.fire(
            Fire::at("ssm/kda.cuh", "::pie::ssm::kda_qkv_prep<::pie::bf16, 128>").apply(
                Launch::grid([n.unsigned_abs(), PLANES, 1], [PREP_BLOCK, 1, 1]),
            ),
            &[
                mixed.ptr.cast::<bf16>().arg(),
                staged.q_norm.arg(),
                staged.k_norm.arg(),
                staged.v.arg(),
                w.arg(),
                eps.arg(),
            ],
        )?;

        kda_gate_beta::<bf16>(
            ctx,
            In {
                ptr: f.ptr.cast::<bf16>(),
                rows: n,
                width: w,
            },
            In {
                ptr: b.ptr.cast::<bf16>(),
                rows: n,
                width: h,
            },
            a_log,
            dt_bias,
            Out {
                ptr: staged.gate,
                rows: n,
                width: w,
            },
            Out {
                ptr: staged.beta,
                rows: n,
                width: h,
            },
            Const::new(d),
        )?;
        Ok(staged)
    }
}

#[derive(Clone, Copy)]
struct GdnShape {
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
}

#[derive(Clone, Copy)]
struct Chunked {
    g: GdnShape,
    r: i32,
    qo_indptr: *const u32,
}

#[derive(Clone, Copy)]
struct Staged {
    q_norm: *mut f32,
    k_norm: *mut f32,
    v: *mut f32,
    g_log: *mut f32,
    beta: *mut f32,
}

impl GdnShape {
    fn of<T: kernels::points::Scalar>(
        qkv: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<GdnShape, Refusal> {
        fn stated(n: u32, what: &'static str) -> Result<i32, Refusal> {
            match i32::try_from(n) {
                Ok(0) => Err(Refusal::Empty { what }),
                Ok(n) => Ok(n),
                Err(_) => Err(Refusal::Wide {
                    what,
                    at: i64::from(n),
                    max: i64::from(i32::MAX),
                }),
            }
        }
        let k_h = stated(k_heads, "the key heads this statement states")?;
        let v_h = stated(v_heads, "the value heads this statement states")?;
        let k_d = stated(k_dim, "the key head width this statement states")?;
        let v_d = stated(v_dim, "the value head width this statement states")?;
        if v_h % k_h != 0 {
            return Err(Refusal::Narrow {
                what: "v_h per k_h",
                at: i64::from(v_h),
            });
        }
        let packed = qkv.all("the post-convolution qkv")?;

        let want = 2 * i64::from(k_h) * i64::from(k_d) + i64::from(v_h) * i64::from(v_d);
        if i64::from(packed.width) != want {
            return Err(Refusal::Narrow {
                what: "the post-convolution qkv's row, against the four stated head numbers",
                at: i64::from(packed.width),
            });
        }
        let fused = gates.all("the fused `[g_log | beta]` row")?;
        if i64::from(fused.width) != 2 * i64::from(v_h) {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the stated value heads",
                at: i64::from(fused.width),
            });
        }
        Ok(GdnShape {
            n: packed.rows,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim: packed.width,
        })
    }

    fn plane(self, heads: i32, width: i32, what: &'static str) -> Result<usize, Refusal> {
        let elems = i64::from(self.n) * i64::from(heads) * i64::from(width);
        usize::try_from(elems).map_err(|_| Refusal::Wide {
            what,
            at: elems,
            max: i64::MAX,
        })
    }

    fn stage(self, ctx: &Ctx<'_>, qkv: *const bf16, gates: *const f32) -> Result<Staged, Refusal> {
        const PREP_BLOCK: u32 = 128;

        let GdnShape {
            n,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim,
        } = self;
        let key = self.plane(k_h, k_d, "the key plane this recurrence stages")?;
        let val = self.plane(v_h, v_d, "the value plane this recurrence stages")?;
        let decay = self.plane(v_h, 1, "the decay plane this recurrence stages")?;

        let qk = ctx
            .scratch("ssm::gdn_chunk_qk", 2 * key * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let v = ctx
            .scratch("ssm::gdn_chunk_v", val * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let gb = ctx
            .scratch(
                "ssm::gdn_chunk_gates",
                2 * decay * core::mem::size_of::<f32>(),
            )?
            .cast::<f32>();
        let staged = Staged {
            q_norm: qk,
            k_norm: unsafe { qk.add(key) },
            v,
            g_log: gb,
            beta: unsafe { gb.add(decay) },
        };

        #[allow(clippy::cast_precision_loss)]
        let q_scale = (k_d as f32).sqrt().recip();
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
            )
            .apply(Launch::grid(
                [n.unsigned_abs(), k_h.unsigned_abs(), 1],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                qkv.arg(),
                staged.q_norm.arg(),
                staged.k_norm.arg(),
                k_h.arg(),
                k_d.arg(),
                conv_dim.arg(),
                q_scale.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_v_gates<::pie::bf16, 128>",
            )
            .apply(Launch::grid(
                [n.unsigned_abs(), v_h.unsigned_abs(), 1],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                qkv.arg(),
                gates.arg(),
                staged.v.arg(),
                staged.g_log.arg(),
                staged.beta.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                conv_dim.arg(),
            ],
        )?;
        Ok(staged)
    }

    fn step(
        self,
        ctx: &Ctx<'_>,
        staged: &Staged,
        rsv: *const crate::views::RecurrentView,
        out: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let GdnShape {
            n,
            k_h,
            v_h,
            k_d,
            v_d,
            ..
        } = self;
        let key = In {
            ptr: staged.q_norm.cast_const(),
            rows: n,
            width: k_h.saturating_mul(k_d),
        };
        recurrent_gated_delta_step_batched_gqa_state_bf16(
            ctx,
            key,
            In {
                ptr: staged.k_norm.cast_const(),
                rows: key.rows,
                width: key.width,
            },
            In {
                ptr: staged.v.cast_const(),
                rows: n,
                width: v_h.saturating_mul(v_d),
            },
            In {
                ptr: staged.g_log.cast_const(),
                rows: n,
                width: v_h,
            },
            In {
                ptr: staged.beta.cast_const(),
                rows: n,
                width: v_h,
            },
            out,
            Const::new(k_h),
            Const::new(v_h),
            Const::new(k_d),
            Const::new(v_d),
            Const::new(n),
            In {
                ptr: rsv,
                rows: 0,
                width: 0,
            },
        )
    }
}

impl Chunked {
    fn of<T: kernels::points::Scalar>(
        qkv: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        gates: In<Tensor<f32>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Chunked, Refusal> {
        let g = GdnShape::of(qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        Ok(Chunked {
            g,
            r: indptr.rows,

            qo_indptr: indptr.ptr.cast::<u32>(),
        })
    }

    fn scan(
        self,
        ctx: &Ctx<'_>,
        staged: &Staged,
        rsv: &crate::views::RecurrentView,
        out: *mut f32,
    ) -> Result<(), Refusal> {
        const BK_MAX_FLA: i32 = 128;

        const BV_FLA: u32 = 128;

        const WARP_TILED_K_MAX: i32 = 256;

        let Chunked { g, r, qo_indptr } = self;
        let GdnShape {
            k_h, v_h, k_d, v_d, ..
        } = g;
        let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
        let state_base = rsv.slab;
        if state_base.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent slab this statement's cache row names",
            });
        }
        let slot_ids = rsv.slot_ids;
        let slot_stride_elems = rsv.slot_stride_elems;

        let write_state = true;

        if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
            return ctx.fire(
                Fire::at(
                    "ssm/gated_delta_net.cuh",
                    "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::state_bf16, 128, 128>",
                )
                .apply(
                    Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                        .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
                ),
                &[
                    staged.q_norm.cast_const().arg(),
                    staged.k_norm.cast_const().arg(),
                    staged.v.cast_const().arg(),
                    staged.g_log.cast_const().arg(),
                    staged.beta.cast_const().arg(),
                    state_base.arg(),
                    slot_ids.arg(),
                    qo_indptr.arg(),
                    slot_stride_elems.arg(),
                    out.arg(),
                    k_h.arg(),
                    v_h.arg(),
                    k_d.arg(),
                    v_d.arg(),
                    write_state.arg(),

                    MaybeConst::<i32>::none().arg(),
                    MaybeConst::<u8>::none().arg(),
                ],
            );
        }

        if k_d <= WARP_TILED_K_MAX {
            return ctx.fire(
                Fire::at(
                    "ssm/gated_delta_net.cuh",
                    "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>",
                )
                .apply(warp_tiled_scan(rows, heads, v_d.unsigned_abs())),
                &[
                    staged.q_norm.cast_const().arg(),
                    staged.k_norm.cast_const().arg(),
                    staged.v.cast_const().arg(),
                    staged.g_log.cast_const().arg(),
                    staged.beta.cast_const().arg(),
                    state_base.arg(),
                    slot_ids.arg(),
                    qo_indptr.arg(),
                    slot_stride_elems.arg(),
                    out.arg(),
                    k_h.arg(),
                    v_h.arg(),
                    k_d.arg(),
                    v_d.arg(),
                    write_state.arg(),
                    MaybeConst::<u8>::none().arg(),
                ],
            );
        }

        let (q_norm, k_norm) = if v_h == k_h {
            (staged.q_norm.cast_const(), staged.k_norm.cast_const())
        } else {
            let wide = g.plane(v_h, k_d, "the repeated key plane this window stages")?;
            let rep = ctx
                .scratch(
                    "ssm::gdn_chunk_repeat",
                    2 * wide * core::mem::size_of::<f32>(),
                )?
                .cast::<f32>();
            let (q, k) = (rep, unsafe { rep.add(wide) });
            for (src, dst) in [(staged.q_norm, q), (staged.k_norm, k)] {
                ctx.fire(
                    Fire::at(
                        "ssm/gated_delta_net_prep.cuh",
                        "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
                    )
                    .apply(gated_rms(g.n.unsigned_abs(), heads)),
                    &[
                        src.cast_const().arg(),
                        dst.arg(),
                        k_h.arg(),
                        v_h.arg(),
                        k_d.arg(),
                        (v_h / k_h).arg(),
                    ],
                )?;
            }
            (q.cast_const(), k.cast_const())
        };
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net.cuh",
                "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
            )
            .apply(
                Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1])
                    .smem(2 * k_d.unsigned_abs() * FLOAT),
            ),
            &[
                q_norm.arg(),
                k_norm.arg(),
                staged.v.cast_const().arg(),
                staged.g_log.cast_const().arg(),
                staged.beta.cast_const().arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx<'_>,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    y: *mut T,
    state_out: *mut T,
    n: i32,
    channels: i32,
    k: i32,
) -> Result<(), Refusal>
where
    T: kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    ctx.fire(
        Fire::at(
            "ssm/causal_conv1d.cuh",
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill<{}, false>",
                T::CPP
            )),
        )
        .apply(Launch::grid([channels.unsigned_abs(), 1, 1], [64, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            state_out.arg(),
            n.arg(),
            channels.arg(),
            k.arg(),
        ],
    )
}

pub fn bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "element count",
        });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::widen<::pie::bf16>",
        )
        .apply(elementwise(count)),
        &[x.arg(), y.arg(), elems.arg()],
    )
}

pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx<'_>,
    in_: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
        )
        .apply(gated_rms(in_.rows.unsigned_abs(), v_h.unsigned_abs())),
        &[
            in_.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            d.arg(),
            (*v_h / *k_h).arg(),
        ],
    )
}

pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
        const PER_ROW_NARROW_BLOCK: u32 = 128;

        Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    let dst = y.all("the normalised row")?;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::l2norm_scale<::pie::bf16, 128>",
        )
        .apply(per_row_narrow(dst.rows.unsigned_abs())),
        &[x.arg(), y.arg(), dst.width.arg(), 1.0f32.arg(), eps.arg()],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn kda_gate_beta<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    raw_g: In<Tensor<T>>,
    raw_beta: In<Tensor<T>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<f32>>,
    gate_out: Out<Tensor<f32>>,
    beta_out: Out<Tensor<f32>>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    let betas = beta_out.all("the KDA head count")?;
    let t = betas.rows;

    let h = betas.width;
    ctx.fire(
        Fire::at(
            "ssm/kda.cuh",
            crate::jit::symbol(&format!("::pie::ssm::kda_gate_beta<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            t.unsigned_abs(),
            h.unsigned_abs(),
            d.unsigned_abs(),
        )),
        &[
            raw_g.arg(),
            raw_beta.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gate_out.arg(),
            beta_out.arg(),
            t.arg(),
            h.arg(),
            d.arg(),
            0.0f32.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    h: Const<i32>,
    d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;

    let r = out.rows;
    const KDA_STEP_BLOCK: u32 = 256;
    ctx.fire(
        Fire::at("ssm/kda.cuh", "::pie::ssm::kda_recurrent_step_batched").apply(
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [KDA_STEP_BLOCK, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            gate.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    let (v_h, k_d, v_d) = (*v_h, *k_d, *v_d);
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        )
        .apply(
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            write_state.arg(),
            MaybeConst::<u8>::none().arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    const SMEM_BV: u32 = 128;

    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }

    let (instantiation, launch) = if *v_d == GDN_SMEM_ARM_WIDTH && *k_d == GDN_SMEM_ARM_WIDTH {
        (
            "::pie::ssm::recurrent_step_batched_gqa_smem<::pie::ssm::gqa_smem_bv>",
            Launch::grid(
                [
                    v_d.unsigned_abs().div_ceil(SMEM_BV),
                    r.unsigned_abs(),
                    v_h.unsigned_abs(),
                ],
                [SMEM_BV, 1, 1],
            )
            .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
        )
    } else {
        (
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::state_bf16, false>",
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(2 * k_d.unsigned_abs() * FLOAT),
        )
    };
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(launch),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    Conv,
    Recurrent,
}

#[derive(Clone, Copy, Debug)]
pub struct Gdn {
    pub k_h: i32,
    pub v_h: i32,
    pub k_d: i32,
    pub v_d: i32,
    pub conv_dim: i32,
    pub conv_k: i32,
    pub n_groups: i32,
    pub conv_stride_elems: i64,
    pub state_stride_elems: i64,
    pub slot_ids_d: *const i32,
    pub write_state: bool,
}
