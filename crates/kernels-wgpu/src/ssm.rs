use kernels::BindMut;
use kernels::plane::{Cache, Refusal};
use kernels::raises::Struct;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use crate::points::{Payload, at_bf16};
use crate::views::{RecurrentState, RecurrentView};

const SCAN_WIDTH: u32 = 128;

const SCAN_HEAD_MAX: i32 = 256;

fn recurrent<'a>(state: Cache<Struct<RecurrentState>>) -> Result<&'a RecurrentView, Refusal> {
    if state.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    Ok(unsafe { &*state.ptr })
}

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

fn head_width(width: i32, what: &'static str) -> Result<(), Refusal> {
    if width > SCAN_HEAD_MAX {
        return Err(Refusal::Wide {
            what,
            at: i64::from(width),
            max: i64::from(SCAN_HEAD_MAX),
        });
    }
    Ok(())
}

fn conv_lanes(channels: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if channels <= 0 {
        return Err(Refusal::Empty {
            what: "the conv's channel count",
        });
    }
    if channels % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "the conv's channel row is not a whole number of bf16 pairs",
            at: i64::from(channels),
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty {
            what: "the rows this conv covers",
        });
    }
    Ok([channels.unsigned_abs() / 2, rows.unsigned_abs(), 1])
}

fn scan_lanes(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty {
            what: "the heads this scan covers",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty {
            what: "the rows this scan covers",
        });
    }
    Ok([SCAN_WIDTH, heads.unsigned_abs(), rows.unsigned_abs()])
}

fn taps_of(conv_width: u32) -> Result<i32, Refusal> {
    stated(conv_width, "the conv width this statement states")
}

fn requests(indptr: In<Payload<i32>>) -> Result<i32, Refusal> {
    if indptr.rows <= 0 {
        return Err(Refusal::Empty {
            what: "the query CSR this statement names",
        });
    }
    Ok(indptr.rows)
}

struct Delta {
    rows: i32,
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}

impl Delta {
    fn of<T: kernels::points::Scalar>(
        qkv: In<Payload<T>>,
        gates: In<Payload<f32>>,
        y: Out<Payload<f32>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Refusal> {
        let k_h = stated(k_heads, "the key heads this statement states")?;
        let v_h = stated(v_heads, "the value heads this statement states")?;
        let k_d = stated(k_dim, "the key head width this statement states")?;
        let v_d = stated(v_dim, "the value head width this statement states")?;
        if v_h % k_h != 0 {
            return Err(Refusal::Narrow {
                what: "the value heads this statement states, per key head",
                at: i64::from(v_h),
            });
        }
        head_width(
            k_d,
            "the key head width, against the shared row this scan stages",
        )?;
        if qkv.rows <= 0 || qkv.width <= 0 {
            return Err(Refusal::Empty {
                what: "the post-convolution qkv",
            });
        }
        let want = 2 * i64::from(k_h) * i64::from(k_d) + i64::from(v_h) * i64::from(v_d);
        if i64::from(qkv.width) != want {
            return Err(Refusal::Narrow {
                what: "the post-convolution qkv's row, against the four stated head numbers",
                at: i64::from(qkv.width),
            });
        }
        if gates.rows != qkv.rows || i64::from(gates.width) != 2 * i64::from(v_h) {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the stated value heads",
                at: i64::from(gates.width),
            });
        }
        if y.rows != qkv.rows || i64::from(y.width) != i64::from(v_h) * i64::from(v_d) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the stated value heads",
                at: i64::from(y.width),
            });
        }
        Ok(Self {
            rows: qkv.rows,
            k_heads: k_h,
            v_heads: v_h,
            k_dim: k_d,
            v_dim: v_d,
        })
    }
}

struct Kda {
    rows: i32,
    heads: i32,
    head_dim: i32,
}

impl Kda {
    fn of<T: kernels::points::Scalar>(
        mixed: In<Payload<T>>,
        f: In<Payload<T>>,
        b: In<Payload<T>>,
        y: Out<Payload<f32>>,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Refusal> {
        let h = stated(heads, "the KDA heads this statement states")?;
        let d = stated(head_dim, "the KDA head width this statement states")?;
        head_width(
            d,
            "the KDA head width, against the shared row this scan stages",
        )?;
        if mixed.rows <= 0 || mixed.width <= 0 {
            return Err(Refusal::Empty {
                what: "the post-convolution `[q | k | v]` row",
            });
        }
        let plane = i64::from(h) * i64::from(d);
        if i64::from(mixed.width) != 3 * plane {
            return Err(Refusal::Narrow {
                what: "the post-convolution `[q | k | v]` row, against the two stated head numbers",
                at: i64::from(mixed.width),
            });
        }
        if f.rows != mixed.rows || i64::from(f.width) != plane {
            return Err(Refusal::Narrow {
                what: "the forget projection's row, against the two stated head numbers",
                at: i64::from(f.width),
            });
        }
        if b.rows != mixed.rows || i64::from(b.width) != i64::from(h) {
            return Err(Refusal::Narrow {
                what: "the beta projection's row, against the stated head count",
                at: i64::from(b.width),
            });
        }
        if y.rows != mixed.rows || i64::from(y.width) != plane {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the two stated head numbers",
                at: i64::from(y.width),
            });
        }
        Ok(Self {
            rows: mixed.rows,
            heads: h,
            head_dim: d,
        })
    }
}

#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    fn causal_conv1d<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d at an element other than bf16")?;
        if y.rows != x.rows || y.width != x.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(y.width),
            });
        }
        let taps = taps_of(conv_width)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/causal_conv1d.wgsl", "causal_conv1d_bfloat16")
                .apply(conv_lanes(x.width, x.rows)?),
            &[
                x.arg(),
                weight.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                x.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn causal_conv1d_chunked<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        indptr: In<Payload<i32>>,
        weight: Const<Payload<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d_chunked at an element other than bf16")?;
        if y.rows != x.rows || y.width != x.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(y.width),
            });
        }
        let taps = taps_of(conv_width)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/causal_conv1d.wgsl", "causal_conv1d_chunked_bfloat16")
                .apply(conv_lanes(x.width, requests(indptr)?)?),
            &[
                x.arg(),
                weight.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                indptr.arg(),
                y.arg(),
                x.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn gdn_prep<T: kernels::points::Scalar>(
        &self,
        ba: In<Payload<T>>,
        dt_bias: Const<Payload<T>>,
        a_log: Const<Payload<f32>>,
        gates: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.gdn_prep at an element other than bf16")?;
        if ba.rows <= 0 || ba.width <= 0 {
            return Err(Refusal::Empty {
                what: "the `[b | a]` projection",
            });
        }
        if ba.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(ba.width),
            });
        }
        let v_heads = ba.width / 2;
        if gates.width != ba.width || gates.rows != ba.rows {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(gates.width),
            });
        }
        self.fire(
            Fire::at("ssm/gdn_gates.wgsl", "gdn_ba_gates_bfloat16").apply([
                v_heads.unsigned_abs(),
                ba.rows.unsigned_abs(),
                1,
            ]),
            &[
                ba.arg(),
                a_log.arg(),
                dt_bias.arg(),
                gates.arg(),
                v_heads.arg(),
            ],
        )
    }

    fn gated_delta<T: kernels::points::Scalar>(
        &self,
        qkv: In<Payload<T>>,
        z: In<Payload<T>>,
        gates: In<Payload<f32>>,
        state: Cache<Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta at an element other than bf16")?;
        let shape = Delta::of(qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/gated_delta.wgsl", "gated_delta_bfloat16")
                .apply(scan_lanes(shape.v_heads, shape.rows)?),
            &[
                qkv.arg(),
                gates.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                shape.k_heads.arg(),
                shape.v_heads.arg(),
                shape.k_dim.arg(),
                shape.v_dim.arg(),
            ],
        )
    }

    fn gated_delta_chunked<T: kernels::points::Scalar>(
        &self,
        qkv: In<Payload<T>>,
        indptr: In<Payload<i32>>,
        z: In<Payload<T>>,
        gates: In<Payload<f32>>,
        state: Cache<Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta_chunked at an element other than bf16")?;
        let shape = Delta::of(qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/gated_delta.wgsl", "gated_delta_chunked_bfloat16")
                .apply(scan_lanes(shape.v_heads, requests(indptr)?)?),
            &[
                qkv.arg(),
                gates.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                indptr.arg(),
                shape.k_heads.arg(),
                shape.v_heads.arg(),
                shape.k_dim.arg(),
                shape.v_dim.arg(),
            ],
        )
    }

    fn kda_step<T: kernels::points::Scalar>(
        &self,
        mixed: In<Payload<T>>,
        f: In<Payload<T>>,
        b: In<Payload<T>>,
        dt_bias: Const<Payload<f32>>,
        a_log: Const<Payload<f32>>,
        state: Cache<Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_step at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, y, heads, head_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/kda.wgsl", "kda_step_bfloat16")
                .apply(scan_lanes(shape.heads, shape.rows)?),
            &[
                mixed.arg(),
                f.arg(),
                b.arg(),
                dt_bias.arg(),
                a_log.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                shape.heads.arg(),
                shape.head_dim.arg(),
                norm_eps.arg(),
            ],
        )
    }

    fn kda_chunked<T: kernels::points::Scalar>(
        &self,
        mixed: In<Payload<T>>,
        indptr: In<Payload<i32>>,
        f: In<Payload<T>>,
        b: In<Payload<T>>,
        dt_bias: Const<Payload<f32>>,
        a_log: Const<Payload<f32>>,
        state: Cache<Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_chunked at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, y, heads, head_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/kda.wgsl", "kda_chunked_bfloat16")
                .apply(scan_lanes(shape.heads, requests(indptr)?)?),
            &[
                mixed.arg(),
                f.arg(),
                b.arg(),
                dt_bias.arg(),
                a_log.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                indptr.arg(),
                shape.heads.arg(),
                shape.head_dim.arg(),
                norm_eps.arg(),
            ],
        )
    }
}
