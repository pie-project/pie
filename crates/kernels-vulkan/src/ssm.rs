use kernels::BindMut;
use kernels::plane::{Cache, Refusal};
use kernels::raises::Struct;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    rows: i32,
    width: i32,
}

use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use crate::points::{Handle, at_bf16, stated};
use crate::views::{RecurrentState, RecurrentView};

pub fn scan_point(lanes: i32, vrows: i32) -> Result<usize, Refusal> {
    match (lanes, vrows) {
        (16, 1) => Ok(0),
        (16, 2) => Ok(1),
        (16, 4) => Ok(2),
        (32, 2) => Ok(3),
        (32, 4) => Ok(4),
        (32, 8) => Ok(5),
        (4, 1) => Ok(6),
        (8, 1) => Ok(7),
        (8, 2) => Ok(8),
        _ => Err(Refusal::Narrow {
            what: "no gdn scan is compiled for this (LANES, VROWS)",
            at: i64::from(lanes) * 100 + i64::from(vrows),
        }),
    }
}

pub fn head_rows(rows: i32, v_heads: i32) -> Result<u32, Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    let n = u64::from(rows.unsigned_abs()) * u64::from(v_heads.unsigned_abs());
    u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "rows * v_heads",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })
}

pub fn gdn_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

pub fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
}

pub fn scan_grid(v_dim: i32, v_heads: i32, lanes: i32, vrows: i32) -> Result<[u32; 3], Refusal> {
    scan_point(lanes, vrows)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    Ok([
        32,
        v_dim.unsigned_abs().div_ceil(per_group),
        v_heads.unsigned_abs(),
    ])
}

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

fn requests(indptr: In<Handle<i32>>) -> Result<u32, Refusal> {
    if indptr.rows <= 0 {
        return Err(Refusal::Empty {
            what: "the query CSR this statement names",
        });
    }
    Ok(indptr.rows.unsigned_abs())
}

fn taps_of(conv_width: u32) -> Result<i32, Refusal> {
    let taps = stated("the conv width this statement states", conv_width)?;
    if taps <= 0 {
        return Err(Refusal::Empty {
            what: "the conv width this statement states",
        });
    }
    Ok(taps)
}

fn per_lane_row(lanes: i32, rows: u32) -> Result<[u32; 3], Refusal> {
    if lanes <= 0 {
        return Err(Refusal::Empty {
            what: "the lanes this map covers",
        });
    }
    if rows == 0 {
        return Err(Refusal::Empty {
            what: "the rows this map covers",
        });
    }
    Ok([lanes.unsigned_abs(), rows, 1])
}

fn per_head_row(v_heads: i32, rows: u32) -> Result<[u32; 3], Refusal> {
    if v_heads <= 0 {
        return Err(Refusal::Empty {
            what: "the value heads this scan covers",
        });
    }
    if rows == 0 {
        return Err(Refusal::Empty {
            what: "the rows this scan covers",
        });
    }
    Ok([SCAN_WIDTH, v_heads.unsigned_abs(), rows])
}

struct Delta {
    rows: i32,
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}

impl Delta {
    fn of(
        qkv: Extent,
        gates: Extent,
        y: Extent,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Refusal> {
        let k_h = stated("the key heads this statement states", k_heads)?;
        let v_h = stated("the value heads this statement states", v_heads)?;
        let k_d = stated("the key head width this statement states", k_dim)?;
        let v_d = stated("the value head width this statement states", v_dim)?;
        if k_h <= 0 || v_h <= 0 || k_d <= 0 || v_d <= 0 {
            return Err(Refusal::Empty {
                what: "the four head numbers this statement states",
            });
        }
        if v_h % k_h != 0 {
            return Err(Refusal::Narrow {
                what: "the value heads this statement states, per key head",
                at: i64::from(v_h),
            });
        }
        if k_d > SCAN_HEAD_MAX {
            return Err(Refusal::Wide {
                what: "the key head width, against the shared row this scan stages",
                at: i64::from(k_d),
                max: i64::from(SCAN_HEAD_MAX),
            });
        }
        if qkv.rows <= 0 {
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

#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    fn causal_conv1d<T: kernels::points::Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d, at an element this plane does not instantiate")?;
        let row = x.all("the convolved row's channel count")?;
        let out = y.all("the conv's result row")?;
        if out.rows != row.rows || out.width != row.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(out.width),
            });
        }
        let taps = taps_of(conv_width)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("causal_conv1d_bfloat16", self.best()),
                "causal_conv1d_bfloat16",
            )
            .apply(per_lane_row(row.width, row.rows.unsigned_abs())?),
            &[
                x.arg(),
                weight.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                row.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn causal_conv1d_chunked<T: kernels::points::Scalar>(
        &self,
        x: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        weight: Const<Handle<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d_chunked, at an element this plane does not instantiate")?;
        let row = x.all("the convolved row's channel count")?;
        let out = y.all("the conv's result row")?;
        if out.rows != row.rows || out.width != row.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(out.width),
            });
        }
        let taps = taps_of(conv_width)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("causal_conv1d_chunked_bfloat16", self.best()),
                "causal_conv1d_chunked_bfloat16",
            )
            .apply(per_lane_row(row.width, requests(indptr)?)?),
            &[
                x.arg(),
                weight.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                indptr.arg(),
                y.arg(),
                row.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn gdn_prep<T: kernels::points::Scalar>(
        &self,
        ba: In<Handle<T>>,
        dt_bias: Const<Handle<T>>,
        a_log: Const<Handle<f32>>,
        gates: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.gdn_prep, at an element this plane does not instantiate")?;
        let row = ba.all("the `[b | a]` projection's row")?;
        let out = gates.all("the fused `[g_log | beta]` row")?;
        if row.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the `[b | a]` projection",
            });
        }
        if row.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(row.width),
            });
        }
        let v_heads = row.width / 2;
        if out.rows != row.rows || out.width != row.width {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(out.width),
            });
        }
        self.fire(
            Fire::at(
                crate::plane::module_path("gdn_ba_gates_bfloat16", self.best()),
                "gdn_ba_gates_bfloat16",
            )
            .apply(per_lane_row(v_heads, row.rows.unsigned_abs())?),
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
        qkv: In<Handle<T>>,
        z: In<Handle<T>>,
        gates: In<Handle<f32>>,
        state: Cache<Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta, at an element this plane does not instantiate")?;
        let row = qkv.all("the post-convolution qkv row")?;
        let decay = gates.all("the fused `[g_log | beta]` row")?;
        let out = y.all("the recurrence's result row")?;
        let shape = Delta::of(
            Extent {
                rows: row.rows,
                width: row.width,
            },
            Extent {
                rows: decay.rows,
                width: decay.width,
            },
            Extent {
                rows: out.rows,
                width: out.width,
            },
            k_heads,
            v_heads,
            k_dim,
            v_dim,
        )?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("gated_delta_bfloat16", self.best()),
                "gated_delta_bfloat16",
            )
            .apply(per_head_row(shape.v_heads, shape.rows.unsigned_abs())?),
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
        qkv: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        z: In<Handle<T>>,
        gates: In<Handle<f32>>,
        state: Cache<Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        let _ = z;
        at_bf16::<T>("ssm.gated_delta_chunked, at an element this plane does not instantiate")?;
        let row = qkv.all("the post-convolution qkv row")?;
        let decay = gates.all("the fused `[g_log | beta]` row")?;
        let out = y.all("the recurrence's result row")?;
        let shape = Delta::of(
            Extent {
                rows: row.rows,
                width: row.width,
            },
            Extent {
                rows: decay.rows,
                width: decay.width,
            },
            Extent {
                rows: out.rows,
                width: out.width,
            },
            k_heads,
            v_heads,
            k_dim,
            v_dim,
        )?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("gated_delta_chunked_bfloat16", self.best()),
                "gated_delta_chunked_bfloat16",
            )
            .apply(per_head_row(shape.v_heads, requests(indptr)?)?),
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
}
