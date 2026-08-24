use kernels::BindMut;
use kernels::Grid;
use kernels::plane::{Cache, Refusal};
use kernels::points::Scalar;
use kernels::raises::Struct;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use crate::points::{self, Handle};
use crate::views::{RecurrentState, RecurrentView};

const CONV_GROUP: u32 = 256;

const SCAN_WIDTH: u32 = 128;

const SCAN_HEAD_MAX: i32 = 256;

fn recurrent<'a>(state: Cache<Struct<RecurrentState>>) -> Result<&'a RecurrentView, Refusal> {
    let row = state.raised();
    if row.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    Ok(unsafe { &*row.ptr })
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

fn rect(rows: i32, width: i32, what: &'static str) -> Result<(u32, u32), Refusal> {
    if rows <= 0 || width <= 0 {
        return Err(Refusal::Empty { what });
    }
    Ok((rows.unsigned_abs(), width.unsigned_abs()))
}

fn conv_grid(channels: u32, rows: u32) -> Grid {
    Grid::of([channels, rows, 1], [channels.min(CONV_GROUP), 1, 1])
}

const fn recurrence_grid(heads: u32, rows: u32) -> Grid {
    Grid::of([SCAN_WIDTH, heads, rows], [SCAN_WIDTH, 1, 1])
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

struct Delta {
    rows: i32,
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}

impl Delta {
    fn of(
        qkv: In<Tensor<bf16>>,
        gates: In<Tensor<f32>>,
        y: Out<Tensor<f32>>,
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
                what: "v_h per k_h",
                at: i64::from(v_h),
            });
        }
        head_width(
            k_d,
            "the key head width, against the shared row this scan stages",
        )?;
        let (_, width) = rect(qkv.rows, qkv.width, "the post-convolution qkv")?;
        let want = 2 * i64::from(k_h) * i64::from(k_d) + i64::from(v_h) * i64::from(v_d);
        if i64::from(width) != want {
            return Err(Refusal::Narrow {
                what: "the post-convolution qkv's row, against the four stated head numbers",
                at: i64::from(width),
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
    fn of(
        mixed: In<Tensor<bf16>>,
        f: In<Tensor<bf16>>,
        b: In<Tensor<bf16>>,
        y: Out<Tensor<f32>>,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Refusal> {
        let h = stated(heads, "the KDA heads this statement states")?;
        let d = stated(head_dim, "the KDA head width this statement states")?;
        head_width(
            d,
            "the KDA head width, against the shared row this scan stages",
        )?;
        let plane = i64::from(h) * i64::from(d);
        let (_, width) = rect(
            mixed.rows,
            mixed.width,
            "the post-convolution `[q | k | v]` row",
        )?;
        if i64::from(width) != 3 * plane {
            return Err(Refusal::Narrow {
                what: "the post-convolution `[q | k | v]` row, against the two stated head numbers",
                at: i64::from(width),
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

fn head_rows(rows: i32, v_heads: i32) -> Result<u32, Refusal> {
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

pub fn core_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

pub fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
}

pub const fn core_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 4, 1]
}

pub const fn simd_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

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
        (4 | 8 | 16 | 32, _) => Err(Refusal::Narrow {
            what: "scan rows per lane group, at this lane width",
            at: i64::from(vrows),
        }),
        _ => Err(Refusal::Narrow {
            what: "scan lane width",
            at: i64::from(lanes),
        }),
    }
}

pub fn scan_grid(v_dim: i32, v_heads: i32, lanes: i32, vrows: i32) -> Result<[u32; 3], Refusal> {
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    if lanes <= 0 || vrows <= 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    let per_y = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    if per_y == 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    Ok([
        32,
        v_dim.unsigned_abs().div_ceil(per_y),
        v_heads.unsigned_abs(),
    ])
}

fn head_lanes(v_heads: i32) -> Result<u32, Refusal> {
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    Ok(v_heads.unsigned_abs().min(256))
}

#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    fn causal_conv1d<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.causal_conv1d`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        let (rows, channels) = rect(x.rows, x.width, "the conv's channel count")?;
        if y.rows != x.rows || y.width != x.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(y.width),
            });
        }
        let taps = stated(conv_width, "the conv width this statement states")?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/causal_conv1d.metal", "causal_conv1d_bfloat16")
                .apply(conv_grid(channels, rows)),
            &[
                x.arg(),
                points::weight::<T, bf16>(weight, WHAT)?.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                x.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn causal_conv1d_chunked<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        weight: Const<Handle<T>>,
        state: Cache<Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.causal_conv1d_chunked`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        let indptr = points::input::<i32, i32>(indptr, "`ssm.causal_conv1d_chunked`'s query CSR")?;
        let (_, channels) = rect(x.rows, x.width, "the conv's channel count")?;
        if y.rows != x.rows || y.width != x.width {
            return Err(Refusal::Narrow {
                what: "the conv's result row, against the row it convolves",
                at: i64::from(y.width),
            });
        }
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        let taps = stated(conv_width, "the conv width this statement states")?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/causal_conv1d.metal", "causal_conv1d_chunked_bfloat16")
                .apply(conv_grid(channels, indptr.rows.unsigned_abs())),
            &[
                x.arg(),
                indptr.arg(),
                points::weight::<T, bf16>(weight, WHAT)?.arg(),
                view.conv_state.arg(),
                view.new_conv_state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                x.width.arg(),
                taps.arg(),
            ],
        )
    }

    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Handle<T>>,
        dt_bias: Const<Handle<T>>,
        a_log: Const<Handle<f32>>,
        gates: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.gdn_prep`, at an element this plane does not stamp";
        let ba = points::input::<T, bf16>(ba, WHAT)?;
        let gates = points::result::<f32, f32>(gates, "`ssm.gdn_prep`'s decay row")?;
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
        let lanes = head_lanes(v_heads)?;
        let rows = u32::try_from(ba.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
        if rows == 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        self.fire(
            Fire::at("ssm/gdn_prep.metal", "qwen_gdn_ba_gates_bfloat16")
                .apply(Grid::of([v_heads.unsigned_abs(), rows, 1], [lanes, 1, 1])),
            &[
                ba.arg(),
                points::weight::<f32, f32>(a_log, "`ssm.gdn_prep`'s decay bank")?.arg(),
                points::weight::<T, bf16>(dt_bias, WHAT)?.arg(),
                gates.arg(),
                v_heads.arg(),
            ],
        )
    }

    fn gated_delta<T: Scalar>(
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
        const WHAT: &str = "`ssm.gated_delta`, at an element this plane does not stamp";
        let qkv = points::input::<T, bf16>(qkv, WHAT)?;
        let gates = points::input::<f32, f32>(gates, "`ssm.gated_delta`'s decay row")?;
        let y = points::result::<f32, f32>(y, "`ssm.gated_delta`'s result")?;
        let shape = Delta::of(qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/gated_delta.metal", "gated_delta_bfloat16").apply(recurrence_grid(
                shape.v_heads.unsigned_abs(),
                shape.rows.unsigned_abs(),
            )),
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

    fn gated_delta_chunked<T: Scalar>(
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
        const WHAT: &str = "`ssm.gated_delta_chunked`, at an element this plane does not stamp";
        let qkv = points::input::<T, bf16>(qkv, WHAT)?;
        let gates = points::input::<f32, f32>(gates, "`ssm.gated_delta_chunked`'s decay row")?;
        let y = points::result::<f32, f32>(y, "`ssm.gated_delta_chunked`'s result")?;
        let indptr = points::input::<i32, i32>(indptr, "`ssm.gated_delta_chunked`'s query CSR")?;
        let shape = Delta::of(qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/gated_delta.metal", "gated_delta_chunked_bfloat16").apply(
                recurrence_grid(shape.v_heads.unsigned_abs(), indptr.rows.unsigned_abs()),
            ),
            &[
                qkv.arg(),
                indptr.arg(),
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

    fn kda_step<T: Scalar>(
        &self,
        mixed: In<Handle<T>>,
        f: In<Handle<T>>,
        b: In<Handle<T>>,
        dt_bias: Const<Handle<f32>>,
        a_log: Const<Handle<f32>>,
        state: Cache<Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.kda_step`, at an element this plane does not stamp";
        let mixed = points::input::<T, bf16>(mixed, WHAT)?;
        let f = points::input::<T, bf16>(f, WHAT)?;
        let b = points::input::<T, bf16>(b, WHAT)?;
        let y = points::result::<f32, f32>(y, "`ssm.kda_step`'s result")?;
        let shape = Kda::of(mixed, f, b, y, heads, head_dim)?;
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/kda.metal", "kda_step_bfloat16").apply(recurrence_grid(
                shape.heads.unsigned_abs(),
                shape.rows.unsigned_abs(),
            )),
            &[
                mixed.arg(),
                f.arg(),
                b.arg(),
                points::weight::<f32, f32>(dt_bias, "`ssm.kda_step`'s decay bias")?.arg(),
                points::weight::<f32, f32>(a_log, "`ssm.kda_step`'s decay bank")?.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                shape.heads.arg(),
                shape.head_dim.arg(),
                norm_eps.arg(),
            ],
        )
    }

    fn kda_chunked<T: Scalar>(
        &self,
        mixed: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        f: In<Handle<T>>,
        b: In<Handle<T>>,
        dt_bias: Const<Handle<f32>>,
        a_log: Const<Handle<f32>>,
        state: Cache<Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.kda_chunked`, at an element this plane does not stamp";
        let mixed = points::input::<T, bf16>(mixed, WHAT)?;
        let f = points::input::<T, bf16>(f, WHAT)?;
        let b = points::input::<T, bf16>(b, WHAT)?;
        let y = points::result::<f32, f32>(y, "`ssm.kda_chunked`'s result")?;
        let indptr = points::input::<i32, i32>(indptr, "`ssm.kda_chunked`'s query CSR")?;
        let shape = Kda::of(mixed, f, b, y, heads, head_dim)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        let view = recurrent(state)?;
        self.fire(
            Fire::at("ssm/kda.metal", "kda_chunked_bfloat16").apply(recurrence_grid(
                shape.heads.unsigned_abs(),
                indptr.rows.unsigned_abs(),
            )),
            &[
                mixed.arg(),
                indptr.arg(),
                f.arg(),
                b.arg(),
                points::weight::<f32, f32>(dt_bias, "`ssm.kda_chunked`'s decay bias")?.arg(),
                points::weight::<f32, f32>(a_log, "`ssm.kda_chunked`'s decay bank")?.arg(),
                view.state.arg_mut(),
                view.slots.arg(),
                y.arg(),
                shape.heads.arg(),
                shape.head_dim.arg(),
                norm_eps.arg(),
            ],
        )
    }
}
