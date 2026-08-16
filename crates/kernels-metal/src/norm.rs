//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine, elementwise, elementwise_rows};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    ("norm/add_bias.metal", "add_bias_bfloat16"),
    ("norm/gated_rms.metal", "gated_rms_bfloat16"),
    ("norm/gated_rms.metal", "gated_rms_strided_bfloat16"),
    ("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16"),
    ("norm/residual_add.metal", "residual_add_bfloat16"),
    ("norm/residual_add.metal", "residual_add_strided_bfloat16"),
    ("norm/rms.metal", "rms_residual_bfloat16"),
    ("norm/rms.metal", "rms_residual_scaled_bfloat16"),
    ("norm/rms.metal", "rms_single_row_bfloat16"),
    ("norm/rms.metal", "rms_strided_head_row_bfloat16"),
    ("norm/rms.metal", "rms_strided_row_bfloat16"),
    ("norm/vector.metal", "vnorm_single_row_bfloat16"),
];

/// Threads per threadgroup for the two elementwise bodies here, for the
/// reason `mlp.rs` states it: Metal declares no group size in the source, so
/// the number lives on this side and reaches the encoder as the second half
/// of `dispatchThreads:threadsPerThreadgroup:`.
const GROUP_X: u32 = 256;

/// The reduction's threadgroup: one lane per `N_READS = 4` channels of the
/// axis, capped at the 1024 a Metal threadgroup may be.
///
/// Rounded UP. The kernel guards its own tail, but a truncating count drops
/// the last partial group of four -- and those channels are not left stale,
/// they are left out of the SUM, so every channel of the row is normalized by
/// a divisor that is too small.
fn rms_threads(axis: i32) -> Result<u32, Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    Ok(axis.unsigned_abs().div_ceil(4).min(1024))
}

/// `LaunchRule::Rms`: one threadgroup per AXIS, not per row.
///
/// The distinction is the whole content of this helper. A hidden-state norm
/// has `axis == width` and this reduces to one threadgroup per row. A QK-norm
/// does not: a per-head norm packs `width / axis` of them into each row, and
/// each is its own reduction. Handing such a fire one threadgroup per row
/// normalizes head 0 and leaves every other head as the projection wrote it
/// -- fully written, not fully normalized, and no read of the output reports
/// it.
fn rms_grid(width: i32, axis: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if axis > width {
        return Err(Refusal::Wide {
            what: "axis",
            at: i64::from(axis),
            max: i64::from(width),
        });
    }
    let t = rms_threads(axis)?;
    let norms = width.unsigned_abs() / axis.unsigned_abs();
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows.unsigned_abs()))
        .ok_or(Refusal::Grid {
            what: "axis threads * norms per row * rows",
            at: i64::from(t) * i64::from(norms) * i64::from(rows),
        })?;
    Ok(([lanes, 1, 1], [t, 1, 1]))
}

/// One threadgroup per `(head, row)` pair, for the two-level bases.
///
/// A token holds `heads` norms packed an axis apart and the next token is a
/// uniform `row_pitch` away, so the base is two-level and one grid axis cannot
/// carry both terms -- flattening would need the two strides commensurate, and
/// the pitch is the whole token while the axis is one head of it. The launch
/// gives it two: y is the head and z is the row, which is what makes the
/// threadgroup's own position the pair.
fn head_row_grid(threads: u32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([threads, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// `out = w * (x / rms(x))`, one threadgroup per axis.
///
/// The order of `x, w, out, params` is the SHADER's and not the trace's. A
/// trace states inputs, outputs, then weights, so binding positionally puts
/// the output where the norm weight belongs -- and nothing reports it, because
/// Metal does not validate a binding. That mismatch is why this was the first
/// row in the tree to state its operands, and here it is the argument order.
///
/// # Errors
///
/// See [`rms_grid`].
pub fn rms_single_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(*width, *axis, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "rms_single_row_bfloat16",
            file: RMS_FILE,
            lanes,
            group,
        },
        &[x.v(), w.v(), out.v(), params.v()],
    )
}

/// [`rms_single_row`] over rows that are a `row_pitch` apart rather than an
/// axis apart.
///
/// One threadgroup per ROW here and not per axis: the base is `gid *
/// row_pitch`, so a row holds exactly one norm and the axis only sizes the
/// threadgroup. That is the difference from [`rms_single_row`], whose single
/// grid axis carries both the norm and the row.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty axis or row count, and [`Refusal::Grid`]
/// when the lane total does not fit a `u32`.
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let t = rms_threads(*axis)?;
    if *rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let lanes = t.checked_mul(rows.unsigned_abs()).ok_or(Refusal::Grid {
        what: "axis threads * rows",
        at: i64::from(t) * i64::from(*rows),
    })?;
    ctx.dispatch(
        Fire {
            entrypoint: "rms_strided_row_bfloat16",
            file: RMS_FILE,
            lanes: [lanes, 1, 1],
            group: [t, 1, 1],
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// The per-head q/k norms over a whole prompt: `heads` norms inside each row,
/// `rows` of them.
///
/// # Errors
///
/// See [`head_row_grid`].
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    axis: Env<i32>,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let t = rms_threads(*axis)?;
    ctx.dispatch(
        Fire {
            entrypoint: "rms_strided_head_row_bfloat16",
            file: RMS_FILE,
            lanes: head_row_grid(t, *heads, *rows)?,
            group: [t, 1, 1],
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// The residual is buffer 4 and arrives AFTER the params struct, which is the
/// order the fold adds it in: the conditional binding comes after the
/// unconditional ones, so folding does not renumber the four every form
/// shares.
///
/// # Errors
///
/// See [`rms_grid`].
pub fn rms_residual(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    r: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(*width, *axis, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "rms_residual_bfloat16",
            file: RMS_FILE,
            lanes,
            group,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v()],
    )
}

/// [`rms_residual`] with a per-layer gain beside the residual.
///
/// # Errors
///
/// See [`rms_grid`].
pub fn rms_residual_scaled(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    r: Buf,
    s: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(*width, *axis, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "rms_residual_scaled_bfloat16",
            file: RMS_FILE,
            lanes,
            group,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v(), s.v()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm, and the absent weight is the whole difference from
/// [`rms_single_row`] -- which is why `out` is buffer 1 here and 2 there.
///
/// # Errors
///
/// See [`rms_grid`].
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(*width, *axis, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "vnorm_single_row_bfloat16",
            file: "norm/vector.metal",
            lanes,
            group,
        },
        &[x.v(), out.v(), params.v()],
    )
}

/// GDN's gated norm: `w * rms(x) * silu(z)`, over each value head.
///
/// The row axis is stated here and `LaunchRule::GatedRms` does not state it.
/// That is a DEFECT this port found, not a difference of spelling: the body
/// indexes `(tgpos.z * tpg.y + tgpos.y) * vd + lid`, so with `grid.z = 1` only
/// row 0 is normalized and every other token keeps whatever the core scan
/// wrote. A decode is one row, which is why the golden `gdn_core` tag has been
/// green over this the whole time; a PREFILL through this symbol would norm
/// its first token and no other. `LaunchRule::RouterLane` records the same
/// finding about the same missing axis, already fixed.
///
/// # Errors
///
/// See [`head_row_grid`], plus [`Refusal::Empty`] for an empty head width.
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    vd: Env<i32>,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let t = head_width(*vd)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gated_rms_bfloat16",
            file: GATED_FILE,
            lanes: head_row_grid(t, *heads, *rows)?,
            group: [t, 1, 1],
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v()],
    )
}

/// [`gated_rms`] over rows a `row_pitch` apart.
///
/// The strided form needs no `threadgroups_per_grid`: its base is `tgpos.z *
/// row_pitch + tgpos.y * vd`, both terms explicit. That is what makes it the
/// prefill form -- the packed one infers the row stride from the grid's own
/// shape, which only holds when the heads are contiguous.
///
/// # Errors
///
/// See [`gated_rms`].
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    vd: Env<i32>,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let t = head_width(*vd)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gated_rms_strided_bfloat16",
            file: GATED_FILE,
            lanes: head_row_grid(t, *heads, *rows)?,
            group: [t, 1, 1],
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// One lane per channel of a value head, which is also the threadgroup: the
/// gated norm reduces across the whole head in one group and has no outer
/// loop, so the head dim IS the group width and 1024 is a hard ceiling.
fn head_width(vd: i32) -> Result<u32, Refusal> {
    if vd <= 0 {
        return Err(Refusal::Empty { what: "vd" });
    }
    if vd > 1024 {
        return Err(Refusal::Wide {
            what: "vd",
            at: i64::from(vd),
            max: 1024,
        });
    }
    Ok(vd.unsigned_abs())
}

/// gemma's per-layer scale: `out = x * scalar[0]`, elementwise.
///
/// `params` is bound and not read. `LayerScalarParams` holds a hidden width
/// the body bounds itself with the grid instead, and the buffer stays because
/// the entrypoint declares it -- the struct's own header says so at length.
///
/// # Errors
///
/// See [`elementwise`].
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: Buf,
    scalar: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "layer_scalar_mul_bfloat16",
            file: "norm/layer_scalar.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[x.v(), scalar.v(), out.v(), params.v()],
    )
}

/// `out = x + residual`, elementwise, and `out` may alias `x`.
///
/// Stated because a MIXTURE demands it: a routed FFN's rows are already
/// down-projected and combined, so all the block owes is the add, where a
/// dense FFN fuses it into the down projection and never names this symbol.
///
/// # Errors
///
/// See [`elementwise`].
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: Buf,
    residual: Buf,
    out: BufMut,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "residual_add_bfloat16",
            file: RESIDUAL_FILE,
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[x.v(), residual.v(), out.v()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart, so a whole prompt is one
/// dispatch instead of one per token.
///
/// The rows go on their own grid axis rather than being folded into a flat
/// count, because a pitch is not a width: the row's own extent is what the
/// threads cover and the pitch is only how far the next one starts.
///
/// # Errors
///
/// See [`elementwise_rows`].
pub fn residual_add_strided(
    ctx: &Ctx<'_>,
    x: Buf,
    residual: Buf,
    out: BufMut,
    row_pitch: i32,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "residual_add_strided_bfloat16",
            file: RESIDUAL_FILE,
            lanes: elementwise_rows(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[x.v(), residual.v(), out.v(), row_pitch.v()],
    )
}

/// The Qwen-2 family's attention biases, added in place.
///
/// The width is an ARGUMENT and not an `Env`: the kernel is told it, because
/// `tid.y * width + tid.x` needs the row stride and the grid only carries the
/// extent. Its threadgroup is the row clamped to 256 rather than a flat 256,
/// so a row narrower than that does not launch threads past its own end and
/// rely on the grid to retire them.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty width or row count.
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: BufMut,
    bias: Buf,
    width: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = elementwise_rows(width, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "add_bias_bfloat16",
            file: "norm/add_bias.metal",
            lanes,
            group: [lanes[0].min(GROUP_X), 1, 1],
        },
        &[out.v(), bias.v(), width.v()],
    )
}

const RMS_FILE: &str = "norm/rms.metal";
const GATED_FILE: &str = "norm/gated_rms.metal";
const RESIDUAL_FILE: &str = "norm/residual_add.metal";

/// The family, in the order the rows above state it.
pub static ROUTINES: &[Routine] = &[
    // The one in-place pair in the family, and a REAL one unlike a rotation's:
    // the trace's `AddBias` states an input and an output, and the kernel
    // binds only the output because they are the same bytes. A rotation binds
    // one `BufMut` too, but its statement has no input to alias -- which is
    // why `rope` states nothing here and this does.
    crate::routine!(add_bias, in_place = &[(0, 0)]),
    crate::routine!(gated_rms),
    crate::routine!(gated_rms_strided),
    crate::routine!(layer_scalar_mul),
    crate::routine!(residual_add),
    crate::routine!(residual_add_strided),
    crate::routine!(rms_residual),
    crate::routine!(rms_residual_scaled),
    crate::routine!(rms_single_row),
    crate::routine!(rms_strided_head_row),
    crate::routine!(rms_strided_row),
    crate::routine!(vnorm_single_row),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The only call the recorder made.
    fn one(seen: &Seen) -> Call {
        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 1, "one norm is one dispatch");
        calls[0].clone()
    }

    /// The shader declares `x, w, out, params` and a trace states inputs,
    /// outputs, then weights. Binding in the trace's order puts the OUTPUT
    /// where the norm weight belongs, which is the defect that made this
    /// family the first to state its operands -- and Metal does not validate a
    /// binding, so the only thing that reports it is a test like this one.
    #[test]
    fn the_row_norm_binds_the_weight_before_its_output() {
        let seen = Seen::default();
        rms_single_row(
            &seen,
            Buf(1),
            Buf(2),
            BufMut(3),
            Buf(4),
            Env(8),
            Env(8),
            Env(1),
        )
        .expect("a launch");
        assert_eq!(
            one(&seen).1,
            [
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::Buffer(3),
                ArgValue::Buffer(4)
            ],
            "x, w, out, params -- the shader's order and not the trace's"
        );
    }

    /// A per-head norm packs `width / axis` reductions into each row and each
    /// is its own threadgroup.
    ///
    /// One threadgroup per ROW instead normalizes head 0 and leaves the other
    /// heads as the projection wrote them. That is not a smaller
    /// normalization: the output is fully written and only partly normalized,
    /// so nothing downstream can tell. gemma-4 is the model that has it --
    /// 32 heads of 256 channels in an 8192-wide Q -- and llama has no QK-norm,
    /// which is why the golden gate was green over it.
    #[test]
    fn a_per_head_norm_gets_a_threadgroup_per_head_and_not_per_row() {
        let (lanes, group) = rms_grid(8192, 256, 2).expect("a grid");
        assert_eq!(group, [64, 1, 1], "256 channels over N_READS = 4");
        assert_eq!(
            lanes,
            [64 * 32 * 2, 1, 1],
            "32 heads a row, 2 rows, each its own threadgroup"
        );

        let (whole, _) = rms_grid(8192, 8192, 2).expect("a grid");
        assert_eq!(
            whole,
            [1024 * 2, 1, 1],
            "a hidden-state norm spans its row, so this is one group a row \
             -- capped at the 1024 threads a threadgroup may be"
        );
    }

    /// `LaunchRule::GatedRms` is `(vd, heads, 1)` and the body indexes
    /// `(tgpos.z * tpg.y + tgpos.y) * vd`, so under that rule a prefill norms
    /// its first token and leaves every other one as the core scan wrote it.
    ///
    /// A decode is one row, which is why `gdn_core` has been green over this.
    /// The routine states the axis. `LaunchRule::RouterLane`'s own doc records
    /// the identical finding about the identical missing axis.
    #[test]
    fn the_gated_norm_puts_its_rows_on_the_axis_the_body_reads() {
        let seen = Seen::default();
        gated_rms(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            BufMut(4),
            Buf(5),
            Env(128),
            Env(16),
            Env(7),
        )
        .expect("a launch");
        let (fire, _) = one(&seen);
        assert_eq!(
            fire.lanes,
            [128, 16, 7],
            "the row is grid.z, and it is 7 and not 1"
        );
        assert_eq!(fire.group, [128, 1, 1], "the head dim is the threadgroup");
    }

    /// The strided forms take their pitch LAST, after every buffer the packed
    /// form binds, so folding a stride in does not renumber what they share.
    #[test]
    fn a_stride_lands_after_the_bindings_it_does_not_change() {
        let seen = Seen::default();
        rms_strided_row(
            &seen,
            Buf(1),
            Buf(2),
            BufMut(3),
            Buf(4),
            4096,
            Env(8),
            Env(2),
        )
        .expect("a launch");
        residual_add_strided(&seen, Buf(1), Buf(2), BufMut(3), 4096, Env(8), Env(2))
            .expect("a launch");
        let calls = seen.0.borrow();
        assert_eq!(
            calls[0].1.last(),
            Some(&ArgValue::I32(4096)),
            "rms_strided_row takes the pitch at 4"
        );
        assert_eq!(
            calls[1].1.last(),
            Some(&ArgValue::I32(4096)),
            "residual_add_strided takes it at 3"
        );
        assert_eq!(
            calls[0].0.lanes,
            [2 * 2, 1, 1],
            "one threadgroup a row here, because the pitch carries the row"
        );
    }

    /// An axis wider than the row it sits in is a mistake in the statement,
    /// not a shape to clamp. Clamping is what the row path does, and it turns
    /// `width / axis` into 1 -- the per-head norm above, launched as a whole
    /// row norm, silently.
    #[test]
    fn an_axis_wider_than_its_row_is_refused_rather_than_clamped() {
        assert!(matches!(
            rms_grid(256, 8192, 1),
            Err(Refusal::Wide { what: "axis", .. })
        ));
        assert!(matches!(
            rms_grid(256, 0, 1),
            Err(Refusal::Empty { what: "axis" })
        ));
        assert!(matches!(
            rms_grid(256, 256, 0),
            Err(Refusal::Empty { what: "rows" })
        ));
    }

    /// The bias is added IN PLACE, so the buffer it reads is the one it
    /// writes and there are only three arguments.
    #[test]
    fn the_bias_binds_the_value_it_biases_once() {
        let seen = Seen::default();
        add_bias(&seen, BufMut(1), Buf(2), 96, Env(4)).expect("a launch");
        let (fire, args) = one(&seen);
        assert_eq!(
            args,
            [ArgValue::Buffer(1), ArgValue::Buffer(2), ArgValue::I32(96)],
            "out, bias, width -- and `out` is the input too"
        );
        assert_eq!(fire.lanes, [96, 4, 1]);
        assert_eq!(
            fire.group,
            [96, 1, 1],
            "the row, clamped to 256 -- not a flat 256 over a 96-wide row"
        );
        add_bias(&seen, BufMut(1), Buf(2), 4096, Env(4)).expect("a launch");
        assert_eq!(seen.0.borrow()[1].0.group, [256, 1, 1], "and clamped");
    }
}
