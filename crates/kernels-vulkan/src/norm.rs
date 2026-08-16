//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.
#![allow(clippy::too_many_arguments)]

use kernels::KernelSig;
use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "gated_rms_bfloat16",
    "gated_rms_strided_bfloat16",
    "layer_scalar_mul_bfloat16",
    "add_bias_bfloat16",
    "residual_add_bfloat16",
    "residual_add_strided_bfloat16",
    "rms_residual_bfloat16",
    "rms_residual_scaled_bfloat16",
    "rms_single_row_bfloat16",
    "rms_strided_head_row_bfloat16",
    "rms_strided_row_bfloat16",
    "vnorm_single_row_bfloat16",
];

pub static KERNELS: &[KernelSig] = &[];

/// The workgroup width every reducing kernel in this file is compiled at.
///
/// `PIE_GROUP_X` in `rms.slang` and `gated_rms.slang`, `PIE_GROUP` in
/// `vector.slang`, and 256 in all three. It appears here because these grids
/// are stated in THREADS and a reduction wants one whole workgroup per row:
/// the lane count for `rows` rows is `rows * 256`, and `driver-vulkan`'s
/// `div_ceil` turns it back into `rows` workgroups. A row is reduced by the
/// workgroup that owns it, so this is the shader's number and not a tuning
/// choice -- `256 * N_READS` is also the span the row-walking loop strides by.
const GROUP: u32 = 256;

/// One whole workgroup per row, which is what a row reduction needs.
///
/// For the STRIDED forms only: their base is `group.x * row_pitch`, so a row
/// holds exactly one norm however wide the axis is.
///
/// # Errors
///
/// [`Refusal::Empty`] for no rows, and [`Refusal::Grid`] for a row count whose
/// lane total does not fit a `u32`. The second is not hypothetical arithmetic:
/// multiplying by 256 costs eight bits of headroom, so a row count that fits
/// comfortably can produce a lane count that does not, and a wrapped grid
/// launches a FRACTION of the rows and returns success.
fn per_row(rows: i32) -> Result<[u32; 3], Refusal> {
    per_axis(1, 1, rows)
}

/// One whole workgroup per AXIS, which is not the same as per row.
///
/// `rms.slang`'s packed base is `group.x * p.axis_size`, so the grid counts
/// AXES and a row holds `width / axis` of them. Handing such a fire one
/// workgroup per row normalizes the first axis of each row and leaves the rest
/// as the projection wrote them -- output fully written and only partly
/// normalized, which no read of it can report.
///
/// A hidden-state norm has `axis == width` and this reduces to one workgroup
/// per row, which is what it said before. A per-head q/k norm does not:
/// gemma-4 normalizes each of 32 heads of an 8192-wide Q over 256 channels,
/// and got one workgroup for the 32. Metal's crossing of this family is where
/// that was found; the two backends' shaders index the same way, so it was
/// true here too.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent, [`Refusal::Wide`] for an axis wider
/// than the row it sits in, and [`Refusal::Grid`] when the lane total does not
/// fit a `u32`.
fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
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
    let axes = (width.unsigned_abs() / axis.unsigned_abs()) * rows.unsigned_abs();
    let lanes = axes.checked_mul(GROUP).ok_or(Refusal::Grid {
        what: "axes * the workgroup width",
        at: i64::from(axes) * i64::from(GROUP),
    })?;
    Ok([lanes, 1, 1])
}

/// One workgroup per `(head, row)` pair, for the two-level forms.
///
/// A token holds `heads` per-head norms packed `axis_size` apart and the next
/// token is a uniform `row_pitch` away, so the base is two-level and one grid
/// axis cannot carry both terms. The launch gives it two -- y is the head and
/// z is the row -- exactly as Metal's does. x is the workgroup, and it is 256
/// lanes rather than `heads * 256` because the head is its own axis here.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty head or row count.
fn per_head_row(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([GROUP, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// `out = w * (x / rms(x))`, one workgroup per row.
///
/// The order of `x, w, out, params` is the SHADER's and not the trace's. A
/// trace states inputs, outputs, then weights, so binding positionally puts
/// the output where the norm weight belongs -- and nothing reports it, because
/// a descriptor write is typed by the layout and every one of these is a
/// storage buffer. That mismatch is the reason this was the first row in the
/// tree to state its operands, and here it is the argument order.
///
/// # Errors
///
/// See [`per_axis`].
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
    ctx.dispatch(
        Fire {
            entrypoint: "rms_single_row_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v()],
    )
}

/// [`rms_single_row`] over rows that are a `row_pitch` apart rather than an
/// axis apart.
///
/// The pitch is a PUSH constant and the axis is a field of the params struct,
/// which is this backend's rule rather than an accident: a scalar rides the
/// push block and a struct stays a buffer. `rms.slang`'s header is where both
/// halves of that rule are written down.
///
/// # Errors
///
/// See [`per_row`].
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "rms_strided_row_bfloat16",
            lanes: per_row(*rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// The per-head q/k norms: `heads` norms inside each row, `rows` of them.
///
/// Two grid axes and not one. The head form cannot be [`rms_strided_row`] with
/// a different pitch, because the base is `row * row_pitch + head *
/// axis_size` and a single axis cannot carry both terms -- flattening it would
/// need the two strides to be commensurate, and `row_pitch` is the whole
/// token while `axis_size` is one head of it.
///
/// # Errors
///
/// See [`per_head_row`].
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "rms_strided_head_row_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// The residual is buffer 4 and arrives AFTER the params struct, which is the
/// order `#if defined(PIE_RESIDUAL)` adds it in -- the conditional bindings
/// come after the unconditional ones, so a fold does not renumber the four
/// every form shares.
///
/// # Errors
///
/// See [`per_axis`].
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
    ctx.dispatch(
        Fire {
            entrypoint: "rms_residual_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v()],
    )
}

/// [`rms_residual`] with a per-layer gain applied AFTER the add.
///
/// The two are separate entrypoints and not one with a gain of one, because
/// the order is the point: the gain multiplies the sum and not the normalised
/// value, and a fused form that scaled before adding would be a different
/// number.
///
/// # Errors
///
/// See [`per_axis`].
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
    ctx.dispatch(
        Fire {
            entrypoint: "rms_residual_scaled_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v(), s.v()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm. The absence of a weight is the whole difference from
/// [`rms_single_row`], and it is why `out` is buffer 1 here and buffer 2
/// there -- so an argument list copied between them binds the output where
/// the other reads its input.
///
/// # Errors
///
/// See [`per_axis`].
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "vnorm_single_row_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), out.v(), params.v()],
    )
}

/// `out = w * rmsnorm(x) * silu(z)`, per head.
///
/// The gate is buffer 1 and the weight is buffer 2, which is not the order the
/// name suggests: `gated_rms` reads as a norm that happens to be gated, and
/// the shader binds the gate before the gain.
///
/// The grid is `[256, heads, rows]`, and the non-strided form builds its base
/// from `group.z * gl_NumWorkGroups.y + group.y` -- it reads the HEAD COUNT
/// back out of the grid it was launched on. So the y extent is not a
/// scheduling choice here; a y that did not equal `heads` would not run fewer
/// heads, it would address the wrong rows.
///
/// # Errors
///
/// See [`per_head_row`].
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gated_rms_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v()],
    )
}

/// [`gated_rms`] over rows a `row_pitch` apart.
///
/// The strided form takes its row base from the pitch rather than from the
/// grid's own y extent, which is the one place the two differ.
///
/// # Errors
///
/// See [`per_head_row`].
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gated_rms_strided_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// gemma's per-layer scale: one number per layer, read from a buffer.
///
/// Read rather than stated because WHICH layer is running is the fire's, not
/// the statement's -- a scalar operand would have to be re-stated per layer
/// and a buffer is bound per layer for free.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: Buf,
    scalar: Buf,
    out: BufMut,
    _params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "layer_scalar_mul_bfloat16",
            lanes: crate::routine::elementwise(*width, *rows)?,
        },
        &[x.v(), scalar.v(), out.v()],
    )
}

/// `out = x + residual`, elementwise, and `out` may alias `x`.
///
/// Filled because a MIXTURE demands it: a routed FFN's rows are already
/// down-projected and combined, so all the block owes is the add, where a
/// dense FFN fuses the add into its down projection and never fires this.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
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
            lanes: crate::routine::elementwise(*width, *rows)?,
        },
        &[x.v(), residual.v(), out.v()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart.
///
/// A RECTANGLE and not a flat run, which is the whole difference: the plain
/// form indexes `gid.x` straight into the buffer and the strided one indexes
/// `gid.y * row_pitch + gid.x`, so the grid has to carry the row on its own
/// axis for the pitch to be applied at all. Firing this on a flat grid adds
/// the first row `width * rows` times.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
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
            lanes: crate::routine::elementwise_rows(*width, *rows)?,
        },
        &[x.v(), residual.v(), out.v(), row_pitch.v()],
    )
}

/// The Qwen-2 family's attention biases, IN PLACE over the value they bias.
///
/// One buffer that is both the input and the result -- which is why `out` is
/// the only activation here and why it is `BufMut`. The bias is one vector of
/// `width`, broadcast down every row.
///
/// `width` is a real argument and not an extent: the shader reads it from the
/// push block to find both the row base and the guard, so the grid and the
/// addressing come from the same number by construction. That is worth having
/// as one argument rather than two -- a grid width that disagreed with the
/// pushed width would bias the wrong columns of every row after the first.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: BufMut,
    bias: Buf,
    width: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "add_bias_bfloat16",
            lanes: crate::routine::elementwise_rows(width, *rows)?,
        },
        &[out.v(), bias.v(), width.v()],
    )
}

/// The twelve, in the order the rows above name them.
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

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// A reducing norm gets a WHOLE workgroup per row; an elementwise one gets
    /// a lane per element.
    ///
    /// The two launch shapes in this file look alike written down and are not
    /// the same kind of number. `rms_single_row` reduces a row and the
    /// workgroup that owns it must be whole, so 3 rows is 768 lanes;
    /// `residual_add` has nothing to reduce and 3 rows of 128 is 384 lanes.
    /// Giving the norm the elementwise count launches `ceil(rows * width /
    /// 256)` workgroups -- fewer than `rows` whenever the row is narrower than
    /// 256 -- and every row past the last one is left exactly as it arrived,
    /// unnormalised.
    #[test]
    fn a_reduction_is_launched_by_the_workgroup_and_an_elementwise_op_by_the_lane() {
        let seen = Seen::default();
        rms_single_row(
            &seen,
            Buf(0),
            Buf(1),
            BufMut(2),
            Buf(3),
            Env(128),
            Env(128),
            Env(3),
        )
        .expect("a launch");
        residual_add(&seen, Buf(0), Buf(1), BufMut(2), Env(128), Env(3)).expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([768, 1, 1], [384, 1, 1]),
            "three rows is three whole 256-lane workgroups for a reduction and \
             `3 * 128` lanes for an elementwise add"
        );
    }

    /// A two-level base needs two grid axes, and the head count is one of
    /// them.
    ///
    /// `rms_strided_head_row` addresses `row * row_pitch + head * axis_size`,
    /// and the two strides are not commensurate -- `row_pitch` is a whole
    /// token and `axis_size` is one head of it -- so the head cannot be folded
    /// into the row axis at any pitch. `gated_rms` is stronger still: its
    /// non-strided form reads the head count back out of `gl_NumWorkGroups.y`,
    /// so a y extent that is not `heads` does not run fewer heads, it
    /// addresses different rows.
    #[test]
    fn a_two_level_norm_carries_the_head_on_its_own_grid_axis() {
        let seen = Seen::default();
        rms_strided_head_row(
            &seen,
            Buf(0),
            Buf(1),
            BufMut(2),
            Buf(3),
            4096,
            Env(8),
            Env(5),
        )
        .expect("a launch");
        gated_rms(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            Buf(4),
            Env(8),
            Env(5),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([256, 8, 5], [256, 8, 5]),
            "one workgroup per (head, row) pair: x is the workgroup, y is the \
             head and z is the row, and the head is NOT multiplied into x"
        );
    }

    /// A strided elementwise op is a rectangle; a flat one is a run.
    ///
    /// `residual_add_strided` indexes `gid.y * row_pitch + gid.x` and
    /// `residual_add` indexes `gid.x`, so the strided form needs the row on
    /// its own axis for the pitch to be applied at all. Firing it flat gives
    /// every invocation `gid.y == 0` and adds the FIRST row `rows` times over,
    /// which for a one-row test -- the shape most of this tree is tested at --
    /// is indistinguishable from correct.
    #[test]
    fn a_strided_elementwise_op_is_a_rectangle_and_a_flat_one_is_a_run() {
        let seen = Seen::default();
        residual_add(&seen, Buf(0), Buf(1), BufMut(2), Env(128), Env(3)).expect("a launch");
        residual_add_strided(&seen, Buf(0), Buf(1), BufMut(2), 4096, Env(128), Env(3))
            .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([384, 1, 1], [128, 3, 1]),
            "the same 3 rows of 128: flat it is one run of 384 and strided it \
             is a 128-by-3 rectangle"
        );
    }

    /// The width a bias is launched on is the width it is TOLD.
    ///
    /// `add_bias` reads `pc.width` for both the row base and the out-of-range
    /// guard, so the grid and the addressing come from one number. Two
    /// arguments could disagree, and the disagreement is quiet: a grid wider
    /// than the pushed width returns early and biases nothing past the guard,
    /// and a grid narrower biases a prefix of every row and leaves the rest
    /// unbiased. Both decode as fluent text.
    #[test]
    fn a_bias_is_launched_on_the_width_it_is_told() {
        let seen = Seen::default();
        add_bias(&seen, BufMut(0), Buf(1), 640, Env(7)).expect("a launch");

        let call = &seen.0.borrow()[0];
        assert_eq!(call.1, [640, 7, 1], "the grid is the pushed width by rows");
        assert_eq!(
            call.2.last(),
            Some(&ArgValue::I32(640)),
            "and the pushed width is that same number, not a second one"
        );
    }

    /// A row count that overflows its lane count is refused, not wrapped.
    ///
    /// Multiplying by the workgroup width costs eight bits of headroom, so a
    /// row count that fits a `u32` comfortably can produce a lane count that
    /// does not. A wrap here is the worst available failure: the grid stays
    /// positive, the dispatch succeeds, and a small FRACTION of the rows is
    /// normalised while the rest pass through unchanged.
    ///
    /// The empty direction matters for the same reason and is quieter still --
    /// a zero grid normalises nothing at all and reports success.
    #[test]
    fn a_row_count_that_does_not_fit_the_lane_count_is_refused() {
        assert!(
            per_row(i32::MAX).is_err(),
            "`i32::MAX` rows is 2^39 lanes and does not fit the grid"
        );
        assert!(
            per_row(0).is_err() && per_row(-1).is_err(),
            "no rows is not a launch"
        );
        assert!(
            per_head_row(0, 4).is_err() && per_head_row(4, 0).is_err(),
            "no heads and no rows are both refused"
        );
        assert_eq!(
            per_row(1).expect("one row"),
            [256, 1, 1],
            "and one row is one whole workgroup"
        );
    }
}
