#![allow(clippy::too_many_arguments)]

use crate::jit::Abi;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::routine::{In, InOut};
use kernels::{Bind, Fire};

use core::ptr::NonNull;

pub const ROTATE_BLOCK: i32 = 256;

pub const FUSED_BLOCK: u32 = 128;

pub const MAX_CACHED_PAIRS: i32 = 4096;

#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK {
        1
    } else {
        ROTATE_BLOCK / half
    }
}

#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

#[must_use]
const fn rotate_launch(num_tokens: i32, total_heads: i32, per_block: i32, smem: u32) -> Launch {
    #[must_use]
    const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
        [
            num_tokens.unsigned_abs(),
            (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
            1,
        ]
    }

    Launch::grid(
        rotate_grid(num_tokens, total_heads, per_block),
        [ROTATE_BLOCK.unsigned_abs(), 1, 1],
    )
    .smem(smem)
}

#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    #[allow(clippy::cast_precision_loss)]
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

fn heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

fn q_heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty {
            what: "the q region's width",
        });
    }
    heads(width, head_dim)
}

fn k_heads<T>(q: *mut T, k: *mut T, width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 && !k.is_null() && !core::ptr::eq(k.cast_const(), q.cast_const()) {
        return Err(Refusal::Empty {
            what: "the k region's width",
        });
    }
    heads(width, head_dim)
}

/// A stated width as the launches below spell one. A declaration states
/// `u32` because a width is not negative; a routine takes `i32` because the
/// device text does.
fn stated(width: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(width).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(width),
        max: i64::from(i32::MAX),
    })
}

/// The element a statement rides, re-marked as the one this plane's rotate
/// rows take, or a refusal naming the point that has no row for it.
///
/// Which axes a plane serves is a runtime truth, never a declared list, and
/// cuda's rotations are rows rather than open templates: `rope.cuh` states
/// that only `standard_table` and `rotate_partial` are templates and that
/// "the other eight stay bf16", because they rotate through
/// `rope_device.cuh`'s `rotate_pair`, which takes `bf16*`. `rotate_partial`
/// IS a template and `rope_partial_f16` is the row that would open the f16
/// axis — no text has ever stated one, so it stays a row nothing fires and
/// this claim stays at the element the family is measured on. The compare
/// is what makes the cast sound: past it, `T` IS `bf16`.
fn rotates_bf16<T: kernels::points::Scalar>(
    r: InOut<Tensor<T>>,
    what: &'static str,
) -> Result<InOut<Tensor<bf16>>, Refusal> {
    if <T as kernels::Elem>::TY_MUT != <bf16 as kernels::Elem>::TY_MUT {
        return Err(Refusal::Absent { what });
    }
    Ok(InOut {
        ptr: r.ptr.cast::<bf16>(),
        rows: r.rows,
        width: r.width,
    })
}

/// The `Rope` family, claimed. Every body is the launch itself, deriving
/// what `rope/rope.cuh` asks for and the declaration does not state:
/// `num_q_heads` and `num_kv_heads` are each operand's row over the STATED
/// head width, which is the whole reason `head_dim` is stated.
///
/// `rope.partial_last` passes [`Yarn::NONE`]'s numbers, and that is the
/// point's definition rather than a guess: the trailing rotation states no
/// YaRN block, `rotate_partial_last` guards its ramp with
/// `if (yarn_factor > 1.f)`, and the interpolated rotation is `rope.yarn` —
/// a different point, not a parameterisation of this one.
#[kernels_macros::claims]
impl kernels::points::Rope for Ctx<'_> {
    fn full<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        k: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let head_dim = stated(head_dim, "the head width this rotation states")?;
        let q = rotates_bf16(q, "rope.full")?;
        let k = rotates_bf16(k, "rope.full")?;
        let (num_q_heads, num_kv_heads) = (
            q_heads(q.width, head_dim)?,
            k_heads(q.ptr, k.ptr, k.width, head_dim)?,
        );
        let half = head_dim / 2;
        let pairs = cache_pairs(half);
        let per_block = heads_per_block(half);
        self.fire(
            Fire::at(
                "rope/rope.cuh",
                "::pie::rope::rotate<::pie::false_type::value, false>",
            )
            .apply(rotate_launch(
                q.rows,
                num_q_heads + num_kv_heads,
                per_block,
                pairs.unsigned_abs() * 2 * 4,
            )),
            &[
                q.arg(),
                k.arg(),
                positions.ptr.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                interleaved.arg(),
                pairs.arg(),
                per_block.arg(),
                MaybeConst::<bf16>::none().arg(),
                None::<NonNull<bf16>>.arg(),
                None::<NonNull<bf16>>.arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u8>::none().arg(),
                0_i32.arg(),
                0_i32.arg(),
            ],
        )
    }

    fn partial<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        k: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let rotary_dim = stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = stated(head_dim, "the head width this rotation states")?;
        let q = rotates_bf16(q, "rope.partial")?;
        let k = rotates_bf16(k, "rope.partial")?;
        rope_partial(
            self,
            "::pie::rope::rotate_partial<::pie::bf16>",
            q.ptr,
            k.ptr,
            positions.ptr,
            q.rows,
            q.width,
            k.width,
            head_dim,
            rotary_dim,
            theta,
        )
    }

    fn partial_q<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let rotary_dim = stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = stated(head_dim, "the head width this rotation states")?;
        let q = rotates_bf16(q, "rope.partial_q")?;
        rope_partial_q_bf16(self, q, rotary_dim, head_dim, theta, positions.ptr)
    }

    fn partial_last<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let rotary_dim = stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = stated(head_dim, "the head width this rotation states")?;
        let q = rotates_bf16(q, "rope.partial_last")?;
        // ONE PLANE, TWO SLOTS. `rotate_partial_last` takes q and k; this
        // point states q alone, so k is q's address with a ZERO width — the
        // head count `heads` derives from it is zero and the kernel's k leg
        // runs over nothing.
        let num_q_heads = q_heads(q.width, head_dim)?;
        // `rope.partial_last` states no YaRN block, and that is the point's
        // definition rather than a guess: `rotate_partial_last` guards its
        // ramp with `if (yarn_factor > 1.f)`, and the interpolated rotation
        // is `rope.yarn` — a different point, not a parameterisation of
        // this one.
        self.fire(
            Fire::at("rope/rope.cuh", "::pie::rope::rotate_partial_last").apply(
                Launch::per_row(q.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            ),
            &[
                q.arg(),
                q.ptr.arg(),
                positions.ptr.arg(),
                num_q_heads.arg(),
                0_i32.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
                false.arg(),
                interleaved.arg(),
                Yarn::NONE.factor.arg(),
                0.0_f32.arg(),
                0.0_f32.arg(),
            ],
        )
    }

    fn yarn<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        k: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let head_dim = stated(head_dim, "the head width this rotation states")?;
        let original_max_position = stated(
            original_max_position,
            "the position span this checkpoint's YaRN block states",
        )?;
        let q = rotates_bf16(q, "rope.yarn")?;
        let k = rotates_bf16(k, "rope.yarn")?;
        if original_max_position <= 0 {
            return Err(Refusal::Unstated {
                what: "the checkpoint's YaRN block",
            });
        }
        let (num_q_heads, num_kv_heads) = (
            q_heads(q.width, head_dim)?,
            k_heads(q.ptr, k.ptr, k.width, head_dim)?,
        );
        let (low_dim, high_dim) =
            ramp_bounds(head_dim, theta, beta_fast, beta_slow, original_max_position);
        let half = head_dim / 2;
        let pairs = cache_pairs(half);
        let per_block = heads_per_block(half);
        self.fire(
            Fire::at("rope/rope.cuh", "::pie::rope::rotate_yarn_original").apply(
                rotate_launch(
                    q.rows,
                    num_q_heads + num_kv_heads,
                    per_block,
                    pairs.unsigned_abs() * 8,
                ),
            ),
            &[
                q.arg(),
                k.arg(),
                positions.ptr.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                attention_factor.arg(),
                interleaved.arg(),
                per_block.arg(),
                pairs.arg(),
            ],
        )
    }
}

fn rope_partial<T>(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    q: *mut T,
    k: *mut T,
    positions: *const i32,
    num_tokens: i32,
    q_width: i32,
    k_width: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
) -> Result<(), Refusal>
where
    T: kernels::Elem,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Write: Abi,
{
    let (num_q_heads, num_kv_heads) = (
        q_heads(q_width, head_dim)?,
        k_heads(q, k, k_width, head_dim)?,
    );
    ctx.fire(
        Fire::at("rope/rope.cuh", instantiation).apply(Launch::per_row(
            num_tokens.unsigned_abs(),
            ROTATE_BLOCK.unsigned_abs(),
        )),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            0i32.arg(),
            num_q_heads.arg(),
            num_kv_heads.arg(),
            head_dim.arg(),
            rotary_dim.arg(),
            theta.arg(),
        ],
    )
}

/// The query-only partial rotation.
///
/// TWO CALLERS, WHICH IS WHY IT IS A FUNCTION: `Rope::partial_q` above, and
/// kimi's MLA prologue, which rotates the `k_pe` lane it just cut. The k
/// slot is q's address at a ZERO width, so the kernel's k leg runs over no
/// heads.
pub(crate) fn rope_partial_q_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    rotary_dim: i32,
    head_dim: i32,
    theta: f32,
    positions: *const i32,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        q.ptr,
        positions,
        q.rows,
        q.width,
        0,
        head_dim,
        rotary_dim,
        theta,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
    pub original_max_position: i32,
}
impl Yarn {
    pub const NONE: Self = Self {
        factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        attention_factor: 1.0,
        original_max_position: 0,
    };
}
