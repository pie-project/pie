use crate::attn::kv;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;
use dtype::Dtype;

const RAGGED: &str = "attn/ragged.hip";

const OP: &str = "attn.ragged";

/// Wavefronts a block runs, 16 query rows each.
const WAVES: u32 = 4;

const BLOCK: u32 = WAVES * 64;

/// Query rows a block answers.
const TILE: u32 = WAVES * 16;

/// The head widths the kernel is stamped at (multiples of 16); a head is
/// zero-padded to the first that holds it.
const STAMPS: [u32; 5] = [64, 80, 96, 128, 256];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaggedMask {
    None,
    /// Keeps group `g`'s pair `(q, k)` where bit `q * kv_len + k` from byte `indptr[g]` is set.
    Custom {
        bits: Tensor,
        indptr: Tensor,
    },
    RelativeBias {
        table: Tensor,
        max_len: u32,
    },
}

/// The kernel's arm, as `attn/ragged.hip` numbers them.
#[derive(Clone, Copy)]
enum Arm {
    Full = 0,
    Custom = 1,
    Bias = 2,
}

/// The arm's operands, as the kernel takes them after the shape.
struct Operands {
    arm: Arm,
    bits: ArgValue<'static>,
    mask_indptr: ArgValue<'static>,
    bias: ArgValue<'static>,
    max_len: u32,
}

impl Operands {
    const fn of(arm: Arm) -> Self {
        Self {
            arm,
            bits: ArgValue::ABSENT,
            mask_indptr: ArgValue::ABSENT,
            bias: ArgValue::ABSENT,
            max_len: 0,
        }
    }
}

fn row_heads(what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// Ragged (unpaged) attention over packed q/k/v, the vision towers' form: as
/// kernels-cuda's, each arm masking or biasing by group-local positions, and
/// a row no group holds left as it was.
#[allow(clippy::too_many_arguments)]
pub fn ragged(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    let Some(stamp) = STAMPS.into_iter().find(|stamp| head_dim <= *stamp) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide accumulator this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads("query", q.width, head_dim)?;
    let num_kv_heads = row_heads("key", k.width, head_dim)?;
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if v.width != k.width {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide value row is not the {}-wide key row",
                v.width, k.width
            ),
        ));
    }
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(k.rows, v.rows, "`{OP}` reads one value row per key row");
    let groups = kv::lanes_of(OP, q_indptr)?;
    if kv_indptr.rows != q_indptr.rows || kv_indptr.dtype != q_indptr.dtype {
        return Err(refuse(
            OP,
            format!(
                "the query table spells {} groups and the key table {}; a group is one entry \
                 of each",
                q_indptr.rows.saturating_sub(1),
                kv_indptr.rows.saturating_sub(1)
            ),
        ));
    }
    let rows = count(OP, "the query rows this attention answers", q.rows)?;
    let sm_scale = if sm_scale > 0.0 {
        sm_scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    };

    let operands = match mask {
        RaggedMask::None => Operands::of(Arm::Full),
        RaggedMask::Custom { bits, indptr } => {
            if bits.dtype != Dtype::U8 || indptr.dtype != Dtype::I32 || indptr.rows != q_indptr.rows
            {
                return Err(refuse(
                    OP,
                    format!(
                        "a custom mask is u8 bits and an i32 byte offset a group; this one is \
                         {:?} bits and {} {:?} offsets for {} groups",
                        bits.dtype,
                        indptr.rows,
                        indptr.dtype,
                        q_indptr.rows.saturating_sub(1)
                    ),
                ));
            }
            Operands {
                bits: bits.arg(),
                mask_indptr: indptr.arg(),
                ..Operands::of(Arm::Custom)
            }
        }
        RaggedMask::RelativeBias { table, max_len } => {
            let span = max_len
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
                .filter(|_| max_len > 0)
                .ok_or_else(|| {
                    refuse(
                        OP,
                        format!("a relative bias over {max_len} positions has no table width"),
                    )
                })?;
            if table.dtype != Dtype::F32 || table.rows < num_q_heads || table.width != span {
                return Err(refuse(
                    OP,
                    format!(
                        "the relative bias table is {} x {} {:?}; the arm reads one f32 row of \
                         {span} (2 · {max_len} − 1) per query head, {num_q_heads} of them",
                        table.rows, table.width, table.dtype
                    ),
                ));
            }
            Operands {
                bias: table.arg(),
                max_len,
                ..Operands::of(Arm::Bias)
            }
        }
    };

    ctx.fire(
        OP,
        Fire::at(kernel(
            RAGGED,
            &format!(
                "::pie::attn::ragged<{stamp}, {WAVES}, {}>",
                operands.arm as u32
            ),
        ))
        .apply(Launch::grid(
            [rows.unsigned_abs().div_ceil(TILE), num_q_heads, 1],
            [BLOCK, 1, 1],
        )),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg(),
            q_indptr.arg(),
            kv_indptr.arg(),
            groups.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
            rows.arg(),
            operands.bits,
            operands.mask_indptr,
            operands.bias,
            stated(OP, operands.max_len)?.arg(),
        ],
    )
}
