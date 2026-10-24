//! DSpark's Markov head over a drafted block's readouts
//! (`sampling.markov_walk`, DeepSeek-V4.1).

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, nonzero, refuse};
use crate::tensor::Tensor;

const FILE: &str = "sampling/markov_walk.cu";

/// The Markov embeddings' rank the kernel holds.
pub const MARKOV_RANK: u32 = 256;

/// A step's shared memory (`markov_walk.cu`'s `kSmem`): the head's 128-row
/// tile and a 32-readout chunk's embeddings, rows padded 16 bytes, and the
/// chunk's f32 products.
const STEP_SMEM: u32 = (128 + 32) * (MARKOV_RANK + 8) * 2 + 32 * (128 + 4) * 4;

/// `biased` (`[readouts, vocab]`, the logits' element): `logits` with each
/// readout's bias `head . embed[t]` added. A readout's depth in its drafted
/// block is `positions - (ends - block + 1)` (`ends` its lane's last key);
/// at depth 0 `t` is its own token in `tokens`, past it the token the
/// readout before it (the block's previous row) picked from its own biased
/// row. `picks` (i32 a readout): each one's pick, its biased row's argmax
/// taken in f32.
#[allow(clippy::too_many_arguments)]
pub fn markov_walk(
    ctx: &Ctx,
    logits: Tensor,
    tokens: Tensor,
    positions: Tensor,
    ends: Tensor,
    embed: Tensor,
    head: Tensor,
    block: u32,
    biased: &mut Tensor,
    picks: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.markov_walk";
    let t = match logits.dtype {
        Dtype::F32 => "float",
        Dtype::Bf16 => "__nv_bfloat16",
        held => return Err(refuse(OP, format!("the logits are {held:?}"))),
    };
    if biased.dtype != logits.dtype || picks.dtype != Dtype::I32 || picks.rows != logits.rows {
        return Err(refuse(
            OP,
            "the biased logits are the logits' element, a pick i32 a readout",
        ));
    }
    if embed.dtype != Dtype::Bf16 || head.dtype != Dtype::Bf16 {
        return Err(refuse(OP, "the Markov embeddings and head are bf16"));
    }
    if embed.width != MARKOV_RANK || head.width != MARKOV_RANK || head.rows != logits.width {
        return Err(refuse(
            OP,
            format!(
                "a rank-{MARKOV_RANK} head over the logits' {} columns, not [{}, {}] and [{}, {}]",
                logits.width, embed.rows, embed.width, head.rows, head.width
            ),
        ));
    }
    if [tokens, positions, ends]
        .iter()
        .any(|t| t.dtype != Dtype::I32 || t.rows != logits.rows)
    {
        return Err(refuse(OP, "a readout's i32 token, position and lane's end"));
    }
    let readouts = nonzero(OP, "readouts", logits.rows)?;
    let steps = nonzero(OP, "the block's rows", block)?;
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let r = readouts as usize;
    let lists = ctx.scratch_stream(OP, "markov.lists", r * steps as usize * 4)? as u64;
    let counts = ctx.scratch_stream(OP, "markov.counts", steps as usize * 4)? as u64;
    let depth = ctx.scratch_stream(OP, "markov.depth", r * 4)? as u64;
    let keys = ctx.scratch_stream(OP, "markov.keys", r * 8)? as u64;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::sampling::plan"))
            .apply(Launch::grid([1, 1, 1], [1024, 1, 1])),
        &[
            positions.arg(),
            ends.arg(),
            int(readouts).arg(),
            int(steps).arg(),
            ArgValue::Ptr(lists),
            ArgValue::Ptr(counts),
            ArgValue::Ptr(depth),
            ArgValue::Ptr(keys),
        ],
    )?;
    let tiles = logits.width.div_ceil(128);
    let gathered = ctx.scratch_stream(OP, "markov.gathered", r * MARKOV_RANK as usize * 2)? as u64;
    for s in 0..steps {
        // The depth's embeddings, a warp a readout, then its tiles.
        ctx.fire(
            OP,
            Fire::at(kernel(FILE, "::pie::sampling::gather"))
                .apply(Launch::grid([readouts.div_ceil(8), 1, 1], [256, 1, 1])),
            &[
                tokens.arg(),
                embed.arg(),
                ArgValue::Ptr(lists),
                ArgValue::Ptr(counts),
                ArgValue::Ptr(depth),
                ArgValue::Ptr(keys),
                ArgValue::Ptr(gathered),
                int(readouts).arg(),
                int(s).arg(),
            ],
        )?;
        ctx.fire(
            OP,
            Fire::at(kernel(FILE, &format!("::pie::sampling::step<{t}>")))
                .apply(Launch::grid([tiles, 1, 1], [256, 1, 1]).smem(STEP_SMEM)),
            &[
                logits.arg(),
                ArgValue::Ptr(gathered),
                head.arg(),
                biased.arg(),
                ArgValue::Ptr(lists),
                ArgValue::Ptr(counts),
                ArgValue::Ptr(keys),
                int(readouts).arg(),
                int(logits.width).arg(),
                int(s).arg(),
            ],
        )?;
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::sampling::picks")).apply(Launch::flat(readouts, 128)),
        &[ArgValue::Ptr(keys), picks.arg(), int(readouts).arg()],
    )
}
