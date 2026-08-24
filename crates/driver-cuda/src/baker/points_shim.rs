//! The point shim: the plane's own claims, by declaration order.
//!
//! # This file is a placeholder for a generator
//!
//! `.wiki/baker.md` says what replaces it: `#[claims]` reads the plane's
//! impl blocks and emits `dispatch(ctx, op)` — the one data→type crossing,
//! with the mechanical `Elem^k` cartesian and no instantiation annotations.
//! W5 is that generator, and `baker-smoke`'s hand shim (which this is
//! lifted from, `smoke.rs:963-1100`) is its spec. Nothing here is meant to
//! be maintained by hand for long; it is here so the fire path can be
//! proven before the generator exists, and so the generator has a target
//! whose output is already known-correct.
//!
//! # The shape every arm has
//!
//! Every arm is the same three moves and nothing else: read the operands
//! and scalars off the statement in the order `model_dsl::kernels` records
//! them, wear them as the marks `kernels::points` declares, call the
//! method. The dtype comes off the SLOT -- there is no default and no
//! cast; a dtype with no arm is a refusal naming the point, which is what
//! a generated dispatch's `Elem^axes` match does for the axes a plane has
//! no instantiation for.
//!
//! The set of points and dtypes answered here is mirrored by
//! [`super::resolve::CLAIMED`], which is what turns "no arm" from a
//! mid-fire refusal into a load-time one. That doubling, and why its drift
//! is fail-safe, is argued there.

use kernels::points::{Gate, Gemm, Layout, Mlp, Norm, Rope, Ssm};
use kernels::routine::Refusal;
use kernels_cuda::jit::abi::bf16;
use model_baker::produce::Dtype;
use model_compiler::program::Dt;
use model_ir::plan::Op;

use super::fire::Fire;
use super::marks::{rin, rio, rout, wconst};

/// Fire one `Call::Point` through the plane's claim.
#[allow(clippy::too_many_lines)]
pub(crate) fn point(f: &Fire<'_>, point: &str, op: &Op) -> Result<(), Refusal> {
    let ctx = f.ctx;
    // A dtype with no arm, named. `Refusal::Absent` wants a `&'static str`
    // and this one is built per call, so it is leaked — a handful of bytes
    // on a path that is about to fail the load or the fire, never in a
    // steady state. The generated dispatch will have a `&'static` per arm
    // and no leak at all.
    let unpointed = |dt: Dt| Refusal::Absent {
        what: Box::leak(format!("`{point}` at {dt:?}").into_boxed_str()),
    };
    match point {
        // The two norms and their OFFSET-BANK twins. The convention is
        // the checkpoint's, so it is the point that picks: `_plus_one`
        // scales by `1 + weight`.
        "norm.rmsnorm" | "norm.rmsnorm_plus_one" => {
            let (x, y, w) = (f.input(op, 0)?, f.output(op, 0)?, f.weight(op, 0)?.ptr);
            let eps = Fire::pf32(op, 0)?;
            let plus = point == "norm.rmsnorm_plus_one";
            match (y.dt, plus) {
                (Dt::Bf16, false) => ctx.rmsnorm::<bf16>(rin(x), wconst(w), eps, rout(y)),
                (Dt::F32, false) => ctx.rmsnorm::<f32>(rin(x), wconst(w), eps, rout(y)),
                (Dt::Bf16, true) => ctx.rmsnorm_plus_one::<bf16>(rin(x), wconst(w), eps, rout(y)),
                (Dt::F32, true) => ctx.rmsnorm_plus_one::<f32>(rin(x), wconst(w), eps, rout(y)),
                (other, _) => Err(unpointed(other)),
            }
        }
        "norm.rmsnorm_per_head" | "norm.rmsnorm_per_head_plus_one" => {
            let (x, y, w) = (f.input(op, 0)?, f.output(op, 0)?, f.weight(op, 0)?.ptr);
            let (head_dim, eps) = (Fire::p32(op, 0)?, Fire::pf32(op, 1)?);
            let plus = point == "norm.rmsnorm_per_head_plus_one";
            match (y.dt, plus) {
                (Dt::Bf16, false) => {
                    ctx.rmsnorm_per_head::<bf16>(rin(x), wconst(w), head_dim, eps, rout(y))
                }
                (Dt::Bf16, true) => {
                    ctx.rmsnorm_per_head_plus_one::<bf16>(rin(x), wconst(w), head_dim, eps, rout(y))
                }
                (other, _) => Err(unpointed(other)),
            }
        }
        "norm.rmsnorm_gated" => {
            // The declaration SPELLS the core and the weight f32 and
            // quantifies only over the gate's element -- which is why
            // `program.rs`'s width rule sizes this result from `like(1)`.
            let (core, gate) = (f.input(op, 0)?, f.input(op, 1)?);
            let (y, w) = (f.output(op, 0)?, f.weight(op, 0)?);
            let (head_dim, eps) = (Fire::p32(op, 0)?, Fire::pf32(op, 1)?);
            if core.dt != Dt::F32 {
                return Err(unpointed(core.dt));
            }
            if w.dtype != Dtype::F32 {
                // The checkpoint ships qwen's gdn norm F32 and the
                // declaration agrees; a bf16 bank here would be a silent
                // halving of every stride inside the kernel.
                return Err(Refusal::Absent {
                    what: "a gated norm weight stored f32",
                });
            }
            match y.dt {
                Dt::Bf16 => ctx.rmsnorm_gated::<bf16>(
                    rin(core),
                    rin(gate),
                    wconst(w.ptr),
                    head_dim,
                    eps,
                    rout(y),
                ),
                other => Err(unpointed(other)),
            }
        }
        "norm.residual_add" => {
            // `(x: In, y: InOut)` -- the ONE point of the family whose
            // `InOut` is not the receiver, which `program.rs`'s width
            // table calls out and sizes from `like(1)`.
            let x = f.input(op, 0)?;
            let y = f.inout(f.input(op, 1)?, f.output(op, 0)?)?;
            match y.dt {
                Dt::Bf16 => ctx.residual_add::<bf16>(rin(x), rio(y)),
                Dt::F32 => ctx.residual_add::<f32>(rin(x), rio(y)),
                other => Err(unpointed(other)),
            }
        }
        "gemm.matmul" | "gemm.lm_head" | "gemm.attention_landing" => {
            let (act, y, w) = (f.input(op, 0)?, f.output(op, 0)?, f.weight(op, 0)?.ptr);
            let layer = op.layer.unwrap_or(0);
            match y.dt {
                Dt::Bf16 => match point {
                    "gemm.matmul" => ctx.matmul::<bf16>(rin(act), wconst(w), rout(y)),
                    "gemm.lm_head" => ctx.lm_head::<bf16>(rin(act), wconst(w), rout(y)),
                    _ => ctx.attention_landing::<bf16>(rin(act), wconst(w), layer, rout(y)),
                },
                other => Err(unpointed(other)),
            }
        }
        "mlp.swiglu" => {
            let (packed, y) = (f.input(op, 0)?, f.output(op, 0)?);
            let intermediate = Fire::p32(op, 0)?;
            match y.dt {
                Dt::Bf16 => ctx.swiglu::<bf16>(rin(packed), intermediate, rout(y)),
                other => Err(unpointed(other)),
            }
        }
        "gate.sigmoid_mul" => {
            let gate = f.input(op, 1)?;
            let x = f.inout(f.input(op, 0)?, f.output(op, 0)?)?;
            match x.dt {
                Dt::Bf16 => ctx.sigmoid_mul::<bf16>(rio(x), rin(gate)),
                other => Err(unpointed(other)),
            }
        }
        // The walk's ROOT: one table row per token id, clamped against the
        // vocab the statement states. It was a `Call::Symbol` until R4a and
        // the staging it needed was that one number, read out of the plan's
        // weight shape; the declaration carries it now.
        "layout.embed" => {
            let (ids, y, table) = (f.input(op, 0)?, f.output(op, 0)?, f.weight(op, 0)?.ptr);
            let vocab = Fire::p32(op, 0)?;
            match y.dt {
                Dt::Bf16 => ctx.embed::<bf16>(rin(ids), wconst(table), vocab, rout(y)),
                other => Err(unpointed(other)),
            }
        }
        "layout.split_q_gate" => {
            let packed = f.input(op, 0)?;
            let (q, gate) = (f.output(op, 0)?, f.output(op, 1)?);
            let head_dim = Fire::p32(op, 0)?;
            match q.dt {
                Dt::Bf16 => ctx.split_q_gate::<bf16>(rin(packed), head_dim, rout(q), rout(gate)),
                other => Err(unpointed(other)),
            }
        }
        "layout.split_rows" => {
            let x = f.input(op, 0)?;
            let (left, right) = (f.output(op, 0)?, f.output(op, 1)?);
            let width = Fire::p32(op, 0)?;
            match x.dt {
                Dt::Bf16 => ctx.split_rows::<bf16>(rin(x), width, rout(left), rout(right)),
                other => Err(unpointed(other)),
            }
        }
        "rope.partial" => {
            let pos = f.input(op, 2)?;
            let q = f.inout(f.input(op, 0)?, f.output(op, 0)?)?;
            let k = f.inout(f.input(op, 1)?, f.output(op, 1)?)?;
            let (rotary_dim, head_dim, theta) = (
                Fire::p32(op, 0)?,
                Fire::p32(op, 1)?,
                Fire::pf32(op, 2)?,
            );
            match q.dt {
                Dt::Bf16 => {
                    ctx.partial::<bf16>(rio(q), rio(k), rin(pos), rotary_dim, head_dim, theta)
                }
                other => Err(unpointed(other)),
            }
        }
        // THE TWO GDN POINTS, and neither carries a rectangle this shim
        // has to cut. That is the whole of W10: `ssm.gdn_prep` takes the
        // packed `[b | a]` projection and states the packed
        // `[g_log | beta]` row, `ssm.gated_delta` takes the packed `qkv`
        // and that same row, and the claim bodies do every packed→compact
        // cut in a kernel. What used to be here — a reach backwards
        // through the plan for a missing operand, three carved scratch
        // columns, and four `Rect::column` pointer offsets — was a row
        // stride of `v_heads` claimed for bytes whose stride is
        // `2 * v_heads`: right at one row, silently wrong at two.
        "ssm.gdn_prep" => {
            let (ba, gates) = (f.input(op, 0)?, f.output(op, 0)?);
            let dt_bias = f.weight(op, 0)?;
            let a_log = f.weight(op, 1)?;
            if a_log.dtype != Dtype::F32 {
                // `A_log` is one f32 decay per value head and the kernel
                // spells its slot `const float*`; a bf16 bank there is
                // half the decays, read as nonsense.
                return Err(Refusal::Absent {
                    what: "an `a_log` bank stored f32",
                });
            }
            match ba.dt {
                Dt::Bf16 => ctx.gdn_prep::<bf16>(
                    rin(ba),
                    wconst(dt_bias.ptr),
                    wconst(a_log.ptr),
                    rout(gates),
                ),
                other => Err(unpointed(other)),
            }
        }
        "ssm.gated_delta" => {
            let (qkv, z, gates) = (f.input(op, 0)?, f.input(op, 1)?, f.input(op, 2)?);
            let y = f.output(op, 0)?;
            let state = f.recurrent(op)?;
            let (k_heads, v_heads) = (Fire::p32(op, 0)?, Fire::p32(op, 1)?);
            let (k_dim, v_dim) = (Fire::p32(op, 2)?, Fire::p32(op, 3)?);
            match qkv.dt {
                Dt::Bf16 => ctx.gated_delta::<bf16>(
                    rin(qkv),
                    rin(z),
                    rin(gates),
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    rout(y),
                ),
                other => Err(unpointed(other)),
            }
        }
        "ssm.causal_conv1d" => {
            let (x, y, w) = (f.input(op, 0)?, f.output(op, 0)?, f.weight(op, 0)?.ptr);
            let state = f.recurrent(op)?;
            let conv_width = Fire::p32(op, 0)?;
            match y.dt {
                Dt::Bf16 => {
                    ctx.causal_conv1d::<bf16>(rin(x), wconst(w), state, conv_width, rout(y))
                }
                other => Err(unpointed(other)),
            }
        }
        other => Err(Refusal::Absent {
            what: Box::leak(
                format!("a point shim for `{other}`; this driver states none").into_boxed_str(),
            ),
        }),
    }
}
