//! The staging shim: the routines that keep their own `canon`.
//!
//! A `Call::Symbol` is a routine for which no honest delegation to a point
//! exists — the plane's `#[claims]` blocks name each one where it is
//! measured, `kernels-cuda/src/ssm.rs`'s family doc being the longest.
//! Those need STAGING: operands the statement does not carry, results it
//! does not state, resident objects it only names. That is what this file
//! is, and unlike [`super::points_shim`] it is NOT a generator's placeholder
//! — a generator has nothing to read here, because the gap between what the
//! statement says and what the routine wants is exactly the thing no
//! declaration captures. Each of these dies when its routine is decomposed
//! into points that state their own operands.
//!
//! Lifted from `baker-smoke/src/smoke.rs:1104-1310`, with every driver
//! citation kept and the four substitutions the driver makes marked
//! `REUSED:`.
//!
//! THREE OF THE FIVE ARMS ARE GONE. `ssm.gdn_prep` and `ssm.gated_delta`
//! (W10) were the ones that needed the most staging — a backwards reach
//! through the plan for an operand the statement never carried, three carved
//! scratch columns for results it never stated, and four column cuts into
//! packed rows — and every one of those was the symptom of a routine whose
//! signature was not the point's. Both are claim bodies now
//! (`kernels-cuda/src/ssm.rs`), so the statement and the launch state the
//! same thing and the cuts happen in a kernel that knows the packing.
//!
//! `layout.embed` (R4a) went the OTHER way and is worth keeping as the
//! second shape this file dies in. Its staging was not a rectangle at all:
//! it read the embedding table's ROW count out of the plan's weight shape,
//! because `embed_bf16` clamps every id against it and the point stated no
//! such number. The declaration states `vocab` now, so the number reaches
//! the plane through the statement like every other geometry and there is
//! nothing left here to stage. An arm dies either by its routine being
//! decomposed into points that state their own operands, or by the one
//! number it reached around the statement for becoming part of it.

use kernels::routine::{Const, In, Refusal};
use model_ir::plan::Op;

use super::fire::Fire;
use super::marks::{rin, rout};

/// Fire one `Call::Symbol` through the routine it names.
#[allow(clippy::too_many_lines)]
pub(crate) fn symbol(f: &Fire<'_>, symbol: &str, op: &Op) -> Result<(), Refusal> {
    let ctx = f.ctx;
    let g = f.geom;
    match symbol {
        // The appender's three runtime planes. `first_token` is a scalar
        // in the pointer channel and `row_valid` is one BYTE per row --
        // both declared `In<Tensor<i32>>` and both read as something
        // else, which is the prefix-agreement the two legs of
        // `attn::write_kv_to_pages` are pinned to
        // (`kernels-cuda/src/attn/mod.rs:2384-2396`).
        //
        // REUSED: all three come off `FireViews::streams`, so the write
        // lands on the pages the scheduler assigned this request, at the
        // offset `kv_and_arrays` computed for this fire.
        "attn::write_kv_to_pages" => {
            let (k, v) = (f.input(op, 0)?, f.input(op, 1)?);
            let pages = f.pages(op)?;
            let qo = f.rect_of_runtime("qo_indptr")?;
            let valid = f.rect_of_runtime("row_valid")?;
            let first_token = f.first_token();
            kernels_cuda::attn::kv_paged::write_kv_to_pages_bf16(
                ctx,
                rin(k),
                rin(v),
                pages,
                Const::new(g.kv_heads),
                Const::new(g.head_dim),
                In {
                    ptr: first_token as usize as *const i32,
                    rows: 0,
                    width: 0,
                },
                In {
                    ptr: qo.ptr.cast(),
                    rows: qo.rows,
                    width: qo.width,
                },
                In {
                    ptr: valid.ptr.cast(),
                    rows: valid.rows,
                    width: valid.width,
                },
            )
        }

        "attn::dispatch_attention_flashinfer_decode" => {
            let (q, o) = (f.input(op, 0)?, f.output(op, 0)?);
            let pages = f.pages(op)?;
            // The statement's window param is `Option<u32>` flattened by
            // `Stmt::window`, which spells `None` as `0`
            // (`model-dsl/src/record.rs:133-135`). flashinfer spells the
            // same absence `-1`, and the driver passes `-1` for every
            // qwen fire (`fire/launch.rs:3209`). A NON-ZERO window would
            // need the `w` -> `window_left` convention pinned, and no
            // shipping text in this tree states one -- so it refuses
            // rather than guessing.
            let window_left = match Fire::p32(op, 0)? {
                0 => -1,
                _ => {
                    return Err(Refusal::Unstated {
                        what: "how a stated sliding window maps to flashinfer's `window_left`",
                    });
                }
            };
            // The point declares no soft cap, so the statement states
            // none, and zero is what "no cap" spells at this routine
            // (`decode_arm`, `kernels-cuda/src/attn/fa2/mod.rs:1069`).
            let logits_soft_cap = 0.0f32;
            let sm_scale = Fire::pf32(op, 2)?;
            // REUSED: the schedule `raise_attn_plans` raised for THIS fire.
            // The smoke carried its own `DecodePlanCache` and replanned it
            // per fire with `full_attention_variant = true, window_left =
            // -1`; the driver raises exactly that from the text's own
            // `PrepKind::DecodeAttention` (`fire/launch.rs:1631-1643`),
            // with the workspaces stamped inside the
            // `begin_plan_update`/`end_plan_update` fence that says the
            // schedule upload landed. Planning a second one here would be
            // a second 48 MB workspace and a second answer to a question
            // the fire already answered.
            if f.decode_plan.is_null() {
                return Err(Refusal::Absent {
                    what: "the fa2 decode schedule this fire was raised on",
                });
            }
            kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode(
                ctx,
                rin(q),
                In {
                    ptr: f.decode_plan,
                    rows: 0,
                    width: 0,
                },
                rout(o),
                Const::new(window_left),
                Const::new(logits_soft_cap),
                Const::new(sm_scale),
                pages,
                // No log-sum-exp: this lane states one attention leg and
                // nothing merges partials across it.
                None,
            )
        }

        other => Err(Refusal::Absent {
            what: Box::leak(
                format!("a staging shim for `{other}`; this driver states none").into_boxed_str(),
            ),
        }),
    }
}
