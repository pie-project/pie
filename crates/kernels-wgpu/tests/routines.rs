//! What a crossed routine does when it is actually called.
//!
//! `kernels/tests/shader_backends_agree.rs` compares a routine's DERIVED row
//! against its `kernel!` row and against the other backends'. That is a check
//! on the signature. This is the check on the BODY, and the two see different
//! things:
//!
//! > an extra `.v()` in a body is not an extra argument in the signature.
//!
//! `kernels-vulkan` found that within an hour of writing the weaker form —
//! giving `neox_freqs_decode` a seventh pushed word turned nothing red,
//! because a pushed scalar is not a descriptor and no other check in either
//! plane counts them. This file is that finding, ported.
//!
//! # What each half of the list costs when it is wrong, on THIS backend
//!
//! A missing or extra BUFFER shifts every binding after it. `driver-wgpu`
//! builds its bind group from the dispatched list in order, so a dropped
//! operand binds the next buffer one slot early — a weight read as an output,
//! or an output written over a weight. `wgpu` validates the bind group against
//! the module's layout, so a COUNT mismatch is caught at dispatch; an
//! order swap of two buffers of the same kind is not caught anywhere.
//!
//! A misplaced SCALAR is worse and quieter. The uniform block is built by
//! walking the same list and appending each scalar's own width, so a word in
//! the wrong place is read as the field that lives at that offset — an `i32`
//! extent read as an `f32` scale, or the reverse, both of which produce
//! numbers rather than errors.
//!
//! # Why the recipes are written out
//!
//! A generic synthesizer cannot supply them. `embed_gather_4bit` refuses any
//! `(group, bits)` that is not one of six real affine points, and a body given
//! a zero extent refuses by design — so stand-in values have to be plausible,
//! and plausible is a fact about each kernel. The guard is that every crossed
//! routine has a recipe and every recipe names a crossed routine: a stale
//! entry is a routine silently running on stand-ins, and a missing one is a
//! body nothing calls.

use std::cell::{Cell, RefCell};

use kernels::routine::Refusal;
use kernels_wgpu::routine::{ArgValue, Encode, Fire, Routine};

/// One dispatch: the entrypoint it named, the lanes it asked for, and the
/// values it bound.
type Dispatched = (String, [u32; 3], Vec<ArgValue>);

/// Every dispatch a body made, in order, and a source of buffer handles for
/// this mock's OWN answers.
///
/// The `asked` counter is separate from the recipe's own `0..` so that a
/// fact this mock resolves inside a body can never be mistaken for one of the
/// recipe's own operands by the order check below — see [`Seen::asked_buffer`].
#[derive(Default)]
struct Seen {
    fired: RefCell<Vec<Dispatched>>,
    asked: Cell<u32>,
}

impl Seen {
    /// A fresh handle, clear of every recipe's own `0..N`.
    fn asked_buffer(&self) -> u32 {
        let at = 900 + self.asked.get();
        self.asked.set(self.asked.get() + 1);
        at
    }
}

impl Encode for Seen {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        self.fired
            .borrow_mut()
            .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
        Ok(())
    }

    /// Answers generically, by [`kernels::Ty`] alone, with one deliberate
    /// exception.
    ///
    /// None of this file's four tests reads the VALUE an asked fact resolves
    /// to — only order, arity, entrypoint names and buffer counts — so a
    /// probe that answers honestly by TYPE, ignoring which specific fact was
    /// named, is enough: a buffer-shaped `Ty` gets a fresh handle, a scalar
    /// `Ty` gets a small positive default. The one place that is not enough
    /// is `ssm::gdn_core_recurrent_prefill`, which turns two asked scalars
    /// into a compiled-shape LOOKUP (`scan_point`) rather than a bare
    /// forward: only nine `(LANES, VROWS)` pairs are compiled, and a generic
    /// default for each would almost certainly name none of them. `(32, 4)`
    /// is the pair this file's own `ssm` recipe named for the same body
    /// before `lanes`/`vrows` became asks rather than parameters.
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        use kernels::{Source as Src, Ty as T};
        // THE SAME TWO NUMBERS BY THE OTHER ROUTE. The tiling was
        // `Source::Named("lanes")` while the body asked for it as a fact; it
        // is the statement's eleventh and twelfth WORDS now -- HEAD's
        // `Param<11>`/`Param<12>` -- because the run this body forwards is the
        // shader's struct and no `Const` mark can name a word inside it.
        if let Src::Slot(kernels::Kind::Param, n) = source {
            return Ok(ArgValue::I32(match n {
                11 => 32,
                12 => 4,
                _ => 4096,
            }));
        }
        if source == Src::Named("lanes") {
            return Ok(ArgValue::I32(32));
        }
        if source == Src::Named("vrows") {
            return Ok(ArgValue::I32(4));
        }
        Ok(match ty {
            T::Buf
            | T::BufMut
            | T::Bf16s
            | T::Bf16sMut
            | T::F16s
            | T::F16sMut
            | T::I32s
            | T::I32sMut
            | T::U32s
            | T::U32sMut
            | T::U8s
            | T::U8sMut
            | T::F32s
            | T::F32sMut => ArgValue::Buffer(self.asked_buffer()),
            T::I32 => ArgValue::I32(8),
            T::U32 => ArgValue::U32(8),
            T::F32 => ArgValue::F32(1.0),
            T::Usize => ArgValue::Usize(4096),
            T::InPacked => ArgValue::U32(8),
            _ => {
                return Err(Refusal::Unstated {
                    what: "a fact this generic mock does not answer",
                });
            }
        })
    }
}

/// A buffer handle, distinct per position so a swap is visible, and SHAPED
/// like a real operand's binding.
///
/// `In`/`Out`/`InOut` read `.rows`/`.width` off exactly the rectangle
/// [`ArgValue::Shaped`] carries when they unpack; a plain `Buffer` gives every
/// one of them `Extent { rows: 0, width: 0 }` and most bodies refuse an empty
/// extent before doing anything else worth checking. `Const<Tensor<_>>`
/// (the weight run) reads only the handle, so the same value serves both.
const fn b(at: u32) -> ArgValue {
    ArgValue::Shaped {
        handle: at,
        rows: 7,
        width: 1024,
    }
}

/// The six affine points, which are the only `(group, bits)` a gather takes.
const POINTS: [(i32, i32); 6] = [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)];

/// One plausible argument list per crossed routine that takes no affine
/// point.
///
/// Most of these lists are far shorter than the tree that grew them: a scalar
/// that used to be `Env` is now either a `Const` signature parameter (still
/// stated here) or a fact the body `ctx.ask`s for from INSIDE itself (never
/// on the signature at all, so nothing is stated for it here — see
/// `Seen::resolve`, which answers whatever a body asks). The comment on each
/// arm is the CURRENT signature, in order, so a wrong value is visible to a
/// reader without counting parentheses against `src/*.rs`.
fn recipe(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let one = |v: Vec<ArgValue>| Some(vec![v]);
    match name {
        // proj, token, out -- `params`, `width` and `rows` are all asked for
        // or read off a buffer's own shape now.
        "ple_combine" => one(vec![b(0), b(1), b(2)]),
        // input, out -- every scalar this body once took as a parameter
        // (`rows`, `params`, `count`, `width`, `row_count`) is asked for now.
        "row_gather" => one(vec![b(0), b(1)]),
        // logits, next_token, params, eos_flag -- `params` here is an
        // ordinary buffer parameter (a `Tensor<bf16>`), not the staged scalar
        // block; `rows` moved to an ask.
        "argmax_logits" => one(vec![b(0), b(1), b(2), b(3)]),
        // source, destination -- `params`, `vocab` (from `source.width`) and
        // `rows` are all either read off a buffer or asked now.
        "copy_logits_bf16" => one(vec![b(0), b(1)]),

        // ---- norm, the first LIVE family crossed here ----
        //
        // x, w, out -- `width`, `axis` and `rows` are all asked for; none of
        // the three is a signature parameter any more.
        "rms_single_row" => one(vec![b(0), b(1), b(2), ArgValue::F32(1e-5), ArgValue::I32(1024)]),
        // x, w, out -- `row_pitch`/`rows` asked.
        "rms_strided_row" => one(vec![b(0), b(1), b(2), ArgValue::F32(1e-5), ArgValue::I32(512)]),
        // x, w, out -- `row_pitch`/`heads`/`rows` asked.
        "rms_strided_head_row" => one(vec![b(0), b(1), b(2), ArgValue::F32(1e-5), ArgValue::I32(512)]),
        // x, w, out, r -- `width`/`axis`/`rows` asked.
        "rms_residual" => one(vec![b(0), b(1), b(2), b(3)]),
        // x, w, out, r, s -- `width`/`axis`/`rows` asked.
        "rms_residual_scaled" => one(vec![b(0), b(1), b(2), b(3), b(4)]),
        // x, out -- `width`/`axis`/`rows` asked.
        "vnorm_single_row" => one(vec![b(0), b(1)]),
        // x, z, w, out, heads -- `rows` asked.
        // x, z, w, out -- `vd`/`heads`/`rows` asked. The head count left the
        // run: this body forwards `ctx.params()` as a STRUCT, so no slot in it
        // is a mark's to take.
        "gated_rms" => one(vec![b(0), b(1), b(2), b(3)]),
        // x, z, w, out -- `row_pitch`/`heads`/`rows` asked, same reason.
        "gated_rms_strided" => one(vec![b(0), b(1), b(2), b(3)]),
        // x, scalar, out -- `width`/`rows` asked.
        "layer_scalar_mul" => one(vec![b(0), b(1), b(2)]),
        // x, residual, out -- `width`/`rows` asked.
        "residual_add" => one(vec![b(0), b(1), b(2)]),
        // x, residual, out -- `row_pitch`/`width`/`rows` asked.
        "residual_add_strided" => one(vec![b(0), b(1), b(2), ArgValue::I32(4096)]),
        // out, bias -- `width`/`rows` asked. `out` is the `InOut` mark, so it
        // is the same address as the separate input this once took.
        "add_bias" => one(vec![b(0), b(1)]),

        // ---- moe ----
        //
        // logits, expert_ids, expert_weights -- `rows` asked;
        // `router_topk_scaled` also takes a fourth `Const` buffer,
        // `per_expert_scale`, that the plain form does not.
        "router_topk" => one(vec![b(0), b(1), b(2)]),
        "router_topk_scaled" => one(vec![b(0), b(1), b(2), b(3)]),
        // expert_ids, perm, row_expert, tile_expert, inv -- `params` asked.
        "route_sort" => one(vec![b(0), b(1), b(2), b(3), b(4)]),
        // x, out, perm -- `params`/`width`/`padded` asked.
        "route_gather" => one(vec![b(0), b(1), b(2)]),
        // y, expert_weights, out, inv -- `params`/`width`/`tokens` asked.
        "combine_sorted" => one(vec![b(0), b(1), b(2), b(3)]),
        // routed, shared, gate, out -- `width`/`rows` asked.
        "shared_expert_combine" => one(vec![b(0), b(1), b(2), b(3)]),
        // ... `row_pitch`/`width`/`rows` asked; still the same four buffers.
        "shared_expert_combine_strided" => one(vec![b(0), b(1), b(2), b(3)]),
        // w, scales, biases, x, y, x_slot_stride, x_row_stride,
        // slots_per_row, expert_ids. `in_vec`/`out_vec` are the operands'
        // own widths and `rows` is asked; the three STRIDES are the
        // mixture's own geometry, which no operand carries and the statement
        // states.
        "qmv_routed" => one(vec![
            b(0), b(1), b(2), b(3), b(4),
            ArgValue::I32(1), ArgValue::I32(4), ArgValue::I32(4),
            b(5),
        ]),
        // ... plus a `bias` buffer, positioned before the strides.
        "qmv_routed_bias" => one(vec![
            b(0), b(1), b(2), b(3), b(4), b(5),
            ArgValue::I32(1), ArgValue::I32(4), ArgValue::I32(4),
            b(6),
        ]),
        // w, scales, x, y, bias, then the strides -- the mxfp4 form has no
        // `biases` buffer at all; its `scales` IS the affine codec.
        "mxfp4_qmv_routed_bias" => one(vec![
            b(0), b(1), b(2), b(3), b(4),
            ArgValue::I32(1), ArgValue::I32(4), ArgValue::I32(4),
            b(5),
        ]),
        // w, scales, biases, x, y, pad, tile_expert, group, bits, tile_m,
        // tile_n -- `pad` is the statement's second input (`row_expert`'s
        // slot), which this GEMM does not read but must still bind
        // positionally; `k`/`n`/`rows` are read off buffers or asked now.
        "qmm_t_routed" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),
        // ... without the group/bits pair: this form is stamped at one codec
        // and only the tile is still a choice.
        "qmm_t_routed_fp16" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),
        // w, exponents, x, y, bias, pad, tile_expert, tile_m, tile_n -- the
        // same unread positional `pad` as the two affine forms above.
        "mxfp4_qmm_t_routed_bias" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),

        // ---- mlp ----
        //
        // gate, up, out -- every one of these now asks for `width`/`rows`;
        // none keeps a `params` buffer or a scalar on its signature.
        "geglu_tanh" | "geglu_tanh_strided" | "gptoss_swiglu" => one(vec![b(0), b(1), b(2)]),
        "silu_mul" => one(vec![b(0), b(1), b(2)]),
        "silu_mul_strided" => one(vec![b(0), b(1), b(2)]),

        // ---- rope ----
        //
        // `rotary` 64 against `head_dim` 128 deliberately: a PARTIAL
        // rotation, so `rope_grid`'s x is 32 rather than half the head. A
        // body that built its grid on the head width instead of the rotary
        // width would give the same number for the full-rotation case and a
        // different one here. Both are still `Const` parameters; `position`
        // and `width` (the third and seventh of the old list) are asked for
        // or read off `x` now.
        //
        // x, scale, base, head_dim, rotary
        // THE STRIDED FORM TAKES ONE MORE: its `row_pitch` is the statement's
        // (`Param<4>` at HEAD), and 4096 is wide enough for the row the stand-in
        // builds -- a narrower one is the refusal `kernels-vulkan`'s
        // `a_stride_narrower_than_the_row_is_refused` exercises on purpose.
        "neox_strided" => one(vec![
            b(0),
            ArgValue::F32(1.0),
            ArgValue::F32(10_000.0),
            ArgValue::I32(128),
            ArgValue::I32(64),
            ArgValue::I32(4096),
        ]),
        "neox_decode" | "neox_prop_decode" | "neox_mb" | "neox_prop_mb" => {
            one(vec![
                b(0),
                ArgValue::F32(1.0),
                ArgValue::F32(10_000.0),
                ArgValue::I32(128),
                ArgValue::I32(64),
            ])
        }
        // x, scale, head_dim, mscale, rotary -- `inv_freq` is the FIRE's
        // frequency table, built once per fire and asked for, not a weight
        // the checkpoint carries.
        "neox_freqs_decode" | "neox_freqs_mb" => one(vec![
            b(0),
            ArgValue::F32(1.0),
            ArgValue::I32(128),
            ArgValue::F32(1.0),
            ArgValue::I32(64),
        ]),
        _ => None,
    }
}

/// The gathers, swept over all six affine points.
///
/// Sweeping rather than picking one is what makes the entrypoint check below a
/// census: all twenty-four spellings a gather can choose are resolved, and
/// mistyping one table entry is caught by the point that names it.
///
/// The `_mb_` split is no longer a signature difference: the M>1 forms ask
/// for `Rows` where the M=1 forms hardcode it (see `layout::embed_gather_mb_
/// 4bit` against `embed_gather_4bit`), but neither keeps a `rows` PARAMETER,
/// so the four routines' signatures differ only in `embed_scale`.
fn affine(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let scaled = name.contains("scaled");
    if !name.starts_with("embed_gather") {
        return None;
    }
    Some(
        POINTS
            .iter()
            .map(|(g, bits)| {
                // w, scales, biases, out, [embed_scale], group, bits -- `id`
                // (the token-ids buffer) is asked for and `hidden` is read
                // off `out.width` now, in every one of the four.
                let mut v = vec![b(0), b(1), b(2), b(3)];
                if scaled {
                    v.push(ArgValue::F32(1.5));
                }
                v.push(ArgValue::I32(*g));
                v.push(ArgValue::I32(*bits));
                v
            })
            .collect(),
    )
}

/// The `attn` family's recipes.
///
/// Sixteen routines whose signatures run at most eight arguments now (down
/// from twenty): most of what a signature used to state for these bodies —
/// the KV planes, the position table, the attention mask, `GqaFactor` — is
/// asked for from inside the body instead (see `attn::sdpa_paged_decode` for
/// the fullest example), and what remains a `Const` parameter is exactly the
/// checkpoint-fixed geometry: `page_size`, `n_kv_heads`, `scale`, `window`,
/// `head_dim`, `q_heads` (or `heads`).
///
/// The head width is the one argument every attention still constrains: the
/// body indexes a literal spelling table with it (`head_point`), which
/// refuses anything not one of the widths its own table names. `HEADS` states
/// those per routine, swept rather than picked once for the same reason the
/// old version gave: it is what makes the entrypoint census below exhaustive.
/// `page_size`, `n_kv_heads`, `scale` and `window` gate nothing —
/// `vector_grid`/`tiled_grid`/`head_grid` take only `head_dim`, `q_heads` (or
/// `heads`) and `rows` — so any value serves for them, and 16/2/0.125/0 are
/// simply distinct from `head_dim`/`q_heads` so a transposition is visible.
fn attn(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    /// The head widths each routine's spelling table actually carries.
    ///
    /// `sdpa_paged_decode` sweeps FOUR where its row states seven: the three
    /// `_p32` tails are not reachable from a routine, for the reason
    /// `kernels-wgpu`'s
    /// `the_entrypoints_that_ignore_the_window_are_exactly_the_ones_named_p32`
    /// gives — they pin the key run's start to zero and answer a windowed
    /// caller with full attention.
    ///
    /// `q_gate_split`, `kv_append` and `kv_append_paged` now take a REAL
    /// `head_dim: Const<i32>` too (`head_grid` refuses a non-positive one),
    /// so they sweep a real width the same as the attentions; `split_qkv_
    /// bf16`, `gate` and `logit_softcap` take none at all, and get one dummy
    /// entry each so the loop below still runs once.
    const HEADS: &[(&str, &[i32])] = &[
        ("sdpa_paged_decode", &[64, 128, 256, 512]),
        ("sdpa_paged_decode_sink", &[64]),
        ("sdpa_paged_tiled", &[64, 128, 256, 512]),
        ("sdpa_paged_tiled_sink", &[64]),
        ("sdpa_paged_tiled_strided", &[256]),
        ("sdpa_paged_mma", &[64]),
        ("sdpa_paged_mma_sink", &[64]),
        ("sdpa_vector_decode", &[64, 128, 256]),
        ("sdpa_vector_decode_sink", &[64]),
        ("sdpa_vector_decode_swa", &[256, 512]),
        ("q_gate_split", &[64]),
        ("kv_append", &[64]),
        ("kv_append_paged", &[64]),
        // Not attentions, and no `head_dim` on their signature at all.
        ("split_qkv_bf16", &[0]),
        ("gate", &[0]),
        ("logit_softcap", &[0]),
    ];

    let heads = HEADS.iter().find(|(n, _)| *n == name)?.1;
    let mut out = Vec::new();
    for &head_dim in heads {
        let hd = ArgValue::I32(head_dim);
        let args = match name {
            // packed, q, k, v -- no scalar at all.
            "split_qkv_bf16" => vec![b(0), b(1), b(2), b(3)],
            // attn, gate -- no scalar at all.
            // attn, gate, row_stride -- the pitch was `Param<0>` at HEAD.
            "gate" => vec![b(0), b(1), ArgValue::I32(4096)],
            // qg, q_out, gate_out, head_dim, q_heads
            "q_gate_split" => vec![b(0), b(1), b(2), hd, ArgValue::I32(8), ArgValue::I32(4096), ArgValue::I32(4096)],
            // k_new, v_new, head_dim, heads
            "kv_append" => vec![b(0), b(1), hd, ArgValue::I32(8)],
            // k_new, v_new, head_dim, n_kv_heads -- `page_size` is asked for
            // now (see `sdpa_paged_decode`'s identical comment, below).
            "kv_append_paged" => vec![b(0), b(1), hd, ArgValue::I32(2)],
            // logits, out -- no scalar at all.
            "logit_softcap" => vec![b(0), b(1)],
            // queries, out, n_kv_heads, scale, window, head_dim, q_heads --
            // `page_size` is a fact only the fire can answer (a property of
            // the allocation, not of the model text) and is asked for from
            // inside the body now, rather than stated on the signature.
            "sdpa_paged_decode" | "sdpa_paged_tiled" | "sdpa_paged_tiled_strided"
            | "sdpa_paged_mma" => vec![
                b(0),
                b(1),
                ArgValue::I32(2),
                ArgValue::F32(0.125),
                ArgValue::I32(0),
                hd,
                ArgValue::I32(8),
            ],
            // ..., with a `sinks` buffer between `window` and `head_dim`.
            "sdpa_paged_decode_sink" | "sdpa_paged_tiled_sink" | "sdpa_paged_mma_sink" => vec![
                b(0),
                b(1),
                ArgValue::I32(2),
                ArgValue::F32(0.125),
                ArgValue::I32(0),
                b(2),
                hd,
                ArgValue::I32(8),
            ],
            // queries, out, scale, head_dim, q_heads
            "sdpa_vector_decode" => {
                vec![b(0), b(1), ArgValue::F32(0.125), hd, ArgValue::I32(8)]
            }
            // queries, out, scale, window, head_dim, q_heads, q/o pitches
            "sdpa_vector_decode_swa" => vec![
                b(0),
                b(1),
                ArgValue::F32(0.125),
                ArgValue::I32(0),
                hd,
                ArgValue::I32(8),
                // The two row pitches, which were `Param<4>`/`Param<5>` and are
                // the statement's again: a stride is the rectangle the text laid
                // out, not something this batch made.
                ArgValue::I32(4096),
                ArgValue::I32(4096),
            ],
            // queries, out, sinks, scale, window, head_dim, q_heads, q/o pitches
            "sdpa_vector_decode_sink" => vec![
                b(0),
                b(1),
                b(2),
                ArgValue::F32(0.125),
                ArgValue::I32(0),
                hd,
                ArgValue::I32(8),
                // The two row pitches, which were `Param<4>`/`Param<5>` and are
                // the statement's again: a stride is the rectangle the text laid
                // out, not something this batch made.
                ArgValue::I32(4096),
                ArgValue::I32(4096),
            ],
            _ => return None,
        };
        out.push(args);
    }
    Some(out)
}

/// The `quant` family's recipes.
///
/// Thirty-one routines whose signatures shrank to at most nine arguments;
/// what remains constrained is exactly what indexes a spelling table — the
/// affine `group`/`bits` (`quant::codec_point`) and the `bm`/`bn` tile
/// (`tile_point`/`wide_point`/`row_tile_point`) — stated per routine below,
/// in the order the signature takes them, with the buffers getting
/// sequential handles.
///
/// `group` 64, `bits` 4 and `bm`/`bn` 32/64 throughout: distinct values on
/// every axis a body could transpose, and all four are real points --
/// `GROUPS`, `BIT_WIDTHS` and `TILES` all carry them.
fn quant(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let one = |v: Vec<ArgValue>| Some(vec![v]);
    match name {
        // w, scales, biases, x, y, group, bits, bm, bn -- `k`/`n`/`rows` are
        // read off buffers or asked now.
        "qmm_t" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, x, y, bias, group, bits, bm, bn
        "qmm_t_bias" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, x, y, residual, group, bits, bm, bn
        "qmm_t_residual" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, y, half_in, bm, bn -- this codec is fixed, so
        // only the tile is still a choice.
        "qmm_t_fp16_precast" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, y, bias, half_in, bm, bn
        "qmm_t_bias_fp16_precast" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, y, residual, half_in, bm, bn
        "qmm_t_residual_fp16_precast" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(32),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, x, out, group, bits, bm -- the split-K forms
        // take one tile edge, not two: see `wide_point`.
        "qmm_t_splitk" | "qmm_t_splitk_f32" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, out, half_in, bm
        "qmm_t_splitk_fp16_precast" | "qmm_t_splitk_fp16_precast_f32" => {
            one(vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(32)])
        }
        // w, scales, biases, x, y, group, bits, bm
        "qmm_t_strided" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, x, y, residual, group, bits, bm
        "qmm_t_strided_residual" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
        ]),
        // w, scales, biases, y, half_in, bm
        "qmm_t_strided_fp16_precast" => {
            one(vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(32)])
        }
        // w, scales, biases, y, residual, half_in, bm
        "qmm_t_strided_fp16_precast_residual" => {
            one(vec![b(0), b(1), b(2), b(3), b(4), b(5), ArgValue::I32(32)])
        }
        // y, partial -- no scalar at all; the f32 and bf16 partials both
        // cross here, only the buffer's element type differs.
        "qmm_splitk_reduce" | "qmm_splitk_reduce_f32" => one(vec![b(0), b(1)]),
        // cast_in, half_out -- `k` is the operand's own width and `count`
        // and `rows` are asked.
        "cast_qmm_input_bfloat16_to_float16" => one(vec![b(0), b(1)]),
        // ... and the STRIDED form takes the source's row pitch, which is the
        // activation's own stride: the text knows it and the fire does not.
        "cast_qmm_input_strided_bfloat16_to_float16" => {
            one(vec![b(0), b(1), ArgValue::I32(1024)])
        }
        // w, scales, biases, x, y, group, bits
        "qmv_fast" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(64),
            ArgValue::I32(4),
        ]),
        // w, scales, biases, x, y, residual, group, bits
        "qmv_fast_residual" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(64),
            ArgValue::I32(4),
        ]),
        // w, scales, biases, x, y, bits -- `vecs` moved to an ask.
        "qmv_tail" => one(vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(4)]),
        // w, scales, biases, x, y, bias, bits
        "qmv_tail_bias" => one(vec![b(0), b(1), b(2), b(3), b(4), b(5), ArgValue::I32(4)]),
        // w, scales, biases, x, y, bits -- this form used to take `vecs`
        // before `bits`, in the one order its siblings do not; `vecs` is gone
        // from the signature entirely, so the old transposed-order trap no
        // longer has anything to transpose.
        "qmv_wide_strided" => one(vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(4)]),
        // w, scales, biases, x, y -- these five are each stamped at one
        // exact tile and codec; nothing is left to state but the buffers.
        "qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"
        | "qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"
        | "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"
        | "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"
        | "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4" => {
            one(vec![b(0), b(1), b(2), b(3), b(4)])
        }
        // input, codes, scales, biases, group_size -- `groups` is asked now.
        "encode_u4_bf16" | "encode_u4_f32" => {
            one(vec![b(0), b(1), b(2), b(3), ArgValue::I32(32)])
        }
        // payload, exponents, out, block_size -- `blocks` is asked now.
        "mxfp4_dequant_bf16" => one(vec![b(0), b(1), b(2), ArgValue::I32(32)]),
        _ => None,
    }
}

/// The `ssm` family's recipes.
///
/// `dv`/`hv`/`v_dim`/`v_heads` are the only scalars any of these nine
/// routines still take on its signature; `rows` and (for the recurrent
/// prefill scan) `lanes`/`vrows` are asked for in the body now -- see
/// `Seen::resolve`, which answers `keys::Lanes`/`keys::Vrows` with `(32, 4)`,
/// the one pair `gdn_core_recurrent_prefill`'s `SCAN` table actually compiles
/// that a generic default would not have found.
fn ssm(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let one = |v: Vec<ArgValue>| Some(vec![v]);
    match name {
        // mixed, core_out, conv_w, conv_b, a_log, dt_bias, a_gate, b_gate,
        // v_heads, v_dim -- `rows` moved to an ask.
        "gdn_core" | "gdn_core_slotted" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
            b(7),
        ]),
        // mixed, conv_w, conv_b, a_log, dt_bias, a_gate, b_gate, pre_q,
        // pre_k, pre_gate, v_heads -- `rows` (and, for the prefill,
        // `row_pitch`/`n_scan`) moved to asks.
        "gdn_prep" | "gdn_prep_slotted" | "gdn_prep_prefill" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
            b(7),
            b(8),
            b(9),
        ]),
        // mixed, core_out, conv_w, conv_b, pre_q, pre_k, pre_gate, v_heads,
        // v_dim -- `rows` moved to an ask.
        "gdn_core_recurrent" | "gdn_core_recurrent_slotted" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            b(6),
        ]),
        // pad, core_out, pre_q, pre_k, pre_gate, dv, hv -- `pad` is `mixed`'s
        // slot, which this scan does not read (the mark is still there
        // because the slot is a POSITION, and without it `pre_q` would bind
        // where `mixed` should); `row_pitch` (read off `pre_q.width` now) and
        // `n_scan`/`lanes`/`vrows` all moved to asks, so only the two
        // channel counts `scan_point`'s grid math still needs directly are
        // left on the signature besides the five buffers.
        "gdn_core_recurrent_prefill" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
        ]),
        _ => None,
    }
}

/// Every crossed routine, called on every recipe, with what it did.
fn fired() -> Vec<(&'static Routine, Vec<ArgValue>, Seen)> {
    let all = kernels_wgpu::routines();
    assert!(
        !all.is_empty(),
        "no routine has crossed, so this file compares nothing"
    );

    let mut out = Vec::new();
    for r in all {
        let recipes = recipe(r.name)
            .or_else(|| affine(r.name))
            .or_else(|| attn(r.name))
            .or_else(|| quant(r.name))
            .or_else(|| ssm(r.name))
            .unwrap_or_else(|| {
                panic!(
                    "`{}` has crossed and this file has no recipe for it. A body \
                 nothing calls is a body nothing checks.",
                    r.name
                )
            });
        for args in recipes {
            assert_eq!(
                args.len(),
                r.args.len(),
                "`{}`'s recipe supplies {} arguments and its signature takes \
                 {}. The recipe is stale.",
                r.name,
                args.len(),
                r.args.len()
            );
            // One `Seen` per list: pooling them would compare a swept gather's
            // second point against its first point's recipe.
            let seen = Seen::default();
            (r.body)(&seen, &args)
                .unwrap_or_else(|e| panic!("`{}` refused {args:?}: {e:?}", r.name));
            out.push((r, args, seen));
        }
    }
    out
}

/// A body passes its dispatch the BUFFERS its recipe gave it, in the same
/// relative order; a buffer the module declares no slot for may be skipped.
///
/// # What this checked before, and why the scalar half of it cannot survive
///
/// This used to be a subsequence over EVERY argument, buffers and scalars
/// alike, with `Provenance::Env` marking the ones a skip could excuse: an
/// `Env` argument sized the grid and was never forwarded to `.arg()`, so its
/// absence from the dispatch was legal and everything else had to appear in
/// order.
///
/// `Provenance` is deleted, and nothing took its place at the type level. A
/// `Const` scalar on today's signature is forwarded to `.arg()` normally, or
/// consumed only for grid math and never forwarded (`neox_decode`'s
/// `rotary`, spent entirely inside `rope_grid`), or removed from the
/// signature altogether and asked for from inside the body instead (`ssm`'s
/// `rows`, `lanes`, `vrows` — see `Seen::resolve`). The second and third
/// cases are indistinguishable, from this test's vantage point, from a
/// dropped or reordered recipe value: there is no marker left to read. A
/// "the value appears in the chosen entrypoint's name" heuristic does not
/// close the gap either -- `rotary` is consumed but never chosen a spelling
/// with, so it would still read as a bare, unexplained absence.
///
/// # What still is checkable, and why
///
/// A recipe's buffers keep their old guarantee: every one [`b`] mints carries
/// a handle unique within its own recipe (`0..N`) and disjoint from every
/// handle [`Seen::resolve`] mints for an asked fact (`900..`), so "did this
/// specific buffer appear in the dispatch, and in what relative order" stays
/// answerable with no provenance signal at all. So the check now covers
/// buffers only: for every buffer handle the recipe supplied that also shows
/// up among the fired arguments, its position among the fired buffers must be
/// strictly increasing in the recipe's own order. A recipe buffer the
/// dispatch never binds -- because the module declares no slot for it, as
/// `moe::mxfp4_qmv_routed_bias`'s `//#if`-gated mxfp4 arm does not for
/// `biases` -- is silently allowed to be missing, exactly as the old
/// `Buffer`-may-be-skipped rule always allowed. Only a genuine REORDERING
/// among the buffers that DO appear is now flagged. Scalars are excluded
/// entirely: with no signal left to tell a legitimately-unforwarded `Const`
/// from a dropped one, a scalar position check would either reject correct
/// bodies or wave through broken ones depending on which stand-in value this
/// file's mock happens to reuse.
///
/// Two routines are EXCUSED from the ordering check outright rather than
/// passing it honestly: `qmm_t_residual_fp16_precast` and
/// `qmm_t_strided_fp16_precast_residual` both declare `half_in` before
/// `residual` -- the statement's input 0 and input 1, in that order -- and
/// both fire `residual` before `half_in`, the order the compiled shader's own
/// buffer table wants. The comment beside each signature says so (the second
/// by reference to the first), in the identical words, in `kernels-vulkan`
/// and `kernels-metal` too, so this is a cross-plane design rather than a
/// slip: with `InSlot<N, _>`/`OutSlot<N, _>` deleted, a mark's declaration
/// position is the only order left that can read as "the statement's", and
/// the shader's own bind order is a separate fact that only the fire array
/// encodes. A body is free to state its operands in one order and fire them
/// in another; these two routines are the ones that actually do, so the
/// ordering check below is not the place to hold them to a stricter rule.
#[test]
fn a_body_passes_the_arguments_its_signature_takes_in_order() {
    /// The handle underneath a buffer-shaped argument, recipe or fired.
    ///
    /// A recipe's own values are always [`b`]'s `Shaped`, and every fired
    /// value a mark forwards is always a plain `Buffer` (`Tensor::arg` reads
    /// only `.handle`, so the two variants never compare equal to each other
    /// even when they name the same buffer) -- this is the projection that
    /// makes the two sides comparable at all.
    fn handle_of(v: &ArgValue) -> Option<u32> {
        match *v {
            ArgValue::Buffer(h) | ArgValue::Shaped { handle: h, .. } => Some(h),
            _ => None,
        }
    }

    const REORDERS_BY_DESIGN: &[&str] =
        &["qmm_t_residual_fp16_precast", "qmm_t_strided_fp16_precast_residual"];

    for (r, args, seen) in fired() {
        if REORDERS_BY_DESIGN.contains(&r.name) {
            continue;
        }
        let want: Vec<u32> = args.iter().filter_map(handle_of).collect();
        for (entrypoint, _, got) in seen.fired.borrow().iter() {
            let got_handles: Vec<u32> = got.iter().filter_map(handle_of).collect();
            let mut last: Option<usize> = None;
            for handle in &want {
                // Not found at all: legitimately skipped, exactly as a
                // `Buffer` was always allowed to be.
                let Some(pos) = got_handles.iter().position(|h| h == handle) else {
                    continue;
                };
                if let Some(before) = last {
                    assert!(
                        pos > before,
                        "`{}` fires `{entrypoint}` with its buffers out of \
                         order: handle {handle} lands at position {pos} \
                         among {got_handles:?}, not after the previous \
                         recipe buffer's position {before}",
                        r.name
                    );
                }
                last = Some(pos);
            }
        }
    }
}

/// No routine ever asks for a zero-length grid.
///
/// `dispatch_workgroups(0, 1, 1)` is legal WebGPU that runs nothing and
/// reports success over a buffer that kept whatever it held. A body with
/// nothing to do must refuse instead, which is a value the caller reads.
#[test]
fn no_routine_dispatches_an_empty_grid() {
    for (r, _, seen) in fired() {
        for (entrypoint, lanes, _) in seen.fired.borrow().iter() {
            assert!(
                !lanes.contains(&0),
                "`{}` fires `{entrypoint}` with lanes {lanes:?}",
                r.name
            );
        }
    }
}

/// Every entrypoint any body can name is one the shader tree carries.
///
/// The crate-wide form of the per-family check: a body composes its spelling
/// from the facts it is given, so a point outside the tree is a name that
/// resolves to nothing at first fire. Sweeping the affine recipes over all six
/// points means all twenty-four gather spellings are resolved here, with no
/// adapter.
#[test]
fn every_entrypoint_a_body_names_is_one_the_tree_carries() {
    let mut checked = 0usize;
    for (r, _, seen) in fired() {
        for (entrypoint, _, _) in seen.fired.borrow().iter() {
            kernels_wgpu::source::entrypoint_source(entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|e| {
                    panic!(
                        "`{}` names `{entrypoint}`, which the tree has not got: {e}",
                        r.name
                    )
                });
            checked += 1;
        }
    }
    assert!(
        checked >= 28,
        "only {checked} spellings were resolved; the affine sweep alone is \
         twenty-four"
    );
}

/// A routine passes a buffer for EVERY `@group(0)` binding its module
/// declares — including the ones the entrypoint never reads.
///
/// # The sibling finding, inverted
///
/// `kernels-vulkan` landed the opposite fix. Its `encode::dispatch` requires
/// exactly `declared.bindings - holes()` buffers, and `slangc` emits no
/// binding decoration for a global nothing reads, so nine of its crossed
/// routines were forwarding operands their modules had compiled out — an
/// arity refusal at the first real dispatch. Its fix was that a body declines
/// to FORWARD what its module deleted, and its
/// `a_body_passes_the_arguments_its_signature_takes_in_order` became
/// a SUBSEQUENCE check to allow the skipping.
///
/// **Do not copy that here.** WGSL declares its bindings in the source and
/// `naga` keeps a `GlobalVariable` the entry point never reads, so nothing is
/// deleted; and `driver-wgpu` builds an EXPLICIT bind group layout from those
/// declarations rather than from the compiled usage. `wgpu` then requires the
/// bind group to MATCH that layout. So on this backend a declared-and-unread
/// binding is a slot the shell must still fill, and a body that skipped it
/// would shift every buffer after it — a weight read as an output. The two
/// backends' bodies are NOT interchangeable at exactly this point, and the
/// ports have already adopted vulkan's tables verbatim once.
///
/// `driver-wgpu::reflect`'s module docs state this reading; this is the same
/// claim asked of every crossed body, which is the half prose cannot do.
///
/// Falsified in both directions: dropping a `.v()` from a body fails with the
/// count short, adding one fails with it long.
#[test]
fn every_routine_binds_a_buffer_for_every_binding_its_module_declares() {
    let mut checked = 0usize;
    let mut unread_slots = 0usize;
    for (r, _, seen) in fired() {
        for (entrypoint, _, dispatched) in seen.fired.borrow().iter() {
            let source = kernels_wgpu::source::entrypoint_source(
                entrypoint,
                kernels_wgpu::Capability::Baseline,
            )
            .unwrap_or_else(|e| panic!("`{entrypoint}`: {e}"));
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("`{entrypoint}` does not parse: {e}"));

            let declared: Vec<naga::Handle<naga::GlobalVariable>> = module
                .global_variables
                .iter()
                .filter(|(_, g)| g.binding.as_ref().is_some_and(|b| b.group == 0))
                .map(|(h, _)| h)
                .collect();
            let passed = dispatched
                .iter()
                .filter(|a| matches!(a, ArgValue::Buffer(_)))
                .count();
            assert_eq!(
                passed,
                declared.len(),
                "`{}` passed {passed} buffers to `{entrypoint}`, whose module \
                 declares {} in `@group(0)`. On this backend the layout comes \
                 from the DECLARATIONS, so every one of them is a slot that \
                 must be filled -- see this test's own docs before copying a \
                 vulkan body that skips one.",
                r.name,
                declared.len()
            );

            // And how many of those the entrypoint never reads, which is the
            // number that is zero on no backend and harmless only on this
            // one. Counted rather than asserted about: it is a fact of the
            // shader tree, and the check that matters is the one above.
            //
            // BY INDEX, not by name. A generated source has exactly one entry
            // point and it is called `main` -- which is why `driver-wgpu`
            // passes `entry_point: None` and lets `wgpu` take the only one
            // there is. The first version of this looked it up by the
            // INSTANTIATION name, found nothing every time, and reported a
            // confident zero. `tests/gpu.rs`'s `writes_agree_with_the_body`
            // made the identical mistake, and ITS falsification passed twice
            // because of it.
            assert_eq!(
                module.entry_points.len(),
                1,
                "`{entrypoint}` compiled to {} entry points; `driver-wgpu` \
                 passes `entry_point: None` and relies on there being one",
                module.entry_points.len()
            );
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            let info = validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("`{entrypoint}` does not validate: {e}"));
            let f = info.get_entry_point(0);
            unread_slots += declared.iter().filter(|h| f[**h].is_empty()).count();
            checked += 1;
        }
    }
    assert!(checked >= 28, "only {checked} dispatches were read");
    println!("{checked} dispatches; {unread_slots} declared-but-unread slots");
}
