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

use std::cell::RefCell;

use kernels::Ty;
use kernels::routine::{Provenance, Refusal};
use kernels_wgpu::routine::{ArgValue, Encode, Fire, Routine};

/// One dispatch: the entrypoint it named, the lanes it asked for, and the
/// values it bound.
type Dispatched = (String, [u32; 3], Vec<ArgValue>);

/// Every dispatch a body made, in order.
#[derive(Default)]
struct Seen(RefCell<Vec<Dispatched>>);

impl Encode for Seen {
    fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
        self.0
            .borrow_mut()
            .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
        Ok(())
    }
}

/// A buffer handle, distinct per position so a swap is visible.
const fn b(at: u32) -> ArgValue {
    ArgValue::Buffer(at)
}

/// The six affine points, which are the only `(group, bits)` a gather takes.
const POINTS: [(i32, i32); 6] = [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)];

/// One plausible argument list per crossed routine that takes no affine point.
fn recipe(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let one = |v: Vec<ArgValue>| Some(vec![v]);
    match name {
        // proj, token, out, params, width, rows
        "ple_combine" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(64),
            ArgValue::I32(7),
        ]),
        // input, out, rows, params, count, width, row_count
        "row_gather" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::U32(4),
            ArgValue::I32(64),
            ArgValue::I32(7),
        ]),
        // logits, next_token, params, eos_flag, rows
        "argmax_logits" => one(vec![b(0), b(1), b(2), b(3), ArgValue::U32(7)]),
        // source, destination, params, vocab, rows
        "copy_logits_bf16" => one(vec![
            b(0),
            b(1),
            b(2),
            ArgValue::U32(1024),
            ArgValue::U32(7),
        ]),

        // ---- norm, the first LIVE family crossed here ----
        //
        // `width` 256 and `axis` 128 deliberately: two axes per row, so
        // `per_axis`'s `width / axis` is not 1 and a body that confused the
        // two would produce a different lane count rather than the same one.
        //
        // x, w, out, params, width, axis, rows
        "rms_single_row" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(256),
            ArgValue::I32(128),
            ArgValue::I32(7),
        ]),
        // x, w, out, params, row_pitch, rows
        "rms_strided_row" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(512),
            ArgValue::I32(7),
        ]),
        // x, w, out, params, row_pitch, heads, rows
        "rms_strided_head_row" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(512),
            ArgValue::I32(8),
            ArgValue::I32(7),
        ]),
        // x, w, out, params, r, width, axis, rows
        "rms_residual" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(256),
            ArgValue::I32(128),
            ArgValue::I32(7),
        ]),
        // x, w, out, params, r, s, width, axis, rows
        "rms_residual_scaled" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(256),
            ArgValue::I32(128),
            ArgValue::I32(7),
        ]),
        // x, out, params, width, axis, rows
        "vnorm_single_row" => one(vec![
            b(0),
            b(1),
            b(2),
            ArgValue::I32(256),
            ArgValue::I32(128),
            ArgValue::I32(7),
        ]),
        // x, z, w, out, params, heads, rows
        "gated_rms" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(8),
            ArgValue::I32(7),
        ]),
        // x, z, w, out, params, row_pitch, heads, rows
        "gated_rms_strided" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(512),
            ArgValue::I32(8),
            ArgValue::I32(7),
        ]),
        // x, scalar, out, params, width, rows
        "layer_scalar_mul" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(256),
            ArgValue::I32(7),
        ]),
        // x, residual, out, width, rows
        "residual_add" => one(vec![b(0), b(1), b(2), ArgValue::I32(256), ArgValue::I32(7)]),
        // x, residual, out, row_pitch, width, rows
        "residual_add_strided" => one(vec![
            b(0),
            b(1),
            b(2),
            ArgValue::I32(512),
            ArgValue::I32(256),
            ArgValue::I32(7),
        ]),
        // out, bias, width, rows
        "add_bias" => one(vec![b(0), b(1), ArgValue::I32(256), ArgValue::I32(7)]),

        // ---- moe ----
        //
        // Tiles 32x64 rather than 16x16: a square tile at the table's first
        // point would let a body that swapped `tile_m` and `tile_n` spell the
        // same entrypoint and build the same grid.
        //
        // logits, expert_ids, expert_weights, params, per_expert_scale, rows
        "router_topk" | "router_topk_scaled" => {
            one(vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(7)])
        }
        // expert_ids, perm, row_expert, tile_expert, params, inv
        "route_sort" => one(vec![b(0), b(1), b(2), b(3), b(4), b(5)]),
        // x, out, perm, params, width, padded
        "route_gather" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(1024),
            ArgValue::I32(16),
        ]),
        // y, expert_weights, out, params, inv, width, tokens
        "combine_sorted" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(1024),
            ArgValue::I32(7),
        ]),
        // routed, shared, gate, out, width, rows
        "shared_expert_combine" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::U32(1024),
            ArgValue::I32(7),
        ]),
        // ..., row_pitch before rows
        "shared_expert_combine_strided" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::U32(1024),
            ArgValue::I32(2048),
            ArgValue::I32(7),
        ]),
        // w, scales, biases, x, y, in_vec, out_vec, bias, expert_ids,
        // x_slot_stride, x_row_stride, slots_per_row, rows
        "qmv_routed" | "qmv_routed_bias" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(2048),
            ArgValue::I32(1024),
            b(5),
            b(6),
            ArgValue::I32(2048),
            ArgValue::I32(4096),
            ArgValue::I32(4),
            ArgValue::I32(7),
        ]),
        // The same list; the mxfp4 form's `biases` is bound and unread.
        "mxfp4_qmv_routed_bias" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            ArgValue::I32(2048),
            ArgValue::I32(1024),
            b(5),
            b(6),
            ArgValue::I32(2048),
            ArgValue::I32(4096),
            ArgValue::I32(4),
            ArgValue::I32(7),
        ]),
        // w, scales, biases, x, y, tile_expert, k, n, rows, group, bits,
        // tile_m, tile_n
        "qmm_t_routed" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(2048),
            ArgValue::I32(1024),
            ArgValue::I32(64),
            ArgValue::I32(64),
            ArgValue::I32(4),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),
        // ... without the group/bits pair
        "qmm_t_routed_fp16" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(2048),
            ArgValue::I32(1024),
            ArgValue::I32(64),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),
        // w, exponents, x, y, bias, tile_expert, k, n, rows, tile_m, tile_n
        "mxfp4_qmm_t_routed_bias" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            b(4),
            b(5),
            ArgValue::I32(2048),
            ArgValue::I32(1024),
            ArgValue::I32(64),
            ArgValue::I32(32),
            ArgValue::I32(64),
        ]),

        // ---- mlp ----
        //
        // gate, up, out, params, width, rows -- and `params` is bound here
        // where `kernels-vulkan` drops it, because WGSL declares the binding
        // and the layout comes from the declaration. See `mlp::geglu_tanh`.
        "geglu_tanh" | "geglu_tanh_strided" | "gptoss_swiglu" => one(vec![
            b(0),
            b(1),
            b(2),
            b(3),
            ArgValue::I32(2048),
            ArgValue::I32(7),
        ]),
        // gate, up, out, width, rows -- no params in this one's module.
        "silu_mul" => one(vec![
            b(0),
            b(1),
            b(2),
            ArgValue::I32(2048),
            ArgValue::I32(7),
        ]),

        // ---- rope ----
        //
        // `rotary` 64 against `head_dim` 128 deliberately: a PARTIAL rotation,
        // so `rope_grid`'s x is 32 rather than half the head. A body that
        // built its grid on the head width instead of the rotary width would
        // give the same number for the full-rotation case and a different one
        // here.
        //
        // x, position, scale, base, head_dim, rotary, width
        "neox_decode" | "neox_prop_decode" => one(vec![
            b(0),
            b(1),
            ArgValue::F32(1.0),
            ArgValue::F32(10_000.0),
            ArgValue::I32(128),
            ArgValue::I32(64),
            ArgValue::I32(1024),
        ]),
        // ..., rows
        "neox_mb" | "neox_prop_mb" => one(vec![
            b(0),
            b(1),
            ArgValue::F32(1.0),
            ArgValue::F32(10_000.0),
            ArgValue::I32(128),
            ArgValue::I32(64),
            ArgValue::I32(1024),
            ArgValue::I32(7),
        ]),
        // x, position, scale, inv_freq, head_dim, mscale, rotary, width
        "neox_freqs_decode" => one(vec![
            b(0),
            b(1),
            ArgValue::F32(1.0),
            b(2),
            ArgValue::I32(128),
            ArgValue::F32(1.0),
            ArgValue::I32(64),
            ArgValue::I32(1024),
        ]),
        // ..., rows
        "neox_freqs_mb" => one(vec![
            b(0),
            b(1),
            ArgValue::F32(1.0),
            b(2),
            ArgValue::I32(128),
            ArgValue::F32(1.0),
            ArgValue::I32(64),
            ArgValue::I32(1024),
            ArgValue::I32(7),
        ]),
        // x, position, scale, base, head_dim, row_pitch, rotary, width, rows
        "neox_strided" => one(vec![
            b(0),
            b(1),
            ArgValue::F32(1.0),
            ArgValue::F32(10_000.0),
            ArgValue::I32(128),
            ArgValue::I32(2048),
            ArgValue::I32(64),
            ArgValue::I32(1024),
            ArgValue::I32(7),
        ]),
        _ => None,
    }
}

/// The gathers, swept over all six affine points.
///
/// Sweeping rather than picking one is what makes the entrypoint check below a
/// census: all twenty-four spellings a gather can choose are resolved, and
/// mistyping one table entry is caught by the point that names it.
fn affine(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    let scaled = name.contains("scaled");
    let mb = name.contains("_mb_");
    if !name.starts_with("embed_gather") {
        return None;
    }
    Some(
        POINTS
            .iter()
            .map(|(g, bits)| {
                // w, scales, biases, id, out, hidden, [embed_scale], [rows], group, bits
                let mut v = vec![b(0), b(1), b(2), b(3), b(4), ArgValue::I32(64)];
                if scaled {
                    v.push(ArgValue::F32(1.5));
                }
                if mb {
                    v.push(ArgValue::I32(7));
                }
                v.push(ArgValue::I32(*g));
                v.push(ArgValue::I32(*bits));
                v
            })
            .collect(),
    )
}

/// The `attn` family's recipes, SYNTHESIZED from each routine's own argument
/// types rather than written out.
///
/// # Why this one is generated and the others are not
///
/// This file's header says a generic synthesizer cannot supply the recipes,
/// and for `layout` that is true: `embed_gather_4bit` refuses any
/// `(group, bits)` outside six real affine points, so a stand-in has to be a
/// fact about the kernel. `attn`'s signatures are twenty arguments long and
/// constrain exactly ONE thing — the head width, which must be a point the
/// row generates, because the body indexes a literal spelling table with it.
///
/// So the head width is stated per routine below and everything else is a
/// shape: buffers get sequential handles so a swap is visible, and the
/// remaining scalars get values that are plausible and not equal to each
/// other. Sixteen hand-written twenty-element lists would be sixteen chances
/// to transpose a pair by hand while checking that bodies do not transpose
/// pairs.
///
/// The `Env` arguments are the last three of every signature —
/// `head_dim, q_heads, rows` for the attentions, `heads` or `rows` alone for
/// the writes — and they are taken from the type walk in order, which is why
/// `HEADS` names only the head width.
fn attn(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    /// The head widths each routine's spelling table actually carries.
    ///
    /// `sdpa_paged_decode` sweeps FOUR where its row states seven: the three
    /// `_p32` tails are not reachable from a routine, for the reason
    /// `kernels-wgpu`'s
    /// `the_entrypoints_that_ignore_the_window_are_exactly_the_ones_named_p32`
    /// gives — they pin the key run's start to zero and answer a windowed
    /// caller with full attention.
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
        // Not attentions, and their first `Env` is not a head width. One
        // entry each so the walk below has something to read.
        ("split_qkv_bf16", &[0]),
        ("gate", &[0]),
        ("q_gate_split", &[0]),
        ("kv_append", &[0]),
        ("kv_append_paged", &[0]),
        ("logit_softcap", &[0]),
    ];

    let heads = HEADS.iter().find(|(n, _)| *n == name)?.1;
    let sig = kernels_wgpu::routines()
        .into_iter()
        .find(|r| r.name == name)?;

    let mut out = Vec::new();
    for &head_dim in heads {
        let mut args = Vec::new();
        let mut buffers = 0u32;
        let mut envs = 0usize;
        for (ty, prov) in sig.args {
            let value = match (ty, prov) {
                (Ty::Buf | Ty::BufMut | Ty::I32s | Ty::U32s | Ty::U8s, _) => {
                    let v = b(buffers);
                    buffers += 1;
                    v
                }
                (_, Provenance::Env) => {
                    let v = match envs {
                        // The head width, where there is one.
                        0 if head_dim != 0 => ArgValue::I32(head_dim),
                        // `q_heads` for the attentions, and the sole `Env` of
                        // the writes -- eight either way.
                        0 | 1 => ArgValue::I32(8),
                        _ => ArgValue::I32(7),
                    };
                    envs += 1;
                    v
                }
                (Ty::F32, _) => ArgValue::F32(0.125),
                (Ty::Usize, _) => ArgValue::Usize(4096),
                (Ty::U32, _) => ArgValue::U32(2048),
                // Every remaining scalar is an i32 the body only forwards.
                _ => ArgValue::I32(64),
            };
            args.push(value);
        }
        out.push(args);
    }
    Some(out)
}

/// The `quant` family's recipes, SYNTHESIZED like `attn`'s.
///
/// Thirty-one routines whose signatures run to fourteen arguments, and whose
/// only constrained values are the ones that INDEX A SPELLING TABLE: the
/// affine `group` and `bits`, and the `bm`/`bn` tile. Those are stated per
/// routine below, in the order the signature takes them; the buffers get
/// sequential handles and the remaining scalars get plausible extents.
///
/// The table was generated by reading each `pub fn`'s `Env<i32>` parameters
/// out of `src/quant.rs` in order — the comment on each line is those
/// parameter NAMES, which is what makes a wrong value visible to a reader
/// rather than something to count out on the signature.
fn quant(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    /// The `Env` values each routine takes, in signature order.
    const ENVS: &[(&str, &[i32])] = &[
        ("qmm_t", &[64, 4, 32, 32, 64]),            // group, bits, bm, bn, m
        ("qmm_t_bias", &[64, 4, 32, 32, 64]),       // group, bits, bm, bn, m
        ("qmm_t_residual", &[64, 4, 32, 32, 64]),   // group, bits, bm, bn, m
        ("qmm_t_fp16_precast", &[32, 32, 64]),      // bm, bn, m
        ("qmm_t_bias_fp16_precast", &[32, 32, 64]), // bm, bn, m
        ("qmm_t_residual_fp16_precast", &[32, 32, 64]), // bm, bn, m
        ("qmm_t_splitk", &[64, 4, 32, 64]),         // group, bits, bm, m
        ("qmm_t_splitk_f32", &[64, 4, 32, 64]),     // group, bits, bm, m
        ("qmm_t_splitk_fp16_precast", &[32, 64]),   // bm, m
        ("qmm_t_splitk_fp16_precast_f32", &[32, 64]), // bm, m
        ("qmm_t_strided", &[64, 4, 32, 64]),        // group, bits, bm, m
        ("qmm_t_strided_residual", &[64, 4, 32, 64]), // group, bits, bm, m
        ("qmm_t_strided_fp16_precast", &[32, 64]),  // bm, m
        ("qmm_t_strided_fp16_precast_residual", &[32, 64]), // bm, m
        ("qmm_splitk_reduce", &[64]),               // m
        ("qmm_splitk_reduce_f32", &[64]),           // m
        ("cast_qmm_input_bfloat16_to_float16", &[]), // no Env
        ("cast_qmm_input_strided_bfloat16_to_float16", &[7]), // rows
        ("qmv_fast", &[64, 4, 7]),                  // group, bits, vecs
        ("qmv_fast_residual", &[64, 4, 7]),         // group, bits, vecs
        ("qmv_tail", &[4, 7]),                      // bits, vecs
        ("qmv_tail_bias", &[4, 7]),                 // bits, vecs
        ("qmv_wide_strided", &[4]),                 // bits
        ("qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4", &[64]), // m
        ("qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2", &[64]), // m
        ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2", &[64]), // m
        ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1", &[64]), // m
        ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4", &[64]), // m
        ("encode_u4_bf16", &[8]),                   // groups
        ("encode_u4_f32", &[8]),                    // groups
        ("mxfp4_dequant_bf16", &[8]),               // blocks
    ];

    let envs = ENVS.iter().find(|(n, _)| *n == name)?.1;
    let sig = kernels_wgpu::routines()
        .into_iter()
        .find(|r| r.name == name)?;

    let mut args = Vec::new();
    let mut buffers = 0u32;
    let mut env_at = 0usize;
    for (ty, prov) in sig.args {
        let value = match (ty, prov) {
            (Ty::Buf | Ty::BufMut | Ty::I32s | Ty::U32s | Ty::U8s, _) => {
                let v = b(buffers);
                buffers += 1;
                v
            }
            (_, Provenance::Env) => {
                let v = ArgValue::I32(envs[env_at]);
                env_at += 1;
                v
            }
            (Ty::F32, _) => ArgValue::F32(1.0),
            (Ty::Usize, _) => ArgValue::Usize(4096),
            (Ty::U32, _) => ArgValue::U32(2048),
            _ => ArgValue::I32(2048),
        };
        args.push(value);
    }
    assert_eq!(
        env_at,
        envs.len(),
        "`{name}`'s recipe states {} `Env` values and its signature takes {env_at}",
        envs.len()
    );
    Some(vec![args])
}

/// The `ssm` family's recipes, synthesized as `attn`'s and `quant`'s are.
///
/// `gdn_core_recurrent_prefill`'s `(lanes, vrows)` must be one of the nine
/// compiled scan shapes, because the body indexes `SCAN` with it; the rest
/// are extents.
fn ssm(name: &str) -> Option<Vec<Vec<ArgValue>>> {
    /// The `Env` values each routine takes, in signature order.
    const ENVS: &[(&str, &[i32])] = &[
        ("gdn_core", &[7, 8, 64]),                       // rows, v_heads, v_dim
        ("gdn_core_slotted", &[7, 8, 64]),               // rows, v_heads, v_dim
        ("gdn_prep", &[7, 8]),                           // rows, v_heads
        ("gdn_prep_slotted", &[7, 8]),                   // rows, v_heads
        ("gdn_prep_prefill", &[7, 8]),                   // rows, v_heads
        ("gdn_core_recurrent", &[7, 8, 64]),             // rows, v_heads, v_dim
        ("gdn_core_recurrent_slotted", &[7, 8, 64]),     // rows, v_heads, v_dim
        ("gdn_core_recurrent_prefill", &[32, 4, 64, 8]), // lanes, vrows, dv, hv
    ];

    let envs = ENVS.iter().find(|(n, _)| *n == name)?.1;
    let sig = kernels_wgpu::routines()
        .into_iter()
        .find(|r| r.name == name)?;

    let mut args = Vec::new();
    let mut buffers = 0u32;
    let mut env_at = 0usize;
    for (ty, prov) in sig.args {
        let value = match (ty, prov) {
            (Ty::Buf | Ty::BufMut | Ty::I32s | Ty::U32s | Ty::U8s | Ty::F32s | Ty::F32sMut, _) => {
                let v = b(buffers);
                buffers += 1;
                v
            }
            (_, Provenance::Env) => {
                let v = ArgValue::I32(envs[env_at]);
                env_at += 1;
                v
            }
            (Ty::F32, _) => ArgValue::F32(1.0),
            (Ty::Usize, _) => ArgValue::Usize(4096),
            (Ty::U32, _) => ArgValue::U32(2048),
            _ => ArgValue::I32(2048),
        };
        args.push(value);
    }
    assert_eq!(env_at, envs.len(), "`{name}`'s recipe is stale");
    Some(vec![args])
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

/// A body passes its dispatch the arguments its signature takes, IN ORDER,
/// and may skip only a BUFFER.
///
/// Not "every buffer is bound" — the whole list, buffers and scalars alike,
/// compared value by value. `Env` arguments are the ones left out by design,
/// and that is the point of the provenance: they size the grid and the kernel
/// never reads them.
///
/// # Why this is a subsequence and not an equality
///
/// It was an equality, and `refactor-bigplan.md` §8c said in so many words
/// that `kernels-vulkan`'s weakening of the same check must not be copied
/// here: WGSL declares its bindings in source and `naga` keeps a global the
/// entrypoint never reads, so a wgpu body has nothing to skip and a skip is
/// how an output ends up bound where a params block belongs.
///
/// `moe::mxfp4_qmv_routed_bias` is the case that argument did not cover. Its
/// ROW states a `biases` slot, because a row is positional and the MXFP4 form
/// shares its template with the affine one; `moe/qmv_routed.wgsl`'s mxfp4 arm
/// is `//#if`-gated and DECLARES SIX bindings where the row states seven. A
/// binding a preprocessor arm never declares is not the same thing as a
/// global an entrypoint never reads, and only the second is what §8c was
/// about.
///
/// # What replaces the equality, which is not vulkan's plain subsequence
///
/// Two rules together, and they are as strong as the equality everywhere
/// except the one shape that has to be allowed:
///
/// * **only a `Buffer` may be skipped.** A skipped scalar has no declaration
///   to justify it — the uniform block is packed by walking this same list —
///   so a dropped `I32` is still a hard failure here.
/// * **the buffers that DO arrive must equal the module's own count**, which
///   is `every_routine_binds_a_buffer_for_every_binding_its_module_declares`,
///   asked of every dispatch in this file. So the number skipped is not the
///   author's judgement; it is the shader's declaration.
///
/// Reordering and inventing remain failures, as before.
#[test]
fn a_body_passes_the_arguments_its_signature_takes_in_order() {
    for (r, args, seen) in fired() {
        let want: Vec<ArgValue> = r
            .args
            .iter()
            .zip(&args)
            .filter(|((_, prov), _)| *prov == Provenance::Trace)
            .map(|(_, value)| *value)
            .collect();

        for (entrypoint, _, got) in seen.0.borrow().iter() {
            // A subsequence walk, keeping what was skipped so it can be
            // judged rather than merely allowed.
            let mut skipped: Vec<ArgValue> = Vec::new();
            let mut at = 0usize;
            for value in &want {
                if got.get(at) == Some(value) {
                    at += 1;
                } else {
                    skipped.push(*value);
                }
            }
            assert_eq!(
                at,
                got.len(),
                "`{}` fires `{entrypoint}` with arguments that are not its \
                 signature's in order: it passed {got:?} where the signature \
                 states {want:?}. Skipping is legal for a buffer the module \
                 does not declare; reordering and inventing are not.",
                r.name
            );
            for value in &skipped {
                assert!(
                    matches!(value, ArgValue::Buffer(_)),
                    "`{}` fires `{entrypoint}` without passing {value:?}, \
                     which is not a buffer. Only a BUFFER may be skipped, and \
                     only because its module declares no binding for it -- a \
                     scalar has no declaration to appeal to, and the uniform \
                     block is packed by walking this very list, so a dropped \
                     scalar is read as the field that lives at that offset.",
                    r.name
                );
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
        for (entrypoint, lanes, _) in seen.0.borrow().iter() {
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
        for (entrypoint, _, _) in seen.0.borrow().iter() {
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
        for (entrypoint, _, dispatched) in seen.0.borrow().iter() {
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
