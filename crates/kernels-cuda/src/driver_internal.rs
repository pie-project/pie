//! Launchers the DRIVER reaches for directly — no DSL statement, no place
//! in the planner's vocabulary, and deliberately not rows of [`crate::KERNELS`]:
//! `model`'s `kernels_table` holds that table and `dsl::cuda` to the same
//! set, and these have no statement a trace could record. The per-family
//! exhaustiveness tests classify them as `DriverInternal` for exactly this
//! reason.
//!
//! They are still LAUNCHES, and the Rust driver still has to make them —
//! which is what this second table is for. Same [`KernelSig`] rows, same
//! `abi::emit_c_shim` proof, same generated bindings; the only difference
//! is which invariant the table answers to. A row joins here when a live
//! seam or the executor needs a launcher the DSL surface correctly lacks.

use kernels::kernel;
use kernels::{KernelSig, Source, operands};

#[rustfmt::skip]
pub static DRIVER_KERNELS: &[KernelSig] = &[
    // The envelope tier: seeded empty at materialize (`KvCacheDeviceOps`),
    // recomputed after eviction, merged after a write. The seed writes
    // +inf/-inf bf16 so the first real merge tightens from the identity.
    kernel!(envelope_seed "layout::launch_envelope_seed_empty_bf16",
        operands = operands![
            env_min: U16sMut, env_max: U16sMut,
            num_pages: I32, num_kv_heads: I32, head_dim: I32, stream: Stream,
        ]),
    kernel!(envelope_recompute "layout::launch_envelope_recompute_bf16",
        operands = operands![
            k_pages: U16s, page_live_lens: I32s,
            env_min: U16sMut, env_max: U16sMut,
            num_pages: I32, page_size: I32, num_kv_heads: I32, head_dim: I32,
            stream: Stream,
        ]),
    kernel!(envelope_merge_written "layout::launch_envelope_merge_written_bf16",
        operands = operands![
            k_curr: U16s, w_page: U32s, w_off: U32s, row_valid: U8s | null,
            env_min: U16sMut, env_max: U16sMut,
            num_tokens: I32, num_kv_heads: I32, head_dim: I32, stream: Stream,
        ]),
    // The QKV split the generated bodies call ~390 times — the loud case
    // the attn exhaustiveness test names.
    kernel!(split_qkv "attn::split_qkv_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_out: BufMut <- Source::Out(1),
            v_out: BufMut <- Source::Out(2),
            n_tokens: I32 <- Source::Rows,
            // The two widths come off what is WRITTEN, not off the packed
            // operand: a `[N, q + 2*kv]` row cannot say where the cut
            // falls, and both results can.
            q_dim: I32 <- Source::OutWidth(0),
            kv_dim: I32 <- Source::OutWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The DEVICE-WINDOW twin. Its own stated symbol, so there is no
    // ambiguity for a binder to resolve — the peel's tail region states
    // this one and the plain body states the other. `CtxNonZero` on the
    // window is the arm's null check: a fire that published no peel
    // window is not one this launcher can run for.
    kernel!(split_qkv_devwin "attn::split_qkv_bf16_devwin",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_out: BufMut <- Source::Out(1),
            v_out: BufMut <- Source::Out(2),
            win_d: U32s <- Source::CtxNonZero("peel_window"),
            n_max: I32 <- Source::Ctx("rows_total"),
            q_dim: I32 <- Source::OutWidth(0),
            kv_dim: I32 <- Source::OutWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The page-mask packers `FirePageMask` fires.
    kernel!(pack_dense_mask "attn::pack_dense_mask",
        operands = operands![
            kvm_dense: U8s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            packed: U8sMut, b: I32, p_page: I32, stream: Stream,
        ]),
    kernel!(pack_structured_mask "attn::pack_structured_mask",
        operands = operands![
            positions: U32s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            masks: StructuredMasks, packed: U8sMut, b: I32, stream: Stream,
        ]),
    // Beam-repair cell moves, per layer, disjoint spans by contract.
    kernel!(copy_kv_cells "attn::copy_kv_cells_bf16",
        operands = operands![
            layer: KvCacheLayerView, dst_page: U32s, dst_off: U32s,
            src_page: U32s, src_off: U32s, n: I32, stream: Stream,
        ]),
    // ── the ones a SEMANTIC op picks ──────────────────────────────
    //
    // No trace records a Launch naming these: the statement carries an
    // `OpKind`, and `lower()` reads the CUDA kernel off it. So the DSL
    // surface correctly lacks them and they are rows here, which is
    // what `every_lowered_kernel_has_a_bridge_row` found on its first
    // run and finds again whenever a semantic kind gains a reading.
    //
    // `norm::rmsnorm_bf16` stood here too and has left: the fan-out
    // pair is stated by `dsl::cuda::rmsnorm` now and its row moved to
    // `norm.rs`, where a text names it. That is the exit this table
    // wants for every row in this block — a row leaves when a statement
    // learns to say it.
    //
    // `gemm::act_x_w` also stood here — the quantized dispatch entry,
    // whose `WeightView` BY VALUE was the operand the handoff predicted
    // would be gemm's friction. It is gone, and the prediction was
    // answered rather than paid: the representation axis is FOUR named
    // rows now (`gemm.rs`'s tensor/channel/grouped/mxfp4 scaled entry
    // points), each one a symbol a statement chose, so nothing crosses
    // this ABI carrying a descriptor for the launcher to route on. What
    // the lowering still spells `gemm::act_x_w` is the DENSE matmul,
    // and the executor binds it to `gemm::act_x_wt_bf16` — which
    // `gemm.hpp` defines as `act_x_w` with `WeightView::raw(W, BF16)`,
    // the one view the dense arm ever built.
    // The first launch of every fire, and the row that forced
    // `WeightNamed`: a vocab table is not something a trace produces, so
    // the embedding's weight is only ever the statement's own NAME and
    // never a slot in the argument run.
    kernel!(embed "layout::embed_bf16",
        operands = operands![
            token_ids: I32s <- Source::Ctx("token_ids"),
            weight: Buf <- Source::WeightNamed,
            y: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            hidden: I32 <- Source::OutWidth(0),
            vocab: I32 <- Source::Ctx("vocab"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // In place over the value it biases — one operand, one result, the
    // same bytes — so `out` binds from `Out(0)` and the staging comes off
    // the pair. The bias is the statement's named weight, like the
    // embedding's table.
    kernel!(add_bias "norm::add_bias_bf16", in_place = &[(0, 0)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            bias: Buf <- Source::WeightNamed,
            num_rows: I32 <- Source::Rows,
            dim: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // qwen3_5's four, all read off a semantic kind the same way. They
    // arrived together because the family's declaration stopped naming
    // kernels and started naming what it MEANS — `GdnPrep`, `SplitQGate`,
    // `SigmoidGateMul`, `RmsnormGated` — which is the direction, and
    // leaves the reading to `lower()`.
    //
    // The post-conv prep, fused: q/k split and L2-normalized, v widened
    // to fp32, and g/beta gated — the three launches that used to sit
    // between the conv and the recurrent step. Its five fp32 outputs are
    // exactly the step's first five inputs, which is the shape of it.
    kernel!(gdn_post_conv_prep "ssm::qwen_gdn_post_conv_prep_bf16",
        operands = operands![
            qkv_post: Buf, a: Buf, b: Buf, a_log: Buf, dt_bias: Buf,
            q_norm_kh: F32sMut, k_norm_kh: F32sMut, v_fp32: F32sMut,
            g_log_out: F32sMut, beta_out: F32sMut,
            n: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, conv_dim: I32,
            stream: Stream,
        ]),
    // Full attention's q_proj packs the query and the per-token output
    // gate PER HEAD — `[N, heads, 2*head_dim]`, query first — so this is
    // strided by head, not a halves cut like `split_gate_up`. Three shape
    // arguments rather than one width, because the stride IS the layout.
    kernel!(split_q_gate "layout::split_q_gate_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            gate_out: BufMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            // Off the QUERY half, not the packed operand: `packed` is
            // `[N, heads, 2*head_dim]` and only the query's half of it
            // lands here, so the head count comes from what is written.
            num_heads: I32 <- Source::OutWidthOver(0, "head_dim"),
            head_dim: I32 <- Source::Ctx("head_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // That gate applied: `a' = a * σ(g)`, IN PLACE on operand 0 — the
    // header spells `x` "bf16, in-place" in as many words.
    kernel!(sigmoid_gate_inplace "mlp::sigmoid_gate_inplace_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut, gate: Buf, num_elements: I32, stream: Stream,
        ]),
    // The gated norm with an FP32 `x`: the GDN recurrent step lands in
    // fp32, so this reads it there and the separate conversion launch
    // goes away. `x` and `weight` hold fp32 and are still `Buf` — the
    // header spells them `const void*`, and this table describes the
    // DECLARATION, not the contents. The shim initialises a function
    // pointer, so the spelling is what has to agree.
    kernel!(rmsnorm_gated_fp32_in "norm::rmsnorm_gated_fp32_in_bf16",
        operands = operands![
            x: Buf, gate: Buf, weight: Buf, y: BufMut,
            num_rows: I32, hidden: I32, eps: F32, stream: Stream,
        ]),
    // The qwen3_vl vision TOWER, bridged at tower granularity — one row
    // that is a whole subgraph, the flashinfer-dispatch precedent (see
    // the retirement wiki's VL judgment). The wrapper rebuilds the C++
    // weights struct from the flat tables; the walk and its host prep
    // (bilinear pos-embed interp, 2-D rope ids, spatial-merge reorder,
    // the f32→bf16 pixel cast) stay `qwen3_vl_tower.cu`'s. The pixel/
    // grid/anchor operands and the pointer tables are HOST pointers —
    // the step hands them over host-side, the C++ shape. `whole`: the
    // tower addresses rows through per-image anchor offsets, and a row
    // window would encode the wrong images.
    kernel!(qwen3vl_tower_scatter "vision::qwen3vl_scatter", whole = true,
        operands = operands![
            patch_w: Buf, patch_b: Buf | null, pos_embed: Buf,
            block_w: BufArray, depth: I32,
            merger_w: BufArray,
            deepstack_w: BufArray, deepstack_layers: I32s,
            hidden: I32, heads: I32, intermediate: I32, patch_size: I32,
            temporal_patch: I32, merge_size: I32, in_channels: I32,
            out_hidden: I32, num_pos_embed: I32, ln_eps: F32,
            rope_theta: F32,
            pixels_h: F32s, pixel_byte_indptr_h: U32s, grids_h: U32s,
            anchor_rows_h: U32s, num_images: I32,
            hidden_rows: BufMut, n_rows: I32,
            deepstack_scratch: BufMut | null, num_deep: I32,
            blas: CublasHandle, stream: Stream,
        ]),
    // gemma-4's STANDALONE towers — the encode-ABI pair (host pixels /
    // log-mel in, HOST bf16 embedding rows out, anchor-segmented CSR).
    // Layer tables are `Ty::Bufs` at stride 41 (vision) / 62 (audio);
    // the field orders live in `vision/gemma4_towers_c.hpp`. The output
    // operands are HOST buffers — `PieEncodeDesc`'s own shape.
    kernel!(gemma4_vision_encode "vision::gemma4_vision_encode", whole = true,
        operands = operands![
            patch_w: Buf, pos_table: Buf, embed_proj: Buf,
            layer_w: BufArray, depth: I32,
            hidden: I32, heads: I32, intermediate: I32,
            pos_table_size: I32, text_hidden: I32, pool_kernel: I32,
            eps: F32, theta: F32,
            pixels_h: F32s, pixel_byte_indptr_h: U32s,
            patch_positions_h: U32s, anchor_rows_h: U32s, num_images: I32,
            output_rows_h: U16sMut, output_bytes: Usize,
            output_row_indptr_h: U32sMut, stream: Stream,
        ]),
    kernel!(gemma4_audio_encode "vision::gemma4_audio_encode", whole = true,
        operands = operands![
            sscp0_conv: Buf, sscp0_norm: Buf, sscp1_conv: Buf,
            sscp1_norm: Buf, sscp_input_proj: Buf,
            output_proj_w: Buf, output_proj_b: Buf, embed_proj: Buf,
            layer_w: BufArray, depth: I32,
            hidden: I32, heads: I32, conv_kernel: I32, n_mel: I32,
            sscp_ch0: I32, sscp_ch1: I32, out_proj_dims: I32,
            text_hidden: I32, chunk_size: I32, context_left: I32,
            context_right: I32, logit_cap: F32, residual_weight: F32,
            eps: F32,
            features_h: F32s, feature_byte_indptr_h: U32s,
            anchor_rows_h: U32s, num_clips: I32,
            output_rows_h: U16sMut, output_bytes: Usize,
            output_row_indptr_h: U32sMut, stream: Stream,
        ]),
];
