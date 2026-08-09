//! `pie_cuda_encode`: the multimodal towers, run outside a fire.
//!
//! Gemma-4's vision and audio encoders write rows the next fire reads as
//! embeddings. It is its own entry point because it is its own pass — no
//! KV, no sampling, no logits.

use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR,
    PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_OK,
    PIE_STATUS_UNSUPPORTED,
    PieCompletion,
    PieDriver,
    PieEncodeDesc,
};
use super::{checked, guard};
use super::state::{LoadedModel, Shell, shell};

/// Awaits: the MULTIMODAL encoders — image/audio features to embedding
/// rows (the vision/audio towers, which stayed hand-written C++). The
/// plan once mislabeled this as the Sampling-IR path; the desc's fields
/// (`image_pixels`, `audio_features`, `output_rows`) say what it is. A
/// text-only shell refuses it honestly.
/// The audio half of the encode arm: `bind_gemma4_audio`'s name map as
/// the stride-62 table (`vision/gemma4_towers_c.hpp`'s layout), the
/// media row width from the embed projection's own bytes, and the
/// `PieEncodeDesc` audio slices passed straight through.
fn encode_gemma4_audio_arm(
    model: &LoadedModel,
    desc: &PieEncodeDesc,
    out_ptr: *mut std::ffi::c_void,
    out_bytes: usize,
    indptr_ptr: *mut u32,
) -> i32 {
    let Some(ac) = model.deployment.towers.audio.as_ref() else {
        eprintln!("[driver-cuda] encode: this deployment carries no audio tower");
        return PIE_STATUS_UNSUPPORTED;
    };
    let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
        model.weights.get(n).map(|b| b.ptr.cast_const()).ok_or_else(|| {
            eprintln!("[driver-cuda] encode: missing audio weight {n}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let opt = |n: String| -> *const std::ffi::c_void {
        model.weights.get(&n).map_or(core::ptr::null(), |b| b.ptr.cast_const())
    };
    let ap = "model.audio_tower";
    let g = |n: &str| need(&format!("{ap}.{n}"));
    let (sscp0_conv, sscp0_norm, sscp1_conv, sscp1_norm, sscp_proj, out_w, out_b, embed) = match (
        g("subsample_conv_projection.layer0.conv.weight"),
        g("subsample_conv_projection.layer0.norm.weight"),
        g("subsample_conv_projection.layer1.conv.weight"),
        g("subsample_conv_projection.layer1.norm.weight"),
        g("subsample_conv_projection.input_proj_linear.weight"),
        g("output_proj.weight"),
        g("output_proj.bias"),
        need("model.embed_audio.embedding_projection.weight"),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f), Ok(gp), Ok(h)) => (a, b, c, d, e, f, gp, h),
        _ => return PIE_STATUS_UNSUPPORTED,
    };
    let depth = ac.layers as usize;
    let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(depth * 62);
    for l in 0..depth {
        let lp = format!("{ap}.layers.{l}");
        let clip = |base: String, table: &mut Vec<*const std::ffi::c_void>| -> Result<(), i32> {
            table.push(need(&format!("{base}.linear.weight"))?);
            for m in ["input_min", "input_max", "output_min", "output_max"] {
                table.push(opt(format!("{base}.{m}")));
            }
            Ok(())
        };
        let ffn = |base: String, table: &mut Vec<*const std::ffi::c_void>| -> Result<(), i32> {
            table.push(need(&format!("{base}.pre_layer_norm.weight"))?);
            table.push(need(&format!("{base}.post_layer_norm.weight"))?);
            clip(format!("{base}.ffw_layer_1"), table)?;
            clip(format!("{base}.ffw_layer_2"), table)?;
            Ok(())
        };
        let r: Result<(), i32> = (|| {
            ffn(format!("{lp}.feed_forward1"), &mut table)?;
            ffn(format!("{lp}.feed_forward2"), &mut table)?;
            table.push(need(&format!("{lp}.norm_pre_attn.weight"))?);
            table.push(need(&format!("{lp}.norm_post_attn.weight"))?);
            clip(format!("{lp}.self_attn.q_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.k_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.v_proj"), &mut table)?;
            clip(format!("{lp}.self_attn.post"), &mut table)?;
            table.push(need(&format!("{lp}.self_attn.relative_k_proj.weight"))?);
            table.push(need(&format!("{lp}.self_attn.per_dim_scale"))?);
            table.push(need(&format!("{lp}.lconv1d.pre_layer_norm.weight"))?);
            table.push(need(&format!("{lp}.lconv1d.conv_norm.weight"))?);
            clip(format!("{lp}.lconv1d.linear_start"), &mut table)?;
            clip(format!("{lp}.lconv1d.linear_end"), &mut table)?;
            table.push(need(&format!("{lp}.lconv1d.depthwise_conv1d.weight"))?);
            table.push(need(&format!("{lp}.norm_out.weight"))?);
            Ok(())
        })();
        if let Err(e) = r {
            return e;
        }
    }
    let text_hidden = model
        .weights
        .get("model.embed_audio.embedding_projection.weight")
        .map_or(0, |b| b.bytes / (ac.output_dims.max(1) as usize * 2));
    let Ok(stream) = crate::device::OwnedStream::new(0) else {
        return PIE_STATUS_DRIVER_ERROR;
    };
    unsafe {
        crate::bind::abi::ffi::pie_k_vision_gemma4_audio_encode(
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_proj,
            out_w,
            out_b,
            embed,
            table.as_ptr(),
            ac.layers as i32,
            ac.hidden as i32,
            ac.heads as i32,
            ac.conv_kernel as i32,
            ac.feature_size as i32,
            ac.subsample_channels_0 as i32,
            ac.subsample_channels_1 as i32,
            ac.output_dims as i32,
            i32::try_from(text_hidden).unwrap_or(0),
            ac.chunk_size as i32,
            ac.context_left as i32,
            ac.context_right as i32,
            ac.logit_cap,
            ac.residual_weight,
            ac.norm_eps,
            desc.audio_features.ptr.cast(),
            desc.audio_feature_indptr.ptr,
            desc.audio_anchor_rows.ptr,
            i32::try_from(desc.audio_anchor_rows.len).unwrap_or(0),
            out_ptr.cast(),
            out_bytes,
            indptr_ptr,
            stream.as_ref().as_raw().cast(),
        );
    }
    if stream.as_ref().synchronize().is_err() {
        return PIE_STATUS_DRIVER_ERROR;
    }
    PIE_STATUS_OK
}

/// The MULTIMODAL encode: image/audio media in, embedding rows out —
/// the towers behind `vision::gemma4_*_encode`. One media kind per call
/// today; mixed batches await the offset plumbing.
pub fn pie_cuda_encode(
    driver: *mut PieDriver,
    encode: *const PieEncodeDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_encode", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            encode,
            |d| unsafe { driver_api::local::validate_encode_desc(d) },
            "encode",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        let Some(model) = state.model.as_ref() else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let num_images = desc.image_anchor_rows.len;
        let num_clips = desc.audio_anchor_rows.len;
        if num_images == 0 && num_clips == 0 {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if desc.output_row_indptr.len < num_images + num_clips + 1 {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let notify_done = |state: &Shell| {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            if let Some(notify) = state.notify {
                unsafe { notify(state.notify_ctx, completion.wait_id, completion.target_epoch) };
            }
        };
        if num_images == 0 {
            // Audio only: the helper writes the whole CSR itself.
            let st = encode_gemma4_audio_arm(
                model,
                desc,
                desc.output_rows.ptr.cast(),
                desc.output_rows.len,
                desc.output_row_indptr.ptr,
            );
            if st != PIE_STATUS_OK {
                return st;
            }
            notify_done(state);
            return PIE_STATUS_OK;
        }
        let Some(vc) = model.deployment.towers.vision.as_ref() else {
            eprintln!("[driver-cuda] encode: this deployment carries no vision tower");
            return PIE_STATUS_UNSUPPORTED;
        };
        // The vision table, `vision/gemma4_towers_c.hpp`'s stride-41 layout,
        // built per call from the loaded weights — name lookups, no stored
        // pointers. The binder mapping is `bind_gemma4_vision`'s.
        let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
            model.weights.get(n).map(|b| b.ptr.cast_const()).ok_or_else(|| {
                eprintln!("[driver-cuda] encode: missing vision weight {n}");
                PIE_STATUS_UNSUPPORTED
            })
        };
        let opt = |n: String| -> *const std::ffi::c_void {
            model.weights.get(&n).map_or(core::ptr::null(), |b| b.ptr.cast_const())
        };
        let vp = "model.vision_tower";
        let patch_w = match need(&format!("{vp}.patch_embedder.input_proj.weight")) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let pos_table = match need(&format!("{vp}.patch_embedder.position_embedding_table")) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let embed_proj = match need("model.embed_vision.embedding_projection.weight") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let depth = vc.layers as usize;
        let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(depth * 41);
        for l in 0..depth {
            let lp = format!("{vp}.encoder.layers.{l}");
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                match need(&format!("{lp}.{norm}.weight")) {
                    Ok(p) => table.push(p),
                    Err(e) => return e,
                }
            }
            for norm in ["self_attn.q_norm", "self_attn.k_norm"] {
                match need(&format!("{lp}.{norm}.weight")) {
                    Ok(p) => table.push(p),
                    Err(e) => return e,
                }
            }
            for clip in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                match need(&format!("{lp}.{clip}.linear.weight")) {
                    Ok(p) => table.push(p),
                    Err(e) => return e,
                }
                for m in ["input_min", "input_max", "output_min", "output_max"] {
                    table.push(opt(format!("{lp}.{clip}.{m}")));
                }
            }
        }
        // pos_table is `[2, S, hidden]` bf16 — S from the buffer itself; the
        // media row width from the projection (`[text_hidden, hidden]`).
        let hidden = vc.hidden.max(1) as usize;
        let pos_table_size = model
            .weights
            .get(&format!("{vp}.patch_embedder.position_embedding_table"))
            .map_or(0, |b| b.bytes / (2 * hidden * 2));
        let text_hidden = model
            .weights
            .get("model.embed_vision.embedding_projection.weight")
            .map_or(0, |b| b.bytes / (hidden * 2));

        let Ok(stream) = crate::device::OwnedStream::new(0) else {
            return PIE_STATUS_DRIVER_ERROR;
        };
        let mut vis_bounds = vec![0u32; num_images + 1];
        unsafe {
            crate::bind::abi::ffi::pie_k_vision_gemma4_vision_encode(
                patch_w,
                pos_table,
                embed_proj,
                table.as_ptr(),
                vc.layers as i32,
                vc.hidden as i32,
                vc.heads as i32,
                vc.intermediate as i32,
                i32::try_from(pos_table_size).unwrap_or(0),
                i32::try_from(text_hidden).unwrap_or(0),
                vc.pooling_kernel as i32,
                vc.norm_eps,
                vc.rope_theta,
                desc.image_pixels.ptr.cast(),
                desc.image_pixel_indptr.ptr,
                desc.image_patch_positions.ptr,
                desc.image_anchor_rows.ptr,
                i32::try_from(num_images).unwrap_or(0),
                desc.output_rows.ptr.cast(),
                desc.output_rows.len,
                vis_bounds.as_mut_ptr(),
                stream.as_ref().as_raw().cast(),
            );
        }
        if stream.as_ref().synchronize().is_err() {
            return PIE_STATUS_DRIVER_ERROR;
        }
        // Compose the shared CSR the C++ `Context::encode` writes: the
        // vision segment's boundaries verbatim, then the audio segment's
        // shifted by the vision row count.
        let indptr = desc.output_row_indptr.ptr;
        unsafe {
            for (i, b) in vis_bounds.iter().enumerate() {
                *indptr.add(i) = *b;
            }
        }
        if num_clips > 0 {
            let row_offset = *vis_bounds.last().unwrap_or(&0) as usize;
            let consumed = row_offset * text_hidden * 2;
            if consumed > desc.output_rows.len {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            let mut audio_bounds = vec![0u32; num_clips + 1];
            let st = encode_gemma4_audio_arm(
                model,
                desc,
                unsafe { desc.output_rows.ptr.add(consumed) }.cast(),
                desc.output_rows.len - consumed,
                audio_bounds.as_mut_ptr(),
            );
            if st != PIE_STATUS_OK {
                return st;
            }
            unsafe {
                for c in 0..num_clips {
                    *indptr.add(num_images + 1 + c) =
                        u32::try_from(row_offset).unwrap_or(u32::MAX) + audio_bounds[c + 1];
                }
            }
        }
        notify_done(state);
        PIE_STATUS_OK
    })
}

