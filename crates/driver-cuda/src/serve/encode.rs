//! `pie_cuda_encode`: the multimodal towers, run outside a fire.
//!
//! A deployment's vision and audio encoders write rows the next fire reads
//! as embeddings. It is its own entry point because it is its own pass — no
//! KV, no sampling, no logits.
//!
//! The tensor NAMES and their launcher order are
//! [`model::shared::tower_names`]'s, not this file's: what is here is the
//! resolution of a name to a device pointer and the call.

use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_OK, PIE_STATUS_UNSUPPORTED,
};
use model::shared::tower_names::{Slot, VISION_SLOTS_PER_LAYER, vision_head, vision_layers};

use super::state::{LoadedModel, Shell};
use super::guard;

/// Awaits: the MULTIMODAL encoders — image/audio features to embedding
/// rows (the vision/audio towers, which stayed hand-written C++). The
/// plan once mislabeled this as the Sampling-IR path; the desc's fields
/// (`image_pixels`, `audio_features`, `output_rows`) say what it is. A
/// text-only shell refuses it honestly.
/// The audio half of the encode arm: the tower's name table as the
/// stride-62 pointer list the launcher indexes, the media row width from the
/// embed projection's own bytes, and the `PieEncodeDesc` audio slices passed
/// straight through.
///
/// THE NAMES ARE NOT THIS CRATE'S. `model::shared::tower_names` states them,
/// in launcher order, for the reason its module doc gives: a tower's tensors
/// are named by the checkpoint and consumed by a launcher, and a backend is
/// neither. What stood here spelled some fifty paths inline.
fn encode_audio_arm(
    model: &LoadedModel,
    desc: &driver_api::MediaEncodePlan,
    out_ptr: *mut std::ffi::c_void,
    out_bytes: usize,
    indptr_ptr: *mut u32,
) -> i32 {
    use model::shared::tower_names::{AUDIO_SLOTS_PER_LAYER, Slot, audio_head, audio_layers};

    let Some(ac) = model.deployment.towers.audio.as_ref() else {
        eprintln!("[driver-cuda] encode: this deployment carries no audio tower");
        return PIE_STATUS_UNSUPPORTED;
    };
    let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
        model
            .weights
            .get(n)
            .map(|b| b.ptr.cast_const())
            .ok_or_else(|| {
                eprintln!("[driver-cuda] encode: missing audio weight {n}");
                PIE_STATUS_UNSUPPORTED
            })
    };
    let opt = |n: &str| -> *const std::ffi::c_void {
        model
            .weights
            .get(n)
            .map_or(core::ptr::null(), |b| b.ptr.cast_const())
    };
    // A slot says which of the two it is; that is the whole reason the list
    // is `Slot` and not `String`.
    let bind = |slot: &Slot| -> Result<*const std::ffi::c_void, i32> {
        match slot {
            Slot::Required(n) => need(n),
            Slot::Optional(n) => Ok(opt(n)),
        }
    };
    let ap = "model.audio_tower";
    let embed = "model.embed_audio.embedding_projection.weight";
    let head = audio_head(ap, embed);
    let mut heads = Vec::with_capacity(head.len());
    for s in &head {
        match bind(s) {
            Ok(p) => heads.push(p),
            Err(e) => return e,
        }
    }
    let [
        sscp0_conv,
        sscp0_norm,
        sscp1_conv,
        sscp1_norm,
        sscp_proj,
        out_w,
        out_b,
        embed_p,
    ] = heads[..]
    else {
        return PIE_STATUS_UNSUPPORTED;
    };
    let slots = audio_layers(ap, ac.layers);
    let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(slots.len());
    for s in &slots {
        match bind(s) {
            Ok(p) => table.push(p),
            Err(e) => return e,
        }
    }
    debug_assert_eq!(table.len(), ac.layers as usize * AUDIO_SLOTS_PER_LAYER);
    let text_hidden = model
        .weights
        .get(embed)
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
            embed_p,
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
            desc.audio_features.as_ptr().cast(),
            desc.audio_feature_indptr.as_ptr(),
            desc.audio_anchor_rows.as_ptr(),
            i32::try_from(desc.audio_anchor_rows.len()).unwrap_or(0),
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
impl Shell {
    /// Encode media into the model's embedding space.
    ///
    /// # Errors
    ///
    /// No encode tower for this model, or a device failure.
    pub fn encode(
        &mut self,
        encode: &mut driver_api::MediaEncodePlan,
        completion: driver_api::completion::CompletionTarget,
    ) -> Result<(), i32> {
        guard("encode", Err(PIE_STATUS_DRIVER_ERROR), move || {
        let state = self;
        // Most of `validate_encode_desc` was NOT about the C shape: the
        // plane counts, the `f32` alignment, the exact partitions. All of it
        // is `MediaEncodePlan::validate`.
        if let Err(why) = encode.validate() {
            eprintln!("[driver-cuda] encode: {why}");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        // The out-params, taken as raw pointers ONCE. The encode towers
        // write through them from the device side, and holding them as raw
        // is what lets the read side stay a shared borrow of the same plan.
        let out_ptr: *mut u8 = encode.output_rows.as_mut_ptr();
        let out_bytes = encode.output_rows.len();
        let out_indptr: *mut u32 = encode.output_row_indptr.as_mut_ptr();
        let desc = &*encode;
        let Some(model) = state.model.as_ref() else {
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        };
        let num_images = desc.image_anchor_rows.len();
        let num_clips = desc.audio_anchor_rows.len();
        if num_images == 0 && num_clips == 0 {
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        if desc.output_row_indptr.len() < num_images + num_clips + 1 {
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        let notify_done = |state: &Shell| {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            state
                .broker
                .notify(completion.wait_id, completion.target_epoch);
        };
        if num_images == 0 {
            // Audio only: the helper writes the whole CSR itself.
            let st = encode_audio_arm(model, desc, out_ptr.cast(), out_bytes, out_indptr);
            if st != PIE_STATUS_OK {
                return Err(st);
            }
            notify_done(state);
            return Err(PIE_STATUS_OK);
        }
        let Some(vc) = model.deployment.towers.vision.as_ref() else {
            eprintln!("[driver-cuda] encode: this deployment carries no vision tower");
            return Err(PIE_STATUS_UNSUPPORTED);
        };
        // The vision table, in the stride-41 layout the launcher indexes,
        // built per call from the loaded weights — name lookups, no stored
        // pointers. The NAMES and their order are
        // `model::shared::tower_names`'s; this resolves them.
        let need = |n: &str| -> Result<*const std::ffi::c_void, i32> {
            model
                .weights
                .get(n)
                .map(|b| b.ptr.cast_const())
                .ok_or_else(|| {
                    eprintln!("[driver-cuda] encode: missing vision weight {n}");
                    PIE_STATUS_UNSUPPORTED
                })
        };
        let opt = |n: &str| -> *const std::ffi::c_void {
            model
                .weights
                .get(n)
                .map_or(core::ptr::null(), |b| b.ptr.cast_const())
        };
        let bind = |slot: &Slot| -> Result<*const std::ffi::c_void, i32> {
            match slot {
                Slot::Required(n) => need(n),
                Slot::Optional(n) => Ok(opt(n)),
            }
        };
        let vp = "model.vision_tower";
        let vembed = "model.embed_vision.embedding_projection.weight";
        let head = vision_head(vp, vembed);
        let mut heads = Vec::with_capacity(head.len());
        for s in &head {
            match bind(s) {
                Ok(p) => heads.push(p),
                Err(e) => return Err(e),
            }
        }
        let [patch_w, pos_table, embed_proj] = heads[..] else {
            return Err(PIE_STATUS_UNSUPPORTED);
        };
        let slots = vision_layers(vp, vc.layers);
        let mut table: Vec<*const std::ffi::c_void> = Vec::with_capacity(slots.len());
        for s in &slots {
            match bind(s) {
                Ok(p) => table.push(p),
                Err(e) => return Err(e),
            }
        }
        debug_assert_eq!(table.len(), vc.layers as usize * VISION_SLOTS_PER_LAYER);
        // pos_table is `[2, S, hidden]` bf16 — S from the buffer itself; the
        // media row width from the projection (`[text_hidden, hidden]`).
        let hidden = vc.hidden.max(1) as usize;
        let pos_table_size = model
            .weights
            .get(head[1].name())
            .map_or(0, |b| b.bytes / (2 * hidden * 2));
        let text_hidden = model
            .weights
            .get(vembed)
            .map_or(0, |b| b.bytes / (hidden * 2));

        let Ok(stream) = crate::device::OwnedStream::new(0) else {
            return Err(PIE_STATUS_DRIVER_ERROR);
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
                desc.image_pixels.as_ptr().cast(),
                desc.image_pixel_indptr.as_ptr(),
                desc.image_patch_positions.as_ptr(),
                desc.image_anchor_rows.as_ptr(),
                i32::try_from(num_images).unwrap_or(0),
                out_ptr.cast(),
                out_bytes,
                vis_bounds.as_mut_ptr(),
                stream.as_ref().as_raw().cast(),
            );
        }
        if stream.as_ref().synchronize().is_err() {
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
        // Compose the shared CSR the C++ `Context::encode` writes: the
        // vision segment's boundaries verbatim, then the audio segment's
        // shifted by the vision row count.
        let indptr = out_indptr;
        unsafe {
            for (i, b) in vis_bounds.iter().enumerate() {
                *indptr.add(i) = *b;
            }
        }
        if num_clips > 0 {
            let row_offset = *vis_bounds.last().unwrap_or(&0) as usize;
            let consumed = row_offset * text_hidden * 2;
            if consumed > out_bytes {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            let mut audio_bounds = vec![0u32; num_clips + 1];
            let st = encode_audio_arm(
                model,
                desc,
                unsafe { out_ptr.add(consumed) }.cast(),
                out_bytes - consumed,
                audio_bounds.as_mut_ptr(),
            );
            if st != PIE_STATUS_OK {
                return Err(st);
            }
            unsafe {
                for c in 0..num_clips {
                    *indptr.add(num_images + 1 + c) =
                        u32::try_from(row_offset).unwrap_or(u32::MAX) + audio_bounds[c + 1];
                }
            }
        }
        notify_done(state);
        Ok(())
    })
    }
}
