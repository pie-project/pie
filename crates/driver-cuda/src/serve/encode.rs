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

use super::guard;
use super::state::{LoadedModel, Shell};
// ALIASED, and `tests/no_family_names.rs` is the reason. Both tower walks are
// Rust now and this file calls them; spelled out, each call would be a line
// naming a family, and the budget this file sits on is a CEILING that only
// ratchets down. The names are still here — once each, at the imports, where
// a reader looks to find out what `vis_tower` and `aud_tower` are — which is
// the same shape the budget's own comment argues for: what is left is the
// name of the thing being called, not a routing decision made on it.
//
// The count did not move when the audio launcher went. It was two before —
// one import and one `pie_k_vision_gemma4_audio_encode` — and it is two now,
// one import each. The difference is that neither is a C++ symbol.
use crate::tower::gemma4_audio as aud_tower;
use crate::tower::gemma4_vision as vis_tower;

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
    features: &[u8],
    feature_indptr: &[u32],
    anchor_rows: &[u32],
    out: &mut [u8],
    indptr: &mut [u32],
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
    // THE STRUCT REBUILD `gemma4_towers_c.cpp` DID, as a call. That file
    // existed to turn the two lists below into the C++ `AudioRawWeights` the
    // walk consumed; the walk is Rust and consumes them directly.
    //
    // Two arrays and not twenty-three positional arguments: `from_flat` took
    // eight head pointers and nine dimensions as `[_; 8]` and `[_; 9]` for
    // the reason a `dim3` pair is written as a struct — a transposed
    // `(sscp_ch0, sscp_ch1)` in a positional list is invisible at the call
    // site and silent at run time.
    let heads_flat = [
        sscp0_conv,
        sscp0_norm,
        sscp1_conv,
        sscp1_norm,
        sscp_proj,
        out_w,
        out_b,
        embed_p,
    ];
    let dims = [
        ac.hidden as i32,
        ac.heads as i32,
        ac.conv_kernel as i32,
        ac.feature_size as i32,
        ac.subsample_channels_0 as i32,
        ac.subsample_channels_1 as i32,
        ac.output_dims as i32,
        i32::try_from(text_hidden).unwrap_or(0),
        ac.chunk_size as i32,
    ];
    let built = aud_tower::Weights::from_flat(
        heads_flat,
        &table,
        ac.layers as usize,
        dims,
        ac.logit_cap,
        ac.residual_weight,
        ac.norm_eps,
    );
    let weights = match built {
        Ok(w) => w.with_context(ac.context_left as i32, ac.context_right as i32),
        // `i32::from(Error)` prints the refusal itself and maps it to a
        // `PIE_STATUS_*`; an `eprintln!` here would say it twice.
        Err(why) => return i32::from(why),
    };
    // The CLIP COUNT is the anchor table's length, which is what the C++
    // `Gemma4AudioInputs::num_clips` carried. `output_row_indptr` is the
    // whole plan's and may be longer — the vision arm's rows come first when
    // both are present — so the tower is handed exactly the window it writes.
    let num_clips = anchor_rows.len();
    if indptr.len() < num_clips + 1 {
        eprintln!("[driver-cuda] encode: audio CSR is shorter than its clip count");
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    let Ok(stream) = crate::device::OwnedStream::new(0) else {
        return PIE_STATUS_DRIVER_ERROR;
    };
    if let Err(why) = aud_tower::encode(
        &weights,
        features,
        feature_indptr,
        out,
        &mut indptr[..num_clips + 1],
        stream.as_ref(),
    ) {
        return i32::from(why);
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
            // THE PLAN, DESTRUCTURED. The vision tower now writes
            // `output_rows` from Rust while it reads `image_pixels`, and one
            // `&mut` to the whole plan cannot say that two of its fields are
            // disjoint. The C++ shape took the out-params as raw pointers to
            // dodge exactly this; a destructure says it in the type system
            // instead, and the towers take slices.
            let driver_api::MediaEncodePlan {
                image_grids: _,
                image_pixels,
                image_pixel_indptr,
                image_patch_positions,
                image_anchor_rows,
                audio_features,
                audio_feature_indptr,
                audio_anchor_rows,
                output_rows,
                output_row_indptr,
            } = encode;
            let Some(model) = state.model.as_ref() else {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            let num_images = image_anchor_rows.len();
            let num_clips = audio_anchor_rows.len();
            if num_images == 0 && num_clips == 0 {
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            if output_row_indptr.len() < num_images + num_clips + 1 {
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
                let st = encode_audio_arm(
                    model,
                    audio_features,
                    audio_feature_indptr,
                    audio_anchor_rows,
                    output_rows,
                    output_row_indptr,
                );
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
            // The tower's weights, marshalled from the flat table by the Rust
            // that replaced `gemma4_towers_c.cpp` — the same stride-41 offsets,
            // in the crate that walks them.
            let weights = match vis_tower::Weights::from_flat(
                patch_w,
                pos_table,
                embed_proj,
                &table,
                vc.layers as usize,
                vc.hidden as i32,
                vc.heads as i32,
                vc.intermediate as i32,
                i32::try_from(pos_table_size).unwrap_or(0),
                i32::try_from(text_hidden).unwrap_or(0),
                vc.pooling_kernel as i32,
                vc.norm_eps,
                vc.rope_theta,
            ) {
                Ok(w) => w,
                Err(why) => return Err(i32::from(why)),
            };
            // ONE cuBLAS handle for the whole encode, bound to this stream.
            // The C++ walk built one per IMAGE (`kernels::gemm::CublasHandle
            // cublas(S)` inside `run_gemma4_vision`), and `Shell::cublas`
            // records what that costs: `cublasDestroy` is 3.2 ms, most of it
            // the workspace. One per call is the same object with the same
            // stream bound, made once.
            let mut cublas_ops = crate::device::cublas::LiveCublas;
            let mut cublas = match crate::device::cublas::CublasHandle::create(
                &mut cublas_ops,
                stream.as_ref().as_raw().cast(),
            ) {
                Ok(h) => h,
                Err(why) => {
                    eprintln!("[driver-cuda] encode: cuBLAS handle: {why}");
                    return Err(PIE_STATUS_DRIVER_ERROR);
                }
            };
            let cublas_raw: *mut std::ffi::c_void =
                cublas.handle().expect("just created").cast();
            let mut vis_bounds = vec![0u32; num_images + 1];
            let walked = vis_tower::encode(
                &weights,
                image_pixels,
                image_pixel_indptr,
                image_patch_positions,
                output_rows,
                &mut vis_bounds,
                cublas_raw,
                stream.as_ref(),
            );
            // Released on BOTH paths: `CublasHandle`'s destructor asserts the
            // token was handed back, because the C++ class's was the leak this
            // port is not repeating.
            cublas.release(&mut cublas_ops);
            if let Err(why) = walked {
                return Err(i32::from(why));
            }
            if stream.as_ref().synchronize().is_err() {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
            // Compose the shared CSR the C++ `Context::encode` writes: the
            // vision segment's boundaries verbatim, then the audio segment's
            // shifted by the vision row count.
            output_row_indptr[..num_images + 1].copy_from_slice(&vis_bounds);
            if num_clips > 0 {
                let row_offset = *vis_bounds.last().unwrap_or(&0) as usize;
                let consumed = row_offset * text_hidden * 2;
                if consumed > output_rows.len() {
                    return Err(PIE_STATUS_INVALID_ARGUMENT);
                }
                let mut audio_bounds = vec![0u32; num_clips + 1];
                let st = encode_audio_arm(
                    model,
                    audio_features,
                    audio_feature_indptr,
                    audio_anchor_rows,
                    &mut output_rows[consumed..],
                    &mut audio_bounds,
                );
                if st != PIE_STATUS_OK {
                    return Err(st);
                }
                for c in 0..num_clips {
                    output_row_indptr[num_images + 1 + c] =
                        u32::try_from(row_offset).unwrap_or(u32::MAX) + audio_bounds[c + 1];
                }
            }
            notify_done(state);
            Ok(())
        })
    }
}
