//! pie:core/audio-out — native audio OUTPUT (CSM-1B + Mimi). The inverse of
//! `media` (perception -> text): the engine EMITS Mimi codec tokens and the
//! Mimi decoder turns them back into a 24 kHz waveform. See AUDIO_OUTPUT.md.
//!
//! **Model-agnostic, mirroring `media`.** The inferlet supplies neutral intent
//! (text + [`Voice`] + an optional target duration) via a [`SpeechRequest`];
//! everything model-specific — the "[speaker]text" prompt framing, the BOS/EOS
//! the CSM processor adds, the 12.5 Hz frame rate, the 24 kHz sample rate — is
//! applied here, dispatched off the bound model's arch. The driver runs the
//! whole frame-stepped loop (backbone prefill -> per-frame depth loop -> Mimi
//! decode) in one `generate_audio` cold-path request (AdapterOp::GenerateAudio).
//! The returned [`Speech`] is self-describing, so the inferlet never hardcodes a
//! model constant.

use crate::api::pie;
use crate::api::pie::core::audio_out::{SpeechRequest, Voice};
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

// ---- CSM (Llama-3 tokenizer) audio-output front-end constants --------------
// CSM is the only audio-output arch today. These are the model-specific values
// that used to leak into the inferlet; they live host-side now.

/// Llama-3 `<|begin_of_text|>` — the CSM processor prepends it (add_special_tokens).
const CSM_BOS: u32 = 128000;
/// Llama-3 `<|end_of_text|>` — the CSM processor appends it. Without the BOS/EOS
/// framing the backbone emits the audio-EOS frame immediately (no speech).
const CSM_EOS: u32 = 128001;
/// Mimi output sample rate (24 kHz mono).
const CSM_SAMPLE_RATE: u32 = 24_000;
/// Mimi frame period: 12.5 Hz => 80 ms per frame (= 1920 samples @ 24 kHz).
const CSM_MS_PER_FRAME: u32 = 80;
/// Frame cap applied when the request gives no `max-duration-ms` (~82 s). The
/// generation still stops early at the all-EOS frame; this is just a safety cap.
const CSM_DEFAULT_MAX_FRAMES: u32 = 1024;

/// A generated audio clip held host-side. Self-describing — carries its own
/// sample rate and channel count (`pcm` is handed to the guest only on request).
pub struct Speech {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Build the CSM audio-output prompt host-side: BOS + "[speaker]text" + EOS,
/// using the bound model's tokenizer. Mirrors the verified prompt the CSM
/// processor produces with `add_special_tokens=True`.
fn csm_frame_prompt(model: &crate::model::Model, text: &str, speaker: u32) -> Vec<u32> {
    let prompt = format!("[{speaker}]{text}");
    let mut ids = Vec::with_capacity(2 + text.len() / 3);
    ids.push(CSM_BOS);
    ids.extend(model.tokenize(&prompt));
    ids.push(CSM_EOS);
    ids
}

impl pie::core::audio_out::Host for InstanceState {}

impl pie::core::audio_out::HostSpeech for InstanceState {
    async fn generate(
        &mut self,
        req: SpeechRequest,
    ) -> Result<Result<Resource<Speech>, String>> {
        if req.text.trim().is_empty() {
            return Ok(Err("audio-out: empty text".into()));
        }
        // Gate on arch and frame the prompt — all host-side.
        let prompt = {
            let m = crate::model::model();
            let arch = m.arch_name();
            // CSM is the only audio-output arch. The driver also guards (negative
            // status when the bound model isn't CSM), but reject early here with a
            // clear message for every other arch.
            if arch != "csm" {
                return Ok(Err(format!(
                    "model '{}' (arch '{arch}') has no audio-output front-end \
                     (requires a CSM checkpoint, e.g. eustlb/csm-1b)",
                    m.name()
                )));
            }
            let speaker = match &req.voice {
                Voice::Speaker(n) => *n,
                Voice::Named(v) => {
                    return Ok(Err(format!(
                        "model '{}' (CSM) selects voices by integer id, not name {v:?}",
                        m.name()
                    )));
                }
            };
            csm_frame_prompt(m, &req.text, speaker)
        };
        // Neutral duration -> model frame count.
        let max_frames = match req.max_duration_ms {
            Some(ms) => ms.div_ceil(CSM_MS_PER_FRAME).max(1),
            None => CSM_DEFAULT_MAX_FRAMES,
        };
        // Default device for the single model (single-driver configs use driver 0).
        let driver_idx = crate::context::get_device(0);
        match crate::driver::generate_audio(driver_idx, &prompt, max_frames).await {
            Ok(pcm) => {
                let speech = Speech {
                    pcm,
                    sample_rate: CSM_SAMPLE_RATE,
                    channels: 1,
                };
                Ok(Ok(self.ctx().table.push(speech)?))
            }
            Err(e) => Ok(Err(format!("audio-out generate failed: {e:#}"))),
        }
    }

    async fn sample_rate(&mut self, this: Resource<Speech>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.sample_rate)
    }

    async fn channels(&mut self, this: Resource<Speech>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.channels)
    }

    async fn duration_ms(&mut self, this: Resource<Speech>) -> Result<u32> {
        let s = self.ctx().table.get(&this)?;
        let frames = if s.channels == 0 {
            0
        } else {
            s.pcm.len() as u64 / s.channels as u64
        };
        Ok((frames * 1000 / s.sample_rate.max(1) as u64) as u32)
    }

    async fn pcm(&mut self, this: Resource<Speech>) -> Result<Vec<f32>> {
        Ok(self.ctx().table.get(&this)?.pcm.clone())
    }

    async fn drop(&mut self, this: Resource<Speech>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
