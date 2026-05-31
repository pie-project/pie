//! pie:core/audio-out — native audio OUTPUT (CSM-1B + Mimi). The inverse of
//! `media` (perception -> text): the engine EMITS Mimi codec tokens and the
//! Mimi decoder turns them back into a 24 kHz waveform. See AUDIO_OUTPUT.md.
//!
//! `generate` is a single synchronous call: the tokenized "[speaker]text"
//! prompt goes in, fully-decoded f32 PCM comes out. The driver runs the whole
//! frame-stepped loop (backbone prefill -> per-frame depth loop -> Mimi decode)
//! in one `generate_audio` cold-path request (AdapterOp::GenerateAudio).

use crate::api::model::Model;
use crate::api::pie;
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::audio_out::Host for InstanceState {
    async fn generate(
        &mut self,
        model: Resource<Model>,
        tokens: Vec<u32>,
        max_frames: u32,
    ) -> Result<Result<Vec<f32>, String>> {
        let (arch, name, model_idx) = {
            let m = self.ctx().table.get(&model)?;
            (
                m.model.arch_name().to_string(),
                m.model.name().to_string(),
                m.model_id,
            )
        };
        // CSM is the only audio-output arch. The driver also guards (returns a
        // negative status when the bound model isn't CSM), but reject early here
        // with a clear message for every other arch.
        if arch != "csm" {
            return Ok(Err(format!(
                "model '{name}' (arch '{arch}') has no audio-output front-end \
                 (requires a CSM checkpoint, e.g. eustlb/csm-1b)"
            )));
        }
        if tokens.is_empty() {
            return Ok(Err("audio-out generate: empty prompt".into()));
        }
        // Default device for the model (single-driver configs use driver 0).
        let driver_idx = crate::context::get_device(model_idx, 0);
        match crate::driver::generate_audio(driver_idx, &tokens, max_frames).await {
            Ok(pcm) => Ok(Ok(pcm)),
            Err(e) => Ok(Err(format!("audio-out generate failed: {e:#}"))),
        }
    }
}
