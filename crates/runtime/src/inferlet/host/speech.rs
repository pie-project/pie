use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pie::inferlet::speech::{SpeechRequest, Voice};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

const CSM_BOS: u32 = 128000;
const CSM_EOS: u32 = 128001;
const CSM_SAMPLE_RATE: u32 = 24_000;
const CSM_MS_PER_FRAME: u32 = 80;
const CSM_DEFAULT_MAX_FRAMES: u32 = 1024;

pub struct Speech {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
}

fn csm_frame_prompt(model: &crate::model::Model, text: &str, speaker: u32) -> Vec<u32> {
    let prompt = format!("[{speaker}]{text}");
    let mut ids = Vec::with_capacity(2 + text.len() / 3);
    ids.push(CSM_BOS);
    ids.extend(model.tokenize(&prompt));
    ids.push(CSM_EOS);
    ids
}

impl pie::inferlet::speech::Host for ProcessCtx {}

impl pie::inferlet::speech::HostSpeech for ProcessCtx {
    async fn generate(&mut self, req: SpeechRequest) -> Result<Result<Resource<Speech>, String>> {
        if req.text.trim().is_empty() {
            return Ok(Err("audio-out: empty text".into()));
        }
        let prompt = {
            let m = crate::model::model();
            let arch = m.arch();
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
        let max_frames = match req.max_duration_ms {
            Some(ms) => ms.div_ceil(CSM_MS_PER_FRAME).max(1),
            None => CSM_DEFAULT_MAX_FRAMES,
        };
        let engine_idx = 0;
        match crate::engine::generate_audio(engine_idx, &prompt, max_frames).await {
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
