pub mod chat;
pub mod forward;
pub mod frames;
pub mod grammar;
pub mod kv_working_set;
pub mod media;
pub mod model;
pub mod pipeline;
pub mod reasoning;
pub mod rs_working_set;
pub mod session;
pub mod speech;
pub mod system;
pub mod tokenizer;
pub mod tools;
pub mod types;

use crate::inferlet::ProcessCtx;
use wasmtime::component::HasSelf;
use wasmtime_wasi::WasiView;
#[cfg(target_arch = "wasm32")]
use wasmtime_wasi as wasmtime_wasi_http;

impl crate::pipeline::fire::FireContext for ProcessCtx {
    fn resources(&mut self) -> &mut wasmtime::component::ResourceTable {
        self.ctx().table
    }

    fn process_id(&self) -> uuid::Uuid {
        self.id()
    }

    async fn settle_pipeline_tail(&mut self) -> anyhow::Result<()> {
        crate::inferlet::process::gate::drain_pending_fires(self).await
    }
}

wasmtime::component::bindgen!({
    path: "../inferlet/wit",
    world: "inferlet",
    anyhow: true,
    with: {
        "wasi:http": wasmtime_wasi_http::p3::bindings::http,
        "wasi:clocks": wasmtime_wasi::p3::bindings::clocks,
        "wasi:filesystem": wasmtime_wasi::p3::bindings::filesystem,
        "pie:inferlet/working-set.kv-working-set": crate::store::kv::working_set::KvWorkingSet,
        "pie:inferlet/grammar.grammar": grammar::Grammar,
        "pie:inferlet/grammar.matcher": grammar::Matcher,
        "pie:inferlet/channel.channel": forward::Channel,
        "pie:inferlet/forward.forward-pass": forward::ForwardPass,
        "pie:inferlet/forward-recurrent.forward-pass": forward::ForwardPass,
        "pie:inferlet/forward-hybrid.forward-pass": forward::ForwardPass,
        "pie:inferlet/forward-diffusion.forward-pass": forward::ForwardPass,
        "pie:inferlet/pipeline.pipeline": pipeline::Pipeline,
        "pie:inferlet/working-set.rs-working-set": crate::store::rs::working_set::RsWorkingSet,
        "pie:inferlet/media.image": media::Image,
        "pie:inferlet/media.video": media::Video,
        "pie:inferlet/media.audio": media::Audio,
        "pie:inferlet/frames.frames": frames::Frames,
        "pie:inferlet/frames.pcm": frames::Pcm,
        "pie:inferlet/speech.speech": speech::Speech,
        "pie:inferlet/chat.decoder": chat::Decoder,
        "pie:inferlet/tools.decoder": tools::Decoder,
        "pie:inferlet/reasoning.decoder": reasoning::Decoder,
    },
    imports: { default: async | trappable },
    exports: { default: async },
});

pub fn add_to_linker(
    linker: &mut wasmtime::component::Linker<ProcessCtx>,
) -> Result<(), wasmtime::Error> {
    type D = HasSelf<ProcessCtx>;
    pie::inferlet::types::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::pipeline::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::working_set::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::model::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::tokenizer::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::grammar::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::channel::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::forward_recurrent::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::forward_hybrid::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::forward_diffusion::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::session::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::media::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::speech::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::frames::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::system::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::chat::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::tools::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::reasoning::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;

    Ok(())
}
