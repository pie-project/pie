//! `pie:inferlet` WIT host boundary: bindgen! + `add_to_linker`, one thin
//! host-glue file per interface.
//!
//! **Accepted layering exception.** `session.rs` calls
//! `crate::server::send_file` and `crate::server::inbox::receive`: guest-to-
//! client I/O goes through the server's facade. This is the one documented
//! upward exception in this crate; revisit only if a second one appears.

pub mod chat;
pub mod forward;
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

/// Implements `pipeline::fire`'s narrow resource-table/process-identity seam
/// for the process context, so the fire engine's orchestration functions
/// (generic over `C: FireContext`) can run with `self: &mut ProcessCtx`
/// without `pipeline/` ever naming `ProcessCtx`/`inferlet` itself.
impl crate::pipeline::fire::FireContext for ProcessCtx {
    fn resources(&mut self) -> &mut wasmtime::component::ResourceTable {
        self.ctx().table
    }

    fn process_id(&self) -> uuid::Uuid {
        self.id()
    }

    fn kv_working_sets(
        &self,
    ) -> std::collections::HashSet<crate::store::kv::page_table::WorkingSetId> {
        self.residency_snapshot()
            .kv_working_sets
            .into_iter()
            .filter_map(|(model, driver, ws)| (model == 0 && driver == 0).then_some(ws))
            .collect()
    }

    async fn honor_preemption(&mut self) -> anyhow::Result<()> {
        crate::inferlet::process::preemption::honor(self).await
    }
}

wasmtime::component::bindgen!({
    path: "../../interface/inferlet",
    world: "inferlet",
    // wasmtime 46 split `wasmtime::Error` from `anyhow::Error`; keep the
    // generated host traits on `anyhow::Result` so the existing host/*.rs
    // impls (anyhow `?`/`bail!`/`anyhow!`) continue to compile unchanged.
    anyhow: true,
    with: {
        // Standard wasi 0.3 surfaces the world imports resolve to the wasmtime
        // p3 host bindings (what the p3 linkers in linker.rs implement) rather
        // than generating fresh host code. Package-level keys cover every
        // reachable sub-interface (http/{client,types}, clocks/{types,
        // monotonic-clock,system-clock}, filesystem/{types,preopens}).
        "wasi:http": wasmtime_wasi_http::p3::bindings::http,
        "wasi:clocks": wasmtime_wasi::p3::bindings::clocks,
        "wasi:filesystem": wasmtime_wasi::p3::bindings::filesystem,
        // pie:inferlet/working-set (kv); rs-working-set below
        "pie:inferlet/working-set.kv-working-set": crate::store::kv::working_set::KvWorkingSet,
        // pie:inferlet/grammar
        "pie:inferlet/grammar.grammar": grammar::Grammar,
        "pie:inferlet/grammar.matcher": grammar::Matcher,
        // pie:inferlet/forward — first-class channels + forward-pass
        // submission (the registry surface folded into forward-pass.new).
        "pie:inferlet/forward.channel": forward::Channel,
        "pie:inferlet/forward.forward-pass": forward::ForwardPass,
        // pie:inferlet/pipeline — the ordering domain (hoisted out of forward
        // so working-set mutators can take borrow<pipeline> without a cycle).
        "pie:inferlet/pipeline.pipeline": pipeline::Pipeline,
        // pie:inferlet/working-set (rs)
        "pie:inferlet/working-set.rs-working-set": crate::store::rs::working_set::RsWorkingSet,
        // pie:inferlet/media
        "pie:inferlet/media.image": media::Image,
        "pie:inferlet/media.video": media::Video,
        "pie:inferlet/media.audio": media::Audio,
        // pie:inferlet/speech (ex audio-out)
        "pie:inferlet/speech.speech": speech::Speech,
        // pie:inferlet chat / tools / reasoning (ex pie:instruct)
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
    // Concrete on ProcessCtx: the async-func imports (execute/receive/pull/
    // subscribe) are generated on `HostWithStore` traits implemented for
    // `HasSelf<ProcessCtx>`, so the linker `D` type must be concrete.
    type D = HasSelf<ProcessCtx>;
    pie::inferlet::types::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::pipeline::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::working_set::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::model::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::tokenizer::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::grammar::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::session::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::media::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::speech::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::system::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::chat::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::tools::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;
    pie::inferlet::reasoning::add_to_linker::<ProcessCtx, D>(linker, |s| s)?;

    Ok(())
}
