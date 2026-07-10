pub mod chat;
pub mod grammar;
pub mod kv_working_set;
pub mod media;
pub mod messaging;
pub mod model;
pub mod reasoning;
pub mod rs_working_set;
pub mod session;
pub mod speech;
pub mod system;
pub mod tokenizer;
pub mod tools;
pub mod types;

use crate::instance::InstanceState;
use wasmtime::component::HasSelf;

wasmtime::component::bindgen!({
    path: "../../interface/inferlet",
    world: "inferlet",
    // wasmtime 46 split `wasmtime::Error` from `anyhow::Error`; keep the
    // generated host traits on `anyhow::Result` so the existing api/*.rs
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
        "pie:inferlet/working-set.kv-working-set": crate::working_set::kv::KvWorkingSet,
        // pie:inferlet/grammar (ex inference)
        "pie:inferlet/grammar.grammar": grammar::Grammar,
        "pie:inferlet/grammar.matcher": grammar::Matcher,
        // pie:inferlet/forward (ex ptir) — first-class channels + forward-pass
        // submission (the registry surface folded into forward-pass.new).
        "pie:inferlet/forward.channel": crate::ptir::ptir_host::Channel,
        "pie:inferlet/forward.forward-pass": crate::ptir::ptir_host::ForwardPass,
        "pie:inferlet/forward.pipeline": crate::ptir::ptir_host::Pipeline,
        // pie:inferlet/working-set (rs)
        "pie:inferlet/working-set.rs-working-set": crate::working_set::rs::RsWorkingSet,
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
    imports: {
        // subscribe returns a host-created stream<string>, which needs store
        // access to build -> generate it on HostWithStore (store flag), matching
        // how wasmtime_wasi p3 handles stream-returning funcs.
        "pie:inferlet/messaging.subscribe": store | async | trappable,
        default: async | trappable,
    },
    exports: { default: async },
});

pub fn add_to_linker(
    linker: &mut wasmtime::component::Linker<InstanceState>,
) -> Result<(), wasmtime::Error> {
    // Concrete on InstanceState: the async-func imports (execute/receive/pull/
    // subscribe) are generated on `HostWithStore` traits implemented for
    // `HasSelf<InstanceState>`, so the linker `D` type must be concrete.
    type D = HasSelf<InstanceState>;
    pie::inferlet::types::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::working_set::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::model::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::tokenizer::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::grammar::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::messaging::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::session::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::media::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::speech::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::system::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::chat::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::tools::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::inferlet::reasoning::add_to_linker::<InstanceState, D>(linker, |s| s)?;

    Ok(())
}
